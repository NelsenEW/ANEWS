use crate::raycaster::ResolvedObstacle;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollisionShape {
    Circle {
        offset_x: f64,
        offset_y: f64,
        radius: f64,
    },
    Rect {
        offset_x: f64,
        offset_y: f64,
        width: f64,
        height: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FurnitureType {
    pub name: String,
    pub visual_shape: String,
    pub default_width: f64,
    pub default_height: f64,
    pub laser_profile: Vec<CollisionShape>,
    pub depth_profile: Vec<CollisionShape>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacedFurniture {
    pub id: String,
    pub type_id: String,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub label: String,
}

#[derive(Deserialize)]
struct FurnitureYaml {
    furniture_types: HashMap<String, FurnitureTypeYaml>,
}

#[derive(Deserialize)]
struct FurnitureTypeYaml {
    name: String,
    visual: VisualYaml,
    #[serde(default)]
    laser_profile: Vec<CollisionShapeYaml>,
    #[serde(default)]
    depth_profile: Vec<CollisionShapeYaml>,
}

#[derive(Deserialize)]
struct VisualYaml {
    shape: String,
    width: f64,
    height: f64,
}

#[derive(Deserialize)]
struct CollisionShapeYaml {
    shape: String,
    #[serde(default)]
    offset: Vec<f64>,
    #[serde(default)]
    radius: f64,
    #[serde(default)]
    width: f64,
    #[serde(default)]
    height: f64,
}

fn parse_shape(y: &CollisionShapeYaml) -> CollisionShape {
    let ox = *y.offset.first().unwrap_or(&0.0);
    let oy = *y.offset.get(1).unwrap_or(&0.0);
    match y.shape.as_str() {
        "circle" => CollisionShape::Circle {
            offset_x: ox,
            offset_y: oy,
            radius: y.radius,
        },
        "rect" => CollisionShape::Rect {
            offset_x: ox,
            offset_y: oy,
            width: y.width,
            height: y.height,
        },
        other => panic!("Unknown collision shape: {other}"),
    }
}

pub fn load_furniture_types(
    yaml_path: &Path,
) -> Result<HashMap<String, FurnitureType>, Box<dyn std::error::Error>> {
    let yaml_str = std::fs::read_to_string(yaml_path)?;
    let raw: FurnitureYaml = serde_yaml::from_str(&yaml_str)?;
    let mut result = HashMap::new();
    for (key, entry) in raw.furniture_types {
        result.insert(
            key,
            FurnitureType {
                name: entry.name,
                visual_shape: entry.visual.shape,
                default_width: entry.visual.width,
                default_height: entry.visual.height,
                laser_profile: entry.laser_profile.iter().map(parse_shape).collect(),
                depth_profile: entry.depth_profile.iter().map(parse_shape).collect(),
            },
        );
    }
    Ok(result)
}

pub fn resolve_collision_shapes(
    ftype: &FurnitureType,
    profile: &str,
    instance_width: Option<f64>,
    instance_height: Option<f64>,
) -> Vec<CollisionShape> {
    let shapes = match profile {
        "laser" => &ftype.laser_profile,
        "depth" => &ftype.depth_profile,
        _ => return vec![],
    };
    let w = instance_width.unwrap_or(ftype.default_width);
    let h = instance_height.unwrap_or(ftype.default_height);
    let scale_x = w / ftype.default_width;
    let scale_y = h / ftype.default_height;
    shapes
        .iter()
        .map(|shape| match shape {
            CollisionShape::Circle {
                offset_x,
                offset_y,
                radius,
            } => CollisionShape::Circle {
                offset_x: offset_x * scale_x,
                offset_y: offset_y * scale_y,
                radius: *radius * scale_x.max(scale_y),
            },
            CollisionShape::Rect {
                offset_x,
                offset_y,
                width,
                height,
            } => CollisionShape::Rect {
                offset_x: offset_x * scale_x,
                offset_y: offset_y * scale_y,
                width: width * scale_x,
                height: height * scale_y,
            },
        })
        .collect()
}

pub fn resolve_furniture_to_obstacles(
    furniture: &[PlacedFurniture],
    types: &HashMap<String, FurnitureType>,
    profile: &str,
) -> Vec<ResolvedObstacle> {
    let mut obstacles = Vec::new();
    for item in furniture {
        let ftype = match types.get(&item.type_id) {
            Some(ft) => ft,
            None => continue,
        };
        let shapes = resolve_collision_shapes(ftype, profile, item.width, item.height);
        let cos_t = item.rotation.cos();
        let sin_t = item.rotation.sin();
        for shape in &shapes {
            match shape {
                CollisionShape::Circle {
                    offset_x,
                    offset_y,
                    radius,
                } => {
                    let world_x = item.x + offset_x * cos_t - offset_y * sin_t;
                    let world_y = item.y + offset_x * sin_t + offset_y * cos_t;
                    obstacles.push(ResolvedObstacle::Circle {
                        cx: world_x,
                        cy: world_y,
                        radius: *radius,
                    });
                }
                CollisionShape::Rect {
                    offset_x,
                    offset_y,
                    width,
                    height,
                } => {
                    let world_x = item.x + offset_x * cos_t - offset_y * sin_t;
                    let world_y = item.y + offset_x * sin_t + offset_y * cos_t;
                    obstacles.push(ResolvedObstacle::Rect {
                        cx: world_x,
                        cy: world_y,
                        half_w: width / 2.0,
                        half_h: height / 2.0,
                        rotation: item.rotation,
                    });
                }
            }
        }
    }
    obstacles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table4_type() -> FurnitureType {
        FurnitureType {
            name: "Table 4-leg".into(),
            visual_shape: "rect".into(),
            default_width: 1.5,
            default_height: 0.8,
            laser_profile: vec![
                CollisionShape::Circle {
                    offset_x: -0.68,
                    offset_y: -0.38,
                    radius: 0.03,
                },
                CollisionShape::Circle {
                    offset_x: 0.68,
                    offset_y: -0.38,
                    radius: 0.03,
                },
                CollisionShape::Circle {
                    offset_x: 0.68,
                    offset_y: 0.38,
                    radius: 0.03,
                },
                CollisionShape::Circle {
                    offset_x: -0.68,
                    offset_y: 0.38,
                    radius: 0.03,
                },
            ],
            depth_profile: vec![CollisionShape::Rect {
                offset_x: 0.0,
                offset_y: 0.0,
                width: 1.5,
                height: 0.8,
            }],
        }
    }

    #[test]
    fn test_resolve_laser_profile_default_size() {
        let ft = make_table4_type();
        let shapes = resolve_collision_shapes(&ft, "laser", None, None);
        assert_eq!(shapes.len(), 4);
        match &shapes[0] {
            CollisionShape::Circle { offset_x, .. } => assert!((*offset_x - (-0.68)).abs() < 1e-9),
            _ => panic!("Expected circle"),
        }
    }

    #[test]
    fn test_resolve_depth_profile_scaled() {
        let ft = make_table4_type();
        let shapes = resolve_collision_shapes(&ft, "depth", Some(3.0), Some(0.8));
        assert_eq!(shapes.len(), 1);
        match &shapes[0] {
            CollisionShape::Rect { width, height, .. } => {
                assert!((*width - 3.0).abs() < 1e-9);
                assert!((*height - 0.8).abs() < 1e-9);
            }
            _ => panic!("Expected rect"),
        }
    }

    #[test]
    fn test_resolve_circle_profile_scaled() {
        let ft = make_table4_type();
        let shapes = resolve_collision_shapes(&ft, "laser", Some(0.75), Some(0.4));
        assert_eq!(shapes.len(), 4);
        match &shapes[0] {
            CollisionShape::Circle {
                offset_x,
                offset_y,
                radius,
            } => {
                assert!((*offset_x - (-0.34)).abs() < 1e-9);
                assert!((*offset_y - (-0.19)).abs() < 1e-9);
                assert!((*radius - 0.015).abs() < 1e-9);
            }
            _ => panic!("Expected circle"),
        }
    }

    #[test]
    fn test_resolve_furniture_to_obstacles_world_coords() {
        let mut types = HashMap::new();
        types.insert("table4".into(), make_table4_type());
        let furniture = vec![PlacedFurniture {
            id: "f1".into(),
            type_id: "table4".into(),
            x: 5.0,
            y: 3.0,
            rotation: 0.0,
            width: None,
            height: None,
            label: "T1".into(),
        }];
        let obstacles = resolve_furniture_to_obstacles(&furniture, &types, "laser");
        assert_eq!(obstacles.len(), 4);
        match &obstacles[0] {
            ResolvedObstacle::Circle { cx, cy, radius } => {
                assert!((*cx - 4.32).abs() < 1e-6, "cx={cx}");
                assert!((*cy - 2.62).abs() < 1e-6, "cy={cy}");
                assert!((*radius - 0.03).abs() < 1e-9);
            }
            _ => panic!("Expected circle"),
        }
    }

    #[test]
    fn test_resolve_furniture_rotated() {
        let mut types = HashMap::new();
        types.insert("table4".into(), make_table4_type());
        let furniture = vec![PlacedFurniture {
            id: "f1".into(),
            type_id: "table4".into(),
            x: 0.0,
            y: 0.0,
            rotation: std::f64::consts::FRAC_PI_2,
            width: None,
            height: None,
            label: "T1".into(),
        }];
        let obstacles = resolve_furniture_to_obstacles(&furniture, &types, "laser");
        match &obstacles[0] {
            ResolvedObstacle::Circle { cx, cy, .. } => {
                assert!((*cx - 0.38).abs() < 1e-6, "cx={cx}");
                assert!((*cy - (-0.68)).abs() < 1e-6, "cy={cy}");
            }
            _ => panic!("Expected circle"),
        }
    }
}
