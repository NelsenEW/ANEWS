use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const SCHEMA: &str = "foodlink.venue/1";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Walls {
    #[serde(default)]
    pub from_map: bool,
    #[serde(default = "default_wall_height")]
    pub height: f64,
    #[serde(default)]
    pub clear_items: bool,
    #[serde(default = "default_clear_margin")]
    pub clear_margin: f64,
}

fn default_wall_height() -> f64 {
    2.6
}

fn default_clear_margin() -> f64 {
    0.15
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "lowercase")]
pub enum Part {
    Box {
        size: [f64; 3],
        z: f64,
        #[serde(default)]
        scale_xy: bool,
        #[serde(default)]
        offset: [f64; 2],
        #[serde(default)]
        material: Option<String>,
    },
    Cylinder {
        radius: f64,
        z0: f64,
        z1: f64,
        #[serde(default)]
        scale_xy: bool,
        #[serde(default)]
        offset: [f64; 2],
        #[serde(default)]
        material: Option<String>,
    },
    Leg4 {
        radius: f64,
        inset: f64,
        z0: f64,
        z1: f64,
        #[serde(default)]
        offset: [f64; 2],
        #[serde(default)]
        material: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemType {
    pub parts: Vec<Part>,
    #[serde(default)]
    pub model: Option<ModelRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub file: String,
    #[serde(default)]
    pub scale_xy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialDef {
    pub color: String,
    #[serde(default = "default_roughness")]
    pub roughness: f64,
    #[serde(default)]
    pub metallic: f64,
}

fn default_roughness() -> f64 {
    0.7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub yaw_deg: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub number: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mobility: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ItemChange {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yaw_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scene {
    #[serde(default)]
    pub remove: Vec<String>,
    #[serde(default, rename = "move")]
    pub moves: BTreeMap<String, ItemChange>,
    #[serde(default)]
    pub add: Vec<Item>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clear_walls: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Venue {
    pub schema: String,
    #[serde(default)]
    pub frame: String,
    #[serde(default)]
    pub map: Option<String>,
    #[serde(default)]
    pub walls: Walls,
    #[serde(default)]
    pub materials: BTreeMap<String, MaterialDef>,
    #[serde(default)]
    pub types: BTreeMap<String, ItemType>,
    #[serde(default)]
    pub items: Vec<Item>,
    #[serde(default)]
    pub scenes: BTreeMap<String, Scene>,
    #[serde(skip)]
    pub dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Solid {
    Box {
        center: [f64; 3],
        half: [f64; 3],
        yaw: f64,
    },
    Cylinder {
        center: [f64; 2],
        radius: f64,
        z0: f64,
        z1: f64,
    },
}

pub fn parse_venue(text: &str) -> Result<Venue, String> {
    let venue: Venue = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    if venue.schema != SCHEMA {
        return Err(format!("schema is {:?}, expected {SCHEMA}", venue.schema));
    }
    let mut seen = std::collections::BTreeSet::new();
    for item in &venue.items {
        if !seen.insert(item.id.clone()) {
            return Err(format!("duplicate item id {}", item.id));
        }
        if !venue.types.contains_key(&item.type_name) {
            return Err(format!("item {} has unknown type {}", item.id, item.type_name));
        }
    }
    for (name, def) in &venue.materials {
        parse_color(&def.color).map_err(|e| format!("material {name}: {e}"))?;
    }
    Ok(venue)
}

pub fn parse_color(hex: &str) -> Result<[f64; 3], String> {
    let digits = hex.strip_prefix('#').unwrap_or(hex);
    if digits.len() != 6 {
        return Err(format!("colour {hex:?} is not #rrggbb"));
    }
    let channel = |i: usize| u8::from_str_radix(&digits[i..i + 2], 16).map(|v| v as f64 / 255.0).map_err(|e| format!("colour {hex:?}: {e}"));
    Ok([channel(0)?, channel(2)?, channel(4)?])
}

impl Venue {
    pub fn model_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.types.values().filter_map(|t| t.model.as_ref().map(|m| m.file.clone())).collect();
        files.sort();
        files.dedup();
        files
    }

    pub fn map_path(&self) -> Option<PathBuf> {
        self.map.as_ref().map(|m| self.dir.join(m))
    }

    pub fn scene_wall_clears(&self, scene: Option<&str>) -> &[[f64; 2]] {
        scene.and_then(|n| self.scenes.get(n)).map(|s| s.clear_walls.as_slice()).unwrap_or(&[])
    }

    pub fn scene_items(&self, scene: Option<&str>) -> Result<Vec<Item>, String> {
        let mut items: Vec<Item> = self.items.clone();
        let Some(name) = scene else { return Ok(items) };
        let spec = self.scenes.get(name).ok_or_else(|| {
            format!("unknown scene {name:?}; known: {:?}", self.scenes.keys().collect::<Vec<_>>())
        })?;
        items.retain(|i| !spec.remove.contains(&i.id));
        for (id, change) in &spec.moves {
            let item = items
                .iter_mut()
                .find(|i| &i.id == id)
                .ok_or_else(|| format!("scene {name} moves unknown item {id}"))?;
            if let Some(v) = change.x {
                item.x = v;
            }
            if let Some(v) = change.y {
                item.y = v;
            }
            if let Some(v) = change.yaw_deg {
                item.yaw_deg = v;
            }
            if let Some(v) = change.size {
                item.size = Some(v);
            }
        }
        for item in &spec.add {
            if items.iter().any(|i| i.id == item.id) {
                return Err(format!("scene {name} adds existing id {}", item.id));
            }
            if !self.types.contains_key(&item.type_name) {
                return Err(format!("scene {name} item {} has unknown type {}", item.id, item.type_name));
            }
            items.push(item.clone());
        }
        Ok(items)
    }
}

fn local_to_world(item: &Item, lx: f64, ly: f64) -> [f64; 2] {
    let yaw = item.yaw_deg.to_radians();
    let (s, c) = yaw.sin_cos();
    [item.x + c * lx - s * ly, item.y + s * lx + c * ly]
}

pub fn expand_item(item: &Item, item_type: &ItemType) -> Vec<Solid> {
    let [width, depth] = item.size.unwrap_or([1.0, 1.0]);
    let yaw = item.yaw_deg.to_radians();
    let mut out = Vec::new();
    for part in &item_type.parts {
        match part {
            Part::Box { size, z, scale_xy, offset, .. } => {
                let (sx, sy) = if *scale_xy { (size[0] * width, size[1] * depth) } else { (size[0], size[1]) };
                let c = local_to_world(item, offset[0], offset[1]);
                out.push(Solid::Box { center: [c[0], c[1], z + size[2] / 2.0], half: [sx / 2.0, sy / 2.0, size[2] / 2.0], yaw });
            }
            Part::Cylinder { radius, z0, z1, scale_xy, offset, .. } => {
                let r = if *scale_xy { radius * width } else { *radius };
                out.push(Solid::Cylinder { center: local_to_world(item, offset[0], offset[1]), radius: r, z0: *z0, z1: *z1 });
            }
            Part::Leg4 { radius, inset, z0, z1, offset, .. } => {
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        let lx = sx * (width / 2.0 - inset) + offset[0];
                        let ly = sy * (depth / 2.0 - inset) + offset[1];
                        out.push(Solid::Cylinder { center: local_to_world(item, lx, ly), radius: *radius, z0: *z0, z1: *z1 });
                    }
                }
            }
        }
    }
    out
}

pub fn scene_solids(venue: &Venue, scene: Option<&str>) -> Result<Vec<(String, Solid)>, String> {
    let mut out = Vec::new();
    for item in venue.scene_items(scene)? {
        for solid in expand_item(&item, &venue.types[&item.type_name]) {
            out.push((item.id.clone(), solid));
        }
    }
    Ok(out)
}
