use crate::map::OccupancyGrid;

#[derive(Debug, Clone)]
pub enum ResolvedObstacle {
    Circle {
        cx: f64,
        cy: f64,
        radius: f64,
    },
    Rect {
        cx: f64,
        cy: f64,
        half_w: f64,
        half_h: f64,
        rotation: f64,
    },
}

pub fn ray_circle_intersect(ox: f64, oy: f64, dx: f64, dy: f64, cx: f64, cy: f64, r: f64) -> f64 {
    let fx = ox - cx;
    let fy = oy - cy;
    let a = dx * dx + dy * dy;
    let b = 2.0 * (fx * dx + fy * dy);
    let c = fx * fx + fy * fy - r * r;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return f64::INFINITY;
    }
    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    if t1 >= 0.0 {
        return t1;
    }
    if t2 >= 0.0 {
        return t2;
    }
    f64::INFINITY
}

pub fn ray_aabb_intersect(
    ox: f64,
    oy: f64,
    dx: f64,
    dy: f64,
    cx: f64,
    cy: f64,
    hw: f64,
    hh: f64,
) -> f64 {
    let min_x = cx - hw;
    let max_x = cx + hw;
    let min_y = cy - hh;
    let max_y = cy + hh;
    let mut tmin = 0.0_f64;
    let mut tmax = f64::INFINITY;
    if dx.abs() < 1e-12 {
        if ox < min_x || ox > max_x {
            return f64::INFINITY;
        }
    } else {
        let mut t1 = (min_x - ox) / dx;
        let mut t2 = (max_x - ox) / dx;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return f64::INFINITY;
        }
    }
    if dy.abs() < 1e-12 {
        if oy < min_y || oy > max_y {
            return f64::INFINITY;
        }
    } else {
        let mut t1 = (min_y - oy) / dy;
        let mut t2 = (max_y - oy) / dy;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return f64::INFINITY;
        }
    }
    if tmin >= 0.0 { tmin } else { f64::INFINITY }
}

pub fn ray_obb_intersect(
    ox: f64,
    oy: f64,
    dx: f64,
    dy: f64,
    cx: f64,
    cy: f64,
    hw: f64,
    hh: f64,
    rotation: f64,
) -> f64 {
    let cos_r = (-rotation).cos();
    let sin_r = (-rotation).sin();
    let local_ox = (ox - cx) * cos_r - (oy - cy) * sin_r;
    let local_oy = (ox - cx) * sin_r + (oy - cy) * cos_r;
    let local_dx = dx * cos_r - dy * sin_r;
    let local_dy = dx * sin_r + dy * cos_r;
    ray_aabb_intersect(local_ox, local_oy, local_dx, local_dy, 0.0, 0.0, hw, hh)
}

pub fn raycast(
    grid: &OccupancyGrid,
    robot_x: f64,
    robot_y: f64,
    robot_theta: f64,
    angle_min: f64,
    angle_max: f64,
    angle_increment: f64,
    range_max: f64,
    range_min: f64,
    obstacles: &[ResolvedObstacle],
) -> Vec<f64> {
    let resolution = grid.metadata.resolution;
    let origin_x = grid.metadata.origin[0];
    let origin_y = grid.metadata.origin[1];
    let width = grid.width as i32;
    let height = grid.height as i32;
    let gx = (robot_x - origin_x) / resolution;
    let gy = (robot_y - origin_y) / resolution;
    let n_rays = ((angle_max - angle_min) / angle_increment).round() as usize + 1;
    let max_steps = range_max / resolution;
    let mut ranges = vec![range_max; n_rays];

    for i in 0..n_rays {
        let frac = if n_rays > 1 {
            i as f64 / (n_rays - 1) as f64
        } else {
            0.0
        };
        let angle = robot_theta + angle_min + frac * (angle_max - angle_min);
        let dx = angle.cos();
        let dy = angle.sin();
        let mut map_x = gx.floor() as i32;
        let mut map_y = gy.floor() as i32;
        let step_x: i32 = if dx >= 0.0 { 1 } else { -1 };
        let step_y: i32 = if dy >= 0.0 { 1 } else { -1 };
        let t_delta_x = if dx != 0.0 { (1.0 / dx).abs() } else { 1e30 };
        let t_delta_y = if dy != 0.0 { (1.0 / dy).abs() } else { 1e30 };
        let mut t_max_x = if dx >= 0.0 {
            (map_x as f64 + 1.0 - gx) * t_delta_x
        } else {
            (gx - map_x as f64) * t_delta_x
        };
        let mut t_max_y = if dy >= 0.0 {
            (map_y as f64 + 1.0 - gy) * t_delta_y
        } else {
            (gy - map_y as f64) * t_delta_y
        };
        let mut min_hit = f64::INFINITY;
        let mut t = 0.0_f64;
        while t < max_steps {
            if map_x < 0 || map_x >= width || map_y < 0 || map_y >= height {
                break;
            }
            let dist = t * resolution;
            if grid.cells[map_y as usize * grid.width + map_x as usize] {
                min_hit = dist;
                break;
            }
            if t_max_x < t_max_y {
                t = t_max_x;
                t_max_x += t_delta_x;
                map_x += step_x;
            } else {
                t = t_max_y;
                t_max_y += t_delta_y;
                map_y += step_y;
            }
        }
        for obs in obstacles {
            let d = match obs {
                ResolvedObstacle::Circle { cx, cy, radius } => {
                    ray_circle_intersect(robot_x, robot_y, dx, dy, *cx, *cy, *radius)
                }
                ResolvedObstacle::Rect {
                    cx,
                    cy,
                    half_w,
                    half_h,
                    rotation,
                } => {
                    if *rotation != 0.0 {
                        ray_obb_intersect(
                            robot_x, robot_y, dx, dy, *cx, *cy, *half_w, *half_h, *rotation,
                        )
                    } else {
                        ray_aabb_intersect(robot_x, robot_y, dx, dy, *cx, *cy, *half_w, *half_h)
                    }
                }
            };
            if d >= 0.0 && d < min_hit {
                min_hit = d;
            }
        }
        const NEAR_BLIND_M: f64 = 0.05;
        if min_hit.is_finite() && min_hit <= range_max {
            let blind_threshold = range_min.max(NEAR_BLIND_M);
            ranges[i] = if min_hit < blind_threshold {
                f64::NAN
            } else {
                min_hit
            };
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_circle_hit() {
        let d = ray_circle_intersect(0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0);
        assert!((d - 4.0).abs() < 1e-6, "Expected 4.0, got {d}");
    }

    #[test]
    fn test_ray_circle_miss() {
        let d = ray_circle_intersect(0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 1.0);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_ray_circle_inside() {
        let d = ray_circle_intersect(5.0, 0.0, 1.0, 0.0, 5.0, 0.0, 2.0);
        assert!((d - 2.0).abs() < 1e-6, "Expected 2.0, got {d}");
    }

    #[test]
    fn test_ray_aabb_hit() {
        let d = ray_aabb_intersect(0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 1.0);
        assert!((d - 4.0).abs() < 1e-6, "Expected 4.0, got {d}");
    }

    #[test]
    fn test_ray_aabb_miss() {
        let d = ray_aabb_intersect(0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 1.0, 1.0);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_ray_obb_rotated() {
        let d = ray_obb_intersect(
            0.0,
            0.0,
            1.0,
            0.0,
            5.0,
            0.0,
            1.0,
            1.0,
            std::f64::consts::FRAC_PI_4,
        );
        assert!(d.is_finite(), "Should hit rotated rect");
        assert!(
            d > 3.0 && d < 5.0,
            "Hit distance {d} should be between 3 and 5"
        );
    }

    #[test]
    fn test_raycast_empty_grid_returns_range_max() {
        let grid = OccupancyGrid {
            cells: vec![false; 100],
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let ranges = raycast(&grid, 5.0, 5.0, 0.0, -0.1, 0.1, 0.1, 50.0, 0.0, &[]);
        assert_eq!(ranges.len(), 3);
        for r in &ranges {
            assert!(*r >= 4.0);
        }
    }

    #[test]
    fn test_raycast_hits_wall() {
        let mut cells = vec![false; 100];
        for row in 0..10 {
            cells[row * 10 + 8] = true;
        }
        let grid = OccupancyGrid {
            cells,
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let ranges = raycast(&grid, 5.0, 5.0, 0.0, 0.0, 0.0, 1.0, 50.0, 0.0, &[]);
        assert_eq!(ranges.len(), 1);
        assert!(
            ranges[0] > 2.0 && ranges[0] < 4.0,
            "Expected ~3m, got {}",
            ranges[0]
        );
    }

    #[test]
    fn test_raycast_obstacle_closer_than_wall() {
        let grid = OccupancyGrid {
            cells: vec![false; 100],
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let obstacles = vec![ResolvedObstacle::Circle {
            cx: 7.0,
            cy: 5.0,
            radius: 0.5,
        }];
        let ranges = raycast(&grid, 5.0, 5.0, 0.0, 0.0, 0.0, 1.0, 50.0, 0.0, &obstacles);
        assert_eq!(ranges.len(), 1);
        assert!(
            (ranges[0] - 1.5).abs() < 0.1,
            "Expected ~1.5, got {}",
            ranges[0]
        );
    }

    #[test]
    fn test_raycast_close_wall_occludes_far_wall() {
        let mut cells = vec![false; 100];
        cells[5 * 10 + 6] = true;
        cells[5 * 10 + 9] = true;
        let grid = OccupancyGrid {
            cells,
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let ranges = raycast(&grid, 5.0, 5.5, 0.0, 0.0, 0.0, 1.0, 50.0, 2.0, &[]);
        assert_eq!(ranges.len(), 1);
        assert!(
            ranges[0].is_nan(),
            "Expected NaN (close wall occludes ray), got {}",
            ranges[0]
        );
    }

    #[test]
    fn test_raycast_close_wall_visible_when_range_min_zero() {
        let mut cells = vec![false; 100];
        cells[5 * 10 + 6] = true;
        cells[5 * 10 + 9] = true;
        let grid = OccupancyGrid {
            cells,
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let ranges = raycast(&grid, 5.0, 5.5, 0.0, 0.0, 0.0, 1.0, 50.0, 0.0, &[]);
        assert_eq!(ranges.len(), 1);
        assert!(
            ranges[0] > 0.5 && ranges[0] < 1.5,
            "Expected close wall at ~1.0, got {}",
            ranges[0]
        );
    }

    #[test]
    fn test_raycast_origin_inside_rect_obstacle_reports_nan_not_far_wall() {
        let mut cells = vec![false; 100];
        cells[5 * 10 + 9] = true;
        let grid = OccupancyGrid {
            cells,
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let obstacles = vec![ResolvedObstacle::Rect {
            cx: 5.0,
            cy: 5.5,
            half_w: 0.3,
            half_h: 0.3,
            rotation: 0.0,
        }];
        let ranges = raycast(&grid, 5.0, 5.5, 0.0, 0.0, 0.0, 1.0, 50.0, 0.0, &obstacles);
        assert_eq!(ranges.len(), 1);
        assert!(
            ranges[0].is_nan(),
            "Expected NaN (sensor overlapping obstacle = blind), got {}",
            ranges[0]
        );
    }

    #[test]
    fn test_raycast_close_obstacle_occludes_far_wall() {
        let mut cells = vec![false; 100];
        cells[5 * 10 + 9] = true;
        let grid = OccupancyGrid {
            cells,
            width: 10,
            height: 10,
            metadata: crate::map::MapMetadata {
                resolution: 1.0,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let obstacles = vec![ResolvedObstacle::Circle {
            cx: 6.0,
            cy: 5.5,
            radius: 0.3,
        }];
        let ranges = raycast(&grid, 5.0, 5.5, 0.0, 0.0, 0.0, 1.0, 50.0, 2.0, &obstacles);
        assert_eq!(ranges.len(), 1);
        assert!(
            ranges[0].is_nan(),
            "Expected NaN (close obstacle occludes ray), got {}",
            ranges[0]
        );
    }
}
