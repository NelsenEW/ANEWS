use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use crate::detectors::{check_collision, check_footprint_collision};
use crate::furniture::{FurnitureType, PlacedFurniture, resolve_furniture_to_obstacles};
use crate::human::HumanActor;
use crate::map::{MapMetadata, OccupancyGrid};
use crate::named_pose::{NamedPose, PoseKind};
use crate::robot_config::Footprint;
use crate::scenario::{ScenarioFurniture, ScenarioGoalRef};

#[derive(Serialize, PartialEq, Debug, Clone)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Serialize, Debug, Clone)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub check: String,
    pub message: String,
    pub position: Option<[f64; 2]>,
}

#[derive(Serialize, Debug)]
pub struct ValidationReport {
    pub valid: bool,
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn to_json(&self) -> serde_json::Value {
        let errors: Vec<&ValidationIssue> = self
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect();
        let warnings: Vec<&ValidationIssue> = self
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect();
        serde_json::json!({
            "valid": self.valid,
            "errors": errors,
            "warnings": warnings,
            "error_count": errors.len(),
            "warning_count": warnings.len(),
        })
    }
}

fn make_empty_grid() -> OccupancyGrid {
    OccupancyGrid {
        cells: vec![false],
        width: 1,
        height: 1,
        metadata: MapMetadata {
            resolution: 1.0,
            origin: [0.0, 0.0, 0.0],
            occupied_thresh: 0.65,
            free_thresh: 0.196,
            negate: false,
            image_path: PathBuf::new(),
        },
    }
}

pub fn validate_scenario_state(
    grid: &OccupancyGrid,
    robot_pose: Option<(f64, f64, f64)>,
    collision_radius: f64,
    footprint: Option<&Footprint>,
    goal: Option<(f64, f64)>,
    placed_furniture: &[PlacedFurniture],
    furniture_types: &HashMap<String, FurnitureType>,
    human_actors: &[HumanActor],
) -> ValidationReport {
    let mut issues: Vec<ValidationIssue> = Vec::new();

    let furniture_obstacles = if !placed_furniture.is_empty() && !furniture_types.is_empty() {
        resolve_furniture_to_obstacles(placed_furniture, furniture_types, "laser")
    } else {
        vec![]
    };

    if let Some((rx, ry, rtheta)) = robot_pose {
        let wall_hit = match footprint {
            Some(fp) => check_footprint_collision(grid, rx, ry, rtheta, fp, &[]),
            None => check_collision(grid, rx, ry, collision_radius, &[]),
        };
        if let Some((cx, cy)) = wall_hit {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                check: "robot_spawn".into(),
                message: format!("Robot spawn collides with wall at ({:.2}, {:.2})", cx, cy),
                position: Some([rx, ry]),
            });
        }

        if !furniture_obstacles.is_empty() {
            let empty_grid = make_empty_grid();
            let furn_hit = match footprint {
                Some(fp) => {
                    check_footprint_collision(&empty_grid, rx, ry, rtheta, fp, &furniture_obstacles)
                }
                None => {
                    check_collision(&empty_grid, rx, ry, collision_radius, &furniture_obstacles)
                }
            };
            if let Some((cx, cy)) = furn_hit {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    check: "robot_spawn".into(),
                    message: format!(
                        "Robot spawn collides with furniture at ({:.2}, {:.2})",
                        cx, cy
                    ),
                    position: Some([rx, ry]),
                });
            }
        }
    }

    if let Some((gx, gy)) = goal {
        let res = grid.metadata.resolution;
        let ox = grid.metadata.origin[0];
        let oy = grid.metadata.origin[1];
        let map_w = grid.width as f64 * res;
        let map_h = grid.height as f64 * res;

        if gx < ox || gx > ox + map_w || gy < oy || gy > oy + map_h {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                check: "goal_position".into(),
                message: format!("Goal ({:.2}, {:.2}) is outside map bounds", gx, gy),
                position: Some([gx, gy]),
            });
        } else {
            let (col, row) = grid.world_to_grid(gx, gy);
            if col >= 0 && (col as usize) < grid.width && row >= 0 && (row as usize) < grid.height {
                if grid.is_occupied(col as usize, row as usize) {
                    issues.push(ValidationIssue {
                        severity: Severity::Error,
                        check: "goal_position".into(),
                        message: format!("Goal ({:.2}, {:.2}) is in an occupied cell", gx, gy),
                        position: Some([gx, gy]),
                    });
                }
            }
        }
    }

    if !placed_furniture.is_empty() && !furniture_types.is_empty() {
        for item in placed_furniture {
            let (col, row) = grid.world_to_grid(item.x, item.y);
            if col >= 0 && (col as usize) < grid.width && row >= 0 && (row as usize) < grid.height {
                if grid.is_occupied(col as usize, row as usize) {
                    issues.push(ValidationIssue {
                        severity: Severity::Warning,
                        check: "furniture_overlap".into(),
                        message: format!("Furniture '{}' center is in an occupied cell", item.id),
                        position: Some([item.x, item.y]),
                    });
                    continue;
                }
            }
        }
    }

    if !human_actors.is_empty() {
        for actor in human_actors {
            for (i, &(wx, wy)) in actor.path.iter().enumerate() {
                if check_collision(grid, wx, wy, actor.radius, &[]).is_some() {
                    issues.push(ValidationIssue {
                        severity: Severity::Error,
                        check: "human_path_wall".into(),
                        message: format!(
                            "Human '{}' waypoint {} at ({:.2}, {:.2}) collides with WALL",
                            actor.name, i, wx, wy
                        ),
                        position: Some([wx, wy]),
                    });
                }
                if check_collision(grid, wx, wy, actor.radius, &furniture_obstacles).is_some() {
                    issues.push(ValidationIssue {
                        severity: Severity::Warning,
                        check: "human_path_furniture".into(),
                        message: format!(
                            "Human '{}' waypoint {} at ({:.2}, {:.2}) collides with furniture",
                            actor.name, i, wx, wy
                        ),
                        position: Some([wx, wy]),
                    });
                }
            }

            let segments: Vec<_> = if actor.loop_path && actor.path.len() >= 2 {
                let mut s: Vec<((f64, f64), (f64, f64))> =
                    actor.path.windows(2).map(|w| (w[0], w[1])).collect();
                s.push((*actor.path.last().unwrap(), actor.path[0]));
                s
            } else {
                actor.path.windows(2).map(|w| (w[0], w[1])).collect()
            };
            for ((ax, ay), (bx, by)) in &segments {
                let dx = bx - ax;
                let dy = by - ay;
                let dist = (dx * dx + dy * dy).sqrt();
                let n_samples = (dist / 0.1).ceil() as usize;
                for s in 1..n_samples {
                    let t = s as f64 / n_samples as f64;
                    let sx = ax + dx * t;
                    let sy = ay + dy * t;
                    if check_collision(grid, sx, sy, actor.radius, &[]).is_some() {
                        issues.push(ValidationIssue {
                            severity: Severity::Error,
                            check: "human_path_wall".into(),
                            message: format!(
                                "Human '{}' path crosses WALL near ({:.2}, {:.2})",
                                actor.name, sx, sy
                            ),
                            position: Some([sx, sy]),
                        });
                        break;
                    }
                    if check_collision(grid, sx, sy, actor.radius, &furniture_obstacles).is_some() {
                        issues.push(ValidationIssue {
                            severity: Severity::Warning,
                            check: "human_path_furniture".into(),
                            message: format!(
                                "Human '{}' path crosses furniture near ({:.2}, {:.2})",
                                actor.name, sx, sy
                            ),
                            position: Some([sx, sy]),
                        });
                        break;
                    }
                }
            }
        }
    }

    let has_errors = issues.iter().any(|i| i.severity == Severity::Error);
    ValidationReport {
        valid: !has_errors,
        issues,
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReachabilityFailure {
    OutOfBounds,
    SpawnUnreachable,
    WaypointInsideWall,
    WaypointTooTight,
    WaypointBehindWall,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestedFix {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub side: String,
    pub method: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PoseReachability {
    pub name: String,
    pub kind: String,
    pub x: f64,
    pub y: f64,
    pub reachable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_class: Option<ReachabilityFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_fix: Option<SuggestedFix>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InfeasibleGoal {
    pub index: usize,
    #[serde(rename = "ref")]
    pub ref_name: String,
    pub failure_class: ReachabilityFailure,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReachabilityReport {
    pub spawn_reachable: bool,
    pub robot_radius_m: f64,
    pub reachable_area_pct: f64,
    pub poses: Vec<PoseReachability>,
    pub tour_feasible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tour_infeasible_goals: Option<Vec<InfeasibleGoal>>,
}

fn compute_suggested_fix(
    pose: &NamedPose,
    reachable: &[bool],
    grid: &OccupancyGrid,
    robot_radius: f64,
    furniture: &[ScenarioFurniture],
) -> Option<SuggestedFix> {
    let is_pixel_reachable = |wx: f64, wy: f64| -> bool {
        let (c, r) = grid.world_to_grid(wx, wy);
        if c < 0 || (c as usize) >= grid.width || r < 0 || (r as usize) >= grid.height {
            return false;
        }
        reachable[r as usize * grid.width + c as usize]
    };

    if pose.kind == PoseKind::Delivery {
        let table = furniture.iter().find(|f| f.label == pose.name)?;

        let table_x = table.pose.first().copied().unwrap_or(0.0);
        let table_y = table.pose.get(1).copied().unwrap_or(0.0);
        let half_w = table.width / 2.0;
        let half_h = table.height / 2.0;
        let clearance = robot_radius + 0.15;

        let candidates = [
            (table_x, table_y - half_h - clearance, -std::f64::consts::FRAC_PI_2, "south"),
            (table_x + half_w + clearance, table_y, std::f64::consts::PI, "east"),
            (table_x - half_w - clearance, table_y, 0.0, "west"),
            (table_x, table_y + half_h + clearance, std::f64::consts::FRAC_PI_2, "north"),
        ];

        for (cx, cy, ctheta, side) in &candidates {
            if is_pixel_reachable(*cx, *cy) {
                return Some(SuggestedFix {
                    x: *cx,
                    y: *cy,
                    theta: *ctheta,
                    side: side.to_string(),
                    method: format!("Reachable side of table {}", pose.name),
                });
            }
        }

        None
    } else {
        let (col, row) = grid.world_to_grid(pose.x, pose.y);
        if col < 0 || (col as usize) >= grid.width || row < 0 || (row as usize) >= grid.height {
            return None;
        }

        let start_c = col as usize;
        let start_r = row as usize;

        let mut visited = vec![false; grid.width * grid.height];
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        visited[start_r * grid.width + start_c] = true;
        queue.push_back((start_c, start_r));

        while let Some((c, r)) = queue.pop_front() {
            if reachable[r * grid.width + c] {
                let (wx, wy) = grid.grid_to_world(c, r);
                return Some(SuggestedFix {
                    x: wx,
                    y: wy,
                    theta: pose.theta,
                    side: "nearest".into(),
                    method: "Nearest reachable point".into(),
                });
            }
            for (dc, dr) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nc = c as i32 + dc;
                let nr = r as i32 + dr;
                if nc >= 0 && (nc as usize) < grid.width && nr >= 0 && (nr as usize) < grid.height
                {
                    let ncu = nc as usize;
                    let nru = nr as usize;
                    let idx = nru * grid.width + ncu;
                    if !visited[idx] {
                        visited[idx] = true;
                        queue.push_back((ncu, nru));
                    }
                }
            }
        }

        None
    }
}

pub fn validate_reachability(
    grid: &OccupancyGrid,
    robot_radius: f64,
    spawn: (f64, f64, f64),
    named_poses: &[NamedPose],
    goals: &[ScenarioGoalRef],
    furniture: &[ScenarioFurniture],
) -> ReachabilityReport {
    let res = grid.metadata.resolution;

    let pixel_radius = (robot_radius / res).ceil() as i32;
    let mut inflated: Vec<bool> = grid.cells.clone();

    for row in 0..grid.height {
        for col in 0..grid.width {
            if grid.cells[row * grid.width + col] {
                let r_min = (row as i32 - pixel_radius).max(0) as usize;
                let r_max = (row as i32 + pixel_radius).min(grid.height as i32 - 1) as usize;
                let c_min = (col as i32 - pixel_radius).max(0) as usize;
                let c_max = (col as i32 + pixel_radius).min(grid.width as i32 - 1) as usize;
                for ir in r_min..=r_max {
                    for ic in c_min..=c_max {
                        let dr = ir as i32 - row as i32;
                        let dc = ic as i32 - col as i32;
                        if dr * dr + dc * dc <= pixel_radius * pixel_radius {
                            inflated[ir * grid.width + ic] = true;
                        }
                    }
                }
            }
        }
    }

    let (spawn_col, spawn_row) = grid.world_to_grid(spawn.0, spawn.1);
    let mut spawn_reachable = false;

    if spawn_col >= 0
        && (spawn_col as usize) < grid.width
        && spawn_row >= 0
        && (spawn_row as usize) < grid.height
    {
        let sc = spawn_col as usize;
        let sr = spawn_row as usize;
        if !inflated[sr * grid.width + sc] {
            spawn_reachable = true;
        }
    }

    let mut reachable: Vec<bool> = vec![false; grid.width * grid.height];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    if spawn_reachable {
        let sc = spawn_col as usize;
        let sr = spawn_row as usize;
        reachable[sr * grid.width + sc] = true;
        queue.push_back((sc, sr));

        while let Some((col, row)) = queue.pop_front() {
            for (dc, dr) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc >= 0 && (nc as usize) < grid.width && nr >= 0 && (nr as usize) < grid.height {
                    let ncu = nc as usize;
                    let nru = nr as usize;
                    let idx = nru * grid.width + ncu;
                    if !inflated[idx] && !reachable[idx] {
                        reachable[idx] = true;
                        queue.push_back((ncu, nru));
                    }
                }
            }
        }
    }

    let mut pose_results: Vec<PoseReachability> = Vec::new();

    for pose in named_poses {
        let (col, row) = grid.world_to_grid(pose.x, pose.y);
        let kind_str = format!("{:?}", pose.kind).to_lowercase();

        if pose.kind == PoseKind::Agvtag {
            pose_results.push(PoseReachability {
                name: pose.name.clone(),
                kind: kind_str,
                x: pose.x,
                y: pose.y,
                reachable: true,
                failure_class: None,
                failure_detail: None,
                suggested_fix: None,
            });
            continue;
        }

        if col < 0 || (col as usize) >= grid.width || row < 0 || (row as usize) >= grid.height {
            pose_results.push(PoseReachability {
                name: pose.name.clone(),
                kind: kind_str,
                x: pose.x,
                y: pose.y,
                reachable: false,
                failure_class: Some(ReachabilityFailure::OutOfBounds),
                failure_detail: Some("Pose coordinates are outside the map boundaries".into()),
                suggested_fix: None,
            });
            continue;
        }

        let cu = col as usize;
        let ru = row as usize;

        if grid.is_occupied(cu, ru) {
            pose_results.push(PoseReachability {
                name: pose.name.clone(),
                kind: kind_str,
                x: pose.x,
                y: pose.y,
                reachable: false,
                failure_class: Some(ReachabilityFailure::WaypointInsideWall),
                failure_detail: Some("Waypoint is inside a wall — check SLAM annotation".into()),
                suggested_fix: None,
            });
            continue;
        }

        if inflated[ru * grid.width + cu] {
            pose_results.push(PoseReachability {
                name: pose.name.clone(),
                kind: kind_str,
                x: pose.x,
                y: pose.y,
                reachable: false,
                failure_class: Some(ReachabilityFailure::WaypointTooTight),
                failure_detail: Some(format!(
                    "Passage too narrow for robot ({}m body) — widen doorway or reduce inflation",
                    robot_radius
                )),
                suggested_fix: None,
            });
            continue;
        }

        if !spawn_reachable {
            pose_results.push(PoseReachability {
                name: pose.name.clone(),
                kind: kind_str,
                x: pose.x,
                y: pose.y,
                reachable: false,
                failure_class: Some(ReachabilityFailure::SpawnUnreachable),
                failure_detail: Some(
                    "Spawn position is blocked — all waypoints are unreachable".into(),
                ),
                suggested_fix: None,
            });
            continue;
        }

        if !reachable[ru * grid.width + cu] {
            pose_results.push(PoseReachability {
                name: pose.name.clone(),
                kind: kind_str,
                x: pose.x,
                y: pose.y,
                reachable: false,
                failure_class: Some(ReachabilityFailure::WaypointBehindWall),
                failure_detail: Some(
                    "No navigable path from spawn to this waypoint — check waypoint placement"
                        .into(),
                ),
                suggested_fix: None,
            });
            continue;
        }

        pose_results.push(PoseReachability {
            name: pose.name.clone(),
            kind: kind_str,
            x: pose.x,
            y: pose.y,
            reachable: true,
            failure_class: None,
            failure_detail: None,
            suggested_fix: None,
        });
    }

    for (idx, pose) in named_poses.iter().enumerate() {
        if pose_results[idx].reachable {
            continue;
        }
        pose_results[idx].suggested_fix = compute_suggested_fix(
            pose, &reachable, grid, robot_radius, furniture,
        );
    }

    let (tour_feasible, tour_infeasible_goals) = if goals.is_empty() {
        (None, None)
    } else {
        let pose_map: HashMap<&str, &PoseReachability> = pose_results
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        let mut infeasible: Vec<InfeasibleGoal> = Vec::new();
        for (idx, goal) in goals.iter().enumerate() {
            if let Some(pr) = pose_map.get(goal.ref_name.as_str()) {
                if !pr.reachable {
                    infeasible.push(InfeasibleGoal {
                        index: idx,
                        ref_name: goal.ref_name.clone(),
                        failure_class: pr.failure_class.clone().unwrap_or(
                            ReachabilityFailure::WaypointBehindWall,
                        ),
                    });
                }
            } else {
                infeasible.push(InfeasibleGoal {
                    index: idx,
                    ref_name: goal.ref_name.clone(),
                    failure_class: ReachabilityFailure::OutOfBounds,
                });
            }
        }

        if infeasible.is_empty() {
            (Some(true), None)
        } else {
            (Some(false), Some(infeasible))
        }
    };

    let total_cells = grid.width * grid.height;
    let reachable_count = reachable.iter().filter(|&&r| r).count();
    let reachable_area_pct = if total_cells > 0 {
        (reachable_count as f64 / total_cells as f64) * 100.0
    } else {
        0.0
    };

    ReachabilityReport {
        spawn_reachable,
        robot_radius_m: robot_radius,
        reachable_area_pct,
        poses: pose_results,
        tour_feasible,
        tour_infeasible_goals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::human::AvoidanceConfig;
    use crate::named_pose::PoseKind;

    fn make_grid(width: usize, height: usize, resolution: f64) -> OccupancyGrid {
        OccupancyGrid {
            cells: vec![false; width * height],
            width,
            height,
            metadata: MapMetadata {
                resolution,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: PathBuf::new(),
            },
        }
    }

    #[test]
    fn test_robot_spawn_in_free_space_passes() {
        let grid = make_grid(100, 100, 0.05);
        let report = validate_scenario_state(
            &grid,
            Some((2.5, 2.5, 0.0)),
            0.35,
            None,
            None,
            &[],
            &HashMap::new(),
            &[],
        );
        assert!(report.valid);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_robot_spawn_in_wall_fails() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(2.5, 2.5);
        for dr in -2i32..=2 {
            for dc in -2i32..=2 {
                let r = (row + dr) as usize;
                let c = (col + dc) as usize;
                if r < grid.height && c < grid.width {
                    grid.cells[r * grid.width + c] = true;
                }
            }
        }
        let report = validate_scenario_state(
            &grid,
            Some((2.5, 2.5, 0.0)),
            0.35,
            None,
            None,
            &[],
            &HashMap::new(),
            &[],
        );
        assert!(!report.valid);
        assert!(report.issues.iter().any(|i| i.check == "robot_spawn"));
    }

    #[test]
    fn test_goal_outside_map_fails() {
        let grid = make_grid(100, 100, 0.05);
        let report = validate_scenario_state(
            &grid,
            None,
            0.35,
            None,
            Some((10.0, 10.0)),
            &[],
            &HashMap::new(),
            &[],
        );
        assert!(!report.valid);
        assert!(report.issues.iter().any(|i| i.check == "goal_position"));
    }

    #[test]
    fn test_goal_in_wall_fails() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(2.5, 2.5);
        grid.cells[row as usize * grid.width + col as usize] = true;
        let report = validate_scenario_state(
            &grid,
            None,
            0.35,
            None,
            Some((2.5, 2.5)),
            &[],
            &HashMap::new(),
            &[],
        );
        assert!(!report.valid);
        assert!(report.issues.iter().any(|i| i.check == "goal_position"));
    }

    #[test]
    fn test_goal_in_free_space_passes() {
        let grid = make_grid(100, 100, 0.05);
        let report = validate_scenario_state(
            &grid,
            None,
            0.35,
            None,
            Some((2.5, 2.5)),
            &[],
            &HashMap::new(),
            &[],
        );
        assert!(report.valid);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_human_waypoint_in_wall_errors() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(2.5, 2.5);
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                let r = (row + dr) as usize;
                let c = (col + dc) as usize;
                if r < grid.height && c < grid.width {
                    grid.cells[r * grid.width + c] = true;
                }
            }
        }
        let actor = HumanActor::new(
            "walker".into(),
            vec![(1.0, 1.0), (2.5, 2.5)],
            0.3,
            1.0,
            false,
            0.0,
            0.0,
            true,
            AvoidanceConfig::default(),
        );
        let report = validate_scenario_state(
            &grid,
            None,
            0.35,
            None,
            None,
            &[],
            &HashMap::new(),
            &[actor],
        );
        assert!(!report.valid);
        assert!(!report.issues.is_empty());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.check == "human_path_wall" && i.severity == Severity::Error)
        );
    }

    #[test]
    fn test_no_checks_when_nothing_provided() {
        let grid = make_grid(100, 100, 0.05);
        let report =
            validate_scenario_state(&grid, None, 0.35, None, None, &[], &HashMap::new(), &[]);
        assert!(report.valid);
        assert!(report.issues.is_empty());
    }

    fn make_pose(name: &str, kind: PoseKind, x: f64, y: f64) -> NamedPose {
        NamedPose {
            name: name.into(),
            kind,
            x,
            y,
            theta: 0.0,
        }
    }

    fn make_goal_ref(name: &str) -> ScenarioGoalRef {
        ScenarioGoalRef {
            ref_name: name.into(),
            tolerance: 0.35,
            timeout: 60.0,
            yaw_tolerance: 0.5,
        }
    }

    #[test]
    fn reachability_all_reachable_on_clean_grid() {
        let grid = make_grid(100, 100, 0.05);
        let poses = vec![
            make_pose("CHARGER", PoseKind::Charger, 0.5, 0.5),
            make_pose("KITCHEN", PoseKind::Production, 1.0, 1.0),
            make_pose("T1", PoseKind::Delivery, 2.0, 2.0),
        ];
        let report = validate_reachability(&grid, 0.22, (0.5, 0.5, 0.0), &poses, &[], &[]);

        assert!(report.spawn_reachable, "spawn should be reachable on clean grid");
        assert!(report.poses.iter().all(|p| p.reachable),
            "all poses should be reachable: {:?}", report.poses);
        assert!(report.reachable_area_pct > 99.0,
            "most of grid should be reachable: got {:.1}%", report.reachable_area_pct);
    }

    #[test]
    fn reachability_spawn_on_wall() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(2.5, 2.5);
        grid.cells[row as usize * grid.width + col as usize] = true;

        let report = validate_reachability(
            &grid,
            0.22,
            (2.5, 2.5, 0.0),
            &[make_pose("T1", PoseKind::Delivery, 1.0, 1.0)],
            &[],
            &[],
        );

        assert!(!report.spawn_reachable, "spawn on wall should be unreachable");
        assert!(!report.poses[0].reachable, "poses should be unreachable when spawn is blocked");
        assert_eq!(
            report.poses[0].failure_class,
            Some(ReachabilityFailure::SpawnUnreachable)
        );
        assert!(report.reachable_area_pct < 1.0, "no cells should be reachable");
    }

    #[test]
    fn reachability_spawn_in_inflation_zone() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(2.5, 2.5);
        grid.cells[row as usize * grid.width + col as usize] = true;
        let spawn_x = 2.6;
        let spawn_y = 2.5;

        let report = validate_reachability(
            &grid,
            0.22,
            (spawn_x, spawn_y, 0.0),
            &[],
            &[],
            &[],
        );

        assert!(!report.spawn_reachable, "spawn in inflation zone should be unreachable");
    }

    #[test]
    fn reachability_waypoint_inside_wall() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(1.0, 1.0);
        grid.cells[row as usize * grid.width + col as usize] = true;

        let report = validate_reachability(
            &grid,
            0.22,
            (0.5, 0.5, 0.0),
            &[make_pose("T1", PoseKind::Delivery, 1.0, 1.0)],
            &[],
            &[],
        );

        assert!(report.spawn_reachable);
        assert!(!report.poses[0].reachable);
        assert_eq!(
            report.poses[0].failure_class,
            Some(ReachabilityFailure::WaypointInsideWall)
        );
        assert!(report.poses[0].failure_detail.as_ref().unwrap().contains("wall"));
    }

    #[test]
    fn reachability_waypoint_too_tight() {
        let mut grid = make_grid(100, 100, 0.05);
        for col in 0..100 {
            grid.cells[30 * grid.width + col] = true;
        }

        let wp_x = 2.5;
        let wp_y = 1.55;

        let report = validate_reachability(
            &grid,
            0.22,
            (2.5, 0.5, 0.0),
            &[make_pose("T1", PoseKind::Delivery, wp_x, wp_y)],
            &[],
            &[],
        );

        assert!(report.spawn_reachable, "spawn should be reachable");
        assert!(!report.poses[0].reachable, "pose in inflation zone should be unreachable");
        assert_eq!(
            report.poses[0].failure_class,
            Some(ReachabilityFailure::WaypointTooTight)
        );
        assert!(report.poses[0].failure_detail.as_ref().unwrap().contains("narrow"));
    }

    #[test]
    fn reachability_waypoint_behind_wall() {
        let mut grid = make_grid(100, 100, 0.05);
        for col in 0..100 {
            if col < 49 || col > 50 {
                grid.cells[50 * grid.width + col] = true;
            }
        }

        let report = validate_reachability(
            &grid,
            0.22,
            (2.5, 1.0, 0.0),
            &[make_pose("T1", PoseKind::Delivery, 2.5, 4.0)],
            &[],
            &[],
        );

        assert!(report.spawn_reachable);
        let pose = &report.poses[0];
        assert!(!pose.reachable, "waypoint should be unreachable: failure = {:?}", pose.failure_class);
        assert!(
            pose.failure_class == Some(ReachabilityFailure::WaypointBehindWall) ||
            pose.failure_class == Some(ReachabilityFailure::WaypointTooTight),
            "expected behind-wall or too-tight, got {:?}", pose.failure_class
        );
    }

    #[test]
    fn reachability_waypoint_behind_wall_separate_component() {
        let mut grid = make_grid(100, 100, 0.05);
        for row in 60..72 {
            for col in 60..72 {
                if row == 60 || row == 71 || col == 60 || col == 71 {
                    grid.cells[row * grid.width + col] = true;
                }
            }
        }
        let wp_col = 65;
        let wp_row = 65;
        let (wp_x, wp_y) = grid.grid_to_world(wp_col, wp_row);

        let report = validate_reachability(
            &grid,
            0.22,
            (1.0, 1.0, 0.0),
            &[make_pose("SEALED", PoseKind::Delivery, wp_x, wp_y)],
            &[],
            &[],
        );

        assert!(report.spawn_reachable);
        let pose = &report.poses[0];
        assert!(!pose.reachable);
        assert!(
            pose.failure_class == Some(ReachabilityFailure::WaypointBehindWall) ||
            pose.failure_class == Some(ReachabilityFailure::WaypointTooTight),
            "expected behind-wall or too-tight, got {:?}", pose.failure_class
        );
    }

    #[test]
    fn reachability_out_of_bounds() {
        let grid = make_grid(100, 100, 0.05);
        let report = validate_reachability(
            &grid,
            0.22,
            (0.5, 0.5, 0.0),
            &[make_pose("OUT", PoseKind::Recycle, 10.0, 10.0)],
            &[],
            &[],
        );

        assert!(report.spawn_reachable);
        assert!(!report.poses[0].reachable);
        assert_eq!(
            report.poses[0].failure_class,
            Some(ReachabilityFailure::OutOfBounds)
        );
        assert!(report.poses[0].failure_detail.as_ref().unwrap().contains("boundaries"));
    }

    #[test]
    fn reachability_tour_feasible_all_reachable() {
        let grid = make_grid(100, 100, 0.05);
        let poses = vec![
            make_pose("A", PoseKind::Charger, 0.5, 0.5),
            make_pose("B", PoseKind::Delivery, 1.0, 1.0),
        ];
        let goals = vec![
            make_goal_ref("A"),
            make_goal_ref("B"),
        ];

        let report = validate_reachability(&grid, 0.22, (0.5, 0.5, 0.0), &poses, &goals, &[]);

        assert_eq!(report.tour_feasible, Some(true));
        assert!(report.tour_infeasible_goals.is_none());
    }

    #[test]
    fn reachability_tour_infeasible_unreachable_pose() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(4.0, 4.0);
        grid.cells[row as usize * grid.width + col as usize] = true;

        let poses = vec![
            make_pose("GOOD", PoseKind::Charger, 0.5, 0.5),
            make_pose("BAD", PoseKind::Delivery, 4.0, 4.0),
        ];
        let goals = vec![
            make_goal_ref("GOOD"),
            make_goal_ref("BAD"),
        ];

        let report = validate_reachability(&grid, 0.22, (0.5, 0.5, 0.0), &poses, &goals, &[]);

        assert_eq!(report.tour_feasible, Some(false));
        let infeasible = report.tour_infeasible_goals.unwrap();
        assert_eq!(infeasible.len(), 1);
        assert_eq!(infeasible[0].index, 1);
        assert_eq!(infeasible[0].ref_name, "BAD");
        assert_eq!(infeasible[0].failure_class, ReachabilityFailure::WaypointInsideWall);
    }

    #[test]
    fn reachability_empty_goals_tour_feasible_is_none() {
        let grid = make_grid(100, 100, 0.05);
        let report = validate_reachability(
            &grid,
            0.22,
            (0.5, 0.5, 0.0),
            &[make_pose("A", PoseKind::Charger, 1.0, 1.0)],
            &[],
            &[],
        );

        assert_eq!(report.tour_feasible, None);
        assert!(report.tour_infeasible_goals.is_none());
    }

    #[test]
    fn reachability_area_percentage_calculated_correctly() {
        let grid = make_grid(100, 100, 0.05);
        let report = validate_reachability(&grid, 0.22, (2.5, 2.5, 0.0), &[], &[], &[]);
        assert!((report.reachable_area_pct - 100.0).abs() < 1.0,
            "expected ~100%, got {:.1}%", report.reachable_area_pct);

        let mut half_grid = make_grid(100, 100, 0.05);
        for row in 50..100 {
            for col in 0..100 {
                half_grid.cells[row * half_grid.width + col] = true;
            }
        }
        let report_half = validate_reachability(
            &half_grid, 0.22, (2.5, 1.0, 0.0), &[], &[], &[]
        );
        assert!(report_half.reachable_area_pct < 50.0,
            "reachable area should be under 50% with half blocked: got {:.1}%",
            report_half.reachable_area_pct);
        assert!(report_half.reachable_area_pct > 0.0,
            "some area should still be reachable: got {:.1}%",
            report_half.reachable_area_pct);
    }

    #[test]
    fn reachability_multiple_failures_mixed() {
        let mut grid = make_grid(100, 100, 0.05);
        let (c1, r1) = grid.world_to_grid(1.0, 1.0);
        grid.cells[r1 as usize * grid.width + c1 as usize] = true;

        let poses = vec![
            make_pose("OK", PoseKind::Charger, 2.0, 2.0),
            make_pose("WALL", PoseKind::Delivery, 1.0, 1.0),
            make_pose("OOB", PoseKind::Recycle, 99.0, 99.0),
        ];

        let report = validate_reachability(&grid, 0.22, (2.5, 2.5, 0.0), &poses, &[], &[]);

        assert!(report.spawn_reachable);
        assert!(report.poses[0].reachable, "OK should be reachable");
        assert_eq!(report.poses[1].failure_class, Some(ReachabilityFailure::WaypointInsideWall));
        assert_eq!(report.poses[2].failure_class, Some(ReachabilityFailure::OutOfBounds));
    }

    #[test]
    fn reachability_serializes_to_json() {
        let grid = make_grid(50, 50, 0.05);
        let poses = vec![make_pose("A", PoseKind::Charger, 0.5, 0.5)];
        let report = validate_reachability(&grid, 0.22, (0.5, 0.5, 0.0), &poses, &[], &[]);

        let json = serde_json::to_value(&report).unwrap();
        assert!(json["spawn_reachable"].is_boolean());
        assert!(json["robot_radius_m"].is_number());
        assert!(json["reachable_area_pct"].is_number());
        assert!(json["poses"].is_array());
        assert!(json["poses"][0]["reachable"].is_boolean());
        assert!(json["poses"][0]["name"].is_string());
        assert!(json["tour_feasible"].is_null());
        assert!(json.get("tour_infeasible_goals").is_none());
    }

    #[test]
    fn reachability_tour_goal_ref_not_in_poses() {
        let grid = make_grid(100, 100, 0.05);
        let poses = vec![make_pose("A", PoseKind::Charger, 0.5, 0.5)];
        let goals = vec![ScenarioGoalRef {
            ref_name: "NONEXISTENT".into(),
            tolerance: 0.35,
            timeout: 60.0,
            yaw_tolerance: 0.5,
        }];

        let report = validate_reachability(&grid, 0.22, (0.5, 0.5, 0.0), &poses, &goals, &[]);

        assert_eq!(report.tour_feasible, Some(false));
        let inf = report.tour_infeasible_goals.unwrap();
        assert_eq!(inf.len(), 1);
        assert_eq!(inf[0].ref_name, "NONEXISTENT");
        assert_eq!(inf[0].failure_class, ReachabilityFailure::OutOfBounds);
    }

    #[test]
    fn reachability_spawn_blocked_gives_spawn_unreachable() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(2.5, 2.5);
        for dr in -3i32..=3 {
            for dc in -3i32..=3 {
                let r = (row + dr) as usize;
                let c = (col + dc) as usize;
                if r < grid.height && c < grid.width {
                    grid.cells[r * grid.width + c] = true;
                }
            }
        }
        let poses = vec![make_pose("T1", PoseKind::Delivery, 0.5, 0.5)];

        let report = validate_reachability(&grid, 0.22, (2.5, 2.5, 0.0), &poses, &[], &[]);

        assert!(!report.spawn_reachable);
        assert_eq!(report.poses.len(), 1);
        assert_eq!(
            report.poses[0].failure_class,
            Some(ReachabilityFailure::SpawnUnreachable)
        );
    }

    fn make_furniture(label: &str, x: f64, y: f64, width: f64, height: f64) -> ScenarioFurniture {
        ScenarioFurniture {
            type_id: "table4".into(),
            pose: vec![x, y, 0.0],
            label: label.into(),
            width,
            height,
        }
    }

    #[test]
    fn suggested_fix_delivery_reachable_side() {
        let mut grid = make_grid(100, 100, 0.05);
        for row in 60..=65 {
            for col in 0..100 {
                grid.cells[row * grid.width + col] = true;
            }
        }
        let furniture = vec![make_furniture("T8", 2.5, 2.0, 0.4, 0.4)];
        let poses = vec![make_pose("T8", PoseKind::Delivery, 2.5, 4.0)];

        let report = validate_reachability(
            &grid, 0.22, (2.5, 1.0, 0.0), &poses, &[], &furniture,
        );

        assert!(report.spawn_reachable, "spawn should be reachable");
        let pose = &report.poses[0];
        assert!(!pose.reachable, "T8 pose above wall should be unreachable");
        let fix = pose.suggested_fix.as_ref()
            .expect("should have a suggested fix from table sides");
        assert_eq!(fix.side, "south", "south side should be picked first (priority)");
        assert!((fix.theta - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-6,
            "theta should face north toward table: got {}", fix.theta);
        assert!(fix.method.contains("T8"), "method should reference table name");
    }

    #[test]
    fn suggested_fix_no_furniture_match() {
        let mut grid = make_grid(100, 100, 0.05);
        let (col, row) = grid.world_to_grid(3.0, 3.0);
        grid.cells[row as usize * grid.width + col as usize] = true;

        let furniture = vec![make_furniture("OTHER_TABLE", 1.0, 1.0, 0.5, 0.5)];
        let poses = vec![make_pose("T99", PoseKind::Delivery, 3.0, 3.0)];

        let report = validate_reachability(
            &grid, 0.22, (0.5, 0.5, 0.0), &poses, &[], &furniture,
        );

        assert!(!report.poses[0].reachable);
        assert!(report.poses[0].suggested_fix.is_none(),
            "no matching furniture → suggested_fix should be None");
    }

    #[test]
    fn suggested_fix_non_delivery_nearest_reachable() {
        let mut grid = make_grid(100, 100, 0.05);
        for col in 0..100 {
            grid.cells[50 * grid.width + col] = true;
        }
        let poses = vec![NamedPose {
            name: "CHARGER".into(),
            kind: PoseKind::Charger,
            x: 2.5,
            y: 4.0,
            theta: 1.57,
        }];

        let report = validate_reachability(
            &grid, 0.22, (2.5, 1.0, 0.0), &poses, &[], &[],
        );

        let pose = &report.poses[0];
        assert!(!pose.reachable, "charger above wall should be unreachable");
        let fix = pose.suggested_fix.as_ref()
            .expect("non-delivery unreachable pose should get nearest-reachable fix");
        assert_eq!(fix.side, "nearest");
        assert_eq!(fix.method, "Nearest reachable point");
        assert!((fix.theta - 1.57).abs() < 1e-6,
            "theta should be preserved from original pose: got {}", fix.theta);
        assert!(fix.y < 2.5,
            "nearest reachable point should be below the wall: got y={}", fix.y);
    }
}
