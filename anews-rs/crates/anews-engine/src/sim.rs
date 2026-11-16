use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::diff_drive::DiffDrive;
use crate::edit_history::{EditAction, EditHistory, GridPatch};
use crate::evaluator::SimEvaluator;
use crate::event_stream::EventBus;
use crate::events::SimEvent;
use crate::furniture::{
    FurnitureType, PlacedFurniture, load_furniture_types, resolve_furniture_to_obstacles,
};
use crate::human::{AvoidanceConfig, HumanActor, actors_to_obstacles};
use crate::map::{self, OccupancyGrid};
use crate::raycaster::{self, ResolvedObstacle};
use crate::robot_config::{self, RobotConfig};
use crate::scenario::{self, Scenario};
use crate::sensor::{self, SensorSuite};

const OVERLAY_SEND_TICKS: u32 = 10;

#[derive(Debug, Default, Clone)]
pub struct CancellationRegistry {
    cancelled: Arc<Mutex<HashSet<String>>>,
}

impl CancellationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self, request_id: &str) -> bool {
        self.cancelled.lock().unwrap().contains(request_id)
    }

    pub fn cancel(&self, request_id: &str) {
        self.cancelled.lock().unwrap().insert(request_id.to_string());
    }

    pub fn clear(&self, request_id: &str) {
        self.cancelled.lock().unwrap().remove(request_id);
    }
}

pub struct SimConfig {
    pub tick_dt: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { tick_dt: 0.05 }
    }
}

pub struct ScanResult {
    pub ranges: Vec<f64>,
    pub angle_min: f64,
    pub angle_max: f64,
    pub angle_increment: f64,
    pub range_min: f64,
    pub range_max: f64,
}

pub struct Simulation {
    pub grid: OccupancyGrid,
    pub drive: DiffDrive,
    pub sensor_suite: SensorSuite,
    pub furniture_types: HashMap<String, FurnitureType>,
    pub placed_furniture: Vec<PlacedFurniture>,
    pub human_actors: Vec<HumanActor>,
    pub robot_config: RobotConfig,
    pub evaluator: Option<SimEvaluator>,
    pub config: SimConfig,

    pub playing: bool,
    pub speed: f64,
    pub sim_time: f64,
    pub cmd_linear: f64,
    pub cmd_angular: f64,
    pub actual_linear: f64,
    pub actual_angular: f64,
    pub goal: Option<(f64, f64)>,
    pub goal_theta: Option<f64>,
    pub goal_epoch: u64,

    pub scan_results: HashMap<String, ScanResult>,
    pub scenario: Option<Scenario>,

    pub costmap_global: Option<serde_json::Value>,
    pub costmap_local: Option<serde_json::Value>,
    pub nav_path_global: Option<serde_json::Value>,
    pub nav_path_local: Option<serde_json::Value>,
    pub nav_trajectory: Option<serde_json::Value>,
    pub nav_events: Vec<serde_json::Value>,
    costmap_dirty: bool,
    nav_paths_dirty: bool,
    nav_trajectory_dirty: bool,
    overlay_ticks_since_send: u32,

    pub edit_history: EditHistory,
    pub grid_dirty: bool,
    original_grid: Vec<bool>,

    furniture_obstacle_cache: HashMap<String, Vec<ResolvedObstacle>>,
    obstacle_cache: HashMap<String, Vec<ResolvedObstacle>>,

    config_dir: std::path::PathBuf,
    map_yaml_path: std::path::PathBuf,

    pub event_logger: Option<crate::event_logger::EventLogger>,

    pub event_bus: EventBus,

    pub stuck_detector: crate::detectors::StuckDetector,
    pub human_approach_detector: crate::detectors::HumanApproachDetector,
    pub path_replanned_detector: crate::detectors::PathReplannedDetector,

    pub cancellation: CancellationRegistry,
}

fn furniture_from_scenario(items: &[scenario::ScenarioFurniture]) -> Vec<PlacedFurniture> {
    items
        .iter()
        .enumerate()
        .map(|(i, f)| PlacedFurniture {
            id: format!("f{i}"),
            type_id: f.type_id.clone(),
            x: f.pose[0],
            y: f.pose[1],
            rotation: *f.pose.get(2).unwrap_or(&0.0),
            width: Some(f.width),
            height: Some(f.height),
            label: f.label.clone(),
        })
        .collect()
}

fn ranges_to_json(ranges: &[f64]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = ranges
        .iter()
        .map(|&r| {
            if r.is_finite() {
                serde_json::json!(r)
            } else {
                serde_json::Value::Null
            }
        })
        .collect();
    serde_json::Value::Array(arr)
}

fn rotate_by_quat(v: [f64; 3], q: [f64; 4]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let [vx, vy, vz] = v;
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
}

fn frustum_floor_corners(
    cfg: &sensor::SensorConfig,
    robot_x: f64,
    robot_y: f64,
    robot_theta: f64,
) -> Option<[(f64, f64); 4]> {
    let pose = cfg.mount_pose.as_ref()?;
    let pitch = pose.pitch();
    const PITCH_THRESHOLD: f64 = 0.0873;
    if pitch.abs() < PITCH_THRESHOLD {
        return None;
    }
    let sensor_z = pose.height();
    if sensor_z <= 0.0 {
        return None;
    }
    let half_h = (cfg.fov_degrees.to_radians()) * 0.5;
    let half_v = (cfg.vfov_degrees.unwrap_or(cfg.fov_degrees).to_radians()) * 0.5;
    let cos_r = robot_theta.cos();
    let sin_r = robot_theta.sin();
    let sensor_wx = robot_x + pose.xyz[0] * cos_r - pose.xyz[1] * sin_r;
    let sensor_wy = robot_y + pose.xyz[0] * sin_r + pose.xyz[1] * cos_r;
    let corner_angles = [
        (half_h, half_v),
        (-half_h, half_v),
        (-half_h, -half_v),
        (half_h, -half_v),
    ];
    let mut out = [(0.0, 0.0); 4];
    for (i, (th, tv)) in corner_angles.iter().enumerate() {
        let cv = tv.cos();
        let dir_body = [cv * th.cos(), cv * th.sin(), tv.sin()];
        let dir_robot = rotate_by_quat(dir_body, pose.quat_xyzw);
        let wx = dir_robot[0] * cos_r - dir_robot[1] * sin_r;
        let wy = dir_robot[0] * sin_r + dir_robot[1] * cos_r;
        let wz = dir_robot[2];
        let t = if wz < -1e-6 {
            (-sensor_z / wz).clamp(cfg.range_min, cfg.range_max)
        } else {
            cfg.range_max
        };
        out[i] = (sensor_wx + t * wx, sensor_wy + t * wy);
    }
    Some(out)
}

fn ray_segment_t(
    ox: f64,
    oy: f64,
    dx: f64,
    dy: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
) -> Option<f64> {
    let sx = bx - ax;
    let sy = by - ay;
    let denom = dx * (-sy) - dy * (-sx);
    if denom.abs() < 1e-12 {
        return None;
    }
    let rx = ax - ox;
    let ry = ay - oy;
    let t = (rx * (-sy) - ry * (-sx)) / denom;
    let s = (dx * ry - dy * rx) / denom;
    if t > 1e-9 && (-1e-9..=1.0 + 1e-9).contains(&s) {
        Some(t)
    } else {
        None
    }
}

fn clip_ray_to_convex_poly(
    ox: f64,
    oy: f64,
    dx: f64,
    dy: f64,
    poly: &[(f64, f64); 4],
) -> Option<(f64, f64)> {
    let mut hits: Vec<f64> = Vec::with_capacity(2);
    for i in 0..4 {
        let (ax, ay) = poly[i];
        let (bx, by) = poly[(i + 1) % 4];
        if let Some(t) = ray_segment_t(ox, oy, dx, dy, ax, ay, bx, by) {
            hits.push(t);
        }
    }
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if hits.is_empty() {
        return None;
    }
    let origin_inside = point_in_convex_poly_2d(ox, oy, poly);
    if origin_inside {
        Some((0.0, *hits.last().unwrap()))
    } else if hits.len() >= 2 {
        Some((hits[0], hits[hits.len() - 1]))
    } else {
        None
    }
}

fn point_in_convex_poly_2d(px: f64, py: f64, poly: &[(f64, f64); 4]) -> bool {
    let mut sign: f64 = 0.0;
    for i in 0..4 {
        let (ax, ay) = poly[i];
        let (bx, by) = poly[(i + 1) % 4];
        let cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        if cross.abs() < 1e-12 {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign != cross.signum() {
            return false;
        }
    }
    true
}

fn humans_from_scenario(humans: &[scenario::ScenarioHuman]) -> Vec<HumanActor> {
    humans
        .iter()
        .map(|h| {
            let path: Vec<(f64, f64)> = h.path.iter().map(|p| (p[0], p[1])).collect();
            HumanActor::new(
                h.name.clone(),
                path,
                h.radius,
                h.speed,
                h.loop_path,
                h.pause_at_waypoints,
                h.start_delay,
                true,
                AvoidanceConfig::default(),
            )
        })
        .collect()
}

impl Simulation {
    pub fn new(
        map_path: &Path,
        scenario_path: Option<&Path>,
        config_dir: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let grid = map::load_map(map_path)?;
        let sensor_suite = sensor::load_sensor_config("default", config_dir)?;
        let robot_config = robot_config::load_robot_config(config_dir);
        let furniture_types =
            load_furniture_types(&config_dir.join("furniture").join("builtin.yaml"))?;

        let original_grid = grid.cells.clone();
        let mut sim = Simulation {
            grid,
            drive: DiffDrive::new(0.0, 0.0, 0.0),
            sensor_suite,
            furniture_types,
            placed_furniture: Vec::new(),
            human_actors: Vec::new(),
            robot_config,
            evaluator: None,
            config: SimConfig::default(),
            playing: true,
            speed: 1.0,
            sim_time: 0.0,
            cmd_linear: 0.0,
            cmd_angular: 0.0,
            actual_linear: 0.0,
            actual_angular: 0.0,
            goal: None,
            goal_theta: None,
            goal_epoch: 0,
            scan_results: HashMap::new(),
            scenario: None,
            costmap_global: None,
            costmap_local: None,
            nav_path_global: None,
            nav_path_local: None,
            nav_trajectory: None,
            nav_events: Vec::new(),
            costmap_dirty: false,
            nav_paths_dirty: false,
            nav_trajectory_dirty: false,
            overlay_ticks_since_send: 0,
            edit_history: EditHistory::new(50),
            grid_dirty: false,
            original_grid,
            furniture_obstacle_cache: HashMap::new(),
            obstacle_cache: HashMap::new(),
            config_dir: config_dir.to_path_buf(),
            map_yaml_path: map_path.to_path_buf(),
            event_logger: None,
            event_bus: EventBus::new(),
            stuck_detector: Default::default(),
            human_approach_detector: Default::default(),
            path_replanned_detector: Default::default(),
            cancellation: CancellationRegistry::new(),
        };

        if let Some(sp) = scenario_path {
            sim.load_scenario_file(sp)?;
        }
        sim.rebuild_furniture_obstacles();
        sim.rebuild_obstacles();
        Ok(sim)
    }

    pub fn load_scenario_file(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let s = scenario::load_scenario(path)?;

        if !s.localization_map.is_empty() {
            let loc = &s.localization_map;
            let resolved = {
                let parent = path.parent().unwrap_or(std::path::Path::new("."));
                let rel = parent.join(loc);
                if rel.exists() {
                    Some(rel)
                } else {
                    let grandparent_rel = parent
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .join(loc);
                    if grandparent_rel.exists() {
                        Some(grandparent_rel)
                    } else {
                        let fname = std::path::Path::new(loc).file_name().unwrap_or_default();
                        let maps_dir = std::path::Path::new("/maps").join(fname);
                        if maps_dir.exists() {
                            Some(maps_dir)
                        } else {
                            None
                        }
                    }
                }
            };
            if let Some(new_map_path) = resolved {
                if new_map_path != self.map_yaml_path {
                    let new_grid = map::load_map(&new_map_path)?;
                    self.original_grid = new_grid.cells.clone();
                    self.grid = new_grid;
                    self.map_yaml_path = new_map_path;
                    self.edit_history = EditHistory::new(50);
                    self.grid_dirty = false;
                    self.costmap_global = None;
                    self.costmap_local = None;
                    self.nav_path_global = None;
                    self.nav_path_local = None;
                    self.nav_trajectory = None;
                    self.costmap_dirty = false;
                    self.nav_paths_dirty = false;
                    self.nav_trajectory_dirty = false;
                }
            }
        }

        self.drive
            .reset(s.robot_pose.0, s.robot_pose.1, s.robot_pose.2);
        self.cmd_linear = 0.0;
        self.cmd_angular = 0.0;
        self.actual_linear = 0.0;
        self.actual_angular = 0.0;

        self.placed_furniture = furniture_from_scenario(&s.furniture);
        self.human_actors = humans_from_scenario(&s.humans);

        if let Some(ref g) = s.goal {
            self.goal = Some((g.x, g.y));
            self.goal_theta = g.theta;
            self.goal_epoch += 1;
        } else {
            self.goal = None;
            self.goal_theta = None;
        }
        self.scenario = Some(s);
        self.rebuild_furniture_obstacles();
        self.rebuild_obstacles();

        if let Some(goal) = self.goal {
            let sc = self.scenario.as_ref();
            self.evaluator = Some(SimEvaluator::new(
                &self.grid,
                Some(goal),
                self.goal_theta,
                sc.and_then(|s| s.goal.as_ref())
                    .map(|g| g.yaw_tolerance)
                    .unwrap_or(0.5),
                self.robot_config.collision_radius,
                sc.and_then(|s| s.goal.as_ref())
                    .map(|g| g.timeout)
                    .unwrap_or(60.0),
                sc.and_then(|s| s.goal.as_ref())
                    .map(|g| g.tolerance)
                    .unwrap_or(0.3),
                0.05,
                self.get_all_obstacles(),
                Some(self.robot_config.footprint.clone()),
            ));
        }
        Ok(())
    }

    pub fn tick(&mut self) {
        self.tick_dt(self.config.tick_dt);
    }

    pub fn tick_dt(&mut self, dt: f64) {
        if !self.playing {
            return;
        }
        let effective_dt = dt * self.speed;

        let max_lin_accel = self.robot_config.max_linear_accel;
        let max_ang_accel = self.robot_config.max_angular_accel;

        let lin_diff = self.cmd_linear - self.actual_linear;
        let max_lin_delta = max_lin_accel * effective_dt;
        self.actual_linear += lin_diff.clamp(-max_lin_delta, max_lin_delta);

        let ang_diff = self.cmd_angular - self.actual_angular;
        let max_ang_delta = max_ang_accel * effective_dt;
        self.actual_angular += ang_diff.clamp(-max_ang_delta, max_ang_delta);

        self.drive
            .update(self.actual_linear, self.actual_angular, effective_dt);
        self.sim_time += effective_dt;

        let (rx, ry, _) = self.drive.pose();
        for i in 0..self.human_actors.len() {
            let mut avoid_list: Vec<(f64, f64, f64)> = vec![(rx, ry, 0.25)];
            for j in 0..self.human_actors.len() {
                if i != j {
                    let other = &self.human_actors[j];
                    if other.enabled && other.active && !other.finished {
                        avoid_list.push((other.x, other.y, other.radius));
                    }
                }
            }
            self.human_actors[i].update(effective_dt, &avoid_list);
        }

        self.rebuild_obstacles();
        self.raycast_sensors();

        let eval_obstacles = self.get_all_obstacles();
        let (rx, ry, rtheta) = self.drive.pose();
        if let Some(ref mut eval) = self.evaluator {
            if !eval.terminated {
                let prev_collision = eval.is_collided();
                let prev_timeout = eval.is_timed_out();
                let prev_oscillation = eval.is_oscillating();
                let prev_near_miss_count = eval.near_miss_count();
                let prev_max_overshoot = eval.max_overshoot();
                let prev_goal_reached = eval.is_goal_reached();

                eval.obstacles = eval_obstacles;
                eval.tick(
                    rx,
                    ry,
                    rtheta,
                    self.sim_time,
                    self.actual_linear,
                    self.actual_angular,
                );

                let current_tick = eval.tick_count();

                if !prev_collision && eval.is_collided() {
                    self.event_bus.emit(SimEvent::Collision {
                        obstacle_id: "unknown".to_string(),
                        tick: current_tick,
                    });
                }
                if !prev_timeout && eval.is_timed_out() {
                    self.event_bus.emit(SimEvent::Timeout { tick: current_tick });
                }
                if !prev_oscillation && eval.is_oscillating() {
                    self.event_bus.emit(SimEvent::Oscillation {
                        period_ticks: 0,
                        amplitude_m: 0.0,
                        tick: current_tick,
                    });
                }
                const OVERSHOOT_THRESHOLD: f64 = 0.3;
                if prev_max_overshoot < OVERSHOOT_THRESHOLD
                    && eval.max_overshoot() >= OVERSHOOT_THRESHOLD
                {
                    self.event_bus.emit(SimEvent::Overshoot {
                        overshoot_m: eval.max_overshoot(),
                        tick: current_tick,
                    });
                }
                if eval.near_miss_count() > prev_near_miss_count {
                    self.event_bus.emit(SimEvent::NearMiss {
                        obstacle_id: "unknown".to_string(),
                        distance_m: 0.0,
                        tick: current_tick,
                    });
                }
                if !prev_goal_reached && eval.is_goal_reached() {
                    let time_s = eval.goal_first_reached_time().unwrap_or(self.sim_time);
                    self.event_bus.emit(SimEvent::GoalReached {
                        time_s,
                        tick: current_tick,
                    });
                }

                let goal_reached = eval.is_goal_reached();
                self.stuck_detector.step(
                    self.config.tick_dt,
                    current_tick,
                    self.actual_linear,
                    self.actual_angular,
                    goal_reached,
                    &self.event_bus,
                );

                let humans_snapshot: Vec<(String, (f64, f64))> = self
                    .human_actors
                    .iter()
                    .map(|h| (h.name.clone(), (h.x, h.y)))
                    .collect();
                let (robot_x, robot_y, _) = self.drive.pose();
                self.human_approach_detector.step(
                    current_tick,
                    (robot_x, robot_y),
                    &humans_snapshot,
                    &self.event_bus,
                );

                let path_points: Option<Vec<(f64, f64)>> =
                    self.nav_path_global.as_ref().and_then(|v| {
                        v.as_array().map(|arr| {
                            arr.iter()
                                .filter_map(|pt| {
                                    let x = pt.get("x")?.as_f64()?;
                                    let y = pt.get("y")?.as_f64()?;
                                    Some((x, y))
                                })
                                .collect()
                        })
                    });
                self.path_replanned_detector.step(
                    current_tick,
                    path_points.as_deref(),
                    &self.event_bus,
                );
            }
        }

        let (rx, ry, rtheta) = self.drive.pose();
        let (collision, collision_wall, collision_furniture_ids, collision_human_names) =
            self.compute_collision_state();

        let humans_json: Vec<serde_json::Value> = self
            .human_actors
            .iter()
            .filter(|h| h.enabled && h.active)
            .map(|h| {
                serde_json::json!({
                    "name": h.name,
                    "x": (h.x * 10000.0).round() / 10000.0,
                    "y": (h.y * 10000.0).round() / 10000.0,
                    "theta": (h.theta * 10000.0).round() / 10000.0,
                })
            })
            .collect();

        let evaluator_summary = self.evaluator.as_ref().map(|eval| {
            let result = eval.result();
            serde_json::json!({
                "min_clearance": if result.min_clearance == f64::INFINITY { serde_json::Value::Null } else { serde_json::json!((result.min_clearance * 10000.0).round() / 10000.0) },
                "near_miss_count": result.near_miss_count,
                "failure_class": result.failure_class(),
            })
        });

        let goal_json = self.goal.map(|(gx, gy)| serde_json::json!([gx, gy]));

        let scan_ranges = self.scan_results.values().next().map(|scan| {
            scan.ranges
                .iter()
                .enumerate()
                .filter_map(|(i, &r)| if i % 10 == 0 { Some(r) } else { None })
                .collect::<Vec<_>>()
        });

        if let Some(ref mut logger) = self.event_logger {
            let tick_data = serde_json::json!({
                "robot": [rx, ry, rtheta],
                "cmd_linear": self.cmd_linear,
                "cmd_angular": self.cmd_angular,
                "actual_linear": self.actual_linear,
                "actual_angular": self.actual_angular,
                "humans": humans_json,
                "collision": collision,
                "collision_wall": collision_wall,
                "collision_furniture_ids": collision_furniture_ids,
                "collision_human_names": collision_human_names,
                "evaluator": evaluator_summary.unwrap_or(serde_json::Value::Null),
                "goal": goal_json.unwrap_or(serde_json::Value::Null),
                "scan_ranges": scan_ranges.as_ref().map(|r| ranges_to_json(r)).unwrap_or(serde_json::Value::Null),
            });

            logger.log(self.sim_time, "tick", tick_data);
        }
    }

    fn raycast_sensors(&mut self) {
        let (rx, ry, rtheta) = self.drive.pose();
        let cos_t = rtheta.cos();
        let sin_t = rtheta.sin();
        for (name, sensor_cfg) in &self.sensor_suite.sensors {
            let obstacles = self
                .obstacle_cache
                .get(&sensor_cfg.profile)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let [odx, ody, dyaw] = sensor_cfg.mount_offset;
            let sensor_x = rx + odx * cos_t - ody * sin_t;
            let sensor_y = ry + odx * sin_t + ody * cos_t;
            let sensor_theta = rtheta + dyaw;

            let frustum_poly = frustum_floor_corners(sensor_cfg, rx, ry, rtheta);
            let (sweep_min, sweep_max, sweep_inc) = if frustum_poly.is_some() {
                let n = sensor_cfg.num_rays.max(2);
                let span = std::f64::consts::TAU;
                let inc = span / (n - 1) as f64;
                (-std::f64::consts::PI, std::f64::consts::PI, inc)
            } else {
                (
                    sensor_cfg.angle_min,
                    sensor_cfg.angle_max,
                    sensor_cfg.angle_increment,
                )
            };

            let mut ranges = raycaster::raycast(
                &self.grid,
                sensor_x,
                sensor_y,
                sensor_theta,
                sweep_min,
                sweep_max,
                sweep_inc,
                sensor_cfg.range_max,
                sensor_cfg.range_min,
                obstacles,
            );
            if let Some(poly) = frustum_poly {
                let no_hit_eps = sensor_cfg.range_max - 1e-6;
                let n_rays = ranges.len();
                for (i, r) in ranges.iter_mut().enumerate() {
                    let frac = if n_rays > 1 {
                        i as f64 / (n_rays - 1) as f64
                    } else {
                        0.0
                    };
                    let angle = sensor_theta + sweep_min + frac * (sweep_max - sweep_min);
                    let dx = angle.cos();
                    let dy = angle.sin();
                    let hit = *r < no_hit_eps;
                    if !hit {
                        *r = f64::NAN;
                        continue;
                    }
                    match clip_ray_to_convex_poly(sensor_x, sensor_y, dx, dy, &poly) {
                        Some((t_in, t_out)) if *r >= t_in && *r <= t_out => {
                        }
                        _ => *r = f64::NAN,
                    }
                }
            }

            self.scan_results.insert(
                name.clone(),
                ScanResult {
                    ranges,
                    angle_min: sweep_min,
                    angle_max: sweep_max,
                    angle_increment: sweep_inc,
                    range_min: sensor_cfg.range_min,
                    range_max: sensor_cfg.range_max,
                },
            );
        }
    }

    pub fn tick_count(&self) -> u64 {
        (self.sim_time / self.config.tick_dt) as u64
    }

    pub fn build_init(&self) -> serde_json::Value {
        json!({
            "type": "init",
            "map": { "width": self.grid.width, "height": self.grid.height, "resolution": self.grid.metadata.resolution, "origin": self.grid.metadata.origin },
            "map_image_base64": self.grid.to_base64_png(),
            "sensors": self.sensor_suite.sensors.iter().map(|(name, s)| {
                (name.clone(), json!({ "name": s.name, "frame_id": s.frame_id, "topic": s.topic,
                    "fov_degrees": s.fov_degrees, "vfov_degrees": s.vfov_degrees,
                    "range_min": s.range_min, "range_max": s.range_max,
                    "num_rays": s.num_rays, "mount_offset": s.mount_offset,
                    "mount_pose": s.mount_pose.as_ref().map(|p| json!({"xyz": p.xyz, "quat_xyzw": p.quat_xyzw})),
                    "rate_hz": s.rate_hz, "profile": s.profile }))
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
            "furniture_types": self.furniture_types.iter().map(|(key, ft)| {
                (key.clone(), json!({ "name": ft.name, "visual_shape": ft.visual_shape, "default_width": ft.default_width, "default_height": ft.default_height }))
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
            "furniture": self.placed_furniture.iter().map(|pf| json!({
                "id": pf.id, "type_id": pf.type_id, "x": pf.x, "y": pf.y, "rotation": pf.rotation,
                "width": pf.width, "height": pf.height, "label": pf.label,
            })).collect::<Vec<_>>(),
            "humans": self.human_actors.iter().map(|a| json!({
                "name": a.name, "path": a.path.iter().map(|(x,y)| vec![*x,*y]).collect::<Vec<_>>(),
                "radius": a.radius, "speed": a.speed, "loop": a.loop_path, "start_delay": a.start_delay, "enabled": a.enabled,
            })).collect::<Vec<_>>(),
            "named_poses": self.scenario.as_ref()
                .map(|s| s.named_poses.iter().map(|p| json!({
                    "name": p.name,
                    "kind": serde_json::to_value(&p.kind).unwrap(),
                    "x": p.x, "y": p.y, "theta": p.theta,
                })).collect::<Vec<_>>())
                .unwrap_or_default(),
            "robot": { "footprint": self.robot_config.footprint.iter().map(|(x,y)| vec![*x,*y]).collect::<Vec<_>>(), "collision_radius": self.robot_config.collision_radius },
        })
    }

    pub fn build_state(&mut self) -> serde_json::Value {
        let (rx, ry, rtheta) = self.drive.pose();
        let mut state = json!({
            "type": "state", "t": (self.sim_time * 10000.0).round() / 10000.0,
            "playing": self.playing, "speed": self.speed,
            "robot": [(rx * 10000.0).round() / 10000.0, (ry * 10000.0).round() / 10000.0, (rtheta * 10000.0).round() / 10000.0],
            "cmd_vel": [(self.cmd_linear * 10000.0).round() / 10000.0, (self.cmd_angular * 10000.0).round() / 10000.0],
            "actual_vel": [(self.actual_linear * 10000.0).round() / 10000.0, (self.actual_angular * 10000.0).round() / 10000.0],
        });
        let m = state.as_object_mut().unwrap();
        for (name, scan) in &self.scan_results {
            m.insert(name.clone(), json!({ "ranges": ranges_to_json(&scan.ranges), "angle_min": scan.angle_min, "angle_max": scan.angle_max, "angle_increment": scan.angle_increment, "range_min": scan.range_min, "range_max": scan.range_max }));
        }
        m.insert(
            "humans".into(),
            self.human_actors
                .iter()
                .map(|a| a.to_state_dict())
                .collect(),
        );
        m.insert("furniture".into(), json!(self.placed_furniture.iter().map(|pf| json!({ "id": pf.id, "type_id": pf.type_id, "x": pf.x, "y": pf.y, "rotation": pf.rotation, "width": pf.width, "height": pf.height, "label": pf.label })).collect::<Vec<_>>()));
        m.insert(
            "goal".into(),
            match self.goal {
                Some((gx, gy)) => {
                    let mut v = vec![json!(gx), json!(gy)];
                    if let Some(gt) = self.goal_theta {
                        v.push(json!(gt));
                    }
                    json!(v)
                }
                None => serde_json::Value::Null,
            },
        );
        m.insert("goal_epoch".into(), json!(self.goal_epoch));
        if let Some(ref eval) = self.evaluator {
            m.insert(
                "evaluator_status".into(),
                if eval.terminated {
                    "terminated".into()
                } else {
                    "running".into()
                },
            );
            m.insert("evaluator_result".into(), eval.result().to_json_value());
        } else {
            m.insert("evaluator_status".into(), serde_json::Value::Null);
            m.insert("evaluator_result".into(), serde_json::Value::Null);
        }
        let (collision, collision_wall, collision_furniture_ids, collision_human_names) =
            self.compute_collision_state();
        m.insert("collision".into(), json!(collision));
        m.insert("collision_wall".into(), json!(collision_wall));
        m.insert(
            "collision_furniture_ids".into(),
            json!(collision_furniture_ids),
        );
        m.insert("collision_human_names".into(), json!(collision_human_names));
        self.overlay_ticks_since_send += 1;
        let send_overlays =
            (self.costmap_dirty || self.nav_paths_dirty || self.nav_trajectory_dirty)
                && self.overlay_ticks_since_send >= OVERLAY_SEND_TICKS;
        if send_overlays {
            if self.costmap_dirty {
                let mut costmap = serde_json::Map::new();
                if let Some(ref cg) = self.costmap_global {
                    costmap.insert("global".into(), cg.clone());
                }
                if let Some(ref cl) = self.costmap_local {
                    costmap.insert("local".into(), cl.clone());
                }
                if !costmap.is_empty() {
                    m.insert("costmap".into(), serde_json::Value::Object(costmap));
                }
            }
            if self.nav_paths_dirty {
                let mut nav_paths = serde_json::Map::new();
                if let Some(ref pg) = self.nav_path_global {
                    nav_paths.insert("global".into(), pg.clone());
                }
                if let Some(ref pl) = self.nav_path_local {
                    nav_paths.insert("local".into(), pl.clone());
                }
                if !nav_paths.is_empty() {
                    m.insert("nav_paths".into(), serde_json::Value::Object(nav_paths));
                }
            }
            if self.nav_trajectory_dirty {
                if let Some(ref tr) = self.nav_trajectory {
                    m.insert("nav_trajectory".into(), tr.clone());
                }
            }
        }
        if !self.nav_events.is_empty() {
            let drained = std::mem::take(&mut self.nav_events);
            m.insert("nav_events".into(), serde_json::Value::Array(drained));
        }
        if self.grid_dirty {
            m.insert(
                "map_image_base64".into(),
                serde_json::Value::String(self.grid.to_base64_png()),
            );
        }
        state
    }

    pub fn flush_dirty(&mut self) {
        self.grid_dirty = false;
        if self.overlay_ticks_since_send >= OVERLAY_SEND_TICKS {
            if self.costmap_dirty || self.nav_paths_dirty || self.nav_trajectory_dirty {
                self.overlay_ticks_since_send = 0;
            }
            self.costmap_dirty = false;
            self.nav_paths_dirty = false;
            self.nav_trajectory_dirty = false;
        }
    }

    pub fn handle_command(&mut self, cmd: &serde_json::Value) -> serde_json::Value {
        let action = cmd
            .get("action")
            .or_else(|| cmd.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let request_id = cmd.get("request_id").cloned();
        let mut result = match action {
            "set_robot_pose" => {
                self.drive.reset(
                    cmd["x"].as_f64().unwrap_or(0.0),
                    cmd["y"].as_f64().unwrap_or(0.0),
                    cmd["theta"].as_f64().unwrap_or(0.0),
                );
                self.cmd_linear = 0.0;
                self.cmd_angular = 0.0;
                self.actual_linear = 0.0;
                self.actual_angular = 0.0;
                self.update_scans();
                json!({"type": "response", "status": "ok"})
            }
            "set_cmd_vel" => {
                let max_lin = self.robot_config.max_linear_vel;
                let max_ang = self.robot_config.max_angular_vel;
                self.cmd_linear = cmd["linear"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .clamp(-max_lin, max_lin);
                self.cmd_angular = cmd["angular"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .clamp(-max_ang, max_ang);
                if let Some(ref mut logger) = self.event_logger {
                    logger.log(
                        self.sim_time,
                        "cmd_vel",
                        json!({"linear": self.cmd_linear, "angular": self.cmd_angular}),
                    );
                }
                json!({"type": "response", "status": "ok"})
            }
            "pause" => {
                self.playing = false;
                json!({"type": "response", "status": "ok"})
            }
            "play" => {
                self.playing = true;
                json!({"type": "response", "status": "ok"})
            }
            "set_speed" => {
                self.speed = cmd["speed"].as_f64().unwrap_or(1.0).clamp(0.1, 10.0);
                json!({"type": "response", "status": "ok"})
            }
            "set_goal" => {
                let x = cmd["x"].as_f64().unwrap_or(0.0);
                let y = cmd["y"].as_f64().unwrap_or(0.0);
                self.goal = Some((x, y));
                self.goal_theta = cmd.get("theta").and_then(|v| v.as_f64());
                self.goal_epoch += 1;
                let timeout = cmd.get("timeout").and_then(|v| v.as_f64()).unwrap_or(60.0);
                let tolerance = cmd.get("tolerance").and_then(|v| v.as_f64()).unwrap_or(0.3);
                let yaw_tolerance = cmd
                    .get("yaw_tolerance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                self.evaluator = Some(SimEvaluator::new(
                    &self.grid,
                    Some((x, y)),
                    self.goal_theta,
                    yaw_tolerance,
                    self.robot_config.collision_radius,
                    timeout,
                    tolerance,
                    0.05,
                    self.get_all_obstacles(),
                    Some(self.robot_config.footprint.clone()),
                ));
                if let Some(ref mut logger) = self.event_logger {
                    logger.log(self.sim_time, "set_goal", json!({"goal": [x, y]}));
                }
                json!({"type": "response", "status": "ok"})
            }
            "tick_n" => {
                let steps = cmd.get("steps").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                for _ in 0..steps {
                    self.tick();
                }
                json!({"type": "response", "status": "ok"})
            }
            "place_furniture" => {
                let id = format!("f{}", self.placed_furniture.len());
                self.placed_furniture.push(PlacedFurniture {
                    id: id.clone(),
                    type_id: cmd["type_id"].as_str().unwrap_or("").to_string(),
                    x: cmd["x"].as_f64().unwrap_or(0.0),
                    y: cmd["y"].as_f64().unwrap_or(0.0),
                    rotation: cmd.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    width: cmd.get("width").and_then(|v| v.as_f64()),
                    height: cmd.get("height").and_then(|v| v.as_f64()),
                    label: cmd
                        .get("label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                });
                self.rebuild_furniture_obstacles();
                self.rebuild_obstacles();
                self.update_scans();
                json!({"type": "response", "status": "ok", "id": id})
            }
            "remove_furniture" => {
                let id = cmd["id"].as_str().unwrap_or("");
                self.placed_furniture.retain(|pf| pf.id != id);
                self.rebuild_furniture_obstacles();
                self.rebuild_obstacles();
                self.update_scans();
                json!({"type": "response", "status": "ok"})
            }
            "move_furniture" => {
                let id = cmd["id"].as_str().unwrap_or("");
                if let Some(pf) = self.placed_furniture.iter_mut().find(|pf| pf.id == id) {
                    pf.x = cmd["x"].as_f64().unwrap_or(pf.x);
                    pf.y = cmd["y"].as_f64().unwrap_or(pf.y);
                    if let Some(r) = cmd.get("rotation").and_then(|v| v.as_f64()) {
                        pf.rotation = r;
                    }
                    if let Some(w) = cmd.get("width").and_then(|v| v.as_f64()) {
                        pf.width = Some(w);
                    }
                    if let Some(h) = cmd.get("height").and_then(|v| v.as_f64()) {
                        pf.height = Some(h);
                    }
                }
                self.rebuild_furniture_obstacles();
                self.rebuild_obstacles();
                json!({"type": "response", "status": "ok"})
            }
            "add_human" => {
                let name = cmd["name"].as_str().unwrap_or("human").to_string();
                let path: Vec<(f64, f64)> = cmd
                    .get("path")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|p| {
                                let a = p.as_array()?;
                                Some((a[0].as_f64()?, a[1].as_f64()?))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                self.human_actors.push(HumanActor::new(
                    name,
                    path,
                    cmd.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.3),
                    cmd.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    cmd.get("loop").and_then(|v| v.as_bool()).unwrap_or(false),
                    cmd.get("pause_at_waypoints")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    cmd.get("start_delay")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    true,
                    AvoidanceConfig::default(),
                ));
                self.rebuild_obstacles();
                json!({"type": "response", "status": "ok"})
            }
            "remove_human" => {
                let name = cmd["name"].as_str().unwrap_or("");
                self.human_actors.retain(|a| a.name != name);
                self.rebuild_obstacles();
                json!({"type": "response", "status": "ok"})
            }
            "snapshot" => self.build_state(),
            "update_human" => {
                let name = cmd["name"].as_str().unwrap_or("");
                let mut rebuild = false;
                if let Some(actor) = self.human_actors.iter_mut().find(|a| a.name == name) {
                    if let Some(speed) = cmd.get("speed").and_then(|v| v.as_f64()) {
                        actor.speed = speed;
                    }
                    if let Some(radius) = cmd.get("radius").and_then(|v| v.as_f64()) {
                        actor.radius = radius;
                        rebuild = true;
                    }
                    if let Some(pause) = cmd.get("pause_at_waypoints").and_then(|v| v.as_f64()) {
                        actor.pause_at_waypoints = pause;
                    }
                    if let Some(enabled) = cmd.get("enabled").and_then(|v| v.as_bool()) {
                        actor.enabled = enabled;
                        rebuild = true;
                    }
                }
                if rebuild {
                    self.rebuild_obstacles();
                }
                json!({"type": "response", "status": "ok"})
            }
            "update_human_path" => {
                let name = cmd["name"].as_str().unwrap_or("");
                let path: Vec<(f64, f64)> = cmd
                    .get("path")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|p| {
                                let a = p.as_array()?;
                                Some((a[0].as_f64()?, a[1].as_f64()?))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if let Some(actor) = self.human_actors.iter_mut().find(|a| a.name == name) {
                    actor.path = path.clone();
                    actor.path_idx = 0;
                    actor.path_t = 0.0;
                    actor.finished = false;
                    if let Some(&(x, y)) = actor.path.first() {
                        actor.x = x;
                        actor.y = y;
                    }
                }
                self.rebuild_obstacles();
                json!({"type": "response", "status": "ok"})
            }
            "set_human_pose" => {
                let name = cmd["name"].as_str().unwrap_or("");
                let x = cmd["x"].as_f64().unwrap_or(0.0);
                let y = cmd["y"].as_f64().unwrap_or(0.0);
                if let Some(actor) = self.human_actors.iter_mut().find(|a| a.name == name) {
                    actor.x = x;
                    actor.y = y;
                }
                self.rebuild_obstacles();
                json!({"type": "response", "status": "ok"})
            }
            "load_scenario" => {
                let path_str = cmd["path"].as_str().unwrap_or("");
                match self.load_scenario_file(std::path::Path::new(path_str)) {
                    Ok(()) => json!({"type": "response", "status": "ok"}),
                    Err(e) => {
                        json!({"type": "response", "status": "error", "error": e.to_string()})
                    }
                }
            }
            "check_free" => {
                let x = cmd["x"].as_f64().unwrap_or(0.0);
                let y = cmd["y"].as_f64().unwrap_or(0.0);
                let (col, row) = self.grid.world_to_grid(x, y);
                let in_bounds = col >= 0
                    && (col as usize) < self.grid.width
                    && row >= 0
                    && (row as usize) < self.grid.height;
                let free = in_bounds && !self.grid.is_occupied(col as usize, row as usize);
                json!({"type": "response", "free": free, "in_bounds": in_bounds, "x": x, "y": y})
            }
            "validate_scenario" => {
                let path_str = cmd.get("path").and_then(|v| v.as_str());
                let (robot_pose, goal_pos, furniture, humans) = if let Some(p) = path_str {
                    match crate::scenario::load_scenario(std::path::Path::new(p)) {
                        Ok(sc) => {
                            let furn = furniture_from_scenario(&sc.furniture);
                            let goal = sc.goal.as_ref().map(|g| (g.x, g.y));
                            let humans = humans_from_scenario(&sc.humans);
                            (Some(sc.robot_pose), goal, furn, humans)
                        }
                        Err(e) => {
                            return json!({"type": "response", "valid": false,
                            "errors": [{"severity": "error", "check": "load", "message": e.to_string()}],
                            "warnings": [], "error_count": 1, "warning_count": 0});
                        }
                    }
                } else {
                    let pose = self.drive.pose();
                    (
                        Some((pose.0, pose.1, pose.2)),
                        self.goal,
                        self.placed_furniture.clone(),
                        self.human_actors.clone(),
                    )
                };
                let report = crate::validator::validate_scenario_state(
                    &self.grid,
                    robot_pose,
                    self.robot_config.collision_radius,
                    Some(&self.robot_config.footprint),
                    goal_pos,
                    &furniture,
                    &self.furniture_types,
                    &humans,
                );
                let mut result = report.to_json();
                if let serde_json::Value::Object(ref mut m) = result {
                    m.insert("type".to_string(), json!("response"));
                }
                result
            }
            "validate_reachability" => {
                match &self.scenario {
                    None => json!({"type": "response", "error": "No scenario loaded"}),
                    Some(sc) => {
                        let report = crate::validator::validate_reachability(
                            &self.grid,
                            0.22,
                            sc.robot_pose,
                            &sc.named_poses,
                            &sc.goals,
                            &sc.furniture,
                        );
                        let mut result = serde_json::to_value(&report)
                            .unwrap_or_else(|e| json!({"error": e.to_string()}));
                        if let serde_json::Value::Object(ref mut m) = result {
                            m.insert("type".to_string(), json!("response"));
                        }
                        result
                    }
                }
            }
            "grade_run" => {
                let sim_result = cmd.get("sim_result").cloned().unwrap_or(json!({}));
                let thresholds: Option<std::collections::HashMap<String, f64>> = cmd
                    .get("thresholds")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let grade = crate::benchmark::grade_run(
                    &sim_result,
                    thresholds.as_ref(),
                );
                json!({"type": "response", "grade": grade})
            }
            "update_costmap" => {
                let layer = cmd
                    .get("layer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("global");
                let mut payload = serde_json::Map::new();
                for key in &[
                    "width",
                    "height",
                    "resolution",
                    "origin_x",
                    "origin_y",
                    "data_b64",
                ] {
                    if let Some(v) = cmd.get(*key) {
                        payload.insert(key.to_string(), v.clone());
                    }
                }
                let data = serde_json::Value::Object(payload);
                match layer {
                    "local" => self.costmap_local = Some(data),
                    _ => self.costmap_global = Some(data),
                }
                self.costmap_dirty = true;
                json!({"type": "response", "status": "ok"})
            }
            "update_nav_path" => {
                let layer = cmd
                    .get("layer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("global");
                let waypoints = cmd
                    .get("waypoints")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                match layer {
                    "local" => self.nav_path_local = Some(waypoints),
                    _ => self.nav_path_global = Some(waypoints),
                }
                self.nav_paths_dirty = true;
                json!({"type": "response", "status": "ok"})
            }
            "update_nav_trajectory" => {
                let data = cmd
                    .get("waypoints")
                    .or_else(|| cmd.get("trajectory"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                self.nav_trajectory = Some(data);
                self.nav_trajectory_dirty = true;
                json!({"type": "response", "status": "ok"})
            }
            "nav_event" => {
                const NAV_EVENTS_BUFFER_CAP: usize = 64;
                let mut ev = serde_json::Map::new();
                for key in &["source", "level", "message", "ts", "node", "kind"] {
                    if let Some(v) = cmd.get(*key) {
                        ev.insert((*key).into(), v.clone());
                    }
                }
                ev.entry("sim_time".to_string())
                    .or_insert(json!(self.sim_time));
                self.nav_events.push(serde_json::Value::Object(ev));
                if self.nav_events.len() > NAV_EVENTS_BUFFER_CAP {
                    let drop = self.nav_events.len() - NAV_EVENTS_BUFFER_CAP;
                    self.nav_events.drain(..drop);
                }
                json!({"type": "response", "status": "ok"})
            }
            "map_edit" => {
                let tool = cmd.get("tool").and_then(|v| v.as_str()).unwrap_or("brush");
                let mode = cmd
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("occupied");
                let occupied = mode == "occupied";
                let points: Vec<(f64, f64)> = cmd
                    .get("points")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|p| {
                                let a = p.as_array()?;
                                Some((a[0].as_f64()?, a[1].as_f64()?))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let radius = cmd
                    .get("radius")
                    .or(cmd.get("width"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.1);

                let mut patches = Vec::new();
                match tool {
                    "brush" => {
                        for (x, y) in &points {
                            patches.push(self.brush_edit(*x, *y, radius, occupied));
                        }
                    }
                    "line" if points.len() >= 2 => {
                        let (x0, y0) = points[0];
                        let (x1, y1) = points[1];
                        let dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                        let steps = (dist / (radius * 0.5)).ceil() as usize;
                        for i in 0..=steps {
                            let t = i as f64 / steps.max(1) as f64;
                            let x = x0 + (x1 - x0) * t;
                            let y = y0 + (y1 - y0) * t;
                            patches.push(self.brush_edit(x, y, radius, occupied));
                        }
                    }
                    "rect" if !points.is_empty() => {
                        let (cx, cy) = points[0];
                        let w = cmd.get("width").and_then(|v| v.as_f64()).unwrap_or(0.5);
                        let h = cmd.get("height").and_then(|v| v.as_f64()).unwrap_or(0.5);
                        let res = self.grid.metadata.resolution;
                        let steps_x = (w / res).ceil() as usize;
                        let steps_y = (h / res).ceil() as usize;
                        for iy in 0..=steps_y {
                            for ix in 0..=steps_x {
                                let x = cx - w / 2.0 + ix as f64 * res;
                                let y = cy - h / 2.0 + iy as f64 * res;
                                let col = ((x - self.grid.metadata.origin[0]) / res) as i32;
                                let row = ((y - self.grid.metadata.origin[1]) / res) as i32;
                                if row >= 0
                                    && (row as usize) < self.grid.height
                                    && col >= 0
                                    && (col as usize) < self.grid.width
                                {
                                    self.grid.cells
                                        [row as usize * self.grid.width + col as usize] = occupied;
                                }
                            }
                        }
                        patches.push(GridPatch {
                            row_min: 0,
                            col_min: 0,
                            before: vec![],
                            after: vec![],
                            width: 0,
                            height: 0,
                        });
                    }
                    _ => {}
                }

                let cells_modified: usize = patches.iter().map(|p| p.after.len()).sum();
                self.edit_history.push(EditAction {
                    action_type: "map_edit".into(),
                    data: serde_json::json!({"tool": tool}),
                    patches,
                });
                self.grid_dirty = true;
                self.update_scans();
                json!({"type": "response", "action": "map_edit", "tool": tool, "mode": mode, "cells_modified": cells_modified, "undo_stack_size": self.edit_history.undo_stack.len()})
            }
            "undo" => match self.edit_history.undo() {
                Some(action) => {
                    for patch in &action.patches {
                        self.apply_patch_before(patch);
                    }
                    self.grid_dirty = true;
                    self.update_scans();
                    json!({"type": "response", "status": "undone", "action_type": action.action_type, "undo_remaining": self.edit_history.undo_stack.len()})
                }
                None => json!({"type": "response", "status": "nothing_to_undo"}),
            },
            "redo" => match self.edit_history.redo() {
                Some(action) => {
                    for patch in &action.patches {
                        self.apply_patch_after(patch);
                    }
                    self.grid_dirty = true;
                    self.update_scans();
                    json!({"type": "response", "status": "redone", "action_type": action.action_type})
                }
                None => json!({"type": "response", "status": "nothing_to_redo"}),
            },
            "map_export" => {
                let path = cmd
                    .get("output_path")
                    .or(cmd.get("path"))
                    .or(cmd.get("output"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("/tmp/anews_map");
                let pgm_path = format!("{path}.pgm");
                let yaml_path = format!("{path}.yaml");
                let mut pgm_data = Vec::with_capacity(self.grid.width * self.grid.height);
                for r in (0..self.grid.height).rev() {
                    for c in 0..self.grid.width {
                        pgm_data.push(if self.grid.cells[r * self.grid.width + c] {
                            0u8
                        } else {
                            254u8
                        });
                    }
                }
                let pgm_header = format!("P5\n{} {}\n255\n", self.grid.width, self.grid.height);
                let mut file_data = pgm_header.into_bytes();
                file_data.extend_from_slice(&pgm_data);
                std::fs::write(&pgm_path, &file_data).ok();
                let yaml_content = format!(
                    "image: {}\nresolution: {}\norigin: [{}, {}, {}]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
                    pgm_path,
                    self.grid.metadata.resolution,
                    self.grid.metadata.origin[0],
                    self.grid.metadata.origin[1],
                    self.grid.metadata.origin[2]
                );
                std::fs::write(&yaml_path, &yaml_content).ok();
                json!({"type": "response", "status": "exported", "path": path})
            }
            "map_reset" => {
                self.grid.cells = self.original_grid.clone();
                self.grid_dirty = true;
                self.update_scans();
                json!({"type": "response", "status": "reset"})
            }
            "history" => {
                json!({"type": "response", "actions": self.edit_history.list_actions()})
            }
            "run_scenario" => {
                let path_str = cmd["path"].as_str().unwrap_or("");
                let goal_x = cmd
                    .get("goal")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(0))
                    .and_then(|v| v.as_f64());
                let goal_y = cmd
                    .get("goal")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(1))
                    .and_then(|v| v.as_f64());
                let goal_theta = cmd
                    .get("goal")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(2))
                    .and_then(|v| v.as_f64());
                let timeout = cmd.get("timeout").and_then(|v| v.as_f64()).unwrap_or(60.0);

                if let Err(e) = self.load_scenario_file(std::path::Path::new(path_str)) {
                    return json!({"type": "response", "status": "error", "error": e.to_string()});
                }
                if let (Some(gx), Some(gy)) = (goal_x, goal_y) {
                    self.goal = Some((gx, gy));
                    self.goal_theta = goal_theta;
                    self.goal_epoch += 1;
                    self.evaluator = Some(crate::evaluator::SimEvaluator::new(
                        &self.grid,
                        Some((gx, gy)),
                        goal_theta,
                        0.5,
                        self.robot_config.collision_radius,
                        timeout,
                        0.3,
                        0.05,
                        self.get_all_obstacles(),
                        Some(self.robot_config.footprint.clone()),
                    ));
                }
                self.playing = true;
                let max_ticks = (timeout / self.config.tick_dt) as usize + 100;
                for _ in 0..max_ticks {
                    self.tick();
                    if let Some(ref eval) = self.evaluator {
                        if eval.terminated {
                            break;
                        }
                    }
                }
                let result = self
                    .evaluator
                    .as_ref()
                    .map(|e| e.result().to_json_value())
                    .unwrap_or(serde_json::json!(null));
                json!({"type": "response", "sim_result": result})
            }
            "check_path" => {
                let x0 = cmd["x0"]
                    .as_f64()
                    .or_else(|| {
                        cmd.get("from")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a[0].as_f64())
                    })
                    .unwrap_or(0.0);
                let y0 = cmd["y0"]
                    .as_f64()
                    .or_else(|| {
                        cmd.get("from")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a[1].as_f64())
                    })
                    .unwrap_or(0.0);
                let x1 = cmd["x1"]
                    .as_f64()
                    .or_else(|| {
                        cmd.get("to")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a[0].as_f64())
                    })
                    .unwrap_or(0.0);
                let y1 = cmd["y1"]
                    .as_f64()
                    .or_else(|| {
                        cmd.get("to")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a[1].as_f64())
                    })
                    .unwrap_or(0.0);
                let res = self.grid.metadata.resolution;
                let origin = &self.grid.metadata.origin;
                let dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                let steps = (dist / res).ceil() as usize;
                let mut clear = true;
                for i in 0..=steps {
                    let t = i as f64 / steps.max(1) as f64;
                    let x = x0 + (x1 - x0) * t;
                    let y = y0 + (y1 - y0) * t;
                    let col = ((x - origin[0]) / res) as i32;
                    let row = ((y - origin[1]) / res) as i32;
                    if row < 0
                        || row as usize >= self.grid.height
                        || col < 0
                        || col as usize >= self.grid.width
                    {
                        clear = false;
                        break;
                    }
                    if self.grid.cells[row as usize * self.grid.width + col as usize] {
                        clear = false;
                        break;
                    }
                }
                json!({"type": "response", "clear": clear, "from": [x0, y0], "to": [x1, y1], "distance": dist})
            }
            "step_and_perceive" => {
                let linear = cmd.get("linear").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let angular = cmd.get("angular").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ticks = cmd.get("ticks").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                self.cmd_linear = linear;
                self.cmd_angular = angular;
                let was_playing = self.playing;
                self.playing = true;
                for _ in 0..ticks {
                    self.tick();
                }
                self.playing = was_playing;
                self.build_state()
            }
            "start_recording" => {
                let output_path = cmd
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("/tmp/anews_recording.jsonl");
                self.event_logger = None;
                if let Ok(mut logger) =
                    crate::event_logger::EventLogger::new(std::path::Path::new(output_path))
                {
                    let (px, py, pt) = self.drive.pose();
                    let init_data: serde_json::Value = json!({"robot": [px, py, pt], "goal": self.goal, "furniture": self.placed_furniture.len(), "humans": self.human_actors.len()});
                    logger.log(self.sim_time, "init", init_data);
                    self.event_logger = Some(logger);
                    json!({"type": "response", "status": "recording", "path": output_path})
                } else {
                    json!({"type": "response", "status": "error", "error": "Failed to create logger"})
                }
            }
            "stop_recording" => {
                self.event_logger = None;
                json!({"type": "response", "status": "stopped"})
            }
            "run_scenario_async" => {
                let path_str = cmd["path"].as_str().unwrap_or("");
                if let Err(e) = self.load_scenario_file(std::path::Path::new(path_str)) {
                    return json!({"type": "response", "status": "error", "error": e.to_string()});
                }
                let goal_x = cmd
                    .get("goal")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(0))
                    .and_then(|v| v.as_f64());
                let goal_y = cmd
                    .get("goal")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(1))
                    .and_then(|v| v.as_f64());
                let goal_theta = cmd
                    .get("goal")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(2))
                    .and_then(|v| v.as_f64());
                let timeout = cmd.get("timeout").and_then(|v| v.as_f64()).unwrap_or(60.0);
                if let (Some(gx), Some(gy)) = (goal_x, goal_y) {
                    self.goal = Some((gx, gy));
                    self.goal_theta = goal_theta;
                    self.goal_epoch += 1;
                    self.evaluator = Some(SimEvaluator::new(
                        &self.grid,
                        Some((gx, gy)),
                        goal_theta,
                        0.5,
                        self.robot_config.collision_radius,
                        timeout,
                        0.3,
                        0.05,
                        self.get_all_obstacles(),
                        Some(self.robot_config.footprint.clone()),
                    ));
                }
                self.playing = true;
                json!({"type": "response", "status": "started"})
            }
            "run_benchmark" => {
                let path_str = cmd.get("path").and_then(|v| v.as_str()).unwrap_or("");
                match crate::benchmark::load_benchmark(std::path::Path::new(path_str)) {
                    Ok(bdef) => {
                        let thresholds: HashMap<String, HashMap<String, f64>> = cmd
                            .get("thresholds")
                            .and_then(|v| serde_json::from_value(v.clone()).ok())
                            .unwrap_or_default();
                        let default_map_path = self.map_yaml_path.clone();
                        let config_dir = self.config_dir.clone();
                        let tick_dt = self.config.tick_dt;
                        let benchmark_dir = std::path::Path::new(path_str)
                            .parent()
                            .unwrap_or(std::path::Path::new("."))
                            .to_path_buf();
                        let results = crate::benchmark::run_benchmark(
                            &bdef,
                            &thresholds,
                            |scenario_path, goal, timeout| {
                                let scenario_file = {
                                    let p = std::path::Path::new(scenario_path);
                                    if p.exists() {
                                        p.to_path_buf()
                                    } else {
                                        let rel = benchmark_dir.join(scenario_path);
                                        if rel.exists() {
                                            rel
                                        } else {
                                            let remapped =
                                                scenario_path.replace("examples/", "/scenarios/");
                                            std::path::PathBuf::from(remapped)
                                        }
                                    }
                                };
                                let scenario_file = scenario_file.as_path();
                                let map_path = match scenario::load_scenario(scenario_file) {
                                    Ok(s) if !s.localization_map.is_empty() => {
                                        let loc = &s.localization_map;
                                        let parent = scenario_file
                                            .parent()
                                            .unwrap_or(std::path::Path::new("."));
                                        let rel = parent.join(loc);
                                        if rel.exists() {
                                            rel
                                        } else {
                                            let grandparent_rel = parent
                                                .parent()
                                                .unwrap_or(std::path::Path::new("."))
                                                .join(loc);
                                            if grandparent_rel.exists() {
                                                grandparent_rel
                                            } else {
                                                let fname = std::path::Path::new(loc)
                                                    .file_name()
                                                    .unwrap_or_default();
                                                let maps_dir =
                                                    std::path::Path::new("/maps").join(fname);
                                                if maps_dir.exists() {
                                                    maps_dir
                                                } else {
                                                    default_map_path.clone()
                                                }
                                            }
                                        }
                                    }
                                    _ => default_map_path.clone(),
                                };
                                let mut sim = match Simulation::new(
                                    &map_path,
                                    Some(scenario_file),
                                    &config_dir,
                                ) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        return serde_json::json!({"failure_class": "incomplete", "error": e.to_string()});
                                    }
                                };
                                sim.goal = Some((goal[0], goal[1]));
                                sim.evaluator = Some(SimEvaluator::new(
                                    &sim.grid,
                                    Some((goal[0], goal[1])),
                                    None,
                                    std::f64::consts::PI,
                                    sim.robot_config.collision_radius,
                                    timeout,
                                    0.3,
                                    0.05,
                                    sim.get_all_obstacles(),
                                    Some(sim.robot_config.footprint.clone()),
                                ));
                                sim.playing = true;
                                let max_ticks = (timeout / tick_dt) as usize + 100;
                                for _ in 0..max_ticks {
                                    sim.tick();
                                    if let Some(ref eval) = sim.evaluator {
                                        if eval.terminated {
                                            break;
                                        }
                                    }
                                }
                                sim.evaluator
                                    .as_ref()
                                    .map(|e| e.result().to_json_value())
                                    .unwrap_or(serde_json::json!(null))
                            },
                        );
                        json!({"type": "response", "benchmark": results.to_json()})
                    }
                    Err(e) => {
                        json!({"type": "response", "status": "error", "error": e.to_string()})
                    }
                }
            }
            "mutate_scenario" => {
                let params = cmd.get("params").cloned().unwrap_or(serde_json::json!({}));
                let mutations_result: Result<Vec<crate::scenario_mutate::Mutation>, _> =
                    serde_json::from_value(
                        params
                            .get("mutations")
                            .cloned()
                            .unwrap_or(serde_json::json!([])),
                    );
                match mutations_result {
                    Err(e) => json!({
                        "type": "response",
                        "status": "ok",
                        "action": "mutate_scenario",
                        "applied": 0,
                        "validity": "infeasible",
                        "errors": [format!("invalid mutations: {e}")],
                    }),
                    Ok(mutations) => {
                        let output_path = params
                            .get("output_scenario_path")
                            .and_then(|v| v.as_str())
                            .map(String::from);

                        let scenario_dir = self
                            .config_dir
                            .parent()
                            .map(|p| p.to_path_buf())
                            .unwrap_or_else(|| self.config_dir.clone());

                        match crate::scenario_mutate::apply_mutations(
                            None,
                            &mutations,
                            &scenario_dir,
                        ) {
                            Ok(new_scenario) => {
                                let final_path = output_path.unwrap_or_else(|| {
                                    "/tmp/mutated_scenario.yaml".to_string()
                                });
                                match serde_yaml::to_string(&new_scenario) {
                                    Err(e) => json!({
                                        "type": "response",
                                        "status": "error",
                                        "action": "mutate_scenario",
                                        "error": format!("yaml serialize failed: {e}"),
                                    }),
                                    Ok(s) => {
                                        let yaml_body = format!(
                                            "scenario:\n{}",
                                            s.lines()
                                                .map(|l| format!("  {l}"))
                                                .collect::<Vec<_>>()
                                                .join("\n")
                                        );
                                        if let Err(e) =
                                            std::fs::write(&final_path, yaml_body)
                                        {
                                            json!({
                                                "type": "response",
                                                "status": "error",
                                                "action": "mutate_scenario",
                                                "error": format!("write failed: {e}"),
                                            })
                                        } else {
                                            let applied = mutations.len();
                                            json!({
                                                "type": "response",
                                                "status": "ok",
                                                "action": "mutate_scenario",
                                                "applied": applied,
                                                "scenario_file_path": final_path,
                                                "validity": "valid",
                                                "errors": serde_json::Value::Array(vec![]),
                                            })
                                        }
                                    }
                                }
                            }
                            Err(errors) => {
                                let error_strings: Vec<String> = errors
                                    .iter()
                                    .map(|e| {
                                        format!(
                                            "[{}] {}: {}",
                                            e.index, e.mutation, e.reason
                                        )
                                    })
                                    .collect();
                                json!({
                                    "type": "response",
                                    "status": "ok",
                                    "action": "mutate_scenario",
                                    "applied": 0,
                                    "validity": "infeasible",
                                    "errors": error_strings,
                                })
                            }
                        }
                    }
                }
            }
            _ => {
                json!({"type": "response", "status": "error", "error": format!("Unknown action: {action}")})
            }
        };
        if let Some(rid) = request_id {
            if let serde_json::Value::Object(ref mut m) = result {
                m.insert("request_id".to_string(), rid);
            }
        }
        result
    }

    fn brush_edit(&mut self, wx: f64, wy: f64, radius: f64, occupied: bool) -> GridPatch {
        let res = self.grid.metadata.resolution;
        let origin = &self.grid.metadata.origin;
        let cx = ((wx - origin[0]) / res) as i32;
        let cy = ((wy - origin[1]) / res) as i32;
        let r_cells = (radius / res).ceil() as i32 + 1;

        let row_min = (cy - r_cells).max(0) as usize;
        let row_max = ((cy + r_cells) as usize).min(self.grid.height - 1);
        let col_min = (cx - r_cells).max(0) as usize;
        let col_max = ((cx + r_cells) as usize).min(self.grid.width - 1);

        let patch_h = row_max - row_min + 1;
        let patch_w = col_max - col_min + 1;

        let mut before = Vec::with_capacity(patch_h * patch_w);
        for r in row_min..=row_max {
            for c in col_min..=col_max {
                before.push(self.grid.cells[r * self.grid.width + c]);
            }
        }

        for r in row_min..=row_max {
            for c in col_min..=col_max {
                let cell_wx = origin[0] + (c as f64 + 0.5) * res;
                let cell_wy = origin[1] + (r as f64 + 0.5) * res;
                if ((cell_wx - wx).powi(2) + (cell_wy - wy).powi(2)).sqrt() <= radius {
                    self.grid.cells[r * self.grid.width + c] = occupied;
                }
            }
        }

        let mut after = Vec::with_capacity(patch_h * patch_w);
        for r in row_min..=row_max {
            for c in col_min..=col_max {
                after.push(self.grid.cells[r * self.grid.width + c]);
            }
        }

        GridPatch {
            row_min,
            col_min,
            before,
            after,
            width: patch_w,
            height: patch_h,
        }
    }

    fn apply_patch_before(&mut self, patch: &GridPatch) {
        for r in 0..patch.height {
            for c in 0..patch.width {
                self.grid.cells[(patch.row_min + r) * self.grid.width + (patch.col_min + c)] =
                    patch.before[r * patch.width + c];
            }
        }
    }

    fn apply_patch_after(&mut self, patch: &GridPatch) {
        for r in 0..patch.height {
            for c in 0..patch.width {
                self.grid.cells[(patch.row_min + r) * self.grid.width + (patch.col_min + c)] =
                    patch.after[r * patch.width + c];
            }
        }
    }

    fn rebuild_furniture_obstacles(&mut self) {
        let profiles: Vec<String> = self
            .sensor_suite
            .sensors
            .values()
            .map(|s| s.profile.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        self.furniture_obstacle_cache.clear();
        for profile in &profiles {
            let obs = resolve_furniture_to_obstacles(
                &self.placed_furniture,
                &self.furniture_types,
                profile,
            );
            self.furniture_obstacle_cache.insert(profile.clone(), obs);
        }
    }

    fn rebuild_obstacles(&mut self) {
        let human_obs = if !self.human_actors.is_empty() {
            actors_to_obstacles(&self.human_actors)
        } else {
            vec![]
        };
        self.obstacle_cache.clear();
        for (profile, furniture_obs) in &self.furniture_obstacle_cache {
            let mut combined = furniture_obs.clone();
            combined.extend(human_obs.iter().cloned());
            self.obstacle_cache.insert(profile.clone(), combined);
        }
    }

    fn get_all_obstacles(&self) -> Vec<ResolvedObstacle> {
        self.obstacle_cache
            .get("depth")
            .cloned()
            .unwrap_or_default()
    }

    fn update_scans(&mut self) {
        self.rebuild_obstacles();
        self.raycast_sensors();
    }

    fn compute_collision_state(&self) -> (bool, bool, Vec<String>, Vec<String>) {
        let (rx, ry, rtheta) = self.drive.pose();
        let fp = &self.robot_config.footprint;
        let mut collision_wall = false;
        let mut collision_furniture_ids: Vec<String> = Vec::new();
        let mut collision_human_names: Vec<String> = Vec::new();
        let empty_grid = OccupancyGrid {
            cells: vec![false; self.grid.cells.len()],
            width: self.grid.width,
            height: self.grid.height,
            metadata: self.grid.metadata.clone(),
        };

        if crate::detectors::check_footprint_collision(&self.grid, rx, ry, rtheta, fp, &[])
            .is_some()
        {
            collision_wall = true;
        }

        let furniture_obs_by_id: Vec<(String, Vec<raycaster::ResolvedObstacle>)> = self
            .placed_furniture
            .iter()
            .map(|pf| {
                let mut obs = resolve_furniture_to_obstacles(
                    std::slice::from_ref(pf),
                    &self.furniture_types,
                    "depth",
                );
                if obs.is_empty() {
                    obs.push(raycaster::ResolvedObstacle::Rect {
                        cx: pf.x,
                        cy: pf.y,
                        half_w: pf.width.unwrap_or(0.0) / 2.0,
                        half_h: pf.height.unwrap_or(0.0) / 2.0,
                        rotation: pf.rotation,
                    });
                }
                (pf.id.clone(), obs)
            })
            .collect();
        let furniture_obs: Vec<_> = furniture_obs_by_id
            .iter()
            .flat_map(|(_, obs)| obs.iter().cloned())
            .collect();
        if crate::detectors::check_footprint_collision(
            &self.grid,
            rx,
            ry,
            rtheta,
            fp,
            &furniture_obs,
        )
        .is_some()
        {
            for (id, obs) in &furniture_obs_by_id {
                if crate::detectors::check_footprint_collision(
                    &empty_grid,
                    rx,
                    ry,
                    rtheta,
                    fp,
                    obs,
                )
                .is_some()
                {
                    collision_furniture_ids.push(id.clone());
                }
            }
        }

        let human_obs = crate::human::actors_to_obstacles(&self.human_actors);
        if !human_obs.is_empty()
            && crate::detectors::check_footprint_collision(
                &self.grid, rx, ry, rtheta, fp, &human_obs,
            )
            .is_some()
        {
            for h in &self.human_actors {
                if !h.enabled || !h.active || h.finished {
                    continue;
                }
                let legs = crate::human::actors_to_obstacles(std::slice::from_ref(h));
                if crate::detectors::check_footprint_collision(
                    &self.grid, rx, ry, rtheta, fp, &legs,
                )
                .is_some()
                {
                    collision_human_names.push(h.name.clone());
                }
            }
        }

        let collision = collision_wall
            || !collision_furniture_ids.is_empty()
            || !collision_human_names.is_empty();
        (
            collision,
            collision_wall,
            collision_furniture_ids,
            collision_human_names,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_sim() -> Simulation {
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
        let original_grid = grid.cells.clone();
        Simulation {
            grid,
            drive: DiffDrive::new(5.0, 5.0, 0.0),
            sensor_suite: SensorSuite {
                sensors: HashMap::new(),
            },
            furniture_types: HashMap::new(),
            placed_furniture: Vec::new(),
            human_actors: Vec::new(),
            robot_config: robot_config::default_config(),
            evaluator: None,
            config: SimConfig::default(),
            playing: true,
            speed: 1.0,
            sim_time: 0.0,
            cmd_linear: 0.0,
            cmd_angular: 0.0,
            actual_linear: 0.0,
            actual_angular: 0.0,
            goal: None,
            goal_theta: None,
            goal_epoch: 0,
            scan_results: HashMap::new(),
            scenario: None,
            costmap_global: None,
            costmap_local: None,
            nav_path_global: None,
            nav_path_local: None,
            nav_trajectory: None,
            nav_events: Vec::new(),
            costmap_dirty: false,
            nav_paths_dirty: false,
            nav_trajectory_dirty: false,
            overlay_ticks_since_send: 0,
            edit_history: EditHistory::new(50),
            grid_dirty: false,
            original_grid,
            furniture_obstacle_cache: HashMap::new(),
            obstacle_cache: HashMap::new(),
            config_dir: std::path::PathBuf::new(),
            map_yaml_path: std::path::PathBuf::new(),
            event_logger: None,
            event_bus: EventBus::new(),
            stuck_detector: Default::default(),
            human_approach_detector: Default::default(),
            path_replanned_detector: Default::default(),
            cancellation: CancellationRegistry::new(),
        }
    }

    #[test]
    fn test_handle_command_set_cmd_vel() {
        let mut sim = make_minimal_sim();
        let resp =
            sim.handle_command(&json!({"action": "set_cmd_vel", "linear": 0.5, "angular": 0.1}));
        assert_eq!(resp["status"], "ok");
        assert!((sim.cmd_linear - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_tick_advances_time() {
        let mut sim = make_minimal_sim();
        sim.tick();
        assert!((sim.sim_time - 0.05).abs() < 1e-9);
        sim.tick();
        assert!((sim.sim_time - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_pause_prevents_tick() {
        let mut sim = make_minimal_sim();
        sim.playing = false;
        sim.cmd_linear = 1.0;
        sim.tick();
        assert!((sim.sim_time - 0.0).abs() < 1e-9);
        assert!((sim.drive.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_state_has_required_fields() {
        let mut sim = make_minimal_sim();
        let state = sim.build_state();
        assert_eq!(state["type"], "state");
        assert!(state["robot"].is_array());
        assert!(state["t"].is_number());
        assert!(state.get("humans").is_some());
        assert!(state.get("furniture").is_some());
    }

    #[test]
    fn test_update_human_changes_speed() {
        let mut sim = make_minimal_sim();
        sim.human_actors.push(crate::human::HumanActor::new(
            "waiter".into(),
            vec![(0.0, 0.0), (1.0, 0.0)],
            0.3,
            0.8,
            false,
            0.0,
            0.0,
            true,
            crate::human::AvoidanceConfig::default(),
        ));
        sim.handle_command(&json!({"action": "update_human", "name": "waiter", "speed": 1.5}));
        assert!((sim.human_actors[0].speed - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_update_human_path_resets_progress() {
        let mut sim = make_minimal_sim();
        sim.human_actors.push(crate::human::HumanActor::new(
            "waiter".into(),
            vec![(0.0, 0.0), (1.0, 0.0)],
            0.3,
            0.8,
            false,
            0.0,
            0.0,
            true,
            crate::human::AvoidanceConfig::default(),
        ));
        sim.human_actors[0].path_idx = 1;
        sim.human_actors[0].path_t = 0.5;
        sim.handle_command(&json!({"action": "update_human_path", "name": "waiter", "path": [[2.0, 0.0], [3.0, 0.0]]}));
        assert_eq!(sim.human_actors[0].path_idx, 0);
        assert!((sim.human_actors[0].path_t).abs() < 1e-9);
        assert!((sim.human_actors[0].x - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_set_human_pose_teleports() {
        let mut sim = make_minimal_sim();
        sim.human_actors.push(crate::human::HumanActor::new(
            "waiter".into(),
            vec![(0.0, 0.0), (1.0, 0.0)],
            0.3,
            0.8,
            false,
            0.0,
            0.0,
            true,
            crate::human::AvoidanceConfig::default(),
        ));
        sim.handle_command(
            &json!({"action": "set_human_pose", "name": "waiter", "x": 5.5, "y": 3.2}),
        );
        assert!((sim.human_actors[0].x - 5.5).abs() < 1e-9);
        assert!((sim.human_actors[0].y - 3.2).abs() < 1e-9);
    }

    #[test]
    fn test_check_free_returns_free_for_empty_cell() {
        let mut sim = make_minimal_sim();
        let resp = sim.handle_command(&json!({"action": "check_free", "x": 5.0, "y": 5.0}));
        assert_eq!(resp["free"], true);
        assert_eq!(resp["in_bounds"], true);
    }

    #[test]
    fn test_check_free_returns_occupied_for_wall() {
        let mut sim = make_minimal_sim();
        sim.grid.cells[3 * sim.grid.width + 3] = true;
        let resp = sim.handle_command(&json!({"action": "check_free", "x": 3.0, "y": 3.0}));
        assert_eq!(resp["free"], false);
        assert_eq!(resp["in_bounds"], true);
    }

    #[test]
    fn test_check_free_out_of_bounds() {
        let mut sim = make_minimal_sim();
        let resp = sim.handle_command(&json!({"action": "check_free", "x": -5.0, "y": -5.0}));
        assert_eq!(resp["in_bounds"], false);
        assert_eq!(resp["free"], false);
    }

    #[test]
    fn test_update_costmap_stores_global() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "update_costmap", "layer": "global", "width": 10, "height": 10, "resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0, "data_b64": "AAAA"}));
        assert!(sim.costmap_global.is_some());
        assert_eq!(sim.costmap_global.as_ref().unwrap()["width"], 10);
        assert_eq!(sim.costmap_global.as_ref().unwrap()["data_b64"], "AAAA");
    }

    #[test]
    fn test_update_nav_path_stores_poses() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "update_nav_path", "layer": "global", "poses": [[1.0, 2.0], [3.0, 4.0]]}));
        assert!(sim.nav_path_global.is_some());
    }

    #[test]
    fn test_build_state_includes_nav_overlays_when_present() {
        let mut sim = make_minimal_sim();
        sim.handle_command(
            &json!({"action": "update_nav_trajectory", "trajectory": [[0.0, 0.0, 0.5]]}),
        );
        for _ in 0..OVERLAY_SEND_TICKS {
            let _ = sim.build_state();
            sim.flush_dirty();
        }
        sim.handle_command(
            &json!({"action": "update_nav_trajectory", "trajectory": [[0.0, 0.0, 0.5]]}),
        );
        for _ in 0..OVERLAY_SEND_TICKS - 1 {
            let _ = sim.build_state();
            sim.flush_dirty();
        }
        let state = sim.build_state();
        assert!(state.get("nav_trajectory").is_some());
    }

    #[test]
    fn test_map_edit_brush_modifies_grid() {
        let mut sim = make_minimal_sim();
        let resp = sim.handle_command(&json!({"action": "map_edit", "tool": "brush", "mode": "occupied", "points": [[5.5, 5.5]], "radius": 1.0}));
        assert_eq!(resp["action"], "map_edit");
        assert!(resp["cells_modified"].as_u64().unwrap() > 0);
        let col = 5;
        let row = 5;
        assert!(sim.grid.cells[row * sim.grid.width + col]);
    }

    #[test]
    fn test_undo_reverts_map_edit() {
        let mut sim = make_minimal_sim();
        assert!(!sim.grid.cells[5 * sim.grid.width + 5]);
        sim.handle_command(&json!({"action": "map_edit", "tool": "brush", "mode": "occupied", "points": [[5.5, 5.5]], "radius": 1.0}));
        assert!(sim.grid.cells[5 * sim.grid.width + 5]);
        let resp = sim.handle_command(&json!({"action": "undo"}));
        assert_eq!(resp["status"], "undone");
        assert!(!sim.grid.cells[5 * sim.grid.width + 5]);
    }

    #[test]
    fn test_redo_reapplies_undone_edit() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "map_edit", "tool": "brush", "mode": "occupied", "points": [[5.5, 5.5]], "radius": 1.0}));
        sim.handle_command(&json!({"action": "undo"}));
        assert!(!sim.grid.cells[5 * sim.grid.width + 5]);
        let resp = sim.handle_command(&json!({"action": "redo"}));
        assert_eq!(resp["status"], "redone");
        assert!(sim.grid.cells[5 * sim.grid.width + 5]);
    }

    #[test]
    fn test_map_reset_restores_original() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "map_edit", "tool": "brush", "mode": "occupied", "points": [[5.5, 5.5]], "radius": 1.0}));
        assert!(sim.grid.cells[5 * sim.grid.width + 5]);
        sim.handle_command(&json!({"action": "map_reset"}));
        assert!(!sim.grid.cells[5 * sim.grid.width + 5]);
    }

    #[test]
    fn test_history_returns_action_list() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "map_edit", "tool": "brush", "mode": "occupied", "points": [[5.5, 5.5]], "radius": 1.0}));
        let resp = sim.handle_command(&json!({"action": "history"}));
        let actions = resp["actions"].as_array().unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0]["type"], "map_edit");
    }

    #[test]
    fn test_check_path_clear_on_empty_grid() {
        let mut sim = make_minimal_sim();
        let resp = sim.handle_command(
            &json!({"action": "check_path", "x0": 2.0, "y0": 2.0, "x1": 8.0, "y1": 8.0}),
        );
        assert_eq!(resp["clear"], true);
    }

    #[test]
    fn test_check_path_blocked_by_wall() {
        let mut sim = make_minimal_sim();
        for c in 0..sim.grid.width {
            sim.grid.cells[5 * sim.grid.width + c] = true;
        }
        let resp = sim.handle_command(
            &json!({"action": "check_path", "x0": 2.0, "y0": 2.0, "x1": 2.0, "y1": 8.0}),
        );
        assert_eq!(resp["clear"], false);
    }

    #[test]
    fn test_step_and_perceive_returns_state() {
        let mut sim = make_minimal_sim();
        let resp = sim.handle_command(
            &json!({"action": "step_and_perceive", "linear": 0.5, "angular": 0.0, "ticks": 5}),
        );
        assert_eq!(resp["type"], "state");
        assert!(resp["robot"].is_array());
    }

    #[test]
    fn test_tick_logger_includes_collision_fields() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().unwrap();
        let mut sim = make_minimal_sim();
        sim.event_logger = Some(crate::event_logger::EventLogger::new(tmp.path()).unwrap());

        sim.placed_furniture.push(PlacedFurniture {
            id: "f0".into(),
            type_id: "table".into(),
            x: 5.0,
            y: 5.0,
            rotation: 0.0,
            width: Some(1.0),
            height: Some(1.0),
            label: "".into(),
        });

        sim.tick();

        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert!(!lines.is_empty());

        let event: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert!(event.get("collision").is_some());
        assert!(event.get("collision_wall").is_some());
        assert!(event.get("collision_furniture_ids").is_some());
        assert!(event.get("collision_human_names").is_some());
    }

    #[test]
    fn test_wall_collision_does_not_mark_every_furniture_item() {
        let mut sim = make_minimal_sim();
        sim.grid.metadata.origin = [-0.5, -0.5, 0.0];
        let (col, row) = sim.grid.world_to_grid(5.0, 5.0);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = (col + dx) as usize;
                let y = (row + dy) as usize;
                sim.grid.cells[y * sim.grid.width + x] = true;
            }
        }
        sim.placed_furniture.push(PlacedFurniture {
            id: "far_a".into(),
            type_id: "chair".into(),
            x: 1.0,
            y: 1.0,
            rotation: 0.0,
            width: Some(0.2),
            height: Some(0.2),
            label: "far_a".into(),
        });
        sim.placed_furniture.push(PlacedFurniture {
            id: "far_b".into(),
            type_id: "chair".into(),
            x: 9.0,
            y: 9.0,
            rotation: 0.0,
            width: Some(0.2),
            height: Some(0.2),
            label: "far_b".into(),
        });

        let (collision, collision_wall, collision_furniture_ids, _) = sim.compute_collision_state();

        assert!(collision);
        assert!(collision_wall);
        assert!(collision_furniture_ids.is_empty());
    }

    #[test]
    fn test_round_furniture_collision_uses_depth_circle_not_bounding_square() {
        let mut sim = make_minimal_sim();
        sim.drive = DiffDrive::new(5.62, 5.62, 0.0);
        sim.furniture_types.insert(
            "round".into(),
            FurnitureType {
                name: "Round".into(),
                visual_shape: "circle".into(),
                default_width: 1.0,
                default_height: 1.0,
                laser_profile: vec![crate::furniture::CollisionShape::Circle {
                    offset_x: 0.0,
                    offset_y: 0.0,
                    radius: 0.05,
                }],
                depth_profile: vec![crate::furniture::CollisionShape::Circle {
                    offset_x: 0.0,
                    offset_y: 0.0,
                    radius: 0.5,
                }],
            },
        );
        sim.placed_furniture.push(PlacedFurniture {
            id: "round_table".into(),
            type_id: "round".into(),
            x: 5.0,
            y: 5.0,
            rotation: 0.0,
            width: Some(1.0),
            height: Some(1.0),
            label: "round_table".into(),
        });

        let (collision, _, collision_furniture_ids, _) = sim.compute_collision_state();

        assert!(!collision);
        assert!(collision_furniture_ids.is_empty());
    }

    #[test]
    fn test_tick_logger_includes_evaluator_summary() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().unwrap();
        let mut sim = make_minimal_sim();
        sim.event_logger = Some(crate::event_logger::EventLogger::new(tmp.path()).unwrap());

        let resp = sim.handle_command(&json!({"action": "set_goal", "x": 8.0, "y": 5.0}));
        assert_eq!(resp["status"], "ok");

        sim.tick();

        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert!(!lines.is_empty());

        let event: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
        assert!(event.get("evaluator").is_some());
        assert!(
            !event["evaluator"].is_null(),
            "evaluator should not be null when goal is active"
        );
        assert!(event["evaluator"].get("min_clearance").is_some());
        assert!(event["evaluator"].get("near_miss_count").is_some());
        assert!(event["evaluator"].get("failure_class").is_some());
    }

    #[test]
    fn test_tick_logger_includes_goal_and_scan_ranges() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().unwrap();
        let mut sim = make_minimal_sim();
        sim.event_logger = Some(crate::event_logger::EventLogger::new(tmp.path()).unwrap());

        let resp = sim.handle_command(&json!({"action": "set_goal", "x": 8.0, "y": 5.0}));
        assert_eq!(resp["status"], "ok");

        sim.tick();

        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert!(!lines.is_empty());

        let event: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert!(event.get("goal").is_some());
    }

    #[test]
    fn test_run_scenario_async_preserves_goal_theta() {
        let dir = tempfile::tempdir().unwrap();
        let scenario_path = dir.path().join("scenario.yaml");
        std::fs::write(
            &scenario_path,
            "scenario:\n  name: theta_goal\n  robot:\n    spawn_pose: [0.0, 0.0, 0.0]\n  furniture: []\n  humans: []\n  events: []\n",
        )
        .unwrap();

        let mut sim = make_minimal_sim();
        let resp = sim.handle_command(&json!({
            "action": "run_scenario_async",
            "path": scenario_path,
            "goal": [3.01, -5.85, -2.51],
            "timeout": 90.0,
        }));
        assert_eq!(resp["status"], "started");
        assert_eq!(sim.goal, Some((3.01, -5.85)));
        assert_eq!(sim.goal_theta, Some(-2.51));
    }

    #[test]
    fn test_run_scenario_async_clears_stale_goal_theta_when_no_theta_supplied() {
        let dir = tempfile::tempdir().unwrap();
        let scenario_path = dir.path().join("scenario.yaml");
        std::fs::write(
            &scenario_path,
            "scenario:\n  name: theta_goal\n  robot:\n    spawn_pose: [0.0, 0.0, 0.0]\n  furniture: []\n  humans: []\n  events: []\n",
        )
        .unwrap();

        let mut sim = make_minimal_sim();
        sim.goal_theta = Some(0.0409);
        let resp = sim.handle_command(&json!({
            "action": "run_scenario_async",
            "path": scenario_path,
            "goal": [3.01, -5.85],
            "timeout": 90.0,
        }));
        assert_eq!(resp["status"], "started");
        assert_eq!(sim.goal, Some((3.01, -5.85)));
        assert_eq!(sim.goal_theta, None);
    }

    #[test]
    fn test_cmd_vel_clamped() {
        let mut sim = make_minimal_sim();
        let resp =
            sim.handle_command(&json!({"action": "set_cmd_vel", "linear": 5.0, "angular": 3.0}));
        assert_eq!(resp["status"], "ok");
        assert!(
            (sim.cmd_linear - 0.6).abs() < 1e-9,
            "linear should be clamped to 0.6, got {}",
            sim.cmd_linear
        );
        assert!(
            (sim.cmd_angular - 0.8).abs() < 1e-9,
            "angular should be clamped to 0.8, got {}",
            sim.cmd_angular
        );
    }

    #[test]
    fn test_cmd_vel_clamped_negative() {
        let mut sim = make_minimal_sim();
        let resp =
            sim.handle_command(&json!({"action": "set_cmd_vel", "linear": -5.0, "angular": -3.0}));
        assert_eq!(resp["status"], "ok");
        assert!(
            (sim.cmd_linear - (-0.6)).abs() < 1e-9,
            "linear should be clamped to -0.6, got {}",
            sim.cmd_linear
        );
        assert!(
            (sim.cmd_angular - (-0.8)).abs() < 1e-9,
            "angular should be clamped to -0.8, got {}",
            sim.cmd_angular
        );
    }

    #[test]
    fn test_acceleration_ramp() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "set_cmd_vel", "linear": 0.6, "angular": 0.0}));
        sim.tick();
        assert!(
            sim.actual_linear < 0.02,
            "actual_linear should ramp slowly, got {}",
            sim.actual_linear
        );
        assert!(
            sim.actual_linear > 0.0,
            "actual_linear should increase from 0"
        );
    }

    #[test]
    fn test_actual_vel_in_state() {
        let mut sim = make_minimal_sim();
        sim.handle_command(&json!({"action": "set_cmd_vel", "linear": 0.3, "angular": 0.1}));
        sim.tick();
        let state = sim.build_state();
        assert!(
            state.get("actual_vel").is_some(),
            "state should include actual_vel"
        );
        let actual_vel = state["actual_vel"].as_array().unwrap();
        assert!(
            actual_vel[0].as_f64().unwrap() > 0.0,
            "actual linear vel should be positive after tick"
        );
    }

    #[test]
    fn test_load_scenario_swaps_map() {
        let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let config_dir = project_root.join("config");
        let restaurant_map = project_root.join("examples/maps/restaurant.yaml");

        let mut sim = Simulation::new(&restaurant_map, None, &config_dir).unwrap();
        assert_ne!((sim.grid.width, sim.grid.height), (240, 60), "the starting map must differ from the corridor");

        let corridor_map = project_root.join("examples/maps/corridor.yaml");
        let tmp_dir = std::env::temp_dir().join("anews_test_swap_map");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let scenario_path = tmp_dir.join("swap_test.yaml");
        std::fs::write(&scenario_path, format!(
            "scenario:\n  name: swap_test\n  localization_map: {}\n  robot:\n    spawn_pose: [0.0, 0.0, 0.0]\n  goal:\n    x: 1.0\n    y: 0.0\n    tolerance: 0.3\n    timeout: 30.0\n  furniture: []\n  humans: []\n  events: []\n",
            corridor_map.display()
        )).unwrap();

        sim.load_scenario_file(&scenario_path).unwrap();

        assert_eq!(sim.grid.width, 240, "grid width should change to corridor");
        assert_eq!(sim.grid.height, 60, "grid height should change to corridor");

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn build_init_emits_named_poses() {
        let dir = tempfile::tempdir().unwrap();
        let scenario_yaml = dir.path().join("scenario.yaml");
        std::fs::write(
            &scenario_yaml,
            r#"
scenario:
  name: init_test
  named_poses:
    - name: CHARGER
      kind: charger
      x: 0.0
      y: 0.0
      theta: 0.0
    - name: T1
      kind: delivery
      x: 2.5
      y: -1.5
      theta: 1.571
"#,
        ).unwrap();

        let mut sim = make_minimal_sim();
        sim.load_scenario_file(&scenario_yaml).unwrap();
        let init = sim.build_init();

        let arr = init["named_poses"].as_array().expect("named_poses missing from init");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["name"], "CHARGER");
        assert_eq!(arr[0]["kind"], "charger");
        assert_eq!(arr[1]["kind"], "delivery");
        assert!((arr[1]["x"].as_f64().unwrap() - 2.5).abs() < 1e-6);
    }
}

#[cfg(test)]
mod event_integration_tests {
    use super::*;
    use crate::events::SimEvent;

    #[tokio::test]
    async fn test_event_bus_emit_subscribe() {
        let sim = {
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
            let original_grid = grid.cells.clone();
            Simulation {
                grid,
                drive: DiffDrive::new(5.0, 5.0, 0.0),
                sensor_suite: SensorSuite {
                    sensors: std::collections::HashMap::new(),
                },
                furniture_types: std::collections::HashMap::new(),
                placed_furniture: Vec::new(),
                human_actors: Vec::new(),
                robot_config: crate::robot_config::default_config(),
                evaluator: None,
                config: SimConfig::default(),
                playing: true,
                speed: 1.0,
                sim_time: 0.0,
                cmd_linear: 0.0,
                cmd_angular: 0.0,
                actual_linear: 0.0,
                actual_angular: 0.0,
                goal: None,
                goal_theta: None,
                goal_epoch: 0,
                scan_results: std::collections::HashMap::new(),
                scenario: None,
                costmap_global: None,
                costmap_local: None,
                nav_path_global: None,
                nav_path_local: None,
                nav_trajectory: None,
                nav_events: Vec::new(),
                costmap_dirty: false,
                nav_paths_dirty: false,
                nav_trajectory_dirty: false,
                overlay_ticks_since_send: 0,
                edit_history: EditHistory::new(50),
                grid_dirty: false,
                original_grid,
                furniture_obstacle_cache: std::collections::HashMap::new(),
                obstacle_cache: std::collections::HashMap::new(),
                config_dir: std::path::PathBuf::new(),
                map_yaml_path: std::path::PathBuf::new(),
                event_logger: None,
                event_bus: EventBus::new(),
                stuck_detector: Default::default(),
                human_approach_detector: Default::default(),
                path_replanned_detector: Default::default(),
                cancellation: CancellationRegistry::new(),
            }
        };

        let mut rx = sim.event_bus.subscribe();

        sim.event_bus.emit(SimEvent::Collision {
            obstacle_id: "test_wall".into(),
            tick: 7,
        });

        let got = rx.recv().await.unwrap();
        assert!(
            matches!(got, SimEvent::Collision { ref obstacle_id, tick: 7 } if obstacle_id == "test_wall"),
            "expected Collision event, got {:?}",
            got
        );
    }

    #[tokio::test]
    async fn test_evaluator_timeout_emits_event() {
        let grid = OccupancyGrid {
            cells: vec![false; 400],
            width: 20,
            height: 20,
            metadata: crate::map::MapMetadata {
                resolution: 0.05,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        };
        let original_grid = grid.cells.clone();
        let mut sim = Simulation {
            grid: grid.clone(),
            drive: DiffDrive::new(0.5, 0.5, 0.0),
            sensor_suite: SensorSuite {
                sensors: std::collections::HashMap::new(),
            },
            furniture_types: std::collections::HashMap::new(),
            placed_furniture: Vec::new(),
            human_actors: Vec::new(),
            robot_config: crate::robot_config::default_config(),
            evaluator: None,
            config: SimConfig::default(),
            playing: true,
            speed: 1.0,
            sim_time: 0.0,
            cmd_linear: 0.0,
            cmd_angular: 0.0,
            actual_linear: 0.0,
            actual_angular: 0.0,
            goal: None,
            goal_theta: None,
            goal_epoch: 0,
            scan_results: std::collections::HashMap::new(),
            scenario: None,
            costmap_global: None,
            costmap_local: None,
            nav_path_global: None,
            nav_path_local: None,
            nav_trajectory: None,
            nav_events: Vec::new(),
            costmap_dirty: false,
            nav_paths_dirty: false,
            nav_trajectory_dirty: false,
            overlay_ticks_since_send: 0,
            edit_history: EditHistory::new(50),
            grid_dirty: false,
            original_grid,
            furniture_obstacle_cache: std::collections::HashMap::new(),
            obstacle_cache: std::collections::HashMap::new(),
            config_dir: std::path::PathBuf::new(),
            map_yaml_path: std::path::PathBuf::new(),
            event_logger: None,
            event_bus: EventBus::new(),
            stuck_detector: Default::default(),
            human_approach_detector: Default::default(),
            path_replanned_detector: Default::default(),
            cancellation: CancellationRegistry::new(),
        };

        sim.evaluator = Some(crate::evaluator::SimEvaluator::new(
            &grid,
            Some((999.0, 999.0)),
            None,
            f64::INFINITY,
            0.25,
            0.1,
            0.3,
            0.15,
            Vec::new(),
            None,
        ));

        let mut rx = sim.event_bus.subscribe();

        sim.cmd_linear = 0.1;
        sim.actual_linear = 0.1;

        for _ in 0..10 {
            sim.tick();
        }

        let got = rx.try_recv().expect("expected a Timeout event in the channel");
        assert!(
            matches!(got, SimEvent::Timeout { .. }),
            "expected Timeout event, got {:?}",
            got
        );
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::CancellationRegistry;

    #[test]
    fn test_cancellation_registry_lifecycle() {
        let r = CancellationRegistry::new();
        assert!(!r.is_cancelled("r1"));
        r.cancel("r1");
        assert!(r.is_cancelled("r1"));
        r.clear("r1");
        assert!(!r.is_cancelled("r1"));
    }

    #[test]
    fn test_cancellation_registry_is_shared_via_clone() {
        let r1 = CancellationRegistry::new();
        let r2 = r1.clone();
        r1.cancel("rX");
        assert!(r2.is_cancelled("rX"), "clone should share inner state");
    }
}

#[cfg(test)]
mod tilted_cam_tests {
    use crate::sensor::{MountPose, SensorConfig};

    fn make_sensor(
        pitch_deg: f64,
        h: f64,
        hfov_deg: f64,
        vfov_deg: Option<f64>,
        range_min: f64,
        range_max: f64,
    ) -> SensorConfig {
        let pitch = pitch_deg.to_radians();
        let qy = (pitch * 0.5).sin();
        let qw = (pitch * 0.5).cos();
        let mut cfg = SensorConfig {
            name: "test".into(),
            frame_id: "test".into(),
            topic: "test".into(),
            fov_degrees: hfov_deg,
            vfov_degrees: vfov_deg,
            range_min,
            range_max,
            num_rays: 16,
            noise_std_dev: 0.0,
            mount_offset: [0.0, 0.0, 0.0],
            mount_pose: Some(MountPose {
                xyz: [0.0, 0.0, h],
                quat_xyzw: [0.0, qy, 0.0, qw],
            }),
            rate_hz: 10,
            profile: "depth".into(),
            fov_radians: 0.0,
            angle_min: 0.0,
            angle_max: 0.0,
            angle_increment: 0.0,
        };
        cfg.compute_angles();
        cfg
    }

    #[test]
    fn no_mount_pose_returns_no_polygon() {
        let mut cfg = SensorConfig {
            name: "lidar".into(),
            frame_id: "laser".into(),
            topic: "scan".into(),
            fov_degrees: 270.0,
            vfov_degrees: None,
            range_min: 0.05,
            range_max: 10.0,
            num_rays: 360,
            noise_std_dev: 0.0,
            mount_offset: [0.22, 0.0, 0.0],
            mount_pose: None,
            rate_hz: 10,
            profile: "laser".into(),
            fov_radians: 0.0,
            angle_min: 0.0,
            angle_max: 0.0,
            angle_increment: 0.0,
        };
        cfg.compute_angles();
        assert!(super::frustum_floor_corners(&cfg, 0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn small_pitch_below_threshold_returns_no_polygon() {
        let cfg = make_sensor(3.0, 1.0, 60.0, Some(45.0), 0.1, 5.0);
        assert!(super::frustum_floor_corners(&cfg, 0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn rear_cam_corners_form_trapezoid_behind_robot() {
        let mut cfg = SensorConfig {
            name: "rear".into(),
            frame_id: "rear".into(),
            topic: "rear".into(),
            fov_degrees: 60.0,
            vfov_degrees: Some(45.0),
            range_min: 0.25,
            range_max: 2.0,
            num_rays: 16,
            noise_std_dev: 0.0,
            mount_offset: [-0.35, 0.0, std::f64::consts::PI],
            mount_pose: Some(MountPose {
                xyz: [-0.35, 0.0, 1.128844],
                quat_xyzw: [0.542170, 0.005427, -0.840214, 0.007916],
            }),
            rate_hz: 10,
            profile: "depth".into(),
            fov_radians: 0.0,
            angle_min: 0.0,
            angle_max: 0.0,
            angle_increment: 0.0,
        };
        cfg.compute_angles();
        let poly = super::frustum_floor_corners(&cfg, 0.0, 0.0, 0.0).unwrap();
        for (x, _) in &poly {
            assert!(*x < 0.0, "corner x={x} should be < 0 (behind robot)");
        }
        let mut indexed: Vec<(usize, f64, f64)> = poly
            .iter()
            .enumerate()
            .map(|(i, (x, y))| (i, x.abs(), *y))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let near_y_spread = (indexed[0].2 - indexed[1].2).abs();
        let far_y_spread = (indexed[2].2 - indexed[3].2).abs();
        assert!(
            far_y_spread > near_y_spread,
            "far edge width {far_y_spread} should exceed near edge width {near_y_spread}"
        );
    }

    #[test]
    fn ray_polygon_clip_outside_origin() {
        let poly = [(1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)];
        let (t_in, t_out) = super::clip_ray_to_convex_poly(0.0, 0.0, 1.0, 0.0, &poly).unwrap();
        assert!((t_in - 1.0).abs() < 1e-6, "t_in={t_in}");
        assert!((t_out - 3.0).abs() < 1e-6, "t_out={t_out}");
    }

    #[test]
    fn ray_polygon_clip_misses() {
        let poly = [(1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)];
        assert!(super::clip_ray_to_convex_poly(0.0, 5.0, 1.0, 0.0, &poly).is_none());
    }

    #[test]
    fn ray_polygon_clip_origin_inside() {
        let poly = [(1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)];
        let (t_in, t_out) = super::clip_ray_to_convex_poly(2.0, 0.0, 1.0, 0.0, &poly).unwrap();
        assert!(t_in.abs() < 1e-6);
        assert!((t_out - 1.0).abs() < 1e-6, "t_out={t_out}");
    }

    #[test]
    fn nan_serialization_roundtrip() {
        let v = super::ranges_to_json(&[1.5, f64::NAN, 2.0, f64::INFINITY]);
        let arr = v.as_array().expect("expected array");
        assert!(arr[0].is_number());
        assert!(arr[1].is_null());
        assert!(arr[2].is_number());
        assert!(arr[3].is_null());
    }
}
