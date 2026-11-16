use crate::detectors::{
    OscillationDetector, check_collision, check_footprint_collision,
    check_goal_reached_with_yaw, check_timeout, compute_circle_clearance,
    compute_footprint_clearance, is_near_miss,
};
use crate::map::OccupancyGrid;
use crate::raycaster::ResolvedObstacle;
use crate::robot_config::Footprint;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    pub success: bool,
    pub time_to_goal: f64,
    pub path_length: f64,
    pub path_efficiency: f64,
    pub collision: bool,
    pub collision_point: Option<(f64, f64)>,
    pub timeout: bool,
    pub oscillation_detected: bool,
    pub oscillation_duration: f64,
    pub min_clearance: f64,
    pub near_miss_count: u32,
    pub replan_count: u32,
    pub recovery_count: u32,
    pub avg_velocity: f64,
    pub steering_smoothness: f64,
    pub max_overshoot: f64,
    pub goal_radius: f64,
    pub trajectory: Vec<(f64, f64, f64, f64)>,
}

impl SimResult {
    fn quality_active(&self) -> Vec<(&'static str, f64)> {
        if !self.success {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(4);
        if self.oscillation_detected {
            out.push(("oscillatory", 0.85));
        }
        if self.max_overshoot > self.goal_radius {
            out.push(("overshooting", 0.85));
        }
        if self.path_efficiency > 0.0 && self.path_efficiency < 0.5 {
            out.push(("inefficient", 0.90));
        }
        if self.min_clearance < 0.05 && self.min_clearance != f64::INFINITY {
            out.push(("near_miss", 0.80));
        }
        out
    }

    pub fn nav_score(&self) -> f64 {
        if !self.success {
            if self.collision {
                return 0.0;
            }
            if self.timeout {
                return 0.2;
            }
            return 0.0;
        }
        self.quality_active()
            .iter()
            .fold(1.0_f64, |acc, (_, mult)| acc * mult)
    }

    pub fn failure_class(&self) -> &'static str {
        if self.collision {
            return "hard_fail_collision";
        }
        if self.timeout {
            return "hard_fail_timeout";
        }
        if !self.success {
            return "incomplete";
        }
        "pass"
    }

    pub fn quality_flags(&self) -> Vec<&'static str> {
        self.quality_active()
            .into_iter()
            .map(|(label, _)| label)
            .collect()
    }

    pub fn to_json_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("success".into(), self.success.into());
        map.insert("time_to_goal".into(), json_f64(self.time_to_goal));
        map.insert("path_length".into(), json_f64(self.path_length));
        map.insert("path_efficiency".into(), json_f64(self.path_efficiency));
        map.insert("collision".into(), self.collision.into());
        map.insert(
            "collision_point".into(),
            match self.collision_point {
                Some((x, y)) => serde_json::json!([x, y]),
                None => serde_json::Value::Null,
            },
        );
        map.insert("timeout".into(), self.timeout.into());
        map.insert(
            "oscillation_detected".into(),
            self.oscillation_detected.into(),
        );
        map.insert(
            "oscillation_duration".into(),
            json_f64(self.oscillation_duration),
        );
        map.insert("min_clearance".into(), json_f64(self.min_clearance));
        map.insert("near_miss_count".into(), self.near_miss_count.into());
        map.insert("replan_count".into(), self.replan_count.into());
        map.insert("recovery_count".into(), self.recovery_count.into());
        map.insert("avg_velocity".into(), json_f64(self.avg_velocity));
        map.insert(
            "steering_smoothness".into(),
            json_f64(self.steering_smoothness),
        );
        map.insert("max_overshoot".into(), json_f64(self.max_overshoot));
        map.insert("nav_score".into(), json_f64(self.nav_score()));
        map.insert("failure_class".into(), self.failure_class().into());
        map.insert(
            "quality_flags".into(),
            serde_json::Value::Array(
                self.quality_flags()
                    .into_iter()
                    .map(|s| serde_json::Value::String(s.into()))
                    .collect(),
            ),
        );
        serde_json::Value::Object(map)
    }
}

fn json_f64(v: f64) -> serde_json::Value {
    if v.is_infinite() || v.is_nan() {
        serde_json::Value::Null
    } else {
        serde_json::json!(v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminateReason {
    Collision,
    GoalReached,
    Timeout,
}

pub struct SimEvaluator {
    pub goal: Option<(f64, f64)>,
    pub goal_theta: Option<f64>,
    pub yaw_tolerance: f64,
    pub robot_radius: f64,
    pub timeout_budget: f64,
    pub goal_radius: f64,
    pub near_miss_threshold: f64,
    pub obstacles: Vec<ResolvedObstacle>,
    pub footprint: Option<Footprint>,
    pub terminated: bool,
    terminate_reason: Option<TerminateReason>,
    collision_point: Option<(f64, f64)>,
    cached_result: Option<SimResult>,
    trajectory: Vec<(f64, f64, f64, f64)>,
    path_length: f64,
    last_pos: Option<(f64, f64)>,
    min_clearance: f64,
    near_miss_count: u32,
    velocity_sum: f64,
    angular_velocities: Vec<f64>,
    tick_count: u64,
    first_motion_time: Option<f64>,
    oscillation: OscillationDetector,
    goal_first_reached_time: Option<f64>,
    settling_ticks_remaining: i32,
    max_overshoot: f64,
    grid_cells: Vec<bool>,
    grid_width: usize,
    grid_height: usize,
    grid_resolution: f64,
    grid_origin: [f64; 3],
}

impl SimEvaluator {
    pub fn new(
        grid: &OccupancyGrid,
        goal: Option<(f64, f64)>,
        goal_theta: Option<f64>,
        yaw_tolerance: f64,
        robot_radius: f64,
        timeout_budget: f64,
        goal_radius: f64,
        near_miss_threshold: f64,
        obstacles: Vec<ResolvedObstacle>,
        footprint: Option<Footprint>,
    ) -> Self {
        Self {
            goal,
            goal_theta,
            yaw_tolerance,
            robot_radius,
            timeout_budget,
            goal_radius,
            near_miss_threshold,
            obstacles,
            footprint,
            terminated: false,
            terminate_reason: None,
            collision_point: None,
            cached_result: None,
            trajectory: Vec::new(),
            path_length: 0.0,
            last_pos: None,
            min_clearance: f64::INFINITY,
            near_miss_count: 0,
            velocity_sum: 0.0,
            angular_velocities: Vec::new(),
            tick_count: 0,
            first_motion_time: None,
            oscillation: OscillationDetector::new(3.0, 3),
            goal_first_reached_time: None,
            settling_ticks_remaining: -1,
            max_overshoot: 0.0,
            grid_cells: grid.cells.clone(),
            grid_width: grid.width,
            grid_height: grid.height,
            grid_resolution: grid.metadata.resolution,
            grid_origin: grid.metadata.origin,
        }
    }

    fn grid(&self) -> OccupancyGrid {
        OccupancyGrid {
            cells: self.grid_cells.clone(),
            width: self.grid_width,
            height: self.grid_height,
            metadata: crate::map::MapMetadata {
                resolution: self.grid_resolution,
                origin: self.grid_origin,
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        }
    }

    pub fn tick(&mut self, x: f64, y: f64, theta: f64, sim_time: f64, linear: f64, angular: f64) {
        if self.terminated {
            return;
        }
        self.tick_count += 1;
        self.trajectory.push((x, y, theta, sim_time));
        self.velocity_sum += linear.abs();
        self.angular_velocities.push(angular);
        if self.first_motion_time.is_none() && (linear.abs() > 1e-4 || angular.abs() > 1e-4) {
            self.first_motion_time = Some(sim_time);
        }
        if let Some((lx, ly)) = self.last_pos {
            self.path_length += ((x - lx).powi(2) + (y - ly).powi(2)).sqrt();
        }
        self.last_pos = Some((x, y));
        let grid = self.grid();
        let collision = if let Some(ref fp) = self.footprint {
            check_footprint_collision(&grid, x, y, theta, fp, &self.obstacles)
        } else {
            check_collision(&grid, x, y, self.robot_radius, &self.obstacles)
        };
        if let Some(point) = collision {
            self.terminated = true;
            self.terminate_reason = Some(TerminateReason::Collision);
            self.collision_point = Some(point);
            return;
        }
        if self.settling_ticks_remaining >= 0 {
            if let Some((gx, gy)) = self.goal {
                let dist = ((x - gx).powi(2) + (y - gy).powi(2)).sqrt();
                if dist > self.max_overshoot {
                    self.max_overshoot = dist;
                }
            }
            self.settling_ticks_remaining -= 1;
            if self.settling_ticks_remaining < 0 {
                self.terminated = true;
                self.terminate_reason = Some(TerminateReason::GoalReached);
                return;
            }
        } else if let Some((gx, gy)) = self.goal {
            if check_goal_reached_with_yaw(
                x, y, theta, gx, gy, self.goal_theta,
                self.goal_radius, self.yaw_tolerance,
            ) {
                self.goal_first_reached_time = Some(sim_time);
                self.settling_ticks_remaining = 40;
                self.max_overshoot = ((x - gx).powi(2) + (y - gy).powi(2)).sqrt();
            }
        }
        let elapsed = match self.first_motion_time {
            Some(t0) => sim_time - t0,
            None => 0.0,
        };
        if check_timeout(elapsed, self.timeout_budget) {
            self.terminated = true;
            self.terminate_reason = Some(TerminateReason::Timeout);
            return;
        }
        if !self.obstacles.is_empty() {
            let clearance = if let Some(ref fp) = self.footprint {
                compute_footprint_clearance(x, y, theta, fp, &self.obstacles)
            } else {
                compute_circle_clearance(x, y, self.robot_radius, &self.obstacles)
            };
            self.min_clearance = self.min_clearance.min(clearance);
            if is_near_miss(clearance, self.near_miss_threshold) {
                self.near_miss_count += 1;
            }
        }
        self.oscillation.update(x, y, sim_time);
    }

    pub fn is_collided(&self) -> bool {
        self.terminate_reason == Some(TerminateReason::Collision)
    }

    pub fn is_timed_out(&self) -> bool {
        self.terminate_reason == Some(TerminateReason::Timeout)
    }

    pub fn is_oscillating(&self) -> bool {
        self.oscillation.oscillating
    }

    pub fn is_goal_reached(&self) -> bool {
        self.terminate_reason == Some(TerminateReason::GoalReached)
    }

    pub fn near_miss_count(&self) -> u32 {
        self.near_miss_count
    }

    pub fn max_overshoot(&self) -> f64 {
        self.max_overshoot
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn goal_first_reached_time(&self) -> Option<f64> {
        self.goal_first_reached_time
    }

    pub fn metrics(&mut self) -> serde_json::Value {
        let elapsed = self.trajectory.last().map_or(0.0, |t| t.3);
        let result_val = if self.terminated {
            self.get_cached_result()
        } else {
            serde_json::Value::Null
        };
        serde_json::json!({
            "status": if self.terminated { "terminated" } else { "running" },
            "elapsed": (elapsed * 10000.0).round() / 10000.0,
            "path_length": (self.path_length * 10000.0).round() / 10000.0,
            "near_miss_count": self.near_miss_count,
            "oscillation_detected": self.oscillation.oscillating,
            "settling": self.settling_ticks_remaining >= 0,
            "max_overshoot": (self.max_overshoot * 10000.0).round() / 10000.0,
            "result": result_val,
        })
    }

    pub fn get_cached_result(&mut self) -> serde_json::Value {
        if let Some(ref r) = self.cached_result {
            return r.to_json_value();
        }
        let r = self.result();
        let val = r.to_json_value();
        self.cached_result = Some(r);
        val
    }

    pub fn result(&self) -> SimResult {
        let success = self.terminate_reason == Some(TerminateReason::GoalReached);
        let start_time = self.trajectory.first().map_or(0.0, |t| t.3);
        let time_to_goal = self
            .goal_first_reached_time
            .map(|t| t - start_time)
            .unwrap_or_else(|| {
                self.trajectory
                    .last()
                    .map_or(f64::INFINITY, |t| t.3 - start_time)
            });
        let optimal = if let (Some(goal), Some(first)) = (self.goal, self.trajectory.first()) {
            ((goal.0 - first.0).powi(2) + (goal.1 - first.1).powi(2)).sqrt()
        } else {
            0.0
        };
        let path_efficiency = if self.path_length > 0.0 {
            (optimal / self.path_length).min(1.0)
        } else {
            0.0
        };
        let avg_vel = if self.tick_count > 0 {
            self.velocity_sum / self.tick_count as f64
        } else {
            0.0
        };
        let steering_smoothness = if self.angular_velocities.len() >= 2 {
            let mean =
                self.angular_velocities.iter().sum::<f64>() / self.angular_velocities.len() as f64;
            let variance = self
                .angular_velocities
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / self.angular_velocities.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        SimResult {
            success,
            time_to_goal: if success { time_to_goal } else { f64::INFINITY },
            path_length: (self.path_length * 10000.0).round() / 10000.0,
            path_efficiency: (path_efficiency * 10000.0).round() / 10000.0,
            collision: self.terminate_reason == Some(TerminateReason::Collision),
            collision_point: self.collision_point,
            timeout: self.terminate_reason == Some(TerminateReason::Timeout),
            oscillation_detected: self.oscillation.oscillating,
            oscillation_duration: (self.oscillation.duration * 10000.0).round() / 10000.0,
            min_clearance: if self.min_clearance == f64::INFINITY {
                f64::INFINITY
            } else {
                (self.min_clearance * 10000.0).round() / 10000.0
            },
            near_miss_count: self.near_miss_count,
            replan_count: 0,
            recovery_count: 0,
            avg_velocity: (avg_vel * 10000.0).round() / 10000.0,
            steering_smoothness: (steering_smoothness * 10000.0).round() / 10000.0,
            max_overshoot: (self.max_overshoot * 10000.0).round() / 10000.0,
            goal_radius: self.goal_radius,
            trajectory: self.trajectory.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_grid() -> OccupancyGrid {
        OccupancyGrid {
            cells: vec![false; 10000],
            width: 100,
            height: 100,
            metadata: crate::map::MapMetadata {
                resolution: 0.1,
                origin: [0.0, 0.0, 0.0],
                occupied_thresh: 0.65,
                free_thresh: 0.196,
                negate: false,
                image_path: std::path::PathBuf::new(),
            },
        }
    }

    #[test]
    fn test_goal_reached_with_settling() {
        let grid = free_grid();
        let mut eval =
            SimEvaluator::new(&grid, Some((5.0, 5.0)), None, std::f64::consts::PI, 0.25, 60.0, 0.3, 0.05, vec![], None);
        eval.tick(4.9, 5.0, 0.0, 1.0, 0.5, 0.0);
        assert!(
            !eval.terminated,
            "should not terminate immediately — settling"
        );
        assert!(eval.settling_ticks_remaining > 0);
        for i in 1..=41 {
            eval.tick(4.9, 5.0, 0.0, 1.0 + i as f64 * 0.05, 0.0, 0.0);
        }
        assert!(eval.terminated);
        let result = eval.result();
        assert!(result.success);
        assert_eq!(result.failure_class(), "pass");
        assert!((result.nav_score() - 1.0).abs() < 1e-6);
        assert!(
            result.max_overshoot < 0.3,
            "no overshoot when staying at goal"
        );
    }

    #[test]
    fn test_overshoot_detected() {
        let grid = free_grid();
        let mut eval =
            SimEvaluator::new(&grid, Some((5.0, 5.0)), None, std::f64::consts::PI, 0.25, 60.0, 0.3, 0.05, vec![], None);
        eval.tick(4.9, 5.0, 0.0, 1.0, 0.5, 0.0);
        assert!(!eval.terminated);
        eval.tick(5.8, 5.0, 0.0, 1.05, 0.5, 0.0);
        for i in 2..=41 {
            eval.tick(5.8, 5.0, 0.0, 1.0 + i as f64 * 0.05, 0.0, 0.0);
        }
        assert!(eval.terminated);
        let result = eval.result();
        assert!(result.success);
        assert!(
            result.max_overshoot > 0.5,
            "max_overshoot={} should be >0.5",
            result.max_overshoot
        );
        assert_eq!(result.failure_class(), "pass");
        let flags = result.quality_flags();
        assert!(
            flags.contains(&"overshooting"),
            "expected overshooting flag, got {:?}",
            flags
        );
        assert!(result.nav_score() > 0.2);
    }

    #[test]
    fn test_timeout() {
        let grid = free_grid();
        let mut eval = SimEvaluator::new(
            &grid,
            Some((50.0, 50.0)),
            None,
            std::f64::consts::PI,
            0.25,
            10.0,
            0.3,
            0.05,
            vec![],
            None,
        );
        eval.tick(1.0, 1.0, 0.0, 0.0, 0.5, 0.0);
        assert!(!eval.terminated);
        eval.tick(1.0, 1.0, 0.0, 10.0, 0.5, 0.0);
        assert!(eval.terminated);
        let result = eval.result();
        assert!(result.timeout);
        assert_eq!(result.failure_class(), "hard_fail_timeout");
    }

    #[test]
    fn test_collision_terminates() {
        let grid = free_grid();
        let obs = vec![ResolvedObstacle::Circle {
            cx: 1.1,
            cy: 1.0,
            radius: 0.15,
        }];
        let mut eval = SimEvaluator::new(&grid, None, None, std::f64::consts::PI, 0.25, 60.0, 0.3, 0.05, obs, None);
        eval.tick(1.0, 1.0, 0.0, 0.1, 0.5, 0.0);
        assert!(eval.terminated);
        let result = eval.result();
        assert!(result.collision);
        assert_eq!(result.failure_class(), "hard_fail_collision");
        assert!((result.nav_score() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_path_length_accumulates() {
        let grid = free_grid();
        let mut eval = SimEvaluator::new(
            &grid,
            Some((50.0, 50.0)),
            None,
            std::f64::consts::PI,
            0.25,
            60.0,
            0.3,
            0.05,
            vec![],
            None,
        );
        eval.tick(1.0, 1.0, 0.0, 0.1, 0.5, 0.0);
        eval.tick(2.0, 1.0, 0.0, 0.2, 0.5, 0.0);
        eval.tick(3.0, 1.0, 0.0, 0.3, 0.5, 0.0);
        let result = eval.result();
        assert!(
            (result.path_length - 2.0).abs() < 0.01,
            "path_length={}",
            result.path_length
        );
    }

    #[test]
    fn test_no_goal_no_terminate_on_goal() {
        let grid = free_grid();
        let mut eval = SimEvaluator::new(&grid, None, None, std::f64::consts::PI, 0.25, 60.0, 0.3, 0.05, vec![], None);
        eval.tick(5.0, 5.0, 0.0, 1.0, 0.0, 0.0);
        assert!(!eval.terminated);
    }
}
