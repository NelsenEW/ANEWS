use crate::raycaster::ResolvedObstacle;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvoidanceConfig {
    pub enabled: bool,
    pub stop_dist: f64,
    pub slow_dist: f64,
    pub slow_factor: f64,
}

impl Default for AvoidanceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stop_dist: 0.1,
            slow_dist: 0.5,
            slow_factor: 0.3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanActor {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub radius: f64,
    pub speed: f64,
    pub path: Vec<(f64, f64)>,
    pub loop_path: bool,
    pub pause_at_waypoints: f64,
    pub start_delay: f64,
    pub path_idx: usize,
    pub path_t: f64,
    pub pause_timer: f64,
    pub delay_timer: f64,
    pub enabled: bool,
    pub active: bool,
    pub finished: bool,
    pub avoidance: AvoidanceConfig,
}

pub type AvoidanceObstacle = (f64, f64, f64);

impl HumanActor {
    pub fn new(
        name: String,
        path: Vec<(f64, f64)>,
        radius: f64,
        speed: f64,
        loop_path: bool,
        pause_at_waypoints: f64,
        start_delay: f64,
        enabled: bool,
        avoidance: AvoidanceConfig,
    ) -> Self {
        let x = path.first().map_or(0.0, |p| p.0);
        let y = path.first().map_or(0.0, |p| p.1);
        let theta = if path.len() >= 2 {
            (path[1].1 - path[0].1).atan2(path[1].0 - path[0].0)
        } else {
            0.0
        };
        let active = start_delay <= 0.0;
        Self {
            name,
            x,
            y,
            theta,
            radius,
            speed,
            path,
            loop_path,
            pause_at_waypoints,
            start_delay,
            path_idx: 0,
            path_t: 0.0,
            pause_timer: 0.0,
            delay_timer: 0.0,
            enabled,
            active,
            finished: false,
            avoidance,
        }
    }

    pub fn update(&mut self, dt: f64, obstacles: &[AvoidanceObstacle]) {
        if !self.enabled || self.finished {
            return;
        }
        if !self.active {
            self.delay_timer += dt;
            if self.delay_timer >= self.start_delay {
                self.active = true;
                if !self.path.is_empty() {
                    self.x = self.path[0].0;
                    self.y = self.path[0].1;
                }
            }
            return;
        }
        if self.path.len() < 2 {
            return;
        }
        if self.pause_timer > 0.0 {
            self.pause_timer -= dt;
            return;
        }
        let mut effective_dt = dt;
        if self.avoidance.enabled && !obstacles.is_empty() {
            let mut min_clearance = f64::INFINITY;
            for &(ox, oy, orad) in obstacles {
                let clearance =
                    ((self.x - ox).powi(2) + (self.y - oy).powi(2)).sqrt() - self.radius - orad;
                min_clearance = min_clearance.min(clearance);
            }
            if min_clearance < self.avoidance.stop_dist {
                return;
            }
            if min_clearance < self.avoidance.slow_dist {
                effective_dt *= self.avoidance.slow_factor;
            }
        }
        let from_pt = self.path[self.path_idx];
        let to_idx = (self.path_idx + 1) % self.path.len();
        let to_pt = self.path[to_idx];
        let dx = to_pt.0 - from_pt.0;
        let dy = to_pt.1 - from_pt.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 0.001 {
            self.advance_waypoint();
            return;
        }
        self.theta = dy.atan2(dx);
        self.path_t += (self.speed * effective_dt) / dist;
        if self.path_t >= 1.0 {
            self.x = to_pt.0;
            self.y = to_pt.1;
            self.advance_waypoint();
        } else {
            self.x = from_pt.0 + dx * self.path_t;
            self.y = from_pt.1 + dy * self.path_t;
        }
    }

    fn advance_waypoint(&mut self) {
        self.path_t = 0.0;
        let next_idx = self.path_idx + 1;
        if !self.loop_path && next_idx >= self.path.len() - 1 {
            self.path_idx = self.path.len() - 1;
            self.finished = true;
            return;
        }
        self.path_idx = next_idx % self.path.len();
        self.pause_timer = self.pause_at_waypoints;
    }

    pub fn to_state_dict(&self) -> serde_json::Value {
        let state_str = if !self.enabled {
            "disabled"
        } else if self.finished {
            "finished"
        } else if !self.active && self.start_delay > 0.0 {
            "delayed"
        } else if !self.active {
            "inactive"
        } else if self.pause_timer > 0.0 {
            "paused"
        } else {
            "walking"
        };
        serde_json::json!({
            "name": self.name, "x": self.x, "y": self.y,
            "theta": (self.theta * 10000.0).round() / 10000.0,
            "radius": self.radius, "enabled": self.enabled, "active": self.active,
            "finished": self.finished, "speed": self.speed, "loop": self.loop_path,
            "state": state_str,
            "path_progress": {
                "waypoint_index": self.path_idx, "total_waypoints": self.path.len(),
                "segment_percent": (self.path_t * 10000.0).round() / 100.0,
            },
        })
    }
}

pub fn actors_to_obstacles(actors: &[HumanActor]) -> Vec<ResolvedObstacle> {
    let mut obstacles = Vec::new();
    for a in actors {
        if a.enabled && a.active && !a.finished {
            let leg_r = a.radius * 0.4;
            let fwd = a.radius * 0.5;
            let perp = a.radius * 0.4;
            let fx = a.theta.cos() * fwd;
            let fy = a.theta.sin() * fwd;
            let px = -a.theta.sin() * perp;
            let py = a.theta.cos() * perp;
            obstacles.push(ResolvedObstacle::Circle {
                cx: a.x + fx + px,
                cy: a.y + fy + py,
                radius: leg_r,
            });
            obstacles.push(ResolvedObstacle::Circle {
                cx: a.x + fx - px,
                cy: a.y + fy - py,
                radius: leg_r,
            });
        }
    }
    obstacles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_actor(path: Vec<(f64, f64)>, loop_path: bool) -> HumanActor {
        HumanActor::new(
            "test".into(),
            path,
            0.3,
            1.0,
            loop_path,
            0.0,
            0.0,
            true,
            AvoidanceConfig::default(),
        )
    }

    #[test]
    fn test_straight_walk() {
        let mut actor = make_actor(vec![(0.0, 0.0), (10.0, 0.0)], false);
        assert!(actor.active);
        actor.update(1.0, &[]);
        assert!((actor.x - 1.0).abs() < 0.01, "x={}", actor.x);
        assert!(actor.y.abs() < 0.01);
    }

    #[test]
    fn test_finish_non_looping() {
        let mut actor = make_actor(vec![(0.0, 0.0), (1.0, 0.0)], false);
        actor.update(2.0, &[]);
        assert!(actor.finished);
        assert!((actor.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_looping_wraps() {
        let mut actor = make_actor(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], true);
        actor.update(1.5, &[]);
        assert!(!actor.finished);
        assert!(actor.path_idx >= 1);
    }

    #[test]
    fn test_start_delay() {
        let mut actor = HumanActor::new(
            "delayed".into(),
            vec![(0.0, 0.0), (10.0, 0.0)],
            0.3,
            1.0,
            false,
            0.0,
            5.0,
            true,
            AvoidanceConfig::default(),
        );
        assert!(!actor.active);
        actor.update(3.0, &[]);
        assert!(!actor.active);
        actor.update(3.0, &[]);
        assert!(actor.active);
    }

    #[test]
    fn test_avoidance_stop() {
        let mut actor = make_actor(vec![(0.0, 0.0), (10.0, 0.0)], false);
        actor.avoidance = AvoidanceConfig {
            enabled: true,
            stop_dist: 0.5,
            slow_dist: 1.0,
            slow_factor: 0.3,
        };
        let old_x = actor.x;
        actor.update(1.0, &[(0.5, 0.0, 0.1)]);
        assert!((actor.x - old_x).abs() < 0.01, "Should have stopped");
    }

    #[test]
    fn test_actors_to_obstacles() {
        let actor = make_actor(vec![(5.0, 5.0), (10.0, 5.0)], false);
        let obs = actors_to_obstacles(&[actor]);
        assert_eq!(obs.len(), 2);
        for o in &obs {
            match o {
                ResolvedObstacle::Circle { radius, .. } => assert!((*radius - 0.12).abs() < 0.01),
                _ => panic!("Expected circle"),
            }
        }
    }

    #[test]
    fn test_disabled_actor_no_obstacles() {
        let mut actor = make_actor(vec![(5.0, 5.0), (10.0, 5.0)], false);
        actor.enabled = false;
        let obs = actors_to_obstacles(&[actor]);
        assert!(obs.is_empty());
    }
}
