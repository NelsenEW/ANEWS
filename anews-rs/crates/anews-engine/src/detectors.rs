use crate::event_stream::EventBus;
use crate::events::SimEvent;
use crate::map::OccupancyGrid;
use crate::raycaster::ResolvedObstacle;
use crate::robot_config::Footprint;
use std::collections::VecDeque;

pub fn rotate_footprint(footprint: &Footprint, x: f64, y: f64, theta: f64) -> Vec<(f64, f64)> {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    footprint
        .iter()
        .map(|(lx, ly)| (x + lx * cos_t - ly * sin_t, y + lx * sin_t + ly * cos_t))
        .collect()
}

pub fn point_in_convex_polygon(px: f64, py: f64, verts: &[(f64, f64)]) -> bool {
    let n = verts.len();
    let mut positive = false;
    let mut negative = false;
    for i in 0..n {
        let j = (i + 1) % n;
        let ex = verts[j].0 - verts[i].0;
        let ey = verts[j].1 - verts[i].1;
        let cross = ex * (py - verts[i].1) - ey * (px - verts[i].0);
        if cross > 0.0 {
            positive = true;
        } else if cross < 0.0 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

fn polygon_circle_intersect(verts: &[(f64, f64)], cx: f64, cy: f64, r: f64) -> bool {
    if point_in_convex_polygon(cx, cy, verts) {
        return true;
    }
    let n = verts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, ay) = verts[i];
        let (bx, by) = verts[j];
        let abx = bx - ax;
        let aby = by - ay;
        let t = ((cx - ax) * abx + (cy - ay) * aby) / (abx * abx + aby * aby + 1e-12);
        let t = t.clamp(0.0, 1.0);
        let nx = ax + t * abx;
        let ny = ay + t * aby;
        if ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt() < r {
            return true;
        }
    }
    false
}

fn polygon_rect_intersect(
    verts: &[(f64, f64)],
    rect_cx: f64,
    rect_cy: f64,
    rect_hw: f64,
    rect_hh: f64,
) -> bool {
    let mut axes: Vec<(f64, f64)> = Vec::new();
    let n = verts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let ex = verts[j].0 - verts[i].0;
        let ey = verts[j].1 - verts[i].1;
        let length = (ex * ex + ey * ey).sqrt();
        if length > 1e-12 {
            axes.push((-ey / length, ex / length));
        }
    }
    axes.push((1.0, 0.0));
    axes.push((0.0, 1.0));
    let rect_verts = [
        (rect_cx - rect_hw, rect_cy - rect_hh),
        (rect_cx + rect_hw, rect_cy - rect_hh),
        (rect_cx + rect_hw, rect_cy + rect_hh),
        (rect_cx - rect_hw, rect_cy + rect_hh),
    ];
    for (ax, ay) in &axes {
        let mut poly_min = f64::INFINITY;
        let mut poly_max = f64::NEG_INFINITY;
        for (vx, vy) in verts {
            let proj = vx * ax + vy * ay;
            poly_min = poly_min.min(proj);
            poly_max = poly_max.max(proj);
        }
        let mut rect_min = f64::INFINITY;
        let mut rect_max = f64::NEG_INFINITY;
        for (vx, vy) in &rect_verts {
            let proj = vx * ax + vy * ay;
            rect_min = rect_min.min(proj);
            rect_max = rect_max.max(proj);
        }
        if poly_max < rect_min || rect_max < poly_min {
            return false;
        }
    }
    true
}

fn obstacle_distance(robot_x: f64, robot_y: f64, obs: &ResolvedObstacle) -> (f64, f64, f64, f64) {
    match obs {
        ResolvedObstacle::Circle { cx, cy, radius } => {
            let dist = ((cx - robot_x).powi(2) + (cy - robot_y).powi(2)).sqrt();
            (dist, *radius, *cx, *cy)
        }
        ResolvedObstacle::Rect {
            cx,
            cy,
            half_w,
            half_h,
            ..
        } => {
            let closest_x = robot_x.clamp(cx - half_w, cx + half_w);
            let closest_y = robot_y.clamp(cy - half_h, cy + half_h);
            let dist = ((closest_x - robot_x).powi(2) + (closest_y - robot_y).powi(2)).sqrt();
            (dist, 0.0, closest_x, closest_y)
        }
    }
}

pub fn check_collision(
    grid: &OccupancyGrid,
    robot_x: f64,
    robot_y: f64,
    radius: f64,
    obstacles: &[ResolvedObstacle],
) -> Option<(f64, f64)> {
    let resolution = grid.metadata.resolution;
    let origin = &grid.metadata.origin;
    let cells_radius = (radius / resolution).ceil() as i32 + 1;
    let cx = ((robot_x - origin[0]) / resolution) as i32;
    let cy = ((robot_y - origin[1]) / resolution) as i32;
    for dy in -cells_radius..=cells_radius {
        for dx in -cells_radius..=cells_radius {
            let gx = cx + dx;
            let gy = cy + dy;
            if gx >= 0 && (gx as usize) < grid.width && gy >= 0 && (gy as usize) < grid.height {
                if grid.cells[gy as usize * grid.width + gx as usize] {
                    let wx = origin[0] + (gx as f64 + 0.5) * resolution;
                    let wy = origin[1] + (gy as f64 + 0.5) * resolution;
                    if ((wx - robot_x).powi(2) + (wy - robot_y).powi(2)).sqrt() < radius {
                        return Some((wx, wy));
                    }
                }
            }
        }
    }
    for obs in obstacles {
        let (dist, obs_radius, contact_x, contact_y) = obstacle_distance(robot_x, robot_y, obs);
        if dist < radius + obs_radius {
            return Some((contact_x, contact_y));
        }
    }
    None
}

pub fn check_footprint_collision(
    grid: &OccupancyGrid,
    robot_x: f64,
    robot_y: f64,
    robot_theta: f64,
    footprint: &Footprint,
    obstacles: &[ResolvedObstacle],
) -> Option<(f64, f64)> {
    let verts = rotate_footprint(footprint, robot_x, robot_y, robot_theta);
    let resolution = grid.metadata.resolution;
    let origin = &grid.metadata.origin;
    let min_x = verts.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
    let min_y = verts.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
    let max_x = verts.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
    let max_y = verts.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);
    let gx_lo = ((min_x - origin[0]) / resolution) as i32 - 1;
    let gx_hi = ((max_x - origin[0]) / resolution) as i32 + 1;
    let gy_lo = ((min_y - origin[1]) / resolution) as i32 - 1;
    let gy_hi = ((max_y - origin[1]) / resolution) as i32 + 1;
    let gx_lo = gx_lo.max(0) as usize;
    let gx_hi = (gx_hi as usize).min(grid.width - 1);
    let gy_lo = gy_lo.max(0) as usize;
    let gy_hi = (gy_hi as usize).min(grid.height - 1);
    for gy in gy_lo..=gy_hi {
        for gx in gx_lo..=gx_hi {
            if grid.cells[gy * grid.width + gx] {
                let wx = origin[0] + (gx as f64 + 0.5) * resolution;
                let wy = origin[1] + (gy as f64 + 0.5) * resolution;
                if point_in_convex_polygon(wx, wy, &verts) {
                    return Some((wx, wy));
                }
            }
        }
    }
    for obs in obstacles {
        match obs {
            ResolvedObstacle::Circle { cx, cy, radius } => {
                if polygon_circle_intersect(&verts, *cx, *cy, *radius) {
                    return Some((*cx, *cy));
                }
            }
            ResolvedObstacle::Rect {
                cx,
                cy,
                half_w,
                half_h,
                ..
            } => {
                if polygon_rect_intersect(&verts, *cx, *cy, *half_w, *half_h) {
                    return Some((*cx, *cy));
                }
            }
        }
    }
    None
}

pub fn compute_footprint_clearance(
    robot_x: f64,
    robot_y: f64,
    robot_theta: f64,
    footprint: &Footprint,
    obstacles: &[ResolvedObstacle],
) -> f64 {
    let verts = rotate_footprint(footprint, robot_x, robot_y, robot_theta);
    let n = verts.len();
    let mut min_clearance = f64::INFINITY;
    for obs in obstacles {
        match obs {
            ResolvedObstacle::Circle { cx, cy, radius } => {
                for i in 0..n {
                    let j = (i + 1) % n;
                    let (ax, ay) = verts[i];
                    let (bx, by) = verts[j];
                    let abx = bx - ax;
                    let aby = by - ay;
                    let t = ((cx - ax) * abx + (cy - ay) * aby) / (abx * abx + aby * aby + 1e-12);
                    let t = t.clamp(0.0, 1.0);
                    let dist = ((ax + t * abx - cx).powi(2) + (ay + t * aby - cy).powi(2)).sqrt();
                    min_clearance = min_clearance.min(dist - radius);
                }
            }
            ResolvedObstacle::Rect {
                cx,
                cy,
                half_w,
                half_h,
                ..
            } => {
                for i in 0..n {
                    let (px, py) = verts[i];
                    let near_x = px.clamp(cx - half_w, cx + half_w);
                    let near_y = py.clamp(cy - half_h, cy + half_h);
                    let dist = ((px - near_x).powi(2) + (py - near_y).powi(2)).sqrt();
                    min_clearance = min_clearance.min(dist);
                }
            }
        }
    }
    min_clearance
}

pub fn compute_circle_clearance(
    robot_x: f64,
    robot_y: f64,
    robot_radius: f64,
    obstacles: &[ResolvedObstacle],
) -> f64 {
    let mut min_clearance = f64::INFINITY;
    for obs in obstacles {
        let (dist, obs_radius, _, _) = obstacle_distance(robot_x, robot_y, obs);
        min_clearance = min_clearance.min(dist - robot_radius - obs_radius);
    }
    min_clearance
}

pub fn check_goal_reached(
    robot_x: f64,
    robot_y: f64,
    goal_x: f64,
    goal_y: f64,
    radius: f64,
) -> bool {
    ((robot_x - goal_x).powi(2) + (robot_y - goal_y).powi(2)).sqrt() <= radius
}

pub fn check_goal_reached_with_yaw(
    robot_x: f64,
    robot_y: f64,
    robot_theta: f64,
    goal_x: f64,
    goal_y: f64,
    goal_theta: Option<f64>,
    radius: f64,
    yaw_tolerance: f64,
) -> bool {
    if !check_goal_reached(robot_x, robot_y, goal_x, goal_y, radius) {
        return false;
    }
    let Some(gt) = goal_theta else { return true };
    if !yaw_tolerance.is_finite() || yaw_tolerance >= std::f64::consts::PI {
        return true;
    }
    let mut diff = robot_theta - gt;
    diff = diff.rem_euclid(2.0 * std::f64::consts::PI);
    if diff > std::f64::consts::PI {
        diff -= 2.0 * std::f64::consts::PI;
    }
    diff.abs() <= yaw_tolerance
}

pub fn check_timeout(sim_time: f64, budget: f64) -> bool {
    sim_time >= budget
}

pub fn is_near_miss(clearance: f64, threshold: f64) -> bool {
    clearance < threshold && clearance != f64::INFINITY
}

pub struct OscillationDetector {
    window_sec: f64,
    min_reversals: usize,
    min_segment_len: f64,
    history: VecDeque<(f64, f64, f64)>,
    pub oscillating: bool,
    pub duration: f64,
    oscillation_start: Option<f64>,
}

impl OscillationDetector {
    pub fn new(window_sec: f64, min_reversals: usize) -> Self {
        Self {
            window_sec,
            min_reversals,
            min_segment_len: 0.05,
            history: VecDeque::new(),
            oscillating: false,
            duration: 0.0,
            oscillation_start: None,
        }
    }
    pub fn update(&mut self, x: f64, y: f64, t: f64) {
        self.history.push_back((x, y, t));
        let cutoff = t - self.window_sec;
        while let Some(front) = self.history.front() {
            if front.2 < cutoff {
                self.history.pop_front();
            } else {
                break;
            }
        }
        if self.history.len() < 4 {
            return;
        }

        let positions: Vec<(f64, f64)> = self.history.iter().map(|(x, y, _)| (*x, *y)).collect();
        let mut segments: Vec<(f64, f64)> = Vec::new();
        for i in 1..positions.len() {
            let dx = positions[i].0 - positions[i - 1].0;
            let dy = positions[i].1 - positions[i - 1].1;
            let len = (dx * dx + dy * dy).sqrt();
            if len >= self.min_segment_len {
                segments.push((dx, dy));
            }
        }

        if segments.len() < 2 {
            self.oscillating = false;
            self.duration = 0.0;
            self.oscillation_start = None;
            return;
        }

        let mut reversals = 0;
        for i in 1..segments.len() {
            let dot = segments[i].0 * segments[i - 1].0 + segments[i].1 * segments[i - 1].1;
            if dot < 0.0 {
                reversals += 1;
            }
        }

        if reversals >= self.min_reversals {
            if !self.oscillating {
                self.oscillation_start = self.history.front().map(|(_, _, t)| *t);
            }
            self.oscillating = true;
            self.duration = t - self.oscillation_start.unwrap_or(t);
        } else {
            self.oscillating = false;
            self.duration = 0.0;
            self.oscillation_start = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(point_in_convex_polygon(0.5, 0.5, &square));
        assert!(!point_in_convex_polygon(2.0, 2.0, &square));
    }

    #[test]
    fn test_check_collision_no_collision() {
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
        assert!(check_collision(&grid, 5.0, 5.0, 0.3, &[]).is_none());
    }

    #[test]
    fn test_check_collision_with_wall() {
        let mut cells = vec![false; 100];
        cells[5 * 10 + 5] = true;
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
        assert!(check_collision(&grid, 5.5, 5.5, 1.0, &[]).is_some());
    }

    #[test]
    fn test_check_collision_with_obstacle() {
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
        let obs = vec![ResolvedObstacle::Circle {
            cx: 5.1,
            cy: 5.0,
            radius: 0.1,
        }];
        assert!(check_collision(&grid, 5.0, 5.0, 0.25, &obs).is_some());
    }

    #[test]
    fn test_goal_reached() {
        assert!(check_goal_reached(5.0, 5.0, 5.1, 5.1, 0.3));
        assert!(!check_goal_reached(5.0, 5.0, 10.0, 10.0, 0.3));
    }

    #[test]
    fn test_oscillation_detector_real_oscillation() {
        let mut det = OscillationDetector::new(3.0, 3);
        let positions = [
            (0.0, 0.0),
            (0.3, 0.0),
            (0.5, 0.0),
            (0.3, 0.0),
            (0.1, 0.0),
            (0.3, 0.0),
            (0.5, 0.0),
            (0.3, 0.0),
            (0.1, 0.0),
            (0.3, 0.0),
        ];
        for (i, (x, y)) in positions.iter().enumerate() {
            det.update(*x, *y, i as f64 * 0.3);
        }
        assert!(det.oscillating);
        assert!(det.duration > 0.0);
    }

    #[test]
    fn test_no_oscillation_moving() {
        let mut det = OscillationDetector::new(3.0, 3);
        for i in 0..40 {
            det.update(i as f64 * 0.1, 0.0, i as f64 * 0.1);
        }
        assert!(!det.oscillating);
    }

    #[test]
    fn test_no_oscillation_point_turn() {
        let mut det = OscillationDetector::new(3.0, 3);
        for i in 0..40 {
            let theta = i as f64 * 0.15;
            det.update(theta.cos() * 0.01, theta.sin() * 0.01, i as f64 * 0.1);
        }
        assert!(!det.oscillating);
    }

    #[test]
    fn test_footprint_collision_with_circle_obstacle() {
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
        let footprint = vec![(-0.3, -0.2), (-0.3, 0.2), (0.3, 0.2), (0.3, -0.2)];
        let obs = vec![ResolvedObstacle::Circle {
            cx: 5.2,
            cy: 5.0,
            radius: 0.2,
        }];
        assert!(check_footprint_collision(&grid, 5.0, 5.0, 0.0, &footprint, &obs).is_some());
    }
}

const STUCK_VELOCITY_EPSILON: f64 = 0.02;
const STUCK_ANGULAR_EPSILON: f64 = 0.05;
const STUCK_THRESHOLD_SECS: f64 = 3.0;

#[derive(Debug, Default)]
pub struct StuckDetector {
    stuck_secs: f64,
    fired: bool,
}

impl StuckDetector {
    pub fn step(
        &mut self,
        dt: f64,
        current_tick: u64,
        linear: f64,
        angular: f64,
        goal_reached: bool,
        bus: &EventBus,
    ) {
        if goal_reached {
            self.reset();
            return;
        }
        if linear.abs() < STUCK_VELOCITY_EPSILON && angular.abs() < STUCK_ANGULAR_EPSILON {
            self.stuck_secs += dt;
            if !self.fired && self.stuck_secs >= STUCK_THRESHOLD_SECS {
                bus.emit(SimEvent::Stuck {
                    stuck_secs: self.stuck_secs,
                    tick: current_tick,
                });
                self.fired = true;
            }
        } else {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.stuck_secs = 0.0;
        self.fired = false;
    }
}

#[cfg(test)]
mod stuck_tests {
    use super::*;

    #[test]
    fn test_fires_after_three_seconds_of_zero_velocity() {
        let mut det = StuckDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        for tick in 0..70 {
            det.step(0.05, tick, 0.0, 0.0, false, &bus);
        }
        assert!(matches!(rx.try_recv(), Ok(SimEvent::Stuck { .. })));
    }

    #[test]
    fn test_does_not_fire_when_moving() {
        let mut det = StuckDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        for tick in 0..70 {
            det.step(0.05, tick, 0.5, 0.0, false, &bus);
        }
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_resets_when_motion_resumes() {
        let mut det = StuckDetector::default();
        let bus = EventBus::new();
        for tick in 0..40 {
            det.step(0.05, tick, 0.0, 0.0, false, &bus);
        }
        det.step(0.05, 40, 0.5, 0.0, false, &bus);
        let mut rx = bus.subscribe();
        for tick in 41..80 {
            det.step(0.05, tick, 0.0, 0.0, false, &bus);
        }
        assert!(rx.try_recv().is_err());
    }
}

const APPROACH_THRESHOLD_M: f64 = 0.5;

#[derive(Debug, Default)]
pub struct HumanApproachDetector {
    fired_for: std::collections::HashSet<String>,
}

impl HumanApproachDetector {
    pub fn step(
        &mut self,
        current_tick: u64,
        robot_pos: (f64, f64),
        humans: &[(String, (f64, f64))],
        bus: &EventBus,
    ) {
        let mut still_near: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for (name, (hx, hy)) in humans {
            let dx = robot_pos.0 - hx;
            let dy = robot_pos.1 - hy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < APPROACH_THRESHOLD_M {
                still_near.insert(name.clone());
                if !self.fired_for.contains(name) {
                    bus.emit(SimEvent::HumanApproach {
                        human_name: name.clone(),
                        distance_m: dist,
                        tick: current_tick,
                    });
                }
            }
        }
        self.fired_for = still_near;
    }
}

#[cfg(test)]
mod human_approach_tests {
    use super::*;

    #[test]
    fn test_fires_when_human_enters_radius() {
        let mut det = HumanApproachDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        det.step(1, (0.0, 0.0), &[("w".into(), (0.3, 0.0))], &bus);
        assert!(matches!(rx.try_recv(), Ok(SimEvent::HumanApproach { .. })));
    }

    #[test]
    fn test_does_not_refire_while_still_near() {
        let mut det = HumanApproachDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        det.step(1, (0.0, 0.0), &[("w".into(), (0.3, 0.0))], &bus);
        let _ = rx.try_recv();
        det.step(2, (0.0, 0.0), &[("w".into(), (0.35, 0.0))], &bus);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_refires_after_leaving_and_returning() {
        let mut det = HumanApproachDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        det.step(1, (0.0, 0.0), &[("w".into(), (0.3, 0.0))], &bus);
        let _ = rx.try_recv();
        det.step(2, (0.0, 0.0), &[("w".into(), (2.0, 0.0))], &bus);
        det.step(3, (0.0, 0.0), &[("w".into(), (0.3, 0.0))], &bus);
        assert!(matches!(rx.try_recv(), Ok(SimEvent::HumanApproach { .. })));
    }
}

#[derive(Debug, Default)]
pub struct PathReplannedDetector {
    last_path_hash: Option<u64>,
}

impl PathReplannedDetector {
    pub fn step(&mut self, current_tick: u64, path: Option<&[(f64, f64)]>, bus: &EventBus) {
        let new_hash = path.map(Self::hash_path);
        if new_hash.is_some() && new_hash != self.last_path_hash {
            if self.last_path_hash.is_some() {
                bus.emit(SimEvent::PathReplanned { tick: current_tick });
            }
            self.last_path_hash = new_hash;
        }
    }

    fn hash_path(path: &[(f64, f64)]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for (x, y) in path {
            x.to_bits().hash(&mut hasher);
            y.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod path_replanned_tests {
    use super::*;

    #[test]
    fn test_no_fire_on_first_path() {
        let mut det = PathReplannedDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        det.step(1, Some(&[(0.0, 0.0), (1.0, 1.0)]), &bus);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_fires_on_path_change() {
        let mut det = PathReplannedDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        det.step(1, Some(&[(0.0, 0.0), (1.0, 1.0)]), &bus);
        det.step(2, Some(&[(0.0, 0.0), (2.0, 2.0)]), &bus);
        assert!(matches!(rx.try_recv(), Ok(SimEvent::PathReplanned { tick: 2 })));
    }

    #[test]
    fn test_no_fire_when_path_identical() {
        let mut det = PathReplannedDetector::default();
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        det.step(1, Some(&[(0.0, 0.0), (1.0, 1.0)]), &bus);
        det.step(2, Some(&[(0.0, 0.0), (1.0, 1.0)]), &bus);
        assert!(rx.try_recv().is_err());
    }
}
