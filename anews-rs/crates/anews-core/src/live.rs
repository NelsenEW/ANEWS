use crate::actors::ActorSet;
use crate::crowd::{CrowdGrid, CrowdTimeline};
use crate::diff_drive::DiffDrive;
use crate::world3d::World3d;

pub const CMD_FRESH_S: f64 = 0.5;
const BODY_HEIGHTS: [f64; 5] = [0.05, 0.3, 0.6, 0.9, 1.2];
pub const ROBOT_HEIGHT: f64 = 1.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contact {
    Free,
    Venue,
    Crowd,
}

#[derive(Debug, Clone)]
pub struct LiveRobot {
    pub drive: DiffDrive,
    pub half: [f64; 2],
    pub bumps: u32,
    pub crowd_bumps: u32,
    pub linear: f64,
    pub angular: f64,
}

impl LiveRobot {
    pub fn new(x: f64, y: f64, yaw: f64, half: [f64; 2]) -> Self {
        LiveRobot { drive: DiffDrive::new(x, y, yaw), half, bumps: 0, crowd_bumps: 0, linear: 0.0, angular: 0.0 }
    }

    pub fn pose(&self) -> (f64, f64, f64) {
        self.drive.pose()
    }

    pub fn reset(&mut self, x: f64, y: f64, yaw: f64) {
        self.drive.reset(x, y, yaw);
        self.linear = 0.0;
        self.angular = 0.0;
    }

    pub fn step(&mut self, contact: &World3d, crowd: Option<&CrowdGrid>, actors: Option<&ActorSet>, linear: f64, angular: f64, dt: f64) -> Contact {
        let crowd_hit = |d: &DiffDrive| {
            let (x, y, yaw) = d.pose();
            crowd.is_some_and(|g| g.touches(x, y, yaw, self.half))
                || actors.is_some_and(|a| a.touches(x, y, yaw, self.half, ROBOT_HEIGHT))
        };
        let before = self.drive.clone();
        let was_touching = footprint_hits(contact, &before, self.half);
        let was_touching_crowd = crowd_hit(&before);
        let mut next = before.clone();
        next.update(linear, angular, dt);
        let outcome = if !was_touching && footprint_hits(contact, &next, self.half) {
            Contact::Venue
        } else if !was_touching_crowd && crowd_hit(&next) {
            Contact::Crowd
        } else {
            Contact::Free
        };
        match outcome {
            Contact::Free => {
                self.drive = next;
                self.linear = linear;
                self.angular = angular;
            }
            Contact::Venue | Contact::Crowd => {
                self.linear = 0.0;
                self.angular = 0.0;
                if outcome == Contact::Venue {
                    self.bumps += 1;
                } else {
                    self.crowd_bumps += 1;
                }
            }
        }
        outcome
    }
}

pub fn footprint_hits(world: &World3d, drive: &DiffDrive, half: [f64; 2]) -> bool {
    let (x, y, yaw) = drive.pose();
    let (s, c) = yaw.sin_cos();
    let mut edge = Vec::new();
    let steps = 6;
    for k in 0..=steps {
        let t = -1.0 + 2.0 * k as f64 / steps as f64;
        edge.extend([[half[0], t * half[1]], [-half[0], t * half[1]], [t * half[0], half[1]], [t * half[0], -half[1]]]);
    }
    edge.iter().any(|p| {
        let reach = p[0].hypot(p[1]);
        let dir = [(c * p[0] - s * p[1]) / reach, (s * p[0] + c * p[1]) / reach, 0.0];
        BODY_HEIGHTS.iter().any(|&z| world.cast([x, y, z], dir, reach) < reach)
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CmdLatch {
    linear: f64,
    angular: f64,
    at: Option<f64>,
}

impl CmdLatch {
    pub fn set(&mut self, linear: f64, angular: f64, now: f64) {
        *self = CmdLatch { linear, angular, at: Some(now) };
    }

    pub fn effective(&self, now: f64) -> (f64, f64) {
        match self.at {
            Some(at) if now - at < CMD_FRESH_S => (self.linear, self.angular),
            _ => (0.0, 0.0),
        }
    }
}

pub struct CrowdReplay {
    pub timeline: CrowdTimeline,
    anchors: Vec<(String, f64)>,
    leg: Option<(f64, f64)>,
}

impl CrowdReplay {
    pub fn new(timeline: CrowdTimeline, mut anchors: Vec<(String, f64)>) -> Self {
        anchors.sort_by(|a, b| a.1.total_cmp(&b.1));
        CrowdReplay { timeline, anchors, leg: None }
    }

    pub fn goal(&mut self, name: &str, now: f64) -> bool {
        let Some((_, recording)) = self.anchors.iter().find(|(n, _)| n == name) else { return false };
        self.leg = Some((*recording, now));
        true
    }

    pub fn time(&self, now: f64) -> Option<f64> {
        let first = self.anchors.first()?;
        Some(match self.leg {
            Some((recording, wall)) => recording + (now - wall),
            None => first.1,
        })
    }

    pub fn grid(&self) -> CrowdGrid {
        let mut grid = self.timeline.grid();
        self.timeline.apply(self.timeline.start, 0.0, &mut grid);
        grid
    }

    pub fn apply(&self, now: f64, grid: &mut CrowdGrid) {
        if let Some(t) = self.time(now) {
            self.timeline.apply(t, 0.0, grid);
        }
    }
}

pub struct LiveSim {
    pub robot: LiveRobot,
    pub contact: World3d,
    pub crowd: Option<(CrowdReplay, CrowdGrid)>,
    pub cmd: CmdLatch,
    pub dt: f64,
    pub start: f64,
    pub tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tick {
    pub tick: u64,
    pub now: f64,
    pub contact: Contact,
}

impl LiveSim {
    pub fn new(robot: LiveRobot, contact: World3d, crowd: Option<CrowdReplay>, dt: f64, start: f64) -> Self {
        let crowd = crowd.map(|c| {
            let grid = c.grid();
            (c, grid)
        });
        LiveSim { robot, contact, crowd, cmd: CmdLatch::default(), dt, start, tick: 0 }
    }

    pub fn now(&self) -> f64 {
        self.start + self.tick as f64 * self.dt
    }

    pub fn command(&mut self, linear: f64, angular: f64) {
        let now = self.now();
        self.cmd.set(linear, angular, now);
    }

    pub fn goal(&mut self, name: &str) -> bool {
        let now = self.now();
        self.crowd.as_mut().is_some_and(|(c, _)| c.goal(name, now))
    }

    pub fn crowd_grid(&self) -> Option<&CrowdGrid> {
        self.crowd.as_ref().map(|(_, g)| g)
    }

    pub fn step(&mut self) -> Tick {
        let now = self.now();
        if let Some((replay, grid)) = self.crowd.as_mut() {
            replay.apply(now, grid);
        }
        let (v, w) = self.cmd.effective(now);
        let crowd = self.crowd.as_ref().map(|(_, g)| g);
        let contact = self.robot.step(&self.contact, crowd, self.contact.actors.as_ref(), v, w, self.dt);
        self.tick += 1;
        Tick { tick: self.tick, now: self.now(), contact }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world3d::box_triangles;

    fn open_floor() -> World3d {
        World3d::from_triangles(Vec::new(), true)
    }

    #[test]
    fn sim_clock_advances_by_dt_per_step() {
        let mut sim = LiveSim::new(LiveRobot::new(0.0, 0.0, 0.0, [0.27, 0.18]), open_floor(), None, 0.05, 1000.0);
        for _ in 0..40 {
            sim.step();
        }
        assert!((sim.now() - 1002.0).abs() < 1e-9, "now {}", sim.now());
    }

    #[test]
    fn fresh_command_drives_and_stale_command_stops() {
        let mut sim = LiveSim::new(LiveRobot::new(0.0, 0.0, 0.0, [0.27, 0.18]), open_floor(), None, 0.05, 0.0);
        sim.command(0.5, 0.0);
        for _ in 0..9 {
            sim.step();
        }
        let (x_fresh, _, _) = sim.robot.pose();
        assert!((x_fresh - 0.225).abs() < 1e-9, "drove {x_fresh}");
        for _ in 0..20 {
            sim.step();
        }
        let (x_stale, _, _) = sim.robot.pose();
        assert!((x_stale - 0.25).abs() < 1e-9, "a command older than 0.5 s must stop the robot, x {x_stale}");
        assert_eq!(sim.robot.linear, 0.0);
    }

    #[test]
    fn driving_into_a_table_is_refused_and_counted() {
        let mut tris = Vec::new();
        box_triangles([1.0, 0.0, 0.4], [0.3, 0.5, 0.4], 0.0, 0, &mut tris);
        let mut sim = LiveSim::new(LiveRobot::new(0.0, 0.0, 0.0, [0.27, 0.18]), World3d::from_triangles(tris, true), None, 0.05, 0.0);
        for _ in 0..60 {
            sim.command(0.5, 0.0);
            sim.step();
        }
        let (x, _, _) = sim.robot.pose();
        assert!(x + 0.27 <= 0.7 + 1e-9, "front edge {} went into the table", x + 0.27);
        assert!(x > 0.35, "robot stopped too early at {x}");
        assert!(sim.robot.bumps > 0);
    }
}
