use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::live::{pace, tick_sensors, wall_now, GoalProtocol, LiveRaycaster, Robot, RobotSpec, Topics, Venue3d, VenueInputs};

#[derive(clap::Args, Debug)]
pub struct BatchArgs {
    spec: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum)]
    raycaster: Option<LiveRaycaster>,
    #[arg(long)]
    realtime_factor: Option<f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GoalSpec {
    pub name: String,
    pub at: [f64; 2],
    #[serde(default)]
    pub yaw: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RunSpec {
    pub name: String,
    pub zenoh: String,
    #[serde(default)]
    pub domain: u32,
    pub start: [f64; 3],
    #[serde(default)]
    pub crowd_anchors: BTreeMap<String, f64>,
    pub goals: Vec<GoalSpec>,
}

fn default_world() -> String {
    "live".into()
}
fn default_lidar_mount() -> Vec<f64> {
    vec![0.22, 0.0, 0.23, 0.0, 0.0, 0.0, 1.0]
}
fn default_camera_mount() -> Vec<f64> {
    vec![0.31, -0.24, 1.086, 0.0, 0.0, 0.0, 1.0]
}
fn default_footprint() -> [f64; 2] {
    [0.299, 0.18]
}
fn default_dt() -> f64 {
    0.05
}
fn default_rtf() -> f64 {
    1.0
}
fn default_seed() -> u32 {
    1
}
fn default_settle() -> f64 {
    45.0
}
fn default_arrive() -> f64 {
    0.6
}
fn default_dwell() -> f64 {
    8.0
}
fn default_resend() -> f64 {
    5.0
}
fn default_goal_timeout() -> f64 {
    240.0
}

#[derive(Deserialize, Debug)]
pub struct BatchSpec {
    pub venue: PathBuf,
    pub scene: String,
    #[serde(default = "default_world")]
    pub world: String,
    pub depth: PathBuf,
    pub crowd: Option<PathBuf>,
    #[serde(default = "default_lidar_mount")]
    pub lidar_mount: Vec<f64>,
    #[serde(default = "default_camera_mount")]
    pub camera_mount: Vec<f64>,
    #[serde(default = "default_footprint")]
    pub footprint: [f64; 2],
    #[serde(default)]
    pub raycaster: Option<LiveRaycaster>,
    #[serde(default = "default_dt")]
    pub dt: f64,
    #[serde(default = "default_rtf")]
    pub realtime_factor: f64,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default = "default_settle")]
    pub settle_s: f64,
    #[serde(default = "default_arrive")]
    pub arrive_m: f64,
    #[serde(default = "default_dwell")]
    pub dwell_s: f64,
    #[serde(default = "default_resend")]
    pub resend_s: f64,
    #[serde(default = "default_goal_timeout")]
    pub goal_timeout_s: f64,
    #[serde(default)]
    pub goals_via: GoalProtocol,
    #[serde(default)]
    pub cmd_topic: Option<String>,
    pub runs: Vec<RunSpec>,
}

impl BatchSpec {
    pub fn parse(text: &str, base: &Path) -> Result<BatchSpec, String> {
        let mut spec: BatchSpec = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
        let rebase = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        };
        rebase(&mut spec.venue);
        rebase(&mut spec.depth);
        if let Some(c) = spec.crowd.as_mut() {
            rebase(c);
        }
        if spec.runs.is_empty() {
            return Err("batch has no runs".into());
        }
        let mut names: Vec<&str> = spec.runs.iter().map(|r| r.name.as_str()).collect();
        names.sort();
        names.dedup();
        if names.len() != spec.runs.len() {
            return Err("run names must be unique".into());
        }
        if spec.dt <= 0.0 || spec.realtime_factor < 0.0 {
            return Err("dt must be positive and realtime_factor non-negative (0 runs unthrottled)".into());
        }
        Ok(spec)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    Settling,
    Requesting { goal: usize, sent: f64, last_sent: f64 },
    Driving { goal: usize, sent: f64, accepted: f64 },
    Dwelling { goal: usize, until: f64 },
    Done,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoalOutcome {
    pub name: String,
    pub outcome: &'static str,
    pub sent: f64,
    pub accepted: Option<f64>,
    pub arrived: Option<f64>,
    pub path_m: f64,
}

pub enum Action {
    Send(String),
    Anchor(String),
}

pub struct Mission {
    pub goals: Vec<GoalSpec>,
    pub phase: Phase,
    pub outcomes: Vec<GoalOutcome>,
    path_m: f64,
    last_xy: Option<[f64; 2]>,
    plans_seen: Option<f64>,
}

pub struct Rules {
    pub settle_s: f64,
    pub arrive_m: f64,
    pub dwell_s: f64,
    pub resend_s: f64,
    pub goal_timeout_s: f64,
}

impl Mission {
    pub fn new(goals: Vec<GoalSpec>) -> Self {
        Mission { goals, phase: Phase::Settling, outcomes: Vec::new(), path_m: 0.0, last_xy: None, plans_seen: None }
    }

    pub fn done(&self) -> bool {
        self.phase == Phase::Done
    }

    fn finish(&mut self, goal: usize, outcome: &'static str, sent: f64, accepted: Option<f64>, arrived: Option<f64>) {
        self.outcomes.push(GoalOutcome { name: self.goals[goal].name.clone(), outcome, sent, accepted, arrived, path_m: self.path_m });
    }

    fn request(&mut self, goal: usize, t: f64, actions: &mut Vec<Action>) {
        if goal >= self.goals.len() {
            self.phase = Phase::Done;
            return;
        }
        self.path_m = 0.0;
        actions.push(Action::Anchor(self.goals[goal].name.clone()));
        actions.push(Action::Send(self.goals[goal].name.clone()));
        self.phase = Phase::Requesting { goal, sent: t, last_sent: t };
    }

    pub fn update(&mut self, t: f64, xy: [f64; 2], plan_at: Option<f64>, rules: &Rules) -> Vec<Action> {
        if let Some(prev) = self.last_xy {
            self.path_m += (xy[0] - prev[0]).hypot(xy[1] - prev[1]);
        }
        self.last_xy = Some(xy);
        let new_plan = plan_at.filter(|p| self.plans_seen.is_none_or(|seen| *p > seen));
        if new_plan.is_some() {
            self.plans_seen = plan_at;
        }
        let mut actions = Vec::new();
        match self.phase.clone() {
            Phase::Settling => {
                if t >= rules.settle_s {
                    self.request(0, t, &mut actions);
                }
            }
            Phase::Requesting { goal, sent, last_sent } => {
                if let Some(p) = new_plan.filter(|p| *p >= sent) {
                    self.phase = Phase::Driving { goal, sent, accepted: p };
                } else if t - sent >= rules.goal_timeout_s {
                    self.finish(goal, "no_plan", sent, None, None);
                    self.phase = Phase::Done;
                } else if t - last_sent >= rules.resend_s {
                    actions.push(Action::Send(self.goals[goal].name.clone()));
                    self.phase = Phase::Requesting { goal, sent, last_sent: t };
                }
            }
            Phase::Driving { goal, sent, accepted } => {
                let at = self.goals[goal].at;
                if (xy[0] - at[0]).hypot(xy[1] - at[1]) < rules.arrive_m {
                    self.finish(goal, "arrived", sent, Some(accepted), Some(t));
                    self.phase = Phase::Dwelling { goal, until: t + rules.dwell_s };
                } else if t - sent >= rules.goal_timeout_s {
                    self.finish(goal, "timeout", sent, Some(accepted), None);
                    self.phase = Phase::Done;
                }
            }
            Phase::Dwelling { goal, until } => {
                if t >= until {
                    self.request(goal + 1, t, &mut actions);
                }
            }
            Phase::Done => {}
        }
        actions
    }
}

fn outcome_json(o: &GoalOutcome) -> Value {
    let r = |x: f64| (x * 100.0).round() / 100.0;
    json!({
        "goal": o.name,
        "outcome": o.outcome,
        "sent_s": r(o.sent),
        "plan_s": o.accepted.map(r),
        "arrived_s": o.arrived.map(r),
        "duration_s": o.arrived.map(|a| r(a - o.sent)),
        "path_m": r(o.path_m),
    })
}

pub fn batch(args: BatchArgs, stop: Arc<AtomicBool>) -> Result<(), String> {
    let text = std::fs::read_to_string(&args.spec).map_err(|e| format!("{}: {e}", args.spec.display()))?;
    let base = args.spec.parent().map(Path::to_path_buf).unwrap_or_default();
    let spec = BatchSpec::parse(&text, &base)?;
    let raycaster = args.raycaster.or(spec.raycaster).unwrap_or(LiveRaycaster::Simd);
    let rtf = args.realtime_factor.unwrap_or(spec.realtime_factor);
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let venue = Venue3d::load(&VenueInputs {
        venue: &spec.venue,
        scene: &spec.scene,
        world: &spec.world,
        depth: &spec.depth,
        crowd: spec.crowd.as_deref(),
        lidar_mount: &spec.lidar_mount,
        camera_mount: &spec.camera_mount,
        raycaster,
        seed: spec.seed,
    })?;
    let topics = Topics {
        protocol: spec.goals_via,
        cmd: spec.cmd_topic.clone().unwrap_or_else(|| spec.goals_via.cmd_topic().into()),
        goal: "/goal_pose".into(),
        plan: "/plan".into(),
        nav_ctrl: "/robot_api/nav_ctrl".into(),
        nav_events: "/robot_api/nav_events".into(),
    };
    let rules = Rules { settle_s: spec.settle_s, arrive_m: spec.arrive_m, dwell_s: spec.dwell_s, resend_s: spec.resend_s, goal_timeout_s: spec.goal_timeout_s };
    let start = wall_now();
    let mut robots = Vec::new();
    let mut missions = Vec::new();
    for run in &spec.runs {
        let replay = args.out.join(format!("replay-{}.jsonl", run.name));
        let node = format!("anews_chassis_{}", run.name.replace(|c: char| !c.is_ascii_alphanumeric(), "_"));
        robots.push(Robot::open(
            &RobotSpec {
                name: &run.name,
                zenoh: &run.zenoh,
                domain: run.domain,
                node: &node,
                start: run.start,
                footprint: spec.footprint,
                crowd_anchors: run.crowd_anchors.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                topics: &topics,
                follow_goals: false,
                replay: Some(&replay),
                dt: spec.dt,
                venue_label: spec.venue.display().to_string(),
                scene: &spec.scene,
            },
            &venue,
            start,
        )?);
        missions.push(Mission::new(run.goals.clone()));
    }
    eprintln!("anews sim batch: {} runs, scene {} on {}, dt {} s, realtime factor {}", robots.len(), spec.scene, venue.backend, spec.dt, rtf);

    let wall0 = Instant::now();
    let mut last_status = start;
    let lidar_xyz = venue.rig_lidar_xyz();
    while !stop.load(Ordering::Relaxed) && !missions.iter().all(Mission::done) {
        let mut stepped = Vec::with_capacity(robots.len());
        for (robot, mission) in robots.iter_mut().zip(missions.iter_mut()) {
            robot.drain()?;
            let t = robot.sim.now() - start;
            let (x, y, _) = robot.sim.robot.pose();
            let before = mission.outcomes.len();
            for action in mission.update(t, [x, y], robot.plan_at.map(|p| p - start), &rules) {
                match action {
                    Action::Anchor(name) => {
                        robot.anchor_goal(&name)?;
                    }
                    Action::Send(name) => {
                        let goal = mission.goals.iter().find(|g| g.name == name).ok_or_else(|| format!("unknown goal {name}"))?;
                        robot.send_goal(&name, [goal.at[0], goal.at[1], goal.yaw])?
                    }
                }
            }
            for o in &mission.outcomes[before..] {
                eprintln!("anews sim batch: {} {} {} at {:.1} s", robot.name, o.name, o.outcome, t);
                robot.write_event(json!({"event": "mission", "result": outcome_json(o)}))?;
            }
            stepped.push(robot.step(lidar_xyz)?);
        }
        let tick = robots[0].sim.tick;
        let mut refs: Vec<&mut Robot> = robots.iter_mut().collect();
        tick_sensors(&venue, &mut refs, &stepped, tick, spec.dt)?;
        for robot in robots.iter_mut() {
            robot.record_crowd(tick)?;
        }
        let now = stepped[0].now;
        if now - last_status >= 1.0 {
            last_status = now;
            let wall = wall0.elapsed().as_secs_f64();
            for robot in &robots {
                println!("{}", robot.status(wall));
            }
        }
        pace(wall0, now - start, rtf);
    }

    let wall = wall0.elapsed().as_secs_f64();
    let sim_s = robots[0].sim.now() - start;
    let runs: Vec<Value> = robots
        .iter()
        .zip(&missions)
        .map(|(robot, mission)| {
            let arrived = mission.outcomes.iter().filter(|o| o.outcome == "arrived").count();
            json!({
                "run": robot.name,
                "passed": arrived == mission.goals.len(),
                "goals": mission.outcomes.iter().map(outcome_json).collect::<Vec<_>>(),
                "unfinished": if mission.done() { Value::Null } else { json!(format!("{:?}", mission.phase)) },
                "bumps": robot.sim.robot.bumps,
                "crowd_bumps": robot.sim.robot.crowd_bumps,
                "scans": robot.scans,
                "replay": format!("replay-{}.jsonl", robot.name),
            })
        })
        .collect();
    let summary = json!({
        "schema": "anews.sim_batch/1",
        "spec": args.spec.display().to_string(),
        "scene": spec.scene,
        "raycaster": venue.backend,
        "dt": spec.dt,
        "realtime_factor": rtf,
        "sim_s": (sim_s * 100.0).round() / 100.0,
        "wall_s": (wall * 100.0).round() / 100.0,
        "speed_x_realtime": if wall > 0.0 { (sim_s / wall * 100.0).round() / 100.0 } else { 0.0 },
        "passed": runs.iter().filter(|r| r["passed"] == json!(true)).count(),
        "runs": runs,
    });
    let path = args.out.join("summary.json");
    std::fs::write(&path, serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())? + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
    println!("{summary}");
    for robot in robots {
        robot.close()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> Rules {
        Rules { settle_s: 10.0, arrive_m: 0.5, dwell_s: 2.0, resend_s: 5.0, goal_timeout_s: 60.0 }
    }

    fn goals() -> Vec<GoalSpec> {
        vec![GoalSpec { name: "11".into(), at: [3.0, 0.0], yaw: 0.0 }, GoalSpec { name: "charging_pile".into(), at: [0.0, 0.0], yaw: 0.0 }]
    }

    fn sends(actions: &[Action]) -> Vec<String> {
        actions.iter().filter_map(|a| if let Action::Send(n) = a { Some(n.clone()) } else { None }).collect()
    }

    #[test]
    fn waits_for_nav2_then_resends_until_a_plan_arrives_and_visits_goals_in_order() {
        let r = rules();
        let mut m = Mission::new(goals());
        assert!(m.update(5.0, [0.0, 0.0], None, &r).is_empty(), "no goal while Nav2 is still settling");
        assert_eq!(sends(&m.update(10.0, [0.0, 0.0], None, &r)), ["11"]);
        assert!(sends(&m.update(12.0, [0.0, 0.0], None, &r)).is_empty());
        assert_eq!(sends(&m.update(15.0, [0.0, 0.0], None, &r)), ["11"], "an unanswered goal is sent again");
        m.update(16.0, [0.0, 0.0], Some(16.0), &r);
        assert!(matches!(m.phase, Phase::Driving { goal: 0, .. }));
        m.update(20.0, [1.5, 0.0], Some(16.0), &r);
        m.update(24.0, [2.8, 0.0], Some(16.0), &r);
        assert_eq!(m.outcomes.len(), 1);
        assert_eq!(m.outcomes[0].outcome, "arrived");
        assert!((m.outcomes[0].path_m - 2.8).abs() < 1e-9);
        assert!(sends(&m.update(25.0, [2.8, 0.0], Some(16.0), &r)).is_empty(), "dwell at the table before the next goal");
        assert_eq!(sends(&m.update(26.0, [2.8, 0.0], Some(16.0), &r)), ["charging_pile"]);
        m.update(27.0, [2.8, 0.0], Some(16.0), &r);
        assert!(matches!(m.phase, Phase::Requesting { goal: 1, .. }), "a plan from the previous goal does not count");
        m.update(28.0, [2.8, 0.0], Some(28.0), &r);
        m.update(35.0, [0.1, 0.0], Some(28.0), &r);
        m.update(38.0, [0.1, 0.0], Some(28.0), &r);
        assert!(m.done());
        assert!(m.outcomes.iter().all(|o| o.outcome == "arrived"));
    }

    #[test]
    fn a_goal_that_never_arrives_times_out_and_ends_the_run() {
        let r = rules();
        let mut m = Mission::new(goals());
        m.update(10.0, [0.0, 0.0], None, &r);
        m.update(11.0, [0.0, 0.0], Some(11.0), &r);
        m.update(69.0, [1.0, 0.0], Some(11.0), &r);
        assert!(!m.done());
        m.update(70.0, [1.0, 0.0], Some(11.0), &r);
        assert!(m.done());
        assert_eq!(m.outcomes[0].outcome, "timeout");
        assert_eq!(m.outcomes.len(), 1, "later goals are not attempted after a timeout");
    }

    #[test]
    fn relative_paths_resolve_against_the_spec_file() {
        let spec = BatchSpec::parse(
            "venue: venues/demo_cafe/venue.yaml\nscene: s\ndepth: /abs/depth.yaml\nruns:\n  - {name: a, zenoh: tcp/h:1, start: [0, 0, 0], goals: [{name: '11', at: [1, 2]}]}\n",
            Path::new("/data/batches"),
        )
        .unwrap();
        assert_eq!(spec.venue, PathBuf::from("/data/batches/venues/demo_cafe/venue.yaml"));
        assert_eq!(spec.depth, PathBuf::from("/abs/depth.yaml"));
        assert!(BatchSpec::parse("venue: v\nscene: s\ndepth: d\nruns: []\n", Path::new("/")).is_err());
    }
}
