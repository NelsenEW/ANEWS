use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anews_core::cdr::{self, Chassis, PlanarTransform, ScanOut};
use anews_core::crowd::CrowdTimeline;
use anews_core::fidelity::{scene_items_world, scene_world, voxel_centroids, LIDAR_ANGLE_MAX, LIDAR_ANGLE_MIN, LIDAR_COUNT};
use anews_core::live::{Contact, CrowdReplay, LiveRobot, LiveSim};
use anews_core::raycast::RayCaster;
use anews_core::replay::REPLAY_SCHEMA;
use anews_core::sensor3d::{mat_vec, quat_matrix};
use anews_core::sensor_batch::{occlude_by_crowd, sense_batch, Pose2, Rig, RobotScan};
use anews_core::trip::Mount;
use anews_core::world3d::World3d;
use anews_link::{types, Link};
use anews_raycast::SimdCaster;
use serde_json::{json, Value};

use crate::offline::{load_depth_sensor, load_venue};

#[derive(Clone, Copy, Debug, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LiveRaycaster {
    Simd,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum GoalProtocol {
    #[default]
    GoalPose,
    RobotApi,
}

impl GoalProtocol {
    pub fn cmd_topic(self) -> &'static str {
        match self {
            GoalProtocol::GoalPose => "/cmd_vel",
            GoalProtocol::RobotApi => "/mobile_base/commands/velocity",
        }
    }
}

#[derive(clap::Args, Debug)]
pub struct LiveArgs {
    #[arg(long)]
    venue: PathBuf,
    #[arg(long)]
    scene: String,
    #[arg(long, default_value = "live")]
    world: String,
    #[arg(long)]
    depth: PathBuf,
    #[arg(long)]
    crowd: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    crowd_anchors: Vec<String>,
    #[arg(long, value_delimiter = ',', allow_negative_numbers = true)]
    start: Vec<f64>,
    #[arg(long, value_delimiter = ',', allow_negative_numbers = true, default_value = "0.22,0,0.23,0,0,0,1")]
    lidar_mount: Vec<f64>,
    #[arg(long, value_delimiter = ',', allow_negative_numbers = true, default_value = "0.31,-0.24,1.086,0,0,0,1")]
    camera_mount: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "0.299,0.18")]
    footprint: Vec<f64>,
    #[arg(long)]
    zenoh: String,
    #[arg(long, default_value = "0")]
    domain: u32,
    #[arg(long, default_value = "anews_chassis")]
    node: String,
    #[arg(long, value_enum, default_value_t = GoalProtocol::GoalPose)]
    goals: GoalProtocol,
    #[arg(long)]
    cmd_topic: Option<String>,
    #[arg(long, default_value = "/goal_pose")]
    goal_topic: String,
    #[arg(long, default_value = "/plan")]
    plan_topic: String,
    #[arg(long, default_value = "/robot_api/nav_ctrl")]
    nav_ctrl_topic: String,
    #[arg(long, default_value = "/robot_api/nav_events")]
    nav_events_topic: String,
    #[arg(long, value_enum, default_value_t = LiveRaycaster::Simd)]
    raycaster: LiveRaycaster,
    #[arg(long, default_value = "0.05")]
    dt: f64,
    #[arg(long, default_value = "1.0")]
    realtime_factor: f64,
    #[arg(long)]
    duration: Option<f64>,
    #[arg(long)]
    replay: Option<PathBuf>,
    #[arg(long, default_value = "1.0")]
    status_every: f64,
    #[arg(long, default_value = "1")]
    seed: u32,
}

const BASE_LINK_Z: f64 = 0.0102;
const SCAN_EVERY: u64 = 2;
const DEPTH_EVERY: u64 = 4;
const CROWD_EVERY: u64 = 10;

fn mount(values: &[f64], what: &str) -> Result<Mount, String> {
    if values.len() != 7 {
        return Err(format!("--{what} is x,y,z,qx,qy,qz,qw; got {} values", values.len()));
    }
    Ok(Mount { xyz: [values[0], values[1], values[2]], rot: quat_matrix([values[3], values[4], values[5], values[6]]) })
}

fn anchors(raw: &[String]) -> Result<Vec<(String, f64)>, String> {
    raw.iter()
        .map(|kv| {
            let (k, v) = kv.split_once(':').ok_or_else(|| format!("crowd anchor {kv} is goal_name:unix_time"))?;
            Ok((k.to_string(), v.parse::<f64>().map_err(|e| format!("crowd anchor {kv}: {e}"))?))
        })
        .collect()
}

pub fn wall_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

enum Sensing {
    Simd(SimdCaster),
    Gpu(anews_raycast::GpuSensors),
}

impl Sensing {
    fn new(kind: LiveRaycaster, world: &World3d, rig: &Rig, seed: u32) -> Result<(Sensing, String), String> {
        match kind {
            LiveRaycaster::Simd => Ok((Sensing::Simd(SimdCaster::new(world)?), "simd".into())),
            LiveRaycaster::Gpu => {
                let ctx = anews_raycast::GpuContext::new()?;
                let label = format!("gpu: {}", ctx.adapter());
                Ok((Sensing::Gpu(anews_raycast::GpuSensors::new(ctx, world, rig, seed)?), label))
            }
        }
    }

    fn sense(&self, rig: &Rig, poses: &[Pose2], seed: u32, tick: u32) -> Vec<RobotScan> {
        match self {
            Sensing::Simd(c) => sense_batch(c as &dyn RayCaster, rig, poses, seed, tick),
            Sensing::Gpu(g) => g.sense(poses, tick),
        }
    }
}

#[derive(Default)]
struct Inbox {
    cmd: Option<(f64, f64)>,
    plans: Vec<Vec<[f64; 2]>>,
    goals: Vec<String>,
    nav_events: Vec<String>,
}

struct Replay {
    out: Option<BufWriter<File>>,
    start: f64,
}

impl Replay {
    fn write(&mut self, now: f64, mut event: Value) -> Result<(), String> {
        let Some(out) = self.out.as_mut() else { return Ok(()) };
        event["t"] = json!(r3(now - self.start));
        writeln!(out, "{event}").map_err(|e| e.to_string())
    }

    fn flush(&mut self) -> Result<(), String> {
        self.out.as_mut().map_or(Ok(()), |o| o.flush().map_err(|e| e.to_string()))
    }
}

pub struct Venue3d {
    rig: Rig,
    leaf: f64,
    venue: anews_core::venue::Venue,
    scene: String,
    sensing: Sensing,
    pub backend: String,
    crowd: Option<CrowdTimeline>,
    seed: u32,
}

pub struct VenueInputs<'a> {
    pub venue: &'a std::path::Path,
    pub scene: &'a str,
    pub world: &'a str,
    pub depth: &'a std::path::Path,
    pub crowd: Option<&'a std::path::Path>,
    pub lidar_mount: &'a [f64],
    pub camera_mount: &'a [f64],
    pub raycaster: LiveRaycaster,
    pub seed: u32,
}

impl Venue3d {
    pub fn load(i: &VenueInputs) -> Result<Venue3d, String> {
        let (venue, map) = load_venue(i.venue)?;
        let sensor = load_depth_sensor(i.depth)?;
        let leaf = sensor.pipeline.voxel_leaf_m;
        let rig = Rig::new(mount(i.lidar_mount, "lidar-mount")?, mount(i.camera_mount, "camera-mount")?, sensor);
        let world = scene_world(&venue, map.as_ref(), i.world, i.scene)?.world;
        let (sensing, backend) = Sensing::new(i.raycaster, &world, &rig, i.seed)?;
        let crowd = match i.crowd {
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
                Some(serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?)
            }
            None => None,
        };
        Ok(Venue3d { rig, leaf, venue, scene: i.scene.to_string(), sensing, backend, crowd, seed: i.seed })
    }

    pub fn rig_lidar_xyz(&self) -> [f64; 3] {
        self.rig.lidar.xyz
    }

    fn contact(&self) -> Result<World3d, String> {
        scene_items_world(&self.venue, &self.scene)
    }

    fn sense(&self, poses: &[Pose2], tick: u32) -> Vec<RobotScan> {
        self.sensing.sense(&self.rig, poses, self.seed, tick)
    }
}

pub struct RobotSpec<'a> {
    pub name: &'a str,
    pub zenoh: &'a str,
    pub domain: u32,
    pub node: &'a str,
    pub start: [f64; 3],
    pub footprint: [f64; 2],
    pub crowd_anchors: Vec<(String, f64)>,
    pub topics: &'a Topics,
    pub follow_goals: bool,
    pub replay: Option<&'a std::path::Path>,
    pub dt: f64,
    pub venue_label: String,
    pub scene: &'a str,
}

pub struct Topics {
    pub protocol: GoalProtocol,
    pub cmd: String,
    pub goal: String,
    pub plan: String,
    pub nav_ctrl: String,
    pub nav_events: String,
}

pub struct Robot {
    pub name: String,
    pub sim: LiveSim,
    link: Link,
    inbox: Arc<Mutex<Inbox>>,
    replay: Replay,
    protocol: GoalProtocol,
    goal_topic: String,
    nav_ctrl: String,
    last_crowd: Option<Vec<(usize, u8)>>,
    pub plan_at: Option<f64>,
    pub scans: u64,
    pub start: f64,
    pub nav_events: Vec<Value>,
}

pub struct Stepped {
    pub now: f64,
    pub pose: Pose2,
}

impl Robot {
    pub fn open(spec: &RobotSpec, venue: &Venue3d, start: f64) -> Result<Robot, String> {
        let crowd = venue.crowd.clone().map(|t| CrowdReplay::new(t, spec.crowd_anchors.clone()));
        let body = LiveRobot::new(spec.start[0], spec.start[1], spec.start[2], spec.footprint);
        let sim = LiveSim::new(body, venue.contact()?, crowd, spec.dt, start);
        let mut link = Link::open(spec.zenoh, spec.domain, spec.node)?;
        if spec.topics.protocol == GoalProtocol::RobotApi {
            link.advertise("/mobile/odom", types::ODOMETRY)?;
        }
        for (topic, ty) in [
            ("/clock", types::CLOCK),
            ("/imu", types::IMU),
            ("/scan", types::LASER_SCAN),
            ("/odom", types::ODOMETRY),
            ("/tf", types::TF),
            ("/camera/depth/points_stvl", types::POINT_CLOUD2),
        ] {
            link.advertise(topic, ty)?;
        }
        let inbox = Arc::new(Mutex::new(Inbox::default()));
        let ib = inbox.clone();
        link.subscribe(&spec.topics.cmd, types::TWIST, move |p| {
            if let Ok((lin, ang)) = cdr::twist(&p) {
                ib.lock().unwrap().cmd = Some((lin[0], ang[2]));
            }
        })?;
        let ib = inbox.clone();
        link.subscribe(&spec.topics.plan, types::PATH, move |p| {
            if let Ok(path) = cdr::path(&p) {
                ib.lock().unwrap().plans.push(path.iter().map(|q| [r3(q.position[0]), r3(q.position[1])]).collect());
            }
        })?;
        match (spec.topics.protocol, spec.follow_goals) {
            (GoalProtocol::RobotApi, true) => {
                let ib = inbox.clone();
                link.subscribe(&spec.topics.nav_ctrl, types::NAVIGATION_CONTROL, move |p| {
                    if let Ok(m) = cdr::navigation_control(&p)
                        && m.control == 1
                    {
                        ib.lock().unwrap().goals.push(m.goal_name);
                    }
                })?;
            }
            (GoalProtocol::RobotApi, false) => link.advertise(&spec.topics.nav_ctrl, types::NAVIGATION_CONTROL)?,
            (GoalProtocol::GoalPose, true) => {
                let ib = inbox.clone();
                link.subscribe(&spec.topics.goal, types::POSE_STAMPED, move |p| {
                    if let Ok((_, pose)) = cdr::pose_stamped(&p) {
                        ib.lock().unwrap().goals.push(format!("{:.2},{:.2}", pose.position[0], pose.position[1]));
                    }
                })?;
            }
            (GoalProtocol::GoalPose, false) => link.advertise(&spec.topics.goal, types::POSE_STAMPED)?,
        }
        let ib = inbox.clone();
        link.subscribe(&spec.topics.nav_events, types::STRING, move |p| {
            if let Ok(s) = cdr::string(&p) {
                ib.lock().unwrap().nav_events.push(s);
            }
        })?;
        let mut replay = Replay {
            out: match spec.replay {
                Some(p) => Some(BufWriter::new(File::create(p).map_err(|e| format!("{}: {e}", p.display()))?)),
                None => None,
            },
            start,
        };
        replay.write(
            start,
            json!({
                "event": "init",
                "schema": REPLAY_SCHEMA,
                "source": "anews live",
                "run": spec.name,
                "start_unix": start,
                "robot": spec.start,
                "mounts": {
                    "lidar": {"xyz": venue.rig.lidar.xyz, "rot": venue.rig.lidar.rot},
                    "camera": {"xyz": venue.rig.camera.xyz, "rot": venue.rig.camera.rot},
                },
                "footprint": spec.footprint,
                "venue": spec.venue_label,
                "scene": spec.scene,
                "raycaster": venue.backend,
            }),
        )?;
        Ok(Robot {
            name: spec.name.to_string(),
            sim,
            link,
            inbox,
            replay,
            protocol: spec.topics.protocol,
            goal_topic: spec.topics.goal.clone(),
            nav_ctrl: spec.topics.nav_ctrl.clone(),
            last_crowd: None,
            plan_at: None,
            scans: 0,
            start,
            nav_events: Vec::new(),
        })
    }

    pub fn send_goal(&mut self, name: &str, pose: [f64; 3]) -> Result<(), String> {
        let now = self.sim.now();
        match self.protocol {
            GoalProtocol::RobotApi => {
                let msg = cdr::NavigationControl { control: 1, mode: 5, xyz: [0.0; 3], goal_name: name.to_string() };
                self.link.publish(&self.nav_ctrl, types::NAVIGATION_CONTROL, cdr::encode_navigation_control(&msg))?;
                self.replay.write(now, json!({"event": "nav_ctrl_sent", "name": name}))
            }
            GoalProtocol::GoalPose => {
                self.link.publish(&self.goal_topic, types::POSE_STAMPED, cdr::encode_pose_stamped(now, "map", pose[0], pose[1], pose[2]))?;
                self.replay.write(now, json!({"event": "goal_pose_sent", "name": name, "pose": [r3(pose[0]), r3(pose[1]), r3(pose[2])]}))
            }
        }
    }

    pub fn anchor_goal(&mut self, name: &str) -> Result<bool, String> {
        let now = self.sim.now();
        let anchored = self.sim.goal(name);
        self.replay.write(now, json!({"event": "goal", "name": name, "crowd_anchored": anchored}))?;
        Ok(anchored)
    }

    pub fn drain(&mut self) -> Result<(), String> {
        let now = self.sim.now();
        let drained = std::mem::take(&mut *self.inbox.lock().unwrap());
        if let Some((v, w)) = drained.cmd {
            self.sim.command(v, w);
        }
        for path in drained.plans {
            self.plan_at = Some(now);
            self.replay.write(now, json!({"event": "plan", "path": path}))?;
        }
        for goal in drained.goals {
            let anchored = self.anchor_goal(&goal)?;
            eprintln!("anews live: {} goal {goal} at sim {:.2} s (crowd anchored: {anchored})", self.name, now - self.start);
        }
        for text in drained.nav_events {
            let data: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
            self.nav_events.push(data.clone());
            self.replay.write(now, json!({"event": "nav_event", "data": data}))?;
        }
        Ok(())
    }

    pub fn step(&mut self, lidar_xyz: [f64; 3]) -> Result<Stepped, String> {
        let tick = self.sim.step();
        let now = tick.now;
        let (x, y, yaw) = self.sim.robot.pose();
        if tick.contact != Contact::Free {
            let with = if tick.contact == Contact::Venue { "venue" } else { "crowd" };
            self.replay.write(now, json!({"event": "bump", "with": with}))?;
        }
        let state = Chassis { stamp: now, x, y, yaw, linear: self.sim.robot.linear, angular: self.sim.robot.angular };
        self.link.publish("/clock", types::CLOCK, cdr::encode_clock(now))?;
        if self.protocol == GoalProtocol::RobotApi {
            self.link.publish("/mobile/odom", types::ODOMETRY, cdr::encode_odometry(&Chassis { x: 0.0, y: 0.0, yaw: 0.0, ..state }, "odom", "base_footprint"))?;
        }
        self.link.publish("/imu", types::IMU, cdr::encode_imu(&state, "imu_link"))?;
        self.link.publish("/odom", types::ODOMETRY, cdr::encode_odometry(&state, "odom", "base_footprint"))?;
        let lm = lidar_xyz;
        let tfs = [
            PlanarTransform { parent: "map".into(), child: "odom".into(), xyz: [0.0, 0.0, 0.0], yaw: 0.0 },
            PlanarTransform { parent: "odom".into(), child: "base_footprint".into(), xyz: [x, y, 0.0], yaw },
            PlanarTransform { parent: "base_footprint".into(), child: "base_link".into(), xyz: [0.0, 0.0, BASE_LINK_Z], yaw: 0.0 },
            PlanarTransform { parent: "base_link".into(), child: "laser_link".into(), xyz: [lm[0], lm[1], lm[2] - BASE_LINK_Z], yaw: 0.0 },
        ];
        self.link.publish("/tf", types::TF, cdr::encode_tf(now, &tfs))?;
        self.replay.write(now, json!({"event": "state", "robot": [r3(x), r3(y), r3(yaw)], "v": r3(state.linear), "w": r3(state.angular)}))?;
        Ok(Stepped { now, pose: Pose2 { x, y, yaw } })
    }

    pub fn publish_sensors(&mut self, venue: &Venue3d, mut scan: RobotScan, pose: Pose2, tick: u64, dt: f64) -> Result<(), String> {
        let now = self.sim.now();
        let sense_tick = tick as u32;
        if let Some(grid) = self.sim.crowd_grid() {
            occlude_by_crowd(&venue.rig, grid, std::slice::from_ref(&pose), venue.seed, sense_tick, std::slice::from_mut(&mut scan));
        }
        let Pose2 { x, y, yaw } = pose;
        let ranges: Vec<f32> = scan.lidar.iter().map(|r| *r as f32).collect();
        self.link.publish(
            "/scan",
            types::LASER_SCAN,
            cdr::encode_laser_scan(&ScanOut {
                stamp: now,
                frame_id: "laser_link",
                angle_min: LIDAR_ANGLE_MIN as f32,
                angle_max: LIDAR_ANGLE_MAX as f32,
                angle_increment: ((LIDAR_ANGLE_MAX - LIDAR_ANGLE_MIN) / (LIDAR_COUNT as f64 - 1.0)) as f32,
                scan_time: (dt * SCAN_EVERY as f64) as f32,
                range_min: 0.05,
                range_max: 30.0,
                ranges: &ranges,
            }),
        )?;
        self.scans += 1;
        let (s, c) = yaw.sin_cos();
        if self.replay.out.is_some() {
            let rig = &venue.rig;
            let rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
            let origin = {
                let m = mat_vec(&rot, rig.lidar.xyz);
                [r3(x + m[0]), r3(y + m[1]), r3(m[2])]
            };
            let pts: Vec<[f64; 3]> = scan
                .lidar
                .iter()
                .enumerate()
                .filter(|(_, r)| r.is_finite() && **r > 0.05 && **r < 8.0)
                .map(|(i, r)| {
                    let a = Rig::lidar_angle(i);
                    let p = mat_vec(&rot, mat_vec(&rig.lidar.rot, [r * a.cos(), r * a.sin(), 0.0]));
                    [r3(origin[0] + p[0]), r3(origin[1] + p[1]), r3(origin[2] + p[2])]
                })
                .collect();
            self.replay.write(now, json!({"event": "lidar", "origin": origin, "points": pts}))?;
        }
        if tick.is_multiple_of(DEPTH_EVERY) {
            let cloud = voxel_centroids(&venue.rig.depth_points(&scan.depth), venue.leaf);
            let xyz: Vec<[f32; 3]> = cloud.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
            self.link.publish("/camera/depth/points_stvl", types::POINT_CLOUD2, cdr::encode_point_cloud(now, "base_footprint", &xyz))?;
            if self.replay.out.is_some() {
                let pts: Vec<[f64; 3]> = cloud.iter().map(|p| [r3(x + c * p[0] - s * p[1]), r3(y + s * p[0] + c * p[1]), r3(p[2])]).collect();
                self.replay.write(now, json!({"event": "depth", "points": pts}))?;
            }
        }
        Ok(())
    }

    pub fn record_crowd(&mut self, tick: u64) -> Result<(), String> {
        if !tick.is_multiple_of(CROWD_EVERY) {
            return Ok(());
        }
        let now = self.sim.now();
        let Some(grid) = self.sim.crowd_grid() else { return Ok(()) };
        let cells = grid.occupied();
        if self.last_crowd.as_ref() == Some(&cells) {
            return Ok(());
        }
        let out: Vec<[f64; 3]> = cells
            .iter()
            .map(|(i, f)| {
                let c = grid.frame.center(*i);
                [r3(c[0]), r3(c[1]), *f as f64]
            })
            .collect();
        self.replay.write(now, json!({"event": "crowd", "cells": out}))?;
        self.last_crowd = Some(cells);
        Ok(())
    }

    pub fn write_event(&mut self, event: Value) -> Result<(), String> {
        let now = self.sim.now();
        self.replay.write(now, event)
    }

    pub fn status(&self, wall_elapsed: f64) -> Value {
        let now = self.sim.now();
        let (x, y, yaw) = self.sim.robot.pose();
        json!({
            "run": self.name,
            "t": r3(now - self.start),
            "x": r3(x), "y": r3(y), "yaw": r3(yaw),
            "v": r3(self.sim.robot.linear), "w": r3(self.sim.robot.angular),
            "bumps": self.sim.robot.bumps, "crowd_bumps": self.sim.robot.crowd_bumps,
            "scans": self.scans,
            "plan_age": self.plan_at.map(|p| r3(now - p)),
            "rtf": if wall_elapsed > 0.0 { r3((now - self.start) / wall_elapsed) } else { 0.0 },
        })
    }

    pub fn close(mut self) -> Result<(), String> {
        self.replay.flush()?;
        self.link.close()
    }
}

pub fn pace(wall0: Instant, sim_elapsed: f64, realtime_factor: f64) {
    if realtime_factor > 0.0 {
        let due = Duration::from_secs_f64(sim_elapsed / realtime_factor);
        let elapsed = wall0.elapsed();
        if due > elapsed {
            std::thread::sleep(due - elapsed);
        }
    }
}

pub fn tick_sensors(venue: &Venue3d, robots: &mut [&mut Robot], stepped: &[Stepped], tick: u64, dt: f64) -> Result<(), String> {
    if !tick.is_multiple_of(SCAN_EVERY) {
        return Ok(());
    }
    let poses: Vec<Pose2> = stepped.iter().map(|s| s.pose).collect();
    let scans = venue.sense(&poses, tick as u32);
    for ((robot, scan), pose) in robots.iter_mut().zip(scans).zip(poses) {
        robot.publish_sensors(venue, scan, pose, tick, dt)?;
    }
    Ok(())
}

pub fn live(args: LiveArgs, stop: Arc<AtomicBool>) -> Result<(), String> {
    if args.start.len() != 3 || args.footprint.len() != 2 {
        return Err("--start is x,y,yaw and --footprint is half_length,half_width".into());
    }
    if args.dt <= 0.0 || args.realtime_factor < 0.0 {
        return Err("--dt must be positive and --realtime-factor non-negative (0 runs unthrottled)".into());
    }
    let venue = Venue3d::load(&VenueInputs {
        venue: &args.venue,
        scene: &args.scene,
        world: &args.world,
        depth: &args.depth,
        crowd: args.crowd.as_deref(),
        lidar_mount: &args.lidar_mount,
        camera_mount: &args.camera_mount,
        raycaster: args.raycaster,
        seed: args.seed,
    })?;
    let topics = Topics {
        protocol: args.goals,
        cmd: args.cmd_topic.clone().unwrap_or_else(|| args.goals.cmd_topic().into()),
        goal: args.goal_topic.clone(),
        plan: args.plan_topic.clone(),
        nav_ctrl: args.nav_ctrl_topic.clone(),
        nav_events: args.nav_events_topic.clone(),
    };
    let start = wall_now();
    let mut robot = Robot::open(
        &RobotSpec {
            name: "live",
            zenoh: &args.zenoh,
            domain: args.domain,
            node: &args.node,
            start: [args.start[0], args.start[1], args.start[2]],
            footprint: [args.footprint[0], args.footprint[1]],
            crowd_anchors: anchors(&args.crowd_anchors)?,
            topics: &topics,
            follow_goals: true,
            replay: args.replay.as_deref(),
            dt: args.dt,
            venue_label: args.venue.display().to_string(),
            scene: &args.scene,
        },
        &venue,
        start,
    )?;
    eprintln!("anews live: {} scene {} on {} via {}, dt {} s, realtime factor {}", args.venue.display(), args.scene, venue.backend, args.zenoh, args.dt, args.realtime_factor);

    let wall0 = Instant::now();
    let mut last_status = start;
    let lidar_xyz = venue.rig.lidar.xyz;
    while !stop.load(Ordering::Relaxed) {
        if args.duration.is_some_and(|d| robot.sim.now() - start >= d) {
            break;
        }
        robot.drain()?;
        let stepped = robot.step(lidar_xyz)?;
        let tick = robot.sim.tick;
        tick_sensors(&venue, &mut [&mut robot], std::slice::from_ref(&stepped), tick, args.dt)?;
        robot.record_crowd(tick)?;
        if stepped.now - last_status >= args.status_every {
            last_status = stepped.now;
            let mut status = robot.status(wall0.elapsed().as_secs_f64());
            status.as_object_mut().map(|o| o.remove("run"));
            println!("{status}");
            robot.replay.flush()?;
        }
        pace(wall0, stepped.now - start, args.realtime_factor);
    }
    robot.close()
}
