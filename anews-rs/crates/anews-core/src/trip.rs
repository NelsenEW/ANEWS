use crate::cdr::{self, LaserScan, Pose, Transform};
use crate::mcap::{read_mcap, Recording};
use crate::sensor3d::{mat_vec, quat_matrix};
use crate::world3d::V3;
use serde::Serialize;
use std::collections::BTreeMap;

pub const POSE_TOPIC: &str = "/robot_api/current_pose";
pub const LIDAR_TOPIC: &str = "/scan";
pub const CAMERA_SCAN_TOPIC: &str = "/camera/scan";
pub const TF_STATIC_TOPIC: &str = "/tf_static";
pub const PLAN_TOPIC: &str = "/plan";
pub const NAV_EVENTS_TOPIC: &str = "/robot_api/nav_events";

pub type Mat3 = [[f64; 3]; 3];

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PlanarPose {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

#[derive(Debug, Clone)]
pub struct TimedScan {
    pub t: f64,
    pub scan: LaserScan,
}

#[derive(Debug, Clone)]
pub struct Mount {
    pub xyz: V3,
    pub rot: Mat3,
}

#[derive(Debug, Default)]
pub struct Trip {
    pub poses: Vec<PlanarPose>,
    pub scans: Vec<TimedScan>,
    pub camera_scans: Vec<TimedScan>,
    pub statics: BTreeMap<String, Transform>,
    pub plans: Vec<(f64, Vec<Pose>)>,
    pub nav_events: Vec<(f64, String)>,
}

fn wanted(topic: &str) -> bool {
    matches!(topic, POSE_TOPIC | LIDAR_TOPIC | CAMERA_SCAN_TOPIC | TF_STATIC_TOPIC | PLAN_TOPIC | NAV_EVENTS_TOPIC)
}

pub fn read_trip(bytes: &[u8]) -> Result<Trip, String> {
    let rec = read_mcap(bytes, &wanted)?;
    trip_from_recording(&rec)
}

pub fn trip_from_recording(rec: &Recording) -> Result<Trip, String> {
    let mut trip = Trip::default();
    for m in &rec.messages {
        let t = m.log_time as f64 * 1e-9;
        match rec.topic_of(m) {
            Some(POSE_TOPIC) => {
                let (_, p) = cdr::pose_stamped(&m.data)?;
                trip.poses.push(PlanarPose { t, x: p.position[0], y: p.position[1], yaw: p.yaw() });
            }
            Some(LIDAR_TOPIC) => trip.scans.push(TimedScan { t, scan: cdr::laser_scan(&m.data)? }),
            Some(CAMERA_SCAN_TOPIC) => trip.camera_scans.push(TimedScan { t, scan: cdr::laser_scan(&m.data)? }),
            Some(TF_STATIC_TOPIC) => {
                for tr in cdr::tf_message(&m.data)? {
                    trip.statics.insert(tr.child.clone(), tr);
                }
            }
            Some(PLAN_TOPIC) => trip.plans.push((t, cdr::path(&m.data)?)),
            Some(NAV_EVENTS_TOPIC) => trip.nav_events.push((t, cdr::string(&m.data)?)),
            _ => {}
        }
    }
    Ok(trip)
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub const BASE_LINK_HEIGHT: f64 = 0.0102;

impl Trip {
    pub fn mount(&self, frame: &str, root: &str) -> Result<Mount, String> {
        let mut xyz = [0.0; 3];
        let mut rot = IDENTITY;
        let mut frame = frame.to_string();
        while frame != root {
            let Some(tr) = self.statics.get(&frame) else {
                if frame == "base_link" && root == "base_footprint" {
                    return Ok(Mount { xyz: [xyz[0], xyz[1], xyz[2] + BASE_LINK_HEIGHT], rot });
                }
                return Err(format!("no static transform for {frame}"));
            };
            let r = quat_matrix(tr.rotation);
            let moved = mat_vec(&r, xyz);
            xyz = [tr.translation[0] + moved[0], tr.translation[1] + moved[1], tr.translation[2] + moved[2]];
            rot = mat_mul(&r, &rot);
            frame = tr.parent.clone();
        }
        Ok(Mount { xyz, rot })
    }

    pub fn pose_at(&self, t: f64) -> Option<(PlanarPose, f64)> {
        let n = self.poses.len();
        if n < 2 {
            return None;
        }
        let i = self.poses.partition_point(|p| p.t < t).clamp(1, n - 1);
        let (a, b) = (self.poses[i - 1], self.poses[i]);
        let w = if b.t == a.t { 0.0 } else { ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0) };
        let dyaw = (b.yaw - a.yaw).sin().atan2((b.yaw - a.yaw).cos());
        let gap = (t - a.t).abs().min((t - b.t).abs());
        Some((PlanarPose { t, x: a.x + w * (b.x - a.x), y: a.y + w * (b.y - a.y), yaw: a.yaw + w * dyaw }, gap))
    }
}
