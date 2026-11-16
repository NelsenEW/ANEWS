use crate::sensor3d::mat_vec;
use crate::trip::{mat_mul, Trip};
use serde_json::{json, Value};

pub const REPLAY_SCHEMA: &str = "anews.replay/1";

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn scan_xyz(ranges: &[f32], amin: f64, inc: f64, origin: [f64; 3], rot: &[[f64; 3]; 3], r_max: f64) -> Vec<[f64; 3]> {
    ranges
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_finite() && **r > 0.05 && (**r as f64) < r_max)
        .map(|(i, r)| {
            let a = amin + inc * i as f64;
            let r = *r as f64;
            let p = mat_vec(rot, [r * a.cos(), r * a.sin(), 0.0]);
            [r3(origin[0] + p[0]), r3(origin[1] + p[1]), r3(origin[2] + p[2])]
        })
        .collect()
}

pub fn import_trip(name: &str, trip: &Trip, r_max: f64) -> Result<Vec<Value>, String> {
    let t0 = trip.poses.first().map(|p| p.t).ok_or("recording has no robot pose")?;
    let lidar = trip.mount("laser_link", "base_footprint")?;
    let camera = trip.mount("camera_link", "base_footprint").ok();
    let mut events = vec![json!({
        "t": 0.0,
        "event": "init",
        "schema": REPLAY_SCHEMA,
        "source": name,
        "start_unix": t0,
        "robot": [trip.poses[0].x, trip.poses[0].y, trip.poses[0].yaw],
        "mounts": {
            "lidar": {"xyz": lidar.xyz, "rot": lidar.rot},
            "camera": camera.map(|m| json!({"xyz": m.xyz, "rot": m.rot})),
        },
    })];
    for p in &trip.poses {
        events.push(json!({"t": p.t - t0, "event": "state", "robot": [r3(p.x), r3(p.y), r3(p.yaw)]}));
    }
    for s in &trip.scans {
        let Some((pose, _)) = trip.pose_at(s.t) else { continue };
        let (sn, cs) = pose.yaw.sin_cos();
        let base_rot = [[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]];
        let m = mat_vec(&base_rot, lidar.xyz);
        let origin = [pose.x + m[0], pose.y + m[1], m[2]];
        let rot = mat_mul(&base_rot, &lidar.rot);
        let pts = scan_xyz(&s.scan.ranges, s.scan.angle_min as f64, s.scan.angle_increment as f64, origin, &rot, r_max);
        events.push(json!({"t": s.t - t0, "event": "lidar", "origin": origin, "points": pts}));
    }
    for s in &trip.camera_scans {
        let Some((pose, _)) = trip.pose_at(s.t) else { continue };
        let (sn, cs) = pose.yaw.sin_cos();
        let base_rot = [[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]];
        let pts = scan_xyz(&s.scan.ranges, s.scan.angle_min as f64, s.scan.angle_increment as f64, [pose.x, pose.y, 0.3], &base_rot, r_max);
        events.push(json!({"t": s.t - t0, "event": "camera_scan", "points": pts}));
    }
    for (t, poses) in &trip.plans {
        let xy: Vec<[f64; 2]> = poses.iter().map(|p| [r3(p.position[0]), r3(p.position[1])]).collect();
        events.push(json!({"t": t - t0, "event": "plan", "path": xy}));
    }
    for (t, text) in &trip.nav_events {
        let parsed: Value = serde_json::from_str(text).unwrap_or(Value::String(text.clone()));
        events.push(json!({"t": t - t0, "event": "nav_event", "data": parsed}));
    }
    events.sort_by(|a, b| a["t"].as_f64().unwrap().partial_cmp(&b["t"].as_f64().unwrap()).unwrap());
    Ok(events)
}

pub fn to_jsonl(events: &[Value]) -> String {
    let mut out = String::new();
    for e in events {
        out.push_str(&e.to_string());
        out.push('\n');
    }
    out
}
