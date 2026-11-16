use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct StateExtra {
    v: Option<f64>,
    w: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Scan {
    pub t: f64,
    pub origin: [f64; 3],
    pub points: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct Goal {
    pub t: f64,
    pub xy: Option<[f64; 2]>,
    pub label: Option<String>,
}

#[derive(Debug, Default)]
pub struct Timeline {
    pub source: String,
    pub start_unix: Option<f64>,
    pub scene: Option<String>,
    states: Vec<(f64, Pose, StateExtra)>,
    pub lidar: Vec<Scan>,
    pub camera: Vec<Scan>,
    pub plans: Vec<(f64, Vec<[f64; 2]>)>,
    pub nav_events: Vec<(f64, Value)>,
    goals: Vec<Goal>,
    venue_bumps: Vec<f64>,
    crowd_bumps: Vec<f64>,
    crowd: Vec<(f64, Vec<[f32; 3]>)>,
    trail: Vec<(f64, [f64; 2])>,
    progress: Vec<f64>,
    pub duration: f64,
}

pub struct Frame<'a> {
    pub t: f64,
    pub pose: Pose,
    pub v: f64,
    pub w: f64,
    pub lidar: Option<&'a Scan>,
    pub camera: Option<&'a Scan>,
    pub plan: Option<(f64, &'a [[f64; 2]])>,
    pub goal: Option<[f64; 2]>,
    pub goal_label: Option<&'a str>,
    pub nav_event: Option<(f64, &'a Value)>,
    pub still_s: f64,
    pub bumps: u64,
    pub crowd_bumps: u64,
    pub nearest: Option<f64>,
    pub trail: &'a [(f64, [f64; 2])],
    pub crowd: &'a [[f32; 3]],
}

const PROGRESS_M: f64 = 0.3;
const TRAIL_STEP_M: f64 = 0.05;
const SPEED_HALF_WINDOW_S: f64 = 0.25;

fn f(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn xyz(v: &Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    Some([f(a.first()?)?, f(a.get(1)?)?, a.get(2).and_then(f).unwrap_or(0.0)])
}

fn xy(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array()?;
    Some([f(a.first()?)?, f(a.get(1)?)?])
}

fn points(v: &Value) -> Vec<[f32; 3]> {
    v.as_array().map(|a| a.iter().filter_map(xyz).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect()).unwrap_or_default()
}

fn wrap(a: f64) -> f64 {
    a.sin().atan2(a.cos())
}

pub fn parse_jsonl(text: &str) -> Result<Vec<Value>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("replay line {}: {e}", i + 1)))
        .collect()
}

impl Timeline {
    pub fn from_events(events: &[Value]) -> Result<Self, String> {
        let mut tl = Timeline::default();
        for e in events {
            let t = e["t"].as_f64().ok_or("replay event without t")?;
            tl.duration = tl.duration.max(t);
            match e["event"].as_str().unwrap_or("") {
                "init" => {
                    tl.source = e["source"].as_str().unwrap_or("").to_string();
                    tl.start_unix = e["start_unix"].as_f64();
                    tl.scene = e["scene"].as_str().map(str::to_string);
                    if let Some(r) = xyz(&e["robot"]) {
                        tl.states.push((t, Pose { x: r[0], y: r[1], yaw: r[2] }, StateExtra::default()));
                    }
                    if let Some(g) = xy(&e["goal"]) {
                        tl.goals.push(Goal { t, xy: Some(g), label: e["goal_label"].as_str().map(str::to_string) });
                    }
                }
                "state" => {
                    let r = xyz(&e["robot"]).ok_or("state event without robot [x, y, yaw]")?;
                    let extra = StateExtra { v: e["v"].as_f64(), w: e["w"].as_f64() };
                    tl.states.push((t, Pose { x: r[0], y: r[1], yaw: r[2] }, extra));
                }
                "lidar" => tl.lidar.push(Scan { t, origin: xyz(&e["origin"]).unwrap_or_default(), points: points(&e["points"]) }),
                "camera_scan" | "depth" => tl.camera.push(Scan { t, origin: xyz(&e["origin"]).unwrap_or_default(), points: points(&e["points"]) }),
                "plan" => {
                    let path = e["path"].as_array().map(|a| a.iter().filter_map(xy).collect()).unwrap_or_default();
                    tl.plans.push((t, path));
                }
                "nav_event" => tl.nav_events.push((t, e["data"].clone())),
                "goal" => tl.goals.push(Goal { t, xy: xy(&e["xy"]), label: e["name"].as_str().or(e["label"].as_str()).map(str::to_string) }),
                "bump" => match e["with"].as_str() {
                    Some("crowd") => tl.crowd_bumps.push(t),
                    _ => tl.venue_bumps.push(t),
                },
                "crowd" => tl.crowd.push((t, points(&e["cells"]))),
                _ => {}
            }
        }
        if tl.states.is_empty() {
            return Err("replay has no robot pose".into());
        }
        let by_t = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        tl.states.sort_by(|a, b| by_t(&a.0, &b.0));
        tl.lidar.sort_by(|a, b| by_t(&a.t, &b.t));
        tl.camera.sort_by(|a, b| by_t(&a.t, &b.t));
        tl.plans.sort_by(|a, b| by_t(&a.0, &b.0));
        tl.nav_events.sort_by(|a, b| by_t(&a.0, &b.0));
        tl.goals.sort_by(|a, b| by_t(&a.t, &b.t));
        tl.venue_bumps.sort_by(by_t);
        tl.crowd_bumps.sort_by(by_t);
        tl.crowd.sort_by(|a, b| by_t(&a.0, &b.0));
        let mut anchor = tl.states[0].1;
        tl.progress.push(tl.states[0].0);
        let mut last = [anchor.x, anchor.y];
        tl.trail.push((tl.states[0].0, last));
        for (t, p, _) in &tl.states[1..] {
            if (p.x - anchor.x).hypot(p.y - anchor.y) > PROGRESS_M {
                anchor = *p;
                tl.progress.push(*t);
            }
            if (p.x - last[0]).hypot(p.y - last[1]) > TRAIL_STEP_M {
                last = [p.x, p.y];
                tl.trail.push((*t, last));
            }
        }
        Ok(tl)
    }

    pub fn pose_at(&self, t: f64) -> Pose {
        let s = &self.states;
        let i = s.partition_point(|e| e.0 < t);
        if i == 0 {
            return s[0].1;
        }
        if i == s.len() {
            return s[i - 1].1;
        }
        let ((ta, a, _), (tb, b, _)) = (s[i - 1], s[i]);
        let w = if tb > ta { (t - ta) / (tb - ta) } else { 0.0 };
        Pose { x: a.x + w * (b.x - a.x), y: a.y + w * (b.y - a.y), yaw: wrap(a.yaw + w * wrap(b.yaw - a.yaw)) }
    }

    fn latest<T>(items: &[T], t: f64, time: impl Fn(&T) -> f64) -> Option<&T> {
        let i = items.partition_point(|e| time(e) <= t);
        if i == 0 { None } else { Some(&items[i - 1]) }
    }

    pub fn window(&self, from: Option<f64>, to: Option<f64>, fps: u32) -> Result<(f64, usize), String> {
        let from = from.unwrap_or(0.0).max(0.0);
        let to = to.unwrap_or(self.duration).min(self.duration);
        if to <= from {
            return Err(format!("empty time window {from:.2}..{to:.2} s (replay lasts {:.2} s)", self.duration));
        }
        Ok((from, ((to - from) * fps as f64 + 1e-9).floor() as usize))
    }

    pub fn frame(&self, t: f64) -> Frame<'_> {
        let pose = self.pose_at(t);
        let state = Self::latest(&self.states, t, |e| e.0).map(|e| e.2).unwrap_or_default();
        let (v, w) = match (state.v, state.w) {
            (Some(v), Some(w)) => (v, w),
            _ => {
                let a = self.pose_at(t - SPEED_HALF_WINDOW_S);
                let b = self.pose_at(t + SPEED_HALF_WINDOW_S);
                let dt = 2.0 * SPEED_HALF_WINDOW_S;
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                let sign = if dx * pose.yaw.cos() + dy * pose.yaw.sin() < 0.0 { -1.0 } else { 1.0 };
                (sign * dx.hypot(dy) / dt, wrap(b.yaw - a.yaw) / dt)
            }
        };
        let lidar = Self::latest(&self.lidar, t, |s| s.t);
        let nearest = lidar.and_then(|s| {
            s.points
                .iter()
                .map(|p| (p[0] as f64 - s.origin[0]).hypot(p[1] as f64 - s.origin[1]))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        });
        let plan = Self::latest(&self.plans, t, |p| p.0).map(|(pt, path)| (*pt, path.as_slice()));
        let goal_event = Self::latest(&self.goals, t, |g| g.t);
        let goal = goal_event.and_then(|g| g.xy).or_else(|| plan.and_then(|(_, p)| p.last().copied()));
        let goal_label = goal_event
            .and_then(|g| g.label.as_deref())
            .or_else(|| {
                let i = self.nav_events.partition_point(|e| e.0 <= t);
                self.nav_events[..i]
                    .iter()
                    .rev()
                    .chain(self.nav_events[i..].iter())
                    .find_map(|e| e.1["goal_name"].as_str())
            });
        let still_from = Self::latest(&self.progress, t, |p| *p).copied().unwrap_or(0.0);
        let trail_end = self.trail.partition_point(|e| e.0 <= t);
        Frame {
            t,
            pose,
            v,
            w,
            lidar,
            camera: Self::latest(&self.camera, t, |s| s.t),
            plan,
            goal,
            goal_label,
            nav_event: Self::latest(&self.nav_events, t, |e| e.0).map(|(et, d)| (*et, d)),
            still_s: (t - still_from).max(0.0),
            bumps: self.venue_bumps.partition_point(|b| *b <= t) as u64,
            crowd_bumps: self.crowd_bumps.partition_point(|b| *b <= t) as u64,
            nearest,
            trail: &self.trail[..trail_end],
            crowd: Self::latest(&self.crowd, t, |c| c.0).map(|c| c.1.as_slice()).unwrap_or(&[]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn events() -> Vec<Value> {
        vec![
            json!({"t": 0.0, "event": "init", "schema": "anews.replay/1", "source": "x.mcap", "start_unix": 1000.0, "robot": [0.0, 0.0, 0.0]}),
            json!({"t": 0.0, "event": "state", "robot": [0.0, 0.0, 0.0]}),
            json!({"t": 1.0, "event": "state", "robot": [1.0, 0.0, 0.0]}),
            json!({"t": 2.0, "event": "state", "robot": [1.0, 1.0, std::f64::consts::FRAC_PI_2]}),
            json!({"t": 0.5, "event": "lidar", "origin": [0.2, 0.0, 0.23], "points": [[2.2, 0.0, 0.23]]}),
            json!({"t": 1.5, "event": "lidar", "origin": [1.2, 0.0, 0.23], "points": [[1.2, 0.5, 0.23], [3.2, 0.0, 0.23]]}),
            json!({"t": 0.8, "event": "plan", "path": [[0.0, 0.0], [4.0, 3.0]]}),
            json!({"t": 1.2, "event": "nav_event", "data": {"event": "give_way_prompt", "goal_name": "11"}}),
        ]
    }

    #[test]
    fn frame_at_time_interpolates_pose_and_holds_the_latest_scan_and_plan() {
        let tl = Timeline::from_events(&events()).unwrap();
        let f = tl.frame(0.5);
        assert!((f.pose.x - 0.5).abs() < 1e-9);
        assert!(f.plan.is_none(), "the plan arrives at 0.8 s and must not show early");
        assert_eq!(f.lidar.unwrap().t, 0.5);
        let f = tl.frame(1.6);
        assert_eq!(f.lidar.unwrap().t, 1.5);
        assert!((f.nearest.unwrap() - 0.5).abs() < 1e-6);
        assert_eq!(f.plan.unwrap().0, 0.8);
        assert_eq!(f.goal, Some([4.0, 3.0]));
        assert_eq!(f.goal_label, Some("11"));
        assert!((f.pose.yaw - 0.6 * std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn live_replay_counts_bumps_and_names_the_goal_up_to_each_frame() {
        let mut ev = events();
        ev.extend([
            json!({"t": 0.2, "event": "goal", "name": "table 11", "crowd_anchored": true}),
            json!({"t": 0.6, "event": "bump", "with": "venue"}),
            json!({"t": 1.1, "event": "bump", "with": "crowd"}),
            json!({"t": 1.4, "event": "bump", "with": "venue"}),
            json!({"t": 0.9, "event": "depth", "points": [[1.5, 0.2, 0.3]]}),
            json!({"t": 0.4, "event": "crowd", "cells": [[2.0, 1.0, 1], [2.05, 1.0, 3]]}),
            json!({"t": 1.3, "event": "crowd", "cells": []}),
        ]);
        let tl = Timeline::from_events(&ev).unwrap();
        let f = tl.frame(1.2);
        assert_eq!((f.bumps, f.crowd_bumps), (1, 1));
        assert_eq!(f.goal_label, Some("table 11"));
        assert_eq!(f.goal, Some([4.0, 3.0]), "a named goal without coordinates falls back to the plan end");
        assert_eq!(f.camera.unwrap().t, 0.9);
        assert_eq!(f.crowd.len(), 2);
        assert!(tl.frame(1.35).crowd.is_empty(), "an empty crowd event clears the cells");
        assert_eq!(tl.frame(1.5).bumps, 2);
        assert_eq!(tl.frame(0.5).bumps, 0);
    }

    #[test]
    fn goal_label_before_the_first_nav_event_comes_from_the_upcoming_request() {
        let tl = Timeline::from_events(&events()).unwrap();
        assert_eq!(tl.frame(0.1).goal_label, Some("11"));
    }

    #[test]
    fn time_window_yields_one_frame_per_tick_inside_the_replay() {
        let tl = Timeline::from_events(&events()).unwrap();
        assert_eq!(tl.window(None, None, 25).unwrap(), (0.0, 50));
        assert_eq!(tl.window(Some(0.5), Some(1.5), 25).unwrap(), (0.5, 25));
        assert_eq!(tl.window(Some(1.0), Some(9.0), 10).unwrap(), (1.0, 10));
        assert!(tl.window(Some(3.0), None, 25).is_err());
    }

    #[test]
    fn speed_from_poses_is_signed_by_heading() {
        let tl = Timeline::from_events(&events()).unwrap();
        let f = tl.frame(0.5);
        assert!((f.v - 1.0).abs() < 1e-9, "{}", f.v);
        let back = Timeline::from_events(&[
            json!({"t": 0.0, "event": "state", "robot": [1.0, 0.0, 0.0]}),
            json!({"t": 1.0, "event": "state", "robot": [0.0, 0.0, 0.0]}),
        ])
        .unwrap();
        assert!(back.frame(0.5).v < -0.9);
    }
}
