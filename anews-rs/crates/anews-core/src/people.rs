use crate::actors::{Actor, ActorTimeline};
use crate::fidelity::{trip_frames, Frame, WallNear, COMPARE_RANGE};
use crate::occupancy::OccupancyMap;
use crate::scene_fit::{Evidence, FitConfig};
use crate::sensor3d::mat_vec;
use crate::trip::{Mount, Trip};
use crate::venue::Venue;
use crate::world3d::WallGrid;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

pub const WALKER_TYPE: &str = "person_standing";
pub const SEATED_TYPE: &str = "person_seated";

#[derive(Debug, Clone)]
pub struct TrackConfig {
    pub link_m: f64,
    pub body_m: f64,
    pub surface_to_center_m: f64,
    pub min_points: usize,
    pub max_spread_m: f64,
    pub gate_m: f64,
    pub max_gate_m: f64,
    pub walk_speed: f64,
    pub max_speed: f64,
    pub clear_frames: u32,
    pub min_detections: usize,
    pub min_life_s: f64,
    pub min_still_life_s: f64,
    pub still_path_m: f64,
    pub join_gap_s: f64,
    pub join_reach_m: f64,
    pub smooth_s: f64,
    pub heading_s: f64,
    pub heading_min_m: f64,
    pub sample_s: f64,
    pub static_dilate: usize,
    pub robot_clear_m: f64,
    pub seat_reach_m: f64,
    pub seated_path_m: f64,
}

impl Default for TrackConfig {
    fn default() -> Self {
        TrackConfig {
            link_m: 0.12,
            body_m: 0.45,
            surface_to_center_m: 0.07,
            min_points: 3,
            max_spread_m: 0.9,
            gate_m: 0.35,
            max_gate_m: 0.9,
            walk_speed: 1.0,
            max_speed: 1.8,
            clear_frames: 3,
            min_detections: 5,
            min_life_s: 0.5,
            min_still_life_s: 3.0,
            still_path_m: 0.3,
            join_gap_s: 2.5,
            join_reach_m: 0.5,
            smooth_s: 0.3,
            heading_s: 0.5,
            heading_min_m: 0.15,
            sample_s: 0.1,
            static_dilate: 2,
            robot_clear_m: 0.15,
            seat_reach_m: 0.3,
            seated_path_m: 0.6,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PeopleReport {
    pub frames: usize,
    pub detections: usize,
    pub raw_tracks: usize,
    pub people: usize,
    pub walking: usize,
    pub seated: usize,
    pub joined: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Copy)]
struct Detection {
    x: f64,
    y: f64,
    legs: Option<[[f64; 2]; 2]>,
}

struct Track {
    obs: Vec<[f64; 3]>,
    legs: Vec<[f64; 5]>,
    vx: f64,
    vy: f64,
    clear: u32,
    closed: Option<f64>,
}

impl Track {
    fn last(&self) -> [f64; 3] {
        *self.obs.last().unwrap()
    }

    fn predict(&self, t: f64) -> [f64; 2] {
        let l = self.last();
        let dt = (t - l[0]).min(1.0);
        [l[1] + self.vx * dt, l[2] + self.vy * dt]
    }
}

fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    let mut j = i;
    while parent[j] != r {
        let next = parent[j];
        parent[j] = r;
        j = next;
    }
    r
}

fn clusters(points: &[[f64; 2]], link: f64) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..points.len()).collect();
    let key = |p: &[f64; 2]| ((p[0] / link).floor() as i64, (p[1] / link).floor() as i64);
    let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        buckets.entry(key(p)).or_default().push(i);
    }
    for (i, p) in points.iter().enumerate() {
        let (kx, ky) = key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(js) = buckets.get(&(kx + dx, ky + dy)) {
                    for &j in js {
                        if j > i && (points[j][0] - p[0]).hypot(points[j][1] - p[1]) <= link {
                            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                            parent[a] = b;
                        }
                    }
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..points.len() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    groups.into_values().collect()
}

fn detect(points: &[([f64; 2], [f64; 2], bool)], tc: &TrackConfig) -> Vec<Detection> {
    let xy: Vec<[f64; 2]> = points.iter().map(|p| p.0).collect();
    let mut parts: Vec<([f64; 2], usize, Vec<usize>, bool)> = Vec::new();
    for g in clusters(&xy, tc.link_m) {
        let n = g.len() as f64;
        let (mut cx, mut cy, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0);
        for &i in &g {
            cx += xy[i][0] / n;
            cy += xy[i][1] / n;
            sx += points[i].1[0] / n;
            sy += points[i].1[1] / n;
        }
        let (dx, dy) = (cx - sx, cy - sy);
        let d = dx.hypot(dy).max(1e-9);
        let lidar = g.iter().any(|&i| points[i].2);
        parts.push(([cx + tc.surface_to_center_m * dx / d, cy + tc.surface_to_center_m * dy / d], g.len(), g, lidar));
    }
    let centers: Vec<[f64; 2]> = parts.iter().map(|p| p.0).collect();
    let mut out = Vec::new();
    for body in clusters(&centers, tc.body_m) {
        let count: usize = body.iter().map(|&k| parts[k].1).sum();
        if count < tc.min_points {
            continue;
        }
        let members: Vec<[f64; 2]> = body.iter().flat_map(|&k| parts[k].2.iter().map(|&i| xy[i])).collect();
        let spread = members.iter().flat_map(|a| members.iter().map(move |b| (a[0] - b[0]).hypot(a[1] - b[1]))).fold(0.0, f64::max);
        if spread > tc.max_spread_m {
            continue;
        }
        let k = body.len() as f64;
        let x = body.iter().map(|&b| centers[b][0]).sum::<f64>() / k;
        let y = body.iter().map(|&b| centers[b][1]).sum::<f64>() / k;
        let mut lidar_parts: Vec<usize> = body.iter().copied().filter(|&b| parts[b].3).collect();
        lidar_parts.sort_by(|a, b| parts[*b].1.cmp(&parts[*a].1));
        let legs = match lidar_parts.as_slice() {
            [a, b, ..] => Some([centers[*a], centers[*b]]),
            [a] => Some([centers[*a], [2.0 * x - centers[*a][0], 2.0 * y - centers[*a][1]]]),
            [] => None,
        };
        out.push(Detection { x, y, legs });
    }
    out
}

fn static_masks(venue: &Venue, map: &OccupancyMap, scene: &str, ev: &Evidence, cfg: &FitConfig, tc: &TrackConfig) -> Result<(Vec<bool>, Vec<bool>), String> {
    let mut low = WallGrid::from_map(map).for_scene(venue, Some(scene)).occupied;
    let mut high = low.clone();
    for item in venue.scene_items(Some(scene))? {
        for i in ev.item_cells(venue, &item, cfg.lidar_height, cfg.lidar_height) {
            low[i] = true;
        }
        for i in ev.item_cells(venue, &item, cfg.camera_band.0, cfg.camera_band.1) {
            high[i] = true;
        }
    }
    for _ in 0..tc.static_dilate {
        low = ev.dilate(&low);
        high = ev.dilate(&high);
    }
    Ok((low, high))
}

fn frame_points(f: &Frame, mount: &Mount, ev: &Evidence, low_static: &[bool], high_static: &[bool], cfg: &FitConfig) -> Vec<([f64; 2], [f64; 2], bool)> {
    let mut out = Vec::new();
    let (s, c) = f.pose.yaw.sin_cos();
    let m = mount.xyz;
    let lo = [f.pose.x + c * m[0] - s * m[1], f.pose.y + s * m[0] + c * m[1]];
    let scan = &f.lidar.scan;
    for (k, r) in scan.ranges.iter().enumerate() {
        let r = *r as f64;
        if !(r.is_finite() && r > 0.05 && r < COMPARE_RANGE) {
            continue;
        }
        let a = scan.angle_min as f64 + scan.angle_increment as f64 * k as f64;
        let local = mat_vec(&mount.rot, [a.cos(), a.sin(), 0.0]);
        let dir = [c * local[0] - s * local[1], s * local[0] + c * local[1]];
        let n = dir[0].hypot(dir[1]);
        let p = [lo[0] + dir[0] / n * r, lo[1] + dir[1] / n * r];
        if ev.cell(p[0], p[1]).is_some_and(|i| !low_static[i]) {
            out.push((p, lo, true));
        }
    }
    if let Some(cam) = f.camera {
        let scan = &cam.scan;
        for (k, r) in scan.ranges.iter().enumerate() {
            let r = *r as f64;
            if !(r.is_finite() && r > 0.05 && r < cfg.camera_max_range) {
                continue;
            }
            let a = f.pose.yaw + scan.angle_min as f64 + scan.angle_increment as f64 * k as f64;
            let p = [f.pose.x + a.cos() * r, f.pose.y + a.sin() * r];
            if ev.cell(p[0], p[1]).is_some_and(|i| !high_static[i]) {
                out.push((p, [f.pose.x, f.pose.y], false));
            }
        }
    }
    out
}

fn smooth(obs: &[[f64; 3]], half: f64) -> Vec<[f64; 3]> {
    obs.iter()
        .map(|o| {
            let (lo, hi) = (obs.partition_point(|p| p[0] < o[0] - half), obs.partition_point(|p| p[0] <= o[0] + half));
            let n = (hi - lo) as f64;
            let (x, y) = obs[lo..hi].iter().fold((0.0, 0.0), |(x, y), p| (x + p[1] / n, y + p[2] / n));
            [o[0], x, y]
        })
        .collect()
}

fn path_length(obs: &[[f64; 3]]) -> f64 {
    obs.windows(2).map(|w| (w[1][1] - w[0][1]).hypot(w[1][2] - w[0][2])).sum()
}

fn samples(obs: &[[f64; 3]], tc: &TrackConfig) -> Vec<[f64; 4]> {
    let sm = smooth(obs, tc.smooth_s);
    let mut picked: Vec<[f64; 3]> = Vec::new();
    for p in &sm {
        if picked.last().is_none_or(|l| p[0] - l[0] >= tc.sample_s) {
            picked.push(*p);
        }
    }
    if let (Some(l), Some(p)) = (picked.last(), sm.last())
        && p[0] > l[0]
    {
        picked.push(*p);
    }
    let at = |t: f64| -> [f64; 2] {
        let k = sm.partition_point(|p| p[0] < t).min(sm.len() - 1);
        [sm[k][1], sm[k][2]]
    };
    let mut yaws: Vec<Option<f64>> = picked
        .iter()
        .map(|p| {
            let (a, b) = (at(p[0] - tc.heading_s), at(p[0] + tc.heading_s));
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            (dx.hypot(dy) >= tc.heading_min_m).then(|| dy.atan2(dx))
        })
        .collect();
    let first = yaws.iter().flatten().next().copied().unwrap_or(0.0);
    let mut last = first;
    for y in yaws.iter_mut() {
        match y {
            Some(v) => last = *v,
            None => *y = Some(last),
        }
    }
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    picked.iter().zip(yaws).map(|(p, y)| [round(p[0]), round(p[1]), round(p[2]), round(y.unwrap())]).collect()
}

pub fn track_people(
    venue: &Venue,
    map: &OccupancyMap,
    scene: &str,
    trips: &[Trip],
    cfg: &FitConfig,
    tc: &TrackConfig,
    refine: Option<&WallNear>,
) -> Result<(ActorTimeline, PeopleReport), String> {
    let walker = venue.types.get(WALKER_TYPE).ok_or_else(|| format!("venue has no {WALKER_TYPE} type; tracked people use it"))?;
    let ev = Evidence::new(map);
    let (low_static, high_static) = static_masks(venue, map, scene, &ev, cfg, tc)?;
    let mut passed = vec![0u32; ev.lidar_hit.len()];
    let mut tracks: Vec<Track> = Vec::new();
    let (mut frames, mut detections, mut start, mut end) = (0usize, 0usize, f64::INFINITY, f64::NEG_INFINITY);
    for trip in trips {
        let mount = trip.mount("laser_link", "base_footprint")?;
        for f in trip_frames(trip, 0.0, cfg.window, refine)? {
            let t = f.lidar.t;
            frames += 1;
            start = start.min(t);
            end = end.max(t);
            let stamp = frames as u32;
            for i in ev.observe(&f, &mount, cfg).lidar_passes {
                passed[i] = stamp;
            }
            let dets = detect(&frame_points(&f, &mount, &ev, &low_static, &high_static, cfg), tc);
            detections += dets.len();
            let open: Vec<usize> = (0..tracks.len()).filter(|&k| tracks[k].closed.is_none()).collect();
            let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
            for &k in &open {
                let tr = &tracks[k];
                let p = tr.predict(t);
                let gate = (tc.gate_m + tc.max_speed * (t - tr.last()[0])).min(tc.max_gate_m);
                for (d, det) in dets.iter().enumerate() {
                    let dist = (det.x - p[0]).hypot(det.y - p[1]);
                    if dist <= gate {
                        pairs.push((dist, k, d));
                    }
                }
            }
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut track_used, mut det_used) = (vec![false; tracks.len()], vec![false; dets.len()]);
            for (_, k, d) in pairs {
                if track_used[k] || det_used[d] {
                    continue;
                }
                track_used[k] = true;
                det_used[d] = true;
                let tr = &mut tracks[k];
                let det = dets[d];
                tr.obs.push([t, det.x, det.y]);
                if let Some(l) = det.legs {
                    tr.legs.push([t, l[0][0], l[0][1], l[1][0], l[1][1]]);
                }
                tr.clear = 0;
                let back = tr.obs.partition_point(|o| o[0] < t - 0.5);
                let o = tr.obs[back.min(tr.obs.len() - 1)];
                let dt = t - o[0];
                if dt > 0.1 {
                    let (vx, vy) = ((det.x - o[1]) / dt, (det.y - o[2]) / dt);
                    let speed = vx.hypot(vy);
                    let k = if speed > tc.max_speed { tc.max_speed / speed } else { 1.0 };
                    tr.vx = vx * k;
                    tr.vy = vy * k;
                }
            }
            let (ys, yc) = f.pose.yaw.sin_cos();
            let under_robot = |p: [f64; 3]| {
                let (dx, dy) = (p[1] - f.pose.x, p[2] - f.pose.y);
                (yc * dx + ys * dy).abs() <= cfg.footprint_half[0] + tc.robot_clear_m && (-ys * dx + yc * dy).abs() <= cfg.footprint_half[1] + tc.robot_clear_m
            };
            for &k in &open {
                if track_used[k] {
                    continue;
                }
                let tr = &mut tracks[k];
                if under_robot(tr.last()) {
                    tr.closed = Some(t);
                    continue;
                }
                if t - tr.last()[0] > 1.0 {
                    tr.vx = 0.0;
                    tr.vy = 0.0;
                }
                let l = tr.last();
                let seen_through = ev.cell(l[1], l[2]).map_or(0, |i| ev.neighbours(i).filter(|&j| passed[j] == stamp).count());
                if seen_through >= 3 {
                    tr.clear += 1;
                    if tr.clear >= tc.clear_frames {
                        tr.closed = Some(t);
                    }
                } else {
                    tr.clear = 0;
                }
            }
            for (d, det) in dets.iter().enumerate() {
                if !det_used[d] {
                    let legs = det.legs.map(|l| vec![[t, l[0][0], l[0][1], l[1][0], l[1][1]]]).unwrap_or_default();
                    tracks.push(Track { obs: vec![[t, det.x, det.y]], legs, vx: 0.0, vy: 0.0, clear: 0, closed: None });
                }
            }
        }
    }
    let raw_tracks = tracks.len();
    let mut kept: Vec<(Vec<[f64; 3]>, f64, Vec<[f64; 5]>)> = tracks
        .into_iter()
        .filter(|tr| {
            let life = tr.last()[0] - tr.obs[0][0];
            let still = path_length(&smooth(&tr.obs, tc.smooth_s)) < tc.still_path_m;
            tr.obs.len() >= tc.min_detections && life >= tc.min_life_s && !(still && life < tc.min_still_life_s)
        })
        .map(|tr| {
            let close = tr.closed.unwrap_or(end);
            (tr.obs, close, tr.legs)
        })
        .collect();
    kept.sort_by(|a, b| a.0[0][0].total_cmp(&b.0[0][0]));
    let mut joined = 0;
    let mut merged: Vec<(Vec<[f64; 3]>, f64, Vec<[f64; 5]>)> = Vec::new();
    for (obs, close, legs) in kept {
        let first = obs[0];
        let best = merged
            .iter()
            .enumerate()
            .filter_map(|(k, (m, _, _))| {
                let l = *m.last().unwrap();
                let gap = first[0] - l[0];
                let dist = (first[1] - l[1]).hypot(first[2] - l[2]);
                (gap > 0.0 && gap <= tc.join_gap_s && dist <= tc.join_reach_m + tc.walk_speed * gap).then_some((gap, k))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            Some((_, k)) => {
                merged[k].0.extend(obs);
                merged[k].1 = close;
                merged[k].2.extend(legs);
                joined += 1;
            }
            None => merged.push((obs, close, legs)),
        }
    }
    let seats: Vec<crate::venue::Item> = venue
        .scene_items(Some(scene))?
        .into_iter()
        .filter(|i| i.size.is_some() && (i.kind == "table" || i.type_name == "booth"))
        .collect();
    let seat_at = |x: f64, y: f64| -> Option<f64> {
        seats.iter().filter(|i| crate::objects_fit::covers(i, x, y, tc.seat_reach_m)).map(|i| (i.y - y).atan2(i.x - x)).next()
    };
    let seated_type = venue.types.get(SEATED_TYPE).cloned();
    let mut actors = Vec::new();
    let mut walking = 0;
    let mut seated = 0;
    for (k, (obs, close, legs)) in merged.iter().enumerate() {
        if path_length(&smooth(obs, tc.smooth_s)) >= 1.0 {
            walking += 1;
        }
        let mut s = samples(obs, tc);
        let last = s.last().map_or(*close, |l| l[0]);
        let mut sorted: Vec<[f64; 2]> = s.iter().map(|p| [p[1], p[2]]).collect();
        let mid = sorted.len() / 2;
        sorted.select_nth_unstable_by(mid, |a, b| a[0].total_cmp(&b[0]));
        let mx = sorted[mid][0];
        sorted.select_nth_unstable_by(mid, |a, b| a[1].total_cmp(&b[1]));
        let my = sorted[mid][1];
        let still = path_length(&smooth(obs, tc.smooth_s)) < tc.seated_path_m;
        let seat = if still && seated_type.is_some() { seat_at(mx, my) } else { None };
        if let Some(facing) = seat {
            seated += 1;
            s = vec![[s[0][0], mx, my, facing], [last, mx, my, facing]];
            let round = |v: f64| (v * 1000.0).round() / 1000.0;
            actors.push(Actor { id: format!("person_{}", k + 1), type_name: SEATED_TYPE.into(), samples: s.iter().map(|p| p.map(round)).collect(), end: close.max(last), legs: Vec::new() });
            continue;
        }
        let round = |v: f64| (v * 1000.0).round() / 1000.0;
        let legs = legs.iter().map(|l| [round(l[0]), round(l[1]), round(l[2]), round(l[3]), round(l[4])]).collect();
        actors.push(Actor { id: format!("person_{}", k + 1), type_name: WALKER_TYPE.into(), samples: s, end: close.max(last), legs });
    }
    let report = PeopleReport { frames, detections, raw_tracks, people: actors.len(), walking, seated, joined, start, end };
    let mut types = BTreeMap::from([(WALKER_TYPE.to_string(), walker.clone())]);
    if let Some(t) = seated_type {
        types.insert(SEATED_TYPE.to_string(), t);
    }
    let timeline = ActorTimeline { start, end, types, actors };
    Ok((timeline, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_leg_arcs_half_a_metre_apart_are_one_person_and_a_far_arc_is_another() {
        let sensor = [0.0, 0.0];
        let arc = |x: f64, y: f64| (0..4).map(move |k| ([x, y + k as f64 * 0.02], sensor, true));
        let pts: Vec<_> = arc(2.0, 0.0).chain(arc(2.0, 0.25)).chain(arc(3.5, 2.0)).collect();
        let dets = detect(&pts, &TrackConfig::default());
        assert_eq!(dets.len(), 2, "{dets:?}");
        let person = dets.iter().find(|d| d.y < 1.0).unwrap();
        assert!((person.y - 0.155).abs() < 0.02, "midway between the legs: {}", person.y);
        assert!(person.x > 2.0, "the centre sits behind the seen surface: {}", person.x);
    }

    #[test]
    fn heading_follows_the_walk_and_holds_while_standing() {
        let mut obs: Vec<[f64; 3]> = (0..20).map(|k| [k as f64 * 0.1, 0.0, k as f64 * 0.1]).collect();
        obs.extend((20..40).map(|k| [k as f64 * 0.1, 0.0, 1.9]));
        let s = samples(&obs, &TrackConfig::default());
        assert!((s[5][3] - std::f64::consts::FRAC_PI_2).abs() < 0.05, "walking north: {}", s[5][3]);
        assert!((s.last().unwrap()[3] - std::f64::consts::FRAC_PI_2).abs() < 0.05, "still faces north after stopping");
    }
}
