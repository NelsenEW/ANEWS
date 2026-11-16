use crate::depth_sensor::{DepthSensor, Rng};
use crate::fidelity::{beam_points, cells, dilate, f64s, linspace, trip_frames, Base, Mode, SimWorld, WallNear, LIDAR_ANGLE_MAX, LIDAR_ANGLE_MIN, LIDAR_COUNT, STEP_S};
use crate::objects_fit::covers;
use crate::occupancy::OccupancyMap;
use crate::trip::{mat_mul, Mount, Trip};
use crate::venue::{Item, Scene, Venue};
use crate::world3d::{WallGrid, World3d};
use serde::Serialize;
use std::collections::HashMap;
use std::f64::consts::PI;

pub const GATE_SCENE: &str = "__object_gate";

#[derive(Debug, Clone)]
pub struct GateConfig {
    pub region_pad_m: f64,
    pub view_m: f64,
    pub tolerance: f64,
    pub replace_pad_m: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { region_pad_m: 0.6, view_m: 4.5, tolerance: 0.0, replace_pad_m: 0.05 }
    }
}

struct GateFrame {
    base: Base,
    lidar: Mount,
    camera: Mount,
    real_lidar: Vec<[f64; 2]>,
    real_camera: Option<Vec<[f64; 2]>>,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GateDecision {
    pub id: String,
    pub accepted: bool,
    pub frames: usize,
    pub before: f64,
    pub after: f64,
    pub replaced_evidence: usize,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    real: usize,
    found: usize,
    sim: usize,
    confirmed: usize,
}

impl Counts {
    fn add(&mut self, real: &[[f64; 2]], sim: &[[f64; 2]], inside: &dyn Fn(&[f64; 2]) -> bool) {
        let (rc, sc) = (cells(real), cells(sim));
        let (rd, sd) = (dilate(&rc), dilate(&sc));
        let centre = |c: &(i64, i64)| [(c.0 as f64 + 0.5) * crate::fidelity::CELL, (c.1 as f64 + 0.5) * crate::fidelity::CELL];
        for c in rc.iter().filter(|c| inside(&centre(c))) {
            self.real += 1;
            self.found += sd.contains(c) as usize;
        }
        for c in sc.iter().filter(|c| inside(&centre(c))) {
            self.sim += 1;
            self.confirmed += rd.contains(c) as usize;
        }
    }

    fn score(&self) -> f64 {
        let r = if self.real == 0 { 1.0 } else { self.found as f64 / self.real as f64 };
        let p = if self.sim == 0 { 1.0 } else { self.confirmed as f64 / self.sim as f64 };
        r + p
    }
}

fn frames(trips: &[Trip], window: Option<(f64, f64)>, refine: Option<&WallNear>) -> Result<Vec<GateFrame>, String> {
    let mut out = Vec::new();
    for trip in trips {
        let lidar = trip.mount("laser_link", "base_footprint")?;
        let camera = trip.mount("camera_link", "base_footprint")?;
        for f in trip_frames(trip, STEP_S, window, refine)? {
            let base = Base::planar(f.pose.x, f.pose.y, f.pose.yaw);
            let lo = add3(base.origin, crate::sensor3d::mat_vec(&base.rot, lidar.xyz));
            let lr = mat_mul(&base.rot, &lidar.rot);
            let s = &f.lidar.scan;
            let real_lidar = beam_points(&f64s(&s.ranges), s.angle_min as f64, s.angle_increment as f64, lo, &lr).into_iter().flatten().collect();
            let real_camera = f.camera.map(|c| {
                beam_points(&f64s(&c.scan.ranges), c.scan.angle_min as f64, c.scan.angle_increment as f64, base.origin, &base.rot).into_iter().flatten().collect()
            });
            let seed = out.len() as u64 + 1;
            out.push(GateFrame { base, lidar: lidar.clone(), camera: camera.clone(), real_lidar, real_camera, seed });
        }
    }
    Ok(out)
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

struct Sim {
    lidar: Vec<[f64; 2]>,
    camera: Vec<[f64; 2]>,
}

fn simulate(w: &SimWorld, f: &GateFrame, sensor: &DepthSensor, rays: &[[f64; 3]]) -> Sim {
    let lo = add3(f.base.origin, crate::sensor3d::mat_vec(&f.base.rot, f.lidar.xyz));
    let lr = mat_mul(&f.base.rot, &f.lidar.rot);
    let angles = linspace(LIDAR_ANGLE_MIN, LIDAR_ANGLE_MAX, LIDAR_COUNT);
    let lidar = beam_points(&w.lidar(&f.base, &f.lidar), angles[0], angles[1] - angles[0], lo, &lr).into_iter().flatten().collect();
    let camera = if f.real_camera.is_some() {
        let mut rng = Rng::new(f.seed);
        let scan = w.camera_scan(&f.base, &f.camera, sensor, rays, Some(&mut rng));
        beam_points(&scan, -PI, sensor.pipeline.angle_increment_rad, f.base.origin, &f.base.rot).into_iter().flatten().collect()
    } else {
        Vec::new()
    };
    Sim { lidar, camera }
}

fn world(venue: &Venue, map: &OccupancyMap, scene: Scene) -> Result<SimWorld, String> {
    let mut v = venue.clone();
    v.scenes.insert(GATE_SCENE.into(), scene);
    Ok(SimWorld::new(GATE_SCENE, World3d::from_venue(&v, Some(GATE_SCENE), Some(WallGrid::from_map(map)))?, Mode::Volume))
}

#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct Totals {
    pub lidar_real: f64,
    pub lidar_found: f64,
    pub lidar_sim: f64,
    pub lidar_confirmed: f64,
    pub camera_real: f64,
    pub camera_found: f64,
    pub camera_sim: f64,
    pub camera_confirmed: f64,
}

impl Totals {
    pub fn scores(&self) -> [f64; 4] {
        [self.lidar_found / self.lidar_real, self.lidar_confirmed / self.lidar_sim, self.camera_found / self.camera_real, self.camera_confirmed / self.camera_sim]
    }

    fn plus(&self, d: &Totals) -> Totals {
        Totals {
            lidar_real: self.lidar_real,
            lidar_found: self.lidar_found + d.lidar_found,
            lidar_sim: self.lidar_sim + d.lidar_sim,
            lidar_confirmed: self.lidar_confirmed + d.lidar_confirmed,
            camera_real: self.camera_real,
            camera_found: self.camera_found + d.camera_found,
            camera_sim: self.camera_sim + d.camera_sim,
            camera_confirmed: self.camera_confirmed + d.camera_confirmed,
        }
    }

    fn slack(&self, bar: &[f64; 4]) -> f64 {
        self.scores().iter().zip(bar).map(|(s, b)| s - b).fold(f64::INFINITY, f64::min)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub id: String,
    pub frames: usize,
    pub delta: Totals,
    pub replaced_evidence: usize,
    pub accepted: bool,
}

fn delta(before: &Counts, after: &Counts, cam_before: &Counts, cam_after: &Counts) -> Totals {
    Totals {
        lidar_real: 0.0,
        lidar_found: after.found as f64 - before.found as f64,
        lidar_sim: after.sim as f64 - before.sim as f64,
        lidar_confirmed: after.confirmed as f64 - before.confirmed as f64,
        camera_real: 0.0,
        camera_found: cam_after.found as f64 - cam_before.found as f64,
        camera_sim: cam_after.sim as f64 - cam_before.sim as f64,
        camera_confirmed: cam_after.confirmed as f64 - cam_before.confirmed as f64,
    }
}

fn overlaps(a: &[Item], b: &[Item]) -> bool {
    a.iter().any(|x| b.iter().any(|y| (x.x - y.x).hypot(x.y - y.y) < 0.4 || covers(x, y.x, y.y, 0.0) || covers(y, x.x, x.y, 0.0)))
}

#[allow(clippy::too_many_arguments)]
pub fn budget_objects(
    venue: &Venue,
    map: &OccupancyMap,
    furniture: &Scene,
    groups: &[Vec<Item>],
    evidence: &[Item],
    trips: &[Trip],
    sensor: &DepthSensor,
    window: Option<(f64, f64)>,
    refine: Option<&WallNear>,
    gc: &GateConfig,
    totals: Totals,
    bar: [f64; 4],
    margin: f64,
) -> Result<(Scene, Vec<Candidate>, [f64; 4]), String> {
    let frames = frames(trips, window, refine)?;
    let rays = sensor.camera.rays();
    let compose = |objects: &[Item], ev: &[Item]| {
        let mut s = furniture.clone();
        s.add.extend(objects.iter().cloned());
        s.add.extend(ev.iter().cloned());
        s
    };
    let base = world(venue, map, compose(&[], evidence))?;
    let mut cache: HashMap<usize, Sim> = HashMap::new();
    let mut cands: Vec<Candidate> = Vec::new();
    for group in groups {
        let cx = group.iter().map(|i| i.x).sum::<f64>() / group.len() as f64;
        let cy = group.iter().map(|i| i.y).sum::<f64>() / group.len() as f64;
        let reach = group.iter().map(|i| i.size.map_or(0.3, |[w, d]| w.hypot(d) / 2.0)).fold(0.0, f64::max) + gc.region_pad_m;
        let inside = |p: &[f64; 2]| (p[0] - cx).hypot(p[1] - cy) <= reach;
        let near: Vec<usize> = (0..frames.len()).filter(|&k| (frames[k].base.origin[0] - cx).hypot(frames[k].base.origin[1] - cy) <= gc.view_m).collect();
        let kept_ev: Vec<Item> = evidence.iter().filter(|e| !group.iter().any(|g| covers(g, e.x, e.y, gc.replace_pad_m))).cloned().collect();
        let trial = world(venue, map, compose(group, &kept_ev))?;
        let (mut b, mut a, mut cb, mut ca) = (Counts::default(), Counts::default(), Counts::default(), Counts::default());
        for &k in &near {
            let f = &frames[k];
            let old = cache.entry(k).or_insert_with(|| simulate(&base, f, sensor, &rays));
            let new = simulate(&trial, f, sensor, &rays);
            b.add(&f.real_lidar, &old.lidar, &inside);
            a.add(&f.real_lidar, &new.lidar, &inside);
            if let Some(real) = &f.real_camera {
                cb.add(real, &old.camera, &inside);
                ca.add(real, &new.camera, &inside);
            }
        }
        cands.push(Candidate { id: group[0].id.clone(), frames: near.len(), delta: delta(&b, &a, &cb, &ca), replaced_evidence: evidence.len() - kept_ev.len(), accepted: false });
    }
    let mut order: Vec<usize> = (0..cands.len()).filter(|&k| cands[k].frames > 0).collect();
    order.sort_by(|&x, &y| totals.plus(&cands[y].delta).slack(&bar).total_cmp(&totals.plus(&cands[x].delta).slack(&bar)));
    let mut projected = totals;
    let mut chosen: Vec<usize> = Vec::new();
    for k in order {
        if chosen.iter().any(|&c| overlaps(&groups[c], &groups[k])) {
            continue;
        }
        let next = projected.plus(&cands[k].delta);
        if next.slack(&bar) >= margin {
            projected = next;
            chosen.push(k);
            cands[k].accepted = true;
        }
    }
    let objects: Vec<Item> = chosen.iter().flat_map(|&k| groups[k].iter().cloned()).collect();
    let kept_ev: Vec<Item> = evidence.iter().filter(|e| !objects.iter().any(|g| covers(g, e.x, e.y, gc.replace_pad_m))).cloned().collect();
    Ok((compose(&objects, &kept_ev), cands, projected.scores()))
}

#[allow(clippy::too_many_arguments)]
pub fn gate_objects(
    venue: &Venue,
    map: &OccupancyMap,
    furniture: &Scene,
    groups: &[Vec<Item>],
    evidence: &[Item],
    trips: &[Trip],
    sensor: &DepthSensor,
    window: Option<(f64, f64)>,
    refine: Option<&WallNear>,
    gc: &GateConfig,
) -> Result<(Scene, Vec<GateDecision>), String> {
    let frames = frames(trips, window, refine)?;
    let rays = sensor.camera.rays();
    let compose = |objects: &[Item], ev: &[Item]| {
        let mut s = furniture.clone();
        s.add.extend(objects.iter().cloned());
        s.add.extend(ev.iter().cloned());
        s
    };
    let mut objects: Vec<Item> = Vec::new();
    let mut ev: Vec<Item> = evidence.to_vec();
    let mut current = world(venue, map, compose(&objects, &ev))?;
    let mut cache: HashMap<usize, Sim> = HashMap::new();
    let mut decisions = Vec::new();
    for group in groups {
        let cx = group.iter().map(|i| i.x).sum::<f64>() / group.len() as f64;
        let cy = group.iter().map(|i| i.y).sum::<f64>() / group.len() as f64;
        let reach = group.iter().map(|i| i.size.map_or(0.3, |[w, d]| w.hypot(d) / 2.0)).fold(0.0, f64::max) + gc.region_pad_m;
        let inside = |p: &[f64; 2]| (p[0] - cx).hypot(p[1] - cy) <= reach;
        let near: Vec<usize> = (0..frames.len()).filter(|&k| (frames[k].base.origin[0] - cx).hypot(frames[k].base.origin[1] - cy) <= gc.view_m).collect();
        let kept_ev: Vec<Item> = ev.iter().filter(|e| !group.iter().any(|g| covers(g, e.x, e.y, gc.replace_pad_m))).cloned().collect();
        let mut trial_objects = objects.clone();
        trial_objects.extend(group.iter().cloned());
        let trial = world(venue, map, compose(&trial_objects, &kept_ev))?;
        let (mut before, mut after) = (Counts::default(), Counts::default());
        let (mut cam_before, mut cam_after) = (Counts::default(), Counts::default());
        let mut trial_sims = Vec::new();
        for &k in &near {
            let f = &frames[k];
            let old = cache.entry(k).or_insert_with(|| simulate(&current, f, sensor, &rays));
            let new = simulate(&trial, f, sensor, &rays);
            before.add(&f.real_lidar, &old.lidar, &inside);
            after.add(&f.real_lidar, &new.lidar, &inside);
            if let Some(real) = &f.real_camera {
                cam_before.add(real, &old.camera, &inside);
                cam_after.add(real, &new.camera, &inside);
            }
            trial_sims.push((k, new));
        }
        let (b, a) = (before.score() + cam_before.score(), after.score() + cam_after.score());
        let accepted = !near.is_empty() && a + 1e-9 >= b - gc.tolerance;
        decisions.push(GateDecision {
            id: group[0].id.clone(),
            accepted,
            frames: near.len(),
            before: (b * 1000.0).round() / 1000.0,
            after: (a * 1000.0).round() / 1000.0,
            replaced_evidence: ev.len() - kept_ev.len(),
        });
        if accepted {
            objects = trial_objects;
            ev = kept_ev;
            current = trial;
            cache.retain(|k, _| !near.contains(k));
            cache.extend(trial_sims);
            let far = |k: &usize| (frames[*k].base.origin[0] - cx).hypot(frames[*k].base.origin[1] - cy) > gc.view_m + 2.0;
            cache.retain(|k, _| far(k) || near.contains(k));
        }
    }
    Ok((compose(&objects, &ev), decisions))
}
