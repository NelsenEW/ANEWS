use crate::mesh::glb_triangles;
use crate::occupancy::OccupancyMap;
use crate::raycast::{Ray, RayCaster};
use crate::sensor3d::{mat_vec, quat_matrix};
use crate::depth_sensor::{DepthSensor, Rng, ScanPipeline};
use crate::trip::{mat_mul, Mount, PlanarPose, Trip, Mat3};
use crate::venue::Venue;
use crate::world3d::{WallGrid, World3d, V3};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;

pub const LIDAR_ANGLE_MIN: f64 = -2.1;
pub const LIDAR_ANGLE_MAX: f64 = 2.1;
pub const LIDAR_COUNT: usize = 1369;
pub const LIDAR_RANGE: f64 = 30.0;
pub const FLAT_DEPTH_FOV_DEG: f64 = 45.0;
pub const FLAT_DEPTH_RANGE: f64 = 2.3;
pub const FLAT_DEPTH_COUNT: usize = 120;
pub const FLAT_DEPTH_MOUNT_X: f64 = 0.3264;
pub const FLAT_HEIGHT: f64 = 0.5;
pub const CELL: f64 = 0.05;
pub const COMPARE_RANGE: f64 = 4.0;
pub const STEP_S: f64 = 1.0;
pub const MAX_POSE_GAP_S: f64 = 0.15;
pub const MAX_CAMERA_GAP_S: f64 = 0.1;
pub const OPTICAL_QUAT: [f64; 4] = [0.5, -0.5, 0.5, -0.5];
pub const REFINE_SHIFT_M: f64 = 0.1;
pub const REFINE_SHIFT_STEP_M: f64 = 0.025;
pub const REFINE_TURN_DEG: f64 = 2.0;
pub const REFINE_TURN_STEP_DEG: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Flat,
    Volume,
}

pub struct SimWorld {
    pub name: String,
    pub world: World3d,
    pub mode: Mode,
    accel: Option<Box<dyn RayCaster>>,
}

pub struct MeshInputs<'a> {
    pub full: Option<&'a [u8]>,
    pub structure: Option<&'a [u8]>,
}

fn grid(map: &OccupancyMap) -> WallGrid {
    WallGrid::from_map(map)
}

pub fn build_worlds(venue: &Venue, map: Option<&OccupancyMap>, meshes: &MeshInputs) -> Result<Vec<SimWorld>, String> {
    let mut worlds = Vec::new();
    if let Some(map) = map {
        worlds.push(SimWorld::new("2d_map", World3d::walls_only(&grid(map), 1e6, false), Mode::Flat));
    }
    worlds.push(SimWorld::new("3d_venue", World3d::from_venue(venue, None, map.map(grid))?, Mode::Volume));
    if venue.scenes.contains_key("no_chairs") {
        worlds.push(SimWorld::new("3d_no_chairs", World3d::from_venue(venue, Some("no_chairs"), map.map(grid))?, Mode::Volume));
    }
    if let Some(full) = meshes.full {
        worlds.push(SimWorld::new("mesh_full", World3d::from_triangles(glb_triangles(full, 0)?, true), Mode::Volume));
    }
    if let Some(structure) = meshes.structure {
        let mut items_only = venue.clone();
        items_only.walls.from_map = false;
        let mut world = World3d::from_venue(&items_only, None, None)?;
        world.add_triangles(glb_triangles(structure, 0)?);
        worlds.push(SimWorld::new("mesh_struct+venue", world, Mode::Volume));
    }
    Ok(worlds)
}

pub fn scene_items_world(venue: &Venue, scene: &str) -> Result<World3d, String> {
    let mut items_only = venue.clone();
    items_only.walls.from_map = false;
    World3d::from_venue(&items_only, Some(scene), None)
}

pub fn scene_world(venue: &Venue, map: Option<&OccupancyMap>, name: &str, scene: &str) -> Result<SimWorld, String> {
    Ok(SimWorld::new(name, World3d::from_venue(venue, Some(scene), map.map(grid))?, Mode::Volume))
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn beam_points(ranges: &[f64], amin: f64, inc: f64, origin: V3, rot: &Mat3) -> Vec<Option<[f64; 2]>> {
    ranges
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            (r.is_finite() && r > 0.05 && r < COMPARE_RANGE).then(|| {
                let a = amin + inc * i as f64;
                let p = add(origin, mat_vec(rot, [r * a.cos(), r * a.sin(), 0.0]));
                [p[0], p[1]]
            })
        })
        .collect()
}

pub const OCCLUSION_MARGIN_M: f64 = 0.1;

fn static_view(
    real: &[Option<[f64; 2]>],
    real_r: &[f64],
    sim: &[Option<[f64; 2]>],
    sim_r: &[f64],
    transient: &dyn Fn([f64; 2]) -> bool,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let real_keep = real.iter().flatten().copied().filter(|p| !transient(*p)).collect();
    let sim_keep = sim
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let p = (*p)?;
            let hidden = match (real.get(i).copied().flatten(), real_r.get(i)) {
                (Some(rp), Some(&rr)) => rr < sim_r[i] - OCCLUSION_MARGIN_M && transient(rp),
                _ => false,
            };
            (!hidden).then_some(p)
        })
        .collect();
    (real_keep, sim_keep)
}

fn p2l_bins(p: &ScanPipeline) -> usize {
    (2.0 * PI / p.angle_increment_rad).round() as usize + 1
}

pub fn voxel_centroids(points: &[V3], leaf: f64) -> Vec<V3> {
    let mut slots: HashMap<[i64; 3], usize> = HashMap::new();
    let mut sums: Vec<(V3, usize)> = Vec::new();
    for pt in points {
        let key = [(pt[0] / leaf).floor() as i64, (pt[1] / leaf).floor() as i64, (pt[2] / leaf).floor() as i64];
        let slot = *slots.entry(key).or_insert_with(|| {
            sums.push(([0.0; 3], 0));
            sums.len() - 1
        });
        let s = &mut sums[slot];
        s.0 = add(s.0, *pt);
        s.1 += 1;
    }
    sums.into_iter().map(|(sum, n)| [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64]).collect()
}

pub fn pointcloud_to_scan(points: &[V3], p: &ScanPipeline) -> Vec<f64> {
    let mut out = vec![f64::INFINITY; p2l_bins(p)];
    for c in voxel_centroids(points, p.voxel_leaf_m) {
        let r = c[0].hypot(c[1]);
        if c[2] < p.min_height_m || c[2] > p.max_height_m || r < p.range_min_m || r > p.range_max_m {
            continue;
        }
        let bin = ((c[1].atan2(c[0]) + PI) / p.angle_increment_rad).floor() as usize;
        if bin < out.len() && r < out[bin] {
            out[bin] = r;
        }
    }
    out
}

pub(crate) fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| if i == n - 1 { b } else { a + step * i as f64 }).collect()
}

pub struct Base {
    pub origin: V3,
    pub rot: Mat3,
}

impl Base {
    pub fn planar(x: f64, y: f64, yaw: f64) -> Self {
        let (s, c) = yaw.sin_cos();
        Base { origin: [x, y, 0.0], rot: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]] }
    }
}

impl SimWorld {
    pub fn new(name: impl Into<String>, world: World3d, mode: Mode) -> Self {
        SimWorld { name: name.into(), world, mode, accel: None }
    }

    pub fn accelerate<F>(&mut self, build: F) -> Result<(), String>
    where
        F: FnOnce(&World3d) -> Result<Box<dyn RayCaster>, String>,
    {
        self.accel = Some(build(&self.world)?);
        Ok(())
    }

    fn cast_batch(&self, rays: &[Ray]) -> Vec<f64> {
        let Some(accel) = &self.accel else { return self.world.cast_batch(rays) };
        let mut out = accel.cast_batch(rays);
        if self.world.crowd.is_some() || self.world.actors.is_some() {
            for (t, r) in out.iter_mut().zip(rays) {
                let n = (r.dir[0] * r.dir[0] + r.dir[1] * r.dir[1] + r.dir[2] * r.dir[2]).sqrt();
                let c = self.world.dynamic_cast(r.origin, [r.dir[0] / n, r.dir[1] / n, r.dir[2] / n], t.min(r.t_max));
                if c < *t {
                    *t = c;
                }
            }
        }
        out
    }

    pub fn lidar(&self, base: &Base, mount: &Mount) -> Vec<f64> {
        let mut origin = add(base.origin, mat_vec(&base.rot, mount.xyz));
        let rot = mat_mul(&base.rot, &mount.rot);
        if self.mode == Mode::Flat {
            origin[2] = FLAT_HEIGHT;
        }
        let rays: Vec<Ray> = linspace(LIDAR_ANGLE_MIN, LIDAR_ANGLE_MAX, LIDAR_COUNT)
            .into_iter()
            .map(|a| Ray { origin, dir: mat_vec(&rot, [a.cos(), a.sin(), 0.0]), t_max: LIDAR_RANGE })
            .collect();
        self.cast_batch(&rays)
    }

    pub fn depth_points(&self, base: &Base, mount: &Mount, sensor: &DepthSensor, rays: &[V3], mut rng: Option<&mut Rng>) -> Vec<V3> {
        let optical = quat_matrix(OPTICAL_QUAT);
        if self.mode == Mode::Flat {
            let fwd_rot = mat_mul(&mount.rot, &optical);
            let yaw0 = fwd_rot[1][2].atan2(fwd_rot[0][2]);
            let half = FLAT_DEPTH_FOV_DEG / 2.0;
            let o_base = [FLAT_DEPTH_MOUNT_X, 0.0, FLAT_HEIGHT];
            let origin = add(base.origin, mat_vec(&base.rot, o_base));
            let dirs: Vec<V3> = linspace(-half, half, FLAT_DEPTH_COUNT)
                .into_iter()
                .map(|deg| {
                    let a = yaw0 + deg.to_radians();
                    [a.cos(), a.sin(), 0.0]
                })
                .collect();
            let cast: Vec<Ray> = dirs.iter().map(|d| Ray { origin, dir: mat_vec(&base.rot, *d), t_max: FLAT_DEPTH_RANGE }).collect();
            return dirs
                .iter()
                .zip(self.cast_batch(&cast))
                .filter(|(_, r)| r.is_finite())
                .map(|(d, r)| [o_base[0] + d[0] * r, o_base[1] + d[1] * r, FLAT_HEIGHT])
                .collect();
        }
        let rot_opt = mat_mul(&mount.rot, &optical);
        let world_rot = mat_mul(&base.rot, &rot_opt);
        let origin = add(base.origin, mat_vec(&base.rot, mount.xyz));
        let reach = sensor.camera.range_max_m * 1.5;
        let cast: Vec<Ray> = rays.iter().map(|ray| Ray { origin, dir: mat_vec(&world_rot, *ray), t_max: reach }).collect();
        rays.iter()
            .zip(self.cast_batch(&cast))
            .filter_map(|(ray, t)| {
                let r = sensor.camera.measure(ray[2], t, rng.as_deref_mut())?;
                Some(add(mount.xyz, mat_vec(&rot_opt, [ray[0] * r, ray[1] * r, ray[2] * r])))
            })
            .collect()
    }

    pub fn camera_scan(&self, base: &Base, mount: &Mount, sensor: &DepthSensor, rays: &[V3], rng: Option<&mut Rng>) -> Vec<f64> {
        pointcloud_to_scan(&self.depth_points(base, mount, sensor, rays, rng), &sensor.pipeline)
    }
}

pub struct WallNear {
    near: Vec<u8>,
    width: usize,
    height: usize,
    resolution: f64,
    origin: [f64; 2],
}

impl WallNear {
    pub fn from_map(map: &OccupancyMap) -> Self {
        let (w, h) = (map.width, map.height);
        let mut near = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                if !map.occupied[row * w + col] {
                    continue;
                }
                for r in row.saturating_sub(1)..=(row + 1).min(h - 1) {
                    for c in col.saturating_sub(1)..=(col + 1).min(w - 1) {
                        near[r * w + c] = near[r * w + c].max(1);
                    }
                }
                near[row * w + col] = 2;
            }
        }
        WallNear { near, width: w, height: h, resolution: map.meta.resolution, origin: [map.meta.origin[0], map.meta.origin[1]] }
    }

    fn hits(&self, pts: &[[f64; 2]]) -> usize {
        pts.iter()
            .map(|p| {
                let c = ((p[0] - self.origin[0]) / self.resolution).floor();
                let r = ((p[1] - self.origin[1]) / self.resolution).floor();
                if c >= 0.0 && r >= 0.0 && (c as usize) < self.width && (r as usize) < self.height {
                    self.near[r as usize * self.width + c as usize] as usize
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn refine(&self, pose: PlanarPose, mount: &Mount, ranges: &[f64], amin: f64, inc: f64) -> PlanarPose {
        let local: Vec<[f64; 2]> = ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_finite() && **r > 0.05 && **r < COMPARE_RANGE)
            .map(|(i, r)| {
                let a = amin + inc * i as f64;
                let p = add(mount.xyz, mat_vec(&mount.rot, [r * a.cos(), r * a.sin(), 0.0]));
                [p[0], p[1]]
            })
            .collect();
        let place = |x: f64, y: f64, yaw: f64| -> Vec<[f64; 2]> {
            let (s, c) = yaw.sin_cos();
            local.iter().map(|p| [x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]]).collect()
        };
        let mut best: (usize, f64, f64, f64) = (self.hits(&place(pose.x, pose.y, pose.yaw)), 0.0, 0.0, 0.0);
        let shifts = (REFINE_SHIFT_M / REFINE_SHIFT_STEP_M).round() as i32;
        let turns = (REFINE_TURN_DEG / REFINE_TURN_STEP_DEG).round() as i32;
        for k in -turns..=turns {
            let dyaw = (k as f64 * REFINE_TURN_STEP_DEG).to_radians();
            for i in -shifts..=shifts {
                for j in -shifts..=shifts {
                    let (dx, dy) = (i as f64 * REFINE_SHIFT_STEP_M, j as f64 * REFINE_SHIFT_STEP_M);
                    let n = self.hits(&place(pose.x + dx, pose.y + dy, pose.yaw + dyaw));
                    let shorter = dx.hypot(dy) < best.1.hypot(best.2);
                    if n > best.0 || (n == best.0 && shorter) {
                        best = (n, dx, dy, dyaw);
                    }
                }
            }
        }
        PlanarPose { t: pose.t, x: pose.x + best.1, y: pose.y + best.2, yaw: pose.yaw + best.3 }
    }
}

fn owner_class(world: &World3d, o: V3, d: V3, r: f64) -> String {
    let n = dot3(d, d).sqrt();
    let d = [d[0] / n, d[1] / n, d[2] / n];
    match world.bvh.cast(o, d, r + 0.01) {
        Some((t, owner)) if (t - r).abs() < 0.01 => {
            let id = world.owners.get(owner as usize).map(String::as_str).unwrap_or("?");
            id.split(|c: char| c.is_ascii_digit()).next().unwrap_or(id).trim_end_matches('_').to_string() + if id.ends_with("_diner") { "_diner" } else { "" }
        }
        _ => if world.actors.is_some() { "people".into() } else if world.crowd.is_some() { "crowd".into() } else { "floor".into() },
    }
}

fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cells(points: &[[f64; 2]]) -> HashSet<(i64, i64)> {
    points.iter().map(|p| ((p[0] / CELL).floor() as i64, (p[1] / CELL).floor() as i64)).collect()
}

pub(crate) fn dilate(s: &HashSet<(i64, i64)>) -> HashSet<(i64, i64)> {
    let mut out = HashSet::with_capacity(s.len() * 9);
    for &(a, b) in s {
        for i in -1..=1 {
            for j in -1..=1 {
                out.insert((a + i, b + j));
            }
        }
    }
    out
}

#[derive(Debug, Clone, Default)]
pub struct Tally {
    pub real: usize,
    pub sim: usize,
    pub real_found: usize,
    pub sim_confirmed: usize,
    pub err: Vec<f64>,
}

impl Tally {
    fn agree(&mut self, real_pts: &[[f64; 2]], sim_pts: &[[f64; 2]]) {
        let (real, sim) = (cells(real_pts), cells(sim_pts));
        let (sim_d, real_d) = (dilate(&sim), dilate(&real));
        self.real += real.len();
        self.sim += sim.len();
        self.real_found += real.iter().filter(|c| sim_d.contains(c)).count();
        self.sim_confirmed += sim.iter().filter(|c| real_d.contains(c)).count();
    }

    fn beam_errors(&mut self, real: &[f64], sim: &[f64]) {
        for (&r, &s) in real.iter().zip(sim) {
            if r.is_finite() && s.is_finite() && r < COMPARE_RANGE && s < COMPARE_RANGE && r > 0.05 {
                self.err.push((r - s).abs());
            }
        }
    }

    fn merge(&mut self, other: &Tally) {
        self.real += other.real;
        self.sim += other.sim;
        self.real_found += other.real_found;
        self.sim_confirmed += other.sim_confirmed;
        self.err.extend_from_slice(&other.err);
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Summary {
    pub real_cells: usize,
    pub sim_cells: usize,
    pub recall: Option<f64>,
    pub precision: Option<f64>,
    pub beams: usize,
    pub median_err_m: Option<f64>,
    pub p90_err_m: Option<f64>,
    pub within_5cm: Option<f64>,
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

impl Tally {
    pub fn summary(&self) -> Summary {
        let mut err = self.err.clone();
        err.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let has = !err.is_empty();
        Summary {
            real_cells: self.real,
            sim_cells: self.sim,
            recall: (self.real > 0).then(|| round3(self.real_found as f64 / self.real as f64)),
            precision: (self.sim > 0).then(|| round3(self.sim_confirmed as f64 / self.sim as f64)),
            beams: err.len(),
            median_err_m: has.then(|| round3(percentile(&err, 0.5))),
            p90_err_m: has.then(|| round3(percentile(&err, 0.9))),
            within_5cm: has.then(|| round3(err.iter().filter(|e| **e <= 0.05).count() as f64 / err.len() as f64)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SensorTallies {
    pub lidar: Tally,
    pub camera: Tally,
    pub lidar_static: Tally,
    pub camera_static: Tally,
}

impl SensorTallies {
    fn merge(&mut self, o: &SensorTallies) {
        self.lidar.merge(&o.lidar);
        self.camera.merge(&o.camera);
        self.lidar_static.merge(&o.lidar_static);
        self.camera_static.merge(&o.camera_static);
    }

    fn summaries(&self, split: bool) -> SensorSummaries {
        SensorSummaries {
            lidar: self.lidar.summary(),
            camera: self.camera.summary(),
            lidar_static: split.then(|| self.lidar_static.summary()),
            camera_static: split.then(|| self.camera_static.summary()),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SensorSummaries {
    pub lidar: Summary,
    pub camera: Summary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_static: Option<Summary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_static: Option<Summary>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TripResult {
    pub name: String,
    pub frames: usize,
    pub worlds: BTreeMap<String, SensorSummaries>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FidelityResult {
    pub worlds: Vec<String>,
    pub trips: usize,
    pub frames: usize,
    pub totals: BTreeMap<String, SensorSummaries>,
    pub per_trip: Vec<TripResult>,
}

pub struct FidelityRun {
    pub worlds: Vec<SimWorld>,
    pub sensor: DepthSensor,
    pub refine: Option<WallNear>,
    pub window: Option<(f64, f64)>,
    pub seed: u64,
    pub persistence: Option<crate::scene_fit::Evidence>,
    pub crowd: Vec<(String, crate::crowd::CrowdTimeline, f64, f64)>,
    pub people: Vec<(String, crate::actors::ActorTimeline, f64)>,
    pub attribution: Option<BTreeMap<String, BTreeMap<String, [usize; 2]>>>,
    rays: Vec<V3>,
    totals: Vec<SensorTallies>,
    per_trip: Vec<TripResult>,
}

pub(crate) fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

pub struct Frame<'a> {
    pub pose: PlanarPose,
    pub lidar: &'a crate::trip::TimedScan,
    pub camera: Option<&'a crate::trip::TimedScan>,
}

pub fn trip_frames<'a>(trip: &'a Trip, step_s: f64, window: Option<(f64, f64)>, refine: Option<&WallNear>) -> Result<Vec<Frame<'a>>, String> {
    if trip.poses.len() < 2 || trip.scans.is_empty() {
        return Ok(Vec::new());
    }
    let lidar_mount = trip.mount("laser_link", "base_footprint")?;
    let (first, last_pose) = (trip.poses[0].t, trip.poses[trip.poses.len() - 1].t);
    let mut out = Vec::new();
    let mut last = -1e9;
    for s in &trip.scans {
        let t = s.t;
        if t - last < step_s || t < first || t > last_pose {
            continue;
        }
        if let Some((a, b)) = window
            && (t < a || t > b)
        {
            continue;
        }
        let Some((pose, gap)) = trip.pose_at(t) else { continue };
        if gap > MAX_POSE_GAP_S {
            continue;
        }
        last = t;
        let pose = match refine {
            Some(r) => r.refine(pose, &lidar_mount, &f64s(&s.scan.ranges), s.scan.angle_min as f64, s.scan.angle_increment as f64),
            None => pose,
        };
        let camera = trip
            .camera_scans
            .iter()
            .min_by(|a, b| (a.t - t).abs().partial_cmp(&(b.t - t).abs()).unwrap())
            .filter(|c| (c.t - t).abs() < MAX_CAMERA_GAP_S);
        out.push(Frame { pose, lidar: s, camera });
    }
    Ok(out)
}

impl FidelityRun {
    pub fn new(worlds: Vec<SimWorld>, sensor: DepthSensor) -> Self {
        let totals = worlds.iter().map(|_| SensorTallies::default()).collect();
        let rays = sensor.camera.rays();
        FidelityRun { worlds, sensor, refine: None, window: None, seed: 1, persistence: None, crowd: Vec::new(), people: Vec::new(), attribution: None, rays, totals, per_trip: Vec::new() }
    }

    pub fn add_world(&mut self, world: SimWorld) {
        self.worlds.retain(|w| w.name != world.name);
        self.worlds.push(world);
        self.totals = self.worlds.iter().map(|_| SensorTallies::default()).collect();
        self.per_trip.clear();
    }

    pub fn add_trip(&mut self, name: &str, trip: &Trip) -> Result<Option<TripResult>, String> {
        let frames = trip_frames(trip, STEP_S, self.window, self.refine.as_ref())?;
        if frames.is_empty() {
            return Ok(None);
        }
        let lidar_mount = trip.mount("laser_link", "base_footprint")?;
        let cam_mount = trip.mount("camera_link", "base_footprint")?;
        let mut tallies: Vec<SensorTallies> = self.worlds.iter().map(|_| SensorTallies::default()).collect();
        let lidar_angles = linspace(LIDAR_ANGLE_MIN, LIDAR_ANGLE_MAX, LIDAR_COUNT);
        let (sim_amin, sim_inc) = (lidar_angles[0], lidar_angles[1] - lidar_angles[0]);
        let mut rng = Rng::new(self.seed);
        let split = self.persistence.is_some();
        let lidar_transient = |p: [f64; 2]| self.persistence.as_ref().is_some_and(|e| e.transient_lidar(p));
        let camera_transient = |p: [f64; 2]| self.persistence.as_ref().is_some_and(|e| e.transient_camera(p));
        for f in &frames {
            let s = f.lidar;
            let base = Base::planar(f.pose.x, f.pose.y, f.pose.yaw);
            let lo = add(base.origin, mat_vec(&base.rot, lidar_mount.xyz));
            let lr = mat_mul(&base.rot, &lidar_mount.rot);
            let real_ranges = f64s(&s.scan.ranges);
            let real_b = beam_points(&real_ranges, s.scan.angle_min as f64, s.scan.angle_increment as f64, lo, &lr);
            let real_l: Vec<[f64; 2]> = real_b.iter().flatten().copied().collect();
            let cam_real = f.camera.map(|c| {
                let r = f64s(&c.scan.ranges);
                let b = beam_points(&r, c.scan.angle_min as f64, c.scan.angle_increment as f64, base.origin, &base.rot);
                (r, b)
            });
            for (w, tally) in self.worlds.iter_mut().zip(tallies.iter_mut()) {
                if let Some((_, timeline, lag, hold)) = self.crowd.iter().find(|(n, _, _, _)| n == &w.name) {
                    let grid = w.world.crowd.get_or_insert_with(|| timeline.grid());
                    timeline.apply(s.t - lag, *hold, grid);
                }
                if let Some((_, timeline, lag)) = self.people.iter().find(|(n, _, _)| n == &w.name) {
                    timeline.apply(s.t - lag, w.world.actors.get_or_insert_with(Default::default));
                }
                let sim_r = w.lidar(&base, &lidar_mount);
                let sim_b = beam_points(&sim_r, sim_amin, sim_inc, lo, &lr);
                let sim_l: Vec<[f64; 2]> = sim_b.iter().flatten().copied().collect();
                tally.lidar.agree(&real_l, &sim_l);
                if let Some(attr) = self.attribution.as_mut() {
                    let real_d = dilate(&cells(&real_l));
                    let per = attr.entry(w.name.clone()).or_default();
                    for (k, p) in sim_b.iter().enumerate() {
                        let Some(p) = p else { continue };
                        let a = sim_amin + sim_inc * k as f64;
                        let dir = mat_vec(&lr, [a.cos(), a.sin(), 0.0]);
                        let owner = owner_class(&w.world, lo, dir, sim_r[k]);
                        let e = per.entry(owner).or_insert([0, 0]);
                        e[0] += 1;
                        if real_d.contains(&((p[0] / CELL).floor() as i64, (p[1] / CELL).floor() as i64)) {
                            e[1] += 1;
                        }
                    }
                }
                if sim_r.len() == real_ranges.len() {
                    tally.lidar.beam_errors(&real_ranges, &sim_r);
                }
                if split {
                    let (r, s2) = static_view(&real_b, &real_ranges, &sim_b, &sim_r, &lidar_transient);
                    tally.lidar_static.agree(&r, &s2);
                }
                if let Some((c_ranges, c_real_b)) = &cam_real {
                    let sim_c = w.camera_scan(&base, &cam_mount, &self.sensor, &self.rays, Some(&mut rng));
                    let sim_cb = beam_points(&sim_c, -PI, self.sensor.pipeline.angle_increment_rad, base.origin, &base.rot);
                    let real_c: Vec<[f64; 2]> = c_real_b.iter().flatten().copied().collect();
                    let sim_cp: Vec<[f64; 2]> = sim_cb.iter().flatten().copied().collect();
                    tally.camera.agree(&real_c, &sim_cp);
                    let n = c_ranges.len().min(sim_c.len());
                    tally.camera.beam_errors(&c_ranges[..n], &sim_c[..n]);
                    if split {
                        let (r, s2) = static_view(c_real_b, c_ranges, &sim_cb, &sim_c, &camera_transient);
                        tally.camera_static.agree(&r, &s2);
                    }
                }
            }
        }
        let mut worlds = BTreeMap::new();
        for ((w, tally), total) in self.worlds.iter().zip(&tallies).zip(self.totals.iter_mut()) {
            total.merge(tally);
            worlds.insert(w.name.clone(), tally.summaries(split));
        }
        let result = TripResult { name: name.to_string(), frames: frames.len(), worlds };
        self.per_trip.push(result.clone());
        Ok(Some(result))
    }

    pub fn result(&self) -> FidelityResult {
        FidelityResult {
            worlds: self.worlds.iter().map(|w| w.name.clone()).collect(),
            trips: self.per_trip.len(),
            frames: self.per_trip.iter().map(|t| t.frames).sum(),
            totals: self
                .worlds
                .iter()
                .zip(&self.totals)
                .map(|(w, t)| (w.name.clone(), t.summaries(self.persistence.is_some())))
                .collect(),
            per_trip: self.per_trip.clone(),
        }
    }
}

#[cfg(test)]
mod refine_tests {
    use super::*;
    use crate::occupancy::{parse_map_meta, OccupancyMap};

    fn corner_room() -> OccupancyMap {
        let meta = parse_map_meta("image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n").unwrap();
        let (w, h) = (80, 80);
        let mut occupied = vec![false; w * h];
        for i in 0..w {
            occupied[10 * w + i] = true;
            occupied[i * w + 10] = true;
        }
        OccupancyMap { meta, width: w, height: h, occupied, free: vec![true; w * h] }
    }

    #[test]
    fn a_scan_taken_off_pose_is_pulled_back_onto_the_walls() {
        let map = corner_room();
        let near = WallNear::from_map(&map);
        let mount = Mount { xyz: [0.0, 0.0, 0.2], rot: crate::trip::IDENTITY };
        let truth = PlanarPose { t: 0.0, x: 1.5, y: 1.5, yaw: 0.0 };
        let (amin, inc, n) = (-PI, 2.0 * PI / 720.0, 720);
        let wall = 0.5 + 0.025;
        let ranges: Vec<f64> = (0..n)
            .map(|i| {
                let a = amin + inc * i as f64;
                let (c, s) = (a.cos(), a.sin());
                let tx = if c < -1e-9 { (wall - truth.x) / c } else { f64::INFINITY };
                let ty = if s < -1e-9 { (wall - truth.y) / s } else { f64::INFINITY };
                tx.min(ty)
            })
            .collect();
        let start = PlanarPose { x: truth.x + 0.075, y: truth.y - 0.05, ..truth };
        let fixed = near.refine(start, &mount, &ranges, amin, inc);
        assert!((fixed.x - truth.x).abs() <= 0.03 && (fixed.y - truth.y).abs() <= 0.03, "refined to ({}, {})", fixed.x, fixed.y);
    }
}
