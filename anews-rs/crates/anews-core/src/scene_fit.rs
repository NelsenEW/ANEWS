use crate::crowd::{CrowdTimeline, GridFrame, TimelineBuilder, HIGH, LOW};
use crate::fidelity::{trip_frames, Frame, WallNear, COMPARE_RANGE};
use crate::occupancy::OccupancyMap;
use crate::sensor3d::mat_vec;
use crate::trip::{Mount, Trip};
use crate::venue::{expand_item, Item, ItemChange, Scene, Solid, Venue};
use crate::world3d::WallGrid;
use serde::Serialize;
use std::collections::BTreeMap;

pub const LOW_TYPE: &str = "observed_low";
pub const HIGH_TYPE: &str = "observed_high";
pub const OBSERVED_KIND: &str = "observed";

#[derive(Debug, Clone)]
pub struct FitConfig {
    pub step_s: f64,
    pub window: Option<(f64, f64)>,
    pub min_hits: u32,
    pub remove_ratio: f64,
    pub min_item_observations: u32,
    pub wall_clear_passes: u32,
    pub lidar_height: f64,
    pub camera_band: (f64, f64),
    pub camera_pass_from: f64,
    pub camera_max_range: f64,
    pub footprint_half: [f64; 2],
    pub bucket_s: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        FitConfig {
            step_s: 0.2,
            window: None,
            min_hits: 3,
            remove_ratio: 0.15,
            min_item_observations: 20,
            wall_clear_passes: 4,
            lidar_height: 0.23,
            camera_band: (0.15, 1.2),
            camera_pass_from: 0.5,
            camera_max_range: 3.5,
            footprint_half: [0.25, 0.16],
            bucket_s: 3.0,
        }
    }
}

pub const PERSISTENT_RATIO: f64 = 0.5;

#[derive(Clone, Default)]
pub struct Buckets {
    pub hit: Vec<u32>,
    pub seen: Vec<u32>,
    last_hit: Vec<u64>,
    last_seen: Vec<u64>,
}

impl Buckets {
    fn new(n: usize) -> Self {
        Buckets { hit: vec![0; n], seen: vec![0; n], last_hit: vec![u64::MAX; n], last_seen: vec![u64::MAX; n] }
    }

    fn mark(&mut self, i: usize, bucket: u64, hit: bool) {
        if self.last_seen[i] != bucket {
            self.last_seen[i] = bucket;
            self.seen[i] += 1;
        }
        if hit && self.last_hit[i] != bucket {
            self.last_hit[i] = bucket;
            self.hit[i] += 1;
        }
    }

    pub fn persistent(&self, i: usize) -> bool {
        self.hit[i] > 0 && self.hit[i] as f64 >= PERSISTENT_RATIO * self.seen[i] as f64
    }
}

#[derive(Default)]
pub struct FrameCells {
    pub lidar_hits: Vec<usize>,
    pub lidar_passes: Vec<usize>,
    pub camera_hits: Vec<usize>,
    pub camera_passes: Vec<usize>,
    pub has_camera: bool,
}

pub struct Evidence {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin: [f64; 2],
    pub lidar_hit: Vec<u32>,
    pub lidar_pass: Vec<u32>,
    pub camera_hit: Vec<u32>,
    pub camera_pass: Vec<u32>,
    pub driven: Vec<bool>,
    pub lidar_buckets: Buckets,
    pub camera_buckets: Buckets,
    pub frames: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ItemEvidence {
    pub id: String,
    pub kind: String,
    pub type_name: String,
    pub lidar_observations: u32,
    pub lidar_hit_ratio: f64,
    pub camera_observations: u32,
    pub camera_hit_ratio: f64,
    pub removed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitReport {
    pub frames: usize,
    pub moved: usize,
    pub items: Vec<ItemEvidence>,
    pub removed: Vec<String>,
    pub walls_cleared: usize,
    pub low_cells: usize,
    pub high_cells: usize,
    pub added_items: usize,
}

impl Evidence {
    pub fn new(map: &OccupancyMap) -> Self {
        let n = map.width * map.height;
        Evidence {
            width: map.width,
            height: map.height,
            resolution: map.meta.resolution,
            origin: [map.meta.origin[0], map.meta.origin[1]],
            lidar_hit: vec![0; n],
            lidar_pass: vec![0; n],
            camera_hit: vec![0; n],
            camera_pass: vec![0; n],
            driven: vec![false; n],
            lidar_buckets: Buckets::new(n),
            camera_buckets: Buckets::new(n),
            frames: 0,
        }
    }

    pub fn cell(&self, x: f64, y: f64) -> Option<usize> {
        let c = ((x - self.origin[0]) / self.resolution).floor();
        let r = ((y - self.origin[1]) / self.resolution).floor();
        (c >= 0.0 && r >= 0.0 && (c as usize) < self.width && (r as usize) < self.height).then(|| r as usize * self.width + c as usize)
    }

    pub fn center(&self, i: usize) -> [f64; 2] {
        [self.origin[0] + ((i % self.width) as f64 + 0.5) * self.resolution, self.origin[1] + ((i / self.width) as f64 + 0.5) * self.resolution]
    }

    pub fn neighbours(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let (c, r) = ((i % self.width) as i64, (i / self.width) as i64);
        (-1..=1).flat_map(move |dr| (-1..=1).map(move |dc| (c + dc, r + dr))).filter_map(move |(c, r)| {
            (c >= 0 && r >= 0 && (c as usize) < self.width && (r as usize) < self.height).then(|| r as usize * self.width + c as usize)
        })
    }

    fn trace(&self, from: [f64; 2], dir: [f64; 2], start: f64, end: f64, out: &mut Vec<usize>) {
        let step = self.resolution / 2.0;
        let mut d = start;
        while d < end {
            if let Some(i) = self.cell(from[0] + dir[0] * d, from[1] + dir[1] * d) {
                out.push(i);
            }
            d += step;
        }
    }

    pub fn add_trip(&mut self, trip: &Trip, cfg: &FitConfig, refine: Option<&WallNear>) -> Result<usize, String> {
        let frames = trip_frames(trip, cfg.step_s, cfg.window, refine)?;
        if frames.is_empty() {
            return Ok(0);
        }
        let mount = trip.mount("laser_link", "base_footprint")?;
        for f in &frames {
            self.frames += 1;
            let bucket = (f.lidar.t / cfg.bucket_s).floor().max(0.0) as u64;
            let body = Solid::Box { center: [f.pose.x, f.pose.y, 0.5], half: [cfg.footprint_half[0], cfg.footprint_half[1], 0.5], yaw: f.pose.yaw };
            let mut under = Vec::new();
            self.raster(&body, 0.0, 1.0, &mut under);
            for i in under {
                self.driven[i] = true;
            }
            let obs = self.observe(f, &mount, cfg);
            for &i in &obs.lidar_hits {
                self.lidar_hit[i] += 1;
                self.lidar_buckets.mark(i, bucket, true);
            }
            for &i in &obs.lidar_passes {
                self.lidar_pass[i] += 1;
                self.lidar_buckets.mark(i, bucket, false);
            }
            for &i in &obs.camera_hits {
                self.camera_hit[i] += 1;
                self.camera_buckets.mark(i, bucket, true);
            }
            for &i in &obs.camera_passes {
                self.camera_pass[i] += 1;
                self.camera_buckets.mark(i, bucket, false);
            }
        }
        Ok(frames.len())
    }

    pub fn observe(&self, f: &Frame, mount: &Mount, cfg: &FitConfig) -> FrameCells {
        let mut out = FrameCells::default();
        let (s, c) = f.pose.yaw.sin_cos();
        let m = mount.xyz;
        let lo = [f.pose.x + c * m[0] - s * m[1], f.pose.y + s * m[0] + c * m[1]];
        let scan = &f.lidar.scan;
        let mut passes = Vec::new();
        for (k, r) in scan.ranges.iter().enumerate() {
            let r = *r as f64;
            let a = scan.angle_min as f64 + scan.angle_increment as f64 * k as f64;
            let local = mat_vec(&mount.rot, [a.cos(), a.sin(), 0.0]);
            let dir = [c * local[0] - s * local[1], s * local[0] + c * local[1]];
            let n = dir[0].hypot(dir[1]);
            let dir = [dir[0] / n, dir[1] / n];
            if r.is_finite() && r > 0.05 && r < COMPARE_RANGE {
                if let Some(i) = self.cell(lo[0] + dir[0] * r, lo[1] + dir[1] * r) {
                    out.lidar_hits.push(i);
                }
                self.trace(lo, dir, 0.0, r - self.resolution * 0.75, &mut passes);
            } else if !r.is_finite() || r >= COMPARE_RANGE {
                self.trace(lo, dir, 0.0, COMPARE_RANGE, &mut passes);
            }
        }
        out.lidar_hits.sort_unstable();
        out.lidar_hits.dedup();
        passes.sort_unstable();
        passes.dedup();
        out.lidar_passes = passes.into_iter().filter(|i| out.lidar_hits.binary_search(i).is_err()).collect();
        let Some(cam) = f.camera else { return out };
        out.has_camera = true;
        let scan = &cam.scan;
        let mut passes = Vec::new();
        for (k, r) in scan.ranges.iter().enumerate() {
            let r = *r as f64;
            if !(r.is_finite() && r > 0.05 && r < cfg.camera_max_range) {
                continue;
            }
            let a = f.pose.yaw + scan.angle_min as f64 + scan.angle_increment as f64 * k as f64;
            let dir = [a.cos(), a.sin()];
            if let Some(i) = self.cell(f.pose.x + dir[0] * r, f.pose.y + dir[1] * r) {
                out.camera_hits.push(i);
            }
            self.trace([f.pose.x, f.pose.y], dir, cfg.camera_pass_from, r - self.resolution * 2.0, &mut passes);
        }
        out.camera_hits.sort_unstable();
        out.camera_hits.dedup();
        passes.sort_unstable();
        passes.dedup();
        out.camera_passes = passes.into_iter().filter(|i| out.camera_hits.binary_search(i).is_err()).collect();
        out
    }

    pub fn grid_frame(&self) -> GridFrame {
        GridFrame { width: self.width, height: self.height, resolution: self.resolution, origin: self.origin }
    }

    pub fn transient_lidar(&self, p: [f64; 2]) -> bool {
        self.cell(p[0], p[1]).is_some_and(|i| self.neighbours(i).all(|j| !self.lidar_buckets.persistent(j)))
    }

    pub fn transient_camera(&self, p: [f64; 2]) -> bool {
        self.cell(p[0], p[1]).is_some_and(|i| self.neighbours(i).all(|j| !self.camera_buckets.persistent(j)))
    }

    pub fn nonzero_cells(&self) -> Vec<[f64; 6]> {
        (0..self.lidar_hit.len())
            .filter(|&i| self.lidar_hit[i] + self.lidar_pass[i] + self.camera_hit[i] + self.camera_pass[i] > 0)
            .map(|i| {
                let c = self.center(i);
                [c[0], c[1], self.lidar_hit[i] as f64, self.lidar_pass[i] as f64, self.camera_hit[i] as f64, self.camera_pass[i] as f64]
            })
            .collect()
    }

    fn raster(&self, solid: &Solid, z_lo: f64, z_hi: f64, out: &mut Vec<usize>) {
        let pad = self.resolution / 2.0;
        let (cx, cy, reach, inside): (f64, f64, f64, Box<dyn Fn(f64, f64) -> bool>) = match *solid {
            Solid::Box { center, half, yaw } => {
                if center[2] + half[2] < z_lo || center[2] - half[2] > z_hi {
                    return;
                }
                let (s, c) = yaw.sin_cos();
                let (hx, hy) = (half[0] + pad, half[1] + pad);
                (center[0], center[1], hx.hypot(hy), Box::new(move |dx, dy| (c * dx + s * dy).abs() <= hx && (-s * dx + c * dy).abs() <= hy))
            }
            Solid::Cylinder { center, radius, z0, z1 } => {
                if z1 < z_lo || z0 > z_hi {
                    return;
                }
                let r = radius + pad;
                (center[0], center[1], r, Box::new(move |dx, dy| dx.hypot(dy) <= r))
            }
        };
        let mut y = cy - reach;
        while y <= cy + reach + self.resolution {
            let mut x = cx - reach;
            while x <= cx + reach + self.resolution {
                if let Some(i) = self.cell(x, y) {
                    let p = self.center(i);
                    if inside(p[0] - cx, p[1] - cy) {
                        out.push(i);
                    }
                }
                x += self.resolution;
            }
            y += self.resolution;
        }
    }

    pub fn item_cells(&self, venue: &Venue, item: &Item, z_lo: f64, z_hi: f64) -> Vec<usize> {
        let mut out = Vec::new();
        for solid in expand_item(item, &venue.types[&item.type_name]) {
            self.raster(&solid, z_lo, z_hi, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn dilate(&self, mask: &[bool]) -> Vec<bool> {
        let mut out = vec![false; mask.len()];
        for (i, &m) in mask.iter().enumerate() {
            if m {
                for j in self.neighbours(i) {
                    out[j] = true;
                }
            }
        }
        out
    }

    fn ratio(&self, cells: &[usize], hit: &[u32], pass: &[u32]) -> (u32, f64) {
        let mut near = vec![];
        for &i in cells {
            near.extend(self.neighbours(i));
        }
        near.sort_unstable();
        near.dedup();
        let h: u32 = near.iter().map(|&i| hit[i]).sum();
        let p: u32 = cells.iter().map(|&i| pass[i]).sum();
        let obs = h + p;
        (obs, if obs == 0 { 0.0 } else { h as f64 / obs as f64 })
    }
}

pub const PLACE_SHIFT_M: f64 = 0.2;
pub const PLACE_SHIFT_STEP_M: f64 = 0.05;
pub const PLACE_TURN_DEG: f64 = 20.0;
pub const PLACE_TURN_STEP_DEG: f64 = 10.0;
pub const PLACE_SCALES: [f64; 3] = [0.8, 0.9, 1.0];

fn placement_score(ev: &Evidence, venue: &Venue, item: &Item, cfg: &FitConfig) -> f64 {
    let low = ev.item_cells(venue, item, cfg.lidar_height, cfg.lidar_height);
    let high = ev.item_cells(venue, item, cfg.camera_band.0, cfg.camera_band.1);
    let mut score = 0.0;
    for &i in &low {
        score += ev.lidar_hit[i] as f64 - ev.lidar_pass[i] as f64;
        if ev.driven[i] {
            score -= 1e4;
        }
    }
    for &i in &high {
        score += 0.5 * (ev.camera_hit[i] as f64 - ev.camera_pass[i] as f64);
        if ev.driven[i] {
            score -= 1e4;
        }
    }
    score
}

fn place(ev: &Evidence, venue: &Venue, item: &Item, cfg: &FitConfig) -> Item {
    let mut best = (placement_score(ev, venue, item, cfg), item.clone());
    let shifts = (PLACE_SHIFT_M / PLACE_SHIFT_STEP_M).round() as i32;
    let turns = (PLACE_TURN_DEG / PLACE_TURN_STEP_DEG).round() as i32;
    let scales: &[f64] = if item.size.is_some() { &PLACE_SCALES } else { &[1.0] };
    for &scale in scales {
        for k in -turns..=turns {
            for i in -shifts..=shifts {
                for j in -shifts..=shifts {
                    let mut cand = item.clone();
                    cand.x += i as f64 * PLACE_SHIFT_STEP_M;
                    cand.y += j as f64 * PLACE_SHIFT_STEP_M;
                    cand.yaw_deg += k as f64 * PLACE_TURN_STEP_DEG;
                    cand.size = item.size.map(|[w, d]| [w * scale, d * scale]);
                    let score = placement_score(ev, venue, &cand, cfg);
                    if score > best.0 + 1e-9 {
                        best = (score, cand);
                    }
                }
            }
        }
    }
    best.1
}

fn posts(ev: &Evidence, mask: &[bool]) -> Vec<Item> {
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    (0..mask.len())
        .filter(|&i| mask[i])
        .map(|i| {
            let c = ev.center(i);
            Item {
                id: format!("fit_low_{}_{}", i / ev.width, i % ev.width),
                kind: OBSERVED_KIND.into(),
                type_name: LOW_TYPE.into(),
                x: round(c[0]),
                y: round(c[1]),
                yaw_deg: 0.0,
                size: None,
                number: None,
                mobility: Some("dynamic".into()),
                extra: BTreeMap::new(),
            }
        })
        .collect()
}

fn runs(ev: &Evidence, mask: &[bool], type_name: &str, prefix: &str) -> Vec<Item> {
    let mut out = Vec::new();
    for row in 0..ev.height {
        let mut col = 0;
        while col < ev.width {
            if !mask[row * ev.width + col] {
                col += 1;
                continue;
            }
            let start = col;
            while col < ev.width && mask[row * ev.width + col] {
                col += 1;
            }
            let len = (col - start) as f64 * ev.resolution;
            let x = ev.origin[0] + start as f64 * ev.resolution + len / 2.0;
            let y = ev.origin[1] + (row as f64 + 0.5) * ev.resolution;
            out.push(Item {
                id: format!("{prefix}_{row}_{start}"),
                kind: OBSERVED_KIND.into(),
                type_name: type_name.into(),
                x: (x * 1000.0).round() / 1000.0,
                y: (y * 1000.0).round() / 1000.0,
                yaw_deg: 0.0,
                size: Some([(len * 1000.0).round() / 1000.0, ev.resolution]),
                number: None,
                mobility: Some("dynamic".into()),
                extra: BTreeMap::new(),
            });
        }
    }
    out
}

pub fn fit_scene(venue: &Venue, map: &OccupancyMap, base_scene: Option<&str>, ev: &Evidence, cfg: &FitConfig) -> Result<(Scene, FitReport), String> {
    for t in [LOW_TYPE, HIGH_TYPE] {
        if !venue.types.contains_key(t) {
            return Err(format!("venue has no {t} type; fitted obstacles need it"));
        }
    }
    let base_items = venue.scene_items(base_scene)?;
    let mut remove: Vec<String> = base_scene.and_then(|n| venue.scenes.get(n)).map(|s| s.remove.clone()).unwrap_or_default();
    let mut kept = Vec::new();
    let mut removed_now = Vec::new();
    let mut items = Vec::new();
    let mut placed: BTreeMap<String, Item> = BTreeMap::new();
    for original in &base_items {
        let movable_kind = matches!(original.mobility.as_deref(), Some("movable") | Some("dynamic")) && original.kind != OBSERVED_KIND;
        let item = &if movable_kind { place(ev, venue, original, cfg) } else { original.clone() };
        if item.x != original.x || item.y != original.y || item.yaw_deg != original.yaw_deg || item.size != original.size {
            placed.insert(item.id.clone(), item.clone());
        }
        let movable = matches!(item.mobility.as_deref(), Some("movable") | Some("dynamic"));
        let mobile = movable && item.kind != "table";
        let low = ev.item_cells(venue, item, cfg.lidar_height, cfg.lidar_height);
        let high = ev.item_cells(venue, item, cfg.camera_band.0, cfg.camera_band.1);
        let (lo_obs, lo_ratio) = ev.ratio(&low, &ev.lidar_hit, &ev.lidar_pass);
        let (hi_obs, hi_ratio) = ev.ratio(&high, &ev.camera_hit, &ev.camera_pass);
        let driven_through = movable && low.iter().chain(high.iter()).any(|&i| ev.driven[i]);
        let unseen = mobile && lo_obs + hi_obs >= cfg.min_item_observations && lo_ratio < cfg.remove_ratio && hi_ratio < cfg.remove_ratio;
        let lidar_sees_through = lo_obs >= cfg.min_item_observations && lo_ratio < cfg.remove_ratio;
        let unsupported = movable_kind && placement_score(ev, venue, item, cfg) <= 0.0;
        let remove = driven_through || unseen || lidar_sees_through || unsupported;
        if item.kind != OBSERVED_KIND {
            items.push(ItemEvidence {
                id: item.id.clone(),
                kind: item.kind.clone(),
                type_name: item.type_name.clone(),
                lidar_observations: lo_obs,
                lidar_hit_ratio: (lo_ratio * 1000.0).round() / 1000.0,
                camera_observations: hi_obs,
                camera_hit_ratio: (hi_ratio * 1000.0).round() / 1000.0,
                removed: remove,
            });
        }
        if remove {
            removed_now.push(item.id.clone());
            continue;
        }
        kept.push(item.clone());
    }
    let base_ids: Vec<&str> = venue.items.iter().map(|i| i.id.as_str()).collect();
    for id in &removed_now {
        if base_ids.contains(&id.as_str()) && !remove.contains(id) {
            remove.push(id.clone());
        }
    }
    let mut grid = WallGrid::from_map(map).for_scene(venue, base_scene);
    let mut clear_walls: Vec<[f64; 2]> = venue.scene_wall_clears(base_scene).to_vec();
    for i in 0..grid.occupied.len() {
        if !grid.occupied[i] {
            continue;
        }
        let observed = ev.lidar_buckets.seen[i] >= 2 && ev.lidar_pass[i] >= cfg.wall_clear_passes;
        let seen_through = observed && !ev.lidar_buckets.persistent(i);
        if seen_through || ev.driven[i] {
            grid.occupied[i] = false;
            let c = ev.center(i);
            clear_walls.push([(c[0] * 1000.0).round() / 1000.0, (c[1] * 1000.0).round() / 1000.0]);
        }
    }
    let (low, high) = unexplained(venue, &grid, &kept, ev, cfg);
    let mut add: Vec<Item> = base_scene.and_then(|n| venue.scenes.get(n)).map(|s| s.add.clone()).unwrap_or_default();
    add.retain(|i| !removed_now.contains(&i.id));
    let new_low = posts(ev, &low);
    let new_high = runs(ev, &high, HIGH_TYPE, "fit_high");
    let report = FitReport {
        frames: ev.frames,
        moved: 0,
        items,
        removed: removed_now.clone(),
        walls_cleared: clear_walls.len(),
        low_cells: low.iter().filter(|b| **b).count(),
        high_cells: high.iter().filter(|b| **b).count(),
        added_items: new_low.len() + new_high.len(),
    };
    add.extend(new_low);
    add.extend(new_high);
    let mut moves = base_scene.and_then(|n| venue.scenes.get(n)).map(|s| s.moves.clone()).unwrap_or_default();
    for (id, item) in &placed {
        if removed_now.contains(id) {
            continue;
        }
        if let Some(a) = add.iter_mut().find(|a| &a.id == id) {
            *a = item.clone();
        } else {
            let round = |v: f64| (v * 1000.0).round() / 1000.0;
            moves.insert(id.clone(), ItemChange { x: Some(round(item.x)), y: Some(round(item.y)), yaw_deg: Some(round(item.yaw_deg)), size: item.size.map(|[w, d]| [round(w), round(d)]) });
        }
    }
    let report = FitReport { moved: moves.len(), ..report };
    Ok((Scene { remove, moves, add, clear_walls }, report))
}

pub fn unexplained(venue: &Venue, grid: &WallGrid, kept: &[Item], ev: &Evidence, cfg: &FitConfig) -> (Vec<bool>, Vec<bool>) {
    let n = ev.lidar_hit.len();
    let mut low_layer = grid.occupied.clone();
    let mut high_layer = grid.occupied.clone();
    for item in kept {
        for i in ev.item_cells(venue, item, cfg.lidar_height, cfg.lidar_height) {
            low_layer[i] = true;
        }
        for i in ev.item_cells(venue, item, cfg.camera_band.0, cfg.camera_band.1) {
            high_layer[i] = true;
        }
    }
    let low_explained = ev.dilate(&low_layer);
    let mut low = vec![false; n];
    for i in 0..n {
        if ev.lidar_hit[i] >= cfg.min_hits && ev.lidar_buckets.persistent(i) && !low_explained[i] && !ev.driven[i] {
            low[i] = true;
        }
    }
    for i in 0..n {
        high_layer[i] |= low[i];
    }
    let high_explained = ev.dilate(&high_layer);
    let mut high = vec![false; n];
    for i in 0..n {
        if ev.camera_hit[i] >= cfg.min_hits && ev.camera_buckets.persistent(i) && !high_explained[i] && !ev.driven[i] {
            high[i] = true;
        }
    }
    (low, high)
}

pub fn observed_items(ev: &Evidence, low: &[bool], high: &[bool]) -> Vec<Item> {
    let mut out = posts(ev, low);
    out.extend(runs(ev, high, HIGH_TYPE, "fit_high"));
    out
}

fn type_band(venue: &Venue, type_name: &str) -> Result<[f64; 2], String> {
    let t = venue.types.get(type_name).ok_or_else(|| format!("venue has no {type_name} type"))?;
    match t.parts.first() {
        Some(crate::venue::Part::Box { size, z, .. }) => Ok([*z, z + size[2]]),
        Some(crate::venue::Part::Cylinder { z0, z1, .. }) | Some(crate::venue::Part::Leg4 { z0, z1, .. }) => Ok([*z0, *z1]),
        None => Err(format!("type {type_name} has no parts")),
    }
}

fn observed_cells(ev: &Evidence, item: &Item) -> Vec<usize> {
    let Some([len, _]) = item.size else { return ev.cell(item.x, item.y).into_iter().collect() };
    let n = (len / ev.resolution).round().max(1.0) as usize;
    (0..n).filter_map(|k| ev.cell(item.x - len / 2.0 + (k as f64 + 0.5) * ev.resolution, item.y)).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct CrowdReport {
    pub frames: usize,
    pub prior_cells: usize,
    pub tracks: usize,
    pub changes: usize,
    pub start: f64,
    pub end: f64,
}

pub fn crowd_timeline(
    venue: &Venue,
    map: &OccupancyMap,
    scene: &str,
    trips: &[Trip],
    cfg: &FitConfig,
    refine: Option<&WallNear>,
) -> Result<(CrowdTimeline, Scene, CrowdReport), String> {
    let ev = Evidence::new(map);
    let items = venue.scene_items(Some(scene))?;
    let (observed, furniture): (Vec<Item>, Vec<Item>) = items.into_iter().partition(|i| i.kind == OBSERVED_KIND);
    let mut low_static = WallGrid::from_map(map).for_scene(venue, Some(scene)).occupied;
    let mut high_static = low_static.clone();
    for item in &furniture {
        for i in ev.item_cells(venue, item, cfg.lidar_height, cfg.lidar_height) {
            low_static[i] = true;
        }
        for i in ev.item_cells(venue, item, cfg.camera_band.0, cfg.camera_band.1) {
            high_static[i] = true;
        }
    }
    let (low_static, high_static) = (ev.dilate(&low_static), ev.dilate(&high_static));
    let mut b = TimelineBuilder::default();
    let mut prior_cells = 0;
    for item in &observed {
        let layer = if item.type_name == HIGH_TYPE { HIGH } else { LOW };
        for i in observed_cells(&ev, item) {
            b.prior(i, layer);
            prior_cells += 1;
        }
    }
    let (mut frames, mut start, mut end) = (0, f64::INFINITY, f64::NEG_INFINITY);
    for trip in trips {
        let mount = trip.mount("laser_link", "base_footprint")?;
        for f in trip_frames(trip, 0.0, cfg.window, refine)? {
            let t = f.lidar.t;
            let obs = ev.observe(&f, &mount, cfg);
            for &i in obs.lidar_hits.iter().filter(|&&i| !low_static[i]) {
                b.observe(t, i, LOW, true);
            }
            for &i in obs.lidar_passes.iter().filter(|&&i| !low_static[i]) {
                b.observe(t, i, LOW, false);
            }
            for &i in obs.camera_hits.iter().filter(|&&i| !high_static[i]) {
                b.observe(t, i, HIGH, true);
            }
            for &i in obs.camera_passes.iter().filter(|&&i| !high_static[i]) {
                b.observe(t, i, HIGH, false);
            }
            frames += 1;
            start = start.min(t);
            end = end.max(t);
        }
    }
    let timeline = b.finish(ev.grid_frame(), type_band(venue, LOW_TYPE)?, type_band(venue, HIGH_TYPE)?, start, end);
    let mut furniture_scene = venue.scenes.get(scene).cloned().unwrap_or_default();
    furniture_scene.add.retain(|i| i.kind != OBSERVED_KIND);
    let report = CrowdReport {
        frames,
        prior_cells,
        tracks: timeline.tracks.len(),
        changes: timeline.tracks.iter().map(|t| t.times.len()).sum(),
        start,
        end,
    };
    Ok((timeline, furniture_scene, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy::{MapMeta, OccupancyMap};

    fn venue_with_chair() -> Venue {
        crate::venue::parse_venue(
            "schema: foodlink.venue/1\n\
             types:\n  chair: {parts: [{shape: cylinder, radius: 0.2, z0: 0.0, z1: 0.9}]}\n  \
             observed_low: {parts: [{shape: box, size: [1, 1, 0.45], z: 0, scale_xy: true}]}\n  \
             observed_high: {parts: [{shape: box, size: [1, 1, 0.8], z: 0.35, scale_xy: true}]}\n\
             items:\n- {id: c1, kind: chair, type: chair, x: 1.0, y: 1.0, mobility: movable}\n",
        )
        .unwrap()
    }

    fn empty_map() -> OccupancyMap {
        let meta: MapMeta = crate::occupancy::parse_map_meta("image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n").unwrap();
        OccupancyMap { width: 60, height: 60, occupied: vec![false; 3600], free: vec![true; 3600], meta }
    }

    #[test]
    fn a_chair_the_lidar_sees_through_is_removed_and_a_persistent_return_becomes_an_obstacle() {
        let venue = venue_with_chair();
        let map = empty_map();
        let mut ev = Evidence::new(&map);
        let chair = ev.cell(1.0, 1.0).unwrap();
        for j in ev.neighbours(chair).collect::<Vec<_>>() {
            ev.lidar_pass[j] = 30;
            ev.camera_pass[j] = 30;
        }
        let post = ev.cell(2.02, 2.02).unwrap();
        ev.lidar_hit[post] = 12;
        ev.lidar_pass[post] = 2;
        ev.lidar_buckets.hit[post] = 6;
        ev.lidar_buckets.seen[post] = 6;
        let walker = ev.cell(0.52, 2.52).unwrap();
        ev.lidar_hit[walker] = 3;
        ev.lidar_pass[walker] = 40;
        ev.lidar_buckets.hit[walker] = 1;
        ev.lidar_buckets.seen[walker] = 8;
        let (scene, report) = fit_scene(&venue, &map, None, &ev, &FitConfig::default()).unwrap();
        assert_eq!(scene.remove, vec!["c1".to_string()]);
        assert_eq!(report.low_cells, 1, "the walker crossing a lit cell is not furniture");
        let added = &scene.add[0];
        assert_eq!(added.type_name, LOW_TYPE);
        assert!((added.x - 2.025).abs() < 1e-9 && (added.y - 2.025).abs() < 1e-9);
    }

    #[test]
    fn a_chair_with_returns_on_it_is_kept() {
        let venue = venue_with_chair();
        let map = empty_map();
        let mut ev = Evidence::new(&map);
        let chair = ev.cell(1.0, 1.0).unwrap();
        ev.lidar_hit[chair] = 25;
        ev.lidar_pass[chair] = 5;
        ev.lidar_buckets.hit[chair] = 6;
        ev.lidar_buckets.seen[chair] = 6;
        let (scene, _) = fit_scene(&venue, &map, None, &ev, &FitConfig::default()).unwrap();
        assert!(scene.remove.is_empty());
        assert!(scene.add.is_empty(), "returns the chair explains need no new obstacle");
    }
}
