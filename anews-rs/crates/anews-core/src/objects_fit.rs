use crate::occupancy::OccupancyMap;
use crate::scene_fit::{observed_items, unexplained, Evidence, FitConfig, OBSERVED_KIND};
use crate::venue::{expand_item, Item, Scene, Solid, Venue};
use crate::world3d::WallGrid;
use serde::Serialize;
use std::collections::{BTreeMap, BinaryHeap};

pub const CHAIR_TYPE: &str = "chair";
pub const SEATED_TYPE: &str = "person_seated";
pub const PEDESTAL_TYPE: &str = "table_pedestal";
pub const TWIN_PEDESTAL_TYPE: &str = "table_twin_pedestal";
pub const CHAIR_SIZE: [f64; 2] = [0.45, 0.45];

#[derive(Debug, Clone)]
pub struct ObjectFitConfig {
    pub table_sizes: Vec<[f64; 2]>,
    pub twin_from_m: f64,
    pub chair_yaw_step_deg: f64,
    pub table_yaw_step_deg: f64,
    pub reach_m: f64,
    pub low_weight: f64,
    pub high_weight: f64,
    pub seen_through_penalty: f64,
    pub area_cost: f64,
    pub min_gain: f64,
    pub min_table_gain: f64,
    pub camera_only_gain: f64,
    pub chair_spacing_m: f64,
    pub chair_tuck_m: f64,
    pub support_m: f64,
    pub min_chair_supports: usize,
    pub refine_shift_m: f64,
    pub refine_turn_deg: f64,
}

impl Default for ObjectFitConfig {
    fn default() -> Self {
        ObjectFitConfig {
            table_sizes: vec![[0.7, 0.7], [0.8, 0.8], [1.2, 0.7], [1.4, 0.75]],
            twin_from_m: 1.1,
            chair_yaw_step_deg: 15.0,
            table_yaw_step_deg: 15.0,
            reach_m: 0.45,
            low_weight: 1.0,
            high_weight: 0.5,
            seen_through_penalty: 1.0,
            area_cost: 0.005,
            min_gain: 3.0,
            min_table_gain: 8.0,
            camera_only_gain: 4.0,
            chair_spacing_m: 0.42,
            chair_tuck_m: 0.12,
            support_m: 0.05,
            min_chair_supports: 3,
            refine_shift_m: 0.06,
            refine_turn_deg: 12.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Shape {
    Chair,
    Diner,
    Table([f64; 2]),
}

struct Template {
    shape: Shape,
    yaw_deg: f64,
    low: Vec<(i64, i64)>,
    high: Vec<(i64, i64)>,
    low_reach: Vec<(i64, i64)>,
    high_reach: Vec<(i64, i64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObjectFitReport {
    pub tables: usize,
    pub chairs: usize,
    pub seated_people: usize,
    pub target_low_cells: usize,
    pub target_high_cells: usize,
    pub residual_low_cells: usize,
    pub residual_high_cells: usize,
    pub residual_items: usize,
}

fn shape_items(shape: Shape, x: f64, y: f64, yaw_deg: f64, twin_from: f64) -> Vec<Item> {
    let item = |type_name: &str, kind: &str, size: Option<[f64; 2]>| Item {
        id: String::new(),
        kind: kind.into(),
        type_name: type_name.into(),
        x,
        y,
        yaw_deg,
        size,
        number: None,
        mobility: Some("movable".into()),
        extra: BTreeMap::new(),
    };
    match shape {
        Shape::Chair => vec![item(CHAIR_TYPE, "chair", Some(CHAIR_SIZE))],
        Shape::Diner => vec![item(CHAIR_TYPE, "chair", Some(CHAIR_SIZE)), item(SEATED_TYPE, "person", None)],
        Shape::Table(size) => {
            let t = if size[0] >= twin_from { TWIN_PEDESTAL_TYPE } else { PEDESTAL_TYPE };
            vec![item(t, "table", Some(size))]
        }
    }
}

fn inside(solid: &Solid, dx: f64, dy: f64, pad: f64) -> bool {
    match *solid {
        Solid::Box { center, half, yaw } => {
            let (s, c) = yaw.sin_cos();
            let (px, py) = (dx - center[0], dy - center[1]);
            (c * px + s * py).abs() <= half[0] + pad && (-s * px + c * py).abs() <= half[1] + pad
        }
        Solid::Cylinder { center, radius, .. } => (dx - center[0]).hypot(dy - center[1]) <= radius + pad,
    }
}

fn in_band(solid: &Solid, lo: f64, hi: f64) -> bool {
    match *solid {
        Solid::Box { center, half, .. } => center[2] + half[2] >= lo && center[2] - half[2] <= hi,
        Solid::Cylinder { z0, z1, .. } => z1 >= lo && z0 <= hi,
    }
}

fn pattern(solids: &[Solid], lo: f64, hi: f64, res: f64) -> Vec<(i64, i64)> {
    let band: Vec<&Solid> = solids.iter().filter(|s| in_band(s, lo, hi)).collect();
    let n = (1.0 / res).ceil() as i64;
    let mut out = Vec::new();
    for dr in -n..=n {
        for dc in -n..=n {
            let (x, y) = (dc as f64 * res, dr as f64 * res);
            if band.iter().any(|s| inside(s, x, y, res / 2.0)) {
                out.push((dc, dr));
            }
        }
    }
    out
}

fn grow(cells: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = cells.iter().flat_map(|&(c, r)| (-1..=1).flat_map(move |a| (-1..=1).map(move |b| (c + a, r + b)))).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn outline(cells: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let set: std::collections::HashSet<(i64, i64)> = cells.iter().copied().collect();
    cells.iter().copied().filter(|&(c, r)| [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|(a, b)| !set.contains(&(c + a, r + b)))).collect()
}

fn templates(venue: &Venue, cfg: &FitConfig, oc: &ObjectFitConfig, res: f64) -> Vec<Template> {
    let mut shapes: Vec<(Shape, f64)> = vec![(Shape::Chair, oc.chair_yaw_step_deg), (Shape::Diner, oc.chair_yaw_step_deg)];
    shapes.extend(oc.table_sizes.iter().map(|s| (Shape::Table(*s), oc.table_yaw_step_deg)));
    let mut out = Vec::new();
    for (shape, step) in shapes {
        let turn = match shape {
            Shape::Table(s) if (s[0] - s[1]).abs() < 1e-9 => 90.0,
            Shape::Table(_) => 180.0,
            _ => 360.0,
        };
        let mut yaw = 0.0;
        while yaw < turn - 1e-9 {
            let solids: Vec<Solid> = shape_items(shape, 0.0, 0.0, yaw, oc.twin_from_m).iter().flat_map(|i| expand_item(i, &venue.types[&i.type_name])).collect();
            let low = pattern(&solids, cfg.lidar_height, cfg.lidar_height, res);
            let high = pattern(&solids, cfg.camera_band.0, cfg.camera_band.1, res);
            out.push(Template { shape, yaw_deg: yaw, low_reach: grow(&low), high_reach: grow(&outline(&high)), low, high });
            yaw += step;
        }
    }
    out
}

struct Placed {
    shape: Shape,
    x: f64,
    y: f64,
    yaw: f64,
}

fn table_contains(t: &Placed, size: [f64; 2], x: f64, y: f64, shrink: f64) -> bool {
    let (s, c) = t.yaw.sin_cos();
    let (dx, dy) = (x - t.x, y - t.y);
    (c * dx + s * dy).abs() < size[0] / 2.0 - shrink && (-s * dx + c * dy).abs() < size[1] / 2.0 - shrink
}

fn corners(x: f64, y: f64, yaw: f64, size: [f64; 2]) -> [[f64; 2]; 4] {
    let (s, c) = yaw.sin_cos();
    let at = |u: f64, v: f64| [x + c * u * size[0] / 2.0 - s * v * size[1] / 2.0, y + s * u * size[0] / 2.0 + c * v * size[1] / 2.0];
    [at(-1.0, -1.0), at(1.0, -1.0), at(1.0, 1.0), at(-1.0, 1.0)]
}

fn rects_overlap(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> bool {
    for poly in [&a, &b] {
        for k in 0..4 {
            let (p, q) = (poly[k], poly[(k + 1) % 4]);
            let n = [q[1] - p[1], p[0] - q[0]];
            let span = |r: &[[f64; 2]; 4]| r.iter().map(|c| c[0] * n[0] + c[1] * n[1]).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
            let ((a0, a1), (b0, b1)) = (span(&a), span(&b));
            if a1 <= b0 || b1 <= a0 {
                return false;
            }
        }
    }
    true
}

fn supports(shape: Shape, twin_from: f64) -> Vec<[f64; 2]> {
    let legs = [[-0.195, -0.195], [-0.195, 0.195], [0.195, -0.195], [0.195, 0.195]];
    match shape {
        Shape::Chair => legs.to_vec(),
        Shape::Diner => legs.iter().copied().chain([[0.33, 0.09], [0.33, -0.09]]).collect(),
        Shape::Table(size) if size[0] >= twin_from => vec![[-0.35, 0.0], [0.35, 0.0]],
        Shape::Table(_) => vec![[0.0, 0.0]],
    }
}

fn required(shape: Shape, twin_from: f64, oc: &ObjectFitConfig) -> usize {
    match shape {
        Shape::Chair | Shape::Diner => oc.min_chair_supports,
        Shape::Table(_) => supports(shape, twin_from).len(),
    }
}

fn support_count(ev: &Evidence, target: &[bool], p: &Placed, oc: &ObjectFitConfig) -> usize {
    let (s, c) = p.yaw.sin_cos();
    supports(p.shape, oc.twin_from_m)
        .iter()
        .filter(|l| {
            let (x, y) = (p.x + c * l[0] - s * l[1], p.y + s * l[0] + c * l[1]);
            ev.cell(x, y).is_some_and(|i| {
                ev.neighbours(i).any(|j| {
                    let q = ev.center(j);
                    target[j] && (q[0] - x).hypot(q[1] - y) <= oc.support_m
                })
            })
        })
        .count()
}

fn refine(ev: &Evidence, target: &[bool], p: &Placed, oc: &ObjectFitConfig) -> (Placed, usize) {
    let steps = (oc.refine_shift_m / 0.01).round() as i32;
    let turns = (oc.refine_turn_deg / 2.0).round() as i32;
    let mut best: (usize, f64, f64, f64) = (support_count(ev, target, p, oc), 0.0, 0.0, 0.0);
    for k in -turns..=turns {
        for a in -steps..=steps {
            for b in -steps..=steps {
                let (dx, dy, dyaw) = (a as f64 * 0.01, b as f64 * 0.01, (k as f64 * 2.0).to_radians());
                let q = Placed { shape: p.shape, x: p.x + dx, y: p.y + dy, yaw: p.yaw + dyaw };
                let n = support_count(ev, target, &q, oc);
                let nearer = dx.hypot(dy) + dyaw.abs() * 0.2 < best.1.hypot(best.2) + best.3.abs() * 0.2;
                if n > best.0 || (n == best.0 && nearer) {
                    best = (n, dx, dy, dyaw);
                }
            }
        }
    }
    (Placed { shape: p.shape, x: p.x + best.1, y: p.y + best.2, yaw: p.yaw + best.3 }, best.0)
}

fn conflicts(placed: &[Placed], cand: &Placed, oc: &ObjectFitConfig) -> bool {
    placed.iter().any(|p| match (p.shape, cand.shape) {
        (Shape::Table(a), Shape::Table(b)) => rects_overlap(corners(p.x, p.y, p.yaw, a), corners(cand.x, cand.y, cand.yaw, b)),
        (Shape::Table(a), _) => table_contains(p, a, cand.x, cand.y, oc.chair_tuck_m),
        (_, Shape::Table(b)) => table_contains(cand, b, p.x, p.y, oc.chair_tuck_m),
        _ => (p.x - cand.x).hypot(p.y - cand.y) < oc.chair_spacing_m,
    })
}

struct State<'a> {
    ev: &'a Evidence,
    low: Vec<bool>,
    high: Vec<bool>,
    blocked: Vec<bool>,
}

impl State<'_> {
    fn at(&self, i: usize, d: (i64, i64)) -> Option<usize> {
        let (c, r) = ((i % self.ev.width) as i64 + d.0, (i / self.ev.width) as i64 + d.1);
        (c >= 0 && r >= 0 && (c as usize) < self.ev.width && (r as usize) < self.ev.height).then(|| r as usize * self.ev.width + c as usize)
    }

    fn gain(&self, i: usize, t: &Template, oc: &ObjectFitConfig) -> Option<f64> {
        let ev = self.ev;
        let mut g = 0.0;
        for &d in t.low.iter().chain(t.high.iter()) {
            let j = self.at(i, d)?;
            if self.blocked[j] {
                return None;
            }
        }
        for &d in &t.low {
            let j = self.at(i, d)?;
            if ev.lidar_pass[j] >= 2 && ev.lidar_pass[j] > 2 * ev.lidar_hit[j] {
                g -= oc.seen_through_penalty;
            }
        }
        for &d in &t.high {
            let j = self.at(i, d)?;
            if ev.camera_pass[j] >= 4 && ev.camera_pass[j] > 3 * ev.camera_hit[j] {
                g -= oc.seen_through_penalty * oc.high_weight;
            }
        }
        g -= oc.area_cost * t.high.len() as f64;
        let mut seen_low = 0usize;
        for &d in &t.low_reach {
            if let Some(j) = self.at(i, d)
                && self.low[j]
            {
                g += oc.low_weight;
                seen_low += 1;
            }
        }
        for &d in &t.high_reach {
            if let Some(j) = self.at(i, d) {
                if self.high[j] {
                    g += oc.high_weight;
                }
            }
        }
        if matches!(t.shape, Shape::Chair | Shape::Diner) && seen_low == 0 && g < oc.camera_only_gain {
            return None;
        }
        Some(g)
    }

    fn explain(&mut self, i: usize, t: &Template) {
        for &d in &t.low_reach {
            if let Some(j) = self.at(i, d) {
                self.low[j] = false;
            }
        }
        for &d in &t.high_reach {
            if let Some(j) = self.at(i, d) {
                self.high[j] = false;
            }
        }
    }
}

#[derive(PartialEq)]
struct Entry(f64, usize, usize);

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(o.1.cmp(&self.1)).then(o.2.cmp(&self.2))
    }
}

pub fn fit_objects(venue: &Venue, map: &OccupancyMap, scene: &str, ev: &Evidence, cfg: &FitConfig, oc: &ObjectFitConfig) -> Result<ObjectFit, String> {
    for t in [CHAIR_TYPE, SEATED_TYPE, PEDESTAL_TYPE, TWIN_PEDESTAL_TYPE] {
        if !venue.types.contains_key(t) {
            return Err(format!("venue has no {t} type; the object fit places it"));
        }
    }
    let kept: Vec<Item> = venue.scene_items(Some(scene))?.into_iter().filter(|i| i.kind != OBSERVED_KIND).collect();
    let grid = WallGrid::from_map(map).for_scene(venue, Some(scene));
    let (low, high) = unexplained(venue, &grid, &kept, ev, cfg);
    let n = low.len();
    let mut blocked: Vec<bool> = (0..n).map(|i| grid.occupied[i] || ev.driven[i]).collect();
    for item in &kept {
        for i in ev.item_cells(venue, item, 0.0, 2.0) {
            blocked[i] = true;
        }
    }
    let (target_low, target_high) = (low.iter().filter(|b| **b).count(), high.iter().filter(|b| **b).count());
    let target_low_mask = low.clone();
    let target_high_mask = high.clone();
    let mut st = State { ev, low, high, blocked };
    let tpl = templates(venue, cfg, oc, ev.resolution);
    let reach = (oc.reach_m / ev.resolution).ceil() as i64;
    let mut near = vec![false; n];
    for i in 0..n {
        if st.low[i] || st.high[i] {
            for dr in -reach..=reach {
                for dc in -reach..=reach {
                    if let Some(j) = st.at(i, (dc, dr)) {
                        near[j] = true;
                    }
                }
            }
        }
    }
    let mut heap = BinaryHeap::new();
    for i in (0..n).filter(|&i| near[i] && !st.blocked[i]) {
        for (k, t) in tpl.iter().enumerate() {
            let floor = if matches!(t.shape, Shape::Table(_)) { oc.min_table_gain } else { oc.min_gain };
            if let Some(g) = st.gain(i, t, oc)
                && g >= floor
            {
                heap.push(Entry(g, i, k));
            }
        }
    }
    let mut placed: Vec<Placed> = Vec::new();
    while let Some(Entry(stale, i, k)) = heap.pop() {
        let t = &tpl[k];
        let floor = if matches!(t.shape, Shape::Table(_)) { oc.min_table_gain } else { oc.min_gain };
        let Some(g) = st.gain(i, t, oc) else { continue };
        if g < floor {
            continue;
        }
        if g + 1e-9 < stale {
            heap.push(Entry(g, i, k));
            continue;
        }
        let c = ev.center(i);
        let (cand, held) = refine(ev, &target_low_mask, &Placed { shape: t.shape, x: c[0], y: c[1], yaw: t.yaw_deg.to_radians() }, oc);
        if held < required(t.shape, oc.twin_from_m, oc) || conflicts(&placed, &cand, oc) {
            continue;
        }
        st.explain(i, t);
        placed.push(cand);
    }
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    let mut add: Vec<Item> = Vec::new();
    let mut groups: Vec<Vec<Item>> = Vec::new();
    let mut counts = [0usize; 3];
    for p in &placed {
        let (prefix, slot) = match p.shape {
            Shape::Table(_) => ("fit_table", 0),
            Shape::Diner => ("fit_seat", 1),
            Shape::Chair => ("fit_chair", 2),
        };
        counts[slot] += 1;
        let id = format!("{prefix}_{}", counts[slot]);
        let mut group = Vec::new();
        for mut item in shape_items(p.shape, round(p.x), round(p.y), round(p.yaw.to_degrees()), oc.twin_from_m) {
            item.id = if item.type_name == SEATED_TYPE { format!("{id}_diner") } else { id.clone() };
            add.push(item.clone());
            group.push(item);
        }
        groups.push(group);
    }
    let residual = observed_items(ev, &st.low, &st.high);
    let report = ObjectFitReport {
        tables: counts[0],
        chairs: counts[1] + counts[2],
        seated_people: counts[1],
        target_low_cells: target_low,
        target_high_cells: target_high,
        residual_low_cells: st.low.iter().filter(|b| **b).count(),
        residual_high_cells: st.high.iter().filter(|b| **b).count(),
        residual_items: residual.len(),
    };
    add.extend(residual);
    let base = venue.scenes.get(scene).cloned().unwrap_or_default();
    let mut base_add: Vec<Item> = base.add.into_iter().filter(|i| i.kind != OBSERVED_KIND).collect();
    let furniture = Scene { remove: base.remove.clone(), moves: base.moves.clone(), add: base_add.clone(), clear_walls: base.clear_walls.clone() };
    base_add.extend(add);
    let evidence = observed_items(ev, &target_low_mask, &target_high_mask);
    Ok(ObjectFit { scene: Scene { remove: base.remove, moves: base.moves, add: base_add, clear_walls: base.clear_walls }, furniture, groups, evidence, report })
}

pub struct ObjectFit {
    pub scene: Scene,
    pub furniture: Scene,
    pub groups: Vec<Vec<Item>>,
    pub evidence: Vec<Item>,
    pub report: ObjectFitReport,
}

pub fn footprint(item: &Item) -> ([f64; 2], [f64; 2], f64) {
    let half = match item.size {
        Some([w, d]) => [w / 2.0, d / 2.0],
        None => [0.2, 0.2],
    };
    ([item.x, item.y], half, item.yaw_deg.to_radians())
}

pub fn covers(item: &Item, x: f64, y: f64, pad: f64) -> bool {
    let (c, half, yaw) = footprint(item);
    let (s, k) = yaw.sin_cos();
    let (dx, dy) = (x - c[0], y - c[1]);
    (k * dx + s * dy).abs() <= half[0] + pad && (-s * dx + k * dy).abs() <= half[1] + pad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy::parse_map_meta;

    fn venue() -> Venue {
        crate::venue::parse_venue(
            "schema: foodlink.venue/1\n\
             types:\n  \
             chair: {parts: [{shape: leg4, radius: 0.015, inset: 0.03, z0: 0, z1: 0.43}, {shape: box, size: [0.45, 0.45, 0.04], z: 0.43}]}\n  \
             person_seated: {parts: [{shape: cylinder, radius: 0.19, z0: 0.45, z1: 1.15}, {shape: cylinder, radius: 0.07, z0: 0, z1: 0.45, offset: [0.25, 0]}]}\n  \
             table_pedestal: {parts: [{shape: box, size: [1, 1, 0.04], z: 0.71, scale_xy: true}, {shape: cylinder, radius: 0.04, z0: 0, z1: 0.71}]}\n  \
             table_twin_pedestal: {parts: [{shape: box, size: [1, 1, 0.04], z: 0.71, scale_xy: true}, {shape: cylinder, radius: 0.04, z0: 0, z1: 0.71, offset: [0.35, 0]}, {shape: cylinder, radius: 0.04, z0: 0, z1: 0.71, offset: [-0.35, 0]}]}\n  \
             observed_low: {parts: [{shape: cylinder, radius: 0.025, z0: 0, z1: 0.45}]}\n  \
             observed_high: {parts: [{shape: box, size: [1, 1, 0.2], z: 0.35, scale_xy: true}]}\n\
             scenes:\n  empty: {}\n",
        )
        .unwrap()
    }

    fn persistent_lidar(ev: &mut Evidence, x: f64, y: f64) {
        let i = ev.cell(x, y).unwrap();
        ev.lidar_hit[i] = 20;
        ev.lidar_buckets.hit[i] = 5;
        ev.lidar_buckets.seen[i] = 5;
    }

    #[test]
    fn four_leg_returns_around_a_seat_become_a_chair_and_a_lone_return_stays_evidence() {
        let v = venue();
        let meta = parse_map_meta("image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n").unwrap();
        let map = OccupancyMap { meta, width: 80, height: 80, occupied: vec![false; 6400], free: vec![true; 6400] };
        let mut ev = Evidence::new(&map);
        for (sx, sy) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            persistent_lidar(&mut ev, 2.0 + sx * 0.195, 2.0 + sy * 0.195);
        }
        persistent_lidar(&mut ev, 0.6, 3.4);
        let ObjectFit { scene, report, .. } = fit_objects(&v, &map, "empty", &ev, &FitConfig::default(), &ObjectFitConfig::default()).unwrap();
        assert_eq!(report.chairs, 1, "{report:?}");
        assert_eq!(report.tables, 0);
        let chair = scene.add.iter().find(|i| i.type_name == CHAIR_TYPE).unwrap();
        assert!((chair.x - 2.0).abs() <= 0.05 && (chair.y - 2.0).abs() <= 0.05, "chair at ({}, {})", chair.x, chair.y);
        assert_eq!(report.residual_low_cells, 1, "the lone return is not a chair");
    }
}
