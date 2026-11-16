use crate::world3d::V3;
use serde::{Deserialize, Serialize};

pub const LOW: u8 = 1;
pub const HIGH: u8 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GridFrame {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin: [f64; 2],
}

impl GridFrame {
    pub fn cell(&self, x: f64, y: f64) -> Option<usize> {
        let c = ((x - self.origin[0]) / self.resolution).floor();
        let r = ((y - self.origin[1]) / self.resolution).floor();
        (c >= 0.0 && r >= 0.0 && (c as usize) < self.width && (r as usize) < self.height).then(|| r as usize * self.width + c as usize)
    }

    pub fn center(&self, i: usize) -> [f64; 2] {
        [
            self.origin[0] + ((i % self.width) as f64 + 0.5) * self.resolution,
            self.origin[1] + ((i / self.width) as f64 + 0.5) * self.resolution,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct CrowdGrid {
    pub frame: GridFrame,
    pub cells: Vec<u8>,
    pub low_z: [f64; 2],
    pub high_z: [f64; 2],
}

impl CrowdGrid {
    pub fn new(frame: GridFrame, low_z: [f64; 2], high_z: [f64; 2]) -> Self {
        let n = frame.width * frame.height;
        CrowdGrid { frame, cells: vec![0; n], low_z, high_z }
    }

    fn hits_in(&self, flags: u8, za: f64, zb: f64) -> Option<f64> {
        let (lo, hi) = (za.min(zb), za.max(zb));
        let mut best: Option<f64> = None;
        for (bit, range) in [(LOW, self.low_z), (HIGH, self.high_z)] {
            if flags & bit == 0 || hi < range[0] || lo > range[1] {
                continue;
            }
            let z = if za >= range[0] && za <= range[1] {
                za
            } else if za < range[0] {
                range[0]
            } else {
                range[1]
            };
            let frac = if (zb - za).abs() < 1e-12 { 0.0 } else { ((z - za) / (zb - za)).clamp(0.0, 1.0) };
            best = Some(best.map_or(frac, |b: f64| b.min(frac)));
        }
        best
    }

    pub fn cast(&self, o: V3, d: V3, t_max: f64) -> f64 {
        let h = d[0].hypot(d[1]);
        if h < 1e-9 {
            return f64::INFINITY;
        }
        let res = self.frame.resolution;
        let gx = (o[0] - self.frame.origin[0]) / res;
        let gy = (o[1] - self.frame.origin[1]) / res;
        let (mut cx, mut cy) = (gx.floor() as i64, gy.floor() as i64);
        let (dx, dy) = (d[0] / res, d[1] / res);
        let step_x = if dx > 0.0 { 1 } else { -1 };
        let step_y = if dy > 0.0 { 1 } else { -1 };
        let t_dx = if dx.abs() < 1e-12 { f64::INFINITY } else { 1.0 / dx.abs() };
        let t_dy = if dy.abs() < 1e-12 { f64::INFINITY } else { 1.0 / dy.abs() };
        let mut t_x = if dx.abs() < 1e-12 { f64::INFINITY } else if dx > 0.0 { ((cx + 1) as f64 - gx) / dx } else { (gx - cx as f64) / -dx };
        let mut t_y = if dy.abs() < 1e-12 { f64::INFINITY } else if dy > 0.0 { ((cy + 1) as f64 - gy) / dy } else { (gy - cy as f64) / -dy };
        let mut t_in = 0.0;
        while t_in < t_max {
            let t_out = t_x.min(t_y).min(t_max);
            if cx >= 0 && cy >= 0 && (cx as usize) < self.frame.width && (cy as usize) < self.frame.height {
                let flags = self.cells[cy as usize * self.frame.width + cx as usize];
                if flags != 0 {
                    let (za, zb) = (o[2] + d[2] * t_in, o[2] + d[2] * t_out);
                    if let Some(frac) = self.hits_in(flags, za, zb) {
                        let t = t_in + frac * (t_out - t_in);
                        if t > 1e-6 {
                            return t;
                        }
                    }
                }
            } else if t_in > 0.0 {
                let outside_x = (cx < 0 && step_x < 0) || (cx >= self.frame.width as i64 && step_x > 0);
                let outside_y = (cy < 0 && step_y < 0) || (cy >= self.frame.height as i64 && step_y > 0);
                if outside_x || outside_y {
                    return f64::INFINITY;
                }
            }
            t_in = t_out;
            if t_x < t_y {
                cx += step_x;
                t_x += t_dx;
            } else {
                cy += step_y;
                t_y += t_dy;
            }
        }
        f64::INFINITY
    }

    pub fn footprint_cells(&self, x: f64, y: f64, yaw: f64, half: [f64; 2]) -> Vec<usize> {
        let (s, c) = yaw.sin_cos();
        let reach = half[0].hypot(half[1]);
        let res = self.frame.resolution;
        let mut out = Vec::new();
        let n = (reach / res).ceil() as i64 + 1;
        let Some(center) = self.frame.cell(x, y) else { return out };
        let (cc, cr) = ((center % self.frame.width) as i64, (center / self.frame.width) as i64);
        for dr in -n..=n {
            for dc in -n..=n {
                let (col, row) = (cc + dc, cr + dr);
                if col < 0 || row < 0 || col as usize >= self.frame.width || row as usize >= self.frame.height {
                    continue;
                }
                let i = row as usize * self.frame.width + col as usize;
                let p = self.frame.center(i);
                let (px, py) = (p[0] - x, p[1] - y);
                if (c * px + s * py).abs() <= half[0] && (-s * px + c * py).abs() <= half[1] {
                    out.push(i);
                }
            }
        }
        out
    }

    pub fn touches(&self, x: f64, y: f64, yaw: f64, half: [f64; 2]) -> bool {
        self.footprint_cells(x, y, yaw, half).into_iter().any(|i| self.cells[i] != 0)
    }

    pub fn occupied(&self) -> Vec<(usize, u8)> {
        self.cells.iter().enumerate().filter(|(_, f)| **f != 0).map(|(i, f)| (i, *f)).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Track {
    pub cell: u32,
    pub layer: u8,
    pub prior: bool,
    pub times: Vec<f64>,
    pub states: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrowdTimeline {
    pub frame: GridFrame,
    pub low_z: [f64; 2],
    pub high_z: [f64; 2],
    pub start: f64,
    pub end: f64,
    pub tracks: Vec<Track>,
}

impl CrowdTimeline {
    pub fn grid(&self) -> CrowdGrid {
        CrowdGrid::new(self.frame.clone(), self.low_z, self.high_z)
    }

    pub fn apply(&self, t: f64, hold: f64, grid: &mut CrowdGrid) {
        grid.cells.iter_mut().for_each(|c| *c = 0);
        for tr in &self.tracks {
            let k = tr.times.partition_point(|&x| x <= t);
            let on = if k == 0 { tr.prior } else { tr.states[k - 1] };
            let recently_on = tr.times[..k].iter().zip(&tr.states[..k]).rev().take_while(|(x, _)| **x > t - hold).any(|(_, s)| !*s);
            if on || recently_on {
                grid.cells[tr.cell as usize] |= tr.layer;
            }
        }
    }
}

impl CrowdTimeline {
    pub fn without_actors(&self, people: &crate::actors::ActorTimeline, radius: f64, step: f64, layers: u8) -> CrowdTimeline {
        let bins = ((self.end - self.start) / step).ceil().max(0.0) as usize + 1;
        let at: Vec<Vec<crate::venue::Solid>> = (0..bins).map(|k| people.solids(self.start + k as f64 * step)).collect();
        let band = |layer: u8| if layer == LOW { self.low_z } else { self.high_z };
        let mut tracks = Vec::new();
        for tr in &self.tracks {
            if tr.prior || tr.layer & layers == 0 {
                tracks.push(tr.clone());
                continue;
            }
            let c = self.frame.center(tr.cell as usize);
            let [z0, z1] = band(tr.layer);
            let near = |k: usize| {
                at[k].iter().any(|solid| match *solid {
                    crate::venue::Solid::Cylinder { center, radius: r, z0: a, z1: b } => b >= z0 && a <= z1 && (center[0] - c[0]).hypot(center[1] - c[1]) <= r + radius,
                    crate::venue::Solid::Box { center, half, yaw } => {
                        let (s, k) = yaw.sin_cos();
                        let (dx, dy) = (c[0] - center[0], c[1] - center[1]);
                        center[2] + half[2] >= z0 && center[2] - half[2] <= z1 && (k * dx + s * dy).abs() <= half[0] + radius && (-s * dx + k * dy).abs() <= half[1] + radius
                    }
                })
            };
            let (mut times, mut states) = (Vec::new(), Vec::new());
            let mut last = false;
            for k in 0..bins {
                let t = self.start + k as f64 * step;
                let j = tr.times.partition_point(|&x| x <= t);
                let on = j > 0 && tr.states[j - 1] && !near(k);
                if on != last {
                    times.push(t);
                    states.push(on);
                    last = on;
                }
            }
            if !times.is_empty() {
                tracks.push(Track { cell: tr.cell, layer: tr.layer, prior: false, times, states });
            }
        }
        CrowdTimeline { frame: self.frame.clone(), low_z: self.low_z, high_z: self.high_z, start: self.start, end: self.end, tracks }
    }
}

#[derive(Default)]
pub struct TimelineBuilder {
    state: std::collections::HashMap<(u32, u8), usize>,
    tracks: Vec<Track>,
}

impl TimelineBuilder {
    pub fn prior(&mut self, cell: usize, layer: u8) {
        let key = (cell as u32, layer);
        if !self.state.contains_key(&key) {
            self.state.insert(key, self.tracks.len());
            self.tracks.push(Track { cell: cell as u32, layer, prior: true, times: Vec::new(), states: Vec::new() });
        }
    }

    pub fn observe(&mut self, t: f64, cell: usize, layer: u8, occupied: bool) {
        let key = (cell as u32, layer);
        let idx = match self.state.get(&key) {
            Some(&i) => i,
            None => {
                if !occupied {
                    return;
                }
                self.state.insert(key, self.tracks.len());
                self.tracks.push(Track { cell: cell as u32, layer, prior: false, times: Vec::new(), states: Vec::new() });
                self.tracks.len() - 1
            }
        };
        let tr = &mut self.tracks[idx];
        let current = tr.states.last().copied().unwrap_or(tr.prior);
        if current != occupied || tr.states.is_empty() {
            tr.times.push(t);
            tr.states.push(occupied);
        }
    }

    pub fn finish(mut self, frame: GridFrame, low_z: [f64; 2], high_z: [f64; 2], start: f64, end: f64) -> CrowdTimeline {
        for tr in &mut self.tracks {
            let mut times = Vec::new();
            let mut states = Vec::new();
            let mut last = tr.prior;
            for (t, s) in tr.times.iter().zip(&tr.states) {
                if *s != last {
                    times.push(*t);
                    states.push(*s);
                    last = *s;
                }
            }
            tr.times = times;
            tr.states = states;
        }
        self.tracks.retain(|t| t.prior || !t.states.is_empty());
        CrowdTimeline { frame, low_z, high_z, start, end, tracks: self.tracks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> GridFrame {
        GridFrame { width: 40, height: 40, resolution: 0.05, origin: [0.0, 0.0] }
    }

    #[test]
    fn a_low_cell_stops_a_lidar_ray_and_a_high_ray_passes_over_it() {
        let mut g = CrowdGrid::new(frame(), [0.0, 0.45], [0.35, 0.55]);
        let i = g.frame.cell(1.02, 0.52).unwrap();
        g.cells[i] = LOW;
        let t = g.cast([0.1, 0.52, 0.23], [1.0, 0.0, 0.0], 10.0);
        assert!((t - 0.9).abs() < 0.03, "hit at {t}");
        assert!(g.cast([0.1, 0.52, 0.8], [1.0, 0.0, 0.0], 10.0).is_infinite());
        g.cells[i] = HIGH;
        assert!(g.cast([0.1, 0.52, 0.23], [1.0, 0.0, 0.0], 10.0).is_infinite());
        let down = g.cast([0.1, 0.52, 1.0], [0.9, 0.0, -0.55], 10.0);
        assert!(down.is_finite(), "a ray sloping down into the high band hits it");
    }

    #[test]
    fn cells_a_tracked_person_stands_on_are_handed_to_the_person_and_others_stay() {
        let mut b = TimelineBuilder::default();
        let near = frame().cell(1.02, 1.02).unwrap();
        let far = frame().cell(1.52, 1.52).unwrap();
        b.observe(0.0, near, LOW, true);
        b.observe(0.0, far, LOW, true);
        let tl = b.finish(frame(), [0.0, 0.45], [0.35, 0.55], 0.0, 4.0);
        let people: crate::actors::ActorTimeline = serde_json::from_str(
            r#"{"start": 0, "end": 4, "types": {"person_standing": {"parts": [{"shape": "cylinder", "radius": 0.08, "z0": 0.0, "z1": 0.85}]}}, "actors": [{"id": "p", "type": "person_standing", "samples": [[1.0, 1.0, 1.0, 0.0], [2.0, 1.0, 1.0, 0.0]], "end": 2.0}]}"#,
        )
        .unwrap();
        let left = tl.without_actors(&people, 0.05, 0.1, LOW | HIGH);
        let mut g = left.grid();
        left.apply(1.5, 0.0, &mut g);
        assert_eq!((g.cells[near], g.cells[far]), (0, LOW), "while the person stands there the cell is theirs");
        left.apply(3.0, 0.0, &mut g);
        assert_eq!(g.cells[near], LOW, "after the person leaves the cell replay resumes");
    }

    #[test]
    fn the_last_observation_before_t_decides_and_unseen_cells_keep_their_prior() {
        let mut b = TimelineBuilder::default();
        b.prior(5, LOW);
        b.prior(9, LOW);
        b.observe(10.0, 5, LOW, false);
        b.observe(12.0, 5, LOW, true);
        b.observe(11.0, 7, LOW, true);
        b.observe(13.0, 7, LOW, false);
        b.observe(14.0, 8, LOW, false);
        let tl = b.finish(frame(), [0.0, 0.45], [0.35, 0.55], 10.0, 14.0);
        let mut g = tl.grid();
        let at = |t: f64, g: &mut CrowdGrid| {
            tl.apply(t, 0.0, g);
            (g.cells[5], g.cells[7], g.cells[8], g.cells[9])
        };
        assert_eq!(at(9.0, &mut g), (LOW, 0, 0, LOW));
        assert_eq!(at(10.5, &mut g), (0, 0, 0, LOW));
        assert_eq!(at(11.5, &mut g), (0, LOW, 0, LOW));
        assert_eq!(at(13.5, &mut g), (LOW, 0, 0, LOW));
        tl.apply(13.5, 1.0, &mut g);
        assert_eq!(g.cells[7], LOW, "a cell seen occupied within the hold time stays on");
        tl.apply(14.5, 1.0, &mut g);
        assert_eq!(g.cells[7], 0);
    }
}
