use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use serde_json::{json, Value};

use crate::occupancy::OccupancyMap;
use crate::venue::{parse_color, Item, MaterialDef, Part, Venue};
use crate::world3d::WallGrid;

pub const WALL_MATERIAL: &str = "wall";
pub const FLOOR_MATERIAL: &str = "floor";
pub const FURNITURE_MATERIAL: &str = "furniture";

const WALL_SIMPLIFY_M: f64 = 0.1;
const WALL_SNAP_DEG: f64 = 15.0;
const FLOOR_SIMPLIFY_M: f64 = 0.12;
const MIN_WALL_BLOB_M2: f64 = 0.05;
const MIN_RING_M2: f64 = 0.02;
const CYLINDER_SEGMENTS: usize = 24;
const LEG_SEGMENTS: usize = 12;

#[derive(Debug, Clone, Serialize)]
pub struct SceneStats {
    pub wall_outlines: usize,
    pub wall_holes: usize,
    pub floor: bool,
    pub items: usize,
    pub modelled_items: usize,
    pub triangles: usize,
    pub materials: Vec<String>,
}

fn default_material(name: &str) -> MaterialDef {
    let (color, roughness) = match name {
        WALL_MATERIAL => ("#ece6dc", 0.92),
        FLOOR_MATERIAL => ("#b8a48a", 0.78),
        _ => ("#a37a58", 0.6),
    };
    MaterialDef { color: color.into(), roughness, metallic: 0.0 }
}

struct Mask {
    w: usize,
    h: usize,
    cells: Vec<bool>,
}

impl Mask {
    fn get(&self, c: i64, r: i64) -> bool {
        c >= 0 && r >= 0 && (c as usize) < self.w && (r as usize) < self.h && self.cells[r as usize * self.w + c as usize]
    }

    fn morph(&self, steps: usize, grow: bool) -> Mask {
        let mut cur = self.cells.clone();
        for _ in 0..steps {
            let prev = Mask { w: self.w, h: self.h, cells: cur };
            let mut next = prev.cells.clone();
            for r in 0..self.h as i64 {
                for c in 0..self.w as i64 {
                    let mut any = false;
                    let mut all = true;
                    for dr in -1..=1 {
                        for dc in -1..=1 {
                            let v = prev.get(c + dc, r + dr);
                            any |= v;
                            all &= v;
                        }
                    }
                    next[r as usize * self.w + c as usize] = if grow { any } else { all };
                }
            }
            cur = next;
        }
        Mask { w: self.w, h: self.h, cells: cur }
    }

    fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.cells.len()];
        let mut out = Vec::new();
        for start in 0..self.cells.len() {
            if !self.cells[start] || seen[start] {
                continue;
            }
            seen[start] = true;
            let mut stack = vec![start];
            let mut comp = Vec::new();
            while let Some(i) = stack.pop() {
                comp.push(i);
                let (c, r) = ((i % self.w) as i64, (i / self.w) as i64);
                for (dc, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if self.get(c + dc, r + dr) {
                        let j = (r + dr) as usize * self.w + (c + dc) as usize;
                        if !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
            out.push(comp);
        }
        out
    }

    fn only(&self, cells: impl IntoIterator<Item = usize>) -> Mask {
        let mut out = vec![false; self.cells.len()];
        for i in cells {
            out[i] = true;
        }
        Mask { w: self.w, h: self.h, cells: out }
    }

    fn rings(&self) -> Vec<Vec<[i64; 2]>> {
        let mut edges: HashMap<[i64; 2], Vec<[i64; 2]>> = HashMap::new();
        let mut add = |a: [i64; 2], b: [i64; 2]| edges.entry(a).or_default().push(b);
        for r in 0..self.h as i64 {
            for c in 0..self.w as i64 {
                if !self.get(c, r) {
                    continue;
                }
                if !self.get(c, r - 1) {
                    add([c, r], [c + 1, r]);
                }
                if !self.get(c + 1, r) {
                    add([c + 1, r], [c + 1, r + 1]);
                }
                if !self.get(c, r + 1) {
                    add([c + 1, r + 1], [c, r + 1]);
                }
                if !self.get(c - 1, r) {
                    add([c, r + 1], [c, r]);
                }
            }
        }
        let mut starts: Vec<[i64; 2]> = edges.keys().copied().collect();
        starts.sort();
        let mut rings = Vec::new();
        for start in starts {
            while edges.get(&start).is_some_and(|v| !v.is_empty()) {
                let mut ring = vec![start];
                let mut cur = start;
                let mut dir = [0i64, 0i64];
                loop {
                    let outs = edges.get_mut(&cur).expect("traced vertex has edges");
                    let pick = if outs.len() == 1 {
                        0
                    } else {
                        let left = [-dir[1], dir[0]];
                        let straight = dir;
                        let rank = |to: &[i64; 2]| {
                            let d = [to[0] - cur[0], to[1] - cur[1]];
                            if d == left {
                                0
                            } else if d == straight {
                                1
                            } else {
                                2
                            }
                        };
                        (0..outs.len()).min_by_key(|&i| rank(&outs[i])).unwrap()
                    };
                    let next = outs.swap_remove(pick);
                    dir = [next[0] - cur[0], next[1] - cur[1]];
                    cur = next;
                    if cur == start {
                        break;
                    }
                    ring.push(cur);
                }
                rings.push(ring);
            }
        }
        rings
    }
}

fn signed_area(ring: &[[f64; 2]]) -> f64 {
    let n = ring.len();
    (0..n).map(|i| {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f64>() / 2.0
}

fn point_in_ring(p: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0);
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

fn simplify_chain(points: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0, points.len() - 1)];
    while let Some((i, j)) = stack.pop() {
        let (mut best, mut at) = (0.0, i);
        for k in i + 1..j {
            let d = segment_distance(points[k], points[i], points[j]);
            if d > best {
                best = d;
                at = k;
            }
        }
        if best > tol {
            keep[at] = true;
            stack.push((i, at));
            stack.push((at, j));
        }
    }
    points.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

fn simplify_ring(ring: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    if ring.len() < 4 {
        return ring.to_vec();
    }
    let far = (1..ring.len())
        .max_by(|&a, &b| {
            let da = (ring[a][0] - ring[0][0]).hypot(ring[a][1] - ring[0][1]);
            let db = (ring[b][0] - ring[0][0]).hypot(ring[b][1] - ring[0][1]);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let first = simplify_chain(&ring[..=far], tol);
    let mut second_src: Vec<[f64; 2]> = ring[far..].to_vec();
    second_src.push(ring[0]);
    let second = simplify_chain(&second_src, tol);
    let mut out = first;
    out.extend_from_slice(&second[1..second.len() - 1]);
    out
}

struct Polygon {
    outer: Vec<[f64; 2]>,
    holes: Vec<Vec<[f64; 2]>>,
}

fn polygons(mask: &Mask, origin: [f64; 2], res: f64, tol: f64, keep_holes: bool) -> Vec<Polygon> {
    let to_world = |v: &[i64; 2]| [origin[0] + v[0] as f64 * res, origin[1] + v[1] as f64 * res];
    let mut outers = Vec::new();
    let mut holes = Vec::new();
    for ring in mask.rings() {
        let world: Vec<[f64; 2]> = ring.iter().map(to_world).collect();
        let simple = simplify_ring(&world, tol);
        if simple.len() < 3 {
            continue;
        }
        let area = signed_area(&simple);
        if area.abs() < MIN_RING_M2 {
            continue;
        }
        if area > 0.0 {
            outers.push(Polygon { outer: simple, holes: Vec::new() });
        } else if keep_holes {
            holes.push(simple);
        }
    }
    for hole in holes {
        let owner = outers
            .iter_mut()
            .filter(|p| point_in_ring(hole[0], &p.outer))
            .min_by(|a, b| signed_area(&a.outer).partial_cmp(&signed_area(&b.outer)).unwrap());
        if let Some(p) = owner {
            p.holes.push(hole);
        }
    }
    outers
}

fn dominant_axis(polys: &[Polygon]) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for ring in polys.iter().flat_map(|p| std::iter::once(&p.outer).chain(&p.holes)) {
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let len = dx.hypot(dy);
            let angle = dy.atan2(dx);
            s += len * (4.0 * angle).sin();
            c += len * (4.0 * angle).cos();
        }
    }
    s.atan2(c) / 4.0
}

fn rectify(ring: &[[f64; 2]], axis: f64) -> Vec<[f64; 2]> {
    let (sn, cs) = axis.sin_cos();
    let local: Vec<[f64; 2]> = ring.iter().map(|p| [cs * p[0] + sn * p[1], -sn * p[0] + cs * p[1]]).collect();
    let n = local.len();
    let snap = WALL_SNAP_DEG.to_radians().sin();
    let class: Vec<u8> = (0..n)
        .map(|i| {
            let (a, b) = (local[i], local[(i + 1) % n]);
            let len = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-12);
            if ((b[1] - a[1]) / len).abs() < snap {
                1
            } else if ((b[0] - a[0]) / len).abs() < snap {
                2
            } else {
                0
            }
        })
        .collect();
    let mut target: Vec<Option<f64>> = vec![None; n];
    let start = (0..n).find(|&i| class[i] != class[(i + n - 1) % n]).unwrap_or(0);
    let mut k = 0;
    while k < n {
        let first = (start + k) % n;
        let mut run = vec![first];
        while k + 1 < n && class[(start + k + 1) % n] == class[first] {
            k += 1;
            run.push((start + k) % n);
        }
        k += 1;
        if class[first] == 0 {
            continue;
        }
        let axis_index = if class[first] == 1 { 1 } else { 0 };
        let (mut sum, mut weight) = (0.0, 0.0);
        for &e in &run {
            let (a, b) = (local[e], local[(e + 1) % n]);
            let len = (b[0] - a[0]).hypot(b[1] - a[1]);
            sum += len * (a[axis_index] + b[axis_index]) / 2.0;
            weight += len;
        }
        let value = if weight > 0.0 { sum / weight } else { local[first][axis_index] };
        for &e in &run {
            target[e] = Some(value);
        }
    }
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let (pe, ne) = ((i + n - 1) % n, i);
        let along = |c: u8| {
            let a = (class[pe] == c).then(|| target[pe]).flatten();
            let b = (class[ne] == c).then(|| target[ne]).flatten();
            match (a, b) {
                (Some(a), Some(b)) => Some((a + b) / 2.0),
                (Some(v), None) | (None, Some(v)) => Some(v),
                (None, None) => None,
            }
        };
        let x = along(2).unwrap_or(local[i][0]);
        let y = along(1).unwrap_or(local[i][1]);
        let p = [cs * x - sn * y, sn * x + cs * y];
        if out.last().is_none_or(|q: &[f64; 2]| (q[0] - p[0]).hypot(q[1] - p[1]) > 1e-3) {
            out.push(p);
        }
    }
    while out.len() > 1 && (out[0][0] - out[out.len() - 1][0]).hypot(out[0][1] - out[out.len() - 1][1]) <= 1e-3 {
        out.pop();
    }
    let mut cleaned = Vec::with_capacity(out.len());
    let m = out.len();
    for i in 0..m {
        let (a, b, c) = (out[(i + m - 1) % m], out[i], out[(i + 1) % m]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() > 1e-6 {
            cleaned.push(b);
        }
    }
    let original = signed_area(ring);
    if cleaned.len() < 3 || signed_area(&cleaned).signum() != original.signum() || signed_area(&cleaned).abs() < MIN_RING_M2 {
        return ring.to_vec();
    }
    cleaned
}

fn rectified(polys: Vec<Polygon>) -> Vec<Polygon> {
    let axis = dominant_axis(&polys);
    polys
        .into_iter()
        .map(|p| Polygon { outer: rectify(&p.outer, axis), holes: p.holes.iter().map(|h| rectify(h, axis)).collect() })
        .collect()
}

#[derive(Default)]
struct Geo {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    idx: Vec<u32>,
}

fn gl(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[2] as f32, -p[1] as f32]
}

impl Geo {
    fn vertex(&mut self, p: [f64; 3], n: [f64; 3]) -> u32 {
        self.pos.push(gl(p));
        self.nrm.push(gl(n));
        (self.pos.len() - 1) as u32
    }

    fn quad(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], n: [f64; 3]) {
        let i = [a, b, c, d].map(|p| self.vertex(p, n));
        self.idx.extend_from_slice(&[i[0], i[1], i[2], i[0], i[2], i[3]]);
    }

    fn cap(&mut self, poly: &Polygon, z: f64) {
        let mut flat: Vec<[f64; 2]> = poly.outer.clone();
        let mut hole_starts = Vec::new();
        for h in &poly.holes {
            hole_starts.push(flat.len() as u32);
            flat.extend_from_slice(h);
        }
        let mut tris: Vec<u32> = Vec::new();
        earcut::Earcut::new().earcut(flat.iter().copied(), &hole_starts, &mut tris);
        let base = self.pos.len() as u32;
        for p in &flat {
            self.vertex([p[0], p[1], z], [0.0, 0.0, 1.0]);
        }
        for t in tris.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| flat[i as usize]);
            let ccw = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0;
            if ccw {
                self.idx.extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
            } else {
                self.idx.extend_from_slice(&[base + t[0], base + t[2], base + t[1]]);
            }
        }
    }

    fn extrude_sides(&mut self, ring: &[[f64; 2]], z0: f64, z1: f64) {
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let len = (b[0] - a[0]).hypot(b[1] - a[1]);
            if len == 0.0 {
                continue;
            }
            let n = [(b[1] - a[1]) / len, -(b[0] - a[0]) / len, 0.0];
            self.quad([a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1], n);
        }
    }

    fn cuboid(&mut self, center: [f64; 3], half: [f64; 3]) {
        let [cx, cy, cz] = center;
        let [hx, hy, hz] = half;
        let p = |sx: f64, sy: f64, sz: f64| [cx + sx * hx, cy + sy * hy, cz + sz * hz];
        self.quad(p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.), [0., 0., 1.]);
        self.quad(p(-1., 1., -1.), p(1., 1., -1.), p(1., -1., -1.), p(-1., -1., -1.), [0., 0., -1.]);
        self.quad(p(1., -1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(1., -1., 1.), [1., 0., 0.]);
        self.quad(p(-1., 1., -1.), p(-1., -1., -1.), p(-1., -1., 1.), p(-1., 1., 1.), [-1., 0., 0.]);
        self.quad(p(1., 1., -1.), p(-1., 1., -1.), p(-1., 1., 1.), p(1., 1., 1.), [0., 1., 0.]);
        self.quad(p(-1., -1., -1.), p(1., -1., -1.), p(1., -1., 1.), p(-1., -1., 1.), [0., -1., 0.]);
    }

    fn cylinder(&mut self, center: [f64; 2], radius: f64, z0: f64, z1: f64, segments: usize) {
        let ring: Vec<[f64; 2]> = (0..segments)
            .map(|i| {
                let a = i as f64 / segments as f64 * std::f64::consts::TAU;
                [a.cos(), a.sin()]
            })
            .collect();
        let at = |d: [f64; 2], z: f64| [center[0] + radius * d[0], center[1] + radius * d[1], z];
        let base = self.pos.len() as u32;
        for d in &ring {
            self.vertex(at(*d, z0), [d[0], d[1], 0.0]);
            self.vertex(at(*d, z1), [d[0], d[1], 0.0]);
        }
        for i in 0..segments as u32 {
            let j = (i + 1) % segments as u32;
            let (a0, a1, b0, b1) = (base + 2 * i, base + 2 * i + 1, base + 2 * j, base + 2 * j + 1);
            self.idx.extend_from_slice(&[a0, b0, b1, a0, b1, a1]);
        }
        for (z, up) in [(z1, 1.0), (z0, -1.0)] {
            let c = self.vertex([center[0], center[1], z], [0.0, 0.0, up]);
            let rim: Vec<u32> = ring.iter().map(|d| self.vertex(at(*d, z), [0.0, 0.0, up])).collect();
            for i in 0..segments {
                let (a, b) = (rim[i], rim[(i + 1) % segments]);
                if up > 0.0 {
                    self.idx.extend_from_slice(&[c, a, b]);
                } else {
                    self.idx.extend_from_slice(&[c, b, a]);
                }
            }
        }
    }
}

struct Glb {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    materials: Vec<Value>,
    material_names: Vec<String>,
    meshes: Vec<Value>,
    nodes: Vec<Value>,
    triangles: usize,
}

impl Glb {
    fn new() -> Self {
        Glb {
            bin: Vec::new(),
            views: Vec::new(),
            accessors: Vec::new(),
            materials: Vec::new(),
            material_names: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            triangles: 0,
        }
    }

    fn view(&mut self, bytes: &[u8], target: u32) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        self.views.push(json!({"buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len(), "target": target}));
        self.bin.extend_from_slice(bytes);
        self.views.len() - 1
    }

    fn vec3(&mut self, data: &[[f32; 3]], bounds: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes())).collect();
        let view = self.view(&bytes, 34962);
        let mut acc = json!({"bufferView": view, "componentType": 5126, "count": data.len(), "type": "VEC3"});
        if bounds {
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for v in data {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            acc["min"] = json!(lo);
            acc["max"] = json!(hi);
        }
        self.accessors.push(acc);
        self.accessors.len() - 1
    }

    fn indices(&mut self, idx: &[u32]) -> usize {
        let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.view(&bytes, 34963);
        self.accessors.push(json!({"bufferView": view, "componentType": 5125, "count": idx.len(), "type": "SCALAR"}));
        self.accessors.len() - 1
    }

    fn material(&mut self, name: &str, color: [f64; 4], roughness: f64, metallic: f64) -> usize {
        if let Some(i) = self.material_names.iter().position(|n| n == name) {
            return i;
        }
        self.materials.push(json!({
            "name": name,
            "pbrMetallicRoughness": {"baseColorFactor": color, "roughnessFactor": roughness, "metallicFactor": metallic},
        }));
        self.material_names.push(name.to_string());
        self.materials.len() - 1
    }

    fn mesh(&mut self, name: &str, prims: Vec<(usize, Geo)>) -> Option<usize> {
        let mut out = Vec::new();
        for (material, geo) in prims {
            if geo.idx.is_empty() {
                continue;
            }
            self.triangles += geo.idx.len() / 3;
            let position = self.vec3(&geo.pos, true);
            let normal = self.vec3(&geo.nrm, false);
            let indices = self.indices(&geo.idx);
            out.push(json!({"attributes": {"POSITION": position, "NORMAL": normal}, "indices": indices, "material": material}));
        }
        if out.is_empty() {
            return None;
        }
        self.meshes.push(json!({"name": name, "primitives": out}));
        Some(self.meshes.len() - 1)
    }

    fn node(&mut self, node: Value) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn finish(self, roots: Vec<usize>) -> Vec<u8> {
        let doc = json!({
            "asset": {"version": "2.0", "generator": "anews-core scene"},
            "scene": 0,
            "scenes": [{"name": "venue", "nodes": roots}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": self.bin.len()}],
        });
        let mut json_bytes = serde_json::to_vec(&doc).unwrap();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let mut bin = self.bin;
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }
}

type Mat4 = [[f64; 4]; 4];

fn mul4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn node_matrix(node: &gltf::Node) -> Mat4 {
    let cols = node.transform().matrix();
    let mut m = [[0.0; 4]; 4];
    for (c, col) in cols.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            m[r][c] = *v as f64;
        }
    }
    m
}

struct VenueMaterials<'a> {
    venue: &'a Venue,
}

impl VenueMaterials<'_> {
    fn register(&self, glb: &mut Glb, name: &str) -> Result<usize, String> {
        let def = self.venue.materials.get(name).cloned().unwrap_or_else(|| default_material(name));
        let [r, g, b] = parse_color(&def.color)?;
        Ok(glb.material(name, [r, g, b, 1.0], def.roughness, def.metallic))
    }
}

fn import_model(glb: &mut Glb, mats: &VenueMaterials, name: &str, bytes: &[u8]) -> Result<usize, String> {
    let doc = gltf::Gltf::from_slice(bytes).map_err(|e| format!("{name}: {e}"))?;
    let blob = doc.blob.as_deref().ok_or_else(|| format!("{name}: not a .glb"))?;
    let scene = doc.default_scene().or_else(|| doc.scenes().next()).ok_or_else(|| format!("{name}: no scene"))?;
    let identity: Mat4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let mut by_material: BTreeMap<usize, Geo> = BTreeMap::new();
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, identity)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = mul4(&parent, &node_matrix(&node));
        for child in node.children() {
            stack.push((child, world));
        }
        let Some(mesh) = node.mesh() else { continue };
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let src = prim.material();
            let mat_name = src.name().map(str::to_string).unwrap_or_else(|| format!("{name}_material"));
            let material = if mats.venue.materials.contains_key(&mat_name) {
                mats.register(glb, &mat_name)?
            } else {
                let pbr = src.pbr_metallic_roughness();
                let c = pbr.base_color_factor();
                glb.material(&mat_name, [c[0] as f64, c[1] as f64, c[2] as f64, c[3] as f64], pbr.roughness_factor() as f64,
                             pbr.metallic_factor() as f64)
            };
            let reader = prim.reader(|b| match b.source() {
                gltf::buffer::Source::Bin => Some(blob),
                gltf::buffer::Source::Uri(_) => None,
            });
            let positions: Vec<[f32; 3]> = reader.read_positions().ok_or_else(|| format!("{name}: primitive without positions"))?.collect();
            let normals: Vec<[f32; 3]> = reader.read_normals().ok_or_else(|| format!("{name}: primitive without normals"))?.collect();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(ix) => ix.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let geo = by_material.entry(material).or_default();
            let base = geo.pos.len() as u32;
            for (p, n) in positions.iter().zip(&normals) {
                let tp: Vec<f64> = (0..3).map(|r| (0..3).map(|k| world[r][k] * p[k] as f64).sum::<f64>() + world[r][3]).collect();
                let tn: Vec<f64> = (0..3).map(|r| (0..3).map(|k| world[r][k] * n[k] as f64).sum::<f64>()).collect();
                let len = (tn[0] * tn[0] + tn[1] * tn[1] + tn[2] * tn[2]).sqrt().max(1e-12);
                geo.pos.push([tp[0] as f32, tp[1] as f32, tp[2] as f32]);
                geo.nrm.push([(tn[0] / len) as f32, (tn[1] / len) as f32, (tn[2] / len) as f32]);
            }
            geo.idx.extend(indices.iter().map(|i| base + i));
        }
    }
    glb.mesh(name, by_material.into_iter().collect()).ok_or_else(|| format!("{name}: no triangles"))
}

fn part_geometry(item: &Item, part: &Part, geo: &mut Geo) {
    let [width, depth] = item.size.unwrap_or([1.0, 1.0]);
    match part {
        Part::Box { size, z, scale_xy, offset, .. } => {
            let (sx, sy) = if *scale_xy { (size[0] * width, size[1] * depth) } else { (size[0], size[1]) };
            geo.cuboid([offset[0], offset[1], z + size[2] / 2.0], [sx / 2.0, sy / 2.0, size[2] / 2.0]);
        }
        Part::Cylinder { radius, z0, z1, scale_xy, offset, .. } => {
            let r = if *scale_xy { radius * width } else { *radius };
            geo.cylinder(*offset, r, *z0, *z1, CYLINDER_SEGMENTS);
        }
        Part::Leg4 { radius, inset, z0, z1, offset, .. } => {
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    let c = [sx * (width / 2.0 - inset) + offset[0], sy * (depth / 2.0 - inset) + offset[1]];
                    geo.cylinder(c, *radius, *z0, *z1, LEG_SEGMENTS);
                }
            }
        }
    }
}

fn part_material(part: &Part) -> &str {
    match part {
        Part::Box { material, .. } | Part::Cylinder { material, .. } | Part::Leg4 { material, .. } => {
            material.as_deref().unwrap_or(FURNITURE_MATERIAL)
        }
    }
}

fn wall_mask(venue: &Venue, scene: Option<&str>, map: &OccupancyMap) -> Mask {
    let grid = WallGrid::from_map(map).for_scene(venue, scene);
    let mask = Mask { w: map.width, h: map.height, cells: grid.occupied };
    let min_cells = (MIN_WALL_BLOB_M2 / (map.meta.resolution * map.meta.resolution)).ceil() as usize;
    let kept: Vec<usize> = mask.components().into_iter().filter(|c| c.len() >= min_cells).flatten().collect();
    mask.only(kept).morph(1, true)
}

fn floor_mask(map: &OccupancyMap) -> Mask {
    let known: Vec<bool> = map.occupied.iter().zip(&map.free).map(|(o, f)| *o || *f).collect();
    let closed = Mask { w: map.width, h: map.height, cells: known }.morph(3, true).morph(3, false).morph(6, false).morph(6, true);
    let largest = closed.components().into_iter().max_by_key(|c| c.len()).unwrap_or_default();
    closed.only(largest)
}

pub fn build_scene_glb(
    venue: &Venue,
    scene: Option<&str>,
    map: Option<&OccupancyMap>,
    models: &BTreeMap<String, Vec<u8>>,
) -> Result<(Vec<u8>, SceneStats), String> {
    let mats = VenueMaterials { venue };
    let mut glb = Glb::new();
    let mut roots = Vec::new();
    let mut stats = SceneStats {
        wall_outlines: 0,
        wall_holes: 0,
        floor: false,
        items: 0,
        modelled_items: 0,
        triangles: 0,
        materials: Vec::new(),
    };

    if let Some(map) = map {
        let origin = [map.meta.origin[0], map.meta.origin[1]];
        let res = map.meta.resolution;
        let floor_mat = mats.register(&mut glb, FLOOR_MATERIAL)?;
        let mut floor = Geo::default();
        for poly in polygons(&floor_mask(map), origin, res, FLOOR_SIMPLIFY_M, false) {
            floor.cap(&poly, 0.0);
        }
        if let Some(mesh) = glb.mesh("floor", vec![(floor_mat, floor)]) {
            stats.floor = true;
            roots.push(glb.node(json!({"name": "floor", "mesh": mesh, "extras": {"kind": "floor"}})));
        }
        if venue.walls.from_map {
            let wall_mat = mats.register(&mut glb, WALL_MATERIAL)?;
            let mut walls = Geo::default();
            for poly in rectified(polygons(&wall_mask(venue, scene, map), origin, res, WALL_SIMPLIFY_M, true)) {
                stats.wall_outlines += 1;
                stats.wall_holes += poly.holes.len();
                walls.extrude_sides(&poly.outer, 0.0, venue.walls.height);
                for h in &poly.holes {
                    walls.extrude_sides(h, 0.0, venue.walls.height);
                }
                walls.cap(&poly, venue.walls.height);
            }
            if let Some(mesh) = glb.mesh("walls", vec![(wall_mat, walls)]) {
                roots.push(glb.node(json!({"name": "walls", "mesh": mesh, "extras": {"kind": "walls"}})));
            }
        }
    }

    let mut model_meshes: BTreeMap<String, usize> = BTreeMap::new();
    for (type_name, item_type) in &venue.types {
        if let Some(model) = &item_type.model
            && let Some(bytes) = models.get(&model.file)
        {
            model_meshes.insert(type_name.clone(), import_model(&mut glb, &mats, type_name, bytes)?);
        }
    }

    for item in venue.scene_items(scene)? {
        let item_type = &venue.types[&item.type_name];
        let yaw = item.yaw_deg.to_radians();
        let mut node = json!({
            "name": item.id,
            "translation": [item.x, 0.0, -item.y],
            "rotation": [0.0, (yaw / 2.0).sin(), 0.0, (yaw / 2.0).cos()],
            "extras": {"kind": item.kind, "type": item.type_name, "number": item.number},
        });
        if let Some(&mesh) = model_meshes.get(&item.type_name) {
            node["mesh"] = json!(mesh);
            if item_type.model.as_ref().is_some_and(|m| m.scale_xy) {
                let [w, d] = item.size.unwrap_or([1.0, 1.0]);
                node["scale"] = json!([w, 1.0, d]);
            }
            stats.modelled_items += 1;
        } else {
            let mut by_material: BTreeMap<usize, Geo> = BTreeMap::new();
            for part in &item_type.parts {
                let material = mats.register(&mut glb, part_material(part))?;
                part_geometry(&item, part, by_material.entry(material).or_default());
            }
            if let Some(mesh) = glb.mesh(&item.id, by_material.into_iter().collect()) {
                node["mesh"] = json!(mesh);
            }
        }
        stats.items += 1;
        roots.push(glb.node(node));
    }

    stats.triangles = glb.triangles;
    stats.materials = glb.material_names.clone();
    Ok((glb.finish(roots), stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_mask(w: usize, h: usize, c0: usize, c1: usize, r0: usize, r1: usize) -> Mask {
        let mut cells = vec![false; w * h];
        for r in r0..r1 {
            for c in c0..c1 {
                cells[r * w + c] = true;
            }
        }
        Mask { w, h, cells }
    }

    #[test]
    fn traced_block_becomes_one_counter_clockwise_rectangle() {
        let polys = polygons(&square_mask(10, 10, 2, 6, 3, 5), [0.0, 0.0], 1.0, 0.1, true);
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0].outer.len(), 4);
        assert!((signed_area(&polys[0].outer) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn ring_with_hole_keeps_the_hole() {
        let mut mask = square_mask(12, 12, 1, 11, 1, 11);
        for r in 4..8 {
            for c in 4..8 {
                mask.cells[r * 12 + c] = false;
            }
        }
        let polys = polygons(&mask, [0.0, 0.0], 1.0, 0.1, true);
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0].holes.len(), 1);
        assert!((signed_area(&polys[0].holes[0]) + 16.0).abs() < 1e-9);
    }

    #[test]
    fn staircase_diagonal_simplifies_to_a_straight_edge() {
        let n = 40;
        let mut cells = vec![false; n * n];
        for r in 0..n {
            for c in 0..=r {
                cells[r * n + c] = true;
            }
        }
        let polys = polygons(&Mask { w: n, h: n, cells }, [0.0, 0.0], 0.05, 0.07, true);
        assert_eq!(polys.len(), 1);
        assert!(polys[0].outer.len() <= 4, "{:?}", polys[0].outer);
    }

    #[test]
    fn rotated_noisy_rectangle_is_squared_to_the_dominant_axis() {
        let axis = 0.6f64;
        let (s, c) = axis.sin_cos();
        let local = [[0.0, 0.0], [2.0, 0.03], [4.0, -0.02], [4.01, 1.0], [3.98, 2.0], [2.0, 2.04], [0.02, 1.97], [-0.03, 1.0]];
        let ring: Vec<[f64; 2]> = local.iter().map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1]]).collect();
        let squared = rectified(vec![Polygon { outer: ring, holes: Vec::new() }]);
        let outer = &squared[0].outer;
        assert_eq!(outer.len(), 4, "{outer:?}");
        for i in 0..4 {
            let (a, b, d) = (outer[i], outer[(i + 1) % 4], outer[(i + 2) % 4]);
            let dot = (b[0] - a[0]) * (d[0] - b[0]) + (b[1] - a[1]) * (d[1] - b[1]);
            assert!(dot.abs() < 1e-9, "corner {i} is not square");
        }
    }

    #[test]
    fn glb_round_trips_through_the_gltf_reader() {
        let venue = crate::venue::parse_venue(
            "schema: foodlink.venue/1\ntypes:\n  t:\n    parts:\n    - {shape: box, size: [1, 1, 0.04], z: 0.71, scale_xy: true, material: wood}\n    - {shape: leg4, radius: 0.02, inset: 0.05, z0: 0, z1: 0.71}\nmaterials:\n  wood: {color: '#8a5a3b'}\nitems:\n- {id: a, type: t, x: 1, y: 2, yaw_deg: 90, size: [1.2, 0.8]}\n",
        )
        .unwrap();
        let (bytes, stats) = build_scene_glb(&venue, None, None, &BTreeMap::new()).unwrap();
        assert_eq!(stats.items, 1);
        let doc = gltf::Gltf::from_slice(&bytes).unwrap();
        let node = doc.nodes().find(|n| n.name() == Some("a")).unwrap();
        let names: Vec<_> = doc.materials().filter_map(|m| m.name().map(str::to_string)).collect();
        assert_eq!(names, vec!["wood", "furniture"]);
        let tris = crate::mesh::glb_triangles(&bytes, 0).unwrap();
        let top = tris.iter().map(|t| t.a[2].max(t.b[2]).max(t.c[2])).fold(0.0, f64::max);
        assert!((top - 0.75).abs() < 1e-5);
        let xs: Vec<f64> = tris.iter().flat_map(|t| [t.a[0], t.b[0], t.c[0]]).collect();
        let span = xs.iter().cloned().fold(f64::MIN, f64::max) - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!((span - 0.8).abs() < 1e-5, "rotated 90 degrees, so the 0.8 m depth spans x; got {span}");
        assert!(node.mesh().is_some());
    }
}
