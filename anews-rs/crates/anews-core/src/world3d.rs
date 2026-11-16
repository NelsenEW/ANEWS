use crate::venue::{Item, Solid, Venue, scene_solids};

pub type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub a: V3,
    pub b: V3,
    pub c: V3,
    pub owner: u32,
}

impl Triangle {
    pub fn hit(&self, o: V3, d: V3) -> Option<f64> {
        let e1 = sub(self.b, self.a);
        let e2 = sub(self.c, self.a);
        let p = cross(d, e2);
        let det = dot(e1, p);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let s = sub(o, self.a);
        let u = dot(s, p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = cross(s, e1);
        let v = dot(d, q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = dot(e2, q) * inv;
        (t > 1e-9).then_some(t)
    }

    fn bounds(&self) -> Aabb {
        let mut b = Aabb::empty();
        for p in [self.a, self.b, self.c] {
            b.grow(p);
        }
        b
    }
}

#[derive(Debug, Clone, Copy)]
struct Aabb {
    lo: V3,
    hi: V3,
}

impl Aabb {
    fn empty() -> Self {
        Aabb { lo: [f64::INFINITY; 3], hi: [f64::NEG_INFINITY; 3] }
    }

    fn grow(&mut self, p: V3) {
        for k in 0..3 {
            self.lo[k] = self.lo[k].min(p[k]);
            self.hi[k] = self.hi[k].max(p[k]);
        }
    }

    fn merge(&mut self, o: &Aabb) {
        self.grow(o.lo);
        self.grow(o.hi);
    }

    fn entry(&self, o: V3, inv: V3, t_max: f64) -> Option<f64> {
        let mut t0 = 0.0_f64;
        let mut t1 = t_max;
        for k in 0..3 {
            let a = (self.lo[k] - o[k]) * inv[k];
            let b = (self.hi[k] - o[k]) * inv[k];
            let (near, far) = if a < b { (a, b) } else { (b, a) };
            if near.is_nan() || far.is_nan() {
                if o[k] < self.lo[k] || o[k] > self.hi[k] {
                    return None;
                }
                continue;
            }
            t0 = t0.max(near);
            t1 = t1.min(far);
            if t0 > t1 {
                return None;
            }
        }
        Some(t0)
    }
}

enum Node {
    Leaf { bounds: Aabb, start: usize, count: usize },
    Inner { bounds: Aabb, left: usize, right: usize },
}

impl Node {
    fn bounds(&self) -> &Aabb {
        match self {
            Node::Leaf { bounds, .. } | Node::Inner { bounds, .. } => bounds,
        }
    }
}

pub struct Bvh {
    tris: Vec<Triangle>,
    nodes: Vec<Node>,
}

const LEAF_SIZE: usize = 4;

impl Bvh {
    pub fn new(mut tris: Vec<Triangle>) -> Self {
        let mut nodes = Vec::new();
        if !tris.is_empty() {
            let n = tris.len();
            build(&mut tris, 0, n, &mut nodes);
        }
        Bvh { tris, nodes }
    }

    pub fn len(&self) -> usize {
        self.tris.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    pub fn triangles(&self) -> &[Triangle] {
        &self.tris
    }

    pub fn cast(&self, o: V3, d: V3, t_max: f64) -> Option<(f64, u32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let mut best: Option<(f64, u32)> = None;
        let mut limit = t_max;
        let mut stack = vec![self.nodes.len() - 1];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if node.bounds().entry(o, inv, limit).is_none() {
                continue;
            }
            match node {
                Node::Leaf { start, count, .. } => {
                    for tri in &self.tris[*start..*start + *count] {
                        if let Some(t) = tri.hit(o, d) {
                            if t < limit {
                                limit = t;
                                best = Some((t, tri.owner));
                            }
                        }
                    }
                }
                Node::Inner { left, right, .. } => {
                    let tl = self.nodes[*left].bounds().entry(o, inv, limit);
                    let tr = self.nodes[*right].bounds().entry(o, inv, limit);
                    match (tl, tr) {
                        (Some(a), Some(b)) if a <= b => {
                            stack.push(*right);
                            stack.push(*left);
                        }
                        (Some(_), Some(_)) => {
                            stack.push(*left);
                            stack.push(*right);
                        }
                        (Some(_), None) => stack.push(*left),
                        (None, Some(_)) => stack.push(*right),
                        (None, None) => {}
                    }
                }
            }
        }
        best
    }
}

fn build(tris: &mut [Triangle], start: usize, end: usize, nodes: &mut Vec<Node>) -> usize {
    let slice = &mut tris[start..end];
    let mut bounds = Aabb::empty();
    let mut centroids = Aabb::empty();
    for t in slice.iter() {
        bounds.merge(&t.bounds());
        centroids.grow(centroid(t));
    }
    if slice.len() <= LEAF_SIZE {
        nodes.push(Node::Leaf { bounds, start, count: slice.len() });
        return nodes.len() - 1;
    }
    let extent = sub(centroids.hi, centroids.lo);
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    let mid = slice.len() / 2;
    slice.select_nth_unstable_by(mid, |a, b| centroid(a)[axis].total_cmp(&centroid(b)[axis]));
    let left = build(tris, start, start + mid, nodes);
    let right = build(tris, start + mid, end, nodes);
    nodes.push(Node::Inner { bounds, left, right });
    nodes.len() - 1
}

fn centroid(t: &Triangle) -> V3 {
    [(t.a[0] + t.b[0] + t.c[0]) / 3.0, (t.a[1] + t.b[1] + t.c[1]) / 3.0, (t.a[2] + t.b[2] + t.c[2]) / 3.0]
}

const CYLINDER_SEGMENTS: usize = 16;

fn push_quad(out: &mut Vec<Triangle>, p: [V3; 4], owner: u32) {
    out.push(Triangle { a: p[0], b: p[1], c: p[2], owner });
    out.push(Triangle { a: p[0], b: p[2], c: p[3], owner });
}

pub fn box_triangles(center: V3, half: V3, yaw: f64, owner: u32, out: &mut Vec<Triangle>) {
    let (s, c) = yaw.sin_cos();
    let corner = |u: f64, v: f64, w: f64| -> V3 {
        let (x, y) = (u * half[0], v * half[1]);
        [center[0] + c * x - s * y, center[1] + s * x + c * y, center[2] + w * half[2]]
    };
    let p = |i: usize| corner(if i & 1 == 0 { -1.0 } else { 1.0 }, if i & 2 == 0 { -1.0 } else { 1.0 }, if i & 4 == 0 { -1.0 } else { 1.0 });
    for face in [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]] {
        push_quad(out, [p(face[0]), p(face[1]), p(face[2]), p(face[3])], owner);
    }
}

pub fn cylinder_triangles(center: [f64; 2], radius: f64, z0: f64, z1: f64, owner: u32, out: &mut Vec<Triangle>) {
    let ring = |k: usize, z: f64| -> V3 {
        let a = std::f64::consts::TAU * k as f64 / CYLINDER_SEGMENTS as f64;
        [center[0] + radius * a.cos(), center[1] + radius * a.sin(), z]
    };
    for k in 0..CYLINDER_SEGMENTS {
        let k1 = (k + 1) % CYLINDER_SEGMENTS;
        push_quad(out, [ring(k, z0), ring(k1, z0), ring(k1, z1), ring(k, z1)], owner);
        out.push(Triangle { a: [center[0], center[1], z1], b: ring(k, z1), c: ring(k1, z1), owner });
        out.push(Triangle { a: [center[0], center[1], z0], b: ring(k1, z0), c: ring(k, z0), owner });
    }
}

pub struct WallGrid {
    pub occupied: Vec<bool>,
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin: [f64; 2],
}

impl WallGrid {
    pub fn from_map(map: &crate::occupancy::OccupancyMap) -> Self {
        WallGrid {
            occupied: map.occupied.clone(),
            width: map.width,
            height: map.height,
            resolution: map.meta.resolution,
            origin: [map.meta.origin[0], map.meta.origin[1]],
        }
    }
}

impl WallGrid {
    fn cell_center(&self, col: usize, row: usize) -> [f64; 2] {
        [self.origin[0] + (col as f64 + 0.5) * self.resolution, self.origin[1] + (row as f64 + 0.5) * self.resolution]
    }

    fn footprint_mask(&self, items: &[Item], grow: f64) -> Vec<bool> {
        let mut mask = vec![false; self.occupied.len()];
        for item in items {
            let [w, d] = item.size.unwrap_or([0.5, 0.5]);
            let (hw, hd) = (w / 2.0 + grow, d / 2.0 + grow);
            if hw <= 0.0 || hd <= 0.0 {
                continue;
            }
            let (s, c) = item.yaw_deg.to_radians().sin_cos();
            let reach = hw.hypot(hd);
            let to_cell = |v: f64, o: f64| ((v - o) / self.resolution).floor();
            let c0 = to_cell(item.x - reach, self.origin[0]).max(0.0) as usize;
            let c1 = (to_cell(item.x + reach, self.origin[0]) + 1.0).clamp(0.0, self.width as f64) as usize;
            let r0 = to_cell(item.y - reach, self.origin[1]).max(0.0) as usize;
            let r1 = (to_cell(item.y + reach, self.origin[1]) + 1.0).clamp(0.0, self.height as f64) as usize;
            for row in r0..r1 {
                for col in c0..c1 {
                    let [wx, wy] = self.cell_center(col, row);
                    let (dx, dy) = (wx - item.x, wy - item.y);
                    if (c * dx + s * dy).abs() <= hw && (-s * dx + c * dy).abs() <= hd {
                        mask[row * self.width + col] = true;
                    }
                }
            }
        }
        mask
    }

    pub fn clear_items(&mut self, items: &[Item], margin: f64) {
        let near = self.footprint_mask(items, margin);
        let core = self.footprint_mask(items, -self.resolution);
        let mut seen = vec![false; self.occupied.len()];
        for start in 0..self.occupied.len() {
            if !self.occupied[start] || seen[start] {
                continue;
            }
            seen[start] = true;
            let mut stack = vec![start];
            let mut blob = Vec::new();
            while let Some(i) = stack.pop() {
                blob.push(i);
                let (col, row) = (i % self.width, i / self.width);
                let neighbours = [
                    (col > 0).then(|| i - 1),
                    (col + 1 < self.width).then(|| i + 1),
                    (row > 0).then(|| i - self.width),
                    (row + 1 < self.height).then(|| i + self.width),
                ];
                for j in neighbours.into_iter().flatten() {
                    if self.occupied[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            let inside = blob.iter().filter(|&&i| near[i]).count();
            let furniture_blob = inside * 10 >= blob.len() * 7;
            for i in blob {
                if furniture_blob || core[i] {
                    self.occupied[i] = false;
                }
            }
        }
    }

    pub fn clear_points(&mut self, points: &[[f64; 2]]) {
        for p in points {
            let col = ((p[0] - self.origin[0]) / self.resolution).floor();
            let row = ((p[1] - self.origin[1]) / self.resolution).floor();
            if col >= 0.0 && row >= 0.0 && (col as usize) < self.width && (row as usize) < self.height {
                self.occupied[row as usize * self.width + col as usize] = false;
            }
        }
    }

    pub fn for_scene(mut self, venue: &Venue, scene: Option<&str>) -> Self {
        if venue.walls.clear_items {
            self.clear_items(&venue.items, venue.walls.clear_margin);
        }
        self.clear_points(venue.scene_wall_clears(scene));
        self
    }

    pub fn triangles(&self, wall_height: f64, owner: u32, out: &mut Vec<Triangle>) {
        let half_z = wall_height / 2.0;
        for row in 0..self.height {
            let mut col = 0;
            while col < self.width {
                if !self.occupied[row * self.width + col] {
                    col += 1;
                    continue;
                }
                let start = col;
                while col < self.width && self.occupied[row * self.width + col] {
                    col += 1;
                }
                let [x0, y0] = self.cell_center(start, row);
                let [x1, _] = self.cell_center(col - 1, row);
                let center = [(x0 + x1) / 2.0, y0, half_z];
                let half = [(x1 - x0) / 2.0 + self.resolution / 2.0, self.resolution / 2.0, half_z];
                box_triangles(center, half, 0.0, owner, out);
            }
        }
    }
}

pub struct World3d {
    pub bvh: Bvh,
    pub floor: bool,
    pub owners: Vec<String>,
    pub crowd: Option<crate::crowd::CrowdGrid>,
    pub actors: Option<crate::actors::ActorSet>,
}

pub const WALL_OWNER: &str = "walls";

impl World3d {
    pub fn from_venue(venue: &Venue, scene: Option<&str>, walls: Option<WallGrid>) -> Result<Self, String> {
        let mut owners = vec![WALL_OWNER.to_string()];
        let mut tris = Vec::new();
        if venue.walls.from_map {
            let grid = walls.ok_or("venue walls.from_map needs the map")?.for_scene(venue, scene);
            grid.triangles(venue.walls.height, 0, &mut tris);
        }
        for (id, solid) in scene_solids(venue, scene)? {
            let owner = match owners.iter().position(|o| o == &id) {
                Some(i) => i as u32,
                None => {
                    owners.push(id);
                    (owners.len() - 1) as u32
                }
            };
            match solid {
                Solid::Box { center, half, yaw } => box_triangles(center, half, yaw, owner, &mut tris),
                Solid::Cylinder { center, radius, z0, z1 } => cylinder_triangles(center, radius, z0, z1, owner, &mut tris),
            }
        }
        Ok(World3d { bvh: Bvh::new(tris), floor: true, owners, crowd: None, actors: None })
    }

    pub fn from_triangles(tris: Vec<Triangle>, floor: bool) -> Self {
        World3d { bvh: Bvh::new(tris), floor, owners: vec![WALL_OWNER.to_string()], crowd: None, actors: None }
    }

    pub fn walls_only(grid: &WallGrid, wall_height: f64, floor: bool) -> Self {
        let mut tris = Vec::new();
        grid.triangles(wall_height, 0, &mut tris);
        Self::from_triangles(tris, floor)
    }

    pub fn add_triangles(&mut self, extra: Vec<Triangle>) {
        let mut tris = std::mem::replace(&mut self.bvh, Bvh::new(Vec::new())).tris;
        tris.extend(extra);
        self.bvh = Bvh::new(tris);
    }

    pub fn dynamic_cast(&self, o: V3, d: V3, t_max: f64) -> f64 {
        let crowd = self.crowd.as_ref().map_or(f64::INFINITY, |c| c.cast(o, d, t_max));
        let actors = self.actors.as_ref().map_or(f64::INFINITY, |a| a.cast(o, d, t_max.min(crowd)));
        crowd.min(actors)
    }

    pub fn cast(&self, o: V3, d: V3, t_max: f64) -> f64 {
        let n = dot(d, d).sqrt();
        let d = [d[0] / n, d[1] / n, d[2] / n];
        let mut best = t_max;
        let mut hit = false;
        if self.floor && d[2] < -1e-12 {
            let t = -o[2] / d[2];
            if t > 0.0 && t < best {
                best = t;
                hit = true;
            }
        }
        if let Some((t, _)) = self.bvh.cast(o, d, best) {
            best = t;
            hit = true;
        }
        let t = self.dynamic_cast(o, d, best);
        if t < best {
            best = t;
            hit = true;
        }
        if hit { best } else { f64::INFINITY }
    }
}

#[cfg(test)]
mod clear_items_tests {
    use super::*;

    fn booth(x: f64, y: f64) -> Item {
        serde_yaml::from_str(&format!("{{id: b, type: booth, x: {x}, y: {y}, yaw_deg: 0, size: [1.2, 0.6]}}")).unwrap()
    }

    #[test]
    fn wall_behind_a_booth_survives_and_a_leg_blob_inside_it_is_removed() {
        let (w, h) = (60, 40);
        let mut grid = WallGrid { occupied: vec![false; w * h], width: w, height: h, resolution: 0.05, origin: [0.0, 0.0] };
        for col in 0..w {
            grid.occupied[10 * w + col] = true;
        }
        let leg = 20 * w + 30;
        grid.occupied[leg] = true;
        grid.clear_items(&[booth(1.5, 0.84)], 0.15);
        assert!((0..w).all(|col| grid.occupied[10 * w + col]), "wall row was cut");
        assert!(!grid.occupied[leg], "leg blob inside the booth was kept");
    }
}
