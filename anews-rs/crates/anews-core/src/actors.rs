use crate::venue::{expand_item, Item, ItemType, Solid};
use crate::world3d::V3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const EPS: f64 = 1e-6;
pub const LEG_SAMPLE_S: f64 = 0.15;
const LEG_TOP_BELOW: f64 = 0.1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Actor {
    pub id: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub samples: Vec<[f64; 4]>,
    pub end: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub legs: Vec<[f64; 5]>,
}

impl Actor {
    pub fn start(&self) -> f64 {
        self.samples.first().map_or(f64::INFINITY, |s| s[0])
    }

    pub fn legs_at(&self, t: f64, within: f64) -> Option<[[f64; 2]; 2]> {
        let k = self.legs.partition_point(|l| l[0] < t);
        [k.checked_sub(1), Some(k)]
            .into_iter()
            .flatten()
            .filter_map(|i| self.legs.get(i))
            .filter(|l| (l[0] - t).abs() <= within)
            .min_by(|a, b| (a[0] - t).abs().total_cmp(&(b[0] - t).abs()))
            .map(|l| [[l[1], l[2]], [l[3], l[4]]])
    }

    pub fn pose_at(&self, t: f64) -> Option<[f64; 3]> {
        if self.samples.is_empty() || t < self.start() || t > self.end {
            return None;
        }
        let k = self.samples.partition_point(|s| s[0] <= t);
        if k == 0 {
            let s = self.samples[0];
            return Some([s[1], s[2], s[3]]);
        }
        if k == self.samples.len() {
            let s = self.samples[k - 1];
            return Some([s[1], s[2], s[3]]);
        }
        let (a, b) = (self.samples[k - 1], self.samples[k]);
        let u = if b[0] - a[0] > EPS { (t - a[0]) / (b[0] - a[0]) } else { 0.0 };
        let turn = (b[3] - a[3] + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        Some([a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * turn])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorTimeline {
    pub start: f64,
    pub end: f64,
    pub types: BTreeMap<String, ItemType>,
    pub actors: Vec<Actor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ActorPose {
    pub index: usize,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

impl ActorTimeline {
    pub fn poses(&self, t: f64) -> Vec<ActorPose> {
        self.actors
            .iter()
            .enumerate()
            .filter_map(|(index, a)| a.pose_at(t).map(|[x, y, yaw]| ActorPose { index, x, y, yaw }))
            .collect()
    }

    pub fn solids(&self, t: f64) -> Vec<Solid> {
        let mut out = Vec::new();
        for p in self.poses(t) {
            let actor = &self.actors[p.index];
            let Some(item_type) = self.types.get(&actor.type_name) else { continue };
            let item = Item {
                id: actor.id.clone(),
                kind: "person".into(),
                type_name: actor.type_name.clone(),
                x: p.x,
                y: p.y,
                yaw_deg: p.yaw.to_degrees(),
                size: None,
                number: None,
                mobility: Some("dynamic".into()),
                extra: BTreeMap::new(),
            };
            let solids = expand_item(&item, item_type);
            match actor.legs_at(t, LEG_SAMPLE_S) {
                Some(legs) => {
                    let mut k = 0;
                    for solid in solids {
                        match solid {
                            Solid::Cylinder { radius, z0, z1, .. } if z0 < LEG_TOP_BELOW && k < 2 => {
                                out.push(Solid::Cylinder { center: legs[k], radius, z0, z1 });
                                k += 1;
                            }
                            other => out.push(other),
                        }
                    }
                }
                None => out.extend(solids),
            }
        }
        out
    }

    pub fn apply(&self, t: f64, set: &mut ActorSet) {
        set.solids = self.solids(t);
    }
}

#[derive(Debug, Clone, Default)]
pub struct ActorSet {
    pub solids: Vec<Solid>,
}

fn cylinder_entry(o: V3, d: V3, center: [f64; 2], radius: f64, z0: f64, z1: f64) -> Option<f64> {
    let (px, py) = (o[0] - center[0], o[1] - center[1]);
    let a = d[0] * d[0] + d[1] * d[1];
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    if a < 1e-12 {
        if px * px + py * py > radius * radius {
            return None;
        }
    } else {
        let b = 2.0 * (px * d[0] + py * d[1]);
        let c = px * px + py * py - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let q = disc.sqrt();
        lo = (-b - q) / (2.0 * a);
        hi = (-b + q) / (2.0 * a);
    }
    if d[2].abs() < 1e-12 {
        if o[2] < z0 || o[2] > z1 {
            return None;
        }
    } else {
        let (ta, tb) = ((z0 - o[2]) / d[2], (z1 - o[2]) / d[2]);
        lo = lo.max(ta.min(tb));
        hi = hi.min(ta.max(tb));
    }
    (lo <= hi && lo > EPS).then_some(lo)
}

fn box_entry(o: V3, d: V3, center: V3, half: V3, yaw: f64) -> Option<f64> {
    let (s, c) = yaw.sin_cos();
    let (px, py) = (o[0] - center[0], o[1] - center[1]);
    let lo_o = [c * px + s * py, -s * px + c * py, o[2] - center[2]];
    let lo_d = [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]];
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        if lo_d[k].abs() < 1e-12 {
            if lo_o[k].abs() > half[k] {
                return None;
            }
            continue;
        }
        let (a, b) = ((-half[k] - lo_o[k]) / lo_d[k], (half[k] - lo_o[k]) / lo_d[k]);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t0 <= t1 && t0 > EPS).then_some(t0)
}

pub fn solid_entry(o: V3, d: V3, solid: &Solid) -> Option<f64> {
    match *solid {
        Solid::Cylinder { center, radius, z0, z1 } => cylinder_entry(o, d, center, radius, z0, z1),
        Solid::Box { center, half, yaw } => box_entry(o, d, center, half, yaw),
    }
}

fn rect_touches_circle(x: f64, y: f64, yaw: f64, half: [f64; 2], center: [f64; 2], radius: f64) -> bool {
    let (s, c) = yaw.sin_cos();
    let (dx, dy) = (center[0] - x, center[1] - y);
    let (lx, ly) = (c * dx + s * dy, -s * dx + c * dy);
    let (qx, qy) = (lx.clamp(-half[0], half[0]), ly.clamp(-half[1], half[1]));
    (lx - qx).hypot(ly - qy) <= radius
}

fn rect_corners(x: f64, y: f64, yaw: f64, half: [f64; 2]) -> [[f64; 2]; 4] {
    let (s, c) = yaw.sin_cos();
    let at = |u: f64, v: f64| [x + c * u * half[0] - s * v * half[1], y + s * u * half[0] + c * v * half[1]];
    [at(-1.0, -1.0), at(1.0, -1.0), at(1.0, 1.0), at(-1.0, 1.0)]
}

fn rects_overlap(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> bool {
    for poly in [&a, &b] {
        for k in 0..4 {
            let (p, q) = (poly[k], poly[(k + 1) % 4]);
            let n = [q[1] - p[1], p[0] - q[0]];
            let proj = |r: &[[f64; 2]; 4]| {
                let v: Vec<f64> = r.iter().map(|c| c[0] * n[0] + c[1] * n[1]).collect();
                (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            };
            let ((a0, a1), (b0, b1)) = (proj(&a), proj(&b));
            if a1 < b0 || b1 < a0 {
                return false;
            }
        }
    }
    true
}

impl ActorSet {
    pub fn cast(&self, o: V3, d: V3, t_max: f64) -> f64 {
        self.solids.iter().filter_map(|s| solid_entry(o, d, s)).filter(|t| *t < t_max).fold(f64::INFINITY, f64::min)
    }

    pub fn touches(&self, x: f64, y: f64, yaw: f64, half: [f64; 2], height: f64) -> bool {
        self.solids.iter().any(|s| match *s {
            Solid::Cylinder { center, radius, z0, .. } => z0 < height && rect_touches_circle(x, y, yaw, half, center, radius),
            Solid::Box { center, half: h, yaw: by } => {
                center[2] - h[2] < height && rects_overlap(rect_corners(x, y, yaw, half), rect_corners(center[0], center[1], by, [h[0], h[1]]))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walker() -> ActorTimeline {
        let ty: ItemType = serde_yaml::from_str(
            "parts:\n- {shape: cylinder, radius: 0.08, z0: 0.0, z1: 0.85, offset: [0.0, 0.1]}\n- {shape: cylinder, radius: 0.08, z0: 0.0, z1: 0.85, offset: [0.0, -0.1]}\n- {shape: cylinder, radius: 0.2, z0: 0.85, z1: 1.5}\n",
        )
        .unwrap();
        ActorTimeline {
            start: 0.0,
            end: 10.0,
            types: BTreeMap::from([("person_standing".to_string(), ty)]),
            actors: vec![Actor { id: "p1".into(), type_name: "person_standing".into(), samples: vec![[2.0, 1.0, 0.0, 0.0], [4.0, 3.0, 0.0, 0.0]], end: 6.0, legs: Vec::new() }],
        }
    }

    #[test]
    fn an_actor_walks_between_samples_and_is_absent_outside_its_span() {
        let tl = walker();
        assert!(tl.poses(1.0).is_empty());
        let p = tl.poses(3.0)[0];
        assert!((p.x - 2.0).abs() < 1e-9 && p.y.abs() < 1e-9);
        assert!((tl.poses(5.5)[0].x - 3.0).abs() < 1e-9, "after the last sample the actor stays where it was last seen");
        assert!(tl.poses(6.5).is_empty());
    }

    #[test]
    fn lidar_height_rays_hit_the_legs_and_pass_between_them_and_the_camera_sees_the_body() {
        let tl = walker();
        let mut set = ActorSet::default();
        tl.apply(3.0, &mut set);
        let leg = set.cast([0.0, 0.1, 0.23], [1.0, 0.0, 0.0], 10.0);
        assert!((leg - 1.92).abs() < 1e-6, "leg front at {leg}");
        assert!(set.cast([0.0, 0.0, 0.23], [1.0, 0.0, 0.0], 10.0).is_infinite(), "the gap between the legs is open at shin height");
        let body = set.cast([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 10.0);
        assert!((body - 1.8).abs() < 1e-6, "torso front at {body}");
        assert!(set.cast([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 1.5).is_infinite(), "beyond t_max is not a hit");
    }

    #[test]
    fn a_robot_footprint_touches_a_person_only_when_their_shapes_meet() {
        let tl = walker();
        let mut set = ActorSet::default();
        tl.apply(3.0, &mut set);
        assert!(set.touches(1.5, 0.0, 0.0, [0.3, 0.2], 1.3));
        assert!(!set.touches(1.0, 0.0, 0.0, [0.3, 0.2], 1.3));
    }
}
