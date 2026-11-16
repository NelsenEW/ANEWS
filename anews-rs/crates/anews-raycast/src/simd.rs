use anews_core::raycast::{Ray, RayCaster};
use anews_core::world3d::World3d;
use parry3d::math::Vector;
use parry3d::query::{Ray as ParryRay, RayCast};
use parry3d::shape::TriMesh;
use rayon::prelude::*;

const MIN_RAYS_PER_TASK: usize = 256;

pub struct SimdCaster {
    mesh: Option<TriMesh>,
    floor: bool,
}

impl SimdCaster {
    pub fn new(world: &World3d) -> Result<Self, String> {
        let tris = world.bvh.triangles();
        let mesh = if tris.is_empty() {
            None
        } else {
            let mut verts = Vec::with_capacity(tris.len() * 3);
            let mut idx = Vec::with_capacity(tris.len());
            for (i, t) in tris.iter().enumerate() {
                for p in [t.a, t.b, t.c] {
                    verts.push(Vector::new(p[0] as f32, p[1] as f32, p[2] as f32));
                }
                let k = (i * 3) as u32;
                idx.push([k, k + 1, k + 2]);
            }
            Some(TriMesh::new(verts, idx).map_err(|e| format!("parry3d trimesh: {e}"))?)
        };
        Ok(SimdCaster { mesh, floor: world.floor })
    }

    fn cast(&self, r: &Ray) -> f64 {
        let n = (r.dir[0] * r.dir[0] + r.dir[1] * r.dir[1] + r.dir[2] * r.dir[2]).sqrt();
        let d = [r.dir[0] / n, r.dir[1] / n, r.dir[2] / n];
        let mut best = r.t_max;
        let mut hit = false;
        if self.floor && d[2] < -1e-12 {
            let t = -r.origin[2] / d[2];
            if t > 0.0 && t < best {
                best = t;
                hit = true;
            }
        }
        if let Some(mesh) = &self.mesh {
            let ray = ParryRay::new(
                Vector::new(r.origin[0] as f32, r.origin[1] as f32, r.origin[2] as f32),
                Vector::new(d[0] as f32, d[1] as f32, d[2] as f32),
            );
            if let Some(t) = mesh.cast_local_ray(&ray, best as f32, false) {
                let t = t as f64;
                if t > 1e-9 && t < best {
                    best = t;
                    hit = true;
                }
            }
        }
        if hit {
            best
        } else {
            f64::INFINITY
        }
    }
}

impl RayCaster for SimdCaster {
    fn cast_batch(&self, rays: &[Ray]) -> Vec<f64> {
        rays.par_iter().with_min_len(MIN_RAYS_PER_TASK).map(|r| self.cast(r)).collect()
    }
}
