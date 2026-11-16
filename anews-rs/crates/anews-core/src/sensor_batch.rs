use crate::crowd::CrowdGrid;
use crate::depth_sensor::{DepthCamera, DepthSensor};
use crate::fidelity::{LIDAR_ANGLE_MAX, LIDAR_ANGLE_MIN, LIDAR_COUNT, LIDAR_RANGE, OPTICAL_QUAT};
use crate::raycast::{Ray, RayCaster};
use crate::sensor3d::{mat_vec, quat_matrix};
use crate::trip::{mat_mul, Mat3, Mount};
use crate::world3d::V3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobotScan {
    pub lidar: Vec<f64>,
    pub depth: Vec<f64>,
}

pub struct Rig {
    pub lidar: Mount,
    pub camera: Mount,
    pub sensor: DepthSensor,
    pub camera_rays: Vec<V3>,
    pub camera_optical: Mat3,
}

impl Rig {
    pub fn new(lidar: Mount, camera: Mount, sensor: DepthSensor) -> Self {
        let camera_rays = sensor.camera.rays();
        let camera_optical = mat_mul(&camera.rot, &quat_matrix(OPTICAL_QUAT));
        Rig { lidar, camera, sensor, camera_rays, camera_optical }
    }

    pub fn lidar_angle(i: usize) -> f64 {
        if i == LIDAR_COUNT - 1 {
            LIDAR_ANGLE_MAX
        } else {
            LIDAR_ANGLE_MIN + (LIDAR_ANGLE_MAX - LIDAR_ANGLE_MIN) / (LIDAR_COUNT as f64 - 1.0) * i as f64
        }
    }

    pub fn depth_reach(&self) -> f64 {
        self.sensor.camera.range_max_m * 1.5
    }

    pub fn rays(&self, pose: &Pose2) -> Vec<Ray> {
        let rot = yaw_matrix(pose.yaw);
        let base = [pose.x, pose.y, 0.0];
        let lidar_rot = mat_mul(&rot, &self.lidar.rot);
        let lo = add(base, mat_vec(&rot, self.lidar.xyz));
        let co = add(base, mat_vec(&rot, self.camera.xyz));
        let cam_rot = mat_mul(&rot, &self.camera_optical);
        let reach = self.depth_reach();
        let mut out = Vec::with_capacity(LIDAR_COUNT + self.camera_rays.len());
        for i in 0..LIDAR_COUNT {
            let a = Self::lidar_angle(i);
            out.push(Ray { origin: lo, dir: mat_vec(&lidar_rot, [a.cos(), a.sin(), 0.0]), t_max: LIDAR_RANGE });
        }
        for r in &self.camera_rays {
            out.push(Ray { origin: co, dir: mat_vec(&cam_rot, *r), t_max: reach });
        }
        out
    }

    pub fn depth_points(&self, depth: &[f64]) -> Vec<V3> {
        self.camera_rays
            .iter()
            .zip(depth)
            .filter(|(_, r)| r.is_finite())
            .map(|(ray, &r)| add(self.camera.xyz, mat_vec(&self.camera_optical, [ray[0] * r, ray[1] * r, ray[2] * r])))
            .collect()
    }
}

pub fn sense_batch(caster: &dyn RayCaster, rig: &Rig, poses: &[Pose2], seed: u32, tick: u32) -> Vec<RobotScan> {
    let per = LIDAR_COUNT + rig.camera_rays.len();
    let rays: Vec<Ray> = poses.iter().flat_map(|p| rig.rays(p)).collect();
    let hits = caster.cast_batch(&rays);
    hits.chunks(per)
        .enumerate()
        .map(|(robot, h)| {
            let depth = rig
                .camera_rays
                .iter()
                .zip(&h[LIDAR_COUNT..])
                .enumerate()
                .map(|(k, (ray, &t))| {
                    let key = noise_key(seed, tick, robot as u32, k as u32);
                    measure_keyed(&rig.sensor.camera, ray[2], t, key).unwrap_or(f64::NAN)
                })
                .collect();
            RobotScan { lidar: h[..LIDAR_COUNT].to_vec(), depth }
        })
        .collect()
}

pub fn occlude_by_crowd(rig: &Rig, crowd: &CrowdGrid, poses: &[Pose2], seed: u32, tick: u32, scans: &mut [RobotScan]) {
    for (robot, (pose, scan)) in poses.iter().zip(scans.iter_mut()).enumerate() {
        let rays = rig.rays(pose);
        for (r, t) in rays[..LIDAR_COUNT].iter().zip(scan.lidar.iter_mut()) {
            let c = crowd.cast(r.origin, unit_dir(r.dir), t.min(r.t_max));
            if c < *t {
                *t = c;
            }
        }
        for (k, (r, d)) in rays[LIDAR_COUNT..].iter().zip(scan.depth.iter_mut()).enumerate() {
            let bound = if d.is_finite() { *d } else { r.t_max };
            let c = crowd.cast(r.origin, unit_dir(r.dir), bound);
            if c.is_finite() && c < bound {
                let key = noise_key(seed, tick, robot as u32, k as u32);
                *d = measure_keyed(&rig.sensor.camera, rig.camera_rays[k][2], c, key).unwrap_or(f64::NAN);
            }
        }
    }
}

fn unit_dir(d: V3) -> V3 {
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / n, d[1] / n, d[2] / n]
}

pub fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

pub fn noise_key(seed: u32, tick: u32, robot: u32, ray: u32) -> u32 {
    pcg(seed ^ pcg(tick ^ pcg(robot ^ pcg(ray))))
}

pub fn unit(h: u32) -> f64 {
    ((h >> 8) as f64 + 0.5) / 16_777_216.0
}

pub fn keyed_gaussian(key: u32) -> f64 {
    let a = pcg(key);
    let b = pcg(a);
    (-2.0 * unit(a).ln()).sqrt() * (2.0 * std::f64::consts::PI * unit(b)).cos()
}

pub fn measure_keyed(camera: &DepthCamera, ray_z: f64, t: f64, key: u32) -> Option<f64> {
    if !t.is_finite() || ray_z <= 0.0 {
        return None;
    }
    let z = t * ray_z;
    if z < camera.range_min_m || z > camera.range_max_m {
        return None;
    }
    let measured = camera.quantize(z + camera.sigma(z) * keyed_gaussian(key));
    (measured >= camera.range_min_m && measured <= camera.range_max_m).then(|| measured / ray_z)
}

fn yaw_matrix(yaw: f64) -> Mat3 {
    let (s, c) = yaw.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crowd::{GridFrame, HIGH, LOW};
    use crate::world3d::{box_triangles, World3d};

    #[test]
    fn crowd_occlusion_after_sensing_matches_casting_through_the_crowd() {
        let sensor = crate::depth_sensor::parse_depth_sensor(include_str!("../../../../config/sensors/depth/orbbec_astra_mini_s.yaml")).unwrap();
        let identity = Mount { xyz: [0.22, 0.0, 0.23], rot: quat_matrix([0.0, 0.0, 0.0, 1.0]) };
        let camera = Mount { xyz: [0.317, -0.18, 1.086], rot: quat_matrix([0.0536, 0.4071, 0.0840, 0.9079]) };
        let rig = Rig::new(identity, camera, sensor);
        let mut tris = Vec::new();
        box_triangles([4.0, 0.0, 1.0], [0.1, 3.0, 1.0], 0.0, 0, &mut tris);
        let static_world = World3d::from_triangles(tris.clone(), true);
        let frame = GridFrame { width: 80, height: 80, resolution: 0.05, origin: [-2.0, -2.0] };
        let mut grid = CrowdGrid::new(frame.clone(), [0.0, 0.45], [0.35, 0.55]);
        for (x, y, flags) in [(1.2, 0.0, LOW), (1.25, 0.05, LOW | HIGH), (1.0, -0.4, HIGH), (0.9, 0.3, LOW)] {
            grid.cells[frame.cell(x, y).unwrap()] = flags;
        }
        let mut crowded = World3d::from_triangles(tris, true);
        crowded.crowd = Some(grid.clone());
        let poses = [Pose2 { x: 0.0, y: 0.0, yaw: 0.0 }, Pose2 { x: 0.2, y: -0.1, yaw: 0.3 }];
        let expected = sense_batch(&crowded, &rig, &poses, 7, 3);
        let bare = sense_batch(&static_world, &rig, &poses, 7, 3);
        let mut got = bare.clone();
        occlude_by_crowd(&rig, &grid, &poses, 7, 3, &mut got);
        let mut crowd_beams = 0;
        let (mut pixels, mut differ) = (0, 0);
        for (e, g) in expected.iter().zip(&got) {
            for (a, b) in e.lidar.iter().zip(&g.lidar) {
                assert!(a == b || (a - b).abs() < 1e-9, "lidar {a} vs {b}");
                crowd_beams += usize::from(*a < 3.0);
            }
            for (a, b) in e.depth.iter().zip(&g.depth) {
                pixels += 1;
                if (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9 {
                    continue;
                }
                differ += 1;
                assert!((a - b).abs() < 0.05, "depth {a} vs {b} differs by more than sensor noise");
            }
        }
        assert!(crowd_beams > 10, "the crowd should block lidar beams, saw {crowd_beams}");
        let occluded = bare.iter().zip(&got).flat_map(|(b, g)| b.depth.iter().zip(&g.depth)).filter(|(b, g)| b.to_bits() != g.to_bits()).count();
        assert!(occluded > 100, "the crowd should occlude depth pixels, saw {occluded}");
        assert!(differ * 200 < pixels, "{differ} of {pixels} depth pixels differ");
    }

    #[test]
    fn keyed_gaussian_has_unit_spread() {
        let n = 200_000u32;
        let xs: Vec<f64> = (0..n).map(|i| keyed_gaussian(noise_key(1, 0, 0, i))).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }
}
