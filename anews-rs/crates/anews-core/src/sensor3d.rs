use crate::world3d::{V3, World3d};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    pub xyz: [f64; 3],
    pub quat_xyzw: [f64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Sensor3dKind {
    Lidar { angle_min: f64, angle_max: f64, count: usize },
    Depth { width: usize, height: usize, hfov_deg: f64, vfov_deg: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensor3dConfig {
    pub name: String,
    pub topic: String,
    pub frame_id: String,
    pub rate_hz: f64,
    pub range_min: f64,
    pub range_max: f64,
    #[serde(default)]
    pub noise_std_dev: f64,
    pub mount: Mount,
    #[serde(flatten)]
    pub kind: Sensor3dKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensor3dSuite {
    pub sensors: Vec<Sensor3dConfig>,
}

pub fn parse_sensor3d_suite(text: &str) -> Result<Sensor3dSuite, String> {
    serde_yaml::from_str(text).map_err(|e| e.to_string())
}

pub fn quat_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

pub fn mat_vec(m: &[[f64; 3]; 3], v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

impl Sensor3dConfig {
    pub fn local_rays(&self) -> Vec<V3> {
        match &self.kind {
            Sensor3dKind::Lidar { angle_min, angle_max, count } => (0..*count)
                .map(|i| {
                    let a = angle_min + (angle_max - angle_min) * i as f64 / (*count as f64 - 1.0).max(1.0);
                    [a.cos(), a.sin(), 0.0]
                })
                .collect(),
            Sensor3dKind::Depth { width, height, hfov_deg, vfov_deg } => {
                let fx = (*width as f64 / 2.0) / (hfov_deg.to_radians() / 2.0).tan();
                let fy = (*height as f64 / 2.0) / (vfov_deg.to_radians() / 2.0).tan();
                let mut out = Vec::with_capacity(width * height);
                for v in 0..*height {
                    for u in 0..*width {
                        let x = (u as f64 + 0.5 - *width as f64 / 2.0) / fx;
                        let y = (v as f64 + 0.5 - *height as f64 / 2.0) / fy;
                        let n = (x * x + y * y + 1.0).sqrt();
                        out.push([x / n, y / n, 1.0 / n]);
                    }
                }
                out
            }
        }
    }

    pub fn cast(&self, world: &World3d, rays: &[V3], base: (f64, f64, f64), noise: &mut impl FnMut() -> f64) -> Vec<f32> {
        let (bx, by, yaw) = base;
        let (s, c) = yaw.sin_cos();
        let mount_rot = quat_matrix(self.mount.quat_xyzw);
        let m = self.mount.xyz;
        let origin = [bx + c * m[0] - s * m[1], by + s * m[0] + c * m[1], m[2]];
        rays.iter()
            .map(|r| {
                let b = mat_vec(&mount_rot, *r);
                let d = [c * b[0] - s * b[1], s * b[0] + c * b[1], b[2]];
                let t = world.cast(origin, d, self.range_max);
                if t.is_finite() && t >= self.range_min {
                    (t + self.noise_std_dev * noise()) as f32
                } else if t.is_finite() {
                    f32::NAN
                } else {
                    f32::INFINITY
                }
            })
            .collect()
    }
}
