use crate::world3d::V3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DepthNoise {
    pub sigma_at_1m_m: f64,
    pub baseline_m: f64,
    pub native_focal_px: f64,
    pub subpixel: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DepthCamera {
    pub model: String,
    pub width: usize,
    pub height: usize,
    pub hfov_deg: f64,
    pub vfov_deg: f64,
    pub range_min_m: f64,
    pub range_max_m: f64,
    pub rate_hz: f64,
    pub noise: DepthNoise,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanPipeline {
    pub voxel_leaf_m: f64,
    pub min_height_m: f64,
    pub max_height_m: f64,
    pub range_min_m: f64,
    pub range_max_m: f64,
    pub angle_increment_rad: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DepthSensor {
    pub source: String,
    pub camera: DepthCamera,
    pub pipeline: ScanPipeline,
}

pub fn parse_depth_sensor(text: &str) -> Result<DepthSensor, String> {
    let sensor: DepthSensor = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    let c = &sensor.camera;
    if c.width == 0 || c.height == 0 || !(c.hfov_deg > 0.0 && c.hfov_deg < 180.0) || !(c.vfov_deg > 0.0 && c.vfov_deg < 180.0) {
        return Err(format!("depth camera {} needs a positive image size and fields of view under 180 degrees", c.model));
    }
    if !(c.range_min_m >= 0.0 && c.range_max_m > c.range_min_m) {
        return Err(format!("depth camera {} range_min_m must be below range_max_m", c.model));
    }
    if sensor.pipeline.voxel_leaf_m <= 0.0 || sensor.pipeline.angle_increment_rad <= 0.0 {
        return Err("scan pipeline voxel_leaf_m and angle_increment_rad must be positive".into());
    }
    Ok(sensor)
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn gaussian(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

impl DepthCamera {
    pub fn rays(&self) -> Vec<V3> {
        let fx = (self.width as f64 / 2.0) / (self.hfov_deg.to_radians() / 2.0).tan();
        let fy = (self.height as f64 / 2.0) / (self.vfov_deg.to_radians() / 2.0).tan();
        let mut out = Vec::with_capacity(self.width * self.height);
        for v in 0..self.height {
            for u in 0..self.width {
                let x = (u as f64 + 0.5 - self.width as f64 / 2.0) / fx;
                let y = (v as f64 + 0.5 - self.height as f64 / 2.0) / fy;
                let n = (x * x + y * y + 1.0).sqrt();
                out.push([x / n, y / n, 1.0 / n]);
            }
        }
        out
    }

    pub fn sigma(&self, z: f64) -> f64 {
        self.noise.sigma_at_1m_m * z * z
    }

    pub fn quantize(&self, z: f64) -> f64 {
        let fb = self.noise.native_focal_px * self.noise.baseline_m;
        if fb <= 0.0 || self.noise.subpixel <= 0.0 || z <= 0.0 {
            return z;
        }
        let step = self.noise.subpixel;
        let disparity = ((fb / z) / step).round().max(1.0) * step;
        fb / disparity
    }

    pub fn measure(&self, ray_z: f64, t: f64, rng: Option<&mut Rng>) -> Option<f64> {
        if !t.is_finite() || ray_z <= 0.0 {
            return None;
        }
        let z = t * ray_z;
        if z < self.range_min_m || z > self.range_max_m {
            return None;
        }
        let noisy = match rng {
            Some(r) => z + self.sigma(z) * r.gaussian(),
            None => z,
        };
        let measured = self.quantize(noisy);
        (measured >= self.range_min_m && measured <= self.range_max_m).then(|| measured / ray_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_s() -> DepthCamera {
        DepthCamera {
            model: "test".into(),
            width: 160,
            height: 120,
            hfov_deg: 60.0,
            vfov_deg: 49.5,
            range_min_m: 0.35,
            range_max_m: 2.5,
            rate_hz: 30.0,
            noise: DepthNoise { sigma_at_1m_m: 0.003, baseline_m: 0.075, native_focal_px: 570.0, subpixel: 0.125 },
        }
    }

    #[test]
    fn noise_grows_with_the_square_of_distance() {
        let cam = mini_s();
        let spread = |z: f64| {
            let mut rng = Rng::new(7);
            let samples: Vec<f64> = (0..4000).filter_map(|_| cam.measure(1.0, z, Some(&mut rng))).collect();
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            (samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
        };
        let (near, far) = (spread(1.0), spread(2.0));
        assert!(near > 0.002 && near < 0.005, "sigma at 1 m was {near}");
        assert!(far / near > 3.0 && far / near < 5.0, "sigma ratio 2 m / 1 m was {}", far / near);
    }

    #[test]
    fn depth_outside_the_working_range_gives_no_return() {
        let cam = mini_s();
        assert!(cam.measure(1.0, 0.2, None).is_none());
        assert!(cam.measure(1.0, 2.8, None).is_none());
        assert!(cam.measure(0.5, 3.0, None).is_some(), "an oblique ray 3 m long is only 1.5 m deep");
    }

    #[test]
    fn quantization_steps_follow_disparity() {
        let cam = mini_s();
        let step_at = |z: f64| {
            let a = cam.quantize(z);
            let mut b = a;
            let mut probe = z;
            while b == a {
                probe += 0.0005;
                b = cam.quantize(probe);
            }
            b - a
        };
        let (s1, s2) = (step_at(1.0), step_at(2.0));
        assert!(s1 > 0.002 && s1 < 0.004, "step at 1 m was {s1}");
        assert!(s2 / s1 > 3.5 && s2 / s1 < 4.5, "step ratio was {}", s2 / s1);
    }
}
