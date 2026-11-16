use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountPose {
    pub xyz: [f64; 3],
    pub quat_xyzw: [f64; 4],
}

impl MountPose {
    pub fn pitch(&self) -> f64 {
        let [x, y, z, w] = self.quat_xyzw;
        let sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
        sinp.asin()
    }

    pub fn height(&self) -> f64 {
        self.xyz[2]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    pub name: String,
    pub frame_id: String,
    pub topic: String,
    pub fov_degrees: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vfov_degrees: Option<f64>,
    pub range_min: f64,
    pub range_max: f64,
    pub num_rays: usize,
    pub noise_std_dev: f64,
    pub mount_offset: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mount_pose: Option<MountPose>,
    pub rate_hz: u32,
    pub profile: String,
    #[serde(skip)]
    pub fov_radians: f64,
    #[serde(skip)]
    pub angle_min: f64,
    #[serde(skip)]
    pub angle_max: f64,
    #[serde(skip)]
    pub angle_increment: f64,
}

impl SensorConfig {
    pub fn compute_angles(&mut self) {
        self.fov_radians = self.fov_degrees.to_radians();
        self.angle_min = -self.fov_radians / 2.0;
        self.angle_max = self.fov_radians / 2.0;
        self.angle_increment = self.fov_radians / (self.num_rays.max(2) - 1) as f64;
    }
}

#[derive(Debug, Clone)]
pub struct SensorSuite {
    pub sensors: HashMap<String, SensorConfig>,
}

#[derive(Deserialize)]
struct SensorYaml {
    sensors: HashMap<String, SensorYamlEntry>,
}

#[derive(Deserialize)]
struct SensorYamlEntry {
    frame_id: String,
    topic: String,
    fov_degrees: f64,
    #[serde(default)]
    vfov_degrees: Option<f64>,
    range_min: f64,
    range_max: f64,
    num_rays: usize,
    noise_std_dev: f64,
    mount_offset: Vec<f64>,
    #[serde(default)]
    mount_pose: Option<MountPose>,
    rate_hz: u32,
    profile: String,
}

pub fn load_sensor_config(
    profile_name: &str,
    config_dir: &Path,
) -> Result<SensorSuite, Box<dyn std::error::Error>> {
    let yaml_path = config_dir
        .join("sensors")
        .join(format!("{profile_name}.yaml"));
    let yaml_str = std::fs::read_to_string(&yaml_path)?;
    let raw: SensorYaml = serde_yaml::from_str(&yaml_str)?;
    let mut sensors = HashMap::new();
    for (name, entry) in raw.sensors {
        let offset = [
            *entry.mount_offset.first().unwrap_or(&0.0),
            *entry.mount_offset.get(1).unwrap_or(&0.0),
            *entry.mount_offset.get(2).unwrap_or(&0.0),
        ];
        let mut cfg = SensorConfig {
            name: name.clone(),
            frame_id: entry.frame_id,
            topic: entry.topic,
            fov_degrees: entry.fov_degrees,
            vfov_degrees: entry.vfov_degrees,
            range_min: entry.range_min,
            range_max: entry.range_max,
            num_rays: entry.num_rays,
            noise_std_dev: entry.noise_std_dev,
            mount_offset: offset,
            mount_pose: entry.mount_pose,
            rate_hz: entry.rate_hz,
            profile: entry.profile,
            fov_radians: 0.0,
            angle_min: 0.0,
            angle_max: 0.0,
            angle_increment: 0.0,
        };
        cfg.compute_angles();
        sensors.insert(name, cfg);
    }
    Ok(SensorSuite { sensors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_sensor_yaml(dir: &Path) {
        let sensors_dir = dir.join("sensors");
        std::fs::create_dir_all(&sensors_dir).unwrap();
        let mut f = std::fs::File::create(sensors_dir.join("test.yaml")).unwrap();
        write!(
            f,
            r#"
sensors:
  laser:
    frame_id: base_laser_link
    topic: /scan
    fov_degrees: 270
    range_min: 0.05
    range_max: 10.0
    num_rays: 360
    noise_std_dev: 0.01
    mount_offset: [0.22, 0.0, 0.0]
    rate_hz: 10
    profile: laser
"#
        )
        .unwrap();
    }

    #[test]
    fn test_load_sensor_config() {
        let dir = tempfile::tempdir().unwrap();
        write_test_sensor_yaml(dir.path());
        let suite = load_sensor_config("test", dir.path()).unwrap();
        assert!(suite.sensors.contains_key("laser"));
        let laser = &suite.sensors["laser"];
        assert_eq!(laser.num_rays, 360);
        assert!((laser.fov_degrees - 270.0).abs() < 1e-9);
        assert!((laser.range_max - 10.0).abs() < 1e-9);
        assert_eq!(laser.profile, "laser");
    }

    #[test]
    fn test_angle_computation() {
        let mut cfg = SensorConfig {
            name: "test".into(),
            frame_id: "".into(),
            topic: "".into(),
            fov_degrees: 180.0,
            range_min: 0.0,
            range_max: 10.0,
            num_rays: 181,
            noise_std_dev: 0.0,
            mount_offset: [0.0; 3],
            mount_pose: None,
            vfov_degrees: None,
            rate_hz: 10,
            profile: "laser".into(),
            fov_radians: 0.0,
            angle_min: 0.0,
            angle_max: 0.0,
            angle_increment: 0.0,
        };
        cfg.compute_angles();
        assert!((cfg.angle_min - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-6);
        assert!((cfg.angle_max - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((cfg.angle_increment - std::f64::consts::PI / 180.0).abs() < 1e-6);
    }
}
