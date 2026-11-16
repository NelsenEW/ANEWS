use serde::{Deserialize, Serialize};
use std::path::Path;

pub type Footprint = Vec<(f64, f64)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    pub footprint: Footprint,
    pub footprint_padding: f64,
    pub collision_radius: f64,
    pub max_linear_vel: f64,
    pub max_angular_vel: f64,
    pub max_linear_accel: f64,
    pub max_angular_accel: f64,
}

pub fn default_config() -> RobotConfig {
    RobotConfig {
        footprint: vec![
            (-0.299, -0.180),
            (-0.299, 0.180),
            (0.299, 0.180),
            (0.299, -0.180),
        ],
        footprint_padding: 0.05,
        collision_radius: 0.35,
        max_linear_vel: 0.6,
        max_angular_vel: 0.8,
        max_linear_accel: 0.22,
        max_angular_accel: 0.6,
    }
}

pub fn load_robot_config(config_dir: &Path) -> RobotConfig {
    let yaml_path = config_dir.join("robot.yaml");
    let yaml_str = match std::fs::read_to_string(&yaml_path) {
        Ok(s) => s,
        Err(_) => return default_config(),
    };
    #[derive(Deserialize)]
    struct RobotYaml {
        #[serde(default)]
        footprint: Vec<Vec<f64>>,
        #[serde(default = "default_padding")]
        footprint_padding: f64,
        #[serde(default = "default_collision_radius")]
        collision_radius: f64,
        #[serde(default = "default_max_linear_vel")]
        max_linear_vel: f64,
        #[serde(default = "default_max_angular_vel")]
        max_angular_vel: f64,
        #[serde(default = "default_max_linear_accel")]
        max_linear_accel: f64,
        #[serde(default = "default_max_angular_accel")]
        max_angular_accel: f64,
    }
    fn default_padding() -> f64 {
        0.05
    }
    fn default_collision_radius() -> f64 {
        0.32
    }
    fn default_max_linear_vel() -> f64 {
        0.6
    }
    fn default_max_angular_vel() -> f64 {
        0.8
    }
    fn default_max_linear_accel() -> f64 {
        0.22
    }
    fn default_max_angular_accel() -> f64 {
        0.6
    }
    let data: RobotYaml = match serde_yaml::from_str(&yaml_str) {
        Ok(d) => d,
        Err(_) => return default_config(),
    };
    let footprint = if data.footprint.is_empty() {
        default_config().footprint
    } else {
        data.footprint
            .iter()
            .map(|v| {
                (
                    v.first().copied().unwrap_or(0.0),
                    v.get(1).copied().unwrap_or(0.0),
                )
            })
            .collect()
    };
    RobotConfig {
        footprint,
        footprint_padding: data.footprint_padding,
        collision_radius: data.collision_radius,
        max_linear_vel: data.max_linear_vel,
        max_angular_vel: data.max_angular_vel,
        max_linear_accel: data.max_linear_accel,
        max_angular_accel: data.max_angular_accel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = default_config();
        assert_eq!(cfg.footprint.len(), 4);
        assert!((cfg.collision_radius - 0.35).abs() < 1e-9);
    }

    #[test]
    fn test_load_missing_file_uses_default() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = load_robot_config(dir.path());
        assert_eq!(cfg.footprint.len(), 4);
    }

    #[test]
    fn test_load_from_yaml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("robot.yaml"),
            r#"
footprint:
  - [-0.3, -0.2]
  - [-0.3, 0.2]
  - [0.3, 0.2]
  - [0.3, -0.2]
footprint_padding: 0.1
collision_radius: 0.4
"#,
        )
        .unwrap();
        let cfg = load_robot_config(dir.path());
        assert_eq!(cfg.footprint.len(), 4);
        assert!((cfg.collision_radius - 0.4).abs() < 1e-9);
        assert!((cfg.footprint_padding - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_load_velocity_limits() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("robot.yaml"),
            r#"
footprint:
  - [-0.3, -0.2]
  - [-0.3, 0.2]
  - [0.3, 0.2]
  - [0.3, -0.2]
max_linear_vel: 0.5
max_angular_vel: 0.7
max_linear_accel: 0.2
max_angular_accel: 0.5
"#,
        )
        .unwrap();
        let cfg = load_robot_config(dir.path());
        assert!((cfg.max_linear_vel - 0.5).abs() < 1e-9);
        assert!((cfg.max_angular_vel - 0.7).abs() < 1e-9);
        assert!((cfg.max_linear_accel - 0.2).abs() < 1e-9);
        assert!((cfg.max_angular_accel - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_default_velocity_limits() {
        let cfg = default_config();
        assert!((cfg.max_linear_vel - 0.6).abs() < 1e-9);
        assert!((cfg.max_angular_vel - 0.8).abs() < 1e-9);
    }
}
