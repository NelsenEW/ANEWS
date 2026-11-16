use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PoseKind {
    Charger,
    Production,
    Delivery,
    Recycle,
    Agvtag,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NamedPose {
    pub name: String,
    pub kind: PoseKind,
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub theta: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_named_poses_from_yaml() {
        let yaml = r#"
- name: charging_pile
  kind: charger
  x: 0.0
  y: 0.0
  theta: 0.0
- name: product_point
  kind: production
  x: 1.5
  y: -2.7
  theta: -1.57
- name: T1
  kind: delivery
  x: 2.5
  y: -1.5
  theta: 1.571
- name: recycling_point
  kind: recycle
  x: 0.0
  y: -2.0
  theta: 3.14
- name: A7
  kind: agvtag
  x: 1.25
  y: -3.0
  theta: -2.5
"#;
        let poses: Vec<NamedPose> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(poses.len(), 5);
        assert_eq!(poses[0].name, "charging_pile");
        assert_eq!(poses[0].kind, PoseKind::Charger);
        assert_eq!(poses[1].kind, PoseKind::Production);
        assert_eq!(poses[2].kind, PoseKind::Delivery);
        assert_eq!(poses[3].kind, PoseKind::Recycle);
        assert_eq!(poses[4].kind, PoseKind::Agvtag);
        assert_eq!(poses[2].x, 2.5);
        assert!((poses[2].theta - 1.571).abs() < 1e-6);
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let yaml = "- name: x\n  kind: bogus\n  x: 0.0\n  y: 0.0\n  theta: 0.0";
        let result: Result<Vec<NamedPose>, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "unknown kind must fail to deserialize");
    }

    #[test]
    fn theta_is_optional_and_defaults_to_zero() {
        let yaml = "- name: a\n  kind: delivery\n  x: 1.0\n  y: 2.0";
        let poses: Vec<NamedPose> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(poses[0].theta, 0.0);
    }

    #[test]
    fn round_trips_through_json() {
        let p = NamedPose {
            name: "CHARGER".into(),
            kind: PoseKind::Charger,
            x: 1.0, y: 2.0, theta: 0.5,
        };
        let j = serde_json::to_value(&p).unwrap();
        assert_eq!(j["name"], "CHARGER");
        assert_eq!(j["kind"], "charger");
        assert_eq!(j["x"], 1.0);
        let back: NamedPose = serde_json::from_value(j).unwrap();
        assert_eq!(back.name, p.name);
        assert_eq!(back.kind, p.kind);
    }
}
