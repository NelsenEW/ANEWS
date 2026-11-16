use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScenarioFurniture {
    #[serde(rename = "type")]
    pub type_id: String,
    pub pose: Vec<f64>,
    #[serde(default)]
    pub label: String,
    #[serde(default = "default_width")]
    pub width: f64,
    #[serde(default = "default_height")]
    pub height: f64,
}

fn default_width() -> f64 {
    1.0
}
fn default_height() -> f64 {
    0.8
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScenarioHuman {
    pub name: String,
    #[serde(default = "default_speed")]
    pub speed: f64,
    #[serde(default = "default_radius")]
    pub radius: f64,
    #[serde(default)]
    pub path: Vec<Vec<f64>>,
    #[serde(rename = "loop", default)]
    pub loop_path: bool,
    #[serde(default)]
    pub pause_at_waypoints: f64,
    #[serde(default)]
    pub start_delay: f64,
}

fn default_speed() -> f64 {
    1.0
}
fn default_radius() -> f64 {
    0.3
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScenarioGoal {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub theta: Option<f64>,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    #[serde(default = "default_yaw_tolerance")]
    pub yaw_tolerance: f64,
}

fn default_tolerance() -> f64 {
    0.35
}
fn default_timeout() -> f64 {
    60.0
}
fn default_yaw_tolerance() -> f64 {
    0.5
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScenarioGoalRef {
    #[serde(rename = "ref")]
    pub ref_name: String,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    #[serde(default = "default_yaw_tolerance")]
    pub yaw_tolerance: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LegFailPolicy {
    #[default]
    Continue,
    Abort,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FurnitureJitter {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_jitter_offset")]
    pub max_offset: f64,
    #[serde(default = "default_jitter_rotation")]
    pub max_rotation: f64,
    #[serde(default)]
    pub applies_to: Vec<String>,
}

fn default_jitter_offset() -> f64 {
    0.1
}
fn default_jitter_rotation() -> f64 {
    0.05
}

impl Default for FurnitureJitter {
    fn default() -> Self {
        Self {
            enabled: false,
            max_offset: 0.1,
            max_rotation: 0.05,
            applies_to: vec!["all".into()],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioRobot {
    #[serde(default)]
    pub spawn_pose: Vec<f64>,
    #[serde(default = "default_sensor_profile")]
    pub sensor_profile: String,
}

fn default_sensor_profile() -> String {
    "default".into()
}

impl Default for ScenarioRobot {
    fn default() -> Self {
        Self {
            spawn_pose: vec![0.0, 0.0, 0.0],
            sensor_profile: "default".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    pub seed: u64,
    pub localization_map: String,
    pub robot_pose: (f64, f64, f64),
    pub sensor_profile: String,
    pub goal: Option<ScenarioGoal>,
    pub furniture: Vec<ScenarioFurniture>,
    pub humans: Vec<ScenarioHuman>,
    pub furniture_jitter: FurnitureJitter,
    pub metadata: HashMap<String, serde_yaml::Value>,
    pub named_poses: Vec<crate::named_pose::NamedPose>,
    pub goals: Vec<ScenarioGoalRef>,
    pub on_leg_fail: LegFailPolicy,
}

#[derive(Deserialize)]
struct ScenarioYaml {
    scenario: ScenarioInner,
}

#[derive(Deserialize)]
struct ScenarioInner {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    localization_map: String,
    #[serde(default)]
    robot: ScenarioRobot,
    #[serde(default)]
    goal: Option<ScenarioGoal>,
    #[serde(default)]
    furniture: Vec<ScenarioFurniture>,
    #[serde(default)]
    humans: Vec<ScenarioHuman>,
    #[serde(default)]
    furniture_jitter: FurnitureJitter,
    #[serde(default)]
    metadata: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    named_poses: Vec<crate::named_pose::NamedPose>,
    #[serde(default)]
    goals: Vec<ScenarioGoalRef>,
    #[serde(default)]
    on_leg_fail: LegFailPolicy,
}

pub fn load_scenario(yaml_path: &Path) -> Result<Scenario, Box<dyn std::error::Error>> {
    let yaml_str = std::fs::read_to_string(yaml_path)?;
    let raw: ScenarioYaml = serde_yaml::from_str(&yaml_str)?;
    let s = raw.scenario;
    let robot_pose = (
        s.robot.spawn_pose.first().copied().unwrap_or(0.0),
        s.robot.spawn_pose.get(1).copied().unwrap_or(0.0),
        s.robot.spawn_pose.get(2).copied().unwrap_or(0.0),
    );
    let has_goal = s.goal.is_some();
    let has_goals = !s.goals.is_empty();
    if has_goal && has_goals {
        eprintln!(
            "WARNING: Scenario '{}' has both 'goal' and 'goals' fields; 'goals' takes precedence",
            s.name
        );
    }
    let scenario = Scenario {
        name: s.name,
        description: s.description,
        seed: s.seed,
        localization_map: s.localization_map,
        robot_pose,
        sensor_profile: s.robot.sensor_profile,
        goal: s.goal,
        furniture: s.furniture,
        humans: s.humans,
        furniture_jitter: s.furniture_jitter,
        metadata: s.metadata,
        named_poses: s.named_poses,
        goals: s.goals,
        on_leg_fail: s.on_leg_fail,
    };
    if let Err(unresolved) = scenario.validate_goal_refs() {
        return Err(format!(
            "Unresolved goal refs in scenario '{}': {}",
            scenario.name,
            unresolved.join(", ")
        ).into());
    }
    Ok(scenario)
}

impl Scenario {
    pub fn empty() -> Self {
        Self {
            name: "empty".into(),
            description: String::new(),
            seed: 0,
            localization_map: String::new(),
            robot_pose: (0.0, 0.0, 0.0),
            sensor_profile: "default".into(),
            goal: None,
            furniture: Vec::new(),
            humans: Vec::new(),
            furniture_jitter: FurnitureJitter::default(),
            metadata: HashMap::new(),
            named_poses: Vec::new(),
            goals: Vec::new(),
            on_leg_fail: LegFailPolicy::default(),
        }
    }

    pub fn validate_goal_refs(&self) -> Result<(), Vec<String>> {
        let pose_names: Vec<&str> = self.named_poses.iter().map(|p| p.name.as_str()).collect();
        let unresolved: Vec<String> = self.goals.iter()
            .filter(|g| !pose_names.contains(&g.ref_name.as_str()))
            .map(|g| g.ref_name.clone())
            .collect();
        if unresolved.is_empty() {
            Ok(())
        } else {
            Err(unresolved)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_minimal_scenario() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("test.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: test_scenario
  robot:
    spawn_pose: [1.0, 2.0, 3.14]
"#,
        )
        .unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert_eq!(s.name, "test_scenario");
        assert!((s.robot_pose.0 - 1.0).abs() < 1e-9);
        assert!((s.robot_pose.1 - 2.0).abs() < 1e-9);
        assert!(s.furniture.is_empty());
        assert!(s.humans.is_empty());
    }

    #[test]
    fn test_load_scenario_with_furniture_and_humans() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("test.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: restaurant
  seed: 42
  robot:
    spawn_pose: [-3.1, 1.0, -1.5708]
  goal:
    x: 5.0
    y: 0.5
    tolerance: 0.35
    timeout: 90.0
  furniture:
    - type: table4
      pose: [1.0, 2.0, 0.0]
      label: T1
      width: 1.5
      height: 0.8
  humans:
    - name: waiter
      speed: 0.6
      radius: 0.3
      path: [[-6.0, 1.3], [5.0, 1.3]]
      loop: true
      start_delay: 5.0
"#,
        )
        .unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert_eq!(s.name, "restaurant");
        assert_eq!(s.seed, 42);
        assert_eq!(s.furniture.len(), 1);
        assert_eq!(s.furniture[0].type_id, "table4");
        assert_eq!(s.humans.len(), 1);
        assert_eq!(s.humans[0].name, "waiter");
        assert!(s.humans[0].loop_path);
        assert!((s.humans[0].start_delay - 5.0).abs() < 1e-9);
        assert!(s.goal.is_some());
        let goal = s.goal.unwrap();
        assert!((goal.x - 5.0).abs() < 1e-9);
        assert!((goal.timeout - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_default_furniture_jitter() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("test.yaml");
        std::fs::write(&yaml, "scenario:\n  name: test\n").unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert!(!s.furniture_jitter.enabled);
    }

    #[test]
    fn scenario_parses_named_poses_block() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("test_poses.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: test_with_waypoints
  localization_map: maps/x.yaml
  named_poses:
    - name: CHARGER
      kind: charger
      x: 0.0
      y: 0.0
      theta: 0.0
    - name: KITCHEN
      kind: production
      x: 1.5
      y: -2.7
      theta: -1.57
"#,
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert_eq!(s.named_poses.len(), 2);
        assert_eq!(s.named_poses[0].name, "CHARGER");
        assert_eq!(s.named_poses[0].kind, crate::named_pose::PoseKind::Charger);
        assert_eq!(s.named_poses[1].x, 1.5);
    }

    #[test]
    fn scenario_without_named_poses_defaults_to_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("bare.yaml");
        std::fs::write(
            &yaml,
            "scenario:\n  name: t\n  localization_map: m.yaml\n",
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert!(s.named_poses.is_empty());
    }

    #[test]
    fn goals_parses_from_yaml_with_valid_refs() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("goals.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: l3_lunch_tour
  localization_map: maps/restaurant.yaml
  robot:
    spawn_pose: [0.0, 0.0, 0.0]
  on_leg_fail: continue
  goals:
    - ref: product_point
      tolerance: 0.35
      timeout: 45.0
    - ref: T1
      tolerance: 0.35
      timeout: 45.0
  named_poses:
    - { name: charging_pile, kind: charger, x: 0.0, y: 0.0, theta: 0.0 }
    - { name: product_point, kind: production, x: -3.5, y: -1.2, theta: -1.57 }
    - { name: T1, kind: delivery, x: 2.5, y: -1.5, theta: 1.571 }
  furniture: []
"#,
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert_eq!(s.goals.len(), 2);
        assert_eq!(s.goals[0].ref_name, "product_point");
        assert!((s.goals[0].tolerance - 0.35).abs() < 1e-9);
        assert!((s.goals[0].timeout - 45.0).abs() < 1e-9);
        assert_eq!(s.goals[1].ref_name, "T1");
        assert!(matches!(s.on_leg_fail, LegFailPolicy::Continue));
    }

    #[test]
    fn goals_with_invalid_refs_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("bad_refs.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: bad_tour
  goals:
    - ref: nonexistent_pose
    - ref: also_missing
  named_poses:
    - { name: charging_pile, kind: charger, x: 0.0, y: 0.0, theta: 0.0 }
"#,
        ).unwrap();
        let result = load_scenario(&yaml);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unresolved goal refs"), "Expected unresolved refs error, got: {}", err_msg);
        assert!(err_msg.contains("nonexistent_pose"), "Error should mention 'nonexistent_pose', got: {}", err_msg);
        assert!(err_msg.contains("also_missing"), "Error should mention 'also_missing', got: {}", err_msg);
    }

    #[test]
    fn backward_compat_goal_only_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("goal_only.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: legacy_goal
  goal:
    x: 5.0
    y: 0.5
    tolerance: 0.35
    timeout: 90.0
"#,
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert!(s.goal.is_some());
        assert!(s.goals.is_empty(), "goals should be empty when only goal is specified");
        let goal = s.goal.unwrap();
        assert!((goal.x - 5.0).abs() < 1e-9);
        assert!(matches!(s.on_leg_fail, LegFailPolicy::Continue));
    }

    #[test]
    fn goals_and_goal_both_present_goals_wins() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("both.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: both_fields
  goal:
    x: 1.0
    y: 2.0
  goals:
    - ref: product_point
  named_poses:
    - { name: product_point, kind: production, x: 3.0, y: 4.0, theta: 0.0 }
"#,
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert!(s.goal.is_some());
        assert_eq!(s.goals.len(), 1);
        assert_eq!(s.goals[0].ref_name, "product_point");
    }

    #[test]
    fn on_leg_fail_defaults_to_continue() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("defaults.yaml");
        std::fs::write(
            &yaml,
            "scenario:\n  name: minimal\n",
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert!(matches!(s.on_leg_fail, LegFailPolicy::Continue));
        assert!(s.goals.is_empty());
    }

    #[test]
    fn on_leg_fail_abort_parses_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("abort.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: abort_test
  on_leg_fail: abort
"#,
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert!(matches!(s.on_leg_fail, LegFailPolicy::Abort));
    }

    #[test]
    fn empty_scenario_has_default_goals_and_leg_fail() {
        let s = Scenario::empty();
        assert!(s.goals.is_empty());
        assert!(matches!(s.on_leg_fail, LegFailPolicy::Continue));
    }

    #[test]
    fn goal_refs_default_tolerance_and_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let yaml = dir.path().join("defaults.yaml");
        std::fs::write(
            &yaml,
            r#"
scenario:
  name: default_vals
  goals:
    - ref: CHARGER
  named_poses:
    - { name: CHARGER, kind: charger, x: 0.0, y: 0.0, theta: 0.0 }
"#,
        ).unwrap();
        let s = load_scenario(&yaml).unwrap();
        assert_eq!(s.goals.len(), 1);
        assert!((s.goals[0].tolerance - 0.35).abs() < 1e-9);
        assert!((s.goals[0].timeout - 60.0).abs() < 1e-9);
    }

    #[test]
    fn validate_goal_refs_ok_when_no_goals() {
        let s = Scenario::empty();
        assert!(s.validate_goal_refs().is_ok());
    }

    #[test]
    fn validate_goal_refs_ok_when_all_resolve() {
        let s = Scenario {
            goals: vec![
                ScenarioGoalRef { ref_name: "A".into(), tolerance: 0.35, timeout: 60.0, yaw_tolerance: 0.5 },
            ],
            named_poses: vec![
                crate::named_pose::NamedPose {
                    name: "A".into(),
                    kind: crate::named_pose::PoseKind::Charger,
                    x: 0.0, y: 0.0, theta: 0.0,
                },
            ],
            ..Scenario::empty()
        };
        assert!(s.validate_goal_refs().is_ok());
    }

    #[test]
    fn validate_goal_refs_err_on_unresolved() {
        let s = Scenario {
            goals: vec![
                ScenarioGoalRef { ref_name: "MISSING".into(), tolerance: 0.35, timeout: 60.0, yaw_tolerance: 0.5 },
            ],
            ..Scenario::empty()
        };
        let result = s.validate_goal_refs();
        assert!(result.is_err());
        let unresolved = result.unwrap_err();
        assert_eq!(unresolved, vec!["MISSING".to_string()]);
    }
}
