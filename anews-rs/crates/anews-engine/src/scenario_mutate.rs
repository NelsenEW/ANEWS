use crate::scenario::{Scenario, ScenarioFurniture, ScenarioHuman};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Mutation {
    LoadBase {
        path: String,
    },
    AddFurniture {
        name: String,
        x: f64,
        y: f64,
        #[serde(default)]
        theta: f64,
        #[serde(default = "default_width")]
        width: f64,
        #[serde(default = "default_height")]
        height: f64,
    },
    MoveFurniture {
        id: String,
        x: f64,
        y: f64,
        #[serde(default)]
        theta: f64,
    },
    RemoveFurniture {
        id: String,
    },
    AddHuman {
        name: String,
        path: Vec<Vec<f64>>,
        #[serde(default = "default_speed")]
        speed: f64,
        #[serde(default = "default_radius")]
        radius: f64,
        #[serde(default, rename = "loop")]
        loop_path: bool,
        #[serde(default)]
        avoidance: Option<HumanAvoidance>,
    },
    RemoveHuman {
        name: String,
    },
    EditMap {
        tool: String,
        points: Vec<Vec<f64>>,
        fill: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HumanAvoidance {
    pub enabled: bool,
    #[serde(default = "default_stop_dist")]
    pub stop_dist: f64,
    #[serde(default = "default_slow_dist")]
    pub slow_dist: f64,
    #[serde(default = "default_slow_factor")]
    pub slow_factor: f64,
}

fn default_width() -> f64 {
    1.0
}
fn default_height() -> f64 {
    0.8
}
fn default_speed() -> f64 {
    1.0
}
fn default_radius() -> f64 {
    0.3
}
fn default_stop_dist() -> f64 {
    0.4
}
fn default_slow_dist() -> f64 {
    0.9
}
fn default_slow_factor() -> f64 {
    0.3
}

#[derive(Debug)]
pub struct MutationError {
    pub index: usize,
    pub mutation: String,
    pub reason: String,
}

pub fn apply_mutations(
    base: Option<Scenario>,
    mutations: &[Mutation],
    scenario_dir: &Path,
) -> Result<Scenario, Vec<MutationError>> {
    let mut scenario = base.unwrap_or_else(Scenario::empty);
    let mut errors: Vec<MutationError> = Vec::new();

    for (idx, mutation) in mutations.iter().enumerate() {
        if let Err(reason) = apply_one(&mut scenario, mutation, scenario_dir) {
            errors.push(MutationError {
                index: idx,
                mutation: mutation_name(mutation).to_string(),
                reason,
            });
            break;
        }
    }

    if errors.is_empty() {
        Ok(scenario)
    } else {
        Err(errors)
    }
}

fn mutation_name(m: &Mutation) -> &'static str {
    match m {
        Mutation::LoadBase { .. } => "load_base",
        Mutation::AddFurniture { .. } => "add_furniture",
        Mutation::MoveFurniture { .. } => "move_furniture",
        Mutation::RemoveFurniture { .. } => "remove_furniture",
        Mutation::AddHuman { .. } => "add_human",
        Mutation::RemoveHuman { .. } => "remove_human",
        Mutation::EditMap { .. } => "edit_map",
    }
}

fn apply_one(
    scenario: &mut Scenario,
    mutation: &Mutation,
    scenario_dir: &Path,
) -> Result<(), String> {
    match mutation {
        Mutation::LoadBase { path } => {
            let full_path = if Path::new(path).is_absolute() {
                Path::new(path).to_path_buf()
            } else {
                scenario_dir.join(path)
            };
            let loaded = crate::scenario::load_scenario(&full_path)
                .map_err(|e| format!("load_base failed: {e}"))?;
            *scenario = loaded;
            Ok(())
        }
        Mutation::AddFurniture {
            name,
            x,
            y,
            theta,
            width,
            height,
        } => {
            scenario.furniture.push(ScenarioFurniture {
                type_id: name.clone(),
                pose: vec![*x, *y, *theta],
                label: name.clone(),
                width: *width,
                height: *height,
            });
            Ok(())
        }
        Mutation::MoveFurniture { id, x, y, theta } => {
            let item = scenario
                .furniture
                .iter_mut()
                .find(|f| f.label == *id)
                .ok_or_else(|| format!("no furniture with id {id}"))?;
            item.pose = vec![*x, *y, *theta];
            Ok(())
        }
        Mutation::RemoveFurniture { id } => {
            let before = scenario.furniture.len();
            scenario.furniture.retain(|f| f.label != *id);
            if scenario.furniture.len() == before {
                return Err(format!("no furniture with id {id}"));
            }
            Ok(())
        }
        Mutation::AddHuman {
            name,
            path,
            speed,
            radius,
            loop_path,
            ..
        } => {
            scenario.humans.push(ScenarioHuman {
                name: name.clone(),
                speed: *speed,
                radius: *radius,
                path: path.clone(),
                loop_path: *loop_path,
                pause_at_waypoints: 0.0,
                start_delay: 0.0,
            });
            Ok(())
        }
        Mutation::RemoveHuman { name } => {
            let before = scenario.humans.len();
            scenario.humans.retain(|h| h.name != *name);
            if scenario.humans.len() == before {
                return Err(format!("no human with name {name}"));
            }
            Ok(())
        }
        Mutation::EditMap { .. } => Err(
            "edit_map mutation not yet supported via mutate_scenario — use map_edit_* WS actions directly"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_atomic_rollback_on_move_missing_id() {
        let muts = vec![
            Mutation::AddFurniture {
                name: "chair".into(),
                x: 1.0,
                y: 2.0,
                theta: 0.0,
                width: 0.5,
                height: 0.5,
            },
            Mutation::MoveFurniture {
                id: "nonexistent".into(),
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            },
        ];
        let dir = tempdir().unwrap();
        let result = apply_mutations(None, &muts, dir.path());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].index, 1);
    }

    #[test]
    fn test_load_base_then_mutate() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("base.yaml");
        std::fs::write(
            &base_path,
            r#"
scenario:
  name: base
  furniture:
    - type: chair
      pose: [1.0, 2.0, 0.0]
      label: chair1
      width: 0.5
      height: 0.5
"#,
        )
        .unwrap();

        let muts = vec![
            Mutation::LoadBase {
                path: "base.yaml".into(),
            },
            Mutation::MoveFurniture {
                id: "chair1".into(),
                x: 5.0,
                y: 6.0,
                theta: 0.0,
            },
        ];
        let scenario = apply_mutations(None, &muts, dir.path()).unwrap();
        assert_eq!(scenario.furniture.len(), 1);
        assert_eq!(scenario.furniture[0].pose[0], 5.0);
        assert_eq!(scenario.furniture[0].pose[1], 6.0);
    }

    #[test]
    fn test_add_and_remove_human() {
        let dir = tempdir().unwrap();
        let muts = vec![
            Mutation::AddHuman {
                name: "waiter".into(),
                path: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
                speed: 0.6,
                radius: 0.3,
                loop_path: true,
                avoidance: None,
            },
            Mutation::RemoveHuman {
                name: "waiter".into(),
            },
        ];
        let scenario = apply_mutations(None, &muts, dir.path()).unwrap();
        assert_eq!(scenario.humans.len(), 0);
    }

    #[test]
    fn test_edit_map_rejected() {
        let dir = tempdir().unwrap();
        let muts = vec![Mutation::EditMap {
            tool: "brush".into(),
            points: vec![vec![0.0, 0.0]],
            fill: "wall".into(),
        }];
        let result = apply_mutations(None, &muts, dir.path());
        assert!(result.is_err());
    }
}
