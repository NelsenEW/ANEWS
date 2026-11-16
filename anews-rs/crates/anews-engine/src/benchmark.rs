use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkEntry {
    pub scenario: String,
    pub goal: Vec<f64>,
    #[serde(default = "default_timeout")]
    pub timeout: f64,
}

fn default_timeout() -> f64 {
    60.0
}

#[derive(Debug, Clone, Deserialize)]
struct BenchmarkYaml {
    benchmark: BenchmarkDefinitionInner,
}

#[derive(Debug, Clone, Deserialize)]
struct BenchmarkDefinitionInner {
    name: String,
    description: String,
    tiers: HashMap<String, Vec<BenchmarkEntry>>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkDefinition {
    pub name: String,
    pub description: String,
    pub tiers: HashMap<String, Vec<BenchmarkEntry>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResults {
    pub name: String,
    pub description: String,
    pub tiers: HashMap<String, Vec<serde_json::Value>>,
    pub wall_time: f64,
}

pub fn load_benchmark(path: &Path) -> Result<BenchmarkDefinition, Box<dyn std::error::Error>> {
    let yaml_str = std::fs::read_to_string(path)?;
    let raw: BenchmarkYaml = serde_yaml::from_str(&yaml_str)?;
    let bdef = raw.benchmark;
    Ok(BenchmarkDefinition {
        name: bdef.name,
        description: bdef.description,
        tiers: bdef.tiers,
    })
}

pub fn grade_run(
    sim_result: &serde_json::Value,
    thresholds: Option<&HashMap<String, f64>>,
) -> serde_json::Value {
    let failure_class = sim_result
        .get("failure_class")
        .and_then(|v| v.as_str())
        .unwrap_or("incomplete");

    if failure_class.starts_with("hard_fail")
        || failure_class == "incomplete"
        || failure_class == "timeout"
    {
        return serde_json::json!({
            "grade": "hard_fail",
            "violations": [failure_class]
        });
    }

    let mut violations: Vec<String> = Vec::new();

    if let Some(thrs) = thresholds {
        if let Some(min_clearance_threshold) = thrs.get("min_clearance") {
            if let Some(actual_clearance) = sim_result.get("min_clearance").and_then(|v| v.as_f64())
            {
                if actual_clearance < *min_clearance_threshold {
                    violations.push("min_clearance".to_string());
                }
            }
        }

        if let Some(max_osc_threshold) = thrs.get("max_oscillation_count") {
            if let Some(actual_osc) = sim_result.get("oscillation_count").and_then(|v| v.as_f64()) {
                if actual_osc > *max_osc_threshold {
                    violations.push("oscillation_count".to_string());
                }
            }
        }
    }

    let sim_success = sim_result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !sim_success && violations.is_empty() {
        return serde_json::json!({
            "grade": "hard_fail",
            "violations": ["success_false"]
        });
    }

    if violations.is_empty() {
        serde_json::json!({
            "grade": "hard_pass",
            "violations": []
        })
    } else {
        serde_json::json!({
            "grade": "hard_pass",
            "violations": violations
        })
    }
}

pub fn run_benchmark<F>(
    bdef: &BenchmarkDefinition,
    thresholds: &HashMap<String, HashMap<String, f64>>,
    mut run_scenario_fn: F,
) -> BenchmarkResults
where
    F: FnMut(&str, &[f64], f64) -> serde_json::Value,
{
    let start_time = Instant::now();
    let mut tier_results: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

    for (tier_name, entries) in &bdef.tiers {
        let mut tier_entries = Vec::new();
        let tier_thresholds = thresholds.get(tier_name);

        for entry in entries {
            let sim_result = run_scenario_fn(&entry.scenario, &entry.goal, entry.timeout);
            let grade = if let Some(thr) = tier_thresholds {
                grade_run(&sim_result, Some(thr))
            } else {
                grade_run(&sim_result, None)
            };

            let entry_result = serde_json::json!({
                "scenario": entry.scenario,
                "goal": entry.goal,
                "timeout": entry.timeout,
                "sim_result": sim_result,
                "grade": grade
            });

            tier_entries.push(entry_result);
        }

        tier_results.insert(tier_name.clone(), tier_entries);
    }

    let wall_time = start_time.elapsed().as_secs_f64();

    BenchmarkResults {
        name: bdef.name.clone(),
        description: bdef.description.clone(),
        tiers: tier_results,
        wall_time,
    }
}

impl BenchmarkResults {
    pub fn to_json(&self) -> serde_json::Value {
        let mut tiers_summary = serde_json::json!({});
        let mut overall_total = 0;
        let mut overall_successes = 0;
        let mut overall_failures = 0;
        let mut overall_time_sum = 0.0;
        let mut overall_time_count = 0;

        for (tier_name, entries) in &self.tiers {
            let mut total_scenarios = 0;
            let mut successes = 0;
            let mut failures = 0;
            let mut time_sum = 0.0;
            let mut time_count = 0;

            for entry in entries {
                total_scenarios += 1;
                let grade = entry
                    .get("grade")
                    .and_then(|g| g.get("grade"))
                    .and_then(|g| g.as_str())
                    .unwrap_or("unknown");

                let success = grade == "hard_pass"
                    || entry
                        .get("sim_result")
                        .and_then(|sr| sr.get("success"))
                        .and_then(|s| s.as_bool())
                        .unwrap_or(false);

                if success {
                    successes += 1;
                } else {
                    failures += 1;
                }

                if let Some(ttg) = entry
                    .get("sim_result")
                    .and_then(|sr| sr.get("time_to_goal"))
                    .and_then(|t| t.as_f64())
                {
                    time_sum += ttg;
                    time_count += 1;
                }
            }

            let success_rate = if total_scenarios > 0 {
                successes as f64 / total_scenarios as f64
            } else {
                0.0
            };

            let avg_time = if time_count > 0 {
                time_sum / time_count as f64
            } else {
                0.0
            };

            let tier_json = serde_json::json!({
                "entries": entries,
                "summary": {
                    "total_scenarios": total_scenarios,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": success_rate,
                    "avg_time_to_goal": avg_time
                }
            });

            tiers_summary[tier_name] = tier_json;

            overall_total += total_scenarios;
            overall_successes += successes;
            overall_failures += failures;
            overall_time_sum += time_sum;
            overall_time_count += time_count;
        }

        let overall_success_rate = if overall_total > 0 {
            overall_successes as f64 / overall_total as f64
        } else {
            0.0
        };

        let overall_avg_time = if overall_time_count > 0 {
            overall_time_sum / overall_time_count as f64
        } else {
            0.0
        };

        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "tiers": tiers_summary,
            "summary": {
                "total_scenarios": overall_total,
                "successes": overall_successes,
                "failures": overall_failures,
                "success_rate": overall_success_rate,
                "avg_time_to_goal": overall_avg_time
            },
            "wall_time": self.wall_time
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_benchmark_definition() {
        let yaml = r#"
benchmark:
  name: test_suite
  description: Test benchmark
  tiers:
    easy:
      - scenario: examples/simple.yaml
        goal: [3.0, 2.0]
        timeout: 30.0
      - scenario: examples/medium.yaml
        goal: [4.0, 3.0]
    hard:
      - scenario: examples/hard.yaml
        goal: [5.0, 4.0]
        timeout: 120.0
"#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{yaml}").unwrap();
        let bdef = load_benchmark(tmp.path()).unwrap();
        assert_eq!(bdef.name, "test_suite");
        assert_eq!(bdef.tiers.len(), 2);
        assert_eq!(bdef.tiers["easy"].len(), 2);
        assert_eq!(bdef.tiers["easy"][0].timeout, 30.0);
        assert_eq!(bdef.tiers["easy"][1].timeout, 60.0);
    }

    #[test]
    fn test_grade_run_collision_hard_fail() {
        let result = serde_json::json!({"failure_class": "hard_fail_collision"});
        let grade = grade_run(&result, None);
        assert_eq!(grade["grade"], "hard_fail");
    }

    #[test]
    fn test_grade_run_incomplete_hard_fail() {
        let result = serde_json::json!({"failure_class": "incomplete"});
        let grade = grade_run(&result, None);
        assert_eq!(grade["grade"], "hard_fail");
    }

    #[test]
    fn test_grade_run_timeout_hard_fail() {
        let result = serde_json::json!({"failure_class": "timeout"});
        let grade = grade_run(&result, None);
        assert_eq!(grade["grade"], "hard_fail");
    }

    #[test]
    fn test_grade_run_success_hard_pass() {
        let result =
            serde_json::json!({"failure_class": "pass", "success": true, "min_clearance": 0.3});
        let grade = grade_run(&result, None);
        assert_eq!(grade["grade"], "hard_pass");
    }

    #[test]
    fn test_grade_run_success_false_without_failure_class() {
        let result = serde_json::json!({"failure_class": "pass", "success": false});
        let grade = grade_run(&result, None);
        assert_eq!(grade["grade"], "hard_fail");
    }

    #[test]
    fn test_grade_run_below_threshold_pass_with_violation() {
        let result =
            serde_json::json!({"failure_class": "pass", "success": true, "min_clearance": 0.05});
        let mut thresholds = HashMap::new();
        thresholds.insert("min_clearance".to_string(), 0.1);
        let grade = grade_run(&result, Some(&thresholds));
        assert_eq!(grade["grade"], "hard_pass");
        assert!(
            grade["violations"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("min_clearance"))
        );
    }

    #[test]
    fn test_grade_run_oscillation_threshold_pass_with_violation() {
        let result = serde_json::json!({"failure_class": "pass", "success": true, "oscillation_count": 15.0});
        let mut thresholds = HashMap::new();
        thresholds.insert("max_oscillation_count".to_string(), 10.0);
        let grade = grade_run(&result, Some(&thresholds));
        assert_eq!(grade["grade"], "hard_pass");
        assert!(
            grade["violations"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("oscillation_count"))
        );
    }

    #[test]
    fn test_run_benchmark_basic() {
        let yaml = r#"
benchmark:
  name: test
  description: Test
  tiers:
    easy:
      - scenario: test.yaml
        goal: [1.0, 1.0]
        timeout: 10.0
"#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{yaml}").unwrap();
        let bdef = load_benchmark(tmp.path()).unwrap();
        let thresholds = HashMap::new();
        let results = run_benchmark(&bdef, &thresholds, |_scenario, _goal, _timeout| {
            serde_json::json!({
                "success": true,
                "failure_class": "pass",
                "time_to_goal": 5.0
            })
        });
        assert_eq!(results.name, "test");
        assert_eq!(results.tiers["easy"].len(), 1);
        let json = results.to_json();
        let easy_summary = &json["tiers"]["easy"]["summary"];
        assert_eq!(easy_summary["successes"], 1);
        assert_eq!(easy_summary["total_scenarios"], 1);
        assert!((easy_summary["avg_time_to_goal"].as_f64().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_benchmark_multiple_tiers() {
        let yaml = r#"
benchmark:
  name: multi_tier
  description: Multi tier test
  tiers:
    easy:
      - scenario: easy1.yaml
        goal: [1.0, 1.0]
      - scenario: easy2.yaml
        goal: [2.0, 2.0]
    hard:
      - scenario: hard1.yaml
        goal: [5.0, 5.0]
"#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{yaml}").unwrap();
        let bdef = load_benchmark(tmp.path()).unwrap();
        let thresholds = HashMap::new();

        let call_count = std::sync::Mutex::new(0);
        let results = run_benchmark(&bdef, &thresholds, |_scenario, _goal, _timeout| {
            let mut count = call_count.lock().unwrap();
            *count += 1;
            let current = *count;
            if current <= 2 {
                serde_json::json!({"success": true, "failure_class": "pass"})
            } else {
                serde_json::json!({"success": false, "failure_class": "timeout"})
            }
        });

        let json = results.to_json();
        assert_eq!(json["summary"]["total_scenarios"], 3);
        assert_eq!(json["summary"]["successes"], 2);
        assert_eq!(json["summary"]["failures"], 1);
    }

    #[test]
    fn test_benchmark_results_to_json_format() {
        let yaml = r#"
benchmark:
  name: json_test
  description: JSON format test
  tiers:
    test:
      - scenario: test.yaml
        goal: [1.0, 1.0]
        timeout: 15.0
"#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{yaml}").unwrap();
        let bdef = load_benchmark(tmp.path()).unwrap();
        let thresholds = HashMap::new();

        let results = run_benchmark(&bdef, &thresholds, |_scenario, _goal, _timeout| {
            serde_json::json!({
                "success": true,
                "failure_class": "pass",
                "time_to_goal": 3.5,
                "min_clearance": 0.25
            })
        });

        let json = results.to_json();
        assert!(json["name"].is_string());
        assert!(json["description"].is_string());
        assert!(json["tiers"]["test"].is_object());
        assert!(json["tiers"]["test"]["entries"].is_array());
        assert!(json["tiers"]["test"]["summary"].is_object());
        assert!(json["summary"].is_object());
        assert!(json["wall_time"].is_number());
    }

    #[test]
    fn test_grade_run_multiple_violations_on_pass() {
        let result = serde_json::json!({
            "failure_class": "pass",
            "success": true,
            "min_clearance": 0.05,
            "oscillation_count": 15.0
        });
        let mut thresholds = HashMap::new();
        thresholds.insert("min_clearance".to_string(), 0.1);
        thresholds.insert("max_oscillation_count".to_string(), 10.0);
        let grade = grade_run(&result, Some(&thresholds));
        assert_eq!(grade["grade"], "hard_pass");
        let violations = grade["violations"].as_array().unwrap();
        assert_eq!(violations.len(), 2);
        assert!(violations.contains(&serde_json::json!("min_clearance")));
        assert!(violations.contains(&serde_json::json!("oscillation_count")));
    }

    #[test]
    fn test_load_nav2_benchmark() {
        let project_root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&project_root)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("config/benchmarks/nav2_benchmark.yaml");
        if !path.exists() {
            return;
        }
        let bdef = load_benchmark(&path).unwrap();
        assert_eq!(bdef.name, "nav2_benchmark");
        assert_eq!(bdef.tiers.len(), 3);
        assert!(bdef.tiers.contains_key("l1_primitive"));
        assert!(bdef.tiers.contains_key("l2_constrained"));
        assert!(bdef.tiers.contains_key("l3_full"));
        assert_eq!(bdef.tiers["l1_primitive"].len(), 3);
        assert_eq!(bdef.tiers["l2_constrained"].len(), 3);
        assert_eq!(bdef.tiers["l3_full"].len(), 2);
    }
}
