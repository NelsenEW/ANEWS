use std::time::{Duration, Instant};

use anews_engine::sim::Simulation;
use serde_json::json;

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn make_sim() -> Simulation {
    let root = project_root();
    let map = root.join("examples/maps/restaurant.yaml");
    let config_dir = root.join("config");
    Simulation::new(&map, None, &config_dir).expect("sim boots")
}

#[test]
fn mutate_scenario_load_base_plus_add_furniture_returns_under_one_second() {
    let mut sim = make_sim();
    let cmd = json!({
        "action": "mutate_scenario",
        "request_id": "t1",
        "params": {
            "mutations": [
                {"op": "load_base", "path": "examples/benchmarks/l2_narrow_passage.yaml"},
                {"op": "add_furniture", "type": "chair", "pose": [2.0, -1.0, 0.0]},
            ],
            "output_scenario_path": "/tmp/mutate_test.yaml",
        },
    });

    let start = Instant::now();
    let resp = sim.handle_command(&cmd);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "mutate_scenario took {:?}, expected under 1s",
        elapsed,
    );
    assert_eq!(
        resp["status"].as_str(),
        Some("ok"),
        "unexpected response: {}",
        resp,
    );
    assert_eq!(
        resp["request_id"].as_str(),
        Some("t1"),
        "response missing request_id echo — Python client will time out: {}",
        resp,
    );
}

#[test]
fn mutate_scenario_with_nonexistent_base_path_errors_fast() {
    let mut sim = make_sim();
    let cmd = json!({
        "action": "mutate_scenario",
        "request_id": "t2",
        "params": {
            "mutations": [
                {"op": "load_base", "path": "examples/does_not_exist.yaml"},
            ],
            "output_scenario_path": "/tmp/mutate_error.yaml",
        },
    });

    let start = Instant::now();
    let resp = sim.handle_command(&cmd);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(500),
        "missing base should error fast, took {:?}",
        elapsed,
    );
    let infeasible = resp["validity"].as_str() == Some("infeasible");
    let has_errors = resp["errors"].as_array().map_or(false, |a| !a.is_empty());
    assert!(infeasible || has_errors, "expected error response: {}", resp);
    assert_eq!(
        resp["request_id"].as_str(),
        Some("t2"),
        "error response missing request_id echo: {}",
        resp,
    );
}

#[test]
fn mutate_scenario_valid_payload_writes_yaml_and_returns_valid() {
    let mut sim = make_sim();
    let out_path = std::env::temp_dir().join("mutate_scenario_valid_test.yaml");
    let _ = std::fs::remove_file(&out_path);

    let cmd = json!({
        "action": "mutate_scenario",
        "request_id": "t3",
        "params": {
            "mutations": [
                {"type": "add_furniture", "name": "chair", "x": 2.0, "y": -1.0, "theta": 0.0},
            ],
            "output_scenario_path": out_path.to_str().unwrap(),
        },
    });

    let start = Instant::now();
    let resp = sim.handle_command(&cmd);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "valid mutate_scenario took {:?}, expected under 1s",
        elapsed,
    );
    assert_eq!(resp["status"].as_str(), Some("ok"), "resp: {}", resp);
    assert_eq!(resp["validity"].as_str(), Some("valid"), "resp: {}", resp);
    assert_eq!(resp["applied"].as_u64(), Some(1), "resp: {}", resp);
    assert_eq!(
        resp["request_id"].as_str(),
        Some("t3"),
        "response missing request_id echo: {}",
        resp,
    );
    assert!(out_path.exists(), "output YAML not written at {:?}", out_path);
    let _ = std::fs::remove_file(&out_path);
}
