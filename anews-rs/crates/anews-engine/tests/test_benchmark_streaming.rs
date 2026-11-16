use anews_engine::event_stream::EventBus;
use anews_engine::events::SimEvent;

#[tokio::test]
async fn test_event_flags_collect_correctly() {
    let bus = EventBus::new();
    let mut rx = bus.subscribe();

    bus.emit(SimEvent::Overshoot {
        overshoot_m: 0.5,
        tick: 1,
    });
    bus.emit(SimEvent::NearMiss {
        obstacle_id: "c".into(),
        distance_m: 0.2,
        tick: 2,
    });

    let mut flags = Vec::new();
    let mut max_overshoot = 0.0f64;
    let mut min_clearance = f64::INFINITY;

    for _ in 0..2 {
        let e = rx.recv().await.unwrap();
        match e {
            SimEvent::Overshoot { overshoot_m, .. } => {
                if overshoot_m > max_overshoot {
                    max_overshoot = overshoot_m;
                }
                flags.push("overshooting".to_string());
            }
            SimEvent::NearMiss { distance_m, .. } => {
                if distance_m < min_clearance {
                    min_clearance = distance_m;
                }
                flags.push("near_miss".to_string());
            }
            _ => {}
        }
    }

    assert_eq!(flags.len(), 2);
    assert!(flags.contains(&"overshooting".to_string()));
    assert!(flags.contains(&"near_miss".to_string()));
    assert_eq!(max_overshoot, 0.5);
    assert_eq!(min_clearance, 0.2);
}

#[tokio::test]
async fn test_grade_logic_goal_reached_is_pass() {
    let final_reason: Option<&'static str> = Some("goal_reached");
    let flags: Vec<String> = vec![];

    let grade = if matches!(final_reason, Some("goal_reached"))
        && !flags.contains(&"collision".to_string())
        && !flags.contains(&"timeout".to_string())
    {
        "hard_pass"
    } else {
        "hard_fail"
    };

    assert_eq!(grade, "hard_pass");
}

#[tokio::test]
async fn test_grade_logic_collision_is_fail() {
    let final_reason: Option<&'static str> = Some("goal_reached");
    let flags: Vec<String> = vec!["collision".to_string()];

    let grade = if matches!(final_reason, Some("goal_reached"))
        && !flags.contains(&"collision".to_string())
        && !flags.contains(&"timeout".to_string())
    {
        "hard_pass"
    } else {
        "hard_fail"
    };

    assert_eq!(grade, "hard_fail");
}

#[tokio::test]
async fn test_grade_logic_timeout_is_fail() {
    let final_reason: Option<&'static str> = Some("timeout");
    let flags: Vec<String> = vec!["timeout".to_string()];

    let grade = if matches!(final_reason, Some("goal_reached"))
        && !flags.contains(&"collision".to_string())
        && !flags.contains(&"timeout".to_string())
    {
        "hard_pass"
    } else {
        "hard_fail"
    };

    assert_eq!(grade, "hard_fail");
}

#[tokio::test]
async fn test_terminal_event_detection() {
    let events = vec![
        SimEvent::Collision {
            obstacle_id: "wall".into(),
            tick: 10,
        },
        SimEvent::GoalReached {
            time_s: 5.0,
            tick: 100,
        },
        SimEvent::Timeout { tick: 2400 },
        SimEvent::NearMiss {
            obstacle_id: "T1".into(),
            distance_m: 0.15,
            tick: 50,
        },
        SimEvent::PathReplanned { tick: 30 },
    ];

    let terminal_kinds = ["collision", "goal_reached", "timeout"];
    let terminal_count = events
        .iter()
        .filter(|e| terminal_kinds.contains(&e.kind()))
        .count();

    assert_eq!(terminal_count, 3);
}
