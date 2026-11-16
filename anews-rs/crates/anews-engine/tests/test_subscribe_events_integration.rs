use anews_engine::event_stream::EventBus;
use anews_engine::events::SimEvent;

#[tokio::test]
async fn test_event_bus_streams_all_event_kinds() {
    let bus = EventBus::new();
    let mut rx = bus.subscribe();
    bus.emit(SimEvent::Collision {
        obstacle_id: "c".into(),
        tick: 1,
    });
    bus.emit(SimEvent::GoalReached {
        time_s: 1.0,
        tick: 2,
    });
    assert!(matches!(rx.recv().await.unwrap(), SimEvent::Collision { .. }));
    assert!(matches!(rx.recv().await.unwrap(), SimEvent::GoalReached { .. }));
}

#[tokio::test]
async fn test_event_bus_kind_filters() {
    let bus = EventBus::new();
    let mut rx = bus.subscribe();
    bus.emit(SimEvent::NearMiss {
        obstacle_id: "T1".into(),
        distance_m: 0.2,
        tick: 5,
    });
    bus.emit(SimEvent::Stuck {
        stuck_secs: 3.0,
        tick: 10,
    });
    bus.emit(SimEvent::GoalReached {
        time_s: 5.0,
        tick: 15,
    });

    let until = vec!["goal_reached".to_string()];

    let mut non_terminal_count = 0usize;
    loop {
        let event = rx.recv().await.unwrap();
        let kind = event.kind().to_string();
        if until.contains(&kind) {
            break;
        }
        non_terminal_count += 1;
    }
    assert_eq!(non_terminal_count, 2, "expected 2 non-terminal events before goal_reached");
}

#[tokio::test]
async fn test_cancellation_registry() {
    use anews_engine::sim::CancellationRegistry;

    let reg = CancellationRegistry::new();
    assert!(!reg.is_cancelled("req-1"));
    reg.cancel("req-1");
    assert!(reg.is_cancelled("req-1"));
    reg.clear("req-1");
    assert!(!reg.is_cancelled("req-1"));
}

#[tokio::test]
async fn test_sim_tick_count_accessor() {
    use std::path::Path;

    let map_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("examples/maps/empty_restaurant.yaml");
    if !map_path.exists() {
        return;
    }
    let config_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("config");
    let sim = anews_engine::sim::Simulation::new(&map_path, None, &config_dir).unwrap();
    assert_eq!(sim.tick_count(), 0);
}
