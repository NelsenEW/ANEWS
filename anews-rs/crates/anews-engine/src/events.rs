use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SimEvent {
    Collision {
        obstacle_id: String,
        tick: u64,
    },
    NearMiss {
        obstacle_id: String,
        distance_m: f64,
        tick: u64,
    },
    Overshoot {
        overshoot_m: f64,
        tick: u64,
    },
    Oscillation {
        period_ticks: u32,
        amplitude_m: f64,
        tick: u64,
    },
    GoalReached {
        time_s: f64,
        tick: u64,
    },
    Timeout {
        tick: u64,
    },
    Stuck {
        stuck_secs: f64,
        tick: u64,
    },
    PathReplanned {
        tick: u64,
    },
    HumanApproach {
        human_name: String,
        distance_m: f64,
        tick: u64,
    },
}

impl SimEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            SimEvent::Collision { .. } => "collision",
            SimEvent::NearMiss { .. } => "near_miss",
            SimEvent::Overshoot { .. } => "overshoot",
            SimEvent::Oscillation { .. } => "oscillation",
            SimEvent::GoalReached { .. } => "goal_reached",
            SimEvent::Timeout { .. } => "timeout",
            SimEvent::Stuck { .. } => "stuck",
            SimEvent::PathReplanned { .. } => "path_replanned",
            SimEvent::HumanApproach { .. } => "human_approach",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_enum_json_snapshot() {
        let cases = vec![
            (
                SimEvent::Collision {
                    obstacle_id: "C6".into(),
                    tick: 42,
                },
                r#"{"event":"collision","obstacle_id":"C6","tick":42}"#,
            ),
            (
                SimEvent::NearMiss {
                    obstacle_id: "T5".into(),
                    distance_m: 0.18,
                    tick: 99,
                },
                r#"{"event":"near_miss","obstacle_id":"T5","distance_m":0.18,"tick":99}"#,
            ),
            (
                SimEvent::Overshoot {
                    overshoot_m: 0.42,
                    tick: 200,
                },
                r#"{"event":"overshoot","overshoot_m":0.42,"tick":200}"#,
            ),
            (
                SimEvent::Oscillation {
                    period_ticks: 12,
                    amplitude_m: 0.05,
                    tick: 300,
                },
                r#"{"event":"oscillation","period_ticks":12,"amplitude_m":0.05,"tick":300}"#,
            ),
            (
                SimEvent::GoalReached {
                    time_s: 21.5,
                    tick: 430,
                },
                r#"{"event":"goal_reached","time_s":21.5,"tick":430}"#,
            ),
            (
                SimEvent::Timeout { tick: 2000 },
                r#"{"event":"timeout","tick":2000}"#,
            ),
            (
                SimEvent::Stuck {
                    stuck_secs: 3.2,
                    tick: 150,
                },
                r#"{"event":"stuck","stuck_secs":3.2,"tick":150}"#,
            ),
            (
                SimEvent::PathReplanned { tick: 88 },
                r#"{"event":"path_replanned","tick":88}"#,
            ),
            (
                SimEvent::HumanApproach {
                    human_name: "waiter".into(),
                    distance_m: 0.35,
                    tick: 220,
                },
                r#"{"event":"human_approach","human_name":"waiter","distance_m":0.35,"tick":220}"#,
            ),
        ];
        for (event, expected) in cases {
            let got = serde_json::to_string(&event).unwrap();
            assert_eq!(got, expected, "snapshot mismatch for {:?}", event.kind());
        }
    }

    #[test]
    fn test_all_nine_kinds_distinct() {
        let kinds: Vec<&str> = vec![
            SimEvent::Collision {
                obstacle_id: "x".into(),
                tick: 0,
            }
            .kind(),
            SimEvent::NearMiss {
                obstacle_id: "x".into(),
                distance_m: 0.0,
                tick: 0,
            }
            .kind(),
            SimEvent::Overshoot {
                overshoot_m: 0.0,
                tick: 0,
            }
            .kind(),
            SimEvent::Oscillation {
                period_ticks: 0,
                amplitude_m: 0.0,
                tick: 0,
            }
            .kind(),
            SimEvent::GoalReached {
                time_s: 0.0,
                tick: 0,
            }
            .kind(),
            SimEvent::Timeout { tick: 0 }.kind(),
            SimEvent::Stuck {
                stuck_secs: 0.0,
                tick: 0,
            }
            .kind(),
            SimEvent::PathReplanned { tick: 0 }.kind(),
            SimEvent::HumanApproach {
                human_name: "x".into(),
                distance_m: 0.0,
                tick: 0,
            }
            .kind(),
        ];
        let mut unique = kinds.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(kinds.len(), 9);
        assert_eq!(unique.len(), 9);
    }
}
