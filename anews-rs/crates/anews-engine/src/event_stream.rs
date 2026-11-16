use crate::events::SimEvent;
use tokio::sync::broadcast;

const CHANNEL_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
pub struct EventBus {
    tx: broadcast::Sender<SimEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(CHANNEL_CAPACITY);
        EventBus { tx }
    }

    pub fn emit(&self, event: SimEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SimEvent> {
        self.tx.subscribe()
    }

    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_emit_without_subscribers_is_ok() {
        let bus = EventBus::new();
        bus.emit(SimEvent::Collision { obstacle_id: "c".into(), tick: 1 });
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[tokio::test]
    async fn test_subscribe_receives_emitted_event() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        bus.emit(SimEvent::Timeout { tick: 42 });
        let got = rx.recv().await.unwrap();
        assert!(matches!(got, SimEvent::Timeout { tick: 42 }));
    }

    #[tokio::test]
    async fn test_multiple_subscribers_all_receive() {
        let bus = EventBus::new();
        let mut r1 = bus.subscribe();
        let mut r2 = bus.subscribe();
        bus.emit(SimEvent::GoalReached { time_s: 1.0, tick: 1 });
        assert!(matches!(r1.recv().await.unwrap(), SimEvent::GoalReached { .. }));
        assert!(matches!(r2.recv().await.unwrap(), SimEvent::GoalReached { .. }));
    }
}
