use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
pub struct StreamChunk<'a> {
    pub request_id: &'a str,
    pub is_final: bool,
    #[serde(rename = "type")]
    pub chunk_type: &'a str,
    pub payload: Value,
}

pub fn chunk(req_id: &str, is_final: bool, chunk_type: &str, payload: Value) -> String {
    serde_json::to_string(&StreamChunk {
        request_id: req_id,
        is_final,
        chunk_type,
        payload,
    })
    .expect("StreamChunk serialize never fails")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_non_final_tick_chunk() {
        let s = chunk("uuid-1", false, "tick", json!({"tick": 42}));
        assert!(s.contains(r#""request_id":"uuid-1""#));
        assert!(s.contains(r#""is_final":false"#));
        assert!(s.contains(r#""type":"tick""#));
        assert!(s.contains(r#""tick":42"#));
    }

    #[test]
    fn test_final_result_chunk() {
        let s = chunk("uuid-2", true, "result", json!({"reason": "goal_reached", "total_ticks": 421}));
        assert!(s.contains(r#""is_final":true"#));
        assert!(s.contains(r#""type":"result""#));
        assert!(s.contains(r#""reason":"goal_reached""#));
    }

    #[test]
    fn test_chunk_with_nested_payload() {
        let s = chunk("r1", false, "event",
            json!({"event": "near_miss", "obstacle_id": "C6", "distance_m": 0.18}));
        assert!(s.contains(r#""event":"near_miss""#));
        assert!(s.contains(r#""obstacle_id":"C6""#));
    }
}
