use serde_json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct EventLogger {
    writer: BufWriter<File>,
}

impl EventLogger {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn log(&mut self, t: f64, event: &str, extra: serde_json::Value) {
        let t_rounded = (t * 10000.0).round() / 10000.0;
        let mut entry = serde_json::json!({"t": t_rounded, "event": event});
        if let serde_json::Value::Object(map) = extra {
            if let serde_json::Value::Object(ref mut obj) = entry {
                obj.extend(map);
            }
        }
        let _ = writeln!(
            self.writer,
            "{}",
            serde_json::to_string(&entry).unwrap_or_default()
        );
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_log_writes_jsonl() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut logger = EventLogger::new(&path).unwrap();
            logger.log(0.0, "init", serde_json::json!({"robot": [1.0, 2.0, 0.0]}));
            logger.log(
                0.0523,
                "cmd_vel",
                serde_json::json!({"linear": 0.5, "angular": 0.1}),
            );
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["t"], 0.0);
        assert_eq!(first["event"], "init");
        assert_eq!(first["robot"], serde_json::json!([1.0, 2.0, 0.0]));
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["t"], 0.0523);
        assert_eq!(second["event"], "cmd_vel");
        assert_eq!(second["linear"], 0.5);
    }

    #[test]
    fn test_log_rounds_timestamp() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut logger = EventLogger::new(&path).unwrap();
            logger.log(1.23456789, "tick", serde_json::json!({}));
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        let entry: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(entry["t"], 1.2346);
    }

    #[test]
    fn test_log_tick_with_humans() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut logger = EventLogger::new(&path).unwrap();
            logger.log(
                1.05,
                "tick",
                serde_json::json!({
                    "robot": [1.0, 2.0, 0.5],
                    "linear": 0.3,
                    "angular": 0.1,
                    "humans": [
                        {"name": "waiter", "x": 3.0, "y": 4.0, "theta": 1.57},
                        {"name": "customer", "x": 5.0, "y": 6.0, "theta": 0.0}
                    ]
                }),
            );
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        let entry: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(entry["event"], "tick");
        let humans = entry["humans"].as_array().unwrap();
        assert_eq!(humans.len(), 2);
        assert_eq!(humans[0]["name"], "waiter");
        assert_eq!(humans[0]["x"], 3.0);
        assert_eq!(humans[1]["name"], "customer");
    }

    #[test]
    fn test_log_creates_parent_dirs() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let deep_path = tmp_dir.path().join("a/b/c/events.jsonl");
        let mut logger = EventLogger::new(&deep_path).unwrap();
        logger.log(0.0, "test", serde_json::json!({}));
        assert!(deep_path.exists());
    }
}
