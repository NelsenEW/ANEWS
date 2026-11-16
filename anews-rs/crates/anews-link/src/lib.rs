use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use zenoh::liveliness::LivelinessToken;
use zenoh::pubsub::{Publisher, Subscriber};
use zenoh::{Config, Session, Wait};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RosType {
    pub name: &'static str,
    pub hash: &'static str,
}

pub mod types {
    use super::RosType;

    pub const CLOCK: RosType = RosType { name: "rosgraph_msgs::msg::dds_::Clock_", hash: "RIHS01_692f7a66e93a3c83e71765d033b60349ba68023a8c689a79e48078bcb5c58564" };
    pub const LASER_SCAN: RosType = RosType { name: "sensor_msgs::msg::dds_::LaserScan_", hash: "RIHS01_64c191398013af96509d518dac71d5164f9382553fce5c1f8cca5be7924bd828" };
    pub const POSE_STAMPED: RosType = RosType { name: "geometry_msgs::msg::dds_::PoseStamped_", hash: "RIHS01_10f3786d7d40fd2b54367835614bff85d4ad3b5dab62bf8bca0cc232d73b4cd8" };
    pub const TWIST: RosType = RosType { name: "geometry_msgs::msg::dds_::Twist_", hash: "RIHS01_9c45bf16fe0983d80e3cfe750d6835843d265a9a6c46bd2e609fcddde6fb8d2a" };
    pub const ODOMETRY: RosType = RosType { name: "nav_msgs::msg::dds_::Odometry_", hash: "RIHS01_3cc97dc7fb7502f8714462c526d369e35b603cfc34d946e3f2eda2766dfec6e0" };
    pub const IMU: RosType = RosType { name: "sensor_msgs::msg::dds_::Imu_", hash: "RIHS01_7d9a00ff131080897a5ec7e26e315954b8eae3353c3f995c55faf71574000b5b" };
    pub const POINT_CLOUD2: RosType = RosType { name: "sensor_msgs::msg::dds_::PointCloud2_", hash: "RIHS01_9198cabf7da3796ae6fe19c4cb3bdd3525492988c70522628af5daa124bae2b5" };
    pub const TF: RosType = RosType { name: "tf2_msgs::msg::dds_::TFMessage_", hash: "RIHS01_e369d0f05a23ae52508854b66f6aa0437f3449d652e8cbf22d5abe85d020f087" };
    pub const PATH: RosType = RosType { name: "nav_msgs::msg::dds_::Path_", hash: "RIHS01_1957a5bb3cee5da65c4e52e52b65a93df227efce4c20f8458b36e73066ca334b" };
    pub const OCCUPANCY_GRID: RosType = RosType { name: "nav_msgs::msg::dds_::OccupancyGrid_", hash: "RIHS01_8d348150c12913a31ee0ec170fbf25089e4745d17035792a1ba94d6f0bc0cfc7" };
    pub const NAVIGATION_CONTROL: RosType = RosType { name: "yocs_msgs::msg::dds_::NavigationControl_", hash: "RIHS01_aec2f10d4675c202225986fb7fb7a188e6310d478d4fe58c99308932f74d4f33" };
    pub const STRING: RosType = RosType { name: "std_msgs::msg::dds_::String_", hash: "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18" };
}

pub const ENTITY_QOS: &str = "::,10:,:,:,,";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Publisher,
    Subscriber,
}

impl EntityKind {
    fn tag(self) -> &'static str {
        match self {
            EntityKind::Publisher => "MP",
            EntityKind::Subscriber => "MS",
        }
    }
}

fn topic_name(topic: &str) -> &str {
    topic.trim_start_matches('/')
}

fn mangle(name: &str) -> String {
    name.replace('/', "%")
}

pub fn data_key(domain: u32, topic: &str, ty: RosType) -> String {
    format!("{domain}/{}/{}/{}", topic_name(topic), ty.name, ty.hash)
}

pub fn node_token(domain: u32, zid: &str, node: &str) -> String {
    format!("@ros2_lv/{domain}/{zid}/0/0/NN/%/%/{node}")
}

pub fn entity_token(domain: u32, zid: &str, entity: u32, kind: EntityKind, node: &str, topic: &str, ty: RosType) -> String {
    format!(
        "@ros2_lv/{domain}/{zid}/0/{entity}/{}/%/%/{node}/%{}/{}/{}/{ENTITY_QOS}",
        kind.tag(),
        mangle(topic_name(topic)),
        ty.name,
        ty.hash
    )
}

pub fn attachment(sequence: i64, timestamp_ns: i64, gid: &[u8; 16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(33);
    out.extend_from_slice(&sequence.to_le_bytes());
    out.extend_from_slice(&timestamp_ns.to_le_bytes());
    out.push(gid.len() as u8);
    out.extend_from_slice(gid);
    out
}

struct Outlet {
    publisher: Publisher<'static>,
    sequence: i64,
}

pub struct Link {
    session: Session,
    domain: u32,
    node: String,
    zid: String,
    gid: [u8; 16],
    next_entity: u32,
    tokens: Vec<LivelinessToken>,
    outlets: HashMap<String, Outlet>,
    subscribers: Vec<Subscriber<()>>,
}

const CONNECT_TIMEOUT_MS: u64 = 120_000;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

impl Link {
    pub fn open(endpoint: &str, domain: u32, node: &str) -> Result<Link, String> {
        let mut config = Config::default();
        config.insert_json5("mode", "\"client\"").map_err(err)?;
        config.insert_json5("connect/endpoints", &format!("[\"{endpoint}\"]")).map_err(err)?;
        config.insert_json5("scouting/multicast/enabled", "false").map_err(err)?;
        config.insert_json5("connect/timeout_ms", &CONNECT_TIMEOUT_MS.to_string()).map_err(err)?;
        eprintln!("anews: connecting to the zenoh router at {endpoint} (waiting up to {} s)", CONNECT_TIMEOUT_MS / 1000);
        let session = zenoh::open(config).wait().map_err(|e| format!("zenoh session to {endpoint}: {e}"))?;
        let zid = session.zid().to_string();
        let mut link = Link {
            session,
            domain,
            node: node.to_string(),
            zid,
            gid: rand::random(),
            next_entity: 1,
            tokens: Vec::new(),
            outlets: HashMap::new(),
            subscribers: Vec::new(),
        };
        link.token(node_token(domain, &link.zid, node))?;
        Ok(link)
    }

    fn token(&mut self, key: String) -> Result<(), String> {
        let token = self.session.liveliness().declare_token(key).wait().map_err(err)?;
        self.tokens.push(token);
        Ok(())
    }

    fn entity(&mut self, kind: EntityKind, topic: &str, ty: RosType) -> Result<(), String> {
        let eid = self.next_entity;
        self.next_entity += 1;
        let key = entity_token(self.domain, &self.zid, eid, kind, &self.node, topic, ty);
        self.token(key)
    }

    pub fn advertise(&mut self, topic: &str, ty: RosType) -> Result<(), String> {
        self.entity(EntityKind::Publisher, topic, ty)?;
        let key = data_key(self.domain, topic, ty);
        let publisher = self.session.declare_publisher(key.clone()).wait().map_err(err)?;
        self.outlets.insert(key, Outlet { publisher, sequence: 0 });
        Ok(())
    }

    pub fn publish(&mut self, topic: &str, ty: RosType, payload: Vec<u8>) -> Result<(), String> {
        let key = data_key(self.domain, topic, ty);
        let outlet = self.outlets.get_mut(&key).ok_or_else(|| format!("{topic} is not advertised"))?;
        outlet.sequence += 1;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0);
        outlet.publisher.put(payload).attachment(attachment(outlet.sequence, now, &self.gid)).wait().map_err(err)
    }

    pub fn subscribe<F>(&mut self, topic: &str, ty: RosType, handler: F) -> Result<(), String>
    where
        F: Fn(Vec<u8>) + Send + Sync + 'static,
    {
        self.entity(EntityKind::Subscriber, topic, ty)?;
        let sub = self
            .session
            .declare_subscriber(data_key(self.domain, topic, ty))
            .callback(move |sample| handler(sample.payload().to_bytes().into_owned()))
            .wait()
            .map_err(err)?;
        self.subscribers.push(sub);
        Ok(())
    }

    pub fn close(self) -> Result<(), String> {
        for s in self.subscribers {
            s.undeclare().wait().map_err(err)?;
        }
        for t in self.tokens {
            t.undeclare().wait().map_err(err)?;
        }
        self.session.close().wait().map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_keys_match_the_typescript_link() {
        assert_eq!(
            data_key(0, "/scan", types::LASER_SCAN),
            "0/scan/sensor_msgs::msg::dds_::LaserScan_/RIHS01_64c191398013af96509d518dac71d5164f9382553fce5c1f8cca5be7924bd828"
        );
        assert_eq!(
            data_key(3, "/mobile_base/commands/velocity", types::TWIST),
            "3/mobile_base/commands/velocity/geometry_msgs::msg::dds_::Twist_/RIHS01_9c45bf16fe0983d80e3cfe750d6835843d265a9a6c46bd2e609fcddde6fb8d2a"
        );
    }

    #[test]
    fn liveliness_tokens_follow_rmw_zenoh_layout() {
        assert_eq!(node_token(0, "abc123", "anews_chassis"), "@ros2_lv/0/abc123/0/0/NN/%/%/anews_chassis");
        assert_eq!(
            entity_token(0, "abc123", 7, EntityKind::Publisher, "anews_chassis", "/camera/depth/points_stvl", types::POINT_CLOUD2),
            "@ros2_lv/0/abc123/0/7/MP/%/%/anews_chassis/%camera%depth%points_stvl/sensor_msgs::msg::dds_::PointCloud2_/RIHS01_9198cabf7da3796ae6fe19c4cb3bdd3525492988c70522628af5daa124bae2b5/::,10:,:,:,,"
        );
        assert_eq!(
            entity_token(0, "abc123", 8, EntityKind::Subscriber, "anews_chassis", "/plan", types::PATH),
            "@ros2_lv/0/abc123/0/8/MS/%/%/anews_chassis/%plan/nav_msgs::msg::dds_::Path_/RIHS01_1957a5bb3cee5da65c4e52e52b65a93df227efce4c20f8458b36e73066ca334b/::,10:,:,:,,"
        );
    }

    #[test]
    fn attachment_matches_the_typescript_link() {
        let gid: [u8; 16] = std::array::from_fn(|i| (i * 7) as u8);
        let hex: String = attachment(42, 1_793_854_588_123_456_789, &gid).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "2a0000000000000015a5cd5f3e0de5181000070e151c232a31383f464d545b6269");
    }
}
