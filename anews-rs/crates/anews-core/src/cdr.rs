use serde::Serialize;

pub struct Cdr<'a> {
    bytes: &'a [u8],
    pos: usize,
    little: bool,
}

impl<'a> Cdr<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < 4 {
            return Err("CDR payload is shorter than its header".into());
        }
        Ok(Cdr { bytes: &bytes[4..], pos: 0, little: bytes[1] & 1 == 1 })
    }

    fn align(&mut self, n: usize) {
        let rem = self.pos % n;
        if rem != 0 {
            self.pos += n - rem;
        }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.align(N);
        let end = self.pos + N;
        let out: [u8; N] = self.bytes.get(self.pos..end).ok_or("CDR payload is truncated")?.try_into().unwrap();
        self.pos = end;
        Ok(out)
    }

    pub fn i8(&mut self) -> Result<i8, String> {
        Ok(self.take::<1>()?[0] as i8)
    }

    pub fn u32(&mut self) -> Result<u32, String> {
        let b = self.take::<4>()?;
        Ok(if self.little { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }

    pub fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
    }

    pub fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn f64(&mut self) -> Result<f64, String> {
        let b = self.take::<8>()?;
        Ok(f64::from_bits(if self.little { u64::from_le_bytes(b) } else { u64::from_be_bytes(b) }))
    }

    pub fn string(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        let end = self.pos + n;
        let raw = self.bytes.get(self.pos..end).ok_or("CDR string is truncated")?;
        self.pos = end;
        Ok(String::from_utf8_lossy(raw.strip_suffix(&[0]).unwrap_or(raw)).into_owned())
    }

    pub fn f32_seq(&mut self) -> Result<Vec<f32>, String> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.f32()).collect()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Header {
    pub stamp: f64,
    pub frame_id: String,
}

fn header(c: &mut Cdr) -> Result<Header, String> {
    let sec = c.i32()?;
    let nanosec = c.u32()?;
    Ok(Header { stamp: sec as f64 + nanosec as f64 * 1e-9, frame_id: c.string()? })
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Pose {
    pub position: [f64; 3],
    pub orientation: [f64; 4],
}

impl Pose {
    pub fn yaw(&self) -> f64 {
        let [x, y, z, w] = self.orientation;
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
    }
}

fn vec3(c: &mut Cdr) -> Result<[f64; 3], String> {
    Ok([c.f64()?, c.f64()?, c.f64()?])
}

fn quat(c: &mut Cdr) -> Result<[f64; 4], String> {
    Ok([c.f64()?, c.f64()?, c.f64()?, c.f64()?])
}

fn pose(c: &mut Cdr) -> Result<Pose, String> {
    Ok(Pose { position: vec3(c)?, orientation: quat(c)? })
}

#[derive(Debug, Clone, Serialize)]
pub struct LaserScan {
    pub header: Header,
    pub angle_min: f32,
    pub angle_max: f32,
    pub angle_increment: f32,
    pub range_min: f32,
    pub range_max: f32,
    pub ranges: Vec<f32>,
}

pub fn laser_scan(data: &[u8]) -> Result<LaserScan, String> {
    let mut c = Cdr::new(data)?;
    let header = header(&mut c)?;
    let angle_min = c.f32()?;
    let angle_max = c.f32()?;
    let angle_increment = c.f32()?;
    let _time_increment = c.f32()?;
    let _scan_time = c.f32()?;
    let range_min = c.f32()?;
    let range_max = c.f32()?;
    let ranges = c.f32_seq()?;
    Ok(LaserScan { header, angle_min, angle_max, angle_increment, range_min, range_max, ranges })
}

#[derive(Debug, Clone, Serialize)]
pub struct Transform {
    pub parent: String,
    pub child: String,
    pub stamp: f64,
    pub translation: [f64; 3],
    pub rotation: [f64; 4],
}

pub fn tf_message(data: &[u8]) -> Result<Vec<Transform>, String> {
    let mut c = Cdr::new(data)?;
    let n = c.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let h = header(&mut c)?;
        let child = c.string()?;
        out.push(Transform { parent: h.frame_id, child, stamp: h.stamp, translation: vec3(&mut c)?, rotation: quat(&mut c)? });
    }
    Ok(out)
}

pub fn pose_stamped(data: &[u8]) -> Result<(Header, Pose), String> {
    let mut c = Cdr::new(data)?;
    let h = header(&mut c)?;
    Ok((h, pose(&mut c)?))
}

pub fn path(data: &[u8]) -> Result<Vec<Pose>, String> {
    let mut c = Cdr::new(data)?;
    let _h = header(&mut c)?;
    let n = c.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let _ph = header(&mut c)?;
        out.push(pose(&mut c)?);
    }
    Ok(out)
}

pub fn string(data: &[u8]) -> Result<String, String> {
    Cdr::new(data)?.string()
}

pub struct CdrWriter {
    buf: Vec<u8>,
}

impl Default for CdrWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CdrWriter {
    pub fn new() -> Self {
        CdrWriter { buf: vec![0, 1, 0, 0] }
    }

    fn align(&mut self, n: usize) {
        let rem = (self.buf.len() - 4) % n;
        if rem != 0 {
            self.buf.resize(self.buf.len() + n - rem, 0);
        }
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn i8(&mut self, v: i8) {
        self.buf.push(v as u8);
    }

    pub fn u32(&mut self, v: u32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i32(&mut self, v: i32) {
        self.u32(v as u32);
    }

    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }

    pub fn f64(&mut self, v: f64) {
        self.align(8);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn string(&mut self, s: &str) {
        self.u32(s.len() as u32 + 1);
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
    }

    pub fn f64_array(&mut self, values: &[f64]) {
        values.iter().for_each(|v| self.f64(*v));
    }

    pub fn f32_seq(&mut self, values: &[f32]) {
        self.u32(values.len() as u32);
        values.iter().for_each(|v| self.f32(*v));
    }

    pub fn u8_seq(&mut self, values: &[u8]) {
        self.u32(values.len() as u32);
        self.buf.extend_from_slice(values);
    }

    pub fn time(&mut self, t: f64) {
        let sec = t.floor();
        self.i32(sec as i32);
        self.u32((((t - sec) * 1e9).round() % 1e9) as u32);
    }

    pub fn header(&mut self, stamp: f64, frame_id: &str) {
        self.time(stamp);
        self.string(frame_id);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chassis {
    pub stamp: f64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub linear: f64,
    pub angular: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanarTransform {
    pub parent: String,
    pub child: String,
    pub xyz: [f64; 3],
    pub yaw: f64,
}

pub struct ScanOut<'a> {
    pub stamp: f64,
    pub frame_id: &'a str,
    pub angle_min: f32,
    pub angle_max: f32,
    pub angle_increment: f32,
    pub scan_time: f32,
    pub range_min: f32,
    pub range_max: f32,
    pub ranges: &'a [f32],
}

pub fn encode_clock(t: f64) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.time(t);
    w.into_bytes()
}

pub fn encode_laser_scan(m: &ScanOut) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.header(m.stamp, m.frame_id);
    w.f32(m.angle_min);
    w.f32(m.angle_max);
    w.f32(m.angle_increment);
    w.f32(0.0);
    w.f32(m.scan_time);
    w.f32(m.range_min);
    w.f32(m.range_max);
    w.f32_seq(m.ranges);
    w.f32_seq(&[]);
    w.into_bytes()
}

pub fn encode_odometry(s: &Chassis, frame_id: &str, child_frame_id: &str) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.header(s.stamp, frame_id);
    w.string(child_frame_id);
    w.f64_array(&[s.x, s.y, 0.0, 0.0, 0.0, (s.yaw / 2.0).sin(), (s.yaw / 2.0).cos()]);
    w.f64_array(&[0.0; 36]);
    w.f64_array(&[s.linear, 0.0, 0.0, 0.0, 0.0, s.angular]);
    w.f64_array(&[0.0; 36]);
    w.into_bytes()
}

pub fn encode_imu(s: &Chassis, frame_id: &str) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.header(s.stamp, frame_id);
    w.f64_array(&[0.0, 0.0, (s.yaw / 2.0).sin(), (s.yaw / 2.0).cos()]);
    w.f64_array(&[0.0; 9]);
    w.f64_array(&[0.0, 0.0, s.angular]);
    w.f64_array(&[0.0; 9]);
    w.f64_array(&[0.0, 0.0, 9.81]);
    w.f64_array(&[0.0; 9]);
    w.into_bytes()
}

pub fn encode_pose_stamped(stamp: f64, frame_id: &str, x: f64, y: f64, yaw: f64) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.header(stamp, frame_id);
    w.f64_array(&[x, y, 0.0, 0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()]);
    w.into_bytes()
}

pub fn encode_tf(stamp: f64, transforms: &[PlanarTransform]) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.u32(transforms.len() as u32);
    for t in transforms {
        w.header(stamp, &t.parent);
        w.string(&t.child);
        w.f64_array(&[t.xyz[0], t.xyz[1], t.xyz[2], 0.0, 0.0, (t.yaw / 2.0).sin(), (t.yaw / 2.0).cos()]);
    }
    w.into_bytes()
}

pub fn encode_point_cloud(stamp: f64, frame_id: &str, xyz: &[[f32; 3]]) -> Vec<u8> {
    let n = xyz.len() as u32;
    let mut w = CdrWriter::new();
    w.header(stamp, frame_id);
    w.u32(1);
    w.u32(n);
    w.u32(3);
    for (i, name) in ["x", "y", "z"].iter().enumerate() {
        w.string(name);
        w.u32(i as u32 * 4);
        w.u8(7);
        w.u32(1);
    }
    w.u8(0);
    w.u32(12);
    w.u32(12 * n);
    let data: Vec<u8> = xyz.iter().flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes())).collect();
    w.u8_seq(&data);
    w.u8(1);
    w.into_bytes()
}

pub fn encode_string(s: &str) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.string(s);
    w.into_bytes()
}

pub fn twist(data: &[u8]) -> Result<([f64; 3], [f64; 3]), String> {
    let mut c = Cdr::new(data)?;
    Ok((vec3(&mut c)?, vec3(&mut c)?))
}

#[derive(Debug, Clone, PartialEq)]
pub struct NavigationControl {
    pub control: i8,
    pub mode: i8,
    pub xyz: [f32; 3],
    pub goal_name: String,
}

pub fn navigation_control(data: &[u8]) -> Result<NavigationControl, String> {
    let mut c = Cdr::new(data)?;
    let control = c.i8()?;
    let mode = c.i8()?;
    let xyz = [c.f32()?, c.f32()?, c.f32()?];
    Ok(NavigationControl { control, mode, xyz, goal_name: c.string()? })
}

pub fn encode_navigation_control(m: &NavigationControl) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.i8(m.control);
    w.i8(m.mode);
    m.xyz.iter().for_each(|v| w.f32(*v));
    w.string(&m.goal_name);
    w.into_bytes()
}
