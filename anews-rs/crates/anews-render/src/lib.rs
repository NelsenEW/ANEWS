pub mod camera;
mod encode;
mod gpu;
pub mod mesh;
mod text;
pub mod timeline;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anews_core::occupancy::OccupancyMap;
use glam::{Mat4, Vec2, Vec3};
use serde_json::{json, Value};

use camera::{layout, robot_local, robot_pose, Camera, EyeMount, Layout, MapView, Viewport};
use gpu::{FlatVertex, Light, OverlayBatch, OverlayInst, PointInst, Renderer, View3d};
use text::{Fonts, Weight};
use timeline::{Frame, Timeline};

pub use encode::nvenc_probe;

pub const CHASE_WALL_SCALE: f32 = 0.42;

pub struct RenderJob {
    pub timeline: Timeline,
    pub scene_glb: Vec<u8>,
    pub map: Option<OccupancyMap>,
    pub robot_glb: Option<Vec<u8>>,
    pub robot_radius: f32,
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub from: Option<f64>,
    pub to: Option<f64>,
    pub label: String,
    pub eye: EyeMount,
    pub font_dir: Option<PathBuf>,
    pub cq: u32,
}

type Rgba = [f32; 4];

fn hex(h: u32, a: f32) -> Rgba {
    [((h >> 16) & 0xff) as f32 / 255.0, ((h >> 8) & 0xff) as f32 / 255.0, (h & 0xff) as f32 / 255.0, a]
}

const BG: u32 = 0x1b2129;
const PANEL: Rgba = [10.0 / 255.0, 14.0 / 255.0, 20.0 / 255.0, 0.75];
const TEXT: u32 = 0xe8edf3;
const BAD: u32 = 0xff7a7a;
const LIDAR: u32 = 0x2fd4ff;
const CAMERA_SCAN: u32 = 0xff9f43;
const PLAN_3D: u32 = 0xffe066;
const PLAN_MAP: u32 = 0x1f6bff;
const GOAL: u32 = 0x2a5bff;
const CROWD: u32 = 0xe8833a;
const CROWD_CELL_M: f32 = 0.05;

fn crowd_band(flags: f32) -> (f32, f32) {
    let f = flags as u32;
    (if f & 1 != 0 { 0.0 } else { 0.35 }, if f & 2 != 0 { 0.55 } else { 0.45 })
}

struct Overlay<'f> {
    fonts: &'f mut Fonts,
    items: Vec<OverlayInst>,
    batches: Vec<OverlayBatch>,
}

impl<'f> Overlay<'f> {
    fn new(fonts: &'f mut Fonts) -> Self {
        Overlay { fonts, items: Vec::new(), batches: vec![OverlayBatch { scissor: None, range: 0..0 }] }
    }

    fn scissor(&mut self, s: Option<Viewport>) {
        let n = self.items.len() as u32;
        self.batches.last_mut().unwrap().range.end = n;
        self.batches.push(OverlayBatch { scissor: s, range: n..n });
    }

    fn finish(mut self) -> (Vec<OverlayInst>, Vec<OverlayBatch>) {
        self.batches.last_mut().unwrap().range.end = self.items.len() as u32;
        (self.items, self.batches)
    }

    fn push(&mut self, bbox: [f32; 4], shape: [f32; 4], color: Rgba, uv: [f32; 4], kind: f32, param: f32) {
        self.items.push(OverlayInst { bbox, shape, color, uv, kind: [kind, param] });
    }

    fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, radius: f32, color: Rgba) {
        self.push([x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0], [x0, y0, x1, y1], color, [0.0; 4], 0.0, radius);
    }

    fn line(&mut self, a: Vec2, b: Vec2, width: f32, color: Rgba) {
        let pad = width / 2.0 + 1.5;
        let (lo, hi) = (a.min(b) - Vec2::splat(pad), a.max(b) + Vec2::splat(pad));
        self.push([lo.x, lo.y, hi.x, hi.y], [a.x, a.y, b.x, b.y], color, [0.0; 4], 1.0, width / 2.0);
    }

    fn polyline(&mut self, pts: &[Vec2], width: f32, color: Rgba) {
        for w in pts.windows(2) {
            self.line(w[0], w[1], width, color);
        }
    }

    fn circle(&mut self, c: Vec2, r: f32, color: Rgba) {
        let p = r + 1.5;
        self.push([c.x - p, c.y - p, c.x + p, c.y + p], [c.x, c.y, r, 0.0], color, [0.0; 4], 2.0, 0.0);
    }

    fn image(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.push([x0, y0, x1, y1], [0.0; 4], [1.0; 4], [0.0, 0.0, 1.0, 1.0], 3.0, 0.0);
    }

    fn text(&mut self, s: &str, x: f32, top: f32, px: f32, weight: Weight, color: Rgba) -> Result<f32, String> {
        for g in self.fonts.layout(s, x, top, px, weight)? {
            self.push([g.x0, g.y0, g.x1, g.y1], [0.0; 4], color, g.uv, 4.0, 0.0);
        }
        Ok(self.fonts.width(s, px, weight))
    }

    fn label(&mut self, s: &str, x: f32, top: f32, px: f32, weight: Weight, anchor_right: bool) -> Result<(), String> {
        let w = self.fonts.width(s, px, weight);
        let pad = px * 0.45;
        let x0 = if anchor_right { x - w - 2.0 * pad } else { x };
        self.rect(x0, top, x0 + w + 2.0 * pad, top + px * 1.45, 5.0, PANEL);
        self.text(s, x0 + pad, top + px * 0.2, px, weight, hex(TEXT, 1.0))?;
        Ok(())
    }
}

fn wall_clock(start_unix: Option<f64>, t: f64) -> String {
    match start_unix {
        Some(s) => {
            let secs = s + t;
            chrono::DateTime::from_timestamp(secs.floor() as i64, ((secs.fract()) * 1e9) as u32)
                .map(|d| d.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
                .unwrap_or_default()
        }
        None => format!("t {t:.1} s"),
    }
}

fn cents(v: f64) -> f64 {
    (v * 100.0).round() / 100.0 + 0.0
}

fn nav_event_text(t: f64, e: Option<(f64, &Value)>) -> Option<String> {
    let (et, data) = e?;
    if t - et > 6.0 {
        return None;
    }
    let name = data["event"].as_str().map(str::to_string).unwrap_or_else(|| data.to_string());
    let still = data["still_sec"].as_f64().map(|s| format!(" (still {s:.1} s)")).unwrap_or_default();
    Some(format!("nav event: {}{still}, {:.1} s ago", name.replace('_', " "), t - et))
}

fn hud(o: &mut Overlay, f: &Frame, tl: &Timeline, lay: &Layout, from: f64, to: f64) -> Result<(), String> {
    let mut lines: Vec<(String, f32, Weight, u32)> = vec![(wall_clock(tl.start_unix, f.t), 26.0, Weight::Bold, 0xffd27a)];
    let goal = match (f.goal_label, f.goal) {
        (Some(l), Some(g)) => format!("Goal: {l} ({:.2}, {:.2})", g[0], g[1]),
        (Some(l), None) => format!("Goal: {l}"),
        (None, Some(g)) => format!("Goal: ({:.2}, {:.2})", g[0], g[1]),
        (None, None) => "Goal: none yet".into(),
    };
    lines.push((goal, 20.0, Weight::Regular, TEXT));
    lines.push((format!("x {:.2}  y {:.2}  yaw {:.0}°", f.pose.x, f.pose.y, f.pose.yaw.to_degrees()), 20.0, Weight::Regular, TEXT));
    lines.push((format!("{:.2} m/s  {:.2} rad/s", cents(f.v), cents(f.w)), 20.0, Weight::Regular, TEXT));
    lines.push((
        match f.plan {
            Some((pt, _)) => format!("last plan {:.1} s ago", f.t - pt),
            None => "no plan yet".into(),
        },
        20.0,
        Weight::Regular,
        TEXT,
    ));
    if f.still_s > 5.0 {
        lines.push((format!("not moved 0.3 m for {:.0} s", f.still_s), 20.0, Weight::Bold, BAD));
    }
    if f.bumps > 0 {
        lines.push((format!("bumped into the venue {}x (motion refused)", f.bumps), 20.0, Weight::Bold, BAD));
    }
    if f.crowd_bumps > 0 {
        lines.push((format!("bumped into replayed people/legs {}x (motion refused)", f.crowd_bumps), 20.0, Weight::Bold, BAD));
    }
    if let Some(s) = nav_event_text(f.t, f.nav_event) {
        lines.push((s, 20.0, Weight::Regular, 0xffd27a));
    }
    if let Some(n) = f.nearest {
        lines.push((format!("nearest obstacle {n:.2} m"), 20.0, Weight::Regular, TEXT));
    }
    lines.push((format!("replay {:.1} s  ({:.1} to {:.1} s)", f.t, from, to), 14.0, Weight::Regular, 0x99aabb));
    let width = lines.iter().map(|(s, px, w, _)| o.fonts.width(s, *px, *w)).fold(400.0f32, f32::max);
    let height: f32 = lines.iter().map(|(_, px, _, _)| px * 1.45).sum();
    let (x0, y1) = (20.0, lay.main.h - 20.0);
    let y0 = y1 - height - 24.0;
    o.rect(x0, y0, x0 + width + 36.0, y1, 6.0, PANEL);
    let mut y = y0 + 12.0;
    for (s, px, w, c) in &lines {
        o.text(s, x0 + 18.0, y + px * 0.2, *px, *w, hex(*c, 1.0))?;
        y += px * 1.45;
    }
    Ok(())
}

fn map_panel(o: &mut Overlay, f: &Frame, map: Option<&OccupancyMap>, lay: &Layout) {
    let vp = lay.map;
    o.rect(vp.x, vp.y, vp.x + vp.w, vp.y + vp.h, 0.0, hex(BG, 1.0));
    o.scissor(Some(vp));
    let mv = MapView::new(vp, [f.pose.x, f.pose.y]);
    if let Some(m) = map {
        let (ox, oy, res) = (m.meta.origin[0], m.meta.origin[1], m.meta.resolution);
        let tl = mv.to_px(ox, oy + m.height as f64 * res);
        let br = mv.to_px(ox + m.width as f64 * res, oy);
        o.image(tl.x, tl.y, br.x, br.y);
    }
    let trail: Vec<Vec2> = f.trail.iter().map(|(_, p)| mv.to_px(p[0], p[1])).chain(std::iter::once(mv.to_px(f.pose.x, f.pose.y))).collect();
    o.polyline(&trail, 2.0, [80.0 / 255.0, 80.0 / 255.0, 80.0 / 255.0, 0.9]);
    let half = CROWD_CELL_M / 2.0 * mv.px_per_m;
    for c in f.crowd {
        let q = mv.to_px(c[0] as f64, c[1] as f64);
        if vp.contains(q) {
            o.rect(q.x - half, q.y - half, q.x + half, q.y + half, 0.0, hex(CROWD, 0.9));
        }
    }
    if let Some(s) = f.lidar {
        for p in &s.points {
            let q = mv.to_px(p[0] as f64, p[1] as f64);
            if vp.contains(q) {
                o.circle(q, 1.8, hex(LIDAR, 1.0));
            }
        }
    }
    if let Some((_, path)) = f.plan {
        let pts: Vec<Vec2> = path.iter().map(|p| mv.to_px(p[0], p[1])).collect();
        o.polyline(&pts, 4.0, hex(PLAN_MAP, 1.0));
    }
    if let Some(g) = f.goal {
        let q = mv.to_px(g[0], g[1]);
        o.rect(q.x - 8.0, q.y - 8.0, q.x + 8.0, q.y + 8.0, 0.0, hex(GOAL, 1.0));
    }
    let c = mv.to_px(f.pose.x, f.pose.y);
    o.circle(c, 0.25 * mv.px_per_m, hex(0x00b000, 1.0));
    let tip = c + Vec2::new(f.pose.yaw.cos() as f32, -f.pose.yaw.sin() as f32) * 0.45 * mv.px_per_m;
    o.line(c, tip, 3.0, hex(0x003300, 1.0));
    o.scissor(None);
}

fn disc(out: &mut Vec<FlatVertex>, c: Vec3, r: f32, inner: Rgba, outer: Rgba) {
    let n = 40;
    for i in 0..n {
        let a0 = i as f32 / n as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
        out.push(FlatVertex { pos: c.to_array(), color: inner });
        out.push(FlatVertex { pos: (c + Vec3::new(a0.cos(), a0.sin(), 0.0) * r).to_array(), color: outer });
        out.push(FlatVertex { pos: (c + Vec3::new(a1.cos(), a1.sin(), 0.0) * r).to_array(), color: outer });
    }
}

fn ribbon(out: &mut Vec<FlatVertex>, path: &[[f64; 2]], z: f32, width: f32, color: Rgba) {
    for w in path.windows(2) {
        let a = Vec3::new(w[0][0] as f32, w[0][1] as f32, z);
        let b = Vec3::new(w[1][0] as f32, w[1][1] as f32, z);
        let d = (b - a).truncate();
        if d.length_squared() < 1e-10 {
            continue;
        }
        let n = Vec2::new(-d.y, d.x).normalize() * (width / 2.0);
        let n = Vec3::new(n.x, n.y, 0.0);
        for p in [a - n, a + n, b + n, a - n, b + n, b - n] {
            out.push(FlatVertex { pos: p.to_array(), color });
        }
    }
}

fn map_rgba(m: &OccupancyMap) -> (u32, u32, Vec<u8>) {
    let (w, h) = (m.width, m.height);
    let mut px = vec![0u8; w * h * 4];
    for row in 0..h {
        let src = h - 1 - row;
        for col in 0..w {
            let i = src * w + col;
            let c: [u8; 3] = if m.occupied[i] {
                [28, 28, 28]
            } else if m.free[i] {
                [255, 255, 255]
            } else {
                [70, 70, 100]
            };
            px[(row * w + col) * 4..(row * w + col) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    (w as u32, h as u32, px)
}

fn light_for(lo: Vec3, hi: Vec3) -> Light {
    let c = Vec3::new((lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0, 0.0);
    let reach = (hi.x - lo.x).max(hi.y - lo.y) / 2.0 + 1.0;
    let sun = c + Vec3::new(reach * 0.35, -reach * 0.5, 14.0);
    let dist = (sun - c).length();
    let view = Mat4::look_at_rh(sun, c, Vec3::Y);
    let proj = Mat4::orthographic_rh(-reach, reach, -reach, reach, 0.5, dist + reach + 5.0);
    Light { view_proj: proj * view, dir: (sun - c).normalize() }
}

pub fn render(job: RenderJob) -> Result<Value, String> {
    let _lock = encode::lock()?;
    let preflight = encode::preflight()?;
    let (from, frames) = job.timeline.window(job.from, job.to, job.fps)?;
    let to = from + frames as f64 / job.fps as f64;

    let loaded = mesh::load_glb(&job.scene_glb, camera::gltf_to_world(), &mesh::venue_look)?;
    let robot = match &job.robot_glb {
        Some(bytes) => {
            let bytes = mesh::without_draco(bytes.clone()).map_err(|e| format!("robot GLB: {e}"))?;
            let r = mesh::load_glb(&bytes, robot_local(), &mesh::robot_look).map_err(|e| format!("robot GLB: {e}"))?;
            let mut m = r.solid;
            m.append(r.floor);
            m
        }
        None => mesh::proxy_robot(job.robot_radius, 1.2),
    };
    let (mut lo, mut hi) = loaded.solid.bounds();
    if !loaded.floor.vertices.is_empty() {
        let (fl, fh) = loaded.floor.bounds();
        lo = lo.min(fl);
        hi = hi.max(fh);
    }
    let (rlo, rhi) = robot.bounds();
    let gpu = gpu::gpu()?;
    let adapter = gpu.adapter.clone();
    let mut fonts = Fonts::new(job.font_dir.as_deref())?;
    let mut r = Renderer::new(
        gpu,
        job.width,
        job.height,
        gpu::Inputs {
            venue_floor: &loaded.floor,
            venue_solid: &loaded.solid,
            robot: &robot,
            light: light_for(lo, hi),
            wall_scales: [CHASE_WALL_SCALE, 1.0],
            map_rgba: job.map.as_ref().map(map_rgba),
        },
    )?;
    let lay = layout(job.width, job.height);
    if let Some(parent) = job.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let monitor = encode::GpuMonitor::start()?;
    let cpu0 = encode::cpu_seconds();
    let started = Instant::now();
    let encoder = encode::Encoder::start(&job.out, job.width, job.height, job.fps, job.cq)?;
    let mut buf = Vec::new();
    let result = (|| -> Result<(), String> {
        for i in 0..frames {
            let t = from + i as f64 / job.fps as f64;
            let f = job.timeline.frame(t);
            let mut points = Vec::new();
            if let Some(s) = f.lidar {
                points.extend(s.points.iter().map(|p| PointInst { center: *p, size: 0.045, color: hex(LIDAR, 1.0), height: 0.0 }));
            }
            if let Some(s) = f.camera {
                points.extend(s.points.iter().map(|p| PointInst { center: *p, size: 0.035, color: hex(CAMERA_SCAN, 1.0), height: 0.0 }));
            }
            points.extend(f.crowd.iter().map(|c| {
                let (z0, z1) = crowd_band(c[2]);
                PointInst { center: [c[0], c[1], (z0 + z1) / 2.0], size: CROWD_CELL_M, color: hex(CROWD, 1.0), height: z1 - z0 }
            }));
            let mut flat = Vec::new();
            disc(&mut flat, Vec3::new(f.pose.x as f32, f.pose.y as f32, 0.006), job.robot_radius * 1.35, [0.0, 0.0, 0.0, 0.45], [0.0, 0.0, 0.0, 0.0]);
            let robot_flat = flat.len() as u32;
            if let Some((_, path)) = f.plan {
                ribbon(&mut flat, path, 0.04, 0.05, hex(PLAN_3D, 1.0));
            }
            if let Some(g) = f.goal {
                let c = Vec3::new(g[0] as f32, g[1] as f32, 0.045);
                disc(&mut flat, c, 0.22, hex(GOAL, 0.95), hex(GOAL, 0.95));
            }
            let n_flat = flat.len() as u32;
            let views = [
                View3d {
                    camera: Camera::chase(f.pose, lay.main.w / lay.main.h),
                    viewport: lay.main,
                    wall_scale: CHASE_WALL_SCALE,
                    robot: true,
                    points: 0..points.len() as u32,
                    flat: 0..n_flat,
                },
                View3d {
                    camera: Camera::robot_eye(f.pose, &job.eye),
                    viewport: lay.eye,
                    wall_scale: 1.0,
                    robot: false,
                    points: 0..0,
                    flat: robot_flat..n_flat,
                },
            ];
            let mut o = Overlay::new(&mut fonts);
            map_panel(&mut o, &f, job.map.as_ref(), &lay);
            o.line(Vec2::new(lay.main.w, 0.0), Vec2::new(lay.main.w, lay.main.h), 2.0, hex(0x0a0e14, 1.0));
            o.line(Vec2::new(lay.eye.x, lay.eye.h), Vec2::new(lay.eye.x + lay.eye.w, lay.eye.h), 2.0, hex(0x0a0e14, 1.0));
            o.label(&job.label, 20.0, 16.0, 30.0, Weight::Bold, false)?;
            o.label("Robot-eye view (3D venue at the recorded pose)", job.width as f32 - 12.0, 8.0, 15.0, Weight::Regular, true)?;
            o.label("Venue map: trail grey, plan blue, lidar cyan, crowd orange", job.width as f32 - 12.0, job.height as f32 - 34.0, 15.0, Weight::Regular, true)?;
            hud(&mut o, &f, &job.timeline, &lay, from, to)?;
            let (items, batches) = o.finish();
            if fonts.dirty {
                r.upload_atlas(&fonts.pixels);
                fonts.dirty = false;
            }
            r.submit(&views, robot_pose(f.pose), &points, &flat, &items, &batches)?;
            while r.in_flight() > r.max_in_flight() {
                r.take(&mut buf)?;
                encoder.push(std::mem::take(&mut buf))?;
            }
        }
        while r.take(&mut buf)? {
            encoder.push(std::mem::take(&mut buf))?;
        }
        Ok(())
    })();
    let finished = encoder.finish();
    let elapsed = started.elapsed().as_secs_f64();
    let gpu_use = monitor.stop();
    result?;
    finished?;
    let cpu = encode::cpu_seconds() - cpu0;
    let clip = frames as f64 / job.fps as f64;
    Ok(json!({
        "ok": true,
        "out": job.out,
        "frames": frames,
        "fps": job.fps,
        "size": [job.width, job.height],
        "window_s": [from, to],
        "clip_s": clip,
        "render_s": elapsed,
        "speed_x_realtime": clip / elapsed,
        "cpu_s": cpu,
        "cpu_pct_of_one_core": 100.0 * cpu / elapsed,
        "peak_rss_mib": encode::peak_rss_mib(),
        "adapter": adapter,
        "encoder": encode::ENCODER,
        "gpu": gpu_use,
        "preflight": preflight,
        "ffprobe": encode::probe_codec(&job.out)?,
        "scene_triangles": (loaded.solid.indices.len() + loaded.floor.indices.len()) / 3,
        "robot_triangles": robot.indices.len() / 3,
        "robot_bounds": [rlo.to_array(), rhi.to_array()],
        "source": job.timeline.source,
    }))
}

pub fn default_label(source: &str) -> String {
    Path::new(source).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "ANEWS replay".into())
}
