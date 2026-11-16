use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::timeline::Pose;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Viewport {
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.y >= self.y && p.x <= self.x + self.w && p.y <= self.y + self.h
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct EyeMount {
    pub forward: f64,
    pub height: f64,
    pub pitch_deg: f64,
    pub hfov_deg: f64,
}

impl Default for EyeMount {
    fn default() -> Self {
        EyeMount { forward: 0.25, height: 1.25, pitch_deg: 12.0, hfov_deg: 100.0 }
    }
}

pub const CHASE_BACK_M: f64 = 3.4;
pub const CHASE_HEIGHT_M: f64 = 2.8;
pub const CHASE_LOOK_AHEAD_M: f64 = 1.6;
pub const CHASE_LOOK_HEIGHT_M: f64 = 0.4;
pub const CHASE_FOV_DEG: f32 = 55.0;
pub const EYE_ASPECT: f32 = 4.0 / 3.0;
pub const MAIN_WIDTH_SHARE: f32 = 0.62;
pub const MAP_SPAN_M: f32 = 8.0;

impl Camera {
    pub fn view_proj(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far) * Mat4::look_at_rh(self.eye, self.target, Vec3::Z)
    }

    pub fn project(&self, vp: &Viewport, world: Vec3) -> Option<Vec2> {
        let clip = self.view_proj() * Vec4::new(world.x, world.y, world.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new(vp.x + (ndc.x + 1.0) * 0.5 * vp.w, vp.y + (1.0 - ndc.y) * 0.5 * vp.h))
    }

    pub fn chase(pose: Pose, aspect: f32) -> Camera {
        let (s, c) = pose.yaw.sin_cos();
        let eye = Vec3::new((pose.x - CHASE_BACK_M * c) as f32, (pose.y - CHASE_BACK_M * s) as f32, CHASE_HEIGHT_M as f32);
        let target = Vec3::new(
            (pose.x + CHASE_LOOK_AHEAD_M * c) as f32,
            (pose.y + CHASE_LOOK_AHEAD_M * s) as f32,
            CHASE_LOOK_HEIGHT_M as f32,
        );
        Camera { eye, target, fov_y: CHASE_FOV_DEG.to_radians(), aspect, near: 0.05, far: 200.0 }
    }

    pub fn robot_eye(pose: Pose, mount: &EyeMount) -> Camera {
        let (s, c) = pose.yaw.sin_cos();
        let eye = Vec3::new((pose.x + mount.forward * c) as f32, (pose.y + mount.forward * s) as f32, mount.height as f32);
        let pitch = mount.pitch_deg.to_radians();
        let dir = Vec3::new((c * pitch.cos()) as f32, (s * pitch.cos()) as f32, -pitch.sin() as f32);
        let half_h = (mount.hfov_deg.to_radians() / 2.0).tan() as f32;
        let fov_y = 2.0 * (half_h / EYE_ASPECT).atan();
        Camera { eye, target: eye + dir, fov_y, aspect: EYE_ASPECT, near: 0.05, far: 60.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub main: Viewport,
    pub eye: Viewport,
    pub map: Viewport,
}

pub fn layout(width: u32, height: u32) -> Layout {
    let (w, h) = (width as f32, height as f32);
    let main_w = (w * MAIN_WIDTH_SHARE).round();
    let pip_w = w - main_w;
    let pip_h = (pip_w / EYE_ASPECT).round();
    Layout {
        main: Viewport { x: 0.0, y: 0.0, w: main_w, h },
        eye: Viewport { x: main_w, y: 0.0, w: pip_w, h: pip_h },
        map: Viewport { x: main_w, y: pip_h, w: pip_w, h: h - pip_h },
    }
}

pub fn robot_local() -> Mat4 {
    Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2) * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

pub fn gltf_to_world() -> Mat4 {
    Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

pub fn robot_pose(pose: Pose) -> Mat4 {
    Mat4::from_translation(Vec3::new(pose.x as f32, pose.y as f32, 0.0)) * Mat4::from_rotation_z(pose.yaw as f32)
}

#[derive(Clone, Copy, Debug)]
pub struct MapView {
    pub vp: Viewport,
    pub center: [f64; 2],
    pub px_per_m: f32,
}

impl MapView {
    pub fn new(vp: Viewport, center: [f64; 2]) -> Self {
        MapView { vp, center, px_per_m: vp.w.min(vp.h) / MAP_SPAN_M }
    }

    pub fn to_px(&self, x: f64, y: f64) -> Vec2 {
        let c = self.vp.center();
        Vec2::new(c.x + (x - self.center[0]) as f32 * self.px_per_m, c.y - (y - self.center[1]) as f32 * self.px_per_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 0.5
    }

    #[test]
    fn chase_camera_centres_the_look_ahead_point_and_keeps_robot_left_on_screen_left() {
        let l = layout(1920, 1080);
        let pose = Pose { x: 1.0, y: 2.0, yaw: std::f64::consts::FRAC_PI_2 };
        let cam = Camera::chase(pose, l.main.w / l.main.h);
        let ahead = Vec3::new(1.0, 2.0 + CHASE_LOOK_AHEAD_M as f32, CHASE_LOOK_HEIGHT_M as f32);
        assert!(close(cam.project(&l.main, ahead).unwrap(), l.main.center()));
        let left = cam.project(&l.main, Vec3::new(0.0, 3.6, 0.4)).unwrap();
        let right = cam.project(&l.main, Vec3::new(2.0, 3.6, 0.4)).unwrap();
        assert!(left.x < l.main.center().x && right.x > l.main.center().x, "{left} {right}");
        let robot_base = cam.project(&l.main, Vec3::new(1.0, 2.0, 0.0)).unwrap();
        let robot_head = cam.project(&l.main, Vec3::new(1.0, 2.0, 1.2)).unwrap();
        assert!(robot_head.y < robot_base.y, "up in the world is up on screen");
        assert!(l.main.contains(robot_base) && l.main.contains(robot_head));
    }

    #[test]
    fn robot_eye_looks_along_heading_pitched_down() {
        let l = layout(1920, 1080);
        let mount = EyeMount::default();
        let pose = Pose { x: 0.0, y: 0.0, yaw: std::f64::consts::PI };
        let cam = Camera::robot_eye(pose, &mount);
        let d = 3.0f64;
        let pitch = mount.pitch_deg.to_radians();
        let on_axis = Vec3::new((-mount.forward - d * pitch.cos()) as f32, 0.0, (mount.height - d * pitch.sin()) as f32);
        assert!(close(cam.project(&l.eye, on_axis).unwrap(), l.eye.center()));
        let hfov_edge = Vec3::new(-mount.forward as f32 - 3.0, (3.0 * (mount.hfov_deg.to_radians() / 2.0).tan()) as f32, mount.height as f32);
        let edge = cam.project(&l.eye, hfov_edge).unwrap();
        assert!((edge.x - (l.eye.x + l.eye.w)).abs() < 40.0, "robot right at half the horizontal FOV lands on the right edge: {edge}");
        assert!(cam.project(&l.eye, Vec3::new(5.0, 0.0, 1.0)).is_none(), "points behind the robot are not drawn");
    }

    #[test]
    fn robot_model_faces_its_heading_and_stands_up() {
        let m = robot_pose(Pose { x: 2.0, y: 1.0, yaw: std::f64::consts::FRAC_PI_2 }) * robot_local();
        let front = m.transform_point3(Vec3::new(0.0, 0.0, 0.3));
        let top = m.transform_point3(Vec3::new(0.0, 1.5, 0.0));
        let left_side = m.transform_point3(Vec3::new(0.2, 0.0, 0.0));
        assert!((front - Vec3::new(2.0, 1.3, 0.0)).length() < 1e-5, "{front}");
        assert!((top - Vec3::new(2.0, 1.0, 1.5)).length() < 1e-5, "{top}");
        assert!((left_side - Vec3::new(1.8, 1.0, 0.0)).length() < 1e-5, "model +x is the robot's left: {left_side}");
    }

    #[test]
    fn venue_gltf_axes_map_to_ros_map_frame() {
        let p = gltf_to_world().transform_point3(Vec3::new(3.0, 0.75, -2.0));
        assert!((p - Vec3::new(3.0, 2.0, 0.75)).length() < 1e-5, "{p}");
    }

    #[test]
    fn map_panel_puts_north_up_and_east_right() {
        let l = layout(1920, 1080);
        let mv = MapView::new(l.map, [1.0, 1.0]);
        let c = l.map.center();
        assert!(close(mv.to_px(1.0, 1.0), c));
        assert!(close(mv.to_px(2.0, 1.0), c + Vec2::new(mv.px_per_m, 0.0)));
        assert!(close(mv.to_px(1.0, 2.0), c - Vec2::new(0.0, mv.px_per_m)));
    }
}
