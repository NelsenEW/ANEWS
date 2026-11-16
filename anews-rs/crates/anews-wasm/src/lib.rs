use anews_core::depth_sensor::{parse_depth_sensor, DepthSensor, Rng};
use anews_core::fidelity::{build_worlds, scene_items_world, scene_world, Base, FidelityRun, MeshInputs};
use anews_core::occupancy::{parse_map_meta, OccupancyMap};
use anews_core::replay::{import_trip, to_jsonl};
use anews_core::scene::build_scene_glb;
use anews_core::sensor3d::{mat_vec, quat_matrix};
use anews_core::trip::{read_trip, Mount, Trip};
use anews_core::venue::{parse_venue, scene_solids, Solid, Venue};
use serde_json::json;
use wasm_bindgen::prelude::*;

fn js(e: String) -> JsError {
    JsError::new(&e)
}

#[wasm_bindgen]
pub struct Session {
    contact: Vec<(String, anews_core::world3d::World3d)>,
    venue: Venue,
    map: Option<OccupancyMap>,
    run: FidelityRun,
    trip: Option<Trip>,
    rng: std::cell::RefCell<Rng>,
    crowd: Vec<(String, anews_core::crowd::CrowdTimeline)>,
    people: Vec<(String, anews_core::actors::ActorTimeline)>,
    overlay: Option<(anews_core::crowd::CrowdTimeline, anews_core::crowd::CrowdGrid)>,
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new(
        venue_yaml: &str,
        map_yaml: Option<String>,
        map_pgm: Option<Vec<u8>>,
        mesh_full: Option<Vec<u8>>,
        mesh_structure: Option<Vec<u8>>,
        depth_sensor_yaml: &str,
    ) -> Result<Session, JsError> {
        let sensor: DepthSensor = parse_depth_sensor(depth_sensor_yaml).map_err(js)?;
        let venue = parse_venue(venue_yaml).map_err(js)?;
        let map = match (map_yaml, map_pgm) {
            (Some(yaml), Some(pgm)) => Some(OccupancyMap::from_pgm(parse_map_meta(&yaml).map_err(js)?, &pgm).map_err(js)?),
            _ => None,
        };
        let meshes = MeshInputs { full: mesh_full.as_deref(), structure: mesh_structure.as_deref() };
        let worlds = build_worlds(&venue, map.as_ref(), &meshes).map_err(js)?;
        Ok(Session { contact: Vec::new(), venue, map, run: FidelityRun::new(worlds, sensor), trip: None, rng: std::cell::RefCell::new(Rng::new(1)), crowd: Vec::new(), people: Vec::new(), overlay: None })
    }

    pub fn add_scene_world(&mut self, name: &str, scene: &str) -> Result<(), JsError> {
        let world = scene_world(&self.venue, self.map.as_ref(), name, scene).map_err(js)?;
        self.run.add_world(world);
        let contact = scene_items_world(&self.venue, scene).map_err(js)?;
        self.contact.retain(|(n, _)| n != name);
        self.contact.push((name.to_string(), contact));
        Ok(())
    }

    pub fn add_crowd(&mut self, world: &str, timeline_json: &str) -> Result<(), JsError> {
        let timeline: anews_core::crowd::CrowdTimeline = serde_json::from_str(timeline_json).map_err(|e| js(e.to_string()))?;
        let w = self.run.worlds.iter_mut().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        let mut grid = timeline.grid();
        timeline.apply(timeline.start, 0.0, &mut grid);
        w.world.crowd = Some(grid);
        self.crowd.retain(|(n, _)| n != world);
        self.crowd.push((world.to_string(), timeline));
        Ok(())
    }

    pub fn set_crowd_time(&mut self, world: &str, t: f64) -> Result<(), JsError> {
        let Some((_, timeline)) = self.crowd.iter().find(|(n, _)| n == world) else { return Ok(()) };
        let w = self.run.worlds.iter_mut().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        if let Some(grid) = w.world.crowd.as_mut() {
            timeline.apply(t, 0.0, grid);
        }
        Ok(())
    }

    pub fn add_people(&mut self, world: &str, timeline_json: &str) -> Result<(), JsError> {
        let timeline: anews_core::actors::ActorTimeline = serde_json::from_str(timeline_json).map_err(|e| js(e.to_string()))?;
        let w = self.run.worlds.iter_mut().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        let mut set = anews_core::actors::ActorSet::default();
        timeline.apply(timeline.start, &mut set);
        w.world.actors = Some(set);
        self.people.retain(|(n, _)| n != world);
        self.people.push((world.to_string(), timeline));
        Ok(())
    }

    pub fn set_people_time(&mut self, world: &str, t: f64) -> Result<(), JsError> {
        let Some((_, timeline)) = self.people.iter().find(|(n, _)| n == world) else { return Ok(()) };
        let w = self.run.worlds.iter_mut().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        timeline.apply(t, w.world.actors.get_or_insert_with(Default::default));
        Ok(())
    }

    pub fn people_poses(&self, world: &str, t: f64) -> Vec<f32> {
        let Some((_, timeline)) = self.people.iter().find(|(n, _)| n == world) else { return Vec::new() };
        timeline.poses(t).into_iter().flat_map(|p| [p.index as f32, p.x as f32, p.y as f32, p.yaw as f32]).collect()
    }

    pub fn set_cell_overlay(&mut self, timeline_json: &str) -> Result<(), JsError> {
        let timeline: anews_core::crowd::CrowdTimeline = serde_json::from_str(timeline_json).map_err(|e| js(e.to_string()))?;
        let grid = timeline.grid();
        self.overlay = Some((timeline, grid));
        Ok(())
    }

    pub fn cell_overlay(&mut self, t: f64) -> Vec<f32> {
        let Some((timeline, grid)) = self.overlay.as_mut() else { return Vec::new() };
        timeline.apply(t, 0.0, grid);
        grid.occupied()
            .into_iter()
            .flat_map(|(i, f)| {
                let c = grid.frame.center(i);
                [c[0] as f32, c[1] as f32, f as f32]
            })
            .collect()
    }

    pub fn crowd_cells(&self, world: &str) -> Vec<f32> {
        let Some(grid) = self.run.worlds.iter().find(|w| w.name == world).and_then(|w| w.world.crowd.as_ref()) else { return Vec::new() };
        grid.occupied()
            .into_iter()
            .flat_map(|(i, f)| {
                let c = grid.frame.center(i);
                [c[0] as f32, c[1] as f32, f as f32]
            })
            .collect()
    }

    pub fn live_lidar(&self, world: &str, robot: &LiveRobot) -> Result<Vec<f32>, JsError> {
        let w = self.run.worlds.iter().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        Ok(w.lidar(&robot.base(), &robot.lidar_mount).into_iter().map(|r| r as f32).collect())
    }

    pub fn live_step(&self, world: &str, robot: &mut LiveRobot, linear: f64, angular: f64, dt: f64) -> Result<Vec<f64>, JsError> {
        let (_, contact) = self.contact.iter().find(|(n, _)| n == world).ok_or_else(|| js(format!("no contact world for {world}")))?;
        let live = self.run.worlds.iter().find(|w| w.name == world);
        let crowd = live.and_then(|w| w.world.crowd.as_ref());
        let actors = live.and_then(|w| w.world.actors.as_ref());
        robot.body.step(contact, crowd, actors, linear, angular, dt);
        let mut out = robot.pose();
        out.push(robot.body.bumps as f64);
        out.push(robot.body.crowd_bumps as f64);
        Ok(out)
    }

    pub fn live_depth(&self, world: &str, robot: &LiveRobot) -> Result<Vec<f32>, JsError> {
        let w = self.run.worlds.iter().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        let rays = self.run.sensor.camera.rays();
        let mut rng = self.rng.borrow_mut();
        let points = w.depth_points(&robot.base(), &robot.camera_mount, &self.run.sensor, &rays, Some(&mut rng));
        let filtered = anews_core::fidelity::voxel_centroids(&points, self.run.sensor.pipeline.voxel_leaf_m);
        Ok(filtered.into_iter().flatten().map(|v| v as f32).collect())
    }

    pub fn model_files(&self) -> Vec<String> {
        self.venue.model_files()
    }

    pub fn scene_glb(&self, scene: Option<String>, model_files: Vec<String>, model_bytes: Vec<js_sys::Uint8Array>) -> Result<Vec<u8>, JsError> {
        let models = model_files.into_iter().zip(model_bytes.iter().map(|b| b.to_vec())).collect();
        let (glb, _) = build_scene_glb(&self.venue, scene.as_deref(), self.map.as_ref(), &models).map_err(js)?;
        Ok(glb)
    }

    pub fn worlds(&self) -> Vec<String> {
        self.run.worlds.iter().map(|w| w.name.clone()).collect()
    }

    pub fn solids(&self, scene: Option<String>) -> Result<String, JsError> {
        let solids = scene_solids(&self.venue, scene.as_deref()).map_err(js)?;
        let out: Vec<serde_json::Value> = solids
            .into_iter()
            .map(|(id, solid)| match solid {
                Solid::Box { center, half, yaw } => json!({"id": id, "shape": "box", "center": center, "half": half, "yaw": yaw}),
                Solid::Cylinder { center, radius, z0, z1 } => {
                    json!({"id": id, "shape": "cylinder", "center": center, "radius": radius, "z0": z0, "z1": z1})
                }
            })
            .collect();
        Ok(serde_json::Value::Array(out).to_string())
    }

    pub fn load_trip(&mut self, name: &str, mcap: &[u8], max_range: f64) -> Result<String, JsError> {
        let trip = read_trip(mcap).map_err(js)?;
        let events = import_trip(name, &trip, max_range).map_err(js)?;
        self.trip = Some(trip);
        Ok(to_jsonl(&events))
    }

    pub fn score_trip(&mut self, name: &str) -> Result<String, JsError> {
        let trip = self.trip.as_ref().ok_or_else(|| js("load a trip first".into()))?;
        let result = self.run.add_trip(name, trip).map_err(js)?;
        Ok(serde_json::to_string(&result).unwrap())
    }

    pub fn result(&self) -> String {
        serde_json::to_string(&self.run.result()).unwrap()
    }

    pub fn lidar_scan_at(&self, world: &str, t_unix: f64) -> Result<Vec<f32>, JsError> {
        let trip = self.trip.as_ref().ok_or_else(|| js("load a trip first".into()))?;
        let w = self.run.worlds.iter().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        let (pose, _) = trip.pose_at(t_unix).ok_or_else(|| js("trip has too few poses".into()))?;
        let mount = trip.mount("laser_link", "base_footprint").map_err(js)?;
        Ok(w.lidar(&Base::planar(pose.x, pose.y, pose.yaw), &mount).into_iter().map(|r| r as f32).collect())
    }

    pub fn simulate_at(&self, world: &str, t_unix: f64) -> Result<String, JsError> {
        let trip = self.trip.as_ref().ok_or_else(|| js("load a trip first".into()))?;
        let w = self.run.worlds.iter().find(|w| w.name == world).ok_or_else(|| js(format!("unknown world {world}")))?;
        let (pose, _) = trip.pose_at(t_unix).ok_or_else(|| js("trip has too few poses".into()))?;
        let base = Base::planar(pose.x, pose.y, pose.yaw);
        let lidar_mount = trip.mount("laser_link", "base_footprint").map_err(js)?;
        let cam_mount = trip.mount("camera_link", "base_footprint").map_err(js)?;
        let lidar_origin = {
            let m = mat_vec(&base.rot, lidar_mount.xyz);
            [base.origin[0] + m[0], base.origin[1] + m[1], m[2]]
        };
        let ranges = w.lidar(&base, &lidar_mount);
        let lidar_rot = anews_core::trip::mat_mul(&base.rot, &lidar_mount.rot);
        let n = ranges.len();
        let lidar: Vec<[f64; 3]> = ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_finite() && **r < 8.0)
            .map(|(i, r)| {
                let a = anews_core::fidelity::LIDAR_ANGLE_MIN
                    + (anews_core::fidelity::LIDAR_ANGLE_MAX - anews_core::fidelity::LIDAR_ANGLE_MIN) * i as f64 / (n - 1) as f64;
                let d = mat_vec(&lidar_rot, [r * a.cos(), r * a.sin(), 0.0]);
                [lidar_origin[0] + d[0], lidar_origin[1] + d[1], lidar_origin[2] + d[2]]
            })
            .collect();
        let rays = self.run.sensor.camera.rays();
        let mut rng = self.rng.borrow_mut();
        let depth: Vec<[f64; 3]> = w
            .depth_points(&base, &cam_mount, &self.run.sensor, &rays, Some(&mut rng))
            .into_iter()
            .step_by(4)
            .map(|p| {
                let q = mat_vec(&base.rot, p);
                [base.origin[0] + q[0], base.origin[1] + q[1], q[2]]
            })
            .collect();
        Ok(json!({"world": world, "robot": [pose.x, pose.y, pose.yaw], "lidar": lidar, "depth": depth}).to_string())
    }
}

fn mount_from(values: &[f64]) -> Result<Mount, JsError> {
    if values.len() != 7 {
        return Err(js(format!("a mount is [x, y, z, qx, qy, qz, qw]; got {} values", values.len())));
    }
    Ok(Mount { xyz: [values[0], values[1], values[2]], rot: quat_matrix([values[3], values[4], values[5], values[6]]) })
}

#[wasm_bindgen]
pub struct LiveRobot {
    body: anews_core::live::LiveRobot,
    lidar_mount: Mount,
    camera_mount: Mount,
}

#[wasm_bindgen]
impl LiveRobot {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, yaw: f64, lidar_mount: Vec<f64>, camera_mount: Vec<f64>, footprint: Vec<f64>) -> Result<LiveRobot, JsError> {
        if footprint.len() != 2 {
            return Err(js("footprint is [half_length, half_width]".into()));
        }
        Ok(LiveRobot {
            body: anews_core::live::LiveRobot::new(x, y, yaw, [footprint[0], footprint[1]]),
            lidar_mount: mount_from(&lidar_mount)?,
            camera_mount: mount_from(&camera_mount)?,
        })
    }

    pub fn step(&mut self, linear: f64, angular: f64, dt: f64) -> Vec<f64> {
        self.body.drive.update(linear, angular, dt);
        self.body.linear = linear;
        self.body.angular = angular;
        self.pose()
    }

    pub fn pose(&self) -> Vec<f64> {
        let (x, y, yaw) = self.body.pose();
        vec![x, y, yaw, self.body.linear, self.body.angular]
    }

    pub fn reset(&mut self, x: f64, y: f64, yaw: f64) {
        self.body.reset(x, y, yaw);
    }
}

impl LiveRobot {
    fn base(&self) -> Base {
        let (x, y, yaw) = self.body.pose();
        Base::planar(x, y, yaw)
    }
}
