use anews_core::depth_sensor::parse_depth_sensor;
use anews_core::fidelity::{scene_world, Base, SimWorld, OPTICAL_QUAT};
use anews_core::occupancy::{parse_map_meta, OccupancyMap};
use anews_core::raycast::{Ray, RayCaster};
use anews_core::sensor3d::{mat_vec, quat_matrix};
use anews_core::trip::{mat_mul, Mount};
use anews_core::venue::parse_venue;
use anews_raycast::SimdCaster;
use std::path::PathBuf;

const SCENE: &str = "dinner_rush";
const POSES: usize = 48;
const TOLERANCE_M: f64 = 2e-3;
const MAX_MISMATCH_FRACTION: f64 = 1e-4;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn dinner_rush() -> (SimWorld, OccupancyMap) {
    let dir = repo().join("venues/demo_cafe");
    let venue = parse_venue(&std::fs::read_to_string(dir.join("venue.yaml")).unwrap()).unwrap();
    let meta = parse_map_meta(&std::fs::read_to_string(dir.join("maps/default.yaml")).unwrap()).unwrap();
    let pgm = std::fs::read(dir.join("maps").join(&meta.image)).unwrap();
    let map = OccupancyMap::from_pgm(meta, &pgm).unwrap();
    (scene_world(&venue, Some(&map), SCENE, SCENE).unwrap(), map)
}

fn poses(map: &OccupancyMap) -> Vec<Base> {
    let free: Vec<(usize, usize)> = (0..map.height).flat_map(|r| (0..map.width).map(move |c| (r, c))).filter(|&(r, c)| map.free[r * map.width + c]).collect();
    let (ox, oy, res) = (map.meta.origin[0], map.meta.origin[1], map.meta.resolution);
    (0..POSES)
        .map(|i| {
            let (r, c) = free[i * free.len() / POSES];
            let x = ox + (c as f64 + 0.5) * res;
            let y = oy + ((map.height - 1 - r) as f64 + 0.5) * res;
            Base::planar(x, y, i as f64 * 2.399)
        })
        .collect()
}

fn lidar_mount() -> Mount {
    Mount { xyz: [0.22, 0.0, 0.2302], rot: quat_matrix([0.0, 0.0, 0.0, 1.0]) }
}

fn sensor_rays(map: &OccupancyMap) -> Vec<Ray> {
    let sensor = parse_depth_sensor(&std::fs::read_to_string(repo().join("config/sensors/depth/orbbec_astra_mini_s.yaml")).unwrap()).unwrap();
    let cam = Mount { xyz: [0.317, -0.1802, 1.0862], rot: quat_matrix([0.0536, 0.4071, 0.0840, 0.9079]) };
    let cam_rot = mat_mul(&cam.rot, &quat_matrix(OPTICAL_QUAT));
    let reach = sensor.camera.range_max_m * 1.5;
    let lidar = lidar_mount();
    let mut rays = Vec::new();
    for base in poses(map) {
        let lo = add(base.origin, mat_vec(&base.rot, lidar.xyz));
        for i in 0..1369 {
            let a = -2.1 + 4.2 * i as f64 / 1368.0;
            rays.push(Ray { origin: lo, dir: mat_vec(&base.rot, [a.cos(), a.sin(), 0.0]), t_max: 30.0 });
        }
        let co = add(base.origin, mat_vec(&base.rot, cam.xyz));
        let wr = mat_mul(&base.rot, &cam_rot);
        for d in sensor.camera.rays() {
            rays.push(Ray { origin: co, dir: mat_vec(&wr, d), t_max: reach });
        }
    }
    rays
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mismatches(a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.is_finite() != y.is_finite() || (x.is_finite() && (*x - *y).abs() > TOLERANCE_M)).count()
}

fn assert_matches_reference(name: &str, caster: &dyn RayCaster) {
    let (world, map) = dinner_rush();
    let rays = sensor_rays(&map);
    let truth = world.world.cast_batch(&rays);
    let hits = truth.iter().filter(|t| t.is_finite()).count();
    assert!(hits > rays.len() / 2, "scene too empty to prove anything: {hits} hits of {}", rays.len());
    let bad = mismatches(&caster.cast_batch(&rays), &truth);
    assert!(
        (bad as f64) <= MAX_MISMATCH_FRACTION * rays.len() as f64,
        "{name}: {bad} of {} ranges differ from the reference by more than {TOLERANCE_M} m",
        rays.len()
    );
}

#[test]
fn simd_backend_matches_reference_on_dinner_rush() {
    let (world, _) = dinner_rush();
    assert_matches_reference("simd", &SimdCaster::new(&world.world).unwrap());
}

struct Sentinel;

impl RayCaster for Sentinel {
    fn cast_batch(&self, rays: &[Ray]) -> Vec<f64> {
        rays.iter().map(|r| r.t_max + 1234.5).collect()
    }
}

#[test]
fn sim_world_sensors_cast_through_the_plugged_in_caster() {
    let (mut world, map) = dinner_rush();
    world.accelerate(|_| Ok(Box::new(Sentinel))).unwrap();
    let base = &poses(&map)[0];
    assert!(world.lidar(base, &lidar_mount()).iter().all(|&r| r == 30.0 + 1234.5));
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_backend_matches_reference_on_dinner_rush() {
    let (world, _) = dinner_rush();
    match anews_raycast::GpuCaster::new(&world.world) {
        Ok(gpu) => assert_matches_reference(gpu.adapter(), &gpu),
        Err(e) if std::env::var_os("ANEWS_REQUIRE_GPU").is_none() => eprintln!("skipping GPU backend check: {e}"),
        Err(e) => panic!("ANEWS_REQUIRE_GPU is set but the GPU backend is unavailable: {e}"),
    }
}
