#![cfg(feature = "gpu")]

use anews_core::depth_sensor::parse_depth_sensor;
use anews_core::fidelity::scene_world;
use anews_core::occupancy::{parse_map_meta, OccupancyMap};
use anews_core::sensor3d::quat_matrix;
use anews_core::sensor_batch::{sense_batch, Pose2, Rig, RobotScan};
use anews_core::trip::Mount;
use anews_core::venue::parse_venue;
use anews_core::world3d::World3d;
use anews_raycast::{GpuContext, GpuSensors, SimdCaster};
use std::path::PathBuf;

const SCENE: &str = "dinner_rush";
const ROBOTS: usize = 220;
const SEED: u32 = 11;
const TICK: u32 = 42;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn setup() -> (World3d, OccupancyMap, Rig) {
    let dir = repo().join("venues/demo_cafe");
    let venue = parse_venue(&std::fs::read_to_string(dir.join("venue.yaml")).unwrap()).unwrap();
    let meta = parse_map_meta(&std::fs::read_to_string(dir.join("maps/default.yaml")).unwrap()).unwrap();
    let pgm = std::fs::read(dir.join("maps").join(&meta.image)).unwrap();
    let map = OccupancyMap::from_pgm(meta, &pgm).unwrap();
    let world = scene_world(&venue, Some(&map), SCENE, SCENE).unwrap().world;
    let sensor = parse_depth_sensor(&std::fs::read_to_string(repo().join("config/sensors/depth/orbbec_astra_mini_s.yaml")).unwrap()).unwrap();
    let rig = Rig::new(
        Mount { xyz: [0.22, 0.0, 0.2302], rot: quat_matrix([0.0, 0.0, 0.0, 1.0]) },
        Mount { xyz: [0.317, -0.1802, 1.0862], rot: quat_matrix([0.0536, 0.4071, 0.0840, 0.9079]) },
        sensor,
    );
    (world, map, rig)
}

fn poses(map: &OccupancyMap) -> Vec<Pose2> {
    let free: Vec<(usize, usize)> = (0..map.height).flat_map(|r| (0..map.width).map(move |c| (r, c))).filter(|&(r, c)| map.free[r * map.width + c]).collect();
    let (ox, oy, res) = (map.meta.origin[0], map.meta.origin[1], map.meta.resolution);
    (0..ROBOTS)
        .map(|i| {
            let (r, c) = free[i * free.len() / ROBOTS];
            Pose2 { x: ox + (c as f64 + 0.5) * res, y: oy + ((map.height - 1 - r) as f64 + 0.5) * res, yaw: i as f64 * 2.399 }
        })
        .collect()
}

fn differs(a: f64, b: f64, tol: f64) -> bool {
    a.is_finite() != b.is_finite() || (a.is_finite() && (a - b).abs() > tol)
}

#[test]
fn gpu_sensing_matches_the_cpu_reference_for_many_robots() {
    let (world, map, rig) = setup();
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) if std::env::var_os("ANEWS_REQUIRE_GPU").is_none() => return eprintln!("skipping GPU sensing check: {e}"),
        Err(e) => panic!("ANEWS_REQUIRE_GPU is set but the GPU is unavailable: {e}"),
    };
    let poses = poses(&map);
    let gpu = GpuSensors::new(ctx, &world, &rig, SEED).unwrap().sense(&poses, TICK);
    let cpu: Vec<RobotScan> = sense_batch(&SimdCaster::new(&world).unwrap(), &rig, &poses, SEED, TICK);
    assert_eq!(gpu.len(), ROBOTS);

    let lidar: Vec<(f64, f64)> = cpu.iter().zip(&gpu).flat_map(|(c, g)| c.lidar.iter().copied().zip(g.lidar.iter().copied())).collect();
    let lidar_bad = lidar.iter().filter(|(c, g)| differs(*c, *g, 2e-3)).count();
    assert!(lidar_bad * 10_000 <= lidar.len(), "lidar: {lidar_bad} of {} beams differ", lidar.len());

    let depth: Vec<(f64, f64)> = cpu.iter().zip(&gpu).flat_map(|(c, g)| c.depth.iter().copied().zip(g.depth.iter().copied())).collect();
    let returns = depth.iter().filter(|(c, _)| c.is_finite()).count();
    assert!(returns * 4 > depth.len(), "too few depth returns ({returns}) to prove anything");
    let depth_bad = depth.iter().filter(|(c, g)| differs(*c, *g, 2e-3)).count();
    assert!(depth_bad * 1_000 <= depth.len(), "depth: {depth_bad} of {} pixels differ", depth.len());

    let last = ROBOTS - 1;
    let noisy = cpu[last].depth.iter().zip(&gpu[last].depth).filter(|(c, g)| c.is_finite() && g.is_finite()).count();
    let same = cpu[last].depth.iter().zip(&gpu[last].depth).filter(|(c, g)| c.is_finite() && !differs(**c, **g, 2e-3)).count();
    assert!(same * 100 >= noisy * 99, "robot {last} in the second chunk drew different noise: {same} of {noisy} agree");
}
