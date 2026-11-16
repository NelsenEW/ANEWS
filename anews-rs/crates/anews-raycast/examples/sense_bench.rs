use anews_core::depth_sensor::parse_depth_sensor;
use anews_core::fidelity::scene_world;
use anews_core::occupancy::{parse_map_meta, OccupancyMap};
use anews_core::sensor3d::quat_matrix;
use anews_core::sensor_batch::{sense_batch, Pose2, Rig};
use anews_core::trip::Mount;
use anews_core::venue::parse_venue;
use anews_raycast::{GpuContext, GpuSensors, SimdCaster};
use std::path::PathBuf;
use std::time::Instant;

const ROUNDS: u32 = 20;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let venue_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("venues/demo_cafe"));
    let scene = args.get(2).map(String::as_str).unwrap_or("dinner_rush");
    let robots: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let sensor_path = args.get(4).map(String::as_str).unwrap_or("config/sensors/depth/orbbec_astra_mini_s.yaml");

    let venue = parse_venue(&std::fs::read_to_string(venue_dir.join("venue.yaml")).unwrap()).unwrap();
    let meta = parse_map_meta(&std::fs::read_to_string(venue_dir.join("maps/default.yaml")).unwrap()).unwrap();
    let pgm = std::fs::read(venue_dir.join("maps").join(&meta.image)).unwrap();
    let map = OccupancyMap::from_pgm(meta, &pgm).unwrap();
    let world = scene_world(&venue, Some(&map), scene, scene).unwrap().world;
    let sensor = parse_depth_sensor(&std::fs::read_to_string(sensor_path).unwrap()).unwrap();
    let rig = Rig::new(
        Mount { xyz: [0.22, 0.0, 0.23], rot: quat_matrix([0.0, 0.0, 0.0, 1.0]) },
        Mount { xyz: [0.31, 0.0, 1.086], rot: quat_matrix([0.0, 0.2231, 0.0, 0.9748]) },
        sensor,
    );
    let free: Vec<(usize, usize)> = (0..map.height).flat_map(|r| (0..map.width).map(move |c| (r, c))).filter(|&(r, c)| map.free[r * map.width + c]).collect();
    let (ox, oy, res) = (map.meta.origin[0], map.meta.origin[1], map.meta.resolution);
    let poses: Vec<Pose2> = (0..robots)
        .map(|i| {
            let (r, c) = free[i * free.len() / robots];
            Pose2 { x: ox + (c as f64 + 0.5) * res, y: oy + ((map.height - 1 - r) as f64 + 0.5) * res, yaw: i as f64 * 2.399 }
        })
        .collect();
    let rays_per_robot = rig.rays(&poses[0]).len();
    println!("scene {scene}: {robots} robots x {rays_per_robot} rays (lidar + depth) per sensor tick");

    let report = |name: &str, secs: f64| {
        let mrays = (robots * rays_per_robot) as f64 / secs / 1e6;
        println!("{name:<28} {:>8.2} ms per tick  {mrays:>8.1} Mrays/s  {:>6.0} robots at 20 Hz", secs * 1e3, robots as f64 / secs / 20.0);
    };

    let simd = SimdCaster::new(&world).unwrap();
    sense_batch(&simd, &rig, &poses, 1, 0);
    let t = Instant::now();
    for tick in 0..ROUNDS {
        sense_batch(&simd, &rig, &poses, 1, tick);
    }
    report("cpu simd (all cores)", t.elapsed().as_secs_f64() / ROUNDS as f64);

    match GpuContext::new() {
        Ok(ctx) => {
            let adapter = ctx.adapter().to_string();
            let gpu = GpuSensors::new(ctx, &world, &rig, 1).unwrap();
            gpu.sense(&poses, 0);
            let t = Instant::now();
            for tick in 0..ROUNDS {
                gpu.sense(&poses, tick);
            }
            report(&format!("gpu ({adapter})"), t.elapsed().as_secs_f64() / ROUNDS as f64);
        }
        Err(e) => println!("gpu: unavailable ({e})"),
    }
}
