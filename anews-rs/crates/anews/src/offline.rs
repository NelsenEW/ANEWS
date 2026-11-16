use std::path::{Path, PathBuf};

use anews_core::depth_sensor::{parse_depth_sensor, DepthSensor};
use anews_core::crowd::CrowdTimeline;
use anews_core::fidelity::{build_worlds, scene_world, FidelityRun, MeshInputs, SensorSummaries, SimWorld, Summary, WallNear};
use anews_core::actors::ActorTimeline;
use anews_core::object_gate::{budget_objects, gate_objects, GateConfig, Totals};
use anews_core::objects_fit::{fit_objects, ObjectFitConfig};
use anews_core::people::{track_people, TrackConfig};
use anews_core::scene_fit::{crowd_timeline, fit_scene, Evidence, FitConfig};
use anews_raycast::{GpuCaster, GpuContext, SimdCaster};
use anews_core::occupancy::{parse_map_meta, OccupancyMap};
use anews_core::replay::{import_trip, to_jsonl};
use anews_core::scene::build_scene_glb;
use anews_core::trip::read_trip;
use anews_core::venue::{parse_venue, scene_solids, Venue};
use anews_core::world3d::{WallGrid, World3d};
use serde_json::json;

#[derive(clap::Subcommand)]
pub enum VenueCommand {
    Check {
        venue: PathBuf,
        #[arg(long)]
        scene: Option<String>,
    },
    Export {
        venue: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        scene: Option<String>,
    },
    Fit {
        venue: PathBuf,
        mcaps: Vec<PathBuf>,
        #[arg(long)]
        name: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        base_scene: Option<String>,
        #[arg(long)]
        from: Option<f64>,
        #[arg(long)]
        to: Option<f64>,
        #[arg(long, default_value = "3")]
        min_hits: u32,
        #[arg(long, default_value = "0.15")]
        remove_ratio: f64,
        #[arg(long)]
        no_refine_pose: bool,
        #[arg(long)]
        evidence_out: Option<PathBuf>,
        #[arg(long, default_value = "3.0")]
        bucket_s: f64,
    },
    Objects {
        venue: PathBuf,
        mcaps: Vec<PathBuf>,
        #[arg(long)]
        scene: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        from: Option<f64>,
        #[arg(long)]
        to: Option<f64>,
        #[arg(long, default_value = "3.0")]
        bucket_s: f64,
        #[arg(long)]
        min_gain: Option<f64>,
        #[arg(long)]
        min_table_gain: Option<f64>,
        #[arg(long)]
        area_cost: Option<f64>,
        #[arg(long)]
        camera_only_gain: Option<f64>,
        #[arg(long)]
        seen_through_penalty: Option<f64>,
        #[arg(long)]
        gate_depth_sensor: Option<PathBuf>,
        #[arg(long, default_value = "0.0")]
        gate_tolerance: f64,
        #[arg(long)]
        budget_from: Option<PathBuf>,
        #[arg(long)]
        budget_world: Option<String>,
        #[arg(long, value_delimiter = ',', default_values_t = [0.8, 0.7, 0.8, 0.65])]
        budget_bar: Vec<f64>,
        #[arg(long, default_value = "0.005")]
        budget_margin: f64,
        #[arg(long)]
        min_chair_supports: Option<usize>,
    },
}

#[derive(clap::Subcommand)]
pub enum ReplayCommand {
    Crowd {
        venue: PathBuf,
        mcaps: Vec<PathBuf>,
        #[arg(long)]
        scene: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        scene_out: PathBuf,
        #[arg(long)]
        from: Option<f64>,
        #[arg(long)]
        to: Option<f64>,
    },
    People {
        venue: PathBuf,
        mcaps: Vec<PathBuf>,
        #[arg(long)]
        scene: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        cells_out: Option<PathBuf>,
        #[arg(long)]
        scene_out: Option<PathBuf>,
        #[arg(long, default_value = "0.05")]
        person_radius: f64,
        #[arg(long)]
        keep_camera_cells: bool,
        #[arg(long)]
        from: Option<f64>,
        #[arg(long)]
        to: Option<f64>,
    },
    Import {
        mcap: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "8.0")]
        max_range: f64,
    },
}

#[derive(clap::Args)]
pub struct FidelityArgs {
    venue: PathBuf,
    mcaps: Vec<PathBuf>,
    #[arg(long)]
    depth_sensor: PathBuf,
    #[arg(long)]
    scene: Vec<String>,
    #[arg(long)]
    only_scenes: bool,
    #[arg(long)]
    refine_pose: bool,
    #[arg(long)]
    persistence: bool,
    #[arg(long)]
    crowd: Option<PathBuf>,
    #[arg(long)]
    crowd_scene: Option<String>,
    #[arg(long, default_value = "1.0")]
    crowd_lag: f64,
    #[arg(long, default_value = "0.0")]
    crowd_hold: f64,
    #[arg(long)]
    people: Option<PathBuf>,
    #[arg(long)]
    people_scene: Option<String>,
    #[arg(long, default_value = "0.0")]
    people_lag: f64,
    #[arg(long)]
    people_with_crowd: bool,
    #[arg(long)]
    attribute: bool,
    #[arg(long, default_value = "3.0")]
    bucket_s: f64,
    #[arg(long)]
    from: Option<f64>,
    #[arg(long)]
    to: Option<f64>,
    #[arg(long, default_value = "0.8")]
    bar_recall: f64,
    #[arg(long, default_value = "0.7")]
    bar_lidar_precision: f64,
    #[arg(long, default_value = "0.65")]
    bar_camera_precision: f64,
    #[arg(long, default_value = "1")]
    seed: u64,
    #[arg(long)]
    mesh_full: Option<PathBuf>,
    #[arg(long)]
    mesh_structure: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Raycaster::Reference)]
    raycaster: Raycaster,
}

#[derive(clap::Args)]
pub struct RenderArgs {
    input: PathBuf,
    #[arg(long)]
    venue: PathBuf,
    #[arg(long)]
    scene: Option<String>,
    #[arg(long)]
    robot: Option<PathBuf>,
    #[arg(long, default_value = "0.27")]
    robot_radius: f32,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "1920")]
    width: u32,
    #[arg(long, default_value = "1080")]
    height: u32,
    #[arg(long, default_value = "25")]
    fps: u32,
    #[arg(long)]
    from: Option<f64>,
    #[arg(long)]
    to: Option<f64>,
    #[arg(long)]
    label: Option<String>,
    #[arg(long, value_delimiter = ',', default_values_t = [0.25, 1.25, 12.0, 100.0])]
    eye: Vec<f64>,
    #[arg(long, default_value = "8.0")]
    max_range: f64,
    #[arg(long)]
    font_dir: Option<PathBuf>,
    #[arg(long, default_value = "21")]
    cq: u32,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Raycaster {
    Reference,
    Simd,
    Gpu,
}

fn accelerate(worlds: &mut [SimWorld], raycaster: Raycaster) -> Result<String, String> {
    match raycaster {
        Raycaster::Reference => Ok("reference".into()),
        Raycaster::Simd => {
            for w in worlds {
                w.accelerate(|world| Ok(Box::new(SimdCaster::new(world)?)))?;
            }
            Ok("simd".into())
        }
        Raycaster::Gpu => {
            let ctx = GpuContext::new()?;
            for w in worlds {
                w.accelerate(|world| Ok(Box::new(GpuCaster::with_context(ctx.clone(), world)?)))?;
            }
            Ok(format!("gpu: {}", ctx.adapter()))
        }
    }
}

fn window(from: Option<f64>, to: Option<f64>) -> Option<(f64, f64)> {
    (from.is_some() || to.is_some()).then(|| (from.unwrap_or(f64::NEG_INFINITY), to.unwrap_or(f64::INFINITY)))
}

pub fn load_depth_sensor(path: &Path) -> Result<DepthSensor, String> {
    parse_depth_sensor(&read_text(path)?).map_err(|e| format!("{}: {e}", path.display()))
}

fn passes(s: &Summary, recall: f64, precision: f64) -> bool {
    s.recall.is_some_and(|r| r >= recall) && s.precision.is_some_and(|p| p >= precision)
}

fn verdict(totals: &std::collections::BTreeMap<String, SensorSummaries>, recall: f64, lidar_precision: f64, camera_precision: f64) -> serde_json::Value {
    let per_world: serde_json::Map<String, serde_json::Value> = totals
        .iter()
        .map(|(name, s)| {
            let lidar = s.lidar_static.as_ref().unwrap_or(&s.lidar);
            let camera = s.camera_static.as_ref().unwrap_or(&s.camera);
            let (l, c) = (passes(lidar, recall, lidar_precision), passes(camera, recall, camera_precision));
            (name.clone(), json!({"lidar": l, "camera": c, "pass": l && c, "scored_on": if s.lidar_static.is_some() { "persistent returns" } else { "all returns" }}))
        })
        .collect();
    json!({"recall": recall, "lidar_precision": lidar_precision, "camera_precision": camera_precision, "match": "same or adjacent 5 cm cell", "worlds": per_world})
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn load_venue(path: &Path) -> Result<(Venue, Option<OccupancyMap>), String> {
    let mut venue = parse_venue(&read_text(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
    venue.dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let map = match venue.map_path() {
        Some(map_yaml) => {
            let meta = parse_map_meta(&read_text(&map_yaml)?)?;
            let image = map_yaml.parent().unwrap_or(Path::new(".")).join(&meta.image);
            Some(OccupancyMap::from_pgm(meta, &read(&image)?)?)
        }
        None => None,
    };
    Ok((venue, map))
}

fn emit(value: serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(&value).unwrap());
}

pub fn venue(cmd: VenueCommand) -> Result<(), String> {
    match cmd {
        VenueCommand::Check { venue: path, scene } => {
            let (venue, map) = load_venue(&path)?;
            let solids = scene_solids(&venue, scene.as_deref())?;
            let world = World3d::from_venue(&venue, scene.as_deref(), map.as_ref().map(WallGrid::from_map))?;
            let mut kinds = std::collections::BTreeMap::<String, usize>::new();
            for item in venue.scene_items(scene.as_deref())? {
                *kinds.entry(item.type_name).or_default() += 1;
            }
            emit(json!({
                "ok": true,
                "venue": path,
                "scene": scene,
                "scenes": venue.scenes.keys().collect::<Vec<_>>(),
                "items_by_type": kinds,
                "solids": solids.len(),
                "triangles": world.bvh.len(),
                "map": map.as_ref().map(|m| json!({"width": m.width, "height": m.height, "resolution": m.meta.resolution})),
            }));
            Ok(())
        }
        VenueCommand::Fit { venue: path, mcaps, name, out, base_scene, from, to, min_hits, remove_ratio, no_refine_pose, evidence_out, bucket_s } => {
            let (venue, map) = load_venue(&path)?;
            let map = map.ok_or("scene fitting needs the venue map")?;
            let cfg = FitConfig { window: window(from, to), min_hits, remove_ratio, bucket_s, ..FitConfig::default() };
            let refine = (!no_refine_pose).then(|| WallNear::from_map(&map));
            let mut ev = Evidence::new(&map);
            let mut trips = Vec::new();
            for p in &mcaps {
                let trip = read_trip(&read(p)?)?;
                let frames = ev.add_trip(&trip, &cfg, refine.as_ref())?;
                trips.push(json!({"mcap": p, "frames": frames}));
            }
            if let Some(path) = &evidence_out {
                let cells = json!({"columns": ["x", "y", "lidar_hit", "lidar_pass", "camera_hit", "camera_pass"], "cells": ev.nonzero_cells()});
                std::fs::write(path, cells.to_string()).map_err(|e| format!("{}: {e}", path.display()))?;
            }
            let (scene, report) = fit_scene(&venue, &map, base_scene.as_deref(), &ev, &cfg)?;
            let mut doc = serde_yaml::Mapping::new();
            doc.insert(name.clone().into(), serde_yaml::to_value(&scene).map_err(|e| e.to_string())?);
            std::fs::write(&out, serde_yaml::to_string(&doc).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", out.display()))?;
            emit(json!({"ok": true, "scene": name, "out": out, "trips": trips, "report": report}));
            Ok(())
        }
        VenueCommand::Objects { venue: path, mcaps, scene, name, out, from, to, bucket_s, min_gain, min_table_gain, area_cost, camera_only_gain, seen_through_penalty, gate_depth_sensor, gate_tolerance, budget_from, budget_world, budget_bar, budget_margin, min_chair_supports } => {
            let (venue, map) = load_venue(&path)?;
            let map = map.ok_or("the object fit needs the venue map")?;
            let cfg = FitConfig { window: window(from, to), bucket_s, ..FitConfig::default() };
            let mut oc = ObjectFitConfig::default();
            if let Some(v) = min_gain {
                oc.min_gain = v;
            }
            if let Some(v) = min_table_gain {
                oc.min_table_gain = v;
            }
            if let Some(v) = area_cost {
                oc.area_cost = v;
            }
            if let Some(v) = camera_only_gain {
                oc.camera_only_gain = v;
            }
            if let Some(v) = seen_through_penalty {
                oc.seen_through_penalty = v;
            }
            if let Some(v) = min_chair_supports {
                oc.min_chair_supports = v;
            }
            let refine = WallNear::from_map(&map);
            let mut ev = Evidence::new(&map);
            let mut trips = Vec::new();
            for p in &mcaps {
                let trip = read_trip(&read(p)?)?;
                ev.add_trip(&trip, &cfg, Some(&refine))?;
                trips.push(trip);
            }
            let fit = fit_objects(&venue, &map, &scene, &ev, &cfg, &oc)?;
            let mut budget = None;
            let (fitted, gate) = match &gate_depth_sensor {
                Some(sensor) if budget_from.is_some() => {
                    let path = budget_from.as_ref().unwrap();
                    let doc: serde_json::Value = serde_json::from_str(&read_text(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
                    let world = budget_world.as_deref().ok_or("--budget-from needs --budget-world")?;
                    let t = &doc["totals"][world];
                    let get = |k: &str, f: &str| t[k][f].as_f64().ok_or_else(|| format!("{}: totals.{world}.{k}.{f} missing", path.display()));
                    let (lr, lrc, ls, lp) = (get("lidar_static", "real_cells")?, get("lidar_static", "recall")?, get("lidar_static", "sim_cells")?, get("lidar_static", "precision")?);
                    let (cr, crc, cs, cp) = (get("camera_static", "real_cells")?, get("camera_static", "recall")?, get("camera_static", "sim_cells")?, get("camera_static", "precision")?);
                    let totals = Totals { lidar_real: lr, lidar_found: lr * lrc, lidar_sim: ls, lidar_confirmed: ls * lp, camera_real: cr, camera_found: cr * crc, camera_sim: cs, camera_confirmed: cs * cp };
                    let bar: [f64; 4] = budget_bar.clone().try_into().map_err(|_| "--budget-bar takes four values: lidar recall, lidar precision, camera recall, camera precision")?;
                    let (s, c, projected) = budget_objects(&venue, &map, &fit.furniture, &fit.groups, &fit.evidence, &trips, &load_depth_sensor(sensor)?, cfg.window, Some(&refine), &GateConfig::default(), totals, bar, budget_margin)?;
                    budget = Some(json!({"candidates": c, "projected": projected, "bar": bar}));
                    (s, None)
                }
                Some(sensor) => {
                    let gc = GateConfig { tolerance: gate_tolerance, ..GateConfig::default() };
                    let (s, d) = gate_objects(&venue, &map, &fit.furniture, &fit.groups, &fit.evidence, &trips, &load_depth_sensor(sensor)?, cfg.window, Some(&refine), &gc)?;
                    (s, Some(d))
                }
                None => (fit.scene, None),
            };
            let mut kinds = std::collections::BTreeMap::<String, usize>::new();
            for item in &fitted.add {
                *kinds.entry(item.type_name.clone()).or_default() += 1;
            }
            let mut doc = serde_yaml::Mapping::new();
            doc.insert(name.clone().into(), serde_yaml::to_value(&fitted).map_err(|e| e.to_string())?);
            std::fs::write(&out, serde_yaml::to_string(&doc).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", out.display()))?;
            let accepted = gate.as_ref().map(|d| d.iter().filter(|x| x.accepted).count());
            emit(json!({"ok": true, "scene": name, "out": out, "frames": ev.frames, "fit": fit.report, "items_by_type": kinds, "gate_accepted": accepted, "gate": gate, "budget": budget}));
            Ok(())
        }
        VenueCommand::Export { venue: path, out, scene } => {
            let (venue, map) = load_venue(&path)?;
            let mut models = std::collections::BTreeMap::new();
            for file in venue.model_files() {
                models.insert(file.clone(), read(&venue.dir.join(&file))?);
            }
            let (glb, stats) = build_scene_glb(&venue, scene.as_deref(), map.as_ref(), &models)?;
            std::fs::write(&out, &glb).map_err(|e| format!("{}: {e}", out.display()))?;
            emit(json!({"ok": true, "out": out, "bytes": glb.len(), "scene": scene, "stats": stats}));
            Ok(())
        }
    }
}

pub fn replay(cmd: ReplayCommand) -> Result<(), String> {
    match cmd {
        ReplayCommand::Crowd { venue: path, mcaps, scene, out, scene_out, from, to } => {
            let (venue, map) = load_venue(&path)?;
            let map = map.ok_or("a crowd replay needs the venue map")?;
            let mut trips = Vec::new();
            for p in &mcaps {
                trips.push(read_trip(&read(p)?)?);
            }
            trips.sort_by(|a, b| a.scans.first().map(|s| s.t).partial_cmp(&b.scans.first().map(|s| s.t)).unwrap());
            let cfg = FitConfig { window: window(from, to), ..FitConfig::default() };
            let refine = WallNear::from_map(&map);
            let (timeline, furniture, report) = crowd_timeline(&venue, &map, &scene, &trips, &cfg, Some(&refine))?;
            std::fs::write(&out, serde_json::to_string(&timeline).unwrap()).map_err(|e| format!("{}: {e}", out.display()))?;
            let name = format!("{scene}_furniture");
            let mut doc = serde_yaml::Mapping::new();
            doc.insert(name.clone().into(), serde_yaml::to_value(&furniture).map_err(|e| e.to_string())?);
            std::fs::write(&scene_out, serde_yaml::to_string(&doc).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", scene_out.display()))?;
            emit(json!({"ok": true, "out": out, "scene_out": scene_out, "furniture_scene": name, "report": report}));
            Ok(())
        }
        ReplayCommand::People { venue: path, mcaps, scene, out, cells_out, scene_out, person_radius, keep_camera_cells, from, to } => {
            let (venue, map) = load_venue(&path)?;
            let map = map.ok_or("tracking people needs the venue map")?;
            let mut trips = Vec::new();
            for p in &mcaps {
                trips.push(read_trip(&read(p)?)?);
            }
            trips.sort_by(|a, b| a.scans.first().map(|s| s.t).partial_cmp(&b.scans.first().map(|s| s.t)).unwrap());
            let cfg = FitConfig { window: window(from, to), ..FitConfig::default() };
            let refine = WallNear::from_map(&map);
            let (timeline, report) = track_people(&venue, &map, &scene, &trips, &cfg, &TrackConfig::default(), Some(&refine))?;
            std::fs::write(&out, serde_json::to_string(&timeline).unwrap()).map_err(|e| format!("{}: {e}", out.display()))?;
            let mut cells = None;
            if let Some(cells_out) = &cells_out {
                let (crowd, furniture, crowd_report) = crowd_timeline(&venue, &map, &scene, &trips, &cfg, Some(&refine))?;
                let layers = if keep_camera_cells { anews_core::crowd::LOW } else { anews_core::crowd::LOW | anews_core::crowd::HIGH };
                let left = crowd.without_actors(&timeline, person_radius, 0.1, layers);
                std::fs::write(cells_out, serde_json::to_string(&left).unwrap()).map_err(|e| format!("{}: {e}", cells_out.display()))?;
                let name = format!("{scene}_furniture");
                if let Some(scene_out) = &scene_out {
                    let mut doc = serde_yaml::Mapping::new();
                    doc.insert(name.clone().into(), serde_yaml::to_value(&furniture).map_err(|e| e.to_string())?);
                    std::fs::write(scene_out, serde_yaml::to_string(&doc).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", scene_out.display()))?;
                }
                let on = |tl: &CrowdTimeline| tl.tracks.iter().filter(|t| !t.prior).map(|t| t.times.len()).sum::<usize>();
                cells = Some(json!({"out": cells_out, "furniture_scene": name, "crowd": crowd_report, "dynamic_changes_before": on(&crowd), "dynamic_changes_after": on(&left)}));
            }
            emit(json!({"ok": true, "out": out, "report": report, "cells": cells}));
            Ok(())
        }
        ReplayCommand::Import { mcap, out, max_range } => {
            let trip = read_trip(&read(&mcap)?)?;
            let name = mcap.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let events = import_trip(&name, &trip, max_range)?;
            std::fs::write(&out, to_jsonl(&events)).map_err(|e| format!("{}: {e}", out.display()))?;
            let duration = events.last().and_then(|e| e["t"].as_f64()).unwrap_or(0.0);
            emit(json!({
                "ok": true,
                "out": out,
                "events": events.len(),
                "duration_s": duration,
                "poses": trip.poses.len(),
                "lidar_scans": trip.scans.len(),
                "camera_scans": trip.camera_scans.len(),
                "plans": trip.plans.len(),
                "nav_events": trip.nav_events.len(),
            }));
            Ok(())
        }
    }
}

pub fn fidelity(args: FidelityArgs) -> Result<(), String> {
    let (venue, map) = load_venue(&args.venue)?;
    let full = args.mesh_full.as_deref().map(read).transpose()?;
    let structure = args.mesh_structure.as_deref().map(read).transpose()?;
    let meshes = MeshInputs { full: full.as_deref(), structure: structure.as_deref() };
    let mut crowd = Vec::new();
    let mut worlds = if args.only_scenes { Vec::new() } else { build_worlds(&venue, map.as_ref(), &meshes)? };
    for scene in &args.scene {
        worlds.push(scene_world(&venue, map.as_ref(), scene, scene)?);
    }
    if let (Some(path), Some(scene)) = (&args.crowd, &args.crowd_scene) {
        let timeline: CrowdTimeline = serde_json::from_str(&read_text(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
        let name = format!("{scene}+crowd");
        let mut world = scene_world(&venue, map.as_ref(), &name, scene)?;
        world.world.crowd = Some(timeline.grid());
        worlds.push(world);
        crowd.push((name, timeline, args.crowd_lag, args.crowd_hold));
    }
    let mut people = Vec::new();
    if let (Some(path), Some(scene)) = (&args.people, &args.people_scene) {
        let timeline: ActorTimeline = serde_json::from_str(&read_text(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
        if args.people_with_crowd {
            let (name, _, _, _) = crowd.first().ok_or("--people-with-crowd needs --crowd and --crowd-scene")?;
            let name = name.clone();
            let w = worlds.iter_mut().find(|w| w.name == name).unwrap();
            w.world.actors = Some(Default::default());
            w.name = format!("{name}+people");
            crowd[0].0 = w.name.clone();
            people.push((w.name.clone(), timeline, args.people_lag));
        } else {
            let name = format!("{scene}+people");
            let mut world = scene_world(&venue, map.as_ref(), &name, scene)?;
            world.world.actors = Some(Default::default());
            worlds.push(world);
            people.push((name, timeline, args.people_lag));
        }
    }
    let raycaster = accelerate(&mut worlds, args.raycaster)?;
    let mut run = FidelityRun::new(worlds, load_depth_sensor(&args.depth_sensor)?);
    run.crowd = crowd;
    run.people = people;
    if args.attribute {
        run.attribution = Some(Default::default());
    }
    run.seed = args.seed;
    run.window = window(args.from, args.to);
    if args.refine_pose {
        run.refine = Some(WallNear::from_map(map.as_ref().ok_or("--refine-pose needs the venue map")?));
    }
    if args.persistence {
        let map = map.as_ref().ok_or("--persistence needs the venue map")?;
        let cfg = FitConfig { window: run.window, bucket_s: args.bucket_s, ..FitConfig::default() };
        let mut ev = Evidence::new(map);
        for path in &args.mcaps {
            ev.add_trip(&read_trip(&read(path)?)?, &cfg, run.refine.as_ref())?;
        }
        run.persistence = Some(ev);
    }
    for path in &args.mcaps {
        let trip = read_trip(&read(path)?)?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(t) = run.add_trip(&name, &trip)? {
            eprintln!("{} frames={}", t.name, t.frames);
        }
    }
    let summary = run.result();
    let mut result = serde_json::to_value(&summary).unwrap();
    result["bar"] = verdict(&summary.totals, args.bar_recall, args.bar_lidar_precision, args.bar_camera_precision);
    result["depth_camera"] = json!(run.sensor.camera.model);
    result["raycaster"] = json!(raycaster);
    if let Some(attr) = &run.attribution {
        let by_owner: std::collections::BTreeMap<&String, std::collections::BTreeMap<&String, serde_json::Value>> =
            attr.iter().map(|(w, per)| (w, per.iter().map(|(o, [n, ok])| (o, json!({"sim": n, "confirmed": ok}))).collect())).collect();
        result["lidar_by_owner"] = json!(by_owner);
    }
    if let Some(out) = &args.out {
        std::fs::write(out, serde_json::to_string_pretty(&result).unwrap()).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    emit(result);
    Ok(())
}

pub fn finish(result: Result<(), String>) -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = result {
        println!("{}", json!({"ok": false, "error": e}));
        std::process::exit(1);
    }
    Ok(())
}

fn replay_events(input: &Path, max_range: f64) -> Result<Vec<serde_json::Value>, String> {
    if input.extension().is_some_and(|e| e == "mcap") {
        let trip = read_trip(&read(input)?)?;
        let name = input.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        import_trip(&name, &trip, max_range)
    } else {
        anews_render::timeline::parse_jsonl(&read_text(input)?)
    }
}

pub fn render(args: RenderArgs) -> Result<(), String> {
    let [forward, height, pitch_deg, hfov_deg] = args.eye[..] else {
        return Err("--eye takes forward,height,pitch_deg,hfov_deg".into());
    };
    let timeline = anews_render::timeline::Timeline::from_events(&replay_events(&args.input, args.max_range)?)?;
    let (venue, map) = load_venue(&args.venue)?;
    let mut models = std::collections::BTreeMap::new();
    for file in venue.model_files() {
        models.insert(file.clone(), read(&venue.dir.join(&file))?);
    }
    let scene = args.scene.or_else(|| timeline.scene.clone());
    let (scene_glb, _) = build_scene_glb(&venue, scene.as_deref(), map.as_ref(), &models)?;
    let label = args.label.unwrap_or_else(|| anews_render::default_label(&timeline.source));
    let report = anews_render::render(anews_render::RenderJob {
        timeline,
        scene_glb,
        map,
        robot_glb: args.robot.as_deref().map(read).transpose()?,
        robot_radius: args.robot_radius,
        out: args.out,
        width: args.width,
        height: args.height,
        fps: args.fps,
        from: args.from,
        to: args.to,
        label,
        eye: anews_render::camera::EyeMount { forward, height, pitch_deg, hfov_deg },
        font_dir: args.font_dir,
        cq: args.cq,
    })?;
    emit(report);
    Ok(())
}
