use anews_core::sensor3d::*;
use anews_core::venue::*;
use anews_core::world3d::*;


fn venue() -> Venue {
    serde_yaml::from_str(
        r#"
schema: foodlink.venue/1
types:
  table:
    parts:
      - {shape: box, size: [1.0, 1.0, 0.04], z: 0.71, scale_xy: true}
      - {shape: leg4, radius: 0.025, inset: 0.06, z0: 0.0, z1: 0.71}
items:
  - {id: a, type: table, x: 0, y: 0, size: [1.2, 0.7]}
  - {id: b, type: table, x: 5, y: 0}
scenes:
  s:
    move: {a: {x: 1.0}}
    remove: [b]
    add: [{id: c, type: table, x: 9, y: 9}]
"#,
    )
    .unwrap()
}

#[test]
fn scene_moves_removes_and_adds() {
    let v = venue();
    let items = v.scene_items(Some("s")).unwrap();
    let ids: Vec<_> = items.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(ids, ["a", "c"]);
    assert_eq!(items[0].x, 1.0);
    assert_eq!(v.items[0].x, 0.0);
}

#[test]
fn unknown_scene_is_an_error() {
    assert!(venue().scene_items(Some("nope")).is_err());
}

#[test]
fn table_expands_to_top_and_four_legs() {
    let v = venue();
    let solids = expand_item(&v.items[0], &v.types["table"]);
    assert_eq!(solids.len(), 5);
    match &solids[0] {
        Solid::Box { center, half, .. } => {
            assert!((center[2] - 0.73).abs() < 1e-9);
            assert!((half[0] - 0.6).abs() < 1e-9 && (half[1] - 0.35).abs() < 1e-9);
        }
        other => panic!("expected box, got {other:?}"),
    }
}


fn table_world(yaw_deg: f64) -> World3d {
    let venue: Venue = serde_yaml::from_str(&format!(
        r#"
schema: foodlink.venue/1
types:
  table:
    parts:
      - {{shape: box, size: [1.0, 1.0, 0.04], z: 0.71, scale_xy: true}}
      - {{shape: leg4, radius: 0.025, inset: 0.06, z0: 0.0, z1: 0.71}}
items:
  - {{id: t, type: table, x: 2.0, y: 0.0, yaw_deg: {yaw_deg}, size: [1.2, 0.7]}}
"#
    ))
    .unwrap();
    assert_eq!(expand_item(&venue.items[0], &venue.types["table"]).len(), 5);
    World3d::from_venue(&venue, None, None).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn low_lidar_passes_between_legs() {
    assert!(table_world(0.0).cast([0.0, 0.0, 0.23], [1.0, 0.0, 0.0], 10.0).is_infinite());
}

#[test]
fn low_lidar_hits_a_leg() {
    let r = table_world(0.0).cast([0.0, 0.35 - 0.06, 0.23], [1.0, 0.0, 0.0], 10.0);
    assert!(close(r, 2.0 - 0.6 + 0.06 - 0.025, 0.002), "{r}");
}

#[test]
fn tabletop_height_hits_edge_and_rotation_applies() {
    assert!(close(table_world(0.0).cast([0.0, 0.0, 0.73], [1.0, 0.0, 0.0], 10.0), 1.4, 1e-6));
    assert!(close(table_world(90.0).cast([0.0, 0.0, 0.73], [1.0, 0.0, 0.0], 10.0), 1.65, 1e-6));
}

#[test]
fn downward_ray_sees_top_then_floor_beside() {
    let w = table_world(0.0);
    assert!(close(w.cast([2.0, 0.0, 1.5], [0.0, 0.0, -1.0], 10.0), 0.75, 1e-6));
    assert!(close(w.cast([3.0, 0.0, 1.5], [0.0, 0.0, -1.0], 10.0), 1.5, 1e-6));
}

#[test]
fn range_limit_returns_infinity() {
    assert!(table_world(0.0).cast([0.0, 0.0, 0.73], [1.0, 0.0, 0.0], 1.0).is_infinite());
}

#[test]
fn bvh_matches_brute_force() {
    let mut tris = Vec::new();
    for k in 0..60 {
        let x = (k % 10) as f64 * 0.7 - 3.0;
        let y = (k / 10) as f64 * 0.9 - 2.5;
        box_triangles([x, y, 0.4], [0.15, 0.1, 0.4], k as f64 * 0.3, k, &mut tris);
    }
    let bvh = Bvh::new(tris.clone());
    for i in 0..400 {
        let a = i as f64 * 0.0157;
        let d = [a.cos(), a.sin(), -0.05 + (i % 7) as f64 * 0.02];
        let o = [0.1, -0.2, 0.3];
        let brute = tris.iter().filter_map(|t| t.hit(o, d)).fold(f64::INFINITY, f64::min);
        let fast = bvh.cast(o, d, f64::INFINITY).map(|h| h.0).unwrap_or(f64::INFINITY);
        assert!(brute == fast || (brute - fast).abs() < 1e-9, "ray {i}: {brute} vs {fast}");
    }
}


#[test]
fn depth_rays_point_along_optical_z() {
    let cfg: Sensor3dConfig = serde_yaml::from_str(
        "{name: d, topic: /d, frame_id: f, rate_hz: 10, range_min: 0.1, range_max: 5, \
         mount: {xyz: [0, 0, 1], quat_xyzw: [0, 0, 0, 1]}, type: depth, width: 4, height: 2, hfov_deg: 60, vfov_deg: 30}",
    )
    .unwrap();
    let rays = cfg.local_rays();
    assert_eq!(rays.len(), 8);
    assert!(rays.iter().all(|r| r[2] > 0.8));
    assert!(rays[0][0] < 0.0 && rays[3][0] > 0.0);
}

#[test]
fn optical_quaternion_maps_z_to_body_x() {
    let m = quat_matrix([0.5, -0.5, 0.5, -0.5]);
    let f = mat_vec(&m, [0.0, 0.0, 1.0]);
    assert!((f[0] - 1.0).abs() < 1e-12 && f[1].abs() < 1e-12 && f[2].abs() < 1e-12);
}
