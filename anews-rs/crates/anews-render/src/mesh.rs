use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub rough: f32,
    pub emit: f32,
    pub wall: f32,
}

#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.vertices.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| {
            let p = Vec3::from(v.pos);
            (lo.min(p), hi.max(p))
        })
    }

    fn push_triangle(&mut self, p: [Vec3; 3], n: Option<[Vec3; 3]>, look: &Look, wall: bool) {
        let flat = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let base = self.vertices.len() as u32;
        for k in 0..3 {
            let normal = n.map(|n| n[k]).filter(|n| n.length_squared() > 1e-12).unwrap_or(flat).normalize_or_zero();
            self.vertices.push(Vertex {
                pos: p[k].to_array(),
                normal: normal.to_array(),
                color: look.color,
                rough: look.rough,
                emit: look.emit,
                wall: if wall { 1.0 } else { 0.0 },
            });
        }
        self.indices.extend([base, base + 1, base + 2]);
    }

    pub fn append(&mut self, other: MeshData) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.indices.extend(other.indices.into_iter().map(|i| i + base));
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub color: [f32; 3],
    pub rough: f32,
    pub emit: f32,
}

pub fn srgb_hex(hex: u32) -> [f32; 3] {
    let c = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0)].map(|v| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) })
}

pub type LookFn = dyn Fn(Option<&str>, [f32; 4], f32) -> Look;

pub struct Loaded {
    pub floor: MeshData,
    pub solid: MeshData,
}

const DRACO: &str = "KHR_draco_mesh_compression";
const GLTF_TRANSFORM: &str = "@gltf-transform/cli@4.5.1";

pub fn requires_draco(bytes: &[u8]) -> bool {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return false;
    }
    let len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let Some(chunk) = bytes.get(20..20 + len) else { return false };
    serde_json::from_slice::<serde_json::Value>(chunk)
        .ok()
        .and_then(|j| j.get("extensionsRequired").cloned())
        .and_then(|r| r.as_array().cloned())
        .is_some_and(|r| r.iter().any(|e| e == DRACO))
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

pub fn without_draco(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    if !requires_draco(&bytes) {
        return Ok(bytes);
    }
    let dir = std::env::temp_dir().join("anews-render-glb");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let key = format!("{:016x}", fnv64(&bytes));
    let out = dir.join(format!("{key}.glb"));
    if let Ok(cached) = std::fs::read(&out) {
        return Ok(cached);
    }
    let src = dir.join(format!("{key}.draco.glb"));
    std::fs::write(&src, &bytes).map_err(|e| format!("{}: {e}", src.display()))?;
    let tmp = dir.join(format!("{key}.{}.glb", std::process::id()));
    let status = std::process::Command::new("npx")
        .args(["--yes", GLTF_TRANSFORM, "copy"])
        .arg(&src)
        .arg(&tmp)
        .stdout(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("the GLB uses Draco compression and decoding it needs npx ({GLTF_TRANSFORM}): {e}"))?;
    if !status.success() {
        return Err(format!("{GLTF_TRANSFORM} could not decode the Draco GLB ({status})"));
    }
    let decoded = std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    if requires_draco(&decoded) {
        return Err(format!("{GLTF_TRANSFORM} left the GLB Draco-compressed"));
    }
    std::fs::rename(&tmp, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    let _ = std::fs::remove_file(&src);
    Ok(decoded)
}

pub fn load_glb(bytes: &[u8], root: Mat4, look: &LookFn) -> Result<Loaded, String> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| {
        let text = e.to_string();
        if text.contains("KHR_draco_mesh_compression") || text.contains("xtension") {
            format!("{text}; decompress the GLB first, e.g. `npx @gltf-transform/cli copy in.glb out.glb`")
        } else {
            text
        }
    })?;
    if gltf.extensions_required().any(|e| e == "KHR_draco_mesh_compression") {
        return Err("GLB uses Draco compression; decompress it first, e.g. `npx @gltf-transform/cli copy in.glb out.glb`".into());
    }
    let blob = gltf.blob.as_deref().ok_or("glTF has no binary chunk; use a .glb")?;
    let scene = gltf.default_scene().or_else(|| gltf.scenes().next()).ok_or("glTF has no scene")?;
    let mut out = Loaded { floor: MeshData::default(), solid: MeshData::default() };
    for node in scene.nodes() {
        let kind = node.name().unwrap_or("");
        visit(&node, root, blob, kind == "walls", kind == "floor", look, &mut out)?;
    }
    Ok(out)
}

fn visit(
    node: &gltf::Node,
    parent: Mat4,
    blob: &[u8],
    wall: bool,
    floor: bool,
    look: &LookFn,
    out: &mut Loaded,
) -> Result<(), String> {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    let normal_m = Mat3::from_mat4(world).inverse().transpose();
    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| match b.source() {
                gltf::buffer::Source::Bin => Some(blob),
                gltf::buffer::Source::Uri(_) => None,
            });
            let positions: Vec<Vec3> =
                reader.read_positions().ok_or("glTF primitive has no positions")?.map(|p| world.transform_point3(Vec3::from(p))).collect();
            let normals: Option<Vec<Vec3>> = reader.read_normals().map(|ns| ns.map(|n| (normal_m * Vec3::from(n)).normalize_or_zero()).collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(ix) => ix.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let mat = prim.material();
            let pbr = mat.pbr_metallic_roughness();
            let l = look(mat.name(), pbr.base_color_factor(), pbr.roughness_factor());
            let target = if floor { &mut out.floor } else { &mut out.solid };
            for tri in indices.chunks_exact(3) {
                let ix = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
                if ix.iter().any(|&i| i >= positions.len()) {
                    return Err("glTF index out of range".into());
                }
                let n = normals.as_ref().map(|n| ix.map(|i| n[i]));
                target.push_triangle(ix.map(|i| positions[i]), n, &l, wall);
            }
        }
    }
    for child in node.children() {
        visit(&child, world, blob, wall, floor, look, out)?;
    }
    Ok(())
}

pub fn venue_look(_: Option<&str>, base: [f32; 4], rough: f32) -> Look {
    Look { color: [base[0], base[1], base[2]], rough, emit: 0.0 }
}

pub fn robot_look(name: Option<&str>, _: [f32; 4], _: f32) -> Look {
    match name.unwrap_or("") {
        "black" | "speaker" => Look { color: srgb_hex(0x15171b), rough: 0.3, emit: 0.0 },
        "lens" => Look { color: srgb_hex(0x0a1424), rough: 0.05, emit: 0.0 },
        "led" => Look { color: srgb_hex(0x35c2ff), rough: 0.4, emit: 1.4 },
        "screen" => Look { color: srgb_hex(0x10305a), rough: 0.2, emit: 0.9 },
        _ => Look { color: srgb_hex(0xf3f5f7), rough: 0.38, emit: 0.0 },
    }
}

pub fn proxy_robot(radius: f32, height: f32) -> MeshData {
    let mut m = MeshData::default();
    let body = robot_look(None, [1.0; 4], 0.5);
    let n = 32;
    for i in 0..n {
        let a0 = i as f32 / n as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
        let (p0, p1) = (Vec3::new(a0.cos(), a0.sin(), 0.0) * radius, Vec3::new(a1.cos(), a1.sin(), 0.0) * radius);
        let up = Vec3::Z * height;
        let side = [p0.normalize(), p1.normalize(), p1.normalize()];
        m.push_triangle([p0, p1, p1 + up], Some(side), &body, false);
        m.push_triangle([p0, p1 + up, p0 + up], Some([side[0], side[1], side[0]]), &body, false);
        m.push_triangle([up, p0 + up, p1 + up], None, &body, false);
    }
    let nose = robot_look(Some("led"), [1.0; 4], 0.5);
    let tip = Vec3::new(radius + 0.06, 0.0, height * 0.7);
    m.push_triangle([Vec3::new(radius, 0.08, height * 0.7), Vec3::new(radius, -0.08, height * 0.7), tip], None, &nose, false);
    m
}

#[cfg(test)]
mod draco_tests {
    use super::*;

    fn glb(json: &str) -> Vec<u8> {
        let mut chunk = json.as_bytes().to_vec();
        while !chunk.len().is_multiple_of(4) {
            chunk.push(b' ');
        }
        let total = 12 + 8 + chunk.len();
        let mut out = b"glTF".to_vec();
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&chunk);
        out
    }

    #[test]
    fn only_glbs_that_require_draco_are_sent_to_the_decoder() {
        assert!(requires_draco(&glb(r#"{"asset":{"version":"2.0"},"extensionsRequired":["KHR_draco_mesh_compression"]}"#)));
        assert!(!requires_draco(&glb(r#"{"asset":{"version":"2.0"},"extensionsUsed":["KHR_draco_mesh_compression"]}"#)));
        assert!(!requires_draco(&glb(r#"{"asset":{"version":"2.0"}}"#)));
        assert!(!requires_draco(b"not a glb at all, just bytes"));
    }
}
