use crate::world3d::{Triangle, V3};

type Mat4 = [[f64; 4]; 4];

const IDENTITY4: Mat4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

fn mul4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn node_matrix(node: &gltf::Node) -> Mat4 {
    let cols = node.transform().matrix();
    let mut m = [[0.0; 4]; 4];
    for (c, col) in cols.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            m[r][c] = *v as f64;
        }
    }
    m
}

fn gltf_to_map(m: &Mat4, p: [f32; 3]) -> V3 {
    let v = [p[0] as f64, p[1] as f64, p[2] as f64];
    let w: Vec<f64> = (0..3).map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3]).collect();
    [w[0], -w[2], w[1]]
}

fn visit(node: gltf::Node, parent: &Mat4, blob: &[u8], owner: u32, out: &mut Vec<Triangle>) -> Result<(), String> {
    let world = mul4(parent, &node_matrix(&node));
    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| match b.source() {
                gltf::buffer::Source::Bin => Some(blob),
                gltf::buffer::Source::Uri(_) => None,
            });
            let positions: Vec<V3> = reader
                .read_positions()
                .ok_or("glTF primitive has no positions")?
                .map(|p| gltf_to_map(&world, p))
                .collect();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(ix) => ix.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            for tri in indices.chunks_exact(3) {
                let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| positions[i as usize]);
                out.push(Triangle { a, b, c, owner });
            }
        }
    }
    for child in node.children() {
        visit(child, &world, blob, owner, out)?;
    }
    Ok(())
}

pub fn glb_triangles(bytes: &[u8], owner: u32) -> Result<Vec<Triangle>, String> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| e.to_string())?;
    let blob = gltf.blob.as_deref().ok_or("glTF has no binary chunk; use a .glb")?;
    let scene = gltf.default_scene().or_else(|| gltf.scenes().next()).ok_or("glTF has no scene")?;
    let mut out = Vec::new();
    for node in scene.nodes() {
        visit(node, &IDENTITY4, blob, owner, &mut out)?;
    }
    Ok(out)
}
