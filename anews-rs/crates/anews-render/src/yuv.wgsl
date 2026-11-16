struct Dims {
    w: u32,
    h: u32,
    y_words: u32,
    c_words: u32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

fn rgb(x: u32, y: u32) -> vec3<f32> {
    return textureLoad(src, vec2<i32>(i32(x), i32(y)), 0).rgb;
}

fn byte(v: f32) -> u32 {
    return u32(clamp(round(v), 0.0, 255.0));
}

@compute @workgroup_size(64)
fn luma(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + id.y * 65535u * 64u;
    if (idx >= dims.y_words) {
        return;
    }
    let per_row = dims.w / 4u;
    let row = idx / per_row;
    let col = (idx % per_row) * 4u;
    var word = 0u;
    for (var k = 0u; k < 4u; k++) {
        let c = rgb(col + k, row);
        let y = 16.0 + 219.0 * dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
        word = word | (byte(y) << (8u * k));
    }
    out[idx] = word;
}

@compute @workgroup_size(64)
fn chroma(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + id.y * 65535u * 64u;
    if (idx >= dims.c_words) {
        return;
    }
    let per_row = dims.w / 8u;
    let row = idx / per_row;
    let col = (idx % per_row) * 4u;
    var uw = 0u;
    var vw = 0u;
    for (var k = 0u; k < 4u; k++) {
        let x = (col + k) * 2u;
        let y = row * 2u;
        let c = (rgb(x, y) + rgb(x + 1u, y) + rgb(x, y + 1u) + rgb(x + 1u, y + 1u)) * 0.25;
        let u = 128.0 + 224.0 * dot(c, vec3<f32>(-0.1146, -0.3854, 0.5));
        let v = 128.0 + 224.0 * dot(c, vec3<f32>(0.5, -0.4542, -0.0458));
        uw = uw | (byte(u) << (8u * k));
        vw = vw | (byte(v) << (8u * k));
    }
    out[dims.y_words + idx] = uw;
    out[dims.y_words + dims.c_words + idx] = vw;
}
