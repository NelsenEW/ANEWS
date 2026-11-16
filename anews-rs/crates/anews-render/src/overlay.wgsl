struct Screen {
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var map_tex: texture_2d<f32>;
@group(0) @binding(2) var map_smp: sampler;
@group(0) @binding(3) var atlas: texture_2d<f32>;
@group(0) @binding(4) var atlas_smp: sampler;

struct Inst {
    @builtin(vertex_index) vi: u32,
    @location(0) bbox: vec4<f32>,
    @location(1) shape: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec4<f32>,
    @location(4) kind: vec2<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) px: vec2<f32>,
    @location(1) shape: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) kind: vec2<f32>,
};

@vertex
fn vs(i: Inst) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[i.vi];
    let px = mix(i.bbox.xy, i.bbox.zw, c);
    var o: VOut;
    o.clip = vec4<f32>(px.x / screen.size.x * 2.0 - 1.0, 1.0 - px.y / screen.size.y * 2.0, 0.0, 1.0);
    o.px = px;
    o.shape = i.shape;
    o.color = i.color;
    o.uv = mix(i.uv.xy, i.uv.zw, c);
    o.kind = i.kind;
    return o;
}

fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

fn round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let kind = i32(round(i.kind.x));
    let param = i.kind.y;
    var a = 1.0;
    var col = i.color;
    if (kind == 0) {
        let c = (i.shape.xy + i.shape.zw) * 0.5;
        let half = (i.shape.zw - i.shape.xy) * 0.5;
        a = clamp(0.5 - round_box(i.px - c, half, param), 0.0, 1.0);
    } else if (kind == 1) {
        a = clamp(0.5 - (segment(i.px, i.shape.xy, i.shape.zw) - param), 0.0, 1.0);
    } else if (kind == 2) {
        a = clamp(0.5 - (length(i.px - i.shape.xy) - i.shape.z), 0.0, 1.0);
    } else if (kind == 3) {
        col = textureSampleLevel(map_tex, map_smp, i.uv, 0.0) * vec4<f32>(1.0, 1.0, 1.0, i.color.a);
    } else if (kind == 4) {
        a = textureSampleLevel(atlas, atlas_smp, i.uv, 0.0).r;
    } else {
        a = clamp(0.5 - (abs(length(i.px - i.shape.xy) - i.shape.z) - param), 0.0, 1.0);
    }
    return vec4<f32>(col.rgb, col.a * a);
}
