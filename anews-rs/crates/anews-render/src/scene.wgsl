struct Cam {
    view_proj: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    eye_pos: vec4<f32>,
    light_dir: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    params: vec4<f32>,
};

struct Model {
    m: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_cmp: sampler_comparison;
@group(1) @binding(0) var<uniform> model: Model;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) rough: f32,
    @location(4) emit: f32,
    @location(5) wall: f32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat: vec2<f32>,
};

fn place(v: VIn) -> vec4<f32> {
    var p = v.pos;
    if (v.wall > 0.5) {
        p.z = p.z * cam.params.x;
    }
    return model.m * vec4<f32>(p, 1.0);
}

@vertex
fn vs_mesh(v: VIn) -> VOut {
    let wp = place(v);
    var o: VOut;
    o.clip = cam.view_proj * wp;
    o.world = wp.xyz;
    o.normal = (model.m * vec4<f32>(v.normal, 0.0)).xyz;
    o.color = v.color;
    o.mat = vec2<f32>(v.rough, v.emit);
    return o;
}

@vertex
fn vs_shadow(v: VIn) -> @builtin(position) vec4<f32> {
    return cam.view_proj * place(v);
}

fn shadow(world: vec3<f32>) -> f32 {
    let lp = cam.light_vp * vec4<f32>(world, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var s = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            s += textureSampleCompareLevel(shadow_map, shadow_cmp, uv + vec2<f32>(f32(x), f32(y)) * texel * 1.25, ndc.z - 0.0004);
        }
    }
    return s / 25.0;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    return pow(aces(c * cam.params.w), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_mesh(i: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
    var n = normalize(i.normal);
    if (!ff) {
        n = -n;
    }
    let l = normalize(cam.light_dir.xyz);
    let v = normalize(cam.eye_pos.xyz - i.world);
    let ndl = max(dot(n, l), 0.0);
    let sh = shadow(i.world + n * 0.02);
    let sky = vec3<f32>(1.0, 0.905, 0.80);
    let ground = vec3<f32>(0.042, 0.030, 0.022);
    let hemi = mix(ground, sky, n.z * 0.5 + 0.5);
    let sun = vec3<f32>(1.0, 0.88, 0.72);
    let rough = i.mat.x;
    let h = normalize(l + v);
    let shin = mix(220.0, 6.0, rough);
    let spec = pow(max(dot(n, h), 0.0), shin) * (1.0 - rough) * 0.5 * ndl * sh;
    let lit = hemi * cam.params.y + vec3<f32>(0.10) + sun * (ndl * sh * cam.params.z);
    let r = reflect(-v, n);
    let fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let env = mix(ground, sky, clamp(r.z * 0.5 + 0.5, 0.0, 1.0)) * fres * (1.0 - rough) * 0.8;
    let c = i.color * lit + vec3<f32>(spec) + env + i.color * i.mat.y;
    return vec4<f32>(encode(c), 1.0);
}

struct PIn {
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) height: f32,
};

struct POut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) round: f32,
};

@vertex
fn vs_point(p: PIn) -> POut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[p.vi];
    var right = cam.right.xyz;
    var up = cam.up.xyz * (p.size * 0.5);
    var round = 1.0;
    if (p.height > 0.0) {
        right = normalize(vec3<f32>(right.x, right.y, 0.0) + vec3<f32>(1e-6, 0.0, 0.0));
        up = vec3<f32>(0.0, 0.0, p.height * 0.5);
        round = 0.0;
    }
    let w = p.center + right * (c.x * p.size * 0.5) + up * c.y;
    var o: POut;
    o.round = round;
    o.clip = cam.view_proj * vec4<f32>(w, 1.0);
    o.uv = c;
    o.color = p.color;
    return o;
}

@fragment
fn fs_point(i: POut) -> @location(0) vec4<f32> {
    if (i.round > 0.5 && dot(i.uv, i.uv) > 1.0) {
        discard;
    }
    return i.color;
}

struct FIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct FOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_flat(v: FIn) -> FOut {
    var o: FOut;
    o.clip = cam.view_proj * vec4<f32>(v.pos, 1.0);
    o.color = v.color;
    return o;
}

@fragment
fn fs_flat(i: FOut) -> @location(0) vec4<f32> {
    return i.color;
}
