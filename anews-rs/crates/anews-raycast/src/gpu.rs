use anews_core::fidelity::{LIDAR_ANGLE_MAX, LIDAR_ANGLE_MIN, LIDAR_COUNT, LIDAR_RANGE};
use anews_core::raycast::{Ray, RayCaster};
use anews_core::sensor_batch::{Pose2, Rig, RobotScan};
use anews_core::world3d::World3d;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const WORKGROUP: u32 = 256;
const MAX_GROUPS_X: u32 = 65535;
const CHUNK_RAYS: usize = 1 << 22;
const RAY_BYTES: u64 = 32;

const COMMON: &str = r#"
enable wgpu_ray_query;
@group(0) @binding(0) var acc: acceleration_structure;
fn trace(o: vec3<f32>, d: vec3<f32>, t_max: f32, floor: bool) -> f32 {
    var tmax = t_max;
    var hit = -1.0;
    if (floor && d.z < -1e-12) {
        let tf = -o.z / d.z;
        if (tf > 0.0 && tf < tmax) { tmax = tf; hit = tf; }
    }
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xFFu, 1e-6, tmax, o, d));
    rayQueryProceed(&rq);
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind == RAY_QUERY_INTERSECTION_TRIANGLE) { hit = h.t; }
    return hit;
}
fn flat_index(id: vec3<u32>, nw: vec3<u32>) -> u32 {
    return id.x + id.y * nw.x * 256u;
}
"#;

const CAST: &str = r#"
struct R { o: vec4<f32>, d: vec4<f32> }
@group(0) @binding(1) var<storage, read> rays: array<R>;
@group(0) @binding(2) var<storage, read_write> hits: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = flat_index(id, nw);
    if (i >= arrayLength(&rays)) { return; }
    let r = rays[i];
    hits[i] = trace(r.o.xyz, r.d.xyz, r.o.w, r.d.w > 0.5);
}
"#;

const SENSE: &str = r#"
struct Params {
    lidar_r0: vec4<f32>, lidar_r1: vec4<f32>, lidar_r2: vec4<f32>, lidar_xyz: vec4<f32>,
    cam_r0: vec4<f32>, cam_r1: vec4<f32>, cam_r2: vec4<f32>, cam_xyz: vec4<f32>,
    lidar: vec4<f32>,
    cam: vec4<f32>,
    quant: vec4<f32>,
    counts: vec4<u32>,
    offset: vec4<u32>,
}
@group(0) @binding(1) var<storage, read> poses: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cam_rays: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> prm: Params;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit(h: u32) -> f32 {
    return (f32(h >> 8u) + 0.5) / 16777216.0;
}
fn gaussian(key: u32) -> f32 {
    let a = pcg(key);
    let b = pcg(a);
    return sqrt(-2.0 * log(unit(a))) * cos(6.283185307179586 * unit(b));
}
fn quantize(z: f32) -> f32 {
    let fb = prm.cam.w;
    let step = prm.quant.x;
    if (fb <= 0.0 || step <= 0.0 || z <= 0.0) { return z; }
    let disparity = max(round((fb / z) / step), 1.0) * step;
    return fb / disparity;
}
fn rz(cs: vec2<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(cs.x * v.x - cs.y * v.y, cs.y * v.x + cs.x * v.y, v.z);
}
fn rot(r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = flat_index(id, nw);
    let nl = prm.counts.x;
    let per = nl + prm.counts.y;
    let robots = arrayLength(&poses);
    if (i >= per * robots) { return; }
    let robot = i / per;
    let k = i % per;
    let p = poses[robot];
    let base = vec3<f32>(p.x, p.y, 0.0);
    let floor = prm.quant.y > 0.5;
    if (k < nl) {
        var a = prm.lidar.x + (prm.lidar.y - prm.lidar.x) / f32(nl - 1u) * f32(k);
        if (k == nl - 1u) { a = prm.lidar.y; }
        let d = rz(p.zw, rot(prm.lidar_r0, prm.lidar_r1, prm.lidar_r2, vec3<f32>(cos(a), sin(a), 0.0)));
        out[i] = trace(base + rz(p.zw, prm.lidar_xyz.xyz), normalize(d), prm.lidar.z, floor);
        return;
    }
    let c = k - nl;
    let r = cam_rays[c].xyz;
    let d = rz(p.zw, rot(prm.cam_r0, prm.cam_r1, prm.cam_r2, r));
    let t = trace(base + rz(p.zw, prm.cam_xyz.xyz), normalize(d), prm.lidar.w, floor);
    out[i] = -1.0;
    if (t < 0.0 || r.z <= 0.0) { return; }
    let z = t * r.z;
    if (z < prm.cam.x || z > prm.cam.y) { return; }
    let key = pcg(prm.counts.z ^ pcg(prm.counts.w ^ pcg((robot + prm.offset.x) ^ pcg(c))));
    let measured = quantize(z + prm.cam.z * z * z * gaussian(key));
    if (measured >= prm.cam.x && measured <= prm.cam.y) { out[i] = measured / r.z; }
}
"#;

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
}

impl GpuContext {
    pub fn new() -> Result<Arc<Self>, String> {
        pollster::block_on(Self::create()).map(Arc::new)
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    async fn create() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            return Err(format!("{name} has no hardware ray query"));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("anews-raycast"),
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: adapter.limits(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(GpuContext { device, queue, adapter: name })
    }

    fn pipeline(&self, source: &str) -> wgpu::ComputePipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anews-raycast"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}{source}").into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anews-raycast"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn storage(&self, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(4), usage, mapped_at_creation: false })
    }

    fn wait(&self) -> Result<(), String> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| format!("{}: {e}", self.adapter))
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, threads: usize, src: &wgpu::Buffer, dst: &wgpu::Buffer, bytes: u64) -> Result<Vec<f32>, String> {
        let groups = (threads as u32).div_ceil(WORKGROUP);
        let (gx, gy) = if groups > MAX_GROUPS_X { (MAX_GROUPS_X, groups.div_ceil(MAX_GROUPS_X)) } else { (groups, 1) };
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = dst.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.wait()?;
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        dst.unmap();
        Ok(out)
    }
}

struct GpuScene {
    tlas: wgpu::Tlas,
    _blas: wgpu::Blas,
    _vertices: wgpu::Buffer,
    floor: bool,
}

impl GpuScene {
    fn new(ctx: &GpuContext, world: &World3d) -> Result<Self, String> {
        let mut verts: Vec<[f32; 3]> = world.bvh.triangles().iter().flat_map(|t| [t.a, t.b, t.c]).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        if verts.is_empty() {
            verts = vec![[0.0, 0.0, -1e6]; 3];
        }
        let vertices = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anews-raycast vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: verts.len() as u32,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = ctx.device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("anews-raycast blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
        );
        let mut tlas = ctx.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("anews-raycast tlas"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        tlas[0] = Some(wgpu::TlasInstance::new(&blas, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0, 0xff));
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.build_acceleration_structures(
            std::iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: 12,
                    index_buffer: None,
                    first_index: None,
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            std::iter::once(&tlas),
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.wait()?;
        Ok(GpuScene { tlas, _blas: blas, _vertices: vertices, floor: world.floor })
    }
}

struct CastBuffers {
    capacity: usize,
    rays: wgpu::Buffer,
    hits: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub struct GpuCaster {
    ctx: Arc<GpuContext>,
    scene: GpuScene,
    pipeline: wgpu::ComputePipeline,
    buffers: Mutex<Option<CastBuffers>>,
}

impl GpuCaster {
    pub fn new(world: &World3d) -> Result<Self, String> {
        Self::with_context(GpuContext::new()?, world)
    }

    pub fn with_context(ctx: Arc<GpuContext>, world: &World3d) -> Result<Self, String> {
        let scene = GpuScene::new(&ctx, world)?;
        let pipeline = ctx.pipeline(CAST);
        Ok(GpuCaster { ctx, scene, pipeline, buffers: Mutex::new(None) })
    }

    pub fn adapter(&self) -> &str {
        self.ctx.adapter()
    }

    fn buffers(&self, n: usize) -> CastBuffers {
        let ctx = &self.ctx;
        let rays = ctx.storage("anews-raycast rays", n as u64 * RAY_BYTES, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let hits = ctx.storage("anews-raycast hits", n as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let read = ctx.storage("anews-raycast readback", n as u64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anews-raycast cast"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&self.scene.tlas) },
                wgpu::BindGroupEntry { binding: 1, resource: rays.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: hits.as_entire_binding() },
            ],
        });
        CastBuffers { capacity: n, rays, hits, read, bind }
    }

    fn cast_chunk(&self, bufs: &mut Option<CastBuffers>, rays: &[Ray], out: &mut Vec<f64>) -> Result<(), String> {
        let n = rays.len();
        if bufs.as_ref().is_none_or(|b| b.capacity < n) {
            *bufs = Some(self.buffers(n.next_power_of_two().min(CHUNK_RAYS)));
        }
        let b = bufs.as_ref().expect("buffers allocated above");
        let floor = if self.scene.floor { 1.0 } else { 0.0 };
        let packed: Vec<[f32; 8]> = rays
            .iter()
            .map(|r| {
                let l = (r.dir[0] * r.dir[0] + r.dir[1] * r.dir[1] + r.dir[2] * r.dir[2]).sqrt();
                [
                    r.origin[0] as f32,
                    r.origin[1] as f32,
                    r.origin[2] as f32,
                    r.t_max as f32,
                    (r.dir[0] / l) as f32,
                    (r.dir[1] / l) as f32,
                    (r.dir[2] / l) as f32,
                    floor,
                ]
            })
            .collect();
        self.ctx.queue.write_buffer(&b.rays, 0, bytemuck::cast_slice(&packed));
        let hits = self.ctx.dispatch(&self.pipeline, &b.bind, n, &b.hits, &b.read, n as u64 * 4)?;
        out.extend(hits.iter().map(|&h| if h < 0.0 { f64::INFINITY } else { h as f64 }));
        Ok(())
    }
}

impl RayCaster for GpuCaster {
    fn cast_batch(&self, rays: &[Ray]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rays.len());
        let mut bufs = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        for chunk in rays.chunks(CHUNK_RAYS) {
            if let Err(e) = self.cast_chunk(&mut bufs, chunk, &mut out) {
                panic!("GPU raycast failed: {e}");
            }
        }
        out
    }
}

struct SenseBuffers {
    robots: usize,
    poses: wgpu::Buffer,
    out: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub struct GpuSensors {
    ctx: Arc<GpuContext>,
    scene: GpuScene,
    pipeline: wgpu::ComputePipeline,
    cam_rays: wgpu::Buffer,
    params: wgpu::Buffer,
    per_robot: usize,
    seed: u32,
    buffers: Mutex<Option<SenseBuffers>>,
}

impl GpuSensors {
    pub fn new(ctx: Arc<GpuContext>, world: &World3d, rig: &Rig, seed: u32) -> Result<Self, String> {
        let scene = GpuScene::new(&ctx, world)?;
        let pipeline = ctx.pipeline(SENSE);
        let cam: Vec<[f32; 4]> = rig.camera_rays.iter().map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, 0.0]).collect();
        let cam_rays = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anews-raycast camera rays"),
            contents: bytemuck::cast_slice(&cam),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let row = |m: &[[f64; 3]; 3], i: usize| [m[i][0] as f32, m[i][1] as f32, m[i][2] as f32, 0.0];
        let v4 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
        let c = &rig.sensor.camera;
        let floats: [[f32; 4]; 11] = [
            row(&rig.lidar.rot, 0),
            row(&rig.lidar.rot, 1),
            row(&rig.lidar.rot, 2),
            v4(rig.lidar.xyz),
            row(&rig.camera_optical, 0),
            row(&rig.camera_optical, 1),
            row(&rig.camera_optical, 2),
            v4(rig.camera.xyz),
            [LIDAR_ANGLE_MIN as f32, LIDAR_ANGLE_MAX as f32, LIDAR_RANGE as f32, rig.depth_reach() as f32],
            [c.range_min_m as f32, c.range_max_m as f32, c.noise.sigma_at_1m_m as f32, (c.noise.native_focal_px * c.noise.baseline_m) as f32],
            [c.noise.subpixel as f32, if scene.floor { 1.0 } else { 0.0 }, 0.0, 0.0],
        ];
        let mut bytes: Vec<u8> = bytemuck::cast_slice(&floats).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&[LIDAR_COUNT as u32, rig.camera_rays.len() as u32, seed, 0u32, 0, 0, 0, 0]));
        let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anews-raycast sensor params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let per_robot = LIDAR_COUNT + rig.camera_rays.len();
        Ok(GpuSensors { ctx, scene, pipeline, cam_rays, params, per_robot, seed, buffers: Mutex::new(None) })
    }

    pub fn adapter(&self) -> &str {
        self.ctx.adapter()
    }

    fn buffers(&self, robots: usize) -> SenseBuffers {
        let ctx = &self.ctx;
        let bytes = (robots * self.per_robot) as u64 * 4;
        let poses = ctx.storage("anews-raycast poses", robots as u64 * 16, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let out = ctx.storage("anews-raycast sensor out", bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let read = ctx.storage("anews-raycast sensor readback", bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anews-raycast sense"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&self.scene.tlas) },
                wgpu::BindGroupEntry { binding: 1, resource: poses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cam_rays.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
            ],
        });
        SenseBuffers { robots, poses, out, read, bind }
    }

    fn sense_chunk(&self, bufs: &mut Option<SenseBuffers>, poses: &[Pose2], first_robot: usize, tick: u32) -> Result<Vec<f32>, String> {
        let n = poses.len();
        if bufs.as_ref().is_none_or(|b| b.robots != n) {
            *bufs = Some(self.buffers(n));
        }
        let b = bufs.as_ref().expect("buffers allocated above");
        let packed: Vec<[f32; 4]> = poses.iter().map(|p| [p.x as f32, p.y as f32, p.yaw.cos() as f32, p.yaw.sin() as f32]).collect();
        self.ctx.queue.write_buffer(&b.poses, 0, bytemuck::cast_slice(&packed));
        let counts = [LIDAR_COUNT as u32, (self.per_robot - LIDAR_COUNT) as u32, self.seed, tick, first_robot as u32, 0, 0, 0];
        self.ctx.queue.write_buffer(&self.params, 11 * 16, bytemuck::cast_slice(&counts));
        let threads = n * self.per_robot;
        self.ctx.dispatch(&self.pipeline, &b.bind, threads, &b.out, &b.read, threads as u64 * 4)
    }

    pub fn sense(&self, poses: &[Pose2], tick: u32) -> Vec<RobotScan> {
        let per_chunk = (CHUNK_RAYS / self.per_robot).max(1);
        let mut bufs = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let mut scans = Vec::with_capacity(poses.len());
        for (c, chunk) in poses.chunks(per_chunk).enumerate() {
            let raw = match self.sense_chunk(&mut bufs, chunk, c * per_chunk, tick) {
                Ok(raw) => raw,
                Err(e) => panic!("GPU sensing failed: {e}"),
            };
            for r in raw.chunks(self.per_robot) {
                let lidar = r[..LIDAR_COUNT].iter().map(|&h| if h < 0.0 { f64::INFINITY } else { h as f64 }).collect();
                let depth = r[LIDAR_COUNT..].iter().map(|&h| if h < 0.0 { f64::NAN } else { h as f64 }).collect();
                scans.push(RobotScan { lidar, depth });
            }
        }
        scans
    }
}
