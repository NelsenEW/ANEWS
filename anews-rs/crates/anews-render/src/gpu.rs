use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::camera::{Camera, Viewport};
use crate::mesh::{MeshData, Vertex};
use crate::text::ATLAS;

pub const MSAA: u32 = 4;
pub const SHADOW_SIZE: u32 = 4096;
const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const STAGING: usize = 3;
const MAX_POINTS: usize = 64 * 1024;
const MAX_FLAT: usize = 64 * 1024;
const MAX_OVERLAY: usize = 32 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CamU {
    view_proj: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    eye_pos: [f32; 4],
    light_dir: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointInst {
    pub center: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub height: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlatVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OverlayInst {
    pub bbox: [f32; 4],
    pub shape: [f32; 4],
    pub color: [f32; 4],
    pub uv: [f32; 4],
    pub kind: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims {
    w: u32,
    h: u32,
    y_words: u32,
    c_words: u32,
}

pub struct Light {
    pub view_proj: Mat4,
    pub dir: Vec3,
}

pub struct View3d {
    pub camera: Camera,
    pub viewport: Viewport,
    pub wall_scale: f32,
    pub robot: bool,
    pub points: std::ops::Range<u32>,
    pub flat: std::ops::Range<u32>,
}

pub struct OverlayBatch {
    pub scissor: Option<Viewport>,
    pub range: std::ops::Range<u32>,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: String,
}

pub fn gpu() -> Result<Gpu, String> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..Default::default() });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let name = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
        if matches!(info.device_type, wgpu::DeviceType::Cpu | wgpu::DeviceType::Other) {
            return Err(format!("{name} is a software adapter; anews render only runs on a GPU"));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("anews-render"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(Gpu { device, queue, adapter: name })
    })
}

fn texture(device: &wgpu::Device, label: &str, w: u32, h: u32, samples: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

struct GpuMesh {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    floor_indices: u32,
    indices: u32,
}

fn upload(device: &wgpu::Device, label: &str, floor: &MeshData, solid: &MeshData) -> GpuMesh {
    let mut all = MeshData::default();
    all.vertices.extend_from_slice(&floor.vertices);
    all.indices.extend_from_slice(&floor.indices);
    let base = all.vertices.len() as u32;
    all.vertices.extend_from_slice(&solid.vertices);
    all.indices.extend(solid.indices.iter().map(|i| i + base));
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&all.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&all.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    GpuMesh { vb, ib, floor_indices: floor.indices.len() as u32, indices: all.indices.len() as u32 }
}

fn uniform(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> =
        entries.iter().enumerate().map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

fn dyn_buffer(device: &wgpu::Device, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some("dynamic"), size: size as u64, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

const MESH_ATTRS: [wgpu::VertexAttribute; 6] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32, 4 => Float32, 5 => Float32];
const POINT_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4, 3 => Float32];
const FLAT_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
const OVERLAY_ATTRS: [wgpu::VertexAttribute; 5] =
    wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x2];

pub struct Renderer {
    pub gpu: Gpu,
    pub width: u32,
    pub height: u32,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    mesh_pipe: wgpu::RenderPipeline,
    point_pipe: wgpu::RenderPipeline,
    flat_pipe: wgpu::RenderPipeline,
    overlay_pipe: wgpu::RenderPipeline,
    luma_pipe: wgpu::ComputePipeline,
    chroma_pipe: wgpu::ComputePipeline,
    venue: GpuMesh,
    robot: GpuMesh,
    cam_bufs: Vec<wgpu::Buffer>,
    mesh_bgs: Vec<wgpu::BindGroup>,
    point_bgs: Vec<wgpu::BindGroup>,
    flat_bgs: Vec<wgpu::BindGroup>,
    identity_bg: wgpu::BindGroup,
    robot_buf: wgpu::Buffer,
    robot_bg: wgpu::BindGroup,
    point_buf: wgpu::Buffer,
    flat_buf: wgpu::Buffer,
    overlay_buf: wgpu::Buffer,
    overlay_bg: wgpu::BindGroup,
    atlas: wgpu::Texture,
    luma_bg: wgpu::BindGroup,
    chroma_bg: wgpu::BindGroup,
    yuv_buf: wgpu::Buffer,
    pub frame_bytes: u64,
    dims: Dims,
    staging: Vec<wgpu::Buffer>,
    pending: std::collections::VecDeque<(usize, wgpu::SubmissionIndex)>,
    light: Light,
    pub look: [f32; 3],
}

pub struct Inputs<'a> {
    pub venue_floor: &'a MeshData,
    pub venue_solid: &'a MeshData,
    pub robot: &'a MeshData,
    pub light: Light,
    pub wall_scales: [f32; 2],
    pub map_rgba: Option<(u32, u32, Vec<u8>)>,
}

impl Renderer {
    pub fn new(gpu: Gpu, width: u32, height: u32, input: Inputs) -> Result<Self, String> {
        if !width.is_multiple_of(8) || !height.is_multiple_of(2) {
            return Err(format!("{width}x{height}: width must be a multiple of 8 and height even for yuv420p"));
        }
        let device = &gpu.device;
        let scene = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scene"), source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()) });
        let overlay = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("overlay"), source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()) });
        let yuv = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("yuv"), source: wgpu::ShaderSource::Wgsl(include_str!("yuv.wgsl").into()) });

        let att = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let color = texture(device, "color", width, height, 1, COLOR, att | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let color_view = color.create_view(&Default::default());
        let msaa_view = texture(device, "msaa", width, height, MSAA, COLOR, att).create_view(&Default::default());
        let depth_view = texture(device, "depth", width, height, MSAA, DEPTH, att).create_view(&Default::default());

        let mesh_layout = wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &MESH_ATTRS };
        let depth_state = |write: bool, bias: wgpu::DepthBiasState| wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: write,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias,
        };
        let msaa = wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false };
        let target = |blend: Option<wgpu::BlendState>| [Some(wgpu::ColorTargetState { format: COLOR, blend, write_mask: wgpu::ColorWrites::ALL })];
        let opaque = target(None);
        let blended = target(Some(wgpu::BlendState::ALPHA_BLENDING));
        let pipe = |label: &str,
                    module: &wgpu::ShaderModule,
                    vs: &str,
                    fs: Option<&str>,
                    buffers: &[wgpu::VertexBufferLayout],
                    depth: Option<wgpu::DepthStencilState>,
                    multisample: wgpu::MultisampleState,
                    targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: depth,
                multisample,
                fragment: fs.map(|fs| wgpu::FragmentState { module, entry_point: Some(fs), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mesh_pipe = pipe("mesh", &scene, "vs_mesh", Some("fs_mesh"), std::slice::from_ref(&mesh_layout), Some(depth_state(true, Default::default())), msaa, &opaque);
        let shadow_bias = wgpu::DepthBiasState { constant: 2, slope_scale: 2.5, clamp: 0.0 };
        let shadow_pipe = pipe("shadow", &scene, "vs_shadow", None, &[mesh_layout], Some(depth_state(true, shadow_bias)), Default::default(), &[]);
        let point_layout = wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<PointInst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &POINT_ATTRS };
        let point_pipe = pipe("points", &scene, "vs_point", Some("fs_point"), &[point_layout], Some(depth_state(true, Default::default())), msaa, &opaque);
        let flat_layout = wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<FlatVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &FLAT_ATTRS };
        let flat_pipe = pipe("flat", &scene, "vs_flat", Some("fs_flat"), &[flat_layout], Some(depth_state(false, Default::default())), msaa, &blended);
        let overlay_layout = wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<OverlayInst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &OVERLAY_ATTRS };
        let overlay_pipe = pipe("overlay", &overlay, "vs", Some("fs"), &[overlay_layout], None, Default::default(), &blended);
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &yuv,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let luma_pipe = compute("luma");
        let chroma_pipe = compute("chroma");

        let venue = upload(device, "venue", input.venue_floor, input.venue_solid);
        let robot = upload(device, "robot", &MeshData::default(), input.robot);

        let model_size = std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        let identity_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("identity"),
            contents: bytemuck::cast_slice(&Mat4::IDENTITY.to_cols_array_2d()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let robot_buf = uniform(device, model_size);
        let identity_bg = bind(device, &mesh_pipe.get_bind_group_layout(1), &[identity_buf.as_entire_binding()]);
        let robot_bg = bind(device, &mesh_pipe.get_bind_group_layout(1), &[robot_buf.as_entire_binding()]);

        let cmp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let cam_size = std::mem::size_of::<CamU>() as u64;
        let mut shadows = Vec::new();
        let mut encoder = device.create_command_encoder(&Default::default());
        let shadow_identity_bg = bind(device, &shadow_pipe.get_bind_group_layout(1), &[identity_buf.as_entire_binding()]);
        let mut shadow_cams = Vec::new();
        for &scale in &input.wall_scales {
            let map = texture(device, "shadow", SHADOW_SIZE, SHADOW_SIZE, 1, DEPTH, att | wgpu::TextureUsages::TEXTURE_BINDING);
            let view = map.create_view(&Default::default());
            let u = CamU {
                view_proj: input.light.view_proj.to_cols_array_2d(),
                light_vp: input.light.view_proj.to_cols_array_2d(),
                eye_pos: [0.0; 4],
                light_dir: [0.0; 4],
                right: [0.0; 4],
                up: [0.0; 4],
                params: [scale, 0.0, 0.0, 0.0],
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("shadow cam"), contents: bytemuck::bytes_of(&u), usage: wgpu::BufferUsages::UNIFORM });
            let bg = bind(device, &shadow_pipe.get_bind_group_layout(0), &[buf.as_entire_binding()]);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&shadow_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.set_bind_group(1, &shadow_identity_bg, &[]);
                pass.set_vertex_buffer(0, venue.vb.slice(..));
                pass.set_index_buffer(venue.ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(venue.floor_indices..venue.indices, 0, 0..1);
            }
            shadow_cams.push(buf);
            shadows.push(view);
        }
        gpu.queue.submit([encoder.finish()]);

        let mut cam_bufs = Vec::new();
        let (mut mesh_bgs, mut point_bgs, mut flat_bgs) = (Vec::new(), Vec::new(), Vec::new());
        for shadow in &shadows {
            let buf = uniform(device, cam_size);
            mesh_bgs.push(bind(device, &mesh_pipe.get_bind_group_layout(0), &[buf.as_entire_binding(), wgpu::BindingResource::TextureView(shadow), wgpu::BindingResource::Sampler(&cmp)]));
            point_bgs.push(bind(device, &point_pipe.get_bind_group_layout(0), &[buf.as_entire_binding()]));
            flat_bgs.push(bind(device, &flat_pipe.get_bind_group_layout(0), &[buf.as_entire_binding()]));
            cam_bufs.push(buf);
        }

        let point_buf = dyn_buffer(device, MAX_POINTS * std::mem::size_of::<PointInst>(), wgpu::BufferUsages::VERTEX);
        let flat_buf = dyn_buffer(device, MAX_FLAT * std::mem::size_of::<FlatVertex>(), wgpu::BufferUsages::VERTEX);
        let overlay_buf = dyn_buffer(device, MAX_OVERLAY * std::mem::size_of::<OverlayInst>(), wgpu::BufferUsages::VERTEX);

        let (mw, mh, mpix) = input.map_rgba.unwrap_or((1, 1, vec![0x1b, 0x21, 0x29, 0xff]));
        let map_tex = texture(device, "map", mw, mh, 1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &map_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &mpix,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(mw * 4), rows_per_image: Some(mh) },
            wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
        );
        let atlas = texture(device, "atlas", ATLAS, ATLAS, 1, wgpu::TextureFormat::R8Unorm, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("nearest"), ..Default::default() });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let screen = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("screen"),
            contents: bytemuck::cast_slice(&[width as f32, height as f32, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let map_view = map_tex.create_view(&Default::default());
        let atlas_view = atlas.create_view(&Default::default());
        let overlay_bg = bind(
            device,
            &overlay_pipe.get_bind_group_layout(0),
            &[
                screen.as_entire_binding(),
                wgpu::BindingResource::TextureView(&map_view),
                wgpu::BindingResource::Sampler(&nearest),
                wgpu::BindingResource::TextureView(&atlas_view),
                wgpu::BindingResource::Sampler(&linear),
            ],
        );

        let dims = Dims { w: width, h: height, y_words: width * height / 4, c_words: width * height / 16 };
        let frame_bytes = (width * height * 3 / 2) as u64;
        let yuv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv"),
            size: frame_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("dims"), contents: bytemuck::bytes_of(&dims), usage: wgpu::BufferUsages::UNIFORM });
        let yuv_entries = [wgpu::BindingResource::TextureView(&color_view), yuv_buf.as_entire_binding(), dims_buf.as_entire_binding()];
        let luma_bg = bind(device, &luma_pipe.get_bind_group_layout(0), &yuv_entries);
        let chroma_bg = bind(device, &chroma_pipe.get_bind_group_layout(0), &yuv_entries);
        let staging = (0..STAGING)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging"),
                    size: frame_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        drop(shadow_cams);
        Ok(Renderer {
            width,
            height,
            color,
            color_view,
            msaa_view,
            depth_view,
            mesh_pipe,
            point_pipe,
            flat_pipe,
            overlay_pipe,
            luma_pipe,
            chroma_pipe,
            venue,
            robot,
            cam_bufs,
            mesh_bgs,
            point_bgs,
            flat_bgs,
            identity_bg,
            robot_buf,
            robot_bg,
            point_buf,
            flat_buf,
            overlay_buf,
            overlay_bg,
            atlas,
            luma_bg,
            chroma_bg,
            yuv_buf,
            frame_bytes,
            dims,
            staging,
            pending: Default::default(),
            light: input.light,
            look: [0.5, 1.4, 0.9],
            gpu,
        })
    }

    pub fn upload_atlas(&self, pixels: &[u8]) {
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.atlas, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(ATLAS), rows_per_image: Some(ATLAS) },
            wgpu::Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: 1 },
        );
    }

    fn cam_uniform(&self, view: &View3d) -> CamU {
        let v = glam::Mat4::look_at_rh(view.camera.eye, view.camera.target, Vec3::Z);
        let r = v.row(0).truncate();
        let u = v.row(1).truncate();
        CamU {
            view_proj: view.camera.view_proj().to_cols_array_2d(),
            light_vp: self.light.view_proj.to_cols_array_2d(),
            eye_pos: view.camera.eye.extend(1.0).to_array(),
            light_dir: self.light.dir.extend(0.0).to_array(),
            right: r.extend(0.0).to_array(),
            up: u.extend(0.0).to_array(),
            params: [view.wall_scale, self.look[0], self.look[1], self.look[2]],
        }
    }

    pub fn submit(
        &mut self,
        views: &[View3d],
        robot_model: Mat4,
        points: &[PointInst],
        flat: &[FlatVertex],
        overlay: &[OverlayInst],
        batches: &[OverlayBatch],
    ) -> Result<(), String> {
        if points.len() > MAX_POINTS || flat.len() > MAX_FLAT || overlay.len() > MAX_OVERLAY {
            return Err(format!("frame too large: {} points, {} flat vertices, {} overlay items", points.len(), flat.len(), overlay.len()));
        }
        let q = &self.gpu.queue;
        for (i, v) in views.iter().enumerate() {
            q.write_buffer(&self.cam_bufs[i], 0, bytemuck::bytes_of(&self.cam_uniform(v)));
        }
        q.write_buffer(&self.robot_buf, 0, bytemuck::cast_slice(&robot_model.to_cols_array_2d()));
        if !points.is_empty() {
            q.write_buffer(&self.point_buf, 0, bytemuck::cast_slice(points));
        }
        if !flat.is_empty() {
            q.write_buffer(&self.flat_buf, 0, bytemuck::cast_slice(flat));
        }
        if !overlay.is_empty() {
            q.write_buffer(&self.overlay_buf, 0, bytemuck::cast_slice(overlay));
        }
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&self.color_view),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.106, g: 0.129, b: 0.161, a: 1.0 }), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (i, v) in views.iter().enumerate() {
                let vp = v.viewport;
                pass.set_viewport(vp.x, vp.y, vp.w, vp.h, 0.0, 1.0);
                pass.set_scissor_rect(vp.x as u32, vp.y as u32, vp.w as u32, vp.h as u32);
                pass.set_pipeline(&self.mesh_pipe);
                pass.set_bind_group(0, &self.mesh_bgs[i], &[]);
                pass.set_bind_group(1, &self.identity_bg, &[]);
                pass.set_vertex_buffer(0, self.venue.vb.slice(..));
                pass.set_index_buffer(self.venue.ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.venue.indices, 0, 0..1);
                if v.robot && self.robot.indices > 0 {
                    pass.set_bind_group(1, &self.robot_bg, &[]);
                    pass.set_vertex_buffer(0, self.robot.vb.slice(..));
                    pass.set_index_buffer(self.robot.ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.robot.indices, 0, 0..1);
                }
                if !v.points.is_empty() {
                    pass.set_pipeline(&self.point_pipe);
                    pass.set_bind_group(0, &self.point_bgs[i], &[]);
                    pass.set_vertex_buffer(0, self.point_buf.slice(..));
                    pass.draw(0..6, v.points.clone());
                }
                if !v.flat.is_empty() {
                    pass.set_pipeline(&self.flat_pipe);
                    pass.set_bind_group(0, &self.flat_bgs[i], &[]);
                    pass.set_vertex_buffer(0, self.flat_buf.slice(..));
                    pass.draw(v.flat.clone(), 0..1);
                }
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.overlay_pipe);
            pass.set_bind_group(0, &self.overlay_bg, &[]);
            pass.set_vertex_buffer(0, self.overlay_buf.slice(..));
            for b in batches {
                if b.range.is_empty() {
                    continue;
                }
                match b.scissor {
                    Some(s) => pass.set_scissor_rect(s.x as u32, s.y as u32, s.w as u32, s.h as u32),
                    None => pass.set_scissor_rect(0, 0, self.width, self.height),
                }
                pass.draw(0..6, b.range.clone());
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.luma_pipe);
            pass.set_bind_group(0, &self.luma_bg, &[]);
            pass.dispatch_workgroups(self.dims.y_words.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.chroma_pipe);
            pass.set_bind_group(0, &self.chroma_bg, &[]);
            pass.dispatch_workgroups(self.dims.c_words.div_ceil(64), 1, 1);
        }
        let slot = (0..STAGING).find(|s| !self.pending.iter().any(|(p, _)| p == s)).ok_or("no free staging buffer")?;
        encoder.copy_buffer_to_buffer(&self.yuv_buf, 0, &self.staging[slot], 0, self.frame_bytes);
        let index = q.submit([encoder.finish()]);
        self.staging[slot].slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.pending.push_back((slot, index));
        let _ = &self.color;
        Ok(())
    }

    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    pub fn max_in_flight(&self) -> usize {
        STAGING - 1
    }

    pub fn take(&mut self, out: &mut Vec<u8>) -> Result<bool, String> {
        let Some((slot, index)) = self.pending.pop_front() else { return Ok(false) };
        self.gpu
            .device
            .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .map_err(|e| format!("GPU wait failed: {e}"))?;
        let buf = &self.staging[slot];
        {
            let data = buf.slice(..).get_mapped_range();
            out.clear();
            out.extend_from_slice(&data);
        }
        buf.unmap();
        Ok(true)
    }
}
