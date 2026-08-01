//! Native wgpu host for the GPU trace kernels (GPU_PLAN.md Track A).
//! Phase 1: synchronous single-job dispatch, validated against the
//! Phase 0 goldens; the pipelined LateBackend ring is Phase 2.

pub mod layout;

use layout::{
    decode_echogram, flatten_mesh, GpuMeshJob, GpuPanel, GpuTraceJob, BINS_LEN, DIRS_LEN,
    MAX_PANELS,
};
use omg_core::material::Material;
use omg_core::mesh::Mesh;
use omg_core::scene::Shoebox;
use omg_core::tracer::Echogram;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;
use wgpu::util::DeviceExt;

pub struct GpuTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    job_buf: wgpu::Buffer,
    bins_buf: wgpu::Buffer,
    dirs_buf: wgpu::Buffer,
    read_bins: wgpu::Buffer,
    read_dirs: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GpuTracer {
    /// None when no usable adapter exists (headless CI, remote box) —
    /// callers fall back to the CPU tracer.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("omg-gpu"),
                ..Default::default()
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace_box"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/trace_box.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_box"),
            layout: None,
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        let job_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("job"),
            size: core::mem::size_of::<GpuTraceJob>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = |label, words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let bins_buf = out_buf("bins", BINS_LEN);
        let dirs_buf = out_buf("dirs", DIRS_LEN);
        let read_buf = |label, words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let read_bins = read_buf("bins-read", BINS_LEN);
        let read_dirs = read_buf("dirs-read", DIRS_LEN);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace_box"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: job_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bins_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dirs_buf.as_entire_binding() },
            ],
        });

        Some(Self {
            device,
            queue,
            pipeline,
            job_buf,
            bins_buf,
            dirs_buf,
            read_bins,
            read_dirs,
            bind_group,
        })
    }

    /// One synchronous trace: dispatch, wait, decode. The Phase 2
    /// backend pipelines this; tests and offline rendering use it as is.
    pub fn trace(
        &self,
        room: &Shoebox,
        src: Vec3,
        lis: Vec3,
        n_rays: u32,
        energy: [f32; NBANDS],
        seed: u32,
        out: &mut Echogram,
    ) {
        let job = GpuTraceJob::new(room, src, lis, n_rays, energy, seed);
        self.queue.write_buffer(&self.job_buf, 0, bytemuck::bytes_of(&job));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("trace") });
        enc.clear_buffer(&self.bins_buf, 0, None);
        enc.clear_buffer(&self.dirs_buf, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(n_rays.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.bins_buf, 0, &self.read_bins, 0, (BINS_LEN * 4) as u64);
        enc.copy_buffer_to_buffer(&self.dirs_buf, 0, &self.read_dirs, 0, (DIRS_LEN * 4) as u64);
        self.queue.submit([enc.finish()]);

        let map = |buf: &wgpu::Buffer| {
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        };
        map(&self.read_bins);
        map(&self.read_dirs);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

        let bins: Vec<u32> = {
            let view = self.read_bins.slice(..).get_mapped_range().expect("mapped bins");
            bytemuck::cast_slice(&view).to_vec()
        };
        let dirs: Vec<i32> = {
            let view = self.read_dirs.slice(..).get_mapped_range().expect("mapped dirs");
            bytemuck::cast_slice(&view).to_vec()
        };
        self.read_bins.unmap();
        self.read_dirs.unmap();
        decode_echogram(&bins, &dirs, out);
    }
}

/// C6d kernel K2: the world-mesh tracer. The BVH, prims and material
/// table upload ONCE (the world is static); per job only the 64-byte
/// uniform and the panel overlay buffer (door leaves swing) change.
pub struct GpuMeshTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    job_buf: wgpu::Buffer,
    panels_buf: wgpu::Buffer,
    bins_buf: wgpu::Buffer,
    dirs_buf: wgpu::Buffer,
    read_bins: wgpu::Buffer,
    read_dirs: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl GpuMeshTracer {
    pub fn new(mesh: &Mesh) -> Option<Self> {
        pollster::block_on(Self::new_async(mesh))
    }

    async fn new_async(mesh: &Mesh) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("omg-gpu-mesh"),
                ..Default::default()
            })
            .await
            .ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace_mesh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/trace_mesh.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_mesh"),
            layout: None,
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (nodes, prims, mats) = flatten_mesh(mesh);
        if nodes.is_empty() || prims.is_empty() {
            return None;
        }
        let static_buf = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let nodes_buf = static_buf("mesh-nodes", bytemuck::cast_slice(&nodes));
        let prims_buf = static_buf("mesh-prims", bytemuck::cast_slice(&prims));
        let mats_buf = static_buf("mesh-mats", bytemuck::cast_slice(&mats));

        let job_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-job"),
            size: core::mem::size_of::<GpuMeshJob>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let panels_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-panels"),
            size: (MAX_PANELS * core::mem::size_of::<GpuPanel>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = |label: &str, words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let bins_buf = out_buf("mesh-bins", BINS_LEN);
        let dirs_buf = out_buf("mesh-dirs", DIRS_LEN);
        let read_buf = |label: &str, words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let read_bins = read_buf("mesh-bins-read", BINS_LEN);
        let read_dirs = read_buf("mesh-dirs-read", DIRS_LEN);

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace_mesh"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: job_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mats_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: panels_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bins_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dirs_buf.as_entire_binding() },
            ],
        });

        Some(Self {
            device,
            queue,
            pipeline,
            job_buf,
            panels_buf,
            bins_buf,
            dirs_buf,
            read_bins,
            read_dirs,
            bind,
        })
    }

    /// One synchronous world trace: dispatch, wait, decode.
    pub fn trace(
        &self,
        src: Vec3,
        lis: Vec3,
        n_rays: u32,
        energy: [f32; NBANDS],
        seed: u32,
        panels: &[GpuPanel],
        out: &mut Echogram,
    ) {
        let n_panels = panels.len().min(MAX_PANELS);
        let job = GpuMeshJob {
            n_rays,
            seed,
            n_panels: n_panels as u32,
            _p0: 0,
            source: [src.x, src.y, src.z],
            _p1: 0,
            listener: [lis.x, lis.y, lis.z],
            _p2: 0,
            energy,
            _p3: 0,
        };
        self.queue.write_buffer(&self.job_buf, 0, bytemuck::bytes_of(&job));
        if n_panels > 0 {
            self.queue
                .write_buffer(&self.panels_buf, 0, bytemuck::cast_slice(&panels[..n_panels]));
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mesh") });
        enc.clear_buffer(&self.bins_buf, 0, None);
        enc.clear_buffer(&self.dirs_buf, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mesh"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(n_rays.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.bins_buf, 0, &self.read_bins, 0, (BINS_LEN * 4) as u64);
        enc.copy_buffer_to_buffer(&self.dirs_buf, 0, &self.read_dirs, 0, (DIRS_LEN * 4) as u64);
        self.queue.submit([enc.finish()]);

        let map = |buf: &wgpu::Buffer| {
            buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        };
        map(&self.read_bins);
        map(&self.read_dirs);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

        let bins: Vec<u32> = {
            let view = self.read_bins.slice(..).get_mapped_range().expect("mapped bins");
            bytemuck::cast_slice(&view).to_vec()
        };
        let dirs: Vec<i32> = {
            let view = self.read_dirs.slice(..).get_mapped_range().expect("mapped dirs");
            bytemuck::cast_slice(&view).to_vec()
        };
        self.read_bins.unmap();
        self.read_dirs.unmap();
        decode_echogram(&bins, &dirs, out);
    }
}

/// The world-late backend for `early=traced` (registered under
/// OMG_GPU=1). The scene's one-trace-per-tick budget stays; the GPU
/// just makes that one trace 8× denser for the same wall-clock class.
pub struct GpuWorldLateBackend {
    tracer: GpuMeshTracer,
    seed: u32,
}

const WORLD_RAY_MULT: u32 = 8;
const WORLD_RAY_CAP: u32 = 8192;

impl GpuWorldLateBackend {
    pub fn new(mesh: &Mesh) -> Option<Self> {
        Some(Self { tracer: GpuMeshTracer::new(mesh)?, seed: 0x5EED_C6D1 })
    }
}

impl omg_scene::late::WorldLateBackend for GpuWorldLateBackend {
    fn trace_world(
        &mut self,
        _id: u32,
        src: Vec3,
        lis: Vec3,
        rays: u32,
        panels: &[(Vec3, Vec3, Material)],
        out: &mut Echogram,
    ) -> bool {
        let gp: Vec<GpuPanel> = panels
            .iter()
            .take(MAX_PANELS)
            .map(|(mn, mx, m)| GpuPanel {
                pmin: [mn.x, mn.y, mn.z],
                scattering: m.scattering,
                pmax: [mx.x, mx.y, mx.z],
                _p0: 0,
                absorption: m.absorption,
                _p1: 0,
            })
            .collect();
        self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
        let n = (rays * WORLD_RAY_MULT).min(WORLD_RAY_CAP);
        self.tracer.trace(src, lis, n, [1.0; NBANDS], self.seed, &gp, out);
        true
    }
}

/// The `LateBackend` the app registers under `OMG_GPU=1`. Synchronous
/// by measurement, not oversight: one dispatch incl. readback is ~1 ms
/// on Apple Metal — 7× cheaper than the CPU trace it replaces — so a
/// pipelined ring would buy a millisecond and cost a tick of staleness.
/// Revisit in phase 5 when one submission batches every source.
pub struct GpuLateBackend {
    tracer: GpuTracer,
    seed: u32,
}

impl GpuLateBackend {
    pub fn new() -> Option<Self> {
        Some(Self { tracer: GpuTracer::new()?, seed: 0xD15E_A5E })
    }
}

/// PT-early chain discovery on the GPU (Track C phase C3): dispatches
/// pt_early.wgsl and decodes the 258-bit chain bitmap. Synchronous by
/// the same measurement as the late backend — the dispatch is tiny.
pub struct GpuEarlyDiscovery {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    job_buf: wgpu::Buffer,
    bitmap: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub const PT_RAYS: u32 = 4096;

impl GpuEarlyDiscovery {
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("omg-gpu-pt"),
                ..Default::default()
            })
            .await
            .ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt_early"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pt_early.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pt_early"),
            layout: None,
            module: &shader,
            entry_point: Some("discover"),
            compilation_options: Default::default(),
            cache: None,
        });
        let job_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt-job"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bitmap = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt-bitmap"),
            size: 36,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt-read"),
            size: 36,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt_early"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: job_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bitmap.as_entire_binding() },
            ],
        });
        Some(Self { device, queue, pipeline, job_buf, bitmap, read, bind })
    }

    pub fn bitmap_for(&self, size: [f32; 3], listener: [f32; 3], rot: u32) -> [u32; 9] {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Job {
            size: [f32; 3],
            n_rays: u32,
            listener: [f32; 3],
            rot: u32,
        }
        let job = Job { size, n_rays: PT_RAYS, listener, rot };
        self.queue.write_buffer(&self.job_buf, 0, bytemuck::bytes_of(&job));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pt") });
        enc.clear_buffer(&self.bitmap, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(PT_RAYS.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.bitmap, 0, &self.read, 0, 36);
        self.queue.submit([enc.finish()]);
        self.read
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let words: [u32; 9] = {
            let view = self.read.slice(..).get_mapped_range().expect("mapped");
            bytemuck::cast_slice(&view).try_into().unwrap()
        };
        self.read.unmap();
        words
    }
}

/// Decode the kernel's chain bitmap into Chain values (shared with the
/// web driver's inject path via omg-web).
pub fn decode_chain_bitmap(words: &[u32; 9], out: &mut Vec<omg_core::pt::Chain>) {
    let bit = |i: usize| words[i >> 5] >> (i & 31) & 1 == 1;
    const NO: u8 = omg_core::pt::NO_WALL;
    for w1 in 0..6usize {
        if bit(w1) {
            out.push(([w1 as u8, NO, NO], 1));
        }
        for w2 in 0..6usize {
            if bit(6 + w1 * 6 + w2) {
                out.push(([w1 as u8, w2 as u8, NO], 2));
            }
            for w3 in 0..6usize {
                if bit(42 + w1 * 36 + w2 * 6 + w3) {
                    out.push(([w1 as u8, w2 as u8, w3 as u8], 3));
                }
            }
        }
    }
}

impl omg_scene::early::EarlyDiscovery for GpuEarlyDiscovery {
    fn discover(
        &mut self,
        _id: u32,
        room: &Shoebox,
        listener: Vec3,
        rot: u32,
        out: &mut Vec<omg_core::pt::Chain>,
    ) -> bool {
        let words = self.bitmap_for(
            [room.size.x, room.size.y, room.size.z],
            [listener.x, listener.y, listener.z],
            rot,
        );
        decode_chain_bitmap(&words, out);
        true
    }
}

impl omg_scene::late::LateBackend for GpuLateBackend {
    fn trace(
        &mut self,
        _id: u32,
        room: &Shoebox,
        src: Vec3,
        lis: Vec3,
        n_rays: u32,
        energy: [f32; NBANDS],
        _rng: &mut omg_core::rng::Rng,
        out: &mut Echogram,
    ) -> bool {
        // fresh stream per trace; the Sim's EMA does the averaging
        self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
        self.tracer.trace(room, src, lis, n_rays, energy, self.seed, out);
        true
    }
}
