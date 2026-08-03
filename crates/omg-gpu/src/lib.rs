//! Native wgpu host for the GPU trace kernels (GPU_PLAN.md Track A).
//! Phase 1: synchronous single-job dispatch, validated against the
//! Phase 0 goldens; the pipelined LateBackend ring is Phase 2.

pub mod layout;

use layout::{
    decode_echogram, flatten_mesh, flatten_surfaces, GpuMeshJob, GpuPanel, GpuSolveJob,
    GpuSolveRec, GpuSolveSrc, GpuTraceJob, BINS_LEN, DIRS_LEN, MAX_PANELS, MAX_SOLVE_CHAINS,
    MAX_SOLVE_EXTRAS, MAX_SOLVE_SOURCES,
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
    // K3: chain discovery over the same BVH buffers
    disc_pipeline: wgpu::ComputePipeline,
    disc_job: wgpu::Buffer,
    disc_chains: wgpu::Buffer,
    disc_count: wgpu::Buffer,
    disc_read: wgpu::Buffer,
    disc_bind: wgpu::BindGroup,
    disc_n_boxes: u32,
    // K4 (C7a): the batched (source × chain) solve over the same BVH,
    // built only when a SurfaceTable is supplied at construction
    solve: Option<SolveState>,
}

struct SolveState {
    pipeline: wgpu::ComputePipeline,
    job: wgpu::Buffer,
    chains: wgpu::Buffer,
    srcs: wgpu::Buffer,
    extras: wgpu::Buffer,
    recs: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// Chain slots in the discovery output list (must match
/// discover_mesh.wgsl CAP).
pub const DISC_CAP: usize = 16384;
/// Discovery rays per dispatch — the CPU fan runs 768/tick; discovery
/// only has to find each chain once per TTL window, so density here is
/// pure freshness.
pub const DISC_RAYS: u32 = 4096;

impl GpuMeshTracer {
    /// `boxes`: overlay boxes (furniture) the DISCOVERY kernel reflects
    /// off, ids `base + box·6 + face` where base is the mesh's surface
    /// count — must match the SurfaceTable the caller solves against.
    pub fn new(mesh: &Mesh, boxes: &[(Vec3, Vec3)]) -> Option<Self> {
        pollster::block_on(Self::new_async(mesh, boxes, None))
    }

    /// Also builds the K4 batched-solve pipeline against `table` (the
    /// SurfaceTable the CPU side solves with — authored planes plus
    /// overlay faces, in the same id order).
    pub fn with_solve(
        mesh: &Mesh,
        boxes: &[(Vec3, Vec3)],
        table: &omg_core::pt_mesh::SurfaceTable,
    ) -> Option<Self> {
        pollster::block_on(Self::new_async(mesh, boxes, Some(table)))
    }

    async fn new_async(
        mesh: &Mesh,
        boxes: &[(Vec3, Vec3)],
        table: Option<&omg_core::pt_mesh::SurfaceTable>,
    ) -> Option<Self> {
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

        // K3: discovery pipeline sharing the SAME nodes/prims buffers
        let disc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("discover_mesh"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/discover_mesh.wgsl").into(),
            ),
        });
        let disc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("discover_mesh"),
            layout: None,
            module: &disc_shader,
            entry_point: Some("discover"),
            compilation_options: Default::default(),
            cache: None,
        });
        let disc_job = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("disc-job"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let disc_chains = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("disc-chains"),
            size: (DISC_CAP * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let disc_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("disc-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let disc_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("disc-read"),
            size: (4 + DISC_CAP * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // overlay boxes (furniture) for discovery — static like the BVH
        let obox_words: Vec<f32> = if boxes.is_empty() {
            vec![0.0; 8]
        } else {
            boxes
                .iter()
                .flat_map(|(mn, mx)| [mn.x, mn.y, mn.z, 0.0, mx.x, mx.y, mx.z, 0.0])
                .collect()
        };
        let obox_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("disc-oboxes"),
            contents: bytemuck::cast_slice(&obox_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let disc_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("discover_mesh"),
            layout: &disc_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: disc_job.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: disc_chains.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: disc_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: obox_buf.as_entire_binding() },
            ],
        });
        let disc_n_boxes = boxes.len() as u32;

        // K4 (C7a): batched solve pipeline over the same nodes/prims/mats
        let solve = table.map(|table| {
            let surfs = flatten_surfaces(table);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("solve_mesh"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/solve_mesh.wgsl").into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("solve_mesh"),
                layout: None,
                module: &shader,
                entry_point: Some("solve"),
                compilation_options: Default::default(),
                cache: None,
            });
            let surfs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("solve-surfs"),
                contents: bytemuck::cast_slice(&surfs),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let job = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solve-job"),
                size: core::mem::size_of::<GpuSolveJob>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dyn_buf = |label: &str, bytes: usize| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let chains = dyn_buf("solve-chains", MAX_SOLVE_CHAINS * 8);
            let srcs = dyn_buf(
                "solve-srcs",
                MAX_SOLVE_SOURCES * core::mem::size_of::<GpuSolveSrc>(),
            );
            let extras = dyn_buf(
                "solve-extras",
                MAX_SOLVE_EXTRAS * core::mem::size_of::<layout::GpuExtra>(),
            );
            let recs_bytes =
                MAX_SOLVE_SOURCES * MAX_SOLVE_CHAINS * core::mem::size_of::<GpuSolveRec>();
            let recs = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solve-recs"),
                size: recs_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solve-read"),
                size: recs_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("solve_mesh"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: job.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: prims_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: mats_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: surfs_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: chains.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: srcs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: extras.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: recs.as_entire_binding() },
                ],
            });
            SolveState { pipeline, job, chains, srcs, extras, recs, read, bind }
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
            disc_pipeline,
            disc_job,
            disc_chains,
            disc_count,
            disc_read,
            disc_bind,
            disc_n_boxes,
            solve,
        })
    }

    /// K4 (C7a): solve every (source × chain) pair in one dispatch.
    /// `out[si * chains.len() + ci]` gets the pair's record or None.
    /// Returns false (untouched `out`) when the solve pipeline wasn't
    /// built or a cap is exceeded — the caller's CPU path takes over.
    pub fn solve_batch(
        &self,
        sources: &[(u16, Vec3)],
        chains: &[omg_core::pt_mesh::MChain],
        listener: Vec3,
        extras: &[omg_core::pt::Aabb],
        out: &mut Vec<Option<omg_core::pt_mesh::MeshRecord>>,
    ) -> bool {
        let Some(s) = &self.solve else { return false };
        // a clamped input would silently change the physics (missing
        // occluders, missing chains) — refuse instead, CPU covers it
        if sources.is_empty()
            || sources.len() > MAX_SOLVE_SOURCES
            || chains.is_empty()
            || chains.len() > MAX_SOLVE_CHAINS
            || extras.len() > MAX_SOLVE_EXTRAS
        {
            return false;
        }
        let job = GpuSolveJob {
            n_sources: sources.len() as u32,
            n_chains: chains.len() as u32,
            n_extras: extras.len() as u32,
            _p0: 0,
            listener: [listener.x, listener.y, listener.z],
            _p1: 0,
        };
        self.queue.write_buffer(&s.job, 0, bytemuck::bytes_of(&job));
        let cw: Vec<u32> = chains
            .iter()
            .flat_map(|(c, order)| {
                [
                    (c[0] as u32) | ((c[1] as u32) << 16),
                    (c[2] as u32) | ((*order as u32) << 16),
                ]
            })
            .collect();
        self.queue.write_buffer(&s.chains, 0, bytemuck::cast_slice(&cw));
        let sw: Vec<GpuSolveSrc> = sources
            .iter()
            .map(|(id, p)| GpuSolveSrc { pos: [p.x, p.y, p.z], id: *id as u32 })
            .collect();
        self.queue.write_buffer(&s.srcs, 0, bytemuck::cast_slice(&sw));
        if !extras.is_empty() {
            let xw: Vec<layout::GpuExtra> = extras
                .iter()
                .map(|x| layout::GpuExtra {
                    bmin: [x.min.x, x.min.y, x.min.z],
                    _p0: 0,
                    bmax: [x.max.x, x.max.y, x.max.z],
                    _p1: 0,
                    trans: x.transmission,
                    _p2: 0,
                })
                .collect();
            self.queue.write_buffer(&s.extras, 0, bytemuck::cast_slice(&xw));
        }

        let n_pairs = sources.len() * chains.len();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("solve") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&s.pipeline);
            pass.set_bind_group(0, &s.bind, &[]);
            pass.dispatch_workgroups((n_pairs as u32).div_ceil(64), 1, 1);
        }
        let bytes = (n_pairs * core::mem::size_of::<GpuSolveRec>()) as u64;
        enc.copy_buffer_to_buffer(&s.recs, 0, &s.read, 0, bytes);
        self.queue.submit([enc.finish()]);
        s.read
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        {
            let view = s.read.slice(..bytes).get_mapped_range().expect("mapped recs");
            let recs: &[GpuSolveRec] = bytemuck::cast_slice(&view);
            out.clear();
            out.reserve(n_pairs);
            for (si, &(id, _)) in sources.iter().enumerate() {
                for (ci, &(chain, order)) in chains.iter().enumerate() {
                    let r = &recs[si * chains.len() + ci];
                    out.push((r.valid != 0).then(|| omg_core::pt_mesh::MeshRecord {
                        source: id,
                        chain,
                        order,
                        delay_s: r.delay,
                        dir: r.dir,
                        gains: r.gains,
                    }));
                }
            }
        }
        s.read.unmap();
        true
    }

    /// One synchronous discovery dispatch: the listener fan over the
    /// BVH, raw chain prefixes appended (duplicates included — the
    /// caller's TTL table dedups).
    pub fn discover(
        &self,
        listener: Vec3,
        rot: u32,
        n_rays: u32,
        base: u16,
        furniture_on: bool,
        out: &mut Vec<omg_core::pt_mesh::MChain>,
    ) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Job {
            n_rays: u32,
            rot: u32,
            n_boxes: u32,
            base: u32,
            listener: [f32; 3],
            _p2: u32,
        }
        let job = Job {
            n_rays,
            rot,
            n_boxes: if furniture_on { self.disc_n_boxes } else { 0 },
            base: base as u32,
            listener: [listener.x, listener.y, listener.z],
            _p2: 0,
        };
        self.queue.write_buffer(&self.disc_job, 0, bytemuck::bytes_of(&job));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("disc") });
        enc.clear_buffer(&self.disc_count, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("disc"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.disc_pipeline);
            pass.set_bind_group(0, &self.disc_bind, &[]);
            pass.dispatch_workgroups(n_rays.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.disc_count, 0, &self.disc_read, 0, 4);
        enc.copy_buffer_to_buffer(&self.disc_chains, 0, &self.disc_read, 4, (DISC_CAP * 8) as u64);
        self.queue.submit([enc.finish()]);
        self.disc_read
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        {
            let view = self.disc_read.slice(..).get_mapped_range().expect("mapped");
            let words: &[u32] = bytemuck::cast_slice(&view);
            let n = (words[0] as usize).min(DISC_CAP);
            for i in 0..n {
                let (w0, w1) = (words[1 + i * 2], words[2 + i * 2]);
                let chain = [
                    (w0 & 0xFFFF) as u16,
                    (w0 >> 16) as u16,
                    (w1 & 0xFFFF) as u16,
                ];
                let order = ((w1 >> 16) as u8).clamp(1, 3);
                out.push((chain, order));
            }
        }
        self.disc_read.unmap();
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
    tracer: std::sync::Arc<GpuMeshTracer>,
    seed: u32,
}

/// Chain discovery (K3) as the world-discovery provider — shares the
/// late backend's device and BVH buffers.
pub struct GpuWorldDiscovery {
    tracer: std::sync::Arc<GpuMeshTracer>,
    base: u16,
}

/// The batched solve (K4) as the world-solve provider — same device,
/// same BVH, same SurfaceTable ids as the CPU side.
pub struct GpuWorldSolve {
    tracer: std::sync::Arc<GpuMeshTracer>,
}

const WORLD_RAY_MULT: u32 = 8;
const WORLD_RAY_CAP: u32 = 8192;

impl GpuWorldLateBackend {
    pub fn new(mesh: &Mesh, boxes: &[(Vec3, Vec3)]) -> Option<Self> {
        Some(Self {
            tracer: std::sync::Arc::new(GpuMeshTracer::new(mesh, boxes)?),
            seed: 0x5EED_C6D1,
        })
    }

    /// Both world backends over ONE device and one BVH upload. `base`
    /// is the mesh's authored-surface count (SurfaceTable::base_overlay).
    pub fn with_discovery(
        mesh: &Mesh,
        boxes: &[(Vec3, Vec3)],
        base: u16,
    ) -> Option<(Self, GpuWorldDiscovery)> {
        let tracer = std::sync::Arc::new(GpuMeshTracer::new(mesh, boxes)?);
        Some((
            Self { tracer: tracer.clone(), seed: 0x5EED_C6D1 },
            GpuWorldDiscovery { tracer, base },
        ))
    }

    /// All three world backends (K2 trace, K3 discovery, K4 batched
    /// solve) over one device and one BVH upload. `furn` are the
    /// significant furniture boxes WITH materials, in the same order
    /// the CPU side appends them to its SurfaceTable — id congruence
    /// between the two tables is what makes the batch replayable.
    pub fn with_discovery_and_solve(
        mesh: &Mesh,
        furn: &[(Vec3, Vec3, Material)],
    ) -> Option<(Self, GpuWorldDiscovery, GpuWorldSolve)> {
        let mut table = omg_core::pt_mesh::SurfaceTable::build(mesh);
        for (mn, mx, m) in furn {
            table.append_box(*mn, *mx, m);
        }
        let base = table.base_overlay;
        let boxes: Vec<(Vec3, Vec3)> = furn.iter().map(|(mn, mx, _)| (*mn, *mx)).collect();
        let tracer = std::sync::Arc::new(GpuMeshTracer::with_solve(mesh, &boxes, &table)?);
        Some((
            Self { tracer: tracer.clone(), seed: 0x5EED_C6D1 },
            GpuWorldDiscovery { tracer: tracer.clone(), base },
            GpuWorldSolve { tracer },
        ))
    }
}

impl omg_scene::early_world::WorldSolve for GpuWorldSolve {
    fn solve_batch(
        &mut self,
        sources: &[(u16, Vec3)],
        chains: &[omg_core::pt_mesh::MChain],
        listener: Vec3,
        extras: &[omg_core::pt::Aabb],
        out: &mut Vec<Option<omg_core::pt_mesh::MeshRecord>>,
    ) -> bool {
        self.tracer.solve_batch(sources, chains, listener, extras, out)
    }
}

impl omg_scene::early_world::WorldDiscovery for GpuWorldDiscovery {
    fn discover(
        &mut self,
        listener: Vec3,
        rot: u32,
        out: &mut Vec<omg_core::pt_mesh::MChain>,
    ) -> bool {
        let furn = omg_scene::quality::furniture_on();
        self.tracer.discover(listener, rot, DISC_RAYS, self.base, furn, out);
        true
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
                transmission: m.transmission,
                _p2: 0,
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
