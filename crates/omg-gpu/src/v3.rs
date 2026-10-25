//! Engine v3 on the GPU (ENGINE_V3.md V3.3, kernel K5): the listener
//! walk of both ray families — diffraction bends included — as
//! `shaders/v3_walk.wgsl`, one thread per ray, contributions binned on
//! the device exactly as v3.rs `deposit` bins them (64-bit fixed-point
//! accumulators, resolved to f32 by a second pass). The CPU keeps the
//! direct and edge paths, the EMA and the field construction
//! (`omg_core::v3::Tracer::trace_walked`); this host turns a flat
//! `WalkJob` into a `Walked`.
//!
//! Two ways to run it: `walk` (synchronous: dispatch, wait, read back —
//! tests, offline) and `submit` + `poll` (one job in flight; its result
//! is collected on a later tick without ever blocking — what the app's
//! walker does, the same one-tick-late contract as the web driver).

use crate::layout::{flatten_mesh, GpuBvhNode};
use omg_core::mesh::Mesh;
use omg_core::v3::fold::{
    decode_debug, output_words, Walk, WalkJob, Walked, DEBUG_WORDS, MAX_EDGES, MAX_PANELS, MAX_POCKETS,
    MAX_SCENE_WORDS, MAX_SOURCES,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

/// Words per probe record (v3_walk.wgsl PROBE_WORDS).
const PROBE_WORDS: usize = 20;

/// One probe answer: (transmission, raycast t (≥ 3e38 = miss), normal,
/// in_earth(a), crosses_earth(a, b), nearest edge (t, distance)).
pub type ProbeResult = ([f32; 3], f32, [f32; 3], bool, bool, Option<(f32, f32)>);

struct InFlight {
    ids: Vec<u32>,
    epoch: u32,
    out_bytes: u64,
    dbg_bytes: u64,
    /// map_async callbacks still outstanding (0 = all mapped).
    pending: Arc<AtomicU32>,
    t0: Instant,
}

pub struct GpuV3Tracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    walk_pipe: wgpu::ComputePipeline,
    resolve_pipe: wgpu::ComputePipeline,
    probe_pipe: wgpu::ComputePipeline,
    nodes: wgpu::Buffer,
    prims: wgpu::Buffer,
    mats: wgpu::Buffer,
    job: wgpu::Buffer,
    scene: wgpu::Buffer,
    acc: wgpu::Buffer,
    out: wgpu::Buffer,
    read: wgpu::Buffer,
    dbg: wgpu::Buffer,
    dbg_read: wgpu::Buffer,
    dbg_words: usize,
    walk_bind: wgpu::BindGroup,
    resolve_bind: wgpu::BindGroup,
    flight: Option<InFlight>,
    /// Wall-clock of the last walk: dispatch to readback (sync), or
    /// submit to the poll that found it done (async — an upper bound).
    pub last_ms: f32,
}

impl GpuV3Tracer {
    /// None when no usable adapter exists — callers keep the CPU walk.
    /// An empty mesh (free field) is fine: one never-hit dummy node.
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
            .request_device(&wgpu::DeviceDescriptor { label: Some("omg-gpu-v3"), ..Default::default() })
            .await
            .ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("v3_walk"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/v3_walk.wgsl").into()),
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (walk_pipe, resolve_pipe, probe_pipe) = (pipe("walk"), pipe("resolve"), pipe("probe"));

        let (mut nodes, mut prims, mut mats) = flatten_mesh(mesh);
        if nodes.is_empty() {
            // free field: a leaf with no prims whose box nothing enters
            nodes.push(GpuBvhNode { bmin: [1e30; 3], a: 0x8000_0000, bmax: [-1e30; 3], b: 0 });
        }
        if prims.is_empty() {
            prims.push(bytemuck::Zeroable::zeroed());
        }
        if mats.is_empty() {
            mats.push(bytemuck::Zeroable::zeroed());
        }
        let static_buf = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let nodes = static_buf("v3-nodes", bytemuck::cast_slice(&nodes));
        let prims = static_buf("v3-prims", bytemuck::cast_slice(&prims));
        let mats = static_buf("v3-mats", bytemuck::cast_slice(&mats));
        use wgpu::BufferUsages as U;
        let buf = |label: &str, bytes: usize, usage: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let job = buf("v3-job", core::mem::size_of::<crate::layout::GpuV3Job>(), U::UNIFORM | U::COPY_DST);
        let scene = buf("v3-scene", MAX_SCENE_WORDS * 4, U::STORAGE | U::COPY_DST);
        let out_bytes = output_words(MAX_SOURCES) * 4;
        let acc = buf("v3-acc", out_bytes * 2, U::STORAGE | U::COPY_DST);
        let out = buf("v3-out", out_bytes, U::STORAGE | U::COPY_SRC);
        let read = buf("v3-read", out_bytes, U::MAP_READ | U::COPY_DST);
        let dbg_words = DEBUG_WORDS;
        let dbg = buf("v3-dbg", dbg_words * 4, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let dbg_read = buf("v3-dbg-read", dbg_words * 4, U::MAP_READ | U::COPY_DST);
        let walk_bind = Self::bind(
            &device,
            &walk_pipe,
            &[(0, &job), (1, &nodes), (2, &prims), (3, &mats), (4, &scene), (5, &acc), (7, &dbg)],
        );
        let resolve_bind = Self::bind(&device, &resolve_pipe, &[(0, &job), (4, &scene), (5, &acc), (6, &out)]);
        Some(Self {
            device,
            queue,
            walk_pipe,
            resolve_pipe,
            probe_pipe,
            nodes,
            prims,
            mats,
            job,
            scene,
            acc,
            out,
            read,
            dbg,
            dbg_read,
            dbg_words,
            walk_bind,
            resolve_bind,
            flight: None,
            last_ms: 0.0,
        })
    }

    fn bind(device: &wgpu::Device, pipe: &wgpu::ComputePipeline, bufs: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("v3"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn fits(job: &WalkJob) -> bool {
        job.n_sources() <= MAX_SOURCES
            && job.n_panels() <= MAX_PANELS
            && job.n_pockets() <= MAX_POCKETS
            && job.n_edges() <= MAX_EDGES
            && job.scene().len() <= MAX_SCENE_WORDS
            && job.n_rays() > 0
    }

    /// Queue one walk (job upload, clear, walk, resolve, copy, map).
    /// False when the job exceeds the buffers or one is still in flight.
    pub fn submit(&mut self, job: &WalkJob) -> bool {
        if self.flight.is_some() || !Self::fits(job) {
            return false;
        }
        let ns = job.n_sources();
        let debug = job.debug();
        if debug {
            let need = job.n_early() as usize * DEBUG_WORDS;
            if need > self.dbg_words {
                let words = need.next_power_of_two();
                use wgpu::BufferUsages as U;
                let mk = |label: &str, usage| {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (words * 4) as u64,
                        usage,
                        mapped_at_creation: false,
                    })
                };
                self.dbg = mk("v3-dbg", U::STORAGE | U::COPY_SRC | U::COPY_DST);
                self.dbg_read = mk("v3-dbg-read", U::MAP_READ | U::COPY_DST);
                self.dbg_words = words;
                self.walk_bind = Self::bind(
                    &self.device,
                    &self.walk_pipe,
                    &[
                        (0, &self.job),
                        (1, &self.nodes),
                        (2, &self.prims),
                        (3, &self.mats),
                        (4, &self.scene),
                        (5, &self.acc),
                        (7, &self.dbg),
                    ],
                );
            }
        }
        self.queue.write_buffer(&self.job, 0, bytemuck::cast_slice(job.header()));
        if !job.scene().is_empty() {
            self.queue.write_buffer(&self.scene, 0, bytemuck::cast_slice(job.scene()));
        }
        let out_words = output_words(ns);
        let out_bytes = (out_words * 4) as u64;
        let dbg_bytes = if debug { (job.n_early() as usize * DEBUG_WORDS * 4) as u64 } else { 0 };
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("v3") });
        enc.clear_buffer(&self.acc, 0, Some(out_bytes * 2));
        if dbg_bytes > 0 {
            enc.clear_buffer(&self.dbg, 0, Some(dbg_bytes));
        }
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("v3"), timestamp_writes: None });
            pass.set_pipeline(&self.walk_pipe);
            pass.set_bind_group(0, &self.walk_bind, &[]);
            pass.dispatch_workgroups(job.n_rays().div_ceil(64), 1, 1);
            pass.set_pipeline(&self.resolve_pipe);
            pass.set_bind_group(0, &self.resolve_bind, &[]);
            pass.dispatch_workgroups((out_words as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.out, 0, &self.read, 0, out_bytes);
        if dbg_bytes > 0 {
            enc.copy_buffer_to_buffer(&self.dbg, 0, &self.dbg_read, 0, dbg_bytes);
        }
        self.queue.submit([enc.finish()]);
        let pending = Arc::new(AtomicU32::new(if dbg_bytes > 0 { 2 } else { 1 }));
        let p = pending.clone();
        self.read.slice(..out_bytes).map_async(wgpu::MapMode::Read, move |r| {
            r.expect("map v3 out");
            p.fetch_sub(1, Ordering::AcqRel);
        });
        if dbg_bytes > 0 {
            let p = pending.clone();
            self.dbg_read.slice(..dbg_bytes).map_async(wgpu::MapMode::Read, move |r| {
                r.expect("map v3 dbg");
                p.fetch_sub(1, Ordering::AcqRel);
            });
        }
        self.flight = Some(InFlight { ids: job.ids.clone(), epoch: job.epoch, out_bytes, dbg_bytes, pending, t0: Instant::now() });
        true
    }

    /// The in-flight walk's result, if it finished — never blocks (unless
    /// `wait`).
    pub fn poll(&mut self, wait: bool) -> Option<Walked> {
        self.flight.as_ref()?;
        let poll = if wait { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll };
        let _ = self.device.poll(poll);
        let f = self.flight.as_ref().unwrap();
        if f.pending.load(Ordering::Acquire) != 0 {
            return None;
        }
        let f = self.flight.take().unwrap();
        let data: Vec<f32> = {
            let view = self.read.slice(..f.out_bytes).get_mapped_range().expect("mapped v3 out");
            bytemuck::cast_slice(&view).to_vec()
        };
        self.read.unmap();
        let debug = if f.dbg_bytes > 0 {
            let list = {
                let view = self.dbg_read.slice(..f.dbg_bytes).get_mapped_range().expect("mapped v3 dbg");
                decode_debug(bytemuck::cast_slice(&view), &f.ids)
            };
            self.dbg_read.unmap();
            list
        } else {
            Vec::new()
        };
        self.last_ms = f.t0.elapsed().as_secs_f32() * 1e3;
        Some(Walked { ids: f.ids, data, debug, epoch: f.epoch })
    }

    /// Walk one job synchronously: dispatch, wait, read the bins back.
    /// None when the job exceeds the buffers (the caller's CPU walk
    /// takes the tick).
    pub fn walk(&mut self, job: &WalkJob) -> Option<Walked> {
        if self.flight.is_some() {
            self.poll(true); // drop a stale async walk
        }
        if !self.submit(job) {
            return None;
        }
        self.poll(true)
    }

    /// Parity hook: for each segment (a, b), the kernel's own
    /// World::transmission(a, b), World::raycast from a toward b, the
    /// earth predicates and the nearest usable edge along a→b, under
    /// `job`'s panels, pockets, grade and edges.
    pub fn probe(
        &mut self,
        job: &WalkJob,
        segs: &[(omg_core::vec3::Vec3, omg_core::vec3::Vec3)],
    ) -> Vec<ProbeResult> {
        let words = segs.len() * PROBE_WORDS;
        let mut w = vec![0u32; words];
        for (i, (a, b)) in segs.iter().enumerate() {
            for (k, x) in [a.x, a.y, a.z, b.x, b.y, b.z].iter().enumerate() {
                w[i * PROBE_WORDS + k] = x.to_bits();
            }
        }
        let mut header = job.header().to_vec();
        header[3] = segs.len() as u32;
        use wgpu::BufferUsages as U;
        let mk = |label: &str, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words.max(16) * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let io = mk("v3-probe-io", U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let read = mk("v3-probe-read", U::MAP_READ | U::COPY_DST);
        self.queue.write_buffer(&io, 0, bytemuck::cast_slice(&w));
        self.queue.write_buffer(&self.job, 0, bytemuck::cast_slice(&header));
        if !job.scene().is_empty() {
            self.queue.write_buffer(&self.scene, 0, bytemuck::cast_slice(job.scene()));
        }
        let bind = Self::bind(
            &self.device,
            &self.probe_pipe,
            &[(0, &self.job), (1, &self.nodes), (2, &self.prims), (3, &self.mats), (4, &self.scene), (7, &io)],
        );
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("probe") });
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("probe"), timestamp_writes: None });
            pass.set_pipeline(&self.probe_pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((segs.len() as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&io, 0, &read, 0, (words.max(16) * 4) as u64);
        self.queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map probe"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = read.slice(..).get_mapped_range().expect("mapped probe");
        let r: &[u32] = bytemuck::cast_slice(&view);
        let f = |i: usize| f32::from_bits(r[i]);
        (0..segs.len())
            .map(|i| {
                let o = i * PROBE_WORDS;
                (
                    [f(o + 6), f(o + 7), f(o + 8)],
                    f(o + 9),
                    [f(o + 10), f(o + 11), f(o + 12)],
                    r[o + 13] != 0,
                    r[o + 14] != 0,
                    (r[o + 15] != 0).then(|| (f(o + 16), f(o + 17))),
                )
            })
            .collect()
    }
}

/// The v3 walker the app registers under OMG_GPU=1 (omg_scene::v3walk)
/// and the engine uses under `Compute::Gpu` (`omg_core::v3::walker`):
/// one walk in flight on the world mesh uploaded once at startup; each
/// tick collects the previous tick's walk (never blocking) and submits
/// this one — the GPU's latency hides behind the 50 ms tick.
pub struct GpuV3Walker {
    pub tracer: GpuV3Tracer,
    /// Synchronous mode (tests, offline renders): this tick's own walk.
    pub sync: bool,
}

impl GpuV3Walker {
    pub fn new(mesh: &Mesh) -> Option<Self> {
        Some(Self { tracer: GpuV3Tracer::new(mesh)?, sync: false })
    }
}

impl omg_core::v3::walker::V3Walker for GpuV3Walker {
    fn walk(&mut self, job: &WalkJob) -> Walk {
        if !GpuV3Tracer::fits(job) {
            return Walk::Refused;
        }
        if self.sync {
            return match self.tracer.walk(job) {
                Some(w) => Walk::Fresh(w),
                None => Walk::Refused,
            };
        }
        let done = self.tracer.poll(false);
        if self.tracer.flight.is_none() {
            self.tracer.submit(job);
        }
        match done {
            Some(w) => Walk::Fresh(w),
            None => Walk::Pending,
        }
    }

    fn last_ms(&self) -> f32 {
        self.tracer.last_ms
    }
}
