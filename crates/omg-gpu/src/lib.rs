//! Native wgpu host for the GPU trace kernels (GPU_PLAN.md Track A).
//! Phase 1: synchronous single-job dispatch, validated against the
//! Phase 0 goldens; the pipelined LateBackend ring is Phase 2.

pub mod layout;

use layout::{decode_echogram, GpuTraceJob, BINS_LEN, DIRS_LEN};
use omg_core::scene::Shoebox;
use omg_core::tracer::Echogram;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

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
