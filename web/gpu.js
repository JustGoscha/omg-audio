// WebGPU driver for the sim's stochastic traces (GPU_PLAN.md phase 3).
// The wasm stays freestanding: this file reads flat trace jobs out of
// wasm memory after each tick, dispatches the SAME trace_box.wgsl the
// native wgpu host runs, decodes the fixed-point output and injects f32
// echograms back for the sim to consume one tick later. Any failure at
// any point = the driver reports disabled and the sim keeps its inline
// CPU tracer — nothing to undo.

const JOB_F32S = 39; // must match omg-web GPU_JOB_F32S
const JOB_VERSION = 1; // must match omg-web GPU_JOB_VERSION
const NBINS = 300;
const BINS_WORDS = NBINS * 3;
const DIRS_WORDS = NBINS * 3;
const ENERGY_SCALE = 2 ** 30;
const DIR_SCALE = 2 ** 28;

export async function initGpu(wasm) {
  if (!navigator.gpu) return null;
  if (wasm.sim_gpu_job_version() !== JOB_VERSION) {
    console.warn('[gpu] job layout version mismatch — staying on CPU');
    return null;
  }
  let device;
  let pipeline;
  let ptPipeline;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    device = await adapter.requestDevice();
    const code = await (await fetch('../crates/omg-gpu/shaders/trace_box.wgsl')).text();
    const module = device.createShaderModule({ code });
    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'trace' },
    });
    const ptCode = await (await fetch('../crates/omg-gpu/shaders/pt_early.wgsl')).text();
    ptPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: ptCode }), entryPoint: 'discover' },
    });
  } catch (e) {
    console.warn('[gpu] init failed — staying on CPU:', e);
    return null;
  }

  const mkOut = (words) => device.createBuffer({
    size: words * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const mkRead = (words) => device.createBuffer({
    size: words * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // A small pool of dispatch slots so several sources' jobs overlap.
  const POOL = 4;
  const slots = Array.from({ length: POOL }, () => ({
    busy: false,
    job: device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    bins: mkOut(BINS_WORDS),
    dirs: mkOut(DIRS_WORDS),
    readBins: mkRead(BINS_WORDS),
    readDirs: mkRead(DIRS_WORDS),
    bind: null,
  }));
  for (const s of slots) {
    s.bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: s.job } },
        { binding: 1, resource: { buffer: s.bins } },
        { binding: 2, resource: { buffer: s.dirs } },
      ],
    });
  }

  let lastMs = 0;
  let ptMs = 0;
  let ptN = 0;
  // rolling duty: GPU-busy wall time per second of real time
  let busyMs = 0;
  let duty = 0;
  let windowT0 = performance.now();
  const noteBusy = (ms) => {
    busyMs += ms;
    const now = performance.now();
    if (now - windowT0 >= 1000) {
      duty = busyMs / (now - windowT0);
      busyMs = 0;
      windowT0 = now;
    }
  };

  // Build the 160-byte Job uniform (layout.rs / trace_box.wgsl v1) from
  // one flat job record.
  const packJob = (f, o) => {
    const buf = new ArrayBuffer(160);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32[0] = f[o + 3]; f32[1] = f[o + 4]; f32[2] = f[o + 5]; // size
    u32[3] = f[o + 1]; // n_rays
    f32[4] = f[o + 6]; f32[5] = f[o + 7]; f32[6] = f[o + 8]; // source
    u32[7] = f[o + 2]; // seed
    f32[8] = f[o + 9]; f32[9] = f[o + 10]; f32[10] = f[o + 11]; // listener
    f32[12] = f[o + 12]; f32[13] = f[o + 13]; f32[14] = f[o + 14]; // energy
    for (let face = 0; face < 6; face++) {
      const src = o + 15 + face * 4;
      const dst = 16 + face * 4;
      f32[dst] = f[src]; f32[dst + 1] = f[src + 1]; f32[dst + 2] = f[src + 2];
      f32[dst + 3] = f[src + 3]; // scattering
    }
    return buf;
  };

  const dispatch = async (slot, id, nRays, jobBytes, inject) => {
    slot.busy = true;
    const t0 = performance.now();
    try {
      device.queue.writeBuffer(slot.job, 0, jobBytes);
      const enc = device.createCommandEncoder();
      enc.clearBuffer(slot.bins);
      enc.clearBuffer(slot.dirs);
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, slot.bind);
      pass.dispatchWorkgroups(Math.ceil(nRays / 64));
      pass.end();
      enc.copyBufferToBuffer(slot.bins, 0, slot.readBins, 0, BINS_WORDS * 4);
      enc.copyBufferToBuffer(slot.dirs, 0, slot.readDirs, 0, DIRS_WORDS * 4);
      device.queue.submit([enc.finish()]);
      await Promise.all([
        slot.readBins.mapAsync(GPUMapMode.READ),
        slot.readDirs.mapAsync(GPUMapMode.READ),
      ]);
      const bins = new Uint32Array(slot.readBins.getMappedRange());
      const dirs = new Int32Array(slot.readDirs.getMappedRange());
      const echo = new Float32Array(BINS_WORDS + DIRS_WORDS);
      for (let i = 0; i < BINS_WORDS; i++) echo[i] = bins[i] / ENERGY_SCALE;
      for (let i = 0; i < DIRS_WORDS; i++) echo[BINS_WORDS + i] = dirs[i] / DIR_SCALE;
      slot.readBins.unmap();
      slot.readDirs.unmap();
      inject(id, echo);
      lastMs = performance.now() - t0;
      noteBusy(lastMs);
    } catch (e) {
      console.warn('[gpu] dispatch failed:', e);
    } finally {
      slot.busy = false;
    }
  };

  // PT-early discovery (Track C phase C4): one tiny slot — jobs are
  // 8 f32 in, 9 u32 out, and the wasm-side seeds keep the early field
  // correct while a bitmap is in flight.
  const PT_RAYS = 4096;
  const pt = {
    busy: false,
    job: device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    bitmap: device.createBuffer({
      size: 36,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
    read: device.createBuffer({ size: 36, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
  };
  pt.bind = device.createBindGroup({
    layout: ptPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pt.job } },
      { binding: 1, resource: { buffer: pt.bitmap } },
    ],
  });

  const ptDispatch = async (id, jobF32, injectPt) => {
    pt.busy = true;
    const t0 = performance.now();
    try {
      const buf = new ArrayBuffer(32);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      f32[0] = jobF32[1]; f32[1] = jobF32[2]; f32[2] = jobF32[3]; // size
      u32[3] = PT_RAYS;
      f32[4] = jobF32[4]; f32[5] = jobF32[5]; f32[6] = jobF32[6]; // listener
      u32[7] = jobF32[7]; // rot
      device.queue.writeBuffer(pt.job, 0, buf);
      const enc = device.createCommandEncoder();
      enc.clearBuffer(pt.bitmap);
      const pass = enc.beginComputePass();
      pass.setPipeline(ptPipeline);
      pass.setBindGroup(0, pt.bind);
      pass.dispatchWorkgroups(Math.ceil(PT_RAYS / 64));
      pass.end();
      enc.copyBufferToBuffer(pt.bitmap, 0, pt.read, 0, 36);
      device.queue.submit([enc.finish()]);
      await pt.read.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(pt.read.getMappedRange()).slice();
      pt.read.unmap();
      injectPt(id, words);
      ptMs = performance.now() - t0;
      ptN++;
      noteBusy(ptMs);
    } catch (e) {
      console.warn('[gpu] pt dispatch failed:', e);
    } finally {
      pt.busy = false;
    }
  };

  return {
    /// Drain wasm's queued jobs and dispatch them. `inject(id, echoF32)`
    /// delivers each decoded result. Jobs with no free slot are dropped —
    /// the trace gate re-fires them.
    pump(wasmExports, inject) {
      const n = wasmExports.sim_gpu_jobs_len();
      if (!n) return;
      const jobs = new Float32Array(wasmExports.memory.buffer, wasmExports.sim_gpu_jobs_ptr(), n);
      for (let o = 0; o + JOB_F32S <= n; o += JOB_F32S) {
        const slot = slots.find((s) => !s.busy);
        if (!slot) break;
        dispatch(slot, jobs[o], jobs[o + 1], packJob(jobs, o), inject);
      }
    },
    /// PT-early discovery jobs (8 f32 each). One in flight at a time;
    /// dropped jobs re-queue next tick and the seeds cover the gap.
    pumpPt(wasmExports, injectPt) {
      const n = wasmExports.sim_pt_jobs_len();
      if (!n || pt.busy) return;
      const jobs = new Float32Array(wasmExports.memory.buffer, wasmExports.sim_pt_jobs_ptr(), n);
      ptDispatch(jobs[0], jobs.slice(0, 8), injectPt);
    },
    stats: () => ({ ms: lastMs, duty, ptMs, ptN }),
  };
}
