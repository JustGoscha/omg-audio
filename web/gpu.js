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

  // C6d kernel K2: the world-mesh tracer. The BVH uploads ONCE from wasm
  // memory; per job only a 64-byte uniform + the panel overlays travel.
  // Any failure = meshPipeline stays null and traced-mode reverb runs on
  // the in-wasm CPU tracer, exactly as without GPU.
  let mesh = null;
  try {
    if (wasm.sim_wlate_version && wasm.sim_wlate_version() === 2) {
      const code = await (await fetch('../crates/omg-gpu/shaders/trace_mesh.wgsl')).text();
      const pipe = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'trace' },
      });
      const staticBuf = (lenFn, ptrFn) => {
        const n = lenFn();
        const buf = device.createBuffer({
          size: n * 4,
          usage: GPUBufferUsage.STORAGE,
          mappedAtCreation: true,
        });
        new Uint32Array(buf.getMappedRange()).set(
          new Uint32Array(wasm.memory.buffer, ptrFn(), n),
        );
        buf.unmap();
        return buf;
      };
      mesh = {
        pipe,
        busy: false,
        nodes: staticBuf(wasm.sim_mesh_nodes_len, wasm.sim_mesh_nodes_ptr),
        prims: staticBuf(wasm.sim_mesh_prims_len, wasm.sim_mesh_prims_ptr),
        mats: staticBuf(wasm.sim_mesh_mats_len, wasm.sim_mesh_mats_ptr),
        job: device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        panels: device.createBuffer({ size: 64 * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      };
      // K3: chain discovery over the same BVH buffers
      const discCode = await (await fetch('../crates/omg-gpu/shaders/discover_mesh.wgsl')).text();
      mesh.discPipe = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: discCode }), entryPoint: 'discover' },
      });
      const DISC_CAP = 16384;
      mesh.discBusy = false;
      mesh.discJob = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      mesh.discChains = device.createBuffer({ size: DISC_CAP * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      mesh.discCount = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      mesh.discRead = device.createBuffer({ size: 4 + DISC_CAP * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      // furniture overlay boxes (static, like the BVH); ≥1 word so the
      // binding is never zero-sized
      const nFurnWords = wasm.sim_wdisc_boxes_len ? wasm.sim_wdisc_boxes_len() : 0;
      mesh.discNBoxes = nFurnWords / 8;
      mesh.discBase = wasm.sim_wdisc_base ? wasm.sim_wdisc_base() : 0;
      mesh.discBoxes = device.createBuffer({
        size: Math.max(32, nFurnWords * 4),
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      if (nFurnWords) {
        new Uint32Array(mesh.discBoxes.getMappedRange()).set(
          new Uint32Array(wasm.memory.buffer, wasm.sim_wdisc_boxes_ptr(), nFurnWords),
        );
      }
      mesh.discBoxes.unmap();
      mesh.discBind = device.createBindGroup({
        layout: mesh.discPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: mesh.discJob } },
          { binding: 1, resource: { buffer: mesh.nodes } },
          { binding: 2, resource: { buffer: mesh.prims } },
          { binding: 3, resource: { buffer: mesh.discChains } },
          { binding: 4, resource: { buffer: mesh.discCount } },
          { binding: 5, resource: { buffer: mesh.discBoxes } },
        ],
      });
      console.info('[gpu] world mesh uploaded:',
        wasm.sim_mesh_prims_len() / 12, 'prims,', wasm.sim_mesh_nodes_len() / 8, 'bvh nodes');
    }
  } catch (e) {
    console.warn('[gpu] mesh kernel unavailable — world late stays on CPU:', e);
    mesh = null;
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

  // one world-late slot: the sim budgets ONE world trace per tick, so a
  // single in-flight dispatch matches the producer exactly
  if (mesh) {
    mesh.bins = mkOut(BINS_WORDS);
    mesh.dirs = mkOut(DIRS_WORDS);
    mesh.readBins = mkRead(BINS_WORDS);
    mesh.readDirs = mkRead(DIRS_WORDS);
    mesh.bind = device.createBindGroup({
      layout: mesh.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mesh.job } },
        { binding: 1, resource: { buffer: mesh.nodes } },
        { binding: 2, resource: { buffer: mesh.prims } },
        { binding: 3, resource: { buffer: mesh.mats } },
        { binding: 4, resource: { buffer: mesh.panels } },
        { binding: 5, resource: { buffer: mesh.bins } },
        { binding: 6, resource: { buffer: mesh.dirs } },
      ],
    });
  }

  const meshDispatch = async (id, nRays, jobBytes, panelF32, inject) => {
    mesh.busy = true;
    const t0 = performance.now();
    try {
      device.queue.writeBuffer(mesh.job, 0, jobBytes);
      if (panelF32.length) device.queue.writeBuffer(mesh.panels, 0, panelF32);
      const enc = device.createCommandEncoder();
      enc.clearBuffer(mesh.bins);
      enc.clearBuffer(mesh.dirs);
      const pass = enc.beginComputePass();
      pass.setPipeline(mesh.pipe);
      pass.setBindGroup(0, mesh.bind);
      pass.dispatchWorkgroups(Math.ceil(nRays / 64));
      pass.end();
      enc.copyBufferToBuffer(mesh.bins, 0, mesh.readBins, 0, BINS_WORDS * 4);
      enc.copyBufferToBuffer(mesh.dirs, 0, mesh.readDirs, 0, DIRS_WORDS * 4);
      device.queue.submit([enc.finish()]);
      await Promise.all([
        mesh.readBins.mapAsync(GPUMapMode.READ),
        mesh.readDirs.mapAsync(GPUMapMode.READ),
      ]);
      const bins = new Uint32Array(mesh.readBins.getMappedRange());
      const dirs = new Int32Array(mesh.readDirs.getMappedRange());
      const echo = new Float32Array(BINS_WORDS + DIRS_WORDS);
      for (let i = 0; i < BINS_WORDS; i++) echo[i] = bins[i] / ENERGY_SCALE;
      for (let i = 0; i < DIRS_WORDS; i++) echo[BINS_WORDS + i] = dirs[i] / DIR_SCALE;
      mesh.readBins.unmap();
      mesh.readDirs.unmap();
      inject(id, echo);
      noteBusy(performance.now() - t0);
    } catch (e) {
      console.warn('[gpu] world dispatch failed:', e);
    } finally {
      mesh.busy = false;
    }
  };

  let wdMs = 0;
  let wdN = 0;
  const DISC_RAYS = 4096;
  const discDispatch = async (jobF32, injectWd) => {
    mesh.discBusy = true;
    const t0 = performance.now();
    try {
      const buf = new ArrayBuffer(32);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      u32[0] = DISC_RAYS;
      u32[1] = jobF32[3]; // rot
      u32[2] = jobF32[4] ? mesh.discNBoxes : 0; // furniture switch
      u32[3] = mesh.discBase;
      f32[4] = jobF32[0]; f32[5] = jobF32[1]; f32[6] = jobF32[2]; // listener
      device.queue.writeBuffer(mesh.discJob, 0, buf);
      const enc = device.createCommandEncoder();
      enc.clearBuffer(mesh.discCount);
      const pass = enc.beginComputePass();
      pass.setPipeline(mesh.discPipe);
      pass.setBindGroup(0, mesh.discBind);
      pass.dispatchWorkgroups(Math.ceil(DISC_RAYS / 64));
      pass.end();
      enc.copyBufferToBuffer(mesh.discCount, 0, mesh.discRead, 0, 4);
      enc.copyBufferToBuffer(mesh.discChains, 0, mesh.discRead, 4, 16384 * 8);
      device.queue.submit([enc.finish()]);
      await mesh.discRead.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(mesh.discRead.getMappedRange());
      const n = Math.min(words[0], 16384);
      const chains = words.slice(1, 1 + n * 2);
      mesh.discRead.unmap();
      injectWd(n, chains);
      wdMs = performance.now() - t0;
      wdN++;
      noteBusy(wdMs);
    } catch (e) {
      console.warn('[gpu] discovery dispatch failed:', e);
    } finally {
      mesh.discBusy = false;
    }
  };

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
    /// World-late jobs (K2): fixed 778-f32 stride — id, n_rays, seed,
    /// source, listener, n_panels, then panels laid out exactly like the
    /// kernel's 48-byte Panel struct (copied verbatim). One in flight;
    /// results ride the SAME inject path as the box traces.
    pumpWorldLate(wasmExports, inject) {
      if (!mesh || mesh.busy || !wasmExports.sim_wlate_jobs_len) return;
      const n = wasmExports.sim_wlate_jobs_len();
      if (!n) return;
      const jobs = new Float32Array(wasmExports.memory.buffer, wasmExports.sim_wlate_jobs_ptr(), n);
      // one slot: take the FIRST job; the others' gates re-fire
      const nPanels = Math.min(jobs[9], 64);
      const buf = new ArrayBuffer(64);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      u32[0] = jobs[1]; // n_rays
      u32[1] = jobs[2]; // seed
      u32[2] = nPanels;
      f32[4] = jobs[3]; f32[5] = jobs[4]; f32[6] = jobs[5]; // source
      f32[8] = jobs[6]; f32[9] = jobs[7]; f32[10] = jobs[8]; // listener
      f32[12] = 1.0; f32[13] = 1.0; f32[14] = 1.0; // unit energy
      meshDispatch(jobs[0], jobs[1], buf, jobs.slice(10, 10 + nPanels * 12), inject);
    },
    /// World-discovery jobs (K3): one 4-f32 job per tick, newest wins;
    /// the raw chain list injects back and the wasm TTL cache dedups.
    pumpWorldDisc(wasmExports, injectWd) {
      if (!mesh || mesh.discBusy || !wasmExports.sim_wdisc_jobs_len) return;
      const n = wasmExports.sim_wdisc_jobs_len();
      if (!n) return;
      const job = new Float32Array(wasmExports.memory.buffer, wasmExports.sim_wdisc_jobs_ptr(), 5).slice();
      discDispatch(job, injectWd);
    },
    /// True when the world-mesh kernel compiled and the BVH uploaded —
    /// the worker registers the wasm-side world proxy only then.
    meshOk: !!mesh,
    stats: () => ({ ms: lastMs, duty, ptMs, ptN, wdMs, wdN }),
  };
}
