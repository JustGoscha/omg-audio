// Simulation worker: wasm WorldSim ticked at 20 Hz with the latest listener
// pose from the main thread; posts flat ParamBlocks (transferred) + a small
// state buffer for the canvas viz. When WebGPU is available, the sim's
// stochastic traces run on the GPU through gpu.js (feature-detected here in
// the worker; any init failure keeps the inline CPU tracer).
let w = null;
let gpu = null;
let gpuOn = false;
let debugOn = false;
let pose = { x: 3.0, y: 3.0, z: 1.6, yaw: 0.0, projs: [] };
// Config sent while `init` is still awaiting wasm/WebGPU would be
// silently dropped (the handlers guard on `w`) — the classic symptom
// was the remembered `early=traced` arriving early and the sim
// starting on ism anyway. Queue everything until init finishes, then
// replay in order.
let preInit = [];

function handle(m) {
  if (m.type === 'pose') {
    pose = m;
  } else if (m.type === 'quality') {
    // sim-side quality ladder: 0 = high, 1 = med, 2 = low
    w.sim_set_quality(m.tier);
  } else if (m.type === 'override') {
    // pin one sim lever (id: 0 rays, 1 gate, 2 dome rays, 3 dome events,
    // 4 ISM order); value 0 hands it back to the tier
    w.sim_set_override(m.id, m.value);
  } else if (m.type === 'debug') {
    debugOn = !!m.on; // ray extraction only runs while the panel is open
  } else if (m.type === 'early') {
    // early-reflections backend: 0 = ism, 1 = traced (PT)
    w.sim_set_early(m.mode);
  } else if (m.type === 'mute') {
    // a muted source is skipped by the whole simulation
    if (w.sim_set_mute) m.srcs.forEach((i) => w.sim_set_mute(i, m.on ? 1 : 0));
  } else if (m.type === 'module') {
    // engine A/B switches: 0 = diffraction, 1 = furniture acoustics
    if (w.sim_set_module) w.sim_set_module(m.id, m.on ? 1 : 0);
  } else if (m.type === 'gpu') {
    // live A/B toggle; only meaningful when the driver initialized
    if (gpu) {
      gpuOn = !!m.on;
      if (gpuOn) {
        w.sim_gpu_enable();
        if (gpu.meshOk && w.sim_wlate_enable) w.sim_wlate_enable();
        if (gpu.meshOk && w.sim_wdisc_enable) w.sim_wdisc_enable();
        if (gpu.v3Ok && w.sim_v3_enable) w.sim_v3_enable();
      } else w.sim_gpu_disable();
    }
  }
}

onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    const { instance } = await WebAssembly.instantiate(m.bytes, {});
    w = instance.exports;
    w.sim_setup();
    // v4 kernels are built for the worklet's rate
    if (m.sampleRate && w.sim_set_rate) w.sim_set_rate(m.sampleRate);
    try {
      const { initGpu } = await import('./gpu.js');
      gpu = await initGpu(w);
      if (gpu) {
        w.sim_gpu_enable();
        // world kernels (K2 late, K3 discovery) only when the mesh
        // pipelines actually built
        if (gpu.meshOk && w.sim_wlate_enable) w.sim_wlate_enable();
        if (gpu.meshOk && w.sim_wdisc_enable) w.sim_wdisc_enable();
        // engine v3's listener walk (K5), over the same uploaded BVH
        if (gpu.v3Ok && w.sim_v3_enable) w.sim_v3_enable();
        gpuOn = true;
      }
      console.log(`[gpu] late field: ${gpu ? 'WebGPU' : 'CPU'}`);
    } catch (err) {
      console.warn('[gpu] unavailable, late field on CPU:', err);
    }
    const q = preInit;
    preInit = null;
    q.forEach(handle);
    setInterval(tick, 50);
  } else if (preInit) {
    preInit.push(m);
  } else {
    handle(m);
  }
};

const inject = (id, echo) => {
  new Float32Array(w.memory.buffer, w.sim_gpu_buf_ptr(), echo.length).set(echo);
  w.sim_gpu_inject(id);
};

const injectWd = (n, words) => {
  new Uint32Array(w.memory.buffer, w.sim_wdisc_buf_ptr(), words.length).set(words);
  w.sim_wdisc_inject(n);
};

// a v3 walk result: resolved bins (f32), then the strongest debug paths
// (raw u32 records); folds into the sim one tick late
const injectV3 = (data, dbg) => {
  const ptr = w.sim_v3_buf_ptr();
  new Float32Array(w.memory.buffer, ptr, data.length).set(data);
  if (dbg.length) new Uint32Array(w.memory.buffer, ptr + data.length * 4, dbg.length).set(dbg);
  w.sim_v3_inject(data.length, dbg.length / 28);
};

const injectPt = (id, words) => {
  new Uint32Array(w.memory.buffer, w.sim_pt_buf_ptr(), 9).set(words);
  w.sim_pt_inject(id);
};

function tick() {
  if (!w) return;
  const doors = new Float32Array(w.memory.buffer, w.sim_door_ptr(), 16);
  (pose.doors || []).forEach((v, i) => { doors[i] = v; });
  const dyn = new Float32Array(w.memory.buffer, w.sim_dyn_ptr(), 24);
  dyn.fill(0);
  (pose.projs || []).forEach((p) => {
    const slot = p[0];
    if (slot >= 0 && slot < 6) dyn.set([p[1], p[2], p[3], p[4] === undefined ? 1 : p[4]], slot * 4);
  });
  const t0 = performance.now();
  w.sim_tick(pose.x, pose.y, pose.z == null ? 1.6 : pose.z, pose.yaw);
  const tickMs = performance.now() - t0;
  if (gpu && gpuOn) {
    gpu.pump(w, inject);
    gpu.pumpPt(w, injectPt);
    gpu.pumpWorldLate(w, inject);
    gpu.pumpWorldDisc(w, injectWd);
    gpu.pumpV3(w, injectV3);
  }
  const blocks = [];
  const nsrc = w.sim_nsrc ? w.sim_nsrc() : 11;
  for (let i = 0; i < nsrc; i++) {
    const len = w.sim_params_len(i);
    const src = new Float32Array(w.memory.buffer, w.sim_params_ptr(i), len);
    blocks.push(src.slice().buffer);
  }
  // engine v4: a finished kernel (at most one a tick) and the sources
  // whose kernel the worklet drops — sent with this tick's blocks, so a
  // block that lost its parametric reverb never arrives before its kernel
  let kernel = null;
  let clears = [];
  if (w.sim_kernel_src) {
    const ks = w.sim_kernel_src();
    if (ks >= 0) {
      const buf = new Float32Array(w.memory.buffer, w.sim_kernel_ptr(), w.sim_kernel_len()).slice().buffer;
      kernel = { src: ks, buf };
    }
    const nc = w.sim_kernel_clears_len();
    if (nc) clears = Array.from(new Uint32Array(w.memory.buffer, w.sim_kernel_clears_ptr(), nc));
  }
  const state = new Float32Array(w.memory.buffer, w.sim_state_ptr(), w.sim_state_len()).slice();
  let rays = null;
  if (debugOn) {
    const rn = w.sim_debug_rays_len();
    if (rn) rays = new Float32Array(w.memory.buffer, w.sim_debug_rays_ptr(), rn).slice().buffer;
  }
  postMessage({
    type: 'tick', blocks, state: state.buffer, envOff: w.sim_env_off(), tickMs,
    gpu: gpu && gpuOn ? 1 : 0, gpuAvail: gpu ? 1 : 0,
    gpuMs: gpu ? gpu.stats().ms : 0, gpuDuty: gpu ? gpu.stats().duty : 0,
    early: w.sim_early_mode(), ptMs: gpu ? gpu.stats().ptMs : 0, ptN: gpu ? gpu.stats().ptN : 0,
    wdMs: gpu ? gpu.stats().wdMs : 0, wdN: gpu ? gpu.stats().wdN : 0,
    v3Ms: gpu ? gpu.stats().v3Ms : 0, v3N: gpu ? gpu.stats().v3N : 0,
    rays, kernel, clears, yaw: pose.yaw || 0, dense: w.sim_dense_count ? w.sim_dense_count() : 0,
  }, [...blocks, state.buffer, ...(rays ? [rays] : []), ...(kernel ? [kernel.buf] : [])]);
}
