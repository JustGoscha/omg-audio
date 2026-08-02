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
    try {
      const { initGpu } = await import('./gpu.js');
      gpu = await initGpu(w);
      if (gpu) {
        w.sim_gpu_enable();
        // world-late kernel (K2) only when its pipeline actually built
        if (gpu.meshOk && w.sim_wlate_enable) w.sim_wlate_enable();
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
  }
  const blocks = [];
  for (let i = 0; i < 11; i++) {
    const len = w.sim_params_len(i);
    const src = new Float32Array(w.memory.buffer, w.sim_params_ptr(i), len);
    blocks.push(src.slice().buffer);
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
    rays,
  }, rays ? [...blocks, state.buffer, rays] : [...blocks, state.buffer]);
}
