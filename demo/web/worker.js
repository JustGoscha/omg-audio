// Simulation worker: the engine's Simulation (omg-audio in wasm — the same
// engine the Godot demo runs) with the demo town, ticked at 20 Hz from the
// latest listener pose the page posts. Each tick posts the frame (flat),
// the convolution kernels built that tick, the sources whose kernel went,
// the page's state and (debug) the traced paths.
let w = null;
let gpu = null;
let gpuOn = false;
// the share of the GPU's time the walk may take: the page's frames first
let gpuBudget = 0.2;
let debugOn = false;
let pose = { x: 3.0, y: 3.0, z: 1.6, yaw: 0.0, projs: [] };
// messages that arrive while init still awaits wasm are replayed after
let preInit = [];

function handle(m) {
  if (m.type === 'pose') {
    pose = m;
  } else if (m.type === 'quality') {
    // the page's tiers: 0 high, 1 med, 2 low
    w.sim_set_quality(m.tier);
  } else if (m.type === 'override') {
    // a pinned tracer lever (5 rays, 6 reflections, 7 late rays); 0 = tier
    w.sim_set_override(m.id, m.value);
  } else if (m.type === 'debug') {
    debugOn = !!m.on;
    w.sim_set_debug(debugOn ? 1 : 0);
  } else if (m.type === 'early') {
    // the engine: 3 = v3, anything else v4
    w.sim_set_v4(m.mode === 3 ? 0 : 1);
  } else if (m.type === 'mute') {
    // a muted source is not traced at all
    m.srcs.forEach((i) => w.sim_set_mute(i, m.on ? 1 : 0));
  } else if (m.type === 'module') {
    // engine switches: 0 = diffraction
    w.sim_set_module(m.id, m.on ? 1 : 0);
  } else if (m.type === 'gpu') {
    // the walk on the GPU or the CPU (only where WebGPU came up)
    if (gpu) {
      gpuOn = !!m.on;
      if (gpuOn) w.sim_gpu_enable();
      else w.sim_gpu_disable();
    }
  } else if (m.type === 'gpuBudget') {
    gpuBudget = m.value;
  }
}

onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    const { instance } = await WebAssembly.instantiate(m.bytes, {});
    w = instance.exports;
    w.sim_setup(m.sampleRate || 48000);
    try {
      const { initGpu } = await import('./gpu.js');
      gpu = await initGpu(w);
      if (gpu) {
        w.sim_gpu_enable();
        gpuOn = true;
      }
      console.log(`[gpu] walk: ${gpu ? 'WebGPU' : 'CPU'}`);
    } catch (err) {
      console.warn('[gpu] unavailable, the walk stays on the CPU:', err);
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

function tick() {
  if (!w) return;
  // the page's door leaves (openness) and dynamic slots (balls, cars, feet)
  const doors = new Float32Array(w.memory.buffer, w.sim_door_ptr(), 16);
  doors.fill(-1);
  (pose.doors || []).forEach((v, i) => { if (i < 16) doors[i] = v; });
  const dyn = new Float32Array(w.memory.buffer, w.sim_dyn_ptr(), 24);
  dyn.fill(0);
  (pose.projs || []).forEach((p) => {
    const slot = p[0];
    if (slot >= 0 && slot < 6) dyn.set([p[1], p[2], p[3], p[4] === undefined ? 1 : p[4]], slot * 4);
  });
  const t0 = performance.now();
  w.sim_tick(pose.x, pose.y, pose.z == null ? 1.6 : pose.z, pose.yaw || 0);
  const tickMs = performance.now() - t0;
  if (gpu && gpuOn) gpu.pump(w, gpuBudget);
  // views after the tick: wasm memory may have grown
  const f32 = (ptr, len) => new Float32Array(w.memory.buffer, ptr, len).slice();
  const frame = f32(w.sim_frame_ptr(), w.sim_frame_len()).buffer;
  const kernels = [];
  for (let i = 0; i < w.sim_kernel_count(); i++) {
    kernels.push({ src: w.sim_kernel_src(i), buf: f32(w.sim_kernel_ptr(i), w.sim_kernel_len(i)).buffer });
  }
  const nc = w.sim_clears_len();
  const clears = nc ? Array.from(new Uint32Array(w.memory.buffer, w.sim_clears_ptr(), nc)) : [];
  const state = f32(w.sim_state_ptr(), w.sim_state_len());
  let rays = null;
  if (debugOn && w.sim_rays_len()) rays = f32(w.sim_rays_ptr(), w.sim_rays_len()).buffer;
  postMessage({
    type: 'tick', frame, kernels, clears, state: state.buffer, envOff: w.sim_env_off(), tickMs,
    gpu: gpu && gpuOn ? 1 : 0, gpuAvail: gpu ? 1 : 0,
    gpuMs: gpu ? gpu.stats().ms : 0, v3Ms: gpu ? gpu.stats().ms : 0, v3N: gpu ? gpu.stats().walks : 0,
    gpuMaxMs: gpu ? gpu.stats().maxMs : 0,
    early: w.sim_v4() ? 4 : 3, rays, dense: w.sim_dense_count(),
  }, [frame, ...kernels.map((k) => k.buf), state.buffer, ...(rays ? [rays] : [])]);
}
