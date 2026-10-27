// WebGPU driver for the engine's ray walk: the same shader the native host
// runs (crates/omg-gpu/shaders/v3_walk.wgsl) over the simulation's own mesh.
// The wasm (lib.rs, "GPU walk") leaves one job per tick; this takes it, runs
// it as a few short dispatches — each submitted alone and followed by a
// rest, so the walk takes at most `budget` of the GPU's time and the page's
// frames come first — and hands the result back, folded the tick after.
// Any failure: the driver reports none and the walk stays on the CPU.

const V3_VERSION = 5; // omg_core::v3::fold::WALK_LAYOUT_VERSION
const V3_SKY = 384;
const V3_SRC = 17164; // SOURCE_WORDS
const V3_MAX_SRC = 64;
const V3_SCENE_WORDS = 64 * 12 + 256 * 16 + 8 * 8 + 16384 * 8; // fold MAX_SCENE_WORDS

export async function initGpu(wasm) {
  if (!navigator.gpu || !wasm.sim_v3_version || wasm.sim_v3_version() !== V3_VERSION) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const code = await (await fetch('../../crates/omg-gpu/shaders/v3_walk.wgsl')).text();
  const module = device.createShaderModule({ code });
  const walkPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'walk' } });
  const resolvePipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'resolve' } });
  // the simulation's mesh, uploaded once
  const staticBuf = (len, ptr) => {
    const n = Math.max(len, 4);
    const b = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    if (len) device.queue.writeBuffer(b, 0, new Uint32Array(wasm.memory.buffer, ptr, len).slice());
    return b;
  };
  const nodes = staticBuf(wasm.sim_mesh_nodes_len(), wasm.sim_mesh_nodes_ptr());
  const prims = staticBuf(wasm.sim_mesh_prims_len(), wasm.sim_mesh_prims_ptr());
  const mats = staticBuf(wasm.sim_mesh_mats_len(), wasm.sim_mesh_mats_ptr());
  const outBytes = (V3_SKY + V3_MAX_SRC * V3_SRC + 1) * 4;
  const job = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const scene = device.createBuffer({ size: V3_SCENE_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const acc = device.createBuffer({ size: outBytes * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const out = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const resolveBind = device.createBindGroup({
    layout: resolvePipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: job } },
      { binding: 4, resource: { buffer: scene } },
      { binding: 5, resource: { buffer: acc } },
      { binding: 6, resource: { buffer: out } },
    ],
  });
  // the rays' state across dispatches (and the late pool across walks)
  let state = null;
  let stateWords = 0;
  let walkBind = null;
  const ensureState = (words) => {
    if (words <= stateWords) return false;
    stateWords = 1 << Math.ceil(Math.log2(Math.max(words, 64)));
    state = device.createBuffer({ size: stateWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    walkBind = device.createBindGroup({
      layout: walkPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: job } },
        { binding: 1, resource: { buffer: nodes } },
        { binding: 2, resource: { buffer: prims } },
        { binding: 3, resource: { buffer: mats } },
        { binding: 4, resource: { buffer: scene } },
        { binding: 5, resource: { buffer: acc } },
        { binding: 7, resource: { buffer: state } },
      ],
    });
    return true;
  };
  ensureState(64);

  let busy = false;
  let lastMs = 0;
  let maxMs = 0;
  let walks = 0;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function run(words, passes, w36, reset, outWords, budget) {
    const nRays = words[3] + words[7];
    device.queue.writeBuffer(scene, 0, words.subarray(40));
    let sum = 0;
    let longest = 0;
    for (let pass = 0; pass < passes; pass++) {
      const hdr = words.slice(0, 40);
      hdr[36] = w36[pass];
      hdr[39] = pass === 0 ? reset : 0;
      device.queue.writeBuffer(job, 0, hdr);
      const enc = device.createCommandEncoder();
      if (pass === 0) enc.clearBuffer(acc, 0, outWords * 8);
      const last = pass + 1 === passes;
      const cp = enc.beginComputePass();
      cp.setPipeline(walkPipe);
      cp.setBindGroup(0, walkBind);
      cp.dispatchWorkgroups(Math.ceil(nRays / 64));
      if (last) {
        cp.setPipeline(resolvePipe);
        cp.setBindGroup(0, resolveBind);
        cp.dispatchWorkgroups(Math.ceil(outWords / 64));
      }
      cp.end();
      if (last) enc.copyBufferToBuffer(out, 0, read, 0, outWords * 4);
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      // submit to done: waiting behind the page's frames counts too
      const busyMs = performance.now() - t0;
      sum += busyMs;
      longest = Math.max(longest, busyMs);
      await sleep(busyMs * (1 / budget - 1));
    }
    await read.mapAsync(GPUMapMode.READ, 0, outWords * 4);
    const data = new Float32Array(read.getMappedRange(0, outWords * 4)).slice();
    read.unmap();
    return { data, sum, longest };
  }

  return {
    /// Run the newest walk job if none is in flight (call once per tick).
    pump(w, budget) {
      if (busy) return;
      const n = w.sim_v3_take();
      if (!n) return;
      busy = true;
      const words = new Uint32Array(w.memory.buffer, w.sim_v3_job_ptr(), n).slice();
      const passes = w.sim_v3_passes();
      const w36 = Array.from({ length: passes }, (_, p) => w.sim_v3_word36(p));
      const reset = w.sim_v3_reset();
      const outWords = w.sim_v3_out_words();
      if (ensureState(w.sim_v3_dbg_words())) w.sim_v3_pool_lost();
      run(words, passes, w36, reset, outWords, Math.min(1, Math.max(0.02, budget)))
        .then(({ data, sum, longest }) => {
          new Float32Array(w.memory.buffer, w.sim_v3_buf_ptr(data.length), data.length).set(data);
          w.sim_v3_inject(data.length, sum, longest);
          lastMs = sum;
          maxMs = longest;
          walks++;
        })
        .catch((e) => console.warn('[gpu] walk failed:', e))
        .finally(() => { busy = false; });
    },
    stats: () => ({ ms: lastMs, maxMs, walks }),
  };
}
