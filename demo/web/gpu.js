// WebGPU driver for the engine's ray walk (the early rays; the late ones are
// the CPU's): the same shader the native host runs
// (crates/omg-gpu/shaders/v3_walk.wgsl) over the simulation's own mesh.
// The wasm (lib.rs, "GPU walk") leaves one job per tick; this takes it, runs
// it as a few short dispatches — each submitted alone and followed by a
// rest, so the walk takes at most `budget` of the GPU's time and the page's
// frames come first — and hands the result back, folded the tick after.
// Any failure: the driver reports none and the walk stays on the CPU.

const V3_VERSION = 7; // omg_core::v3::fold::WALK_LAYOUT_VERSION
const V3_SKY = 384;
const V3_SRC = 17164; // SOURCE_WORDS
const V3_MAX_SRC = 64;
const V3_SCENE_WORDS = 64 * 12 + 256 * 16 + 8 * 8 + 16384 * 8; // fold MAX_SCENE_WORDS

export async function initGpu(wasm) {
  if (!navigator.gpu || !wasm.sim_v3_version || wasm.sim_v3_version() !== V3_VERSION) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  // GPU timestamps where the adapter has them: the walk's own time
  const stamps = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice(stamps ? { requiredFeatures: ['timestamp-query'] } : undefined);
  const MAX_STAMPED = 16; // passes timed per walk
  const query = stamps && {
    set: device.createQuerySet({ type: 'timestamp', count: 2 * MAX_STAMPED }),
    resolved: device.createBuffer({ size: 16 * MAX_STAMPED, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
    read: device.createBuffer({ size: 16 * MAX_STAMPED, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
  };
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
  const outBytes = (V3_SKY + V3_MAX_SRC * V3_SRC) * 4;
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
  // the early rays' state across a walk's dispatches
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
  // what a pass's wait holds besides the walk's own GPU time — the round
  // trip, the dispatch's setup: idle GPU time, not the walk's share. With
  // timestamps, a walk's waits less its GPU time per pass; without, an
  // empty submission's round trip probed every 16 walks (the lowest seen,
  // drifting up if it slows for good)
  let latency = null;
  const floorLatency = (l) => {
    latency = latency === null || l < latency ? Math.max(0, l) : latency + 0.1 * (l - latency);
  };
  async function probeLatency() {
    const t0 = performance.now();
    device.queue.submit([]);
    await device.queue.onSubmittedWorkDone();
    floorLatency(performance.now() - t0);
  }

  async function run(words, passes, w36, outWords, budget) {
    const nRays = words[3] + words[7];
    if (!query && walks % 16 === 0) await probeLatency();
    device.queue.writeBuffer(scene, 0, words.subarray(40));
    let sum = 0;
    let longest = 0;
    let waited = 0;
    const stamped = query ? Math.min(passes, MAX_STAMPED) : 0;
    for (let pass = 0; pass < passes; pass++) {
      const hdr = words.slice(0, 40);
      hdr[36] = w36[pass];
      device.queue.writeBuffer(job, 0, hdr);
      const enc = device.createCommandEncoder();
      if (pass === 0) enc.clearBuffer(acc, 0, outWords * 8);
      const last = pass + 1 === passes;
      const cp = enc.beginComputePass(pass < stamped
        ? { timestampWrites: { querySet: query.set, beginningOfPassWriteIndex: 2 * pass, endOfPassWriteIndex: 2 * pass + 1 } }
        : undefined);
      cp.setPipeline(walkPipe);
      cp.setBindGroup(0, walkBind);
      cp.dispatchWorkgroups(Math.ceil(nRays / 64));
      if (last) {
        cp.setPipeline(resolvePipe);
        cp.setBindGroup(0, resolveBind);
        cp.dispatchWorkgroups(Math.ceil(outWords / 64));
      }
      cp.end();
      if (last) {
        enc.copyBufferToBuffer(out, 0, read, 0, outWords * 4);
        if (stamped) {
          enc.resolveQuerySet(query.set, 0, 2 * stamped, query.resolved, 0);
          enc.copyBufferToBuffer(query.resolved, 0, query.read, 0, 16 * stamped);
        }
      }
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      // submit to done less the round trip: waiting behind the page's
      // frames counts too
      const wait = performance.now() - t0;
      waited += wait;
      const busyMs = Math.max(0, wait - (latency || 0));
      sum += busyMs;
      longest = Math.max(longest, busyMs);
      await sleep(busyMs * (1 / budget - 1));
    }
    await read.mapAsync(GPUMapMode.READ, 0, outWords * 4);
    const data = new Float32Array(read.getMappedRange(0, outWords * 4)).slice();
    read.unmap();
    if (stamped) {
      await query.read.mapAsync(GPUMapMode.READ, 0, 16 * stamped);
      const t = new BigUint64Array(query.read.getMappedRange(0, 16 * stamped));
      let gpu = 0;
      let max = 0;
      for (let p = 0; p < stamped; p++) {
        const ms = t[2 * p + 1] > t[2 * p] ? Number(t[2 * p + 1] - t[2 * p]) * 1e-6 : 0;
        gpu += ms;
        max = Math.max(max, ms);
      }
      query.read.unmap();
      if (stamped === passes && gpu > 0) {
        floorLatency((waited - gpu) / passes);
        return { data, sum: gpu, longest: max };
      }
    }
    return { data, sum, longest };
  }

  return {
    /// Run the newest walk job if none is in flight (call once per tick).
    pump(w, budget) {
      if (busy) return;
      const n = w.sim_v3_take();
      if (!n) return;
      busy = true;
      ensureState(w.sim_v3_dbg_words());
      const words = new Uint32Array(w.memory.buffer, w.sim_v3_job_ptr(), n).slice();
      const passes = w.sim_v3_passes();
      const w36 = Array.from({ length: passes }, (_, p) => w.sim_v3_word36(p));
      const outWords = w.sim_v3_out_words();
      run(words, passes, w36, outWords, Math.min(1, Math.max(0.02, budget)))
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
