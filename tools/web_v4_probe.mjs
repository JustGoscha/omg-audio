#!/usr/bin/env node
// Engine v4 in the wasm build, driven exactly as worker.js + worklet.js
// do (kernels, clears and the listener's yaw ride with each tick's
// blocks): a click every 2 s from the narrator in the Great Hall, the
// listener a few metres away. Checks, for early = v4, that kernels are
// built and handed over within the budget and that the click leaves a
// decaying tail; prints the sim tick and the worklet's render cost next
// to v3's.
//
//   node tools/web_v4_probe.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmBytes = readFileSync(join(root, 'web/omg_web.wasm'));
const FS = 48000;
const SECS = 8;
const VOICE = 1; // the narrator's source index

async function run(mode) {
  const sim = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
  const eng = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
  eng.eng_init(FS);
  const put = (bytes, allocFn, doneFn) => {
    const ptr = allocFn(bytes.byteLength);
    new Uint8Array(eng.memory.buffer, ptr, bytes.byteLength).set(bytes);
    doneFn();
  };
  put(readFileSync(join(root, 'assets/hrir_grid.bin')), eng.eng_hrir_grid_alloc, eng.eng_hrir_grid_done);
  put(readFileSync(join(root, 'assets/hrir_dodeca20.bin')), eng.eng_hrir_speakers_alloc, eng.eng_hrir_speakers_done);
  const nsrc = sim.sim_nsrc ? 0 : 0;
  // every source silent but the voice: a 3 ms click every 2 s
  for (let i = 0; i < 15; i++) {
    const n = 2 * FS;
    const ptr = eng.eng_source_alloc(i, n);
    const buf = new Float32Array(eng.memory.buffer, ptr, n);
    buf.fill(0);
    if (i === VOICE) for (let k = 0; k < 144; k++) buf[k] = (1 - k / 144) * 0.9 * (k % 2 ? -1 : 1);
  }
  void nsrc;
  sim.sim_setup();
  sim.sim_set_rate(FS);
  sim.sim_set_early(mode);

  let kernels = [];
  let tickMs = 0;
  let renderMs = 0;
  let maxBlockMs = 0;
  const out = [];
  for (let t = 0; t < SECS * 20; t++) {
    const t0 = performance.now();
    sim.sim_tick(4.0, 20.0, 1.6, 0.3);
    tickMs += performance.now() - t0;
    // worker → worklet: clears, kernel, yaw, then the blocks
    const nc = sim.sim_kernel_clears_len();
    if (nc) Array.from(new Uint32Array(sim.memory.buffer, sim.sim_kernel_clears_ptr(), nc)).forEach((s) => eng.eng_clear_kernel(s));
    const ks = sim.sim_kernel_src();
    if (ks >= 0) {
      const words = new Float32Array(sim.memory.buffer, sim.sim_kernel_ptr(), sim.sim_kernel_len()).slice();
      const c0 = performance.now();
      const ptr = eng.eng_kernel_alloc(words.length);
      new Float32Array(eng.memory.buffer, ptr, words.length).set(words);
      eng.eng_set_kernel(ks, words.length);
      kernels.push({ tick: t, src: ks, mb: (words.length * 4 / 1e6), copyMs: performance.now() - c0 });
    }
    eng.eng_set_listener_yaw(0.3);
    for (let i = 0; i < 15; i++) {
      const len = sim.sim_params_len(i);
      const src = new Float32Array(sim.memory.buffer, sim.sim_params_ptr(i), len);
      new Float32Array(eng.memory.buffer, eng.eng_param_buf_ptr(), len).set(src);
      eng.eng_set_params(i, len);
    }
    if (eng.eng_params_commit) eng.eng_params_commit();
    for (let blk = 0; blk < FS / 20 / 128; blk++) {
      const r0 = performance.now();
      eng.eng_process(128);
      const dt = performance.now() - r0;
      renderMs += dt;
      if (t > 40) maxBlockMs = Math.max(maxBlockMs, dt);
      const l = new Float32Array(eng.memory.buffer, eng.eng_out_l(), 128);
      const r = new Float32Array(eng.memory.buffer, eng.eng_out_r(), 128);
      for (let k = 0; k < 128; k++) out.push(0.5 * (l[k] * l[k] + r[k] * r[k]));
    }
  }
  return { sim, out, kernels, tickMs: tickMs / (SECS * 20), load: renderMs / (SECS * 1000), maxBlockMs, dense: sim.sim_dense_count() };
}

const db = (x) => 10 * Math.log10(Math.max(x, 1e-30));
// the click that starts at 6 s (kernels long since in), windows after it
const tail = (out) => [[0.05, 0.15], [0.15, 0.3], [0.3, 0.6], [0.6, 1.0], [1.0, 1.5], [1.5, 1.95]].map(([a, b]) => {
  const i0 = Math.floor((6 + a) * FS);
  const i1 = Math.floor((6 + b) * FS);
  let s = 0;
  for (let i = i0; i < i1; i++) s += out[i];
  return db(s / (i1 - i0));
});

const v3 = await run(3);
const v4 = await run(4);
for (const [name, r] of [['v3', v3], ['v4', v4]]) {
  console.log(`${name}: sim tick ${r.tickMs.toFixed(1)} ms · worklet render ${(r.load * 100).toFixed(1)} % of real time, worst 128-sample block ${r.maxBlockMs.toFixed(2)} ms · convolving ${r.dense}`);
  console.log(`    tail after a click (dB, 0.05-0.15 … 1.5-1.95 s): ${tail(r.out).map((x) => x.toFixed(0)).join(' ')}`);
}
console.log(`v4 kernels: ${v4.kernels.map((k) => `tick ${k.tick} src ${k.src} ${k.mb.toFixed(2)} MB copy ${k.copyMs.toFixed(1)} ms`).join(' · ')}`);

const t4 = tail(v4.out);
if (v4.kernels.length === 0) throw new Error('v4: no kernel handed over');
if (!v4.kernels.some((k) => k.src === VOICE)) throw new Error('v4: the clicking source got no kernel');
for (let i = 1; i < v4.kernels.length; i++) {
  if (v4.kernels[i].tick - v4.kernels[i - 1].tick < 10) throw new Error('v4: kernels faster than one per 10 ticks');
}
if (!(t4[0] > t4[3] + 10 && t4[3] > t4[5])) throw new Error(`v4: the tail does not decay: ${t4}`);
if (!(t4[3] > -120)) throw new Error(`v4: no tail at 0.6-1 s: ${t4}`);
console.log('V4 PROBE PASSED');
