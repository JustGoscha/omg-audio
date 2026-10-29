#!/usr/bin/env node
// Smoke + performance test of the web demo's wasm (omg-audio): drives the
// sim and engine exports exactly as worker.js and worklet.js do — frames
// and kernels from the sim instance into the engine instance — and checks
// the audio that comes out. Prints the sim tick and the render cost.
//
//   node tools/web_smoke.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmBytes = readFileSync(join(root, 'demo/web/omg_web.wasm'));
const FS = 48000;

// two instances, like the worker and the worklet
const sim = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
const eng = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
const put = (f32) => {
  const p = eng.eng_stage(f32.length);
  new Float32Array(eng.memory.buffer, p, f32.length).set(f32);
};

eng.eng_init(FS);
// synthetic loops on every static source: a click train (music), 220 Hz
// bursts (the rest); the dynamic slots stay silent
for (let i = 0; i < 15; i++) {
  if (i >= 8 && i < 14) continue;
  const n = FS;
  const buf = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    buf[k] = i === 0
      ? (k % 24000 < 200 ? (1 - (k % 24000) / 200) * 0.8 : 0)
      : Math.sin((2 * Math.PI * 220 * k) / FS) * (k % 12000 < 6000 ? 0.4 : 0);
  }
  put(buf);
  eng.eng_source(i);
}
// a clap fx for a one-shot later
const clap = new Float32Array(480).map((_, k) => (Math.random() * 2 - 1) * Math.exp(-k / 60));
put(clap);
const clapKind = eng.eng_fx();

const t0 = performance.now();
sim.sim_setup(FS);
if (process.env.SMOKE_TIER) sim.sim_set_quality(Number(process.env.SMOKE_TIER));
console.log(`sim setup ${(performance.now() - t0).toFixed(0)} ms`);

const f32 = (mem, ptr, len) => new Float32Array(mem.buffer, ptr, len).slice();
let kernelsSent = 0;
const feed = () => {
  const n = sim.sim_clears_len();
  if (n) for (const s of new Uint32Array(sim.memory.buffer, sim.sim_clears_ptr(), n)) eng.eng_kernel_clear(s);
  for (let i = 0; i < sim.sim_kernel_count(); i++) {
    put(f32(sim.memory, sim.sim_kernel_ptr(i), sim.sim_kernel_len(i)));
    eng.eng_kernel(sim.sim_kernel_src(i));
    kernelsSent++;
  }
  put(f32(sim.memory, sim.sim_frame_ptr(), sim.sim_frame_len()));
  if (!eng.eng_frame()) throw new Error('frame rejected');
};

// 2 s in the living room by the piano, 2 s in the corridor (a clap), 2 s
// in the square
const spots = [[3.0, 3.0], [3.0, 3.0], [4.0, 9.0], [4.0, 9.0], [12, 29], [12, 29]];
let peak = 0, sumSq = 0, n = 0, nans = 0;
let tickMs = 0, ticks = 0, renderMs = 0, rendered = 0;
const secRms = [];
for (let sec = 0; sec < spots.length; sec++) {
  const [x, y] = spots[sec];
  let secSq = 0;
  for (let tick = 0; tick < 20; tick++) {
    const a = performance.now();
    sim.sim_tick(x, y, 1.7, 0.0);
    tickMs += performance.now() - a;
    ticks++;
    feed();
    if (sec === 2 && tick === 0) eng.eng_fx_play(13, clapKind, 1.0, 1.0);
    if (sec === 3) eng.eng_head((tick / 20) * Math.PI, 0.1, 0.0);
    for (let blk = 0; blk < FS / 20 / 128; blk++) {
      const b = performance.now();
      eng.eng_process(128);
      const dt = performance.now() - b;
      renderMs += dt;
      rendered += 128;
      eng.eng_report(dt, 128);
      const l = new Float32Array(eng.memory.buffer, eng.eng_out_l(), 128);
      const r = new Float32Array(eng.memory.buffer, eng.eng_out_r(), 128);
      for (let k = 0; k < 128; k++) {
        if (Number.isNaN(l[k]) || Number.isNaN(r[k])) nans++;
        peak = Math.max(peak, Math.abs(l[k]), Math.abs(r[k]));
        sumSq += l[k] * l[k] + r[k] * r[k];
        secSq += l[k] * l[k] + r[k] * r[k];
        n++;
      }
    }
  }
  secRms.push(Math.sqrt(secSq / (2 * FS)));
}
const state = f32(sim.memory, sim.sim_state_ptr(), 4);
console.log(`state: listener (${state[0].toFixed(1)}, ${state[1].toFixed(1)}) room ${state[2]} rt60 ${state[3].toFixed(2)} s`);
console.log(`audio: peak ${peak.toFixed(3)} rms ${Math.sqrt(sumSq / (2 * n)).toFixed(4)} nans ${nans}`);
console.log(`per-second rms: ${secRms.map((x) => x.toFixed(4)).join(' ')}`);
console.log(`perf: sim tick ${(tickMs / ticks).toFixed(1)} ms (budget 50) · render ${((renderMs / (rendered / FS)) / 10).toFixed(1)} % of real time (rung ${eng.eng_rung()}) · ${kernelsSent} kernels, ${sim.sim_dense_count()} convolved`);
if (nans > 0) throw new Error('NaNs in output');
if (peak < 0.01) throw new Error('output is silent');
if (peak > 0.999) throw new Error('output clipping hard');
console.log('SMOKE TEST PASSED');
