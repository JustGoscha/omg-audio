#!/usr/bin/env node
// Smoke test for the wasm build: drives the sim + engine exports exactly the
// way worker.js + worklet.js do, and checks the audio that comes out.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmBytes = readFileSync(join(root, 'web/omg_web.wasm'));

const FS = 48000;

// Two instances, like the two browser contexts.
const sim = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
const eng = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;

// --- engine setup (worklet role) ---
eng.eng_init(FS);
const put = (bytes, allocFn, doneFn) => {
  const ptr = allocFn(bytes.byteLength);
  new Uint8Array(eng.memory.buffer, ptr, bytes.byteLength).set(bytes);
  doneFn();
};
put(readFileSync(join(root, 'assets/hrir_grid.bin')), eng.eng_hrir_grid_alloc, eng.eng_hrir_grid_done);
put(readFileSync(join(root, 'assets/hrir_dodeca20.bin')), eng.eng_hrir_speakers_alloc, eng.eng_hrir_speakers_done);

// Synthetic sources: click train (music slot) + 220 Hz tone bursts (voice).
for (let i = 0; i < 10; i++) {
  const n = FS; // 1 s loop
  const ptr = eng.eng_source_alloc(i, n);
  const buf = new Float32Array(eng.memory.buffer, ptr, n);
  for (let k = 0; k < n; k++) {
    buf[k] = i === 0
      ? (k % 24000 < 200 ? (1 - k % 24000 / 200) * 0.8 : 0)
      : Math.sin(2 * Math.PI * 220 * k / FS) * (k % 12000 < 6000 ? 0.4 : 0);
  }
}

// --- sim setup (worker role) ---
sim.sim_setup();

const setParamsFromSim = () => {
  for (let i = 0; i < 10; i++) {
    const len = sim.sim_params_len(i);
    const src = new Float32Array(sim.memory.buffer, sim.sim_params_ptr(i), len);
    new Float32Array(eng.memory.buffer, eng.eng_param_buf_ptr(), len).set(src);
    eng.eng_set_params(i, len);
  }
};

// Walk: 2 s in the living room near the music, then 2 s in the corridor.
const positions = [
  [3.0, 3.0], [3.0, 3.0],   // living room
  [4.0, 9.0], [4.0, 9.0],   // corridor
];
let peak = 0;
let sumSq = 0;
let n = 0;
let nans = 0;
const secResults = [];

for (let sec = 0; sec < positions.length; sec++) {
  const [px, py] = positions[sec];
  let secSq = 0;
  for (let tick = 0; tick < 20; tick++) {
    sim.sim_tick(px, py, 1.6, 0.0);
    setParamsFromSim();
    // head turns slowly during second 3 to exercise rotation + reselection
    if (sec === 3) eng.eng_set_head((tick / 20) * Math.PI, 0.15, 0.0);
    for (let blk = 0; blk < FS / 20 / 128; blk++) {
      eng.eng_process(128);
      const l = new Float32Array(eng.memory.buffer, eng.eng_out_l(), 128);
      const r = new Float32Array(eng.memory.buffer, eng.eng_out_r(), 128);
      for (let k = 0; k < 128; k++) {
        if (Number.isNaN(l[k]) || Number.isNaN(r[k])) nans++;
        const m = Math.max(Math.abs(l[k]), Math.abs(r[k]));
        peak = Math.max(peak, m);
        sumSq += l[k] * l[k] + r[k] * r[k];
        secSq += l[k] * l[k] + r[k] * r[k];
        n++;
      }
    }
  }
  secResults.push(Math.sqrt(secSq / (2 * FS)));
}

const rms = Math.sqrt(sumSq / (2 * n));
const state = new Float32Array(sim.memory.buffer, sim.sim_state_ptr(), 32);
console.log(`state: listener=(${state[0].toFixed(1)},${state[1].toFixed(1)}) room=${state[2]} rt60=${state[3].toFixed(2)}s`);
console.log(`audio: peak=${peak.toFixed(3)} rms=${rms.toFixed(4)} nans=${nans}`);
console.log(`per-second rms: ${secResults.map((x) => x.toFixed(4)).join(' ')}`);

if (nans > 0) throw new Error('NaNs in output');
if (peak < 0.01) throw new Error('output is silent');
if (peak > 0.999) throw new Error('output clipping hard');
console.log('SMOKE TEST PASSED');
