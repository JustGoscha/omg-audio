#!/usr/bin/env node
// Realtime-factor benchmark for the wasm engine, used to set the
// point-render budget (renderer.rs / eng_set_point_budget).
//
// Drives sim + engine exactly like worker.js + worklet.js, finds the
// heaviest listener position (most incoming taps), then measures
// eng_process throughput at a range of point budgets.
//
//   node tools/bench_web.mjs
//
// Interpretation: "rtf" = seconds of audio rendered per second of wall
// clock, single thread. Browser wasm on the same machine is comparable;
// budget defaults should keep rtf ≥ ~3 on the dev machine so a phone
// (≈3-5× slower) and Low Power Mode still clear 1× with margin.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmBytes = readFileSync(join(root, 'web/omg_web.wasm'));
const FS = 48000;

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

for (let i = 0; i < 6; i++) {
  const n = FS;
  const ptr = eng.eng_source_alloc(i, n);
  const buf = new Float32Array(eng.memory.buffer, ptr, n);
  for (let k = 0; k < n; k++) buf[k] = Math.sin(2 * Math.PI * (110 + i * 70) * k / FS) * 0.3;
}

sim.sim_setup();

const setParamsFromSim = () => {
  let taps = 0;
  for (let i = 0; i < 6; i++) {
    const len = sim.sim_params_len(i);
    taps += Math.max(0, (len - 24) / 9); // FLAT_HEADER 24, FLAT_PER_TAP 9
    const src = new Float32Array(sim.memory.buffer, sim.sim_params_ptr(i), len);
    new Float32Array(eng.memory.buffer, eng.eng_param_buf_ptr(), len).set(src);
    eng.eng_set_params(i, len);
  }
  return Math.round(taps);
};

// Find the heaviest spot among representative positions (room interiors,
// doorway blend zones, the square between the buildings). Positions must be
// walkable — inside wall thickness the room lookup rightly rejects them.
const candidates = [
  [4, 3],            // living room, near the music
  [4, 10],           // corridor
  [7, 19],           // great hall
  [21, 31],          // entrance
  [27, 32],          // club interior, 4-speaker rig
  [22.5, 31],        // club/entrance doorway blend (both rooms live)
  [21, 27.2],        // entrance/outside blend
  [27.5, 19.5],      // old house
  [18, 25],          // uni square: facades + corners + apertures
  [33, 39],          // outside behind the club, windows + diffraction
  [10, 8],           // outside west, multiple buildings in view
];
let worst = { taps: -1, pos: [0, 0] };
for (const [x, y] of candidates) {
  sim.sim_tick(x, y, 1.6, 0.0);
  const taps = setParamsFromSim();
  if (taps > worst.taps) worst = { taps, pos: [x, y] };
}
console.log(`worst-case position: (${worst.pos}) with ~${worst.taps} taps`);

const bench = (budget) => {
  eng.eng_set_point_budget(budget);
  // settle at the worst spot so tap slots are live in their final mode
  for (let t = 0; t < 10; t++) {
    sim.sim_tick(worst.pos[0], worst.pos[1], 0.0);
    setParamsFromSim();
    for (let b = 0; b < FS / 20 / 128; b++) eng.eng_process(128);
  }
  const SECONDS = 3;
  const blocks = Math.round(SECONDS * FS / 128);
  const t0 = performance.now();
  for (let b = 0; b < blocks; b++) eng.eng_process(128);
  const wall = (performance.now() - t0) / 1000;
  return SECONDS / wall;
};

console.log('budget  rtf (audio-sec / wall-sec, single thread)');
for (const budget of [0, 8, 16, 24, 32, 48, 64]) {
  const rtf = bench(budget);
  console.log(`${String(budget).padStart(6)}  ${rtf.toFixed(2)}x`);
}
