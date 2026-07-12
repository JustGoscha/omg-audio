#!/usr/bin/env node
// Environment-audio probe: renders REAL audio from the wasm build while
// (a) walking from open air through the Great Hall door and (b) swinging
// the Old House door shut, and asserts the ambience+rain level trajectory
// is smooth. This is the regression harness for the two user-facing
// properties the environment routing exists for:
//   - room transitions must never step the ambient/rain sound
//   - a door swing must sweep the filter with the moving leaf
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

// silent sources: only the environment speaks in this probe
for (let i = 0; i < 6; i++) eng.eng_source_alloc(i, 256);

// ambience loop: 1 s of stereo noise (any loop works — we measure levels)
{
  let seed = 1;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x40000000 - 1);
  const n = FS * 2;
  const ptr = eng.eng_ambient_alloc(n);
  const buf = new Float32Array(eng.memory.buffer, ptr, n);
  for (let k = 0; k < n; k++) buf[k] = rnd() * 0.3;
  eng.eng_ambient_commit(2);
}
eng.eng_set_rain(1.0);

sim.sim_setup();
const ENV_OFF = sim.sim_env_off();
const stateLen = sim.sim_state_len();
// NB: create views fresh at every use — wasm memory growth detaches them
const setDoor = (i, openness) => {
  new Float32Array(sim.memory.buffer, sim.sim_door_ptr(), 16)[i] = openness;
};

const forwardEnv = () => {
  const st = new Float32Array(sim.memory.buffer, sim.sim_state_ptr(), stateLen);
  const env = st.slice(ENV_OFF);
  new Float32Array(eng.memory.buffer, eng.eng_param_buf_ptr(), env.length).set(env);
  eng.eng_set_env(env.length);
};

// render `secs` at pose, returning rms of the last `measure` seconds
const renderAt = (x, y, secs, measure = secs) => {
  let sq = 0;
  let n = 0;
  const skip = Math.round((secs - measure) * 20);
  for (let tick = 0; tick < Math.round(secs * 20); tick++) {
    sim.sim_tick(x, y, 1.6, 0.0);
    forwardEnv();
    for (let blk = 0; blk < FS / 20 / 128; blk++) {
      eng.eng_process(128);
      const l = new Float32Array(eng.memory.buffer, eng.eng_out_l(), 128);
      const r = new Float32Array(eng.memory.buffer, eng.eng_out_r(), 128);
      if (tick >= skip) {
        for (let k = 0; k < 128; k++) {
          if (Number.isNaN(l[k]) || Number.isNaN(r[k])) throw new Error('NaN in output');
          sq += l[k] * l[k] + r[k] * r[k];
          n += 2;
        }
      }
    }
  }
  return Math.sqrt(sq / n);
};

// --- (a) walk open air → deep into the Great Hall via the door at (7, 24) ---
renderAt(7.0, 27.0, 10.0); // let rain ramp in and everything settle
const walk = [];
for (let y = 27.0; y >= 16.0 - 1e-6; y -= 0.5) {
  walk.push({ y, rms: renderAt(7.0, y, 1.0, 0.5) });
}
console.log('walk outside→hall:', walk.map((p) => p.rms.toFixed(4)).join(' '));
for (let i = 1; i < walk.length; i++) {
  const ratio = Math.max(walk[i].rms / walk[i - 1].rms, walk[i - 1].rms / walk[i].rms);
  if (ratio > 1.9) {
    throw new Error(
      `level step ${(20 * Math.log10(ratio)).toFixed(1)} dB at y=${walk[i].y} — room transition audible as a switch`);
  }
}
const outdoorRms = walk[0].rms;
const indoorRms = walk[walk.length - 1].rms;
if (!(indoorRms < 0.7 * outdoorRms)) {
  throw new Error(`indoors should be clearly quieter: ${indoorRms} vs ${outdoorRms}`);
}

// --- (b) swing the Old House door shut while standing just inside it ---
const AT = [26.5, 22.3]; // 0.7 m inside the doorway at (26.5, 23)
renderAt(AT[0], AT[1], 4.0);
const swing = [];
for (let step = 0; step <= 12; step++) {
  setDoor(5, 1.0 - step / 12); // leaf animates over ~1.2 s of simulated time
  swing.push(renderAt(AT[0], AT[1], 0.1));
}
// keep sampling equal windows while the smoothers settle — a snap would
// show as a jump between consecutive windows anywhere in this tail
for (let k = 0; k < 15; k++) swing.push(renderAt(AT[0], AT[1], 0.1));
console.log('door swing open→closed:', swing.map((x) => x.toFixed(4)).join(' '));
for (let i = 1; i < swing.length; i++) {
  const ratio = Math.max(swing[i] / swing[i - 1], swing[i - 1] / swing[i]);
  if (ratio > 1.6) {
    throw new Error(`door swing stepped ${(20 * Math.log10(ratio)).toFixed(1)} dB — filter snapped`);
  }
}
if (!(swing[swing.length - 1] < 0.75 * swing[0])) {
  throw new Error(`closing the door must matter: ${swing[swing.length - 1]} vs ${swing[0]}`);
}

console.log('ENV PROBE PASSED');
