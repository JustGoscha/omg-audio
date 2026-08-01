// First-person walking simulator over the omg-audio engine.
// Desktop: pointer-lock mouse look + WASD. Mobile: left-thumb joystick to
// walk, device orientation (or right-half drag) to look.
//
// World coords (match the Rust scene): x/y ground plane, z up.
// three.js coords: X = wx, Y = wz (height), Z = -wy.

import * as THREE from './vendor/three.module.js';

// Scene geometry mirrored from crates/omg-scene/src/walkthrough.rs (visuals
// + movement only; the acoustic truth lives in the wasm sim).
const ROOMS = [
  { name: 'Living Room', min: [0, 0], max: [8, 6], h: 2.7, floor: 0x4a3b2e },
  { name: 'Corridor', min: [3.2, 6], max: [4.8, 14], h: 2.4, floor: 0x323844 },
  { name: 'Great Hall', min: [0, 14], max: [14, 24], h: 7, floor: 0x2a3140 },
  { name: 'Entrance', min: [20, 28], max: [22, 34], h: 2.6, floor: 0x3a2a3a },
  { name: 'Club', min: [22, 26], max: [32, 38], h: 4.5, floor: 0x2e1f33 },
  { name: 'Old House', min: [24, 16], max: [31, 23], h: 5.6, floor: 0x39322a }, // 2 storeys
  { name: 'Colonnade', min: [16, 15], max: [16.5, 21], h: 2.5, solid: true },
  { name: 'Kiosk', min: [14.5, 34], max: [16.5, 36], h: 2.7, solid: true },
  { name: 'Old House Upper', min: [24, 16], max: [31, 23], upper: true }, // index-aligned with the sim
  { name: 'Cathedral', min: [0, 52], max: [16, 74], h: 15, floor: 0x2b2b36 },
  { name: 'Bunker', min: [34, 6], max: [40, 14], h: 2.2, fz: -3, floor: 0x20241f },
  { name: 'Outside', min: [-28, -1900], max: [48, 2000], outdoor: true, floor: 0x1c2a20 },
];
// axis: 0 = opening in an x=const wall, 1 = opening in a y=const wall
const DOORS = [
  { pos: [4, 6], axis: 1 },
  { pos: [4, 14], axis: 1 },
  { pos: [7, 24], axis: 1 },
  { pos: [20, 31], axis: 0 },
  { pos: [22, 31], axis: 0 },
  { pos: [26.5, 23], axis: 1 }, // old house front door
  { pos: [8, 52], axis: 1, hw: 1.2, h: 3.5 }, // cathedral portal (grand)
  { pos: [31.4, 7], axis: 0, hw: 0.7, h: 2.0, steel: true }, // bunker blast door
];
// Furniture (PT-early C5): world-coord mirror of walkthrough.rs
// furniture() — solid boxes the traced early engine shadows sources
// behind. [min x,y,z, max x,y,z, color]
const FURNITURE = [
  // cathedral colonnades (+0,+52), 9 m stone pillars
  [3.75, 56.75, 0, 5.25, 58.25, 9, 0x3d3d49], [3.75, 62.25, 0, 5.25, 63.75, 9, 0x3d3d49],
  [3.75, 67.75, 0, 5.25, 69.25, 9, 0x3d3d49],
  [10.75, 56.75, 0, 12.25, 58.25, 9, 0x3d3d49], [10.75, 62.25, 0, 12.25, 63.75, 9, 0x3d3d49],
  [10.75, 67.75, 0, 12.25, 69.25, 9, 0x3d3d49],
  // club bar counter (+22,+26)
  [23, 29, 0, 24, 33.5, 1.1, 0x3a2a1e],
  // living room: sofa, coffee table, armchair, sideboard, bookshelf
  [2, 4.4, 0, 4.2, 5.3, 0.75, 0x584032],
  [2.5, 3.3, 0, 3.7, 3.9, 0.45, 0x4a3626],
  [4.9, 4.3, 0, 5.7, 5.1, 0.75, 0x584032],
  [0.3, 0.2, 0, 2.3, 0.7, 0.9, 0x4a3626],
  [5.4, 0.4, 0, 5.8, 3.4, 2.2, 0x5a4530],
  // great hall seminar (+0,+14): tables, chairs, lectern, whiteboard
  [1.8, 15.6, 0, 3.6, 16.3, 0.74, 0x4a3b2e], [1.8, 18.4, 0, 3.6, 19.1, 0.74, 0x4a3b2e],
  [1.8, 21.2, 0, 3.6, 21.9, 0.74, 0x4a3b2e], [4.6, 15.6, 0, 6.4, 16.3, 0.74, 0x4a3b2e],
  [4.6, 18.4, 0, 6.4, 19.1, 0.74, 0x4a3b2e], [4.6, 21.2, 0, 6.4, 21.9, 0.74, 0x4a3b2e],
  [1.2, 15.7, 0, 1.65, 16.15, 0.45, 0x35404f], [1.2, 18.5, 0, 1.65, 18.95, 0.45, 0x35404f],
  [1.2, 21.3, 0, 1.65, 21.75, 0.45, 0x35404f], [4.0, 15.7, 0, 4.45, 16.15, 0.45, 0x35404f],
  [4.0, 18.5, 0, 4.45, 18.95, 0.45, 0x35404f], [4.0, 21.3, 0, 4.45, 21.75, 0.45, 0x35404f],
  [10.2, 19.9, 0, 10.8, 20.3, 1.2, 0x5a4530],
  [12.0, 16.5, 0, 12.3, 19.5, 2.2, 0xd8d8d0],
  // old house ground (+24,+16): couch, table, TV, kitchen, fridge, pillows
  [25.0, 18.0, 0, 27.2, 18.9, 0.75, 0x584032],
  [25.6, 19.4, 0, 26.6, 20.0, 0.4, 0x4a3626],
  [24.3, 20.4, 0.4, 24.55, 21.9, 1.4, 0x14161c],
  [28.4, 16.3, 0, 30.7, 16.95, 0.95, 0x4a4038],
  [30.0, 17.3, 0, 30.7, 18.0, 1.9, 0xb8bcc0],
  [26.2, 20.8, 0, 26.8, 21.4, 0.25, 0x6a4a5a],
  [27.4, 21.2, 0, 28.0, 21.8, 0.25, 0x4a5a6a],
  // old house upper (+24,+16, floor at z=3): bed, big TV, dresser, pillows
  [24.8, 16.8, 3.0, 26.6, 18.8, 3.6, 0x707a8a],
  [24.25, 19.6, 3.8, 24.45, 21.6, 5.0, 0x14161c],
  [29.8, 16.4, 3.0, 30.6, 18.0, 4.0, 0x4a3626],
  [28.6, 20.6, 3.0, 29.2, 21.2, 3.25, 0x6a5a4a],
];
const DOOR_HALF = 0.55;
const DOOR_H = 2.1;
// windows: glass — visible, not walkable; direct sound passes lightly damped
const WINDOWS = [
  { pos: [3.0, 0.0], axis: 1, hw: 1.3 },
  { pos: [32.0, 32.0], axis: 0, hw: 1.8 },
  { pos: [26.0, 38.0], axis: 1, hw: 1.8 },
  { pos: [29.3, 23.0], axis: 1, hw: 1.4 }, // house ground floor → square
  { pos: [24.0, 19.5], axis: 0, hw: 1.4 }, // house ground floor → hall
];
const SOURCES = [
  { name: 'music', pos: [2, 3], color: 0xffaa3c },
  { name: 'voice', pos: [10.5, 20.5], color: 0x6ee0a0 },
  { name: 'club', pos: [27, 32], color: 0xff5a9e,
    emitters: [[23.5, 27.5], [30.5, 27.5], [23.5, 36.5], [30.5, 36.5]] },
  { name: 'flute', pos: [8, 66], color: 0x9ad2ff },
  { name: 'radio', pos: [37.5, 12.5], z: -2.2, color: 0xd2b06e },
];
const MARGIN = 0.35;
const EYE = 1.6;

const state = {
  projs: [],
  cars: [],
  fx: null,
  meters: { l: 0, r: 0, agc: 1, tts: 0, pts: 0, hist: [] },
  pose: { x: 3, y: 3 },
  heading: Math.PI / 2, // world math angle; π/2 = facing +y (into the scene)
  pitch: 0,
  keys: new Set(),
  joy: null,
  look: null,
  orientation: null,
  orientationOffset: null,
  simState: null,
  running: false,
  // camera face tracking: smoothed head offsets (rotation rad,
  // translation m in camera terms: dx left, dy up, dz forward) applied
  // on top of mouse/touch heading — audio only, the view stays put
  face: { yaw: 0, pitch: 0, roll: 0, dx: 0, dy: 0, dz: 0 },
  faceTarget: null,
  faceTrack: null,
  // field-debug panel: latest engine snapshots + decimated history
  debug: { on: false, hist: [], chans: null, amb: null, dbg: null, load: 0, tickMs: 0 },
  // black-box recorder: rolling fine-grained snapshots + collapse events
  bbox: { ring: [], collapses: [], cooldownUntil: 0 },
  rainLevel: 0, // index into RAIN_LEVELS
  chanHist: [], // per-channel meter frames, ~1 s
  mixerRows: null,
  doors: DOORS.map(() => true), // open/closed, index-matched to DOORS
  doorMeshes: [],
};

// simulated weather: intensity targets the engine ramps toward (~6 s),
// so rain starts and stops like weather, not like a fader
const RAIN_LEVELS = [
  { label: '☂ off', intensity: 0.0 },
  { label: '☂ drizzle', intensity: 0.3 },
  { label: '☂ rain', intensity: 0.65 },
  { label: '☂ downpour', intensity: 1.0 },
];

function cycleRain() {
  state.rainLevel = (state.rainLevel + 1) % RAIN_LEVELS.length;
  const lvl = RAIN_LEVELS[state.rainLevel];
  document.getElementById('rain').textContent = lvl.label;
  if (state.node) state.node.port.postMessage({ type: 'rain', intensity: lvl.intensity });
}

// ------------------------------------------------------------- mixer panel
//
// Source faders are POWER faders on a real SPL scale (dB SPL @ 1 m):
// 20 ≈ a needle drop, 60 ≈ speech, 90 ≈ fortissimo piano, 110 ≈ club PA,
// 130 ≈ jet engine. The engine's calibration anchor is gain 1.0 ≈ 90 dB
// SPL, and each source's authored gain gives its baseline; the fader sets
// the REAL emitted energy: gain = 10^((SPL − baseline) / 20).
// Ambience/rain/master are trim faders in plain dB.
// `base` is the calibration anchor (authored gain = that SPL); `def` is
// the default fader position (tuned by ear for the demo mix).
const MIXER = [
  { name: 'music', srcs: [0], base: 90, def: 72, meters: [0], spl: true },
  { name: 'voice', srcs: [1], base: 84, def: 65, meters: [1], spl: true },
  { name: 'club', srcs: [2], base: 104, def: 104, meters: [2], spl: true },
  { name: 'flute', srcs: [3], base: 88, def: 76, meters: [3], spl: true },
  { name: 'radio', srcs: [4], base: 80, def: 64, meters: [4], spl: true },
  { name: 'balls', srcs: [5, 6, 7], base: 89, def: 89, meters: [5, 6, 7], spl: true },
  { name: 'cars', srcs: [8, 9], base: 92, def: 86, meters: [8, 9], spl: true },
  { name: 'steps', srcs: [10], base: 78, def: 62, meters: [10], spl: true },
  { name: 'ambience', target: 'ambient', def: -16, meters: [11] },
  { name: 'rain', target: 'rainGain', def: 0, meters: [12] },
  { name: 'master', target: 'master', def: 0, meters: 'lr' },
];
const SPL_MIN = 20, SPL_MAX = 130, TRIM_MIN = -30, TRIM_MAX = 12;

window.__omg = state; // field-debug handle
function buildMixer() {
  const panel = document.getElementById('mixer');
  panel.innerHTML = '<div class="scale"><span>20</span><span>needle</span>' +
    '<span>60 speech</span><span>club 110</span><span>jet 130</span></div>';
  state.mixerRows = MIXER.map((ch) => {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('div');
    label.innerHTML = ch.name + '<br><span class="spl"></span>';
    const mid = document.createElement('div');
    const fader = document.createElement('input');
    fader.type = 'range';
    fader.min = 0; fader.max = 1000; fader.className = 'fader';
    fader.value = ch.spl
      ? ((ch.def - SPL_MIN) / (SPL_MAX - SPL_MIN)) * 1000
      : ((ch.def - TRIM_MIN) / (TRIM_MAX - TRIM_MIN)) * 1000;
    const meter = document.createElement('div');
    meter.className = 'meter';
    meter.innerHTML = '<div class="rms"></div><div class="pk"></div>';
    mid.append(fader, meter);
    const lvl = document.createElement('div');
    lvl.className = 'lvl';
    lvl.textContent = '−∞';
    row.append(label, mid, lvl);
    panel.append(row);
    const apply = () => {
      const v = fader.value / 1000;
      let gain, text;
      if (ch.spl) {
        const spl = SPL_MIN + v * (SPL_MAX - SPL_MIN);
        gain = 10 ** ((spl - ch.base) / 20);
        text = `${spl.toFixed(0)} dB SPL`;
      } else {
        const db = TRIM_MIN + v * (TRIM_MAX - TRIM_MIN);
        gain = v === 0 ? 0 : 10 ** (db / 20);
        text = `${db >= 0 ? '+' : ''}${db.toFixed(0)} dB`;
      }
      label.querySelector('.spl').textContent = text;
      if (state.node) {
        state.node.port.postMessage({ type: 'mixer', target: ch.target, srcs: ch.srcs, gain });
      }
    };
    fader.oninput = apply;
    fader.onpointerup = () => fader.blur();
    apply();
    return { ch, row, lvl, rms: meter.querySelector('.rms'), pk: meter.querySelector('.pk') };
  });
}

// ------------------------------------------------------------ quality panel
// One tier selector + every budget lever as its own slider. Sim levers pin
// through sim_set_override (0 = follow tier); touching an audio lever
// freezes the worklet governor and holds both audio values manually.
// Mirrors of the tier tables in crates/omg-scene/src/quality.rs.
const QUAL_TIER_VALS = {
  rays: [4096, 2048, 1024], gate: [8, 12, 16], domeRays: [384, 256, 160],
  domeEvents: [6, 5, 4], ism: [3, 3, 2],
};
const QUAL_SLIDERS = [
  { key: 'rays', id: 0, label: 'trace rays', min: 256, max: 32768, step: 256, section: 'simulation · 20 Hz',
    tip: 'Stochastic rays per reverb trace, per source. Fewer rays = a noisier RT60/level estimate that the running average smooths out — more flutter, never a different sound. The single biggest sim-CPU lever.' },
  { key: 'ism', id: 4, label: 'ism order', min: 2, max: 3, step: 1,
    tip: 'Image-source reflection order: how many wall bounces the early echoes are traced through. 3 ≈ 63 discrete reflections per source, 2 ≈ 15 — roughly halving the tap count the audio thread must render. 2 keeps the audible early pattern; the late reverb covers the rest.' },
  { key: 'gate', id: 1, label: 'refresh age', min: 4, max: 32, step: 1,
    tip: 'How many 20 Hz ticks a settled scene may keep its last reverb estimate before re-tracing. Motion, door swings and level changes always re-trace immediately — this only slows the idle refresh.' },
  { key: 'domeRays', id: 2, label: 'dome rays', min: 32, max: 384, step: 32,
    tip: 'Rays sampling the outdoor-ambience dome against the real geometry every tick (how rain and city noise find windows and doorways). Fewer rays = noisier direction bins under the dome’s own smoothing.' },
  { key: 'domeEvents', id: 3, label: 'dome events', min: 2, max: 6, step: 1,
    tip: 'Bounces and glass panes a dome ray may cross before giving up. Deep events carry little energy (every surface multiplies the loss), so trimming them mostly cuts cost, not sound.' },
  { key: 'points', label: 'point hrtf', min: 0, max: 32, step: 1, audio: true, section: 'audio render · per sample',
    tip: 'EVERY path is binaural: bus taps encode at their arrival direction into order-2 ambisonics decoded through 20 virtual speakers, each with its measured KEMAR HRIR — full ILD/ITD, slightly blurred (±20–30°). The strongest N paths additionally get a dedicated nearest-HRIR convolution (710-direction grid): pin-sharp. The precedence effect means the loudest early paths carry localization, so sharpness goes where hearing can use it; the weak tail reads as envelopment either way. Each point tap is a per-sample convolution on the AUDIO thread — the one place the GPU cannot help — hence the budget.' },
  { key: 'ceiling', label: 'tap ceiling', min: 16, max: 160, step: 8, audio: true,
    tip: 'Maximum early-reflection taps rendered per source. A doorway can carry ~145/source and the weakest half sit below the masking threshold of the strongest. Evicted taps fade out — lowering this mid-play is click-free.' },
];
const QUAL_TIER_TIPS = {
  auto: 'Starts at low and earns its way up: climbs a tier after ~10 s of clean running, sheds the moment a sim tick blows 40 ms or the audio thread gaps.',
  high: 'Full budgets — 4096 trace rays, reflection order 3, 384 dome rays. The reference sound.',
  med: 'Half the trace rays, slower idle refresh, leaner dome. Statistically the same field with slightly more reverb flutter.',
  low: 'Quarter rays, reflection order 2, lean dome. For throttled machines — everything stays spatialized, the estimate just gets coarser.',
};
const QUAL_STAT_TIPS = {
  'sim tick': 'Worker-side cost of one 20 Hz simulation tick. The budget is 50 ms; above ~40 the auto governor sheds a tier.',
  'render load': 'Audio-thread render time as a share of realtime. Above 85% the worklet governor sheds to its floor.',
  gaps: 'Browser-inserted silences: missed audio deadlines. Any nonzero count means audible dropouts happened.',
  'late field': 'Where the reverb ray traces run, with the wall-clock cost of the last trace dispatch (incl. readback). GPU runs 8× the ray budget for a fraction of the CPU cost. Click to A/B live — statistically identical sound.',
  early: 'The ACTIVE early-reflections engine as the sim reports it. traced shows where chain discovery runs and the last dispatch cost. ism and traced sound identical in open space; behind furniture (cathedral pillars, the club bar) only traced shadows.',
};

function buildQuality() {
  const panel = document.getElementById('quality');
  state.qual = { over: {}, manual: false, manualVals: {}, rows: [] };
  panel.innerHTML = '<h2>engine quality</h2>';

  const seg = document.createElement('div');
  seg.className = 'seg';
  const segBtns = ['auto', 'high', 'med', 'low'].map((mode) => {
    const b = document.createElement('button');
    b.textContent = mode;
    b.title = QUAL_TIER_TIPS[mode];
    b.onclick = () => { state.setQuality(mode); b.blur(); };
    seg.append(b);
    return { mode, b };
  });
  panel.append(seg);

  for (const s of QUAL_SLIDERS) {
    if (s.section) {
      const h = document.createElement('div');
      h.className = 'qsection';
      h.textContent = s.section;
      panel.append(h);
    }
    const row = document.createElement('div');
    row.className = 'qrow';
    const label = document.createElement('label');
    label.innerHTML = `<span>${s.label}</span>`;
    label.dataset.tip = s.tip;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = s.min; input.max = s.max; input.step = s.step;
    const val = document.createElement('div');
    val.className = 'qval';
    row.append(label, input, val);
    panel.append(row);
    input.oninput = () => {
      const v = +input.value;
      if (s.audio) {
        state.qual.manual = true;
        state.qual.manualVals[s.key] = v;
        state.setAudioManual(true,
          state.qual.manualVals.points ?? state.meters.pts,
          state.qual.manualVals.ceiling ?? (state.meters.ceil || 160));
      } else {
        state.qual.over[s.key] = v;
        state.setOverride(s.id, v);
      }
    };
    input.onpointerup = () => input.blur();
    state.qual.rows.push({ s, input, val });
  }

  // early-reflections backend (Track C): live A/B, same tap contract
  const eh = document.createElement('div');
  eh.className = 'qsection';
  eh.textContent = 'early reflections';
  panel.append(eh);
  const eseg = document.createElement('div');
  eseg.className = 'seg';
  eseg.style.gridTemplateColumns = 'repeat(2, 1fr)';
  const esegBtns = [['ism', 0], ['traced', 1]].map(([name, mode]) => {
    const b = document.createElement('button');
    b.textContent = name;
    b.title = mode === 0
      ? 'Image-source method: the exact closed-form solution for empty shoebox rooms. The classic engine.'
      : 'Path-traced (Track C): rays discover reflection chains, each solved exactly by mirror reconstruction. Identical here; built for rooms with things in them.';
    b.onclick = () => { state.setEarly(mode); b.blur(); };
    eseg.append(b);
    return { mode, b };
  });
  panel.append(eseg);

  const stats = document.createElement('div');
  stats.className = 'qstats';
  const cells = ['sim tick', 'render load', 'gaps', 'late field', 'early'].map((k) => {
    const c = document.createElement('div');
    c.className = 'cell';
    c.title = QUAL_STAT_TIPS[k];
    c.innerHTML = `<span class="k">${k}</span><span class="v">–</span>`;
    if (k === 'late field') {
      c.style.cursor = 'pointer';
      c.onclick = () => {
        if (state.debug.gpuAvail && state.setGpu) state.setGpu(!state.debug.gpu);
      };
    }
    stats.append(c);
    return c.querySelector('.v');
  });
  panel.append(stats);

  const reset = document.createElement('div');
  reset.className = 'qreset';
  reset.innerHTML = '<a href="#" title="Clear every pinned slider; the tier and the load governors take back control of all levers.">release all pins → governors</a>';
  reset.querySelector('a').onclick = (e) => {
    e.preventDefault();
    for (const s of QUAL_SLIDERS) if (!s.audio) state.setOverride(s.id, 0);
    state.qual.over = {};
    state.qual.manual = false;
    state.qual.manualVals = {};
    state.setAudioManual(false);
  };
  panel.append(reset);

  // live refresh: effective values track the tier/governor unless pinned
  setInterval(() => {
    if (panel.hidden) return;
    const q = state.quality || { auto: true, tier: 0, mode: 'auto' };
    for (const { mode, b } of segBtns) b.classList.toggle('on', q.mode === mode);
    for (const { mode, b } of esegBtns) b.classList.toggle('on', (state.earlyMode || 0) === mode);
    for (const { s, input, val } of state.qual.rows) {
      let v; let pinned;
      if (s.audio) {
        pinned = state.qual.manual;
        v = pinned ? (state.qual.manualVals[s.key]
              ?? (s.key === 'points' ? state.meters.pts : state.meters.ceil))
          : (s.key === 'points' ? state.meters.pts : state.meters.ceil || 160);
      } else {
        pinned = state.qual.over[s.key] != null;
        v = pinned ? state.qual.over[s.key] : QUAL_TIER_VALS[s.key][q.tier];
        // a GPU late-field backend runs the tier ray budget at 8x
        if (!pinned && s.key === 'rays' && state.debug.gpu) v = Math.min(v * 8, 32768);
      }
      if (document.activeElement !== input) input.value = v;
      val.textContent = `${v}${pinned ? ' ⏚' : ''}`;
      val.classList.toggle('pinned', !!pinned);
    }
    const tick = state.debug.tickMs || 0;
    const load = state.debug.load || 0;
    cells[0].textContent = `${tick.toFixed(1)}ms · ${(tick / 50 * 100).toFixed(0)}%`;
    cells[0].classList.toggle('hot', tick > 40);
    cells[1].textContent = `${(load * 100).toFixed(0)} %`;
    cells[1].classList.toggle('hot', load > 0.85);
    cells[2].textContent = `${state.debug.gaps || 0}`;
    cells[2].classList.toggle('hot', (state.debug.gaps || 0) > 0);
    cells[3].textContent = state.debug.gpu
      ? `gpu · ${(state.debug.gpuMs || 0).toFixed(1)}ms ⇄`
      : (state.debug.gpuAvail ? 'cpu ⇄' : 'cpu');
    cells[4].textContent = !state.debug.early ? 'ism'
      : state.debug.gpu && state.debug.ptN
        ? `traced · gpu ${(state.debug.ptMs || 0).toFixed(1)}ms` : 'traced · cpu';
    const btn = document.getElementById('qualbtn');
    btn.textContent = `⚙ ${state.qual.manual || Object.keys(state.qual.over).length
      ? 'custom' : ['high', 'med', 'low'][q.tier] + (q.auto ? ' ·auto' : '')}`;
  }, 350);
}

// meter history: ~43 frames ≈ 1 s
function updateMixerMeters() {
  if (!state.mixerRows || document.getElementById('mixer').hidden) return;
  const hist = state.chanHist;
  if (!hist.length) return;
  for (const r of state.mixerRows) {
    let rms = 0, pk = 0;
    if (r.ch.meters === 'lr') {
      // master: current output block peaks (post-AGC, post-master)
      rms = (state.meters.l + state.meters.r) / 2;
      pk = Math.max(state.meters.l, state.meters.r);
    } else {
      let n = 0;
      for (const frame of hist) {
        for (const m of r.ch.meters) {
          rms += frame[m * 2 + 1] ** 2;
          pk = Math.max(pk, frame[m * 2]);
          n++;
        }
      }
      rms = Math.sqrt(rms / Math.max(1, n));
    }
    const toDb = (x) => Math.max(-60, 20 * Math.log10(x + 1e-9));
    const rdb = toDb(rms);
    const pdb = toDb(pk);
    r.rms.style.width = `${((rdb + 60) / 60) * 100}%`;
    r.pk.style.left = `${((pdb + 60) / 60) * 100}%`;
    r.lvl.textContent = rdb <= -59.5 ? '−∞' : `${rdb.toFixed(1)} dB`;
  }
}

// ------------------------------------------------------------ walkability

// Collision: actual wall segments (built alongside the 3D walls; door
// gaps are simply absent). Room-union tests stopped working once Outside
// surrounds all buildings — everything was "inside a room".
const COLLIDERS = []; // { axis, plane, lo, hi }  axis 0: x=plane, 1: y=plane
const PLAYER_R = 0.22;
const WORLD = { min: [-27.6, -31.6], max: [47.6, 95.6] }; // playable region (the road runs on)

function crossesWall(x0, y0, x1, y1, z = 1.6) {
  for (const c of COLLIDERS) {
    if (c.off) continue; // opened door panel
    if (z > c.h || z < (c.zlo || 0)) continue; // above wall / below upper storey
    const [p0, p1] = c.axis === 0 ? [x0, x1] : [y0, y1];
    const d = p1 - p0;
    if (Math.abs(d) < 1e-9) {
      // moving parallel: keep a standoff from the wall plane
      if (Math.abs(p1 - c.plane) < PLAYER_R) {
        const along = c.axis === 0 ? y1 : x1;
        if (along > c.lo - PLAYER_R && along < c.hi + PLAYER_R) continue; // grazing ok
      }
      continue;
    }
    const t = (c.plane - p0) / d;
    if (t < 0 || t > 1) {
      // also block ending too close to the plane
      if (Math.abs(p1 - c.plane) > PLAYER_R) continue;
      const along = c.axis === 0 ? y1 : x1;
      if (along > c.lo - PLAYER_R && along < c.hi + PLAYER_R) return true;
      continue;
    }
    const along = c.axis === 0 ? y0 + t * (y1 - y0) : x0 + t * (x1 - x0);
    if (along > c.lo - PLAYER_R && along < c.hi + PLAYER_R) return true;
  }
  return false;
}

function walkableMove(x0, y0, x1, y1) {
  if (x1 < WORLD.min[0] || x1 > WORLD.max[0] || y1 < WORLD.min[1] || y1 > WORLD.max[1]) return false;
  const z = state.pose.z || 0;
  for (const f of FURNITURE) {
    if (x1 > f[0] - 0.25 && x1 < f[3] + 0.25 && y1 > f[1] - 0.25 && y1 < f[4] + 0.25
        && z < f[5] - 0.2 && z > f[2] - 1.2) return false;
  }
  return !crossesWall(x0, y0, x1, y1, z + 1.0);
}

// ------------------------------------------------------------ three scene

const glCanvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d12);
scene.fog = new THREE.Fog(0x0a0d12, 24, 100);
const camera = new THREE.PerspectiveCamera(72, 1, 0.05, 120);
camera.rotation.order = 'YXZ';

const v3 = (wx, wy, wz) => new THREE.Vector3(wx, wz, -wy);

function fit() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', fit);
fit();

scene.add(new THREE.HemisphereLight(0x8899bb, 0x1a1410, 0.9));
scene.add(new THREE.AmbientLight(0x404860, 0.5));

const wallMat = new THREE.MeshLambertMaterial({ color: 0x39465c });
const glassMat = new THREE.MeshLambertMaterial({
  color: 0x7ac8ff, transparent: true, opacity: 0.28, side: THREE.DoubleSide,
});
const edgeMat = new THREE.LineBasicMaterial({ color: 0x6a7f9f });
const ceilMat = new THREE.MeshLambertMaterial({ color: 0x232b38, side: THREE.DoubleSide });

function addBox(cx, cy, cz, sx, sy, sz) {
  const g = new THREE.BoxGeometry(sx, sz, sy); // (wx, wz, wy) sizes
  const m = new THREE.Mesh(g, wallMat);
  m.position.copy(v3(cx, cy, cz));
  scene.add(m);
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), edgeMat);
  e.position.copy(m.position);
  scene.add(e);
}

// Wall along y = const, spanning x0..x1, with door cuts (axis-1 doors)
// and window insets (glass panes; collision keeps the full span solid).
function wallY(y, x0, x1, h) {
  let spans = [[x0, x1]];
  for (const d of DOORS) {
    const [dx, dy] = d.pos;
    const hw = d.hw ?? DOOR_HALF;
    const dh = d.h ?? DOOR_H;
    if (d.axis !== 1 || Math.abs(dy - y) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      dx - hw > a && dx + hw < b ? [[a, dx - hw], [dx + hw, b]] : [[a, b]],
    );
    if (dx - hw > x0 && dx + hw < x1 && h > dh) {
      addBox(dx, y, (dh + h) / 2, 2 * hw, 0.15, h - dh); // lintel
    }
  }
  for (const { pos: [wx, wy], axis, hw } of WINDOWS) {
    if (axis !== 1 || Math.abs(wy - y) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      wx - hw > a && wx + hw < b ? [[a, wx - hw], [wx + hw, b]] : [[a, b]],
    );
    if (wx - hw > x0 && wx + hw < x1) {
      addBox(wx, y, 0.45, 2 * hw, 0.15, 0.9);              // sill
      addBox(wx, y, (2.2 + h) / 2, 2 * hw, 0.15, h - 2.2); // head
      const pane = new THREE.Mesh(new THREE.BoxGeometry(2 * hw, 1.3, 0.06), glassMat);
      pane.position.copy(v3(wx, y, 1.55));
      scene.add(pane);
    }
  }
  for (const [a, b] of spans) {
    addBox((a + b) / 2, y, h / 2, b - a, 0.15, h);
  }
  COLLIDERS.push({ axis: 1, plane: y, lo: x0, hi: x1, h });
  // reopen the doorway gaps in the collider
  for (const d of DOORS) {
    if (d.axis === 1 && Math.abs(d.pos[1] - y) < 0.01 && d.pos[0] > x0 && d.pos[0] < x1) {
      openColliderGap(1, y, d.pos[0], d.hw ?? DOOR_HALF, d.h ?? DOOR_H);
    }
  }
}

// Wall along x = const, spanning y0..y1, with door cuts (axis-0 doors)
// and window insets.
function wallX(x, y0, y1, h) {
  let spans = [[y0, y1]];
  for (const d of DOORS) {
    const [dx, dy] = d.pos;
    const hw = d.hw ?? DOOR_HALF;
    const dh = d.h ?? DOOR_H;
    if (d.axis !== 0 || Math.abs(dx - x) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      dy - hw > a && dy + hw < b ? [[a, dy - hw], [dy + hw, b]] : [[a, b]],
    );
    if (dy - hw > y0 && dy + hw < y1 && h > dh) {
      addBox(x, dy, (dh + h) / 2, 0.15, 2 * hw, h - dh); // lintel
    }
  }
  for (const { pos: [wx, wy], axis, hw } of WINDOWS) {
    if (axis !== 0 || Math.abs(wx - x) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      wy - hw > a && wy + hw < b ? [[a, wy - hw], [wy + hw, b]] : [[a, b]],
    );
    if (wy - hw > y0 && wy + hw < y1) {
      addBox(x, wy, 0.45, 0.15, 2 * hw, 0.9);
      addBox(x, wy, (2.2 + h) / 2, 0.15, 2 * hw, h - 2.2);
      const pane = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.3, 2 * hw), glassMat);
      pane.position.copy(v3(x, wy, 1.55));
      scene.add(pane);
    }
  }
  for (const [a, b] of spans) {
    addBox(x, (a + b) / 2, h / 2, 0.15, b - a, h);
  }
  COLLIDERS.push({ axis: 0, plane: x, lo: y0, hi: y1, h });
  for (const d of DOORS) {
    if (d.axis === 0 && Math.abs(d.pos[0] - x) < 0.01 && d.pos[1] > y0 && d.pos[1] < y1) {
      openColliderGap(0, x, d.pos[1], d.hw ?? DOOR_HALF, d.h ?? DOOR_H);
    }
  }
}

// split a collider around a doorway so it stays walkable
// Romanesque interior: arcades with rounded arches, a triforium tier,
// rib-vaulted ceiling, an apse with altar, checkered stone floor.
// Visual only — acoustically the arcades are the cathedral material's
// high scattering fraction (the tracer diffuses reflections off it).
function buildCathedralInterior() {
  const stone = new THREE.MeshLambertMaterial({ color: 0x4c4c58 });
  const stoneD = new THREE.MeshLambertMaterial({ color: 0x3a3a44 });
  const rows = [4.0, 12.0];
  const ys = [];
  for (let y = 55.0; y <= 71.01; y += 3.2) ys.push(y);

  // checkered stone floor (replaces the flat color inside the nave)
  const fc = document.createElement('canvas');
  fc.width = fc.height = 256;
  const fg = fc.getContext('2d');
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      fg.fillStyle = (i + j) % 2 ? '#3c3c46' : '#55555f';
      fg.fillRect(i * 32, j * 32, 32, 32);
    }
  }
  const fTex = new THREE.CanvasTexture(fc);
  fTex.wrapS = fTex.wrapT = THREE.RepeatWrapping;
  fTex.repeat.set(4, 5.5);
  const cathFloor = new THREE.Mesh(
    new THREE.PlaneGeometry(16, 22),
    new THREE.MeshLambertMaterial({ map: fTex }),
  );
  cathFloor.rotation.x = -Math.PI / 2;
  cathFloor.position.copy(v3(8, 63, 0.03));
  scene.add(cathFloor);

  const colGeo = new THREE.CylinderGeometry(0.45, 0.52, 8.0, 10);
  const capGeo = new THREE.BoxGeometry(1.3, 0.5, 1.3);
  for (const x of rows) {
    for (const y of ys) {
      const c = new THREE.Mesh(colGeo, stone);
      c.position.copy(v3(x, y, 4.0));
      scene.add(c);
      const cap = new THREE.Mesh(capGeo, stoneD);
      cap.position.copy(v3(x, y, 8.25));
      scene.add(cap);
      const base = new THREE.Mesh(capGeo, stoneD);
      base.position.copy(v3(x, y, 0.25));
      scene.add(base);
      COLLIDERS.push({ axis: 0, plane: x, lo: y - 0.45, hi: y + 0.45, h: 8 });
      COLLIDERS.push({ axis: 1, plane: y, lo: x - 0.45, hi: x + 0.45, h: 8 });
    }
    // rounded arcade arches between neighboring columns
    const archGeo = new THREE.TorusGeometry(1.6, 0.22, 8, 14, Math.PI);
    for (let i = 0; i + 1 < ys.length; i++) {
      const a = new THREE.Mesh(archGeo, stone);
      a.position.copy(v3(x, (ys[i] + ys[i + 1]) / 2, 8.5));
      a.rotation.y = Math.PI / 2;
      scene.add(a);
    }
    // triforium: a second, smaller arch tier above the arcade
    const triGeo = new THREE.TorusGeometry(0.8, 0.12, 6, 10, Math.PI);
    for (let i = 0; i + 1 < ys.length; i++) {
      const mid = (ys[i] + ys[i + 1]) / 2;
      for (const off of [-0.85, 0.85]) {
        const a = new THREE.Mesh(triGeo, stoneD);
        a.position.copy(v3(x, mid + off, 10.6));
        a.rotation.y = Math.PI / 2;
        scene.add(a);
      }
      const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.18, 3.2), stoneD);
      shelf.position.copy(v3(x, mid, 9.9));
      scene.add(shelf);
    }
  }
  // transverse arches + vault ribs across the nave at every bay line
  const bigArch = new THREE.TorusGeometry(4.0, 0.3, 8, 18, Math.PI);
  const rib = new THREE.TorusGeometry(4.6, 0.16, 6, 18, Math.PI);
  for (const y of ys) {
    const a = new THREE.Mesh(bigArch, stoneD);
    a.position.copy(v3(8, y, 8.5));
    scene.add(a);
    const r = new THREE.Mesh(rib, stone);
    r.position.copy(v3(8, y, 10.2));
    scene.add(r);
  }
  // longitudinal ridge line along the vault crest
  const ridge = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.25, 19), stoneD);
  ridge.position.copy(v3(8, 63.5, 14.8));
  scene.add(ridge);

  // apse: a faceted alcove at the north end with the altar
  for (let k = 0; k < 5; k++) {
    const ang = Math.PI * (0.15 + (0.7 * k) / 4);
    const px = 8 + 4.2 * Math.cos(ang);
    const py = 74 - 3.4 * Math.abs(Math.sin(ang));
    const panel = new THREE.Mesh(new THREE.BoxGeometry(2.4, 12, 0.3), stone);
    panel.position.copy(v3(px, py, 6));
    panel.rotation.y = ang + Math.PI / 2;
    scene.add(panel);
  }
  const altar = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.1, 1.2), stoneD);
  altar.position.copy(v3(8, 70.5, 0.55));
  scene.add(altar);
  const cloth = new THREE.Mesh(
    new THREE.BoxGeometry(2.5, 0.06, 1.3),
    new THREE.MeshLambertMaterial({ color: 0x7a1f2b }),
  );
  cloth.position.copy(v3(8, 70.5, 1.13));
  scene.add(cloth);
  COLLIDERS.push({ axis: 1, plane: 70.5, lo: 6.8, hi: 9.2, h: 1.2 });
  // portal surround
  const pArch = new THREE.Mesh(new THREE.TorusGeometry(1.35, 0.25, 8, 14, Math.PI), stoneD);
  pArch.position.copy(v3(8, 52, 3.5));
  scene.add(pArch);
  // candle-warm apse glow, crossing light, cool vault wash
  const apseGlow = new THREE.PointLight(0xffc37a, 50, 22);
  apseGlow.position.copy(v3(8, 70, 3));
  scene.add(apseGlow);
  const warm = new THREE.PointLight(0xffd9a0, 35, 28);
  warm.position.copy(v3(8, 60, 5));
  scene.add(warm);
  const cool = new THREE.PointLight(0x8faddf, 30, 45);
  cool.position.copy(v3(8, 64, 12));
  scene.add(cool);
}

// Passing cars: dynamic sources on the west street. Spawn beyond
// audibility, drive through, despawn — the pass-by fade and the Doppler
// both come from the engine (distance loss + tap-delay glide).
const CAR_LANES = [-18.9, -17.1]; // southbound, northbound
function spawnCar() {
  const free = [3, 4].find((slot) => !state.cars.some((c) => c.slot === slot));
  if (free === undefined) return;
  const north = free === 4;
  const body = new THREE.Group();
  const shell = new THREE.Mesh(
    new THREE.BoxGeometry(1.7, 0.55, 4.0),
    new THREE.MeshLambertMaterial({ color: north ? 0x4a5a72 : 0x6a4a3a }),
  );
  shell.position.y = 0.55;
  body.add(shell);
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.5, 2.0),
    new THREE.MeshLambertMaterial({ color: 0x22262c }),
  );
  cabin.position.set(0, 1.05, -0.3);
  body.add(cabin);
  const head = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.14, 0.06),
    new THREE.MeshBasicMaterial({ color: 0xffe9b0 }),
  );
  head.position.set(0, 0.6, north ? -2.0 : 2.0);
  body.add(head);
  const tail = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.1, 0.06),
    new THREE.MeshBasicMaterial({ color: 0xd23a2e }),
  );
  tail.position.set(0, 0.62, north ? 2.0 : -2.0);
  body.add(tail);
  scene.add(body);
  if (state.motors) {
    // every car its own vehicle: a random motor loop, resampled for a
    // per-spawn engine pitch, swapped into the slot's source
    const m = state.motors[(Math.random() * state.motors.length) | 0];
    const rate = 0.85 + Math.random() * 0.4;
    const n = Math.floor(m.length / rate);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * rate;
      const j = p | 0;
      out[i] = m[j % m.length] * (1 - (p - j)) + m[(j + 1) % m.length] * (p - j);
    }
    state.node.port.postMessage({ type: 'motor', src: free + 5, buf: out.buffer }, [out.buffer]);
  }
  state.cars.push({
    slot: free,
    mesh: body,
    x: CAR_LANES[north ? 1 : 0],
    // spawn KILOMETERS out — genuinely inaudible, the approach is a
    // true fade-in from nothing (distance loss + air absorption)
    y: (north ? -1 : 1) * (900 + Math.random() * 900),
    north,
    vy: (north ? 1 : -1) * (12 + Math.random() * 12),
    vol: 0.6 + Math.random(), // some cars are just louder
  });
}

function updateCars(dt, now) {
  if (now > (state.nextCarAt || 0)) {
    spawnCar();
    if (Math.random() < 0.35) spawnCar(); // sometimes two come at once
    state.nextCarAt = now + 5000 + Math.random() * 10000;
  }
  for (const c of state.cars) {
    c.y += c.vy * dt;
    c.mesh.position.copy(v3(c.x, c.y, 0));
  }
  state.cars = state.cars.filter((c) => {
    // free the slot once it is inaudible again on the far side — no
    // need to drive the remaining kilometers in silence
    const gone = c.north ? c.y > 560 : c.y < -500;
    if (gone) scene.remove(c.mesh);
    return !gone;
  });
}

// Openable door panels: a wood slab in each doorway with its own
// toggleable collider. E toggles the nearest door; the sim hears the
// closed panel as mass-law transmission instead of a free aperture.
const doorMat = new THREE.MeshLambertMaterial({ color: 0x6b4a2e });
const steelMat = new THREE.MeshLambertMaterial({ color: 0x59616b });
function buildDoorPanels() {
  DOORS.forEach((d) => {
    const [dx, dy] = d.pos;
    const hw = d.hw ?? DOOR_HALF;
    const dh = d.h ?? DOOR_H;
    // hinge pivot at one jamb; the leaf hangs off it and swings ~112°
    const hinge = new THREE.Group();
    const leaf = new THREE.Mesh(
      d.axis === 0
        ? new THREE.BoxGeometry(d.steel ? 0.16 : 0.08, dh, 2 * hw)
        : new THREE.BoxGeometry(2 * hw, dh, d.steel ? 0.16 : 0.08),
      d.steel ? steelMat : doorMat,
    );
    if (d.axis === 0) {
      hinge.position.copy(v3(dx, dy - hw, dh / 2));
      leaf.position.set(0, 0, -hw); // three -z = world +y
    } else {
      hinge.position.copy(v3(dx - hw, dy, dh / 2));
      leaf.position.set(hw, 0, 0);
    }
    hinge.add(leaf);
    scene.add(hinge);
    const collider = {
      axis: d.axis, plane: d.axis === 0 ? dx : dy,
      lo: (d.axis === 0 ? dy : dx) - hw,
      hi: (d.axis === 0 ? dy : dx) + hw,
      h: dh, off: true,
    };
    COLLIDERS.push(collider);
    // doors start open: leaf swung back against the wall
    state.doorMeshes.push({ hinge, collider, pos: d.pos, openness: 1, target: 1 });
  });
}

const DOOR_SWING = 1.95; // rad, ~112°
function updateDoors(dt) {
  for (const dm of state.doorMeshes) {
    dm.openness += (dm.target - dm.openness) * Math.min(1, dt * 7);
    dm.hinge.rotation.y = dm.openness * DOOR_SWING;
    // passable once it's swung well out of the frame
    dm.collider.off = dm.openness > 0.45;
  }
}

function toggleNearestDoor() {
  let best = -1;
  let bestD = 2.2; // reach
  state.doorMeshes.forEach((dm, i) => {
    const d = Math.hypot(dm.pos[0] - state.pose.x, dm.pos[1] - state.pose.y);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best < 0) return;
  state.doors[best] = !state.doors[best];
  state.doorMeshes[best].target = state.doors[best] ? 1 : 0;
}

function openColliderGap(axis, plane, at, hw = DOOR_HALF, dh = DOOR_H) {
  for (let i = COLLIDERS.length - 1; i >= 0; i--) {
    const c = COLLIDERS[i];
    if (c.axis !== axis || Math.abs(c.plane - plane) > 0.01) continue;
    if (at - hw > c.lo && at + hw < c.hi) {
      COLLIDERS.splice(i, 1,
        { ...c, hi: at - hw },
        { ...c, lo: at + hw },
        // the wall above the doorway stays solid
        { ...c, lo: at - hw, hi: at + hw, zlo: dh });
    }
  }
}

function buildWorld() {
  // floors + ceilings
  for (const r of ROOMS) {
    if (r.solid || r.upper) continue;
    const w = r.max[0] - r.min[0];
    const d = r.max[1] - r.min[1];
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(w, d),
      new THREE.MeshLambertMaterial({ color: r.floor }),
    );
    floor.rotation.x = -Math.PI / 2;
    // outdoor ground sits below room floors (they overlap: z-fighting)
    floor.position.copy(v3(r.min[0] + w / 2, r.min[1] + d / 2,
      r.fz !== undefined ? r.fz + 0.02 : (r.outdoor ? -0.03 : 0.02)));
    scene.add(floor);
    if (!r.outdoor) {
      const ceil = new THREE.Mesh(new THREE.PlaneGeometry(w, d), ceilMat);
      ceil.rotation.x = Math.PI / 2;
      ceil.position.copy(v3(r.min[0] + w / 2, r.min[1] + d / 2, (r.fz || 0) + r.h));
      scene.add(ceil);
    }
  }
  const grid = new THREE.GridHelper(100, 100, 0x24402e, 0x16281d);
  grid.position.copy(v3(20, 36, 0.0));
  scene.add(grid);

  // hand-authored wall list (avoids coplanar duplicates between rooms)
  wallY(0, 0, 8, 2.7);            // living south
  wallX(0, 0, 6, 2.7);            // living west
  wallX(8, 0, 6, 2.7);            // living east
  wallY(6, 0, 8, 2.7);            // living north (door 0)
  wallX(3.2, 6, 14, 2.4);         // corridor west
  wallX(4.8, 6, 14, 2.4);         // corridor east
  wallY(14, 0, 14, 7);            // hall south (door 1)
  wallX(0, 14, 24, 7);            // hall west
  wallX(14, 14, 24, 7);           // hall east
  wallY(24, 0, 14, 7);            // hall north (door 2, to outside)

  // club building (entered from outside via the entrance lobby)
  wallX(20, 28, 34, 2.6);         // entrance west (door 3)
  wallY(28, 20, 22, 2.6);         // entrance south
  wallY(34, 20, 22, 2.6);         // entrance north
  wallX(22, 26, 38, 4.5);         // club west (door 4, shared with entrance)
  wallX(32, 26, 38, 4.5);         // club east
  wallY(26, 22, 32, 4.5);         // club south
  wallY(38, 22, 32, 4.5);         // club north

  // cathedral, far north across the field (portal = door 6)
  wallY(52, 0, 16, 15);
  wallY(74, 0, 16, 15);
  wallX(0, 52, 74, 15);           // stained-glass west flank
  wallX(16, 52, 74, 15);          // stained-glass east flank
  buildCathedralInterior();

  // bunker: an underground box; only the stair ramp touches the surface.
  // Collider tops sit at -0.8 (the buried ceiling): surface walkers pass
  // clean over them, anyone at -3 is properly boxed in.
  addBox(37, 6, -1.9, 6, 0.2, 2.2);
  addBox(37, 14, -1.9, 6, 0.2, 2.2);
  addBox(40, 10, -1.9, 0.2, 8, 2.2);
  addBox(34, 11, -1.9, 0.2, 6, 2.2); // west wall, hatch gap at y≈7
  COLLIDERS.push({ axis: 1, plane: 6, lo: 34, hi: 40, h: -0.8 });
  COLLIDERS.push({ axis: 1, plane: 14, lo: 34, hi: 40, h: -0.8 });
  COLLIDERS.push({ axis: 0, plane: 40, lo: 6, hi: 14, h: -0.8 });
  COLLIDERS.push({ axis: 0, plane: 34, lo: 6, hi: 14, h: -0.8 });
  openColliderGap(0, 34, 7);
  // entrance blockhouse over the stair head: heavy door on its west face
  wallY(6, 31.2, 34, 2.4);
  wallY(8, 31.2, 34, 2.4);
  wallX(31.4, 6, 8, 2.4);          // blast door (door 7) lives here
  addBox(32.7, 7, 2.45, 3.0, 2.2, 0.18); // roof slab
  const pitMat = new THREE.MeshBasicMaterial({ color: 0x05070a });
  const pit = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.9), pitMat);
  pit.rotation.x = -Math.PI / 2;
  pit.position.copy(v3(33.25, 7, 0.015));
  scene.add(pit);
  // furniture: the traced early engine's occluders, made visible
  for (const f of FURNITURE) {
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(f[3] - f[0], f[5] - f[2], f[4] - f[1]),
      new THREE.MeshLambertMaterial({ color: f[6] }),
    );
    box.position.copy(v3((f[0] + f[3]) / 2, (f[1] + f[4]) / 2, (f[2] + f[5]) / 2));
    scene.add(box);
  }
  // stairs down the shaft (walkable height stays the smooth ramp)
  const stepMat = new THREE.MeshLambertMaterial({ color: 0x40453f });
  for (let i = 0; i < 10; i++) {
    const step = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.32, 1.5), stepMat);
    step.position.copy(v3(32.55 + i * 0.15, 7, -0.15 - i * 0.3));
    scene.add(step);
  }
  // sign: BUNKER, at the door
  const signC = document.createElement('canvas');
  signC.width = 256; signC.height = 96;
  const sg = signC.getContext('2d');
  sg.fillStyle = '#2a2e26'; sg.fillRect(0, 0, 256, 96);
  sg.strokeStyle = '#c8b060'; sg.lineWidth = 6; sg.strokeRect(4, 4, 248, 88);
  sg.fillStyle = '#c8b060'; sg.font = 'bold 52px sans-serif';
  sg.textAlign = 'center'; sg.textBaseline = 'middle';
  sg.fillText('BUNKER', 128, 50);
  const sign = new THREE.Mesh(
    new THREE.PlaneGeometry(1.3, 0.5),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(signC) }),
  );
  sign.position.copy(v3(31.32, 5.4, 1.8));
  sign.rotation.y = -Math.PI / 2; // faces west, toward the square
  scene.add(sign);
  const post = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.8, 0.08), stepMat);
  post.position.copy(v3(31.35, 5.4, 0.9));
  scene.add(post);

  // the street: an asphalt strip along the west side; cars pass on it
  const asphalt = new THREE.Mesh(
    new THREE.PlaneGeometry(4.4, 3900),
    new THREE.MeshLambertMaterial({ color: 0x23252a }),
  );
  asphalt.rotation.x = -Math.PI / 2;
  asphalt.position.copy(v3(-18, 50, 0.01));
  scene.add(asphalt);
  const dashMat = new THREE.MeshBasicMaterial({ color: 0x8a8f7a });
  for (let y = -300; y < 360; y += 6) {
    const dash = new THREE.Mesh(new THREE.PlaneGeometry(0.18, 2.2), dashMat);
    dash.rotation.x = -Math.PI / 2;
    dash.position.copy(v3(-18, y, 0.02));
    scene.add(dash);
  }

  // old house: ground floor enterable, second storey visual mass
  wallY(16, 24, 31, 2.8);          // south
  wallX(24, 16, 23, 2.8);          // west (window)
  wallX(31, 16, 23, 2.8);          // east
  wallY(23, 24, 31, 2.8);          // north (door + window, faces the square)
  // upper storey: solid shell + decorative panes, blocks flying balls only
  const upper = (cx, cy, sx, sy) => {
    addBox(cx, cy, 4.2, sx, sy, 2.8);
    COLLIDERS.push({
      axis: sx < sy ? 0 : 1,
      plane: sx < sy ? cx : cy,
      lo: (sx < sy ? cy : cx) - Math.max(sx, sy) / 2,
      hi: (sx < sy ? cy : cx) + Math.max(sx, sy) / 2,
      h: 5.6, zlo: 2.8,
    });
  };
  upper(27.5, 16, 7, 0.15);
  upper(24, 19.5, 0.15, 7);
  upper(31, 19.5, 0.15, 7);
  upper(27.5, 23, 7, 0.15);
  {
    // upper slab (top at z = 3.0) with a stairwell opening along the
    // west wall; stairs rise south → north to the landing
    addBox(28.35, 19.5, 2.9, 5.3, 7, 0.2);      // main slab east of the stairwell
    addBox(24.85, 16.45, 2.9, 1.7, 0.9, 0.2);   // south sliver
    const steps = 13;
    for (let i = 0; i < steps; i++) {
      const z = ((i + 1) / steps) * 3.0;
      addBox(24.9, 17.2 + (i / (steps - 1)) * 4.0, z - 0.1, 1.2, 4.0 / steps + 0.05, 0.2);
    }
    addBox(24.9, 22.0, 2.9, 1.2, 1.2, 0.2);      // landing
    const roof = new THREE.Mesh(new THREE.PlaneGeometry(7, 7), ceilMat);
    roof.rotation.x = Math.PI / 2;
    roof.position.copy(v3(27.5, 19.5, 5.6));
    scene.add(roof);
    for (const [wx, wy, ax] of [[26, 23, 1], [29.3, 23, 1], [24, 19.5, 0]]) {
      const pane = new THREE.Mesh(
        ax ? new THREE.BoxGeometry(2.2, 1.2, 0.06) : new THREE.BoxGeometry(0.06, 1.2, 2.2),
        glassMat,
      );
      pane.position.copy(v3(wx, wy, 4.1)); // upper-floor windows (visual)
      scene.add(pane);
    }
  }
  // solid street furniture (reflectors/occluders)
  for (const r of ROOMS.filter((r) => r.solid)) {
    const w = r.max[0] - r.min[0];
    const d = r.max[1] - r.min[1];
    addBox(r.min[0] + w / 2, r.min[1] + d / 2, r.h / 2, w, d, r.h);
    COLLIDERS.push({ axis: 0, plane: r.min[0], lo: r.min[1], hi: r.max[1], h: r.h });
    COLLIDERS.push({ axis: 0, plane: r.max[0], lo: r.min[1], hi: r.max[1], h: r.h });
    COLLIDERS.push({ axis: 1, plane: r.min[1], lo: r.min[0], hi: r.max[0], h: r.h });
    COLLIDERS.push({ axis: 1, plane: r.max[1], lo: r.min[0], hi: r.max[0], h: r.h });
  }

  // sources: pulsing emissive markers + labels + local light
  for (const src of SOURCES) {
    src.meshes = [];
    for (const [ex, ey] of src.emitters ?? [src.pos]) {
      const isRig = !!src.emitters;
      const mesh = new THREE.Mesh(
        isRig
          ? new THREE.BoxGeometry(0.7, 1.1, 0.55) // PA speaker cabinet
          : new THREE.SphereGeometry(0.16, 24, 16),
        new THREE.MeshBasicMaterial({ color: src.color }),
      );
      mesh.position.copy(v3(ex, ey, src.z ?? (isRig ? 1.9 : 1.5)));
      scene.add(mesh);
      src.meshes.push(mesh);
    }
    const light = new THREE.PointLight(src.color, 6, 8);
    light.position.copy(v3(src.pos[0], src.pos[1], (src.z ?? 1.5) + 0.3));
    scene.add(light);

    const c = document.createElement('canvas');
    c.width = 256;
    c.height = 64;
    const cc = c.getContext('2d');
    cc.font = '40px sans-serif';
    cc.textAlign = 'center';
    cc.fillStyle = '#' + src.color.toString(16).padStart(6, '0');
    cc.fillText(src.name, 128, 44);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true }),
    );
    sprite.scale.set(1.6, 0.4, 1);
    sprite.position.copy(v3(src.pos[0], src.pos[1], 2.15));
    scene.add(sprite);
  }
}
buildWorld();
buildDoorPanels();

// ------------------------------------------------------------ start

const statusEl = document.getElementById('status');
const hintEl = document.getElementById('controls-hint');

document.getElementById('start').onclick = async (ev) => {
  const err = document.getElementById('err');
  // The load takes seconds; an impatient second tap must NOT start a
  // second audio graph (it played everything twice, one copy without
  // its mixer defaults — the "super loud on load" bug).
  if (state.starting) return;
  state.starting = true;
  ev.target.disabled = true;
  ev.target.textContent = 'loading…';
  try {
    await startAudio();
    document.getElementById('overlay').remove();
    document.getElementById('recenter').hidden = false;
    document.getElementById('rain').hidden = false;
    document.getElementById('mixerbtn').hidden = false;
    document.getElementById('face').hidden = false;
    document.getElementById('face').onclick = toggleFaceTracking;
    document.getElementById('dbgbtn').hidden = false;
    document.getElementById('dbgbtn').onclick = (e) => {
      e.target.blur();
      state.debug.on = !state.debug.on;
      document.getElementById('debug').hidden = !state.debug.on;
    };
    buildMixer();
    document.getElementById('mixerbtn').onclick = (e) => {
      e.target.blur();
      const p = document.getElementById('mixer');
      p.hidden = !p.hidden;
      if (!p.hidden) document.getElementById('quality').hidden = true;
      e.target.classList.toggle('on', !p.hidden);
      document.getElementById('qualbtn').classList.remove('on');
    };
    buildQuality();
    document.getElementById('qualbtn').hidden = false;
    document.getElementById('qualbtn').onclick = (e) => {
      e.currentTarget.blur();
      const p = document.getElementById('quality');
      p.hidden = !p.hidden;
      if (!p.hidden) document.getElementById('mixer').hidden = true;
      e.currentTarget.classList.toggle('on', !p.hidden);
      document.getElementById('mixerbtn').classList.remove('on');
    };
    setupControls();
    state.running = true;
  } catch (e) {
    err.textContent = String(e);
    console.error(e);
    state.starting = false;
    ev.target.disabled = false;
    ev.target.textContent = 'start';
  }
};

// Render from the first frame: idle aerial orbit until the user starts.
requestAnimationFrame(frame);

async function startAudio() {
  // Native device rate: forcing 48 kHz makes some Android audio paths
  // run silent. The engine is rate-parametric (assets resample on
  // decode; the worklet passes its real sampleRate to eng_init).
  const audio = new AudioContext({ latencyHint: 'interactive' });
  state.audio = audio; // debug panel reads live output latency from it
  // "Sound stopped and never came back": under sustained overload (or a
  // device switch / long tab freeze) Chrome can move the context to
  // 'suspended'/'interrupted' — and nothing resumes it by itself. Log
  // every transition, retry on a timer, and retry on the next user
  // gesture (resume() is gesture-gated in some states).
  audio.onstatechange = () => {
    console.warn(`[audio] context → ${audio.state}`);
    if (audio.state !== 'running') audio.resume().catch(() => {});
  };
  setInterval(() => {
    if (audio.state !== 'running') audio.resume().catch(() => {});
  }, 1000);
  const gestureResume = () => {
    if (audio.state !== 'running') audio.resume().catch(() => {});
  };
  addEventListener('pointerdown', gestureResume, true);
  addEventListener('keydown', gestureResume, true);
  const fetchBuf = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
    return r.arrayBuffer();
  };
  statusEl.textContent = 'loading…';
  const [wasm1, wasm2, grid, speakers, ariaRaw, aliceRaw, clubRaw, fluteRaw, radioRaw, fxW, fxT, fxB, ambRaw, dropsRaw, mot0, mot1, mot2, mot3] =
    await Promise.all([
    fetchBuf('omg_web.wasm'),
    fetchBuf('omg_web.wasm'),
    fetchBuf('../assets/hrir_grid.bin'),
    fetchBuf('../assets/hrir_dodeca20.bin'),
    fetchBuf('../assets/aria48.ogg'),
    fetchBuf('../assets/alice48.ogg'),
    fetchBuf('../assets/club48.ogg'),
    fetchBuf('../assets/flute48.ogg'),
    fetchBuf('../assets/radio48.ogg'),
    fetchBuf('../assets/fx_whistle.ogg'),
    fetchBuf('../assets/fx_thump.ogg'),
    fetchBuf('../assets/fx_boom.ogg'),
    fetchBuf('../assets/night-nature48.ogg'),
    fetchBuf('../assets/drops48.ogg'),
    fetchBuf('../assets/motor048.ogg'),
    fetchBuf('../assets/motor148.ogg'),
    fetchBuf('../assets/motor248.ogg'),
    fetchBuf('../assets/motor348.ogg'),
  ]);

  // Sources go to the engine raw — import loudness is normalized there
  // (gated RMS), so recording level doesn't matter. FX keep per-type
  // peak targets: their relative energies ARE the type calibration.
  const decodeMono = async (buf, target = 0) => {
    const ab = await audio.decodeAudioData(buf);
    const ch = ab.getChannelData(0);
    const out = new Float32Array(ch.length);
    if (!target) {
      out.set(ch);
      return out;
    }
    let peak = 1e-6;
    for (let i = 0; i < ch.length; i++) peak = Math.max(peak, Math.abs(ch[i]));
    const g = target / peak;
    for (let i = 0; i < ch.length; i++) out[i] = ch[i] * g;
    return out;
  };
  const [aria, alice, club, flute, radio, whistle, thumpFx, boomFx, drops, m0, m1, m2, m3, ambience] = await Promise.all([
    decodeMono(ariaRaw),
    decodeMono(aliceRaw),
    decodeMono(clubRaw),
    decodeMono(fluteRaw),
    decodeMono(radioRaw),
    decodeMono(fxW, 0.18), // whistle: background-y
    decodeMono(fxT, 0.55),
    decodeMono(fxB, 1.9), // boom: BIG (AGC + tanh keep it safe)
    decodeMono(dropsRaw), // recorded splat bank (pre-normalized slices)
    decodeMono(mot0),
    decodeMono(mot1),
    decodeMono(mot2),
    decodeMono(mot3),
    (async () => {
      const ab = await audio.decodeAudioData(ambRaw);
      const L = ab.getChannelData(0);
      let R = ab.numberOfChannels > 1 ? ab.getChannelData(1) : null;
      if (!R) {
        R = new Float32Array(L.length);
        const off = Math.floor(L.length / 3);
        for (let i = 0; i < L.length; i++) R[i] = L[(i + off) % L.length];
      }
      const out = new Float32Array(L.length * 2);
      for (let i = 0; i < L.length; i++) {
        out[i * 2] = L[i];
        out[i * 2 + 1] = R[i];
      }
      return out;
    })(),
  ]);
  state.motors = [m0, m1, m2, m3]; // CC0 motor loops, swapped per spawn
  // Footsteps (CC0, Kenney "Impact Sounds"): 4 surfaces × 5 variants,
  // played as fx one-shots on the dedicated feet source, so each step
  // picks up the room's real acoustics (parquet knock + hall reverb).
  // Shaped on decode: the raw files carry a heavy LF thump that reads
  // as boots-on-a-drum through the engine — highpass at ~160 Hz, trim
  // the silence so steps land ON the stride, keep the peak modest.
  const decodeStep = async (buf) => {
    const raw = await decodeMono(buf, 0.32);
    const k = Math.exp(-2 * Math.PI * 160 / audio.sampleRate);
    let lp = 0;
    for (let i = 0; i < raw.length; i++) {
      lp += (1 - k) * (raw[i] - lp);
      raw[i] -= lp; // one-pole highpass: kill the boom, keep the knock
    }
    let a = 0;
    while (a < raw.length && Math.abs(raw[a]) < 0.01) a++;
    a = Math.max(0, a - 32);
    let b = raw.length - 1;
    while (b > a && Math.abs(raw[b]) < 0.004) b--;
    const out = raw.slice(a, Math.min(b + 480, raw.length));
    for (let i = 0; i < 48 && i < out.length; i++) out[i] *= i / 48;
    for (let i = 0; i < 480 && i < out.length; i++) out[out.length - 1 - i] *= i / 480;
    return out;
  };
  const steps = await Promise.all(
    STEP_SURFACES.flatMap((surf) => [0, 1, 2, 3, 4].map((v) =>
      fetchBuf(`../assets/steps/${surf}_${v}.ogg`).then(decodeStep))),
  );
  // projectile slots + feet: fx voices only (one buffer each — transferables)
  const silents = [new Float32Array(480), new Float32Array(480), new Float32Array(480),
                   new Float32Array(480)];

  await audio.audioWorklet.addModule('worklet.js');
  const node = new AudioWorkletNode(audio, 'omg-engine', {
    numberOfInputs: 0,
    outputChannelCount: [2],
  });
  node.connect(audio.destination);
  state.node = node;
  document.getElementById('rain').onclick = (e) => { e.target.blur(); cycleRain(); };
  const analyser = audio.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.35;
  node.connect(analyser);
  state.analyser = analyser;
  state.specBins = new Uint8Array(analyser.frequencyBinCount);
  node.port.postMessage(
    { type: 'wasm', bytes: wasm1, grid, speakers,
      sources: [aria.buffer, alice.buffer, club.buffer, flute.buffer, radio.buffer,
                silents[0].buffer, silents[1].buffer, silents[2].buffer,
                m0.buffer, m1.buffer, silents[3].buffer],
      fx: [whistle.buffer, thumpFx.buffer, boomFx.buffer, ...steps.map((s) => s.buffer)],
      ambient: ambience.buffer, drops: drops.buffer },
    [wasm1, grid, speakers, aria.buffer, alice.buffer, club.buffer, flute.buffer,
     radio.buffer, silents[0].buffer, silents[1].buffer, silents[2].buffer, silents[3].buffer,
     whistle.buffer, thumpFx.buffer, boomFx.buffer, ...steps.map((s) => s.buffer),
     ambience.buffer, drops.buffer],
  );
  await new Promise((res, rej) => {
    const watchdog = setTimeout(
      () => rej(new Error('engine init timed out (worklet never became ready)')), 12000);
    node.port.onmessage = (e) => {
      if (e.data.type === 'ready') { clearTimeout(watchdog); res(); }
      if (e.data.type === 'error') { clearTimeout(watchdog); rej(new Error('worklet: ' + e.data.message)); }
      if (e.data.type === 'error') console.error('worklet:', e.data.message);
      if (e.data.type === 'meters') {
        state.meters.l = e.data.l;
        state.meters.r = e.data.r;
        state.meters.agc = e.data.agc;
        state.meters.pts = e.data.pts || 0;
        state.meters.ceil = e.data.ceil || 0;
        state.meters.tts = e.data.tts || 0;
        if (e.data.chans) {
          state.chanHist.push(e.data.chans);
          if (state.chanHist.length > 43) state.chanHist.shift();
        }
        state.meters.hist.push(e.data.agc);
        if (state.meters.hist.length > 220) state.meters.hist.shift();
        // Black-box recorder for the "sound suddenly stops" class of
        // field report: keep ~4 s of fine-grained context; when the
        // OUTPUT collapses >30 dB below its own recent level while the
        // engine still believes sources are audible, dump the window to
        // the console — one repro turns ear-memory into data.
        {
          const bb = state.bbox;
          const now = performance.now() / 1000;
          const peak = Math.max(e.data.l, e.data.r);
          let srcRms = 0;
          for (let i = 0; i < 22; i += 2) srcRms = Math.max(srcRms, e.data.chans[i + 1]);
          bb.ring.push({
            t: +now.toFixed(2), peak: +peak.toFixed(5), agc: +e.data.agc.toFixed(3),
            tts: +(e.data.tts || 0).toFixed(2), src: +srcRms.toFixed(5),
            x: +state.pose.x.toFixed(1), y: +state.pose.y.toFixed(1),
            room: state.simState ? state.simState[2] | 0 : -1,
            tick: +state.debug.tickMs.toFixed(1),
            gaps: e.data.gaps || 0, load: +(e.data.load || 0).toFixed(2),
            raf: +((state.debug.rafGap || 0)).toFixed(0),
            lat: state.audio ? +(state.audio.outputLatency * 1000).toFixed(0) : -1,
            ctx: state.audio ? state.audio.state[0] : '?', // r/s/i/c
          });
          state.debug.rafGap = 0;
          if (bb.ring.length > 180) bb.ring.shift(); // ~4 s at 43 Hz
          const ago = bb.ring.filter((s) => now - s.t > 1.0 && now - s.t < 3.0);
          if (ago.length > 20 && now > bb.cooldownUntil) {
            const ref = ago.map((s) => s.peak).sort((a, b) => a - b)[ago.length >> 1];
            const recent = bb.ring.slice(-12); // ~280 ms
            const collapsed = ref > 0.02
              && recent.every((s) => s.peak < ref / 32 && s.src > 1e-4);
            if (collapsed) {
              bb.cooldownUntil = now + 10;
              bb.collapses.push(now);
              console.warn('[blackbox] output collapse — last 4 s:\n'
                + bb.ring.map((s) => JSON.stringify(s)).join('\n'));
            }
          }
        }
        // field-debug snapshots (chans/amb/dbg arrive ~43 Hz; the panel
        // keeps a decimated ~11 Hz history, ≈45 s deep)
        state.debug.chans = e.data.chans;
        state.debug.amb = e.data.amb;
        state.debug.dbg = e.data.dbg;
        state.debug.load = e.data.load || 0;
        state.debug.gaps = e.data.gaps || 0;
        state.debug.gapMs = e.data.gapMs || 0;
        if ((state.debug.seq = (state.debug.seq || 0) + 1) % 4 === 0) {
          const d = e.data.dbg || [];
          let taps = 0;
          let gain = 0;
          for (let i = 0; i < 11; i++) {
            taps += d[i * 5] || 0;
            gain += d[i * 5 + 2] || 0;
          }
          state.debug.hist.push({
            rms: (e.data.chans || []).filter((_, k) => k % 2 === 1),
            taps,
            gain,
            pts: e.data.pts || 0,
            load: e.data.load || 0,
            tickMs: state.debug.tickMs || 0,
          });
          if (state.debug.hist.length > 500) state.debug.hist.shift();
        }
      }
    };
  });

  const worker = new Worker('worker.js');
  worker.postMessage({ type: 'init', bytes: wasm2 }, [wasm2]);
  // Sim-side quality ladder (GPU_PLAN.md Track B): ?q=high|med|low pins a
  // tier, ?q=auto (default) walks it from the measured tick cost. The
  // audio-side ladder (point budget + tap ceiling) lives in the worklet's
  // own governor — this one only paces the 20 Hz sim.
  {
    const qp = (new URLSearchParams(location.search).get('q') || 'auto').toLowerCase();
    const tiers = { high: 0, med: 1, medium: 1, low: 2 };
    // Auto starts at LOW and earns its way up: the cost of starting too
    // high is audible (breakup until the first shed), the cost of
    // starting too low is a few seconds of soft focus nobody notices.
    state.quality = {
      auto: !(qp in tiers), tier: tiers[qp] ?? 2, calm: 0, hot: 0, forced: '',
      mode: qp in tiers ? (qp === 'medium' ? 'med' : qp) : 'auto',
    };
    worker.postMessage({ type: 'quality', tier: state.quality.tier });
    // Manual override (Q key / console): pins the sim tier AND floors the
    // worklet's tap-ceiling ladder to match, so a pinned "low" is low on
    // both clocks. "auto" hands both governors back their freedom.
    state.setQuality = (mode) => {
      const q = state.quality;
      q.mode = mode;
      q.calm = 0;
      q.forced = '';
      if (mode === 'auto') {
        q.auto = true;
      } else {
        q.auto = false;
        q.tier = tiers[mode];
        worker.postMessage({ type: 'quality', tier: q.tier });
      }
      node.port.postMessage({ type: 'ceilfloor', idx: mode === 'med' ? 1 : mode === 'low' ? 2 : 0 });
    };
    // quality-panel plumbing: pin one sim lever (value 0 = follow tier),
    // or hold both audio levers manually (freezes the worklet governor)
    state.setOverride = (id, value) => worker.postMessage({ type: 'override', id, value });
    state.setAudioManual = (on, points, ceiling) =>
      node.port.postMessage({ type: 'manual', on, points, ceiling });
    state.setGpu = (on) => worker.postMessage({ type: 'gpu', on });
    state.setEarly = (mode) => {
      state.earlyMode = mode;
      try { localStorage.setItem('omg-early', mode ? 'traced' : 'ism'); } catch (_) {}
      worker.postMessage({ type: 'early', mode });
    };
    // a real setting: ?early=traced|ism wins, else the remembered
    // choice, else ism (the classic engine)
    {
      const p = (new URLSearchParams(location.search).get('early') || '').toLowerCase();
      let stored = null;
      try { stored = localStorage.getItem('omg-early'); } catch (_) {}
      const pick = p === 'traced' || p === 'pt' ? 1
        : p === 'ism' ? 0
          : stored === 'traced' ? 1 : 0;
      state.earlyMode = pick;
      if (pick) worker.postMessage({ type: 'early', mode: pick });
    }
  }
  worker.onmessage = (e) => {
    if (e.data.type !== 'tick') return;
    node.port.postMessage({ type: 'params', blocks: e.data.blocks }, e.data.blocks);
    state.simState = new Float32Array(e.data.state);
    state.debug.tickMs = Math.max(state.debug.tickMs || 0, e.data.tickMs || 0) * 0.9
      + (e.data.tickMs || 0) * 0.1; // decaying peak-ish: spikes stay visible
    state.debug.gpu = e.data.gpu || 0;
    state.debug.gpuAvail = e.data.gpuAvail || 0;
    state.debug.gpuMs = e.data.gpuMs || 0;
    state.debug.gpuDuty = e.data.gpuDuty || 0;
    state.debug.early = e.data.early || 0;
    state.debug.ptMs = e.data.ptMs || 0;
    state.debug.ptN = e.data.ptN || 0;
    const q = state.quality;
    if (q.auto) {
      // shed on a blown budget immediately (a 20 Hz tick has 50 ms; at
      // >40 the worker starts eating the next tick's slot), recover only
      // after ~10 s of calm so the ladder can't oscillate at a doorway
      if (e.data.tickMs > 40 && q.tier < 2) {
        q.tier++;
        q.calm = 0;
        q.forced = `tick ${e.data.tickMs.toFixed(0)}ms`;
        worker.postMessage({ type: 'quality', tier: q.tier });
      } else if (state.debug.load > 0.8 && q.tier < 2) {
        // sustained audio pressure sheds the sim tier too — a lower
        // tier means fewer taps at the source of the pipeline
        if (++q.hot >= 20) { // ~1 s of ticks over 80%
          q.hot = 0;
          q.tier++;
          q.calm = 0;
          q.forced = `load ${(state.debug.load * 100).toFixed(0)}%`;
          worker.postMessage({ type: 'quality', tier: q.tier });
        }
      } else if (e.data.tickMs < 15 && q.tier > 0) {
        // climb only with AUDIO headroom too: a higher tier raises tap
        // counts, and the render load must never be pushed past 90%
        q.hot = 0;
        if (state.debug.load >= 0.8) {
          q.calm = 0;
        } else if (++q.calm >= 200) { // 200 ticks ≈ 10 s
          q.tier--;
          q.calm = 0;
          q.forced = '';
          worker.postMessage({ type: 'quality', tier: q.tier });
        }
      } else {
        q.calm = 0;
      }
    }
    // environment block — offset comes from the sim (ONE source of
    // truth for the state layout; a hardcoded 58 once parsed car
    // positions as ambience gains)
    state.envOff = e.data.envOff;
    const env = state.simState.slice(e.data.envOff).buffer;
    node.port.postMessage({ type: 'env', env }, [env]);
  };

  state.fx = (src, kind, action = 'play') =>
    node.port.postMessage({ type: 'fx', src, kind, action });
  setInterval(() => {
    // face-tracked head translation: a small world-space offset of the
    // listening position (lean left → ears move left). A lean must never
    // put the ears through geometry — crossing a jamb wall would slam
    // every source behind masonry at once — so it holds at the body when
    // the offset segment crosses a wall.
    const ch = Math.cos(state.heading);
    const sh = Math.sin(state.heading);
    const f = state.face;
    let px = state.pose.x + ch * f.dz - sh * f.dx;
    let py = state.pose.y + sh * f.dz + ch * f.dx;
    if ((f.dx || f.dz)
        && crossesWall(state.pose.x, state.pose.y, px, py, (state.pose.z || 0) + 1.0)) {
      px = state.pose.x;
      py = state.pose.y;
    }
    worker.postMessage({
      type: 'pose',
      x: px,
      y: py,
      z: EYE + (state.pose.z || 0) + (state.airZ || 0) + f.dy - 0.7 * (state.crouch || 0),
      yaw: 0,
      // animated leaf positions: the sim prices the swing continuously
      doors: state.doorMeshes.map((dm) => dm.openness),
      projs: [
        ...state.projs.map((p) => [p.slot, p.x, p.y, p.z]),
        ...state.cars.map((c) => [c.slot, c.x, c.y, 0.7, c.vol]),
        // feet: always-active silent source at ground level under the
        // body (step one-shots mix into it as fx voices)
        [5, state.pose.x, state.pose.y, Math.max(0.2, (state.pose.z || 0) + 0.2), 1],
      ],
    });
  }, 50);
  setInterval(() => {
    node.port.postMessage({
      type: 'head',
      yaw: state.heading + state.face.yaw,
      pitch: Math.max(-1.5, Math.min(1.5, state.pitch + state.face.pitch)),
      roll: state.face.roll,
    });
  }, 16);

  await audio.resume();
}

// ------------------------------------------------------------ controls

function setupControls() {
  const isTouch = matchMedia('(pointer: coarse)').matches;

  // Q cycles the quality ladder: auto → high → med → low. Pinning a tier
  // sets the sim budgets AND floors the audio-side tap ceiling (see
  // state.setQuality); auto returns both governors to their own metering.
  const cycleQuality = () => {
    if (!state.setQuality) return; // audio not started yet
    const order = ['auto', 'high', 'med', 'low'];
    const next = order[(order.indexOf(state.quality.mode) + 1) % order.length];
    state.setQuality(next);
    hintEl.textContent = `quality: ${next}${next === 'auto' ? '' : ' (pinned)'}`;
  };

  if (!isTouch) {
    hintEl.textContent = 'click: capture mouse · WASD walk · Shift run · Space jump · C crouch · click throw · E door · R rain';
    glCanvas.addEventListener('click', () => {
      if (document.pointerLockElement !== glCanvas) glCanvas.requestPointerLock();
      else throwBall();
    });
    addEventListener('mousemove', (e) => {
      if (document.pointerLockElement !== glCanvas) return;
      state.heading -= e.movementX * 0.0025;
      state.pitch = Math.max(-1.4, Math.min(1.4, state.pitch - e.movementY * 0.0025));
    });
    addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); state.wantJump = true; }
      else if (e.code === 'KeyR') cycleRain();
      else if (e.code === 'KeyE') toggleNearestDoor();
      else if (e.code === 'KeyQ') cycleQuality();
      else state.keys.add(e.code);
    });
    addEventListener('keyup', (e) => state.keys.delete(e.code));
  } else {
    hintEl.textContent = 'left thumb: walk · turn your body (or drag right) to look';
    glCanvas.addEventListener('pointerdown', (e) => {
      if (e.clientX < innerWidth / 2 && !state.joy) {
        state.joy = { id: e.pointerId, sx: e.clientX, sy: e.clientY, dx: 0, dy: 0 };
      } else if (!state.look) {
        state.look = { id: e.pointerId, sx: e.clientX, sy: e.clientY,
                       heading0: state.heading, pitch0: state.pitch };
      }
      glCanvas.setPointerCapture(e.pointerId);
    });
    glCanvas.addEventListener('pointermove', (e) => {
      if (state.joy && e.pointerId === state.joy.id) {
        state.joy.dx = e.clientX - state.joy.sx;
        state.joy.dy = e.clientY - state.joy.sy;
      } else if (state.look && e.pointerId === state.look.id) {
        state.heading = state.look.heading0 - (e.clientX - state.look.sx) * 0.008;
        state.pitch = Math.max(-1.4, Math.min(1.4,
          state.look.pitch0 - (e.clientY - state.look.sy) * 0.008));
      }
    });
    const up = (e) => {
      if (state.joy && e.pointerId === state.joy.id) state.joy = null;
      if (state.look && e.pointerId === state.look.id) state.look = null;
    };
    glCanvas.addEventListener('pointerup', up);
    glCanvas.addEventListener('pointercancel', up);

    const btn = document.createElement('button');
    btn.textContent = '🎯 throw';
    btn.style.cssText =
      'position:absolute;right:12px;bottom:270px;z-index:6;padding:10px 16px;'
      + 'background:var(--panel-2);color:var(--ink);border:1px solid var(--line);'
      + 'border-radius:8px;font-size:12px;';
    document.body.appendChild(btn);
    btn.addEventListener('click', throwBall);

    // Android Chrome: compass-fused orientation, no permission prompt.
    addEventListener('deviceorientationabsolute', (e) => {
      if (e.alpha == null) return;
      state.orientation = e.alpha;
      if (state.orientationOffset == null) {
        state.orientationOffset = state.heading + (e.alpha * Math.PI) / 180;
      }
      state.heading = state.orientationOffset - (e.alpha * Math.PI) / 180;
      // pitch from device tilt (portrait: upright beta ≈ 90°) — tilting
      // the phone up looks up, like turning the body drives heading
      if (e.beta != null) {
        state.pitch = Math.max(-1.4, Math.min(1.4, ((e.beta - 90) * Math.PI) / 180));
      }
    });
  }

  document.getElementById('recenter').onclick = () => {
    if (state.orientation != null) {
      state.orientationOffset = Math.PI / 2 + (state.orientation * Math.PI) / 180;
    } else {
      state.heading = Math.PI / 2;
      state.pitch = 0;
    }
    // face tracking re-zeros on the current sitting pose
    state.faceTrack?.recenter();
  };
}

// Camera face tracking: real head movements drive the engine's head
// (audio only — the view stays put, that's the point: turn your head
// and HEAR the world stay fixed while the screen keeps facing you).
async function toggleFaceTracking() {
  const btn = document.getElementById('face');
  btn.blur();
  if (state.faceTrack) {
    state.faceTrack.stop();
    state.faceTrack = null;
    state.faceTarget = null;
    state.face = { yaw: 0, pitch: 0, roll: 0, dx: 0, dy: 0, dz: 0 };
    btn.textContent = '🎥 face';
    return;
  }
  btn.disabled = true;
  btn.textContent = '🎥 …';
  try {
    const { startFaceTracking } = await import('./facetrack.js');
    const clamp = (v, r) => Math.max(-r, Math.min(r, v));
    state.faceTrack = await startFaceTracking((p) => {
      state.faceTarget = {
        yaw: clamp(p.yaw, 1.2),
        pitch: clamp(p.pitch, 1.2),
        roll: clamp(p.roll, 1.0),
        dx: clamp(p.dx, 0.35),
        dy: clamp(p.dy, 0.35),
        dz: clamp(p.dz, 0.5),
      };
    });
    btn.textContent = '🎥 on';
  } catch (e) {
    console.error('face tracking unavailable:', e);
    btn.textContent = '🎥 ✗';
    setTimeout(() => { btn.textContent = '🎥 face'; }, 2500);
  }
  btn.disabled = false;
}

function throwBall() {
  if (state.projs.length >= 3 || !state.fx) return;
  const usedSlots = new Set(state.projs.map((p) => p.slot));
  let slot = 0;
  while (usedSlots.has(slot)) slot++;
  const ch = Math.cos(state.heading);
  const sh = Math.sin(state.heading);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const speed = 13;
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xfff1a8 }),
  );
  const light = new THREE.PointLight(0xffcf7a, 3, 6);
  scene.add(mesh);
  scene.add(light);
  state.projs.push({
    slot,
    x: state.pose.x + ch * 0.4, y: state.pose.y + sh * 0.4, z: EYE + (state.pose.z || 0) + (state.airZ || 0) - 0.7 * (state.crouch || 0),
    vx: ch * cp * speed, vy: sh * cp * speed, vz: sp * speed + 2.5,
    landedAt: 0, boomAt: 0, mesh, light,
  });
  state.fx(5 + slot, 0, 'play'); // whistle
}

function roomHeightAt(x, y) {
  for (const r of ROOMS) {
    if (!r.outdoor && x > r.min[0] && x < r.max[0] && y > r.min[1] && y < r.max[1]) return r.h;
  }
  return Infinity;
}

function updateProjectiles(dt, now) {
  for (const p of [...state.projs]) updateProjectile(p, dt, now);
}

function updateProjectile(p, dt, now) {
  const fx = (kind, action = 'play') => state.fx(5 + p.slot, kind, action);

  if (p.boomAt) {
    // explosion flash decays, then the source goes silent and disappears
    const k = (now - p.boomAt) / 1000;
    p.light.intensity = Math.max(0, 60 * (1 - k / 0.5));
    p.mesh.scale.setScalar(1 + k * 14);
    p.mesh.material.opacity = Math.max(0, 1 - k / 0.6);
    if (k > 2.4) {
      scene.remove(p.mesh);
      scene.remove(p.light);
      state.projs = state.projs.filter((q) => q !== p);
    }
    return;
  }

  p.vz -= 9.8 * dt;
  const nx = p.x + p.vx * dt;
  const ny = p.y + p.vy * dt;
  let nz = p.z + p.vz * dt;

  // wall bounce (height-aware: clears walls it flies over)
  if (crossesWall(p.x, p.y, nx, p.y, p.z)) {
    p.vx = -p.vx * 0.6;
    if (Math.hypot(p.vx, p.vy, p.vz) > 1.5) fx(1);
  } else p.x = nx;
  if (crossesWall(p.x, p.y, p.x, ny, p.z)) {
    p.vy = -p.vy * 0.6;
    if (Math.hypot(p.vx, p.vy, p.vz) > 1.5) fx(1);
  } else p.y = ny;

  // ceiling (from inside), roof landing (from above), ground
  const hRoom = roomHeightAt(p.x, p.y);
  if (hRoom !== Infinity) {
    if (p.z <= hRoom - 0.15 && nz > hRoom - 0.15 && p.vz > 0) {
      p.vz = -p.vz * 0.5;
      nz = hRoom - 0.15;
      fx(1);
    } else if (p.z >= hRoom + 0.1 && nz < hRoom + 0.1) {
      nz = hRoom + 0.12;
      if (Math.abs(p.vz) > 1.2) fx(1);
      p.vz = -p.vz * 0.5;
      p.vx *= 0.8;
      p.vy *= 0.8;
      if (!p.landedAt) p.landedAt = now;
    }
  }
  if (nz < 0.12) {
    nz = 0.12;
    if (Math.abs(p.vz) > 1.2) fx(1);
    p.vz = -p.vz * 0.55;
    p.vx *= 0.75;
    p.vy *= 0.75;
    if (!p.landedAt) p.landedAt = now;
  }
  p.z = nz;

  if (p.landedAt && now - p.landedAt > 3000 && !p.boomAt) {
    p.boomAt = now;
    fx(0, 'stop');
    fx(2, 'play');
    p.mesh.material = new THREE.MeshBasicMaterial({
      color: 0xffb46a, transparent: true, opacity: 1,
    });
  }

  p.mesh.position.copy(v3(p.x, p.y, p.z));
  p.light.position.copy(v3(p.x, p.y, p.z + 0.1));
}

const WALK_SPEED = 3.0;
let lastT = 0;

// Feet height at a position: the Old House stairs rise to the upper
// floor (slab top z = 3.0); everywhere else is ground.
function floorHeightAt(x, y, curZ) {
  const inHouse = x > 24.05 && x < 30.95 && y > 16.05 && y < 22.95;
  const inStairCol = x > 24.25 && x < 25.55;
  if (inHouse && inStairCol && y > 16.9) {
    if (y >= 21.3) return 3.0; // landing
    return 3.0 * Math.min(1, Math.max(0, (y - 17.0) / (21.3 - 17.0)));
  }
  if (inHouse && curZ > 1.5) return 3.0; // upper floor
  // bunker: the ramp descends eastward into the pit; the surface above
  // the buried box stays solid ground
  if (x > 32.4 && x < 34.1 && y > 6.15 && y < 7.85) {
    return -3.0 * Math.min(1, Math.max(0, (x - 32.5) / 1.4));
  }
  if (x > 34.05 && x < 39.95 && y > 6.05 && y < 13.95 && curZ < -0.5) return -3.0;
  return 0;
}

function movePose(t) {
  const dt = Math.min((t - lastT) / 1000, 0.05);
  lastT = t;
  // crouch (hold C / Ctrl): ears sink ~0.7 m — behind the club bar or
  // the sofa that puts your head in the traced engine's shadow
  const wantCrouch = state.keys.has('KeyC') || state.keys.has('ControlLeft') ? 1 : 0;
  state.crouch = (state.crouch || 0) + (wantCrouch - (state.crouch || 0)) * Math.min(1, dt * 8);
  // jump (Space): simple ballistic hop above the floor-follow height
  if (state.wantJump) {
    state.wantJump = false;
    if (!(state.airZ > 0) && (state.crouch || 0) < 0.4) state.vz = 4.2;
  }
  if (state.vz || state.airZ) {
    state.airZ = (state.airZ || 0) + (state.vz || 0) * dt;
    state.vz = (state.vz || 0) - 9.8 * dt;
    if (state.airZ <= 0) {
      state.airZ = 0;
      state.vz = 0;
      strideAcc = strideNext; // landing thud via the next stride check
    }
  }
  // follow the floor (stairs are a ramp; snap gently)
  const target = floorHeightAt(state.pose.x, state.pose.y, state.pose.z || 0);
  state.pose.z = (state.pose.z || 0) + (target - (state.pose.z || 0)) * Math.min(1, dt * 10);
  let fwd = 0;
  let strafe = 0;

  if (state.keys.has('KeyW') || state.keys.has('ArrowUp')) fwd += 1;
  if (state.keys.has('KeyS') || state.keys.has('ArrowDown')) fwd -= 1;
  if (state.keys.has('KeyD') || state.keys.has('ArrowRight')) strafe += 1;
  if (state.keys.has('KeyA') || state.keys.has('ArrowLeft')) strafe -= 1;

  if (state.joy) {
    const mag = Math.hypot(state.joy.dx, state.joy.dy);
    if (mag > 8) {
      const k = Math.min(mag, 80) / 80;
      fwd += (-state.joy.dy / Math.max(mag, 1)) * k;
      strafe += (state.joy.dx / Math.max(mag, 1)) * k;
    }
  }
  if (!fwd && !strafe) return;

  // forward = along heading, strafe-right = heading − 90°
  const mag = Math.hypot(fwd, strafe);
  const len = Math.min(mag, 1);
  const f = (fwd / mag) * len;
  const r = (strafe / mag) * len;
  const ch = Math.cos(state.heading);
  const sh = Math.sin(state.heading);
  const sprint = state.keys.has('ShiftLeft') || state.keys.has('ShiftRight') ? 1.8 : 1;
  const step = WALK_SPEED * dt * sprint * (1 - 0.45 * (state.crouch || 0));
  const nx = state.pose.x + (ch * f + sh * r) * step;
  const ny = state.pose.y + (sh * f - ch * r) * step;
  const { x, y } = state.pose;
  if (walkableMove(x, y, nx, ny)) {
    state.pose.x = nx;
    state.pose.y = ny;
  } else if (walkableMove(x, y, nx, y)) {
    state.pose.x = nx;
  } else if (walkableMove(x, y, x, ny)) {
    state.pose.y = ny;
  }
  strideStep(Math.hypot(state.pose.x - x, state.pose.y - y));
}

// ------------------------------------------------------------ footsteps
// Distance-driven, not time-driven: every ~stride of actual horizontal
// travel lands one step at the feet source (dyn slot 5, source 10),
// which sits at ground level under the body — the engine gives each
// step the room's own acoustics for free. Surface from the room map.
const STEP_SURFACES = ['wood', 'carpet', 'concrete', 'grass'];
const ROOM_SURFACE = {
  'Living Room': 'wood', Corridor: 'wood', 'Great Hall': 'concrete',
  Entrance: 'concrete', Club: 'concrete', 'Old House': 'wood',
  'Old House Upper': 'carpet', Cathedral: 'concrete', Bunker: 'concrete',
  Outside: 'grass',
};
let strideAcc = 0;
let strideNext = 1.5;
let lastStepT = 0;
let lastVariant = -1;
function strideStep(d) {
  if (state.airZ > 0) return; // feet are off the ground
  strideAcc += d;
  if (strideAcc < strideNext || !state.fx) return;
  // human cadence gate: even sprinting, feet don't land more than
  // ~3 times a second — distance alone over-triggers at game speeds
  const now = performance.now();
  if (now - lastStepT < 320) return;
  lastStepT = now;
  strideAcc = 0;
  strideNext = 1.35 + Math.random() * 0.3; // no metronome feet
  const room = state.simState ? ROOMS[state.simState[2] | 0] : null;
  const surf = ROOM_SURFACE[room ? room.name : 'Outside'] || 'concrete';
  let v = (Math.random() * 5) | 0;
  if (v === lastVariant) v = (v + 1) % 5; // never the same step twice
  lastVariant = v;
  state.fx(10, 3 + STEP_SURFACES.indexOf(surf) * 5 + v);
}

// ------------------------------------------------------------ render loop

const mm = document.getElementById('minimap').getContext('2d');

function frame(t) {
  if (!state.running) {
    // idle flyover while the start overlay is up
    const a = t / 12000;
    camera.position.copy(v3(17 + 26 * Math.cos(a), 19 + 26 * Math.sin(a), 16));
    camera.lookAt(v3(17, 19, 1));
  } else {
    const dt = Math.min((t - (state.prevT || t)) / 1000, 0.05);
    // main-thread stall detector: the longest rAF gap since the last
    // black-box snapshot (face tracking, GC and layout all show up here)
    state.debug.rafGap = Math.max(state.debug.rafGap || 0, t - (state.prevT || t));
    state.prevT = t;
    movePose(t);
    updateProjectiles(dt, t);
    updateDoors(dt);
    updateCars(dt, t);
    if (state.faceTarget) {
      // one-pole toward the last camera-frame pose (τ 45 ms) bridges
      // 30–60 fps detections to render rate; the engine smooths 30 ms
      // more on top
      const k = 1 - Math.exp(-dt / 0.045);
      const f = state.face;
      const g = state.faceTarget;
      for (const key of ['yaw', 'pitch', 'roll', 'dx', 'dy', 'dz']) {
        f[key] += (g[key] - f[key]) * k;
      }
    }
    camera.position.copy(v3(state.pose.x, state.pose.y, EYE + (state.pose.z || 0) + (state.airZ || 0) - 0.7 * (state.crouch || 0)));
    camera.rotation.y = state.heading - Math.PI / 2;
    camera.rotation.x = state.pitch;
  }

  const tt = t / 1000;
  for (const src of SOURCES) {
    const s = 1 + 0.25 * Math.sin(tt * 2 * Math.PI * 1.6);
    for (const m of src.meshes) m.scale.setScalar(s);
  }

  renderer.render(scene, camera);
  if (state.running) {
    drawMinimap();
    drawMeters();
    drawSpec();
    drawFaceWire();
    drawSoundRays();
    if (state.debug.on) drawDebug();
  }

  const st = state.simState;
  if (st) {
    let line = `${ROOMS[st[2] | 0].name} · RT60 ${st[3].toFixed(2)}s`;
    if (state.faceTrack) {
      // live sign check in the field: turn left → yaw +, look up → pitch +
      const deg = (r) => (r * 57.2958).toFixed(0).padStart(3);
      const s = state.faceTrack.stats;
      line += s.tracking
        ? ` · 🎥 y${deg(state.face.yaw)}° p${deg(state.face.pitch)}° ${s.fps.toFixed(0)}fps ${s.ms.toFixed(0)}ms`
        : ' · 🎥 no face';
    }
    statusEl.textContent = line;
  }
  requestAnimationFrame(frame);
}

// ------------------------------------------------------- debug panel
// "What is actually playing right now, and why": per-source live state
// straight from the engine (levels, tap slots, reverb sends) plus
// scrolling history of the totals — pile-ups, leaks and perf spikes are
// visible at a glance instead of reconstructed from ear-memory.

const DBG_NAMES = ['music', 'voice', 'club', 'flute', 'radio',
  'ball0', 'ball1', 'ball2', 'car0', 'car1', 'feet', 'amb', 'rain'];
const DBG_COLORS = ['#ffaa3c', '#6ee0a0', '#ff5a9e', '#9ad2ff', '#d2b06e',
  '#fff1a8', '#fff1a8', '#fff1a8', '#7ad7ff', '#7ad7ff', '#ddd6c9', '#8a95ff', '#59c9e8'];

function dbgSourceDist(i) {
  if (i < 5) {
    const s = SOURCES[i];
    return Math.hypot(s.pos[0] - state.pose.x, s.pos[1] - state.pose.y);
  }
  if (i < 8) {
    const p = state.projs.find((q) => q.slot === i - 5);
    return p ? Math.hypot(p.x - state.pose.x, p.y - state.pose.y) : null;
  }
  const c = state.cars.find((q) => q.slot === i - 5);
  return c ? Math.hypot(c.x - state.pose.x, c.y - state.pose.y) : null;
}

function drawDebug() {
  const c = document.getElementById('debug');
  if (!c) return;
  const g = c.getContext('2d');
  const W = c.width;
  const H = c.height;
  g.clearRect(0, 0, W, H);
  const chans = state.debug.chans || [];
  const dbg = state.debug.dbg || [];
  const db = (x) => 20 * Math.log10((x || 0) + 1e-6);

  // --- per-source table: level, tap slots (points), reverb sends, dist
  g.font = '17px monospace';
  g.textAlign = 'left';
  g.fillStyle = '#7a8496';
  g.fillText('source    dB  taps  fdn  rem    d', 14, 30);
  let y = 54;
  for (let i = 0; i < 13; i++) {
    const rms = chans[i * 2 + 1] || 0;
    const audible = rms > 1e-5;
    g.fillStyle = audible ? DBG_COLORS[i] : '#3a4456';
    let line = DBG_NAMES[i].padEnd(7)
      + db(rms).toFixed(0).padStart(5);
    if (i < 11) {
      const live = dbg[i * 5] || 0;
      const pts = dbg[i * 5 + 1] || 0;
      const fdn = dbg[i * 5 + 3] || 0;
      const rem = dbg[i * 5 + 4] || 0;
      line += ` ${String(live).padStart(2)}/${String(pts).padEnd(2)}`
        + ` ${fdn > 1e-4 ? db(fdn).toFixed(0).padStart(4) : '   ·'}`
        + ` ${rem > 1e-4 ? db(rem).toFixed(0).padStart(4) : '   ·'}`;
      const d = dbgSourceDist(i);
      line += d == null ? '    —' : `${d.toFixed(0).padStart(5)}m`;
    }
    g.fillText(line, 14, y);
    y += 24;
  }
  // ambience inlets (routes through the dome bins + seep)
  const amb = state.debug.amb;
  if (amb) {
    g.fillStyle = '#8a95ff';
    const slots = amb.slice(4).filter((v) => v > 1e-4).length;
    g.fillText(
      `amb inlets ${slots}/8  seep ${amb.slice(1, 4).map((v) => v.toFixed(2)).join(' ')}`,
      14, y);
  }
  if (state.bbox.collapses.length) {
    g.fillStyle = '#ff6a5a';
    const last = state.bbox.collapses[state.bbox.collapses.length - 1];
    g.fillText(
      `⚠ ${state.bbox.collapses.length} output collapse(s) — last @${last.toFixed(0)}s (console)`,
      280, y);
  }
  y += 34;

  // --- scrolling graphs
  const hist = state.debug.hist;
  const graph = (label, h, fn, opts = {}) => {
    g.fillStyle = '#161c26';
    g.fillRect(10, y, W - 20, h);
    g.strokeStyle = opts.color || '#50dcff';
    g.lineWidth = 2;
    g.beginPath();
    for (let k = 0; k < hist.length; k++) {
      const gx = 10 + (k / 499) * (W - 20);
      const v = Math.max(0, Math.min(1, fn(hist[k])));
      const gy = y + h - v * (h - 4) - 2;
      k ? g.lineTo(gx, gy) : g.moveTo(gx, gy);
    }
    g.stroke();
    g.fillStyle = '#7a8496';
    g.fillText(label, 14, y + 20);
    y += h + 10;
  };
  // per-source levels overlaid (audible channels only)
  {
    const h = 120;
    g.fillStyle = '#161c26';
    g.fillRect(10, y, W - 20, h);
    for (let i = 0; i < 12; i++) {
      const latest = hist.length ? hist[hist.length - 1].rms[i] || 0 : 0;
      if (latest < 1e-5) continue;
      g.strokeStyle = DBG_COLORS[i];
      g.lineWidth = 2;
      g.beginPath();
      for (let k = 0; k < hist.length; k++) {
        const gx = 10 + (k / 499) * (W - 20);
        const v = Math.max(0, Math.min(1, (db(hist[k].rms[i]) + 80) / 80));
        const gy = y + h - v * (h - 4) - 2;
        k ? g.lineTo(gx, gy) : g.moveTo(gx, gy);
      }
      g.stroke();
    }
    g.fillStyle = '#7a8496';
    g.fillText('levels −80…0 dB', 14, y + 20);
    y += h + 10;
  }
  const last = hist.length ? hist[hist.length - 1] : null;
  graph(`render taps ${last ? last.taps : 0} · budget ${state.meters.pts}/src`
    + ` · ceil ${state.meters.ceil || 160}/src`,
    90, (s) => s.taps / 96, { color: '#6ee0a0' });
  const q = state.quality || { auto: true, tier: 0 };
  graph(`sim tick ${state.debug.tickMs.toFixed(1)} ms (budget 50)`
    + ` · quality ${['high', 'med', 'low'][q.tier]}${q.auto ? ' (auto)' : ''}`
    + (q.forced ? ` ← ${q.forced}` : ''),
    90, (s) => s.tickMs / 50, { color: '#ffaa3c' });
  // outputLatency is the device-side buffer: stable = healthy output
  // path; jumping around = the DEVICE is struggling (Bluetooth), not us
  const lat = state.audio ? (state.audio.outputLatency * 1000).toFixed(0) : '?';
  graph(`render load ${(state.debug.load * 100).toFixed(0)}% of realtime`
    + ` · ${state.debug.gaps || 0} gaps (${(state.debug.gapMs || 0).toFixed(0)}ms silent)`
    + ` · out-lat ${lat}ms`,
    90, (s) => s.load, { color: '#ff6a5a' });
}

// ------------------------------------------------------------ face wireframe
// A small wire head mirroring what the tracker believes: yaw/pitch/roll
// rotate it, lean (dx/dy) shifts it, lean-in (dz) scales it. If sound and
// wireframe disagree with your actual head, the tracker is the problem.
const FACE_WIRE = (() => {
  const ring = (fn, n = 24) =>
    Array.from({ length: n + 1 }, (_, i) => fn((i / n) * 2 * Math.PI));
  return [
    ring((a) => [Math.cos(a), 0, Math.sin(a)]),                       // equator
    ring((a) => [0.86 * Math.cos(a), 0.5, 0.86 * Math.sin(a)]),       // upper lat
    ring((a) => [0.86 * Math.cos(a), -0.5, 0.86 * Math.sin(a)]),      // lower lat
    ring((a) => [0, Math.cos(a), Math.sin(a)]),                       // profile ring
    [[0, 0.05, 1.0], [0, -0.12, 1.28], [0, -0.22, 1.0]],              // nose
    [[-0.3, -0.42, 0.88], [0, -0.5, 0.94], [0.3, -0.42, 0.88]],       // mouth
    [[-0.5, 0.22, 0.82], [-0.2, 0.22, 0.95]],                          // left eye
    [[0.2, 0.22, 0.95], [0.5, 0.22, 0.82]],                            // right eye
  ];
})();

function drawFaceWire() {
  const c = document.getElementById('facewire');
  if (!c) return;
  const on = !!state.faceTrack;
  if (c.hidden !== !on) c.hidden = !on;
  if (!on) return;
  const g = c.getContext('2d');
  const W = c.width;
  const H = c.height;
  g.clearRect(0, 0, W, H);
  const f = state.face;
  const tracking = state.faceTrack.stats && state.faceTrack.stats.tracking;
  const cy = Math.cos(f.yaw); const sy = Math.sin(f.yaw);
  const cp = Math.cos(f.pitch); const sp = Math.sin(f.pitch);
  const cr = Math.cos(f.roll); const sr = Math.sin(f.roll);
  // engine conventions: yaw+ = turned left, pitch+ = up, roll+ = right
  // ear down; drawn mirror-style (your left = screen left)
  const rot = ([x, y, z]) => {
    let a = x * cr - y * sr; let b = x * sr + y * cr; // roll about z
    let d = b * cp + z * sp; let e = -b * sp + z * cp; // pitch about x
    return [a * cy + e * sy, d, -a * sy + e * cy]; // yaw about y
  };
  const s = (1 + (f.dz || 0) * 1.5) * W * 0.3;
  const proj = ([x, y, z]) => [
    W / 2 + (f.dx || 0) * W * 0.8 + x * s,
    H * 0.54 - (f.dy || 0) * H * 0.8 - y * s,
    z,
  ];
  for (const line of FACE_WIRE) {
    const pts = line.map((p) => proj(rot(p)));
    for (let i = 1; i < pts.length; i++) {
      // depth cue: segments facing the camera draw brighter
      const zm = (pts[i][2] + pts[i - 1][2]) / 2;
      const a = Math.max(0.10, 0.30 + zm * 0.45) * (tracking ? 1 : 0.35);
      g.strokeStyle = tracking ? `rgba(221,214,201,${a})` : `rgba(179,87,74,${a})`;
      g.lineWidth = zm > 0.4 ? 1.6 : 1;
      g.beginPath();
      g.moveTo(pts[i - 1][0], pts[i - 1][1]);
      g.lineTo(pts[i][0], pts[i][1]);
      g.stroke();
    }
  }
  g.fillStyle = '#55524a';
  g.font = '10px IBM Plex Mono, monospace';
  g.fillText(tracking ? 'head' : 'no face', 8, 14);
}

function drawMeters() {
  updateMixerMeters();
  const c = document.getElementById('meters');
  if (!c) return;
  const g = c.getContext('2d');
  const W = c.width;
  const H = c.height;
  g.clearRect(0, 0, W, H);
  g.fillStyle = '#0e1116cc';
  g.fillRect(0, 0, W, H);
  const db = (x) => Math.max(-60, 20 * Math.log10(x + 1e-6));
  const bar = (x0, val, color, label) => {
    const h = ((db(val) + 60) / 60) * (H - 56);
    g.fillStyle = '#1c2430';
    g.fillRect(x0, 28, 26, H - 56);
    g.fillStyle = color;
    g.fillRect(x0, 28 + (H - 56) - h, 26, h);
    g.fillStyle = '#7a8496';
    g.font = '13px sans-serif';
    g.textAlign = 'center';
    g.fillText(label, x0 + 13, H - 12);
  };
  bar(12, state.meters.l, '#50dcff', 'L');
  bar(48, state.meters.r, '#50dcff', 'R');
  // AGC ("ear gain") history: dB line, −24…+14
  g.strokeStyle = '#ffaa3c';
  g.lineWidth = 2;
  g.beginPath();
  const hist = state.meters.hist;
  for (let i = 0; i < hist.length; i++) {
    const gx = 86 + (i / 220) * (W - 98);
    const gdb = 20 * Math.log10(hist[i]);
    const gy = 28 + (1 - (gdb + 24) / 38) * (H - 56);
    i ? g.lineTo(gx, gy) : g.moveTo(gx, gy);
  }
  g.stroke();
  g.fillStyle = '#ffaa3c';
  g.textAlign = 'left';
  g.fillText(`ear ${(20 * Math.log10(state.meters.agc)).toFixed(1)} dB`, 86, 18);
  if (state.meters.tts > 0.02) {
    // hearing fatigue after ultra-loud exposure (muffle depth)
    g.fillStyle = '#ff6a5a';
    g.fillText(`muffled ${Math.round(state.meters.tts * 100)}%`, 200, 18);
  }
  {
    // ambience field meter: channel level + the loudest inlets — so a
    // misbehaving burst is measurable on sight, not by ear-memory
    const c = state.chanHist && state.chanHist.length
      ? state.chanHist[state.chanHist.length - 1] : null;
    const st = state.simState;
    if (c && st && state.envOff) {
      const eo = state.envOff;
      const db = 20 * Math.log10(c[21] + 1e-6);
      const n = Math.min(st[eo + 5] | 0, 2);
      let line = `amb ${db.toFixed(0)}dB`;
      for (let i = 0; i < n; i++) {
        const o = eo + 6 + i * 8;
        line += ` ${st[o] | 0}:${st[o + 5].toFixed(2)}`;
      }
      line += ` seep:${st[eo + 1].toFixed(2)}`;
      g.fillStyle = '#8fb8d8';
      g.fillText(line, 86, H - 26);
    }
  }
  if (state.meters.pts) {
    // adaptive point-HRTF budget (per source), set by the worklet from load
    g.fillStyle = '#7a8496';
    g.fillText(`hrtf ×${state.meters.pts}`, 86, H - 12);
  }
  // 0 dB reference line
  g.strokeStyle = '#3a4658';
  g.lineWidth = 1;
  const y0 = 28 + (1 - 24 / 38) * (H - 56);
  g.beginPath();
  g.moveTo(86, y0);
  g.lineTo(W - 12, y0);
  g.stroke();
}

// Inferno colormap (perceptually uniform: black → purple → crimson →
// orange → yellow), piecewise-linear between matplotlib anchor stops.
const SPEC_LUT = (() => {
  const stops = [
    [0.0, 0, 0, 4], [0.125, 22, 11, 57], [0.25, 66, 10, 104],
    [0.375, 106, 23, 110], [0.5, 147, 38, 103], [0.625, 188, 55, 84],
    [0.75, 221, 81, 58], [0.875, 243, 120, 25], [0.9375, 249, 189, 38],
    [1.0, 252, 255, 164],
  ];
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (stops[k + 1][0] < t) k++;
    const u = (t - stops[k][0]) / (stops[k + 1][0] - stops[k][0]);
    for (let ch = 0; ch < 3; ch++) {
      lut[i * 3 + ch] = Math.round(stops[k][ch + 1] + u * (stops[k + 1][ch + 1] - stops[k][ch + 1]));
    }
  }
  return lut;
})();

function drawSpec() {
  const c = document.getElementById('spec');
  if (!c || !state.analyser) return;
  const g = c.getContext('2d');
  const W = c.width;
  const H = c.height;
  state.analyser.getByteFrequencyData(state.specBins);
  // scroll left, draw the new column at the right edge
  g.drawImage(c, -2, 0);
  const bins = state.specBins;
  const fs = 48000;
  const nyq = fs / 2;
  if (!state.specCol) state.specCol = g.createImageData(2, H);
  const px = state.specCol.data;
  for (let y = 0; y < H; y++) {
    // log frequency axis: 40 Hz (bottom) … 16 kHz (top)
    const frac = 1 - y / H;
    const f = 40 * Math.pow(16000 / 40, frac);
    const bin = Math.min(bins.length - 1, Math.round((f / nyq) * bins.length));
    // analyser bytes are already dB-scaled; mild gamma keeps the floor dark
    const li = Math.round(Math.pow(bins[bin] / 255, 1.25) * 255) * 3;
    for (let x = 0; x < 2; x++) {
      const o = (y * 2 + x) * 4;
      px[o] = SPEC_LUT[li];
      px[o + 1] = SPEC_LUT[li + 1];
      px[o + 2] = SPEC_LUT[li + 2];
      px[o + 3] = 255;
    }
  }
  g.putImageData(state.specCol, W - 2, 0);
  // frequency gridlines: 100 Hz, 1 kHz, 10 kHz
  g.fillStyle = '#7a849644';
  for (const f of [100, 1000, 10000]) {
    const y = H * (1 - Math.log(f / 40) / Math.log(16000 / 40));
    g.fillRect(0, y, W, 1);
  }
}

// ------------------------------------------------------------ sound rays
// Debug-mode 3D view of the sim's active propagation routes: the state
// buffer carries ≤4 route points per source (portal hops, bends); a
// glowing line runs source → hops → ears, colored per source.
const rayLines = [];
function drawSoundRays() {
  if (!state.simState) return;
  const show = !!state.debug.on;
  for (let i = 0; i < 11; i++) {
    if (!rayLines[i]) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6 * 3), 3));
      const color = parseInt((DBG_COLORS[i] || '#ffffff').slice(1), 16);
      rayLines[i] = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.65, depthTest: false,
      }));
      rayLines[i].renderOrder = 9;
      scene.add(rayLines[i]);
    }
    const line = rayLines[i];
    const off = 4 + i * 9;
    const n = Math.min(4, state.simState[off] | 0);
    line.visible = show && n > 0;
    if (!line.visible) continue;
    const pos = line.geometry.attributes.position;
    const ez = EYE + (state.pose.z || 0) + (state.airZ || 0) - 0.7 * (state.crouch || 0);
    let k = 0;
    for (let p = 0; p < n; p++) {
      const x = state.simState[off + 1 + p * 2];
      const y = state.simState[off + 2 + p * 2];
      const z = 1.5 + (ez - 1.5) * (p / Math.max(1, n));
      pos.setXYZ(k++, x, z, -y);
    }
    pos.setXYZ(k++, state.pose.x, ez, -state.pose.y);
    line.geometry.setDrawRange(0, k);
    pos.needsUpdate = true;
    line.geometry.computeBoundingSphere();
  }
}

function drawMinimap() {
  const W = 300;
  const H = 480;
  mm.clearRect(0, 0, W, H);
  mm.fillStyle = '#0e1116cc';
  mm.fillRect(0, 0, W, H);
  const xs = ROOMS.flatMap((r) => [r.min[0], r.max[0]]);
  const ys = ROOMS.flatMap((r) => [r.min[1], r.max[1]]);
  // map the interesting region, not the 500 m road corridor
  const [mx0, mx1, my0, my1] = [Math.max(Math.min(...xs), -28), Math.min(Math.max(...xs), 48),
    Math.max(Math.min(...ys), -32), Math.min(Math.max(...ys), 96)];
  const s = Math.min(W / (mx1 - mx0 + 2), H / (my1 - my0 + 2));
  const sx = (wx) => (wx - mx0 + 1) * s;
  const sy = (wy) => H - (wy - my0 + 1) * s;
  const room = state.simState ? state.simState[2] | 0 : -1;

  ROOMS.forEach((r, i) => {
    if (r.upper && i !== room) return;
    if (r.solid) {
      mm.fillStyle = '#3a4658';
      mm.fillRect(sx(r.min[0]), sy(r.max[1]),
                  (r.max[0] - r.min[0]) * s, (r.max[1] - r.min[1]) * s);
      return;
    }
    mm.strokeStyle = i === room ? '#78c8ff' : '#5a6980';
    mm.lineWidth = i === room ? 3 : 1.5;
    mm.strokeRect(sx(r.min[0]), sy(r.max[1]),
                  (r.max[0] - r.min[0]) * s, (r.max[1] - r.min[1]) * s);
  });
  DOORS.forEach(({ pos: [dx, dy], axis }) => {
    mm.strokeStyle = '#0e1116';
    mm.lineWidth = 5;
    mm.beginPath();
    if (axis === 1) {
      mm.moveTo(sx(dx - 0.55), sy(dy));
      mm.lineTo(sx(dx + 0.55), sy(dy));
    } else {
      mm.moveTo(sx(dx), sy(dy - 0.55));
      mm.lineTo(sx(dx), sy(dy + 0.55));
    }
    mm.stroke();
  });
  WINDOWS.forEach(({ pos: [wx, wy], axis, hw }) => {
    mm.strokeStyle = '#7ac8ff';
    mm.lineWidth = 3;
    mm.beginPath();
    if (axis === 1) {
      mm.moveTo(sx(wx - hw), sy(wy));
      mm.lineTo(sx(wx + hw), sy(wy));
    } else {
      mm.moveTo(sx(wx), sy(wy - hw));
      mm.lineTo(sx(wx), sy(wy + hw));
    }
    mm.stroke();
  });
  SOURCES.forEach((src) => {
    mm.fillStyle = '#' + src.color.toString(16).padStart(6, '0');
    mm.beginPath();
    mm.arc(sx(src.pos[0]), sy(src.pos[1]), 4, 0, Math.PI * 2);
    mm.fill();
  });
  const lx = sx(state.pose.x);
  const ly = sy(state.pose.y);
  const a = -state.heading;
  mm.fillStyle = '#50dcff';
  mm.beginPath();
  mm.moveTo(lx + 9 * Math.cos(a), ly + 9 * Math.sin(a));
  mm.lineTo(lx + 6 * Math.cos(a + 2.5), ly + 6 * Math.sin(a + 2.5));
  mm.lineTo(lx + 6 * Math.cos(a - 2.5), ly + 6 * Math.sin(a - 2.5));
  mm.closePath();
  mm.fill();
}
