#!/usr/bin/env node
// Source-scaling benchmark: how many sources can be FULLY simulated and
// rendered inside the real-time budget, and where each source's cost goes.
// Cross-platform (macOS / Windows / Linux), Node ≥ 18, no dependencies.
//
//   node tools/bench_sources.mjs            full run (~2-4 min)
//   node tools/bench_sources.mjs --quick    shorter windows, fewer sim ticks
//   node tools/bench_sources.mjs --engine   audio side only
//   node tools/bench_sources.mjs --sim      sim side only
//   node tools/bench_sources.mjs --json out.json   also dump raw results
//   node tools/bench_sources.mjs --pos 18,25       a chosen spot instead of the heaviest
//
// Drives the same wasm (web/omg_web.wasm) the demo ships, the way
// worker.js (sim) and worklet.js (engine) do. Run `sh tools/build_web.sh`
// first if the Rust changed.
//
// AUDIO side: one engine instance (one AudioWorklet), N sources (N past
// the demo's 14 only works since the engine grew its mixer per source —
// see eng_source_alloc), each with its own test signal and a REAL sim
// ParamBlock: the 14 demo sources' blocks at the heaviest listener
// position, cycled to N (tap keys repeat across renderers; each renderer
// is independent). Blocks are re-applied every 50 ms of audio like the
// 20 Hz worker stream, so set_params cost is included. "rtf" = audio
// seconds rendered per wall second on ONE thread; the AudioWorklet must
// stay ≥ 1, and ≥ 1.5 is the margin we call safe (GC, other tabs, the
// browser's own audio graph, thermal throttling).
//
// SIM side: wall ms per sim_tick at the same position, per quality tier,
// and vs. number of ACTIVE sources via the mixer kill switch
// (sim_set_mute). Two backends: "cpu" (ISM early + CPU-traced late field,
// the no-WebGPU fallback) and "gpu-proxy" (late traces queued for WebGPU
// and answered with a synthetic echogram — the worker's CPU share when a
// GPU carries the late field). Real GPU dispatch time only exists in the
// browser: read it from the page's quality panel ("gpu ms").
import { readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const has = (f) => argv.includes(f);
const QUICK = has('--quick');
const DO_ENGINE = !has('--sim');
const DO_SIM = !has('--engine');
const jsonIdx = argv.indexOf('--json');
const JSON_OUT = jsonIdx >= 0 ? argv[jsonIdx + 1] : null;
const posIdx = argv.indexOf('--pos'); // --pos x,y: use this spot instead of the heaviest
const POS = posIdx >= 0 ? argv[posIdx + 1].split(',').map(Number) : null;
const TTY = process.stdout.isTTY;

const FS = 48000;
const BLOCK = 128; // AudioWorklet render quantum
const TICK_HZ = 20; // worker.js tick rate
const SIG_LEN = FS; // 1 s loop per source
const NSRC_DEMO = 14;
const NS = [1, 2, 4, 8, 14, 24, 32, 48, 64, 96, 128];
const SAFE_RTF = 1.5;
const STOP_RTF = 0.5;
const FLAT_HEADER = 24;
const FLAT_PER_TAP = 9;
const SRC_NAMES = ['music', 'voice', 'club', 'flute', 'radio', 'owl·cath', 'owl·house', 'bells',
  'ball0', 'ball1', 'ball2', 'car0', 'car1', 'feet'];

const wasmBytes = readFileSync(join(root, 'web/omg_web.wasm'));
const hrirGrid = readFileSync(join(root, 'assets/hrir_grid.bin'));
const hrirSpk = readFileSync(join(root, 'assets/hrir_dodeca20.bin'));
const wasmMod = await WebAssembly.compile(wasmBytes);
const instantiate = async () => (await WebAssembly.instantiate(wasmMod, {})).exports;

const results = { machine: {}, sim: {}, engine: {} };
const cpus = os.cpus();
results.machine = {
  os: `${os.platform()} ${os.release()} ${os.arch()}`,
  cpu: cpus.length ? cpus[0].model.trim() : 'unknown',
  threads: cpus.length,
  node: process.version,
  mem_gb: +(os.totalmem() / 2 ** 30).toFixed(1),
};
const M = results.machine;
console.log(`machine: ${M.os} | ${M.cpu} (${M.threads} threads) | node ${M.node} | ${M.mem_gb} GB`);
console.log(`wasm: web/omg_web.wasm (${wasmBytes.length} bytes)${QUICK ? ' | --quick' : ''}`);

const pad = (s, n) => String(s).padStart(n);
const padr = (s, n) => String(s).padEnd(n);
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const pct = (xs, p) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
// least squares y = a + b·x
const linfit = (xs, ys) => {
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const b = sxx > 0 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
};

// =================================================================== SIM
// The demo's scene, set up like main.js + worker.js: doors open except
// 4 and 6 (the page's initial state), all six dynamic slots live around
// the listener (three thrown balls, two cars on the west street, feet).
const sim = await instantiate();
sim.sim_setup();
sim.sim_set_quality(0); // high: worst case, and where auto settles on a fast desktop

const setScene = (lx, ly) => {
  const doors = new Float32Array(sim.memory.buffer, sim.sim_door_ptr(), 16);
  for (let i = 0; i < 8; i++) doors[i] = i !== 4 && i !== 6 ? 1 : 0;
  const dyn = new Float32Array(sim.memory.buffer, sim.sim_dyn_ptr(), 24);
  dyn.set([
    lx + 1.5, ly + 1.0, 1.2, 1,  // ball0
    lx - 1.2, ly + 2.0, 1.5, 1,  // ball1
    lx + 0.5, ly - 1.8, 0.8, 1,  // ball2
    -18.9, ly + 4.0, 1.5, 1,     // car0 (southbound lane)
    -17.1, ly - 6.0, 1.5, 1,     // car1 (northbound lane)
    lx, ly, 0.2, 1,              // feet
  ]);
};
// GPU-offload proxy: with sim_gpu_enable + sim_wlate_enable the sim
// queues its stochastic late traces (per-room and world) for WebGPU
// instead of tracing on the CPU. Node has no WebGPU, so after each tick
// we drain the job queues and inject a synthetic decaying echogram per
// job id — the sim consumes it exactly like a real GPU result. What
// sim_tick then costs is the worker's CPU share when the GPU carries
// the late field (the GPU time itself is not modelled).
let gpuStub = false;
const ECHO = new Float32Array(1800);
for (let b = 0; b < 300; b++) for (let k = 0; k < 3; k++) ECHO[b * 3 + k] = 1e-3 * Math.exp(-b / (30 + 15 * k));
const GPU_JOB_F32S = 57;
const WLATE_JOB_F32S = 10 + 64 * 16;
const pumpStub = () => {
  for (const [lenFn, ptrFn, stride] of [
    [sim.sim_gpu_jobs_len, sim.sim_gpu_jobs_ptr, GPU_JOB_F32S],
    [sim.sim_wlate_jobs_len, sim.sim_wlate_jobs_ptr, WLATE_JOB_F32S],
  ]) {
    const n = lenFn();
    if (!n) continue;
    const jobs = new Float32Array(sim.memory.buffer, ptrFn(), n).slice();
    for (let o = 0; o < n; o += stride) {
      new Float32Array(sim.memory.buffer, sim.sim_gpu_buf_ptr(), ECHO.length).set(ECHO);
      sim.sim_gpu_inject(jobs[o]);
    }
  }
  sim.sim_pt_jobs_len(); // drained, not answered: ISM mode seeds the early field
};
const tick = (lx, ly) => {
  setScene(lx, ly);
  const t0 = performance.now();
  sim.sim_tick(lx, ly, 1.6, 0.0);
  const ms = performance.now() - t0;
  if (gpuStub) pumpStub();
  return ms;
};
const readBlocks = () => {
  const out = [];
  for (let i = 0; i < NSRC_DEMO; i++) {
    const len = sim.sim_params_len(i);
    out.push(new Float32Array(sim.memory.buffer, sim.sim_params_ptr(i), len).slice());
  }
  return out;
};
const blockStats = (b) => ({
  taps: b.length >= FLAT_HEADER ? b[1] : 0,
  level: b.length >= FLAT_HEADER ? Math.max(b[5], b[6], b[7]) : 0,
  remote: b.length >= FLAT_HEADER && b[8] > 0.5,
});

// heaviest position = most incoming taps over all 14 sources
const candidates = [
  [4, 3], [4, 10], [7, 19], [21, 31], [27, 32], [22.5, 31], [21, 27.2],
  [27.5, 19.5], [18, 25], [33, 39], [10, 8],
];
let heavy = { taps: -1, pos: [0, 0] };
const seen = [];
for (const [x, y] of candidates) {
  for (let t = 0; t < 6; t++) tick(x, y);
  const taps = readBlocks().reduce((s, b) => s + blockStats(b).taps, 0);
  seen.push(`(${x},${y}) ${taps}`);
  if (taps > heavy.taps) heavy = { taps, pos: [x, y] };
}
console.log(`\ntotal incoming taps per candidate spot (14 sources, dyn slots around the listener):\n  ${seen.join(' | ')}`);
if (POS) heavy = { taps: 0, pos: POS };
const [HX, HY] = heavy.pos;
for (let t = 0; t < 10; t++) tick(HX, HY);
const heavyBlocks = readBlocks();
const stats = heavyBlocks.map(blockStats);
heavy.taps = stats.reduce((s, b) => s + b.taps, 0);
console.log(`\n${POS ? 'chosen' : 'heaviest'} listener position: (${HX}, ${HY}), sim tier high, ${heavy.taps} incoming taps over ${NSRC_DEMO} sources`);
console.log('  src        taps  rev.level  remote');
heavyBlocks.forEach((b, i) => {
  const s = stats[i];
  console.log(`  ${padr(SRC_NAMES[i] || i, 10)} ${pad(s.taps, 4)}  ${pad(s.level.toFixed(3), 9)}  ${s.remote ? 'yes' : '-'}`);
});
const active = heavyBlocks.filter((b, i) => stats[i].taps > 0 || stats[i].level > 0 || stats[i].remote);
console.log(`  active ${active.length}/${NSRC_DEMO}; mean ${(heavy.taps / NSRC_DEMO).toFixed(1)} taps/source`);
results.sim.heavy = { pos: heavy.pos, taps: heavy.taps, per_source: stats };

if (DO_SIM) {
  console.log('\n=== SIM: sim_tick wall ms (20 Hz budget = 50 ms, governor sheds at 40) ===');
  const WARM = QUICK ? 8 : 15;
  const NT = QUICK ? 40 : 100;
  // walking: a 0.4 m circle at ~1.4 m/s around the heavy spot — a static
  // listener lets the sim's LOD replay cached blocks, which flatters it
  let phase = 0;
  const walkPos = () => { phase += 0.07 / 0.4; return [HX + 0.4 * Math.cos(phase), HY + 0.4 * Math.sin(phase)]; };
  const run = (n, walk) => {
    const ms = [];
    for (let t = 0; t < n; t++) {
      const [x, y] = walk ? walkPos() : [HX, HY];
      ms.push(tick(x, y));
    }
    return ms;
  };
  const summary = (ms) => ({ mean: mean(ms), p50: pct(ms, 0.5), p95: pct(ms, 0.95), max: Math.max(...ms) });
  console.log('backend    tier  listener   mean    p50    p95    max');
  results.sim.tiers = [];
  const tierTable = (backend, walks) => {
    for (const tier of [0, 1, 2]) {
      sim.sim_set_quality(tier);
      for (const walk of walks) {
        run(WARM, walk);
        const s = summary(run(NT, walk));
        results.sim.tiers.push({ backend, tier, walk, ...s });
        console.log(`${padr(backend, 10)} ${padr(['high', 'med', 'low'][tier], 5)} ${padr(walk ? 'walking' : 'static', 8)} ${pad(s.mean.toFixed(2), 6)} ${pad(s.p50.toFixed(2), 6)} ${pad(s.p95.toFixed(2), 6)} ${pad(s.max.toFixed(2), 6)}`);
      }
    }
  };
  tierTable('cpu', [false, true]);
  gpuStub = true;
  sim.sim_gpu_enable();
  sim.sim_wlate_enable();
  tierTable('gpu-proxy', [true]);
  sim.sim_gpu_disable();
  gpuStub = false;

  // per-source cost via the kill switch (tier high, walking)
  const perSource = (backend) => {
    sim.sim_set_quality(0);
    const muteAll = (on) => { for (let i = 0; i < NSRC_DEMO; i++) sim.sim_set_mute(i, on ? 1 : 0); };
    const NS_T = QUICK ? 20 : 40;
    const measureSet = (activeSet) => {
      muteAll(true);
      for (const i of activeSet) sim.sim_set_mute(i, 0);
      run(WARM, true);
      return mean(run(NS_T, true));
  };
  const base = measureSet([]);
  console.log(`\n[${backend}] per-source sim cost (tier high, walking; each source ALONE minus all-muted ${base.toFixed(2)} ms):`);
  const alone = [];
  for (let i = 0; i < NSRC_DEMO; i++) alone.push(measureSet([i]) - base);
  console.log('  ' + SRC_NAMES.map((n) => pad(n, 10)).join(''));
  console.log('  ' + alone.map((d) => pad(d.toFixed(2), 10)).join(''));
  // cumulative: unmute in order, fit ms = a + b·k
  const ks = [], ys = [];
  const order = [...Array(NSRC_DEMO).keys()];
  for (let k = 0; k <= NSRC_DEMO; k += k < 4 ? 1 : 2) {
    ks.push(k);
    ys.push(measureSet(order.slice(0, k)));
  }
  if (ks[ks.length - 1] !== NSRC_DEMO) { ks.push(NSRC_DEMO); ys.push(measureSet(order)); }
  muteAll(false);
  const fit = linfit(ks, ys);
  console.log(`[${backend}] sim tick vs active sources (tier high, walking):`);
  console.log('  active ' + ks.map((k) => pad(k, 6)).join(''));
  console.log('  ms     ' + ys.map((y) => pad(y.toFixed(2), 6)).join(''));
  console.log(`  fit: ${fit.a.toFixed(2)} ms fixed + ${fit.b.toFixed(2)} ms/active source`
    + ` → ~${Math.floor((40 - fit.a) / Math.max(fit.b, 1e-6))} sources before the 40 ms shed line`);
  results.sim[backend] = { alone, ks, ms: ys, fixed_ms: fit.a, ms_per_source: fit.b };
  };
  perSource('cpu');
  gpuStub = true;
  sim.sim_gpu_enable();
  sim.sim_wlate_enable();
  perSource('gpu-proxy');
  sim.sim_gpu_disable();
  gpuStub = false;
  console.log('  (gpu-proxy = worker CPU only; real GPU dispatch time shows as "gpu ms" in the page\'s quality panel)');
  sim.sim_set_quality(0);
}

// ================================================================ ENGINE
if (DO_ENGINE) {
  // distinct, never-silent test signals (FDNs and taps never idle)
  const signal = (i) => {
    const out = new Float32Array(SIG_LEN);
    let s = (0x9E3779B9 * (i + 1)) >>> 0 || 1;
    let lp = 0;
    const f = 70 * Math.pow(1.13, i % 40);
    for (let k = 0; k < SIG_LEN; k++) {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      lp += 0.15 * ((s / 4294967296) * 2 - 1 - lp);
      out[k] = 0.25 * lp + 0.2 * Math.sin(2 * Math.PI * f * k / FS);
    }
    return out;
  };

  const makeEngine = async (points, ceiling) => {
    const w = await instantiate();
    w.eng_init(FS);
    const put = (bytes, alloc, done) => {
      const p = alloc(bytes.byteLength);
      new Uint8Array(w.memory.buffer, p, bytes.byteLength).set(bytes);
      done();
    };
    put(hrirGrid, w.eng_hrir_grid_alloc, w.eng_hrir_grid_done);
    put(hrirSpk, w.eng_hrir_speakers_alloc, w.eng_hrir_speakers_done);
    w.eng_set_point_budget(points);
    w.eng_set_tap_ceiling(ceiling);
    return { w, n: 0 };
  };
  const growTo = (e, n) => {
    while (e.n < n) {
      const i = e.n++;
      const p = e.w.eng_source_alloc(i, SIG_LEN);
      new Float32Array(e.w.memory.buffer, p, SIG_LEN).set(signal(i));
      e.w.eng_source_commit(i);
    }
  };
  const apply = (e, blocks) => {
    for (let i = 0; i < e.n; i++) {
      const b = blocks[i % blocks.length];
      new Float32Array(e.w.memory.buffer, e.w.eng_param_buf_ptr(), b.length).set(b);
      e.w.eng_set_params(i, b.length);
    }
  };
  const PER_TICK = FS / TICK_HZ / BLOCK; // 18.75 quanta per sim tick
  // render `sec` of audio, re-applying params at 20 Hz; returns wall s
  const render = (e, blocks, sec) => {
    const nb = Math.max(1, Math.round(sec * FS / BLOCK));
    let next = 0;
    const t0 = performance.now();
    for (let b = 0; b < nb; b++) {
      if (b >= next) { apply(e, blocks); next += PER_TICK; }
      e.w.eng_process(BLOCK);
    }
    return { audio: nb * BLOCK / FS, wall: (performance.now() - t0) / 1000 };
  };
  const WIN_WALL = QUICK ? 0.25 : 0.5; // target wall seconds per window
  const measure = (e, blocks) => {
    apply(e, blocks);
    render(e, blocks, QUICK ? 0.4 : 0.8); // settle: fades, FDN fill, HRIR xfades
    let est = render(e, blocks, 0.2);
    let rtfEst = est.audio / est.wall;
    const rtfs = [];
    for (let k = 0; k < 3; k++) {
      const sec = Math.min(6, Math.max(0.2, WIN_WALL * rtfEst));
      const r = render(e, blocks, sec);
      rtfs.push(r.audio / r.wall);
      rtfEst = rtfs[rtfs.length - 1];
    }
    return median(rtfs);
  };
  const occupancy = (e) => {
    const n = Math.min(e.n, NSRC_DEMO);
    const d = new Float32Array(e.w.memory.buffer, e.w.eng_debug_render(), NSRC_DEMO * 5);
    let live = 0, pts = 0;
    for (let i = 0; i < n; i++) { live += d[i * 5]; pts += d[i * 5 + 1]; }
    return { live: live / n, points: pts / n };
  };
  // CPU fraction of one core per audio second = 1/rtf
  const load = (rtf) => 1 / rtf;

  console.log(`\n=== ENGINE: realtime factor vs sources (single thread, ${BLOCK}-sample quanta, params @${TICK_HZ} Hz) ===`);
  console.log(`(blocks: the ${NSRC_DEMO} demo blocks at (${HX}, ${HY}) cycled to N; rtf ≥ ${SAFE_RTF} = safe, sweep stops below ${STOP_RTF})`);

  // fixed cost: output stage (20-speaker HRIR decode + AGC) with no sources
  const e0 = await makeEngine(24, 160);
  const rtf0 = measure(e0, heavyBlocks);
  console.log(`fixed (0 sources: 20-speaker decode + output stage; ambience/rain not loaded): rtf ${rtf0.toFixed(1)}x = ${(100 * load(rtf0)).toFixed(1)}% core`);
  results.engine.fixed_rtf = rtf0;

  const configs = [
    { id: 'a', label: 'pb24 c160 default', points: 24, ceiling: 160 },
    { id: 'b', label: 'pb0 c160 bus-only', points: 0, ceiling: 160 },
    { id: 'c', label: 'pb4 c32 shed', points: 4, ceiling: 32 },
    { id: 'd', label: 'pb1 c160', points: 1, ceiling: 160 },
  ];
  const table = {};
  for (const c of configs) {
    const e = await makeEngine(c.points, c.ceiling);
    table[c.id] = [];
    for (const n of NS) {
      growTo(e, n);
      const rtf = measure(e, heavyBlocks);
      table[c.id].push({ n, rtf });
      if (TTY) process.stdout.write(`  (${c.id}) N=${n} rtf ${rtf.toFixed(2)}   \r`);
      if (rtf < STOP_RTF) break;
    }
    if (TTY) process.stdout.write(' '.repeat(40) + '\r');
  }
  results.engine.sweep = table;

  console.log('\n    N' + configs.map((c) => pad(`(${c.id}) ${c.label}`, 24)).join(''));
  console.log('     (pb = point budget, c = tap ceiling)');
  for (const n of NS) {
    const cells = configs.map((c) => {
      const r = table[c.id].find((x) => x.n === n);
      return pad(r ? `${r.rtf.toFixed(2)}x${r.rtf < SAFE_RTF ? (r.rtf < 1 ? ' !!' : ' ! ') : '   '}` : '-   ', 24);
    });
    console.log(`${pad(n, 5)}${cells.join('')}`);
  }
  console.log(`  ("!" = below ${SAFE_RTF}x safe margin, "!!" = below realtime)`);

  console.log(`\nmax sources at rtf ≥ ${SAFE_RTF} (measured / linear-fit estimate), per-source load from the fit:`);
  results.engine.max_safe = {};
  for (const c of configs) {
    const rows = table[c.id];
    const ok = rows.filter((r) => r.rtf >= SAFE_RTF).map((r) => r.n);
    const measured = ok.length ? Math.max(...ok) : 0;
    // load(N) = fixed + per·N  (fit on N ≥ 4 to skip cache-warm small-N noise)
    const fr = rows.filter((r) => r.n >= 4);
    const fit = linfit(fr.map((r) => r.n), fr.map((r) => load(r.rtf)));
    const est = Math.floor((1 / SAFE_RTF - fit.a) / fit.b);
    const est1 = Math.floor((1 - fit.a) / fit.b);
    results.engine.max_safe[c.id] = { measured, est, est_realtime: est1, per_source_core: fit.b, fixed_core: fit.a };
    console.log(`  (${c.id}) ${padr(c.label, 18)} measured ${pad(measured, 3)} | fit ${pad(est, 3)} (realtime 1.0x: ${pad(est1, 3)}) | ${(100 * fit.b).toFixed(2)}% core/source, ${(1000 * fit.b * BLOCK / FS * 1000).toFixed(1)} µs/source/quantum`);
  }

  // ---------------------------------------------------- cost breakdown
  // Same N as the demo; each variant zeroes parts of the real blocks so
  // the renderer's sleep paths skip them (0 taps → no tap loop, level 0
  // → own FDN idles, remote flag 0 → remote FDN idles), then differences.
  const NB = NSRC_DEMO;
  const strip = (b, { taps = true, level = true, remote = true }) => {
    const n = taps ? b.length : FLAT_HEADER;
    const o = b.slice(0, n);
    if (!taps) o[1] = 0;
    if (!level) { o[5] = 0; o[6] = 0; o[7] = 0; }
    if (!remote) o[8] = 0;
    return o;
  };
  const variant = async (points, ceiling, opts) => {
    const e = await makeEngine(points, ceiling);
    growTo(e, NB);
    const blocks = heavyBlocks.map((b) => strip(b, opts));
    const rtf = measure(e, blocks);
    return { rtf, occ: occupancy(e) };
  };
  console.log(`\n=== ENGINE: per-source cost split at N=${NB} (the demo's ${NSRC_DEMO} blocks, ${(100 / rtf0).toFixed(1)}% core fixed subtracted) ===`);
  const V = {
    empty: await variant(24, 160, { taps: false, level: false, remote: false }),
    own: await variant(24, 160, { taps: false, level: true, remote: false }),
    remote: await variant(24, 160, { taps: false, level: false, remote: true }),
    bus: await variant(0, 160, { taps: true, level: false, remote: false }),
    pts: await variant(24, 160, { taps: true, level: false, remote: false }),
    full: await variant(24, 160, { taps: true, level: true, remote: true }),
    full32: await variant(4, 32, { taps: true, level: true, remote: true }),
    bus32: await variant(0, 32, { taps: true, level: false, remote: false }),
  };
  // per-source % of one core
  const per = (v) => 100 * (load(v.rtf) - load(rtf0)) / NB;
  const nRemote = stats.filter((s) => s.remote).length;
  const nLevel = stats.filter((s) => s.level > 0).length;
  const rows = [
    ['base (0 taps, FDNs asleep)', per(V.empty), 'per-sample scan of all 448 tap slots + smoothers'],
    ['own FDN (late field)', per(V.own) - per(V.empty), `${nLevel}/${NB} sources send`],
    ['remote FDN (coupled room)', per(V.remote) - per(V.empty), `${nRemote}/${NB} sources send`],
    ['taps on the SH bus (pb0, ceil160)', per(V.bus) - per(V.empty), `${V.bus.occ.live.toFixed(1)} live taps/src`],
    ['point HRIR convs (pb24 on top)', per(V.pts) - per(V.bus), `${V.pts.occ.points.toFixed(1)} points/src`],
  ];
  const sum = rows.reduce((s, r) => s + r[1], 0);
  console.log('  component                                   % core/source   note');
  for (const [name, v, note] of rows) console.log(`  ${padr(name, 44)} ${pad(v.toFixed(2), 8)}       ${note}`);
  console.log(`  ${padr('sum of parts', 44)} ${pad(sum.toFixed(2), 8)}`);
  console.log(`  ${padr('measured full default (pb24, ceil160)', 44)} ${pad(per(V.full).toFixed(2), 8)}       rtf ${V.full.rtf.toFixed(2)}x at N=${NB}`);
  console.log(`  ${padr('measured shed (pb4, ceil32)', 44)} ${pad(per(V.full32).toFixed(2), 8)}       rtf ${V.full32.rtf.toFixed(2)}x, ${V.full32.occ.live.toFixed(1)} live taps/src`);
  const busTap = (per(V.bus) - per(V.empty)) / Math.max(V.bus.occ.live, 1e-6);
  const ptTap = (per(V.pts) - per(V.bus)) / Math.max(V.pts.occ.points, 1e-6);
  const fdnEach = (per(V.own) - per(V.empty)) * NB / Math.max(nLevel, 1);
  const remEach = (per(V.remote) - per(V.empty)) * NB / Math.max(nRemote, 1);
  console.log(`  unit costs (% core): bus tap ${busTap.toFixed(4)} | point conv ${ptTap.toFixed(4)} (= ${(ptTap / Math.max(busTap, 1e-9)).toFixed(1)} bus taps)`
    + ` | own FDN ${fdnEach.toFixed(3)} | remote FDN ${remEach.toFixed(3)} (per sending source)`);
  console.log(`  bus taps at ceil32: ${V.bus32.occ.live.toFixed(1)} live/src, ${(per(V.bus32) - per(V.empty)).toFixed(2)}% core/src`);
  results.engine.split = {
    n: NB, rtf: Object.fromEntries(Object.entries(V).map(([k, v]) => [k, v.rtf])),
    occ: Object.fromEntries(Object.entries(V).map(([k, v]) => [k, v.occ])),
    per_source_pct: Object.fromEntries(rows.map((r) => [r[0], r[1]])),
    unit: { bus_tap: busTap, point_conv: ptTap, own_fdn: fdnEach, remote_fdn: remEach },
  };
}

console.log(`\nsummary: ${M.os} | ${M.cpu} | ${M.threads} threads | node ${M.node}`
  + (results.engine.max_safe ? ' | max N @1.5x: ' + Object.entries(results.engine.max_safe).map(([k, v]) => `${k}=${v.measured}(~${v.est})`).join(' ') : '')
  + ['cpu', 'gpu-proxy'].filter((b) => results.sim[b]).map((b) => ` | sim[${b}] ${results.sim[b].fixed_ms.toFixed(1)}+${results.sim[b].ms_per_source.toFixed(2)}ms/src`).join(''));
if (JSON_OUT) {
  writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));
  console.log(`raw results → ${JSON_OUT}`);
}
