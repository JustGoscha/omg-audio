// The web demo's Doppler, measured in a browser: headless Chrome opens the
// page (served: python3 tools/serve.py), mutes everything but one car,
// gives it a 500 Hz tone, drives it past the listener 30 m off at 100
// km/h (assess cartone's pass; WALK=v: the car stands and the listener
// walks past at v m/s) and records the worklet's output — a WAV to read
// with tools/demod_tone.mjs (pitch and level per ms against their own
// 25 ms course, as assess cartone reads its own).
// usage: node tools/web_doppler.mjs <out.wav> [port]
//   LIS=x,y ROAD=x Y0=y SPEED=m/s DUR=s WALK=m/s
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const [,, out = 'doppler.wav', port = '9334'] = process.argv;
// the geometry (assess cartone's default): the listener at LIS, the car
// on x = ROAD from y = Y0 at SPEED m/s for DUR s
const LIS = (process.env.LIS || '-100,100').split(',').map(Number);
const ROAD = Number(process.env.ROAD || -130), Y0 = Number(process.env.Y0 || -50);
const SPEED = Number(process.env.SPEED || 27.78), DUR = Number(process.env.DUR || 11.8);
// WALK=v: the car stands at (ROAD, Y0) and the listener walks along +y
// at v m/s from LIS instead (moved every rendered frame)
const WALK = process.env.WALK ? Number(process.env.WALK) : null;
const prof = mkdtempSync(join(tmpdir(), 'omg-doppler-'));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--mute-audio', `--user-data-dir=${prof}`,
  '--autoplay-policy=no-user-gesture-required', `--remote-debugging-port=${port}`, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page'); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
ws.binaryType = 'arraybuffer';
await new Promise((r) => ws.addEventListener('open', r));
let id = 0; const pending = new Map(); const errors = [];
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push('exception: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 300));
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 500));
  return r.result?.result?.value;
};
await send('Runtime.enable'); await send('Page.enable');
// the frames rendered and the poses posted while recording; WALK moves
// the listener at the start of every frame
await send('Page.addScriptToEvaluateOnNewDocument', { source: `(() => {
  let rafT = null;
  window.__raf = [];
  window.__posts = [];
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => raf((t) => {
    if (t !== rafT) {
      rafT = t;
      if (window.__recOn) window.__raf.push(t);
      const w = window.__walk;
      if (w && window.__omg) { if (w.t0 == null) w.t0 = t; window.__omg.pose.y = w.y0 + w.v * (t - w.t0) / 1000; }
    }
    cb(t);
  });
  const pm = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function (m, tr) {
    if (m && m.type === 'pose' && window.__recOn) window.__posts.push(performance.now());
    return tr === undefined ? pm.call(this, m) : pm.call(this, m, tr);
  };
})();` });
await send('Page.navigate', { url: 'http://localhost:8000/demo/web/' });
await sleep(4000);
console.log('start:', await evaluate("document.getElementById('start') ? (document.getElementById('start').click(), 'clicked') : 'no start button'"));
// the engine up and the bed decoded (it arrives ~10 s in)
for (let i = 0; i < 60; i++) {
  await sleep(500);
  if (await evaluate('!!(window.__omg && window.__omg.node && window.__omg.setMute && window.__omg.running)')) break;
}
await sleep(15000);
const setup = await evaluate(`(async () => {
  const s = window.__omg;
  const ctx = s.node.context;
  s.nextCarAt = Infinity;
  s.cars.forEach((c) => { c.mesh.visible = false; });
  s.cars.length = 0;
  const others = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14];
  s.setMute(others, true);
  s.node.port.postMessage({ type: 'mixer', srcs: others, gain: 0 });
  s.node.port.postMessage({ type: 'mixer', target: 'ambient', gain: 0 });
  s.node.port.postMessage({ type: 'mixer', target: 'rainGain', gain: 0 });
  const SR = ctx.sampleRate;
  const tone = new Float32Array(SR); // one second, 500 whole cycles
  for (let k = 0; k < SR; k++) tone[k] = 0.5 * Math.sin(2 * Math.PI * 500 * k / SR);
  s.node.port.postMessage({ type: 'motor', src: 11, buf: tone.buffer }, [tone.buffer]);
  s.pose.x = ${LIS[0]}; s.pose.y = ${LIS[1]};
  const code = "class R extends AudioWorkletProcessor{constructor(){super();this.on=false;this.port.onmessage=(e)=>{this.on=e.data};this.l=new Float32Array(4096);this.r=new Float32Array(4096);this.n=0;this.f0=0}process(i){const a=i[0];if(!this.on||a.length<2)return true;if(this.n===0)this.f0=currentFrame;this.l.set(a[0],this.n);this.r.set(a[1],this.n);this.n+=a[0].length;if(this.n>=4096){this.port.postMessage({l:this.l.slice(),r:this.r.slice(),f:this.f0});this.n=0}return true}}registerProcessor('rec',R)";
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code], { type: 'application/javascript' })));
  const rec = new AudioWorkletNode(ctx, 'rec', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2], channelCount: 2, channelCountMode: 'explicit' });
  s.node.connect(rec);
  rec.connect(ctx.destination);
  window.__chunks = [];
  rec.port.onmessage = (e) => window.__chunks.push(e.data);
  window.__rec = rec;
  return { SR, state: ctx.state, t: ctx.currentTime };
})()`);
console.log('setup:', JSON.stringify(setup));
await sleep(3000); // the muted sources' tails die out
await evaluate(`(() => {
  const s = window.__omg;
  s.cars.push({ slot: 3, mesh: { position: { copy() {} }, visible: false }, x: ${ROAD}, y: ${Y0}, north: true, vy: ${WALK == null ? SPEED : 0}, baseVol: 1, vol: 0 });
  ${WALK == null ? '' : `window.__walk = { v: ${WALK}, y0: ${LIS[1]} };`}
  window.__recOn = true;
  window.__rec.port.postMessage(true);
  return 1;
})()`);
await sleep(DUR * 1000);
await evaluate('(window.__rec.port.postMessage(false), window.__recOn = false, 1)');
await sleep(300);
const got = await evaluate(`(() => {
  const ch = window.__chunks;
  const n = ch.reduce((a, c) => a + c.l.length, 0);
  const all = new Float32Array(2 * n);
  let o = 0, gaps = 0;
  for (let k = 0; k < ch.length; k++) {
    if (k && ch[k].f !== ch[k - 1].f + 4096) gaps++;
    for (let i = 0; i < ch[k].l.length; i++) { all[2 * (o + i)] = ch[k].l[i]; all[2 * (o + i) + 1] = ch[k].r[i]; }
    o += ch[k].l.length;
  }
  const u8 = new Uint8Array(all.buffer);
  let bin = '';
  for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  const d = (a) => a.slice(1).map((v, i) => v - a[i]);
  const st = (a) => a.length ? { n: a.length, mean: +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(2), max: +Math.max(...a).toFixed(2), p95: +[...a].sort((x, y) => x - y)[Math.floor(0.95 * a.length)].toFixed(2) } : null;
  const s = window.__omg;
  return { b64: btoa(bin), SR: window.__rec.context.sampleRate, gaps, raf: st(d(window.__raf)), posts: st(d(window.__posts)),
           tickMs: s.debug && s.debug.tickMs, level: s.quality && s.quality.level, pose: [s.pose.x, s.pose.y, s.pose.z] };
})()`);
ws.close(); chrome.kill();
const buf = Buffer.from(got.b64, 'base64');
const f = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const SR = got.SR;
const n = f.length / 2;
// WAV (float32 stereo)
const wav = Buffer.alloc(44 + buf.length);
wav.write('RIFF', 0); wav.writeUInt32LE(36 + buf.length, 4); wav.write('WAVE', 8); wav.write('fmt ', 12);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(3, 20); wav.writeUInt16LE(2, 22); wav.writeUInt32LE(SR, 24);
wav.writeUInt32LE(SR * 8, 28); wav.writeUInt16LE(8, 32); wav.writeUInt16LE(32, 34); wav.write('data', 36); wav.writeUInt32LE(buf.length, 40);
buf.copy(wav, 44);
writeFileSync(out, wav);
console.log(`${out}: ${(n / SR).toFixed(2)} s at ${SR} Hz, recorder gaps ${got.gaps}`);
console.log(`  frames: ${JSON.stringify(got.raf)}; pose posts: ${JSON.stringify(got.posts)}; tick ${got.tickMs?.toFixed?.(1)} ms, level ${got.level}, pose ${JSON.stringify(got.pose)}`);
console.log(errors.length ? errors.join('\n') : '  no exceptions');
process.exit(0);
