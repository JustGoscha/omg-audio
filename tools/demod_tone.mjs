// Demodulate a recording at 500 Hz as `assess cartone` does: pitch and
// level per ms against their own 25 ms course. Reports the whole pass and
// the loud part (within 20 dB of the peak), where nothing else is heard.
// usage: node tools/demod_tone.mjs file.wav [TRACE]   (SPEC=1: the pitch
// error by band)
import { readFileSync } from 'node:fs';
const b = readFileSync(process.argv[2]);
const SR = b.readUInt32LE(24);
const fmt = b.readUInt16LE(20);
const bits = b.readUInt16LE(34);
let off = 12, data = null;
while (off < b.length) { const id = b.toString('ascii', off, off + 4), len = b.readUInt32LE(off + 4); if (id === 'data') { data = [off + 8, len]; break; } off += 8 + len; }
const n = data[1] / (2 * bits / 8);
const at = (k, c) => fmt === 3 ? b.readFloatLE(data[0] + (2 * k + c) * 4) : b.readInt16LE(data[0] + (2 * k + c) * 2) / 32768;
const f0 = 500;
const c = 1 - Math.exp(-2 * Math.PI * 60 / SR);
const w0 = 2 * Math.PI * f0 / SR;
let i1 = 0, q1 = 0, i2 = 0, q2 = 0;
const ph = [], amp = [], ks = [];
const step = SR / 1000; // ms grid, fractional at 44.1 kHz
let next = 0;
for (let k = 0; k < n; k++) {
  const y = 0.5 * (at(k, 0) + at(k, 1));
  i1 += c * (y * Math.cos(w0 * k) - i1);
  q1 += c * (-y * Math.sin(w0 * k) - q1);
  i2 += c * (i1 - i2);
  q2 += c * (q1 - q2);
  if (k >= next) {
    // (the grid is fractional at 44.1 kHz: frequencies use the exact
    // sample spacing)
    ph.push(Math.atan2(q2, i2));
    ks.push(k);
    amp.push(10 * Math.log10(Math.max(i2 * i2 + q2 * q2, 1e-30)));
    next += step;
  }
}
const fr = [];
for (let k = 1; k < ph.length; k++) {
  let d = ph[k] - ph[k - 1];
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  fr.push(f0 + d * SR / (2 * Math.PI * (ks[k] - ks[k - 1])));
}
const course = (x, k) => { const a = Math.max(0, k - 12), e = Math.min(x.length, k + 13); let s = 0; for (let i = a; i < e; i++) s += x[i]; return s / (e - a); };
const a0 = 1000, b0 = fr.length - 200;
let peak = -Infinity;
for (let k = a0; k < b0; k++) peak = Math.max(peak, amp[k + 1]);
const span = (lo) => {
  let fj = 0, aj = 0, m = 0, jumps = 0, fmin = Infinity, fmax = -Infinity;
  for (let k = a0; k < b0; k++) {
    if (amp[k + 1] < lo) continue;
    const df = fr[k] - course(fr, k);
    const da = amp[k + 1] - course(amp, k + 1);
    fj += df * df; aj += da * da; m++;
    if (Math.abs(da) > 2) jumps++;
    fmin = Math.min(fmin, course(fr, k)); fmax = Math.max(fmax, course(fr, k));
    if (process.argv[3] && (Math.abs(df) > 3 || Math.abs(da) > 2)) console.log(`   t ${((k + 1) / 1000).toFixed(3)}  level ${da.toFixed(1)} dB  pitch ${df.toFixed(2)} Hz  (${amp[k + 1].toFixed(1)} dB)`);
  }
  return `pitch ${Math.sqrt(fj / m).toFixed(2)} Hz rms, level ${Math.sqrt(aj / m).toFixed(2)} dB rms, level off by >2 dB in ${jumps} of ${m} ms (course ${fmin.toFixed(1)}..${fmax.toFixed(1)} Hz)`;
};
if (process.env.SPEC) {
  // the pitch error's spectrum (1 kHz series): where it repeats
  const df = [];
  for (let k = a0; k < b0; k++) df.push(fr[k] - course(fr, k));
  const out = [];
  for (let hz = 2; hz <= 250; hz += 0.5) {
    let re = 0, im = 0;
    for (let i = 0; i < df.length; i++) { re += df[i] * Math.cos(2 * Math.PI * hz * i / 1000); im += df[i] * Math.sin(2 * Math.PI * hz * i / 1000); }
    out.push([hz, Math.sqrt(re * re + im * im) * 2 / df.length]);
  }
  const bands = [[2, 15], [15, 25], [25, 45], [45, 70], [70, 110], [110, 140], [140, 250]];
  console.log('  pitch error by band (Hz rms):', bands.map(([lo, hi]) => `${lo}-${hi}: ${Math.sqrt(out.filter(([h]) => h >= lo && h < hi).reduce((s, [, v]) => s + v * v / 2, 0) * 2).toFixed(2)}`).join('  '));
  console.log('  peaks:', out.slice().sort((x, y) => y[1] - x[1]).slice(0, 8).map(([h, v]) => `${h} Hz ${v.toFixed(2)}`).join(', '));
}
console.log(`${process.argv[2]}: SR ${SR}, ${(n / SR).toFixed(2)} s, tone peak ${peak.toFixed(1)} dB`);
console.log(`  whole pass:          ${span(-Infinity)}`);
console.log(`  within 20 dB of peak: ${span(peak - 20)}`);
