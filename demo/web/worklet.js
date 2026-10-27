// AudioWorkletProcessor hosting the engine's Mixer (omg-audio in wasm — the
// same mixer the Godot demo runs). Everything arrives via port messages:
// the wasm, the decoded sources, fx and beds once; then frames (20 Hz, with
// the convolution kernels the worker built) and head poses (~60 Hz). The
// engine's governor walks the render budgets by the time each quantum
// takes (measured here: a worklet's wasm has no clock).
class OmgProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ready = false;
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  async onMessage(m) {
    try {
      await this.handle(m);
    } catch (e) {
      // surface failures to the page (mobile has no console)
      this.port.postMessage({ type: 'error', message: String(e && e.stack || e) });
    }
  }

  // copy `f32` into the engine's staging buffer (the next call takes it)
  put(f32) {
    const p = this.w.eng_stage(f32.length);
    new Float32Array(this.w.memory.buffer, p, f32.length).set(f32);
  }

  async handle(m) {
    if (m.type === 'wasm') {
      if (this.w) return; // never re-init a live engine
      const { instance } = await WebAssembly.instantiate(m.bytes, {});
      this.w = instance.exports;
      this.w.eng_init(sampleRate);
      m.sources.forEach((buf, i) => {
        this.put(new Float32Array(buf));
        this.w.eng_source(i);
      });
      if (m.drops) {
        this.put(new Float32Array(m.drops));
        this.w.eng_rain_bank();
      }
      if (m.ambient) {
        this.put(new Float32Array(m.ambient));
        this.w.eng_ambience(1); // interleaved stereo
      }
      (m.fx || []).forEach((buf) => {
        this.put(new Float32Array(buf));
        this.w.eng_fx();
      });
      this.ready = true;
      this.port.postMessage({ type: 'ready' });
    } else if (!this.ready) {
      // nothing before the engine
    } else if (m.type === 'frame') {
      // kernels first: the frame names the sources they belong to
      (m.clears || []).forEach((s) => this.w.eng_kernel_clear(s));
      (m.kernels || []).forEach((k) => {
        this.put(new Float32Array(k.buf));
        this.w.eng_kernel(k.src);
      });
      this.put(new Float32Array(m.frame));
      this.w.eng_frame();
    } else if (m.type === 'head') {
      this.w.eng_head(m.yaw, m.pitch || 0, m.roll || 0);
    } else if (m.type === 'rain') {
      this.w.eng_rain(m.intensity);
    } else if (m.type === 'motor') {
      // a car's motor, swapped per spawn
      this.put(new Float32Array(m.buf));
      this.w.eng_source(m.src);
    } else if (m.type === 'mixer') {
      if (m.target === 'ambient') this.w.eng_ambience_gain(m.gain);
      else if (m.target === 'rainGain') this.w.eng_rain_gain(m.gain);
      else if (m.target === 'master') this.w.eng_master(m.gain);
      else for (const i of m.srcs) this.w.eng_gain(i, m.gain);
    } else if (m.type === 'fx') {
      if (m.action === 'play') this.w.eng_fx_play(m.src, m.kind, m.gain ?? 1, m.rate ?? 1);
      else this.w.eng_fx_stop(m.src, m.kind);
    } else if (m.type === 'whisper') {
      // a near-field voice at one ear (not part of the scene)
      this.w.eng_whisper(m.kind, m.right ? 1 : 0, m.dist, m.gain);
    } else if (m.type === 'ceilfloor' || m.type === 'manual') {
      // the tuning panel: pin the governor's rung (the ladder's 0..4), or
      // hand it back
      const rung = m.type === 'manual'
        ? (m.on ? Math.max(0, Math.min(4, m.rung | 0)) : -1)
        : (m.idx ? 4 - 2 * m.idx : -1);
      this.w.eng_pin(rung);
    }
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    if (!this.ready || out.length < 2) return true;
    const n = out[0].length;
    // a quantum the browser had to fill with silence: the engine sheds
    if (this.lastFrame !== undefined && currentFrame - this.lastFrame > n) {
      this.gaps = (this.gaps || 0) + 1;
      this.gapFrames = (this.gapFrames || 0) + (currentFrame - this.lastFrame - n);
      this.w.eng_gap();
    }
    this.lastFrame = currentFrame;
    const t0 = Date.now();
    this.w.eng_process(n);
    this.w.eng_report(Date.now() - t0, n);
    const l = new Float32Array(this.w.memory.buffer, this.w.eng_out_l(), n);
    const r = new Float32Array(this.w.memory.buffer, this.w.eng_out_r(), n);
    out[0].set(l);
    out[1].set(r);

    // level meters + engine state → the page, ~every 23 ms
    this.mL = Math.max(this.mL || 0, ...l.map(Math.abs));
    this.mR = Math.max(this.mR || 0, ...r.map(Math.abs));
    this.mN = (this.mN || 0) + 1;
    if (this.mN >= 8) {
      const chans = Array.from(new Float32Array(this.w.memory.buffer, this.w.eng_meters(), 2 * (15 + 2)));
      this.port.postMessage({
        type: 'meters', l: this.mL, r: this.mR, agc: this.w.eng_agc(), tts: this.w.eng_fatigue(),
        rung: this.w.eng_rung(), load: this.w.eng_load(), rendering: this.w.eng_rendering(),
        gaps: this.gaps || 0, gapMs: ((this.gapFrames || 0) / sampleRate) * 1000, chans,
      });
      this.mL = 0;
      this.mR = 0;
      this.mN = 0;
    }
    return true;
  }
}
registerProcessor('omg-engine', OmgProcessor);
