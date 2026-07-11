// AudioWorkletProcessor hosting the wasm engine. Everything arrives via
// port messages: the wasm binary, HRIR assets, decoded source audio, then a
// steady stream of flat ParamBlocks (20 Hz) and head-yaw updates (~60 Hz).
class OmgProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ready = false;
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  async onMessage(m) {
    if (m.type === 'wasm') {
      const { instance } = await WebAssembly.instantiate(m.bytes, {});
      this.w = instance.exports;
      this.w.eng_init(sampleRate);
      this.pending = m; // grid/speakers/sources arrive in this same message
      const put = (allocName, doneName, bytes) => {
        const ptr = this.w[allocName](bytes.byteLength);
        new Uint8Array(this.w.memory.buffer, ptr, bytes.byteLength).set(new Uint8Array(bytes));
        this.w[doneName]();
      };
      put('eng_hrir_grid_alloc', 'eng_hrir_grid_done', m.grid);
      put('eng_hrir_speakers_alloc', 'eng_hrir_speakers_done', m.speakers);
      m.sources.forEach((buf, i) => {
        const f = new Float32Array(buf);
        const ptr = this.w.eng_source_alloc(i, f.length);
        new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
      });
      if (m.ambient) {
        const f = new Float32Array(m.ambient);
        const ptr = this.w.eng_ambient_alloc(f.length);
        new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
        this.w.eng_ambient_commit(2); // interleaved stereo
      }
      (m.fx || []).forEach((buf) => {
        const f = new Float32Array(buf);
        const ptr = this.w.eng_fx_alloc(f.length);
        new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
        this.w.eng_fx_commit();
      });
      this.ready = true;
      this.port.postMessage({ type: 'ready' });
    } else if (m.type === 'params' && this.ready) {
      m.blocks.forEach((buf, i) => {
        const f = new Float32Array(buf);
        const ptr = this.w.eng_param_buf_ptr();
        new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
        this.w.eng_set_params(i, f.length);
      });
    } else if (m.type === 'head' && this.ready) {
      this.w.eng_set_head(m.yaw);
    } else if (m.type === 'ambient' && this.ready) {
      this.w.eng_set_ambient(m.gain, m.fc);
    } else if (m.type === 'fx' && this.ready) {
      if (m.action === 'play') this.w.eng_fx_play(m.src, m.kind);
      else this.w.eng_fx_stop(m.src, m.kind);
    }
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    if (!this.ready || out.length < 2) {
      return true;
    }
    const n = out[0].length;
    this.w.eng_process(n);
    const l = new Float32Array(this.w.memory.buffer, this.w.eng_out_l(), n);
    const r = new Float32Array(this.w.memory.buffer, this.w.eng_out_r(), n);
    out[0].set(l);
    out[1].set(r);

    // level meters + AGC state → main thread, ~every 23 ms
    this.mL = this.mL || 0;
    this.mR = this.mR || 0;
    this.mN = (this.mN || 0) + 1;
    for (let i = 0; i < n; i++) {
      this.mL = Math.max(this.mL, Math.abs(l[i]));
      this.mR = Math.max(this.mR, Math.abs(r[i]));
    }
    if (this.mN >= 8) {
      this.port.postMessage({ type: 'meters', l: this.mL, r: this.mR, agc: this.w.eng_agc_gain() });
      this.mL = 0;
      this.mR = 0;
      this.mN = 0;
    }
    return true;
  }
}
registerProcessor('omg-engine', OmgProcessor);
