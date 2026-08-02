// AudioWorkletProcessor hosting the wasm engine. Everything arrives via
// port messages: the wasm binary, HRIR assets, decoded source audio, then a
// steady stream of flat ParamBlocks (20 Hz) and head-pose updates (~60 Hz,
// yaw/pitch/roll — mouse look, device tilt and camera face tracking).
class OmgProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ready = false;
    // Adaptive point-render budget: strongest N taps per source get their
    // own HRIR convolution (eng_set_point_budget). The worklet measures its
    // own render load and walks N between the floor and cap with hysteresis
    // — a throttled CPU (Low Power Mode, phones) settles low, a fast
    // desktop climbs to the cap. Bench: tools/bench_web.mjs.
    // Cold-start at the FLOOR and earn the way up: starting high is
    // audible (breakup until the first shed), starting low is a few
    // seconds of soft focus while the grow path (one rung per clean
    // 1 s window) observes that quality holds.
    this.budget = 2;
    // floor 2: point rendering is a sharpness tier, not a requirement —
    // the order-2 bus keeps everything spatialized when a squeezed CPU
    // (camera on, thermal throttle) needs the headroom back
    this.BUDGET_MIN = 2;
    this.BUDGET_MAX = 32;
    // Tap-ceiling ladder (eng_set_tap_ceiling): the per-source cap on
    // incoming taps. The point budget is a *sharpness* tier; this is a
    // *density* tier — the real lever when a doorway carries ~145
    // taps/source and the device can't render them all. Evicted taps
    // release with the slot fade, so stepping down is click-free.
    // deepest rung 32: at a doorway (5 live sources) that is ~160 total
    // taps — the strongest direct paths and first reflections survive,
    // which is all a strained device can honestly render anyway
    this.CEILINGS = [160, 112, 64, 32];
    this.ceilIdx = this.CEILINGS.length - 1; // start dense-shedded too
    this.loadMs = 0;
    this.loadFrames = 0;
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  async onMessage(m) {
    try {
      await this.handle(m);
    } catch (e) {
      // surface init failures to the page (mobile has no console)
      this.port.postMessage({ type: 'error', message: String(e && e.stack || e) });
    }
  }

  async handle(m) {
    if (m.type === 'wasm') {
      if (this.w) return; // never re-init a live engine
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
        this.w.eng_source_commit(i); // import-normalize (gated RMS)
      });
      if (m.drops) {
        const f = new Float32Array(m.drops);
        const ptr = this.w.eng_rain_bank_alloc(f.length);
        new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
        this.w.eng_rain_bank_commit();
      }
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
      this.w.eng_set_point_budget(this.budget);
      this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
      this.ready = true;
      this.port.postMessage({ type: 'ready' });
    } else if (m.type === 'params' && this.ready) {
      let incoming = 0;
      m.blocks.forEach((buf, i) => {
        const f = new Float32Array(buf);
        incoming += Math.max(0, (f.length - 24) / 9); // taps in this block
        const ptr = this.w.eng_param_buf_ptr();
        new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
        this.w.eng_set_params(i, f.length);
      });
      // FEED-FORWARD shed: a doorway surge (club vestibule: ~700 taps
      // vs ~100 in the open) is visible HERE, before a single sample
      // renders at the new density — don't wait for the load window to
      // measure the damage. The ceiling governor recovers afterwards.
      if (!this.manual && incoming > 420 && (this.budget > this.BUDGET_MIN
          || this.ceilIdx < this.CEILINGS.length - 1)) {
        this.budget = this.BUDGET_MIN;
        this.ceilIdx = this.CEILINGS.length - 1;
        this.w.eng_set_point_budget(this.budget);
        this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
        this.loadMs = 0;
        this.loadFrames = 0;
      }
    } else if (m.type === 'head' && this.ready) {
      this.w.eng_set_head(m.yaw, m.pitch || 0, m.roll || 0);
    } else if (m.type === 'rain' && this.ready) {
      this.w.eng_set_rain(m.intensity);
    } else if (m.type === 'env' && this.ready) {
      // flat Environment block: geometry-priced ambience/rain routing
      const f = new Float32Array(m.env);
      new Float32Array(this.w.memory.buffer, this.w.eng_param_buf_ptr(), f.length).set(f);
      this.w.eng_set_env(f.length);
    } else if (m.type === 'motor' && this.ready) {
      // per-spawn motor swap for a car source
      const f = new Float32Array(m.buf);
      const ptr = this.w.eng_source_replace_alloc(m.src, f.length);
      new Float32Array(this.w.memory.buffer, ptr, f.length).set(f);
      this.w.eng_source_commit(m.src);
    } else if (m.type === 'mixer' && this.ready) {
      if (m.target === 'ambient') this.w.eng_set_ambient_user(m.gain);
      else if (m.target === 'rainGain') this.w.eng_set_rain_gain(m.gain);
      else if (m.target === 'master') this.w.eng_set_master(m.gain);
      else for (const i of m.srcs) this.w.eng_set_mixer(i, m.gain);
    } else if (m.type === 'fx' && this.ready) {
      if (m.action === 'play') this.w.eng_fx_play(m.src, m.kind);
      else this.w.eng_fx_stop(m.src, m.kind);
    } else if (m.type === 'ceilfloor' && this.ready) {
      // manual quality pin: the governor may shed below this ladder index
      // but never recover above it (0 = unpinned, full ladder)
      this.ceilFloor = m.idx || 0;
      if (this.ceilIdx < this.ceilFloor) {
        this.ceilIdx = this.ceilFloor;
        this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
      }
    } else if (m.type === 'manual' && this.ready) {
      // tuning-panel override: freeze the governor and set both audio
      // levers directly. {on: false} hands control back to the meters.
      this.manual = !!m.on;
      if (this.manual) {
        if (m.points != null) {
          this.budget = Math.max(0, m.points | 0);
          this.w.eng_set_point_budget(this.budget);
        }
        if (m.ceiling != null) {
          this.manualCeil = Math.max(1, m.ceiling | 0);
          this.w.eng_set_tap_ceiling(this.manualCeil);
        }
      } else {
        this.manualCeil = null;
        this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
      }
    }
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    if (!this.ready || out.length < 2) {
      return true;
    }
    const n = out[0].length;
    // Underrun detection: process() is called once per rendered quantum;
    // when the audio thread gets starved (main-thread GPU/CPU pressure,
    // e.g. face tracking), currentFrame jumps by more than one quantum
    // between calls and the browser has inserted SILENCE for the gap.
    // Counting them makes "the sound stopped" measurable.
    if (this.lastFrame !== undefined && currentFrame - this.lastFrame > n) {
      this.gaps = (this.gaps || 0) + 1;
      this.gapFrames = (this.gapFrames || 0) + (currentFrame - this.lastFrame - n);
      // A gap is ground truth that we missed the deadline — don't wait
      // for the 1 s load window to notice; shed to the floor NOW and
      // restart the window so the grow path re-earns the budget.
      if (this.manual) {
        // tuning panel holds the levers: count the gap, touch nothing
      } else if (this.budget > this.BUDGET_MIN) {
        this.budget = this.BUDGET_MIN;
        this.w.eng_set_point_budget(this.budget);
      } else if (this.ceilIdx < this.CEILINGS.length - 1) {
        // already at the sharpness floor and still gapping: shed density
        this.ceilIdx++;
        this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
      }
      this.loadMs = 0;
      this.loadFrames = 0;
    }
    this.lastFrame = currentFrame;
    const t0 = Date.now();
    this.w.eng_process(n);
    this.loadMs += Date.now() - t0; // 1 ms quantization averages out over the window
    this.loadFrames += n;
    // 0.25 s adaptation window: a sudden squeeze (doorway tap surge,
    // camera turns on, thermal throttle) must shed before it audibly
    // starves the output — a 1 s window measured the damage instead of
    // preventing it (field report: club doors, load pinned over 90).
    if (this.loadFrames >= sampleRate / 4) {
      const ratio = this.loadMs / (this.loadFrames / sampleRate * 1000);
      this.loadRatio = ratio; // exposed via the meters message
      if (this.manual) {
        // tuning panel holds the levers: keep metering, don't govern
      } else if (ratio > 0.80) {
        // emergency: shed BOTH levers to the floor NOW — audible
        // breakup costs more than soft focus, and a doorway surge
        // outruns one-rung-per-window politeness
        if (this.budget > this.BUDGET_MIN || this.ceilIdx < this.CEILINGS.length - 1) {
          this.budget = this.BUDGET_MIN;
          this.ceilIdx = this.CEILINGS.length - 1;
          this.w.eng_set_point_budget(this.budget);
          this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
        }
      } else if (ratio > 0.55 && this.budget > this.BUDGET_MIN) {
        this.budget = Math.max(this.BUDGET_MIN, this.budget - 4);
        this.w.eng_set_point_budget(this.budget);
      } else if (ratio < 0.30) {
        // recover density before sharpness: full tap coverage at soft
        // focus beats razor focus on a thinned field. Climb at the old
        // ~1 s cadence (4 consecutive calm windows) — shedding got 4×
        // faster, recovery must not oscillate against it.
        this.calmWins = (this.calmWins || 0) + 1;
        if (this.calmWins >= 4) {
          this.calmWins = 0;
          if (this.ceilIdx > (this.ceilFloor || 0)) {
            this.ceilIdx--;
            this.w.eng_set_tap_ceiling(this.CEILINGS[this.ceilIdx]);
          } else if (this.budget < this.BUDGET_MAX) {
            this.budget = Math.min(this.BUDGET_MAX, this.budget + 4);
            this.w.eng_set_point_budget(this.budget);
          }
        }
      } else {
        this.calmWins = 0;
      }
      this.loadMs = 0;
      this.loadFrames = 0;
    }
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
      // copy each snapshot before the next call — amb_debug reuses the
      // meter buffer
      const mp = this.w.eng_meters_commit();
      const chans = Array.from(new Float32Array(this.w.memory.buffer, mp, 32));
      const amb = Array.from(new Float32Array(this.w.memory.buffer, this.w.eng_amb_debug(), 12));
      const dbg = Array.from(new Float32Array(this.w.memory.buffer, this.w.eng_debug_render(), 55));
      this.port.postMessage({
        type: 'meters', l: this.mL, r: this.mR, agc: this.w.eng_agc_gain(),
        tts: this.w.eng_ear_fatigue(), pts: this.budget,
        ceil: this.manualCeil || this.CEILINGS[this.ceilIdx],
        manual: !!this.manual,
        load: this.loadRatio || 0, gaps: this.gaps || 0,
        gapMs: ((this.gapFrames || 0) / sampleRate) * 1000, chans, amb, dbg,
      });
      this.mL = 0;
      this.mR = 0;
      this.mN = 0;
    }
    return true;
  }
}
registerProcessor('omg-engine', OmgProcessor);
