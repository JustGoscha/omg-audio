//! The single decode stage for the summed SH bus of all sources: rotates
//! the field by (smoothed) head orientation — yaw, pitch and roll, so
//! mouse look, device tilt and camera face tracking all move the field —
//! then decodes: measured-HRIR binaural when speaker HRIRs are provided,
//! virtual-cardioid stereo otherwise — and mixes in the point-rendered
//! stereo. Shared by native and web builds.

use crate::ambi::{HeadRotation, StereoDecoder, NCH};
use crate::hrtf::BinauralDecoder;
use crate::smooth::Smoothed;

enum Decode {
    Hrtf(BinauralDecoder),
    Cardioid(StereoDecoder),
}

pub struct OutputStage {
    decode: Decode,
    /// Smoothed head orientation [yaw, pitch, roll] (see `HeadRotation`
    /// for the conventions).
    head: [Smoothed; 3],
    // The ear (`process`): levels in dB SPL from the engine's calibration
    // (`SPL_REF_RMS`); updated every `CONTROL` samples.
    /// Short-term level (dB SPL): the mean square over ~`SHORT_S`.
    short_ms: f32,
    short_coef: f32,
    /// The masker's mean square (`LOUDNESS_S`).
    loud_ms: f32,
    loud_coef: f32,
    /// The ear's adaptation level (dB SPL): follows the short-term level
    /// up within `ADAPT_UP_S`, back down within `ADAPT_DOWN_S`.
    adapt_db: f32,
    adapt_up: f32,
    adapt_slew: f32,
    adapt_down: f32,
    /// The acoustic reflex (dB of attenuation below `REFLEX_HZ`).
    reflex_db: f32,
    reflex_on: f32,
    reflex_off: f32,
    /// The share of the reflex lost to its decay (0 … `REFLEX_DECAY`).
    reflex_tired: f32,
    reflex_tire: f32,
    reflex_rest: f32,
    reflex_lp: (f32, f32),
    reflex_coef: f32,
    /// Linear gains now (per sample, gliding to the control's targets):
    /// the adaptation and the reflex's low-band share.
    gain: Smoothed,
    reflex: Smoothed,
    /// The safety limiter's peak envelope (output, after the adaptation).
    peak: f32,
    peak_att: f32,
    peak_rel: f32,
    countdown: u32,
    // Temporary threshold shift ("club ears"): sustained levels far above
    // a safe exposure accumulate fatigue; fatigue muffles the output — a
    // lowpass that deepens with exposure and fully lets go within ~20 s,
    // the way hearing stays dulled after stepping away from a PA.
    // Protection-shaped only: exactly zero effect at normal levels.
    tts: f32,
    tts_up: f32,
    tts_down: f32,
    tts_lp: (f32, f32),
    tts_coef: f32,
    // Tinnitus: a faint ring after blasts. Fed by peaks far above even the
    // fatigue threshold, decays over ~6 s. Diotic (identical in both ears
    // — it reads as inside the head), and NOT scaled by the adaptation:
    // the ear cannot quiet a phantom sound.
    tin: f32,
    tin_phase: f32,
    tin_step: f32,
    tin_decay: f32,
    /// The ear on (adaptation, reflex, fatigue, tinnitus) — off, the
    /// stage only decodes and limits (the levels still measured).
    ear: bool,
    sample_rate: f32,
}

/// The engine's calibration: a clip at the reference loudness
/// (`level::REF_CLIP_RMS`) from a gain-1 source 1 m away is 90 dB SPL.
const SPL_REF_RMS: f32 = crate::level::REF_CLIP_RMS;
const SPL_REF_DB: f32 = 90.0;
/// Control updates every this many samples (~1.3 kHz).
const CONTROL: u32 = 36;
/// The short-term level's integration (loudness: ~50 ms, Glasberg &
/// Moore 2002's short-term loudness release).
const SHORT_S: f32 = 0.05;
/// Adaptation: the level the ear is set to follows the scene up within
/// ~0.6 s (a loud room is a blast at first, then settles) and back down
/// within ~8 s (stepping out of it, the quiet world returns over
/// seconds) — the range of measured loudness adaptation and recovery.
const ADAPT_UP_S: f32 = 0.6;
const ADAPT_DOWN_S: f32 = 8.0;
/// …and up by at most this many dB a second: the ear adapts to a loud
/// place, not to a shot (a 150 ms blast moves it ~3 dB — the reflex and
/// the limiter take the blast; a club's +30 dB settles over ~3 s).
const ADAPT_SLEW_DB_S: f32 = 10.0;
/// The masker's level for the audible window: the mix's mean square over
/// the ear's loudness integration (simultaneous masking acts now, not
/// after the adaptation).
const LOUDNESS_S: f32 = 0.4;
/// The adaptation plays a scene's level change at this fraction (in dB):
/// a quiet street 40 dB under a club plays ~20 dB under it — quiet things
/// in quiet places audible, loud places loud, neither off the scale. How
/// much of the ear's range a playback system shows is a display choice,
/// like HDR tone mapping (no physics decides it).
const ADAPT_SHARE: f32 = 0.5;
/// The adapted level that plays as it is (dB SPL): the engine's anchor
/// (a gain-1 source at 1 m plays as recorded) — louder places play
/// lower, quieter ones higher.
const ADAPT_REF_DB: f32 = SPL_REF_DB;
/// The comfortable band (dB SPL the ear is adapted to): inside it nothing
/// adapts — a door shut between two ordinary places plays its full
/// difference, at once and for good (a share of every change made a shut
/// door half as quiet as it is, then lifted it back over seconds); above
/// it loud places duck, below it quiet ones come up (at `ADAPT_SHARE`).
/// The band plays at the gain the share gives at its top, so loud places
/// play as they did.
const ADAPT_BAND_LOW_DB: f32 = 50.0;
const ADAPT_BAND_HIGH_DB: f32 = 80.0;
/// The adaptation's reach (dB): at most this far down / up.
const ADAPT_MAX_CUT_DB: f32 = 30.0;
const ADAPT_MAX_BOOST_DB: f32 = 24.0;
/// The acoustic (stapedius) reflex: from ~90 dB SPL it stiffens the
/// middle ear — up to 15 dB less below ~1.5 kHz, within ~0.1 s, letting go
/// within ~1 s.
const REFLEX_ONSET_DB: f32 = 90.0;
const REFLEX_SLOPE: f32 = 0.75;
const REFLEX_MAX_DB: f32 = 15.0;
const REFLEX_HZ: f32 = 1_500.0;
const REFLEX_ON_S: f32 = 0.1;
const REFLEX_OFF_S: f32 = 1.0;
/// …and held, it decays ("reflex decay", the audiology test: a normal
/// ear keeps over half of it through 10 s of sustained sound): down to
/// half within ~10 s while engaged, back within ~2 s of rest — a club is
/// a blast at first, its bass loud again once you are in it.
const REFLEX_DECAY: f32 = 0.5;
const REFLEX_DECAY_S: f32 = 10.0;
const REFLEX_REST_S: f32 = 2.0;
/// The safety limiter: output peaks kept under this (before the tanh),
/// reaching for them within ~1 ms (an instant gain step is a click of its
/// own; what overshoots in that millisecond, the tanh rounds).
const LIMIT: f32 = 0.9;
const LIMIT_ATTACK_S: f32 = 0.001;
/// Fatigue builds above this short-term level (dB SPL), fully at +10 dB.
const TTS_DB: f32 = 100.0;
/// Muffle cutoff range: log-swept 18 kHz (none) → 1.4 kHz (full fatigue).
const TTS_FC_HI: f32 = 18_000.0;
const TTS_FC_LO: f32 = 1_400.0;
/// Tinnitus: peaks above this (per sample, before the adaptation: the
/// mix in the engine's units, ≈ 112 dB SPL peak) feed the ring.
const TIN_THRESH: f32 = 2.2;
/// Ring pitch and ceiling level (≈ −44 dBFS — audible only in quiet).
const TIN_HZ: f32 = 5_400.0;
const TIN_LEVEL: f32 = 0.006;

fn spl_db(ms: f32) -> f32 {
    SPL_REF_DB + 10.0 * (ms.max(1e-20) / (SPL_REF_RMS * SPL_REF_RMS)).log10()
}

impl OutputStage {
    pub fn from_speaker_bytes(bytes: Option<&[u8]>, sample_rate: f32) -> Self {
        let decode = match bytes {
            Some(b) => Decode::Hrtf(BinauralDecoder::from_bytes(b)),
            None => Decode::Cardioid(StereoDecoder::new()),
        };
        let tc = |tau: f32| 1.0 - (-1.0 / (tau * sample_rate)).exp();
        let tc_ctl = |tau: f32| 1.0 - (-(CONTROL as f32) / (tau * sample_rate)).exp();
        Self {
            decode,
            head: core::array::from_fn(|_| Smoothed::new(0.0, 0.03, sample_rate)),
            short_ms: 0.0,
            short_coef: tc(SHORT_S),
            loud_ms: 0.0,
            loud_coef: tc(LOUDNESS_S),
            adapt_db: ADAPT_REF_DB,
            adapt_up: tc_ctl(ADAPT_UP_S),
            adapt_slew: ADAPT_SLEW_DB_S * CONTROL as f32 / sample_rate,
            adapt_down: tc_ctl(ADAPT_DOWN_S),
            reflex_db: 0.0,
            reflex_on: tc_ctl(REFLEX_ON_S),
            reflex_off: tc_ctl(REFLEX_OFF_S),
            reflex_tired: 0.0,
            reflex_tire: tc_ctl(REFLEX_DECAY_S),
            reflex_rest: tc_ctl(REFLEX_REST_S),
            reflex_lp: (0.0, 0.0),
            reflex_coef: 1.0 - (-core::f32::consts::TAU * REFLEX_HZ / sample_rate).exp(),
            gain: Smoothed::new(1.0, 0.01, sample_rate),
            reflex: Smoothed::new(1.0, 0.01, sample_rate),
            peak: 0.0,
            peak_att: tc(LIMIT_ATTACK_S),
            peak_rel: (-1.0 / (0.2 * sample_rate)).exp(),
            countdown: 0,
            tts: 0.0,
            tts_up: tc_ctl(4.0),
            tts_down: tc_ctl(6.0),
            tts_lp: (0.0, 0.0),
            tts_coef: 1.0,
            tin: 0.0,
            tin_phase: 0.0,
            tin_step: core::f32::consts::TAU * TIN_HZ / sample_rate,
            tin_decay: (-1.0 / (2.5 * sample_rate)).exp(),
            ear: true,
            sample_rate,
        }
    }

    /// The ear on or off. Off, the scene plays at its own level: no
    /// adaptation, no reflex, no fatigue (the safety limiter stays).
    pub fn set_ear(&mut self, on: bool) {
        self.ear = on;
    }

    pub fn is_binaural(&self) -> bool {
        matches!(self.decode, Decode::Hrtf(_))
    }

    pub fn speaker_count(&self) -> usize {
        match &self.decode {
            Decode::Hrtf(d) => d.speaker_count(),
            Decode::Cardioid(_) => 2,
        }
    }

    pub fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.head[0].set(yaw);
        self.head[1].set(pitch);
        self.head[2].set(roll);
    }

    /// What the audible window sits under, as the mix's RMS in the
    /// engine's units: the level the ear is adapted to, or the mix as it
    /// is now when louder (it masks at once).
    pub fn loudness(&self) -> f32 {
        (SPL_REF_RMS * 10f32.powf((self.adapt_db - SPL_REF_DB) / 20.0)).max(self.loud_ms.sqrt())
    }

    /// The ear's gain now (the adaptation; linear).
    pub fn agc_gain(&self) -> f32 {
        self.gain.current()
    }

    /// The level the ear is adapted to (dB SPL) and the reflex's
    /// low-band attenuation now (dB).
    pub fn adaptation(&self) -> (f32, f32) {
        (self.adapt_db, self.reflex_db)
    }

    /// Current hearing-fatigue amount, 0 (fresh) … 1 (fully dulled).
    pub fn ear_fatigue(&self) -> f32 {
        self.tts
    }

    /// Current tinnitus excitation, 0 … 1.
    pub fn tinnitus(&self) -> f32 {
        self.tin
    }

    /// The ear's state, every `CONTROL` samples: the short-term level
    /// moves the adaptation, the reflex and the fatigue; their gains
    /// glide there sample by sample.
    fn control(&mut self) {
        let short = spl_db(self.short_ms);
        if !self.ear {
            self.adapt_db = short.max(self.adapt_db - 0.1);
            self.gain.set(1.0);
            self.reflex.set(1.0);
            self.reflex_db = 0.0;
            self.tts = 0.0;
            self.tts_coef = 1.0;
            return;
        }
        self.adapt_db += if short > self.adapt_db {
            (self.adapt_up * (short - self.adapt_db)).min(self.adapt_slew)
        } else {
            self.adapt_down * (short - self.adapt_db)
        };
        let band = -ADAPT_SHARE * (ADAPT_BAND_HIGH_DB - ADAPT_REF_DB);
        let g_db = if self.adapt_db > ADAPT_BAND_HIGH_DB {
            -ADAPT_SHARE * (self.adapt_db - ADAPT_REF_DB)
        } else if self.adapt_db < ADAPT_BAND_LOW_DB {
            band + ADAPT_SHARE * (ADAPT_BAND_LOW_DB - self.adapt_db)
        } else {
            band
        }
        .clamp(-ADAPT_MAX_CUT_DB, ADAPT_MAX_BOOST_DB);
        self.gain.set(10f32.powf(g_db / 20.0));
        let want = ((short - REFLEX_ONSET_DB) * REFLEX_SLOPE).clamp(0.0, REFLEX_MAX_DB);
        // engaged, it tires; at rest it recovers
        let (k, tired) = if want > 0.0 { (self.reflex_tire, REFLEX_DECAY) } else { (self.reflex_rest, 0.0) };
        self.reflex_tired += k * (tired - self.reflex_tired);
        let want = want * (1.0 - self.reflex_tired);
        let k = if want > self.reflex_db { self.reflex_on } else { self.reflex_off };
        self.reflex_db += k * (want - self.reflex_db);
        self.reflex.set(10f32.powf(-self.reflex_db / 20.0));
        // hearing fatigue: builds over seconds above `TTS_DB`, clears
        // within ~20 s of relief
        let excess = ((short - TTS_DB) / 10.0).clamp(0.0, 1.0);
        let k = if excess > self.tts { self.tts_up } else { self.tts_down };
        self.tts += k * (excess - self.tts);
        let fc = TTS_FC_HI * (TTS_FC_LO / TTS_FC_HI).powf(self.tts);
        self.tts_coef = 1.0 - (-core::f32::consts::TAU * fc / self.sample_rate).exp();
    }

    /// Rotate + decode the diffuse bus, mix in point-rendered stereo, and
    /// play it through the ear: adaptation, reflex, fatigue, a safety
    /// limiter, tanh.
    #[inline]
    pub fn process(&mut self, sh: &[f32; NCH], pl: f32, pr: f32) -> (f32, f32) {
        let mut rotated = *sh;
        HeadRotation::new(self.head[0].tick(), self.head[1].tick(), self.head[2].tick())
            .apply(&mut rotated);
        let (l, r) = match &mut self.decode {
            Decode::Hrtf(d) => d.process(&rotated),
            Decode::Cardioid(d) => d.decode(&rotated),
        };
        let (l, r) = (l + pl, r + pr);
        let ms = 0.5 * (l * l + r * r);
        self.short_ms += self.short_coef * (ms - self.short_ms);
        self.loud_ms += self.loud_coef * (ms - self.loud_ms);
        if self.countdown == 0 {
            self.countdown = CONTROL;
            self.control();
        }
        self.countdown -= 1;

        // tinnitus excitation: blasts far beyond the fatigue threshold.
        // The ring sits slightly sharp while strong and settles in pitch
        // as it fades (~8 s to silence) — a static tone reads as a bug,
        // a drifting one reads as your ears.
        let m = l.abs().max(r.abs());
        let excite = if self.ear { (m - TIN_THRESH).max(0.0) * 2e-4 } else { 0.0 };
        self.tin = (self.tin + excite).min(1.0) * self.tin_decay;
        let ring = if self.tin > 1e-4 {
            let step = self.tin_step * (1.0 + 0.08 * self.tin);
            self.tin_phase = (self.tin_phase + step) % core::f32::consts::TAU;
            self.tin_phase.sin() * self.tin * TIN_LEVEL
        } else {
            0.0
        };

        // the adaptation, and the reflex taking the low band down
        let g = self.gain.tick();
        let rx = self.reflex.tick();
        self.reflex_lp.0 += self.reflex_coef * (l - self.reflex_lp.0);
        self.reflex_lp.1 += self.reflex_coef * (r - self.reflex_lp.1);
        let gl = g * (l - (1.0 - rx) * self.reflex_lp.0);
        let gr = g * (r - (1.0 - rx) * self.reflex_lp.1);
        // dry/lowpassed blend by fatigue — bit-transparent at tts = 0
        self.tts_lp.0 += self.tts_coef * (gl - self.tts_lp.0);
        self.tts_lp.1 += self.tts_coef * (gr - self.tts_lp.1);
        let ol = gl + self.tts * (self.tts_lp.0 - gl);
        let or_ = gr + self.tts * (self.tts_lp.1 - gr);
        // the safety limiter: within ~1 ms on a peak over `LIMIT`, letting
        // go over ~0.2 s (what the adaptation has not caught yet: a shot)
        let p = ol.abs().max(or_.abs());
        self.peak = if p > self.peak { self.peak + self.peak_att * (p - self.peak) } else { self.peak * self.peak_rel + p * (1.0 - self.peak_rel) };
        let lim = if self.peak > LIMIT { LIMIT / self.peak } else { 1.0 };
        ((ol * lim + ring).tanh(), (or_ * lim + ring).tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ratio of an 8 kHz probe to a 1.5 kHz probe through the stage — the
    /// adaptation scales both equally and the reflex acts below ~1 kHz, so
    /// this isolates the muffle filter.
    fn hf_lf_ratio(out: &mut OutputStage) -> f32 {
        let mut peak = |freq: f32| -> f32 {
            let mut p = 0.0f32;
            for k in 0..4800 {
                let x = 0.02 * (core::f32::consts::TAU * freq * k as f32 / 48_000.0).sin();
                let (l, _) = out.process(&[0.0; NCH], x, x);
                if k > 960 {
                    p = p.max(l.abs());
                }
            }
            p
        };
        peak(8000.0) / peak(1500.0).max(1e-9)
    }

    #[test]
    fn loud_exposure_muffles_highs_then_recovers() {
        let mut out = OutputStage::from_speaker_bytes(None, 48_000.0);
        let fresh = hf_lf_ratio(&mut out);

        // 10 s of club-at-the-PA level (demand ≫ AGC target, below the
        // tinnitus blast threshold so the ring can't pollute the probe)
        for k in 0..48_000 * 10 {
            let x = 2.1 * (core::f32::consts::TAU * 700.0 * k as f32 / 48_000.0).sin();
            out.process(&[0.0; NCH], x, x);
        }
        assert!(out.ear_fatigue() > 0.5, "fatigue {}", out.ear_fatigue());
        let dulled = hf_lf_ratio(&mut out);
        assert!(
            dulled < 0.6 * fresh,
            "highs should dull after exposure: {dulled} vs fresh {fresh}"
        );

        // ~25 s of near-silence: hearing comes back
        for _ in 0..48_000 * 25 {
            out.process(&[0.0; NCH], 0.0, 0.0);
        }
        let later = hf_lf_ratio(&mut out);
        assert!(
            later > 0.85 * fresh,
            "should recover: {later} vs fresh {fresh}"
        );
    }

    #[test]
    fn blast_rings_then_fades() {
        let mut out = OutputStage::from_speaker_bytes(None, 48_000.0);
        // 120 ms explosion-grade blast
        for k in 0..4800 {
            let x = 4.5 * (core::f32::consts::TAU * 90.0 * k as f32 / 48_000.0).sin();
            out.process(&[0.0; NCH], x, x);
        }
        assert!(out.tinnitus() > 0.05, "blast should excite: {}", out.tinnitus());
        // in silence right after: the ring is audible (nonzero output)
        let mut e = 0.0f64;
        for _ in 0..4800 {
            let (l, _) = out.process(&[0.0; NCH], 0.0, 0.0);
            e += (l * l) as f64;
        }
        assert!(e > 1e-9, "ring should be audible in silence: {e}");
        // and it lets go over ~20 s
        for _ in 0..48_000 * 20 {
            out.process(&[0.0; NCH], 0.0, 0.0);
        }
        assert!(out.tinnitus() < 0.01, "ring should fade: {}", out.tinnitus());
    }

    /// The reflex must protect during a blast but RELEASE like a reflex:
    /// moderate content is back within ~2.5 s, not dulled for ~7 s (the
    /// "walking around stops the sounds" field report).
    #[test]
    fn reflex_releases_quickly_after_a_blast() {
        let mut out = OutputStage::from_speaker_bytes(None, 48_000.0);
        let tone = |k: usize, a: f32, f: f32| {
            a * (core::f32::consts::TAU * f * k as f32 / 48_000.0).sin()
        };
        // settle on moderate content (the ear adapts over seconds) and
        // take the reference level
        let mut before = 0.0f32;
        for k in 0..48_000 * 6 {
            let x = tone(k, 0.25, 400.0);
            let (l, _) = out.process(&[0.0; NCH], x, x);
            if k > 48_000 * 5 {
                before = before.max(l.abs());
            }
        }
        // 150 ms explosion
        for k in 0..7200 {
            let x = tone(k, 4.0, 90.0);
            out.process(&[0.0; NCH], x, x);
        }
        // protection is real: right after, the same content is well down
        // (after 10 ms: the cut from blast to tone is a step of its own)
        let mut ducked = 0.0f32;
        for k in 0..9600 {
            let x = tone(k, 0.25, 400.0);
            let (l, _) = out.process(&[0.0; NCH], x, x);
            if k >= 480 {
                ducked = ducked.max(l.abs());
            }
        }
        assert!(ducked < 0.6 * before, "blast should duck: {ducked} vs {before}");
        // ...and the reflex lets go: within 2.5 s the world is back
        let mut after = 0.0f32;
        for k in 0..48_000 * 5 / 2 {
            let x = tone(k, 0.25, 400.0);
            let (l, _) = out.process(&[0.0; NCH], x, x);
            if k > 48_000 * 2 {
                after = after.max(l.abs());
            }
        }
        assert!(
            after > 0.65 * before,
            "reflex must release within seconds: {after} vs {before}"
        );
    }

    #[test]
    fn normal_levels_never_muffle() {
        let mut out = OutputStage::from_speaker_bytes(None, 48_000.0);
        for k in 0..48_000 * 10 {
            let x = 0.3 * (core::f32::consts::TAU * 500.0 * k as f32 / 48_000.0).sin();
            out.process(&[0.0; NCH], x, x);
        }
        assert!(out.ear_fatigue() < 1e-3, "fatigue at normal level: {}", out.ear_fatigue());
        assert!(out.tinnitus() < 1e-6, "tinnitus at normal level: {}", out.tinnitus());
    }
}
