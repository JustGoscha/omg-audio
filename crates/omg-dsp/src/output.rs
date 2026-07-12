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
    // "HDR audio": slow automatic gain riding, like the ear adapting —
    // loud scenes get pulled down quickly (acoustic reflex), quiet scenes
    // get eased up slowly, tanh stays as the final safety.
    env: f32,
    agc_gain: f32,
    env_att: f32,
    env_rel: f32,
    gain_down: f32,
    gain_up_fast: f32,
    gain_up_slow: f32,
    // Temporary threshold shift ("club ears"): sustained demand far above
    // the AGC target accumulates fatigue; fatigue muffles the output —
    // a lowpass that deepens with exposure and fully lets go within
    // ~20 s, the way hearing stays dulled after stepping away from a PA.
    // Protection-shaped only: exactly zero effect at normal levels.
    tts: f32,
    tts_up: f32,
    tts_down: f32,
    tts_lp: (f32, f32),
    tts_coef: f32,
    tts_refresh: u32,
    // Tinnitus: a faint ring after blasts. Fed by demand spikes far above
    // even the fatigue threshold, decays over ~6 s. Diotic (identical in
    // both ears — it reads as inside the head), and NOT scaled by the AGC:
    // the acoustic reflex cannot quiet a phantom sound.
    tin: f32,
    tin_phase: f32,
    tin_step: f32,
    tin_decay: f32,
    sample_rate: f32,
}

const AGC_TARGET: f32 = 0.35;
const AGC_MAX_BOOST: f32 = 1.15;
const AGC_MAX_CUT: f32 = 0.06;
/// Demand (pre-AGC envelope) where fatigue starts: ≈ +10 dB over target.
const TTS_LOUD: f32 = AGC_TARGET * 3.2;
/// Muffle cutoff range: log-swept 18 kHz (none) → 1.4 kHz (full fatigue).
const TTS_FC_HI: f32 = 18_000.0;
const TTS_FC_LO: f32 = 1_400.0;
/// Tinnitus: demand above this (per sample, pre-AGC) feeds the ring.
const TIN_THRESH: f32 = 2.2;
/// Ring pitch and ceiling level (≈ −44 dBFS — audible only in quiet).
const TIN_HZ: f32 = 5_400.0;
const TIN_LEVEL: f32 = 0.006;

impl OutputStage {
    pub fn from_speaker_bytes(bytes: Option<&[u8]>, sample_rate: f32) -> Self {
        let decode = match bytes {
            Some(b) => Decode::Hrtf(BinauralDecoder::from_bytes(b)),
            None => Decode::Cardioid(StereoDecoder::new()),
        };
        let tc = |tau: f32| 1.0 - (-1.0 / (tau * sample_rate)).exp();
        Self {
            decode,
            head: core::array::from_fn(|_| Smoothed::new(0.0, 0.03, sample_rate)),
            env: AGC_TARGET,
            agc_gain: 1.0,
            env_att: tc(0.008),
            env_rel: tc(1.0),
            gain_down: tc(0.05),
            gain_up_fast: tc(4.0),
            gain_up_slow: tc(45.0),
            tts: 0.0,
            tts_up: tc(4.0),
            tts_down: tc(6.0),
            tts_lp: (0.0, 0.0),
            tts_coef: 1.0,
            tts_refresh: 0,
            tin: 0.0,
            tin_phase: 0.0,
            tin_step: core::f32::consts::TAU * TIN_HZ / sample_rate,
            tin_decay: (-1.0 / (2.5 * sample_rate)).exp(),
            sample_rate,
        }
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

    pub fn agc_gain(&self) -> f32 {
        self.agc_gain
    }

    /// Current hearing-fatigue amount, 0 (fresh) … 1 (fully dulled).
    pub fn ear_fatigue(&self) -> f32 {
        self.tts
    }

    /// Current tinnitus excitation, 0 … 1.
    pub fn tinnitus(&self) -> f32 {
        self.tin
    }

    /// Rotate + decode the diffuse bus, mix in point-rendered stereo,
    /// soft-limit.
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

        // ear adaptation
        let m = l.abs().max(r.abs());
        let coef = if m > self.env { self.env_att } else { self.env_rel };
        self.env += coef * (m - self.env);
        let desired = (AGC_TARGET / self.env.max(1e-4)).clamp(AGC_MAX_CUT, AGC_MAX_BOOST);
        // Recovery is exponential in perception: coming back from deep
        // protection is quick at first, but the last ~6 dB of sensitivity
        // returns much later (fast tc while far below the target gain,
        // sliding toward the slow tc as it closes in).
        let gcoef = if desired < self.agc_gain {
            self.gain_down
        } else {
            let r = (self.agc_gain / desired.max(1e-6)).clamp(0.0, 1.0);
            self.gain_up_fast + (self.gain_up_slow - self.gain_up_fast) * r * r
        };
        self.agc_gain += gcoef * (desired - self.agc_gain);

        // hearing fatigue: builds over seconds of ultra-loud demand,
        // clears within ~20 s of relief
        let excess = (self.env / TTS_LOUD - 1.0).clamp(0.0, 1.0);
        let tcoef = if excess > self.tts { self.tts_up } else { self.tts_down };
        self.tts += tcoef * (excess - self.tts);
        if self.tts_refresh == 0 {
            self.tts_refresh = 64;
            let fc = TTS_FC_HI * (TTS_FC_LO / TTS_FC_HI).powf(self.tts);
            self.tts_coef =
                1.0 - (-core::f32::consts::TAU * fc / self.sample_rate).exp();
        }
        self.tts_refresh -= 1;

        // tinnitus excitation: blasts far beyond the fatigue threshold.
        // The ring sits slightly sharp while strong and settles in pitch
        // as it fades (~8 s to silence) — a static tone reads as a bug,
        // a drifting one reads as your ears.
        self.tin = (self.tin + (m - TIN_THRESH).max(0.0) * 2e-4).min(1.0) * self.tin_decay;
        let ring = if self.tin > 1e-4 {
            let step = self.tin_step * (1.0 + 0.08 * self.tin);
            self.tin_phase = (self.tin_phase + step) % core::f32::consts::TAU;
            self.tin_phase.sin() * self.tin * TIN_LEVEL
        } else {
            0.0
        };

        let gl = l * self.agc_gain;
        let gr = r * self.agc_gain;
        // dry/lowpassed blend by fatigue — bit-transparent at tts = 0
        self.tts_lp.0 += self.tts_coef * (gl - self.tts_lp.0);
        self.tts_lp.1 += self.tts_coef * (gr - self.tts_lp.1);
        let ol = gl + self.tts * (self.tts_lp.0 - gl) + ring;
        let or_ = gr + self.tts * (self.tts_lp.1 - gr) + ring;
        (ol.tanh(), or_.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ratio of a 6 kHz probe to a 200 Hz probe through the stage — the
    /// AGC scales both equally, so this isolates the muffle filter.
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
        peak(6000.0) / peak(200.0).max(1e-9)
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
