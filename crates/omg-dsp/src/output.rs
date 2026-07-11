//! The single decode stage for the summed SH bus of all sources: rotates
//! the field by (smoothed) head yaw, then decodes — measured-HRIR binaural
//! when speaker HRIRs are provided, virtual-cardioid stereo otherwise —
//! and mixes in the point-rendered stereo. Shared by native and web builds.

use crate::ambi::{rotate_z, StereoDecoder, NCH};
use crate::hrtf::BinauralDecoder;
use crate::smooth::Smoothed;

enum Decode {
    Hrtf(BinauralDecoder),
    Cardioid(StereoDecoder),
}

pub struct OutputStage {
    decode: Decode,
    head_yaw: Smoothed,
    // "HDR audio": slow automatic gain riding, like the ear adapting —
    // loud scenes get pulled down quickly (acoustic reflex), quiet scenes
    // get eased up slowly, tanh stays as the final safety.
    env: f32,
    agc_gain: f32,
    env_att: f32,
    env_rel: f32,
    gain_down: f32,
    gain_up: f32,
    // Temporary threshold shift ("club ears"): sustained demand far above
    // the AGC target accumulates fatigue; fatigue muffles the output —
    // a lowpass that deepens with exposure and lets go over ~25 s, the
    // way hearing stays dulled after stepping away from a loud PA.
    // Protection-shaped only: exactly zero effect at normal levels.
    tts: f32,
    tts_up: f32,
    tts_down: f32,
    tts_lp: (f32, f32),
    tts_coef: f32,
    tts_refresh: u32,
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

impl OutputStage {
    pub fn from_speaker_bytes(bytes: Option<&[u8]>, sample_rate: f32) -> Self {
        let decode = match bytes {
            Some(b) => Decode::Hrtf(BinauralDecoder::from_bytes(b)),
            None => Decode::Cardioid(StereoDecoder::new()),
        };
        let tc = |tau: f32| 1.0 - (-1.0 / (tau * sample_rate)).exp();
        Self {
            decode,
            head_yaw: Smoothed::new(0.0, 0.03, sample_rate),
            env: AGC_TARGET,
            agc_gain: 1.0,
            env_att: tc(0.008),
            env_rel: tc(1.0),
            gain_down: tc(0.05),
            gain_up: tc(30.0),
            tts: 0.0,
            tts_up: tc(4.0),
            tts_down: tc(25.0),
            tts_lp: (0.0, 0.0),
            tts_coef: 1.0,
            tts_refresh: 0,
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

    pub fn set_head_yaw(&mut self, yaw: f32) {
        self.head_yaw.set(yaw);
    }

    pub fn agc_gain(&self) -> f32 {
        self.agc_gain
    }

    /// Current hearing-fatigue amount, 0 (fresh) … 1 (fully dulled).
    pub fn ear_fatigue(&self) -> f32 {
        self.tts
    }

    /// Rotate + decode the diffuse bus, mix in point-rendered stereo,
    /// soft-limit.
    #[inline]
    pub fn process(&mut self, sh: &[f32; NCH], pl: f32, pr: f32) -> (f32, f32) {
        let mut rotated = *sh;
        rotate_z(&mut rotated, self.head_yaw.tick());
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
        let gcoef = if desired < self.agc_gain { self.gain_down } else { self.gain_up };
        self.agc_gain += gcoef * (desired - self.agc_gain);

        // hearing fatigue: builds over seconds of ultra-loud demand,
        // releases over ~25 s of relief
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

        let gl = l * self.agc_gain;
        let gr = r * self.agc_gain;
        // dry/lowpassed blend by fatigue — bit-transparent at tts = 0
        self.tts_lp.0 += self.tts_coef * (gl - self.tts_lp.0);
        self.tts_lp.1 += self.tts_coef * (gr - self.tts_lp.1);
        let ol = gl + self.tts * (self.tts_lp.0 - gl);
        let or_ = gr + self.tts * (self.tts_lp.1 - gr);
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

        // 8 s of club-at-the-PA level (demand ≫ AGC target)
        for k in 0..48_000 * 8 {
            let x = 2.5 * (core::f32::consts::TAU * 700.0 * k as f32 / 48_000.0).sin();
            out.process(&[0.0; NCH], x, x);
        }
        assert!(out.ear_fatigue() > 0.5, "fatigue {}", out.ear_fatigue());
        let dulled = hf_lf_ratio(&mut out);
        assert!(
            dulled < 0.6 * fresh,
            "highs should dull after exposure: {dulled} vs fresh {fresh}"
        );

        // ~50 s of near-silence: hearing comes back
        for _ in 0..48_000 * 50 {
            out.process(&[0.0; NCH], 0.0, 0.0);
        }
        let later = hf_lf_ratio(&mut out);
        assert!(
            later > 0.85 * fresh,
            "should recover: {later} vs fresh {fresh}"
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
    }
}
