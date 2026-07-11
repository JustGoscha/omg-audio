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
}

const AGC_TARGET: f32 = 0.35;
const AGC_MAX_BOOST: f32 = 1.15;
const AGC_MAX_CUT: f32 = 0.06;

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

        ((l * self.agc_gain).tanh(), (r * self.agc_gain).tanh())
    }
}
