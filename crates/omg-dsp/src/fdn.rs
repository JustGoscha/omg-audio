//! 8-line feedback delay network (Jot-style) for the late field.
//! Per-line gain sets the mid-band RT60; a one-pole lowpass in each loop
//! makes high frequencies decay faster, matched to the traced rt60[high].
//! An FDN is preferred over convolving a traced impulse response because
//! its parameters can change smoothly in real time with zero artifacts.

use crate::delay::DelayLine;
use crate::filter::OnePoleLp;
use crate::smooth::Smoothed;

pub const NLINES: usize = 8;

/// Mutually prime-ish lengths, ~23–81 ms at 48 kHz, scaled for actual rate.
const LEN_MS: [f32; NLINES] = [23.4, 32.5, 39.7, 46.9, 55.9, 62.9, 72.0, 81.4];

pub struct Fdn {
    lines: Vec<DelayLine>,
    lens: [f32; NLINES],
    damps: [OnePoleLp; NLINES],
    gains: [Smoothed; NLINES],
    damp_coef: [Smoothed; NLINES],
    sample_rate: f32,
}

impl Fdn {
    pub fn new(sample_rate: f32) -> Self {
        let lens: [f32; NLINES] = core::array::from_fn(|i| LEN_MS[i] * 1e-3 * sample_rate);
        Self {
            lines: (0..NLINES)
                .map(|i| DelayLine::new(lens[i] as usize + 4))
                .collect(),
            lens,
            damps: [OnePoleLp::default(); NLINES],
            gains: [Smoothed::new(0.7, 0.1, sample_rate); NLINES],
            damp_coef: [Smoothed::new(1.0, 0.15, sample_rate); NLINES],
            sample_rate,
        }
    }

    /// Configure decay from traced per-band RT60 (bands: low/mid/high).
    pub fn set_rt60(&mut self, rt60_mid: f32, rt60_high: f32) {
        for i in 0..NLINES {
            let len_s = self.lens[i] / self.sample_rate;
            let g_mid = 10f32.powf(-3.0 * len_s / rt60_mid.max(0.05));
            let g_high = 10f32.powf(-3.0 * len_s / rt60_high.max(0.05));
            self.gains[i].set(g_mid.min(0.9999));
            // One-pole in the loop: choose cutoff so gain at ~6 kHz is
            // roughly g_high/g_mid. |H(w)| of one-pole lp ≈ fc/f above fc,
            // so fc ≈ 6 kHz * ratio (clamped) — crude but stable.
            let ratio = (g_high / g_mid).clamp(0.05, 1.0);
            let fc = if ratio >= 0.999 {
                20_000.0
            } else {
                (6_000.0 * ratio / (1.0 - ratio * ratio).sqrt()).clamp(500.0, 20_000.0)
            };
            self.damp_coef[i].set(OnePoleLp::coef(fc, self.sample_rate));
        }
    }

    /// One sample in → 8 decorrelated line outputs (for spatial spread).
    #[inline]
    pub fn process(&mut self, input: f32, outs: &mut [f32; NLINES]) {
        for i in 0..NLINES {
            let raw = self.lines[i].read(self.lens[i]);
            let damped = self.damps[i].tick(raw, self.damp_coef[i].tick());
            outs[i] = damped * self.gains[i].tick();
        }
        // Fast 8×8 Hadamard (butterfly), normalized → energy-preserving.
        let mut v = *outs;
        for stride in [1usize, 2, 4] {
            let mut w = [0.0f32; NLINES];
            for i in 0..NLINES {
                let j = i ^ stride;
                w[i] = if i & stride == 0 { v[i] + v[j] } else { v[j] - v[i] };
            }
            v = w;
        }
        let norm = 1.0 / (NLINES as f32).sqrt();
        for i in 0..NLINES {
            self.lines[i].write(input * 0.35 + v[i] * norm);
        }
    }
}
