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

    /// Drop everything ringing (the field it held no longer reaches the
    /// listener; see the renderer's output-scaled late level).
    pub fn clear(&mut self) {
        for l in &mut self.lines {
            l.clear();
        }
        self.damps = [OnePoleLp::default(); NLINES];
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

    /// Most samples `process_block` takes at once: a little under the
    /// shortest loop, so a block's reads (two neighbours each) all come
    /// from before its writes.
    pub fn max_block(&self) -> usize {
        self.lens.iter().fold(f32::MAX, |a, &b| a.min(b)) as usize - 2
    }

    /// `input.len()` (≤ `max_block()`, ≤ N) samples at once: `outs[i][j]`
    /// is line i's output at sample j — what `process` returns sample by
    /// sample. Reads for the whole block first (they precede its writes),
    /// the eight damping loops side by side, the mixing and writes after.
    pub fn process_block<const N: usize>(&mut self, input: &[f32], outs: &mut [[f32; N]; NLINES]) {
        let m = input.len().min(N).min(self.max_block());
        for (i, o) in outs.iter_mut().enumerate() {
            self.lines[i].read_ahead_block(self.lens[i], &mut o[..m]);
        }
        let mut damp: [f32; NLINES] = core::array::from_fn(|i| self.damps[i].state);
        // the loop filters' and gains' smoothers as plain lanes (`tick`
        // inlined), so the eight loops vectorize side by side
        let lanes = |g: &[Smoothed; NLINES]| -> [[f32; NLINES]; 3] {
            [
                core::array::from_fn(|i| g[i].current()),
                core::array::from_fn(|i| g[i].target_val()),
                core::array::from_fn(|i| g[i].coef()),
            ]
        };
        let [mut dc, dt, dk] = lanes(&self.damp_coef);
        let [mut gc, gt, gk] = lanes(&self.gains);
        for j in 0..m {
            for i in 0..NLINES {
                dc[i] = dt[i] + (dc[i] - dt[i]) * dk[i];
                gc[i] = gt[i] + (gc[i] - gt[i]) * gk[i];
                // OnePoleLp::tick
                damp[i] += dc[i] * (outs[i][j] - damp[i]);
                outs[i][j] = damp[i] * gc[i];
            }
        }
        for i in 0..NLINES {
            self.damp_coef[i].set_current(dc[i]);
            self.gains[i].set_current(gc[i]);
            self.damps[i].state = damp[i];
        }
        // 8×8 Hadamard per sample (as in `process`), vectorized over samples
        let mut v = *outs;
        for stride in [1usize, 2, 4] {
            for lo in 0..NLINES {
                if lo & stride != 0 {
                    continue;
                }
                let hi = lo | stride;
                for j in 0..m {
                    let (a, b) = (v[lo][j], v[hi][j]);
                    v[lo][j] = a + b;
                    v[hi][j] = a - b;
                }
            }
        }
        let norm = 1.0 / (NLINES as f32).sqrt();
        for (i, v) in v.iter_mut().enumerate() {
            for (v, x) in v[..m].iter_mut().zip(&input[..m]) {
                *v = x * 0.35 + *v * norm;
            }
            self.lines[i].write_block(&v[..m]);
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
