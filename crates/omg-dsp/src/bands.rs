//! Three-band splitter matching the simulation's band layout
//! (BAND_CENTER_HZ = 125 / 790 / 5600): two one-pole crossovers at the
//! geometric band edges. Perfect reconstruction (lo + mid + hi = input),
//! so unity gains are transparent — apply the sim's per-band amplitudes
//! and sum.

/// Geometric means of adjacent band centers.
const XOVER_LO_HZ: f32 = 314.0;
const XOVER_HI_HZ: f32 = 2100.0;

#[derive(Clone, Copy)]
pub struct BandSplit {
    c_lo: f32,
    c_hi: f32,
    lp_lo: f32,
    lp_hi: f32,
}

impl BandSplit {
    pub fn new(sample_rate: f32) -> Self {
        let coef = |fc: f32| 1.0 - (-core::f32::consts::TAU * fc / sample_rate).exp();
        Self { c_lo: coef(XOVER_LO_HZ), c_hi: coef(XOVER_HI_HZ), lp_lo: 0.0, lp_hi: 0.0 }
    }

    /// One sample in → shaped sample out, with per-band amplitude gains.
    #[inline]
    pub fn process(&mut self, x: f32, gains: &[f32; 3]) -> f32 {
        self.lp_lo += self.c_lo * (x - self.lp_lo);
        self.lp_hi += self.c_hi * (x - self.lp_hi);
        gains[0] * self.lp_lo + gains[1] * (self.lp_hi - self.lp_lo) + gains[2] * (x - self.lp_hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_gains_are_transparent() {
        let mut s = BandSplit::new(48_000.0);
        let mut x = 0.7f32;
        for _ in 0..1000 {
            let y = s.process(x, &[1.0, 1.0, 1.0]);
            assert!((y - x).abs() < 1e-6);
            x = -x * 0.99;
        }
    }

    #[test]
    fn band_gains_shape_the_spectrum() {
        // low-band-only must pass a 100 Hz tone and kill an 8 kHz tone
        let energy = |freq: f32, gains: [f32; 3]| -> f32 {
            let mut s = BandSplit::new(48_000.0);
            let mut e = 0.0f32;
            for k in 0..48_000 {
                let x = (core::f32::consts::TAU * freq * k as f32 / 48_000.0).sin();
                let y = s.process(x, &gains);
                if k > 4800 {
                    e += y * y;
                }
            }
            e
        };
        let lo_only = [1.0, 0.0, 0.0];
        assert!(energy(100.0, lo_only) > 50.0 * energy(8000.0, lo_only));
        let hi_only = [0.0, 0.0, 1.0];
        assert!(energy(8000.0, hi_only) > 50.0 * energy(100.0, hi_only));
    }
}
