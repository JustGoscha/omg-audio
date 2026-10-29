//! Three-band splitter matching the simulation's band layout
//! (BAND_CENTER_HZ = 125 / 790 / 5600): Linkwitz–Riley crossovers (4th
//! order, 24 dB/octave) at the geometric band edges, the low band through
//! the upper crossover's all-pass so the three sum to an all-pass — unity
//! gains keep the magnitude flat. Apply the sim's per-band amplitudes and
//! sum: each band centre plays its band's gain.
//!
//! (One-pole crossovers, 6 dB/octave, let the loudest band's skirt over
//! the others: a wall's transmission — bass 0, mids −14, treble −34 dB —
//! played its mids at −7 and its treble at −18 dB, a closed door sounding
//! far brighter than traced.)

/// Geometric means of adjacent band centers.
pub const XOVER_LO_HZ: f32 = 314.0;
pub const XOVER_HI_HZ: f32 = 2100.0;

/// A second-order section, transposed direct form II.
#[derive(Clone, Copy, Default)]
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
}

/// Butterworth (Q = 1/√2) low-pass, high-pass or all-pass coefficients
/// (RBJ), normalized: (b0, b1, b2, a1, a2).
fn coefs(kind: u8, fc: f32, sample_rate: f32) -> ([f64; 3], [f64; 2]) {
    let w = core::f64::consts::TAU * fc as f64 / sample_rate as f64;
    let (cw, sw) = (w.cos(), w.sin());
    let alpha = sw / (2.0 * core::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (a1, a2) = (-2.0 * cw / a0, (1.0 - alpha) / a0);
    let b = match kind {
        0 => [(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0],
        1 => [(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0],
        _ => [1.0 - alpha, -2.0 * cw, 1.0 + alpha],
    };
    ([b[0] / a0, b[1] / a0, b[2] / a0], [a1, a2])
}

impl Biquad {
    fn new(kind: u8, fc: f32, sample_rate: f32) -> Self {
        let (b, a) = coefs(kind, fc, sample_rate);
        Self { b: b.map(|v| v as f32), a: a.map(|v| v as f32), z: [0.0; 2] }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// A complex number (re, im).
type C = (f64, f64);

fn mul(x: C, y: C) -> C {
    (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0)
}

/// A section's response at angular frequency `w`.
fn section(kind: u8, fc: f32, sample_rate: f32, w: f64) -> C {
    let (b, a) = coefs(kind, fc, sample_rate);
    let z1: C = (w.cos(), -w.sin());
    let z2: C = mul(z1, z1);
    let num = (b[0] + b[1] * z1.0 + b[2] * z2.0, b[1] * z1.1 + b[2] * z2.1);
    let den = (1.0 + a[0] * z1.0 + a[1] * z2.0, a[0] * z1.1 + a[1] * z2.1);
    let d = den.0 * den.0 + den.1 * den.1;
    ((num.0 * den.0 + num.1 * den.1) / d, (num.1 * den.0 - num.0 * den.1) / d)
}

/// The three bands' complex responses at `f` Hz (they sum to an
/// all-pass): low = AP₂·LP₁², mid = LP₂²·HP₁², high = HP₂²·HP₁².
pub fn responses(sample_rate: f32, f: f32) -> [(f32, f32); 3] {
    let w = core::f64::consts::TAU * f as f64 / sample_rate as f64;
    let s = |k: u8, fc: f32| section(k, fc, sample_rate, w);
    let sq = |x: C| mul(x, x);
    let (lp1, hp1, lp2, hp2, ap2) = (s(0, XOVER_LO_HZ), s(1, XOVER_LO_HZ), s(0, XOVER_HI_HZ), s(1, XOVER_HI_HZ), s(2, XOVER_HI_HZ));
    let lo = mul(ap2, sq(lp1));
    let mid = mul(sq(lp2), sq(hp1));
    let hi = mul(sq(hp2), sq(hp1));
    [lo, mid, hi].map(|c| (c.0 as f32, c.1 as f32))
}

#[derive(Clone, Copy)]
pub struct BandSplit {
    lp1: [Biquad; 2],
    hp1: [Biquad; 2],
    lp2: [Biquad; 2],
    hp2: [Biquad; 2],
    ap2: Biquad,
}

impl BandSplit {
    pub fn new(sample_rate: f32) -> Self {
        let q = |k: u8, fc: f32| Biquad::new(k, fc, sample_rate);
        Self {
            lp1: [q(0, XOVER_LO_HZ); 2],
            hp1: [q(1, XOVER_LO_HZ); 2],
            lp2: [q(0, XOVER_HI_HZ); 2],
            hp2: [q(1, XOVER_HI_HZ); 2],
            ap2: q(2, XOVER_HI_HZ),
        }
    }

    /// One sample in → its three bands (low, mid, high; they sum to an
    /// all-pass of it).
    #[inline]
    pub fn split(&mut self, x: f32) -> [f32; 3] {
        let two = |q: &mut [Biquad; 2], x: f32| {
            let y = q[0].tick(x);
            q[1].tick(y)
        };
        let l = two(&mut self.lp1, x);
        let lo = self.ap2.tick(l);
        let r = two(&mut self.hp1, x);
        let mid = two(&mut self.lp2, r);
        let hi = two(&mut self.hp2, r);
        [lo, mid, hi]
    }

    /// One sample in → shaped sample out, the low and high bands at
    /// `g_low`, `g_high` times the mid's — the bands as the kernel
    /// synthesizes them (`dense.rs`), so whatever is shaped here sounds
    /// like the same energies convolved.
    #[inline]
    pub fn shape(&mut self, x: f32, g_low: f32, g_high: f32) -> f32 {
        let [lo, mid, hi] = self.split(x);
        g_low * lo + mid + g_high * hi
    }

    /// Every internal state below `eps` (fed silence, nothing left).
    pub fn quiet(&self, eps: f32) -> bool {
        let q = |b: &Biquad| b.z[0].abs() < eps && b.z[1].abs() < eps;
        self.lp1.iter().chain(&self.hp1).chain(&self.lp2).chain(&self.hp2).all(q) && q(&self.ap2)
    }

    /// One sample in → shaped sample out, with per-band amplitude gains.
    #[inline]
    pub fn process(&mut self, x: f32, gains: &[f32; 3]) -> f32 {
        let [lo, mid, hi] = self.split(x);
        gains[0] * lo + gains[1] * mid + gains[2] * hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steady-state amplitude of a tone through `gains`.
    fn tone(freq: f32, gains: [f32; 3]) -> f32 {
        let mut s = BandSplit::new(48_000.0);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let n0 = 24_000;
        for k in 0..48_000 {
            let ph = core::f64::consts::TAU * freq as f64 * k as f64 / 48_000.0;
            let y = s.process(ph.sin() as f32, &gains);
            if k >= n0 {
                re += y as f64 * ph.sin();
                im += y as f64 * ph.cos();
            }
        }
        (2.0 * (re * re + im * im).sqrt() / 24_000.0) as f32
    }

    #[test]
    fn unity_gains_keep_the_magnitude() {
        for f in [50.0f32, 125.0, 314.0, 790.0, 2100.0, 5600.0, 15_000.0] {
            let a = tone(f, [1.0; 3]);
            assert!((a - 1.0).abs() < 0.01, "{f} Hz: {a}");
        }
    }

    #[test]
    fn band_centres_play_their_band_gains() {
        // a wall's transmission: bass 0, mids −14, treble −34 dB — each
        // centre within 1.5 dB (the treble's: the mids' skirt 1.4 octaves
        // up, −34 dB, under a band 20 dB quieter: +1.2 dB; one-pole bands
        // put it +16 dB)
        let g = [1.0f32, 0.2, 0.02];
        for (k, f) in [125.0f32, 790.0, 5600.0].into_iter().enumerate() {
            let a = tone(f, g);
            let d = 20.0 * (a / g[k]).log10();
            assert!(d.abs() < 1.5, "{f} Hz: {d:+.1} dB off its band's gain");
        }
        // the computed responses agree with the filters
        for f in [125.0f32, 790.0, 5600.0] {
            let r = responses(48_000.0, f);
            let (re, im) = r.iter().zip(g).fold((0.0, 0.0), |a, (c, g)| (a.0 + c.0 * g, a.1 + c.1 * g));
            let a = (re * re + im * im).sqrt();
            assert!((a / tone(f, g) - 1.0).abs() < 0.01, "{f} Hz: computed {a}");
        }
    }
}
