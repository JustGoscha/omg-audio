//! Near-field ear stage: a sound source a few centimetres from one ear
//! (a whisper), rendered straight into the two ear signals and bypassing
//! propagation entirely.
//!
//! HRTFs are far-field measurements (≥ 1 m); inside that radius the
//! inverse-distance law ACROSS THE HEAD dominates anything the grid can
//! express: a whisper 6 cm off the left ear is ~14 dB louder there than
//! at the right ear before head shadow, and the shadow then removes most
//! of a whisper's sibilant band on top — which is why a whisper into one
//! ear is near-inaudible at the other.
//!
//! Model, per voice:
//!  - per-ear 1/r, the far ear's path measured around the head
//!    (`HEAD_DETOUR_M` added to the distance);
//!  - a fixed interaural delay (`ITD_S`) on the far ear;
//!  - a one-pole head-shadow lowpass (`SHADOW_HZ`) on the far ear;
//!  - a linear onset declick over `DECLICK` samples.
//!
//! Room send is omitted on purpose — at whisper level the reflections sit
//! far below audibility. The engine runs this stage POST-AGC: a whisper's
//! whole identity is its absolute closeness, and the scene's loudness
//! governor must not ride its level up or down.

/// Simultaneous near-field voices the engine keeps (further plays drop).
pub const MAX_VOICES: usize = 4;
/// Reference distance: `gain` IS the near-ear amplitude at this distance.
pub const REF_DIST_M: f32 = 0.06;
/// Distance clamp range (mouth-to-ear), metres.
pub const MIN_DIST_M: f32 = 0.02;
pub const MAX_DIST_M: f32 = 0.6;
/// Extra path length to the far ear: the detour around the head.
pub const HEAD_DETOUR_M: f32 = 0.25;
/// Far-ear interaural time difference, seconds (~0.65 ms around the head).
pub const ITD_S: f32 = 0.00065;
/// Head-shadow corner: the whisper's sibilance (2–8 kHz) dies crossing
/// the head, the low murmur survives.
pub const SHADOW_HZ: f32 = 700.0;
/// Onset declick ramp length, samples.
pub const DECLICK: usize = 256;

const RING: usize = 64;
const RING_MASK: usize = RING - 1;

/// One near-field voice's ear processing (no buffer ownership: the caller
/// feeds it one mono sample at a time and mixes the returned ear pair).
#[derive(Clone)]
pub struct NearVoice {
    right: bool,
    near_g: f32,
    far_g: f32,
    /// far-ear head-shadow one-pole coefficient + state
    lp_k: f32,
    lp: f32,
    /// far-ear ITD ring
    dl: [f32; RING],
    dpos: usize,
    itd: usize,
    /// declick ramp, samples remaining
    fade: usize,
}

impl NearVoice {
    /// A voice `dist_m` from one ear (`right` = right ear). `gain` is the
    /// near-ear amplitude at `REF_DIST_M`; the far ear derives from the
    /// around-the-head path. `dist_m` is clamped to
    /// [`MIN_DIST_M`, `MAX_DIST_M`].
    pub fn new(sample_rate: f32, right: bool, dist_m: f32, gain: f32) -> Self {
        let d = dist_m.clamp(MIN_DIST_M, MAX_DIST_M);
        Self {
            right,
            near_g: gain * REF_DIST_M / d,
            far_g: gain * REF_DIST_M / (d + HEAD_DETOUR_M),
            lp_k: (-2.0 * core::f32::consts::PI * SHADOW_HZ / sample_rate).exp(),
            lp: 0.0,
            dl: [0.0; RING],
            dpos: 0,
            itd: ((ITD_S * sample_rate) as usize).clamp(1, RING - 1),
            fade: DECLICK,
        }
    }

    /// True when the voice sits at the right ear.
    pub fn is_right(&self) -> bool {
        self.right
    }

    /// Near-ear amplitude (linear).
    pub fn near_gain(&self) -> f32 {
        self.near_g
    }

    /// Far-ear amplitude before head shadow (linear).
    pub fn far_gain(&self) -> f32 {
        self.far_g
    }

    /// Far-ear delay in samples.
    pub fn itd_samples(&self) -> usize {
        self.itd
    }

    /// One mono sample in → (left, right) ear contributions.
    #[inline]
    pub fn process(&mut self, x: f32) -> (f32, f32) {
        let mut x = x;
        if self.fade > 0 {
            x *= 1.0 - self.fade as f32 / DECLICK as f32;
            self.fade -= 1;
        }
        let near = x * self.near_g;
        self.dl[self.dpos] = x;
        let far_x = self.dl[(self.dpos + RING - self.itd) & RING_MASK];
        self.dpos = (self.dpos + 1) & RING_MASK;
        self.lp = self.lp * self.lp_k + far_x * (1.0 - self.lp_k);
        let far = self.lp * self.far_g;
        if self.right {
            (far, near)
        } else {
            (near, far)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Render `x` through a voice; returns (near ear, far ear).
    fn render(v: &mut NearVoice, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let right = v.is_right();
        let (mut near, mut far) = (Vec::with_capacity(x.len()), Vec::with_capacity(x.len()));
        for &s in x {
            let (l, r) = v.process(s);
            if right {
                near.push(r);
                far.push(l);
            } else {
                near.push(l);
                far.push(r);
            }
        }
        (near, far)
    }

    fn sine(f: f32, n: usize) -> Vec<f32> {
        (0..n).map(|k| (core::f32::consts::TAU * f * k as f32 / SR).sin()).collect()
    }

    fn peak(xs: &[f32]) -> f32 {
        xs.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    /// Steady-state DC: the shadow lowpass has unity DC gain, so the ear
    /// ratio is exactly the 1/r law with the far ear +0.25 m around the
    /// head — and the near ear is exactly `gain` at the 6 cm reference.
    #[test]
    fn near_ear_louder_by_the_around_the_head_1_over_r_law() {
        for &d in &[0.02f32, 0.06, 0.15, 0.4, 0.6] {
            let gain = 0.5;
            let mut v = NearVoice::new(SR, false, d, gain);
            let (near, far) = render(&mut v, &vec![1.0; 4000]);
            let (n_ss, f_ss) = (near[3999], far[3999]);
            assert!((n_ss - gain * 0.06 / d).abs() < 1e-6, "d={d}: near {n_ss}");
            assert!((f_ss - gain * 0.06 / (d + 0.25)).abs() < 1e-5, "d={d}: far {f_ss}");
            let ratio = n_ss / f_ss;
            let want = (d + 0.25) / d;
            assert!((ratio / want - 1.0).abs() < 1e-3, "d={d}: ratio {ratio} want {want}");
        }
        // the documented "~14 dB louder at 6 cm"
        let v = NearVoice::new(SR, false, 0.06, 1.0);
        let db = 20.0 * (v.near_gain() / v.far_gain()).log10();
        assert!((db - 14.26).abs() < 0.05, "6 cm ILD {db} dB");
        assert!((v.near_gain() - 1.0).abs() < 1e-6, "gain is the 6 cm near level");
    }

    #[test]
    fn distance_is_clamped() {
        let a = NearVoice::new(SR, false, 0.001, 1.0);
        let b = NearVoice::new(SR, false, MIN_DIST_M, 1.0);
        assert_eq!(a.near_gain(), b.near_gain());
        assert_eq!(a.far_gain(), b.far_gain());
        let a = NearVoice::new(SR, false, 9.0, 1.0);
        let b = NearVoice::new(SR, false, MAX_DIST_M, 1.0);
        assert_eq!(a.near_gain(), b.near_gain());
        assert_eq!(a.far_gain(), b.far_gain());
    }

    /// Cross-correlating broadband noise between the ears: the peak sits
    /// exactly at the ITD (the one-pole's impulse response peaks at lag 0,
    /// so the shadow filter adds no peak shift), ≈ 0.65 ms.
    #[test]
    fn far_ear_delayed_by_itd() {
        for &sr in &[44_100.0f32, 48_000.0, 96_000.0] {
            let x = noise(8192, 3);
            let mut v = NearVoice::new(sr, true, 0.06, 1.0);
            let mut near = Vec::new();
            let mut far = Vec::new();
            for &s in &x {
                let (l, r) = v.process(s);
                near.push(r);
                far.push(l);
            }
            let (mut best, mut best_lag) = (f32::MIN, 0usize);
            for lag in 0..64 {
                let c: f32 = (0..8192 - 64).map(|k| near[k] * far[k + lag]).sum();
                if c > best {
                    best = c;
                    best_lag = lag;
                }
            }
            assert_eq!(best_lag, v.itd_samples(), "sr {sr}");
            let ms = best_lag as f32 / sr * 1e3;
            assert!((ms - 0.65).abs() < 0.03, "sr {sr}: ITD {ms} ms");
        }
    }

    /// Head shadow: the far/near ratio collapses at 6 kHz relative to
    /// 200 Hz by the one-pole's response at 700 Hz corner; the near ear
    /// is flat.
    #[test]
    fn far_ear_is_darker_by_the_700hz_shadow() {
        let ratio_at = |f: f32| {
            let mut v = NearVoice::new(SR, false, 0.06, 1.0);
            let (near, far) = render(&mut v, &sine(f, 24_000));
            let tail = 12_000..24_000;
            (peak(&near[tail.clone()]), peak(&far[tail]))
        };
        let (n_lo, f_lo) = ratio_at(200.0);
        let (n_hi, f_hi) = ratio_at(6000.0);
        // near ear: flat (gain 1 at 6 cm)
        assert!((n_lo - 1.0).abs() < 1e-3 && (n_hi - 1.0).abs() < 2e-2, "near {n_lo} {n_hi}");
        let k = (-core::f32::consts::TAU * SHADOW_HZ / SR).exp();
        let h = |f: f32| {
            let w = core::f32::consts::TAU * f / SR;
            (1.0 - k) / ((1.0 - k * w.cos()).powi(2) + (k * w.sin()).powi(2)).sqrt()
        };
        let shadow = (f_hi / n_hi) / (f_lo / n_lo);
        let want = h(6000.0) / h(200.0);
        assert!((shadow / want - 1.0).abs() < 0.05, "shadow {shadow} want {want}");
        assert!(shadow < 0.15, "6 kHz must lose ≥ 16 dB vs 200 Hz at the far ear: {shadow}");
    }

    /// The onset ramps linearly over 256 samples: the first output is
    /// exactly silent and no sample-to-sample step exceeds gain/256 — a
    /// step input lands as a ramp, not a click.
    #[test]
    fn onset_is_declicked() {
        let g = 0.8;
        let mut v = NearVoice::new(SR, false, 0.06, g);
        let (near, far) = render(&mut v, &vec![1.0; 600]);
        assert_eq!(near[0], 0.0);
        assert_eq!(far[0], 0.0);
        let max_step = near.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_step <= g / DECLICK as f32 + 1e-6, "step {max_step}");
        assert!((near[128] - g * 0.5).abs() < 1e-5, "half-way {}", near[128]);
        assert!((near[DECLICK] - g).abs() < 1e-6 && (near[599] - g).abs() < 1e-6);
        let max_far_step = far.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_far_step <= v.far_gain() / DECLICK as f32 + 1e-6, "far step {max_far_step}");
    }

    /// Left and right voices are exact mirror images.
    #[test]
    fn left_right_mirror_symmetry() {
        let x = noise(3000, 11);
        let mut a = NearVoice::new(SR, false, 0.1, 0.7);
        let mut b = NearVoice::new(SR, true, 0.1, 0.7);
        for &s in &x {
            let (al, ar) = a.process(s);
            let (bl, br) = b.process(s);
            assert_eq!(al.to_bits(), br.to_bits());
            assert_eq!(ar.to_bits(), bl.to_bits());
        }
    }
}
