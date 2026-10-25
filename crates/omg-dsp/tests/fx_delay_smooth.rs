//! Effect: the two primitives every moving parameter rides on —
//! `DelayLine` (delay.rs: power-of-two ring, linear-interpolated
//! fractional read) and `Smoothed` (smooth.rs: one-pole parameter
//! smoother).
//!
//! Claims:
//!  - `read(d)` returns the input `d` samples ago (write-then-read:
//!    integer d is sample-exact; fractional d linearly interpolates, so
//!    a ramp is exact and a tone is within the (ωT)²/8 interpolation
//!    bound);
//!  - delays are clamped to [1, len − 2], capacity is the next power of
//!    two ≥ the requested length;
//!  - `Smoothed` reaches 1 − 1/e of a step after exactly τ, `snap`
//!    jumps, `set` doesn't move `current` until ticked.
//!
//! Isolation: the structs alone. Combined: every renderer test (Doppler
//! glide, FDN lines, tap gains) builds on these; see fx_doppler.rs.

mod common;
use common::*;
use omg_dsp::delay::DelayLine;
use omg_dsp::smooth::Smoothed;

#[test]
fn integer_and_fractional_reads() {
    let mut dl = DelayLine::new(1000); // → 1024
    for n in 0..5000 {
        dl.write(n as f32);
        if n > 1100 {
            for d in [1.0f32, 2.0, 17.0, 500.0, 1022.0] {
                assert_eq!(dl.read(d), n as f32 - d, "n {n}, integer delay {d}");
            }
            for d in [1.25f32, 7.5, 333.875, 1021.5] {
                // values here stay < 2^13, so the ramp is exact in f32
                assert!((dl.read(d) - (n as f32 - d)).abs() < 1e-3, "n {n}, frac delay {d}");
            }
        }
    }
}

#[test]
fn fractional_tone_read_within_interp_bound() {
    for f in [200.0f32, 1000.0, 4000.0] {
        let mut dl = DelayLine::new(4096);
        let w = std::f32::consts::TAU * f / SR;
        let bound = w * w / 8.0 + 1e-5;
        let mut worst = 0.0f32;
        for n in 0..8000 {
            dl.write(sine_at(f, n));
            if n > 4000 {
                let d = 100.37f32;
                let want = ((std::f64::consts::TAU * f as f64 * (n as f64 - d as f64)) / SR as f64).sin() as f32;
                worst = worst.max((dl.read(d) - want).abs());
            }
        }
        assert!(worst <= bound, "{f} Hz: err {worst} > bound {bound}");
    }
}

#[test]
fn delay_is_clamped_to_the_buffer() {
    let mut dl = DelayLine::new(100); // → 128
    for n in 0..300 {
        dl.write(n as f32);
    }
    assert_eq!(dl.read(0.2), 298.0, "below 1 clamps to 1");
    assert_eq!(dl.read(10_000.0), 299.0 - 126.0, "above len−2 clamps to 126");
}

#[test]
fn smoother_time_constant() {
    for tau in [0.01f32, 0.05, 0.2] {
        let mut s = Smoothed::new(0.0, tau, SR);
        s.set(1.0);
        assert_eq!(s.current(), 0.0, "set alone must not move current");
        assert_eq!(s.target_val(), 1.0);
        let n = (tau * SR).round() as usize;
        let mut v = 0.0;
        for _ in 0..n {
            v = s.tick();
        }
        assert!((v - (1.0 - (-1.0f32).exp())).abs() < 2e-3, "tau {tau}: {v} after τ");
        for _ in 0..(10 * n) {
            v = s.tick();
        }
        // f32 dead band: the one-pole stops moving once
        // |cur − target|·(1 − coef) < ½ ulp — ≈ 3e-4 for τ = 0.2 s at
        // unit level (harmless for gains, −70 dB)
        assert!((v - 1.0).abs() < 5e-4, "tau {tau}: settles, {v}");
        s.snap(-3.0);
        assert_eq!((s.current(), s.target_val()), (-3.0, -3.0));
        assert_eq!(s.tick(), -3.0);
    }
}
