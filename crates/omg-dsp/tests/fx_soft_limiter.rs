//! Effect: final tanh soft limiter in `OutputStage::process`.
//!
//! Claim: whatever reaches the output stage, the samples leaving it are
//! bounded to |y| ≤ 1 (no wrap, no NaN), while ordinary
//! program level passes ~linearly (tanh(x) ≈ x for small x) — the
//! limiter is a safety, not a coloration.
//!
//! Isolation: cardioid `OutputStage` (no HRIR), input injected on the
//! point-stereo inputs with an empty bus. The ear's adaptation sits
//! upstream; a safety limiter (instant on a peak over 0.9) holds the line
//! before it has reacted (first samples) and after, the tanh is the last
//! guard. Not duplicated: adaptation/reflex/fatigue/tinnitus have their
//! own unit tests in src/output.rs.
//! Combined in the demo: every build's master output.

mod common;
use common::*;
use omg_dsp::ambi::NCH;
use omg_dsp::output::OutputStage;

#[test]
fn huge_input_is_bounded() {
    let mut out = OutputStage::from_speaker_bytes(None, SR);
    out.settle();
    let mut peak_early = 0.0f32;
    for k in 0..48_000 {
        let x = 100.0 * sine_at(90.0, k + 20); // full-scale ×100, from sample 0
        let (l, r) = out.process(&[0.0; NCH], x, -x);
        assert!(l.is_finite() && r.is_finite(), "k {k}");
        assert!(l.abs() <= 1.0 && r.abs() <= 1.0, "k {k}: {l} {r}");
        // (no exact L = −R check: the diotic tinnitus ring a blast like
        // this excites is added to both ears before the tanh)
        if k < 48 {
            peak_early = peak_early.max(l.abs());
        }
    }
    // before the adaptation has reacted, the limiter holds the line at
    // once: the output plays at its ceiling (tanh 0.9 ≈ 0.72), no wrap
    assert!(peak_early > 0.6, "the limiter plays at its ceiling early: {peak_early}");
}

#[test]
fn small_input_is_linear() {
    // first 10 ms: the AGC gain has moved < 0.2 %, so out/in is the
    // limiter's small-signal slope, ≈ 1
    for amp in [0.01f32, 0.05, 0.1] {
        let mut out = OutputStage::from_speaker_bytes(None, SR);
        out.settle();
        let mut worst = 0.0f32;
        for k in 0..480 {
            let x = amp * sine_at(440.0, k);
            let (l, _) = out.process(&[0.0; NCH], x, x);
            if x.abs() > 0.5 * amp {
                worst = worst.max((l / x - 1.0).abs());
            }
        }
        // tanh(0.1) / 0.1 = 0.9967; allow AGC drift on top
        assert!(worst < 0.006, "amp {amp}: max deviation from linear {worst}");
    }
}
