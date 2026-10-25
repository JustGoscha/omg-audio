//! Effect: path-identity crossfade (renderer.rs tap lifecycle).
//!
//! Claim: when a propagation path is replaced by a DIFFERENT path (new
//! `key`) or vanishes, the renderer never slides one delay into the
//! other (a pitch chirp) and never hard-switches (a click). The old slot
//! fades out from where it is, the new slot fades in from silence at
//! its own final delay. Measured on a 1 kHz tone: no sample step larger
//! than the tone's own steepest slope, no broadband (2nd-difference)
//! burst, every 10 ms window at the source pitch, and the output lands
//! exactly on the new path. The metrics are self-checked: a synthetic
//! hard switch must fail them.
//!
//! Isolation: bus-only `Renderer::new`, single tap, W channel.
//! Combined in the demo: omg-scene tests/early_gate.rs (zero tap-key
//! churn while stationary), tests/bend_panels.rs and
//! tests/world_traced_gate.rs (step-free walks through re-keying
//! geometry) exercise the key stream that drives this.

mod common;
use common::*;
use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::NCH;
use omg_dsp::Renderer;

const F0: f32 = 1000.0;

fn block(version: u64, taps: &[(u32, f32)]) -> ParamBlock {
    ParamBlock {
        taps: taps
            .iter()
            .map(|&(key, delay_s)| Tap { key, delay_s, dir: [1.0, 0.0, 0.0], gains: [1.0, 1.0, 1.0] })
            .collect(),
        version,
        ..Default::default()
    }
}

/// Tone through key 1 at 10 ms; at t = 0.5 s switch to `after`.
fn render(after: &[(u32, f32)], n: usize) -> Vec<f32> {
    let mut r = Renderer::new(SR);
    r.set_params(&block(1, &[(1, 0.010)]));
    let mut y = Vec::with_capacity(n);
    for k in 0..n {
        if k == 24_000 {
            r.set_params(&block(2, after));
        }
        let mut bus = [0.0f32; NCH];
        r.process(sine_at(F0, k), &mut bus);
        y.push(bus[0]);
    }
    y
}

/// Steady-state reference metrics on the settled tone.
fn steady(y: &[f32]) -> (f32, f64) {
    let s = &y[19_200..24_000];
    (max_step(s), d2_energy(s) / s.len() as f64)
}

fn assert_click_free(y: &[f32], what: &str) {
    let (step_ref, d2_ref) = steady(y);
    let t = &y[24_000..24_000 + 9_600]; // 200 ms after the switch
    assert!(
        max_step(t) <= 1.05 * step_ref,
        "{what}: max step {} vs steady {step_ref}",
        max_step(t)
    );
    for w in t.chunks(480) {
        let d2 = d2_energy(w) / w.len() as f64;
        assert!(d2 <= 1.1 * d2_ref, "{what}: 2nd-difference burst {d2} vs steady {d2_ref}");
    }
}

/// Re-key to a different path 20.25 ms longer (90° out of phase, so the
/// crossfade sum never cancels): click-free, chirp-free, ends exactly on
/// the new path.
#[test]
fn rekey_crossfades_without_click_or_chirp() {
    let d2 = 0.030_25f32;
    let y = render(&[(2, d2)], 72_000);
    assert_click_free(&y, "re-key");

    let j = 24_000;
    for w in 0..20 {
        let f = zc_freq(&y[j + w * 480..j + (w + 1) * 480]);
        assert!((f / F0 - 1.0).abs() < 0.01, "window {w}: {f} Hz — a re-key must not chirp");
    }

    // ~1 s later the old ghost is gone: output = tone at the NEW delay.
    let ds = d2 * SR;
    let err = (67_200..72_000)
        .map(|k| {
            let t = k as f64 - ds as f64;
            (y[k] - (std::f64::consts::TAU * F0 as f64 * t / SR as f64).sin() as f32).abs()
        })
        .fold(0.0, f32::max);
    assert!(err < 3e-3, "should settle on the new path: max err {err}");
}

/// A vanishing path fades out (not cut): click-free, and still clearly
/// present 10 ms after it vanished, gone after a second.
#[test]
fn vanished_path_fades_out() {
    let y = render(&[], 72_000);
    assert_click_free(&y, "vanish");
    let before = rms(&y[19_200..24_000]);
    let just_after = rms(&y[24_000 + 240..24_000 + 720]);
    assert!(just_after > 0.6 * before, "must fade, not cut: {just_after} vs {before}");
    assert!(rms(&y[67_200..72_000]) < 1e-4 * before, "must be gone after ~1 s");
}

/// A brand-new path fades in from silence at its final delay (it does not
/// pop on at full level).
#[test]
fn new_path_fades_in() {
    let mut r = Renderer::new(SR);
    r.set_params(&block(1, &[(9, 0.005)]));
    let mut y = Vec::new();
    for k in 0..24_000 {
        let mut bus = [0.0f32; NCH];
        r.process(sine_at(F0, k), &mut bus);
        y.push(bus[0]);
    }
    assert!(max_step(&y) < 0.14, "fade-in click: {}", max_step(&y));
    let early = rms(&y[240..480]) * 2f32.sqrt();
    let late = rms(&y[19_200..24_000]) * 2f32.sqrt();
    assert!(early < 0.15 && (late - 1.0).abs() < 0.01, "fade-in: early {early}, late {late}");
}

/// Self-check of the metrics: a hard switch (instant 90° phase jump, what
/// a snap-to-new-delay would do) must fail the click bounds.
#[test]
fn metrics_catch_a_hard_switch() {
    let mut y: Vec<f32> = (0..36_000).map(|k| sine_at(F0, k)).collect();
    // switch at an arbitrary sample to the 90°-lagged path
    for (k, v) in y.iter_mut().enumerate().skip(24_013) {
        *v = (std::f64::consts::TAU * F0 as f64 * (k as f64 / SR as f64) - std::f64::consts::FRAC_PI_2)
            .sin() as f32;
    }
    let (step_ref, d2_ref) = steady(&y);
    let t = &y[24_000..33_600];
    let burst = t.chunks(480).map(|w| d2_energy(w) / w.len() as f64).fold(0.0, f64::max);
    assert!(max_step(t) > 1.05 * step_ref && burst > 1.1 * d2_ref, "metrics too lax");
}
