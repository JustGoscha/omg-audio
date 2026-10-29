//! Effect: per-tap band EQ (filter.rs `Shelves`: second-order shelves at
//! the tracer's band edges, 314 Hz and 2.1 kHz, at the low and high
//! band's gains over the mid's; renderer.rs `targets_of`).
//!
//! Claims:
//!  - flat band gains are transparent (pure broadband gain);
//!  - each band centre (125 / 790 / 5600 Hz) plays its band's gain —
//!    the traced absorption heard as tone, a wall's transmission (bass 0,
//!    mids −14, treble −34 dB) as muffled as traced, deep cuts reached;
//!  - a transmission-shaped tap (lows ≫ mids, through a wall) slopes on
//!    through the mid band — "muffled", not merely "quieter".
//!
//! (Before: first-order shelves capped at ±12 dB, and a one-pole
//! low-pass fitted through 800 Hz for bass-heavy taps that ignored the
//! treble gain — a −20 dB band played near −8.)
//!
//! Isolation: bus-only `Renderer::new`, single tap, steady tones, W
//! channel; plus the raw filter structs.
//! Combined in the demo: omg-scene tests/sealed_club.rs and
//! world_traced_gate.rs `closing_a_door_muffles_traced_chains` assert the
//! traced band gains that feed this stage; src/world.rs
//! `deep_shadow_favors_lows` likewise.

mod common;
use common::*;
use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::NCH;
use omg_dsp::filter::{OnePoleLp, TapEq};
use omg_dsp::Renderer;

/// Steady-state amplitude gain of one tap with `gains` at `freq`.
fn tap_gain(gains: [f32; 3], freq: f32) -> f32 {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock {
        taps: vec![Tap { key: 1, delay_s: 0.001, dir: [1.0, 0.0, 0.0], gains }],
        version: 1,
        ..Default::default()
    });
    let mut y = Vec::with_capacity(4800);
    for k in 0..24_000 {
        let mut bus = [0.0f32; NCH];
        r.process(sine_at(freq, k), &mut bus);
        if k >= 19_200 {
            y.push(bus[0]);
        }
    }
    tone_amp(&y, freq) // 0.1 s window: whole cycles for every probe below
}

const PROBES: [f32; 9] = [50.0, 100.0, 400.0, 800.0, 1000.0, 2000.0, 6000.0, 12_000.0, 15_000.0];

#[test]
fn flat_band_gains_are_transparent() {
    for f in PROBES {
        let g = tap_gain([0.3, 0.3, 0.3], f);
        assert!((db(g) - db(0.3)).abs() < 0.05, "{f} Hz: {g} (want 0.3)");
    }
}

/// Moderate shelves [0.5, 1, 0.5] (±6 dB, the common case): the low end
/// approaches 0.5, the top end 0.5, the mid band stays near unity —
/// highs reduced relative to mids/lows exactly when the high band is
/// weaker, and a reversed tilt brightens.
#[test]
fn shelves_shape_the_band_extremes() {
    let g: Vec<f32> = PROBES.iter().map(|&f| tap_gain([0.5, 1.0, 0.5], f)).collect();
    let at = |f: f32| g[PROBES.iter().position(|&p| p == f).unwrap()];
    assert!((db(at(50.0)) - db(0.5)).abs() < 1.0, "low band: {} (want 0.5)", at(50.0));
    assert!((db(at(15_000.0)) - db(0.5)).abs() < 2.0, "high band: {} (want 0.5)", at(15_000.0));
    assert!(db(at(800.0)).abs() < 1.0, "mid band ~unity: {}", at(800.0));

    let dark: Vec<f32> = PROBES.iter().map(|&f| tap_gain([1.0, 1.0, 0.25], f)).collect();
    assert!(dark[8] < 0.5 * dark[0], "weaker high band must darken: {dark:?}");
    for w in dark.windows(2) {
        assert!(w[1] <= w[0] + 1e-3, "pure high cut must be monotone: {dark:?}");
    }
    let lo = tap_gain([0.25, 1.0, 2.0], 50.0);
    let hi = tap_gain([0.25, 1.0, 2.0], 15_000.0);
    assert!(hi > 4.0 * lo, "boosted highs vs cut lows: {hi} vs {lo}");
}

/// Deep band cuts ([0.1, 1, 0.1], −20 dB, e.g. a far source's air
/// absorption or a heavy absorber) are rendered as roughly −20 dB across
/// their band (first-order shelves bottomed out near −8 dB).
#[test]
fn deep_band_cuts_reach_their_band_gain() {
    let band_mean_db = |probes: &[f32]| -> f32 {
        probes.iter().map(|&f| db(tap_gain([0.1, 1.0, 0.1], f))).sum::<f32>() / probes.len() as f32
    };
    let low = band_mean_db(&[40.0, 60.0, 100.0, 160.0]);
    let high = band_mean_db(&[3000.0, 4000.0, 6000.0, 8000.0, 12_000.0, 16_000.0]);
    assert!(
        (low + 20.0).abs() < 4.0 && (high + 20.0).abs() < 4.0,
        "band means: low {low:.1} dB, high {high:.1} dB; traced −20 dB each"
    );
}

/// Through-wall tap [1, 0.2, 0.02] (low/mid = 5): the low band at 1,
/// 0.2 at 800 Hz, and a monotone slope that keeps falling INSIDE the mid
/// band.
#[test]
fn through_wall_tap_is_a_continuous_lowpass() {
    let g: Vec<f32> = PROBES.iter().map(|&f| tap_gain([1.0, 0.2, 0.02], f)).collect();
    let at = |f: f32| g[PROBES.iter().position(|&p| p == f).unwrap()];
    assert!((db(at(50.0))).abs() < 1.0, "anchored at lows: {}", at(50.0));
    assert!((db(at(800.0)) - db(0.2)).abs() < 1.5, "fit point 800 Hz: {} (want 0.2)", at(800.0));
    for w in g.windows(2) {
        assert!(w[1] < w[0], "must fall monotonically: {g:?}");
    }
    assert!(at(400.0) / at(2000.0) > 3.0, "mid band must slope, not sit flat: {g:?}");
    assert!(at(6000.0) < 0.05, "treble dies: {}", at(6000.0));

}

/// The raw filter primitives: one-pole −3 dB at its cutoff, unity DC;
/// TapEq with unit gains is bit-transparent.
#[test]
fn filter_primitives() {
    for fc in [200.0f32, 1000.0, 4000.0] {
        let c = OnePoleLp::coef(fc, SR);
        let mut lp = OnePoleLp::default();
        let y: Vec<f32> = (0..48_000).map(|k| lp.tick(sine_at(fc, k), c)).collect();
        let g = tone_amp(&y[24_000..], fc);
        // bilinear-free one-pole: −3 dB within ~0.5 dB below a few kHz
        assert!((db(g) + 3.0).abs() < 0.6, "fc {fc}: {} dB", db(g));
        let mut lp = OnePoleLp::default();
        let dc = (0..48_000).map(|_| lp.tick(1.0, c)).last().unwrap();
        assert!((dc - 1.0).abs() < 1e-5);
    }
    let mut eq = TapEq::default();
    let (cl, ch) = (OnePoleLp::coef(250.0, SR), OnePoleLp::coef(2500.0, SR));
    for x in noise(3, 4800, 1.0) {
        assert_eq!(eq.tick(x, 1.0, 1.0, cl, ch), x);
    }
}


/// Every band centre plays its band's gain: within 2 dB for steps up to
/// 20 dB between neighbouring bands; a wall's whole span (bass 28 dB over
/// the mids, treble 20 under) within 4.5 dB (measured −2.1 / −2.6 / +4.3:
/// the shelves not quite turned by the next centre). The fitted low-pass
/// before left that wall's treble 28 dB high.
#[test]
fn band_centres_play_their_band_gains() {
    let mut worst = 0.0f32;
    for (g, bound) in [
        ([1.0f32, 0.2, 0.02], 2.0),
        ([1.0, 0.5, 0.25], 2.0),
        ([0.1, 1.0, 0.1], 2.0),
        ([0.02, 0.2, 1.0], 2.0),
        ([1.0, 0.04, 0.0004], 4.5),
    ] {
        for (k, f) in [125.0f32, 790.0, 5600.0].into_iter().enumerate() {
            let d = db(tap_gain(g, f)) - db(g[k]);
            worst = worst.max(d.abs());
            assert!(d.abs() < bound, "gains {g:?}, {f} Hz: {d:+.1} dB off");
        }
    }
    println!("worst band centre: {worst:.1} dB");
}
