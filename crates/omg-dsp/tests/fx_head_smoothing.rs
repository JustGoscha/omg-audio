//! Effect: head-pose smoothing (`OutputStage::set_head`,
//! `Renderer::set_head`; one-pole smoothers, τ = 30 ms).
//!
//! Claim: a head-orientation step (mouse look, device orientation, face
//! tracking) never jumps the sound field: the bus rotation follows the
//! one-pole trajectory yaw(t) = ψ·(1 − e^(−t/30 ms)) — ~63 % at 30 ms,
//! settled by ~200 ms, essentially nothing on the first sample. The same
//! smoothed pose drives point-tap HRIR re-selection in the renderer, so
//! a point source flips sides only when the SMOOTHED head passes the
//! midpoint (~τ·ln 2 ≈ 21 ms for a 180° step), plus the 10 ms HRIR fade.
//!
//! Isolation: cardioid `OutputStage` with a constant frontal SH frame
//! (pan read as (L−R)/(L+R), invariant to the AGC gain); and a renderer
//! with a synthetic left/right `HrirGrid`. No scene.
//! Combined in the demo: omg-web/omg-app mouse-look and device
//! orientation feed these setters; no scene test pins it.

mod common;
use common::*;
use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::{encode_gains, NCH};
use omg_dsp::hrtf::HrirGrid;
use omg_dsp::output::OutputStage;
use omg_dsp::Renderer;
use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

/// Cardioid pan of a frontal source seen with head yaw `yaw`.
fn pan_for_yaw(yaw: f32) -> f32 {
    let (s60, c60) = (60f32.to_radians().sin(), 60f32.to_radians().cos());
    let (x, y) = (yaw.cos(), -yaw.sin()); // head-frame direction
    y * s60 / (1.0 + x * c60)
}

#[test]
fn bus_rotation_follows_a_30ms_one_pole() {
    let mut out = OutputStage::from_speaker_bytes(None, SR);
    let enc = encode_gains([1.0, 0.0, 0.0]);
    let sh: [f32; NCH] = core::array::from_fn(|k| 0.1 * enc[k]);
    for _ in 0..480 {
        out.process(&sh, 0.0, 0.0);
    }
    out.set_head(FRAC_PI_2, 0.0, 0.0); // turn head 90° left → source to the right
    let pans: Vec<f32> = (0..9600)
        .map(|_| {
            let (l, r) = out.process(&sh, 0.0, 0.0);
            (l - r) / (l + r)
        })
        .collect();
    let coef = (-1.0f32 / (0.03 * SR)).exp();
    let model = |k: usize| pan_for_yaw(FRAC_PI_2 * (1.0 - coef.powi(k as i32 + 1)));

    assert!(pans[0].abs() < 0.005, "not instant: first-sample pan {}", pans[0]);
    for k in [240usize, 720, 1440, 2880] {
        assert!((pans[k] - model(k)).abs() < 0.01, "t={} ms: pan {} vs model {}", k / 48, pans[k], model(k));
    }
    // at τ the head is 63 % of the way: pan is between the endpoints
    let p_tau = pans[1439];
    assert!(p_tau < 0.5 * pan_for_yaw(FRAC_PI_2) && p_tau > pan_for_yaw(FRAC_PI_2), "pan at τ {p_tau}");
    assert!((pans[9599] - pan_for_yaw(FRAC_PI_2)).abs() < 0.005, "settled by 200 ms");
    for w in pans.windows(2) {
        assert!(w[1] <= w[0] + 1e-6, "rotation must be monotone, no overshoot");
    }
}

#[test]
fn point_tap_flips_sides_when_the_smoothed_head_passes_midpoint() {
    let rec = |dir, gl, gr| HrirRec { dir, l: impulse(8, 0, gl), r: impulse(8, 0, gr) };
    let grid = HrirGrid::from_bytes(&hrir_bytes(
        8,
        &[rec([0.0, 1.0, 0.0], 1.0, 0.1), rec([0.0, -1.0, 0.0], 0.1, 1.0)],
    ));
    let mut r = Renderer::with_grid(SR, Some(Arc::new(grid)));
    r.set_params(&ParamBlock {
        taps: vec![Tap { key: 1, delay_s: 0.001, dir: [0.0, 1.0, 0.0], gains: [1.0; 3] }],
        version: 1,
        ..Default::default()
    });
    let mut src = Noise::new(9);
    let mut block = |r: &mut Renderer| -> (f64, f64) {
        let (mut el, mut er) = (0.0f64, 0.0f64);
        for _ in 0..48 {
            let mut bus = [0.0f32; NCH];
            let (l, rr) = r.process(src.next(), &mut bus);
            el += (l as f64).powi(2);
            er += (rr as f64).powi(2);
        }
        (el, er)
    };
    for _ in 0..100 {
        block(&mut r);
    }
    let (el, er) = block(&mut r);
    assert!(el > 10.0 * er, "starts on the left");

    r.set_head(PI, 0.0, 0.0); // about-face: the source ends up on the right
    let flip_ms = (1..=100)
        .find(|_| {
            let (el, er) = block(&mut r);
            er > el
        })
        .expect("never flipped");
    assert!((15..=35).contains(&flip_ms), "flip after {flip_ms} ms (smoothed midpoint ≈ 21 ms + fade)");
    for _ in 0..100 {
        block(&mut r);
    }
    let (el, er) = block(&mut r);
    assert!(er > 10.0 * el, "ends on the right");
}
