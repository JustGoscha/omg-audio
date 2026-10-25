//! Effect: virtual-cardioid stereo fallback decode (ambi.rs
//! `StereoDecoder`; `OutputStage::from_speaker_bytes(None)`).
//!
//! Claim: without an HRIR asset the bus still images correctly on two
//! channels: two cardioids aimed ±60° — a left source is louder left
//! (L = 0.5·(1 + sin 60°) vs R = 0.5·(1 − sin 60°), ≈ 23 dB), a right
//! source mirrored, a front source equal in both, panning monotone with
//! azimuth, and a rear source equal but quieter than front (cardioids
//! face forward).
//!
//! Isolation: SH frames from `encode_gains`, decoded directly and through
//! a cardioid `OutputStage` (rotation at rest). No renderer, no scene.
//! Combined in the demo: native/web builds without the HRIR asset
//! (omg-app / omg-web fallback path).

mod common;
use common::*;
use omg_dsp::ambi::{encode_gains, StereoDecoder, NCH};
use omg_dsp::output::OutputStage;

fn dir_az(az_deg: f32) -> [f32; 3] {
    let a = az_deg.to_radians();
    [a.cos(), a.sin(), 0.0] // +az = toward the left (+y)
}

#[test]
fn left_right_front_back() {
    let d = StereoDecoder::new();
    let s60 = 60f32.to_radians().sin();
    let (l, r) = d.decode(&encode_gains([0.0, 1.0, 0.0]));
    assert!((l - 0.5 * (1.0 + s60)).abs() < 1e-6 && (r - 0.5 * (1.0 - s60)).abs() < 1e-6, "left: {l} {r}");
    assert!(db(l / r) > 20.0);
    let (l2, r2) = d.decode(&encode_gains([0.0, -1.0, 0.0]));
    assert!((l2 - r).abs() < 1e-6 && (r2 - l).abs() < 1e-6, "right mirrors left");
    let (lf, rf) = d.decode(&encode_gains([1.0, 0.0, 0.0]));
    assert_eq!(lf, rf, "front is centered");
    let (lb, rb) = d.decode(&encode_gains([-1.0, 0.0, 0.0]));
    assert_eq!(lb, rb, "back is centered");
    assert!(lb < 0.5 * lf, "cardioids face forward: back {lb} vs front {lf}");
}

#[test]
fn pan_is_monotone_in_azimuth() {
    let d = StereoDecoder::new();
    let mut prev = f32::MIN;
    for az in (-90..=90).step_by(10) {
        let (l, r) = d.decode(&encode_gains(dir_az(az as f32)));
        let pan = (l - r) / (l + r);
        assert!(pan > prev, "az {az}: pan {pan} not above {prev}");
        prev = pan;
    }
}

/// The output stage without speaker bytes uses this decoder.
#[test]
fn output_stage_falls_back_to_cardioids() {
    let mut out = OutputStage::from_speaker_bytes(None, SR);
    assert!(!out.is_binaural());
    assert_eq!(out.speaker_count(), 2);
    let enc = encode_gains([0.0, 1.0, 0.0]);
    let sh: [f32; NCH] = core::array::from_fn(|k| 0.05 * enc[k]);
    let (l, r) = out.process(&sh, 0.0, 0.0);
    let s60 = 60f32.to_radians().sin();
    // (tanh compresses the louder ear by < 0.1 % at this level)
    let want = (1.0 + s60) / (1.0 - s60);
    assert!((l / r / want - 1.0).abs() < 2e-3, "L/R {} vs {want}", l / r);
}
