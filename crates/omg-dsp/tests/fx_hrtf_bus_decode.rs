//! Effect: binaural decode of the order-2 SH bus (hrtf.rs
//! `BinauralDecoder`: SH → virtual speakers → per-speaker HRIR pair),
//! also via `OutputStage::from_speaker_bytes(Some(..))`.
//!
//! Claim: a source encoded on the bus at +90° (left) comes out LOUDER in
//! the left ear (ILD) and EARLIER in the left ear (ITD); mirrored for
//! the right; a frontal source is balanced with zero ITD; and a source
//! behind is distinguishable from one in front by the spectral cue the
//! speaker HRIRs carry.
//!
//! Isolation: synthetic 12-speaker icosahedron with analytic HRIRs built
//! in-test (no KEMAR asset): per speaker, ITD = ±10·y samples, ILD gain
//! 1 ± 0.5·y, and a 3-tap pinna-shadow lowpass blended in for rear
//! speakers (weight −x). Noise encoded with `encode_gains`.
//! Combined in the demo: omg-scene tests/wildlife.rs (owls arrive from
//! above) asserts the arrival directions that this stage renders.

mod common;
use common::*;
use omg_dsp::ambi::{encode_gains, NCH};
use omg_dsp::hrtf::BinauralDecoder;
use omg_dsp::output::OutputStage;

const TAPS: usize = 64;
const BASE: i32 = 24;

fn kernel(delay: i32, gain: f32, shadow: f32) -> Vec<f32> {
    let mut h = vec![0.0f32; TAPS];
    let d = delay as usize;
    h[d] += gain * (1.0 - shadow);
    for (o, w) in [(-1i32, 0.25f32), (0, 0.5), (1, 0.25)] {
        h[(d as i32 + o) as usize] += gain * shadow * w;
    }
    h
}

fn speaker_bytes() -> Vec<u8> {
    let recs: Vec<HrirRec> = icosahedron()
        .into_iter()
        .map(|d| {
            let itd = (10.0 * d[1]).round() as i32;
            let shadow = (-d[0]).max(0.0);
            HrirRec {
                dir: d,
                l: kernel(BASE - itd, 1.0 + 0.5 * d[1], shadow),
                r: kernel(BASE + itd, 1.0 - 0.5 * d[1], shadow),
            }
        })
        .collect();
    hrir_bytes(TAPS, &recs)
}

fn decode(dir: [f32; 3]) -> (Vec<f32>, Vec<f32>) {
    let mut dec = BinauralDecoder::from_bytes(&speaker_bytes());
    assert_eq!(dec.speaker_count(), 12);
    let enc = encode_gains(dir);
    let x = noise(11, 12_000, 0.5);
    let (mut l, mut r) = (Vec::new(), Vec::new());
    for s in x {
        let sh: [f32; NCH] = core::array::from_fn(|k| s * enc[k]);
        let (a, b) = dec.process(&sh);
        l.push(a);
        r.push(b);
    }
    (l, r)
}

fn ild_db(l: &[f32], r: &[f32]) -> f64 {
    db_pow(energy(l) / energy(r))
}

#[test]
fn left_source_is_louder_and_earlier_in_the_left_ear() {
    let (l, r) = decode([0.0, 1.0, 0.0]);
    let ild = ild_db(&l, &r);
    let lag = xcorr_peak_lag(&l, &r, 30);
    assert!(ild > 4.0, "ILD {ild:.1} dB");
    assert!((6..=24).contains(&lag), "right ear must lag the left: {lag} samples");

    let (l, r) = decode([0.0, -1.0, 0.0]);
    let (ild_r, lag_r) = (ild_db(&l, &r), xcorr_peak_lag(&l, &r, 30));
    assert!((ild_r + ild).abs() < 0.5, "mirror ILD: {ild_r:.2} vs {ild:.2}");
    assert_eq!(lag_r, -lag, "mirror ITD");
}

#[test]
fn front_source_is_centered() {
    let (l, r) = decode([1.0, 0.0, 0.0]);
    assert!(ild_db(&l, &r).abs() < 0.3, "front ILD {:.2} dB", ild_db(&l, &r));
    assert_eq!(xcorr_peak_lag(&l, &r, 30), 0, "front ITD");
}

/// Behind vs in front: same interaural cues, but the rear-speaker pinna
/// shadow makes the back source clearly darker (first-difference energy
/// relative to total).
#[test]
fn back_is_distinguishable_from_front() {
    let bright = |(l, _r): (Vec<f32>, Vec<f32>)| -> f64 {
        let d: Vec<f32> = l.windows(2).map(|w| w[1] - w[0]).collect();
        energy(&d) / energy(&l)
    };
    let (f, b) = (bright(decode([1.0, 0.0, 0.0])), bright(decode([-1.0, 0.0, 0.0])));
    assert!(b < 0.8 * f, "back must be darker than front: {b:.3} vs {f:.3}");
}

/// Through the shared output stage: HRTF decode is selected by the
/// bytes, and the left source stays left after rotation/AGC/limiter.
#[test]
fn output_stage_uses_the_binaural_decoder() {
    let bytes = speaker_bytes();
    let mut out = OutputStage::from_speaker_bytes(Some(&bytes), SR);
    assert!(out.is_binaural());
    assert_eq!(out.speaker_count(), 12);
    let enc = encode_gains([0.0, 1.0, 0.0]);
    let (mut el, mut er) = (0.0f64, 0.0f64);
    for s in noise(5, 9600, 0.1) {
        let sh: [f32; NCH] = core::array::from_fn(|k| s * enc[k]);
        let (l, r) = out.process(&sh, 0.0, 0.0);
        el += (l as f64).powi(2);
        er += (r as f64).powi(2);
    }
    assert!(db_pow(el / er) > 4.0, "left source via OutputStage: {:.1} dB", db_pow(el / er));
}
