//! Effect: the convolution kernel plays a traced band shape at its band
//! gains — a wall's transmission (bass 0, mids −14, treble −34 dB) heard
//! that much duller, a closed door closed.
//!
//! Claim: tone gains at the band centres (125 / 790 / 5600 Hz) of a
//! kernel synthesized from a response with band energies E_b are √E_b
//! within 1.5 dB. (One-pole bands: the bass's skirt over the rest, the
//! wall's mids at −7 and treble at −18 dB.)

mod common;
use common::*;
use omg_core::v3::{DenseResponse, DENSE_BIN_S, HISTORY_WINDOWS};
use omg_dsp::dense::impulse_response_max;

fn kernel_tones(e: [f32; 3]) -> [f32; 3] {
    // a flat 0.5 s plateau of energy (no decay: the colour alone)
    let n = (0.5 / DENSE_BIN_S) as usize;
    let bins = vec![[e[0] / n as f32, e[1] / n as f32, e[2] / n as f32, 0.0, 0.0, 0.0]; n];
    let resp = DenseResponse { bin_s: DENSE_BIN_S, bins, noise: [0.0; HISTORY_WINDOWS], impulses: Vec::new() };
    let ir = &impulse_response_max(&resp, SR, 1.0, 3, 10.0)[0];
    // the power response at f: |H(f)|² summed over the IR (Parseval on a
    // tone: the IR's energy at that frequency ∝ |Σ h e^{-jωn}|²)
    core::array::from_fn(|k| {
        let f = [125.0f32, 790.0, 5600.0][k];
        // average |H|² over a third of an octave around f (noise IR)
        let mut acc = 0.0f64;
        let m = 24;
        for j in 0..m {
            let ff = f * 2f32.powf((j as f32 / (m - 1) as f32 - 0.5) / 3.0);
            let w = std::f64::consts::TAU * ff as f64 / SR as f64;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, &h) in ir.iter().enumerate() {
                re += h as f64 * (w * n as f64).cos();
                im -= h as f64 * (w * n as f64).sin();
            }
            acc += re * re + im * im;
        }
        ((acc / m as f64).sqrt()) as f32
    })
}

#[test]
fn the_kernel_plays_its_band_gains() {
    let flat = kernel_tones([1.0, 1.0, 1.0]);
    let mut worst = 0.0f32;
    for e in [[1.0f32, 0.04, 0.0004], [1.0, 0.1, 0.01], [0.01, 0.1, 1.0]] {
        let t = kernel_tones(e);
        let d: Vec<f32> = (0..3).map(|k| db(t[k] / flat[k]) - 10.0 * e[k].log10()).collect();
        println!("E {e:?}: off by {:+.1} {:+.1} {:+.1} dB", d[0], d[1], d[2]);
        worst = d.iter().fold(worst, |w, x| w.max(x.abs()));
    }
    assert!(worst < 1.5, "a band centre off by {worst:.1} dB");
}
