//! Effect: the parametric reverb plays each band at the energy it is given
//! — a source's stand-in when the mixer doesn't convolve its response, so
//! a hand-over between the two must not change its level or its colour.
//!
//! Claim: a click through a renderer with reverb levels L_b plays, per
//! band, the energy the click shaped by the tracer's bands at L_b holds
//! (`BandSplit::process`, the kernel's own synthesis) — whatever the
//! decays: the high band faster than the mid, as traced, or not. Equal
//! decays: within 0.2 dB. A faster high band: within 1.7 dB (measured)
//! for every band within 10 dB of the loudest — the sends are solved from
//! a model of the network's tail per frequency (`Fdn::band_make_up`),
//! not exact for its first passes. A band far under the loudest carries
//! that band's skirt through a network decaying differently there:
//! measured up to +4.3 dB on a low band 30 dB under a treble-heavy field
//! (rare in a scattered field). (Two shelves relative to the mid, as
//! before: up to +25 dB.)
//!
//! Isolation: bus-only `Renderer::new`, no taps, the W channel split into
//! the tracer's bands (its scale against energy taken from a flat case).

mod common;
use common::*;
use omg_core::params::{ParamBlock, ReverbParams};
use omg_dsp::ambi::NCH;
use omg_dsp::bands::BandSplit;
use omg_dsp::Renderer;

/// Band energies of `x` (the tracer's one-pole bands).
fn split_energy(x: &[f32]) -> [f64; 3] {
    let mut s = BandSplit::new(SR);
    let mut e = [0.0f64; 3];
    for &v in x {
        let b = s.split(v);
        for i in 0..3 {
            e[i] += (b[i] as f64).powi(2);
        }
    }
    e
}

/// The W channel after a click through the reverb alone.
fn reverb_w(level: [f32; 3], rt60: [f32; 3]) -> Vec<f32> {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock { reverb: ReverbParams { rt60, level }, version: 1, onset_s: 0.001, ..Default::default() });
    let n = (SR * (rt60.iter().fold(0.0f32, |a, &b| a.max(b)) * 1.6 + 0.5)) as usize;
    // let the level glides land first, then the click
    let settle = 24_000;
    let mut w = Vec::with_capacity(n);
    for k in 0..settle + n {
        let mut bus = [0.0f32; NCH];
        r.process(if k == settle { 1.0 } else { 0.0 }, &mut bus);
        if k >= settle {
            w.push(bus[0]);
        }
    }
    w
}

/// The click shaped at `level` per band, as the kernel would play it.
fn shaped_click(level: [f32; 3]) -> Vec<f32> {
    let mut s = BandSplit::new(SR);
    (0..4800).map(|k| s.process(if k == 0 { 1.0 } else { 0.0 }, &level)).collect()
}

#[test]
fn each_band_plays_the_energy_it_is_given() {
    // the W channel's share of the network's energy (flat, equal decays)
    let flat = split_energy(&reverb_w([0.1; 3], [1.0; 3]));
    let w_scale = flat.iter().sum::<f64>() / split_energy(&shaped_click([0.1; 3])).iter().sum::<f64>();
    let (mut worst, mut worst_far) = (0.0f64, 0.0f64);
    for rt60 in [[1.0f32, 1.0, 1.0], [1.2, 1.2, 0.4], [3.0, 3.0, 0.8], [0.5, 0.5, 0.2]] {
        for level in [[0.1f32, 0.1, 0.1], [1.0, 0.1, 0.01], [0.03, 0.1, 0.3], [0.5, 0.1, 0.002]] {
            let got = split_energy(&reverb_w(level, rt60));
            let want = split_energy(&shaped_click(level));
            let mut line = String::new();
            for b in 0..3 {
                let top = want.iter().fold(0.0f64, |a, &x| a.max(x));
                let d = db_pow(got[b] / w_scale) - db_pow(want[b]);
                if want[b] > top * 0.1 {
                    worst = worst.max(d.abs());
                } else {
                    worst_far = worst_far.max(d.abs());
                }
                line += &format!(" {d:+5.1}");
            }
            println!("rt60 {rt60:?} levels {level:?}: band error dB{line}");
        }
    }
    assert!(worst < 2.0, "a band within 10 dB of the loudest off by {worst:.1} dB");
    assert!(worst_far < 6.0, "a band far under the loudest off by {worst_far:.1} dB");
}
