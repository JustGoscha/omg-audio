//! Effect: a source's scattered response sounds the same convolved and
//! played by its parametric stand-in (the mixer's convolution budget
//! hands sources between the two as they come and go: a level or colour
//! step at each hand-over is heard as the source popping).
//!
//! Claim: a click through a traced response's kernel and through the
//! reverb the simulation derives from it (sim.rs: its whole energy per
//! band, the decay that gives it the response's centre time, the
//! one-sided share `a` from where it comes) put the same energy per band
//! into the bus — the W channel and the first-order channels alike —
//! within 2 dB (measured ≤ 1.6: the network's sends are solved from a
//! model of its tail, `fx_fdn_bands`).
//!
//! Claim: two sources playing one signal (a club's speakers) ring
//! uncorrelated either way — their kernels of different noise seeds,
//! their stand-ins of different networks (`Renderer::decorrelate`) — so
//! both sums hold the sum of the energies.
//!
//! Isolation: bus-only `Renderer::new`, no taps; a synthetic response (a
//! decay, bass-heavy as through a wall, partly from one side).

mod common;
use common::*;
use omg_core::params::{ParamBlock, RemoteReverb, ReverbParams};
use omg_core::v3::{DenseResponse, DENSE_BIN_S, HISTORY_WINDOWS};
use omg_dsp::ambi::{encode_gains, NCH};
use omg_dsp::bands::BandSplit;
use omg_dsp::dense::build_kernel_max;
use omg_dsp::renderer::{BusBlock, BLOCK};
use omg_dsp::Renderer;
use std::sync::Arc;

/// A response: per band energy `e` decaying at `rt60`, a share `a` of it
/// arriving from `dir`.
fn response(e: [f32; 3], rt60: [f32; 3], a: f32, dir: [f32; 3]) -> DenseResponse {
    let n = (3.0 / DENSE_BIN_S) as usize;
    let mut bins = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * DENSE_BIN_S;
        let b: [f32; 3] = core::array::from_fn(|k| {
            let k60 = 6.907_755 / rt60[k];
            e[k] * (1.0 - (-k60 * DENSE_BIN_S).exp()) * (-k60 * t).exp()
        });
        bins.push([b[0], b[1], b[2], a * b[1] * dir[0], a * b[1] * dir[1], a * b[1] * dir[2]]);
    }
    DenseResponse { bin_s: DENSE_BIN_S, bins, noise: [0.0; HISTORY_WINDOWS], impulses: Vec::new() }
}

/// Per bus channel group (W; X, Y, Z), the band energies of a click.
fn render(r: &mut Renderer) -> [[f64; 3]; 2] {
    render_all(&mut [r])
}

/// The same for renderers playing the same click, summed in one bus.
fn render_all(rs: &mut [&mut Renderer]) -> [[f64; 3]; 2] {
    let total = 48_000 * 4;
    let mut split: Vec<BandSplit> = (0..4).map(|_| BandSplit::new(SR)).collect();
    let mut e = [[0.0f64; 3]; 2];
    let mut done = 0;
    // let the glides land, then the click
    let settle = 24_064;
    while done < settle + total {
        let mut bus: BusBlock = [[0.0; BLOCK]; NCH];
        let (mut pl, mut pr) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let x: Vec<f32> = (0..BLOCK).map(|j| if done + j == settle { 1.0 } else { 0.0 }).collect();
        for r in rs.iter_mut() {
            r.process_block(&x, &mut bus, &mut pl, &mut pr);
        }
        if done >= settle {
            for j in 0..BLOCK {
                for c in 0..4 {
                    let b = split[c].split(bus[c][j]);
                    for k in 0..3 {
                        e[(c > 0) as usize][k] += (b[k] as f64).powi(2);
                    }
                }
            }
        }
        done += BLOCK;
    }
    e
}

#[test]
fn convolved_and_parametric_put_the_same_energy_in_the_bus() {
    let mut worst = 0.0f64;
    for (e, rt60, a) in [
        ([1.0f32, 1.0, 1.0], [1.0f32, 1.0, 1.0], 0.0f32),
        ([1.0, 1.0, 1.0], [1.5, 1.2, 0.5], 0.0),
        ([1.0, 0.1, 0.01], [1.5, 1.2, 0.5], 0.6),
        ([1.0, 0.3, 0.05], [0.8, 0.7, 0.3], 0.3),
    ] {
        let dir = [0.6f32, 0.8, 0.0];
        let resp = response(e, rt60, a, dir);
        // convolved
        let mut rc = Renderer::new(SR);
        rc.set_params(&ParamBlock { version: 1, onset_s: 0.0005, ..Default::default() });
        rc.set_dense_kernel(Some(Arc::new(build_kernel_max(&resp, SR, 1.0, 7, 10.0))));
        rc.set_convolving(true);
        let conv = render(&mut rc);
        // the stand-in, as sim.rs derives it
        let pb = stand_in(&resp);
        let mut rp = Renderer::new(SR);
        rp.set_params(&pb);
        rp.set_convolving(false);
        let par = render(&mut rp);
        let mut line = String::new();
        for (g, name) in ["W", "XYZ"].iter().enumerate() {
            line += &format!(" {name}:");
            for k in 0..3 {
                let d = db_pow(par[g][k]) - db_pow(conv[g][k]);
                let top = conv[g].iter().fold(0.0f64, |m, &x| m.max(x));
                if conv[g][k] > top * 0.1 {
                    worst = worst.max(d.abs());
                }
                line += &format!(" {d:+5.1}");
            }
        }
        println!("e {e:?} rt60 {rt60:?} a {a}: parametric − convolved dB{line}");
    }
    assert!(worst < 2.0, "a band off by {worst:.1} dB");
}

/// The block sim.rs derives from a response for a mixer that doesn't
/// convolve it.
fn stand_in(resp: &DenseResponse) -> ParamBlock {
    let lv: [f32; 3] = core::array::from_fn(|b| resp.bins.iter().map(|x| x[b]).sum::<f32>().sqrt());
    let ts = resp.centre_time();
    let rt: [f32; 3] = core::array::from_fn(|b| omg_dsp::fdn::rt60_for_centre_time(ts[b]));
    let (mut v, mut em) = ([0.0f32; 3], 0.0f32);
    for x in &resp.bins {
        for k in 0..3 {
            v[k] += x[3 + k];
        }
        em += x[1];
    }
    let vl = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let an = (vl / em).clamp(0.0, 0.9);
    let mut pb = ParamBlock { version: 1, onset_s: 0.0005, ..Default::default() };
    pb.reverb = ReverbParams { rt60: rt, level: core::array::from_fn(|b| lv[b] * (1.0 - an).sqrt()) };
    if an > 0.0 {
        pb.remote = Some(RemoteReverb {
            rt60: rt,
            send: core::array::from_fn(|b| lv[b] * an.sqrt()),
            sh: encode_gains([v[0] / vl, v[1] / vl, v[2] / vl]),
        });
    }
    pb
}

#[test]
fn two_sources_of_one_signal_ring_uncorrelated_either_way() {
    let resp = response([1.0, 0.3, 0.05], [1.2, 1.0, 0.4], 0.4, [0.0, 1.0, 0.0]);
    let kernel = |seed: u32| {
        let mut r = Renderer::new(SR);
        r.set_params(&ParamBlock { version: 1, onset_s: 0.0005, ..Default::default() });
        r.set_dense_kernel(Some(Arc::new(build_kernel_max(&resp, SR, 1.0, seed, 10.0))));
        r.set_convolving(true);
        r
    };
    let param = |seed: u32| {
        let mut r = Renderer::new(SR);
        r.decorrelate(seed);
        r.set_params(&stand_in(&resp));
        r.set_convolving(false);
        r
    };
    let (mut k1, mut k2) = (kernel(1), kernel(2));
    let one = render(&mut kernel(1));
    let conv = render_all(&mut [&mut k1, &mut k2]);
    let (mut p1, mut p2) = (param(1), param(2));
    let par = render_all(&mut [&mut p1, &mut p2]);
    let mut worst = 0.0f64;
    let mut line = String::new();
    for g in 0..2 {
        for k in 0..3 {
            // the sum of two uncorrelated copies: twice one's energy
            let dc = db_pow(conv[g][k]) - db_pow(2.0 * one[g][k]);
            let dp = db_pow(par[g][k]) - db_pow(conv[g][k]);
            worst = worst.max(dc.abs()).max(dp.abs());
            line += &format!(" [{dc:+4.1} {dp:+4.1}]");
        }
    }
    println!("two sources: [kernels' sum − 2×one, stand-ins' − kernels'] dB, W then XYZ:{line}");
    assert!(worst < 1.0, "off by {worst:.1} dB");
}
