//! Effect: the dense early field (`ParamBlock::early`, engine v6.2) — the
//! early energy past the strongest early taps, played by the renderer's
//! short early network from the first arrival on.
//!
//! Claims:
//! - it plays the energy it is given (level L: energy L², per band);
//! - it arrives within the early window: ~35 ms after the first arrival
//!   on average (the early network's fixed decay — `centre_s` is not
//!   followed: a decay swinging with each block's estimate burst), little
//!   of it past 80 ms;
//! - changing every block (level, direction, centre) it never bursts —
//!   the web demo's smoke test played +20 dB when the decay followed the
//!   estimate;
//! - it comes from where it is said to (`dir`): one-sided, the side it
//!   comes from louder; from all round (dir 0), even.
//!
//! Isolation: bus-only `Renderer::new`, no taps, no reverb, one impulse.
//! Combined in the demo: inside the club the first 80 ms played 4–6 dB
//! under the traced sum while the taps carried a quarter of the early
//! energy (OPEN.md, ENGINE_V6.md: "The dense early field").

mod common;
use common::*;
use omg_core::params::{DenseEarly, ParamBlock};
use omg_dsp::ambi::NCH;
use omg_dsp::Renderer;

/// An impulse through a block carrying only `early`, once its level has
/// glided in (0.2 s of silence first): the bus (W, Y) for `secs`.
fn impulse(early: DenseEarly, secs: f32) -> (Vec<f32>, Vec<f32>) {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock { version: 1, early, ..Default::default() });
    for _ in 0..(0.2 * SR) as usize {
        let mut bus = [0.0f32; NCH];
        r.process(0.0, &mut bus);
    }
    let n = (secs * SR) as usize;
    let (mut w, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        let mut bus = [0.0f32; NCH];
        r.process(if k == 0 { 1.0 } else { 0.0 }, &mut bus);
        w.push(bus[0]);
        y.push(bus[1]);
    }
    (w, y)
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64) * (*v as f64)).sum()
}

#[test]
fn the_dense_early_field_plays_its_energy_early() {
    for centre in [0.003f32, 0.02, 0.06] {
        let (w, _) = impulse(DenseEarly { level: [0.5; 3], centre_s: centre, dir: [0.0; 3] }, 1.0);
        let e = energy(&w);
        let t: f64 = w.iter().enumerate().map(|(k, v)| k as f64 / SR as f64 * (*v as f64) * (*v as f64)).sum::<f64>() / e;
        let early = energy(&w[..(0.08 * SR) as usize]) / e;
        eprintln!("centre said {centre:.3} s: energy {:+.2} dB of L², centre time {:.1} ms, within 80 ms {:.0} %", 10.0 * (e / 0.25).log10(), t * 1e3, early * 100.0);
        assert!((10.0 * (e / 0.25).log10()).abs() < 1.0, "energy {:.2} dB off L²", 10.0 * (e / 0.25).log10());
        assert!((t - 0.035).abs() < 0.008, "centre time {:.1} ms", t * 1e3);
        assert!(early > 0.85, "within 80 ms {:.0} %", early * 100.0);
    }
}

#[test]
fn the_dense_early_field_comes_from_where_it_is_said_to() {
    // ACN: the bus's channel 1 is Y (left, +y). Lines uncorrelated: the
    // energy-weighted Y over W is the field's mean y-direction
    let lean = |dir: [f32; 3]| -> f64 {
        let (w, y) = impulse(DenseEarly { level: [0.5; 3], centre_s: 0.03, dir }, 0.5);
        w.iter().zip(&y).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() / energy(&w)
    };
    let (left, round, right) = (lean([0.0, 0.8, 0.0]), lean([0.0; 3]), lean([0.0, -0.8, 0.0]));
    eprintln!("Y/W: from the left {left:.2}, from all round {round:.2}, from the right {right:.2}");
    assert!(left > 0.4 && right < -0.4, "one-sided: {left:.2} / {right:.2}");
    assert!(round.abs() < 0.15, "from all round: {round:.2}");
}

#[test]
fn the_dense_early_field_plays_each_bands_energy() {
    // per band, what the field puts there against what one path of the
    // same gains does (the bands overlap: a bright path's treble skirt is
    // part of its mids) — the field's band balance is a path's
    let split = |w: &[f32]| -> [f64; 3] {
        let mut b = omg_dsp::bands::BandSplit::new(SR);
        let mut e = [0.0f64; 3];
        for v in w {
            let s = b.split(*v);
            for k in 0..3 {
                e[k] += (s[k] as f64) * (s[k] as f64);
            }
        }
        e
    };
    let path = |gains: [f32; 3]| -> [f64; 3] {
        let mut r = Renderer::new(SR);
        let tap = omg_core::params::Tap { key: 1, delay_s: 0.01, dir: [1.0, 0.0, 0.0], gains };
        r.set_params(&ParamBlock { version: 1, taps: vec![tap], ..Default::default() });
        r.snap_next();
        let mut w = Vec::new();
        for k in 0..(0.3 * SR) as usize {
            let mut bus = [0.0f32; NCH];
            r.process(if k == (0.1 * SR) as usize { 1.0 } else { 0.0 }, &mut bus);
            w.push(bus[0]);
        }
        split(&w)
    };
    for level in [[1.0f32, 0.3, 0.1], [0.1, 0.3, 1.0], [0.5; 3]] {
        let (w, _) = impulse(DenseEarly { level, centre_s: 0.03, dir: [0.0; 3] }, 0.5);
        let (e, p) = (split(&w), path(level));
        let rel: Vec<f64> = (0..3).map(|k| 10.0 * (e[k] / p[k]).log10()).collect();
        eprintln!("levels {level:?}: each band against a path's, dB: {:+.2} {:+.2} {:+.2}", rel[0], rel[1], rel[2]);
        assert!(rel.iter().all(|r| r.abs() < 1.5), "band balance off: {rel:?}");
    }
}

#[test]
fn a_dense_early_field_changing_every_block_never_bursts() {
    // steady noise; every 128 samples a new field — level ±3 dB, from
    // anywhere, its centre said anywhere from 3 to 60 ms
    let mut r = Renderer::new(SR);
    let mut rng = omg_core::rng::Rng::new(7);
    let x = noise(3, (2.0 * SR) as usize, 0.3);
    let (mut w, mut l2) = (Vec::with_capacity(x.len()), 0.0f64);
    for (k, &v) in x.iter().enumerate() {
        if k % 128 == 0 {
            let l = 0.3 * 2f32.powf(rng.next_f32() - 0.5);
            let d = [rng.next_f32() - 0.5, rng.next_f32() - 0.5, rng.next_f32() - 0.5];
            let early = DenseEarly { level: [l; 3], centre_s: 0.003 + 0.057 * rng.next_f32(), dir: d };
            r.set_params(&ParamBlock { version: 1 + k as u64, early, ..Default::default() });
            if k >= (0.5 * SR) as usize {
                l2 += (l * l) as f64 * 128.0;
            }
        }
        let mut bus = [0.0f32; NCH];
        r.process(v, &mut bus);
        w.push(bus[0]);
    }
    // after 0.5 s: the energy as given (level² × the input's), and no
    // 10 ms window far over the others
    let a = (0.5 * SR) as usize;
    let got = energy(&w[a..]);
    let want = l2 / (x.len() - a) as f64 * energy(&x[a..]);
    let win: Vec<f64> = w[a..].chunks(480).map(energy).collect();
    let mut sorted = win.clone();
    sorted.sort_by(|p, q| p.total_cmp(q));
    let (median, top) = (sorted[sorted.len() / 2], sorted[sorted.len() - 1]);
    eprintln!("changing every block: {:+.2} dB of the energy given, loudest 10 ms {:+.1} dB over the median", 10.0 * (got / want).log10(), 10.0 * (top / median).log10());
    assert!((10.0 * (got / want).log10()).abs() < 1.5, "energy {:+.2} dB", 10.0 * (got / want).log10());
    assert!(10.0 * (top / median).log10() < 6.0, "a burst: {:+.1} dB over the median", 10.0 * (top / median).log10());
}
