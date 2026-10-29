//! Effect: a moving path's treble stays steady — Doppler is the motion's
//! and nothing else (CLAUDE.md: no warble).
//!
//! Claim: a 10 kHz tone read through the play log along a path whose
//! delay slides steadily (a source passing at walking pace, 1.5 m/s, and
//! at a car's, 30 m/s) keeps its level: its envelope (five-cycle
//! windows) moves by less than 1 dB. (A read interpolated linearly loses treble by how far
//! between two samples it falls — at 10 kHz 2 dB at half a sample, none
//! on one — so a sliding delay modulates the treble at the rate the
//! fraction cycles: ~200 Hz at a walk.)
//!
//! Isolation: `PlayLog::read_block` alone, a looped tone, emission times
//! stepping by 1 − v/c a sample.

mod common;
use common::*;
use omg_dsp::playlog::PlayLog;

/// The tone's envelope's spread (dB, max − min; five-cycle windows) read
/// along a delay changing at `v` m/s.
fn spread(freq: f32, v: f32) -> f32 {
    // a loop of whole cycles (a 10 kHz tone at 48 kHz repeats every 24)
    let cycle = (SR / freq).round() as usize * 5;
    let clip: Vec<f32> = (0..cycle).map(|k| (std::f32::consts::TAU * freq * k as f32 / SR).sin()).collect();
    let mut log = PlayLog::new();
    log.set_loop(clip.into(), f64::NEG_INFINITY);
    let step = 1.0 - v as f64 / 343.0;
    let n = 48_000;
    let t: Vec<f64> = (0..n).map(|k| 1000.0 + k as f64 * step + 0.123).collect();
    let mut out = vec![0.0f32; n];
    for (tc, oc) in t.chunks(128).zip(out.chunks_mut(128)) {
        log.read_block(tc, &vec![1.0; tc.len()], oc);
    }
    // windows of whole cycles, much shorter than the fraction's cycle
    let w = (5.0 * SR / freq).round() as usize;
    let env: Vec<f32> = out.chunks(w).map(|c| db(rms(c))).collect();
    let (lo, hi) = env.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    hi - lo
}

#[test]
fn a_moving_paths_treble_stays_steady() {
    for v in [1.5f32, 30.0] {
        let s = spread(10_000.0, v);
        println!("10 kHz at {v} m/s: envelope spread {s:.2} dB");
        assert!(s < 1.0, "{v} m/s: the treble moves by {s:.2} dB");
    }
}
