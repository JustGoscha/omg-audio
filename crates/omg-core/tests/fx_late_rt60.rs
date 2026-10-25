//! Effect: late-field decay time (RT60) ACCURACY of the stochastic
//! tracer (`tracer::trace` + `tracer::estimate_reverb`) against the
//! diffuse-field closed forms (Sabine / Eyring).
//!
//! Existing tests (lib.rs `rt60_tracks_absorption`, trace_golden) only
//! check ordering and loose ranges; this file pins the absolute value.
//!
//! Physical claim guaranteed: in an empty shoebox with frequency-flat,
//! partly-scattering walls the traced RT60 lands within ±15 % of
//! Eyring/Sabine, scales with room size (RT ∝ V/S), and is
//! frequency-flat (low/mid) when the walls are. The ignored test documents that
//! LIVE rooms (α ≲ 0.1) come out far too short: the tracer's hard
//! 64-bounce cap truncates the decay before the −25 dB fit point.
//!
//! Combined in the demo: every room's late reverb in
//! `omg_scene::world::WorldSim` (hall / club / living room RT60s).

mod common;

use common::{flat_opaque, v};
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::NBANDS;

fn traced_rt60(size: (f32, f32, f32), alpha: f32, seed: u64) -> [f32; NBANDS] {
    let room = Shoebox::new(v(size.0, size.1, size.2), [flat_opaque(alpha, 0.3); 6]);
    let mut rng = Rng::new(seed);
    let mut e = Echogram::new();
    let src = v(1.5, 1.5, 1.5);
    let lis = v(size.0 - 1.5, size.1 - 1.5, 1.6);
    trace(&room, src, lis, 12_000, [1.0; 3], &mut rng, &mut e);
    estimate_reverb(&e).rt60
}

fn eyring(size: (f32, f32, f32), alpha: f32) -> f32 {
    let vol = size.0 * size.1 * size.2;
    let s = 2.0 * (size.0 * size.1 + size.0 * size.2 + size.1 * size.2);
    0.161 * vol / (-s * (1.0 - alpha).ln())
}

fn sabine(size: (f32, f32, f32), alpha: f32) -> f32 {
    let vol = size.0 * size.1 * size.2;
    let s = 2.0 * (size.0 * size.1 + size.0 * size.2 + size.1 * size.2);
    0.161 * vol / (s * alpha)
}

#[test]
fn damped_rooms_match_diffuse_field_theory() {
    // α where 64 bounces reach ≥ −40 dB: the decay is fully resolved
    for (size, alpha) in [((8.0, 6.0, 3.0), 0.2f32), ((8.0, 6.0, 3.0), 0.35), ((20.0, 15.0, 10.0), 0.2)] {
        let rt = traced_rt60(size, alpha, 11)[1];
        let (lo, hi) = (eyring(size, alpha), sabine(size, alpha));
        assert!(
            rt > 0.85 * lo && rt < 1.15 * hi,
            "{size:?} α={alpha}: traced {rt:.3} s, Eyring {lo:.3} s, Sabine {hi:.3} s"
        );
    }
}

#[test]
fn rt60_scales_with_room_size() {
    // same walls, 2.5× linear scale ⇒ V/S (and RT) 2.5× longer
    let small = traced_rt60((8.0, 6.0, 3.0), 0.25, 12)[1];
    let big = traced_rt60((20.0, 15.0, 7.5), 0.25, 13)[1];
    let ratio = big / small;
    assert!((ratio - 2.5).abs() < 0.4, "RT ratio big/small {ratio:.2}, want ≈ 2.5");
}

#[test]
fn flat_walls_give_flat_rt60() {
    // low vs mid only: the high band may legitimately be shortened by
    // air absorption (see fx_air_absorption's ignored test); below
    // 2.5 kHz air costs < 2 % of RT60 in a room this size
    let rt = traced_rt60((8.0, 6.0, 3.0), 0.2, 14);
    assert!((rt[1] - rt[0]).abs() < 0.03 * rt[0], "rt60 not flat: {rt:?}");
}

/// Small live room, α = 0.04 flat: Eyring ≈ 3.2 s. The tracer stops
/// every ray after 64 bounces — in an 8×6×3 m room that is ≈ 0.6 s of
/// travel and only −11 dB of decay — so the Schroeder fit sees an
/// artificial cliff and reports ≈ 1.1 s (measured). The tracer's own
/// comment rejects Russian roulette because "the decay curve IS the long
/// tail"; the fixed bounce cap is the same truncation in disguise.
/// Fix direction: cap by TIME (dist_total/c > MAX_TIME already breaks)
/// or by energy, not by a fixed bounce count.
#[test]
#[ignore = "BUG: 64-bounce cap in tracer::trace truncates live-room tails — α=0.04 8×6×3 room traces RT60 ≈1.1 s vs Eyring 3.2 s"]
fn live_room_rt60_is_not_truncated() {
    let size = (8.0, 6.0, 3.0);
    let alpha = 0.04;
    let rt = traced_rt60(size, alpha, 15)[1];
    let want = eyring(size, alpha);
    assert!(
        (rt / want - 1.0).abs() < 0.2,
        "traced {rt:.2} s vs Eyring {want:.2} s (Sabine {:.2} s)",
        sabine(size, alpha)
    );
}
