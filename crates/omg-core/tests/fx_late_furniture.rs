//! Effect: furniture absorption in the LATE field — absorbing boxes in a
//! room (sofas, shelves, a bar counter) overlaid on the room geometry via
//! `scene::WithPanels` and traced by `tracer::trace`.
//!
//! Physical claim guaranteed: adding absorbing furniture to a room
//! shortens its RT60; more absorbing area ⇒ shorter, roughly as Sabine
//! predicts (RT ∝ V / ΣαS); furniture made of the walls' own material
//! leaves the RT nearly unchanged (it's the ABSORPTION, not mere
//! presence, that deadens the room); and a panel only matters where it
//! is — rays genuinely bounce off it (a reflective panel between source
//! and listener changes the echogram).
//!
//! Combined in the demo: `omg_scene::world::WorldSim` pushes per-room
//! furniture into its late panels (`late_panels`, quality::furniture_on);
//! the living room / club RT60 checks in omg-scene see it combined with
//! door leaves and wall transmission.

mod common;

use common::{flat_opaque, sabine, v};
use omg_core::material::Material;
use omg_core::rng::Rng;
use omg_core::scene::{Shoebox, WithPanels};
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::vec3::Vec3;

const SIZE: Vec3 = Vec3::new(8.0, 6.0, 3.0);
const SRC: Vec3 = Vec3::new(1.5, 1.5, 1.5);
const LIS: Vec3 = Vec3::new(6.5, 4.5, 1.6);

/// Walls flat α = 0.15 — Sabine RT ≈ 0.86 s empty. (Kept damped enough
/// that the tracer's 64-bounce cap reaches −45 dB: livelier rooms hit the
/// truncation bug documented in fx_late_rt60.rs and would skew ratios.)
const WALL: Material = flat_opaque(0.15, 0.3);
/// Upholstery: strongly absorbing in every band.
const SOFA: Material = flat_opaque(0.6, 0.5);

/// Sofa-sized boxes standing on the floor, clear of SRC and LIS.
fn sofas(n: usize) -> Vec<(Vec3, Vec3, Material)> {
    let spots = [
        (v(3.0, 0.2, 0.0), v(5.0, 1.2, 0.8)),
        (v(3.0, 4.8, 0.0), v(5.0, 5.8, 0.8)),
        (v(0.2, 3.5, 0.0), v(1.2, 5.5, 0.8)),
        (v(6.8, 0.5, 0.0), v(7.8, 2.5, 0.8)),
    ];
    spots[..n].iter().map(|&(a, b)| (a, b, SOFA)).collect()
}

fn rt60_mid(panels: &[(Vec3, Vec3, Material)], seed: u64) -> f32 {
    let room = Shoebox::new(SIZE, [WALL; 6]);
    let geo = WithPanels { base: &room, panels };
    let mut rng = Rng::new(seed);
    let mut e = Echogram::new();
    trace(&geo, SRC, LIS, 12_000, [1.0; 3], &mut rng, &mut e);
    estimate_reverb(&e).rt60[1]
}

/// Sabine with furniture: room surface minus the floor footprint the
/// boxes cover, plus the boxes' five exposed faces at their α.
fn sabine_with(panels: &[(Vec3, Vec3, Material)]) -> f32 {
    let s_room = 2.0 * (SIZE.x * SIZE.y + SIZE.x * SIZE.z + SIZE.y * SIZE.z);
    let mut a = s_room * WALL.absorption[1];
    let mut vol = SIZE.x * SIZE.y * SIZE.z;
    for (mn, mx, m) in panels {
        let (dx, dy, dz) = (mx.x - mn.x, mx.y - mn.y, mx.z - mn.z);
        let foot = dx * dy;
        let exposed = foot + 2.0 * (dx * dz + dy * dz);
        a += exposed * m.absorption[1] - foot * WALL.absorption[1];
        vol -= dx * dy * dz;
    }
    sabine(vol, a)
}

#[test]
fn furniture_shortens_rt60_monotonically() {
    let rts: Vec<f32> = (0..=4).map(|n| rt60_mid(&sofas(n), 100 + n as u64)).collect();
    for n in 1..rts.len() {
        assert!(
            rts[n] < rts[n - 1],
            "adding sofa #{n} did not shorten RT60: {rts:?}"
        );
    }
    // four sofas: Sabine 0.86 s → 0.55 s (×0.64)
    assert!(rts[4] < 0.75 * rts[0], "4 sofas only took RT60 {:.2} → {:.2} s", rts[0], rts[4]);
}

#[test]
fn rt60_tracks_sabine_with_furniture() {
    for n in [0usize, 2, 4] {
        let p = sofas(n);
        let got = rt60_mid(&p, 7 + n as u64);
        let want = sabine_with(&p);
        let ratio = got / want;
        assert!(
            (0.7..1.3).contains(&ratio),
            "{n} sofas: traced RT60 {got:.2} s vs Sabine {want:.2} s (ratio {ratio:.2})"
        );
    }
}

#[test]
fn more_absorbent_furniture_is_deader() {
    let make = |alpha: f32| -> Vec<(Vec3, Vec3, Material)> {
        sofas(4).into_iter().map(|(a, b, _)| (a, b, flat_opaque(alpha, 0.5))).collect()
    };
    let rt_soft = rt60_mid(&make(0.3), 21);
    let rt_softer = rt60_mid(&make(0.9), 21);
    assert!(rt_softer < 0.85 * rt_soft, "α 0.3 → 0.9 furniture: {rt_soft:.2} → {rt_softer:.2} s");
}

#[test]
fn furniture_of_wall_material_barely_changes_rt60() {
    // same α as the walls: only volume/area bookkeeping changes
    // (Sabine predicts ×0.86 for these four boxes; absorbing sofas ×0.64)
    let p: Vec<_> = sofas(4).into_iter().map(|(a, b, _)| (a, b, WALL)).collect();
    let empty = rt60_mid(&[], 31);
    let same = rt60_mid(&p, 31);
    let ratio = same / empty;
    assert!(
        (0.78..1.05).contains(&ratio),
        "inert furniture moved RT60 {empty:.2} → {same:.2} s (ratio {ratio:.2})"
    );
}

#[test]
fn panels_are_really_hit_by_rays() {
    // a fully absorbing wall-to-wall slab between source and listener
    // must kill almost everything (they share no air path any more)
    let slab = [(v(3.9, 0.0, 0.0), v(4.1, SIZE.y, SIZE.z), flat_opaque(1.0, 0.0))];
    let room = Shoebox::new(SIZE, [WALL; 6]);
    let geo = WithPanels { base: &room, panels: &slab };
    let mut rng = Rng::new(3);
    let mut e = Echogram::new();
    trace(&geo, SRC, LIS, 12_000, [1.0; 3], &mut rng, &mut e);
    let total: f32 = e.bins.iter().map(|b| b[1]).sum();
    assert_eq!(total, 0.0, "energy crossed an absorbing, opaque full-height slab: {total}");
}
