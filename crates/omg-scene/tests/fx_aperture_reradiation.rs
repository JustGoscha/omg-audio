//! Effect: APERTURE RE-RADIATION (Huygens secondary source) —
//! `Sim::update_routed(room, virt, lis, yaw, extra_dist, muffle, &[])`
//! (APERTURE_RERADIATION = 0.6 m) and the exit aperture's Lambert
//! obliquity `Routed::lambert_toward`.
//!
//! Physical claim guaranteed: a source heard through a doorway is
//! re-radiated by the opening. The field arriving at the door falls as
//! 1/extra beyond the 0.6 m near-field radius (flat inside it), then
//! spreads with the IN-ROOM distance — every tap of the in-room image
//! set is scaled by muffle · 0.6/max(extra, 0.6) and delayed by
//! extra/c, so doubling the pre-door path halves the amplitude (−6 dB)
//! regardless of where in the room the listener stands; the reflections
//! additionally pay the aperture decoherence (×0.6) while the direct tap
//! keeps its strength. The room's late send drops with the pre-door
//! path too. The diffuse radiator is Lambertian: 1 on-axis, |cos θ|
//! off-axis, floored at 0.15 edge-on (energy the jambs diffract).
//!
//! Combined in the demo: every portal-routed source in `WorldSim`'s ISM
//! pipeline (`ism_source_block` → `update_routed(.., routed.extra_dist,
//! routed.muffle, ..)`) and the remote-wet radiators (`lambert_toward`);
//! `world.rs` tests `closing_a_door_muffles_the_portal`,
//! `distant_cars_stay_distant_indoors`.

mod common;

use common::*;
use omg_core::material::Material;
use omg_core::params::ParamBlock;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};
use omg_scene::sim::Sim;
use omg_scene::walkthrough::price_chain;

const R0: f32 = 0.6; // APERTURE_RERADIATION

fn room() -> Shoebox {
    Shoebox::new(Vec3::new(8.0, 6.0, 3.0), [Material::DRYWALL; 6])
}

fn routed(lis: Vec3, extra: f32, muffle: [f32; NBANDS]) -> ParamBlock {
    // virtual source on the door plane (x ≈ 0 wall), as the router places it
    let virt = Vec3::new(0.06, 3.0, 1.5);
    Sim::new().update_routed(&room(), virt, lis, 0.0, extra, muffle, &[])
}

fn direct_idx(pb: &ParamBlock) -> usize {
    let d = direct(&pb.taps);
    pb.taps.iter().position(|t| t.key == d.key).unwrap()
}

#[test]
fn huygens_gain_falls_as_one_over_extra_and_delay_grows() {
    let lis = Vec3::new(5.0, 2.0, 1.6);
    let base = routed(lis, 0.0, [1.0; NBANDS]);
    let di = direct_idx(&base);
    let mut prev_gain = f32::MAX;
    for extra in [0.2f32, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2] {
        let pb = routed(lis, extra, [1.0; NBANDS]);
        assert_eq!(pb.taps.len(), base.taps.len(), "same in-room image set");
        let a_in = R0 / extra.max(R0);
        for (i, (t, b0)) in pb.taps.iter().zip(base.taps.iter()).enumerate() {
            assert_eq!(t.key, b0.key);
            assert_rel(t.delay_s - b0.delay_s, extra / SPEED_OF_SOUND, 1e-3, "pre-door delay");
            let decoh = if i == di { 1.0 } else { 0.6 };
            for b in 0..NBANDS {
                assert_rel(t.gains[b], b0.gains[b] * a_in * decoh, 1e-5, "Huygens tap gain");
            }
        }
        let g = pb.taps[di].gains[1];
        assert!(g <= prev_gain, "growing extra_dist must never raise the level");
        prev_gain = g;
    }
    // −6 dB per doubling beyond the near field; flat inside it
    let g = |e: f32| routed(lis, e, [1.0; NBANDS]).taps[di].gains[1];
    assert_rel(g(4.8) / g(2.4), 0.5, 1e-4, "doubling halves");
    assert_rel(g(0.2), g(0.6), 1e-6, "near field (≤ 0.6 m) is the incoming field");
    assert_rel(g(0.6), base.taps[di].gains[1], 1e-6, "at the radius: unity re-radiation");
}

#[test]
fn reradiation_decays_with_in_room_distance_not_total_path() {
    // The aperture factor is the same wherever the listener stands: the
    // door re-radiates, the room's own 1/d does the rest. Folding extra
    // into each path (1/(d_room + extra)) would make near and far
    // listeners converge — it must not.
    let extra = 12.0;
    let near = Vec3::new(1.0, 3.0, 1.6);
    let far = Vec3::new(7.5, 1.0, 1.6);
    let ratio = |lis: Vec3| {
        let (b, r) = (routed(lis, 0.0, [1.0; NBANDS]), routed(lis, extra, [1.0; NBANDS]));
        let di = direct_idx(&b);
        r.taps[di].gains[1] / b.taps[di].gains[1]
    };
    assert_rel(ratio(near), ratio(far), 1e-5, "aperture factor independent of in-room distance");
    assert_rel(ratio(near), R0 / extra, 1e-5, "aperture factor = 0.6/extra");
    let gn = direct(&routed(near, extra, [1.0; NBANDS]).taps).gains[1];
    let gf = direct(&routed(far, extra, [1.0; NBANDS]).taps).gains[1];
    let (dn, df) = ((near - Vec3::new(0.06, 3.0, 1.5)).length(), (far - Vec3::new(0.06, 3.0, 1.5)).length());
    assert!(gn / gf > 0.9 * df / dn, "in-room 1/d survives: near {gn} far {gf}");
}

#[test]
fn muffle_is_applied_per_band() {
    let lis = Vec3::new(5.0, 2.0, 1.6);
    let muffle = [0.9, 0.4, 0.1];
    let open = routed(lis, 3.0, [1.0; NBANDS]);
    let bent = routed(lis, 3.0, muffle);
    for (o, m) in open.taps.iter().zip(bent.taps.iter()) {
        for b in 0..NBANDS {
            assert_rel(m.gains[b], o.gains[b] * muffle[b], 1e-5, "per-band muffle");
        }
    }
}

#[test]
fn late_send_drops_with_the_pre_door_path() {
    // the room is excited with energy muffle²/(1 + extra²): its measured
    // late level (∝ √energy) must follow (the tracer is linear in energy)
    let lis = Vec3::new(5.0, 2.0, 1.6);
    let l0 = routed(lis, 0.0, [1.0; NBANDS]).reverb.level[1];
    let mut prev = l0;
    for extra in [1.0f32, 3.0, 9.0] {
        let l = routed(lis, extra, [1.0; NBANDS]).reverb.level[1];
        let expect = l0 / (1.0 + extra * extra).sqrt();
        assert!(l < prev, "late send must fall with extra_dist: {l} after {prev}");
        eprintln!("extra {extra}: late {l:.5} expect {expect:.5}");
        assert_rel(l, expect, 0.02, "late send ∝ 1/√(1 + extra²)");
        prev = l;
    }
}

#[test]
fn lambert_obliquity_of_the_exit_aperture() {
    // a one-door chain through an x = 6 wall: exit axis 0 (normal ±x)
    let d = door((0, 1), (6.0, 2.0), 0);
    let r = price_chain((2.0, 2.0), &[0], (9.0, 2.0), &[d]);
    assert_eq!(r.exit_axis, Some(0));
    let v = r.virt_world;
    let at = |deg: f32, side: f32| {
        let th = deg.to_radians();
        r.lambert_toward((v.0 + side * 4.0 * th.cos(), v.1 + 4.0 * th.sin()))
    };
    assert_rel(at(0.0, 1.0), 1.0, 1e-6, "on-axis");
    assert_rel(at(0.0, -1.0), 1.0, 1e-6, "radiates both faces alike");
    let mut prev = f32::MAX;
    for deg in (0..=90).step_by(5) {
        let l = at(deg as f32, 1.0);
        assert!(l <= prev + 1e-6, "monotone toward grazing: {deg}° {l}");
        let expect = (deg as f32).to_radians().cos().max(0.15);
        assert_rel(l, expect, 1e-3, "Lambert |cos θ| with floor");
        prev = l;
    }
    assert_rel(at(90.0, 1.0), 0.15, 1e-3, "edge-on floor");
    // an axis-1 opening (normal ±y) measures obliquity from y
    let d1 = door((0, 1), (2.0, 6.0), 1);
    let r1 = price_chain((2.0, 2.0), &[0], (2.0, 9.0), &[d1]);
    assert_rel(r1.lambert_toward((r1.virt_world.0, r1.virt_world.1 + 3.0)), 1.0, 1e-6, "axis-1 on-axis");
    assert_rel(r1.lambert_toward((r1.virt_world.0 + 3.0, r1.virt_world.1)), 0.15, 1e-6, "axis-1 edge-on");
    // an unrouted source radiates omnidirectionally
    let same = price_chain((2.0, 2.0), &[], (9.0, 5.0), &[]);
    assert_eq!(same.exit_axis, None);
    assert_eq!(same.lambert_toward((100.0, -40.0)), 1.0);
}
