//! Effect: SPEAKER RIG — `Sim::update_multi` (N same-room emitters playing
//! one sample-locked signal).
//!
//! Physical claim guaranteed: every emitter gets its own order-2 image
//! set whose taps equal a lone source's at that position, scaled by
//! 1/√N — the rig is POWER-split: the incoherent power sum over all
//! taps equals one source of the same gain (exactly, when the emitters
//! coincide), so adding speakers spreads the sound, it does not make it
//! louder. Each emitter's direct tap is sample-exact (|E_i − L|/c), tap
//! keys are disjoint per emitter (64 slots each, stable identities), and
//! one shared late-field estimate (traced from the first emitter) is
//! independent of the rig size.
//!
//! Combined in the demo: the Club PA (`CLUB_SPEAKERS`, 4 corners) via
//! `WorldSim::ism_source_block` when the listener is inside the club
//! (`tests/club_boom.rs`, `tests/sealed_club.rs` exercise the club).

mod common;

use common::*;
use omg_core::ism::image_source_taps;
use omg_core::material::Material;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};
use omg_scene::sim::Sim;

const SR: f32 = 48_000.0;

fn club() -> Shoebox {
    Shoebox::new(Vec3::new(10.0, 12.0, 4.5), [Material::CONCRETE; 6])
}

fn corners() -> [Vec3; 4] {
    [
        Vec3::new(1.5, 1.5, 1.5),
        Vec3::new(8.5, 1.5, 1.5),
        Vec3::new(1.5, 10.5, 1.5),
        Vec3::new(8.5, 10.5, 1.5),
    ]
}

#[test]
fn each_emitter_is_a_scaled_image_set_with_sample_exact_delays() {
    let room = club();
    let lis = Vec3::new(4.0, 5.0, 1.6);
    let ems = corners();
    let pb = Sim::new().update_multi(&room, &ems, lis, 0.0);
    let amp = 1.0 / (ems.len() as f32).sqrt();
    let mut lone = Vec::new();
    image_source_taps(&room, ems[0], lis, 2, &mut lone);
    assert_eq!(lone.len(), 25, "order-2 shoebox: 1 + 6 + 18 images");
    assert_eq!(pb.taps.len(), ems.len() * lone.len());

    for (ei, e) in ems.iter().enumerate() {
        image_source_taps(&room, *e, lis, 2, &mut lone);
        let base = ei as u32 * 64;
        let mine: Vec<_> = pb.taps.iter().filter(|t| t.key >= base && t.key < base + 64).collect();
        assert_eq!(mine.len(), lone.len(), "emitter {ei}: its own full image set in its key slot");
        for l in &lone {
            let t = tap(&pb.taps, base + l.key).expect("same path identity, offset key");
            assert_eq!(t.delay_s, l.delay_s, "emitter {ei} key {}: geometry untouched", l.key);
            for b in 0..NBANDS {
                assert_rel(t.gains[b], l.gains[b] * amp, 1e-6, "power-split gain");
            }
        }
        // direct path: sample-exact
        let d = mine.iter().min_by(|a, b| a.delay_s.total_cmp(&b.delay_s)).unwrap();
        let exact = (*e - lis).length() / SPEED_OF_SOUND * SR;
        assert!((d.delay_s * SR - exact).abs() < 1e-3, "emitter {ei}: {} vs {exact} samples", d.delay_s * SR);
    }
    // keys are unique across the whole rig
    let mut keys: Vec<u32> = pb.taps.iter().map(|t| t.key).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), pb.taps.len(), "no two rig paths share an identity");
}

#[test]
fn rig_power_equals_one_source_of_the_same_gain() {
    let room = club();
    let lis = Vec3::new(4.0, 5.0, 1.6);
    let spot = Vec3::new(6.0, 8.0, 1.5);
    let one = Sim::new().update_multi(&room, &[spot], lis, 0.0);
    for n in [2usize, 4, 8] {
        let rig = Sim::new().update_multi(&room, &vec![spot; n], lis, 0.0);
        for b in 0..NBANDS {
            assert_rel(power(&rig.taps, b), power(&one.taps, b), 1e-5, "co-located rig power");
        }
    }
    // spread out, the rig's power is the MEAN of its emitters' — never
    // the sum (four speakers are not 6 dB louder than one)
    let ems = corners();
    let rig = Sim::new().update_multi(&room, &ems, lis, 0.0);
    let mut mean = [0.0f32; NBANDS];
    for e in &ems {
        let s = Sim::new().update_multi(&room, &[*e], lis, 0.0);
        for b in 0..NBANDS {
            mean[b] += power(&s.taps, b) / ems.len() as f32;
        }
    }
    for b in 0..NBANDS {
        assert_rel(power(&rig.taps, b), mean[b], 1e-5, "spread rig power = mean emitter power");
    }
}

#[test]
fn rig_shares_one_reverb_independent_of_size() {
    let room = club();
    let lis = Vec3::new(4.0, 5.0, 1.6);
    let ems = corners();
    let one = Sim::new().update_multi(&room, &ems[..1], lis, 0.0).reverb;
    let four = Sim::new().update_multi(&room, &ems, lis, 0.0).reverb;
    assert_eq!(one.rt60, four.rt60, "late field traced once from the first emitter");
    assert_eq!(one.level, four.level, "the late send is not power-split again");
    assert!(four.level[1] > 0.0 && four.rt60[1] > 0.3, "a concrete club rings: {four:?}");
}

#[test]
fn rig_yaw_rotates_every_emitter() {
    let room = club();
    let lis = Vec3::new(4.0, 5.0, 1.6);
    let ems = corners();
    let a = Sim::new().update_multi(&room, &ems, lis, 0.0);
    let b = Sim::new().update_multi(&room, &ems, lis, core::f32::consts::FRAC_PI_2);
    for (ta, tb) in a.taps.iter().zip(b.taps.iter()) {
        assert_eq!(ta.key, tb.key);
        assert!((tb.dir[0] - ta.dir[1]).abs() < 1e-5 && (tb.dir[1] + ta.dir[0]).abs() < 1e-5);
        assert_eq!(ta.gains, tb.gains);
    }
}
