//! Effect: through-body SEEP of furniture in the early field —
//! `pt::record_for_occ` with a partly transmissive `pt::Aabb`
//! (a sofa / upholstered partition, not stone).
//!
//! The opaque-occluder case (shadow, knife-edge hand-off, no level step)
//! is covered by tests/pt_occluder.rs; this file covers what changes when
//! the piece lets sound THROUGH.
//!
//! Physical claim guaranteed: a blocked direct path takes, per band,
//! the louder of (a) the straight seep through the body, exactly
//! air(d)/d × T_b, and (b) the knife-edge bend around the silhouette —
//! so bass seeps through a sofa (straight delay, ≥ 5 dB above stone)
//! while treble still bends past it (identical to stone); the tap's
//! single delay follows the low band's winner; and a blocked
//! REFLECTION is not deleted but carries exactly the body's
//! transmission (air/d × wall reflection × T).
//!
//! Combined in the demo: `omg_scene::world::WorldSim` furniture
//! (`walkthrough::furniture`, extras with per-piece transmission) in the
//! living room / club early fields.

mod common;

use common::{db, v};
use omg_core::material::{air_attenuation, Material};
use omg_core::pt::{record_for_occ, solve_chain_trans, Aabb, OPAQUE};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};

/// Upholstery-ish amplitude transmission: bass passes, treble dies.
const SEEP: [f32; NBANDS] = [0.6, 0.15, 0.02];
const SRC: Vec3 = Vec3::new(1.5, 3.0, 1.2);
const LIS: Vec3 = Vec3::new(6.5, 3.0, 1.2);

fn room() -> Shoebox {
    Shoebox::new(v(8.0, 6.0, 3.0), [Material::CONCRETE; 6])
}

/// A tall, wide, 1 m deep piece squarely between SRC and LIS.
fn piece(transmission: [f32; NBANDS]) -> Aabb {
    Aabb { min: v(3.5, 1.0, 0.0), max: v(4.5, 5.0, 2.5), transmission }
}

#[test]
fn bass_seeps_straight_through_at_exactly_air_over_r_times_t() {
    let r = record_for_occ(&room(), &[], 0, SRC, LIS, &[piece(SEEP)]).expect("direct must survive");
    let d = 5.0;
    let air = air_attenuation(d);
    let seep0 = air[0] / d * SEEP[0];
    assert!((r.gains[0] - seep0).abs() < 1e-4 * seep0, "low {} want seep {seep0}", r.gains[0]);
    // low band wins by seeping ⇒ the tap keeps the STRAIGHT delay
    assert!((r.delay_s - d / SPEED_OF_SOUND).abs() < 1e-6, "delay {} not straight", r.delay_s);
    // every band at least the seep
    for b in 0..NBANDS {
        assert!(r.gains[b] >= air[b] / d * SEEP[b] * (1.0 - 1e-4), "band {b} below its seep");
    }
}

#[test]
fn seep_beats_stone_in_the_bass_but_treble_still_bends() {
    let soft = record_for_occ(&room(), &[], 0, SRC, LIS, &[piece(SEEP)]).unwrap();
    let stone = record_for_occ(&room(), &[], 0, SRC, LIS, &[piece(OPAQUE)]).unwrap();
    let clear = record_for_occ(&room(), &[], 0, SRC, LIS, &[]).unwrap();
    // per band: a more transmissive body can only be louder
    for b in 0..NBANDS {
        assert!(soft.gains[b] >= stone.gains[b] * (1.0 - 1e-5), "band {b}: soft < stone");
        assert!(soft.gains[b] < clear.gains[b], "band {b}: blocked louder than clear");
    }
    // bass: seep (−4.4 dB) clearly above the stone's bend (≈ −7.8 dB rel.)
    let gain_db = db(soft.gains[0]) - db(stone.gains[0]);
    assert!(gain_db > 5.0, "bass seep only {gain_db:.1} dB over stone");
    // treble: T=0.02 seep loses to the bend ⇒ same as stone, bit-for-bit
    assert!(
        (soft.gains[2] - stone.gains[2]).abs() < 1e-6 * stone.gains[2].max(1e-9),
        "treble should be bend-dominated in both: {} vs {}",
        soft.gains[2],
        stone.gains[2]
    );
    // stone's low band also bends ⇒ its tap moves to the LONGER bent delay
    assert!(stone.delay_s > soft.delay_s + 1e-4, "stone delay {} vs soft {}", stone.delay_s, soft.delay_s);
    // and the seep spectrum is bass-heavy relative to the clear path
    let tilt = |g: [f32; NBANDS]| db(g[0]) - db(g[2]);
    assert!(tilt(soft.gains) > tilt(clear.gains) + 6.0, "seep not bass-heavy");
}

#[test]
fn blocked_reflection_carries_the_body_transmission() {
    // x·max wall bounce: LIS → wall (clear) → back across the room
    // THROUGH the piece to SRC. Not deleted: it pays T exactly once.
    let room = room();
    let occ = [piece(SEEP)];
    let (_, _, _, trans) = solve_chain_trans(&room, &[1], SRC, LIS, &occ).expect("valid chain");
    for b in 0..NBANDS {
        assert!((trans[b] - SEEP[b]).abs() < 1e-6, "band {b}: trans {} want {}", trans[b], SEEP[b]);
    }
    let r = record_for_occ(&room, &[1], 0, SRC, LIS, &occ).expect("seeping reflection must exist");
    // image of SRC across x = 8 → (14.5, 3, 1.2): 8 m from LIS
    let d = 8.0;
    let air = air_attenuation(d);
    let refl = Material::CONCRETE.reflection_amplitude();
    for b in 0..NBANDS {
        let want = air[b] / d * refl[b] * SEEP[b];
        assert!((r.gains[b] - want).abs() < 1e-4 * want, "band {b}: {} want {want}", r.gains[b]);
    }
    assert!((r.delay_s - d / SPEED_OF_SOUND).abs() < 1e-6);
}
