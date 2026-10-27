//! Effect: MOVING-SOURCE WET STALENESS COMPENSATION — `Sim::reverb_world`
//! between traces (budget 0).
//!
//! Physical claim guaranteed: the world late field is an absolute
//! echogram measured at one source↔listener separation d₀ and refreshed
//! round-robin. Between refreshes the reported wet level follows the
//! CURRENT separation: level(d) = level₀ · min(1, (d₀ + 1)/(d + 1)) per
//! band — a source driving away takes its wet bed along, a source coming
//! closer is never boosted (the next trace corrects upward). RT60 and the
//! late direction are untouched (same echogram), no trace budget is
//! spent, and the next trace re-anchors d₀.
//!
//! Combined in the demo: `WorldSim` world-late path for every source
//! (`world_late[si].reverb_world`, one trace per tick round-robin) —
//! `tests/car_drone.rs` (`departing_car_takes_its_wet_bed_along`).

mod common;

use common::*;
use omg_core::material::Material;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;
use omg_scene::sim::Sim;

fn hall() -> Shoebox {
    Shoebox::new(Vec3::new(30.0, 12.0, 6.0), [Material::CONCRETE; 6])
}

#[test]
fn wet_level_follows_separation_between_traces() {
    let world = hall();
    let rcv = Vec3::new(2.0, 6.0, 1.6);
    let src0 = Vec3::new(4.0, 6.0, 1.5);
    let d0 = (src0 - rcv).length();
    let mut sim = Sim::new();

    let mut budget = 1;
    let (rp0, dir0, an0) = sim.reverb_world(&world, &[], src0, rcv, &mut budget);
    assert_eq!(budget, 0, "the first call traces and spends one unit");
    assert!(rp0.level.iter().all(|&l| l > 1e-4), "a concrete hall rings: {:?}", rp0.level);

    // drive away along +x, no trace budget
    let mut prev = rp0.level;
    for x in [4.0f32, 6.0, 10.0, 16.0, 28.0] {
        let src = Vec3::new(x, 6.0, 1.5);
        let d = (src - rcv).length();
        let mut zero = 0;
        let (rp, dir, an) = sim.reverb_world(&world, &[], src, rcv, &mut zero);
        assert_eq!(zero, 0, "budget 0: no trace, no debt");
        let k = ((d0 + 1.0) / (d + 1.0)).min(1.0);
        for b in 0..NBANDS {
            assert_rel(rp.level[b], rp0.level[b] * k, 1e-5, "level ∝ (d₀+1)/(d+1)");
            assert!(rp.level[b] <= prev[b] + 1e-9, "departing never raises the wet level");
        }
        assert_eq!(rp.rt60, rp0.rt60, "decay time is the room's, not the distance's");
        assert_eq!((dir, an), (dir0, an0), "late direction unchanged without a trace");
        prev = rp.level;
    }

    // coming closer than d₀: attenuate only, never boost
    let mut zero = 0;
    let (rp, _, _) = sim.reverb_world(&world, &[], Vec3::new(2.6, 6.0, 1.5), rcv, &mut zero);
    assert_eq!(rp.level, rp0.level, "closer than the anchor: k clamps at 1");

    // a fresh trace re-anchors: far away and measured, no extra scaling
    let far = Vec3::new(28.0, 6.0, 1.5);
    let mut one = 1;
    let (rp_far, _, _) = sim.reverb_world(&world, &[], far, rcv, &mut one);
    assert_eq!(one, 0, "moved > 0.25 m: the gate fires and spends the unit");
    let mut zero = 0;
    let (rp_same, _, _) = sim.reverb_world(&world, &[], far, rcv, &mut zero);
    assert_eq!(rp_same.level, rp_far.level, "anchored at the new separation: k = 1");
}
