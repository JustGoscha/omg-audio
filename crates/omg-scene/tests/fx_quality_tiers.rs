//! Effect: QUALITY TIERS — `quality::set_tier`, `Tier::*` budgets,
//! `set_override`, `set_gpu_backend`.
//!
//! Claims guaranteed: a higher tier never has a smaller budget than a
//! lower one (trace rays, world rays, dome rays/events, ISM order) and
//! never a staler trace gate; out-of-range tiers clamp to Low; a manual
//! override wins over every tier (clamped to its lever's range) and 0
//! hands the lever back; the GPU backend multiplies trace rays 8×
//! (capped) but never beats an override; and the levers are LIVE — the
//! ISM order lever changes the tap count `Sim::update` emits (order 3 =
//! 63 images, 2 = 25, 1 = 7). Everything is restored at the end.
//!
//! One test fn: TIER / OVERRIDES / GPU are process-global atomics.
//!
//! Combined in the demo: the web/godot quality panel drives these for the
//! whole `WorldSim` (every Sim, dome, world-late trace) —
//! `tests/module_switches.rs` covers the sibling module switches.

use omg_core::material::Material;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_scene::quality::{self, Tier};
use omg_scene::sim::Sim;

fn ladder(t: Tier) -> (u32, u32, u32, usize, usize, u32) {
    (t.trace_rays(), t.world_rays(), t.gate_max_age(), t.dome_rays(), t.dome_events(), t.ism_order())
}

fn taps_now() -> usize {
    let room = Shoebox::new(Vec3::new(7.0, 5.0, 3.0), [Material::DRYWALL; 6]);
    Sim::new().update(&room, Vec3::new(2.0, 2.0, 1.5), Vec3::new(5.0, 3.0, 1.6), 0.0).taps.len()
}

#[test]
fn tiers_order_budgets_overrides_win_and_everything_restores() {
    // --- tier selection
    for (i, want) in [(0u32, Tier::High), (1, Tier::Med), (2, Tier::Low), (7, Tier::Low)] {
        quality::set_tier(i);
        assert_eq!(quality::tier(), want, "set_tier({i})");
    }
    quality::set_tier(0);

    // --- monotone ladder: High ≥ Med ≥ Low; gate: High refreshes soonest
    let (h, m, l) = (ladder(Tier::High), ladder(Tier::Med), ladder(Tier::Low));
    for (hi, lo) in [(h, m), (m, l)] {
        assert!(hi.0 >= lo.0 && hi.1 >= lo.1 && hi.3 >= lo.3 && hi.4 >= lo.4 && hi.5 >= lo.5, "{hi:?} vs {lo:?}");
        assert!(hi.2 <= lo.2, "higher tier never has a staler gate: {hi:?} vs {lo:?}");
    }
    assert!(h.0 > l.0 && h.3 > l.3, "the ladder actually sheds work: {h:?} vs {l:?}");
    assert_eq!(h, (4096, 512, 8, 384, 6, 3), "High is the historical baseline");

    // --- the ISM lever is live
    assert_eq!(taps_now(), 63, "High: order 3");
    quality::set_tier(2);
    assert_eq!(taps_now(), 25, "Low: order 2");

    // --- overrides win over every tier, clamp to range, 0 restores
    quality::set_override(4, 1);
    for t in 0..3 {
        quality::set_tier(t);
        assert_eq!(quality::tier().ism_order(), 1);
        assert_eq!(taps_now(), 7, "override order 1 under tier {t}");
    }
    quality::set_override(0, 777);
    quality::set_override(1, 3);
    quality::set_override(2, 100);
    quality::set_override(3, 2);
    for t in [Tier::High, Tier::Med, Tier::Low] {
        assert_eq!((t.trace_rays(), t.gate_max_age(), t.dome_rays(), t.dome_events()), (777, 3, 100, 2));
    }
    quality::set_override(0, 1);
    assert_eq!(Tier::High.trace_rays(), 64, "trace-ray override clamps to ≥ 64");
    quality::set_override(2, 10_000);
    assert_eq!(Tier::High.dome_rays(), 384, "dome override clamps to the fan size");
    quality::set_override(4, 9);
    assert_eq!(Tier::High.ism_order(), 3, "ISM override clamps to ≤ 3");
    quality::set_override(99, 5); // out of range: ignored, no panic
    for id in 0..5 {
        quality::set_override(id, 0);
    }
    quality::set_tier(1);
    assert_eq!(ladder(Tier::Med), m, "0 hands every lever back to the tier");

    // --- GPU backend: 8× trace rays (capped), world rays unchanged, override absolute
    quality::set_gpu_backend(true);
    assert_eq!(Tier::High.trace_rays(), 32768);
    assert_eq!(Tier::Low.trace_rays(), 8192);
    assert_eq!(Tier::High.world_rays(), 512, "world rays are CPU-budgeted either way");
    quality::set_override(0, 2000);
    assert_eq!(Tier::High.trace_rays(), 2000, "override beats the GPU multiplier");
    quality::set_override(0, 0);
    quality::set_gpu_backend(false);

    // --- restored
    quality::set_tier(0);
    assert_eq!(quality::tier(), Tier::High);
    assert_eq!(ladder(Tier::High), h);
    assert_eq!(taps_now(), 63);
}
