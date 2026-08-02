//! Engine-module A/B switches (quality panel): diffraction and
//! furniture acoustics must each change the field the way their
//! tooltip claims, live, and restore cleanly. One test fn — the
//! switches are process-global, so the sequence must not interleave.

use omg_core::params::ParamBlock;
use omg_scene::quality;
use omg_scene::walkthrough;
use omg_scene::world::WorldSim;

/// Furniture geometry and material tables are parallel arrays —
/// authored side by side, asserted here so they can never drift.
#[test]
fn furniture_tables_are_parallel() {
    for room in 0..12 {
        assert_eq!(
            walkthrough::furniture(room).len(),
            walkthrough::furniture_mats(room).len(),
            "room {room}: furniture/material tables out of step"
        );
    }
}

fn taps_mid(pb: &ParamBlock) -> f32 {
    pb.taps.iter().map(|t| t.gains[1]).sum::<f32>()
}

#[test]
fn module_switches_shape_the_field_and_restore() {
    quality::set_early(1);
    let mut w = WorldSim::new();

    // --- diffraction: off-axis outside the hall door the voice arrives
    // partly by bending around the jamb
    const VOICE: usize = 1;
    let read = |w: &mut WorldSim, x: f32, y: f32, src: usize| {
        for _ in 0..12 {
            let _ = w.tick_at(x, y, 0.0);
        }
        let (blocks, _) = w.tick_at(x, y, 0.0);
        taps_mid(&blocks[src])
    };
    let base = read(&mut w, 3.0, 25.2, VOICE);
    quality::set_module(0, false);
    let hard = read(&mut w, 3.0, 25.2, VOICE);
    quality::set_module(0, true);
    let back = read(&mut w, 3.0, 25.2, VOICE);
    eprintln!("diffraction: on {base:.5} off {hard:.5} restored {back:.5}");
    assert!(hard < base, "diffraction off must harden the shadow: {hard} vs {base}");
    assert!(back > 0.85 * base, "re-enabling must restore: {back} vs {base}");

    // --- furniture: the piano heard through the living room's
    // bookshelf divider (local 5.4–5.8 × 0.4–3.4, 2.2 m tall)
    const PIANO: usize = 0;
    let occluded = read(&mut w, 6.8, 1.5, PIANO);
    quality::set_module(1, false);
    let transparent = read(&mut w, 6.8, 1.5, PIANO);
    quality::set_module(1, true);
    let re_occluded = read(&mut w, 6.8, 1.5, PIANO);
    eprintln!("furniture: on {occluded:.5} off {transparent:.5} restored {re_occluded:.5}");
    assert!(
        transparent > 1.3 * occluded,
        "transparent furniture must be markedly louder: {transparent} vs {occluded}"
    );
    assert!(
        re_occluded < 0.85 * transparent,
        "re-enabling must shadow again: {re_occluded} vs {transparent}"
    );

    // --- late field: the FURNISHED living room must decay faster than
    // the empty one (sofa/armchair/books absorb the tail), reversibly
    // one world trace per tick round-robins 11 sources and the
    // echogram EMA unwinds slowly — give each state time to converge
    let rt60_at = |w: &mut WorldSim, settle: usize| {
        for _ in 0..settle {
            let _ = w.tick_at(6.0, 2.0, 0.0);
        }
        let (blocks, _) = w.tick_at(6.0, 2.0, 0.0);
        blocks[PIANO].reverb.rt60[1]
    };
    let rt_on = rt60_at(&mut w, 120);
    quality::set_module(1, false);
    let rt_off = rt60_at(&mut w, 240);
    quality::set_module(1, true);
    let rt_back = rt60_at(&mut w, 240);
    eprintln!("late rt60 mid: furnished {rt_on:.2}s empty {rt_off:.2}s restored {rt_back:.2}s");
    assert!(
        rt_off > 1.05 * rt_on,
        "the empty room must ring longer: furnished {rt_on} vs empty {rt_off}"
    );
    assert!(
        (rt_back - rt_on).abs() < 0.25 * rt_on,
        "re-furnishing must restore the decay: {rt_back} vs {rt_on}"
    );
}
