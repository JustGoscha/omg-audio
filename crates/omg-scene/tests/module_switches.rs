//! Engine-module A/B switches (quality panel): diffraction and
//! furniture acoustics must each change the field the way their
//! tooltip claims, live, and restore cleanly. One test fn — the
//! switches are process-global, so the sequence must not interleave.

use omg_core::params::ParamBlock;
use omg_scene::quality;
use omg_scene::world::WorldSim;

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
}
