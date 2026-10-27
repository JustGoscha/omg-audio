//! Regression: the bells were LOUDER inside houses (and in the buried
//! bunker) than in the open with the tower in plain view. Three faults
//! stacked up: the bell hung inside the solid shaft instead of the open
//! belfry stage; door-routed paths never paid the source → doorway leg
//! (the tower walls an outdoor listener pays); and the over-silhouette
//! bend dropped through roofs, parapets and earth for free. In both
//! early-reflection backends the peal must now be loudest in the open,
//! and every indoor room — the bunker included — must be quieter.

use omg_scene::{quality, world::WorldSim};

const BELLS: usize = 7;

fn level_db(w: &mut WorldSim, x: f32, y: f32, z: f32) -> f32 {
    let mut out = None;
    for _ in 0..40 {
        out = Some(w.tick_at_z(x, y, z, 0.0));
    }
    let (blocks, _) = out.unwrap();
    let pb = &blocks[BELLS];
    let taps: f32 = pb.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum();
    let wet = pb.reverb.level[1] * pb.reverb.level[1];
    10.0 * (taps + wet).max(1e-20).log10()
}

#[test]
fn bells_are_loudest_in_the_open() {
    for mode in [0u32, 1] {
        quality::set_early(mode);
        let mut w = WorldSim::new();
        // open field, tower in plain view, ~260 m out
        let open = level_db(&mut w, 20.0, 60.0, 1.6);
        // free field + ground at 261 m for gain 30 is ≈ −15 dB: the
        // peal must reach the open field essentially unobstructed
        assert!(open > -22.0, "mode {mode}: open field too quiet: {open:.1} dB");
        for (name, x, y, z) in [
            ("living room", 4.0, 3.0, 1.6),
            ("great hall", 7.0, 19.0, 1.6),
            ("old house", 27.5, 19.5, 1.6),
            ("old house upper", 27.5, 19.5, 4.6),
            ("cathedral", 8.0, 63.0, 1.6),
            ("bunker", 37.0, 10.0, -1.4),
        ] {
            let inside = level_db(&mut w, x, y, z);
            eprintln!("mode {mode}: {name:>16} {inside:7.1} dB  (open {open:.1})");
            assert!(
                inside < open - 10.0,
                "mode {mode}: {name} ({inside:.1} dB) must be well under the open field ({open:.1} dB)"
            );
        }
    }
}
