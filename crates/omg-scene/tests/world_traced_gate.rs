//! C6c gates (behavior): with `early = traced` the portal machinery is
//! off this path entirely — no routing, no virtual sources, no aperture
//! re-radiation, no crossing blend — and the sound field must still be
//! CONTINUOUS through every doorway: walking in through the hall door,
//! walking past it outside, and closing a leaf must all behave, powered
//! by nothing but world-mesh chains + knife-edge diffraction + the room
//! late field.

use omg_core::params::ParamBlock;
use omg_core::NBANDS;
use omg_scene::quality;
use omg_scene::world::WorldSim;

fn total_mid(pb: &ParamBlock) -> f32 {
    // everything audible: dry+reflections, directional wet, diffuse wet
    pb.taps.iter().map(|t| t.gains[1]).sum::<f32>()
        + pb.remote.as_ref().map_or(0.0, |r| r.send[1])
        + pb.reverb.level[1]
}

fn db(v: f32) -> f32 {
    20.0 * v.max(1e-9).log10()
}

/// Walking from the field straight through the hall's north door to the
/// voice: audible everywhere, no level step anywhere along the walk.
#[test]
fn doorway_walk_in_is_continuous() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const VOICE: usize = 1; // Great Hall, (10.5, 20.5)
    let mut prev = f32::NAN;
    let mut y = 27.0f32;
    while y >= 17.0 {
        for _ in 0..6 {
            let _ = w.tick_at(7.0, y, 0.0);
        }
        let (blocks, _) = w.tick_at(7.0, y, 0.0);
        let cur = total_mid(&blocks[VOICE]);
        assert!(cur > 1e-4, "voice inaudible at (7, {y})");
        if prev.is_finite() {
            let ratio = (cur / prev).max(prev / cur);
            assert!(
                ratio < 3.2,
                "level jump {:.1} dB between y={} and y={}",
                db(ratio),
                y + 0.25,
                y
            );
        }
        prev = cur;
        y -= 0.25;
    }
}

/// Walking PAST the door outside: the opening must still dominate over a
/// position beside the wall (aperture contrast survives the portal
/// delete), and the profile must stay step-free (bend taps carry the
/// off-axis shadow).
#[test]
fn doorway_contrast_survives_portal_delete() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const VOICE: usize = 1;
    let mut profile = Vec::new();
    let mut x = 1.0f32;
    let mut prev = f32::NAN;
    while x <= 13.0 + 1e-6 {
        for _ in 0..8 {
            let _ = w.tick_at(x, 25.2, 0.0);
        }
        let (blocks, _) = w.tick_at(x, 25.2, 0.0);
        let cur = total_mid(&blocks[VOICE]);
        assert!(cur > 1e-5, "voice fully dead at ({x}, 25.2)");
        if prev.is_finite() {
            let ratio = (cur / prev).max(prev / cur);
            assert!(
                ratio < 3.2,
                "shadow step {:.1} dB between x={} and x={}",
                db(ratio),
                x - 0.5,
                x
            );
        }
        prev = cur;
        profile.push((x, cur));
        x += 0.5;
    }
    eprintln!(
        "traced walk-past profile @y=25.2: {}",
        profile
            .iter()
            .map(|(x, v)| format!("x{x:.1}:{:.1}dB", db(*v)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // "at the door" is the aperture's lit cone as seen from the voice
    // (the source sits deep in the hall at an angle, so the cone lands
    // west of the door's center line)
    let at_door = profile
        .iter()
        .filter(|(x, _)| (*x - 6.2).abs() <= 1.0)
        .map(|(_, v)| *v)
        .fold(0.0f32, f32::max);
    let beside = profile
        .iter()
        .filter(|(x, _)| (*x - 3.0).abs() <= 0.8)
        .map(|(_, v)| *v)
        .fold(0.0f32, f32::max);
    let deep = profile
        .iter()
        .filter(|(x, _)| *x <= 2.0)
        .map(|(_, v)| *v)
        .fold(0.0f32, f32::max);
    eprintln!(
        "traced walk-past: at door {:.1} dB, beside wall {:.1} dB (Δ{:.1}), deep shadow {:.1} dB (Δ{:.1})",
        db(at_door),
        db(beside),
        db(at_door) - db(beside),
        db(deep),
        db(at_door) - db(deep)
    );
    // The traced field legitimately renders paths the portal model
    // discarded — the voice bouncing off the hall's east wall and out
    // the door raises the off-axis shoulder. The opening still
    // dominates: ≥ 5 dB over the 4 m shoulder, ≥ 10 dB over the deep
    // shadow at the building's end.
    assert!(
        at_door > 1.78 * beside, // ≥ 5 dB
        "the opening must dominate the shoulder: at door {at_door} vs beside wall {beside}"
    );
    assert!(
        at_door > 3.16 * deep, // ≥ 10 dB
        "the opening must dominate the deep shadow: at door {at_door} vs {deep}"
    );
}

/// Closing the living-room door in traced mode: the leaf is an extras
/// box over the hole — markedly quieter, bass-favoring, reversible. No
/// portal transmission code involved.
#[test]
fn closing_a_door_muffles_traced_chains() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    let read = |w: &mut WorldSim| -> [f32; NBANDS] {
        for _ in 0..6 {
            let _ = w.tick_at(4.0, 9.0, 0.0);
        }
        let (blocks, _) = w.tick_at(4.0, 9.0, 0.0);
        let mut sum = [0.0f32; NBANDS];
        for t in &blocks[0].taps {
            for b in 0..NBANDS {
                sum[b] += t.gains[b];
            }
        }
        sum
    };
    let open = read(&mut w);
    w.set_door(0, 0.0); // Living ↔ Corridor
    let closed = read(&mut w);
    assert!(
        closed[1] < 0.45 * open[1],
        "closed leaf should muffle: {} vs {}",
        closed[1],
        open[1]
    );
    assert!(
        closed[0] > 2.0 * closed[2],
        "leaf transmission must favor lows: {closed:?}"
    );
    w.set_door(0, 1.0);
    let reopened = read(&mut w);
    assert!(
        reopened[1] > 0.75 * open[1],
        "reopening should restore: {} vs {}",
        reopened[1],
        open[1]
    );
}
