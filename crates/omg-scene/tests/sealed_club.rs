//! Field report: sharp pockets outside the sealed club (and cathedral),
//! traced mode. Three ghost classes were killed, all "flat spectrum
//! paying nothing": (1) near-straight corner "bends" (deflection weight
//! in paths.rs), (2) apexes on coincident wall planes grazing legs at
//! t=1 (legs price extended past the apex), (3) solve records bouncing
//! off a wall PLANE beyond the wall's actual extent (bounce must land
//! on real mesh at its own distance). Gate: with both vestibule doors
//! shut, everywhere on the club's perimeter the spectrum is mass law —
//! bass rules, treble dies.

use omg_scene::quality;
use omg_scene::world::WorldSim;

#[test]
fn sealed_club_is_boomy_from_every_angle() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const CLUB: usize = 2;
    w.set_door(3, 0.0);
    w.set_door(4, 0.0);
    let mut spots: Vec<(f32, f32)> = Vec::new();
    for i in 0..14 {
        spots.push((19.2, 25.0 + i as f32));
        spots.push((33.2, 25.0 + i as f32));
    }
    for i in 0..10 {
        spots.push((23.0 + i as f32, 39.2));
        spots.push((23.0 + i as f32, 24.8));
    }
    for (x, y) in spots {
        for _ in 0..12 {
            let _ = w.tick_at(x, y, 0.0);
        }
        let (blocks, _) = w.tick_at(x, y, 0.0);
        let mut s = [0.0f32; 3];
        for t in &blocks[CLUB].taps {
            for b in 0..3 {
                s[b] += t.gains[b];
            }
        }
        if s[0] < 1e-4 {
            continue; // effectively silent here — nothing to pin
        }
        assert!(
            s[2] < s[0] * 0.178, // high ≥ 15 dB under bass
            "sharp pocket at ({x},{y}): {s:?}"
        );
        assert!(
            s[1] < s[0] * 0.6, // mid ≥ ~4.5 dB under bass
            "mid-heavy pocket at ({x},{y}): {s:?}"
        );
    }
}
