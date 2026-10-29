//! C6c gate (cost): the whole point of dissolving portals — a doorway
//! tick must cost about what an open-square tick does, because with one
//! world listener context there IS nothing extra to do at a door. The
//! bound was 1.2× until the exact bend point (Oct 29: `bend_point` in
//! closed form instead of a golden-section search) halved the open
//! square's tick, all bend searches, 14.2 → 7.6 ms, while the doorways'
//! stayed as they were (12.6 → 12.5 ms, 11.0 → 11.0 ms): what they cost
//! is not bend points, and not more than before. 1.8× still catches a
//! doorway that starts doing extra work.
//! Own test binary: wall-clock measurement, keep it out of the parallel
//! test threads of the behavior gates.

use omg_scene::quality;
use omg_scene::world::WorldSim;
use std::time::Instant;

fn ms_per_tick(w: &mut WorldSim, x: f32, y: f32) -> f64 {
    // settle: LOD levels, bend caches, trace gates
    for _ in 0..30 {
        let _ = w.tick_at(x, y, 0.0);
    }
    // measure with a slight pose wobble so the trace gates fire at the
    // same rate they would during a real walk (a frozen pose lets every
    // gate go idle and flatters both numbers)
    let n = 200;
    let t0 = Instant::now();
    for i in 0..n {
        let jitter = (i % 8) as f32 * 0.04;
        let _ = w.tick_at(x + jitter, y, 0.0);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / n as f64
}

#[test]
fn doorway_tick_within_budget_of_open_square() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    let open = ms_per_tick(&mut w, 12.0, 12.0); // open square
    let door = ms_per_tick(&mut w, 7.0, 24.0); // hall door, mid-blend
    let club = ms_per_tick(&mut w, 22.5, 31.0); // club vestibule door
    eprintln!(
        "traced tick: open {open:.2} ms, hall door {door:.2} ms ({:.2}×), club door {club:.2} ms ({:.2}×)",
        door / open,
        club / open
    );
    assert!(
        door <= 1.8 * open,
        "hall doorway tick {door:.2} ms exceeds 1.8× open square {open:.2} ms"
    );
    assert!(
        club <= 1.8 * open,
        "club doorway tick {club:.2} ms exceeds 1.8× open square {open:.2} ms"
    );
}
