//! Owls on rooftops + the distant belfry (task #20). The interesting
//! physics is free: the belfry is ~250 m out, so its strikes arrive
//! bass-tilted by air absorption and spreading; the cathedral owl
//! perches at z = 17, so its call arrives FROM ABOVE — elevation is
//! native to the traced engine and the binaural renderer.

use omg_scene::quality;
use omg_scene::world::WorldSim;

#[test]
fn bells_carry_bass_and_owls_arrive_from_above() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const OWL_CATHEDRAL: usize = 5;
    const BELLS: usize = 7;

    // town square listener
    for _ in 0..30 {
        let _ = w.tick_at(12.0, 12.0, 0.0);
    }
    let (blocks, _) = w.tick_at(12.0, 12.0, 0.0);
    let bells = &blocks[BELLS];
    let mut s = [0.0f32; 3];
    for t in &bells.taps {
        for b in 0..3 {
            s[b] += t.gains[b];
        }
    }
    eprintln!("bells at the square: [{:.4} {:.4} {:.4}]", s[0], s[1], s[2]);
    assert!(s[1] > 1e-3, "the belfry must be audible in town: {s:?}");
    assert!(
        s[2] < 0.75 * s[1] && s[1] < s[0],
        "250 m of air must tilt the peal toward bass: {s:?}"
    );

    // stand before the cathedral facade: the owl call comes from UP
    for _ in 0..30 {
        let _ = w.tick_at(4.0, 48.0, 0.0);
    }
    let (blocks, _) = w.tick_at(4.0, 48.0, 0.0);
    let owl = &blocks[OWL_CATHEDRAL];
    let top = owl
        .taps
        .iter()
        .max_by(|a, b| a.gains[1].total_cmp(&b.gains[1]))
        .expect("the rooftop owl must reach the forecourt");
    eprintln!(
        "owl strongest tap: gain {:.4} dir [{:.2} {:.2} {:.2}]",
        top.gains[1], top.dir[0], top.dir[1], top.dir[2]
    );
    assert!(
        top.dir[2] > 0.3,
        "a rooftop owl arrives from above: dir {:?}",
        top.dir
    );
}
