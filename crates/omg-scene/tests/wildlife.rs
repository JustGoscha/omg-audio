//! Owls on rooftops + the distant belfry (task #20). The interesting
//! physics is free: the belfry is ~250 m out, so its strikes arrive
//! bass-tilted by air absorption and spreading; the cathedral owl
//! perches on the nave roof at z = 24.25, so its call arrives FROM
//! ABOVE — elevation is native to the traced engine and the binaural
//! renderer.

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
    // (12, 12) stands 2 m south of the Great Hall's 7.4 m wall with the
    // Cathedral beyond it: deep shadow, the peal arrives over two roof
    // lines (≈ −42 dB of knife-edge at 1 kHz on top of 300 m). Audible,
    // not loud — the old 1e-3 floor was met only by a silhouette path
    // that dropped THROUGH parapets for free.
    assert!(s[1] > 2e-4, "the belfry must be audible in town: {s:?}");
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
    // Where a sound is heard FROM is set by its first significant arrival
    // (precedence effect), not by its loudest echo. The owl sits a meter
    // behind the cathedral's portal gable: from the forecourt its call bends
    // over the gable edge — first, from above — while a later bounce
    // off the hall facade and the ground can legitimately be louder.
    let loudest = owl
        .taps
        .iter()
        .map(|t| t.gains[1])
        .fold(0.0f32, f32::max);
    assert!(loudest > 0.0, "the rooftop owl must reach the forecourt");
    let top = owl
        .taps
        .iter()
        .filter(|t| t.gains[1] > 0.1 * loudest)
        .min_by(|a, b| a.delay_s.total_cmp(&b.delay_s))
        .expect("a significant owl arrival");
    eprintln!(
        "owl first arrival: gain {:.4} delay {:.1} ms dir [{:.2} {:.2} {:.2}]",
        top.gains[1], top.delay_s * 1e3, top.dir[0], top.dir[1], top.dir[2]
    );
    assert!(
        top.dir[2] > 0.3,
        "a rooftop owl arrives from above: dir {:?}",
        top.dir
    );
}
