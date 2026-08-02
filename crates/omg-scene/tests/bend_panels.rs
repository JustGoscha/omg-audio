//! Field reports on traced-mode diffraction artifacts. (1) "When two
//! surfaces touch": adjacent rooms author coincident wall planes, so
//! AutoPaths extracts duplicate edges and the bend stage emitted the
//! SAME physical path twice (+50% energy, flappy keys) - candidates now
//! dedupe geometrically. (2) "Around windows / closed doors": bend
//! paths were priced over the mesh, whose apertures are holes - they
//! now pay the panel overlays (panes, leaves, furniture) their
//! segments cross. Gate: no duplicate-delay bend pairs, and the walk
//! past the club's sound-lock stays step-free.

use omg_scene::quality;
use omg_scene::world::WorldSim;

fn taps_mid(pb: &omg_core::params::ParamBlock) -> f32 {
    pb.taps.iter().map(|t| t.gains[1]).sum::<f32>()
}

#[test]
fn no_duplicate_bends_and_smooth_walk_past_the_lock() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const CLUB: usize = 2;

    // beside the sound-lock, where the over-roof bend dominates: the
    // coincident-plane duplicate used to sit at the identical delay
    for _ in 0..40 {
        let _ = w.tick_at(19.2, 32.4, 0.0);
    }
    let (blocks, _) = w.tick_at(19.2, 32.4, 0.0);
    let taps = &blocks[CLUB].taps;
    for i in 0..taps.len() {
        for j in i + 1..taps.len() {
            let (a, b) = (&taps[i], &taps[j]);
            if a.gains[1] < 1e-3 || b.gains[1] < 1e-3 {
                continue;
            }
            let dot = a.dir[0] * b.dir[0] + a.dir[1] * b.dir[1] + a.dir[2] * b.dir[2];
            assert!(
                (a.delay_s - b.delay_s).abs() > 1e-4 || dot < 0.995,
                "duplicate path pair: keys {} / {} at {:.2} ms, gains {} / {}",
                a.key,
                b.key,
                a.delay_s * 1000.0,
                a.gains[1],
                b.gains[1]
            );
        }
    }

    // the reported cliff: sliding past the outer door 0.6 m out (the
    // real player path — never through a wall) must be step-free
    let mut prev = f32::NAN;
    let mut y = 27.0f32;
    while y <= 36.0 {
        for _ in 0..10 {
            let _ = w.tick_at(19.4, y, 0.0);
        }
        let (blocks, _) = w.tick_at(19.4, y, 0.0);
        let cur = taps_mid(&blocks[CLUB]).max(1e-6);
        if prev.is_finite() {
            let ratio = (cur / prev).max(prev / cur);
            assert!(
                ratio < 4.0, // 12 dB per 0.5 m step
                "club steps {:.1} dB between y={} and y={}",
                20.0 * ratio.log10(),
                y - 0.5,
                y
            );
        }
        prev = cur;
        y += 0.5;
    }
}
