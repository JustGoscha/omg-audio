//! Field report: "the club bass is almost not audible in adjacent
//! buildings" (traced mode) — and ISM estimated the outside boom
//! better. Root cause: the real-world boom IS the club's reverberant
//! field leaking through walls, and the stochastic tracer's rays only
//! reflected — the dominant mechanism had no representation. Rays now
//! BRANCH through surfaces with mass-law energy (one ray, one branch,
//! per-band re-weighted, unbiased), so the measured echogram carries
//! the through-wall wet. Gate: inside the Old House (across a 3 m
//! street from the club, both vestibule doors SHUT) the club's wet bed
//! is present and bass-dominant.

use omg_scene::quality;
use omg_scene::world::WorldSim;

#[test]
fn club_wet_leaks_into_the_house_bass_first() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const CLUB: usize = 2;
    w.set_door(3, 0.0);
    w.set_door(4, 0.0);
    // Old House interior, ground floor
    for _ in 0..160 {
        let _ = w.tick_at(27.0, 20.0, 0.0);
    }
    let (blocks, _) = w.tick_at(27.0, 20.0, 0.0);
    let pb = &blocks[CLUB];
    let wet: [f32; 3] = core::array::from_fn(|b| {
        pb.reverb.level[b] + pb.remote.as_ref().map_or(0.0, |r| r.send[b])
    });
    eprintln!(
        "club wet in the Old House: [{:.4} {:.4} {:.4}]",
        wet[0], wet[1], wet[2]
    );
    assert!(
        wet[0] > 5e-3,
        "the boom must come through the walls: {wet:?}"
    );
    assert!(
        wet[0] > 2.0 * wet[1] && wet[1] > wet[2],
        "and it must be BASS (mass law): {wet:?}"
    );
}
