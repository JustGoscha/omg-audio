//! Field report: "traced sometimes has a constant car motor going."
//! Root cause: the world-late echogram is an absolute measurement
//! refreshed round-robin — a car that passed CLOSE left its loud wet
//! level playing while it drove away, until its turn to re-trace came
//! around and the EMA slowly unwound. The estimate now decays with the
//! source–listener separation between traces. This test drives a car
//! past the listener and away, and pins that the wet bed dies with it.

use omg_scene::quality;
use omg_scene::world::WorldSim;

#[test]
fn departing_car_takes_its_wet_bed_along() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const CAR0: usize = 11; // dyn slot 3 (8 placed + 3 balls)
    let wet = |b: &omg_core::params::ParamBlock| -> f32 {
        b.reverb.level[1] + b.remote.as_ref().map_or(0.0, |r| r.send[1])
    };

    // car idles right next to the listener long enough to be traced
    let (lx, ly) = (16.0, 12.0);
    w.set_dynamic(3, 17.5, 12.0, 0.7, 1.0);
    for _ in 0..80 {
        let _ = w.tick_at(lx, ly, 0.0);
    }
    let (blocks, _) = w.tick_at(lx, ly, 0.0);
    let near = wet(&blocks[CAR0]);

    // it drives off fast; within a couple of seconds of sim time the
    // wet bed must be gone even if its trace turn hasn't come round
    let mut y = 12.0f32;
    let mut last = near;
    for _ in 0..40 {
        y += 12.0 * 0.05 * 20.0 / 20.0; // ~12 m/s
        w.set_dynamic(3, 17.5, y.min(900.0), 0.7, 1.0);
        let (blocks, _) = w.tick_at(lx, ly, 0.0);
        last = wet(&blocks[CAR0]);
    }
    eprintln!("car wet: near {near:.5} · after 2 s of driving away {last:.5}");

    // the OTHER constant-motor bug: an INACTIVE slot ships a default
    // block, whose reverb level must be SILENCE — 0.05 here meant every
    // idle car slot fed its motor loop into the reverb network forever
    w.set_dynamic(3, 17.5, 700.0, 0.7, 0.0);
    let (blocks, _) = w.tick_at(lx, ly, 0.0);
    let idle = &blocks[CAR0];
    assert!(idle.taps.is_empty(), "inactive slot must carry no taps");
    assert!(
        idle.reverb.level.iter().all(|&l| l == 0.0),
        "inactive slot must carry NO reverb level: {:?}",
        idle.reverb.level
    );
    assert!(idle.remote.is_none(), "inactive slot must carry no remote wet");

    // the mixer kill switch takes the same silent path for ANY source,
    // and unmuting recomputes immediately (no stale LOD replay)
    w.set_muted(0, true);
    let (blocks, _) = w.tick_at(lx, ly, 0.0);
    assert!(blocks[0].taps.is_empty(), "muted source must be skipped");
    w.set_muted(0, false);
    let (blocks, _) = w.tick_at(lx, ly, 0.0);
    assert!(!blocks[0].taps.is_empty(), "unmuted source must come back at once");
    if near > 1e-4 {
        assert!(
            last < 0.25 * near,
            "the motor bed must leave with the car: near {near} vs departed {last}"
        );
    } else {
        eprintln!("(near level below floor — nothing to pin)");
    }
}
