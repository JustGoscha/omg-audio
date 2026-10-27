//! Demo-world check of the shaped ambience bed (`ambient.rs`, isolated
//! in `fx_ambient_shapes.rs`): the crickets live in the outdoor grass
//! (`walkthrough::ambient_emitters`). On the ground they sound like the
//! old dome bed did — level and indoor routing included — and flying
//! they come from below and fade; far away they point at the town and
//! vanish. Module switch 2 off restores the classic dome exactly; the
//! rain's routes never change.

use omg_dsp::env::{EnvRoute, Environment};
use omg_scene::quality;
use omg_scene::world::WorldSim;
use std::sync::Mutex;

/// The module switch is process-global: tests touching it serialize.
static SWITCH: Mutex<()> = Mutex::new(());

fn env_at(x: f32, y: f32, z: f32) -> Environment {
    let mut w = WorldSim::new();
    let mut env = None;
    for _ in 0..6 {
        env = Some(w.tick_at_z(x, y, z, 0.0).1.env);
    }
    env.unwrap()
}

fn energy(r: &[EnvRoute], b: usize) -> f32 {
    r.iter().map(|r| r.gains[b] * r.gains[b]).sum()
}

/// Energy-weighted mean arrival direction (mid band), normalized.
fn mean_dir(r: &[EnvRoute]) -> [f32; 3] {
    let mut v = [0.0f32; 3];
    for r in r {
        let e = r.gains[1] * r.gains[1];
        for i in 0..3 {
            v[i] += r.dir[i] * e;
        }
    }
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-30);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-30).log10()
}

#[test]
fn on_the_square_the_crickets_sound_like_the_old_bed() {
    let _g = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let env = env_at(18.0, 8.0, 1.6);
    let (amb, dome) = (energy(&env.amb_routes, 1), energy(&env.routes, 1));
    println!("square: crickets {:+.1} dB re the dome bed", db(amb / dome));
    assert!((db(amb / dome)).abs() < 2.0, "same loudness on the ground: {amb} vs {dome}");
    assert!(env.amb_routes.len() >= 6, "they surround: {} inlets", env.amb_routes.len());
    assert!(mean_dir(&env.amb_routes)[2] < 0.0, "from the grass, not the sky");
}

#[test]
fn flying_up_they_come_from_below_and_fade() {
    let _g = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let ground = energy(&env_at(18.0, 8.0, 1.6).amb_routes, 1);
    let mut prev = ground;
    for h in [3.0f32, 30.0, 300.0] {
        let env = env_at(18.0, 8.0, h);
        let e = energy(&env.amb_routes, 1);
        let d = mean_dir(&env.amb_routes);
        println!("{h:>5} m over the square: {:+.1} dB re ear height, mean dir z {:+.2}", db(e / ground), d[2]);
        assert!(e < prev, "{h} m: quieter than below");
        if h >= 30.0 {
            assert!(d[2] < -0.95, "{h} m: from below: {d:?}");
        }
        prev = e;
        // the dome bed would have stayed put (it is the sky)
        assert!(energy(&env.routes, 1) > 0.5 * ground, "rain/dome routes untouched");
    }
    assert!(db(prev / ground) < -12.0, "300 m up they are far down: {} dB", db(prev / ground));
}

#[test]
fn ten_km_away_they_point_at_the_town_and_vanish() {
    let _g = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let ground = energy(&env_at(18.0, 8.0, 1.6).amb_routes, 1);
    let env = env_at(10_018.0, 8.0, 300.0);
    let e = energy(&env.amb_routes, 1);
    let d = mean_dir(&env.amb_routes);
    println!("10 km east, 300 m up: {:+.1} dB, dir {d:?}", db(e / ground));
    assert!(d[0] < -0.95, "toward the town (west): {d:?}");
    assert!(db(e / ground) < -50.0, "all but gone: {} dB", db(e / ground));
    assert!(env.amb_seep.iter().all(|&s| s < 0.01), "no indoor seep of crickets out here: {:?}", env.amb_seep);
}

#[test]
fn indoors_they_arrive_through_the_doorway_like_the_old_bed() {
    let _g = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    // Great Hall, door to the north (+y)
    let env = env_at(7.0, 18.0, 1.6);
    let (amb, dome) = (energy(&env.amb_routes, 1), energy(&env.routes, 1));
    println!("hall: crickets {:+.1} dB re the dome bed, seep {:?} vs {:?}", db(amb / dome), env.amb_seep, env.seep);
    assert!(db(amb / dome).abs() < 4.0, "indoors as loud as the old bed: {amb} vs {dome}");
    let best = env.amb_routes.iter().max_by(|a, b| a.gains[1].total_cmp(&b.gains[1])).unwrap();
    assert!(best.dir[1] > 0.6, "strongest inlet faces the door: {:?}", best.dir);
    for b in 0..3 {
        assert!(env.amb_seep[b] > 0.3 * env.seep[b] && env.amb_seep[b] <= env.seep[b] * 1.01, "shell seep kept");
    }
}

#[test]
fn switch_off_restores_the_classic_dome_exactly() {
    let _g = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    quality::set_module(2, false);
    let a = env_at(7.0, 18.0, 1.6);
    let b = env_at(18.0, 8.0, 300.0);
    quality::set_module(2, true);
    for env in [&a, &b] {
        assert_eq!(env.amb_routes.len(), env.routes.len());
        for (x, y) in env.amb_routes.iter().zip(&env.routes) {
            assert_eq!(x.id, y.id);
            for k in 0..3 {
                assert!((x.gains[k] - y.gains[k]).abs() < 1e-5 && (x.dir[k] - y.dir[k]).abs() < 1e-4);
            }
        }
        for k in 0..3 {
            assert!((env.amb_seep[k] - env.seep[k]).abs() < 1e-6);
        }
    }
    let on = env_at(7.0, 18.0, 1.6);
    assert_eq!(on.routes, a.routes, "rain routes do not depend on the switch");
}
