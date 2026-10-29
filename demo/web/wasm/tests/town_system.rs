//! The live engine (omg-audio's simulation and mixer, as the hosts run
//! them) in the demo town, measured with impulses and exact paths — the
//! regressions these guard were heard in the demo while the legacy
//! engine's tests stayed green.
use omg_audio::{clip, Compute, Config, Hrir, Mixer, Play, Quality, Simulation, SourceId};
use omg_web::town::Town;

const SR: f32 = 48_000.0;
const C: f32 = omg_audio::SPEED_OF_SOUND;
const BELLS: usize = 7;
const BALL: usize = 8;
const CAR: usize = 11;

fn sim() -> Simulation {
    Simulation::new(Config { quality: Quality::High, compute: Compute::Cpu, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() })
}

fn only(town: &mut Town, sim: &mut Simulation, keep: usize) {
    for i in 0..town.defs.len() {
        town.set_muted(sim, i, i != keep);
    }
}

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-30).log10()
}

/// The bell 290 m north of the square, the cathedral's nave between them
/// from most of it: what reaches the square is the bell bent around the
/// nave (two edges: its corners, or its gables), exact and steady — not a
/// few random rays (it was −116 dB of bends and ±10 dB of noise).
#[test]
fn the_bell_reaches_the_square_around_the_cathedral() {
    for at in [[10.0f32, 35.0, 1.6], [12.0, 42.0, 1.6], [8.0, 30.0, 1.6], [15.0, 32.0, 1.6]] {
        let mut sim = sim();
        let mut town = Town::new(&mut sim);
        only(&mut town, &mut sim, BELLS);
        let id = town.sources[BELLS][0];
        let bell = [town.defs[BELLS].pos.0, town.defs[BELLS].pos.1, town.defs[BELLS].z];
        let d = ((bell[0] - at[0]).powi(2) + (bell[1] - at[1]).powi(2) + (bell[2] - at[2]).powi(2)).sqrt();
        sim.set_listener(at, 0.0, 0.0, 0.0);
        let mut exact = Vec::new();
        for _ in 0..12 {
            let f = sim.tick();
            let pb = f.block(id).expect("the bell is traced");
            exact.push(pb.taps.iter().filter(|t| t.key < 9_000_000).map(|t| t.gains[1] * t.gains[1]).sum::<f32>());
        }
        // unit source: free field 1/d²; the gain the town gives the bell
        let g = town.defs[BELLS].gain;
        let free = db(g * g / (d * d));
        let got = db(*exact.last().unwrap());
        assert!(got > free - 35.0, "at {at:?}: the bell's exact paths {got:.1} dB, free field {free:.1} dB");
        let span = exact[4..].iter().map(|&e| db(e)).fold(f32::MIN, f32::max) - exact[4..].iter().map(|&e| db(e)).fold(f32::MAX, f32::min);
        assert!(span < 0.5, "at {at:?}: the exact paths are steady ({span:.2} dB tick to tick)");
    }
}

/// A car 300 m off at 100 km/h: its direct path reaches it where it was
/// when it emitted what arrives now (c·τ = |x(t − τ) − listener|), not
/// where it is (that was up to 70 ms off).
#[test]
fn a_moving_source_is_heard_from_where_it_emitted() {
    let mut sim = sim();
    let mut town = Town::new(&mut sim);
    only(&mut town, &mut sim, CAR);
    let id = town.sources[CAR][0];
    let lis = [-100.0f32, 100.0, 1.6];
    let v = 27.78f32;
    let road = |t: f32| [-400.0f32, -150.0 + v * t, 0.7];
    let slot = CAR - omg_scene::walkthrough::DYN_BASE;
    let mut worst = 0.0f32;
    for k in 0..120 {
        let t = k as f32 * 0.05;
        sim.set_time(t as f64);
        town.set_dynamic(&mut sim, slot, Some((road(t), 1.0)));
        sim.set_listener_at(lis, 0.0, 0.0, 0.0, t as f64);
        let f = sim.tick();
        if t < 3.0 {
            continue; // the track reaches back further than the first path
        }
        let tap = f.block(id).and_then(|pb| pb.taps.iter().find(|t| t.key == 1).copied()).expect("a direct path");
        // solve the light cone on the true trajectory
        let mut tau = 1.0f32;
        for _ in 0..20 {
            let p = road(t - tau);
            tau = ((p[0] - lis[0]).powi(2) + (p[1] - lis[1]).powi(2) + (p[2] - lis[2]).powi(2)).sqrt() / C;
        }
        worst = worst.max((tap.delay_s - tau).abs());
    }
    assert!(worst < 2e-4, "the direct path's delay is off its emission point's by {:.2} ms", worst * 1e3);
}

/// Mixer and simulation as the hosts run them, offline: `secs` of output
/// with a source moving by `at(t)` (None: switched off), clicks at the
/// given times. Returns the output (both ears) and the clicks' positions.
fn render(slot_i: usize, at: impl Fn(f32) -> Option<[f32; 3]>, clicks: &[f32], secs: f32, lis: [f32; 3], yaw: f32) -> (Vec<f32>, Vec<f32>) {
    let mut sim = sim();
    let mut town = Town::new(&mut sim);
    only(&mut town, &mut sim, slot_i);
    let mut mix = Mixer::with_capacity(SR, Hrir::builtin(), 32);
    mix.set_ear(false);
    let ids: Vec<SourceId> = town.sources[slot_i].clone();
    let slot = slot_i - omg_scene::walkthrough::DYN_BASE;
    let click = clip(&[1.0]);
    let block = 400usize;
    let (mut l, mut r) = (Vec::new(), Vec::new());
    let n = (secs * SR) as usize / block;
    let mut next = 0;
    for b in 0..n {
        let t = (b * block) as f32 / SR;
        sim.set_time(t as f64);
        town.set_dynamic(&mut sim, slot, at(t).map(|p| (p, 1.0)));
        sim.set_listener_at(lis, yaw, 0.0, 0.0, t as f64);
        if b % 6 == 0 {
            mix.apply(&sim.tick());
        } else if let Some(f) = sim.tick_exact() {
            mix.apply(&f);
        }
        while next < clicks.len() && clicks[next] <= t {
            for &id in &ids {
                mix.play(id, &click, Play::ONCE.gain(4.0));
            }
            next += 1;
        }
        let (mut a, mut c) = (vec![0.0f32; block], vec![0.0f32; block]);
        mix.render(&mut a, &mut c);
        l.extend_from_slice(&a);
        r.extend_from_slice(&c);
    }
    (l, r)
}

fn energy(x: &[f32], t0: f32, t1: f32) -> f32 {
    x[(t0 * SR) as usize..(t1 * SR) as usize].iter().map(|v| v * v).sum()
}

/// A click from 100 m west is on its way when the source jumps 4 m north
/// of the listener: it still arrives on time, from the west, at its
/// distance's level (the place a jumped source left goes on as a ghost);
/// the source's next click comes from the north.
#[test]
fn a_jumped_source_still_delivers_what_it_emitted_before() {
    let lis = [-100.0f32, 100.0, 1.6];
    let (a, b) = ([-200.0f32, 100.0, 1.6], [-100.0f32, 104.0, 1.6]);
    let (l, r) = render(BALL, |t| Some(if t < 3.1 { a } else { b }), &[3.0, 3.2], 4.0, lis, std::f32::consts::FRAC_PI_2);
    let (t1, t2) = (3.0 + 100.0 / C, 3.2 + 4.0 / C);
    let (l1, r1) = (energy(&l, t1 - 0.004, t1 + 0.006), energy(&r, t1 - 0.004, t1 + 0.006));
    let (l2, r2) = (energy(&l, t2 - 0.004, t2 + 0.006), energy(&r, t2 - 0.004, t2 + 0.006));
    assert!(db(l1) - db(r1) > 6.0, "the click in flight arrives from the west (left): L−R {:.1} dB", db(l1) - db(r1));
    let fall = db(l1 + r1) - db(l2 + r2);
    assert!((fall + 28.0).abs() < 5.0, "100 m against 4 m: {fall:.1} dB (spreading alone −28 dB)");
}

/// A click from 100 m, the source switched off 50 ms later while the
/// click is on its way: it arrives whole (nothing is cut because the
/// source stopped).
#[test]
fn a_source_switched_off_still_delivers_what_it_emitted() {
    let lis = [-100.0f32, 100.0, 1.6];
    let a = [-200.0f32, 100.0, 1.6];
    let (on_l, on_r) = render(BALL, |_| Some(a), &[3.0], 4.0, lis, 0.0);
    let (off_l, off_r) = render(BALL, |t| (t < 3.05).then_some(a), &[3.0], 4.0, lis, 0.0);
    let t1 = 3.0 + 100.0 / C;
    let on = energy(&on_l, t1 - 0.004, t1 + 0.006) + energy(&on_r, t1 - 0.004, t1 + 0.006);
    let off = energy(&off_l, t1 - 0.004, t1 + 0.006) + energy(&off_r, t1 - 0.004, t1 + 0.006);
    assert!((db(off) - db(on)).abs() < 3.0, "switched off in flight: {:.1} dB against {:.1} dB left on", db(off), db(on));
}

/// Standing still with nothing changing, the club heard from the street
/// north of it (through its walls and windows) is the same truth sampled
/// tick after tick: the estimate settles and stays (it wandered ±2.5 dB
/// for as long as one stood there, rare strong paths restarting or
/// pulling a short average).
#[test]
fn standing_still_the_estimate_settles_and_stays() {
    let mut sim = sim();
    let mut town = Town::new(&mut sim);
    let club = 2;
    only(&mut town, &mut sim, club);
    let ids = town.sources[club].clone();
    sim.set_listener([27.0, 48.0, 1.6], 0.0, 0.0, 0.0);
    let total = |sim: &Simulation| -> f32 { ids.iter().filter_map(|id| sim.dense_response(*id)).map(|d| d.bins.iter().map(|b| b[1]).sum::<f32>()).sum() };
    for _ in 0..300 {
        sim.tick();
    }
    let mut seen = Vec::new();
    for _ in 0..200 {
        sim.tick();
        seen.push(db(total(&sim)));
    }
    let span = seen.iter().cloned().fold(f32::MIN, f32::max) - seen.iter().cloned().fold(f32::MAX, f32::min);
    assert!(span < 1.0, "the settled estimate moves by {span:.2} dB over 10 s of standing still");
}

/// The live engine's port of the legacy `bells_open_vs_indoor`: the bell is
/// loudest in the open field with the tower in view, and every room —
/// the bunker under the field included — hears it well under that.
#[test]
fn the_bell_is_loudest_in_the_open() {
    let level = |at: [f32; 3]| -> f32 {
        let mut sim = sim();
        let mut town = Town::new(&mut sim);
        only(&mut town, &mut sim, BELLS);
        let id = town.sources[BELLS][0];
        sim.set_listener(at, 0.0, 0.0, 0.0);
        let mut e = 0.0;
        for _ in 0..30 {
            let f = sim.tick();
            let pb = f.block(id).expect("traced");
            let taps: f32 = pb.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum();
            e = taps + pb.reverb.level[1] * pb.reverb.level[1];
        }
        db(e)
    };
    let open = level([20.0, 60.0, 1.6]);
    let g = 30.0f32; // the town's bell gain; free field at ~261 m
    assert!(open > db(g * g / (261.0 * 261.0)) - 6.0, "the open field hears the tower: {open:.1} dB");
    for (name, at) in [
        ("living room", [4.0, 3.0, 1.6]),
        ("great hall", [7.0, 19.0, 1.6]),
        ("old house", [27.5, 19.5, 1.6]),
        ("cathedral", [8.0, 63.0, 1.6]),
        ("bunker", [37.0, 10.0, -1.4]),
    ] {
        let inside = level(at);
        assert!(inside < open - 10.0, "{name}: {inside:.1} dB must be well under the open field's {open:.1} dB");
    }
}
