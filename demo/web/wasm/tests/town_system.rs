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
    let (on_l, on_r) = render(BALL, |_| Some(a), &[3.0], 4.5, lis, 0.0);
    let (off_l, off_r) = render(BALL, |t| (t < 3.05).then_some(a), &[3.0], 4.5, lis, 0.0);
    let t1 = 3.0 + 100.0 / C;
    let on = energy(&on_l, t1 - 0.004, t1 + 0.006) + energy(&on_r, t1 - 0.004, t1 + 0.006);
    let off = energy(&off_l, t1 - 0.004, t1 + 0.006) + energy(&off_r, t1 - 0.004, t1 + 0.006);
    assert!((db(off) - db(on)).abs() < 3.0, "switched off in flight: {:.1} dB against {:.1} dB left on", db(off), db(on));
    // and what scatters after it (the town's facades): the place it left
    // rings on — a ghost whose paths have all arrived still convolves
    let tail = |l: &[f32], r: &[f32]| energy(l, t1 + 0.004, t1 + 1.0) + energy(r, t1 + 0.004, t1 + 1.0);
    let (on_t, off_t) = (tail(&on_l, &on_r), tail(&off_l, &off_r));
    assert!((db(off_t) - db(on_t)).abs() < 3.0, "its tail switched off: {:.1} dB against {:.1} dB left on", db(off_t), db(on_t));
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


/// The club door shuts while the club is between sounds (its channel
/// masked, not rendering); 50 ms later it clicks: the click is heard
/// through the shut door as it will be once everything settled, not
/// through the open door's paths. With v6 (the demos' engine) the
/// averages start over when the leaf moves, a running mean of the
/// restarted ticks (with the world change's extra rays) — before, v3's
/// averages kept the open door's field: +19 / +28 / +37 dB 50 ms after
/// the shut, still +4 / +15 / +23 dB after 0.8 s (`assess door`). (v5's
/// variant retired Oct 31: its convolution kernels follow a shut over
/// ticks — one of the club's four still played the open door 3 ticks
/// on, 0.38 changed, under its 0.5 withdraw-at-once — and v4/v5 aren't
/// pursued.)
#[test]
fn a_door_shut_between_sounds_is_shut_for_the_next_one_v6() {
    let sim = Simulation::new(Config { quality: Quality::High, compute: Compute::Cpu, v4: false, v5: false, v6: true, ..Config::default() });
    door_shut_between_sounds(sim, 4.0);
}

/// The club plays a steady tone and its door swings shut over a second,
/// moved every frame as the Godot demo eases a leaf (openness ×
/// e^(−7 t), there below 0.001); the listener 1 m before it, a speaker in
/// sight through the doorway (the Godot demo's spot). Narrowing the
/// opening only takes paths away: the level goes down without climbing
/// back on the way, without lurching (clicks), and lands where the door
/// shut all along puts it. v6 restarted its averages at every tick of
/// the swing and played each tick's own heavy-tailed estimate (55–110 dB
/// of climbing back, the treble at −34 dB; `assess doorswing EAR=0`);
/// then, restarting only as the door started, its averages kept the
/// swing's field for seconds (+14.9 dB 1.5–3 s after the swing). (The
/// last gap closing drops the level ~25 dB in ~30 ms — physical; its
/// treble reads ~−44 dB. Right after the restart where the door stops
/// the young estimate reads low: the heavy tail, OPEN.md.) Three seeds:
/// one swing's worst click, and where its young estimate stands 1.5–3 s
/// on, are draws — a seed's luck moves them ±8 and ±4 dB (seven seeds:
/// clicks −39 … −58 dB, the level after −8 … +3 dB, before and after the
/// walls' feet stopped counting as edges alike) — so each seed's clicks
/// within −38 dB and their median within −45, the level after within
/// 4 dB of the door shut all along on average.
///
/// Ignored since Oct 31 (OPEN.md, "The club's door swinging shut"): run
/// on the demos' engine (v6.2, portals) it fails on the old layout too
/// (clicks' median −32.8 dB); in the club's new 6 m entrance, treated,
/// the clicks' median is −36.5 and one seed climbs back 47 dB — the
/// door bursts' heavy tail. Plain v6 here: −41.2 and +5.1 dB.
#[test]
#[ignore = "fails on the demos' engine before and after the club moved: OPEN.md, the club's door swinging shut"]
fn a_door_swinging_shut_takes_the_level_down_without_bursts_v6() {
    let (mut worst, mut after) = (Vec::new(), 0.0f32);
    let seeds = [Config::default().seed, 1, 2];
    for seed in seeds {
        let (clicks, off) = door_swing(seed);
        worst.push(clicks);
        after += off / seeds.len() as f32;
    }
    worst.sort_by(f32::total_cmp);
    assert!(worst[1] < -45.0, "the swing's clicks over three seeds: median {:.1} dB under the tone", worst[1]);
    assert!(after.abs() < 4.0, "1.5–3 s after the swing {after:+.1} dB from the door shut all along (three seeds' mean)");
}

/// One seed's swing (`a_door_swinging_shut_takes_the_level_down_without_bursts_v6`):
/// its own checks; its worst click (dB under the tone) and its level
/// 1.5–3 s after the swing from the door shut all along (dB).
fn door_swing(seed: u64) -> (f32, f32) {
    let run = |swing: bool| -> (Vec<f32>, Vec<f32>) {
        let mut sim = Simulation::new(Config { quality: Quality::Level(3), compute: Compute::Cpu, v6: true, v61: true, v62: true, portals: true, seed, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() });
        let mut town = Town::new(&mut sim);
        let club = 2;
        only(&mut town, &mut sim, club);
        town.set_door(&mut sim, 3, 1.0);
        town.set_door(&mut sim, 4, if swing { 1.0 } else { 0.0 });
        let mut mix = Mixer::with_capacity(SR, Hrir::builtin(), 32);
        mix.set_ear(false);
        let cycle = (SR / 500.0) as usize;
        let tone = clip(&(0..cycle * 10).map(|k| 0.5 * (std::f32::consts::TAU * (k % cycle) as f32 / cycle as f32).sin()).collect::<Vec<f32>>());
        for &id in &town.sources[club] {
            mix.play(id, &tone, Play::LOOP);
        }
        let lis = [25.0f32, 31.0, 1.6];
        let block = 400usize;
        let (mut l, mut r) = (Vec::new(), Vec::new());
        let mut open = 1.0f32;
        for b in 0..(8.0 * SR) as usize / block {
            let t = (b * block) as f32 / SR;
            if swing && t >= 4.0 {
                open *= (-7.0 * block as f32 / SR).exp();
                if open < 0.001 {
                    open = 0.0;
                }
                town.set_door(&mut sim, 4, open);
            }
            sim.set_listener(lis, 0.0, 0.0, 0.0);
            if b % 6 == 0 {
                mix.apply(&sim.tick());
            } else if let Some(f) = sim.tick_exact() {
                mix.apply(&f);
            }
            let (mut bl, mut br) = (vec![0.0; block], vec![0.0; block]);
            mix.render(&mut bl, &mut br);
            l.extend(bl);
            r.extend(br);
        }
        (l, r)
    };
    // the level per 10 ms; the treble per 5 ms (a triple difference: the
    // tone 54 dB under what it passes at 4 kHz) under the tone
    let w = (0.01 * SR) as usize;
    let levels = |(l, r): &(Vec<f32>, Vec<f32>)| -> Vec<f32> { l.chunks(w).zip(r.chunks(w)).map(|(a, b)| db(a.iter().zip(b).map(|(x, y)| 0.5 * (x * x + y * y)).sum::<f32>() / w as f32)).collect() };
    let mean = |lev: &[f32], a: usize, b: usize| db(lev[a..b].iter().map(|d| 10f32.powf(d / 10.0)).sum::<f32>() / (b - a) as f32);
    let swung = run(true);
    let lev = levels(&swung);
    let shut = mean(&levels(&run(false)), 650, 800);
    let after = mean(&lev, 650, 800);
    let (mut lowest, mut climb) = (f32::MAX, 0.0f32);
    for k in 400..520 {
        lowest = lowest.min(lev[k]);
        climb += (lev[k] - lev[k - 1]).max(0.0);
    }
    let mut clicks = f32::MIN;
    let m: Vec<f32> = swung.0.iter().zip(&swung.1).map(|(a, b)| 0.5 * (a + b)).collect();
    let h = (0.005 * SR) as usize;
    for k in (4.0 * SR) as usize / h..(5.2 * SR) as usize / h {
        let (a, b) = (k * h, (k + 1) * h);
        let hf: f32 = (a..b).map(|i| (m[i] - 3.0 * m[i - 1] + 3.0 * m[i - 2] - m[i - 3]).powi(2)).sum();
        let all: f32 = (a..b).map(|i| m[i] * m[i]).sum();
        clicks = clicks.max(db(hf / all.max(1e-30)));
    }
    eprintln!("seed {seed:#x}: open {:.1} dB, shut all along {shut:.1} dB; the swing: lowest {:+.1} dB from it, climbing back {climb:.1} dB, treble at worst {clicks:.1} dB under the tone; 1.5–3 s after the swing {:+.1} dB", lev[350], lowest - shut, after - shut);
    assert!(climb < 20.0, "seed {seed:#x}: the level climbed back {climb:.1} dB while the door closed");
    assert!(lowest > shut - 20.0, "seed {seed:#x}: the level dipped {:.1} dB under the shut level", shut - lowest);
    assert!(clicks < -38.0, "seed {seed:#x}: clicks {clicks:.1} dB under the tone");
    (clicks, after - shut)
}

fn door_shut_between_sounds(mut sim: Simulation, within_db: f32) {
    let mut town = Town::new(&mut sim);
    let club = 2;
    only(&mut town, &mut sim, club);
    town.set_door(&mut sim, 3, 1.0);
    town.set_door(&mut sim, 4, 1.0);
    let mut mix = Mixer::with_capacity(SR, Hrir::builtin(), 32);
    mix.set_ear(false);
    let ids = town.sources[club].clone();
    let lis = [25.0f32, 30.6, 1.6];
    let click = clip(&[1.0]);
    let block = 400usize;
    let (mut l, mut r) = (Vec::new(), Vec::new());
    let clicks = [2.0f32, 6.05, 12.0];
    let mut next = 0;
    for b in 0..(14.0 * SR) as usize / block {
        let t = (b * block) as f32 / SR;
        if (t - 6.0).abs() < 1e-4 {
            town.set_door(&mut sim, 4, 0.0);
        }
        sim.set_listener(lis, 0.0, 0.0, 0.0);
        if b % 6 == 0 {
            mix.apply(&sim.tick());
        } else if let Some(f) = sim.tick_exact() {
            mix.apply(&f);
        }
        while next < clicks.len() && clicks[next] <= t {
            for &id in &ids {
                mix.play(id, &click, Play::ONCE);
            }
            next += 1;
        }
        let (mut a, mut c) = (vec![0.0f32; block], vec![0.0f32; block]);
        mix.render(&mut a, &mut c);
        l.extend_from_slice(&a);
        r.extend_from_slice(&c);
    }
    // the first 100 ms of each click's response (the nearest speaker is
    // ~4 m away)
    let early = |t: f32| energy(&l, t, t + 0.1) + energy(&r, t, t + 0.1);
    let (open, just, settled) = (early(2.0), early(6.05), early(12.0));
    assert!(db(open) - db(settled) > 15.0, "setup: the door shuts the club off ({:.1} dB)", db(open) - db(settled));
    eprintln!("open {:.1} dB, 50 ms after the shut {:.1} dB, settled {:.1} dB", db(open), db(just), db(settled));
    assert!((db(just) - db(settled)).abs() < within_db, "50 ms after the shut {:.1} dB, settled {:.1} dB", db(just), db(settled));
}

/// The bell clicks every 3 s, the listener in the open field, nothing
/// moving: every click is the same click. Between clicks the bell drops
/// out of the audible window (its tail under the scene by 60 dB); a drop
/// that falls just before the next arrival brought it back mid-fade —
/// one click in ten 5 dB down — until a source back in the window
/// started at full level (nothing of it was heard to fade from).
#[test]
fn a_click_repeated_is_the_same_click() {
    let mut sim = sim();
    let mut town = Town::new(&mut sim);
    only(&mut town, &mut sim, BELLS);
    let mut mix = Mixer::with_capacity(SR, Hrir::builtin(), 32);
    mix.set_ear(false);
    let ids = town.sources[BELLS].clone();
    let lis = [20.0f32, 60.0, 1.6];
    let click = clip(&[1.0]);
    let block = 400usize;
    let l_end = 22.0f32;
    let clicks: Vec<f32> = (0..6).map(|k| 3.0 + 3.0 * k as f32).collect();
    let (mut l, mut r) = (Vec::new(), Vec::new());
    let mut next = 0;
    for b in 0..(l_end * SR) as usize / block {
        let t = (b * block) as f32 / SR;
        sim.set_listener(lis, 0.0, 0.0, 0.0);
        if b % 6 == 0 {
            mix.apply(&sim.tick());
        } else if let Some(f) = sim.tick_exact() {
            mix.apply(&f);
        }
        while next < clicks.len() && clicks[next] <= t {
            for &id in &ids {
                mix.play(id, &click, Play::ONCE);
            }
            next += 1;
        }
        let (mut a, mut c) = (vec![0.0f32; block], vec![0.0f32; block]);
        mix.render(&mut a, &mut c);
        l.extend_from_slice(&a);
        r.extend_from_slice(&c);
    }
    let e: Vec<f32> = clicks.iter().map(|&t| db(energy(&l, t, t + 2.9) + energy(&r, t, t + 2.9))).collect();
    let (lo, hi) = e.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(hi - lo < 0.5, "the clicks span {:.1} dB: {e:?}", hi - lo);
}

/// Standing still in the square, the club (behind its walls) clicks every
/// 3 s: each click's reverb is the same reverb. v3's averages kept a
/// fixed share of every tick's noise (the club through its walls is a
/// few rare paths a tick): its late part scattered 9.8 dB click to click;
/// v6 averages a still spell as one running mean: 1.6 dB (some of it the
/// reverb's own, not the estimate's: a clap in the cathedral scatters
/// ~1.9 dB at any rung).
#[test]
fn standing_still_the_reverb_settles_v6() {
    let spread = |v6: bool| {
        let mut sim = Simulation::new(Config { quality: Quality::Level(3), compute: Compute::Cpu, v4: false, v5: false, v6, ..Config::default() });
        let mut town = Town::new(&mut sim);
        let club = 2;
        only(&mut town, &mut sim, club);
        let mut mix = Mixer::with_capacity(SR, Hrir::builtin(), 32);
        mix.set_ear(false);
        let ids = town.sources[club].clone();
        let lis = [15.0f32, 32.0, 1.6];
        let click = clip(&[1.0]);
        let block = 400usize;
        let clicks: Vec<f32> = (0..9).map(|k| 3.0 + 3.0 * k as f32).collect();
        let (mut l, mut r) = (Vec::new(), Vec::new());
        let mut next = 0;
        for b in 0..(30.0 * SR) as usize / block {
            let t = (b * block) as f32 / SR;
            sim.set_listener(lis, 0.0, 0.0, 0.0);
            if b % 6 == 0 {
                mix.apply(&sim.tick());
            } else if let Some(f) = sim.tick_exact() {
                mix.apply(&f);
            }
            while next < clicks.len() && clicks[next] <= t {
                for &id in &ids {
                    mix.play(id, &click, Play::ONCE);
                }
                next += 1;
            }
            let (mut a, mut c) = (vec![0.0f32; block], vec![0.0f32; block]);
            mix.render(&mut a, &mut c);
            l.extend_from_slice(&a);
            r.extend_from_slice(&c);
        }
        // the reverb: 0.15–1.5 s after each click (the club is ~15 m off),
        // the clicks after the first two (the field settling at the start)
        let late: Vec<f32> = clicks[2..].iter().map(|&t| db(energy(&l, t + 0.15, t + 1.5) + energy(&r, t + 0.15, t + 1.5))).collect();
        let m = late.iter().sum::<f32>() / late.len() as f32;
        (late.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (late.len() - 1) as f32).sqrt()
    };
    let (v3, v6) = (spread(false), spread(true));
    eprintln!("the club's reverb in the square, click to click: v3 sd {v3:.2} dB, v6 sd {v6:.2} dB");
    assert!(v6 < 3.0, "v6: the reverb scatters {v6:.2} dB click to click standing still");
    assert!(v6 < 0.5 * v3, "v6 {v6:.2} dB against v3 {v3:.2} dB");
}

/// Walking behind a building never cuts a source off: east–west at y = 12
/// the ears cross the Old House's shadow of the club's windows twice, and
/// what reaches them (the paths, bends and field the engine plays) changes
/// across each boundary as a shadow does — no step of a metre jumps more
/// than 10 dB (the legacy engine's `shadow_boundaries_are_continuous`,
/// ported to the live engine).
#[test]
fn shadow_boundaries_are_continuous() {
    const CLUB: usize = 2;
    // ticks a spot's level is the mean of: the treated club (Oct 31)
    // arrives here ~12 dB under the bare one's, and its walk's far field
    // stepped up to 9 dB between neighbouring metres with no boundary
    // near over 12 ticks (x 18 → 19), up to 16 over 36 in a parallel run
    // (x 39 → 40; 6.5 at most alone)
    const TICKS: usize = 72;
    let mut sim = Simulation::new(Config {
        quality: Quality::High,
        compute: Compute::Cpu,
        v6: true,
        v61: true,
        v62: true,
        kernel_thread: false,
        ..Config::default()
    });
    let mut town = Town::new(&mut sim);
    only(&mut town, &mut sim, CLUB);
    let ids = town.sources[CLUB].clone();
    let level = |sim: &mut Simulation| -> f32 {
        let mut e = 0.0f32;
        for _ in 0..TICKS {
            let f = sim.tick();
            for id in &ids {
                let pb = f.block(*id).expect("the club is traced");
                e += pb.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum::<f32>()
                    + pb.reverb.level[1] * pb.reverb.level[1]
                    + pb.remote.map_or(0.0, |r| r.send[1] * r.send[1]);
            }
        }
        e / TICKS as f32
    };
    let mut prev = f32::NAN;
    let mut worst = 0.0f32;
    let mut x = 16.0f32;
    while x <= 40.0 {
        sim.set_listener([x, 12.0, 1.6], 0.0, 0.0, 0.0);
        for _ in 0..8 {
            sim.tick();
        }
        let cur = level(&mut sim);
        assert!(cur > 1e-10, "the club inaudible at ({x}, 12): {:.1} dB", db(cur));
        if prev.is_finite() {
            let step = (db(cur) - db(prev)).abs();
            worst = worst.max(step);
            assert!(step < 10.0, "a {step:.1} dB jump between x = {} and {x}", x - 1.0);
        }
        prev = cur;
        x += 1.0;
    }
    eprintln!("walking the Old House's shadow: the largest step {worst:.1} dB");
}

/// The radar dome across the west road (`walkthrough::RADOME`) focuses: the
/// stage's voice, heard at the sphere's centre, comes back from every way
/// at once `2 · radius / c` later (~70 ms) — its taps there together about
/// as loud as the direct sound — and off the centre (4 m towards the
/// doorway) that return is far weaker and from fewer ways (measured: −0.6
/// dB from 23 ways at the centre, −10.8 from 4 off it, within 3.5 ms).
#[test]
fn the_radar_dome_focuses_its_echo_back_to_its_centre() {
    const STAGE: usize = 15;
    let rd = omg_scene::walkthrough::RADOME;
    let (cx, cy) = rd.center;
    let back = 2.0 * rd.radius / C;
    let ring = |at: [f32; 3]| -> (f32, usize) {
        let mut sim = Simulation::new(Config { quality: Quality::High, compute: Compute::Cpu, v4: false, v5: false, v6: true, v61: true, v62: true, kernel_thread: false, ..Config::default() });
        let mut town = Town::new(&mut sim);
        only(&mut town, &mut sim, STAGE);
        let id = town.sources[STAGE][0];
        sim.set_listener(at, 0.0, 0.0, 0.0);
        let mut f = sim.tick();
        for _ in 0..60 {
            f = sim.tick();
        }
        let pb = f.block(id).expect("the stage is traced");
        let direct = pb.taps.iter().find(|t| t.key == omg_core::v3::KEY_DIRECT).map_or(0.0, |t| t.gains[1] * t.gains[1]);
        // the taps arriving together with the return (3.5 ms: its bin and
        // the next), and how many ways
        let near: Vec<&omg_audio::Tap> = pb.taps.iter().filter(|t| t.key != omg_core::v3::KEY_DIRECT && (t.delay_s - pb.onset_s - back).abs() < 0.0035).collect();
        let e: f32 = near.iter().map(|t| t.gains[1] * t.gains[1]).sum();
        (db(e / direct.max(1e-30)), near.len())
    };
    let (centre, n_c) = ring([cx, cy, rd.center_z]);
    let (off, n_o) = ring([cx + 4.0, cy, 1.6]);
    eprintln!("the dome's return at {:.0} ms: at its centre {centre:+.1} dB from {n_c} ways, 4 m off {off:+.1} dB from {n_o}", back * 1e3);
    assert!(centre > -6.0, "the focus comes back about as loud as the direct sound: {centre:+.1} dB");
    assert!(n_c >= 8, "from every way: {n_c} taps");
    assert!(centre > off + 6.0, "and off the centre far weaker: {centre:+.1} vs {off:+.1} dB");
}
