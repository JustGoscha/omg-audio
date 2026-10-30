//! Effect: steady Doppler under a game's timing, through the whole facade
//! (Simulation → Frame → Mixer).
//!
//! Claim: a listener running at constant speed toward a tone hears a
//! CONSTANT Doppler pitch even though the game samples the motion
//! irregularly: the pose only moves on 60 Hz physics steps, simulation
//! ticks run long (45–62 ms against a 50 ms period: the next starts late)
//! and frames reach the mixer on 256-sample block boundaries. Frames carry
//! the time their pose was current (`Simulation::set_listener_at`), and
//! the mixer moves every path's delay over the time that really passed.
//! (Divided by the average update interval instead, a tick that caught
//! three physics steps and the next that caught four were heard as a
//! speed change: ±20–30 % of the shift, ±1.8 % pitch at a 27 m/s sprint.)
//!
//! Isolation: free field (one path: no room interference in the
//! measurement), bus-only mixer, 1 kHz loop.
//! Combined: omg-dsp tests/fx_doppler.rs `reflections_born_mid_motion_
//! stay_in_step` (a room's re-keying paths on one timeline).

use omg_audio::{clip, Config, Frame, Hrir, Mixer, Play, Simulation, SourceDesc, SPEED_OF_SOUND};

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const F0: f32 = 1000.0;

fn zc_freq(x: &[f32]) -> f32 {
    let mut first = None;
    let mut last = 0.0f64;
    let mut count = 0usize;
    for n in 1..x.len() {
        let (a, b) = (x[n - 1], x[n]);
        if a < 0.0 && b >= 0.0 {
            let t = (n - 1) as f64 + (-a / (b - a)) as f64;
            if first.is_none() {
                first = Some(t);
            }
            last = t;
            count += 1;
        }
    }
    match first {
        Some(f) if count >= 2 => ((count - 1) as f64 / (last - f) * SR as f64) as f32,
        _ => 0.0,
    }
}

/// Listener runs at `v` m/s straight at a 1 kHz loop under game timing;
/// the pitch of every 20 ms window after the first second, relative to
/// the Doppler prediction.
fn run(v: f32, secs: f32) -> Vec<f32> {
    const PHYSICS_HZ: f32 = 60.0;
    const PERIOD: f32 = 0.05;
    let mut sim = Simulation::new(Config::default());
    let src = sim.add_source(SourceDesc::at([0.0, 0.0, 1.6]));
    let mut m = Mixer::new(SR, Hrir::none());
    m.settle_ear();
    let tone: Vec<f32> = (0..4800).map(|k| (std::f32::consts::TAU * F0 * k as f32 / SR).sin() * 0.5).collect();
    let x0 = -(v * secs + 30.0);
    // the pose at wall time t, and the physics step it has been current since
    let pose = |t: f32| -> (f32, f32) {
        let step = (t * PHYSICS_HZ).floor() / PHYSICS_HZ;
        (x0 + v * step, step)
    };
    // tick k starts at s and publishes after its duration; one that runs
    // past the period delays the next
    let mut ticks: Vec<(f32, f32)> = Vec::new();
    let (mut s, mut rng) = (0.0f32, 0x9e37_79b9u32);
    while s < secs + 1.0 {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let d = 0.045 + 0.017 * (rng as f32 / u32::MAX as f32);
        ticks.push((s, s + d));
        s += PERIOD.max(d);
    }
    let n = (secs * SR) as usize;
    let (mut l, mut r) = (vec![0.0f32; n], vec![0.0f32; n]);
    let (mut next, mut started, mut b) = (0usize, false, 0usize);
    while b < n {
        let t = b as f32 / SR;
        // every tick published by now ran; only the newest frame is taken
        let mut newest: Option<Frame> = None;
        while next < ticks.len() && ticks[next].1 <= t {
            let (x, since) = pose(ticks[next].0);
            sim.set_listener_at([x, 0.0, 1.6], 0.0, 0.0, 0.0, since as f64);
            newest = Some(sim.tick());
            next += 1;
        }
        if let Some(f) = newest {
            m.apply(&f);
            if !started {
                m.play(src, &clip(&tone), Play::LOOP);
                started = true;
            }
        }
        let e = (b + BLOCK).min(n);
        m.render(&mut l[b..e], &mut r[b..e]);
        b = e;
    }
    let want = F0 * (1.0 + v / SPEED_OF_SOUND);
    (SR as usize..n - 960).step_by(960).map(|w| zc_freq(&l[w..w + 960]) / want - 1.0).collect()
}

#[test]
fn a_running_listener_hears_a_steady_doppler_under_game_timing() {
    for v in [5.0f32, 10.0, 27.0] {
        let p = run(v, 4.0);
        let mean = p.iter().sum::<f32>() / p.len() as f32;
        let worst = p.iter().map(|x| x.abs()).fold(0.0, f32::max);
        eprintln!("v {v} m/s: mean {:+.3} %, worst 20 ms deviation {:.3} %", mean * 100.0, worst * 100.0);
        let limit = 0.004 + 0.02 * v / SPEED_OF_SOUND;
        assert!(worst < limit, "v={v} m/s: pitch wobbles {:.2} % (limit {:.2} %)", worst * 100.0, limit * 100.0);
    }
}

/// A source running at `v` m/s straight at a still listener, its place
/// moved on 60 Hz physics steps and sent every 120 Hz frame — each send
/// stamped with when that place was first sent (`Engine::set_source`), or
/// (`each`) with the send's own time, as before. The pitch of every 20 ms
/// window after the first second, relative to the Doppler prediction.
fn run_source(v: f32, secs: f32, each: bool) -> Vec<f32> {
    const PHYSICS_HZ: f64 = 60.0;
    const FRAME_HZ: f64 = 120.0;
    let mut sim = Simulation::new(Config::default());
    let x0 = v * (secs + 1.0) + 30.0;
    let src = sim.add_source(SourceDesc::at([x0, 0.0, 1.6]));
    sim.set_listener_at([0.0, 0.0, 1.6], 0.0, 0.0, 0.0, 0.0);
    let mut m = Mixer::new(SR, Hrir::none());
    m.settle_ear();
    let tone: Vec<f32> = (0..4800).map(|k| (std::f32::consts::TAU * F0 * k as f32 / SR).sin() * 0.5).collect();
    let n = (secs * SR) as usize;
    let (mut l, mut r) = (vec![0.0f32; n], vec![0.0f32; n]);
    let (mut frame, mut placed, mut started) = (0u64, (f32::NAN, 0.0f64), false);
    let mut b = 0usize;
    while b < n {
        let t = b as f64 / SR as f64;
        // the frames up to now: each sends the place of the last physics step
        while frame as f64 / FRAME_HZ <= t {
            let tf = frame as f64 / FRAME_HZ;
            let x = x0 - v * ((tf * PHYSICS_HZ).floor() / PHYSICS_HZ) as f32;
            let stamp = if !each && x == placed.0 { placed.1 } else { tf };
            placed = (x, stamp);
            sim.set_time(tf);
            sim.set_source_at(src, [x, 0.0, 1.6], true, stamp);
            if frame % 6 == 0 {
                m.apply(&sim.tick());
                if !started {
                    m.play(src, &clip(&tone), Play::LOOP);
                    started = true;
                }
            }
            frame += 1;
        }
        let e = (b + BLOCK).min(n);
        m.render(&mut l[b..e], &mut r[b..e]);
        b = e;
    }
    let want = F0 / (1.0 - v / SPEED_OF_SOUND);
    (SR as usize..n - 960).step_by(960).map(|w| zc_freq(&l[w..w + 960]) / want - 1.0).collect()
}

/// A source moved by a game's physics at half its frame rate keeps a
/// steady Doppler: a place sent again is stamped with when it was first
/// sent there (`Engine::set_source`). Stamped with each send, its track
/// stood still for a frame and jumped at twice its speed the next: the
/// town's passing car's pitch jittered 16.5 Hz rms against 4.3 moving
/// smoothly (`assess cartone PHYS=60 STAMP=each`).
#[test]
fn a_running_source_sounds_a_steady_doppler_under_game_timing() {
    for v in [10.0f32, 27.0] {
        let worst = |p: &[f32]| p.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let p = run_source(v, 4.0, false);
        let mean = p.iter().sum::<f32>() / p.len() as f32;
        let old = worst(&run_source(v, 4.0, true));
        eprintln!("v {v} m/s: mean {:+.3} %, worst 20 ms deviation {:.3} % (each send stamped: {:.3} %)", mean * 100.0, worst(&p) * 100.0, old * 100.0);
        let limit = 0.004 + 0.02 * v / SPEED_OF_SOUND;
        assert!(worst(&p) < limit, "v={v} m/s: pitch wobbles {:.2} % (limit {:.2} %)", worst(&p) * 100.0, limit * 100.0);
    }
}

/// Who runs in `run_threaded`.
#[derive(Clone, Copy, PartialEq)]
enum Runner {
    /// The source runs at the still listener.
    Source,
    /// The listener runs at the still source, its pose stamped with the
    /// physics step that moved it (`Engine::set_listener_at`).
    Listener,
    /// `Listener`, its pose stamped with each frame's own time instead
    /// (`Engine::set_listener`).
    ListenerEachFrame,
}

/// The threaded engine's timing, as the Godot demo feeds it: the game
/// renders at 120 Hz and moves the runner on 60 Hz physics steps, sending
/// the listener's pose and the source's place every frame (a source's
/// place stamped when first sent, `Engine::set_source`; the still
/// listener with each frame's time, `Engine::set_listener`); the
/// simulation thread traces exact frames at 120 Hz on its own phase
/// between 20 Hz ticks, with the commands that arrived by then; the audio
/// thread takes the newest frame at each 512-sample callback. The pitch
/// of every 20 ms window after the first second, relative to the Doppler
/// prediction.
fn run_threaded(v: f32, secs: f32, runner: Runner) -> Vec<f32> {
    const FRAME_HZ: f64 = 120.0;
    const PHYSICS_HZ: f64 = 60.0;
    let mut sim = Simulation::new(Config::default());
    let x0 = v * (secs + 1.0) + 30.0;
    let src_at = if runner == Runner::Source { x0 } else { 0.0 };
    let src = sim.add_source(SourceDesc::at([src_at, 0.0, 1.6]));
    let mut m = Mixer::new(SR, Hrir::none());
    m.settle_ear();
    let tone: Vec<f32> = (0..4800).map(|k| (std::f32::consts::TAU * F0 * k as f32 / SR).sin() * 0.5).collect();
    // the game's frame f at time f/FRAME_HZ: the runner where the last
    // physics step left it (two frames a step), and that step's time
    let state = |f: u64| -> (f64, f32, f64) {
        let step = (f / 2) as f64 / PHYSICS_HZ;
        (f as f64 / FRAME_HZ, v * step as f32, step)
    };
    let (mut frame, mut placed) = (0u64, (f32::NAN, 0.0f64));
    let (mut exact_k, mut tick_k) = (0u64, 0u64);
    let mut published: Option<Frame> = None;
    let n = (secs * SR) as usize;
    let (mut l, mut r) = (vec![0.0f32; n], vec![0.0f32; n]);
    let (mut b, mut started) = (0usize, false);
    while b < n {
        let cb = b as f64 / SR as f64;
        // the simulation thread up to this callback: its exact frames and
        // ticks in time order, each after the game's commands sent by then
        loop {
            let te = 0.0031 + exact_k as f64 / FRAME_HZ;
            // (each tick just after a frame without a physics step)
            let tt = 0.0091 + tick_k as f64 / 20.0;
            let at = te.min(tt);
            if at > cb {
                break;
            }
            while (frame as f64 / FRAME_HZ) <= at {
                let (t, run, step) = state(frame);
                match runner {
                    Runner::Source => {
                        let x = x0 - run;
                        let stamp = if x == placed.0 { placed.1 } else { t };
                        placed = (x, stamp);
                        sim.set_listener_at([0.0, 0.0, 1.6], 0.0, 0.0, 0.0, t);
                        sim.set_source_at(src, [x, 0.0, 1.6], true, stamp);
                    }
                    Runner::Listener => sim.set_listener_at([-x0 + run, 0.0, 1.6], 0.0, 0.0, 0.0, step),
                    Runner::ListenerEachFrame => sim.set_listener_at([-x0 + run, 0.0, 1.6], 0.0, 0.0, 0.0, t),
                }
                frame += 1;
            }
            if tt <= te {
                published = Some(sim.tick());
                tick_k += 1;
            } else {
                if let Some(f) = sim.tick_exact() {
                    published = Some(f);
                }
                exact_k += 1;
            }
        }
        if let Some(f) = published.take() {
            m.apply(&f);
            if !started {
                m.play(src, &clip(&tone), Play::LOOP);
                started = true;
            }
        }
        let e = (b + 512).min(n);
        m.render(&mut l[b..e], &mut r[b..e]);
        b = e;
    }
    let want = if runner == Runner::Source { F0 / (1.0 - v / SPEED_OF_SOUND) } else { F0 * (1.0 + v / SPEED_OF_SOUND) };
    (SR as usize..n - 960).step_by(960).map(|w| zc_freq(&l[w..w + 960]) / want - 1.0).collect()
}

/// A frame is the world at the simulation's clock (`Frame::time`). Under
/// the threaded engine's timing a source running past a still listener
/// keeps a steady Doppler: before, frames carried the still listener's
/// first stamp, no time passed between any two of them, and the mixer
/// divided each frame's motion by the average interval between arrivals
/// — a car's pitch in the Godot demo jittered 9.0 Hz rms on a 500 Hz
/// tone (0.3 since, as headless: `record.gd` carfar). A listener running
/// on physics steps keeps it too when its pose is stamped with the step
/// (`Engine::set_listener_at`, as the Godot node does): sent again with
/// each frame's time, a tick after a frame without a step traced it
/// standing still and the next frame ran twice as fast (the demo's walk
/// 1.1 Hz rms, 0.06 stamped). (The mixer's update windows also had to
/// begin where the previous one ends: unequal steps tore the timeline.)
#[test]
fn doppler_stays_steady_under_the_threaded_engines_timing() {
    let worst = |p: &[f32]| p.iter().map(|x| x.abs()).fold(0.0, f32::max);
    for v in [10.0f32, 27.0] {
        let limit = 0.004 + 0.02 * v / SPEED_OF_SOUND;
        for runner in [Runner::Source, Runner::Listener] {
            let p = run_threaded(v, 4.0, runner);
            let mean = p.iter().sum::<f32>() / p.len() as f32;
            let each = if runner == Runner::Listener { format!(" (each frame stamped: {:.3} %)", worst(&run_threaded(v, 4.0, Runner::ListenerEachFrame)) * 100.0) } else { String::new() };
            let who = if runner == Runner::Source { "source" } else { "listener" };
            eprintln!("{who} at {v} m/s: mean {:+.3} %, worst 20 ms deviation {:.3} %{each}", mean * 100.0, worst(&p) * 100.0);
            assert!(worst(&p) < limit, "{who} at {v} m/s: pitch wobbles {:.2} % (limit {:.2} %)", worst(&p) * 100.0, limit * 100.0);
        }
    }
}
