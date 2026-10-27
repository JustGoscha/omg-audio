//! End to end through `threaded::Engine`: a simulation thread, the
//! latest-frame handoff and the command ring deliver audible output.

use omg_audio::threaded::{Engine, EngineConfig};
use omg_audio::{clip, Geometry, Material, Play, SourceDesc, VoiceId};
use std::time::{Duration, Instant};

#[test]
fn the_engine_renders_a_source_end_to_end() {
    let sr = 48_000.0;
    let (mut engine, mut out) = Engine::new(sr, EngineConfig::default());
    let mut g = Geometry::new();
    let floor = g.material(Material::WOOD);
    g.add_box([-10.0, -10.0, -0.2], [10.0, 10.0, 0.0], floor);
    engine.set_geometry(g);
    let src = engine.add_source(SourceDesc::at([4.0, 1.0, 1.5]));
    engine.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    let tone: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.0577).sin() * 0.5).collect();
    assert_ne!(engine.play(src, &clip(&tone), Play::LOOP), VoiceId::NONE);

    let (mut l, mut r) = (vec![0.0f32; 480], vec![0.0f32; 480]);
    let t0 = Instant::now();
    let mut heard = 0.0f32;
    // ~1 s of audio in 10 ms blocks, paced like a real callback
    while t0.elapsed() < Duration::from_secs(5) {
        out.render(&mut l, &mut r);
        heard = heard.max(l.iter().chain(&r).fold(0.0f32, |a, x| a.max(x.abs())));
        if heard > 1e-3 && out.frame().is_some_and(|f| f.version > 3) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let (tick_ms, ticks) = engine.sim_stats();
    eprintln!("threaded: peak {heard}, {ticks} ticks, last {tick_ms:.2} ms");
    assert!(heard > 1e-3, "the engine is heard (peak {heard})");
    assert!(ticks >= 3);
    drop(engine); // stops and joins the sim thread
    out.render(&mut l, &mut r); // the audio side keeps working on its own
}

/// The listener pose is stamped with the time it was first sent at its
/// position: a game that renders faster than its physics moves the
/// listener sends the same pose a few times, and it has been there since
/// the physics step (Doppler is timed by these stamps).
#[test]
fn a_repeated_pose_keeps_the_time_it_was_first_sent() {
    let (mut engine, mut out) = Engine::new(48_000.0, EngineConfig::default());
    let (mut l, mut r) = (vec![0.0f32; 480], vec![0.0f32; 480]);
    let mut frame_at = |out: &mut omg_audio::threaded::AudioOut, x: f32| -> f64 {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(5) {
            out.render(&mut l, &mut r);
            if let Some(f) = out.frame().filter(|f| f.listener.position[0] == x) {
                return f.time.expect("the threaded engine stamps every pose");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("no frame with the listener at x = {x}");
    };
    engine.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    std::thread::sleep(Duration::from_millis(60));
    engine.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0); // the same pose again
    let first = frame_at(&mut out, 0.0);
    std::thread::sleep(Duration::from_millis(60));
    engine.set_listener([1.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    let moved = frame_at(&mut out, 1.0);
    eprintln!("stamps: first {first:.3} s, moved {moved:.3} s");
    assert!(first < 0.03, "a repeated pose keeps its first stamp: {first}");
    assert!(moved - first > 0.1, "a new pose gets a new stamp: {first} → {moved}");
}

/// Debug paths cross from the simulation thread to the game thread only
/// while enabled.
#[test]
fn debug_paths_reach_the_game_thread_while_enabled() {
    let (mut engine, _out) = Engine::new(48_000.0, EngineConfig::default());
    let mut g = Geometry::new();
    let floor = g.material(Material::WOOD);
    g.add_box([-10.0, -10.0, -0.2], [10.0, 10.0, 0.0], floor);
    engine.set_geometry(g);
    let src = engine.add_source(SourceDesc::at([4.0, 1.0, 1.5]));
    engine.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    std::thread::sleep(Duration::from_millis(150));
    assert!(engine.debug_paths().is_empty(), "off by default");
    engine.set_debug_paths(true);
    let t0 = Instant::now();
    let mut paths = Vec::new();
    while paths.is_empty() && t0.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
        paths = engine.debug_paths();
    }
    assert!(!paths.is_empty(), "paths while enabled");
    assert!(paths.iter().all(|p| p.source == src && p.points.len() >= 2));
    engine.set_debug_paths(false);
    std::thread::sleep(Duration::from_millis(150));
    assert!(engine.debug_paths().is_empty(), "none once disabled");
}

/// A source switched on together with its sound (a thrown ball's whoosh):
/// the sound is not lost while the source waits for its first trace —
/// the simulation traces it at once and the mixer plays what it emitted
/// from the beginning.
#[test]
fn a_sound_on_a_source_just_switched_on_is_heard_from_its_start() {
    let sr = 48_000.0;
    let (mut engine, mut out) = Engine::new(sr, EngineConfig::default());
    let mut g = Geometry::new();
    let m = g.material(Material::CONCRETE);
    g.add_box([-10.0, -10.0, -0.2], [10.0, 10.0, 0.0], m);
    engine.set_geometry(g);
    let pos = [0.4, 0.0, 1.6];
    let src = engine.add_source(SourceDesc { position: pos, gain: 1.0, active: false });
    engine.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    let (mut l, mut r) = (vec![0.0f32; 256], vec![0.0f32; 256]);
    let block = Duration::from_secs_f32(256.0 / sr);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(300) {
        out.render(&mut l, &mut r);
        std::thread::sleep(block);
    }
    // a 1 ms click: shorter than a tick, lost before
    let click: Vec<f32> = (0..48).map(|_| 0.8).collect();
    engine.set_source(src, pos, true);
    engine.play(src, &clip(&click), Play::ONCE);
    let start = Instant::now();
    let mut n = 0usize;
    let mut onset = None;
    while start.elapsed() < Duration::from_millis(1000) && onset.is_none() {
        out.render(&mut l, &mut r);
        onset = l.iter().zip(&r).position(|(a, b)| a.abs().max(b.abs()) > 1e-3).map(|i| (n + i) as f32 / sr);
        n += 256;
        std::thread::sleep(block);
    }
    eprintln!("onset {:?} s after the play", onset);
    let onset = onset.expect("the click is heard");
    // the first trace of a near source in an empty scene: well under a tick (50 ms)
    assert!(onset < 0.05, "heard {onset} s late");
}
