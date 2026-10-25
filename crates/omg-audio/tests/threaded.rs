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
