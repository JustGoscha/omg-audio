//! Effect tests for the engine facade's `Mixer`: a simulation frame in,
//! audio out.
//!
//! - a click played on a source 34.3 m away reaches the ears 100 ms
//!   later (the play log is read at the path's emission time);
//! - mix-wide budgets: of many playing sources only ~`voices` render,
//!   muted and finished sources cost nothing;
//! - stopping a loop ends it (after the paths in flight play out).

use omg_audio::{clip, Config, Hrir, Mixer, Play, Simulation, SourceDesc, VoiceId};

const SR: f32 = 48_000.0;

fn render(m: &mut Mixer, secs: f32) -> (Vec<f32>, Vec<f32>) {
    let n = (secs * SR) as usize;
    let (mut l, mut r) = (vec![0.0; n], vec![0.0; n]);
    for (cl, cr) in l.chunks_mut(256).zip(r.chunks_mut(256)) {
        m.render(cl, cr);
    }
    (l, r)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

#[test]
fn a_click_arrives_one_path_delay_later() {
    let mut sim = Simulation::new(Config::default());
    let src = sim.add_source(SourceDesc::at([34.3, 0.0, 1.6]));
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let frame = sim.tick();

    let mut m = Mixer::new(SR, Hrir::none());
    m.apply(&frame);
    // a 1 ms pulse (a single-sample impulse would land between samples:
    // the path delay is fractional and the log interpolates)
    let mut click = vec![0.0f32; 96];
    click[..48].fill(1.0);
    let v = m.play(src, &clip(&click), Play::ONCE);
    assert_ne!(v, VoiceId::NONE);
    let (l, r) = render(&mut m, 0.3);
    let first = l.iter().zip(&r).position(|(a, b)| a.abs().max(b.abs()) > 1e-4).expect("the click arrives");
    let want = (34.3 / omg_audio::SPEED_OF_SOUND * SR) as usize;
    eprintln!("click: first sample {first}, path delay {want}");
    assert!(first.abs_diff(want) <= 48, "arrives at {first}, path delay {want} samples");
    assert!(l[..want - 48].iter().all(|x| x.abs() < 1e-6), "nothing before the path delay");
}

#[test]
fn only_the_voice_budget_renders_and_silence_is_free() {
    let mut sim = Simulation::new(Config { max_traced_sources: 64, ..Config::default() });
    let ids: Vec<_> = (0..40)
        .map(|i| {
            let a = i as f32 * 0.5;
            sim.add_source(SourceDesc::at([a.cos() * (4.0 + i as f32), a.sin() * (4.0 + i as f32), 1.5]))
        })
        .collect();
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let frame = sim.tick();
    let mut m = Mixer::new(SR, Hrir::none());
    m.set_budget(128, 8, 12);
    m.apply(&frame);
    let noise: Vec<f32> = (0..4800u32).map(|i| ((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0 - 1.0) * 0.3).collect();
    let bed = clip(&noise);
    // nothing playing: nothing renders
    render(&mut m, 0.05);
    assert_eq!(m.rendering(), 0, "silent sources cost nothing");
    for &id in &ids {
        m.play(id, &bed, Play::LOOP);
    }
    m.apply(&frame); // re-rank: the budget only sees sounding sources
    render(&mut m, 0.6); // the virtualized voices fade out
    // the 12 loudest, plus up to 25% hysteresis for voices already playing
    assert!((12..=15).contains(&m.rendering()), "{} rendering", m.rendering());
    let (l, _) = render(&mut m, 0.1);
    assert!(rms(&l) > 1e-3, "and they are heard");
    // mute everything: rendering stops once the faders are down
    for &id in &ids {
        m.set_source_gain(id, 0.0);
    }
    render(&mut m, 0.6);
    assert_eq!(m.rendering(), 0, "muted sources cost nothing");
}

#[test]
fn stopping_a_loop_silences_it() {
    let mut sim = Simulation::new(Config::default());
    let src = sim.add_source(SourceDesc::at([3.0, 0.0, 1.6]));
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let frame = sim.tick();
    let mut m = Mixer::new(SR, Hrir::none());
    m.apply(&frame);
    let tone: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.0577).sin() * 0.5).collect();
    let v = m.play(src, &clip(&tone), Play::LOOP.gain(0.8));
    let (l, _) = render(&mut m, 0.2);
    assert!(rms(&l[4800..]) > 1e-3, "the loop plays");
    m.stop(v);
    render(&mut m, 0.2); // the path in flight plays out
    let (l, _) = render(&mut m, 0.1);
    assert!(rms(&l) < 1e-5, "stopped: {}", rms(&l));
    assert!(m.take_meter(src).peak > 0.0, "the channel meter saw it");
}
