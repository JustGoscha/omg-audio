//! Effect tests for the engine facade's `Mixer`: a simulation frame in,
//! audio out.
//!
//! - a click played on a source 34.3 m away reaches the ears 100 ms
//!   later (the play log is read at the path's emission time);
//! - mix-wide budgets: of many playing sources only ~`voices` render,
//!   muted and finished sources cost nothing;
//! - stopping a loop ends it (after the paths in flight play out);
//! - a one-shot on a silent source starts at full level: a clap's 3 ms
//!   transient is not swallowed by a voice fade-in;
//! - the outdoor ambience plays on the SH bus before the loudness stage:
//!   outdoors it is heard, sealed in only its seep, and a loud bed is
//!   ridden down by the ear adaptation like any source.

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

/// A hand clap 0.6 m away, the first after silence vs one while the
/// source still sounds: (first, second) output peaks. `rerank`: a frame
/// arrives just after the first clap starts (the mixer re-ranks the now
/// sounding source: its direct path moves to point-HRTF rendering).
fn clap_after_silence(hrir: Hrir, rerank: bool) -> (f32, f32) {
    // a 3 ms decaying burst (quiet: the output stage's gain stays put)
    let burst: Vec<f32> = (0..144u32)
        .map(|i| (((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0) - 1.0) * 0.05 * (-(i as f32) / 40.0).exp())
        .collect();
    let clap = clip(&burst);
    let mut sim = Simulation::new(Config::default());
    let src = sim.add_source(SourceDesc::at([0.6, 0.0, 1.6]));
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let frame = sim.tick();
    let mut m = Mixer::new(SR, hrir);
    m.apply(&frame);
    render(&mut m, 0.5);
    m.apply(&frame); // ranked while silent
    render(&mut m, 0.1);
    let peak = |l: &[f32], r: &[f32]| l.iter().zip(r).fold(0.0f32, |p, (a, b)| p.max(a.abs()).max(b.abs()));
    m.play(src, &clap, Play::ONCE);
    let mut first = 0.0f32;
    if rerank {
        let (l, r) = render(&mut m, 0.0005);
        first = peak(&l, &r);
        m.apply(&sim.tick()); // a fresh trace
    }
    let (l, r) = render(&mut m, 0.02);
    let first = first.max(peak(&l, &r));
    m.play(src, &clap, Play::ONCE);
    let (l, r) = render(&mut m, 0.02);
    (first, peak(&l, &r))
}

#[test]
fn a_clap_on_a_silent_source_is_not_faded_in() {
    for (name, hrir, rerank) in [
        ("no hrir", Hrir::none(), false),
        ("hrir", Hrir::builtin(), false),
        ("hrir, re-ranked mid-clap", Hrir::builtin(), true),
    ] {
        let (first, second) = clap_after_silence(hrir, rerank);
        let db = 20.0 * (first / second).log10();
        eprintln!("{name}: clap after silence {first:.4}, while sounding {second:.4}: {db:+.1} dB");
        assert!(db.abs() < 1.0, "{name}: the first clap {db:+.1} dB from the second");
    }
}

/// The outdoor bed rides the same loudness stage as the scene (before,
/// in Godot, it was added after it and only clipped).
#[test]
fn the_ambience_rides_the_loudness_stage() {
    use omg_audio::env::{environment, EnvContext};
    use omg_audio::{Frame, SkyCell};
    // an open field: sky all around and above
    let open: Vec<SkyCell> = (0..32)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 32.0;
            let (z, h) = if i % 4 == 0 { (0.95, 0.31) } else { (0.3, 0.95) };
            SkyCell { dir: [a.cos() * h, a.sin() * h, z], energy: [0.6 / 32.0; 3] }
        })
        .collect();
    let level = |amp: f32, sky: &[SkyCell]| -> (f32, f32) {
        let mut m = Mixer::new(SR, Hrir::none());
        m.set_ambience_gain(16.0);
        let noise: Vec<f32> = (0..2 * 48_000u32)
            .map(|i| amp * (((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0) - 1.0))
            .collect();
        m.set_ambience(noise, false);
        let frame = Frame { env: environment(sky, &EnvContext::default()), ..Default::default() };
        m.apply(&frame);
        let (l, r) = render(&mut m, 3.0);
        let n = l.len();
        (rms(&l[n / 2..]).max(rms(&r[n / 2..])), m.agc_gain())
    };
    let (quiet, g_quiet) = level(0.01, &open);
    let (loud, g_loud) = level(2.0, &open);
    let (sealed, _) = level(0.01, &[]);
    eprintln!("outdoors: quiet bed {quiet:.4} (agc {g_quiet:.2}), loud bed {loud:.4} (agc {g_loud:.2}); sealed in {sealed:.5}");
    assert!(quiet > 1e-3, "the bed is heard outdoors");
    assert!(sealed < 0.5 * quiet, "sealed in, only its seep");
    // 200× (46 dB) more bed: the ear adapts, and the bed is ridden by
    // exactly that gain (it is on the bus the stage rides)
    assert!(g_loud < 0.6 * g_quiet, "a loud bed pulls the gain down");
    let ridden = 200.0 * g_loud / g_quiet;
    assert!((loud / quiet / ridden - 1.0).abs() < 0.2, "ridden: {quiet} → {loud} (×{ridden:.0} expected)");
}

/// The audible window: a quiet source renders alone; a loud one pulls the
/// ear's adaptation down and the quiet one drops out of the window (not
/// rendered: costs nothing); the loud one stops, the ear recovers, the
/// quiet one is back.
#[test]
fn a_loud_scene_drops_the_sounds_it_masks() {
    use omg_audio::{Frame, ParamBlock, SourceId, Tap};
    let block = |g: f32| {
        let mut pb = ParamBlock { version: 1, ..Default::default() };
        pb.taps.push(Tap { key: 0, delay_s: 0.003, dir: [1.0, 0.0, 0.0], gains: [g; 3] });
        pb
    };
    let (loud, quiet) = (SourceId(0), SourceId(1));
    let frame = Frame { version: 1, blocks: vec![(loud, block(30.0)), (quiet, block(3e-3))], ..Default::default() };
    let noise: Vec<f32> = (0..48_000u32).map(|i| ((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0) - 1.0).collect();
    let mut m = Mixer::new(SR, Hrir::none());
    m.apply(&frame);
    // ticks of the simulation (the mixer re-ranks at each frame); the
    // quiet source's own level over the last second
    let run = |m: &mut Mixer, secs: f32| -> (usize, f32) {
        let mut e = 0.0f32;
        let n = (secs * 20.0) as usize;
        for k in 0..n {
            m.apply(&frame);
            render(m, 0.05);
            let r = m.take_meter(quiet).rms;
            if k + 20 >= n {
                e += r * r / 20.0;
            }
        }
        (m.occupancy().0, e.sqrt())
    };
    m.play(quiet, &clip(&noise), Play::LOOP);
    let alone = run(&mut m, 2.0);
    let v = m.play(loud, &clip(&noise), Play::LOOP);
    let with_loud = run(&mut m, 3.0);
    let agc_loud = m.agc_gain();
    m.stop(v);
    let after = run(&mut m, 12.0);
    eprintln!(
        "rendering / quiet source's level: alone {alone:?}, with the loud one {with_loud:?} (agc {agc_loud:.2}), after it {after:?} (agc {:.2})",
        m.agc_gain()
    );
    assert!(alone.0 == 1 && alone.1 > 0.0, "alone, the quiet source is heard");
    assert!(with_loud.0 == 1 && with_loud.1 == 0.0, "the loud one renders, the masked quiet one doesn't");
    assert!(after.1 > 0.0, "the loud one gone, the quiet one is back");
}
