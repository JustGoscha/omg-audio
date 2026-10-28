//! Engine v4 end to end: a clap traced, its dense response convolved,
//! rendered by the mixer.
//!
//! - in a canyon (two rock cliffs 40 m apart) the clap comes back: the
//!   output peaks at the round trips (~116, ~233 ms), above the
//!   rock's scattered field between them;
//! - in a closed room the tail decays smoothly, with nothing after it;
//! - the dense convolution's render cost stays a fraction of real time.

use omg_audio::{clip, Config, Geometry, Hrir, Material, Mixer, Play, Simulation, SourceDesc};

const SR: f32 = 48_000.0;

fn render(m: &mut Mixer, secs: f32) -> Vec<f32> {
    let n = (secs * SR) as usize;
    let (mut l, mut r) = (vec![0.0; n], vec![0.0; n]);
    for (cl, cr) in l.chunks_mut(256).zip(r.chunks_mut(256)) {
        m.render(cl, cr);
    }
    l.iter().zip(&r).map(|(a, b)| 0.5 * (a * a + b * b)).collect()
}

/// Mean power in `[t0, t1)` seconds.
fn power(p: &[f32], t0: f32, t1: f32) -> f32 {
    let (a, b) = ((t0 * SR) as usize, ((t1 * SR) as usize).min(p.len()));
    p[a..b].iter().sum::<f32>() / (b - a).max(1) as f32
}

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-30).log10()
}

/// A hand clap: a 3 ms decaying noise burst.
fn clap() -> Vec<f32> {
    (0..144u32)
        .map(|i| {
            let n = ((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0) - 1.0;
            n * (-(i as f32) / 40.0).exp()
        })
        .collect()
}

/// Settle the simulation standing still, then clap once and render.
fn clap_in(g: Geometry, lis: [f32; 3], src: [f32; 3], secs: f32) -> (Vec<f32>, f32) {
    let mut sim = Simulation::new(Config::default());
    assert!(sim.config().v4, "v4 is the default engine");
    sim.set_geometry(g);
    let id = sim.add_source(SourceDesc::at(src));
    sim.set_listener(lis, 0.0, 0.0, 0.0);
    for _ in 0..40 {
        sim.tick();
    }
    sim.flush_kernels();
    let frame = sim.tick();
    assert!(!frame.kernels.is_empty(), "a dense kernel was built");
    let mut m = Mixer::new(SR, Hrir::none());
    m.apply(&frame);
    m.play(id, &clip(&clap()), Play::ONCE);
    let t0 = std::time::Instant::now();
    let p = render(&mut m, secs);
    let load = t0.elapsed().as_secs_f32() / secs;
    (p, load)
}

#[test]
fn a_clap_comes_back_from_the_canyon() {
    let mut g = Geometry::new();
    let rock = g.material(Material::ROCK);
    g.add_box([-20.5, -25.0, 0.0], [-20.0, 25.0, 24.0], rock);
    g.add_box([20.0, -25.0, 0.0], [20.5, 25.0, 24.0], rock);
    g.ground_plane(0.0, rock);
    let (p, load) = clap_in(g, [0.0, 0.0, 1.6], [0.4, 0.0, 1.15], 1.2);
    // the round trips to either cliff: ~2·20 m / 343
    let rt = |k: f32| k * 40.0 / omg_audio::SPEED_OF_SOUND;
    // 3 ms windows (the clap's length): the echo's loudest one near the
    // round trip against the mean of the ones in the gap before it
    let win = |t: f32| power(&p, t, t + 0.003);
    for k in [1.0f32, 2.0] {
        let echo = (0..8).map(|i| win(rt(k) - 0.006 + i as f32 * 0.0015)).fold(0.0f32, f32::max);
        let gap = (0..10).map(|i| win(rt(k) - 0.06 + i as f32 * 0.003)).sum::<f32>() / 10.0;
        eprintln!("echo {k}: {:.1} dB, quiet before it {:.1} dB", db(echo), db(gap));
        // the first return clearly (6 dB); later ones sink toward the
        // rock's scattered field (the offline trace shows the same)
        let need = if k == 1.0 { 4.0 } else { 2.0 };
        assert!(echo > need * gap, "round trip {k}: the echo stands out of the scattered field before it");
    }
    eprintln!("render load {:.3}× real time", load);
}

#[test]
fn a_rooms_tail_decays_and_ends() {
    let (lx, ly, lz) = (6.0, 5.0, 3.0);
    let mut g = Geometry::new();
    let c = g.material(Material::CONCRETE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], c);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], c);
    g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], c);
    g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], c);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], c);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], c);
    let (p, load) = clap_in(g, [4.0, 3.0, 1.6], [1.5, 1.5, 1.5], 4.0);
    let windows: Vec<f32> = (0..16).map(|i| db(power(&p, 0.05 + i as f32 * 0.1, 0.15 + i as f32 * 0.1))).collect();
    eprintln!("room tail dB per 100 ms: {:?}", windows.iter().map(|x| x.round() as i32).collect::<Vec<_>>());
    eprintln!("render load {:.3}× real time", load);
    // it decays, monotonically within noise, by at least 20 dB over a second
    assert!(windows[0] - windows[10] > 20.0, "the tail decays");
    for w in windows.windows(2).take(10) {
        assert!(w[1] < w[0] + 3.0, "no bump in the tail: {windows:?}");
    }
}

/// The mixer's convolution budget: a source outside it plays its traced
/// response as the parametric reverb — the same energy in the mid band
/// (a 1 kHz burst: within 4 dB), nothing dry. Broadband it comes out
/// duller: the reverb's high band is a first-order shelf in each loop, so
/// a white click's top octaves fall short of the convolution's.
#[test]
fn a_source_outside_the_convolution_budget_keeps_its_reverb() {
    let (lx, ly, lz) = (6.0, 5.0, 3.0);
    let mut g = Geometry::new();
    let c = g.material(Material::CONCRETE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], c);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], c);
    g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], c);
    g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], c);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], c);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], c);
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(g);
    let id = sim.add_source(SourceDesc::at([1.5, 1.5, 1.5]));
    sim.set_listener([4.0, 3.0, 1.6], 0.0, 0.0, 0.0);
    for _ in 0..40 {
        sim.tick();
    }
    sim.flush_kernels();
    let frame = sim.tick();
    assert!(!frame.kernels.is_empty());
    let tone: Vec<f32> = (0..480)
        .map(|i| {
            let t = i as f32 / 480.0;
            (std::f32::consts::TAU * 1000.0 * i as f32 / SR).sin() * (std::f32::consts::PI * t).sin().powi(2)
        })
        .collect();
    // 20 dB under the ear's reflex (~90 dB SPL): the reverbs compared,
    // not how the ear takes them
    let tail = |budget: usize, sig: &[f32]| -> f32 {
        let mut m = Mixer::new(SR, Hrir::none());
        m.set_convolution_budget(budget);
        m.apply(&frame);
        m.play(id, &clip(sig), Play::ONCE.gain(0.1));
        let p = render(&mut m, 2.0);
        power(&p, 0.05, 1.0)
    };
    let (conv, param) = (tail(4, &tone), tail(0, &tone));
    let (conv_w, param_w) = (tail(4, &clap()), tail(0, &clap()));
    eprintln!(
        "tail 50 ms – 1 s, 1 kHz: convolved {:.1} dB, parametric {:.1} dB; white click: {:.1} / {:.1} dB",
        db(conv),
        db(param),
        db(conv_w),
        db(param_w)
    );
    assert!(param > 0.0 && param_w > 0.0, "not dry");
    assert!((db(conv) - db(param)).abs() < 4.0, "the same reverb energy in the mid band within 4 dB");
}

/// A flutter between bare parallel walls: the echoes come as a train,
/// on and on past the early window — the solver keeps the chains that
/// stand out of the field (not the loudest: early chains via absorbing
/// surfaces are louder, buried in it), they play as taps and, past the
/// tap budget, as impulses in the kernel. Rendered, the clap's output
/// over 120–300 ms has an echo in every few cells, not a noise envelope.
#[test]
fn a_flutter_train_runs_on() {
    let mut g = Geometry::new();
    let hard = g.material(Material::CONCRETE);
    let tile = g.material(Material::ACOUSTIC_TILE);
    let carpet = g.material(Material::CARPET);
    let (lx, ly, lz, t) = (4.0, 6.0, 2.7, 0.2);
    g.add_box([-t, 0.0, 0.0], [0.0, ly, lz], hard);
    g.add_box([lx, 0.0, 0.0], [lx + t, ly, lz], hard);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], tile);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], tile);
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], carpet);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], tile);
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(g);
    let id = sim.add_source(SourceDesc::at([1.5, 3.15, 1.3]));
    sim.set_listener([1.3, 3.0, 1.6], 0.0, 0.0, 0.0);
    for _ in 0..80 {
        sim.tick();
    }
    sim.flush_kernels();
    let frame = sim.tick();
    let mut m = Mixer::new(SR, Hrir::none());
    m.apply(&frame);
    render(&mut m, 0.5);
    m.play(id, &clip(&clap()), Play::ONCE);
    let p = render(&mut m, 0.4);
    // power in 0.5 ms cells over 120–300 ms, each against the median of
    // the 20 ms around it (the tail decays through the window): echoes
    // stand 5 dB out of their surroundings; a diffuse tail's cells do so
    // in ~2 % (noise: one cell's power is exponentially distributed)
    let cell = 24usize;
    let env: Vec<f32> = p[(0.12 * SR) as usize..(0.30 * SR) as usize].chunks(cell).map(|c| c.iter().sum::<f32>()).collect();
    let at: Vec<f32> = (20..env.len() - 20)
        .filter(|&i| {
            let mut w = env[i - 20..i + 20].to_vec();
            w.sort_by(|x, y| x.total_cmp(y));
            env[i] > 3.16 * w[w.len() / 2]
        })
        .map(|i| 120.0 + i as f32 * 0.5)
        .collect();
    eprintln!("120–300 ms: echoes 5 dB over their surroundings at {at:?} ms");
    // (before, folded into the noise past the taps' few: none) — a train
    // through the window, not a few stragglers at its start
    assert!(at.len() >= 6, "the flutter runs on as a train: {at:?}");
    assert!(at.last().unwrap() - at[0] > 100.0, "through the window: {at:?}");
}
