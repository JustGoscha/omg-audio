//! Headless A/B scenes: the simulation and the mixer in lockstep (a tick
//! every 50 ms, rendered in 10 ms blocks), measured — so two versions of
//! the engine can be compared on the same geometry. Lines marked `// HEAD`
//! use API the morning baseline (6f2d5f3) lacks; strip them to run there.
//!
//!     cargo run --release -p omg-audio --example ab_scenes [-- v3]
//!
//! Scenes: two rooms (concrete, A 8×6×3 m, B 6×6×3 m) joined by a door
//! hole with a door leaf (an object), B open to a grass field.
//!  - clap: an impulse in A — the source's own level, 10 ms windows, until
//!    it is gone: where it crosses −20/−40/−60 dB, its decay rate, any cut.
//!  - door: noise in A heard from B; the leaf closes, then opens: the
//!    level change and how long it takes to settle.
//!  - walk: noise in A; the listener walks out through B into the field at
//!    1.4 m/s and at 0.2 m/s (near the steady state): how far the level
//!    lags (dB off the slow walk at the same spot), and clicks.
//!  - tone: 500 Hz from 60 m away in the field, the listener still: the
//!    level's wobble (flanging, warble) in 20 ms windows.
//!  - fly: noise in A; the listener flies through the A | B wall (not the
//!    door) at 6 m/s, into A and out of it: how long the level takes to
//!    settle where it stops, and how far off it is the first 0.5 s.
//!
//!  - crackle: an organ-like bass (70–280 Hz) in A — the output above
//!    4 kHz is nothing the source holds: every burst there is a
//!    discontinuity (a kernel swapped or dropped, a jump in a path or the
//!    reverb's input). A door between A and B swinging shut and open, the
//!    walk out, the fly through the wall, standing still: bursts over
//!    −45 / −35 dB (5 ms windows against the output's level).
//!  - church: a clap 0.4 m in front of the listener in a stone nave
//!    (16 × 22 × 15 m): the tail's level per 20 ms against the same clap
//!    with the response frozen at the clap (no new frames) — what the
//!    re-traced response adds on top of the tail's own grain (flicker),
//!    standing still and turning (the hands circle with the look).
//!
//! GPU=1 walks on the GPU (whole walks, synchronous); GPU=live runs the
//! live walk the apps use (pooled late rays, paced, a tick or more late)
//! in real time — the ticks then wait for the wall clock.
use omg_audio::{clip, Config, Geometry, Hrir, Material, Mixer, Play, Simulation, SourceDesc, SourceId};

const SR: f32 = 48_000.0;
const BLOCK: usize = 480; // 10 ms
const TICK_BLOCKS: usize = 5; // 50 ms

fn geometry() -> Geometry {
    let mut g = Geometry::new();
    let wall = g.material(Material::CONCRETE);
    let grass = g.material(Material::GRASS);
    let (t, h) = (0.2, 3.0);
    g.ground_plane(0.0, grass);
    // ceilings
    g.add_box([0.0, 0.0, h], [14.0, 6.0, h + t], wall);
    // A: west, south, north walls
    g.add_box([-t, 0.0, 0.0], [0.0, 6.0, h], wall);
    g.add_box([0.0, -t, 0.0], [14.0, 0.0, h], wall);
    g.add_box([0.0, 6.0, 0.0], [14.0, 6.0 + t, h], wall);
    // A | B with a door hole (y 2.5..3.5, z 0..2.1)
    g.add_box([7.9, 0.0, 0.0], [8.1, 2.5, h], wall);
    g.add_box([7.9, 3.5, 0.0], [8.1, 6.0, h], wall);
    g.add_box([7.9, 2.5, 2.1], [8.1, 3.5, h], wall);
    // B's east wall with an opening to the field (y 2..4, z 0..2.5)
    g.add_box([14.0, 0.0, 0.0], [14.0 + t, 2.0, h], wall);
    g.add_box([14.0, 4.0, 0.0], [14.0 + t, 6.0, h], wall);
    g.add_box([14.0, 2.0, 2.5], [14.0 + t, 4.0, h], wall);
    g
}

struct Rig {
    sim: Simulation,
    mix: Mixer,
    blocks: usize,
    exact: bool,
    /// GPU=live: the wall clock of the last tick (ticks keep real time).
    live: Option<std::time::Instant>,
}

impl Rig {
    fn new(v4: bool) -> Self {
        let gpu = std::env::var("GPU").ok();
        let live = gpu.as_deref() == Some("live");
        let compute = match gpu {
            Some(_) if live => omg_audio::Compute::Gpu,
            Some(_) => omg_audio::Compute::GpuSync,
            None => omg_audio::Compute::Cpu,
        };
        let cfg = Config {
            v4,
            compute,
            kernel_thread: false,
            kernel_step_samples: 1 << 24, // HEAD: whole kernels in the tick, like a fast kernel thread
            ..Config::default()
        };
        let mut sim = Simulation::new(cfg);
        sim.set_geometry(geometry());
        let mix = Mixer::new(SR, if std::env::var("HRIR").is_ok() { Hrir::builtin() } else { Hrir::none() });
        let exact = std::env::var("NOEXACT").is_err();
        Self { sim, mix, blocks: 0, exact, live: live.then(std::time::Instant::now) }
    }

    /// Render one 10 ms block (ticking every 50 ms; exact frames between
    /// the ticks at HEAD, as the engine thread publishes them): the
    /// output's (left, right).
    fn block(&mut self) -> ([f32; BLOCK], [f32; BLOCK]) {
        if self.blocks % TICK_BLOCKS == 0 {
            if let Some(t) = &mut self.live {
                let left = 0.05 - t.elapsed().as_secs_f32();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f32(left));
                }
                *t = std::time::Instant::now();
            }
            let f = self.sim.tick();
            self.mix.apply(&f);
        } else if self.exact {
            if let Some(f) = self.sim.tick_exact() { // HEAD
                self.mix.apply(&f); // HEAD
            } // HEAD
        }
        self.blocks += 1;
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        self.mix.render(&mut l, &mut r);
        (l, r)
    }

    fn run(&mut self, secs: f32) {
        for _ in 0..(secs * SR / BLOCK as f32) as usize {
            self.block();
        }
    }
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

fn clap(v4: bool) {
    let mut rig = Rig::new(v4);
    let hands = rig.sim.add_source(SourceDesc { position: [3.3, 3.0, 1.4], gain: 1.0, active: true });
    rig.sim.set_listener([3.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    rig.run(3.0);
    let burst: Vec<f32> = noise(240, 7).iter().enumerate().map(|(i, x)| x * (-(i as f32) / 40.0).exp()).collect();
    rig.mix.play(hands, &clip(&burst), Play::ONCE);
    let mut env = Vec::new();
    for _ in 0..400 {
        rig.block();
        env.push(rig.mix.take_meter(hands).rms);
    }
    let peak = env.iter().cloned().fold(0.0f32, f32::max);
    let rel: Vec<f32> = env.iter().map(|&e| db(e / peak)).collect();
    let cross = |lvl: f32| rel.iter().position(|&d| d < lvl).map_or("never".into(), |i| format!("{} ms", i * 10));
    // the decay rate between −5 and −35 dB (a line fit over those windows)
    let pts: Vec<(f32, f32)> = rel.iter().enumerate().skip_while(|(_, &d)| d > -5.0).take_while(|(_, &d)| d > -35.0).map(|(i, &d)| (i as f32 * 0.01, d)).collect();
    let slope = if pts.len() > 3 {
        let n = pts.len() as f32;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f32>() / n, pts.iter().map(|p| p.1).sum::<f32>() / n);
        pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f32>() / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f32>()
    } else {
        f32::NAN
    };
    // a cut: a window more than 30 dB under the one before, after 50 ms
    let cut = (6..rel.len()).find(|&i| rel[i - 1] > -70.0 && rel[i] < rel[i - 1] - 30.0);
    #[allow(unused)]
    let traced: Option<f32> = None;
    let traced = rig.sim.dense_response(hands).map(|d| d.rt60()[1]); // HEAD
    println!(
        "clap: −20 dB at {}, −40 at {}, −60 at {} · decay → RT60 {:.2} s (traced {}) · {}",
        cross(-20.0),
        cross(-40.0),
        cross(-60.0),
        -60.0 / slope,
        traced.map_or("-".into(), |t| format!("{t:.2} s")),
        cut.map_or("no cut".into(), |i| format!("CUT at {} ms ({:.0} → {:.0} dB)", i * 10, rel[i - 1], rel[i]))
    );
    let marks: Vec<String> = [5, 10, 20, 50, 100, 150, 200, 300, 399].iter().map(|&i| format!("{}ms {:.0}", i * 10, rel[i])).collect();
    println!("      {}", marks.join(" · "));
}

fn steady(rig: &mut Rig, id: SourceId, secs: f32) -> f32 {
    let n = (secs * 100.0) as usize;
    let mut e = 0.0;
    for _ in 0..n {
        rig.block();
        let r = rig.mix.take_meter(id).rms;
        e += r * r;
    }
    (e / n as f32).sqrt()
}

/// Per-100 ms levels (dB) for `secs`.
fn trace(rig: &mut Rig, id: SourceId, secs: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for _ in 0..(secs * 10.0) as usize {
        let mut e = 0.0;
        for _ in 0..10 {
            rig.block();
            let r = rig.mix.take_meter(id).rms;
            e += r * r / 10.0;
        }
        out.push(db(e.sqrt()));
    }
    out
}

fn settle(levels: &[f32], fin: f32) -> String {
    match levels.iter().rposition(|&l| (l - fin).abs() > 1.0) {
        None => "at once".into(),
        Some(i) if i + 1 >= levels.len() => "not settled".into(),
        Some(i) => format!("{} ms", (i + 1) * 100),
    }
}

/// Per tick: the block's energy by kind (direct, bends, chains), the
/// dense response's energy — what moves tick to tick.
fn parts(rig: &mut Rig, id: SourceId, ticks: usize) {
    for _ in 0..ticks {
        let f = rig.sim.tick();
        rig.mix.apply(&f);
        for _ in 0..TICK_BLOCKS {
            let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            rig.mix.render(&mut l, &mut r);
        }
        rig.blocks += TICK_BLOCKS;
        let b = f.block(id).unwrap();
        let (mut d, mut e, mut c, mut nc) = (0.0f32, 0.0f32, 0.0f32, 0);
        for t in &b.taps {
            let g = t.gains[1] * t.gains[1];
            match t.key {
                1 => d += g,
                20_000..=8_999_999 => e += g,
                _ => {
                    c += g;
                    nc += 1
                }
            }
        }
        let dense: f32 = rig.sim.dense_response(id).map_or(0.0, |r| r.bins.iter().map(|b| b[1]).sum());
        println!("  direct {:6.1} dB · bends {:6.1} · chains {:6.1} ({nc}) · dense {:6.1} dB", db(d.sqrt()), db(e.sqrt()), db(c.sqrt()), db(dense.sqrt()));
    }
}

fn door(v4: bool) {
    let mut rig = Rig::new(v4);
    let src = rig.sim.add_source(SourceDesc { position: [3.0, 3.0, 1.5], gain: 1.0, active: true });
    let leaf = rig.sim.add_object([7.95, 2.5, 0.0], [8.05, 3.5, 2.1], Material::DOOR);
    rig.sim.set_object_enabled(leaf, false);
    rig.sim.set_listener([11.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    rig.mix.play(src, &clip(&noise(96_000, 3)), Play::LOOP);
    rig.run(3.0);
    let open = db(steady(&mut rig, src, 1.0));
    rig.sim.set_object_enabled(leaf, true);
    if std::env::var("PARTS").is_ok() {
        parts(&mut rig, src, 30);
    }
    let closing = trace(&mut rig, src, 4.0);
    let closed = closing[30..].iter().sum::<f32>() / 10.0;
    rig.sim.set_object_enabled(leaf, false);
    let opening = trace(&mut rig, src, 4.0);
    let reopened = opening[30..].iter().sum::<f32>() / 10.0;
    println!(
        "door: open {open:.1} dB, closed {closed:.1} dB ({:+.1} dB) — settles in {}; reopened {reopened:.1} dB in {}",
        closed - open,
        settle(&closing, closed),
        settle(&opening, reopened)
    );
    let show = |v: &[f32]| v.iter().take(12).map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" ");
    println!("      closing, per 100 ms: {}", show(&closing));
    println!("      opening, per 100 ms: {}", show(&opening));
}

/// Walk from x 5 to 20 along y 3 at `speed`: (x, level dB) per 100 ms,
/// and the clicks in the output.
fn walk(v4: bool, speed: f32) -> (Vec<(f32, f32)>, usize) {
    let mut rig = Rig::new(v4);
    let src = rig.sim.add_source(SourceDesc { position: [2.0, 3.0, 1.5], gain: 1.0, active: true });
    rig.sim.set_listener([5.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    rig.mix.play(src, &clip(&noise(96_000, 5)), Play::LOOP);
    rig.run(3.0);
    let (mut x, mut out, mut clicks) = (5.0f32, Vec::new(), 0usize);
    let (mut e, mut k) = (0.0f32, 0);
    let (mut p1, mut p2) = (0.0f32, 0.0f32);
    let mut d2_rms = 1e-6f32;
    while x < 20.0 {
        x += speed * BLOCK as f32 / SR;
        rig.sim.set_listener([x, 3.0, 1.7], 0.0, 0.0, 0.0);
        let (l, _) = rig.block();
        for &s in &l {
            // a click: the second difference far over its running RMS
            let d2 = s - 2.0 * p1 + p2;
            if d2.abs() > 12.0 * d2_rms && d2.abs() > 1e-4 {
                clicks += 1;
            }
            d2_rms = (0.999 * d2_rms * d2_rms + 0.001 * d2 * d2).sqrt().max(1e-6);
            p2 = p1;
            p1 = s;
        }
        let r = rig.mix.take_meter(src).rms;
        e += r * r;
        k += 1;
        if k == 10 {
            out.push((x, db((e / 10.0).sqrt())));
            e = 0.0;
            k = 0;
        }
    }
    (out, clicks)
}

fn walk_out(v4: bool) {
    let (fast, clicks) = walk(v4, 1.4);
    let (slow, slow_clicks) = walk(v4, 0.2);
    let at = |x: f32| slow.iter().min_by(|a, b| (a.0 - x).abs().total_cmp(&(b.0 - x).abs())).unwrap().1;
    let err: Vec<f32> = fast.iter().map(|&(x, l)| l - at(x)).collect();
    let rms = (err.iter().map(|e| e * e).sum::<f32>() / err.len() as f32).sqrt();
    let (wi, worst) = err.iter().cloned().enumerate().fold((0, 0.0f32), |a, (i, b)| if b.abs() > a.1.abs() { (i, b) } else { a });
    println!(
        "walk out: 1.4 m/s vs 0.2 m/s — off by {rms:.1} dB RMS (worst {worst:+.1} dB at x {:.1}) · clicks {clicks} (slow walk {slow_clicks})",
        fast[wi].0
    );
    let show: Vec<String> = [6.0, 8.0, 9.0, 11.0, 13.0, 14.5, 16.0, 19.0]
        .iter()
        .map(|&x| {
            let f = fast.iter().min_by(|a, b| (a.0 - x).abs().total_cmp(&(b.0 - x).abs())).unwrap().1;
            format!("x{x}: {f:.0}/{:.0}", at(x))
        })
        .collect();
    println!("      level fast/slow: {}", show.join(" · "));
}

fn tone(v4: bool) {
    let mut rig = Rig::new(v4);
    let src = rig.sim.add_source(SourceDesc { position: [80.0, 3.0, 10.0], gain: 1.0, active: true });
    rig.sim.set_listener([20.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    let t: Vec<f32> = (0..48_000).map(|i| (2.0 * std::f32::consts::PI * 500.0 * i as f32 / SR).sin() * 0.5).collect();
    rig.mix.play(src, &clip(&t), Play::LOOP);
    rig.run(3.0);
    // 20 ms windows of the left output
    let mut lv = Vec::new();
    for _ in 0..200 {
        let (a, _) = rig.block();
        let (b, _) = rig.block();
        let e = a.iter().chain(b.iter()).map(|x| x * x).sum::<f32>() / (2 * BLOCK) as f32;
        lv.push(db(e.sqrt()));
    }
    let mean = lv.iter().sum::<f32>() / lv.len() as f32;
    let sd = (lv.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lv.len() as f32).sqrt();
    let (lo, hi) = lv.iter().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
    println!("tone 60 m, still: level {mean:.1} dB, wobble ±{sd:.2} dB (range {:.1} dB)", hi - lo);
}

/// The outdoor bed (an ambience loop) heard inside B by its opening to
/// the field; a door leaf closes the opening, then opens it: the bed's
/// level per 100 ms and how long it takes to settle.
fn bed(v4: bool) {
    let mut rig = Rig::new(v4);
    rig.mix.set_ambience(noise(96_000, 11), false); // HEAD
    let leaf = rig.sim.add_object([13.95, 2.0, 0.0], [14.25, 4.0, 2.5], Material::DOOR);
    rig.sim.set_object_enabled(leaf, false);
    rig.sim.set_listener([12.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    let level = |rig: &mut Rig, secs: f32| -> Vec<f32> {
        let mut out = Vec::new();
        for _ in 0..(secs * 10.0) as usize {
            for _ in 0..10 {
                rig.block();
            }
            out.push(db(rig.mix.take_env_meters().0.rms)); // HEAD
        }
        out
    };
    let open = level(&mut rig, 3.0);
    let open = open[20..].iter().sum::<f32>() / 10.0;
    rig.sim.set_object_enabled(leaf, true);
    let closing = level(&mut rig, 4.0);
    let closed = closing[30..].iter().sum::<f32>() / 10.0;
    rig.sim.set_object_enabled(leaf, false);
    let opening = level(&mut rig, 4.0);
    let reopened = opening[30..].iter().sum::<f32>() / 10.0;
    println!(
        "outdoor bed: open {open:.1} dB, closed {closed:.1} dB ({:+.1} dB) — settles in {}; reopened {reopened:.1} dB in {}",
        closed - open,
        settle(&closing, closed),
        settle(&opening, reopened)
    );
    let show = |v: &[f32]| v.iter().take(15).map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" ");
    println!("      closing, per 100 ms: {}", show(&closing));
    println!("      opening, per 100 ms: {}", show(&opening));
}

/// A noise source in room A, the listener still in it: the level's
/// wobble per 100 ms (the trace's noise, tick to tick) and the kernels
/// rebuilt per second.
fn still(v4: bool) {
    let mut rig = Rig::new(v4);
    let src = rig.sim.add_source(SourceDesc { position: [2.0, 2.0, 1.5], gain: 1.0, active: true });
    rig.sim.set_listener([5.5, 4.0, 1.7], 0.0, 0.0, 0.0);
    rig.mix.play(src, &clip(&noise(96_000, 9)), Play::LOOP);
    rig.run(3.0);
    let lv = trace(&mut rig, src, 6.0);
    let mean = lv.iter().sum::<f32>() / lv.len() as f32;
    let sd = (lv.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lv.len() as f32).sqrt();
    let (lo, hi) = lv.iter().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
    let mut seen = std::collections::HashSet::new();
    for _ in 0..40 {
        let f = rig.sim.tick();
        for (_, k) in &f.kernels {
            seen.insert(std::sync::Arc::as_ptr(k) as usize);
        }
        rig.mix.apply(&f);
    }
    let kernels = seen.len().saturating_sub(1);
    println!("still in the room: level {mean:.1} dB, wobble ±{sd:.2} dB (range {:.1} dB) · {} kernels rebuilt in 2 s", hi - lo, kernels);
}

/// Fly through the A | B wall at 6 m/s (y 1, away from the door) from
/// `from` to `to` and stop there: the level per 100 ms from the crossing
/// on, against where it settles.
fn fly_through(v4: bool, from: f32, to: f32) -> String {
    let mut rig = Rig::new(v4);
    let src = rig.sim.add_source(SourceDesc { position: [3.0, 3.0, 1.5], gain: 1.0, active: true });
    rig.sim.set_listener([from, 1.0, 1.7], 0.0, 0.0, 0.0);
    rig.mix.play(src, &clip(&noise(96_000, 13)), Play::LOOP);
    rig.run(4.0);
    let before = db(steady(&mut rig, src, 1.0));
    let mut x = from;
    let step = 6.0 * BLOCK as f32 / SR * (to - from).signum();
    while (to - x) * step.signum() > 0.0 {
        x = if (to - x).abs() < step.abs() { to } else { x + step };
        rig.sim.set_listener([x, 1.0, 1.7], 0.0, 0.0, 0.0);
        rig.block();
        rig.mix.take_meter(src);
    }
    let after = trace(&mut rig, src, 5.0);
    let fin = after[40..].iter().sum::<f32>() / 10.0;
    let off = after[..5].iter().map(|l| (l - fin).abs()).fold(0.0f32, f32::max);
    let show = after.iter().take(15).map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" ");
    format!(
        "x {from} → {to}: {before:.1} → {fin:.1} dB, settles in {} (first 0.5 s off by up to {off:.1} dB)\n      per 100 ms: {show}",
        settle(&after, fin)
    )
}

fn fly(v4: bool) {
    println!("fly into A: {}", fly_through(v4, 11.0, 5.0));
    println!("fly out of A: {}", fly_through(v4, 5.0, 11.0));
}

/// High-pass biquad (RBJ, Q 0.707).
struct Hp {
    b: [f32; 3],
    a: [f32; 2],
    x: [f32; 2],
    y: [f32; 2],
}

impl Hp {
    fn new(fc: f32) -> Self {
        let w = 2.0 * std::f32::consts::PI * fc / SR;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + al;
        Self { b: [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0], a: [-2.0 * c / a0, (1.0 - al) / a0], x: [0.0; 2], y: [0.0; 2] }
    }

    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1] - self.a[0] * self.y[0] - self.a[1] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }
}

/// Bursts above 4 kHz in the output: 5 ms windows of the high-passed
/// signal against the output's mean square over the last 200 ms.
struct Crackle {
    hp: [Hp; 2],
    win: Vec<(f32, f32)>,
    acc: (f32, f32, usize),
}

impl Crackle {
    fn new() -> Self {
        Self { hp: [Hp::new(4000.0), Hp::new(4000.0)], win: Vec::new(), acc: (0.0, 0.0, 0) }
    }

    fn feed(&mut self, out: &[f32]) {
        for &x in out {
            let h0 = self.hp[0].tick(x);
            let h = self.hp[1].tick(h0);
            self.acc.0 += h * h;
            self.acc.1 += x * x;
            self.acc.2 += 1;
            if self.acc.2 == 240 {
                self.win.push((self.acc.0 / 240.0, self.acc.1 / 240.0));
                self.acc = (0.0, 0.0, 0);
            }
        }
    }

    /// (windows over −45 dB, over −35 dB, the worst in dB, seconds)
    fn report(&self) -> (usize, usize, f32, f32) {
        let (mut a, mut b, mut worst) = (0, 0, -200.0f32);
        for i in 40..self.win.len() {
            let r = self.win[i - 40..i].iter().map(|w| w.1).sum::<f32>() / 40.0;
            let d = 10.0 * (self.win[i].0.max(1e-20) / r.max(1e-20)).log10();
            a += (d > -45.0) as usize;
            b += (d > -35.0) as usize;
            worst = worst.max(d);
        }
        (a, b, worst, self.win.len() as f32 * 0.005)
    }
}

fn organ(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            [70.0f32, 140.0, 210.0, 280.0].iter().map(|f| (2.0 * std::f32::consts::PI * f * t).sin() * 0.12).sum()
        })
        .collect()
}

fn crackle(v4: bool) {
    let show = |name: &str, c: &Crackle| {
        let (a, b, w, secs) = c.report();
        println!("crackle {name:22} bursts over −45 dB {:5.1}/s, over −35 dB {:5.1}/s, worst {w:6.1} dB", a as f32 / secs, b as f32 / secs);
    };
    // the bass a second long: a whole number of periods of every partial
    let bass = clip(&organ(48_000));
    // still
    {
        let mut rig = Rig::new(v4);
        let src = rig.sim.add_source(SourceDesc { position: [2.0, 2.0, 1.5], gain: 1.0, active: true });
        rig.sim.set_listener([5.5, 4.0, 1.7], 0.0, 0.0, 0.0);
        rig.mix.play(src, &bass, Play::LOOP);
        rig.run(3.0);
        let mut c = Crackle::new();
        for _ in 0..600 {
            let (l, _) = rig.block();
            c.feed(&l);
        }
        show("still", &c);
    }
    // a door swinging shut over 0.8 s, 1.2 s closed, open over 0.8 s,
    // 1.2 s open — four times; the listener in B
    for (name, at) in [("door, listener in B", [11.0f32, 3.0]), ("door, listener in A", [5.0, 4.0])] {
        let mut rig = Rig::new(v4);
        let src = rig.sim.add_source(SourceDesc { position: [3.0, 3.0, 1.5], gain: 1.0, active: true });
        let leaf = rig.sim.add_object([7.95, 2.5, 0.0], [8.05, 2.5, 2.1], Material::DOOR);
        rig.sim.set_listener([at[0], at[1], 1.7], 0.0, 0.0, 0.0);
        rig.mix.play(src, &bass, Play::LOOP);
        rig.run(3.0);
        let mut c = Crackle::new();
        for _ in 0..4 {
            for (frames, from, to) in [(80, 1.0f32, 0.0f32), (120, 0.0, 0.0), (80, 0.0, 1.0), (120, 1.0, 1.0)] {
                for k in 0..frames {
                    let open = from + (to - from) * (k + 1) as f32 / frames as f32;
                    rig.sim.set_object(leaf, [7.95, 2.5, 0.0], [8.05, 2.5 + (1.0 - open) * 1.0, 2.1]);
                    rig.sim.set_object_enabled(leaf, open < 0.999);
                    let (l, _) = rig.block();
                    c.feed(&l);
                }
            }
        }
        show(name, &c);
    }
    // the walk out of A through B into the field, and back
    {
        let mut rig = Rig::new(v4);
        let src = rig.sim.add_source(SourceDesc { position: [2.0, 3.0, 1.5], gain: 1.0, active: true });
        rig.sim.set_listener([5.0, 3.0, 1.7], 0.0, 0.0, 0.0);
        rig.mix.play(src, &bass, Play::LOOP);
        rig.run(3.0);
        let mut c = Crackle::new();
        let mut x = 5.0f32;
        for dir in [1.0f32, -1.0] {
            for _ in 0..(15.0 / 1.4 * 100.0) as usize {
                x += dir * 1.4 * BLOCK as f32 / SR;
                rig.sim.set_listener([x, 3.0, 1.7], 0.0, 0.0, 0.0);
                let (l, _) = rig.block();
                c.feed(&l);
            }
        }
        show("walk out and back", &c);
    }
    // flying through the A | B wall at 6 m/s, to and fro
    {
        let mut rig = Rig::new(v4);
        let src = rig.sim.add_source(SourceDesc { position: [3.0, 3.0, 1.5], gain: 1.0, active: true });
        rig.sim.set_listener([5.0, 1.0, 1.7], 0.0, 0.0, 0.0);
        rig.mix.play(src, &bass, Play::LOOP);
        rig.run(3.0);
        let mut c = Crackle::new();
        let mut x = 5.0f32;
        for _ in 0..3 {
            for (to, hold) in [(11.0f32, 100), (5.0, 100)] {
                while (to - x).abs() > 1e-3 {
                    let step = 6.0 * BLOCK as f32 / SR;
                    x = if (to - x).abs() < step { to } else { x + step * (to - x).signum() };
                    rig.sim.set_listener([x, 1.0, 1.7], 0.0, 0.0, 0.0);
                    let (l, _) = rig.block();
                    c.feed(&l);
                }
                for _ in 0..hold {
                    let (l, _) = rig.block();
                    c.feed(&l);
                }
            }
        }
        show("fly through the wall", &c);
    }
}

fn church_geometry() -> Geometry {
    let mut g = Geometry::new();
    let stone = g.material(Material::STONE);
    let (x1, y1, h, t) = (16.0, 22.0, 15.0, 0.6);
    g.add_box([-t, -t, -t], [x1 + t, y1 + t, 0.0], stone);
    g.add_box([-t, -t, h], [x1 + t, y1 + t, h + t], stone);
    g.add_box([-t, -t, 0.0], [0.0, y1 + t, h], stone);
    g.add_box([x1, -t, 0.0], [x1 + t, y1 + t, h], stone);
    g.add_box([0.0, -t, 0.0], [x1, 0.0, h], stone);
    g.add_box([0.0, y1, 0.0], [x1, y1 + t, h], stone);
    g
}

/// The tail of a clap in the nave, 20 ms levels (dB) for 3 s: the
/// simulation ticking on (`live`), or frozen at the clap.
fn church_tail(v4: bool, live: bool, turn: f32) -> (Vec<f32>, usize) {
    let mut rig = Rig::new(v4);
    rig.sim.set_geometry(church_geometry());
    let ear = [8.0f32, 9.0, 1.7];
    let hands_at = |yaw: f32| [ear[0] + 0.4 * yaw.cos(), ear[1] + 0.4 * yaw.sin(), ear[2] - 0.45];
    let mut yaw = 0.0f32;
    let hands = rig.sim.add_source(SourceDesc { position: hands_at(yaw), gain: 1.0, active: true });
    rig.sim.set_listener(ear, yaw, 0.0, 0.0);
    rig.run(4.0);
    // TONAL=1: a 60 ms bass burst instead of the noise clap — its tail
    // holds nothing above 4 kHz but what a discontinuity adds
    let tonal = std::env::var("TONAL").is_ok();
    let burst: Vec<f32> = if tonal {
        (0..2880).map(|i| {
            let t = i as f32 / SR;
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / 2880.0).cos();
            w * ((2.0 * std::f32::consts::PI * 150.0 * t).sin() + 0.5 * (2.0 * std::f32::consts::PI * 300.0 * t).sin())
        }).collect()
    } else {
        noise(240, 7).iter().enumerate().map(|(i, x)| x * (-(i as f32) / 40.0).exp()).collect()
    };
    rig.mix.play(hands, &clip(&burst), Play::ONCE);
    let mut cr = Crackle::new();
    let mut lv = Vec::new();
    let mut kernels = std::collections::HashSet::new();
    for _ in 0..150 {
        let mut e = 0.0;
        for _ in 0..2 {
            if live {
                yaw += turn * BLOCK as f32 / SR;
                rig.sim.set_source(hands, hands_at(yaw), true);
                rig.sim.set_listener(ear, yaw, 0.0, 0.0);
                let (l, r) = rig.block();
                cr.feed(&l);
                e += l.iter().chain(r.iter()).map(|x| x * x).sum::<f32>();
            } else {
                rig.blocks += 1;
                let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
                rig.mix.render(&mut l, &mut r);
                e += l.iter().chain(r.iter()).map(|x| x * x).sum::<f32>();
            }
            if let Some(k) = rig.sim.dense_response(hands) {
                kernels.insert(k.bins.len() as u64 ^ (k.bins.iter().map(|b| b[1]).sum::<f32>().to_bits() as u64) << 20);
            }
        }
        lv.push(10.0 * (e / (4 * BLOCK) as f32).max(1e-20).log10());
    }
    if live && tonal {
        let (a, b, w, secs) = cr.report();
        println!("      tonal clap tail: bursts over −45 dB {:.1}/s, over −35 dB {:.1}/s, worst {w:.1} dB", a as f32 / secs, b as f32 / secs);
    }
    (lv, kernels.len())
}

fn church(v4: bool) {
    for (name, turn) in [("still", 0.0f32), ("turning 90°/s", std::f32::consts::FRAC_PI_2)] {
        let (frozen, _) = church_tail(v4, false, 0.0);
        let (live, n) = church_tail(v4, true, turn);
        let peak = frozen.iter().cloned().fold(f32::MIN, f32::max);
        // the tail from 100 ms on, down to 50 dB under the peak
        let d: Vec<f32> = (5..frozen.len()).filter(|&i| frozen[i] > peak - 50.0).map(|i| live[i] - frozen[i]).collect();
        let mean = d.iter().sum::<f32>() / d.len().max(1) as f32;
        let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d.len().max(1) as f32).sqrt();
        // window to window: how much faster the live tail jumps
        let jump = |v: &[f32]| (6..v.len()).filter(|&i| frozen[i] > peak - 50.0).map(|i| (v[i] - v[i - 1]).abs()).sum::<f32>() / d.len().max(1) as f32;
        println!(
            "church clap, {name:14}: live − frozen {mean:+.2} dB, flicker ±{sd:.2} dB · 20 ms steps {:.2} dB (frozen {:.2}) · {} responses during the tail",
            jump(&live),
            jump(&frozen),
            n
        );
        let show = |v: &[f32]| v.iter().step_by(10).take(15).map(|x| format!("{:.0}", x - peak)).collect::<Vec<_>>().join(" ");
        println!("      live   per 200 ms: {}", show(&live));
        println!("      frozen per 200 ms: {}", show(&frozen));
    }
}

fn main() {
    let v4 = !std::env::args().any(|a| a == "v3");
    let only = std::env::var("ONLY").unwrap_or_default();
    println!("engine {}", if v4 { "v4" } else { "v3" });
    if only.is_empty() || only == "clap" {
        clap(v4);
    }
    if only.is_empty() || only == "door" {
        door(v4);
    }
    if only.is_empty() || only == "walk" {
        walk_out(v4);
    }
    if only.is_empty() || only == "bed" {
        bed(v4);
    }
    if only.is_empty() || only == "still" {
        still(v4);
    }
    if only.is_empty() || only == "tone" {
        tone(v4);
    }
    if only.is_empty() || only == "fly" {
        fly(v4);
    }
    if only.is_empty() || only == "crackle" {
        crackle(v4);
    }
    if only.is_empty() || only == "church" {
        church(v4);
    }
}
