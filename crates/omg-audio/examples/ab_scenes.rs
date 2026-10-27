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
}

impl Rig {
    fn new(v4: bool) -> Self {
        let compute = if std::env::var("GPU").is_ok() { omg_audio::Compute::GpuSync } else { omg_audio::Compute::Cpu };
        let cfg = Config {
            v4,
            compute,
            kernel_thread: false,
            kernel_step_samples: 1 << 24, // HEAD: whole kernels in the tick, like a fast kernel thread
            ..Config::default()
        };
        let mut sim = Simulation::new(cfg);
        sim.set_geometry(geometry());
        let mix = Mixer::new(SR, Hrir::none());
        let exact = std::env::var("NOEXACT").is_err();
        Self { sim, mix, blocks: 0, exact }
    }

    /// Render one 10 ms block (ticking every 50 ms; exact frames between
    /// the ticks at HEAD, as the engine thread publishes them): the
    /// output's (left, right).
    fn block(&mut self) -> ([f32; BLOCK], [f32; BLOCK]) {
        if self.blocks % TICK_BLOCKS == 0 {
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
    let worst = err.iter().cloned().fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a });
    println!("walk out: 1.4 m/s vs 0.2 m/s — off by {rms:.1} dB RMS (worst {worst:+.1} dB) · clicks {clicks} (slow walk {slow_clicks})");
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
}
