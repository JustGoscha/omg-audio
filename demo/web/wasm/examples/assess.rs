//! The engine measured with IMPULSES: single-sample clicks played on the
//! demo town's sources through the whole engine (simulation + mixer + the
//! render governor), the output split into the tracer's three bands and
//! read as impulse responses — energy per band, early (≤ 80 ms after the
//! free-field arrival) and late, arrival times, envelope steps. No music,
//! no loops: what comes out is the engine's response and nothing else.
//!
//! By default paced in real time like the Godot host (a 20 Hz tick, exact
//! frames at 120 Hz, kernels on their own thread), so latencies land as in
//! the game; `PACE=0` runs as fast as it renders and builds kernels in the
//! tick (deterministic).
//!
//!   door     the club clicks; the listener in the entrance; the club door
//!            shuts at 6 s: the response DELTA s after the shut against the
//!            settled closed one (DELTA=0.1 …)
//!   still    a source clicks every 3 s, nothing moves: how much each
//!            click's response differs from the others (SRC=club|bell|
//!            hands, SPOT=square|field|hall|entrance|north)
//!   walkout  a clap (click) in the hall, then the listener walks out of
//!            the hall's door while it rings: the tail's envelope steps
//!   moving   a car clicks 4 times a second driving past at 100 km/h,
//!            300 m away: each click's arrival against when it physically
//!            arrives (emitted where the car was)
//!   crowd    the club clicks every 3 s while the town's other sources
//!            click around it (CROWD=0: alone): the club's own channel
//!            per click (SPOT=square|north|entrance|inside|field)
//!
//!     MODE=v3|v4|v5  RUNG=0..4 (unset: governed)  QUALITY=0..3  EAR=0
//!     PACE=0  WAV=dir (write the output)  SEED=n  NEE=n (next-event
//!     connections a bounce; 0: every source)
//!
//!     cargo run --release -p omg-web --example assess -- door
use omg_audio::governor::Governor;
use omg_audio::{clip, Clip, Compute, Config, Hrir, Mixer, Play, Quality, Simulation, SourceId};
use omg_web::town::Town;
use std::time::{Duration, Instant};

const SR: f32 = 48_000.0;
const C: f32 = omg_audio::SPEED_OF_SOUND;
/// Render block (the Godot output thread's is 256; any size behaves alike).
const BLOCK: usize = 400;
/// Exact frames every block (120 Hz), a full tick every 6 (20 Hz).
const TICK_EVERY: usize = 6;
/// The early part of a response: up to this long after its free-field
/// arrival (s).
const EARLY: f32 = 0.08;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn mode() -> String {
    env("MODE").unwrap_or_else(|| "v5".into())
}

/// Godot's level → engine gain (sounds.gd: gain = 10^((dB − 90)/20)).
fn db_gain(level_db: f32) -> f32 {
    10f32.powf((level_db - 90.0) / 20.0)
}

/// sounds.gd's spl(authored, anchor, fader).
fn spl(authored: f32, anchor: f32, fader: f32) -> f32 {
    90.0 + 20.0 * authored.log10() + fader - anchor
}

fn db(x: f64) -> f32 {
    (10.0 * x.max(1e-30).log10()) as f32
}

/// One-pole band split at the tracer's crossovers (314 Hz, 2.1 kHz): the
/// bands sum to the input.
struct Bands {
    lo: f32,
    hi: f32,
    c_lo: f32,
    c_hi: f32,
}

impl Bands {
    fn new() -> Self {
        let c = |f: f32| 1.0 - (-std::f32::consts::TAU * f / SR).exp();
        Self { lo: 0.0, hi: 0.0, c_lo: c(314.0), c_hi: c(2100.0) }
    }
    fn split(&mut self, x: f32) -> [f32; 3] {
        self.lo += self.c_lo * (x - self.lo);
        self.hi += self.c_hi * (x - self.hi);
        [self.lo, self.hi - self.lo, x - self.hi]
    }
}

const CLUB: usize = 2;
const BELLS: usize = 7;
const CAR: usize = 11;
const FEET: usize = 13;
const CLUB_DOOR: usize = 4;

/// The engine as the Godot host runs it, and what came out.
struct Rig {
    sim: Simulation,
    mix: Mixer,
    town: Town,
    gov: Governor,
    blocks: usize,
    start: Instant,
    pace: bool,
    l: Vec<f32>,
    r: Vec<f32>,
    pos: [f32; 3],
    yaw: f32,
    ids: Vec<Vec<SourceId>>,
    /// The impulse: one sample.
    click: Clip,
    /// Clicks played: (emission time s, logical source, where it was).
    clicks: Vec<(f32, usize, [f32; 3])>,
    /// Per block: where the listener was (for reading responses).
    track: Vec<[f32; 3]>,
    /// Moving logical sources (dynamic slots): their position over time
    /// (None: switched off).
    movers: Vec<(usize, Box<dyn Fn(f32) -> Option<[f32; 3]>>)>,
    /// The listener's own motion (None: still).
    walk: Option<Box<dyn Fn(f32) -> [f32; 3]>>,
    /// A dynamic slot that follows the listener (0.3 m ahead, chest high).
    hands: Option<usize>,
    bands: Option<[Vec<[f32; 3]>; 2]>,
    /// Engine sources whose own channel energy is read per block (the
    /// mixer's per-source meter, before the ear): `watched`.
    watch: Vec<SourceId>,
    watched: Vec<f64>,
    meters: Vec<(SourceId, omg_audio::Meter)>,
}

impl Rig {
    fn new() -> Self {
        let mode = mode();
        let quality = match env("QUALITY").as_deref() {
            Some("0") => Quality::Low,
            Some("1") => Quality::Medium,
            Some("3") => Quality::Ultra,
            _ => Quality::High,
        };
        // NEE=n: next-event connections per bounce (0: every source)
        let quality = match env("NEE").and_then(|n| n.parse::<usize>().ok()) {
            Some(n) => {
                let mut b = quality.budget(false);
                b.nee = n;
                b.late_nee = n;
                Quality::Custom(b)
            }
            None => quality,
        };
        let pace = env("PACE").as_deref() != Some("0");
        let mut cfg = Config {
            quality,
            compute: Compute::Cpu,
            v4: mode != "v3",
            v5: mode != "v3" && mode != "v4",
            sample_rate: SR,
            ..Config::default()
        };
        if let Some(s) = env("SEED").and_then(|s| s.parse().ok()) {
            cfg.seed = s;
        }
        if !pace {
            cfg.kernel_thread = false;
            cfg.kernel_step_samples = 1 << 24;
        }
        let mut sim = Simulation::new(cfg);
        let town = Town::new(&mut sim);
        let hrir = if env("HRIR").as_deref() == Some("0") { Hrir::none() } else { Hrir::builtin() };
        let mut mix = Mixer::with_capacity(SR, hrir, 64);
        if env("EAR").as_deref() == Some("0") {
            mix.set_ear(false);
        }
        let mut gov = Governor::new(SR);
        gov.apply(&mut mix);
        if let Some(r) = env("RUNG").and_then(|r| r.parse().ok()) {
            gov.pin(&mut mix, Some(r));
        }
        let ids = town.sources.clone();
        let mut rig = Self {
            sim,
            mix,
            town,
            gov,
            blocks: 0,
            start: Instant::now(),
            pace,
            l: Vec::new(),
            r: Vec::new(),
            pos: [0.0; 3],
            yaw: 0.0,
            ids,
            click: clip(&[1.0]),
            clicks: Vec::new(),
            track: Vec::new(),
            movers: Vec::new(),
            walk: None,
            hands: None,
            bands: None,
            watch: Vec::new(),
            watched: Vec::new(),
            meters: Vec::new(),
        };
        rig.levels();
        rig
    }

    /// sounds.gd's levels (dB SPL at 1 m).
    fn levels(&mut self) {
        let set = [
            (0usize, spl(1.0, 90.0, 72.0)),
            (1, spl(0.5, 84.0, 65.0)),
            (3, spl(1.0, 88.0, 76.0)),
            (4, spl(0.5, 80.0, 64.0)),
            (5, spl(1.6, 74.0, 66.0)),
            (6, spl(1.4, 74.0, 66.0)),
            (BELLS, spl(30.0, 100.0, 104.0)),
            (14, spl(0.2, 76.0, 76.0)),
            (CLUB, spl(5.0, 104.0, 112.0) - 6.0206),
            (FEET, spl(1.0, 78.0, 62.0)),
            (CAR, spl(2.2, 92.0, 86.0)),
        ];
        for (i, lv) in set {
            for &id in &self.ids[i].clone() {
                self.sim.set_source_gain(id, db_gain(lv));
            }
        }
    }

    fn t(&self) -> f32 {
        (self.blocks * BLOCK) as f32 / SR
    }

    /// Only these logical sources are traced (the rest muted).
    fn only(&mut self, keep: &[usize]) {
        for i in 0..self.town.defs.len() {
            self.town.set_muted(&mut self.sim, i, !keep.contains(&i));
        }
    }

    fn source_pos(&self, i: usize) -> [f32; 3] {
        let d = &self.town.defs[i];
        [d.pos.0, d.pos.1, d.z]
    }

    fn current_pos(&self, i: usize) -> [f32; 3] {
        if let Some((_, f)) = self.movers.iter().find(|(m, _)| *m == i) {
            return f(self.t()).unwrap_or([0.0; 3]);
        }
        if self.hands == Some(i) {
            return [self.pos[0] + 0.3 * self.yaw.cos(), self.pos[1] + 0.3 * self.yaw.sin(), 1.2];
        }
        self.source_pos(i)
    }

    /// A click on logical source `i` now (every emitter of it), at `gain`.
    fn click(&mut self, i: usize, gain: f32) {
        let at = self.current_pos(i);
        self.clicks.push((self.t(), i, at));
        let c = self.click.clone();
        for &id in &self.ids[i].clone() {
            self.mix.play(id, &c, Play::ONCE.gain(gain));
        }
    }

    /// Render `secs`.
    fn run(&mut self, secs: f32) {
        let n = (secs * SR / BLOCK as f32).round() as usize;
        let base = omg_scene::walkthrough::DYN_BASE;
        for _ in 0..n {
            let t = self.t();
            // the engine's clock is the audio clock: positions set below
            // were current now
            self.sim.set_time(t as f64);
            if let Some(w) = &self.walk {
                let p = w(t);
                let q = w(t + 0.05);
                if (q[0] - p[0]).abs() + (q[1] - p[1]).abs() > 1e-4 {
                    self.yaw = (q[1] - p[1]).atan2(q[0] - p[0]);
                }
                self.pos = p;
            }
            for k in 0..self.movers.len() {
                let (i, p) = (self.movers[k].0, (self.movers[k].1)(t));
                self.town.set_dynamic(&mut self.sim, i - base, p.map(|p| (p, 1.0)));
            }
            if let Some(h) = self.hands {
                let p = [self.pos[0] + 0.3 * self.yaw.cos(), self.pos[1] + 0.3 * self.yaw.sin(), 1.2];
                self.town.set_dynamic(&mut self.sim, h - base, Some((p, 1.0)));
            }
            self.sim.set_listener_at(self.pos, self.yaw, 0.0, 0.0, t as f64);
            let dbg_win = env("DEBUG_TAPS").and_then(|v| v.parse::<f32>().ok());
            {
                let full = self.blocks % TICK_EVERY == 0;
                let f = if full { Some(self.sim.tick()) } else { self.sim.tick_exact() };
                if let (Some(w), Some(f)) = (dbg_win, &f) {
                    if (t - w).abs() < 0.12 {
                        for (id, pb) in &f.blocks {
                            let floor = if env("DEBUG_ALL").is_some() { 0.0 } else { 0.003 };
                            let loud: Vec<String> = pb.taps.iter().filter(|x| x.gains[1] > floor).take(8).map(|x| format!("{}:{:.1}ms {:.1}dB", x.key, x.delay_s * 1e3, 20.0 * x.gains[1].max(1e-12).log10())).collect();
                            if !loud.is_empty() || pb.reverb.level[1] > 0.0 {
                                let k = f.kernels.iter().find(|(kid, _)| kid == id).map_or("none".to_string(), |(_, k)| format!("{:x}/{:.2}s", std::sync::Arc::as_ptr(k) as usize & 0xffffff, k.len as f32 / SR));
                                eprintln!("t {t:.3} {} id {} v{} rev {:.1}dB rt60 {:.2} kernel {k} {}", if full { "tick " } else { "exact" }, id.0, pb.version, 20.0 * pb.reverb.level[1].max(1e-9).log10(), pb.reverb.rt60[1], loud.join(" "));
                            }
                        }
                    }
                }
                if let Some(f) = f {
                    self.mix.apply(&f);
                }
            }
            let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            let t0 = Instant::now();
            self.mix.render(&mut l, &mut r);
            self.gov.report(&mut self.mix, t0.elapsed().as_secs_f64(), BLOCK);
            if let Some(w) = env("DEBUG_TAPS").and_then(|v| v.parse::<f32>().ok()) {
                if (t - w).abs() < 0.12 {
                    let (n, live, pts) = self.mix.occupancy();
                    let e: f32 = l.iter().chain(r.iter()).map(|v| v * v).sum();
                    let mut ms = Vec::new();
                    self.mix.take_meters(&mut ms);
                    let m: Vec<String> = ms.iter().filter(|(_, m)| m.rms > 0.0).map(|(id, m)| format!("{}:{:.0}", id.0, 20.0 * m.rms.log10())).collect();
                    eprintln!("t {t:.4} render: {n} sources, {live} live paths, {pts} hrtf; block energy {:.1} dB; meters {}", 10.0 * e.max(1e-30).log10(), m.join(" "));
                }
            }
            if !self.watch.is_empty() {
                self.mix.take_meters(&mut self.meters);
                let e: f64 = self.meters.iter().filter(|(id, _)| self.watch.contains(id)).map(|(_, m)| (m.rms as f64).powi(2) * BLOCK as f64).sum();
                self.watched.push(e);
            }
            self.l.extend_from_slice(&l);
            self.r.extend_from_slice(&r);
            self.track.push(self.pos);
            self.blocks += 1;
            if self.pace {
                let due = self.start + Duration::from_secs_f32(self.t());
                if let Some(w) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(w);
                }
            }
        }
        self.bands = None;
    }

    fn split(&mut self) -> &[Vec<[f32; 3]>; 2] {
        if self.bands.is_none() {
            let mut out: [Vec<[f32; 3]>; 2] = [Vec::with_capacity(self.l.len()), Vec::with_capacity(self.r.len())];
            for (k, x) in [&self.l, &self.r].into_iter().enumerate() {
                let mut b = Bands::new();
                out[k].extend(x.iter().map(|&v| b.split(v)));
            }
            self.bands = Some(out);
        }
        self.bands.as_ref().unwrap()
    }

    /// Energy per band (both ears) over output times `t0..t1` (s).
    fn energy(&mut self, t0: f32, t1: f32) -> [f64; 3] {
        let a = (t0.max(0.0) * SR) as usize;
        let s = self.split();
        let b = ((t1 * SR) as usize).min(s[0].len());
        let mut e = [0.0f64; 3];
        for i in a..b {
            for k in 0..3 {
                e[k] += (s[0][i][k] * s[0][i][k] + s[1][i][k] * s[1][i][k]) as f64;
            }
        }
        e
    }

    /// The free-field arrival of click `k` (s): emission + the straight
    /// distance from where it was emitted to where the listener was then.
    fn arrival(&self, k: usize) -> f32 {
        let (t, _, p) = self.clicks[k];
        let blk = ((t * SR) as usize / BLOCK).min(self.track.len().saturating_sub(1));
        let l = self.track.get(blk).copied().unwrap_or(self.pos);
        t + ((p[0] - l[0]).powi(2) + (p[1] - l[1]).powi(2) + (p[2] - l[2]).powi(2)).sqrt() / C
    }

    /// Click `k`'s response: (early, late) energy per band, the late part
    /// up to `len` s after its arrival.
    fn response(&mut self, k: usize, len: f32) -> ([f64; 3], [f64; 3]) {
        let a = self.arrival(k);
        (self.energy(a - 0.002, a + EARLY), self.energy(a + EARLY, a + len))
    }

    fn wav(&self, name: &str) {
        let Some(dir) = env("WAV") else { return };
        let path = format!("{dir}/{name}.wav");
        let n = self.l.len();
        let mut b: Vec<u8> = Vec::with_capacity(44 + n * 8);
        let data = (n * 8) as u32;
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes()); // float
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&(SR as u32).to_le_bytes());
        b.extend_from_slice(&((SR as u32) * 8).to_le_bytes());
        b.extend_from_slice(&8u16.to_le_bytes());
        b.extend_from_slice(&32u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data.to_le_bytes());
        for i in 0..n {
            b.extend_from_slice(&self.l[i].to_le_bytes());
            b.extend_from_slice(&self.r[i].to_le_bytes());
        }
        std::fs::write(&path, b).expect("write wav");
        eprintln!("wrote {path}");
    }
}

fn bands_db(e: [f64; 3]) -> String {
    format!("{:6.1} {:6.1} {:6.1}", db(e[0]), db(e[1]), db(e[2]))
}

fn delta_db(a: [f64; 3], b: [f64; 3]) -> String {
    format!("{:+5.1} {:+5.1} {:+5.1}", db(a[0]) - db(b[0]), db(a[1]) - db(b[1]), db(a[2]) - db(b[2]))
}

/// The club door shuts at 6 s: the response `DELTA` s later against the
/// open one (2 s) and the settled closed one (12 s).
fn door() {
    let mut rig = Rig::new();
    rig.only(&[CLUB]);
    rig.town.set_door(&mut rig.sim, 3, 1.0);
    rig.town.set_door(&mut rig.sim, CLUB_DOOR, 1.0);
    rig.pos = [21.0, 30.2, 1.6];
    let delta: f32 = env("DELTA").and_then(|d| d.parse().ok()).unwrap_or(0.1);
    rig.run(2.0);
    rig.click(CLUB, 1.0);
    rig.run(4.0);
    rig.town.set_door(&mut rig.sim, CLUB_DOOR, 0.0);
    rig.run(delta);
    rig.click(CLUB, 1.0);
    rig.run(12.0 - 6.0 - delta);
    rig.click(CLUB, 1.0);
    rig.run(3.0);
    rig.wav(&format!("door_{}_{delta}", mode()));
    let len = 2.0;
    let (oe, ol) = rig.response(0, len);
    let (de, dl) = rig.response(1, len);
    let (se, sl) = rig.response(2, len);
    println!("door ({}): club door shut at 6 s, clicked {delta} s later; energy dB low/mid/high", mode());
    println!("   open        early {}  late {}", bands_db(oe), bands_db(ol));
    println!("   shut+{delta:<6} early {}  late {}", bands_db(de), bands_db(dl));
    println!("   settled     early {}  late {}", bands_db(se), bands_db(sl));
    println!("   shut+{delta} − settled: early {}  late {}", delta_db(de, se), delta_db(dl, sl));
    println!("   settled − open:     early {}  late {}", delta_db(se, oe), delta_db(sl, ol));
}

/// A source clicks every `PERIOD` s, nothing moves: the spread of the
/// clicks' responses.
fn still() {
    let mut rig = Rig::new();
    let src = env("SRC").unwrap_or_else(|| "club".into());
    let spot = env("SPOT").unwrap_or_else(|| "square".into());
    let i = match src.as_str() {
        "bell" => BELLS,
        "hands" => FEET,
        _ => CLUB,
    };
    rig.only(&[i]);
    rig.pos = match spot.as_str() {
        "field" => [20.0, 60.0, 1.6],
        "hall" => [10.0, 19.0, 1.6],
        "entrance" => [21.0, 30.2, 1.6],
        "north" => [27.0, 48.0, 1.6],
        _ => [15.0, 32.0, 1.6],
    };
    if i == FEET {
        rig.hands = Some(FEET);
    }
    let n: usize = env("N").and_then(|n| n.parse().ok()).unwrap_or(10);
    let period: f32 = env("PERIOD").and_then(|n| n.parse().ok()).unwrap_or(3.0);
    rig.run(3.0);
    for _ in 0..n {
        rig.click(i, 1.0);
        rig.run(period);
    }
    rig.wav(&format!("still_{src}_{spot}_{}", mode()));
    println!("still ({}): {src} at {spot}, {n} clicks every {period} s; dB low/mid/high", mode());
    let mut early = Vec::new();
    let mut late = Vec::new();
    for k in 0..n {
        let (e, l) = rig.response(k, period - 0.1);
        println!("   click {k:2}  early {}  late {}", bands_db(e), bands_db(l));
        early.push(e);
        late.push(l);
    }
    let spread = |v: &[[f64; 3]], b: usize| -> (f32, f32) {
        let d: Vec<f32> = v.iter().map(|e| db(e[b])).collect();
        let m = d.iter().sum::<f32>() / d.len() as f32;
        let sd = (d.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / d.len() as f32).sqrt();
        let span = d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min);
        (sd, span)
    };
    for (name, v) in [("early", &early), ("late", &late)] {
        let s: Vec<String> = (0..3)
            .map(|b| {
                let (sd, span) = spread(v, b);
                format!("sd {sd:4.2} span {span:4.1}")
            })
            .collect();
        println!("   {name:5} spread  low {}  mid {}  high {}", s[0], s[1], s[2]);
    }
}

/// The club clicks every 3 s while every other source in the town clicks
/// around it (CROWD=0: the club alone): the club's own channel energy per
/// click (the mixer's meter), so what the mix-wide budgets take from it
/// shows against the club alone.
fn crowd() {
    let mut rig = Rig::new();
    let crowd = env("CROWD").as_deref() != Some("0");
    let others = [0usize, 1, 3, 4, 5, 6, 7, 14];
    let mut keep = vec![CLUB];
    if crowd {
        keep.extend_from_slice(&others);
    }
    rig.only(&keep);
    let spot = env("SPOT").unwrap_or_else(|| "square".into());
    rig.pos = match spot.as_str() {
        "field" => [20.0, 60.0, 1.6],
        "entrance" => [21.0, 30.2, 1.6],
        "north" => [27.0, 48.0, 1.6],
        "inside" => [27.0, 36.0, 1.6],
        _ => [15.0, 32.0, 1.6],
    };
    rig.watch = rig.ids[CLUB].clone();
    let n: usize = env("N").and_then(|n| n.parse().ok()).unwrap_or(10);
    let period = 3.0f32;
    // the others click every 0.5 s, each at its own offset
    let step = 0.05f32;
    let per = (period / step).round() as usize;
    let mut tick = 0usize;
    let mut round = |rig: &mut Rig, clickit: bool| {
        for k in 0..per {
            if clickit && k == 0 {
                rig.click(CLUB, 1.0);
            }
            if crowd {
                for (j, &o) in others.iter().enumerate() {
                    if (tick + 3 * j) % 10 == 0 {
                        rig.click(o, 1.0);
                    }
                }
            }
            tick += 1;
            rig.run(step);
        }
    };
    round(&mut rig, false);
    let starts: Vec<usize> = (0..n)
        .map(|_| {
            let b = rig.blocks;
            round(&mut rig, true);
            b
        })
        .collect();
    println!("crowd ({}): the club at {spot}, {n} clicks every {period} s, {}; its own channel, dB", mode(), if crowd { "the town clicking around it" } else { "alone" });
    let len = rig.watched.len();
    let mut d = Vec::new();
    for (k, &b) in starts.iter().enumerate() {
        let e = |a: f32, z: f32| -> f64 {
            let (i, j) = (b + (a * SR / BLOCK as f32) as usize, (b + (z * SR / BLOCK as f32) as usize).min(len));
            rig.watched[i..j].iter().sum()
        };
        let (early, late) = (e(0.0, 0.25), e(0.25, period - 0.05));
        println!("   click {k:2}  first 0.25 s {:6.1}  after {:6.1}", db(early), db(late));
        d.push((db(early), db(late)));
    }
    let stats = |v: Vec<f32>| -> (f32, f32) {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt())
    };
    let (me, se) = stats(d.iter().map(|x| x.0).collect());
    let (ml, sl) = stats(d.iter().map(|x| x.1).collect());
    println!("   mean  first {me:6.1} (sd {se:4.2})  after {ml:6.1} (sd {sl:4.2})");
}

/// A clap (click) in the hall, then the listener walks out of the hall's
/// door while it rings: the tail's 10 ms envelope, its steps.
fn walkout() {
    let mut rig = Rig::new();
    rig.only(&[FEET]);
    rig.hands = Some(FEET);
    // the hall (0..14, 14..24); its door to the square at (7, 24)
    let path: Vec<([f32; 3], f32)> = vec![
        ([9.0, 20.0, 1.6], 0.0),
        ([9.0, 20.0, 1.6], 3.2),
        ([7.2, 23.0, 1.6], 5.4),
        ([7.0, 26.0, 1.6], 7.5),
        ([7.0, 30.0, 1.6], 10.0),
    ];
    let walk = move |t: f32| -> [f32; 3] {
        for w in path.windows(2) {
            let ((a, ta), (b, tb)) = (w[0], w[1]);
            if t <= tb {
                let k = ((t - ta) / (tb - ta).max(1e-6)).clamp(0.0, 1.0);
                return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2]];
            }
        }
        path.last().unwrap().0
    };
    rig.walk = Some(Box::new(walk));
    rig.yaw = 1.57;
    rig.run(3.0);
    rig.click(FEET, 8.0);
    rig.run(8.0);
    rig.wav(&format!("walkout_{}", mode()));
    let a = rig.arrival(0);
    let peak = db({
        let e = rig.energy(a - 0.002, a + 0.01);
        e[0] + e[1] + e[2]
    });
    // the 10 ms envelope from 30 ms after the clap until 60 dB down
    let w = 0.01;
    let mut env_db = Vec::new();
    let mut t = a + 0.03;
    while t < a + 7.5 {
        let e = rig.energy(t, t + w);
        env_db.push((t - a, db(e[0] + e[1] + e[2])));
        t += w;
    }
    let end = env_db.iter().position(|x| x.1 < peak - 60.0).unwrap_or(env_db.len());
    let env_db = &env_db[..end];
    // steps against the envelope's own course (median of 9 windows)
    let mut off = Vec::new();
    for k in 4..env_db.len().saturating_sub(4) {
        let mut m: Vec<f32> = env_db[k - 4..=k + 4].iter().map(|x| x.1).collect();
        m.sort_by(|a, b| a.total_cmp(b));
        off.push((env_db[k].0, env_db[k].1 - m[4]));
    }
    let big: Vec<&(f32, f32)> = off.iter().filter(|x| x.1.abs() > 4.0).collect();
    println!("walkout ({}): clap at 3 s in the hall, the listener walks out of its door 3.2–7.5 s", mode());
    println!("   tail: {:.2} s to −60 dB; 10 ms windows more than 4 dB off its course: {}", env_db.len() as f32 * w, big.len());
    let s: Vec<String> = env_db.iter().step_by(10).map(|x| format!("{:.1}s {:.0}", x.0, x.1 - peak)).collect();
    println!("   envelope re peak every 0.1 s: {}", s.join("  "));
    for x in big.iter().take(12) {
        println!("   off its course at {:.2} s: {:+.1} dB", x.0, x.1);
    }
}

/// A car clicks 4× a second driving past at 100 km/h, 300 m away: each
/// click's arrival against the physical one (emitted where the car was).
fn moving() {
    let mut rig = Rig::new();
    rig.only(&[CAR]);
    let v: f32 = env("SPEED").and_then(|v| v.parse().ok()).unwrap_or(27.78);
    // open land west of the town: the listener at (−100, 100), the road
    // x = −400, driving north
    rig.pos = [-100.0, 100.0, 1.6];
    let road = move |t: f32| -> Option<[f32; 3]> { Some([-400.0, -150.0 + v * t, 0.7]) };
    rig.movers.push((CAR, Box::new(road)));
    rig.run(2.0);
    let n = 40;
    for _ in 0..n {
        rig.click(CAR, 1.0);
        rig.run(0.25);
    }
    rig.run(2.0);
    rig.wav(&format!("moving_{}", mode()));
    // the listener is still: a click emitted at t from p arrives at
    // t + |p − listener| / c; each click's loudest sample near it
    let s: Vec<f32> = {
        let b = rig.split();
        b[0].iter().zip(&b[1]).map(|(x, y)| x.iter().chain(y.iter()).map(|v| v * v).sum::<f32>()).collect()
    };
    println!("moving ({}): a car at 100 km/h, 300 m off; click arrival − physical arrival (ms)", mode());
    let mut errs = Vec::new();
    for k in 0..n {
        let want = rig.arrival(k);
        // the onset: the first sample within 10 dB of the window's loudest
        // (a strong late reflection can outdo the direct sound)
        let (a, b) = (((want - 0.12) * SR) as usize, ((want + 0.12) * SR) as usize);
        let w = &s[a..b.min(s.len())];
        let peak = w.iter().fold(0.0f32, |m, &v| m.max(v));
        let i = w.iter().position(|&v| v >= 0.1 * peak).unwrap_or(0);
        errs.push(((a + i) as f32 / SR - want) * 1e3);
    }
    let e: Vec<String> = errs.iter().map(|e| format!("{e:+.0}")).collect();
    println!("   {}", e.join(" "));
    // the direct sound's energy at its physical arrival (±3 ms), per click:
    // a blocked line of sight shows here, not as a timing error
    let d: Vec<String> = (0..n)
        .map(|k| {
            let a = rig.arrival(k);
            let e = rig.energy(a - 0.003, a + 0.003);
            format!("{:.0}", db(e[0] + e[1] + e[2]))
        })
        .collect();
    println!("   direct dB: {}", d.join(" "));
    let worst = errs.iter().fold(0.0f32, |m, e| m.max(e.abs()));
    println!("   worst {worst:.1} ms");
}

/// Energy per ear (L, R) over output times `t0..t1`.
fn ears(rig: &Rig, t0: f32, t1: f32) -> (f64, f64) {
    let (a, b) = ((t0.max(0.0) * SR) as usize, ((t1 * SR) as usize).min(rig.l.len()));
    let l: f64 = rig.l[a..b].iter().map(|v| (v * v) as f64).sum();
    let r: f64 = rig.r[a..b].iter().map(|v| (v * v) as f64).sum();
    (l, r)
}

/// A click from 100 m west; while it is on its way the source jumps 4 m
/// north of the listener and clicks again: the first must still arrive
/// from the west on time, the second from the north.
fn teleport() {
    let mut rig = Rig::new();
    const BALL: usize = 8;
    rig.only(&[BALL]);
    rig.pos = [-100.0, 100.0, 1.6];
    rig.yaw = std::f32::consts::FRAC_PI_2; // facing north: west is on the left
    let (a, b) = ([-200.0f32, 100.0, 1.6], [-100.0f32, 104.0, 1.6]);
    rig.movers.push((BALL, Box::new(move |t: f32| Some(if t < 3.1 { a } else { b }))));
    rig.run(3.0);
    rig.click(BALL, 4.0);
    rig.run(0.2);
    eprintln!("sources after the jump: {}", rig.sim.source_count());
    rig.click(BALL, 4.0);
    rig.run(1.5);
    rig.wav(&format!("teleport_{}", mode()));
    let (t1, t2) = (rig.arrival(0), rig.arrival(1));
    let e = |rig: &Rig, t: f32| ears(rig, t - 0.002, t + 0.004);
    let (l1, r1) = e(&rig, t1);
    let (l2, r2) = e(&rig, t2);
    let (lw, rw) = ears(&rig, t1 - 0.06, t1 - 0.004);
    println!("teleport ({}): a click from 100 m west in flight while the source jumps 4 m north", mode());
    println!("   click 1 (west, due {t1:.3} s): L {:.1} dB  R {:.1} dB  (L−R {:+.1}); before it {:.1} dB", db(l1), db(r1), db(l1) - db(r1), db(lw + rw));
    println!("   click 2 (north, due {t2:.3} s): L {:.1} dB  R {:.1} dB  (L−R {:+.1})", db(l2), db(r2), db(l2) - db(r2));
    let expect = 20.0 * (4.0f32 / 100.0).log10();
    println!("   click 1 − click 2: {:+.1} dB (spreading alone: {expect:+.1} dB)", db(l1 + r1) - db(l2 + r2));
}

/// A click from 100 m, the source switched off 50 ms later: the click is
/// still on its way and must arrive whole.
fn ringout() {
    let mut rig = Rig::new();
    const BALL: usize = 8;
    rig.only(&[BALL]);
    rig.pos = [-100.0, 100.0, 1.6];
    let a = [-200.0f32, 100.0, 1.6];
    let off: f32 = env("OFF").and_then(|v| v.parse().ok()).unwrap_or(3.05);
    rig.movers.push((BALL, Box::new(move |t: f32| (t < off).then_some(a))));
    rig.run(1.0);
    rig.click(BALL, 4.0);
    rig.run(2.0);
    rig.click(BALL, 4.0);
    rig.run(2.0);
    rig.wav(&format!("ringout_{}", mode()));
    let (t1, t2) = (rig.arrival(0), rig.arrival(1));
    let e1 = rig.energy(t1 - 0.002, t1 + 0.004);
    let e2 = rig.energy(t2 - 0.002, t2 + 0.004);
    let (l1, l2) = (rig.energy(t1 + 0.004, t1 + 1.0), rig.energy(t2 + 0.004, t2 + 1.0));
    println!("ringout ({}): a click from 100 m; the source switched off {:.2} s after the second", mode(), off - 3.0);
    println!("   direct: on {:.1} dB, switched off in flight {:.1} dB", db(e1[0] + e1[1] + e1[2]), db(e2[0] + e2[1] + e2[2]));
    println!("   after it (1 s): on {:.1} dB, switched off {:.1} dB", db(l1[0] + l1[1] + l1[2]), db(l2[0] + l2[1] + l2[2]));
}

fn main() {
    let scene = std::env::args().nth(1).unwrap_or_else(|| "door".into());
    eprintln!(
        "assess {scene}: mode {} rung {} quality {} pace {}",
        mode(),
        env("RUNG").unwrap_or("gov".into()),
        env("QUALITY").unwrap_or("2".into()),
        env("PACE").unwrap_or("1".into())
    );
    match scene.as_str() {
        "door" => door(),
        "still" => still(),
        "walkout" => walkout(),
        "moving" => moving(),
        "teleport" => teleport(),
        "ringout" => ringout(),
        "crowd" => crowd(),
        s => panic!("unknown scene {s}"),
    }
}
