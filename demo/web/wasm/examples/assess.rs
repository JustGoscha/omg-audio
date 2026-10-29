//! The engine measured with IMPULSES: single-sample clicks played on the
//! demo town's sources through the whole engine (simulation + mixer + the
//! render governor), the output split into the tracer's three bands
//! (Linkwitz–Riley; before Oct 29's band change, one-pole) and
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
//!            hands, SPOT=square|field|hall|entrance|north|inside|
//!            cathedral)
//!   walkout  a clap (click) in the hall, then the listener walks out of
//!            the hall's door while it rings: the tail's envelope steps
//!   moving   a car clicks 4 times a second driving past at 100 km/h,
//!            300 m away: each click's arrival against when it physically
//!            arrives (emitted where the car was)
//!   crowd    the club clicks every 3 s while the town's other sources
//!            click around it (CROWD=0: alone): the club's own channel
//!            per click (SPOT=square|north|entrance|inside|field)
//!
//!   fly      the listener flies (8 m/s) through the cathedral's wall to
//!            the flute, clicking every 50 ms: pops (periods off course)
//!   cartone  a car passes 30 m off at 100 km/h playing a steady tone:
//!            its pitch and level, demodulated, and how they jitter
//!   doorswing the club plays a steady tone; its door swings shut at 4 s
//!            as the Godot demo eases it (~1 s): the level through the
//!            closing door — dips under the shut level, rises against
//!            the closing, how near the shut level it lands
//!
//!     MODE=v3|v4|v5|v6  RUNG=0..4 (unset: governed)  QUALITY=0..3 | LEVEL=0..10  EAR=0
//!     PACE=0  WAV=dir (write the output)  SEED=n  NEE=n (next-event
//!     connections a bounce; 0: every source)  PHYS=hz (movers move at a
//!     game's physics rate, sent every frame)
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

/// The tracer's bands, split by Linkwitz–Riley crossovers (4th order)
/// at 314 Hz and 2.1 kHz — what each band holds, without one band's skirt
/// over the next. Kept here, not taken from the engine, so engines of any
/// age are read alike.
struct Bands {
    lp1: [Biq; 2],
    hp1: [Biq; 2],
    lp2: [Biq; 2],
    hp2: [Biq; 2],
    ap2: Biq,
}

#[derive(Clone, Copy)]
struct Biq {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Biq {
    /// Butterworth low-pass (0), high-pass (1) or all-pass (2).
    fn new(kind: u8, fc: f32) -> Self {
        let w = std::f64::consts::TAU * fc as f64 / SR as f64;
        let (cw, alpha) = (w.cos(), w.sin() / std::f64::consts::SQRT_2);
        let a0 = 1.0 + alpha;
        let b = match kind {
            0 => [(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0],
            1 => [(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0],
            _ => [1.0 - alpha, -2.0 * cw, 1.0 + alpha],
        };
        Self { b: b.map(|v| v / a0), a: [-2.0 * cw / a0, (1.0 - alpha) / a0], z: [0.0; 2] }
    }
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

impl Bands {
    fn new() -> Self {
        Self {
            lp1: [Biq::new(0, 314.0); 2],
            hp1: [Biq::new(1, 314.0); 2],
            lp2: [Biq::new(0, 2100.0); 2],
            hp2: [Biq::new(1, 2100.0); 2],
            ap2: Biq::new(2, 2100.0),
        }
    }
    fn split(&mut self, x: f32) -> [f32; 3] {
        let two = |q: &mut [Biq; 2], x: f64| {
            let y = q[0].tick(x);
            q[1].tick(y)
        };
        let x = x as f64;
        let l = two(&mut self.lp1, x);
        let lo = self.ap2.tick(l);
        let r = two(&mut self.hp1, x);
        let mid = two(&mut self.lp2, r);
        let hi = two(&mut self.hp2, r);
        [lo as f32, mid as f32, hi as f32]
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
    /// Each mover's last place sent and when it was first sent there.
    placed: Vec<Option<([f32; 3], f64)>>,
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
        // LEVEL=n: a rung of the quality ladder (QUALITY=0..3: the tiers)
        let quality = match (env("LEVEL").and_then(|l| l.parse::<usize>().ok()), env("QUALITY").as_deref()) {
            (Some(l), _) => Quality::Level(l),
            (_, Some("0")) => Quality::Low,
            (_, Some("1")) => Quality::Medium,
            (_, Some("3")) => Quality::Ultra,
            _ => Quality::High,
        };
        // LATE=n: late rays a tick over the quality's
        let quality = match env("LATE").and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Quality::Custom(omg_audio::Budget { late_rays: n, ..quality.budget(false) }),
            None => quality,
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
            // COMPUTE=gpu: the walk on the GPU, as the Godot host runs it
            compute: if env("COMPUTE").as_deref() == Some("gpu") { Compute::Gpu } else { Compute::Cpu },
            v4: mode == "v4" || mode == "v5",
            v5: mode == "v5",
            v6: mode == "v6",
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
        // GEOMETRY=<file>: a host's level instead of the town's (the
        // Godot demo's: OMG_DUMP_GEOMETRY), the town's sources and doors
        // where they are
        if let Some(path) = env("GEOMETRY") {
            sim.set_geometry(omg_audio::Geometry::load(&path).expect("GEOMETRY: a dumped geometry"));
            eprintln!("geometry: {path}");
        }
        if env("COMPUTE").as_deref() == Some("gpu") {
            eprintln!("gpu walk: {}", if sim.gpu_active() { "on" } else { "unavailable (build with --features omg-audio/gpu)" });
        }
        let hrir = if env("HRIR").as_deref() == Some("0") { Hrir::none() } else { Hrir::builtin() };
        let mut mix = Mixer::with_capacity(SR, hrir, 64);
        if env("EAR").as_deref() == Some("0") {
            mix.set_ear(false);
        }
        let mut gov = Governor::new(SR);
        gov.apply(&mut mix);
        if let Some(r) = env("RUNG").and_then(|r| r.parse().ok()) {
            gov.pin(&mut mix, Some(r));
            // single levers on the pinned rung: TAPS=n (mix-wide paths),
            // CONV=n (convolved sources)
            let lad = omg_audio::governor::LADDER[r];
            if let Some(t) = env("TAPS").and_then(|t| t.parse().ok()) {
                mix.set_budget(t, lad.1, lad.2);
            }
            if let Some(c) = env("CONV").and_then(|c| c.parse().ok()) {
                mix.set_convolution_budget(c);
            }
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
            placed: Vec::new(),
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
            // PHYS=hz: movers move at a game's physics rate, sent every
            // frame (the Godot demo: 60 Hz physics, 120 Hz frames)
            // — stamped as the engine stamps them (a repeated place with
            // its first send; STAMP=each: every send now, as before Oct 29)
            let tm = env("PHYS").and_then(|v| v.parse::<f32>().ok()).map_or(t, |hz| (t * hz).floor() / hz);
            self.placed.resize(self.movers.len(), None);
            for k in 0..self.movers.len() {
                let (i, p) = (self.movers[k].0, (self.movers[k].1)(tm));
                let stamp = match (self.placed[k], p) {
                    (Some((q, t0)), Some(p)) if q == p && env("STAMP").as_deref() != Some("each") => t0,
                    _ => t as f64,
                };
                self.placed[k] = p.map(|p| (p, stamp));
                self.sim.set_time(stamp);
                self.town.set_dynamic(&mut self.sim, i - base, p.map(|p| (p, 1.0)));
                self.sim.set_time(t as f64);
            }
            if let Some(h) = self.hands {
                let p = [self.pos[0] + 0.3 * self.yaw.cos(), self.pos[1] + 0.3 * self.yaw.sin(), 1.2];
                self.town.set_dynamic(&mut self.sim, h - base, Some((p, 1.0)));
            }
            self.sim.set_listener_at(self.pos, self.yaw, 0.0, 0.0, t as f64);
            // SWITCH=t:level — the quality ladder's rung changes at t (as
            // quality auto changes it while playing)
            if let Some((at, level)) = env("SWITCH").and_then(|v| v.split_once(':').and_then(|(a, l)| Some((a.parse::<f32>().ok()?, l.parse::<usize>().ok()?)))) {
                if t >= at && t - BLOCK as f32 / SR < at {
                    let cfg = Config { quality: Quality::Level(level), ..*self.sim.config() };
                    self.sim.set_config(cfg);
                    eprintln!("t {t:.3}: quality → rung {level}");
                }
            }
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

    /// How impulsive the output is over `t0..t1` (s), per band: the
    /// kurtosis E[x⁴]/E[x²]² in 20 ms windows (both ears), weighted by
    /// each window's energy. A dense reverberant tail is noise-like (≈ 3);
    /// isolated clicks over a quieter field read far higher — heard as
    /// crackle.
    fn kurtosis(&mut self, t0: f32, t1: f32) -> [f64; 3] {
        let w = (0.02 * SR) as usize;
        let a = (t0.max(0.0) * SR) as usize;
        let s = self.split();
        let b = ((t1 * SR) as usize).min(s[0].len());
        let (mut num, mut den) = ([0.0f64; 3], [0.0f64; 3]);
        let mut i = a;
        while i + w <= b {
            for k in 0..3 {
                let (mut m2, mut m4) = (0.0f64, 0.0f64);
                for ear in 0..2 {
                    for x in &s[ear][i..i + w] {
                        let v = (x[k] as f64).powi(2);
                        m2 += v;
                        m4 += v * v;
                    }
                }
                let n = (2 * w) as f64;
                if m2 > 0.0 {
                    let kurt = (m4 / n) / (m2 / n).powi(2);
                    num[k] += kurt * m2;
                    den[k] += m2;
                }
            }
            i += w;
        }
        core::array::from_fn(|k| if den[k] > 0.0 { num[k] / den[k] } else { 0.0 })
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
        "inside" => [27.0, 30.0, 1.6],
        "cathedral" => [8.0, 60.0, 1.6],
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
    let mut kurt = [0.0f64; 3];
    for k in 0..n {
        let (e, l) = rig.response(k, period - 0.1);
        println!("   click {k:2}  early {}  late {}", bands_db(e), bands_db(l));
        early.push(e);
        late.push(l);
        // the tail's grain: from the end of the early part over the next
        // 0.5 s (the reverb, before it sinks under the floor)
        let a = rig.arrival(k);
        let ku = rig.kurtosis(a + EARLY, a + EARLY + 0.5);
        for b in 0..3 {
            kurt[b] += ku[b] / n as f64;
        }
    }
    println!("   tail kurtosis (noise ≈ 3)  low {:5.1}  mid {:5.1}  high {:5.1}", kurt[0], kurt[1], kurt[2]);
    let spread = |v: &[[f64; 3]], b: usize| -> (f32, f32) {
        let d: Vec<f32> = v.iter().map(|e| db(e[b])).collect();
        let m = d.iter().sum::<f32>() / d.len() as f32;
        let sd = (d.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / d.len() as f32).sqrt();
        let span = d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min);
        (sd, span)
    };
    if env("KERNEL_LAG").is_some() {
        // the club's current estimate against what its kernels play
        let (mut now_e, mut ker_e) = ([0.0f64; 3], [0.0f64; 3]);
        for &id in &rig.ids[i] {
            for (dst, d) in [(&mut now_e, rig.sim.dense_response(id)), (&mut ker_e, rig.sim.kernel_response(id))] {
                if let Some(d) = d {
                    for x in &d.bins {
                        for b in 0..3 {
                            dst[b] += x[b] as f64;
                        }
                    }
                }
            }
        }
        println!("   estimate now {}  kernel's {}", bands_db(now_e), bands_db(ker_e));
    }
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

/// The listener flies (8 m/s, Godot's fly) from the field north through
/// the cathedral's south wall to 4 m from the flute, which clicks every
/// 50 ms (a click train: the flute, as impulses), then hovers. Per click
/// period, the output's level (both ears, all bands); a pop is a period
/// off the course of its neighbours (their median over five) — counted
/// outside, crossing the wall, and inside. (SPEED=m/s, X=the flight's x.)
fn fly() {
    let mut rig = Rig::new();
    let flute = 3usize;
    rig.only(&[flute]);
    let speed: f32 = env("SPEED").and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let x: f32 = env("X").and_then(|v| v.parse().ok()).unwrap_or(4.0);
    let (y0, y1, t0) = (40.0f32, 62.0f32, 3.0f32);
    rig.pos = [x, y0, 2.0];
    rig.walk = Some(Box::new(move |t: f32| [x, (y0 + (t - t0).max(0.0) * speed).min(y1), 2.0]));
    let period = 0.05f32;
    let total = t0 + (y1 - y0) / speed + 3.0;
    let n = (total / period) as usize;
    if env("SIGNAL").as_deref() == Some("tone") {
        // a steady 500 Hz tone instead of clicks: whatever the output holds
        // above 3 kHz, the engine made
        let f0 = 500.0f32;
        let cycle = (SR / f0).round() as usize;
        let tone: Vec<f32> = (0..cycle * 10).map(|k| 0.5 * (std::f32::consts::TAU * (k % cycle) as f32 / cycle as f32).sin()).collect();
        let tone = clip(&tone);
        for &id in &rig.ids[flute].clone() {
            rig.mix.play(id, &tone, Play::LOOP);
        }
        rig.run(total);
        rig.wav(&format!("fly_tone_{}", mode()));
        let cross = t0 + (52.0 - y0) / speed;
        println!("fly ({}): through the cathedral wall at {speed} m/s, the flute playing a steady 500 Hz; artifacts = output above 3 kHz, under the tone", mode());
        println!("   {}", artifact_line("outside", &artifacts(&rig, 1.0, cross - 0.15)));
        println!("   {}", artifact_line("crossing", &artifacts(&rig, cross - 0.15, cross + 0.4)));
        println!("   {}", artifact_line("inside", &artifacts(&rig, cross + 0.4, total)));
        if env("TRACE").is_some() {
            for (k, v) in artifacts(&rig, 1.0, total).iter().enumerate() {
                if *v > -40.0 {
                    println!("   t {:6.3}  {:6.1} dB", 1.0 + k as f32 * 0.005, v);
                }
            }
        }
        return;
    }
    for _ in 0..n {
        rig.click(flute, 1.0);
        rig.run(period);
    }
    rig.wav(&format!("fly_{}", mode()));
    let at_y = |t: f32| (y0 + (t - t0).max(0.0) * speed).min(y1);
    let lv: Vec<(f32, f32, f32)> = (0..n.saturating_sub(1))
        .map(|k| {
            let a = rig.arrival(k);
            let e = rig.energy(a, a + period);
            (a, at_y(a), db(e[0] + e[1] + e[2]))
        })
        .collect();
    let mut res = Vec::new();
    for k in 2..lv.len().saturating_sub(2) {
        let mut w: Vec<f32> = lv[k - 2..=k + 2].iter().map(|v| v.2).collect();
        w.sort_by(|a, b| a.total_cmp(b));
        res.push((lv[k].0, lv[k].1, lv[k].2, lv[k].2 - w[2]));
    }
    println!("fly ({}): through the cathedral wall at {speed} m/s, the flute clicking every 50 ms; pops = periods off their neighbours' median", mode());
    let wall = 52.0f32;
    let phases: [(&str, Box<dyn Fn(f32) -> bool>); 3] = [
        ("outside", Box::new(move |y| y < wall - 1.0)),
        ("crossing", Box::new(move |y| (wall - 1.0..wall + 1.5).contains(&y))),
        ("inside", Box::new(move |y| y >= wall + 1.5)),
    ];
    for (name, f) in phases.iter() {
        let r: Vec<f32> = res.iter().filter(|v| f(v.1)).map(|v| v.3.abs()).collect();
        if r.is_empty() {
            continue;
        }
        let over = |d: f32| r.iter().filter(|&&x| x > d).count();
        let mx = r.iter().cloned().fold(0.0f32, f32::max);
        println!("   {name:8} {:3} periods: off course >2 dB {:3}, >4 dB {:3}, worst {mx:4.1} dB", r.len(), over(2.0), over(4.0));
    }
    if env("TRACE").is_some() {
        for v in &res {
            println!("   t {:6.2} y {:5.1}  {:6.1} dB  {:+5.1}", v.0, v.1, v.2, v.3);
        }
    }
}

/// What a pure tone came out with besides itself: the output above
/// 3 kHz (4th-order high-pass) against the whole, in 5 ms windows from
/// `t0` to `t1` (s) — a 500 Hz tone through a linear engine has nothing
/// there but what the engine made: pops, clicks, crackle. Returns (the
/// windows' artifact levels in dB under the tone).
fn artifacts(rig: &Rig, t0: f32, t1: f32) -> Vec<f32> {
    let (mut hl1, mut hl2, mut hr1, mut hr2) = (Biq::new(1, 3000.0), Biq::new(1, 3000.0), Biq::new(1, 3000.0), Biq::new(1, 3000.0));
    let w = (0.005 * SR) as usize;
    let (a, b) = ((t0 * SR) as usize, ((t1 * SR) as usize).min(rig.l.len()));
    let mut out = Vec::new();
    let (mut eh, mut et, mut k) = (0.0f64, 0.0f64, 0usize);
    for i in 0..b {
        let (l, r) = (rig.l[i] as f64, rig.r[i] as f64);
        let hl = {
            let y = hl1.tick(l);
            hl2.tick(y)
        };
        let hr = {
            let y = hr1.tick(r);
            hr2.tick(y)
        };
        if i < a {
            continue;
        }
        eh += hl * hl + hr * hr;
        et += l * l + r * r;
        k += 1;
        if k == w {
            // a silent window (the source not yet in the audible window)
            // holds no artifact: NaN, left out of the counts
            out.push(if et < 1e-20 { f32::NAN } else { (10.0 * (eh.max(1e-30) / et).log10()) as f32 });
            eh = 0.0;
            et = 0.0;
            k = 0;
        }
    }
    out
}

fn artifact_line(name: &str, v: &[f32]) -> String {
    let silent = v.iter().filter(|x| x.is_nan()).count();
    let v: Vec<f32> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return format!("{name}: — ({silent} silent windows)");
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let over = |d: f32| v.iter().filter(|&&x| x > d).count();
    format!(
        "{name:8} {:4} windows{}: median {:6.1} dB, worst {:6.1} dB, over −40 dB {:3}, over −30 dB {:3}",
        v.len(),
        if silent > 0 { format!(" (+{silent} silent)") } else { String::new() },
        s[s.len() / 2],
        s[s.len() - 1],
        over(-40.0),
        over(-30.0)
    )
}

/// A car passes 30 m off (ROAD=x of the road) at 100 km/h playing a steady
/// 500 Hz tone (a sweep at one frequency). The output demodulated at the
/// tone: its pitch and level, millisecond by millisecond, against their
/// own 25 ms course — what moves faster than the motion can is jitter.
/// (The ground's reflection beats with the direct sound — real, and the
/// same in every mode: compare modes, not against zero.)
fn cartone() {
    let mut rig = Rig::new();
    rig.only(&[CAR]);
    let v: f32 = env("SPEED").and_then(|v| v.parse().ok()).unwrap_or(27.78);
    let road: f32 = env("ROAD").and_then(|v| v.parse().ok()).unwrap_or(-130.0);
    // LIS=x,y,z: the listener elsewhere; Y0: where the car starts (the
    // Godot demo's car scenario: ROAD=-17.1 LIS=-13.1,0,2 Y0=-150)
    rig.pos = env("LIS")
        .and_then(|v| {
            let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 3).then(|| [p[0], p[1], p[2]])
        })
        .unwrap_or([-100.0, 100.0, 1.6]);
    let y0: f32 = env("Y0").and_then(|v| v.parse().ok()).unwrap_or(-50.0);
    let at = move |t: f32| -> Option<[f32; 3]> { Some([road, y0 + v * t, 0.7]) };
    rig.movers.push((CAR, Box::new(at)));
    let f0 = 500.0f32;
    let cycle = (SR / f0).round() as usize;
    let tone: Vec<f32> = (0..cycle * 10).map(|k| 0.5 * (std::f32::consts::TAU * (k % cycle) as f32 / cycle as f32).sin()).collect();
    let tone = clip(&tone);
    for &id in &rig.ids[CAR].clone() {
        rig.mix.play(id, &tone, Play::LOOP);
    }
    let total = 300.0 / v + 1.0;
    rig.run(total);
    rig.wav(&format!("cartone_{}", mode()));
    // quadrature demodulation at f0, two one-pole low-passes (60 Hz)
    let c = 1.0 - (-std::f32::consts::TAU * 60.0 / SR).exp();
    let w0 = std::f64::consts::TAU * f0 as f64 / SR as f64;
    let (mut i1, mut q1, mut i2, mut q2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut ph = Vec::new();
    let mut amp = Vec::new();
    for (n, (&l, &r)) in rig.l.iter().zip(&rig.r).enumerate() {
        let y = 0.5 * (l + r);
        let (cs, sn) = ((w0 * n as f64).cos() as f32, (w0 * n as f64).sin() as f32);
        i1 += c * (y * cs - i1);
        q1 += c * (-y * sn - q1);
        i2 += c * (i1 - i2);
        q2 += c * (q1 - q2);
        if n % 48 == 0 {
            ph.push(q2.atan2(i2) as f64);
            amp.push(10.0 * (i2 * i2 + q2 * q2).max(1e-30).log10());
        }
    }
    // unwrap → frequency per ms
    let mut f = Vec::new();
    let mut acc = 0.0f64;
    for k in 1..ph.len() {
        let mut d = ph[k] - ph[k - 1];
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        acc += d;
        f.push(f0 as f64 + d * 1000.0 / std::f64::consts::TAU);
    }
    let _ = acc;
    let course = |x: &[f64], k: usize| -> f64 {
        let (a, b) = (k.saturating_sub(12), (k + 13).min(x.len()));
        x[a..b].iter().sum::<f64>() / (b - a) as f64
    };
    // the pass: 1 s in until the end, while the tone is loud enough to read
    let (a, b) = (1000usize, f.len().saturating_sub(200));
    let amp64: Vec<f64> = amp.iter().map(|&v| v as f64).collect();
    let (mut fj, mut aj, mut n) = (0.0f64, 0.0f64, 0.0f64);
    let mut jumps = 0;
    for k in a..b {
        let df = f[k] - course(&f, k);
        let da = amp64[k + 1] - course(&amp64, k + 1);
        fj += df * df;
        aj += da * da;
        n += 1.0;
        if da.abs() > 2.0 {
            jumps += 1;
            if env("TRACE").is_some() {
                println!("   t {:7.3}  level {:+5.1} dB  pitch {:+6.2} Hz", (k + 1) as f32 * 48.0 / SR, da, df);
            }
        }
    }
    println!(
        "cartone ({}): a car {:.0} m off at {:.0} km/h playing 500 Hz; jitter against the 25 ms course: pitch {:.2} Hz rms, level {:.2} dB rms, level off by >2 dB in {} of {} ms",
        mode(),
        (road + 100.0).abs(),
        v * 3.6,
        (fj / n).sqrt(),
        (aj / n).sqrt(),
        jumps,
        n as usize
    );
    println!("   artifacts (output above 3 kHz, under the tone): {}", artifact_line("pass", &artifacts(&rig, 1.0, total - 0.2)));
    if env("TRACE").is_some() {
        for (k, v) in artifacts(&rig, 1.0, total - 0.2).iter().enumerate() {
            if *v > -40.0 {
                println!("   art t {:6.3}  {:6.1} dB", 1.0 + k as f32 * 0.005, v);
            }
        }
    }
}

/// The club plays a steady 500 Hz tone; its door, open, swings shut at
/// 4 s as the Godot demo eases a leaf (openness × e^(−7 t), there below
/// 0.001), moved every frame. Narrowing the opening only takes paths
/// away: the level falls to the shut level and should neither dip under
/// it nor climb back on the way.
fn doorswing() {
    let mut rig = Rig::new();
    rig.only(&[CLUB]);
    rig.town.set_door(&mut rig.sim, 3, 1.0);
    // REF=1: the door shut all along (the shut level, converged)
    let reference = env("REF").is_some();
    rig.town.set_door(&mut rig.sim, CLUB_DOOR, if reference { 0.0 } else { 1.0 });
    // LIS=x,y,z: elsewhere (the Godot demo's spot, 1 m before the door:
    // LIS=21,31,1.6 — a speaker in sight through the doorway)
    rig.pos = env("LIS")
        .and_then(|v| {
            let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 3).then(|| [p[0], p[1], p[2]])
        })
        .unwrap_or([21.0, 30.2, 1.6]);
    let f0 = 500.0f32;
    let cycle = (SR / f0).round() as usize;
    let tone: Vec<f32> = (0..cycle * 10).map(|k| 0.5 * (std::f32::consts::TAU * (k % cycle) as f32 / cycle as f32).sin()).collect();
    let tone = clip(&tone);
    for &id in &rig.ids[CLUB].clone() {
        rig.mix.play(id, &tone, Play::LOOP);
    }
    let (shut_at, total) = (4.0f32, 12.0f32);
    rig.run(shut_at);
    let mut open = if reference { 0.0f32 } else { 1.0 };
    let frame = BLOCK as f32 / SR;
    while rig.t() < total {
        open *= (-7.0 * frame).exp();
        if open < 0.001 {
            open = 0.0;
        }
        rig.town.set_door(&mut rig.sim, CLUB_DOOR, open);
        rig.run(frame);
    }
    rig.wav(&format!("doorswing_{}", mode()));
    // the level per 10 ms
    let w = (0.01 * SR) as usize;
    let lev: Vec<f32> = rig
        .l
        .chunks(w)
        .zip(rig.r.chunks(w))
        .map(|(a, b)| {
            let e: f32 = a.iter().zip(b).map(|(x, y)| 0.5 * (x * x + y * y)).sum::<f32>() / a.len() as f32;
            10.0 * e.max(1e-20).log10()
        })
        .collect();
    let at = |t: f32| lev[((t / 0.01) as usize).min(lev.len() - 1)];
    let mean = |a: f32, b: f32| {
        let v = &lev[(a / 0.01) as usize..(b / 0.01) as usize];
        10.0 * (v.iter().map(|d| 10f32.powf(d / 10.0)).sum::<f32>() / v.len() as f32).log10()
    };
    let (open_db, shut_db) = (mean(3.0, 4.0), mean(10.0, 12.0));
    let swing = ((shut_at / 0.01) as usize, ((shut_at + 1.2) / 0.01) as usize);
    let (mut lowest, mut rise) = (f32::MAX, 0.0f32);
    for k in swing.0..swing.1 {
        lowest = lowest.min(lev[k]);
        rise += (lev[k] - lev[k - 1]).max(0.0);
    }
    println!("doorswing ({}): the club's tone through its door swinging shut at {shut_at} s (~1 s)", mode());
    println!("   open {open_db:.1} dB, shut {shut_db:.1} dB (10–12 s)");
    println!("   during the swing: lowest {:+.1} dB from shut, climbing back {rise:.1} dB in all (10 ms steps)", lowest - shut_db);
    println!(
        "   after: +0.1 s {:+.1}  +0.3 s {:+.1}  +1 s {:+.1}  +2 s {:+.1}  +4 s {:+.1} dB from shut",
        at(shut_at + 1.3) - shut_db,
        at(shut_at + 1.5) - shut_db,
        at(shut_at + 2.2) - shut_db,
        at(shut_at + 3.2) - shut_db,
        at(shut_at + 5.2) - shut_db
    );
    let course: Vec<String> = (0..=60).map(|k| format!("{:.0}", at(shut_at - 0.1 + k as f32 * 0.05))).collect();
    println!("   level every 50 ms from {:.1} s: {}", shut_at - 0.1, course.join(" "));
    println!("   artifacts (output above 3 kHz, under the tone): {}", artifact_line("swing", &artifacts(&rig, shut_at, shut_at + 1.2)));
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
        "fly" => fly(),
        "cartone" => cartone(),
        "doorswing" => doorswing(),
        s => panic!("unknown scene {s}"),
    }
}
