//! The Godot host around omg-audio's `threaded::Engine`.
//!
//!  - The ENGINE (game-thread handle) and its simulation thread come from
//!    omg-audio unchanged.
//!  - Its `AudioOut` lives on an OUTPUT thread here, rendering 256-frame
//!    blocks ahead into a small ring (≈ 21 ms) — the outdoor beds
//!    (ambience, rain) included, in the mixer; Godot's audio thread only
//!    copies from the ring
//!    (nodes.rs: OmgOutputStream). Rendering on Godot's mix callback
//!    itself would put the whole mixer on a thread that has other
//!    deadlines; the ring isolates them for one block of latency.
//!  - A load governor walks the mixer's mix-wide budgets like the web
//!    worklet's applyBudget: shed at once when a render window runs over
//!    budget, earn density back after calm windows.
//!
//! Without `start_output` the owner renders synchronously (`render`):
//! tests and offline renders.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use omg_audio::threaded::{AudioOut, Engine, EngineConfig};
use omg_audio::governor::{Governor, LADDER};
use omg_audio::{Compute, Config, Hrir, Quality};


const BLOCK: usize = 256;


/// Commands for the output thread's own state (beds, budgets).
pub enum OutCmd {
    /// A levelled ambience bed (`Host::set_ambience` levels it first).
    Ambience(Vec<f32>, bool),
    AmbienceGain(f32),
    RainBank(Vec<f32>),
    Rain(f32),
    RainGain(f32),
    /// Pin a ladder rung (None = governor).
    Budget(Option<usize>),
    /// The ear on or off (`Mixer::set_ear`).
    Ear(bool),
}

#[derive(Default)]
struct Shared {
    ring: Mutex<VecDeque<(f32, f32)>>,
    ring_target: AtomicUsize,
    /// The largest mix callback seen (frames): the ring keeps three.
    read_max: AtomicUsize,
    underruns: AtomicUsize,
    stop: AtomicBool,
    // stats published by whoever renders
    load: AtomicU32,
    rung: AtomicUsize,
    rendering: AtomicUsize,
    paths: AtomicUsize,
    peak: AtomicU32,
    enclosure: AtomicU32,
    rt60: AtomicU32,
    /// Levels for a mixer view, ~10 times a second: (source id, peak,
    /// RMS) per source, the output (ambience and rain included) as
    /// `u32::MAX`.
    meters: Mutex<Vec<(u32, f32, f32)>>,
}

/// The consumer end of the output ring (Godot's audio thread).
#[derive(Clone)]
pub struct OutputReader {
    sh: Arc<Shared>,
    out: Arc<Mutex<Out>>,
}

impl OutputReader {
    /// Frames the output thread hadn't rendered when they were taken
    /// (rendered on the taker's thread instead), so far.
    pub fn late(&self) -> usize {
        self.sh.underruns.load(Ordering::Relaxed)
    }

    /// Fill `out` from the ring. When the ring runs short (a mix callback
    /// bigger than the ring, a stalled output thread) the rest is rendered
    /// right here instead of played as silence, and counted as `late`.
    pub fn read(&self, out: &mut [(f32, f32)]) {
        // a host whose callbacks pull more than the ring holds (the web's
        // audio worklet bridge asks for big chunks) would find it short on
        // every call: keep three callbacks ahead (two left no slack for a
        // busy machine: the output thread naps 1.5 ms between blocks)
        if out.len() > self.sh.read_max.fetch_max(out.len(), Ordering::Relaxed) {
            let want = (3 * out.len()).div_ceil(BLOCK) * BLOCK;
            self.sh.ring_target.fetch_max(want, Ordering::Relaxed);
        }
        let take = |out: &mut [(f32, f32)]| {
            let mut ring = self.sh.ring.lock().unwrap();
            let n = out.len().min(ring.len());
            for (o, f) in out.iter_mut().zip(ring.drain(..n)) {
                *o = f;
            }
            n
        };
        let n = take(out);
        if n < out.len() {
            // the producer pushes under the renderer's lock: holding it,
            // the ring is complete up to the mixer's clock
            let mut o = self.out.lock().unwrap();
            let n = n + take(&mut out[n..]);
            if n < out.len() {
                self.sh.underruns.fetch_add(out.len() - n, Ordering::Relaxed);
                o.render(&self.sh, &mut out[n..]);
            }
        }
    }
}

/// Everything the renderer owns.
struct Out {
    audio: AudioOut,
    rx: Receiver<OutCmd>,
    sr: f32,
    frame_seen: u64,
    l: Vec<f32>,
    r: Vec<f32>,
    gov: Governor,
    // meters
    meter_frames: usize,
    out_peak: f32,
    out_sum2: f64,
    meter_scratch: Vec<(omg_audio::SourceId, omg_audio::Meter)>,
}

impl Out {
    fn render(&mut self, sh: &Shared, out: &mut [(f32, f32)]) {
        let t0 = Instant::now();
        while let Ok(c) = self.rx.try_recv() {
            match c {
                OutCmd::Ambience(d, st) => self.audio.mixer().set_ambience_levelled(d, st),
                OutCmd::AmbienceGain(g) => self.audio.mixer().set_ambience_gain(g),
                OutCmd::RainBank(b) => self.audio.mixer().set_rain_bank(b),
                OutCmd::Rain(x) => self.audio.mixer().set_rain(x),
                OutCmd::RainGain(g) => self.audio.mixer().set_rain_gain(g),
                OutCmd::Budget(b) => {
                    self.gov.pin(self.audio.mixer(), b);
                    sh.rung.store(self.gov.rung(), Ordering::Relaxed);
                }
                OutCmd::Ear(on) => self.audio.mixer().set_ear(on),
            }
        }
        let n = out.len();
        self.l.resize(n, 0.0);
        self.r.resize(n, 0.0);
        self.audio.render(&mut self.l, &mut self.r);
        if let Some(f) = self.audio.frame() {
            if f.version != self.frame_seen {
                self.frame_seen = f.version;
                sh.enclosure.store(f.env.enclosure.to_bits(), Ordering::Relaxed);
                // the room the listener hears: the loudest field's decay
                let mut best = (0.0f32, 0.0f32);
                for (_, b) in &f.blocks {
                    let lvl = b.taps.iter().map(|t| t.gains[1]).sum::<f32>() + b.reverb.level[1];
                    if lvl > best.0 {
                        best = (lvl, b.reverb.rt60[1]);
                    }
                }
                sh.rt60.store(best.1.to_bits(), Ordering::Relaxed);
            }
        }
        let mut peak = 0.0f32;
        let mut sum2 = 0.0f32;
        for (o, (a, b)) in out.iter_mut().zip(self.l.iter().zip(&self.r)) {
            let (a, b) = (a.clamp(-1.0, 1.0), b.clamp(-1.0, 1.0));
            peak = peak.max(a.abs()).max(b.abs());
            sum2 += 0.5 * (a * a + b * b);
            *o = (a, b);
        }
        sh.peak.store(peak.to_bits(), Ordering::Relaxed);
        self.meter(sh, peak, sum2, n);
        self.govern(sh, t0.elapsed().as_secs_f64(), n);
    }

    fn meter(&mut self, sh: &Shared, peak: f32, sum2: f32, frames: usize) {
        self.out_peak = self.out_peak.max(peak);
        self.out_sum2 += sum2 as f64;
        self.meter_frames += frames;
        if (self.meter_frames as f32) < self.sr / 10.0 {
            return;
        }
        self.audio.mixer().take_meters(&mut self.meter_scratch);
        let out_rms = (self.out_sum2 / self.meter_frames as f64).sqrt() as f32;
        let (amb, rain) = self.audio.mixer().take_env_meters();
        if let Ok(mut m) = sh.meters.try_lock() {
            m.clear();
            m.extend(self.meter_scratch.iter().map(|(id, x)| (id.0, x.peak, x.rms)));
            m.push((u32::MAX, self.out_peak, out_rms));
            m.push((u32::MAX - 1, amb.peak, amb.rms));
            m.push((u32::MAX - 2, rain.peak, rain.rms));
        }
        self.meter_frames = 0;
        self.out_peak = 0.0;
        self.out_sum2 = 0.0;
    }

    /// The shared governor (omg-audio), and the stats a HUD reads.
    fn govern(&mut self, sh: &Shared, secs: f64, frames: usize) {
        if self.gov.report(self.audio.mixer(), secs, frames) {
            sh.load.store(self.gov.load().to_bits(), Ordering::Relaxed);
            let (rendering, paths, _) = self.audio.mixer().occupancy();
            sh.rendering.store(rendering, Ordering::Relaxed);
            sh.paths.store(paths, Ordering::Relaxed);
            sh.rung.store(self.gov.rung(), Ordering::Relaxed);
        }
    }
}

/// Live numbers for a HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub tick_ms: f32,
    /// The last tick by part (ms): direct, one-edge, two-edge, early,
    /// late, chains, fold.
    pub tick_parts: [f32; 7],
    /// The last tick traced a world change's extra rays (a door moved).
    pub tick_burst: bool,
    /// The last GPU walk, submit to read-back (ms; 0 on the CPU).
    pub gpu_ms: f32,
    pub ticks: u32,
    pub load: f32,
    pub budget: (usize, usize, usize, usize),
    pub rendering: usize,
    pub paths: usize,
    pub peak: f32,
    pub enclosure: f32,
    pub rt60: f32,
    pub underruns: usize,
    /// Largest mix callback (frames) and the ring kept ahead of it.
    pub read_max: usize,
    pub ring: usize,
}

pub struct Host {
    pub sample_rate: f32,
    /// omg-audio's game-thread handle: geometry, sources, objects,
    /// listener, play/stop.
    pub engine: Engine,
    sh: Arc<Shared>,
    out: Arc<Mutex<Out>>,
    tx: Sender<OutCmd>,
    thread: Option<JoinHandle<()>>,
    scratch: Vec<(f32, f32)>,
    pub gpu: bool,
}

impl Host {
    pub fn new(sample_rate: f32, hrir_grid: &[u8], hrir_speakers: &[u8], gpu: bool) -> Self {
        let cfg = EngineConfig {
            sim: Config {
                quality: Quality::High,
                compute: if gpu { Compute::Gpu } else { Compute::Cpu },
                ..Config::default()
            },
            tick_hz: 20.0,
            hrir: Hrir::from_bytes(Some(hrir_speakers), Some(hrir_grid)),
            mixer_capacity: 64,
            ..EngineConfig::default()
        };
        let (engine, audio) = Engine::new(sample_rate, cfg);
        let (tx, rx) = channel();
        let sh = Arc::new(Shared { ring_target: AtomicUsize::new(1024), ..Default::default() });
        let mut out = Out {
            audio,
            rx,
            sr: sample_rate,
            frame_seen: 0,
            l: Vec::with_capacity(4096),
            r: Vec::with_capacity(4096),
            gov: Governor::new(sample_rate),
            meter_frames: 0,
            out_peak: 0.0,
            out_sum2: 0.0,
            meter_scratch: Vec::new(),
        };
        out.gov.apply(out.audio.mixer());
        sh.rung.store(1, Ordering::Relaxed);
        Self {
            sample_rate,
            engine,
            sh,
            out: Arc::new(Mutex::new(out)),
            tx,
            thread: None,
            scratch: Vec::new(),
            gpu,
        }
    }

    /// Render ahead on a dedicated thread; read it back with `reader()`.
    pub fn start_output(&mut self, ring_frames: usize) {
        if self.thread.is_some() {
            return;
        }
        self.sh.ring_target.store(ring_frames.max(BLOCK), Ordering::Relaxed);
        let (sh, out) = (self.sh.clone(), self.out.clone());
        self.thread = Some(
            std::thread::Builder::new()
                .name("omg-godot-out".into())
                .spawn(move || {
                    let mut buf = vec![(0.0f32, 0.0f32); BLOCK];
                    while !sh.stop.load(Ordering::Relaxed) {
                        let fill = sh.ring.lock().unwrap().len();
                        if fill + BLOCK <= sh.ring_target.load(Ordering::Relaxed) {
                            // render AND push under the renderer's lock
                            // (OutputReader::read relies on it)
                            let mut o = out.lock().unwrap();
                            o.render(&sh, &mut buf);
                            sh.ring.lock().unwrap().extend(buf.iter().copied());
                        } else {
                            std::thread::sleep(std::time::Duration::from_micros(1500));
                        }
                    }
                })
                .expect("spawn the output thread"),
        );
    }

    pub fn output_running(&self) -> bool {
        self.thread.is_some()
    }

    pub fn reader(&self) -> OutputReader {
        OutputReader { sh: self.sh.clone(), out: self.out.clone() }
    }

    /// Synchronous render (no output thread).
    pub fn render(&mut self, frames: usize, out: &mut Vec<(f32, f32)>) {
        let mut o = self.out.lock().unwrap();
        let mut left = frames;
        while left > 0 {
            let n = left.min(BLOCK);
            self.scratch.resize(n, (0.0, 0.0));
            o.render(&self.sh, &mut self.scratch);
            out.extend_from_slice(&self.scratch);
            left -= n;
        }
    }

    pub fn send(&self, c: OutCmd) {
        let _ = self.tx.send(c);
    }

    /// A new ambience bed (interleaved stereo when `stereo`): levelled on
    /// a thread of its own (`flatten_slow_loudness`, ~45 ms for the demo's
    /// bed — in the render it held the lock the audio callback takes: a
    /// 23 ms callback, a dropout), then handed to the renderer to swap in.
    pub fn set_ambience(&self, mut data: Vec<f32>, stereo: bool, sample_rate: f32) {
        let tx = self.tx.clone();
        let _ = std::thread::Builder::new().name("omg-ambience".into()).spawn(move || {
            omg_dsp::level::flatten_slow_loudness(&mut data, if stereo { 2 } else { 1 }, sample_rate);
            let _ = tx.send(OutCmd::Ambience(data, stereo));
        });
    }

    /// Head orientation at audio rate.
    pub fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.engine.set_head(yaw, pitch, roll);
    }

    /// The latest levels: (source id, peak, RMS); the output as
    /// `u32::MAX`, the ambience bed `u32::MAX - 1`, the rain `u32::MAX - 2`.
    pub fn meters(&self) -> Vec<(u32, f32, f32)> {
        self.sh.meters.lock().map(|m| m.clone()).unwrap_or_default()
    }

    pub fn stats(&self) -> Stats {
        let (tick_ms, ticks) = self.engine.sim_stats();
        let f = |a: &AtomicU32| f32::from_bits(a.load(Ordering::Relaxed));
        Stats {
            tick_ms,
            tick_parts: self.engine.tick_parts(),
            tick_burst: self.engine.tick_burst(),
            gpu_ms: self.engine.gpu_walk_ms(),
            ticks,
            load: f(&self.sh.load),
            budget: LADDER[self.sh.rung.load(Ordering::Relaxed).min(LADDER.len() - 1)],
            rendering: self.sh.rendering.load(Ordering::Relaxed),
            paths: self.sh.paths.load(Ordering::Relaxed),
            peak: f(&self.sh.peak),
            enclosure: f(&self.sh.enclosure),
            rt60: f(&self.sh.rt60),
            underruns: self.sh.underruns.load(Ordering::Relaxed),
            read_max: self.sh.read_max.load(Ordering::Relaxed),
            ring: self.sh.ring_target.load(Ordering::Relaxed),
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omg_audio::{Geometry, Material, Play, SourceDesc};

    const GRID: &[u8] = omg_audio::HRIR_GRID;
    const SPK: &[u8] = omg_audio::HRIR_SPEAKERS;

    /// End to end without Godot: a looping source in a brick room is
    /// heard, bounded, and the room rings.
    #[test]
    fn a_source_in_a_room_is_heard() {
        let mut h = Host::new(48_000.0, GRID, SPK, false);
        let mut g = Geometry::new();
        let brick = g.material(Material::BRICK);
        let t = 0.2;
        for (a, b) in [
            ([-t, -t, -t], [6.0 + t, 5.0 + t, 0.0]),
            ([-t, -t, 3.0], [6.0 + t, 5.0 + t, 3.0 + t]),
            ([-t, -t, 0.0], [0.0, 5.0 + t, 3.0]),
            ([6.0, -t, 0.0], [6.0 + t, 5.0 + t, 3.0]),
            ([0.0, -t, 0.0], [6.0, 0.0, 3.0]),
            ([0.0, 5.0, 0.0], [6.0, 5.0 + t, 3.0]),
        ] {
            g.add_box(a, b, brick);
        }
        h.engine.set_geometry(g);
        let src = h.engine.add_source(SourceDesc::at([2.0, 2.5, 1.5]));
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.0575).sin() * 0.3).collect();
        h.engine.set_listener([4.5, 2.5, 1.6], 0.0, 0.0, 0.0);
        let mut out = Vec::new();
        h.render(256, &mut out); // drain: the mixer takes commands while rendering
        h.engine.play(src, &omg_audio::clip(&tone), Play::LOOP);
        std::thread::sleep(std::time::Duration::from_millis(400));
        out.clear();
        h.render(24_000, &mut out);
        let peak = out.iter().fold(0.0f32, |m, (l, r)| m.max(l.abs()).max(r.abs()));
        assert!(peak > 0.01 && peak < 1.0, "peak {peak}");
        let st = h.stats();
        assert!(st.rt60 > 0.2, "a brick room should ring: {st:?}");
    }
}
