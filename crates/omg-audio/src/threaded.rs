//! `Engine`: the batteries-included native host. A simulation thread
//! ticks the `Simulation` at a fixed rate and hands each `Frame` to the
//! audio side through a lock-free latest-value slot; mixer commands
//! (play, stop, volumes, head) travel through a lock-free single-producer
//! single-consumer ring. Your audio callback owns the `AudioOut` and
//! calls `render`.
//!
//! ```no_run
//! use omg_audio::{threaded::{Engine, EngineConfig}, Geometry, Material, Play, SourceDesc};
//! let (mut engine, mut out) = Engine::new(48_000.0, EngineConfig::default());
//! let mut g = Geometry::new();
//! let brick = g.material(Material::BRICK);
//! g.add_box([-5.0, -5.0, -0.2], [5.0, 5.0, 0.0], brick);
//! engine.set_geometry(g);
//! let src = engine.add_source(SourceDesc::at([3.0, 0.0, 1.5]));
//! let clip = omg_audio::clip(&vec![0.1; 48_000]);
//! engine.play(src, &clip, Play::LOOP);
//! engine.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
//! // in the audio callback:
//! let (mut l, mut r) = (vec![0.0; 256], vec![0.0; 256]);
//! out.render(&mut l, &mut r);
//! ```

use crate::frame::{Frame, SourceId};
use crate::geometry::Geometry;
use crate::mixer::{Clip, Hrir, Mixer, Play, VoiceId};
use crate::sim::{Config, DebugPath, ObjectId, Simulation, SourceDesc};
use omg_core::material::Material;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Engine settings.
#[derive(Clone)]
pub struct EngineConfig {
    pub sim: Config,
    /// Simulation ticks per second (10–30).
    pub tick_hz: f32,
    pub hrir: Hrir,
    /// Mixer channels preallocated (sources beyond allocate on arrival).
    pub mixer_capacity: usize,
    /// Exact frames per second between the ticks (0: none): the direct
    /// paths, edge bends and mirror chains for where everything is now
    /// (`Simulation::tick_exact`) — moving sounds follow at this rate, the
    /// reverb at the tick's.
    pub exact_hz: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { sim: Config::default(), tick_hz: 20.0, hrir: Hrir::none(), mixer_capacity: 64, exact_hz: 120.0 }
    }
}

// ---- lock-free single-producer single-consumer ring ----------------------

struct Ring<T> {
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Next slot to read (consumer-owned).
    head: AtomicUsize,
    /// Next slot to write (producer-owned).
    tail: AtomicUsize,
}

unsafe impl<T: Send> Send for Ring<T> {}
unsafe impl<T: Send> Sync for Ring<T> {}

impl<T> Ring<T> {
    fn new(cap: usize) -> Self {
        let buf = (0..cap + 1).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
        Self { buf, head: AtomicUsize::new(0), tail: AtomicUsize::new(0) }
    }

    /// Producer side only. Gives the value back when full.
    fn push(&self, v: T) -> Result<(), T> {
        let t = self.tail.load(Ordering::Relaxed);
        let next = (t + 1) % self.buf.len();
        if next == self.head.load(Ordering::Acquire) {
            return Err(v);
        }
        unsafe { (*self.buf[t].get()).write(v) };
        self.tail.store(next, Ordering::Release);
        Ok(())
    }

    /// Consumer side only.
    fn pop(&self) -> Option<T> {
        let h = self.head.load(Ordering::Relaxed);
        if h == self.tail.load(Ordering::Acquire) {
            return None;
        }
        let v = unsafe { (*self.buf[h].get()).assume_init_read() };
        self.head.store((h + 1) % self.buf.len(), Ordering::Release);
        Some(v)
    }
}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Mixer commands (game thread → audio thread).
enum MixCmd {
    Play(VoiceId, SourceId, Clip, Play),
    Near(VoiceId, Clip, bool, f32, f32),
    Stop(VoiceId),
    Gain(SourceId, f32),
    Head(f32, f32, f32),
    FollowFramesHead,
    Master(f32),
}

/// Simulation commands (game thread → sim thread).
enum SimCmd {
    Geometry(Box<Geometry>),
    AddSource(SourceId, SourceDesc),
    SetSource(SourceId, [f32; 3], bool),
    SourceGain(SourceId, f32),
    RemoveSource(SourceId),
    AddObject(ObjectId, [f32; 3], [f32; 3], Material),
    SetObject(ObjectId, [f32; 3], [f32; 3]),
    ObjectMaterial(ObjectId, Material),
    ObjectEnabled(ObjectId, bool),
    RemoveObject(ObjectId),
    Listener([f32; 3], f32, f32, f32, f64),
    Config(Config),
    DebugPaths(bool),
}

/// Latest-value handoff of frames, plus the return path that sends a
/// spent frame back so the audio thread never frees one.
struct FrameSlot {
    latest: AtomicPtr<Frame>,
    spent: AtomicPtr<Frame>,
}

impl FrameSlot {
    /// Free what the audio side handed back, then make `frame` the latest.
    fn publish(&self, frame: Box<Frame>) {
        self.reclaim();
        let old = self.latest.swap(Box::into_raw(frame), Ordering::AcqRel);
        if !old.is_null() {
            drop(unsafe { Box::from_raw(old) }); // never taken: superseded
        }
    }

    fn reclaim(&self) {
        let p = self.spent.swap(ptr::null_mut(), Ordering::AcqRel);
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
    }
}

impl Drop for FrameSlot {
    fn drop(&mut self) {
        for p in [self.latest.swap(ptr::null_mut(), Ordering::AcqRel), self.spent.swap(ptr::null_mut(), Ordering::AcqRel)] {
            if !p.is_null() {
                drop(unsafe { Box::from_raw(p) });
            }
        }
    }
}

/// Live numbers from the simulation thread.
#[derive(Default)]
struct Stats {
    tick_us: AtomicU32,
    ticks: AtomicU32,
    walk_us: AtomicU32,
}

/// The game-thread handle: geometry, sources, objects, listener, and
/// the mixer's controls. Dropping it stops the simulation thread.
pub struct Engine {
    sim_tx: mpsc::Sender<SimCmd>,
    mix: Arc<Ring<MixCmd>>,
    slot: Arc<FrameSlot>,
    stats: Arc<Stats>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// The sim thread's last debug paths (written only while enabled).
    paths: Arc<Mutex<Vec<DebugPath>>>,
    next_source: u32,
    next_object: u32,
    next_voice: u32,
    /// The listener stamp clock, and the last position sent with the time
    /// it was first sent (see `set_listener`).
    clock: Instant,
    pose: Option<([f32; 3], f64)>,
}

/// The audio-thread half: call `render` from your audio callback.
pub struct AudioOut {
    mixer: Mixer,
    mix: Arc<Ring<MixCmd>>,
    slot: Arc<FrameSlot>,
    current: Option<Box<Frame>>,
    spent: Option<Box<Frame>>,
}

const MIX_RING: usize = 1024;

impl Engine {
    pub fn new(sample_rate: f32, cfg: EngineConfig) -> (Engine, AudioOut) {
        let (sim_tx, sim_rx) = mpsc::channel::<SimCmd>();
        let mix = Arc::new(Ring::new(MIX_RING));
        let slot = Arc::new(FrameSlot { latest: AtomicPtr::new(ptr::null_mut()), spent: AtomicPtr::new(ptr::null_mut()) });
        let stats = Arc::new(Stats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let period = Duration::from_secs_f32(1.0 / cfg.tick_hz.clamp(1.0, 200.0));
        let exact = (cfg.exact_hz > cfg.tick_hz).then(|| Duration::from_secs_f32(1.0 / cfg.exact_hz.min(1000.0)));
        let paths = Arc::new(Mutex::new(Vec::new()));
        let thread = {
            let (slot, stats, stop, sim_cfg) = (slot.clone(), stats.clone(), stop.clone(), cfg.sim);
            let paths = paths.clone();
            std::thread::Builder::new()
                .name("omg-audio-sim".into())
                .spawn(move || sim_loop(Simulation::new(Config { sample_rate, ..sim_cfg }), sim_rx, slot, stats, paths, stop, period, exact))
                .expect("spawn the simulation thread")
        };
        let mixer = Mixer::with_capacity(sample_rate, cfg.hrir, cfg.mixer_capacity);
        let out = AudioOut { mixer, mix: mix.clone(), slot: slot.clone(), current: None, spent: None };
        let engine = Engine {
            sim_tx,
            mix,
            slot,
            stats,
            stop,
            thread: Some(thread),
            paths,
            next_source: 0,
            next_object: 0,
            next_voice: 1 << 31, // disjoint from ids the Mixer hands out itself
            clock: Instant::now(),
            pose: None,
        };
        (engine, out)
    }

    fn sim(&self, c: SimCmd) {
        let _ = self.sim_tx.send(c);
    }

    /// Mixer commands are dropped when the audio side is not draining
    /// them (no audio callback running); returns false then.
    fn mix(&self, c: MixCmd) -> bool {
        self.mix.push(c).is_ok()
    }

    pub fn set_geometry(&mut self, geometry: Geometry) {
        self.sim(SimCmd::Geometry(Box::new(geometry)));
    }

    pub fn set_config(&mut self, cfg: Config) {
        self.sim(SimCmd::Config(cfg));
    }

    /// Record each tick's strongest traced paths for a debug view (see
    /// `Simulation::set_debug_paths`); read them with `debug_paths`.
    pub fn set_debug_paths(&mut self, on: bool) {
        self.sim(SimCmd::DebugPaths(on));
        if !on {
            self.paths.lock().unwrap().clear();
        }
    }

    /// The simulation's latest debug paths (empty while disabled).
    pub fn debug_paths(&self) -> Vec<DebugPath> {
        self.paths.lock().unwrap().clone()
    }

    pub fn add_source(&mut self, desc: SourceDesc) -> SourceId {
        let id = SourceId(self.next_source);
        self.next_source += 1;
        self.sim(SimCmd::AddSource(id, desc));
        id
    }

    pub fn set_source(&mut self, id: SourceId, position: [f32; 3], active: bool) {
        self.sim(SimCmd::SetSource(id, position, active));
    }

    /// Acoustic loudness (see `SourceDesc::gain`).
    pub fn set_source_loudness(&mut self, id: SourceId, gain: f32) {
        self.sim(SimCmd::SourceGain(id, gain));
    }

    pub fn remove_source(&mut self, id: SourceId) {
        self.sim(SimCmd::RemoveSource(id));
    }

    pub fn add_object(&mut self, min: [f32; 3], max: [f32; 3], material: Material) -> ObjectId {
        let id = ObjectId(self.next_object);
        self.next_object += 1;
        self.sim(SimCmd::AddObject(id, min, max, material));
        id
    }

    pub fn set_object(&mut self, id: ObjectId, min: [f32; 3], max: [f32; 3]) {
        self.sim(SimCmd::SetObject(id, min, max));
    }

    pub fn set_object_material(&mut self, id: ObjectId, material: Material) {
        self.sim(SimCmd::ObjectMaterial(id, material));
    }

    pub fn set_object_enabled(&mut self, id: ObjectId, enabled: bool) {
        self.sim(SimCmd::ObjectEnabled(id, enabled));
    }

    pub fn remove_object(&mut self, id: ObjectId) {
        self.sim(SimCmd::RemoveObject(id));
    }

    /// Listener pose for the simulation; the head orientation reaches
    /// the mixer with the next frame (or call `set_head` for audio-rate
    /// head tracking). Call it every frame: the pose is stamped with the
    /// time it was first sent at this position — a game that renders
    /// faster than its physics moves the listener sends the same position
    /// a few times, and it has been there since the physics step (see
    /// `Simulation::set_listener_at`; Doppler depends on it).
    pub fn set_listener(&mut self, position: [f32; 3], yaw: f32, pitch: f32, roll: f32) {
        let time = match self.pose {
            Some((p, t)) if p == position => t,
            _ => self.clock.elapsed().as_secs_f64(),
        };
        self.set_listener_at(position, yaw, pitch, roll, time);
    }

    /// `set_listener` with your own time stamp (seconds, a steady clock:
    /// when the pose was current, e.g. your physics step's time).
    pub fn set_listener_at(&mut self, position: [f32; 3], yaw: f32, pitch: f32, roll: f32, time: f64) {
        self.pose = Some((position, time));
        self.sim(SimCmd::Listener(position, yaw, pitch, roll, time));
    }

    fn voice(&mut self) -> VoiceId {
        let v = self.next_voice;
        self.next_voice = self.next_voice.wrapping_add(1).max(1 << 31);
        VoiceId(v)
    }

    /// Play a clip on a source (see `Mixer::play`).
    pub fn play(&mut self, source: SourceId, clip: &Clip, play: Play) -> VoiceId {
        let v = self.voice();
        if self.mix(MixCmd::Play(v, source, clip.clone(), play)) { v } else { VoiceId::NONE }
    }

    pub fn play_near(&mut self, clip: &Clip, right: bool, dist_m: f32, gain: f32) -> VoiceId {
        let v = self.voice();
        if self.mix(MixCmd::Near(v, clip.clone(), right, dist_m, gain)) { v } else { VoiceId::NONE }
    }

    pub fn stop(&mut self, voice: VoiceId) {
        self.mix(MixCmd::Stop(voice));
    }

    /// A source's volume on the mixer (see `Mixer::set_source_gain`).
    pub fn set_volume(&mut self, source: SourceId, gain: f32) {
        self.mix(MixCmd::Gain(source, gain));
    }

    /// Audio-rate head orientation (switches the mixer to manual head
    /// control; `follow_listener_head` switches back).
    pub fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.mix(MixCmd::Head(yaw, pitch, roll));
    }

    pub fn follow_listener_head(&mut self) {
        self.mix(MixCmd::FollowFramesHead);
    }

    pub fn set_master(&mut self, gain: f32) {
        self.mix(MixCmd::Master(gain));
    }

    /// Wall-clock of the last simulation tick (ms) and ticks so far.
    pub fn sim_stats(&self) -> (f32, u32) {
        (self.stats.tick_us.load(Ordering::Relaxed) as f32 / 1000.0, self.stats.ticks.load(Ordering::Relaxed))
    }

    /// Wall-clock of the last finished GPU walk (ms; 0 on the CPU).
    pub fn gpu_walk_ms(&self) -> f32 {
        self.stats.walk_us.load(Ordering::Relaxed) as f32 / 1000.0
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.slot.reclaim();
    }
}

/// One command into the simulation.
fn handle(sim: &mut Simulation, c: SimCmd) {
    match c {
        SimCmd::Geometry(g) => sim.set_geometry(*g),
        SimCmd::AddSource(id, d) => sim.add_source_as(id, d),
        SimCmd::SetSource(id, p, a) => sim.set_source(id, p, a),
        SimCmd::SourceGain(id, g) => sim.set_source_gain(id, g),
        SimCmd::RemoveSource(id) => sim.remove_source(id),
        SimCmd::AddObject(id, a, b, m) => sim.add_object_as(id, a, b, m),
        SimCmd::SetObject(id, a, b) => sim.set_object(id, a, b),
        SimCmd::ObjectMaterial(id, m) => sim.set_object_material(id, m),
        SimCmd::ObjectEnabled(id, e) => sim.set_object_enabled(id, e),
        SimCmd::RemoveObject(id) => sim.remove_object(id),
        SimCmd::Listener(p, y, pi, r, t) => sim.set_listener_at(p, y, pi, r, t),
        SimCmd::Config(c) => sim.set_config(c),
        SimCmd::DebugPaths(on) => sim.set_debug_paths(on),
    }
}

fn sim_loop(
    mut sim: Simulation,
    rx: mpsc::Receiver<SimCmd>,
    slot: Arc<FrameSlot>,
    stats: Arc<Stats>,
    paths: Arc<Mutex<Vec<DebugPath>>>,
    stop: Arc<AtomicBool>,
    period: Duration,
    exact: Option<Duration>,
) {
    let mut next = Instant::now();
    let mut held: Vec<SimCmd> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        for c in held.drain(..).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
            handle(&mut sim, c);
        }
        let t0 = Instant::now();
        let frame = Box::new(sim.tick());
        stats.tick_us.store(t0.elapsed().as_micros().min(u32::MAX as u128) as u32, Ordering::Relaxed);
        stats.ticks.fetch_add(1, Ordering::Relaxed);
        stats.walk_us.store((sim.gpu_walk_ms() * 1000.0) as u32, Ordering::Relaxed);
        if sim.debug_paths_enabled() {
            *paths.lock().unwrap() = sim.debug_paths();
        }
        slot.publish(frame);
        next += period;
        let mut next_exact = Instant::now() + exact.unwrap_or(period);
        // wait for the next tick — exact frames meanwhile (the direct
        // paths, bends and mirror chains where everything is now: moving
        // sounds follow at this rate) — or not: a source switched on is
        // traced at once (a thrown ball's sound waits for its first paths)
        loop {
            let now = Instant::now();
            if next <= now {
                next = now; // fell behind: don't try to catch up in a burst
                break;
            }
            if let Some(e) = exact {
                if now >= next_exact {
                    for c in held.drain(..) {
                        handle(&mut sim, c);
                    }
                    if let Some(f) = sim.tick_exact() {
                        slot.publish(Box::new(f));
                    }
                    next_exact = Instant::now() + e;
                    continue;
                }
            }
            let until = if exact.is_some() { next.min(next_exact) } else { next };
            match rx.recv_timeout(until.saturating_duration_since(now)) {
                Ok(c) => {
                    let wakes = match &c {
                        SimCmd::AddSource(_, d) => d.active,
                        SimCmd::SetSource(id, _, on) => *on && !sim.is_active(*id),
                        _ => false,
                    };
                    held.push(c);
                    if wakes {
                        next = Instant::now();
                        break;
                    }
                }
                // the tick's time (or an exact frame's: the loop decides)
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if exact.is_none() {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

impl AudioOut {
    /// Render the next `l.len()` stereo samples: applies queued mixer
    /// commands and the newest frame, then renders. Real-time safe except
    /// for `Mixer::apply`'s small allocations when a new frame arrives
    /// (see the `mixer` module docs).
    pub fn render(&mut self, l: &mut [f32], r: &mut [f32]) {
        while let Some(c) = self.mix.pop() {
            match c {
                MixCmd::Play(v, s, clip, p) => {
                    self.mixer.play_as(v, s, &clip, p);
                }
                MixCmd::Near(v, clip, right, d, g) => {
                    self.mixer.play_near_as(v, &clip, right, d, g);
                }
                MixCmd::Stop(v) => self.mixer.stop(v),
                MixCmd::Gain(s, g) => self.mixer.set_source_gain(s, g),
                MixCmd::Head(y, p, ro) => self.mixer.set_head(y, p, ro),
                MixCmd::FollowFramesHead => self.mixer.follow_frames_head(),
                MixCmd::Master(g) => self.mixer.set_master(g),
            }
        }
        // hand the previous frame back for the sim thread to free
        if let Some(f) = self.spent.take() {
            let p = Box::into_raw(f);
            if self.slot.spent.compare_exchange(ptr::null_mut(), p, Ordering::AcqRel, Ordering::Acquire).is_err() {
                self.spent = Some(unsafe { Box::from_raw(p) });
            }
        }
        let p = self.slot.latest.swap(ptr::null_mut(), Ordering::AcqRel);
        if !p.is_null() {
            let f = unsafe { Box::from_raw(p) };
            self.mixer.apply(&f);
            if let Some(old) = self.current.replace(f) {
                if self.spent.is_none() {
                    self.spent = Some(old);
                } else {
                    drop(old); // rare: the return slot was still full twice over
                }
            }
        }
        self.mixer.render(l, r);
    }

    /// The mixer (meters, budgets, direct control).
    pub fn mixer(&mut self) -> &mut Mixer {
        &mut self.mixer
    }

    /// The last frame applied (version, sky, blocks).
    pub fn frame(&self) -> Option<&Frame> {
        self.current.as_deref()
    }
}
