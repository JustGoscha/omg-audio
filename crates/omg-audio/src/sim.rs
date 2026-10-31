//! `Simulation`: the non-real-time half. Owns the level geometry, the
//! moving objects, the sources and the listener; `tick()` traces how
//! sound gets from every source to the listener and returns a `Frame`
//! for the `Mixer`. No threads, no clocks: call `tick()` at 10–30 Hz from
//! wherever suits your host (a game thread, a worker, `threaded::Engine`).
//!
//! The tracer is engine v3 (ENGINE_V3.md): rays leave the listener and
//! bounce through the level; at every bounce the surface connects to the
//! sources it can see; the direct path is one exact segment, paying every
//! wall it passes through (mass law); rays that pass close to an edge
//! bend around it. The ray budget is per tick — shared by all sources —
//! and an exponential average across ticks carries the variance.

use crate::frame::{Frame, Listener, SkyCell, SourceId};
use crate::geometry::Geometry;
use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::params::{ParamBlock, RemoteReverb, Tap};
use omg_core::v3::walker::{trace_with, V3Walker};
use omg_core::v3::{Aperture, Budget, DenseResponse, Motion, Source, Track, Tracer, World};
use std::collections::VecDeque;
use omg_dsp::dense::{build_kernel_max, DenseKernel, KernelBuilder};
use std::sync::Arc;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

/// How much ray tracing a tick may do: a level on the quality ladder
/// (`LADDER`), the named tiers being four of its rungs: low 4, medium
/// 6, high 8, ultra 10.
#[derive(Clone, Copy, Debug)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
    /// A rung of `LADDER` (0 cheapest … `LEVELS` − 1): what quality auto
    /// walks (`governor::LevelGovernor`).
    Level(usize),
    /// Full control over the tracer's budget.
    Custom(Budget),
}

/// The quality ladder, cheapest first: (early rays, early bounces, late
/// rays, next-event connections a bounce, one-bend edge searches a tick,
/// two-bend searches a tick). Each rung costs about `LEVEL_STEP`× the one
/// below. Every rung simulates the same physics — every path reaches the
/// ears at every level, nothing is cut: the early family's bounces past
/// its own count are the late family's, a source not searching keeps the
/// edges it found, solved exactly. What a lower rung gives up is how
/// many samples estimate what matters least to the ear, shed first and
/// hardest going down: the two-bend searches (off below rung 3), then
/// the early family's high bounces and the reverb's late rays, then the
/// rays and the one-bend searches; the direct paths, their occlusion and
/// the edges found are never shed.
pub const LADDER: [(u32, u32, u32, usize, usize, usize); LEVELS] = [
    (128, 2, 16, 1, 1, 0),
    (192, 2, 24, 2, 1, 0),
    (256, 3, 32, 2, 2, 0),
    (384, 3, 32, 2, 2, 1),
    (512, 3, 48, 3, 2, 1),
    (640, 4, 64, 3, 3, 1),
    (768, 4, 96, 3, 4, 1),
    (896, 5, 112, 4, 6, 2),
    (1024, 6, 128, 4, 8, 2),
    (1536, 7, 192, 4, 12, 3),
    (2048, 8, 256, 4, 16, 4),
];
pub const LEVELS: usize = 11;
/// About how much more a rung costs than the one below (measured:
/// tick_move, the town flying: 4.0, 6.1, 8.9, 12.4, 17.9 ms … on an M1
/// Max).
pub const LEVEL_STEP: f32 = 1.45;

/// Where the ray walk runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    Cpu,
    /// The early rays on the GPU through wgpu (feature `gpu`): 8× the
    /// CPU tier's for the same tick, arriving a tick late (the GPU never
    /// blocks the tick). The late rays stay on the CPU, walked whole with
    /// each walk — hundreds of bounces in a row are what a GPU is slowest
    /// at. Without an adapter, without the feature, or for a tick beyond
    /// the kernel's caps (> 64 traced sources or > 64 objects), the tick
    /// runs on the CPU instead.
    Gpu,
    /// `Gpu`, but each tick waits for its own walk (offline renders,
    /// tests: deterministic timing, no frame of latency).
    GpuSync,
}

/// The share of a tick a live GPU walk's own time may fill at the GPU
/// budget: the rest is its passes' round trips.
const GPU_TICK_SHARE: f32 = 0.7;
/// The fewest of the tier's GPU rays a live walk shrinks to.
const GPU_RAYS_MIN: f32 = 0.125;
/// Walks between two re-measurements of what the rays cost.
const GPU_SIZING_PROBE: u32 = 200;

impl Quality {
    /// The ladder's rung of this quality (None: a custom budget).
    pub fn level(self) -> Option<usize> {
        match self {
            Quality::Low => Some(4),
            Quality::Medium => Some(6),
            Quality::High => Some(8),
            Quality::Ultra => Some(10),
            Quality::Level(n) => Some(n.min(LEVELS - 1)),
            Quality::Custom(_) => None,
        }
    }

    /// The tracer budget of this quality on the CPU (`gpu = false`) or
    /// GPU (8× the early rays; the late rays are the CPU's either way).
    pub fn budget(self, gpu: bool) -> Budget {
        let (rays, bounces, late, nee, edge, edge2) = match (self, self.level()) {
            (Quality::Custom(b), _) => return b,
            (_, Some(l)) => LADDER[l],
            (_, None) => unreachable!("only a custom budget has no level"),
        };
        let rays = if gpu { (rays * 8).min(65536) } else { rays };
        Budget {
            rays,
            bounces,
            k: 0.06,
            ema: 0.35,
            nee,
            late_rays: late,
            late_ema: 0.1,
            late_nee: 1,
            diffraction: true,
            edge_searches: edge,
            edge2_searches: edge2,
        }
    }
}

/// Simulation settings.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub quality: Quality,
    pub compute: Compute,
    /// Most sources traced per tick. Beyond it, sources take turns by
    /// priority — loud, near and moving sources every tick, quiet distant
    /// ones less often (they keep their last result meanwhile) — so the
    /// tick cost stays flat however many sources there are. 64 is the
    /// GPU kernel's per-tick cap.
    pub max_traced_sources: usize,
    /// Seed of the tracer's random numbers (same seed, same inputs →
    /// same frames on the CPU).
    pub seed: u64,
    /// Engine v4 (ENGINE_V4.md): render the traced response as traced —
    /// coherent arrivals as taps at any delay, the scattered rest as a
    /// convolved dense response (no early/late split, no parametric
    /// reverb). CPU and GPU walks alike.
    pub v4: bool,
    /// Engine v5 (on v4): the late family's paths are reused across ticks,
    /// each reconnected exactly to where the listener is now — ×8 samples
    /// for the reverb's estimate, standing still or moving (`Tracer::v5`).
    pub v5: bool,
    /// Engine v6 (ENGINE_V6.md): v3's estimate and reverb — discrete early
    /// arrivals and a parametric reverb gliding with the trace, steady in
    /// motion — with v6's refinements. On, `v4` and `v5` are off.
    pub v6: bool,
    /// Engine v6.1 (on v6): the reverberation time as the traced decay
    /// says, however long — the echogram's cut-off tail carried on — and
    /// the late level uncapped (`omg_core::tracer::fit_reverb`; v6 clamps
    /// the time at 5 s and the level under 0.7).
    pub v61: bool,
    /// Engine v6.2 (on v6.1): an object with an aperture
    /// (`Simulation::set_object_aperture`: a door's leaf in its doorway, a
    /// curtain in its window) prices the paths through it at how much of
    /// it is covered now — the reverb and the early field behind a door
    /// follow it every tick and every frame between, not the averages.
    pub v62: bool,
    /// Portals (on v6.2, the walk on the CPU): what reaches the listener
    /// from a source enclosed in a room through the room's openings (the
    /// apertures: `set_object_aperture`) estimated from the source's side
    /// — rays from it record what leaves each opening, the listener's walk
    /// takes it from where it first crosses one. A loud room heard from
    /// outside comes steady instead of in bursts (its rare paths in).
    pub portals: bool,
    /// Path guiding (on v6): a share of the listener's rays sent where the
    /// last ticks' energy came from, every ray weighed by its direction's
    /// odds — the estimate unbiased, a rare path found once found again.
    pub guide: bool,
    /// A walk's sphere hit weighed by the share of its disk the rays can
    /// reach (on v6): mirror paths near a surface as loud as image
    /// sources say, not 1–4.5 dB low; a door's swing noisier. Off by
    /// default: to be judged by ear.
    pub clearance: bool,
    /// The mixer's sample rate (v4 builds its convolution kernels for it).
    pub sample_rate: f32,
    /// v4: build convolution kernels on a thread of their own (off the
    /// tick); false (or no threads: wasm) builds them in the tick, one at
    /// a time, `kernel_step_samples` of impulse response per tick.
    pub kernel_thread: bool,
    /// v4: the most sources with a convolution kernel — the loudest at the
    /// ears (a member counts double, so the set doesn't flicker); the
    /// others' blocks carry the same response as a parametric reverb.
    pub max_kernels: usize,
    /// v4: kernels at most this long (s); a longer tail is cut (the
    /// native cap is `omg_dsp::dense::MAX_KERNEL_S`).
    pub max_kernel_s: f32,
    /// v4, building in the tick: impulse-response samples synthesized
    /// and transformed per tick.
    pub kernel_step_samples: usize,
    /// v4, building in the tick: a finished kernel at most every this
    /// many ticks (a web host copies each into its audio thread).
    pub kernel_every_ticks: u64,
    /// `Compute::Gpu`: the share of the GPU's time the walk may take
    /// (0..1). The game's frames come first: the walk runs as short
    /// dispatches, paced so its share stays under this — waiting behind
    /// the game's frames counts against it — and its results arrive
    /// later when the GPU is busy drawing.
    pub gpu_budget: f32,
    /// The host's tick period (s). `Compute::Gpu` sizes its walk to land
    /// every tick within `gpu_budget`.
    pub tick_s: f32,
    /// Threads the CPU ray walk is spread over each tick (the tick's own
    /// counts; wasm walks on one). The result is the same for any count.
    /// Default: half the machine's hardware threads, at most 4 — the game
    /// keeps the rest.
    pub walk_threads: usize,
}

/// `Config::walk_threads`' default for this machine (the environment's
/// `OMG_WALK_THREADS` when set).
pub fn default_walk_threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(n) = std::env::var("OMG_WALK_THREADS").ok().and_then(|v| v.parse::<usize>().ok()) {
            return n.max(1);
        }
        std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).clamp(1, 4))
    }
}

impl Config {
    /// The engine switches made consistent: v6 runs v3's estimate (v4
    /// and v5 off); v5 needs v4, v6.1 needs v6.
    pub fn engine(mut self) -> Self {
        if self.v6 {
            self.v4 = false;
        }
        self.v5 &= self.v4;
        self.v61 &= self.v6;
        self.v62 &= self.v61;
        self.portals &= self.v62;
        self.guide &= self.v6;
        self.clearance &= self.v6;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            quality: Quality::High,
            compute: Compute::Cpu,
            max_traced_sources: 64,
            seed: 0x0e3_5eed,
            v4: true,
            v5: true,
            v6: false,
            v61: false,
            v62: false,
            portals: false,
            guide: false,
            clearance: false,
            sample_rate: 48_000.0,
            kernel_thread: true,
            max_kernels: 64,
            max_kernel_s: omg_dsp::dense::MAX_KERNEL_S,
            kernel_step_samples: 4 * 4096,
            kernel_every_ticks: 1,
            gpu_budget: 0.2,
            tick_s: 0.05,
            walk_threads: default_walk_threads(),
        }
    }
}

/// A sound source's description.
#[derive(Clone, Copy, Debug)]
pub struct SourceDesc {
    pub position: [f32; 3],
    /// Acoustic loudness (amplitude, 1 = reference). Scales every path
    /// and steers where the tracer spends its connections; per-channel
    /// volume belongs on the mixer (`Mixer::set_source_gain`).
    pub gain: f32,
    /// Inactive sources are not traced and render silence.
    pub active: bool,
}

impl SourceDesc {
    pub fn at(position: [f32; 3]) -> Self {
        Self { position, gain: 1.0, active: true }
    }
}

/// A moving or changing box in the level: a door leaf, a window pane, a
/// crate, a vehicle. Its material's absorption and scattering apply to
/// rays bouncing off it; its transmission applies once to every path
/// passing through it (as given — the material's 15 cm reference value,
/// which is what the `GLASS`, `DOOR` … presets are tuned for). Open a
/// door by shrinking or moving its box: openness is geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u32);

/// One traced sound path, for a debug view: the listener's ears, then
/// every reflection or edge bend, then the source (world coordinates).
#[derive(Clone, Debug)]
pub struct DebugPath {
    pub source: SourceId,
    /// Mid-band energy arriving along it (relative: ranks the paths).
    pub energy: f32,
    pub points: Vec<[f32; 3]>,
    /// What it is: exactly solved (the same path tick after tick, moving
    /// continuously) or one of the walk's random samples.
    pub kind: PathKind,
    /// Stable for an exact path within (source, kind); 0 for samples.
    pub key: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    Direct,
    /// Around an edge (diffraction).
    Bend,
    /// A solved chain of mirror reflections.
    Mirror,
    /// A sampled path of the walk (a different set every tick).
    Sampled,
    /// Light-traced through an opening (portals): ears → a source ray's
    /// bounce past the opening → the opening → the source (a different
    /// set every tick).
    Portal,
}

struct Src {
    id: SourceId,
    pos: Vec3,
    gain: f32,
    active: bool,
    block: ParamBlock,
    /// Position and tick of its last trace (None = never traced).
    traced: Option<(Vec3, u64)>,
    /// v4: the latest dense response (unit source), and the kernel built
    /// from one (with the response, gain and tick it was built from).
    dense: Option<DenseResponse>,
    kernel: Option<(Arc<DenseKernel>, DenseResponse, f32, u64)>,
    /// v6.2: by how much its block's estimated parts (early taps, reverb)
    /// were repriced since its last tick, per band (`Tracer::reprice`).
    repriced: [[f32; NBANDS]; 2],
    /// Where it was (simulation clock, position), oldest first: every
    /// path reaches it where it emitted what arrives now (`Track`).
    track: VecDeque<(f64, Vec3)>,
    /// When and where the host last set it (it stood there since).
    seen: (f64, Vec3),
    /// What it emitted before `since` left from somewhere else (it jumped
    /// then: the old place's emissions are a ghost's).
    since: f64,
    /// Switched off at this time (still traced while what it emitted
    /// before arrives and rings out): None while on.
    off_at: Option<f64>,
    /// A ghost: the place a jumped source left (its parent's id), traced
    /// until `ghost.1`; it emitted only before `until`.
    ghost: Option<(SourceId, f64)>,
    until: f64,
}

struct Obj {
    id: ObjectId,
    min: Vec3,
    max: Vec3,
    material: Material,
    enabled: bool,
    /// The opening it fills (`set_object_aperture`).
    aperture: Option<(Vec3, Vec3)>,
}

impl Obj {
    /// v6.2: its opening, and itself as the leaf in it while enabled.
    fn aperture(&self) -> Option<Aperture> {
        let (min, max) = self.aperture?;
        let leaf = self.enabled.then_some((self.min, self.max));
        Some(Aperture::new(self.id.0, min, max, leaf, self.material))
    }
}

/// The acoustic world: geometry, objects, sources, listener.
pub struct Simulation {
    cfg: Config,
    mesh: Mesh,
    grade: Option<f32>,
    tracer: Tracer,
    walker: Option<Box<dyn V3Walker>>,
    sources: Vec<Src>,
    objects: Vec<Obj>,
    listener: Listener,
    /// The simulation's clock (s): the host's time (`set_time`,
    /// `set_listener_at`), or a tick period per tick when it gives none.
    /// A frame is the world at this instant (`Frame::time`): the paths
    /// reach every moving source where it emitted what arrives now.
    clock: f64,
    clock_given: bool,
    /// Ghosts made since the last frame (ghost, parent), and the next
    /// ghost id.
    new_ghosts: Vec<(SourceId, SourceId)>,
    next_ghost: u32,
    tick_no: u64,
    /// Frames out (full ticks and exact ones): their and their blocks'
    /// versions.
    frame_no: u64,
    /// The last full tick's sky and environment (exact frames resend
    /// them), and where everything was at the last frame, and when.
    last_sky: Vec<SkyCell>,
    last_env: omg_dsp::env::Environment,
    exact_at: Option<(Vec3, Vec<Vec3>, u64, f64)>,
    next_source: u32,
    next_object: u32,
    panels: Vec<(Vec3, Vec3, Material)>,
    /// v6.2: the objects' apertures as they are (`Obj::aperture`).
    apertures: Vec<Aperture>,
    picks: Vec<(f32, usize)>,
    traced_last: usize,
    debug_paths: bool,
    /// v4: the kernel-building thread (when `Config::kernel_thread` and
    /// threads are available), started at the first build.
    kernels: Option<KernelWorker>,
    /// v4: the sources that get kernels (`Config::max_kernels`).
    kernel_members: Vec<SourceId>,
    /// v4, building in the tick: the kernel under construction (source,
    /// builder, the response and gain it is built from), and when the
    /// last one was handed over.
    kernel_job: Option<(SourceId, KernelBuilder, DenseResponse, f32)>,
    kernel_sent: u64,
    /// `Compute::Gpu`: the share of the tier's GPU rays a walk carries —
    /// what lets it land every tick within `gpu_budget` — sized from the
    /// walk's time against its rays (`size_gpu_walk`).
    gpu_rays: f32,
    gpu_seen: f32,
    gpu_prev: Option<(f32, f32)>,
    gpu_slope: Option<f32>,
    gpu_walks: u32,
}

/// v4's kernel builder on a thread of its own: responses in, kernels out.
struct KernelWorker {
    jobs: std::sync::mpsc::Sender<(SourceId, DenseResponse, f32, f32, f32, u64)>,
    done: std::sync::mpsc::Receiver<(SourceId, Arc<DenseKernel>, DenseResponse, f32, u64)>,
    /// Sources with a build in flight.
    busy: Vec<SourceId>,
}

impl KernelWorker {
    fn start() -> Option<Self> {
        let (jt, jr) = std::sync::mpsc::channel::<(SourceId, DenseResponse, f32, f32, f32, u64)>();
        let (dt, dr) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("omg-kernels".into())
            .spawn(move || {
                while let Ok((id, resp, gain, sr, max_s, tick)) = jr.recv() {
                    let k = Arc::new(build_kernel_max(&resp, sr, gain, id.0, max_s));
                    if dt.send((id, k, resp, gain, tick)).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
        Some(Self { jobs: jt, done: dr, busy: Vec::new() })
    }
}

fn v3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// Ghosts' ids (`Src::ghost`): their own range, never a host's.
const GHOST_ID_BASE: u32 = 0x4000_0000;
const GHOST_ID_SPAN: u32 = 0x1000_0000;

/// How long after its last emission a source at `pos` still sounds at the
/// listener (s): its direct sound's travel, then the longest response a
/// path or kernel carries (`omg_dsp::dense::MAX_KERNEL_S`).
fn ring_out(pos: Vec3, ear: Vec3) -> f64 {
    ((pos - ear).length() / omg_core::SPEED_OF_SOUND) as f64 + omg_dsp::dense::MAX_KERNEL_S as f64 + 0.5
}

/// How long a source's track reaches back (s): its paths' emission times
/// (the longest traced response after the farthest direct arrival).
const TRACK_KEEP_S: f64 = 40.0;
/// A source that moved within this long (s) still has paths moving: what
/// arrives now along them left it while it moved.
const RECENT_MOVE_S: f64 = 10.0;

/// Is `s` traced now: on, or switched off (or a ghost) while what it
/// emitted is still arriving and ringing.
fn sounding(s: &Src, now: f64, ear: Vec3) -> bool {
    match (s.ghost, s.off_at) {
        (Some((_, until)), _) => now <= until,
        (None, None) => true,
        (None, Some(off)) => now - off < ring_out(s.pos, ear),
    }
}

/// A source at `p` at `t` on its track (a move of more than a millimetre
/// from where it was last `seen`; a time already on the track updates its
/// last place).
fn record(track: &mut VecDeque<(f64, Vec3)>, seen: (f64, Vec3), t: f64, p: Vec3) {
    if (seen.1 - p).length() <= 1e-3 {
        return;
    }
    // it stood where it was last seen until then
    if track.back().is_some_and(|b| b.0 < seen.0) {
        track.push_back(seen);
    }
    match track.back_mut() {
        Some(last) if last.0 >= t => last.1 = p,
        _ => track.push_back((t, p)),
    }
    // forget what no path reaches any more (keeping where it stood then)
    while track.len() > 2 && track[1].0 < t - TRACK_KEEP_S {
        track.pop_front();
    }
}

/// The motion of `sources` (in a trace's order) as the tracer takes it:
/// the moving ones' tracks, at `now`.
fn motion(now: f64, sources: &[&Src]) -> Motion {
    Motion {
        now,
        tracks: sources
            .iter()
            .map(|s| {
                (s.track.len() > 1 || s.since.is_finite() || s.until.is_finite())
                    .then(|| Track { knots: s.track.iter().copied().collect(), from: s.since, until: s.until })
            })
            .collect(),
    }
}

/// v4 kernel rebuilds (crossfaded): a dense response whose envelope moved
/// by this share (`DenseResponse::difference_from`, ~0.6 dB) is rebuilt.
/// Smaller differences don't land: standing still they are the estimate's
/// own noise (its history tests restart a window that really changed),
/// and every rebuild steps a steady tone's level.
const KERNEL_CHANGE: f32 = 0.15;
/// A kernel this far from its source's current response beyond the
/// estimates' noise (`DenseResponse::change_from`: a quarter of the
/// envelope's energy moved — the listener stepped into another space) is
/// withdrawn at once and rebuilt before anything else: meanwhile the
/// block's parametric reverb plays the current response (it has been fed
/// all along, so it takes over warm), and the rebuilt kernel crossfades
/// back in over the input the convolver kept.
const KERNEL_STALE: f32 = 0.5;
/// Builds queued on the kernel thread at once.
const KERNEL_QUEUE: usize = 4;

/// A moved source counts as this many ticks overdue for a re-trace.
const MOVED_BOOST: f32 = 4.0;
/// Displacement (m) that makes a source's last trace stale.
const MOVED_M: f32 = 0.25;

impl Simulation {
    /// An empty world (free field: no geometry, no ground) until
    /// `set_geometry`.
    pub fn new(cfg: Config) -> Self {
        let cfg = cfg.engine();
        let mut s = Self {
            cfg,
            mesh: MeshBuilder::new().build(),
            grade: None,
            tracer: {
                let mut t = Tracer::new(cfg.seed);
                t.v4 = cfg.v4;
                t.v5 = cfg.v5;
                t.v6 = cfg.v6;
                t.v61 = cfg.v61;
                t.v62 = cfg.v62;
                t.portals = cfg.portals;
                t.guide = cfg.guide;
                t.clearance = cfg.clearance;
                t.threads = cfg.walk_threads;
                t
            },
            walker: None,
            sources: Vec::new(),
            objects: Vec::new(),
            listener: Listener::default(),
            clock: 0.0,
            clock_given: false,
            new_ghosts: Vec::new(),
            next_ghost: 0,
            tick_no: 0,
            frame_no: 0,
            last_sky: Vec::new(),
            last_env: Default::default(),
            exact_at: None,
            next_source: 0,
            next_object: 0,
            panels: Vec::new(),
            apertures: Vec::new(),
            picks: Vec::new(),
            traced_last: 0,
            debug_paths: false,
            kernels: None,
            kernel_members: Vec::new(),
            kernel_job: None,
            kernel_sent: 0,
            gpu_rays: 1.0,
            gpu_seen: 0.0,
            gpu_prev: None,
            gpu_slope: None,
            gpu_walks: 0,
        };
        s.make_walker();
        s
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Change quality or compute on the fly (takes effect next tick).
    pub fn set_config(&mut self, cfg: Config) {
        let cfg = cfg.engine();
        let rebuild = cfg.compute != self.cfg.compute || cfg.v4 != self.cfg.v4 || cfg.gpu_budget != self.cfg.gpu_budget;
        self.cfg = cfg;
        self.tracer.v4 = cfg.v4;
        self.tracer.v5 = cfg.v5;
        self.tracer.v6 = cfg.v6;
        self.tracer.v61 = cfg.v61;
        self.tracer.v62 = cfg.v62;
        self.tracer.portals = cfg.portals;
        self.tracer.guide = cfg.guide;
        self.tracer.clearance = cfg.clearance;
        self.tracer.threads = cfg.walk_threads;
        if rebuild {
            self.make_walker();
        }
    }

    /// Replace the level geometry (builds its acceleration structure:
    /// milliseconds for a building, ~a second for a town — do it at load).
    pub fn set_geometry(&mut self, geometry: Geometry) {
        let (mesh, grade) = geometry.build();
        self.mesh = mesh;
        self.grade = grade;
        // the smoothed fields belong to the old world
        self.tracer = Tracer::new(self.cfg.seed);
        self.tracer.v4 = self.cfg.v4;
        self.tracer.v5 = self.cfg.v5;
        self.tracer.v6 = self.cfg.v6;
        self.tracer.v61 = self.cfg.v61;
        self.tracer.v62 = self.cfg.v62;
        self.tracer.portals = self.cfg.portals;
        self.tracer.guide = self.cfg.guide;
        self.tracer.clearance = self.cfg.clearance;
        self.tracer.threads = self.cfg.walk_threads;
        self.tracer.debug = self.debug_paths;
        for s in &mut self.sources {
            s.traced = None;
        }
        self.make_walker();
    }

    /// Install an external ray walker built on the CURRENT geometry (a
    /// web host's WebGPU proxy, a custom backend); None returns to the
    /// configured compute. Replaced again by `set_geometry`.
    pub fn set_walker(&mut self, walker: Option<Box<dyn V3Walker>>) {
        match walker {
            Some(w) => self.walker = Some(w),
            None => self.make_walker(),
        }
    }

    /// Is the walk running on the GPU (an adapter was found)?
    pub fn gpu_active(&self) -> bool {
        self.walker.is_some()
    }

    /// The acoustic mesh the tracer walks (its BVH), as built from the
    /// geometry — for a host driving the walk on its own GPU (the web's
    /// WebGPU, through `set_walker`).
    pub fn mesh(&self) -> &omg_core::mesh::Mesh {
        &self.mesh
    }

    /// Wall-clock of the last finished GPU walk, submit to read-back (ms;
    /// 0 on the CPU).
    pub fn gpu_walk_ms(&self) -> f32 {
        self.walker.as_ref().map_or(0.0, |w| w.last_ms())
    }

    /// A walk every tick within the GPU's share: as many of the tier's GPU
    /// rays as that leaves room for. A walk's time is its rays' (when the
    /// GPU is full) plus its bounces' in a row (one thread's latency, the
    /// same for 96 rays as for 6144): fewer rays only help in the first
    /// case, so the rays shrink only while the time measurably follows
    /// them — else the walk keeps them all and lands when it can.
    fn size_gpu_walk(&mut self) {
        let ms = self.walker.as_ref().map_or(0.0, |w| w.last_ms());
        if ms <= 0.0 || ms == self.gpu_seen {
            return;
        }
        self.gpu_seen = ms;
        self.gpu_walks = self.gpu_walks.wrapping_add(1);
        // the time one share of rays costs, from two walks at different
        // shares (the results land a walk late: shares move slowly)
        if let Some((s0, m0)) = self.gpu_prev.filter(|(s0, _)| (s0 - self.gpu_rays).abs() > 0.05) {
            self.gpu_slope = Some(((ms - m0) / (self.gpu_rays - s0)).max(0.0));
        }
        self.gpu_prev = Some((self.gpu_rays, ms));
        let fits = GPU_TICK_SHARE * self.cfg.gpu_budget * self.cfg.tick_s * 1e3;
        // re-measure now and then: the scene (and with it the slope) moves
        let probe = self.gpu_slope.is_none() || self.gpu_walks % GPU_SIZING_PROBE == 0;
        self.gpu_rays = if ms <= fits {
            self.gpu_rays * 1.1
        } else if probe || self.gpu_slope.is_some_and(|k| k * self.gpu_rays > 0.25 * ms) {
            self.gpu_rays * (fits / ms).clamp(0.7, 0.9)
        } else {
            1.0
        }
        .clamp(GPU_RAYS_MIN, 1.0);
    }

    /// The share of the tier's GPU rays the live walk carries (1: all; it
    /// shrinks so a walk lands every tick within the GPU's share).
    pub fn gpu_rays_share(&self) -> f32 {
        self.gpu_rays
    }

    /// The last GPU walk's longest single dispatch (ms): how long it held
    /// the GPU at once (a live walk is several short dispatches).
    pub fn gpu_walk_max_ms(&self) -> f32 {
        self.walker.as_ref().map_or(0.0, |w| w.last_max_ms())
    }

    fn make_walker(&mut self) {
        self.walker = None;
        if matches!(self.cfg.compute, Compute::Gpu | Compute::GpuSync) {
            #[cfg(feature = "gpu")]
            {
                let w = if self.cfg.compute == Compute::GpuSync {
                    omg_gpu::GpuV3Walker::new(&self.mesh)
                } else {
                    omg_gpu::GpuV3Walker::live(&self.mesh, self.cfg.gpu_budget)
                };
                self.walker = w.map(|w| Box::new(w) as Box<dyn V3Walker>);
            }
        }
    }

    // ---- sources ---------------------------------------------------------

    pub fn add_source(&mut self, desc: SourceDesc) -> SourceId {
        let id = SourceId(self.next_source);
        self.add_source_as(id, desc);
        id
    }

    /// `add_source` with an id chosen by the caller (hosts that hand out
    /// ids before the simulation sees them, e.g. across a thread).
    pub(crate) fn add_source_as(&mut self, id: SourceId, desc: SourceDesc) {
        self.next_source = self.next_source.max(id.0 + 1);
        self.sources.push(Src {
            id,
            pos: v3(desc.position),
            gain: desc.gain.max(0.0),
            active: desc.active,
            block: ParamBlock::default(),
            traced: None,
            dense: None,
            kernel: None,
            repriced: [[1.0; NBANDS]; 2],
            track: VecDeque::from([(self.clock, v3(desc.position))]),
            seen: (self.clock, v3(desc.position)),
            since: f64::NEG_INFINITY,
            off_at: (!desc.active).then_some(f64::NEG_INFINITY),
            ghost: None,
            until: f64::INFINITY,
        });
    }

    /// Move a source and/or switch it on or off.
    pub fn set_source(&mut self, id: SourceId, position: [f32; 3], active: bool) {
        self.set_source_at(id, position, active, self.clock);
    }

    /// `set_source` at `time` (the clock `set_time` keeps): when the place
    /// was current — a game whose physics moves a source at a lower rate
    /// than it renders sends the same place a few times, and it has been
    /// there since the physics step (`Engine::set_source` stamps a
    /// repeated place with its first send). The clock itself stays.
    pub fn set_source_at(&mut self, id: SourceId, position: [f32; 3], active: bool, time: f64) {
        let now = time;
        let ear = v3(self.listener.position);
        let Some(i) = self.sources.iter().position(|s| s.id == id && s.ghost.is_none()) else { return };
        let p = v3(position);
        let s = &mut self.sources[i];
        // a jump (faster than sound: a teleport, a reused slot): what it
        // emitted at the old place plays on from there in a ghost, until
        // it has all arrived and rung out
        let (t_seen, q) = s.seen;
        let jumped = (p - q).length() > omg_core::SPEED_OF_SOUND * (now - t_seen).max(1e-3) as f32;
        if jumped && s.off_at.is_none_or(|t| now - t < ring_out(s.pos, ear)) {
            let gid = SourceId(GHOST_ID_BASE + self.next_ghost);
            self.next_ghost = self.next_ghost.wrapping_add(1) % GHOST_ID_SPAN;
            // the ghost is the place it left: what was learnt there is its
            let ghost = Src {
                id: gid,
                pos: s.pos,
                gain: s.gain,
                active: true,
                block: s.block.clone(),
                traced: s.traced,
                dense: s.dense.take(),
                kernel: s.kernel.take(),
                repriced: s.repriced,
                track: s.track.clone(),
                seen: s.seen,
                since: s.since,
                off_at: s.off_at,
                ghost: Some((id, now + ring_out(s.pos, ear) as f64)),
                until: now,
            };
            s.track = VecDeque::from([(now, p)]);
            s.seen = (now, p);
            s.since = now;
            self.tracer.fork(id.0, gid.0);
            self.new_ghosts.push((gid, id));
            self.sources.push(ghost);
        }
        let s = &mut self.sources[i];
        s.pos = p;
        if active {
            s.off_at = None;
        } else if s.active || s.off_at.is_none() {
            // (never traced while on: nothing of it is on its way; off
            // now, whenever it stood there since)
            s.off_at = Some(if s.traced.is_some() { self.clock } else { f64::NEG_INFINITY });
        }
        s.active = active;
        record(&mut s.track, s.seen, now, s.pos);
        s.seen = (now, s.pos);
    }

    /// The host's clock (s, any steady clock — the one `set_listener_at`
    /// stamps poses with): positions set from now on were current at
    /// `time`. Where each source was when is what lets every path reach it
    /// where it emitted what arrives (a car 1 km off is heard from where it
    /// was ~3 s ago). Without it the simulation counts a tick period per
    /// tick.
    pub fn set_time(&mut self, time: f64) {
        self.clock = time;
        self.clock_given = true;
    }

    /// Is the source traced and heard (see `SourceDesc::active`)?
    pub fn is_active(&self, id: SourceId) -> bool {
        self.sources.iter().any(|s| s.id == id && s.active)
    }

    pub fn set_source_gain(&mut self, id: SourceId, gain: f32) {
        if let Some(s) = self.sources.iter_mut().find(|s| s.id == id) {
            s.gain = gain.max(0.0);
        }
    }

    pub fn remove_source(&mut self, id: SourceId) {
        self.sources.retain(|s| s.id != id);
        self.tracer.reset(id.0);
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    // ---- objects ---------------------------------------------------------

    /// Add a box (axis-aligned, world frame) — see `ObjectId`.
    pub fn add_object(&mut self, min: [f32; 3], max: [f32; 3], material: Material) -> ObjectId {
        let id = ObjectId(self.next_object);
        self.add_object_as(id, min, max, material);
        id
    }

    pub(crate) fn add_object_as(&mut self, id: ObjectId, min: [f32; 3], max: [f32; 3], material: Material) {
        self.next_object = self.next_object.max(id.0 + 1);
        let (a, b) = (v3(min), v3(max));
        self.objects.push(Obj { id, min: a.min(b), max: a.max(b), material, enabled: true, aperture: None });
    }

    /// Move / resize an object (a door swinging shut: its box grows
    /// across the doorway).
    pub fn set_object(&mut self, id: ObjectId, min: [f32; 3], max: [f32; 3]) {
        if let Some(o) = self.objects.iter_mut().find(|o| o.id == id) {
            let (a, b) = (v3(min), v3(max));
            o.min = a.min(b);
            o.max = a.max(b);
        }
    }

    pub fn set_object_material(&mut self, id: ObjectId, material: Material) {
        if let Some(o) = self.objects.iter_mut().find(|o| o.id == id) {
            o.material = material;
        }
    }

    /// The opening the object fills (a doorway for its leaf, a window for
    /// its curtain): a box, thin across it, that the object's box covers
    /// in part or whole as it moves (disabled: none of it). v6.2 prices the
    /// paths through the opening by how much is covered (`Config::v62`);
    /// keep the object (disable it) while the opening is bare, so what
    /// was learnt about it stays.
    pub fn set_object_aperture(&mut self, id: ObjectId, min: [f32; 3], max: [f32; 3]) {
        if let Some(o) = self.objects.iter_mut().find(|o| o.id == id) {
            let (a, b) = (v3(min), v3(max));
            o.aperture = Some((a.min(b), a.max(b)));
        }
    }

    /// Disabled objects are acoustically absent (a door fully open).
    pub fn set_object_enabled(&mut self, id: ObjectId, enabled: bool) {
        if let Some(o) = self.objects.iter_mut().find(|o| o.id == id) {
            o.enabled = enabled;
        }
    }

    pub fn remove_object(&mut self, id: ObjectId) {
        self.objects.retain(|o| o.id != id);
    }

    // ---- listener --------------------------------------------------------

    /// Where the listener is and where they look (see `Listener`). The
    /// sim traces from the position; the orientation travels in the frame
    /// to the mixer, which turns the sound field (between frames, call
    /// `Mixer::set_head` directly for the fastest response). The pose is
    /// current at the clock as it stands (`set_time`).
    pub fn set_listener(&mut self, position: [f32; 3], yaw: f32, pitch: f32, roll: f32) {
        self.listener = Listener { position, yaw, pitch, roll };
    }

    /// `set_listener` with the time (seconds, any steady clock) at which
    /// the pose was current — for a game, the physics step that moved the
    /// listener: the clock is set to it (`set_time`). Frames carry their
    /// instant, and the mixer moves every path's delay over the time that
    /// really passed between two traced instants: steady Doppler even
    /// though ticks sample the motion irregularly.
    pub fn set_listener_at(&mut self, position: [f32; 3], yaw: f32, pitch: f32, roll: f32, time: f64) {
        self.listener = Listener { position, yaw, pitch, roll };
        self.set_time(time);
    }

    /// Sources traced in the last tick (≤ `max_traced_sources`).
    pub fn traced_last_tick(&self) -> usize {
        self.traced_last
    }

    /// Keep each tick's strongest traced paths (up to 64, early
    /// reflections, edge bends and solved echo chains) for a debug view.
    /// Off by default: then the tracer records nothing.
    pub fn set_debug_paths(&mut self, on: bool) {
        self.debug_paths = on;
        self.tracer.debug = on;
        if !on {
            self.tracer.clear_debug_paths();
        }
    }

    pub fn debug_paths_enabled(&self) -> bool {
        self.debug_paths
    }

    /// The exactly solved paths (direct, edge bends, mirror chains) and
    /// the last tick's strongest sampled ones, strongest first (empty
    /// when off). Sources that did not trace that tick keep no sampled
    /// paths in it.
    pub fn debug_paths(&self) -> Vec<DebugPath> {
        let pts = |v: &[omg_core::vec3::Vec3]| v.iter().map(|p| [p.x, p.y, p.z]).collect();
        let mut out: Vec<DebugPath> = self
            .tracer
            .exact_paths()
            .map(|p| DebugPath {
                source: SourceId(p.source),
                energy: p.energy,
                points: pts(&p.points),
                kind: [PathKind::Direct, PathKind::Bend, PathKind::Mirror][p.kind.min(2) as usize],
                key: p.key,
            })
            .collect();
        out.extend(self.tracer.debug_paths.iter().map(|(id, e, v)| DebugPath {
            source: SourceId(*id),
            energy: *e,
            points: pts(v),
            kind: PathKind::Sampled,
            key: 0,
        }));
        out.extend(self.tracer.debug_lit.iter().map(|(id, e, v)| DebugPath {
            source: SourceId(*id),
            energy: *e,
            points: pts(v),
            kind: PathKind::Portal,
            key: 0,
        }));
        out.sort_by(|a, b| b.energy.total_cmp(&a.energy));
        out
    }

    /// v4: how far a source's kernel is from its current response (the
    /// envelope's L1 over its total, `DenseResponse::change_from`: 0 the
    /// same, 2 disjoint); None without a kernel.
    pub fn kernel_staleness(&self, id: SourceId) -> Option<f32> {
        let s = self.sources.iter().find(|s| s.id == id)?;
        let (_, from, _, _) = s.kernel.as_ref()?;
        Some(from.change_from(s.dense.as_ref()?))
    }

    /// v4: a source's latest dense (scattered) response, unit source
    /// (see `DenseResponse`); None in v3 or before its first trace.
    /// v4: the dense response's measured per-tick scatter, per history
    /// window (variance of ln fresh/mean: the trace's own noise).
    pub fn dense_scatter(&self, id: SourceId) -> Option<[f32; omg_core::v3::HISTORY_WINDOWS]> {
        self.tracer.dense_scatter(id.0)
    }

    /// The last tick's tracer cost by part (ms; zeros in wasm).
    pub fn tick_costs(&self) -> omg_core::v3::TickCosts {
        self.tracer.costs
    }

    /// What the last tick's ray walk deposited for `id` (per band, before
    /// any averaging over ticks): the raw estimate, for comparing
    /// estimators (CPU walks; None for an unknown source).
    pub fn walked_energy(&self, id: SourceId) -> Option<[f32; omg_core::NBANDS]> {
        self.tracer.walked.iter().find(|(s, _)| *s == id.0).map(|(_, e)| *e)
    }

    pub fn dense_response(&self, id: SourceId) -> Option<&DenseResponse> {
        self.sources.iter().find(|s| s.id == id).and_then(|s| s.dense.as_ref())
    }

    /// v4: the response a source's current kernel was built from (it is
    /// rebuilt when the response has changed audibly since).
    pub fn kernel_response(&self, id: SourceId) -> Option<&DenseResponse> {
        self.sources.iter().find(|s| s.id == id).and_then(|s| s.kernel.as_ref().map(|k| &k.1))
    }

    /// v4: (re)build the convolution kernels of the members (the
    /// `max_kernels` loudest) whose dense response changed audibly since
    /// theirs was built — the most changed (× loudness) first: queued on
    /// the kernel thread, or one at a time in the tick.
    /// Kernels the thread finished since the last look, onto their
    /// sources. True when one landed.
    fn collect_kernels(&mut self) -> bool {
        let mut landed = false;
        if let Some(w) = &mut self.kernels {
            while let Ok((id, k, resp, gain, at)) = w.done.try_recv() {
                w.busy.retain(|b| *b != id);
                if let Some(s) = self.sources.iter_mut().find(|s| s.id == id && s.dense.is_some()) {
                    s.kernel = Some((k, resp, gain, at));
                    landed = true;
                }
            }
        }
        landed
    }

    fn build_kernels(&mut self, tick: u64) {
        // finished builds land first
        if self.kernels.is_some() {
            self.collect_kernels();
        } else if self.cfg.kernel_thread {
            self.kernels = KernelWorker::start();
        }
        // the members: the loudest at the ears (energy of the block's
        // paths and its scattered field), a member counting double
        let loud = |s: &Src| -> f32 {
            let taps: f32 = s.block.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum();
            taps + s.block.reverb.level[1] * s.block.reverb.level[1] + s.block.remote.as_ref().map_or(0.0, |r| r.send[1] * r.send[1])
        };
        let mut cand: Vec<(f32, SourceId)> = self
            .sources
            .iter()
            .filter(|s| s.dense.as_ref().is_some_and(|d| !d.bins.is_empty()))
            .map(|s| (loud(s) * if self.kernel_members.contains(&s.id) { 2.0 } else { 1.0 }, s.id))
            .filter(|c| c.0 > 0.0)
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0));
        cand.truncate(self.cfg.max_kernels);
        self.kernel_members.clear();
        self.kernel_members.extend(cand.iter().map(|c| c.1));
        let members = &self.kernel_members;
        for s in &mut self.sources {
            let stale = match (&s.kernel, &s.dense) {
                (Some((_, from, _, _)), Some(d)) => from.change_from(d) > KERNEL_STALE,
                _ => false,
            };
            if !members.contains(&s.id) || stale {
                s.kernel = None;
            }
        }
        if self.kernel_job.as_ref().is_some_and(|j| !members.contains(&j.0)) {
            self.kernel_job = None;
        }
        // which members need a kernel, the most needed first
        let ear = v3(self.listener.position);
        let busy: &[SourceId] = self.kernels.as_ref().map_or(&[], |w| &w.busy);
        let mut todo: Vec<(f32, usize)> = Vec::new();
        for (i, s) in self.sources.iter().enumerate() {
            let Some(d) = &s.dense else { continue };
            if !members.contains(&s.id) || busy.contains(&s.id) {
                continue;
            }
            let change = match &s.kernel {
                None => f32::INFINITY,
                Some((_, from, g, _)) => {
                    let gain = if *g > 0.0 { (s.gain / g - 1.0).abs() } else { 1.0 };
                    let c = from.difference_from(d).max(gain);
                    // small drifts still land, just later
                    if c > KERNEL_CHANGE {
                        c
                    } else {
                        continue;
                    }
                }
            };
            let loud = s.gain / (s.pos - ear).length().max(1.0);
            todo.push((change.min(1e6) * loud, i));
        }
        todo.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (sr, max_s) = (self.cfg.sample_rate, self.cfg.max_kernel_s);
        if let Some(w) = &mut self.kernels {
            // the thread's queue stays short: the most needed first
            for &(_, i) in &todo {
                if w.busy.len() >= KERNEL_QUEUE {
                    break;
                }
                let s = &mut self.sources[i];
                let d = s.dense.clone().expect("listed above");
                if w.jobs.send((s.id, d, s.gain, sr, max_s, tick)).is_ok() {
                    w.busy.push(s.id);
                }
            }
            return;
        }
        // in the tick: one kernel at a time, a step a tick (no clock: the
        // same work on any host)
        if self.kernel_job.is_none() && (self.kernel_sent == 0 || tick >= self.kernel_sent + self.cfg.kernel_every_ticks) {
            if let Some(&(_, i)) = todo.first() {
                let s = &mut self.sources[i];
                let d = s.dense.clone().expect("listed above");
                let b = KernelBuilder::new(&d, sr, s.gain, s.id.0, max_s);
                self.kernel_job = Some((s.id, b, d, s.gain));
            }
        }
        if let Some((_, b, _, _)) = &mut self.kernel_job {
            if b.step(self.cfg.kernel_step_samples.max(1)) {
                let (id, b, d, g) = self.kernel_job.take().expect("checked");
                if let Some(s) = self.sources.iter_mut().find(|s| s.id == id && s.dense.is_some()) {
                    s.kernel = Some((Arc::new(b.finish()), d, g, tick));
                }
                self.kernel_sent = tick;
            }
        }
    }

    /// v4: wait for every kernel build in flight and take it in (offline
    /// renders and tests: the next `tick` carries them).
    pub fn flush_kernels(&mut self) {
        let Some(w) = &mut self.kernels else { return };
        while !w.busy.is_empty() {
            let Ok((id, k, resp, gain, at)) = w.done.recv() else { break };
            w.busy.retain(|b| *b != id);
            if let Some(s) = self.sources.iter_mut().find(|s| s.id == id && s.dense.is_some()) {
                s.kernel = Some((k, resp, gain, at));
            }
        }
    }

    // ---- the tick --------------------------------------------------------

    /// Trace once and return every source's paths.
    pub fn tick(&mut self) -> Frame {
        if !self.clock_given {
            self.clock += self.cfg.tick_s as f64;
        }
        self.tick_no += 1;
        let tick = self.tick_no;
        self.frame_no += 1;
        let version = self.frame_no;
        let ear = v3(self.listener.position);
        self.exact_at = Some((ear, self.sources.iter().map(|s| s.pos).collect(), self.objects_sig(), self.clock));

        self.collect_panels();

        // which sources trace this tick: all, or the most overdue by
        // loudness × age (a moved source is overdue at once)
        self.picks.clear();
        for (i, s) in self.sources.iter().enumerate() {
            if !sounding(s, self.clock, ear) {
                continue;
            }
            let score = match s.traced {
                None => f32::INFINITY,
                Some((p, t)) => {
                    let d = (s.pos - ear).length().max(1.0);
                    let age = (tick - t) as f32;
                    let moved = if (s.pos - p).length() > MOVED_M { MOVED_BOOST } else { 1.0 };
                    (s.gain / d) * age * moved
                }
            };
            self.picks.push((score, i));
        }
        let k = self.cfg.max_traced_sources.max(1);
        if self.picks.len() > k {
            self.picks.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            self.picks.truncate(k);
        }
        self.picks.sort_unstable_by_key(|p| p.1);
        self.traced_last = self.picks.len();

        let srcs: Vec<Source> = self
            .picks
            .iter()
            .map(|&(_, i)| {
                let s = &self.sources[i];
                Source { id: s.id.0, pos: s.pos, weight: s.gain * s.gain }
            })
            .collect();
        let gpu = self.walker.is_some();
        let mut budget = self.cfg.quality.budget(gpu);
        if gpu && self.cfg.compute == Compute::Gpu {
            self.size_gpu_walk();
            budget.rays = ((budget.rays as f32 * self.gpu_rays) as u32).max(64);
        }
        let cpu_budget = self.cfg.quality.budget(false);
        let world = World { grade: self.grade, pockets: &[], apertures: &self.apertures, ..World::new(&self.mesh, &self.panels) };
        let traced: Vec<&Src> = self.picks.iter().map(|&(_, i)| &self.sources[i]).collect();
        let mo = motion(self.clock, &traced);
        let onsets: Vec<f32> = traced
            .iter()
            .enumerate()
            .map(|(k, s)| (mo.emit(k, s.pos, ear, 0.0).unwrap_or(s.pos) - ear).length() / omg_core::SPEED_OF_SOUND)
            .collect();
        self.tracer.set_motion(mo);
        let walker = self.walker.as_mut().map(|w| &mut **w as &mut dyn V3Walker);
        let fields = trace_with(&mut self.tracer, &world, ear, &srcs, budget, cpu_budget, walker);

        // fields → blocks (the generic half of the demo's WorldSim::v3_blocks)
        let mut first_traced = Vec::new();
        for ((&(_, i), f), onset_s) in self.picks.iter().zip(fields).zip(onsets) {
            let s = &mut self.sources[i];
            if s.traced.is_none() {
                first_traced.push(s.id);
            }
            let g = s.gain;
            let mut pb = ParamBlock { version, onset_s, ..Default::default() };
            pb.taps.reserve(f.taps.len());
            for t in f.taps {
                pb.taps.push(Tap { gains: core::array::from_fn(|b| t.gains[b] * g), ..t });
            }
            pb.reverb.rt60 = f.reverb.rt60;
            let mut lv = f.reverb.level;
            if let Some(d) = f.dense {
                // v4: the scattered response is convolved (`kernels`); the
                // block also carries it as a parametric reverb for a mixer
                // that does not convolve this source (its convolution
                // budget): its whole energy, arriving on average as late
                // as the response's (the decay that gives the network's
                // output the response's centre time — a field heard
                // through a door arrives sooner than its tail decays)
                let ts = d.centre_time();
                pb.reverb.rt60 = core::array::from_fn(|b| omg_dsp::fdn::rt60_for_centre_time(ts[b]));
                lv = core::array::from_fn(|b| d.bins.iter().map(|x| x[b].max(0.0)).sum::<f32>().sqrt());
                s.dense = Some(d);
            }
            // a one-sided late field plays from where it comes (the
            // doorway, the lit facade); the rest feeds the diffuse FDN
            let a = f.aniso.clamp(0.0, 0.9);
            pb.reverb.level = core::array::from_fn(|b| lv[b] * (1.0 - a).sqrt() * g);
            if lv[1] * lv[1] * a > 1e-10 {
                let d = f.late_dir;
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
                pb.remote = Some(RemoteReverb {
                    rt60: pb.reverb.rt60,
                    send: core::array::from_fn(|b| lv[b] * a.sqrt() * g),
                    sh: omg_dsp::ambi::encode_gains([d[0] / l, d[1] / l, d[2] / l]),
                });
            }
            // the dense early field (v6.2), at the source's gain
            pb.early = omg_core::params::DenseEarly { level: f.early.level.map(|l| l * g), ..f.early };
            s.block = pb;
            s.repriced = [[1.0; NBANDS]; 2];
            s.traced = Some((s.pos, tick));
        }
        let now = self.clock;
        for s in &mut self.sources {
            let on = sounding(s, now, ear);
            if !on && (!s.block.taps.is_empty() || s.block.reverb.level[1] > 0.0 || s.block.remote.is_some() || s.block.early.level[1] > 0.0) {
                // rung out: silent from now on (the mixer fades it out)
                s.block = ParamBlock { version, ..Default::default() };
                s.traced = None;
            }
            if !self.cfg.v4 || !on {
                s.dense = None;
                s.kernel = None;
            }
        }
        // ghosts that have rung out leave
        let gone: Vec<SourceId> = self.sources.iter().filter(|s| s.ghost.is_some_and(|g| now > g.1)).map(|s| s.id).collect();
        for id in gone {
            self.sources.retain(|s| s.id != id);
            self.tracer.reset(id.0);
        }
        if self.cfg.v4 {
            self.build_kernels(tick);
        }

        let sky: Vec<SkyCell> = self.tracer.sky().into_iter().map(|(dir, energy)| SkyCell { dir, energy }).collect();
        let env = crate::env::environment(&sky, &self.outdoor_context(ear));
        self.last_sky = sky.clone();
        self.last_env = env.clone();
        Frame {
            version,
            listener: self.listener,
            time: self.clock_given.then_some(self.clock),
            blocks: self.sources.iter().map(|s| (s.id, s.block.clone())).collect(),
            kernels: self.sources.iter().filter_map(|s| s.kernel.as_ref().map(|k| (s.id, k.0.clone()))).collect(),
            sky,
            first_traced,
            ghosts: std::mem::take(&mut self.new_ghosts),
            env,
        }
    }

    /// Between full ticks (as often as a host likes — every frame, or
    /// faster): the exact paths for where the listener and the sources are
    /// now — the direct path, the edge bends re-timed through their bend
    /// points, every cached mirror chain re-solved — in the last full
    /// tick's blocks; the stochastic part (the reverb, the dense response)
    /// stays the full tick's. Close, moving sounds follow at this rate
    /// instead of the tick's. None when nothing moved since the last frame.
    pub fn tick_exact(&mut self) -> Option<Frame> {
        let ear = v3(self.listener.position);
        let sig = self.objects_sig();
        let now = self.clock;
        // kernels finished since the last frame go out with this one (a
        // response that changed is heard as soon as its kernel exists,
        // not at the next tick)
        let landed = self.collect_kernels();
        // something moved: the listener, a source, or an object (a door
        // shutting is heard at the next exact frame, not the next tick) —
        // or time passed since a source moved lately: what arrives now
        // left it along its track, so its paths go on moving after it
        // stopped (the same instant again: nothing new to trace)
        let moved = match &self.exact_at {
            Some((l, ps, o, t)) => {
                (*l - ear).length() > 1e-4
                    || *o != sig
                    || ps.len() != self.sources.len()
                    || self.sources.iter().zip(ps).any(|(s, p)| (s.pos - *p).length() > 1e-4)
                    || (now > *t
                        && self.sources.iter().any(|s| s.track.len() > 1 && s.track.back().is_some_and(|b| b.0 > now - RECENT_MOVE_S)))
            }
            None => true,
        };
        // (a source jumped: the next full tick traces its ghost and hands
        // both over together)
        if (!moved && !landed) || self.frame_no == 0 || !self.new_ghosts.is_empty() {
            return None;
        }
        self.exact_at = Some((ear, self.sources.iter().map(|s| s.pos).collect(), sig, now));
        self.collect_panels();
        let world = World { grade: self.grade, pockets: &[], apertures: &self.apertures, ..World::new(&self.mesh, &self.panels) };
        let idx: Vec<usize> =
            (0..self.sources.len()).filter(|&i| sounding(&self.sources[i], now, ear) && self.sources[i].traced.is_some()).collect();
        let srcs: Vec<Source> = idx
            .iter()
            .map(|&i| {
                let s = &self.sources[i];
                Source { id: s.id.0, pos: s.pos, weight: s.gain * s.gain }
            })
            .collect();
        let traced: Vec<&Src> = idx.iter().map(|&i| &self.sources[i]).collect();
        let mo = motion(now, &traced);
        let onsets: Vec<f32> = traced
            .iter()
            .enumerate()
            .map(|(k, s)| (mo.emit(k, s.pos, ear, 0.0).unwrap_or(s.pos) - ear).length() / omg_core::SPEED_OF_SOUND)
            .collect();
        self.tracer.set_motion(mo);
        let taps = self.tracer.exact_taps(&world, ear, &srcs);
        self.frame_no += 1;
        let version = self.frame_no;
        for (k, &i) in idx.iter().enumerate() {
            let s = &mut self.sources[i];
            let g = s.gain;
            let onset_s = onsets[k];
            let mut changed = s.block.onset_s != onset_s;
            s.block.onset_s = onset_s;
            for t in s.block.taps.iter_mut() {
                if let Some(n) = taps[k].iter().find(|n| n.key == t.key) {
                    *t = Tap { gains: core::array::from_fn(|b| n.gains[b] * g), ..*n };
                    changed = true;
                }
            }
            // v6.2: the estimated parts priced at the apertures as they
            // are now (a door on its way between two ticks)
            if let Some(r) = self.tracer.reprice(s.id.0, &self.apertures) {
                let step = |part: usize| -> [f32; NBANDS] { core::array::from_fn(|b| (r[part][b] / s.repriced[part][b].max(1e-9)).sqrt()) };
                let (early, late) = (step(0), step(1));
                if early.iter().chain(&late).any(|&x| (x - 1.0).abs() > 1e-4) {
                    for t in s.block.taps.iter_mut().filter(|t| omg_core::v3::estimated_key(t.key)) {
                        for b in 0..NBANDS {
                            t.gains[b] *= early[b];
                        }
                    }
                    for b in 0..NBANDS {
                        s.block.early.level[b] *= early[b];
                    }
                    for b in 0..NBANDS {
                        s.block.reverb.level[b] *= late[b];
                    }
                    if let Some(rm) = s.block.remote.as_mut() {
                        for b in 0..NBANDS {
                            rm.send[b] *= late[b];
                        }
                    }
                    s.repriced = r;
                    changed = true;
                }
            }
            if changed {
                s.block.version = version;
            }
        }
        Some(Frame {
            version,
            listener: self.listener,
            time: self.clock_given.then_some(self.clock),
            blocks: self.sources.iter().map(|s| (s.id, s.block.clone())).collect(),
            kernels: self.sources.iter().filter_map(|s| s.kernel.as_ref().map(|k| (s.id, k.0.clone()))).collect(),
            sky: self.last_sky.clone(),
            first_traced: Vec::new(),
            ghosts: Vec::new(),
            env: self.last_env.clone(),
        })
    }

    /// The enabled objects as the walk's panels, and (v6.2) every object's
    /// aperture as it is.
    fn collect_panels(&mut self) {
        self.panels.clear();
        for o in self.objects.iter().filter(|o| o.enabled) {
            self.panels.push((o.min, o.max, o.material));
        }
        self.apertures.clear();
        if self.cfg.v62 {
            self.apertures.extend(self.objects.iter().filter_map(Obj::aperture));
        }
    }

    /// A signature of the enabled objects (where they are, what they are).
    fn objects_sig(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for o in self.objects.iter().filter(|o| o.enabled) {
            for x in [o.min.x, o.min.y, o.min.z, o.max.x, o.max.y, o.max.z, o.material.transmission[1]] {
                h = (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }

    /// Around the listener, for the outdoor field: a roof probe
    /// (something overhead with open sky above it: rain drums on it) and
    /// the glass panes near by (rain patters on them).
    fn outdoor_context(&self, ear: Vec3) -> crate::env::EnvContext {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let roof = match self.mesh.raycast(ear + up * 0.3, up) {
            Some((t, _)) if t < 40.0 => {
                let above = ear + up * (0.3 + t + 0.4);
                self.mesh.raycast(above, up).is_none()
            }
            _ => false,
        };
        let windows = self
            .objects
            .iter()
            .filter(|o| o.enabled && o.material == Material::GLASS)
            .filter_map(|o| {
                let v = (o.min + o.max) * 0.5 - ear;
                let d = v.length();
                (d < crate::env::WINDOW_REACH).then(|| {
                    let k = 1.0 / d.max(1e-3);
                    ([v.x * k, v.y * k, v.z * k], d)
                })
            })
            .collect();
        crate::env::EnvContext { roof, windows }
    }
}
