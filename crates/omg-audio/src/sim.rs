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
use omg_core::v3::{Budget, DenseResponse, Source, Tracer, World};
use omg_dsp::dense::{build_kernel_max, DenseKernel, KernelBuilder};
use std::sync::Arc;
use omg_core::vec3::Vec3;

/// How much ray tracing a tick may do. Each step roughly doubles the
/// cost; `High` is the demo's default.
#[derive(Clone, Copy, Debug)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
    /// Full control over the tracer's budget.
    Custom(Budget),
}

/// Where the ray walk runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    Cpu,
    /// The walk on the GPU through wgpu (feature `gpu`): 8× the rays of
    /// the CPU tier for the same tick. Results arrive one tick late (the
    /// GPU never blocks the tick). Without an adapter, without the
    /// feature, or for a tick beyond the kernel's caps (> 64 traced
    /// sources or > 64 objects), the tick runs on the CPU instead.
    Gpu,
    /// `Gpu`, but each tick waits for its own walk (offline renders,
    /// tests: deterministic timing, no frame of latency).
    GpuSync,
}

impl Quality {
    /// The tracer budget of this tier on the CPU (`gpu = false`) or GPU.
    pub fn budget(self, gpu: bool) -> Budget {
        let (rays, bounces, late) = match self {
            Quality::Low => (512, 3, 48),
            Quality::Medium => (768, 4, 96),
            Quality::High => (1024, 6, 128),
            Quality::Ultra => (2048, 8, 256),
            Quality::Custom(b) => return b,
        };
        let (rays, late) = if gpu { ((rays * 8).min(65536), (late * 8).min(8192)) } else { (rays, late) };
        Budget {
            rays,
            bounces,
            k: 0.06,
            ema: 0.35,
            nee: 4,
            late_rays: late,
            late_ema: 0.1,
            late_nee: 1,
            diffraction: true,
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            quality: Quality::High,
            compute: Compute::Cpu,
            max_traced_sources: 64,
            seed: 0x0e3_5eed,
            v4: true,
            sample_rate: 48_000.0,
            kernel_thread: true,
            max_kernels: 64,
            max_kernel_s: omg_dsp::dense::MAX_KERNEL_S,
            kernel_step_samples: 4 * 4096,
            kernel_every_ticks: 1,
            gpu_budget: 0.2,
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
}

struct Obj {
    id: ObjectId,
    min: Vec3,
    max: Vec3,
    material: Material,
    enabled: bool,
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
    listener_time: Option<f64>,
    tick_no: u64,
    next_source: u32,
    next_object: u32,
    panels: Vec<(Vec3, Vec3, Material)>,
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

/// v4 kernel rebuilds: a dense response whose envelope moved by this
/// share (L1 of the mid band over its total, ~0.6 dB) is rebuilt; a
/// quarter of it still lands after `KERNEL_REFRESH_TICKS`.
const KERNEL_CHANGE: f32 = 0.15;
const KERNEL_REFRESH_TICKS: u64 = 40;
/// A kernel this far from its source's current response (a quarter of the
/// envelope's energy moved: the listener stepped into another space) is
/// dropped at once — the block's parametric copy of the current response
/// plays until the new kernel lands, never the old room's.
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
        let mut s = Self {
            cfg,
            mesh: MeshBuilder::new().build(),
            grade: None,
            tracer: {
                let mut t = Tracer::new(cfg.seed);
                t.v4 = cfg.v4;
                t
            },
            walker: None,
            sources: Vec::new(),
            objects: Vec::new(),
            listener: Listener::default(),
            listener_time: None,
            tick_no: 0,
            next_source: 0,
            next_object: 0,
            panels: Vec::new(),
            picks: Vec::new(),
            traced_last: 0,
            debug_paths: false,
            kernels: None,
            kernel_members: Vec::new(),
            kernel_job: None,
            kernel_sent: 0,
        };
        s.make_walker();
        s
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Change quality or compute on the fly (takes effect next tick).
    pub fn set_config(&mut self, cfg: Config) {
        let rebuild = cfg.compute != self.cfg.compute || cfg.v4 != self.cfg.v4 || cfg.gpu_budget != self.cfg.gpu_budget;
        self.cfg = cfg;
        self.tracer.v4 = cfg.v4;
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
        });
    }

    /// Move a source and/or switch it on or off.
    pub fn set_source(&mut self, id: SourceId, position: [f32; 3], active: bool) {
        if let Some(s) = self.sources.iter_mut().find(|s| s.id == id) {
            s.pos = v3(position);
            s.active = active;
        }
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
        self.objects.push(Obj { id, min: a.min(b), max: a.max(b), material, enabled: true });
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
    /// `Mixer::set_head` directly for the fastest response).
    pub fn set_listener(&mut self, position: [f32; 3], yaw: f32, pitch: f32, roll: f32) {
        self.listener = Listener { position, yaw, pitch, roll };
        self.listener_time = None;
    }

    /// `set_listener` with the time (seconds, any steady clock) at which
    /// the pose was current — for a game, the physics step that moved the
    /// listener. Frames carry it, and the mixer moves every path's delay
    /// over the time that really passed between two traced poses: steady
    /// Doppler even though ticks sample the motion irregularly.
    pub fn set_listener_at(&mut self, position: [f32; 3], yaw: f32, pitch: f32, roll: f32, time: f64) {
        self.listener = Listener { position, yaw, pitch, roll };
        self.listener_time = Some(time);
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
    pub fn dense_response(&self, id: SourceId) -> Option<&DenseResponse> {
        self.sources.iter().find(|s| s.id == id).and_then(|s| s.dense.as_ref())
    }

    /// v4: (re)build the convolution kernels of the members (the
    /// `max_kernels` loudest) whose dense response changed audibly since
    /// theirs was built — the most changed (× loudness) first: queued on
    /// the kernel thread, or one at a time in the tick.
    fn build_kernels(&mut self, tick: u64) {
        // finished builds land first
        if let Some(w) = &mut self.kernels {
            while let Ok((id, k, resp, gain, at)) = w.done.try_recv() {
                w.busy.retain(|b| *b != id);
                if let Some(s) = self.sources.iter_mut().find(|s| s.id == id && s.dense.is_some()) {
                    s.kernel = Some((k, resp, gain, at));
                }
            }
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
            if !members.contains(&s.id) {
                s.kernel = None;
            }
            let stale = match (&s.kernel, &s.dense) {
                (Some((_, from, _, _)), Some(d)) => from.change_from(d) > KERNEL_STALE,
                _ => false,
            };
            if stale {
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
                Some((_, from, g, at)) => {
                    let gain = if *g > 0.0 { (s.gain / g - 1.0).abs() } else { 1.0 };
                    let c = from.change_from(d).max(gain);
                    // small drifts still land, just later
                    if c > KERNEL_CHANGE || (c > KERNEL_CHANGE / 4.0 && tick > at + KERNEL_REFRESH_TICKS) {
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
                let s = &self.sources[i];
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
                let s = &self.sources[i];
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
        self.tick_no += 1;
        let tick = self.tick_no;
        let ear = v3(self.listener.position);

        self.panels.clear();
        for o in self.objects.iter().filter(|o| o.enabled) {
            self.panels.push((o.min, o.max, o.material));
        }

        // which sources trace this tick: all, or the most overdue by
        // loudness × age (a moved source is overdue at once)
        self.picks.clear();
        for (i, s) in self.sources.iter().enumerate() {
            if !s.active {
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
        let budget = self.cfg.quality.budget(gpu);
        let cpu_budget = self.cfg.quality.budget(false);
        let world = World { grade: self.grade, pockets: &[], ..World::new(&self.mesh, &self.panels) };
        let walker = self.walker.as_mut().map(|w| &mut **w as &mut dyn V3Walker);
        let fields = trace_with(&mut self.tracer, &world, ear, &srcs, budget, cpu_budget, walker);

        // fields → blocks (the generic half of the demo's WorldSim::v3_blocks)
        let mut first_traced = Vec::new();
        for (&(_, i), f) in self.picks.iter().zip(fields) {
            let s = &mut self.sources[i];
            if s.traced.is_none() {
                first_traced.push(s.id);
            }
            let g = s.gain;
            let mut pb = ParamBlock { version: tick, ..Default::default() };
            pb.taps.reserve(f.taps.len());
            for t in f.taps {
                pb.taps.push(Tap { gains: core::array::from_fn(|b| t.gains[b] * g), ..t });
            }
            pb.reverb.rt60 = f.reverb.rt60;
            let mut lv = f.reverb.level;
            if let Some(d) = f.dense {
                // v4: the scattered response is convolved (`kernels`); the
                // block also carries it as a parametric reverb — its whole
                // energy and its measured decay — for a mixer that does
                // not convolve this source (its convolution budget)
                pb.reverb.rt60 = d.rt60();
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
            s.block = pb;
            s.traced = Some((s.pos, tick));
        }
        for s in &mut self.sources {
            if !s.active && (!s.block.taps.is_empty() || s.block.reverb.level[1] > 0.0 || s.block.remote.is_some()) {
                // silent from now on (the mixer fades it out)
                s.block = ParamBlock { version: tick, ..Default::default() };
                s.traced = None;
            }
            if !self.cfg.v4 || !s.active {
                s.dense = None;
                s.kernel = None;
            }
        }
        if self.cfg.v4 {
            self.build_kernels(tick);
        }

        let sky: Vec<SkyCell> = self.tracer.sky().into_iter().map(|(dir, energy)| SkyCell { dir, energy }).collect();
        let env = crate::env::environment(&sky, &self.outdoor_context(ear));
        Frame {
            version: tick,
            listener: self.listener,
            time: self.listener_time,
            blocks: self.sources.iter().map(|s| (s.id, s.block.clone())).collect(),
            kernels: self.sources.iter().filter_map(|s| s.kernel.as_ref().map(|k| (s.id, k.0.clone()))).collect(),
            sky,
            first_traced,
            env,
        }
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
