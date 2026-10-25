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
use omg_core::v3::{Budget, Source, Tracer, World};
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
}

impl Default for Config {
    fn default() -> Self {
        Self { quality: Quality::High, compute: Compute::Cpu, max_traced_sources: 64, seed: 0x0e3_5eed }
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

struct Src {
    id: SourceId,
    pos: Vec3,
    gain: f32,
    active: bool,
    block: ParamBlock,
    /// Position and tick of its last trace (None = never traced).
    traced: Option<(Vec3, u64)>,
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
    tick_no: u64,
    next_source: u32,
    next_object: u32,
    panels: Vec<(Vec3, Vec3, Material)>,
    picks: Vec<(f32, usize)>,
    traced_last: usize,
}

fn v3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

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
            tracer: Tracer::new(cfg.seed),
            walker: None,
            sources: Vec::new(),
            objects: Vec::new(),
            listener: Listener::default(),
            tick_no: 0,
            next_source: 0,
            next_object: 0,
            panels: Vec::new(),
            picks: Vec::new(),
            traced_last: 0,
        };
        s.make_walker();
        s
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Change quality or compute on the fly (takes effect next tick).
    pub fn set_config(&mut self, cfg: Config) {
        let rebuild = cfg.compute != self.cfg.compute;
        self.cfg = cfg;
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

    fn make_walker(&mut self) {
        self.walker = None;
        if matches!(self.cfg.compute, Compute::Gpu | Compute::GpuSync) {
            #[cfg(feature = "gpu")]
            {
                let sync = self.cfg.compute == Compute::GpuSync;
                self.walker = omg_gpu::GpuV3Walker::new(&self.mesh).map(|mut w| {
                    w.sync = sync;
                    Box::new(w) as Box<dyn V3Walker>
                });
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
        });
    }

    /// Move a source and/or switch it on or off.
    pub fn set_source(&mut self, id: SourceId, position: [f32; 3], active: bool) {
        if let Some(s) = self.sources.iter_mut().find(|s| s.id == id) {
            s.pos = v3(position);
            s.active = active;
        }
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
    }

    /// Sources traced in the last tick (≤ `max_traced_sources`).
    pub fn traced_last_tick(&self) -> usize {
        self.traced_last
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
        for (&(_, i), f) in self.picks.iter().zip(fields) {
            let s = &mut self.sources[i];
            let g = s.gain;
            let mut pb = ParamBlock { version: tick, ..Default::default() };
            pb.taps.reserve(f.taps.len());
            for t in f.taps {
                pb.taps.push(Tap { gains: core::array::from_fn(|b| t.gains[b] * g), ..t });
            }
            pb.reverb.rt60 = f.reverb.rt60;
            // a one-sided late field plays from where it comes (the
            // doorway, the lit facade); the rest feeds the diffuse FDN
            let a = f.aniso.clamp(0.0, 0.9);
            let lv = f.reverb.level;
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
        }

        let sky = self.tracer.sky().into_iter().map(|(dir, energy)| SkyCell { dir, energy }).collect();
        Frame {
            version: tick,
            listener: self.listener,
            blocks: self.sources.iter().map(|s| (s.id, s.block.clone())).collect(),
            sky,
        }
    }
}
