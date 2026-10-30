//! The web demo's wasm: one module, two instances — the worker runs the
//! simulation (`sim_*`), the AudioWorklet the mixer (`eng_*`). Both are
//! omg-audio, the engine the Godot demo runs; this file is only the
//! browser glue: flat buffers in and out of wasm memory, the demo's
//! sources by their logical index (town.rs), the browser's clock for the
//! load governor (a worklet has none of its own).
//!
//! Worker, per tick: the page writes door openness (`sim_door_ptr`) and
//! the dynamic slots (`sim_dyn_ptr`), calls `sim_tick`, and posts the
//! frame (`sim_frame_*`, flat), the kernels built this tick
//! (`sim_kernel_*`), the sources whose kernel went (`sim_clears_*`), the
//! page's state (`sim_state_*`) and the debug paths (`sim_rays_*`).
//!
//! Worklet: uploads go through one staging buffer (`eng_stage(n)`, then
//! the call that takes it); `eng_process(n)` renders `n` samples into
//! `eng_out_l` / `eng_out_r`.

pub mod town;

use omg_audio::governor::{Governor, LevelGovernor};
use omg_audio::sim::LEVELS;
use omg_core::v3::fold::{self as walkfmt, LiveWalk, Walk, WalkJob, Walked};
use omg_audio::{clip, normalize, Clip, Compute, Config, Frame, Hrir, Mixer, Play, Quality, Simulation, SourceId, VoiceId};
use omg_dsp::dense::DenseKernel;
use std::collections::HashMap;
use std::sync::Arc;
use town::Town;

/// The walkthrough's logical sources (town.rs; the page's mixer indices).
const NSRC: usize = 16;
/// Doors the page reports openness for, and the dynamic slots (balls,
/// cars, the footsteps): x, y, z, gain each (gain 0: idle).
const DOORS: usize = 16;
const DYN: usize = 6;

/// One `static mut` per instance: wasm here is single-threaded.
fn leak<T>(v: T) -> &'static mut T {
    Box::leak(Box::new(v))
}

// =================================================================== sim

struct SimHost {
    sim: Simulation,
    town: Town,
    cfg: Config,
    /// The quality tier (0 high, 1 med, 2 low) and pinned levers.
    tier: u32,
    /// Quality auto: the level governor walks the whole ladder (the
    /// page's tier otherwise).
    auto: bool,
    gov: LevelGovernor,
    rays: u32,
    bounces: u32,
    late_rays: u32,
    diffraction: bool,
    doors: [f32; DOORS],
    dynamic: [f32; DYN * 4],
    frame: Vec<f32>,
    /// Kernels built since the last tick: (engine source id, flat words).
    kernels: Vec<(u32, Vec<f32>)>,
    sent: HashMap<u32, usize>,
    clears: Vec<u32>,
    state: Vec<f32>,
    rays_out: Vec<f32>,
    debug: bool,
    /// The page's times (s, the worker's clock): when the pose and the
    /// dynamic slots it posted were current, and now (`sim_set_times`).
    times: Option<(f64, f64)>,
}

static mut SIM: Option<&'static mut SimHost> = None;

fn sim() -> &'static mut SimHost {
    unsafe { (*(&raw mut SIM)).as_deref_mut().expect("sim_setup first") }
}

/// The page's state per tick: [x, y, room, rt60], NSRC route entries the
/// page reads for the old engine's portal drawing (zero: none here), then
/// the flat Environment (the ambience line in the page's panel).
const ENV_OFF: usize = 4 + NSRC * 9;

impl SimHost {
    fn config(&self) -> Config {
        Config { quality: Quality::Custom(self.budget(self.cfg.compute != Compute::Cpu)), ..self.cfg }
    }

    /// The tracer budget in effect: auto's level or the tier, the pinned
    /// levers over it.
    fn budget(&self, gpu: bool) -> omg_audio::Budget {
        let tier = match (self.auto, self.tier) {
            (true, _) => Quality::Level(self.gov.level()),
            (_, 0) => Quality::High,
            (_, 2) => Quality::Low,
            _ => Quality::Medium,
        };
        let mut b = tier.budget(gpu);
        if self.rays > 0 {
            b.rays = self.rays;
        }
        if self.bounces > 0 {
            b.bounces = self.bounces;
        }
        if self.late_rays > 0 {
            b.late_rays = self.late_rays;
        }
        b.diffraction = self.diffraction;
        b
    }

    fn reconfigure(&mut self) {
        let c = self.config();
        self.sim.set_config(c);
        // a rebuild (v3/v4) drops the walker: the GPU walk stays on
        if self.cfg.compute != Compute::Cpu && !self.sim.gpu_active() {
            self.sim.set_walker(Some(Box::new(JsWalker)));
        }
    }
}

#[no_mangle]
pub extern "C" fn sim_setup(sample_rate: f32) {
    let cfg = Config {
        quality: Quality::Medium,
        compute: Compute::Cpu,
        // v6.2: v3's estimate and reverb with v6's refinements, the
        // reverb as long and loud as traced, a door heard where it is
        // (the page's engine switch picks others)
        v6: true,
        v61: true,
        v62: true,
        sample_rate,
        // no threads in a worker's wasm: kernels a step a tick
        kernel_thread: false,
        max_kernels: 8,
        max_kernel_s: 2.5,
        // a 2.5 s kernel in two ticks (~5 ms native a step, a few × that
        // in wasm): after a move, all eight convolve again within a second
        kernel_step_samples: 65536,
        kernel_every_ticks: 1,
        ..Config::default()
    };
    let mut sim = Simulation::new(cfg);
    let town = Town::new(&mut sim);
    let host = SimHost {
        sim,
        town,
        cfg,
        tier: 1,
        auto: true,
        gov: LevelGovernor::new(LEVELS - 1),
        rays: 0,
        bounces: 0,
        late_rays: 0,
        diffraction: true,
        doors: [-1.0; DOORS],
        dynamic: [0.0; DYN * 4],
        frame: Vec::new(),
        kernels: Vec::new(),
        sent: HashMap::new(),
        clears: Vec::new(),
        state: vec![0.0; ENV_OFF + omg_dsp::env::ENV_FLAT_LEN],
        rays_out: Vec::new(),
        debug: false,
        times: None,
    };
    unsafe { *(&raw mut SIM) = Some(leak(host)) };
    self::sim().reconfigure();
}

#[no_mangle]
pub extern "C" fn sim_door_ptr() -> *mut f32 {
    sim().doors.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn sim_dyn_ptr() -> *mut f32 {
    sim().dynamic.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn sim_set_quality(tier: u32) {
    let h = sim();
    h.tier = tier.min(2);
    h.auto = false;
    h.reconfigure();
}

/// What a tick may cost under quality auto (ms): the worker shares the
/// machine with the page's frames.
const TICK_TARGET_MS: f32 = 16.0;

/// Quality auto: the level governor walks the ladder from here on.
#[no_mangle]
pub extern "C" fn sim_set_auto() {
    let h = sim();
    h.auto = true;
    h.reconfigure();
}

/// Quality auto: a tick took `tick_ms` with the audio thread at `load`
/// (render time over audio time). The level in effect (−1: a tier is
/// pinned).
#[no_mangle]
pub extern "C" fn sim_govern(tick_ms: f32, load: f32) -> i32 {
    let h = sim();
    if !h.auto {
        return -1;
    }
    // a door moved: the extra rays are the change's cost, not the level's
    if !h.sim.tick_costs().burst && h.gov.report(tick_ms, TICK_TARGET_MS, load).is_some() {
        h.reconfigure();
    }
    h.gov.level() as i32
}

/// The CPU tracer budget in effect: 0 rays, 1 reflections, 2 late rays.
#[no_mangle]
pub extern "C" fn sim_budget(i: u32) -> u32 {
    let b = sim().budget(false);
    match i {
        0 => b.rays,
        1 => b.bounces,
        _ => b.late_rays,
    }
}

/// Pin a tracer lever (0 hands it back to the tier): 5 rays from the
/// listener, 6 reflections per ray, 7 late rays (the page's ids).
#[no_mangle]
pub extern "C" fn sim_set_override(id: u32, value: u32) {
    let h = sim();
    match id {
        5 => h.rays = value,
        6 => h.bounces = value,
        7 => h.late_rays = value,
        _ => return,
    }
    h.reconfigure();
}

/// Engine switches: 0 diffraction.
#[no_mangle]
pub extern "C" fn sim_set_module(id: u32, on: u32) {
    let h = sim();
    if id == 0 {
        h.diffraction = on != 0;
        h.reconfigure();
    }
}

/// Engine v4 (1) or v3 (0).
#[no_mangle]
pub extern "C" fn sim_set_v4(on: u32) {
    let h = sim();
    h.cfg.v4 = on != 0;
    h.reconfigure();
}

/// Engine v5 (on v4): the late paths reused across ticks.
#[no_mangle]
pub extern "C" fn sim_set_v5(on: u32) {
    let h = sim();
    h.cfg.v5 = on != 0;
    h.reconfigure();
}

/// Engine v6 (v3's estimate and reverb, with v6's refinements).
#[no_mangle]
pub extern "C" fn sim_set_v6(on: u32) {
    let h = sim();
    h.cfg.v6 = on != 0;
    h.reconfigure();
}

/// Engine v6.1 (on v6): the reverberation time as long as the traced
/// decay says, the late level as loud (`Config::v61`).
#[no_mangle]
pub extern "C" fn sim_set_v61(on: u32) {
    let h = sim();
    h.cfg.v61 = on != 0;
    h.reconfigure();
}

/// Engine v6.2 (on v6.1): the paths that met a door priced again with
/// its leaf where it is now, every tick and frame (`Config::v62`).
#[no_mangle]
pub extern "C" fn sim_set_v62(on: u32) {
    let h = sim();
    h.cfg.v62 = on != 0;
    h.reconfigure();
}

#[no_mangle]
pub extern "C" fn sim_v4() -> u32 {
    sim().cfg.v4 as u32
}

/// Mute a logical source: it is not traced at all.
#[no_mangle]
pub extern "C" fn sim_set_mute(i: u32, on: u32) {
    let h = sim();
    h.town.set_muted(&mut h.sim, i as usize, on != 0);
}

#[no_mangle]
pub extern "C" fn sim_set_debug(on: u32) {
    let h = sim();
    h.debug = on != 0;
    h.sim.set_debug_paths(h.debug);
}

/// The times for the next tick or exact frame (s, one steady clock): when
/// the page's pose and dynamic slots were current, and now. The listener
/// and the moving sources are stamped with the first, the simulation's
/// clock is the second — where each was when is what Doppler and far
/// paths are timed by (without, the simulation counts 50 ms a tick,
/// however late the worker's timer fired).
#[no_mangle]
pub extern "C" fn sim_set_times(pose: f64, now: f64) {
    sim().times = Some((pose, now.max(pose)));
}

/// The page's dynamic slots and listener into the simulation, stamped
/// with the pose's time when the page gave one.
fn place(h: &mut SimHost, x: f32, y: f32, z: f32, yaw: f32) {
    if let Some((pose, _)) = h.times {
        h.sim.set_time(pose);
    }
    for s in 0..DYN {
        let d = &h.dynamic[s * 4..s * 4 + 4];
        let at = (d[3] > 0.0).then(|| ([d[0], d[1], d[2]], d[3]));
        h.town.set_dynamic(&mut h.sim, s, at);
    }
    match h.times {
        Some((pose, now)) => {
            h.sim.set_listener_at([x, y, z], yaw, 0.0, 0.0, pose);
            h.sim.set_time(now);
        }
        None => h.sim.set_listener([x, y, z], yaw, 0.0, 0.0),
    }
}

/// One tick from the listener at (x, y, z), z up, yaw about +z.
#[no_mangle]
pub extern "C" fn sim_tick(x: f32, y: f32, z: f32, yaw: f32) {
    let h = sim();
    for i in 0..DOORS.min(h.town.doors.len()) {
        if h.doors[i] >= 0.0 {
            h.town.set_door(&mut h.sim, i, h.doors[i]);
        }
    }
    place(h, x, y, z, yaw);
    let frame = h.sim.tick();
    frame.write_flat(&mut h.frame);
    // kernels new since the last tick (by their Arc), and those gone
    h.kernels.clear();
    h.clears.clear();
    for (id, k) in &frame.kernels {
        let key = Arc::as_ptr(k) as usize;
        if h.sent.get(&id.0) != Some(&key) {
            h.sent.insert(id.0, key);
            h.kernels.push((id.0, k.to_flat()));
        }
    }
    let have: Vec<u32> = frame.kernels.iter().map(|(id, _)| id.0).collect();
    let gone: Vec<u32> = h.sent.keys().copied().filter(|k| !have.contains(k)).collect();
    for k in gone {
        h.sent.remove(&k);
        h.clears.push(k);
    }
    // the page's state: where, which room, the loudest field's decay
    h.state[0] = x;
    h.state[1] = y;
    h.state[2] = h.town.room_at([x, y, z]).map_or(0.0, |r| r as f32);
    let mut best = (0.0f32, 0.0f32);
    for (_, b) in &frame.blocks {
        let lvl = b.taps.iter().map(|t| t.gains[1]).sum::<f32>() + b.reverb.level[1];
        if lvl > best.0 {
            best = (lvl, b.reverb.rt60[1]);
        }
    }
    h.state[3] = best.1;
    frame.env.write_flat(&mut h.state[ENV_OFF..]);
    // debug: the traced paths, [logical source, n, xyz × n] …
    h.rays_out.clear();
    if h.debug {
        for p in h.sim.debug_paths() {
            let Some(i) = h.town.logical(p.source) else { continue };
            if h.rays_out.len() + 2 + p.points.len() * 3 > 6000 {
                break;
            }
            h.rays_out.push(i as f32);
            h.rays_out.push(p.points.len() as f32);
            for q in &p.points {
                h.rays_out.extend_from_slice(q);
            }
        }
    }
}

/// Between ticks: the exact paths (direct, bends, mirror chains) for the
/// listener at (x, y, z) and the dynamic slots as the page left them —
/// 1 and a frame in `sim_frame_*` when anything moved, else 0.
#[no_mangle]
pub extern "C" fn sim_tick_exact(x: f32, y: f32, z: f32, yaw: f32) -> u32 {
    let h = sim();
    place(h, x, y, z, yaw);
    match h.sim.tick_exact() {
        Some(f) => {
            f.write_flat(&mut h.frame);
            1
        }
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn sim_frame_ptr() -> *const f32 {
    sim().frame.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_frame_len() -> u32 {
    sim().frame.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_kernel_count() -> u32 {
    sim().kernels.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_kernel_src(i: u32) -> u32 {
    sim().kernels[i as usize].0
}

#[no_mangle]
pub extern "C" fn sim_kernel_ptr(i: u32) -> *const f32 {
    sim().kernels[i as usize].1.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_kernel_len(i: u32) -> u32 {
    sim().kernels[i as usize].1.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_clears_len() -> u32 {
    sim().clears.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_clears_ptr() -> *const u32 {
    sim().clears.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_state_ptr() -> *const f32 {
    sim().state.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_state_len() -> u32 {
    sim().state.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_env_off() -> u32 {
    ENV_OFF as u32
}

#[no_mangle]
pub extern "C" fn sim_rays_ptr() -> *const f32 {
    sim().rays_out.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_rays_len() -> u32 {
    sim().rays_out.len() as u32
}

/// Sources convolved (with a kernel) in the last frame.
#[no_mangle]
pub extern "C" fn sim_dense_count() -> u32 {
    sim().sent.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_nsrc() -> u32 {
    NSRC as u32
}

// ============================================================ GPU walk
//
// With WebGPU, the tracer's ray walk runs in gpu.js on the same shader the
// native host runs (crates/omg-gpu/shaders/v3_walk.wgsl), over the
// simulation's own mesh (`sim_mesh_*`). Each tick the walker leaves its
// job here; gpu.js takes it (`sim_v3_take`), runs it as `sim_v3_passes`
// short dispatches paced to the GPU budget, and hands the result back
// (`sim_v3_inject`) — folded in the tick after. The GPU walks the early
// rays; the late family is the CPU's, walked whole with each fresh walk
// (`V3Walker::walks_late`), as natively.

#[derive(Default)]
struct Bridge {
    /// The newest job not yet taken (newest wins).
    job: Option<WalkJob>,
    /// The job gpu.js took (source ids, epoch) and its plan.
    taken: Option<(Vec<u32>, u32, LiveWalk, omg_core::vec3::Vec3)>,
    result: Option<Walked>,
    words: Vec<u32>,
    stage: Vec<f32>,
    gpu_ms: f32,
    gpu_max_ms: f32,
}

static mut BRIDGE: Option<&'static mut Bridge> = None;

fn bridge() -> &'static mut Bridge {
    unsafe { (*(&raw mut BRIDGE)).get_or_insert_with(|| leak(Bridge::default())) }
}

struct JsWalker;

impl omg_audio::V3Walker for JsWalker {
    fn walk(&mut self, job: &WalkJob) -> Walk {
        if job.n_edges() != 0 && job.n_edges() != edges_flat()[0] as usize {
            return Walk::Refused; // another edge set than the kernel holds
        }
        let b = bridge();
        b.job = Some(job.clone());
        match b.result.take() {
            Some(w) => Walk::Fresh(w),
            None => Walk::Pending,
        }
    }

    fn last_ms(&self) -> f32 {
        bridge().gpu_ms
    }

    fn last_max_ms(&self) -> f32 {
        bridge().gpu_max_ms
    }

    fn walks_late(&self) -> bool {
        false
    }
}

/// The world mesh flattened for the walk kernel (as omg-gpu's
/// `flatten_mesh`): nodes 8 words, prims 12, materials 8.
static mut MESH_FLAT: Option<(Vec<u32>, Vec<u32>, Vec<u32>)> = None;

fn mesh_flat() -> &'static (Vec<u32>, Vec<u32>, Vec<u32>) {
    unsafe {
        (*(&raw mut MESH_FLAT)).get_or_insert_with(|| {
            let mesh = sim().sim.mesh();
            let (mut nodes, mut prims) = (Vec::new(), Vec::new());
            let f = |x: f32| x.to_bits();
            mesh.visit_bvh(
                &mut |bmin, bmax, a, b| nodes.extend_from_slice(&[f(bmin.x), f(bmin.y), f(bmin.z), a, f(bmax.x), f(bmax.y), f(bmax.z), b]),
                &mut |a, e1, e2, m, surf| {
                    prims.extend_from_slice(&[f(a.x), f(a.y), f(a.z), m as u32, f(e1.x), f(e1.y), f(e1.z), surf as u32, f(e2.x), f(e2.y), f(e2.z), 0])
                },
            );
            let mut mats = Vec::new();
            for (i, m) in mesh.materials.iter().enumerate() {
                let t = mesh.material_thickness(i as u16).unwrap_or(0.0);
                mats.extend_from_slice(&[f(m.absorption[0]), f(m.absorption[1]), f(m.absorption[2]), f(m.scattering)]);
                mats.extend_from_slice(&[f(m.transmission[0]), f(m.transmission[1]), f(m.transmission[2]), f(t)]);
            }
            (nodes, prims, mats)
        })
    }
}

#[no_mangle]
pub extern "C" fn sim_mesh_nodes_ptr() -> *const u32 {
    mesh_flat().0.as_ptr()
}
#[no_mangle]
pub extern "C" fn sim_mesh_nodes_len() -> u32 {
    mesh_flat().0.len() as u32
}
#[no_mangle]
pub extern "C" fn sim_mesh_prims_ptr() -> *const u32 {
    mesh_flat().1.as_ptr()
}
#[no_mangle]
pub extern "C" fn sim_mesh_prims_len() -> u32 {
    mesh_flat().1.len() as u32
}
#[no_mangle]
pub extern "C" fn sim_mesh_mats_ptr() -> *const u32 {
    mesh_flat().2.as_ptr()
}
#[no_mangle]
pub extern "C" fn sim_mesh_mats_len() -> u32 {
    mesh_flat().2.len() as u32
}

/// The mesh's diffraction edges and their search grid for the walk
/// kernel (`EdgeSet::flat`, binding 8): built like the tracer's own, so a
/// job's filled mask indexes them. Word 0 is the edge count.
static mut EDGES_FLAT: Option<Vec<u32>> = None;

fn edges_flat() -> &'static Vec<u32> {
    unsafe { (*(&raw mut EDGES_FLAT)).get_or_insert_with(|| omg_core::v3::EdgeSet::build(sim().sim.mesh()).flat()) }
}

#[no_mangle]
pub extern "C" fn sim_mesh_edges_ptr() -> *const u32 {
    edges_flat().as_ptr()
}
#[no_mangle]
pub extern "C" fn sim_mesh_edges_len() -> u32 {
    edges_flat().len() as u32
}

/// The walk kernel's layout version (gpu.js refuses another).
#[no_mangle]
pub extern "C" fn sim_v3_version() -> u32 {
    walkfmt::WALK_LAYOUT_VERSION
}

/// gpu.js is ready: the walk goes to it (the GPU's ray budget, ×8).
#[no_mangle]
pub extern "C" fn sim_gpu_enable() {
    let h = sim();
    h.cfg.compute = Compute::Gpu;
    h.reconfigure();
    h.sim.set_walker(Some(Box::new(JsWalker)));
}

/// Back to the CPU walk.
#[no_mangle]
pub extern "C" fn sim_gpu_disable() {
    let h = sim();
    h.cfg.compute = Compute::Cpu;
    h.reconfigure();
    let b = bridge();
    b.job = None;
    b.result = None;
}

#[no_mangle]
pub extern "C" fn sim_gpu_on() -> u32 {
    (sim().cfg.compute != Compute::Cpu) as u32
}

/// The share of the tier's GPU rays the walk carries (it shrinks so a
/// walk lands every tick within the GPU's share).
#[no_mangle]
pub extern "C" fn sim_gpu_rays_share() -> f32 {
    sim().sim.gpu_rays_share()
}

/// Take the newest walk job: its words (header + scene) in
/// `sim_v3_job_ptr`, the count returned (0: none waiting).
#[no_mangle]
pub extern "C" fn sim_v3_take() -> u32 {
    let b = bridge();
    let Some(job) = b.job.take() else { return 0 };
    let live = job.live(0);
    b.words.clear();
    b.words.extend_from_slice(&live.header);
    b.words.extend_from_slice(job.scene());
    b.taken = Some((job.ids.clone(), job.epoch, live, job.listener));
    b.words.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_v3_job_ptr() -> *const u32 {
    bridge().words.as_ptr()
}

/// The taken job's dispatches, and header word 36 for pass `pass`.
#[no_mangle]
pub extern "C" fn sim_v3_passes() -> u32 {
    bridge().taken.as_ref().map_or(0, |t| t.2.passes)
}

#[no_mangle]
pub extern "C" fn sim_v3_word36(pass: u32) -> u32 {
    walkfmt::slice_word(pass)
}

/// Words the kernel's state buffer (binding 7) needs for the taken job.
#[no_mangle]
pub extern "C" fn sim_v3_dbg_words() -> u32 {
    bridge().taken.as_ref().map_or(0, |t| t.2.state_words) as u32
}

/// Output words of the taken job.
#[no_mangle]
pub extern "C" fn sim_v3_out_words() -> u32 {
    bridge().taken.as_ref().map_or(0, |t| walkfmt::output_words(t.0.len())) as u32
}

/// Staging for the result: `n` f32 words.
#[no_mangle]
pub extern "C" fn sim_v3_buf_ptr(n: u32) -> *mut f32 {
    let b = bridge();
    b.stage.resize(n as usize, 0.0);
    b.stage.as_mut_ptr()
}

/// The staged result of the taken job, and its GPU times (summed over
/// the dispatches, the longest one).
#[no_mangle]
pub extern "C" fn sim_v3_inject(n: u32, gpu_ms: f32, gpu_max_ms: f32) {
    let b = bridge();
    let Some((ids, epoch, _, listener)) = b.taken.take() else { return };
    let out = walkfmt::output_words(ids.len());
    if (n as usize) < out {
        return; // malformed: the next walk refreshes
    }
    let data = b.stage[..out].to_vec();
    b.result = Some(Walked { ids, data, debug: Vec::new(), epoch, listener });
    b.gpu_ms = gpu_ms;
    b.gpu_max_ms = gpu_max_ms;
}

// ================================================================ engine

struct EngHost {
    mixer: Mixer,
    gov: Governor,
    map: Vec<Vec<SourceId>>,
    stage: Vec<f32>,
    out_l: Vec<f32>,
    out_r: Vec<f32>,
    /// Per logical source: its loop's voices (one per engine source).
    loops: Vec<Vec<VoiceId>>,
    fx: Vec<Clip>,
    /// One-shots in flight per (logical source, fx kind) — to stop them.
    shots: HashMap<(usize, usize), Vec<VoiceId>>,
    kernels: HashMap<u32, Arc<DenseKernel>>,
    meters: Vec<f32>,
}

static mut ENG: Option<&'static mut EngHost> = None;

fn eng() -> &'static mut EngHost {
    unsafe { (*(&raw mut ENG)).as_deref_mut().expect("eng_init first") }
}

#[no_mangle]
pub extern "C" fn eng_init(sample_rate: f32) {
    let map = town::source_map();
    let n = map.iter().map(Vec::len).sum();
    let mut mixer = Mixer::with_capacity(sample_rate, Hrir::builtin(), n);
    let gov = Governor::new(sample_rate);
    gov.apply(&mut mixer);
    let host = EngHost {
        mixer,
        gov,
        loops: vec![Vec::new(); map.len()],
        map,
        stage: Vec::new(),
        out_l: vec![0.0; 128],
        out_r: vec![0.0; 128],
        fx: Vec::new(),
        shots: HashMap::new(),
        kernels: HashMap::new(),
        meters: vec![0.0; 2 * (NSRC + 2)],
    };
    unsafe { *(&raw mut ENG) = Some(leak(host)) };
}

/// The staging buffer for the next upload, `n` floats.
#[no_mangle]
pub extern "C" fn eng_stage(n: u32) -> *mut f32 {
    let e = eng();
    e.stage.resize(n as usize, 0.0);
    e.stage.as_mut_ptr()
}

/// The staged samples as logical source `i`'s loop (normalized, gated
/// RMS; empty: no loop), replacing the one it had (a car's motor swap).
#[no_mangle]
pub extern "C" fn eng_source(i: u32) {
    let e = eng();
    let i = i as usize;
    if i >= e.map.len() {
        return;
    }
    for v in e.loops[i].drain(..) {
        e.mixer.stop(v);
    }
    if e.stage.iter().all(|x| *x == 0.0) {
        return;
    }
    normalize(&mut e.stage);
    let c = clip(&e.stage);
    e.mixer.prime_clip(&c);
    for &id in &e.map[i] {
        let v = e.mixer.play(id, &c, Play::LOOP);
        e.loops[i].push(v);
    }
}

/// The staged samples as the next one-shot clip (its index: the page's
/// fx kind).
#[no_mangle]
pub extern "C" fn eng_fx() -> u32 {
    let e = eng();
    let c = clip(&e.stage);
    e.mixer.prime_clip(&c);
    e.fx.push(c);
    (e.fx.len() - 1) as u32
}

#[no_mangle]
pub extern "C" fn eng_fx_play(i: u32, kind: u32, gain: f32, rate: f32) {
    let e = eng();
    let (i, kind) = (i as usize, kind as usize);
    let (Some(ids), Some(c)) = (e.map.get(i), e.fx.get(kind)) else { return };
    let play = Play::ONCE.gain(gain).rate(rate);
    let vs: Vec<VoiceId> = ids.iter().map(|&id| e.mixer.play(id, c, play)).collect();
    let list = e.shots.entry((i, kind)).or_default();
    list.extend(vs);
    if list.len() > 64 {
        list.drain(..list.len() - 64);
    }
}

#[no_mangle]
pub extern "C" fn eng_fx_stop(i: u32, kind: u32) {
    let e = eng();
    if let Some(vs) = e.shots.remove(&(i as usize, kind as usize)) {
        for v in vs {
            e.mixer.stop(v);
        }
    }
}

/// A near-field voice: fx `kind` at `dist` m from one ear.
#[no_mangle]
pub extern "C" fn eng_whisper(kind: u32, right: u32, dist: f32, gain: f32) {
    let e = eng();
    if let Some(c) = e.fx.get(kind as usize) {
        e.mixer.play_near(c, right != 0, dist, gain);
    }
}

/// The staged samples as the ambience bed (interleaved stereo when
/// `stereo`).
#[no_mangle]
pub extern "C" fn eng_ambience(stereo: u32) {
    let e = eng();
    let data = core::mem::take(&mut e.stage);
    e.mixer.set_ambience(data, stereo != 0);
}

#[no_mangle]
pub extern "C" fn eng_rain_bank() {
    let e = eng();
    let data = core::mem::take(&mut e.stage);
    e.mixer.set_rain_bank(data);
}

#[no_mangle]
pub extern "C" fn eng_rain(intensity: f32) {
    eng().mixer.set_rain(intensity);
}

#[no_mangle]
pub extern "C" fn eng_ambience_gain(g: f32) {
    eng().mixer.set_ambience_gain(g);
}

#[no_mangle]
pub extern "C" fn eng_rain_gain(g: f32) {
    eng().mixer.set_rain_gain(g);
}

#[no_mangle]
pub extern "C" fn eng_master(g: f32) {
    eng().mixer.set_master(g);
}

/// Logical source `i`'s mixer volume (amplitude; 0 stops its cost).
#[no_mangle]
pub extern "C" fn eng_gain(i: u32, g: f32) {
    let e = eng();
    if let Some(ids) = e.map.get(i as usize) {
        for &id in ids {
            e.mixer.set_source_gain(id, g);
        }
    }
}

#[no_mangle]
pub extern "C" fn eng_head(yaw: f32, pitch: f32, roll: f32) {
    eng().mixer.set_head(yaw, pitch, roll);
}

/// The staged words as engine source `src`'s convolution kernel.
#[no_mangle]
pub extern "C" fn eng_kernel(src: u32) {
    let e = eng();
    let words = core::mem::take(&mut e.stage);
    if let Some(k) = DenseKernel::from_flat(words) {
        e.kernels.insert(src, Arc::new(k));
    }
}

#[no_mangle]
pub extern "C" fn eng_kernel_clear(src: u32) {
    eng().kernels.remove(&src);
}

/// The staged words as the next frame (flat), with the kernels uploaded.
#[no_mangle]
pub extern "C" fn eng_frame() -> u32 {
    let e = eng();
    let Some(mut f) = Frame::read_flat(&e.stage) else { return 0 };
    f.kernels = f.blocks.iter().filter_map(|(id, _)| e.kernels.get(&id.0).map(|k| (*id, k.clone()))).collect();
    e.mixer.apply(&f);
    1
}

#[no_mangle]
pub extern "C" fn eng_process(n: u32) {
    let e = eng();
    let n = n as usize;
    e.out_l.resize(n, 0.0);
    e.out_r.resize(n, 0.0);
    e.mixer.render(&mut e.out_l, &mut e.out_r);
}

#[no_mangle]
pub extern "C" fn eng_out_l() -> *const f32 {
    eng().out_l.as_ptr()
}

#[no_mangle]
pub extern "C" fn eng_out_r() -> *const f32 {
    eng().out_r.as_ptr()
}

/// Rendering `frames` samples took `ms` (the worklet's clock): the
/// governor walks the budgets.
#[no_mangle]
pub extern "C" fn eng_report(ms: f32, frames: u32) {
    let e = eng();
    e.gov.report(&mut e.mixer, ms as f64 * 1e-3, frames as usize);
}

/// The audio thread missed a deadline: shed now.
#[no_mangle]
pub extern "C" fn eng_gap() {
    let e = eng();
    e.gov.missed_deadline(&mut e.mixer);
}

/// Pin the governor's rung (-1: hand it back).
#[no_mangle]
pub extern "C" fn eng_pin(rung: i32) {
    let e = eng();
    e.gov.pin(&mut e.mixer, (rung >= 0).then_some(rung as usize));
}

#[no_mangle]
pub extern "C" fn eng_load() -> f32 {
    eng().gov.load()
}

#[no_mangle]
pub extern "C" fn eng_rung() -> u32 {
    eng().gov.rung() as u32
}

#[no_mangle]
pub extern "C" fn eng_agc() -> f32 {
    eng().mixer.agc_gain()
}

#[no_mangle]
pub extern "C" fn eng_fatigue() -> f32 {
    eng().mixer.ear_fatigue()
}

/// The levels since the last call: per logical source (peak, RMS — its
/// engine sources' summed), then the ambience bed and the rain.
#[no_mangle]
pub extern "C" fn eng_meters() -> *const f32 {
    let e = eng();
    for (i, ids) in e.map.iter().enumerate().take(NSRC) {
        let (mut pk, mut e2) = (0.0f32, 0.0f32);
        for &id in ids {
            let m = e.mixer.take_meter(id);
            pk = pk.max(m.peak);
            e2 += m.rms * m.rms;
        }
        e.meters[2 * i] = pk;
        e.meters[2 * i + 1] = e2.sqrt();
    }
    let (a, r) = e.mixer.take_env_meters();
    e.meters[2 * NSRC] = a.peak;
    e.meters[2 * NSRC + 1] = a.rms;
    e.meters[2 * NSRC + 2] = r.peak;
    e.meters[2 * NSRC + 3] = r.rms;
    e.meters.as_ptr()
}

/// (sources rendering, paths, point-HRTF paths) last render.
#[no_mangle]
pub extern "C" fn eng_rendering() -> u32 {
    eng().mixer.occupancy().0 as u32
}
