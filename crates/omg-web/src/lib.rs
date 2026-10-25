//! omg-web: wasm exports for the browser build. One module, two halves,
//! instantiated in two different contexts:
//!
//!  - `sim_*`  — runs in a Web Worker at ~20 Hz. Takes the listener pose,
//!    runs the WorldSim tick, exposes one flat ParamBlock buffer per source
//!    plus a small state buffer for the canvas viz.
//!  - `eng_*`  — runs inside the AudioWorkletProcessor. Receives HRIR
//!    assets, decoded source audio, flat ParamBlocks and head yaw; renders
//!    stereo.
//!
//! Deliberately no wasm-bindgen: AudioWorkletGlobalScope is a hostile
//! environment for JS glue, and every value crossing the boundary here is a
//! number or a buffer of numbers. Plain `extern "C"` + linear memory.

use std::sync::Arc;

use omg_core::params::ParamBlock;
use omg_dsp::ambi::NCH;
use omg_dsp::hrtf::HrirGrid;
use omg_dsp::output::OutputStage;
use omg_dsp::Renderer;
use omg_scene::world::WorldSim;

const NSRC: usize = 14;
const MAX_FLAT: usize = 4096; // f32s per param buffer (~450 taps headroom)
/// State layout: [0..4] pose/room/rt60, then NSRC route-viz entries of
/// 9 floats each, then the flat Environment block (see omg_dsp::env).
/// ENV_OFF derives from NSRC — it silently overflowed once when NSRC
/// grew (car positions parsed as ambience route gains: the loud-burst
/// bug), so the layout now has ONE source of truth, exported to JS.
const ENV_OFF: usize = 4 + NSRC * 9;
const STATE_LEN: usize = ENV_OFF + omg_dsp::env::ENV_FLAT_LEN;
const MAX_BLOCK: usize = 4096;

// ------------------------------------------------------------------ helpers

/// Leak a boxed buffer and hand its pointer to JS (lives for the page).
fn leak_f32(n: usize) -> &'static mut [f32] {
    Box::leak(vec![0.0f32; n].into_boxed_slice())
}

fn leak_u8(n: usize) -> &'static mut [u8] {
    Box::leak(vec![0u8; n].into_boxed_slice())
}

// ================================================================ SIM SIDE

struct SimCtx {
    world: WorldSim,
    params: [&'static mut [f32]; NSRC],
    param_lens: [usize; NSRC],
    state: &'static mut [f32],
    /// JS-written dynamic-source inputs: [x, y, z, active] × DYN_SLOTS.
    dyn_in: &'static mut [f32],
    /// JS-written door states (1 = open), indices into the scene doors.
    door_in: &'static mut [f32],
    flat_tmp: Vec<f32>,
}

static mut SIM: Option<SimCtx> = None;

/// Both contexts are single-threaded (a Worker, an AudioWorklet), so one
/// mutable global per context is sound; raw-pointer access keeps the
/// Rust-2024 `static_mut_refs` lint honest about that.
fn sim() -> &'static mut SimCtx {
    unsafe { (*(&raw mut SIM)).as_mut().expect("sim_setup first") }
}

#[no_mangle]
pub extern "C" fn sim_setup() {
    let ctx = SimCtx {
        world: WorldSim::new(),
        params: core::array::from_fn(|_| leak_f32(MAX_FLAT)),
        param_lens: [0; NSRC],
        state: leak_f32(STATE_LEN),
        dyn_in: leak_f32(omg_scene::walkthrough::DYN_SLOTS * 4),
        door_in: {
            let b = leak_f32(16);
            b.fill(1.0);
            b
        },
        flat_tmp: Vec::with_capacity(MAX_FLAT),
    };
    unsafe { *(&raw mut SIM) = Some(ctx) };
}

/// One simulation tick for listener pose (world coords, walk yaw).
/// Fills the per-source param buffers and the state buffer:
///   state = [lx, ly, room, rt60_mid,
///            src0_route_n, x0,y0,x1,y1,x2,y2,x3,y3,   (≤4 route points)
///            src1_route_n, x0,y0,...]
#[no_mangle]
pub extern "C" fn sim_dyn_ptr() -> *mut f32 {
    let ctx = sim();
    ctx.dyn_in.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn sim_door_ptr() -> *mut f32 {
    sim().door_in.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn sim_state_len() -> u32 {
    STATE_LEN as u32
}

#[no_mangle]
pub extern "C" fn sim_env_off() -> u32 {
    ENV_OFF as u32
}

/// Source count — JS sizes its block loop and meter reads from this.
#[no_mangle]
pub extern "C" fn sim_nsrc() -> u32 {
    NSRC as u32
}

/// Quality ladder for the sim-side ray budgets (GPU_PLAN.md Track B):
/// 0 = High, 1 = Med, 2 = Low. Scales trace rays, gate refresh, dome
/// rays/events and ISM order — variance/staleness, never bias.
#[no_mangle]
pub extern "C" fn sim_set_quality(t: u32) {
    omg_scene::quality::set_tier(t);
}

/// Early-reflections backend (Track C): 0 = ism, 1 = traced. Live
/// switch — both emit the same Tap contract, PT keys are namespaced,
/// so flipping crossfades through the renderer's slot release.
#[no_mangle]
pub extern "C" fn sim_set_early(mode: u32) {
    omg_scene::quality::set_early(mode);
}

/// The ACTIVE early backend as the sim sees it (0 = ism, 1 = traced) —
/// ground truth for the UI, not an echo of what was requested.
#[no_mangle]
pub extern "C" fn sim_early_mode() -> u32 {
    omg_scene::quality::early()
}

// ------------------------------------------------------- GPU trace bridge
// GPU_PLAN.md phase 3: the wasm stays freestanding, so the WebGPU driver
// is plain JS (web/gpu.js) running in the worker. When enabled, the
// registered proxy backend queues each gate-opened trace as a flat job
// for JS to fetch after the tick; JS dispatches trace_box.wgsl, decodes
// the fixed-point output to f32, and injects the echogram back, which
// `Sim` consumes via poll_into one tick later. Never enabled = the
// inline CPU tracer exactly as before (node harnesses unaffected).

/// Flat job layout, `GPU_JOB_F32S` f32 words per job:
/// [0] sim id · [1] n_rays · [2] seed · [3..6] room size ·
/// [6..9] source · [9..12] listener · [12..15] band energy ·
/// [15..57] 6 faces × (absorption ×3, scattering, transmission ×3).
/// Bump `GPU_JOB_VERSION` on ANY change — gpu.js checks it and refuses
/// to enable on mismatch (CPU fallback beats decoding garbage).
pub const GPU_JOB_F32S: usize = 57;
pub const GPU_JOB_VERSION: u32 = 2;
/// Echogram injection: 300 bins × 3 bands, then 300 bins × xyz.
const GPU_ECHO_F32S: usize = 300 * 3 + 300 * 3;
const GPU_MAX_JOBS: usize = 32;

struct WebGpuProxy;

static GPU_JOBS: std::sync::Mutex<Vec<f32>> = std::sync::Mutex::new(Vec::new());
static GPU_RESULTS: std::sync::Mutex<Vec<(u32, Vec<f32>)>> = std::sync::Mutex::new(Vec::new());

impl omg_scene::late::LateBackend for WebGpuProxy {
    fn trace(
        &mut self,
        id: u32,
        room: &omg_core::scene::Shoebox,
        src: omg_core::vec3::Vec3,
        lis: omg_core::vec3::Vec3,
        n_rays: u32,
        energy: [f32; omg_core::NBANDS],
        rng: &mut omg_core::rng::Rng,
        _out: &mut omg_core::tracer::Echogram,
    ) -> bool {
        let mut jobs = GPU_JOBS.lock().unwrap();
        if jobs.len() >= GPU_MAX_JOBS * GPU_JOB_F32S {
            return false; // JS stalled; drop, the gate will re-fire
        }
        let seed = (rng.next_u64() >> 16) as u32 as f32;
        jobs.extend_from_slice(&[id as f32, n_rays as f32, seed]);
        jobs.extend_from_slice(&[room.size.x, room.size.y, room.size.z]);
        jobs.extend_from_slice(&[src.x, src.y, src.z]);
        jobs.extend_from_slice(&[lis.x, lis.y, lis.z]);
        jobs.extend_from_slice(&energy);
        for w in &room.walls {
            jobs.extend_from_slice(&w.absorption);
            jobs.push(w.scattering);
            jobs.extend_from_slice(&w.transmission);
        }
        false // result arrives via poll_into after JS injects it
    }

    fn poll_into(&mut self, id: u32, out: &mut omg_core::tracer::Echogram) -> bool {
        let mut results = GPU_RESULTS.lock().unwrap();
        let Some(i) = results.iter().position(|(rid, _)| *rid == id) else {
            return false;
        };
        let (_, data) = results.swap_remove(i);
        for bin in 0..300 {
            for b in 0..3 {
                out.bins[bin][b] = data[bin * 3 + b];
            }
            for k in 0..3 {
                out.dirs[bin][k] = data[900 + bin * 3 + k];
            }
        }
        true
    }
}

static mut RAYS_OUT: Option<&'static mut [f32]> = None;

/// Debug rays (called by the worker ONLY while the debug panel is
/// open): world-space traced-path polylines, flat
/// [src, n_verts, x,y,z × n]…; returns the f32 count.
#[no_mangle]
pub extern "C" fn sim_debug_rays_len() -> u32 {
    let out = unsafe { (*(&raw mut RAYS_OUT)).get_or_insert_with(|| leak_f32(6144)) };
    let ctx = sim();
    let mut buf = Vec::with_capacity(4096);
    ctx.world.debug_rays(&mut buf);
    let n = buf.len().min(out.len());
    out[..n].copy_from_slice(&buf[..n]);
    n as u32
}

#[no_mangle]
pub extern "C" fn sim_debug_rays_ptr() -> *const f32 {
    let out = unsafe { (*(&raw mut RAYS_OUT)).get_or_insert_with(|| leak_f32(6144)) };
    out.as_ptr()
}

/// Version handshake for the flat job format (gpu.js checks this).
#[no_mangle]
pub extern "C" fn sim_gpu_job_version() -> u32 {
    GPU_JOB_VERSION
}

// -------------------------------------------------- PT-early bridge (C4)
// Same proxy pattern as the late-field bridge: when the JS driver is
// live and `early = traced`, each PathCache's discovery call queues a
// tiny job (id, box size, listener, rot — 8 f32) and consumes chain
// bitmaps injected from earlier dispatches. The permanent ≤2-order
// seeds keep the early field correct while a bitmap is in flight;
// with no driver the in-wasm CPU fan runs as always.

/// One PT job: [id, sx, sy, sz, lx, ly, lz, rot].
pub const PT_JOB_F32S: usize = 8;
const PT_MAX_JOBS: usize = 16;

struct WebPtProxy;

static PT_JOBS: std::sync::Mutex<Vec<f32>> = std::sync::Mutex::new(Vec::new());
static PT_BITMAPS: std::sync::Mutex<Vec<(u32, [u32; 9])>> = std::sync::Mutex::new(Vec::new());

/// Decode a pt_early.wgsl chain bitmap (layout v1: 6 + 36 + 216 slots).
fn decode_bitmap(words: &[u32; 9], out: &mut Vec<omg_core::pt::Chain>) {
    let bit = |i: usize| words[i >> 5] >> (i & 31) & 1 == 1;
    const NO: u8 = omg_core::pt::NO_WALL;
    for w1 in 0..6usize {
        if bit(w1) {
            out.push(([w1 as u8, NO, NO], 1));
        }
        for w2 in 0..6usize {
            if bit(6 + w1 * 6 + w2) {
                out.push(([w1 as u8, w2 as u8, NO], 2));
            }
            for w3 in 0..6usize {
                if bit(42 + w1 * 36 + w2 * 6 + w3) {
                    out.push(([w1 as u8, w2 as u8, w3 as u8], 3));
                }
            }
        }
    }
}

impl omg_scene::early::EarlyDiscovery for WebPtProxy {
    fn discover(
        &mut self,
        id: u32,
        room: &omg_core::scene::Shoebox,
        listener: omg_core::vec3::Vec3,
        rot: u32,
        out: &mut Vec<omg_core::pt::Chain>,
    ) -> bool {
        {
            let mut jobs = PT_JOBS.lock().unwrap();
            if jobs.len() < PT_MAX_JOBS * PT_JOB_F32S {
                jobs.extend_from_slice(&[
                    id as f32,
                    room.size.x,
                    room.size.y,
                    room.size.z,
                    listener.x,
                    listener.y,
                    listener.z,
                    rot as f32,
                ]);
            }
        }
        let mut maps = PT_BITMAPS.lock().unwrap();
        if let Some(i) = maps.iter().position(|(mid, _)| *mid == id) {
            let (_, words) = maps.swap_remove(i);
            decode_bitmap(&words, out);
        }
        true // seeds carry the early field while dispatches are in flight
    }
}

/// Drain queued PT discovery jobs (f32 count, multiple of PT_JOB_F32S).
#[no_mangle]
pub extern "C" fn sim_pt_jobs_len() -> u32 {
    let out = unsafe {
        (*(&raw mut PT_JOBS_OUT)).get_or_insert_with(|| leak_f32(PT_MAX_JOBS * PT_JOB_F32S))
    };
    let mut jobs = PT_JOBS.lock().unwrap();
    let n = jobs.len().min(out.len());
    out[..n].copy_from_slice(&jobs[..n]);
    jobs.clear();
    n as u32
}

#[no_mangle]
pub extern "C" fn sim_pt_jobs_ptr() -> *const f32 {
    let out = unsafe {
        (*(&raw mut PT_JOBS_OUT)).get_or_insert_with(|| leak_f32(PT_MAX_JOBS * PT_JOB_F32S))
    };
    out.as_ptr()
}

static mut PT_JOBS_OUT: Option<&'static mut [f32]> = None;
static mut PT_INJECT: Option<&'static mut [u32]> = None;

fn leak_u32(n: usize) -> &'static mut [u32] {
    Box::leak(vec![0u32; n].into_boxed_slice())
}

/// Staging for one chain bitmap (9 u32 words) before sim_pt_inject.
#[no_mangle]
pub extern "C" fn sim_pt_buf_ptr() -> *mut u32 {
    let buf = unsafe { (*(&raw mut PT_INJECT)).get_or_insert_with(|| leak_u32(9)) };
    buf.as_mut_ptr()
}

/// Deliver the staged bitmap as cache `id`'s discovery result.
#[no_mangle]
pub extern "C" fn sim_pt_inject(id: u32) {
    let buf = unsafe { (*(&raw mut PT_INJECT)).get_or_insert_with(|| leak_u32(9)) };
    let words: [u32; 9] = buf[..9].try_into().unwrap();
    let mut maps = PT_BITMAPS.lock().unwrap();
    maps.retain(|(mid, _)| *mid != id);
    maps.push((id, words));
}

// ------------------------------------------ C6d world-late bridge (K2)
// Same proxy pattern once more: with `early=traced` + GPU on, each
// budgeted world trace queues a flat job; JS dispatches trace_mesh.wgsl
// against the ONCE-uploaded world BVH and injects the echogram through
// the SAME sim_gpu_inject staging — Sim ids are globally unique, so
// `reverb_world`'s poll_into finds its result with zero new plumbing.

/// One world job, fixed stride: [0] sim id · [1] n_rays · [2] seed ·
/// [3..6] source · [6..9] listener · [9] n_panels · [10..] panels,
/// 12 f32 each laid out EXACTLY like omg-gpu's GpuPanel (48 B):
/// min xyz, scattering, max xyz, 0, absorption xyz, 0, transmission
/// xyz, 0 — JS memcpies the slice straight into the panels buffer.
pub const WLATE_MAX_PANELS: usize = 64;
pub const WLATE_JOB_F32S: usize = 10 + WLATE_MAX_PANELS * 16;
const WLATE_MAX_JOBS: usize = 4;
/// Mirrors omg-gpu layout::MESH_LAYOUT_VERSION — gpu.js refuses the
/// mesh pipeline on mismatch.
pub const WLATE_VERSION: u32 = 3;

struct WebWorldLateProxy;

static WLATE_JOBS: std::sync::Mutex<Vec<f32>> = std::sync::Mutex::new(Vec::new());

impl omg_scene::late::WorldLateBackend for WebWorldLateProxy {
    fn trace_world(
        &mut self,
        id: u32,
        src: omg_core::vec3::Vec3,
        lis: omg_core::vec3::Vec3,
        rays: u32,
        panels: &[(omg_core::vec3::Vec3, omg_core::vec3::Vec3, omg_core::material::Material)],
        _out: &mut omg_core::tracer::Echogram,
    ) -> bool {
        let mut jobs = WLATE_JOBS.lock().unwrap();
        if jobs.len() >= WLATE_MAX_JOBS * WLATE_JOB_F32S {
            return false; // JS stalled; the gate re-fires
        }
        // the GPU runs the one budgeted trace 8× denser (variance only)
        let n = (rays * 8).min(8192);
        static SEED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xC6D2);
        let seed = SEED
            .fetch_add(0x9E37_79B9, std::sync::atomic::Ordering::Relaxed);
        let base = jobs.len();
        jobs.extend_from_slice(&[id as f32, n as f32, seed as f32]);
        jobs.extend_from_slice(&[src.x, src.y, src.z]);
        jobs.extend_from_slice(&[lis.x, lis.y, lis.z]);
        let np = panels.len().min(WLATE_MAX_PANELS);
        jobs.push(np as f32);
        for (mn, mx, m) in panels.iter().take(np) {
            jobs.extend_from_slice(&[mn.x, mn.y, mn.z, m.scattering]);
            jobs.extend_from_slice(&[mx.x, mx.y, mx.z, 0.0]);
            jobs.extend_from_slice(&[m.absorption[0], m.absorption[1], m.absorption[2], 0.0]);
            jobs.extend_from_slice(&[m.transmission[0], m.transmission[1], m.transmission[2], 0.0]);
        }
        jobs.resize(base + WLATE_JOB_F32S, 0.0); // fixed stride
        false // result arrives via sim_gpu_inject → poll_into
    }
}

// -------------------------------------- C6d world-discovery bridge (K3)
// Discovery is listener-launched and source-independent, so the whole
// world needs ONE job per tick: [lx, ly, lz, rot]. JS dispatches
// discover_mesh.wgsl over the same uploaded BVH and injects the raw
// chain list (2 u32 per chain); the TTL cache dedups. While a job is in
// flight the provider reports pending — the CPU fan backstops after a
// grace window and the direct path never depends on discovery at all.

const WDISC_CAP: usize = 16384;

struct WebWorldDiscProxy;

static WDISC_JOB: std::sync::Mutex<Option<[f32; 5]>> = std::sync::Mutex::new(None);
static WDISC_CHAINS: std::sync::Mutex<Vec<omg_core::pt_mesh::MChain>> =
    std::sync::Mutex::new(Vec::new());

impl omg_scene::early_world::WorldDiscovery for WebWorldDiscProxy {
    fn discover(
        &mut self,
        listener: omg_core::vec3::Vec3,
        rot: u32,
        out: &mut Vec<omg_core::pt_mesh::MChain>,
    ) -> bool {
        let had = {
            let mut got = WDISC_CHAINS.lock().unwrap();
            let had = !got.is_empty();
            out.append(&mut got);
            had
        };
        // single job slot, newest pose wins; the 5th float carries the
        // live furniture switch (0 = kernel skips the overlay boxes)
        let furn = if omg_scene::quality::furniture_on() { 1.0 } else { 0.0 };
        *WDISC_JOB.lock().unwrap() =
            Some([listener.x, listener.y, listener.z, rot as f32, furn]);
        had
    }
}

static mut WDISC_JOB_OUT: Option<&'static mut [f32]> = None;
static mut WDISC_INJECT: Option<&'static mut [u32]> = None;

/// This tick's discovery job, if any: 5 f32
/// [lx, ly, lz, rot, furniture_on]; 0 = none.
#[no_mangle]
pub extern "C" fn sim_wdisc_jobs_len() -> u32 {
    let out =
        unsafe { (*(&raw mut WDISC_JOB_OUT)).get_or_insert_with(|| leak_f32(5)) };
    match WDISC_JOB.lock().unwrap().take() {
        Some(j) => {
            out.copy_from_slice(&j);
            5
        }
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn sim_wdisc_jobs_ptr() -> *const f32 {
    let out =
        unsafe { (*(&raw mut WDISC_JOB_OUT)).get_or_insert_with(|| leak_f32(5)) };
    out.as_ptr()
}

// Furniture overlay boxes for the discovery kernel: 8 u32 words per box
// (min xyz as f32 bits, 0, max xyz, 0) + the first pseudo-surface id.
// Static like the BVH — uploaded once by gpu.js.
static mut FURN_FLAT: Option<(Vec<u32>, u32)> = None;

fn furn_flat() -> &'static (Vec<u32>, u32) {
    unsafe {
        (*(&raw mut FURN_FLAT)).get_or_insert_with(|| {
            let rooms = omg_scene::walkthrough::rooms();
            let doors = omg_scene::walkthrough::doors();
            let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
            let base = omg_core::pt_mesh::SurfaceTable::build(&mesh).base_overlay as u32;
            let mut words = Vec::new();
            for (ri, r) in rooms.iter().enumerate() {
                for a in omg_scene::walkthrough::furniture(ri) {
                    let d = a.max - a.min;
                    if d.x * d.y * d.z <= omg_scene::walkthrough::FURN_REFLECTOR_MIN_VOL {
                        continue;
                    }
                    let (ox, oy, oz) = (r.min.0, r.min.1, r.floor_z);
                    words.extend_from_slice(&[
                        (a.min.x + ox).to_bits(),
                        (a.min.y + oy).to_bits(),
                        (a.min.z + oz).to_bits(),
                        0,
                        (a.max.x + ox).to_bits(),
                        (a.max.y + oy).to_bits(),
                        (a.max.z + oz).to_bits(),
                        0,
                    ]);
                }
            }
            (words, base)
        })
    }
}

#[no_mangle]
pub extern "C" fn sim_wdisc_boxes_len() -> u32 {
    furn_flat().0.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_wdisc_boxes_ptr() -> *const u32 {
    furn_flat().0.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_wdisc_base() -> u32 {
    furn_flat().1
}

fn leak_u32_buf(n: usize) -> &'static mut [u32] {
    Box::leak(vec![0u32; n].into_boxed_slice())
}

/// Staging JS writes the raw chain list into (2 u32 per chain:
/// (s0 | s1<<16), (s2 | order<<16)) before sim_wdisc_inject(n_chains).
#[no_mangle]
pub extern "C" fn sim_wdisc_buf_ptr() -> *mut u32 {
    let buf = unsafe {
        (*(&raw mut WDISC_INJECT)).get_or_insert_with(|| leak_u32_buf(WDISC_CAP * 2))
    };
    buf.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn sim_wdisc_inject(n: u32) {
    let buf = unsafe {
        (*(&raw mut WDISC_INJECT)).get_or_insert_with(|| leak_u32_buf(WDISC_CAP * 2))
    };
    let n = (n as usize).min(WDISC_CAP);
    let mut chains = WDISC_CHAINS.lock().unwrap();
    chains.clear();
    for i in 0..n {
        let (w0, w1) = (buf[i * 2], buf[i * 2 + 1]);
        let chain = [(w0 & 0xFFFF) as u16, (w0 >> 16) as u16, (w1 & 0xFFFF) as u16];
        let order = ((w1 >> 16) as u8).clamp(1, 3);
        chains.push((chain, order));
    }
}

/// Route world discovery to the JS driver — like sim_wlate_enable, only
/// after gpu.js confirmed the discovery pipeline built.
#[no_mangle]
pub extern "C" fn sim_wdisc_enable() {
    omg_scene::early_world::set_world_discovery(Box::new(WebWorldDiscProxy));
}

static mut WLATE_JOBS_OUT: Option<&'static mut [f32]> = None;

/// Drain this tick's world-trace jobs; returns the f32 count (a
/// multiple of WLATE_JOB_F32S).
#[no_mangle]
pub extern "C" fn sim_wlate_jobs_len() -> u32 {
    let out = unsafe {
        (*(&raw mut WLATE_JOBS_OUT))
            .get_or_insert_with(|| leak_f32(WLATE_MAX_JOBS * WLATE_JOB_F32S))
    };
    let mut jobs = WLATE_JOBS.lock().unwrap();
    let n = jobs.len().min(out.len());
    out[..n].copy_from_slice(&jobs[..n]);
    jobs.clear();
    n as u32
}

#[no_mangle]
pub extern "C" fn sim_wlate_jobs_ptr() -> *const f32 {
    let out = unsafe {
        (*(&raw mut WLATE_JOBS_OUT))
            .get_or_insert_with(|| leak_f32(WLATE_MAX_JOBS * WLATE_JOB_F32S))
    };
    out.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_wlate_version() -> u32 {
    WLATE_VERSION
}

// The world mesh, flattened for the kernel: u32 words with f32 payloads
// stored as bits (JS reads Float32Array + Uint32Array views over the
// same range). Node = 8 words [bmin×3, a, bmax×3, b]; prim = 12 words
// [a×3, mat, e1×3, 0, e2×3, 0]; mat = 4 words [absorption×3,
// scattering]. Built lazily from the same rooms/doors the sim uses.
static mut MESH_FLAT: Option<(Vec<u32>, Vec<u32>, Vec<u32>)> = None;

fn mesh_flat() -> &'static (Vec<u32>, Vec<u32>, Vec<u32>) {
    unsafe {
        (*(&raw mut MESH_FLAT)).get_or_insert_with(|| {
            let rooms = omg_scene::walkthrough::rooms();
            let doors = omg_scene::walkthrough::doors();
            let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
            let mut nodes = Vec::new();
            let mut prims = Vec::new();
            mesh.visit_bvh(
                &mut |bmin, bmax, a, b| {
                    nodes.extend_from_slice(&[
                        bmin.x.to_bits(),
                        bmin.y.to_bits(),
                        bmin.z.to_bits(),
                        a,
                        bmax.x.to_bits(),
                        bmax.y.to_bits(),
                        bmax.z.to_bits(),
                        b,
                    ]);
                },
                &mut |a, e1, e2, m, surf| {
                    prims.extend_from_slice(&[
                        a.x.to_bits(),
                        a.y.to_bits(),
                        a.z.to_bits(),
                        m as u32,
                        e1.x.to_bits(),
                        e1.y.to_bits(),
                        e1.z.to_bits(),
                        surf as u32,
                        e2.x.to_bits(),
                        e2.y.to_bits(),
                        e2.z.to_bits(),
                        0,
                    ]);
                },
            );
            let mut mats = Vec::new();
            for m in &mesh.materials {
                mats.extend_from_slice(&[
                    m.absorption[0].to_bits(),
                    m.absorption[1].to_bits(),
                    m.absorption[2].to_bits(),
                    m.scattering.to_bits(),
                    m.transmission[0].to_bits(),
                    m.transmission[1].to_bits(),
                    m.transmission[2].to_bits(),
                    0,
                ]);
            }
            (nodes, prims, mats)
        })
    }
}

#[no_mangle]
pub extern "C" fn sim_mesh_nodes_len() -> u32 {
    mesh_flat().0.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_mesh_nodes_ptr() -> *const u32 {
    mesh_flat().0.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_mesh_prims_len() -> u32 {
    mesh_flat().1.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_mesh_prims_ptr() -> *const u32 {
    mesh_flat().1.as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_mesh_mats_len() -> u32 {
    mesh_flat().2.len() as u32
}

#[no_mangle]
pub extern "C" fn sim_mesh_mats_ptr() -> *const u32 {
    mesh_flat().2.as_ptr()
}

/// Route traces through the JS WebGPU driver. Call once, before the
/// first tick, and only after gpu.js initialized successfully.
#[no_mangle]
pub extern "C" fn sim_gpu_enable() {
    omg_scene::late::set_late_backend(Box::new(WebGpuProxy));
    omg_scene::early::set_early_discovery(Box::new(WebPtProxy));
    omg_scene::quality::set_gpu_backend(true);
}

/// Route WORLD late traces to the JS driver — called SEPARATELY, and
/// only after gpu.js confirmed the mesh pipeline compiled and the BVH
/// uploaded. Registering it blind would queue jobs nobody dispatches
/// and starve traced-mode reverb into silence.
#[no_mangle]
pub extern "C" fn sim_wlate_enable() {
    omg_scene::late::set_world_late_backend(Box::new(WebWorldLateProxy));
}

/// Live A/B back to the CPU tracer (tuning-panel toggle). In-flight
/// jobs/results are dropped; the trace gate re-fires them on CPU.
#[no_mangle]
pub extern "C" fn sim_gpu_disable() {
    omg_scene::late::clear_late_backend();
    omg_scene::late::clear_world_late_backend();
    omg_scene::early::clear_early_discovery();
    omg_scene::early_world::clear_world_discovery();
    omg_scene::quality::set_gpu_backend(false);
    GPU_JOBS.lock().unwrap().clear();
    GPU_RESULTS.lock().unwrap().clear();
    PT_JOBS.lock().unwrap().clear();
    PT_BITMAPS.lock().unwrap().clear();
    WLATE_JOBS.lock().unwrap().clear();
    *WDISC_JOB.lock().unwrap() = None;
    WDISC_CHAINS.lock().unwrap().clear();
}

static mut GPU_JOBS_OUT: Option<&'static mut [f32]> = None;
static mut GPU_INJECT: Option<&'static mut [f32]> = None;

/// Drain this tick's queued trace jobs into the export buffer; returns
/// the f32 count (a multiple of GPU_JOB_F32S).
#[no_mangle]
pub extern "C" fn sim_gpu_jobs_len() -> u32 {
    let out = unsafe {
        (*(&raw mut GPU_JOBS_OUT)).get_or_insert_with(|| leak_f32(GPU_MAX_JOBS * GPU_JOB_F32S))
    };
    let mut jobs = GPU_JOBS.lock().unwrap();
    let n = jobs.len().min(out.len());
    out[..n].copy_from_slice(&jobs[..n]);
    jobs.clear();
    n as u32
}

#[no_mangle]
pub extern "C" fn sim_gpu_jobs_ptr() -> *const f32 {
    let out = unsafe {
        (*(&raw mut GPU_JOBS_OUT)).get_or_insert_with(|| leak_f32(GPU_MAX_JOBS * GPU_JOB_F32S))
    };
    out.as_ptr()
}

/// Staging buffer JS writes one decoded echogram into (1800 f32:
/// bins[300×3] then dirs[300×3]) before calling sim_gpu_inject.
#[no_mangle]
pub extern "C" fn sim_gpu_buf_ptr() -> *mut f32 {
    let buf =
        unsafe { (*(&raw mut GPU_INJECT)).get_or_insert_with(|| leak_f32(GPU_ECHO_F32S)) };
    buf.as_mut_ptr()
}

/// Deliver the staged echogram as sim `id`'s trace result.
#[no_mangle]
pub extern "C" fn sim_gpu_inject(id: u32) {
    let buf =
        unsafe { (*(&raw mut GPU_INJECT)).get_or_insert_with(|| leak_f32(GPU_ECHO_F32S)) };
    let mut results = GPU_RESULTS.lock().unwrap();
    // one pending result per sim id: newest wins
    results.retain(|(rid, _)| *rid != id);
    results.push((id, buf.to_vec()));
}

/// Pin one quality lever independently of the tier (a tuning panel's
/// sliders). Ids: 0 trace rays, 1 gate age, 2 dome rays, 3 dome events,
/// 4 ISM order. `value` 0 hands the lever back to the tier.
#[no_mangle]
pub extern "C" fn sim_set_override(id: u32, value: u32) {
    omg_scene::quality::set_override(id, value);
}

/// Mixer kill switch: a muted source is skipped by the whole
/// simulation and ships silent blocks.
#[no_mangle]
pub extern "C" fn sim_set_mute(i: u32, on: u32) {
    sim().world.set_muted(i as usize, on != 0);
}

/// A/B module switches (quality panel): 0 = edge diffraction,
/// 1 = furniture acoustics. All default on.
#[no_mangle]
pub extern "C" fn sim_set_module(id: u32, on: u32) {
    omg_scene::quality::set_module(id, on != 0);
}

#[no_mangle]
pub extern "C" fn sim_tick(lx: f32, ly: f32, lz: f32, yaw: f32) {
    let ctx = sim();
    for i in 0..8 {
        // animated leaf position — the swing sweeps the filters
        ctx.world.set_door(i, ctx.door_in[i]);
    }
    for slot in 0..omg_scene::walkthrough::DYN_SLOTS {
        let o = slot * 4;
        ctx.world.set_dynamic(
            slot,
            ctx.dyn_in[o],
            ctx.dyn_in[o + 1],
            ctx.dyn_in[o + 2],
            ctx.dyn_in[o + 3],
        );
    }
    let (blocks, info) = ctx.world.tick_at_z(lx, ly, lz, yaw);
    for (i, pb) in blocks.iter().enumerate().take(NSRC) {
        pb.write_flat(&mut ctx.flat_tmp);
        let n = ctx.flat_tmp.len().min(MAX_FLAT);
        ctx.params[i][..n].copy_from_slice(&ctx.flat_tmp[..n]);
        ctx.param_lens[i] = n;
    }
    let st = &mut ctx.state;
    st[0] = info.listener.0;
    st[1] = info.listener.1;
    st[2] = info.room as f32;
    st[3] = info.rt60_mid;
    // Environment: geometry-priced routing of the outdoor field (ambience
    // + rain) — aperture inlets, shell seep, roof exposure. No per-room
    // constants: the power balance and the blend zones decide.
    info.env.write_flat(&mut st[ENV_OFF..]);
    let mut o = 4;
    for route in info.routes.iter().take(NSRC) {
        let n = route.len().min(4);
        st[o] = n as f32;
        o += 1;
        for p in route.iter().take(4) {
            st[o] = p.0;
            st[o + 1] = p.1;
            o += 2;
        }
        o += (4 - n) * 2;
    }
}

#[no_mangle]
pub extern "C" fn sim_params_ptr(i: u32) -> *const f32 {
    let ctx = sim();
    ctx.params[i as usize].as_ptr()
}

#[no_mangle]
pub extern "C" fn sim_params_len(i: u32) -> u32 {
    let ctx = sim();
    ctx.param_lens[i as usize] as u32
}

#[no_mangle]
pub extern "C" fn sim_state_ptr() -> *const f32 {
    let ctx = sim();
    ctx.state.as_ptr()
}

// ============================================================== ENGINE SIDE

struct SourceState {
    data: Vec<f32>,
    pos: usize,
}

/// One-shot playback of an fx buffer into a source's signal.
struct Voice {
    src: usize,
    buf: usize,
    pos: usize,
    /// Per-voice level and playback rate (1, 1 = the buffer as recorded:
    /// sample-identical to plain playback). `phase` is the fractional
    /// read position between `pos` and `pos + 1`.
    gain: f32,
    rate: f32,
    phase: f32,
}

impl Voice {
    fn plain(src: usize, buf: usize) -> Self {
        Self { src, buf, pos: 0, gain: 1.0, rate: 1.0, phase: 0.0 }
    }

    /// Current sample (linear interpolation; exact at phase 0).
    fn sample(&self, b: &[f32]) -> f32 {
        let a = b[self.pos];
        if self.phase == 0.0 {
            return a * self.gain;
        }
        let c = b.get(self.pos + 1).copied().unwrap_or(0.0);
        (a + (c - a) * self.phase) * self.gain
    }

    fn advance(&mut self) {
        self.phase += self.rate;
        while self.phase >= 1.0 {
            self.phase -= 1.0;
            self.pos += 1;
        }
    }
}

/// A NEAR-FIELD one-shot (a whisper at the ear): an fx-bank buffer fed
/// through `omg_dsp::near::NearVoice`, rendered straight into the ear
/// buffers and bypassing propagation entirely (see that module for the
/// model: per-ear 1/r around the head, ITD, head-shadow lowpass).
struct NearPlay {
    buf: usize,
    pos: usize,
    ear: omg_dsp::near::NearVoice,
}

struct EngCtx {
    renderers: Vec<Renderer>,
    sources: Vec<SourceState>,
    out: Option<OutputStage>,
    sample_rate: f32,
    point_budget: usize,
    tap_ceiling: usize,
    /// Mix-wide budgets (0 = off: per-source budgets as before). When set,
    /// every committed tick ranks ALL sources' taps by loudness at the
    /// ears and hands each source its share — render cost is bounded by
    /// these totals, not by how many sources play (SOURCES_PLAN S1).
    mix_taps: usize,
    mix_points: usize,
    /// Latest block per source, for the mix-wide ranking.
    latest: Vec<Option<ParamBlock>>,
    /// Current mix-wide shares per source: (taps, points).
    shares: Vec<(usize, usize)>,
    grid: Option<Arc<HrirGrid>>,
    // staging buffers JS writes into
    hrir_grid_buf: Option<&'static mut [u8]>,
    hrir_spk_buf: Option<&'static mut [u8]>,
    param_buf: &'static mut [f32],
    out_l: &'static mut [f32],
    out_r: &'static mut [f32],
    fx_bufs: Vec<Vec<f32>>,
    fx_stage: Option<&'static mut [f32]>,
    voices: Vec<Voice>,
    near_voices: Vec<NearPlay>,
    ambient_stage: Option<&'static mut [f32]>,
    ambience: omg_dsp::ambience::Ambience,
    rain: omg_dsp::rain::Rain,
    /// Mixer: per-source user gains (smoothed toward targets), plus
    /// ambience and master. Source faders are POWER faders — the UI maps
    /// an SPL scale (needle drop … jet engine) to these linear gains.
    /// NSRC faders exist from init (the page may set them before the
    /// sources load); a source allocated past NSRC (benchmarks only —
    /// tools/bench_sources.mjs) grows its own unit fader.
    mixer: Vec<omg_dsp::smooth::Smoothed>,
    master: omg_dsp::smooth::Smoothed,
    /// Per-channel meter accumulators (NSRC sources + ambience + rain):
    /// (peak², Σm², n) since the last commit.
    meter_acc: [(f32, f64, u32); NSRC + 2],
    meter_out: &'static mut [f32],
    /// Field-debug snapshot buffer (eng_debug_render).
    debug_out: &'static mut [f32],
}

static mut ENG: Option<EngCtx> = None;

#[no_mangle]
pub extern "C" fn eng_init(sample_rate: f32) {
    let ctx = EngCtx {
        renderers: Vec::new(),
        sources: Vec::new(),
        out: None,
        sample_rate,
        point_budget: 8,
        tap_ceiling: 160,
        mix_taps: 0,
        mix_points: 0,
        latest: Vec::new(),
        shares: Vec::new(),
        grid: None,
        hrir_grid_buf: None,
        hrir_spk_buf: None,
        param_buf: leak_f32(MAX_FLAT),
        out_l: leak_f32(MAX_BLOCK),
        out_r: leak_f32(MAX_BLOCK),
        fx_bufs: Vec::new(),
        fx_stage: None,
        voices: Vec::new(),
        near_voices: Vec::new(),
        ambient_stage: None,
        ambience: omg_dsp::ambience::Ambience::new(sample_rate),
        rain: omg_dsp::rain::Rain::new(sample_rate),
        mixer: (0..NSRC).map(|_| omg_dsp::smooth::Smoothed::new(1.0, 0.02, sample_rate)).collect(),
        master: omg_dsp::smooth::Smoothed::new(1.0, 0.02, sample_rate),
        meter_acc: [(0.0, 0.0, 0); NSRC + 2],
        meter_out: leak_f32(2 * (NSRC + 2)),
        debug_out: leak_f32(NSRC * 5),
    };
    unsafe { *(&raw mut ENG) = Some(ctx) };
}

fn eng() -> &'static mut EngCtx {
    unsafe { (*(&raw mut ENG)).as_mut().expect("eng_init first") }
}

#[no_mangle]
pub extern "C" fn eng_hrir_grid_alloc(nbytes: u32) -> *mut u8 {
    let ctx = eng();
    ctx.hrir_grid_buf = Some(leak_u8(nbytes as usize));
    ctx.hrir_grid_buf.as_mut().unwrap().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn eng_hrir_grid_done() {
    let ctx = eng();
    let buf = ctx.hrir_grid_buf.take().expect("alloc first");
    ctx.grid = Some(Arc::new(HrirGrid::from_bytes(buf)));
}

#[no_mangle]
pub extern "C" fn eng_hrir_speakers_alloc(nbytes: u32) -> *mut u8 {
    let ctx = eng();
    ctx.hrir_spk_buf = Some(leak_u8(nbytes as usize));
    ctx.hrir_spk_buf.as_mut().unwrap().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn eng_hrir_speakers_done() {
    let ctx = eng();
    let buf = ctx.hrir_spk_buf.take().expect("alloc first");
    ctx.out = Some(OutputStage::from_speaker_bytes(Some(buf), ctx.sample_rate));
}

/// Import-normalize source `i`'s buffer (gated RMS → reference): how a
/// clip was recorded stops mattering; its mixer SPL type sets the energy.
#[no_mangle]
pub extern "C" fn eng_source_commit(i: u32) {
    let ctx = eng();
    if let Some(s) = ctx.sources.get_mut(i as usize) {
        omg_dsp::level::normalize_rms(&mut s.data, omg_dsp::level::REF_CLIP_RMS);
    }
}

/// Allocate the loop buffer for source `i` (mono samples at engine rate)
/// and create its renderer. Call in source-index order.
#[no_mangle]
pub extern "C" fn eng_source_alloc(i: u32, nsamples: u32) -> *mut f32 {
    let ctx = eng();
    assert_eq!(i as usize, ctx.sources.len(), "sources in order");
    ctx.sources.push(SourceState { data: vec![0.0; nsamples as usize], pos: 0 });
    let mut r = Renderer::with_grid(ctx.sample_rate, ctx.grid.clone());
    r.set_point_budget(ctx.point_budget);
    r.set_tap_ceiling(ctx.tap_ceiling);
    ctx.renderers.push(r);
    if ctx.mixer.len() < ctx.renderers.len() {
        ctx.mixer.push(omg_dsp::smooth::Smoothed::new(1.0, 0.02, ctx.sample_rate));
    }
    ctx.sources.last_mut().unwrap().data.as_mut_ptr()
}

/// Per-source point-render budget (strongest N taps get their own HRIR
/// convolution). The page sets this from measured platform headroom.
#[no_mangle]
pub extern "C" fn eng_set_point_budget(n: u32) {
    let ctx = eng();
    ctx.point_budget = n as usize;
    for r in &mut ctx.renderers {
        r.set_point_budget(n as usize);
    }
}

/// Per-source cap on incoming taps kept per ParamBlock. The load governor
/// lowers it when the render misses deadlines; evicted taps fade out.
#[no_mangle]
pub extern "C" fn eng_set_tap_ceiling(n: u32) {
    let ctx = eng();
    ctx.tap_ceiling = n as usize;
    for r in &mut ctx.renderers {
        r.set_tap_ceiling(n as usize);
    }
}

/// Stage an fx buffer (call in kind order 0,1,2,…), then eng_fx_commit.
#[no_mangle]
pub extern "C" fn eng_fx_alloc(nsamples: u32) -> *mut f32 {
    let ctx = eng();
    ctx.fx_stage = Some(leak_f32(nsamples as usize));
    ctx.fx_stage.as_mut().unwrap().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn eng_fx_commit() {
    let ctx = eng();
    let buf = ctx.fx_stage.take().expect("alloc first");
    ctx.fx_bufs.push(buf.to_vec());
}

#[no_mangle]
pub extern "C" fn eng_fx_play(src: u32, kind: u32) {
    eng_fx_play_var(src, kind, 1.0, 1.0);
}

/// One-shot with its own level and playback rate (rate 1.03 = 3% faster
/// and a third of a semitone higher) — natural variation for repeated
/// sounds like footsteps.
#[no_mangle]
pub extern "C" fn eng_fx_play_var(src: u32, kind: u32, gain: f32, rate: f32) {
    let ctx = eng();
    if (kind as usize) < ctx.fx_bufs.len() && ctx.voices.len() < 24 {
        let mut v = Voice::plain(src as usize, kind as usize);
        v.gain = gain.clamp(0.0, 4.0);
        v.rate = rate.clamp(0.5, 2.0);
        ctx.voices.push(v);
    }
}

#[no_mangle]
pub extern "C" fn eng_fx_stop(src: u32, kind: u32) {
    let ctx = eng();
    ctx.voices.retain(|v| !(v.src == src as usize && v.buf == kind as usize));
}

/// Near-field play: an fx-bank buffer whispered `dist_m` from one ear
/// (`right != 0` = right ear). `gain` is the near-ear amplitude at the
/// reference 6 cm; the far ear derives from the around-the-head path.
#[no_mangle]
pub extern "C" fn eng_whisper_play(kind: u32, right: u32, dist_m: f32, gain: f32) {
    let ctx = eng();
    if (kind as usize) >= ctx.fx_bufs.len() || ctx.near_voices.len() >= omg_dsp::near::MAX_VOICES {
        return;
    }
    ctx.near_voices.push(NearPlay {
        buf: kind as usize,
        pos: 0,
        ear: omg_dsp::near::NearVoice::new(ctx.sample_rate, right != 0, dist_m, gain),
    });
}

#[no_mangle]
pub extern "C" fn eng_ambient_alloc(nsamples: u32) -> *mut f32 {
    let ctx = eng();
    ctx.ambient_stage = Some(leak_f32(nsamples as usize));
    ctx.ambient_stage.as_mut().unwrap().as_mut_ptr()
}

/// channels: 1 = mono, 2 = interleaved stereo.
#[no_mangle]
pub extern "C" fn eng_ambient_commit(channels: u32) {
    let ctx = eng();
    let buf = ctx.ambient_stage.take().expect("alloc first");
    let mut data = buf.to_vec();
    // Beds play at their AUTHORED loudness — no gated re-normalization
    // (a gate over a sparse cricket bed measures only the chirps and
    // boosts the whole file to chirp level). Slow-loudness flattening
    // still applies: a passage recorded next to a cricket must not
    // surge out of the background.
    omg_dsp::level::flatten_slow_loudness(&mut data, channels as usize, ctx.sample_rate);
    ctx.ambience.set_loop(data, channels == 2);
}

/// Mixer: source fader (linear gain; the UI's SPL scale maps to this).
#[no_mangle]
pub extern "C" fn eng_set_mixer(i: u32, gain: f32) {
    if let Some(m) = eng().mixer.get_mut(i as usize) {
        m.set(gain.clamp(0.0, 256.0));
    }
}

#[no_mangle]
pub extern "C" fn eng_set_ambient_user(gain: f32) {
    eng().ambience.set_user(gain);
}

#[no_mangle]
pub extern "C" fn eng_set_rain_gain(gain: f32) {
    eng().rain.set_gain(gain);
}

#[no_mangle]
pub extern "C" fn eng_set_master(gain: f32) {
    eng().master.set(gain.clamp(0.0, 4.0));
}

/// Commit per-channel meters (peak, rms) × 8 into the meter buffer and
/// reset the accumulators. Returns the buffer pointer.
#[no_mangle]
pub extern "C" fn eng_meters_commit() -> *const f32 {
    let ctx = eng();
    for (ch, acc) in ctx.meter_acc.iter_mut().enumerate() {
        let (p2, s2, n) = *acc;
        ctx.meter_out[ch * 2] = p2.sqrt();
        ctx.meter_out[ch * 2 + 1] = if n > 0 { ((s2 / n as f64) as f32).sqrt() } else { 0.0 };
        *acc = (0.0, 0.0, 0);
    }
    ctx.meter_out.as_ptr()
}

/// Stage + commit the recorded-splat bank for the rain (mono f32,
/// uniform 150 ms slices — see tools/make_drops.py).
#[no_mangle]
pub extern "C" fn eng_rain_bank_alloc(nsamples: u32) -> *mut f32 {
    let ctx = eng();
    ctx.fx_stage = Some(leak_f32(nsamples as usize));
    ctx.fx_stage.as_mut().unwrap().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn eng_rain_bank_commit() {
    let ctx = eng();
    let buf = ctx.fx_stage.take().expect("alloc first");
    ctx.rain.set_bank(buf.to_vec());
}

/// Rain intensity 0…1 (ramped inside; rain starts/stops like weather).
#[no_mangle]
pub extern "C" fn eng_set_rain(intensity: f32) {
    eng().rain.set_intensity(intensity);
}

/// Environment block staged in the param buffer (the flat form of
/// omg_dsp::env::Environment, copied from the sim state): geometry-priced
/// routing for ambience and rain.
#[no_mangle]
pub extern "C" fn eng_set_env(len: u32) {
    let ctx = eng();
    if (len as usize) < omg_dsp::env::ENV_FLAT_LEN {
        return;
    }
    let env = omg_dsp::env::Environment::read_flat(ctx.param_buf);
    ctx.ambience.set_environment(&env);
    ctx.rain.set_environment(&env);
}

#[no_mangle]
pub extern "C" fn eng_param_buf_ptr() -> *mut f32 {
    eng().param_buf.as_mut_ptr()
}

/// Apply the flat ParamBlock currently staged in the param buffer.
#[no_mangle]
pub extern "C" fn eng_set_params(src: u32, len: u32) {
    let ctx = eng();
    let pb = ParamBlock::read_flat(&ctx.param_buf[..len as usize]);
    if let Some(r) = ctx.renderers.get_mut(src as usize) {
        r.set_params(&pb);
    }
    let i = src as usize;
    if ctx.latest.len() <= i {
        ctx.latest.resize(i + 1, None);
    }
    ctx.latest[i] = Some(pb);
}

/// Mix-wide render budgets: at most `taps` taps and `points` point-HRTF
/// convolutions across ALL sources together (0, 0 = back to the
/// per-source budgets). Shares take effect from the next committed tick.
#[no_mangle]
pub extern "C" fn eng_set_mix_budget(taps: u32, points: u32) {
    let ctx = eng();
    ctx.mix_taps = taps as usize;
    ctx.mix_points = points as usize;
    ctx.shares.clear();
    if taps == 0 {
        let (pb, tc) = (ctx.point_budget, ctx.tap_ceiling);
        for r in &mut ctx.renderers {
            r.set_point_budget(pb);
            r.set_tap_ceiling(tc);
        }
    }
}

/// After a tick's blocks are in: rank every tap of every source by its
/// loudness at the ears (path gain × mixer fader) and give each source
/// the count of its taps among the strongest `mix_taps` (≥ 1, so a
/// source always keeps its strongest path) and among the strongest
/// `mix_points` for point-HRTF rendering. Applied with each renderer's
/// next block. Linear time (selection, not a sort).
#[no_mangle]
pub extern "C" fn eng_params_commit() {
    let ctx = eng();
    if ctx.mix_taps == 0 {
        return;
    }
    let mut all: Vec<(f32, u16)> = Vec::new();
    for (i, pb) in ctx.latest.iter().enumerate() {
        let Some(pb) = pb else { continue };
        let fader = ctx.mixer.get(i).map_or(1.0, |m| m.target_val()).max(0.0).sqrt();
        for t in &pb.taps {
            all.push((t.gains[1] * fader, i as u16));
        }
    }
    let n_src = ctx.renderers.len();
    let mut taps = vec![0usize; n_src];
    let mut points = vec![0usize; n_src];
    let mut count_top = |k: usize, out: &mut Vec<usize>, all: &mut Vec<(f32, u16)>| {
        if k == 0 {
            return;
        }
        if all.len() > k {
            all.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        }
        for &(_, s) in all.iter().take(k) {
            if let Some(c) = out.get_mut(s as usize) {
                *c += 1;
            }
        }
    };
    count_top(ctx.mix_taps, &mut taps, &mut all);
    count_top(ctx.mix_points, &mut points, &mut all);
    if ctx.shares.len() < n_src {
        ctx.shares.resize(n_src, (usize::MAX, usize::MAX));
    }
    for (i, r) in ctx.renderers.iter_mut().enumerate() {
        let share = (taps[i].max(1), points[i]);
        if ctx.shares[i] == share {
            continue;
        }
        ctx.shares[i] = share;
        r.set_tap_ceiling(share.0);
        r.set_point_budget(share.1);
        // a changed share takes effect now, not one tick later
        if let Some(Some(pb)) = ctx.latest.get(i) {
            r.invalidate_params();
            r.set_params(pb);
        }
    }
}

/// Per-source render occupancy → [live_taps, point_taps, tap_gain_mid,
/// fdn_send, remote_send] × NSRC. The debug panel's "what is actually
/// playing right now, and why" — read alongside the per-channel meters.
#[no_mangle]
pub extern "C" fn eng_debug_render() -> *const f32 {
    let ctx = eng();
    for (i, r) in ctx.renderers.iter().enumerate().take(NSRC) {
        let (live, points, gain, fdn, remote) = r.debug_stats();
        let o = i * 5;
        ctx.debug_out[o] = live as f32;
        ctx.debug_out[o + 1] = points as f32;
        ctx.debug_out[o + 2] = gain;
        ctx.debug_out[o + 3] = fdn;
        ctx.debug_out[o + 4] = remote;
    }
    ctx.debug_out.as_ptr()
}

/// Fast head orientation (yaw/pitch/roll, see `HeadRotation` for the
/// conventions) — applied at the DSP without waiting for a sim tick.
#[no_mangle]
pub extern "C" fn eng_set_head(yaw: f32, pitch: f32, roll: f32) {
    let ctx = eng();
    for r in &mut ctx.renderers {
        r.set_head(yaw, pitch, roll);
    }
    if let Some(o) = &mut ctx.out {
        o.set_head(yaw, pitch, roll);
    }
}

/// Replace source `i`'s loop buffer (the demo swaps car motors per
/// spawn). Write samples at the returned pointer, then
/// eng_source_commit(i) to import-normalize. The old buffer is freed.
#[no_mangle]
pub extern "C" fn eng_source_replace_alloc(i: u32, nsamples: u32) -> *mut f32 {
    let ctx = eng();
    let s = &mut ctx.sources[i as usize];
    s.data = vec![0.0; nsamples as usize];
    s.pos = 0;
    s.data.as_mut_ptr()
}

/// Ambience internals for field debugging: [user, seep×3, slot mid ×8].
#[no_mangle]
pub extern "C" fn eng_amb_debug() -> *const f32 {
    let ctx = eng();
    ctx.ambience.debug_state(&mut ctx.meter_out[..12]);
    ctx.meter_out.as_ptr()
}

/// Current ear-adaptation (AGC) gain, for the HUD meters.
#[no_mangle]
pub extern "C" fn eng_agc_gain() -> f32 {
    eng().out.as_ref().map_or(1.0, |o| o.agc_gain())
}

/// Hearing fatigue 0…1 (temporary threshold shift after ultra-loud).
#[no_mangle]
pub extern "C" fn eng_ear_fatigue() -> f32 {
    eng().out.as_ref().map_or(0.0, |o| o.ear_fatigue())
}

#[no_mangle]
pub extern "C" fn eng_out_l() -> *const f32 {
    eng().out_l.as_ptr()
}

#[no_mangle]
pub extern "C" fn eng_out_r() -> *const f32 {
    eng().out_r.as_ptr()
}

/// Render `n` samples into the output buffers.
#[no_mangle]
pub extern "C" fn eng_process(n: u32) {
    let ctx = eng();
    let n = (n as usize).min(MAX_BLOCK);
    for i in 0..n {
        let mut bus = [0.0f32; NCH];
        let mut pl = 0.0f32;
        let mut pr = 0.0f32;
        for (si, (src, ren)) in
            ctx.sources.iter_mut().zip(ctx.renderers.iter_mut()).enumerate()
        {
            // muted and fully faded: keep the stream and any one-shots
            // ticking (so unmute resumes in time) but skip ALL rendering
            // — taps, HRIRs, buses. A muted channel costs nothing.
            if ctx.mixer[si].target_val() == 0.0 && ctx.mixer[si].current() < 1e-4 {
                if !src.data.is_empty() {
                    src.pos = (src.pos + 1) % src.data.len();
                }
                for v in &mut ctx.voices {
                    if v.src == si && v.pos < ctx.fx_bufs[v.buf].len() {
                        v.advance();
                    }
                }
                continue;
            }
            let mut x = if src.data.is_empty() {
                0.0
            } else {
                let s = src.data[src.pos];
                src.pos = (src.pos + 1) % src.data.len();
                s
            };
            for v in &mut ctx.voices {
                if v.src == si && v.pos < ctx.fx_bufs[v.buf].len() {
                    x += v.sample(&ctx.fx_bufs[v.buf]);
                    v.advance();
                }
            }
            let w0 = bus[0];
            let (a, b) = ren.process(x * ctx.mixer[si].tick(), &mut bus);
            pl += a;
            pr += b;
            // channel meter: point stereo + this source's bus contribution
            // (W-channel delta, roughly calibrated to the decode)
            // (sources past NSRC exist only in benchmarks: unmetered)
            if si < NSRC {
                let dw = bus[0] - w0;
                let m2 = 0.5 * (a * a + b * b) + 2.0 * dw * dw;
                let acc = &mut ctx.meter_acc[si];
                acc.0 = acc.0.max(m2);
                acc.1 += m2 as f64;
                acc.2 += 1;
            }
        }
        // Environment audio: rain and ambience are outdoor fields routed
        // through the same geometry-priced inlets (apertures, shell seep,
        // horizon sectors) — world-anchored on the SH bus, no per-room
        // constants and no listener-glued bed.
        {
            let w0 = bus[0];
            ctx.rain.process(&mut bus);
            let dw = bus[0] - w0;
            let m2 = 2.0 * dw * dw;
            let acc = &mut ctx.meter_acc[NSRC + 1];
            acc.0 = acc.0.max(m2);
            acc.1 += m2 as f64;
            acc.2 += 1;
        }
        {
            let w0 = bus[0];
            ctx.ambience.process(&mut bus);
            let dw = bus[0] - w0;
            let m2 = 2.0 * dw * dw;
            let acc = &mut ctx.meter_acc[NSRC];
            acc.0 = acc.0.max(m2);
            acc.1 += m2 as f64;
            acc.2 += 1;
        }
        let (mut l, mut r) = match &mut ctx.out {
            Some(o) => o.process(&bus, pl, pr),
            None => (pl.tanh(), pr.tanh()),
        };
        // near-field ear stage — POST-AGC on purpose: a whisper's whole
        // identity is its absolute closeness; the scene's loudness
        // governor must not ride its level up or down
        for v in &mut ctx.near_voices {
            let src = &ctx.fx_bufs[v.buf];
            if v.pos >= src.len() {
                continue;
            }
            let (a, b) = v.ear.process(src[v.pos]);
            v.pos += 1;
            l += a;
            r += b;
        }
        let mg = ctx.master.tick();
        ctx.out_l[i] = (l * mg).clamp(-1.0, 1.0);
        ctx.out_r[i] = (r * mg).clamp(-1.0, 1.0);
    }
    ctx.voices.retain(|v| v.pos < ctx.fx_bufs[v.buf].len());
    ctx.near_voices.retain(|v| v.pos < ctx.fx_bufs[v.buf].len());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dynamic slot JS can write must reach the sim. The feet ride
    /// the LAST slot (5); a bridge that only forwarded slots 0..5 left
    /// source 13 with an empty param block, so every footstep fx voice
    /// mixed into it rendered silence.
    #[test]
    fn every_dynamic_slot_reaches_the_sim() {
        sim_setup();
        let n = omg_scene::walkthrough::DYN_SLOTS;
        let feet_slot = n - 1;
        let feet_src = NSRC - 1;
        let (lx, ly) = (3.0, 3.0);
        let dy = unsafe { core::slice::from_raw_parts_mut(sim_dyn_ptr(), n * 4) };
        dy[feet_slot * 4..feet_slot * 4 + 4].copy_from_slice(&[lx, ly, 0.2, 1.0]);
        for _ in 0..4 {
            sim_tick(lx, ly, 1.6, 0.0);
        }
        let len = sim_params_len(feet_src as u32) as usize;
        let flat = unsafe { core::slice::from_raw_parts(sim_params_ptr(feet_src as u32), len) };
        let pb = omg_core::params::ParamBlock::read_flat(flat);
        assert!(!pb.taps.is_empty(), "feet source (slot {feet_slot}) got no taps: the bridge dropped it");
    }

    // ------------------------------------------------------- engine side

    /// ENG is one global: every engine test holds this lock for its whole
    /// body (eng_init leaks a few small buffers per call — fine in tests).
    static ENG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn eng_lock() -> std::sync::MutexGuard<'static, ()> {
        ENG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Deterministic xorshift noise in [-1, 1).
    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn fx_load(data: &[f32]) {
        let p = eng_fx_alloc(data.len() as u32);
        unsafe { core::slice::from_raw_parts_mut(p, data.len()) }.copy_from_slice(data);
        eng_fx_commit();
    }

    fn out_lr(n: usize) -> (Vec<f32>, Vec<f32>) {
        unsafe {
            (
                core::slice::from_raw_parts(eng_out_l(), n).to_vec(),
                core::slice::from_raw_parts(eng_out_r(), n).to_vec(),
            )
        }
    }

    fn fnv(xs: &[f32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in xs {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    const SR: f32 = 48_000.0;
    /// Meter channels: NSRC sources + ambience + rain, (peak, rms) each.
    const NMETER: usize = NSRC + 2;

    fn source_with(i: u32, data: &[f32]) {
        let p = eng_source_alloc(i, data.len() as u32);
        unsafe { core::slice::from_raw_parts_mut(p, data.len()) }.copy_from_slice(data);
    }

    /// Route source `src` through ONE unit-ish tap: direct path, 0 s delay
    /// (the delay line clamps to 1 sample), flat band gains `g`, no reverb.
    /// Without an HRIR grid the tap lands on the SH bus, so its W channel
    /// (= the source's channel meter) carries exactly x·fader·g.
    fn route_tap(src: u32, g: f32) {
        let pb = ParamBlock {
            taps: vec![omg_core::params::Tap { key: 1, delay_s: 0.0, dir: [1.0, 0.0, 0.0], gains: [g; 3] }],
            reverb: Default::default(),
            remote: None,
            version: 1,
        };
        let mut flat = Vec::new();
        pb.write_flat(&mut flat);
        unsafe { core::slice::from_raw_parts_mut(eng_param_buf_ptr(), flat.len()) }
            .copy_from_slice(&flat);
        eng_set_params(src, flat.len() as u32);
    }

    fn meters() -> Vec<(f32, f32)> {
        let m = unsafe { core::slice::from_raw_parts(eng_meters_commit(), 2 * NMETER) };
        (0..NMETER).map(|c| (m[2 * c], m[2 * c + 1])).collect()
    }

    /// Render `n` samples in ≤ 512-sample blocks; returns (l, r).
    fn render(n: usize) -> (Vec<f32>, Vec<f32>) {
        let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut left = n;
        while left > 0 {
            let b = left.min(512);
            eng_process(b as u32);
            let (bl, br) = out_lr(b);
            l.extend(bl);
            r.extend(br);
            left -= b;
        }
        (l, r)
    }

    /// Per-sample peak meter of channel `ch` (one-sample blocks).
    fn probe(ch: usize, n: usize) -> Vec<f32> {
        meters();
        (0..n)
            .map(|_| {
                eng_process(1);
                meters()[ch].0
            })
            .collect()
    }

    fn max_step(xs: &[f32]) -> f32 {
        xs.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
    }

    fn whisper_bufs() -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = noise(4800, 7)
            .iter()
            .enumerate()
            .map(|(k, x)| 0.15 * x + 0.1 * (k as f32 * 0.07).sin())
            .collect();
        let b: Vec<f32> = noise(1000, 99).iter().map(|x| 0.2 * x).collect();
        (a, b)
    }

    /// A fixed whisper scenario through the whole engine (no sources, no
    /// output stage ⇒ the ear buffers carry ONLY the near-field stage):
    /// both ears, clamped distances, overlapping voices, the 4-voice cap,
    /// voices ending mid-block and a new one starting after. Returns
    /// interleaved [l, r] for 6000 samples.
    fn render_whisper_scenario() -> Vec<f32> {
        eng_init(SR);
        let (a, b) = whisper_bufs();
        fx_load(&a);
        fx_load(&b);
        eng_whisper_play(0, 0, 0.06, 0.5);
        eng_whisper_play(1, 1, 0.2, 0.8);
        eng_whisper_play(0, 1, 0.01, 0.1); // clamps to 0.02 m
        eng_whisper_play(1, 0, 1.0, 1.0); // clamps to 0.6 m
        eng_whisper_play(0, 0, 0.1, 1.0); // 5th: rejected by the cap
        eng_whisper_play(7, 0, 0.1, 1.0); // unknown kind: ignored
        let mut out = Vec::with_capacity(12_000);
        for blk in 0..44 {
            if blk == 20 {
                eng_whisper_play(1, 1, 0.3, 0.6); // after the short voices end
            }
            eng_process(128);
            let (l, r) = out_lr(128);
            for k in 0..128 {
                out.push(l[k]);
                out.push(r[k]);
            }
        }
        eng_process(0);
        let tail = 6000 - 44 * 128;
        eng_process(tail as u32);
        let (l, r) = out_lr(tail);
        for k in 0..tail {
            out.push(l[k]);
            out.push(r[k]);
        }
        out
    }

    /// VERBATIM copy of the near-field stage as it lived inline in this
    /// file before it moved to `omg_dsp::near` — the reference the
    /// extraction is held to, sample for sample.
    struct LegacyNear {
        buf: usize,
        pos: usize,
        right: bool,
        near_g: f32,
        far_g: f32,
        lp_k: f32,
        lp: f32,
        dl: [f32; 64],
        dpos: usize,
        itd: usize,
        fade: usize,
    }

    struct Legacy {
        bufs: Vec<Vec<f32>>,
        voices: Vec<LegacyNear>,
        out: Vec<f32>,
    }

    impl Legacy {
        fn play(&mut self, kind: u32, right: u32, dist_m: f32, gain: f32) {
            if (kind as usize) >= self.bufs.len() || self.voices.len() >= 4 {
                return;
            }
            let d = dist_m.clamp(0.02, 0.6);
            let near_g = gain * 0.06 / d;
            let far_g = gain * 0.06 / (d + 0.25);
            let sr = SR;
            self.voices.push(LegacyNear {
                buf: kind as usize,
                pos: 0,
                right: right != 0,
                near_g,
                far_g,
                lp_k: (-2.0 * core::f32::consts::PI * 700.0 / sr).exp(),
                lp: 0.0,
                dl: [0.0; 64],
                dpos: 0,
                itd: ((0.00065 * sr) as usize).clamp(1, 63),
                fade: 256,
            });
        }

        fn process(&mut self, n: usize) {
            for _ in 0..n {
                let (mut l, mut r) = (0.0f32.tanh(), 0.0f32.tanh());
                for v in &mut self.voices {
                    let src = &self.bufs[v.buf];
                    if v.pos >= src.len() {
                        continue;
                    }
                    let mut x = src[v.pos];
                    v.pos += 1;
                    if v.fade > 0 {
                        x *= 1.0 - v.fade as f32 / 256.0;
                        v.fade -= 1;
                    }
                    let near = x * v.near_g;
                    v.dl[v.dpos & 63] = x;
                    let far_x = v.dl[(v.dpos + 64 - v.itd) & 63];
                    v.dpos += 1;
                    v.lp = v.lp * v.lp_k + far_x * (1.0 - v.lp_k);
                    let far = v.lp * v.far_g;
                    if v.right {
                        r += near;
                        l += far;
                    } else {
                        l += near;
                        r += far;
                    }
                }
                let mg = 1.0f32;
                self.out.push((l * mg).clamp(-1.0, 1.0));
                self.out.push((r * mg).clamp(-1.0, 1.0));
            }
            let bufs = &self.bufs;
            self.voices.retain(|v| v.pos < bufs[v.buf].len());
        }
    }

    /// The NearVoice extraction is bit-identical to the inline code it
    /// replaced (verified once against a captured dump of the old build,
    /// fnv 0xa478dab20be6f914 on aarch64-darwin; held here against the
    /// verbatim legacy model so it stays platform-independent).
    #[test]
    fn whisper_stage_is_bit_identical_to_the_pre_extraction_code() {
        let _g = eng_lock();
        let got = render_whisper_scenario();
        let (a, b) = whisper_bufs();
        let mut lg = Legacy { bufs: vec![a, b], voices: Vec::new(), out: Vec::new() };
        lg.play(0, 0, 0.06, 0.5);
        lg.play(1, 1, 0.2, 0.8);
        lg.play(0, 1, 0.01, 0.1);
        lg.play(1, 0, 1.0, 1.0);
        lg.play(0, 0, 0.1, 1.0);
        lg.play(7, 0, 0.1, 1.0);
        for blk in 0..44 {
            if blk == 20 {
                lg.play(1, 1, 0.3, 0.6);
            }
            lg.process(128);
        }
        lg.process(6000 - 44 * 128);
        assert_eq!(got.len(), lg.out.len());
        assert!(got.iter().any(|x| x.abs() > 0.05), "scenario must be audible");
        for (k, (g, w)) in got.iter().zip(&lg.out).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "sample {} ear {}: {g} vs {w}", k / 2, k % 2);
        }
        if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            assert_eq!(fnv(&got), 0xa478_dab2_0be6_f914);
        }
    }

    /// (a) FX one-shots: a voice sums into ITS source's signal only, at
    /// unit gain; overlapping voices add; stop removes it at once; a voice
    /// lasts exactly its buffer; the 24-voice cap holds; unknown kinds are
    /// ignored.
    #[test]
    fn fx_voice_level_and_rate_vary_per_play() {
        // eng_fx_play_var: level scales the voice, rate shortens or
        // stretches it — the per-step variation of footsteps
        let _g = eng_lock();
        eng_init(SR);
        source_with(0, &[]);
        route_tap(0, 1.0);
        let amp = 0.01f32;
        fx_load(&vec![amp; 3000]); // kind 0
        render(SR as usize);
        meters();
        eng_fx_play_var(0, 0, 0.5, 1.0);
        render(1024);
        let unit = 2f32.sqrt() * amp;
        let peak = meters()[0].0;
        assert!((peak / (0.5 * unit) - 1.0).abs() < 1e-3, "gain 0.5: peak {peak} want {}", 0.5 * unit);
        eng_fx_stop(0, 0);
        // rate 1.5: 3000 samples play out in ~2000
        eng_fx_play_var(0, 0, 1.0, 1.5);
        render(1990);
        assert_eq!(eng().voices.len(), 1, "still playing at 1990 samples");
        render(20);
        assert!(eng().voices.is_empty(), "a 1.5x voice ends after ~2000 samples");
        // rate 1.0 / gain 1.0 is plain playback
        eng_fx_play_var(0, 0, 1.0, 1.0);
        render(2999);
        assert_eq!(eng().voices.len(), 1);
        render(1);
        assert!(eng().voices.is_empty());
    }

    #[test]
    fn fx_voices_mix_into_their_source_stop_end_and_cap() {
        let _g = eng_lock();
        eng_init(SR);
        source_with(0, &[]);
        source_with(1, &[]);
        route_tap(0, 1.0);
        route_tap(1, 1.0);
        let amp = 0.01f32;
        fx_load(&vec![amp; 4000]); // kind 0
        fx_load(&vec![amp; 300]); // kind 1
        render(SR as usize); // tap gains glide in (τ 50 ms)
        meters();
        render(1024);
        let m = meters();
        assert_eq!((m[0].0, m[1].0), (0.0, 0.0), "silent before play");

        // one voice → source 0 only; W meter = √2·|x|
        eng_fx_play(0, 0);
        render(1024);
        let m = meters();
        let unit = 2f32.sqrt() * amp;
        assert!((m[0].0 / unit - 1.0).abs() < 5e-4, "src0 peak {} want {unit}", m[0].0);
        assert_eq!(m[1].0, 0.0, "voice leaked into source 1");
        // 1-sample tap latency: 1023 of 1024 samples carry the voice
        let want_rms = unit * (1023.0f32 / 1024.0).sqrt();
        assert!((m[0].1 / want_rms - 1.0).abs() < 5e-4, "rms {} want {want_rms}", m[0].1);

        // a second voice on the same source adds
        eng_fx_play(0, 0);
        render(256);
        let _ = meters();
        render(256);
        assert!((meters()[0].0 / (2.0 * unit) - 1.0).abs() < 5e-4, "two voices sum");

        // stop removes both voices of (src 0, kind 0) at once
        eng_fx_stop(0, 0);
        assert!(eng().voices.is_empty());
        render(8); // flush the 1-sample tap latency
        meters();
        render(1024);
        assert_eq!(meters()[0].0, 0.0, "stopped voice still sounding");

        // a voice lasts exactly its buffer, then is dropped
        eng_fx_play(0, 1);
        render(1024);
        assert!(eng().voices.is_empty(), "finished voice not retired");
        let m = meters();
        let played = m[0].1 * m[0].1 * 1024.0 / (unit * unit);
        assert!((played - 300.0).abs() < 0.3, "voice lasted {played} samples, buffer is 300");

        // unknown kind: ignored
        eng_fx_play(0, 9);
        assert!(eng().voices.is_empty());

        // cap: 30 plays → 24 voices → 24× the level, not 30×
        for _ in 0..30 {
            eng_fx_play(0, 0);
        }
        assert_eq!(eng().voices.len(), 24);
        render(64);
        meters();
        render(512);
        let m = meters();
        assert!((m[0].0 / (24.0 * unit) - 1.0).abs() < 5e-4, "capped level {}", m[0].0 / unit);
    }

    /// BUG: a voice played on a source index that has no allocated source
    /// is never advanced (only allocated sources tick their voices), so it
    /// never reaches its buffer end and holds one of the 24 slots forever.
    #[test]
    #[ignore = "BUG: eng_fx_play on an unallocated source leaks a voice slot forever"]
    fn fx_voice_on_unallocated_source_does_not_leak() {
        let _g = eng_lock();
        eng_init(SR);
        source_with(0, &[]);
        fx_load(&[0.1; 100]);
        eng_fx_play(5, 0); // no source 5
        render(1024);
        assert!(eng().voices.is_empty(), "voice on a missing source never retires");
    }

    /// (b) Mixer: source faders and master glide (τ 20 ms, no step), clamp
    /// to their ranges, and a muted, faded source is skipped entirely
    /// while its stream and one-shots keep time.
    #[test]
    fn mixer_faders_ramp_clamp_and_mute_skips() {
        let _g = eng_lock();
        eng_init(SR);
        let dc = 0.1f32;
        source_with(0, &vec![dc; 4800]);
        route_tap(0, 1.0);
        fx_load(&vec![0.05; 300]);
        render(SR as usize);
        // unity fader: W meter = √2·dc (the f32 tap-gain smoother parks
        // ~1e-4 short of its target — inaudible, so normalize by the
        // measured steady level from here on)
        let unit = probe(0, 8)[7];
        assert!((unit / (2f32.sqrt() * dc) - 1.0).abs() < 5e-4, "unity fader {unit}");
        let lvl = |p: &[f32]| p.iter().map(|x| x / unit).collect::<Vec<f32>>();

        // 1.0 → 0.25: one-pole glide, never a step
        eng_set_mixer(0, 0.25);
        let p = lvl(&probe(0, 9600));
        assert!(max_step(&p) < 2e-3, "fader step {}", max_step(&p));
        // p[k] carries the fader of sample k−1 (1-sample tap latency)
        let tau = (0.02 * SR) as usize;
        let at_tau = 0.25 + 0.75 * (-1.0f32).exp();
        assert!((p[tau + 1] - at_tau).abs() < 2e-3, "after 1 τ: {} want {at_tau}", p[tau + 1]);
        assert!((p[9599] - 0.25).abs() < 1e-3, "settled at {}", p[9599]);

        // ranges: 0…256, out-of-range index ignored
        eng_set_mixer(0, 1000.0);
        assert_eq!(eng().mixer[0].target_val(), 256.0);
        eng_set_mixer(0, -1.0);
        assert_eq!(eng().mixer[0].target_val(), 0.0);
        eng_set_mixer(NSRC as u32 + 5, 1.0);

        // mute: the fade-out is rendered (no cut) ...
        eng_set_mixer(0, 0.0);
        let p = lvl(&probe(0, 480));
        assert!(p[1] > 0.2 && max_step(&p) < 2e-3, "mute must fade, not cut");
        // ... and once faded below −80 dB the source is skipped entirely
        render(SR as usize / 2);
        assert!(eng().mixer[0].current() < 1e-4);
        let pos0 = eng().sources[0].pos;
        eng().meter_acc[0] = (0.0, 0.0, 0);
        eng_fx_play(0, 0);
        render(1000);
        assert_eq!(eng().meter_acc[0].2, 0, "muted+faded source was still rendered");
        assert_eq!(meters()[0], (0.0, 0.0));
        // stream and one-shot kept time while skipped
        assert_eq!(eng().sources[0].pos, (pos0 + 1000) % 4800, "loop lost its place");
        assert!(eng().voices.is_empty(), "one-shot did not tick while muted");

        // unmute glides back up
        eng_set_mixer(0, 1.0);
        let p = lvl(&probe(0, 9600));
        assert!(max_step(&p) < 2e-3, "unmute step {}", max_step(&p));
        assert!((p[9599] - 1.0).abs() < 1e-3);
    }

    /// (b) Master: glides (no step) and clamps to 0…4. Driven by a DC
    /// whisper, which lands in the ear buffers at a known level.
    #[test]
    fn master_glides_and_clamps() {
        let _g = eng_lock();
        eng_init(SR);
        fx_load(&vec![1.0; 200_000]);
        eng_whisper_play(0, 0, 0.06, 0.1); // near (left) ear: 0.1
        render(1024);
        let (l, _) = render(16);
        assert!((l[15] - 0.1).abs() < 1e-6, "whisper level {}", l[15]);

        eng_set_master(2.0);
        let (l, _) = render(9600);
        assert!(max_step(&l) < 0.1 / 900.0, "master step {}", max_step(&l));
        let tau = (0.02 * SR) as usize;
        let at_tau = 0.1 * (2.0 - (-1.0f32).exp());
        assert!((l[tau - 1] - at_tau).abs() < 2e-4, "after 1 τ {} want {at_tau}", l[tau - 1]);
        assert!((l[9599] - 0.2).abs() < 1e-5);

        eng_set_master(10.0); // clamps to 4
        let (l, _) = render(19_200);
        // (an f32 one-pole parks ~τ·ulp/2 short of its target: 3e-5 rel.)
        assert!((l[19_199] - 0.4).abs() < 2e-5 * 4.0 * 0.1 * 10.0, "master 10 → {}", l[19_199]);
        eng_set_master(-3.0); // clamps to 0
        let (l, r) = render(19_200);
        assert!(l[19_199].abs() < 1e-6 && r[19_199].abs() < 1e-6);
    }

    /// (d) The final hard clamp: after the master, every output sample is
    /// within ±1, overs are pinned exactly to ±1, and signal below 1 passes
    /// untouched.
    #[test]
    fn final_hard_clamp_after_master() {
        let _g = eng_lock();
        eng_init(SR);
        fx_load(&vec![1.0; 4000]); // kind 0: +DC
        fx_load(&vec![-1.0; 4000]); // kind 1: −DC
        fx_load(&noise(4000, 5)); // kind 2: noise
        // near ear 3.0 → pinned; far ear 3·0.06/0.31 ≈ 0.58 → untouched
        eng_whisper_play(0, 0, 0.06, 3.0);
        render(1024);
        let (l, r) = render(64);
        assert!(l.iter().all(|&x| x == 1.0), "over not pinned to +1");
        let far = 3.0 * 0.06 / 0.31;
        assert!(r.iter().all(|&x| (x - far).abs() < 1e-5), "under-1 ear altered: {}", r[0]);
        render(4000);
        eng_whisper_play(1, 1, 0.06, 3.0);
        render(1024);
        let (_, r) = render(64);
        assert!(r.iter().all(|&x| x == -1.0), "over not pinned to −1");
        render(4000);
        // the clamp sits AFTER the master: 0.3 × 4 = 1.2 → 1
        eng_set_master(4.0);
        render(19_200);
        eng_whisper_play(0, 0, 0.06, 0.3);
        render(1024);
        let (l, r) = render(64);
        assert!(l.iter().all(|&x| x == 1.0), "post-master over not clamped: {}", l[0]);
        assert!(r.iter().all(|&x| (x - 4.0 * 0.3 * 0.06 / 0.31).abs() < 1e-5));
        render(4000);
        // loud noise: nothing escapes ±1, and the clamp is engaged
        eng_whisper_play(2, 0, 0.02, 4.0);
        eng_whisper_play(2, 1, 0.02, 4.0);
        let (l, r) = render(4000);
        assert!(l.iter().chain(&r).all(|x| x.abs() <= 1.0 && x.is_finite()));
        assert!(l.iter().filter(|x| x.abs() == 1.0).count() > 1000);
    }

    /// (c) Level meters: a known sine on a unit tap reads peak √2·A and
    /// RMS A (the W-channel calibration, 0.5·(l²+r²) + 2·dw²), scales with
    /// the fader, and a commit resets the accumulators.
    #[test]
    fn meters_report_peak_and_rms_of_a_known_signal() {
        let _g = eng_lock();
        eng_init(SR);
        let a = 0.2f32;
        // 1 kHz, exactly 48 samples per cycle, 100 cycles per loop
        let sine: Vec<f32> =
            (0..4800).map(|k| a * (core::f32::consts::TAU * k as f32 / 48.0).sin()).collect();
        source_with(0, &sine);
        source_with(1, &[]);
        route_tap(0, 1.0);
        route_tap(1, 1.0);
        render(SR as usize);
        meters();
        render(4800);
        let m = meters();
        let (peak, rms) = m[0];
        assert!((peak / (2f32.sqrt() * a) - 1.0).abs() < 5e-4, "peak {peak}");
        assert!((rms / a - 1.0).abs() < 5e-4, "rms {rms} want {a}");
        assert_eq!(m[1], (0.0, 0.0), "silent source reads silent");
        // a commit resets: nothing rendered since ⇒ zeros
        assert_eq!(meters()[0], (0.0, 0.0));
        // meters are post-fader: −6 dB fader ⇒ half the reading
        eng_set_mixer(0, 0.5);
        render(SR as usize / 2);
        meters();
        render(4800);
        let (peak, rms) = meters()[0];
        assert!((peak / (2f32.sqrt() * a * 0.5) - 1.0).abs() < 1e-3, "faded peak {peak}");
        assert!((rms / (a * 0.5) - 1.0).abs() < 1e-3, "faded rms {rms}");
    }

    /// (e) Whispers ride POST-AGC: with a loud scene driving the ear
    /// adaptation deep into cut, a whisper still adds its exact near-field
    /// level to the ears (not level × AGC), and it does not move the AGC.
    #[test]
    fn whisper_is_post_agc() {
        let _g = eng_lock();
        let spk = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/hrir_dodeca20.bin"))
            .expect("speaker HRIRs");
        let run = |whisper: bool| {
            eng_init(SR);
            let p = eng_hrir_speakers_alloc(spk.len() as u32);
            unsafe { core::slice::from_raw_parts_mut(p, spk.len()) }.copy_from_slice(&spk);
            eng_hrir_speakers_done();
            source_with(0, &noise(48_000, 21));
            route_tap(0, 1.0);
            eng_set_mixer(0, 16.0);
            fx_load(&vec![1.0; 20_000]);
            render(2 * SR as usize);
            if whisper {
                eng_whisper_play(0, 0, 0.06, 0.1);
            }
            let out = render(4800);
            (out, eng_agc_gain())
        };
        let ((bl, br), agc_b) = run(false);
        let ((wl, wr), agc_w) = run(true);
        assert!(agc_b < 0.3, "scene must drive the AGC into cut: {agc_b}");
        assert_eq!(agc_b.to_bits(), agc_w.to_bits(), "whisper moved the AGC");
        let far = 0.1 * 0.06 / 0.31;
        let mut checked = 0;
        for k in 1024..4800 {
            if bl[k].abs() > 0.85 || br[k].abs() > 0.85 {
                continue; // keep clear of the final clamp
            }
            assert!((wl[k] - bl[k] - 0.1).abs() < 1e-5, "near ear Δ {} at {k}", wl[k] - bl[k]);
            assert!((wr[k] - br[k] - far).abs() < 1e-5, "far ear Δ {} at {k}", wr[k] - br[k]);
            checked += 1;
        }
        assert!(checked > 3000, "only {checked} samples clear of the clamp");
    }

    /// (e) At most 4 whispers sound at once; the rest are dropped, and the
    /// ears carry exactly the sum of the 4 admitted voices.
    #[test]
    fn whisper_voice_cap_is_four() {
        let _g = eng_lock();
        eng_init(SR);
        fx_load(&vec![1.0; 10_000]);
        for _ in 0..6 {
            eng_whisper_play(0, 0, 0.06, 0.01);
        }
        assert_eq!(eng().near_voices.len(), omg_dsp::near::MAX_VOICES);
        assert_eq!(omg_dsp::near::MAX_VOICES, 4);
        eng_whisper_play(3, 0, 0.06, 0.01); // unknown kind
        assert_eq!(eng().near_voices.len(), 4);
        render(1024);
        let (l, _) = render(16);
        assert!((l[15] - 0.04).abs() < 1e-6, "4 voices × 0.01 → {}", l[15]);
        render(10_000);
        assert!(eng().near_voices.is_empty(), "finished whispers not retired");
        eng_whisper_play(0, 1, 0.06, 0.01); // slots free again
        assert_eq!(eng().near_voices.len(), 1);
    }

    /// Sources past NSRC (tools/bench_sources.mjs scales the engine to
    /// 128) get their own unit fader, render, and never spill into the
    /// ambience/rain meter channels that follow the NSRC source meters.
    #[test]
    fn mix_budget_bounds_the_total_and_favors_the_loud() {
        // SOURCES_PLAN S1: with a mix-wide budget the number of rendered
        // taps is bounded by the TOTAL, however many sources play, and
        // the loudest paths at the ears win it (mixer fader included).
        let _g = eng_lock();
        eng_init(SR);
        let n = 20u32;
        for i in 0..n {
            source_with(i, &vec![0.1; 4800]);
        }
        // source i: 10 taps of gain (i+1)/100 — the last sources loudest
        let block = |src: u32, version: u64| {
            let g = (src + 1) as f32 / 100.0;
            let pb = ParamBlock {
                taps: (0..10)
                    .map(|k| omg_core::params::Tap {
                        key: 100 + k,
                        delay_s: 0.001 * k as f32,
                        dir: [1.0, 0.0, 0.0],
                        gains: [g; 3],
                    })
                    .collect(),
                reverb: Default::default(),
                remote: None,
                version,
            };
            let mut flat = Vec::new();
            pb.write_flat(&mut flat);
            unsafe { core::slice::from_raw_parts_mut(eng_param_buf_ptr(), flat.len()) }
                .copy_from_slice(&flat);
            eng_set_params(src, flat.len() as u32);
        };
        eng_set_mix_budget(40, 0);
        for v in 1..=2u64 {
            for i in 0..n {
                block(i, v);
            }
            eng_params_commit();
        }
        for i in 0..n {
            block(i, 3);
        }
        render(2400);
        let live: Vec<u32> = (0..n as usize).map(|i| eng().renderers[i].debug_stats().0).collect();
        let total: u32 = live.iter().sum();
        // 40 shared + the guaranteed strongest path of every other source
        assert!(total <= 40 + n, "mix budget exceeded: {total} live taps {live:?}");
        assert!(total >= 40, "budget not used: {total} {live:?}");
        assert_eq!(live[n as usize - 1], 10, "the loudest source keeps all its taps: {live:?}");
        assert!(live[0] <= 1, "the quietest source keeps at most its strongest path: {live:?}");
        // muting the loudest via its fader hands its share to the others
        eng_set_mixer(n - 1, 0.0);
        for v in 4..=5u64 {
            for i in 0..n {
                block(i, v);
            }
            eng_params_commit();
        }
        render(2400);
        let live2: Vec<u32> = (0..n as usize).map(|i| eng().renderers[i].debug_stats().0).collect();
        assert!(live2[n as usize - 2] == 10 && live2[n as usize - 5] == 10,
            "the freed share goes to the next loudest: {live2:?}");
    }

    #[test]
    fn sources_past_nsrc_render_unmetered() {
        let _g = eng_lock();
        eng_init(SR);
        let n = NSRC + 3;
        for i in 0..n as u32 {
            source_with(i, &vec![0.1; 4800]);
        }
        assert_eq!(eng().mixer.len(), n);
        let last = n as u32 - 1;
        route_tap(last, 1.0);
        render(4800);
        // no output stage in tests (bus is dropped): read the renderer —
        // its tap gain only glides up while process() actually runs it
        let (live, _, gain, _, _) = eng().renderers[last as usize].debug_stats();
        assert!(live >= 1 && gain > 0.5, "source {last} past NSRC not rendered: {live} {gain}");
        let m = meters();
        assert_eq!(m[NSRC], (0.0, 0.0), "overflow source metered as ambience");
        assert_eq!(m[NSRC + 1], (0.0, 0.0), "overflow source metered as rain");
    }
}
