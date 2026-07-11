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

const NSRC: usize = 6;
const MAX_FLAT: usize = 4096; // f32s per param buffer (~450 taps headroom)
const STATE_LEN: usize = 64;
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
        params: [
            leak_f32(MAX_FLAT), leak_f32(MAX_FLAT), leak_f32(MAX_FLAT),
            leak_f32(MAX_FLAT), leak_f32(MAX_FLAT), leak_f32(MAX_FLAT),
        ],
        param_lens: [0; NSRC],
        state: leak_f32(STATE_LEN),
        dyn_in: leak_f32(12),
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
pub extern "C" fn sim_tick(lx: f32, ly: f32, yaw: f32) {
    let ctx = sim();
    for slot in 0..3 {
        let o = slot * 4;
        ctx.world.set_dynamic(
            slot,
            ctx.dyn_in[o],
            ctx.dyn_in[o + 1],
            ctx.dyn_in[o + 2],
            ctx.dyn_in[o + 3] > 0.5,
        );
    }
    let (blocks, info) = ctx.world.tick_at(lx, ly, yaw);
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
    // ambient bed control by enclosure: (gain, lowpass fc)
    let (ag, afc) = match ctx.world.rooms[info.room].name {
        "Outside" => (0.085, 18_000.0),
        "Living Room" => (0.018, 900.0),
        "Corridor" => (0.005, 400.0),
        "Great Hall" => (0.010, 600.0),
        "Entrance" => (0.024, 1200.0),
        "Old House" => (0.020, 900.0),
        _ => (0.004, 300.0), // Club: thick concrete
    };
    st[62] = ag;
    st[63] = afc;
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
}

struct EngCtx {
    renderers: Vec<Renderer>,
    sources: Vec<SourceState>,
    out: Option<OutputStage>,
    sample_rate: f32,
    point_budget: usize,
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
    ambient: Vec<f32>,
    ambient_stereo: bool,
    ambient_stage: Option<&'static mut [f32]>,
    ambient_pos: usize,
    ambient_gain: omg_dsp::smooth::Smoothed,
    ambient_lp_coef: omg_dsp::smooth::Smoothed,
    ambient_lp: [f32; 4],
    ambient_enc: [[f32; NCH]; 4],
    rain: omg_dsp::rain::Rain,
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
        grid: None,
        hrir_grid_buf: None,
        hrir_spk_buf: None,
        param_buf: leak_f32(MAX_FLAT),
        out_l: leak_f32(MAX_BLOCK),
        out_r: leak_f32(MAX_BLOCK),
        fx_bufs: Vec::new(),
        fx_stage: None,
        voices: Vec::new(),
        ambient: Vec::new(),
        ambient_stereo: false,
        ambient_stage: None,
        ambient_pos: 0,
        ambient_gain: omg_dsp::smooth::Smoothed::new(0.0, 0.8, sample_rate),
        ambient_lp_coef: omg_dsp::smooth::Smoothed::new(1.0, 0.8, sample_rate),
        ambient_lp: [0.0; 4],
        rain: omg_dsp::rain::Rain::new(sample_rate),
        // world-anchored feed directions (N/E/S/W): the bed lives on the
        // rotating SH bus so it counter-rotates with head turns like the
        // rest of the world, instead of sticking to the ears.
        ambient_enc: [
            omg_dsp::ambi::encode_gains([0.0, 1.0, 0.0]),
            omg_dsp::ambi::encode_gains([1.0, 0.0, 0.0]),
            omg_dsp::ambi::encode_gains([0.0, -1.0, 0.0]),
            omg_dsp::ambi::encode_gains([-1.0, 0.0, 0.0]),
        ],
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

/// Allocate the loop buffer for source `i` (mono samples at engine rate)
/// and create its renderer. Call in source-index order.
#[no_mangle]
pub extern "C" fn eng_source_alloc(i: u32, nsamples: u32) -> *mut f32 {
    let ctx = eng();
    assert_eq!(i as usize, ctx.sources.len(), "sources in order");
    ctx.sources.push(SourceState { data: vec![0.0; nsamples as usize], pos: 0 });
    let mut r = Renderer::with_grid(ctx.sample_rate, ctx.grid.clone());
    r.set_point_budget(ctx.point_budget);
    ctx.renderers.push(r);
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
    let ctx = eng();
    if (kind as usize) < ctx.fx_bufs.len() && ctx.voices.len() < 24 {
        ctx.voices.push(Voice { src: src as usize, buf: kind as usize, pos: 0 });
    }
}

#[no_mangle]
pub extern "C" fn eng_fx_stop(src: u32, kind: u32) {
    let ctx = eng();
    ctx.voices.retain(|v| !(v.src == src as usize && v.buf == kind as usize));
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
    ctx.ambient = buf.to_vec();
    ctx.ambient_stereo = channels == 2;
}

/// Rain intensity 0…1 (ramped inside; rain starts/stops like weather).
#[no_mangle]
pub extern "C" fn eng_set_rain(intensity: f32) {
    eng().rain.set_intensity(intensity);
}

/// Enclosure-dependent bed control: gain + lowpass cutoff (Hz).
#[no_mangle]
pub extern "C" fn eng_set_ambient(gain: f32, fc: f32) {
    let ctx = eng();
    ctx.ambient_gain.set(gain);
    ctx.ambient_lp_coef
        .set(1.0 - (-core::f32::consts::TAU * fc.clamp(100.0, 20000.0) / ctx.sample_rate).exp());
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
}

#[no_mangle]
pub extern "C" fn eng_set_head(yaw: f32) {
    let ctx = eng();
    for r in &mut ctx.renderers {
        r.set_head_yaw(yaw);
    }
    if let Some(o) = &mut ctx.out {
        o.set_head_yaw(yaw);
    }
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
            let mut x = if src.data.is_empty() {
                0.0
            } else {
                let s = src.data[src.pos];
                src.pos = (src.pos + 1) % src.data.len();
                s
            };
            for v in &mut ctx.voices {
                if v.src == si && v.pos < ctx.fx_bufs[v.buf].len() {
                    x += ctx.fx_bufs[v.buf][v.pos];
                    v.pos += 1;
                }
            }
            let (a, b) = ren.process(x, &mut bus);
            pl += a;
            pr += b;
        }
        // rain: world-anchored on the SH bus like the ambience bed;
        // enclosure derived from the room's bed gain (outside = 0.085)
        {
            let ag = ctx.ambient_gain.current();
            let enclosure = (1.0 - ag / 0.085).clamp(0.0, 1.0);
            ctx.rain.process(&mut bus, enclosure, ctx.ambient_lp_coef.current());
        }
        // ambient bed: four decorrelated reads of the loop, encoded at
        // world N/E/S/W onto the SH bus (pre-rotation) — world-anchored,
        // ducked + darkened indoors.
        if !ctx.ambient.is_empty() {
            let frames = if ctx.ambient_stereo { ctx.ambient.len() / 2 } else { ctx.ambient.len() };
            ctx.ambient_pos = (ctx.ambient_pos + 1) % frames;
            let g = ctx.ambient_gain.tick() * 0.55;
            let c = ctx.ambient_lp_coef.tick();
            for d in 0..4 {
                // stereo: N/S take L at offset reads, E/W take R —
                // real channel decorrelation, world-anchored.
                let frame = (ctx.ambient_pos + (d / 2) * frames / 2) % frames;
                let sample = if ctx.ambient_stereo {
                    ctx.ambient[frame * 2 + (d % 2)]
                } else {
                    ctx.ambient[(ctx.ambient_pos + d * frames / 4) % frames]
                };
                ctx.ambient_lp[d] += c * (sample - ctx.ambient_lp[d]);
                let v = ctx.ambient_lp[d] * g;
                for k in 0..NCH {
                    bus[k] += v * ctx.ambient_enc[d][k];
                }
            }
        }
        let (l, r) = match &mut ctx.out {
            Some(o) => o.process(&bus, pl, pr),
            None => (pl.tanh(), pr.tanh()),
        };
        ctx.out_l[i] = l;
        ctx.out_r[i] = r;
    }
    ctx.voices.retain(|v| v.pos < ctx.fx_bufs[v.buf].len());
}
