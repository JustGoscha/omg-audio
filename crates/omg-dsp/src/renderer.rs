//! Per-source renderer: one shared source delay line, a pool of spatialized
//! early taps (direct + image sources), an FDN late field, everything summed
//! on a FOA bus and decoded to stereo.
//!
//! Where the source's past comes from: a live/generated source writes its
//! samples into the delay line (`process`); a source made of recordings
//! keeps a `PlayLog` of what it played when, and every path reads that at
//! its own emission time (`process_log`) — no audio stored, no distance
//! limit.
//!
//! Tap lifecycle: incoming taps carry a stable `key` identifying their
//! propagation path. Same key ⇒ the slot glides (delay motion = Doppler,
//! physically correct). New key ⇒ a free slot fades in from silence at the
//! final delay. Vanished key ⇒ the slot fades out and is recycled. Delay is
//! never slid between two different paths — that would be a pitch chirp.
//!
//! Two ways to drive it, same signal: per sample (`process`, `process_log`
//! — the reference path), or a block at a time (`process_block`,
//! `process_log_block`, what the hosts use): paths in groups rendered side
//! by side, each running its whole block with its state in registers —
//! several times cheaper per path. tests/fx_block_render.rs holds the two
//! together.

use crate::ambi::{encode_gains, rotate_z, HeadRotation, NCH};
use crate::delay::{DelayLine, READ_SLACK};
use crate::fdn::{Fdn, NLINES};
use crate::filter::{OnePoleLp, TapEq};
use crate::hrtf::{HrirGrid, PointConv};
use crate::playlog::PlayLog;
use crate::smooth::{PathDelay, Smoothed};
use omg_core::params::ParamBlock;
use std::sync::Arc;

/// Pool is sized ~2× the largest expected tap count so fading-out ghosts
/// of a previous acoustic state can coexist with the new state.
pub const MAX_TAPS: usize = 448;
/// Delay retarget threshold: same-key delay changes above this are treated
/// as a path identity change (crossfade), below it as motion (glide).
const RETARGET_MS: f32 = 6.0;
/// …or, when updates come slowly, half the time since the last one: a
/// path may glide at up to Mach 0.5 relative speed (a fast flyover at
/// 20 Hz updates moves its delay ~25 ms per tick and must still be
/// Doppler, not a crossfade every tick); a teleport still crossfades.
const RETARGET_FRACTION_OF_TICK: f64 = 0.5;
/// How fast the shared update clock follows the updates' arrivals (per
/// update): slow, so arrival jitter barely moves it, and drift between
/// the host's clock and the sample clock is still followed.
const CLOCK_PULL: f64 = 1.0 / 16.0;
/// Delay line length for live / generated sources (`process`). Paths
/// longer than this clamp to its end; recorded sources use a `PlayLog`
/// and have no such limit.
pub const LIVE_LINE_S: f32 = 2.0;
/// Loudness match of point-rendered direct paths vs. the bus decode.
const POINT_GAIN: f32 = 0.8;
/// Default per-source point-render budget: how many of the strongest taps
/// get their own nearest-HRIR convolution (full-sharpness localization)
/// instead of the order-2 bus. A measured CPU budget, not a quality
/// ceiling: with the SIMD convolution kernel, 24 holds 2.06× realtime at
/// the worst-case scene position on an M1 (tools/bench_web.mjs). The web
/// worklet overrides this adaptively from its own measured load.
const DEFAULT_POINT_TAPS: usize = 24;
/// Default per-source cap on incoming taps kept from a ParamBlock. The
/// weakest image sources are inaudible under the ones we keep; this
/// bounds worst-case render cost on throttled devices. Measured: a club
/// doorway carries ~145 taps/source vs ~20 in the open square, and the
/// bottom half sits far below the masking threshold of the top.
const DEFAULT_TAP_CEILING: usize = 160;
/// Time constant of every per-tap parameter smoother.
const TAP_TAU_S: f32 = 0.05;
/// Closing linear tail of a released slot (see `TapState::release_floor`).
const TAIL_SAMPLES: u32 = 240;
/// Angular spread of the remote-reverb lines around the doorway (±~30°).
const REMOTE_SPREAD_RAD: [f32; NLINES] =
    [-0.55, -0.38, -0.22, -0.08, 0.08, 0.22, 0.38, 0.55];
const LOW_SHELF_HZ: f32 = 250.0;
/// Most samples one `process_block` / `process_log_block` call renders.
pub const BLOCK: usize = 128;
/// A block of the SH bus, channel-major (`bus[k][i]`: channel k, sample
/// i) so per-path accumulation vectorizes over samples.
pub type BusBlock = [[f32; BLOCK]; NCH];
/// Head-pose HRIR re-selection period (samples).
const RESELECT_EVERY: u32 = 128;
const _: () = assert!(BLOCK <= READ_SLACK);
const HIGH_SHELF_HZ: f32 = 2500.0;

struct TapState {
    key: Option<u32>,
    /// Point-rendered: nearest-HRIR convolution instead of the SH bus.
    /// The strongest tap (direct path) per source gets this — full-sharpness
    /// localization where it matters; a slot never changes mode while live.
    point: bool,
    /// Walk-frame arrival direction (pre-head-rotation), for HRIR selection.
    base_dir: [f32; 3],
    delay: PathDelay,     // samples
    g_low: Smoothed,      // low/mid amplitude ratio
    g_high: Smoothed,     // high/mid amplitude ratio
    g_sh: [Smoothed; NCH], // anchor gain × SH encoding, per channel
    lp_coef: Smoothed,     // through-wall lowpass (≈1 = bypass)
    lp: OnePoleLp,
    eq: TapEq,
    conv: PointConv,
    /// A releasing slot fades on its normal one-pole (complementary to
    /// the path fading in, so a crossfade keeps its level) until it is
    /// 56 dB under where the fade began, then a short linear tail
    /// (`TAIL_SAMPLES`) takes it to exactly zero and frees it: ~0.33 s
    /// in all instead of the ~0.5 s an absolute 1e-6 floor took. With
    /// paths coming and going every tick, those ghosts ran at ~3× the
    /// tap budget.
    release_floor: f32,
    /// Samples left in the closing linear tail (0 = not in it).
    tail: u32,
}

impl TapState {
    fn free(&self) -> bool {
        self.key.is_none()
            && self.g_sh[0].target_val() == 0.0
            && self.g_sh[0].current().abs() < 1e-6
            && self.tail == 0
    }

    /// A releasing slot under its floor closes with a linear tail; at the
    /// tail's end it is freed. For the next `m` samples: how many it still
    /// renders (`m` = all; fewer = freed at that sample) and whether
    /// `fade[..end]` holds tail gains — sample-exact with the per-sample
    /// path (`pw` as in `Renderer::pw`).
    fn lifecycle(&mut self, pw: &[f32; BLOCK], m: usize, fade: &mut [f32; BLOCK]) -> (usize, bool) {
        let mut end = m;
        let mut faded = false;
        if self.key.is_none() && self.g_sh[0].target_val() == 0.0 {
            let g0 = self.g_sh[0].current();
            let floor = self.release_floor.max(1e-6);
            // |g| only falls: no tail in this stretch unless it is
            // already running or the gain reaches the floor by its end
            if self.tail > 0 || (g0 * pw[m - 1]).abs() < floor || g0.abs() < floor {
                let mut tail = self.tail;
                for i in 0..m {
                    let cur = if i == 0 { g0 } else { g0 * pw[i - 1] };
                    if tail == 0 && cur.abs() < floor {
                        tail = TAIL_SAMPLES;
                    }
                    if tail > 0 {
                        tail -= 1;
                        if tail == 0 {
                            end = i;
                            break;
                        }
                        fade[i] = tail as f32 / TAIL_SAMPLES as f32;
                        faded = true;
                    } else {
                        fade[i] = 1.0;
                    }
                }
                self.tail = tail;
            }
        }
        if self.free() {
            end = 0;
        }
        (end, faded)
    }

    fn release(&mut self) {
        self.key = None;
        self.release_floor = 0.0015 * self.g_sh[0].current().abs();
        self.tail = 0;
        for g in &mut self.g_sh {
            g.set(0.0);
        }
    }
}

struct TapTargets {
    delay: f64,
    g_low: f32,
    g_high: f32,
    g_sh: [f32; NCH],
    /// One-pole lowpass coefficient; ~1.0 = bypass. Transmission-shaped
    /// taps (lows ≫ mids, i.e. through-wall) get a true continuous LP
    /// slope — band-relative shelves alone leave the mid band internally
    /// flat, which made walls sound like "quieter", not "muffled".
    lp_coef: f32,
}

fn targets_of(t: &omg_core::params::Tap, sample_rate: f32) -> TapTargets {
    let g_mid = t.gains[1];
    let g_low_raw = if g_mid > 1e-9 { t.gains[0] / g_mid } else { 1.0 };
    let enc = encode_gains(t.dir);
    let safe_ratio = |x: f32| if g_mid > 1e-9 { (x / g_mid).clamp(0.0, 4.0) } else { 1.0 };

    if g_low_raw > 2.0 && t.gains[0] > 1e-9 {
        // Transmission character: anchor at the LOW band and fit a one-pole
        // lowpass through the band gains: |H(800)| = 1/r → fc = 800/√(r²−1).
        let fc = (800.0 / (g_low_raw * g_low_raw - 1.0).sqrt()).clamp(60.0, 2000.0);
        TapTargets {
            delay: t.delay_s as f64 * sample_rate as f64,
            g_low: 1.0,
            g_high: 1.0,
            g_sh: core::array::from_fn(|k| t.gains[0] * enc[k]),
            lp_coef: OnePoleLp::coef(fc, sample_rate),
        }
    } else {
        TapTargets {
            delay: t.delay_s as f64 * sample_rate as f64,
            g_low: safe_ratio(t.gains[0]),
            g_high: safe_ratio(t.gains[2]),
            g_sh: core::array::from_fn(|k| g_mid * enc[k]),
            lp_coef: 1.0, // bypass
        }
    }
}

/// Where a block's paths read the source.
enum Src<'a> {
    /// The live line, with this sub-block already written: `last` is the
    /// block index of its newest sample.
    Line { line: &'a DelayLine, last: usize },
    /// A recorded source; block sample j is engine time `now + j`, heard
    /// at fader `gain[j]`.
    Log { log: &'a PlayLog, now: f64, gain: &'a [f32] },
}

impl Src<'_> {
    /// `out[i]`: the source as heard at block sample `j0 + i` over a path
    /// of `max(d[i][lane], 1)` samples.
    #[inline]
    fn block<const L: usize>(&self, j0: usize, d: &[[f64; L]], lane: usize, out: &mut [f32]) {
        match *self {
            Src::Line { line, last } => {
                for (i, (o, x)) in out.iter_mut().zip(d).enumerate() {
                    *o = line.read_back(x[lane].max(1.0) as f32, last - (j0 + i));
                }
            }
            Src::Log { log, now, gain } => {
                let n = out.len();
                let mut t = [0.0f64; BLOCK];
                let base = now + j0 as f64;
                for (i, (t, x)) in t[..n].iter_mut().zip(d).enumerate() {
                    let x = x[lane];
                    // max(x, 1) without f64::max's NaN select chain (wasm)
                    *t = base + i as f64 - if x < 1.0 { 1.0 } else { x };
                }
                log.read_block(&t[..n], &gain[j0..j0 + n], out);
            }
        }
    }
}

/// What every group of one block shares.
struct GroupCx<'a, 'b> {
    pw: &'a [f32; BLOCK],
    grid: Option<&'a HrirGrid>,
    c_low: f32,
    c_high: f32,
    src: &'a Src<'b>,
    off: usize,
    m: usize,
    bus: &'a mut BusBlock,
    pl: &'a mut [f32],
    pr: &'a mut [f32],
}

/// The tap slots `group` (≤ L of them) over one block, side by side:
/// their per-sample recursions — delay motion (f64; its slope is the
/// Doppler pitch, so it is stepped per sample) and lowpass + shelves —
/// run interleaved in `L` lanes, so the slots' dependency chains overlap
/// and vectorize instead of each path waiting on its own. The rest runs
/// per slot over the block: the source read for the whole window, and the
/// mono result spread onto the nine bus channels with the gain smoothers
/// in closed form (`target + (cur − target)·pw[i]`: two multiply-adds per
/// channel and sample, vectorized over samples). Spare lanes repeat
/// lane 0 and are thrown away. True when a slot was freed.
fn tap_group<const L: usize>(taps: &mut [TapState], group: &[u16], cx: &mut GroupCx) -> bool {
    let (pw, m, off) = (cx.pw, cx.m, cx.off);
    // block scratch on the stack (measured: faster than keeping it in
    // the renderer, zeroing included)
    let mut dd = [[0.0f64; L]; BLOCK]; // delays, then emission times
    let mut ss = [[0.0f32; BLOCK]; L]; // source reads
    let mut shsh = [[0.0f32; L]; BLOCK]; // filtered
    let mut fade = [[1.0f32; BLOCK]; L]; // release-tail gains
    let mut mono = [0.0f32; BLOCK];
    let mut mp = [0.0f32; BLOCK];
    let c = group.len();
    let idx: [usize; L] = core::array::from_fn(|l| group[if l < c { l } else { 0 }] as usize);
    let mut end = [0usize; L];
    let mut faded = [false; L];
    let mut loud = [false; L];
    for l in 0..c {
        let tap = &mut taps[idx[l]];
        (end[l], faded[l]) = tap.lifecycle(pw, m, &mut fade[l]);
        if end[l] > 0 {
            // skip silent stretches (smoothers still move); gains are
            // monotone, so the ends bound the stretch
            let (t0, c0) = (tap.g_sh[0].target_val(), tap.g_sh[0].current());
            let (a, b) = (t0 + (c0 - t0) * pw[0], t0 + (c0 - t0) * pw[end[l] - 1]);
            loud[l] = a.abs().max(b.abs()) >= 1e-7;
        }
    }

    // delay motion, sample-accurate, all lanes at once (a freed slot's
    // delay is snapped when it is reused)
    let d = &mut dd[..m];
    let mut pd: [PathDelay; L] = core::array::from_fn(|l| taps[idx[l]].delay);
    PathDelay::tick_lanes(&mut pd, d);
    for l in 0..c {
        taps[idx[l]].delay = pd[l];
    }

    // the source at each path's emission times
    for l in 0..c {
        if loud[l] {
            cx.src.block(off, &d[..end[l]], l, &mut ss[l][..end[l]]);
        }
    }

    // through-wall lowpass + band shelves, lanes interleaved
    let tg = |l: usize, f: fn(&TapState) -> &Smoothed| {
        let g = f(&taps[idx[l]]);
        (g.target_val(), g.current() - g.target_val())
    };
    let lpc: [(f32, f32); L] = core::array::from_fn(|l| tg(l, |t| &t.lp_coef));
    let glo: [(f32, f32); L] = core::array::from_fn(|l| tg(l, |t| &t.g_low));
    let ghi: [(f32, f32); L] = core::array::from_fn(|l| tg(l, |t| &t.g_high));
    let mut lp: [OnePoleLp; L] = core::array::from_fn(|l| taps[idx[l]].lp);
    let mut eq: [TapEq; L] = core::array::from_fn(|l| taps[idx[l]].eq);
    let shaped = &mut shsh[..m];
    let s = &ss;
    let (c_low, c_high) = (cx.c_low, cx.c_high);
    for (i, sh) in shaped.iter_mut().enumerate() {
        let p = pw[i];
        for l in 0..L {
            let x = lp[l].tick(s[l][i], lpc[l].0 + lpc[l].1 * p);
            sh[l] = eq[l].tick(x, glo[l].0 + glo[l].1 * p, ghi[l].0 + ghi[l].1 * p, c_low, c_high);
        }
    }

    let mut freed = false;
    for l in 0..c {
        let e = end[l];
        let tap = &mut taps[idx[l]];
        if e > 0 {
            let pe = pw[e - 1];
            if loud[l] {
                tap.lp = lp[l];
                tap.eq = eq[l];
                tap.lp_coef.advance(pe);
                let mono = &mut mono[..e];
                for (v, sh) in mono.iter_mut().zip(shaped.iter()) {
                    *v = sh[l];
                }
                if faded[l] {
                    for (v, f) in mono.iter_mut().zip(&fade[l][..e]) {
                        *v *= f;
                    }
                }
                match (cx.grid, tap.point) {
                    (Some(grid), true) => {
                        // g[0] is the broadband mid gain (W of SN3D SH = 1).
                        let t0 = tap.g_sh[0].target_val();
                        let d0 = tap.g_sh[0].current() - t0;
                        for (i, v) in mono.iter_mut().enumerate() {
                            *v = *v * (t0 + d0 * pw[i]) * POINT_GAIN;
                        }
                        tap.conv.process_block(grid, mono, &mut cx.pl[off..off + e], &mut cx.pr[off..off + e]);
                    }
                    _ => {
                        // bus[k] += mono · (Tₖ + Dₖ·pw) = Tₖ·mono + Dₖ·(mono·pw)
                        let mp = &mut mp[..e];
                        for ((q, v), p) in mp.iter_mut().zip(mono.iter()).zip(pw) {
                            *q = v * p;
                        }
                        for (k, g) in tap.g_sh.iter().enumerate() {
                            let tk = g.target_val();
                            let dk = g.current() - tk;
                            let row = &mut cx.bus[k][off..off + e];
                            for ((r, v), q) in row.iter_mut().zip(mono.iter()).zip(mp.iter()) {
                                *r += tk * v + dk * q;
                            }
                        }
                    }
                }
            }
            tap.g_low.advance(pe);
            tap.g_high.advance(pe);
            for g in &mut tap.g_sh {
                g.advance(pe);
            }
        }
        if e < m {
            // freed: leaves the active list after the block
            for g in &mut tap.g_sh {
                g.snap(0.0);
            }
            freed = true;
        }
    }
    freed
}

/// One FDN step of `late`: wake on input, run while awake, scale the lines
/// by `pre·lvl`, spread them onto `foa` along `dirs`, and fall asleep after
/// `Fdn::idle_after` silent samples (clearing it when its level is gone, so
/// what it held cannot resume as a ghost). True when it ran.
#[allow(clippy::too_many_arguments)]
#[inline]
fn fdn_step(
    fdn: &mut Fdn,
    idle: &mut u32,
    input: f32,
    pre: f32,
    lvl: f32,
    gate: f32,
    dirs: &[[f32; NCH]; NLINES],
    foa: &mut [f32; NCH],
) -> bool {
    let idle_after = fdn.idle_after();
    if *idle >= idle_after {
        // asleep (or waking now): nothing in the lines, their decay may jump
        fdn.settle();
    }
    if input.abs() > 1e-7 {
        *idle = 0;
    }
    if *idle >= idle_after {
        return false;
    }
    let mut lines = [0.0f32; NLINES];
    fdn.process(input, &mut lines);
    let mut e = 0.0f32;
    for i in 0..NLINES {
        let g = lines[i] * pre * lvl;
        e += g.abs();
        for k in 0..NCH {
            foa[k] += g * dirs[i][k];
        }
    }
    if e < 1e-6 && input.abs() <= 1e-7 {
        *idle += 1;
        if *idle == idle_after && gate == 0.0 {
            fdn.clear();
        }
    } else {
        *idle = 0;
    }
    true
}

/// An FDN's per-sample feed over a block (see `fdn_step`).
struct FdnFeed<'a> {
    input: &'a [f32],
    pre: f32,
    lvl: &'a [f32],
    gate: &'a [f32],
}

/// `fdn_step` over block samples `off..off + f.input.len()`, onto `bus`.
/// Awake and unable to fall asleep within the block (fewer samples than
/// it still needs to idle): the FDN runs a block at a time and its lines
/// spread onto the bus vectorized over samples. Asleep with no input: it
/// stays asleep. Otherwise per sample, exactly as `late`.
fn fdn_block(fdn: &mut Fdn, idle: &mut u32, f: &FdnFeed, dirs: &[[f32; NCH]; NLINES], bus: &mut BusBlock, off: usize) {
    let m = f.input.len();
    let wakes = f.input.iter().any(|v| v.abs() > 1e-7);
    let idle_after = fdn.idle_after();
    if *idle >= idle_after {
        // asleep (or waking now): nothing in the lines, their decay may jump
        fdn.settle();
        if !wakes {
            return;
        }
    }
    if (*idle as usize).saturating_add(m) < idle_after as usize && m <= fdn.max_block() {
        let mut g = [[0.0f32; BLOCK]; NLINES];
        fdn.process_block(f.input, &mut g);
        let mut e = [0.0f32; BLOCK];
        for g in &mut g {
            for ((g, l), e) in g[..m].iter_mut().zip(f.lvl).zip(&mut e[..m]) {
                *g = *g * f.pre * l;
                *e += g.abs();
            }
        }
        for (k, row) in bus.iter_mut().enumerate() {
            let row = &mut row[off..off + m];
            for (g, d) in g.iter().zip(dirs) {
                let d = d[k];
                for (r, g) in row.iter_mut().zip(&g[..m]) {
                    *r += g * d;
                }
            }
        }
        for (e, x) in e[..m].iter().zip(f.input) {
            if *e < 1e-6 && x.abs() <= 1e-7 {
                *idle += 1;
            } else {
                *idle = 0;
            }
        }
        return;
    }
    for j in 0..m {
        let mut foa: [f32; NCH] = core::array::from_fn(|k| bus[k][off + j]);
        if fdn_step(fdn, idle, f.input[j], f.pre, f.lvl[j], f.gate[j], dirs, &mut foa) {
            for k in 0..NCH {
                bus[k][off + j] = foa[k];
            }
        }
    }
}

/// Advance each smoother `n` samples at once (exact one-pole jump).
fn advance_n(s: &mut [Smoothed], n: usize) {
    if n == 0 {
        return;
    }
    for g in s {
        if g.current() != g.target_val() {
            g.advance(g.coef().powi(n as i32));
        }
    }
}

pub struct Renderer {
    grid: Option<Arc<HrirGrid>>,
    src_line: DelayLine,
    taps: Vec<TapState>,
    /// Indices of slots that are live or still fading out. The audio loop
    /// walks only these: scanning all MAX_TAPS slots per sample cost
    /// ~4% of a core per source even with every slot idle
    /// (tools/bench_sources.mjs).
    active: Vec<u16>,
    /// First arrival (smoothed least tap delay, samples): the late field
    /// and the remote wet are fed the source as heard THEN, so a far
    /// source's reverb never starts before its direct sound.
    onset: PathDelay,
    /// Engine v4: the dense (scattered) response, convolved.
    dense: crate::dense::DenseConv,
    /// The engine time the convolver's input continues at.
    dense_at: f64,
    /// Turn the dense output about z by this (a host whose bus is in the
    /// listener's yaw frame while its kernels are in the world's: the
    /// web), smoothed. 0: the bus and the kernels share a frame.
    dense_yaw: Smoothed,
    /// Samples since the last applied ParamBlock (retarget window).
    since_params: u32,
    /// Average samples between fresh ParamBlocks (simulation updates):
    /// the interval path delays plan their constant-rate motion over.
    update_interval: f64,
    /// The shared update clock every path delay is timed on: how many
    /// samples after its arrival the current update's window starts (one
    /// motion after the previous window, pulled gently toward arrivals).
    window_lead: f64,
    /// Version of the last FRESH block (re-applied blocks — budget
    /// changes — must not restart the delay motion).
    last_version: u64,
    updates_seen: u32,
    fdn: Fdn,
    fdn_level: Smoothed,
    send_low: Smoothed,
    send_high: Smoothed,
    send_eq: TapEq,
    // Coupled-room wet path: the source room's reverb heard through the
    // portal — its own FDN, output encoded at the doorway direction.
    remote_fdn: Fdn,
    remote_level: Smoothed,
    remote_low: Smoothed,
    remote_high: Smoothed,
    remote_eq: TapEq,
    remote_lp: OnePoleLp,
    remote_lp_coef: Smoothed,
    remote_sh: [Smoothed; NCH],
    /// Doorway aperture: the remote FDN's lines are spread over an angular
    /// arc around the door direction instead of collapsing to mono at a
    /// point — a mono-summed FDN combs audibly ("metallic").
    remote_spread: [[f32; NCH]; NLINES],
    fdn_dirs: [[f32; NCH]; NLINES],
    c_low: f32,
    c_high: f32,
    /// `pw[i]` = coefⁱ⁺¹ of the per-tap parameter smoothers (all share one
    /// time constant): block rendering evaluates them in closed form,
    /// `target + (cur − target)·pw[i]` — the same one-pole curve, without
    /// a per-sample dependency chain.
    pw: [f32; BLOCK],
    sample_rate: f32,
    retarget_thresh: f32,
    /// Per-source point-render budget (see `DEFAULT_POINT_TAPS`).
    point_budget: usize,
    /// Per-source cap on incoming taps kept per ParamBlock (see
    /// `DEFAULT_TAP_CEILING`). The load governor lowers it on weak
    /// devices: the weakest image sources are inaudible under the ones
    /// kept, and evicted taps release with the normal slot fade.
    tap_ceiling: usize,
    /// Fast head rotation [yaw, pitch, roll] (device orientation, mouse
    /// look, face tracking), applied to point-tap HRIR selection here and
    /// to the bus at the decode stage.
    head: [Smoothed; 3],
    /// Samples of joint input+tail silence per FDN; at `Fdn::idle_after` the
    /// FDN sleeps (see process). Starts asleep.
    fdn_idle: u32,
    remote_idle: u32,
    reselect_countdown: u32,
    version_seen: u64,
    first_params: bool,
    /// The next update's new paths start at full level instead of fading
    /// in (`snap_next`).
    snap_born: bool,
    /// Every glide is at its target (see `settle`); cleared by new params.
    settled: bool,
    /// Diagnostic mutes (set from env in the demo app).
    pub mute_taps: bool,
    pub mute_own_fdn: bool,
    pub mute_remote: bool,
}

impl Renderer {
    pub fn new(sample_rate: f32) -> Self {
        Self::with_grid(sample_rate, None)
    }

    pub fn with_grid(sample_rate: f32, grid: Option<Arc<HrirGrid>>) -> Self {
        // Late-field arrival directions: spread over the sphere (cube corners).
        let corner_dirs: [[f32; 3]; NLINES] = core::array::from_fn(|i| {
            let s = 1.0 / (1.0f32 + 1.0 + 0.25).sqrt();
            [
                if i & 1 == 0 { s } else { -s },
                if i & 2 == 0 { s } else { -s },
                if i & 4 == 0 { 0.5 * s } else { -0.5 * s },
            ]
        });
        let conv_taps = grid.as_ref().map_or(1, |g| g.taps);
        let tap_coef = Smoothed::new(0.0, TAP_TAU_S, sample_rate).coef() as f64;
        Self {
            src_line: DelayLine::new((sample_rate * LIVE_LINE_S) as usize + READ_SLACK),
            active: Vec::with_capacity(MAX_TAPS),
            taps: (0..MAX_TAPS)
                .map(|_| TapState {
                    key: None,
                    point: false,
                    base_dir: [1.0, 0.0, 0.0],
                    delay: PathDelay::new(480.0),
                    g_low: Smoothed::new(1.0, TAP_TAU_S, sample_rate),
                    g_high: Smoothed::new(1.0, TAP_TAU_S, sample_rate),
                    g_sh: [Smoothed::new(0.0, TAP_TAU_S, sample_rate); NCH],
                    lp_coef: Smoothed::new(1.0, TAP_TAU_S, sample_rate),
                    lp: OnePoleLp::default(),
                    eq: TapEq::default(),
                    conv: PointConv::new(conv_taps, sample_rate),
                    release_floor: 0.0,
                    tail: 0,
                })
                .collect(),
            grid,
            onset: PathDelay::new(1.0),
            dense: crate::dense::DenseConv::new(),
            dense_at: f64::NAN,
            dense_yaw: Smoothed::new(0.0, 0.03, sample_rate),
            since_params: 0,
            update_interval: 0.05 * sample_rate as f64,
            window_lead: 0.0,
            last_version: u64::MAX,
            updates_seen: 0,
            fdn: Fdn::new(sample_rate),
            fdn_level: Smoothed::new(0.0, 0.2, sample_rate),
            send_low: Smoothed::new(1.0, 0.2, sample_rate),
            send_high: Smoothed::new(1.0, 0.2, sample_rate),
            send_eq: TapEq::default(),
            remote_fdn: Fdn::new(sample_rate),
            remote_level: Smoothed::new(0.0, 0.2, sample_rate),
            remote_low: Smoothed::new(1.0, 0.2, sample_rate),
            remote_high: Smoothed::new(1.0, 0.2, sample_rate),
            remote_eq: TapEq::default(),
            remote_lp: OnePoleLp::default(),
            remote_lp_coef: Smoothed::new(1.0, 0.1, sample_rate),
            remote_sh: [Smoothed::new(0.0, 0.1, sample_rate); NCH],
            remote_spread: [[0.0; NCH]; NLINES],
            fdn_dirs: core::array::from_fn(|i| encode_gains(corner_dirs[i])),
            c_low: OnePoleLp::coef(LOW_SHELF_HZ, sample_rate),
            c_high: OnePoleLp::coef(HIGH_SHELF_HZ, sample_rate),
            pw: core::array::from_fn(|i| tap_coef.powi(i as i32 + 1) as f32),
            sample_rate,
            retarget_thresh: RETARGET_MS * 1e-3 * sample_rate,
            point_budget: DEFAULT_POINT_TAPS,
            tap_ceiling: DEFAULT_TAP_CEILING,
            head: core::array::from_fn(|_| Smoothed::new(0.0, 0.03, sample_rate)),
            fdn_idle: u32::MAX,
            remote_idle: u32::MAX,
            reselect_countdown: 0,
            version_seen: 0,
            first_params: true,
            snap_born: false,
            settled: false,
            mute_taps: false,
            mute_own_fdn: false,
            mute_remote: false,
        }
    }

    /// Drop the live delay line: this renderer will only ever be driven
    /// by `process_log` (its source is recordings, read from a PlayLog).
    pub fn without_line(mut self) -> Self {
        self.src_line = DelayLine::empty();
        self
    }

    /// How far back (samples) any path may still read, now or once its
    /// delay has glided to target. A PlayLog may forget what lies before
    /// `now − horizon()`.
    /// Engine v4: the source's dense response as a convolution kernel
    /// (`dense::build_kernel`, built off the audio thread); None stops it.
    pub fn set_dense_kernel(&mut self, k: Option<std::sync::Arc<crate::dense::DenseKernel>>) {
        self.dense.set_kernel(k);
    }

    /// `set_dense_kernel` for a response that changed beyond its noise:
    /// what the old one still had to play is dropped (`DenseConv::cut`).
    pub fn cut_dense_kernel(&mut self, k: Option<std::sync::Arc<crate::dense::DenseKernel>>) {
        self.dense.cut(k);
    }

    /// The listener's yaw for the dense output (see `dense_yaw`).
    pub fn set_dense_yaw(&mut self, yaw: f32) {
        // the shorter way round
        let cur = self.dense_yaw.target_val();
        let mut d = (yaw - cur) % core::f32::consts::TAU;
        if d > core::f32::consts::PI {
            d -= core::f32::consts::TAU;
        } else if d < -core::f32::consts::PI {
            d += core::f32::consts::TAU;
        }
        self.dense_yaw.set(cur + d);
    }

    /// How long the source keeps sounding after its last sample, beyond
    /// the paths' delays: the dense kernel's length (samples).
    pub fn dense_tail(&self) -> f64 {
        self.dense.kernel_len() as f64
    }

    pub fn horizon(&self) -> f64 {
        let mut h = self.onset.current().max(self.onset.target_val());
        for &i in &self.active {
            let d = &self.taps[i as usize].delay;
            h = h.max(d.current()).max(d.target_val());
        }
        h
    }

    pub fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.head[0].set(yaw);
        self.head[1].set(pitch);
        self.head[2].set(roll);
    }

    /// Live render occupancy for the field-debug panel: (live tap slots,
    /// point-rendered among them, summed mid-channel gain, own-FDN send,
    /// remote-FDN send). "Live" includes releasing slots still fading out
    /// — exactly what still costs CPU and still sounds.
    pub fn debug_stats(&self) -> (u32, u32, f32, f32, f32) {
        let mut live = 0u32;
        let mut points = 0u32;
        let mut gain = 0.0f32;
        for t in &self.taps {
            if t.key.is_some() || t.g_sh[0].current().abs() > 1e-6 {
                live += 1;
                if t.point {
                    points += 1;
                }
                gain += t.g_sh[0].current().abs();
            }
        }
        (
            live,
            points,
            gain,
            self.fdn_level.current(),
            self.remote_level.current(),
        )
    }

    /// Set how many of the strongest taps are point-rendered. Takes effect
    /// on the next ParamBlock; taps whose mode changes re-slot with a
    /// crossfade (a live slot never switches mode in place).
    pub fn set_point_budget(&mut self, n: usize) {
        self.point_budget = n;
    }

    /// Set the per-source cap on incoming taps. Takes effect on the next
    /// ParamBlock; taps that fall out release with the normal slot fade,
    /// so lowering it mid-play is click-free.
    pub fn set_tap_ceiling(&mut self, n: usize) {
        self.tap_ceiling = n.max(1);
    }

    /// Jump every glide to where it is headed — for a source that renders
    /// nothing now (silent): there is no signal a jump could click, and a
    /// path still fading in when the next sound starts would swallow its
    /// onset (a clap is over in a few ms; the fades take tens). Released
    /// paths are freed. Cheap when already settled.
    pub fn settle(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        for slot in &mut self.taps {
            slot.delay.snap(slot.delay.target_val());
            slot.g_low.snap(slot.g_low.target_val());
            slot.g_high.snap(slot.g_high.target_val());
            slot.lp_coef.snap(slot.lp_coef.target_val());
            for g in &mut slot.g_sh {
                g.snap(g.target_val());
            }
            slot.tail = 0;
        }
        let taps = &self.taps;
        self.active.retain(|&i| !taps[i as usize].free());
        self.onset.snap(self.onset.target_val());
        for g in [
            &mut self.dense_yaw,
            &mut self.fdn_level,
            &mut self.send_low,
            &mut self.send_high,
            &mut self.remote_level,
            &mut self.remote_low,
            &mut self.remote_high,
            &mut self.remote_lp_coef,
        ] {
            g.snap(g.target_val());
        }
        for g in &mut self.remote_sh {
            g.snap(g.target_val());
        }
    }

    /// Forget which ParamBlock version was applied, so the next
    /// `set_params` re-applies even the same block — used when a budget
    /// share changes and must take effect without waiting a tick.
    pub fn invalidate_params(&mut self) {
        self.version_seen = u64::MAX;
    }

    /// Called from the audio thread whenever a new ParamBlock is available.
    pub fn set_params(&mut self, pb: &ParamBlock) {
        self.set_params_after(pb, None);
    }

    /// `set_params` for a block traced `motion` samples of simulated time
    /// after the previous one. Path velocities are the delay changes over
    /// that time instead of over the average update interval — exact
    /// Doppler when the simulation samples the motion irregularly (game
    /// physics steps, simulation ticks that run long or get superseded).
    pub fn set_params_after(&mut self, pb: &ParamBlock, motion: Option<f64>) {
        if pb.version == self.version_seen {
            return;
        }
        self.version_seen = pb.version;
        self.settled = false;
        // judged against the TYPICAL interval too: an update arriving
        // early must not turn fast motion into a crossfade
        let thresh = (self.retarget_thresh as f64).max(
            RETARGET_FRACTION_OF_TICK
                * (self.since_params as f64).max(self.update_interval).max(motion.unwrap_or(0.0)),
        );
        let fresh = pb.version != self.last_version;
        let arrived_after = self.since_params as f64;
        if fresh {
            if !self.first_params {
                let since = (self.since_params as f64).clamp(64.0, 0.5 * self.sample_rate as f64);
                // a long-run mean (running mean at first, then a slow
                // EMA): velocities are divided by it, so its jitter
                // would be pitch jitter
                self.updates_seen += 1;
                let a = (1.0 / self.updates_seen as f64).max(0.03);
                self.update_interval += (since - self.update_interval) * a;
            }
            self.last_version = pb.version;
            self.since_params = 0;
        }
        let interval = self.update_interval;
        // a host time far off the cadence (a pause, a stalled clock) says
        // nothing about this update's motion
        let motion = motion.filter(|m| *m >= 0.25 * interval && *m <= 4.0 * interval).unwrap_or(interval);
        // where this update's window starts on the shared clock (as the
        // samples already into it): every path alive now and every path
        // born later moves on the same timeline, whenever updates arrive
        let age = if fresh && !self.first_params {
            let lead = self.window_lead + motion - arrived_after;
            // a stall or a pause: start over at the arrivals
            self.window_lead = if lead.abs() > 2.0 * interval { 0.0 } else { lead * (1.0 - CLOCK_PULL) };
            -self.window_lead
        } else {
            0.0
        };

        // CPU ceiling: keep only the strongest incoming taps (see
        // `DEFAULT_TAP_CEILING`; the load governor lowers it live).
        let mut incoming: Vec<&omg_core::params::Tap> = pb.taps.iter().collect();
        if incoming.len() > self.tap_ceiling {
            incoming.sort_by(|a, b| b.gains[1].total_cmp(&a.gains[1]));
            incoming.truncate(self.tap_ceiling);
        }

        // Fade out slots whose path no longer exists.
        for slot in &mut self.taps {
            if let Some(k) = slot.key {
                if !incoming.iter().any(|t| t.key == k) {
                    slot.release();
                }
            }
        }

        // The strongest taps (direct + prominent early reflections) are
        // point-rendered when a dense HRIR grid is available.
        let mut point_keys: Vec<u32> = Vec::new();
        if self.grid.is_some() {
            let mut top: Vec<(f32, u32)> =
                incoming.iter().map(|t| (t.gains[1], t.key)).collect();
            top.sort_by(|a, b| b.0.total_cmp(&a.0));
            point_keys.extend(top.iter().take(self.point_budget).map(|(_, k)| *k));
        }

        // paths that glide on (their slots), and paths born this update
        let mut moved: Vec<usize> = Vec::new();
        let mut born: Vec<&omg_core::params::Tap> = Vec::new();
        for t in incoming {
            let tg = targets_of(t, self.sample_rate);
            let want_point = self.grid.is_some() && point_keys.contains(&t.key);
            let existing = self.taps.iter().position(|s| s.key == Some(t.key));

            match existing {
                Some(i)
                    if self.taps[i].point == want_point
                        && (self.first_params
                            || (self.taps[i].delay.target_val() - tg.delay).abs() < thresh) =>
                {
                    // Same path: glide (motion → Doppler).
                    moved.push(i);
                    let slot = &mut self.taps[i];
                    slot.base_dir = t.dir;
                    if self.first_params {
                        slot.delay.snap(tg.delay);
                        slot.g_low.snap(tg.g_low);
                        slot.g_high.snap(tg.g_high);
                        slot.lp_coef.snap(tg.lp_coef);
                        for k in 0..NCH {
                            slot.g_sh[k].snap(tg.g_sh[k]);
                        }
                    } else {
                        if fresh {
                            slot.delay.retarget_over(tg.delay, motion, interval, age);
                        }
                        slot.g_low.set(tg.g_low);
                        slot.g_high.set(tg.g_high);
                        slot.lp_coef.set(tg.lp_coef);
                        for k in 0..NCH {
                            slot.g_sh[k].set(tg.g_sh[k]);
                        }
                    }
                }
                other => {
                    // Path identity changed (or new): crossfade, never glide.
                    if let Some(i) = other {
                        self.taps[i].release();
                    }
                    born.push(t);
                }
            }
        }

        // A path born mid-motion joins the timeline of the earliest path
        // that glides on — the direct sound, most often: the listener's
        // and the source's motion, which most of a source's paths share
        // (see `PathDelay::join`). Nothing glides on: it starts still.
        let like = moved
            .iter()
            .map(|&i| self.taps[i].delay)
            .min_by(|a, b| a.target_val().total_cmp(&b.target_val()));
        for t in born {
            let tg = targets_of(t, self.sample_rate);
            let want_point = self.grid.is_some() && point_keys.contains(&t.key);
            if let Some(j) = self.taps.iter().position(|s| s.free()) {
                if !self.active.contains(&(j as u16)) {
                    self.active.push(j as u16);
                }
                let slot = &mut self.taps[j];
                slot.key = Some(t.key);
                slot.point = want_point;
                slot.base_dir = t.dir;
                match &like {
                    Some(l) if !self.first_params => slot.delay.join(tg.delay, l),
                    _ => slot.delay.snap(tg.delay),
                }
                slot.g_low.snap(tg.g_low);
                slot.g_high.snap(tg.g_high);
                slot.lp_coef.snap(tg.lp_coef);
                slot.lp = OnePoleLp::default();
                slot.eq = TapEq::default();
                if want_point {
                    if let Some(g) = &self.grid {
                        let rot = HeadRotation::new(
                            self.head[0].target_val(),
                            self.head[1].target_val(),
                            self.head[2].target_val(),
                        );
                        slot.conv.reset(g.nearest(rot.rotate_dir(t.dir)));
                    }
                }
                for k in 0..NCH {
                    if self.snap_born {
                        slot.g_sh[k].snap(tg.g_sh[k]);
                    } else {
                        slot.g_sh[k].snap(0.0);
                        slot.g_sh[k].set(tg.g_sh[k]);
                    }
                }
            }
            // Pool exhausted: drop the tap this update; a slot frees
            // up within ~1 smoothing constant.
        }

        if let Some(first) = pb.taps.iter().map(|t| t.delay_s as f64 * self.sample_rate as f64).reduce(f64::min) {
            if self.first_params || (self.onset.target_val() - first).abs() >= thresh {
                self.onset.snap(first);
            } else {
                if fresh {
                    self.onset.retarget_over(first, motion, interval, age);
                }
            }
        }

        self.fdn.set_rt60(pb.reverb.rt60[1], pb.reverb.rt60[2]);
        let lv = pb.reverb.level;
        let ratio = |x: f32| if lv[1] > 1e-6 { (x / lv[1]).clamp(0.0, 4.0) } else { 1.0 };
        self.fdn_level.set(lv[1]);
        self.send_low.set(ratio(lv[0]));
        // each band's energy as traced: the high band's faster decay
        // gives it less tail energy per send, made up here
        self.send_high.set((ratio(lv[2]) * self.fdn.high_make_up()).min(8.0));

        match &pb.remote {
            Some(r) => {
                self.remote_fdn.set_rt60(r.rt60[1], r.rt60[2]);
                let rl_raw = if r.send[1] > 1e-9 { r.send[0] / r.send[1] } else { 1.0 };
                if rl_raw > 2.0 {
                    // wall-leaked wet: continuous lowpass, anchored at lows
                    let fc = (800.0 / (rl_raw * rl_raw - 1.0).sqrt()).clamp(60.0, 2000.0);
                    self.remote_level.set(r.send[0]);
                    self.remote_low.set(1.0);
                    self.remote_high.set(1.0);
                    self.remote_lp_coef.set(OnePoleLp::coef(fc, self.sample_rate));
                } else {
                    let rr =
                        |x: f32| if r.send[1] > 1e-6 { (x / r.send[1]).clamp(0.0, 4.0) } else { 1.0 };
                    self.remote_level.set(r.send[1]);
                    self.remote_low.set(rr(r.send[0]));
                    self.remote_high.set(rr(r.send[2]));
                    self.remote_lp_coef.set(1.0);
                }
                for k in 0..NCH {
                    self.remote_sh[k].set(r.sh[k]);
                }
            }
            // Source entered our room: stop feeding the remote FDN; its
            // tail rings out through the last doorway direction.
            None => self.remote_level.set(0.0),
        }
        self.first_params = false;
        self.snap_born = false;
    }

    /// Let the next update's new paths start at full level: a source
    /// heard for the first time whose sounds all begin now (nothing
    /// sounding to crossfade from — a fade-in would swallow a click).
    pub fn snap_next(&mut self) {
        self.snap_born = true;
    }

    /// One mono source sample in (a live or generated source: its past
    /// lives in the delay line). Diffuse content (reflections, reverb)
    /// accumulates onto the shared SH bus; point-rendered taps return their
    /// binaural stereo directly. Bus decoding happens once, downstream.
    #[inline]
    pub fn process(&mut self, x: f32, bus: &mut [f32; NCH]) -> (f32, f32) {
        self.src_line.write(x);
        let line = core::mem::replace(&mut self.src_line, DelayLine::empty());
        let out = self.render(|d| line.read_back(d as f32, 0), bus);
        self.src_line = line;
        out
    }

    /// One output sample of a source made of recordings: every path reads
    /// `log` at its own emission time `now − delay`, scaled by `gain` (the
    /// mixer fader, applied as heard). `now` is the engine clock in
    /// samples; it must advance by one per call.
    #[inline]
    pub fn process_log(&mut self, log: &PlayLog, now: f64, gain: f32, bus: &mut [f32; NCH]) -> (f32, f32) {
        self.render(|d| log.at(now - d) * gain, bus)
    }

    /// Block form of [`process`](Self::process): `x.len()` (≤ [`BLOCK`])
    /// live source samples in; the bus contribution is ADDED to `bus`
    /// (`bus[k][i]`), point-rendered stereo ADDED to `pl[i]`, `pr[i]`.
    /// Renders what `x.len()` calls of `process` would, path by path
    /// instead of sample by sample (see `render_block`).
    pub fn process_block(&mut self, x: &[f32], bus: &mut BusBlock, pl: &mut [f32], pr: &mut [f32]) {
        let n = x.len().min(BLOCK).min(pl.len()).min(pr.len());
        let mut line = core::mem::replace(&mut self.src_line, DelayLine::empty());
        let mut off = 0;
        while off < n {
            let m = self.begin_sub_block(n - off);
            for &v in &x[off..off + m] {
                line.write(v);
            }
            let src = Src::Line { line: &line, last: off + m - 1 };
            self.render_block(&src, off, m, bus, pl, pr);
            off += m;
        }
        self.src_line = line;
    }

    /// Block form of [`process_log`](Self::process_log): `gain.len()`
    /// (≤ [`BLOCK`]) samples starting at engine time `now`, `gain[i]` the
    /// fader at sample i. Adds onto `bus`, `pl`, `pr` like
    /// [`process_block`](Self::process_block).
    pub fn process_log_block(
        &mut self,
        log: &PlayLog,
        now: f64,
        gain: &[f32],
        bus: &mut BusBlock,
        pl: &mut [f32],
        pr: &mut [f32],
    ) {
        let n = gain.len().min(BLOCK).min(pl.len()).min(pr.len());
        // a pause (the mixer didn't run this source): the convolver's
        // input history is not the recent input any more
        if self.dense_at != now {
            self.dense.discontinuity();
        }
        self.dense_at = now + n as f64;
        let src = Src::Log { log, now, gain };
        let mut off = 0;
        while off < n {
            let m = self.begin_sub_block(n - off);
            self.render_block(&src, off, m, bus, pl, pr);
            off += m;
        }
    }

    /// Head pose, HRIR re-selection and bookkeeping for the next sub-block
    /// of at most `left` samples; returns its length. Sub-blocks end where
    /// the per-sample path would re-select (every `RESELECT_EVERY`
    /// samples), so the re-selection lands on the same sample.
    fn begin_sub_block(&mut self, left: usize) -> usize {
        let m;
        if self.reselect_countdown == 0 {
            let yaw = self.head[0].tick();
            let pitch = self.head[1].tick();
            let roll = self.head[2].tick();
            self.reselect(yaw, pitch, roll);
            self.reselect_countdown = RESELECT_EVERY;
            m = left.min(RESELECT_EVERY as usize);
            advance_n(&mut self.head, m - 1);
        } else {
            m = left.min(self.reselect_countdown as usize);
            advance_n(&mut self.head, m);
        }
        self.reselect_countdown -= m as u32;
        self.since_params = self.since_params.saturating_add(m as u32);
        m
    }

    /// Re-select point-tap HRIRs against the head pose (PointConv
    /// crossfades index changes) and refresh the remote-reverb aperture
    /// spread from the smoothed door direction.
    fn reselect(&mut self, yaw: f32, pitch: f32, roll: f32) {
        if let Some(grid) = &self.grid {
            let rot = HeadRotation::new(yaw, pitch, roll);
            for tap in &mut self.taps {
                if tap.point && tap.key.is_some() {
                    tap.conv.set_dir(grid, rot.rotate_dir(tap.base_dir));
                }
            }
        }
        let center: [f32; NCH] = core::array::from_fn(|k| self.remote_sh[k].current());
        for (i, offs) in REMOTE_SPREAD_RAD.iter().enumerate() {
            let mut v = center;
            rotate_z(&mut v, *offs);
            self.remote_spread[i] = v;
        }
    }

    /// Samples `off..off + m` of a block (`m` ≤ `BLOCK`, no re-selection
    /// inside), path by path: each tap runs its whole stretch with its
    /// state in registers — delay motion per sample (the slope is the
    /// Doppler pitch), the source read for the whole window at once,
    /// lowpass + shelves, then a mono block spread onto the nine bus
    /// channels with the gain smoothers in closed form (two vectorizable
    /// multiply-adds per channel and sample). Same signal as `m` calls of
    /// `render`, up to float rounding (tests/fx_block_render.rs).
    fn render_block(&mut self, src: &Src, off: usize, m: usize, bus: &mut BusBlock, pl: &mut [f32], pr: &mut [f32]) {
        if !self.mute_taps {
            self.taps_block(src, off, m, bus, pl, pr);
        }
        // Late field + remote wet. Both FDNs asleep with their sends shut
        // and their send filters run down: the per-sample pass would only
        // move smoothers and push zeros, so jump them instead (most
        // sources, most of the time).
        let quiet = self.fdn_idle >= self.fdn.idle_after()
            && self.remote_idle >= self.remote_fdn.idle_after()
            && self.fdn_level.current() <= 1e-6
            && self.fdn_level.target_val() <= 1e-6
            && self.remote_level.current() <= 1e-6
            && self.remote_level.target_val() <= 1e-6
            && self.send_eq.quiet(1e-9)
            && self.remote_eq.quiet(1e-9)
            && self.remote_lp.state.abs() < 1e-9;
        let dense = self.dense.has_kernel();
        if quiet && !dense {
            for _ in 0..m {
                self.onset.tick();
            }
            for g in [
                &mut self.fdn_level,
                &mut self.send_low,
                &mut self.send_high,
                &mut self.remote_level,
                &mut self.remote_low,
                &mut self.remote_high,
                &mut self.remote_lp_coef,
            ] {
                advance_n(core::slice::from_mut(g), m);
            }
        } else {
            // the source as of the first arrival, read for the block
            let mut d = [[0.0f64; 1]; BLOCK];
            let mut x = [0.0f32; BLOCK];
            PathDelay::tick_lanes(core::array::from_mut(&mut self.onset), &mut d[..m]);
            src.block(off, &d[..m], 0, &mut x[..m]);
            if quiet {
                for g in [
                    &mut self.fdn_level,
                    &mut self.send_low,
                    &mut self.send_high,
                    &mut self.remote_level,
                    &mut self.remote_low,
                    &mut self.remote_high,
                    &mut self.remote_lp_coef,
                ] {
                    advance_n(core::slice::from_mut(g), m);
                }
            } else {
                self.late_block(&x[..m], off, bus);
            }
            // engine v4: the traced dense response, convolved (turned into
            // the bus's frame when that is the listener's)
            if dense {
                let yaw = self.dense_yaw.current();
                let turning = yaw != 0.0 || self.dense_yaw.target_val() != 0.0;
                if !turning {
                    self.dense.process(&x[..m], bus, off);
                } else {
                    let mut t: BusBlock = [[0.0; BLOCK]; NCH];
                    self.dense.process(&x[..m], &mut t, off);
                    advance_n(core::slice::from_mut(&mut self.dense_yaw), m);
                    let (s, c) = yaw.sin_cos();
                    for j in off..off + m {
                        let (y, z, x) = (t[1][j], t[2][j], t[3][j]);
                        bus[0][j] += t[0][j];
                        bus[1][j] += -s * x + c * y;
                        bus[2][j] += z;
                        bus[3][j] += c * x + s * y;
                    }
                }
            }
        }
        // the remote direction smoothers are only read at re-selection
        let pm = self.remote_sh[0].coef().powi(m as i32);
        for g in &mut self.remote_sh {
            g.advance(pm);
        }
    }

    /// The early paths over samples `off..off + m`, in groups of 8 (4 for
    /// a short tail) rendered side by side (see `tap_group`).
    fn taps_block(&mut self, src: &Src, off: usize, m: usize, bus: &mut BusBlock, pl: &mut [f32], pr: &mut [f32]) {
        let Renderer { taps, active, pw, grid, c_low, c_high, .. } = self;
        let mut cx = GroupCx { pw, grid: grid.as_deref(), c_low: *c_low, c_high: *c_high, src, off, m, bus, pl, pr };
        let mut freed = false;
        let mut a0 = 0;
        while a0 < active.len() {
            let left = active.len() - a0;
            let c = if left > 4 { left.min(8) } else { left };
            let g = &active[a0..a0 + c];
            freed |= if c > 4 { tap_group::<8>(taps, g, &mut cx) } else { tap_group::<4>(taps, g, &mut cx) };
            a0 += c;
        }
        if freed {
            let taps = &self.taps;
            self.active.retain(|&i| !taps[i as usize].free());
        }
    }

    /// `read(d)`: the source as emitted `d` samples ago (d ≥ 1).
    #[inline]
    fn render(&mut self, read: impl Fn(f64) -> f32, bus: &mut [f32; NCH]) -> (f32, f32) {
        self.since_params = self.since_params.saturating_add(1);
        let foa = bus;
        let mut pl = 0.0f32;
        let mut pr = 0.0f32;
        let yaw = self.head[0].tick();
        let pitch = self.head[1].tick();
        let roll = self.head[2].tick();

        // Every 128 samples (~375 Hz): re-select point-tap HRIRs against the
        // current head pose (PointConv crossfades index changes), and refresh
        // the remote-reverb aperture spread from the smoothed door direction.
        if self.reselect_countdown == 0 {
            self.reselect_countdown = RESELECT_EVERY;
            self.reselect(yaw, pitch, roll);
        }
        self.reselect_countdown -= 1;

        let mut ai = 0;
        while ai < self.active.len() && !self.mute_taps {
            let idx = self.active[ai] as usize;
            let tap = &mut self.taps[idx];
            // A releasing slot keeps ticking until its fade-out finishes;
            // then it leaves the active list and costs nothing.
            // a releasing slot under its floor closes with a linear tail
            let mut fade = 1.0f32;
            if tap.key.is_none() && tap.g_sh[0].target_val() == 0.0 {
                if tap.tail == 0 && tap.g_sh[0].current().abs() < tap.release_floor.max(1e-6) {
                    tap.tail = TAIL_SAMPLES;
                }
                if tap.tail > 0 {
                    tap.tail -= 1;
                    if tap.tail == 0 {
                        for g in &mut tap.g_sh {
                            g.snap(0.0);
                        }
                        self.active.swap_remove(ai);
                        continue;
                    }
                    fade = tap.tail as f32 / TAIL_SAMPLES as f32;
                }
            }
            if tap.free() {
                self.active.swap_remove(ai);
                continue;
            }
            ai += 1;
            let d = tap.delay.tick();
            let gl = tap.g_low.tick();
            let gh = tap.g_high.tick();
            let mut g = [0.0f32; NCH];
            for k in 0..NCH {
                g[k] = tap.g_sh[k].tick() * fade;
            }
            // Skip silent slots entirely (their smoothers still ticked).
            if g[0].abs() < 1e-7 {
                continue;
            }
            let s = read(d.max(1.0));
            let shaped = tap.lp.tick(s, tap.lp_coef.tick());
            let toned = tap.eq.tick(shaped, gl, gh, self.c_low, self.c_high);
            if tap.point {
                if let Some(grid) = &self.grid {
                    // g[0] is the broadband mid gain (W of SN3D SH = 1).
                    let (l, r) = tap.conv.process(grid, toned * g[0] * POINT_GAIN);
                    pl += l;
                    pr += r;
                    continue;
                }
            }
            for k in 0..NCH {
                foa[k] += toned * g[k];
            }
        }

        let x = read(self.onset.tick().max(1.0));
        self.late(x, foa);
        (pl, pr)
    }

    /// Late field and coupled-room wet for one sample, `x` = the source
    /// as of its first arrival. Adds onto `foa`.
    #[inline]
    fn late(&mut self, x: f32, foa: &mut [f32; NCH]) {
        // Late field: source signal (as of the first arrival) through the
        // FDN, scaled by the traced per-band late level — independent of
        // source distance, like a real diffuse field. The level applies at
        // the OUTPUT: it is the field at the listener, so when the rays
        // stop reaching (you leave the cathedral) its tail goes with them
        // instead of ringing on for an RT60 outside; a source that falls
        // silent still rings out, since its signal stopped, not the level.
        // The input is gated off once the level is negligible, so an FDN
        // whose input AND tail are silent goes to sleep — most sources are
        // inaudible at any moment; waking is instant.
        let own_level = self.fdn_level.tick();
        let own_gate = if own_level > 1e-6 || self.fdn_level.target_val() > 1e-6 { 1.0 } else { 0.0 };
        let wet_in = self.send_eq.tick(
            x * own_gate,
            self.send_low.tick(),
            self.send_high.tick(),
            self.c_low,
            self.c_high,
        );
        let own_in = if self.mute_own_fdn { 0.0 } else { wet_in };
        let own_norm = self.fdn.energy_norm();
        fdn_step(&mut self.fdn, &mut self.fdn_idle, own_in, 1.0, own_level * own_norm, own_gate, &self.fdn_dirs, foa);

        // Coupled-room wet: source room's reverb as a directional emitter
        // at the doorway (mono sum — a distant aperture, not a diffuse field
        // around the listener).
        // remote wet: likewise scaled at the output (its level is how much
        // of that room reaches you through the portal, right now)
        let rem_level = self.remote_level.tick();
        let rem_gate = if rem_level > 1e-6 || self.remote_level.target_val() > 1e-6 { 1.0 } else { 0.0 };
        let rpre = self.remote_lp.tick(x * rem_gate, self.remote_lp_coef.tick());
        let rin = self.remote_eq.tick(
            rpre,
            self.remote_low.tick(),
            self.remote_high.tick(),
            self.c_low,
            self.c_high,
        );
        // Tick the direction smoothers (spread cache reads them at 375 Hz).
        for k in 0..NCH {
            self.remote_sh[k].tick();
        }
        let rem_in = if self.mute_remote { 0.0 } else { rin };
        let rem_norm = self.remote_fdn.energy_norm();
        fdn_step(&mut self.remote_fdn, &mut self.remote_idle, rem_in, 1.0, rem_level * rem_norm, rem_gate, &self.remote_spread, foa);
    }

    /// `late` for block samples `off..off + x.len()` (`x[j]` = the source
    /// as of the first arrival): the send chains and smoothers per sample,
    /// each FDN a block at a time while it is awake and stays awake
    /// through the block (around falling asleep or waking it steps per
    /// sample, exactly as `late`).
    fn late_block(&mut self, x: &[f32], off: usize, bus: &mut BusBlock) {
        let m = x.len();
        let (c_low, c_high) = (self.c_low, self.c_high);
        let mut own_in = [0.0f32; BLOCK];
        let mut own_lvl = [0.0f32; BLOCK];
        let mut own_gate = [0.0f32; BLOCK];
        let mut rem_in = [0.0f32; BLOCK];
        let mut rem_lvl = [0.0f32; BLOCK];
        let mut rem_gate = [0.0f32; BLOCK];
        // energy-calibrated: a level L plays L² (see `Fdn::energy_norm`)
        let (own_norm, rem_norm) = (self.fdn.energy_norm(), self.remote_fdn.energy_norm());
        for (j, &x) in x.iter().enumerate() {
            let own_level = self.fdn_level.tick();
            let gate = if own_level > 1e-6 || self.fdn_level.target_val() > 1e-6 { 1.0 } else { 0.0 };
            let wet_in = self.send_eq.tick(x * gate, self.send_low.tick(), self.send_high.tick(), c_low, c_high);
            own_in[j] = if self.mute_own_fdn { 0.0 } else { wet_in };
            own_lvl[j] = own_level * own_norm;
            own_gate[j] = gate;
            let rem_level = self.remote_level.tick();
            let gate = if rem_level > 1e-6 || self.remote_level.target_val() > 1e-6 { 1.0 } else { 0.0 };
            let rpre = self.remote_lp.tick(x * gate, self.remote_lp_coef.tick());
            let rin = self.remote_eq.tick(rpre, self.remote_low.tick(), self.remote_high.tick(), c_low, c_high);
            rem_in[j] = if self.mute_remote { 0.0 } else { rin };
            rem_lvl[j] = rem_level * rem_norm;
            rem_gate[j] = gate;
        }
        let own = FdnFeed { input: &own_in[..m], pre: 1.0, lvl: &own_lvl[..m], gate: &own_gate[..m] };
        fdn_block(&mut self.fdn, &mut self.fdn_idle, &own, &self.fdn_dirs, bus, off);
        let rem = FdnFeed { input: &rem_in[..m], pre: 1.0, lvl: &rem_lvl[..m], gate: &rem_gate[..m] };
        fdn_block(&mut self.remote_fdn, &mut self.remote_idle, &rem, &self.remote_spread, bus, off);
    }
}

