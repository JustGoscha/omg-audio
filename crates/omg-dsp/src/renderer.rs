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

use crate::ambi::{encode_gains, rotate_z, HeadRotation, NCH};
use crate::delay::DelayLine;
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
/// Samples of joint input+tail silence before an FDN goes to sleep
/// (~21 ms at 48 kHz: long enough to never clip a real tail — the energy
/// check keeps it awake while anything rings — short enough that silent
/// sources stop costing within a frame).
const FDN_IDLE_AFTER: u32 = 1024;
/// Angular spread of the remote-reverb lines around the doorway (±~30°).
const REMOTE_SPREAD_RAD: [f32; NLINES] =
    [-0.55, -0.38, -0.22, -0.08, 0.08, 0.22, 0.38, 0.55];
const LOW_SHELF_HZ: f32 = 250.0;
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
}

impl TapState {
    fn free(&self) -> bool {
        self.key.is_none()
            && self.g_sh[0].target_val() == 0.0
            && self.g_sh[0].current().abs() < 1e-6
    }

    fn release(&mut self) {
        self.key = None;
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
    /// Samples since the last applied ParamBlock (retarget window).
    since_params: u32,
    /// Average samples between fresh ParamBlocks (simulation updates):
    /// the interval path delays plan their constant-rate motion over.
    update_interval: f64,
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
    /// Samples of joint input+tail silence per FDN; at FDN_IDLE_AFTER the
    /// FDN sleeps (see process). Starts asleep.
    fdn_idle: u32,
    remote_idle: u32,
    reselect_countdown: u32,
    version_seen: u64,
    first_params: bool,
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
        Self {
            src_line: DelayLine::new((sample_rate * LIVE_LINE_S) as usize),
            active: Vec::with_capacity(MAX_TAPS),
            taps: (0..MAX_TAPS)
                .map(|_| TapState {
                    key: None,
                    point: false,
                    base_dir: [1.0, 0.0, 0.0],
                    delay: PathDelay::new(480.0),
                    g_low: Smoothed::new(1.0, 0.05, sample_rate),
                    g_high: Smoothed::new(1.0, 0.05, sample_rate),
                    g_sh: [Smoothed::new(0.0, 0.05, sample_rate); NCH],
                    lp_coef: Smoothed::new(1.0, 0.05, sample_rate),
                    lp: OnePoleLp::default(),
                    eq: TapEq::default(),
                    conv: PointConv::new(conv_taps, sample_rate),
                })
                .collect(),
            grid,
            onset: PathDelay::new(1.0),
            since_params: 0,
            update_interval: 0.05 * sample_rate as f64,
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
            sample_rate,
            retarget_thresh: RETARGET_MS * 1e-3 * sample_rate,
            point_budget: DEFAULT_POINT_TAPS,
            tap_ceiling: DEFAULT_TAP_CEILING,
            head: core::array::from_fn(|_| Smoothed::new(0.0, 0.03, sample_rate)),
            fdn_idle: FDN_IDLE_AFTER,
            remote_idle: FDN_IDLE_AFTER,
            reselect_countdown: 0,
            version_seen: 0,
            first_params: true,
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

    /// Forget which ParamBlock version was applied, so the next
    /// `set_params` re-applies even the same block — used when a budget
    /// share changes and must take effect without waiting a tick.
    pub fn invalidate_params(&mut self) {
        self.version_seen = u64::MAX;
    }

    /// Called from the audio thread whenever a new ParamBlock is available.
    pub fn set_params(&mut self, pb: &ParamBlock) {
        if pb.version == self.version_seen {
            return;
        }
        self.version_seen = pb.version;
        // judged against the TYPICAL interval too: an update arriving
        // early must not turn fast motion into a crossfade
        let thresh = (self.retarget_thresh as f64).max(
            RETARGET_FRACTION_OF_TICK * (self.since_params as f64).max(self.update_interval),
        );
        let fresh = pb.version != self.last_version;
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
                            slot.delay.retarget(tg.delay, interval);
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
                    if let Some(j) = self.taps.iter().position(|s| s.free()) {
                        if !self.active.contains(&(j as u16)) {
                            self.active.push(j as u16);
                        }
                        let slot = &mut self.taps[j];
                        slot.key = Some(t.key);
                        slot.point = want_point;
                        slot.base_dir = t.dir;
                        slot.delay.snap(tg.delay);
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
                            slot.g_sh[k].snap(0.0);
                            slot.g_sh[k].set(tg.g_sh[k]);
                        }
                    }
                    // Pool exhausted: drop the tap this update; a slot frees
                    // up within ~1 smoothing constant.
                }
            }
        }

        if let Some(first) = pb.taps.iter().map(|t| t.delay_s as f64 * self.sample_rate as f64).reduce(f64::min) {
            if self.first_params || (self.onset.target_val() - first).abs() >= thresh {
                self.onset.snap(first);
            } else {
                if fresh {
                    self.onset.retarget(first, interval);
                }
            }
        }

        self.fdn.set_rt60(pb.reverb.rt60[1], pb.reverb.rt60[2]);
        let lv = pb.reverb.level;
        let ratio = |x: f32| if lv[1] > 1e-6 { (x / lv[1]).clamp(0.0, 4.0) } else { 1.0 };
        self.fdn_level.set(lv[1]);
        self.send_low.set(ratio(lv[0]));
        self.send_high.set(ratio(lv[2]));

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
    }

    /// One mono source sample in (a live or generated source: its past
    /// lives in the delay line). Diffuse content (reflections, reverb)
    /// accumulates onto the shared SH bus; point-rendered taps return their
    /// binaural stereo directly. Bus decoding happens once, downstream.
    #[inline]
    pub fn process(&mut self, x: f32, bus: &mut [f32; NCH]) -> (f32, f32) {
        self.src_line.write(x);
        let line = core::mem::replace(&mut self.src_line, DelayLine::empty());
        let out = self.render(|d| line.read(d as f32), bus);
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
            self.reselect_countdown = 128;
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
        self.reselect_countdown -= 1;

        let mut ai = 0;
        while ai < self.active.len() && !self.mute_taps {
            let idx = self.active[ai] as usize;
            let tap = &mut self.taps[idx];
            // A releasing slot keeps ticking until its fade-out finishes;
            // then it leaves the active list and costs nothing.
            if tap.key.is_none()
                && tap.g_sh[0].target_val() == 0.0
                && tap.g_sh[0].current().abs() < 1e-6
            {
                self.active.swap_remove(ai);
                continue;
            }
            ai += 1;
            let d = tap.delay.tick();
            let gl = tap.g_low.tick();
            let gh = tap.g_high.tick();
            let mut g = [0.0f32; NCH];
            for k in 0..NCH {
                g[k] = tap.g_sh[k].tick();
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
        let x = read(self.onset.tick().max(1.0));
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
        if own_in.abs() > 1e-7 {
            self.fdn_idle = 0;
        }
        if self.fdn_idle < FDN_IDLE_AFTER {
            let mut lines = [0.0f32; NLINES];
            self.fdn.process(own_in, &mut lines);
            let mut e = 0.0f32;
            let lvl = 0.5 * own_level;
            for i in 0..NLINES {
                let g = lines[i] * lvl;
                e += g.abs();
                for k in 0..NCH {
                    foa[k] += g * self.fdn_dirs[i][k];
                }
            }
            if e < 1e-6 && own_in.abs() <= 1e-7 {
                self.fdn_idle += 1;
                // asleep with its level gone: what it held is inaudible
                // and must not resume as a ghost when the level returns
                if self.fdn_idle == FDN_IDLE_AFTER && own_gate == 0.0 {
                    self.fdn.clear();
                }
            } else {
                self.fdn_idle = 0;
            }
        }

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
        if rem_in.abs() > 1e-7 {
            self.remote_idle = 0;
        }
        if self.remote_idle < FDN_IDLE_AFTER {
            let mut rlines = [0.0f32; NLINES];
            self.remote_fdn.process(rem_in, &mut rlines);
            let mut e = 0.0f32;
            for (i, line) in rlines.iter().enumerate() {
                let g = line * 0.9 * rem_level;
                e += g.abs();
                for k in 0..NCH {
                    foa[k] += g * self.remote_spread[i][k];
                }
            }
            if e < 1e-6 && rem_in.abs() <= 1e-7 {
                self.remote_idle += 1;
                if self.remote_idle == FDN_IDLE_AFTER && rem_gate == 0.0 {
                    self.remote_fdn.clear();
                }
            } else {
                self.remote_idle = 0;
            }
        }
        (pl, pr)
    }
}

