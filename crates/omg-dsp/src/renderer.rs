//! Per-source renderer: one shared source delay line, a pool of spatialized
//! early taps (direct + image sources), an FDN late field, everything summed
//! on a FOA bus and decoded to stereo.
//!
//! Tap lifecycle: incoming taps carry a stable `key` identifying their
//! propagation path. Same key ⇒ the slot glides (delay motion = Doppler,
//! physically correct). New key ⇒ a free slot fades in from silence at the
//! final delay. Vanished key ⇒ the slot fades out and is recycled. Delay is
//! never slid between two different paths — that would be a pitch chirp.

use crate::ambi::{encode_gains, rotate_z, NCH};
use crate::delay::DelayLine;
use crate::fdn::{Fdn, NLINES};
use crate::filter::{OnePoleLp, TapEq};
use crate::hrtf::{HrirGrid, PointConv};
use crate::smooth::Smoothed;
use omg_core::params::ParamBlock;
use std::sync::Arc;

/// Pool is sized ~2× the largest expected tap count so fading-out ghosts
/// of a previous acoustic state can coexist with the new state.
pub const MAX_TAPS: usize = 448;
/// Delay retarget threshold: same-key delay changes above this are treated
/// as a path identity change (crossfade), below it as motion (glide).
const RETARGET_MS: f32 = 6.0;
/// Loudness match of point-rendered direct paths vs. the bus decode.
const POINT_GAIN: f32 = 0.8;
/// Default per-source point-render budget: how many of the strongest taps
/// get their own nearest-HRIR convolution (full-sharpness localization)
/// instead of the order-2 bus. A measured CPU budget, not a quality
/// ceiling: with the SIMD convolution kernel, 24 holds 2.06× realtime at
/// the worst-case scene position on an M1 (tools/bench_web.mjs). The web
/// worklet overrides this adaptively from its own measured load.
const DEFAULT_POINT_TAPS: usize = 24;
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
    delay: Smoothed,      // samples
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
    delay: f32,
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
            delay: t.delay_s * sample_rate,
            g_low: 1.0,
            g_high: 1.0,
            g_sh: core::array::from_fn(|k| t.gains[0] * enc[k]),
            lp_coef: OnePoleLp::coef(fc, sample_rate),
        }
    } else {
        TapTargets {
            delay: t.delay_s * sample_rate,
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
    /// Fast head rotation (device orientation / mouse look), applied to
    /// point-tap HRIR selection here and to the bus at the decode stage.
    head_yaw: Smoothed,
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
            src_line: DelayLine::new((sample_rate * 0.7) as usize),
            taps: (0..MAX_TAPS)
                .map(|_| TapState {
                    key: None,
                    point: false,
                    base_dir: [1.0, 0.0, 0.0],
                    delay: Smoothed::new(480.0, 0.05, sample_rate),
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
            head_yaw: Smoothed::new(0.0, 0.03, sample_rate),
            reselect_countdown: 0,
            version_seen: 0,
            first_params: true,
            mute_taps: false,
            mute_own_fdn: false,
            mute_remote: false,
        }
    }

    pub fn set_head_yaw(&mut self, yaw: f32) {
        self.head_yaw.set(yaw);
    }

    /// Set how many of the strongest taps are point-rendered. Takes effect
    /// on the next ParamBlock; taps whose mode changes re-slot with a
    /// crossfade (a live slot never switches mode in place).
    pub fn set_point_budget(&mut self, n: usize) {
        self.point_budget = n;
    }

    /// Called from the audio thread whenever a new ParamBlock is available.
    pub fn set_params(&mut self, pb: &ParamBlock) {
        if pb.version == self.version_seen {
            return;
        }
        self.version_seen = pb.version;

        // CPU ceiling: keep only the strongest incoming taps. The weakest
        // image sources are inaudible under the ones we keep; this bounds
        // worst-case render cost on throttled devices.
        const MAX_INCOMING: usize = 160;
        let mut incoming: Vec<&omg_core::params::Tap> = pb.taps.iter().collect();
        if incoming.len() > MAX_INCOMING {
            incoming.sort_by(|a, b| b.gains[1].total_cmp(&a.gains[1]));
            incoming.truncate(MAX_INCOMING);
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
                            || (self.taps[i].delay.target_val() - tg.delay).abs()
                                < self.retarget_thresh) =>
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
                        slot.delay.set(tg.delay);
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
                                let psi = self.head_yaw.target_val();
                                slot.conv.reset(g.nearest(rotate_dir(t.dir, psi)));
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

    /// One mono source sample in. Diffuse content (reflections, reverb)
    /// accumulates onto the shared SH bus; point-rendered taps return their
    /// binaural stereo directly. Bus decoding happens once, downstream.
    #[inline]
    pub fn process(&mut self, x: f32, bus: &mut [f32; NCH]) -> (f32, f32) {
        self.src_line.write(x);

        let foa = bus;
        let mut pl = 0.0f32;
        let mut pr = 0.0f32;
        let psi = self.head_yaw.tick();

        // Every 128 samples (~375 Hz): re-select point-tap HRIRs against the
        // current head yaw (PointConv crossfades index changes), and refresh
        // the remote-reverb aperture spread from the smoothed door direction.
        if self.reselect_countdown == 0 {
            self.reselect_countdown = 128;
            if let Some(grid) = &self.grid {
                for tap in &mut self.taps {
                    if tap.point && tap.key.is_some() {
                        tap.conv.set_dir(grid, rotate_dir(tap.base_dir, psi));
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

        for tap in &mut self.taps {
            if self.mute_taps {
                break;
            }
            // Settled-inactive slots cost nothing (the dominant case for
            // idle sources and unused pool headroom). A releasing slot
            // keeps ticking until its fade-out finishes.
            if tap.key.is_none()
                && tap.g_sh[0].target_val() == 0.0
                && tap.g_sh[0].current().abs() < 1e-6
            {
                continue;
            }
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
            let s = self.src_line.read(d);
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

        // Late field: raw source signal scaled by the traced per-band late
        // level — independent of source distance, like a real diffuse field.
        // Scaling the FDN *input* (not output) lets the tail ring out
        // naturally when the level parameter drops (e.g. stepping outdoors).
        let wet_in = self.send_eq.tick(
            x * self.fdn_level.tick(),
            self.send_low.tick(),
            self.send_high.tick(),
            self.c_low,
            self.c_high,
        );
        let mut lines = [0.0f32; NLINES];
        self.fdn.process(if self.mute_own_fdn { 0.0 } else { wet_in }, &mut lines);

        // Coupled-room wet: source room's reverb as a directional emitter
        // at the doorway (mono sum — a distant aperture, not a diffuse field
        // around the listener).
        let rpre = self.remote_lp.tick(x * self.remote_level.tick(), self.remote_lp_coef.tick());
        let rin = self.remote_eq.tick(
            rpre,
            self.remote_low.tick(),
            self.remote_high.tick(),
            self.c_low,
            self.c_high,
        );
        let mut rlines = [0.0f32; NLINES];
        self.remote_fdn.process(if self.mute_remote { 0.0 } else { rin }, &mut rlines);
        // Tick the direction smoothers (spread cache reads them at 375 Hz).
        for k in 0..NCH {
            self.remote_sh[k].tick();
        }
        for (i, line) in rlines.iter().enumerate() {
            let g = line * 0.9;
            for k in 0..NCH {
                foa[k] += g * self.remote_spread[i][k];
            }
        }

        let lvl = 0.5;
        for i in 0..NLINES {
            let g = lines[i] * lvl;
            for k in 0..NCH {
                foa[k] += g * self.fdn_dirs[i][k];
            }
        }
        (pl, pr)
    }
}

/// Listener-frame direction under head yaw ψ: dir' = Rz(-ψ)·dir.
#[inline]
fn rotate_dir(d: [f32; 3], psi: f32) -> [f32; 3] {
    let (s, c) = psi.sin_cos();
    [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]
}
