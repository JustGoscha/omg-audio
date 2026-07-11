//! Rain, synthesized from its statistics — the same philosophy as the late
//! reverb field: individually audible events rendered exactly, the dense
//! remainder as a texture with the right aggregate properties.
//!
//!  - Drops: a Poisson process whose rate follows intensity. WHERE a drop
//!    lands decides its sound (modal impact synthesis per material): glass
//!    ticks on the actual window panes of the listener's room, stone thuds
//!    and metal pings on the roof when the sky is really overhead, splash
//!    pings on the ground outdoors.
//!  - Downpour noise: what the drops fuse into. It emanates from the
//!    surfaces collecting the rain — outdoors a low ring of splash field
//!    around the listener; indoors it pours through the environment's
//!    aperture inlets and seeps through the shell, both band-shaped by the
//!    simulation's geometry pricing.
//!
//! Everything location-dependent arrives via [`Environment`] — the same
//! geometry-priced routing the ambience uses — so room transitions and
//! door swings move the rain continuously. Intensity is slew-limited
//! (~6 s), so rain starts and stops like weather, not like a fader.

use crate::ambi::{encode_gains, NCH};
use crate::bands::BandSplit;
use crate::env::{Environment, RouteSlots, MAX_ENV_ROUTES, MAX_ENV_WINDOWS};
use crate::smooth::Smoothed;
use omg_core::rng::Rng;

const MAX_DROPS: usize = 32;
/// Poisson rate at full intensity (drops audible as events; beyond this
/// they'd fuse anyway, so the hiss carries the rest).
const RATE_FULL_HZ: f32 = 260.0;
/// Overall level trim (calibrated against the ambience bed).
const MASTER: f32 = 0.22;
/// Structure-borne impact level: what reaches the interior through a
/// roof, pane or wall is a transmitted knock, far below the airborne
/// splash outside — scales every indoor surface impact.
const STRUCTURE_LEVEL: f32 = 0.24;
/// Splash-field noise through an aperture inlet, per unit route gain.
/// An opening passes a slice of the field, not the field itself.
const ROUTE_HISS: f32 = 0.3;
/// Diffuse noise residual per unit seep amplitude: the shell transmits
/// pressure, but the splash noise loses its coherence going through.
const SEEP_HISS: f32 = 0.3;
/// Sample bank slot length (150 ms @ 48 kHz) — keep in sync with
/// tools/make_drops.py.
pub const BANK_SLOT: usize = 7200;

/// Modal impact synthesis: a surface is a small set of resonant modes;
/// a drop is a soft noise burst driving them. The practical middle path
/// between samples and physical simulation — material-parametric, so
/// glass, metal and stone genuinely sound like themselves.
/// Per mode: (frequency Hz, decay tau s, relative amplitude).
type ModeTable = [(f32, f32, f32); 4];

const GLASS_MODES: ModeTable =
    [(650.0, 0.030, 0.5), (1750.0, 0.026, 0.9), (3050.0, 0.018, 1.0), (4700.0, 0.010, 0.5)];
const METAL_MODES: ModeTable =
    [(820.0, 0.090, 0.7), (2200.0, 0.120, 1.0), (3600.0, 0.080, 0.8), (5400.0, 0.050, 0.4)];
const STONE_MODES: ModeTable =
    [(380.0, 0.011, 1.0), (900.0, 0.008, 0.8), (1550.0, 0.006, 0.5), (2400.0, 0.004, 0.25)];

struct Drop {
    phase: f32,
    step: f32,   // radians/sample
    env: f32,    // exponential amplitude
    decay: f32,  // per-sample multiplier
    /// Impact click transient (very fast noise burst) — the "tap".
    click: f32,
    /// Structure-borne (roof knock / window tick): heard at full level
    /// indoors instead of being shell-attenuated like airborne drops.
    surface: bool,
    /// Sample-bank playback (real recorded splats): slice offset, read
    /// position and rate (pitch variation), or `usize::MAX` for synth.
    bank_off: usize,
    bank_pos: f32,
    bank_rate: f32,
    /// Modal resonators: per mode [c1, c2, y1, y2] and a drive gain —
    /// active when `modal` is set (excitation drives the modes).
    modal: bool,
    modes: [[f32; 4]; 4],
    mode_gain: [f32; 4],
    exc: f32,
    exc_lp: f32,
    exc_coef: f32,
    enc: [f32; NCH],
}

pub struct Rain {
    intensity: Smoothed,
    /// User mixer gain (smoothed), on top of the weather intensity.
    gain: Smoothed,
    rng: Rng,
    drops: [Drop; MAX_DROPS],
    next: usize,
    /// Normalized time to the next drop (exponential inter-arrival at
    /// unit rate) — advanced by rate/sr each sample, so the schedule
    /// adapts as the rate ramps instead of freezing a stale interval.
    spawn_in: f32,
    // downpour hiss shaping: noise → band between the two poles
    hiss_lo: f32,
    hiss_hi: f32,
    // independent noise chain for the shell seep: the structure decoheres
    // what it transmits, so seep must sum with the inlets in POWER
    seep_lo: f32,
    seep_hi: f32,
    // roof drumming: heavy lowpass on its own drop layer
    drum_lp: f32,
    /// Bank of real recorded drop/splat hits (uniform BANK_SLOT slices).
    bank: Vec<f32>,
    /// Environment inlets (apertures / horizon sectors), id-keyed and
    /// smoothed — the outdoor rain heard THROUGH the room's openings.
    routes: RouteSlots,
    /// Glass panes of the listener's room: (encode, level) — drops land
    /// ON these, at their real directions.
    windows: [([f32; NCH], f32); MAX_ENV_WINDOWS],
    n_windows: usize,
    /// 0 = open sky … 1 = sealed (continuous through blend zones).
    enclosure: Smoothed,
    /// Sky-exposed ceiling fraction overhead — gates roof drops/drumming.
    roof: Smoothed,
    /// Shell seep spectrum (amplitude per band) for the diffuse wash.
    seep: [Smoothed; 3],
    seep_split: BandSplit,
    sample_rate: f32,
    up_enc: [f32; NCH],
    /// Splash-field encode outdoors: the noise comes from the wet ground
    /// AROUND the listener, not from overhead.
    ground_enc: [f32; NCH],
}

impl Rain {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            intensity: Smoothed::new(0.0, 6.0, sample_rate),
            gain: Smoothed::new(1.0, 0.02, sample_rate),
            rng: Rng::new(0x5EED_5EED),
            drops: core::array::from_fn(|_| Drop {
                phase: 0.0,
                step: 0.0,
                env: 0.0,
                decay: 0.0,
                click: 0.0,
                surface: false,
                bank_off: usize::MAX,
                bank_pos: 0.0,
                bank_rate: 1.0,
                modal: false,
                modes: [[0.0; 4]; 4],
                mode_gain: [0.0; 4],
                exc: 0.0,
                exc_lp: 0.0,
                exc_coef: 0.3,
                enc: [0.0; NCH],
            }),
            next: 0,
            spawn_in: 0.0,
            hiss_lo: 0.0,
            hiss_hi: 0.0,
            seep_lo: 0.0,
            seep_hi: 0.0,
            drum_lp: 0.0,
            bank: Vec::new(),
            routes: RouteSlots::new(sample_rate, 0.25),
            windows: [([0.0; NCH], 0.0); MAX_ENV_WINDOWS],
            n_windows: 0,
            enclosure: Smoothed::new(0.0, 0.5, sample_rate),
            roof: Smoothed::new(0.0, 0.5, sample_rate),
            seep: core::array::from_fn(|_| Smoothed::new(0.0, 0.4, sample_rate)),
            seep_split: BandSplit::new(sample_rate),
            sample_rate,
            up_enc: encode_gains([0.0, 0.0, 1.0]),
            ground_enc: encode_gains([0.0, 0.0, -0.4]),
        }
    }

    /// Target intensity 0 (dry) … 1 (downpour); ramped internally.
    pub fn set_intensity(&mut self, v: f32) {
        self.intensity.set(v.clamp(0.0, 1.0));
    }

    pub fn intensity(&self) -> f32 {
        self.intensity.current()
    }

    /// Mixer fader for the rain channel (1 = calibrated default).
    pub fn set_gain(&mut self, g: f32) {
        self.gain.set(g.clamp(0.0, 8.0));
    }

    /// Load the recorded-splat bank (uniform BANK_SLOT slices). Without
    /// it, everything falls back to synthesis. Slices are dulled at load
    /// (one-pole lowpass ~3 kHz): raw drip recordings are bubbly plinks;
    /// rain against a building reads duller.
    pub fn set_bank(&mut self, mut samples: Vec<f32>) {
        let coef = 1.0 - (-core::f32::consts::TAU * 3_000.0 / self.sample_rate).exp();
        for slice in samples.chunks_mut(BANK_SLOT) {
            let mut lp = 0.0f32;
            for x in slice.iter_mut() {
                lp += coef * (*x - lp);
                *x = lp;
            }
        }
        self.bank = samples;
    }

    /// Fresh geometry-priced routing from the simulation: where the sky
    /// is, which panes face the listener, what pours through which
    /// opening. Everything downstream reads this.
    pub fn set_environment(&mut self, env: &Environment) {
        self.routes.update(env);
        self.n_windows = env.windows.len().min(MAX_ENV_WINDOWS);
        for (slot, w) in self.windows.iter_mut().zip(env.windows.iter()) {
            *slot = (encode_gains(w.dir), w.gain);
        }
        self.enclosure.set(env.enclosure);
        self.roof.set(env.roof_gain);
        for (s, &v) in self.seep.iter_mut().zip(env.seep.iter()) {
            s.set(v);
        }
    }

    /// Configure drop `next` as a modal impact on the given material.
    /// `fc_mul` scales the impactor brightness: 1 for a pane radiating
    /// straight at you, well below 1 for knocks arriving through the
    /// structure (the slab lowpasses the drive).
    fn setup_modal(
        d: &mut Drop,
        table: &ModeTable,
        level: f32,
        fc_mul: f32,
        rng: &mut Rng,
        sr: f32,
    ) {
        d.modal = true;
        d.bank_off = usize::MAX;
        d.env = 1.0;
        d.decay = 1.0;
        d.click = 0.0;
        let mut slowest = 0.0f32;
        for (m, &(f, tau, amp)) in table.iter().enumerate() {
            let f = f * (0.97 + 0.06 * rng.next_f32()); // per-drop detune
            let r = (-1.0 / (tau * sr)).exp();
            slowest = slowest.max(r);
            let th = core::f32::consts::TAU * f / sr;
            d.modes[m] = [2.0 * r * th.cos(), -r * r, 0.0, 0.0];
            // (1−r) keeps the resonator's impulse peak ≈ the drive level
            d.mode_gain[m] = amp * (1.0 - r) * level * (0.8 + 0.4 * rng.next_f32());
        }
        // lifetime rides the slowest mode; kill when everything rang out
        d.decay = slowest;
        // soft impactor: small drops excite brighter
        d.exc = 1.0;
        d.exc_lp = 0.0;
        let fc = (1200.0 + 3200.0 * rng.next_f32()) * fc_mul;
        d.exc_coef = 1.0 - (-core::f32::consts::TAU * fc / sr).exp();
    }

    fn spawn_drop(&mut self, enclosure: f32, roof: f32) {
        let d = &mut self.drops[self.next];
        self.next = (self.next + 1) % MAX_DROPS;
        d.phase = 0.0;
        let az = self.rng.next_f32() * core::f32::consts::TAU;
        let n_slices = self.bank.len() / BANK_SLOT;

        // Through-the-opening drops: audible individual drops AT the
        // doorway/window direction, weighted by what pours through —
        // sparse: the fused mass rides the inlet hiss.
        let route_w = self.routes.total_mid();
        if enclosure > 0.3 && self.rng.next_f32() < (route_w * 0.25).min(0.25) {
            let pick = self.rng.next_f32() * route_w;
            let mut acc = 0.0;
            let mut enc = self.routes.slots[0].enc;
            let mut g = 0.0;
            for s in &self.routes.slots {
                acc += s.gains[1].current();
                if pick <= acc {
                    enc = s.enc;
                    g = s.gains[1].current();
                    break;
                }
            }
            d.surface = true; // not shell-attenuated: it comes through the opening
            if n_slices > 0 && self.rng.next_f32() < 0.4 {
                d.bank_off = (self.rng.next_u64() as usize % n_slices) * BANK_SLOT;
                d.bank_pos = 0.0;
                d.bank_rate = 0.85 + self.rng.next_f32() * 0.3;
                d.env = (0.012 + 0.03 * self.rng.next_f32()) * g.min(1.2);
                d.decay = 1.0;
                d.click = 0.0;
            } else {
                d.bank_off = usize::MAX;
                d.modal = false;
                let f = 1400.0 + self.rng.next_f32().powi(2) * 4200.0;
                d.step = core::f32::consts::TAU * f / self.sample_rate;
                d.env = (0.012 + 0.03 * self.rng.next_f32().powi(2)) * g.min(1.2);
                d.click = d.env * 0.4;
                let tau = 0.004 + 0.012 * self.rng.next_f32();
                d.decay = (-1.0 / (tau * self.sample_rate)).exp();
            }
            d.enc = enc;
            return;
        }

        if self.rng.next_f32() < enclosure * 0.9 {
            // Structure impact: WHERE the drop lands decides the sound.
            // Real panes of this room, the roof when the sky is actually
            // overhead, the rest of the shell as a rare dull knock — and
            // a FUSED share that stays silent as a discrete event: only
            // the heavy drops read as ticks, the mass already rides the
            // hiss inlets (a pane is not a hailstorm).
            d.surface = true;
            let wsum: f32 =
                0.35 * self.windows[..self.n_windows].iter().map(|w| w.1).sum::<f32>();
            let total = wsum + roof + 0.15 + 0.45;
            let pick = self.rng.next_f32() * total;
            if pick >= wsum + roof + 0.15 {
                d.env = 0.0; // fused into the noise — no discrete event
                return;
            }
            if pick < wsum {
                // window tick / splat, anchored at the pane's direction
                let mut acc = 0.0;
                let mut wi = 0;
                for (i, w) in self.windows[..self.n_windows].iter().enumerate() {
                    acc += w.1;
                    if pick <= acc {
                        wi = i;
                        break;
                    }
                }
                let (enc, wg) = self.windows[wi];
                // distant panes read quieter; all of it structure-borne
                let lg = (1.6 * wg).min(1.0) * STRUCTURE_LEVEL;
                if n_slices > 0 && self.rng.next_f32() < 0.3 {
                    // a REAL recorded hit, hot and harsh, pitch/gain varied
                    d.bank_off = (self.rng.next_u64() as usize % n_slices) * BANK_SLOT;
                    d.bank_pos = 0.0;
                    d.bank_rate = 0.8 + self.rng.next_f32() * 0.3;
                    d.env = (0.025 + 0.05 * self.rng.next_f32().powi(2)) * lg;
                    d.decay = 1.0;
                    d.click = d.env * 0.25;
                    d.modal = false;
                } else {
                    let level = (0.02 + 0.045 * self.rng.next_f32().powi(2)) * lg;
                    Self::setup_modal(
                        d, &GLASS_MODES, level, 1.0, &mut self.rng, self.sample_rate,
                    );
                }
                d.enc = enc;
            } else if pick < wsum + roof {
                // roof overhead: stone/concrete with the occasional
                // metal ledge/gutter ping
                let (table, level): (&ModeTable, f32) = if self.rng.next_f32() < 0.15 {
                    (&METAL_MODES, 0.012 + 0.03 * self.rng.next_f32().powi(2))
                } else {
                    (&STONE_MODES, 0.025 + 0.05 * self.rng.next_f32().powi(2))
                };
                Self::setup_modal(
                    d,
                    table,
                    level * STRUCTURE_LEVEL,
                    0.35,
                    &mut self.rng,
                    self.sample_rate,
                );
                let el = 1.0 + 0.4 * self.rng.next_f32();
                let (se, ce) = el.min(1.5).sin_cos();
                d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
            } else {
                // shell knock: rain on the walls around you — dull,
                // quiet, structure-borne, near the horizon
                let level = (0.012 + 0.025 * self.rng.next_f32().powi(2)) * STRUCTURE_LEVEL;
                Self::setup_modal(
                    d, &STONE_MODES, level, 0.35, &mut self.rng, self.sample_rate,
                );
                let el = -0.1 + 0.5 * self.rng.next_f32();
                let (se, ce) = el.sin_cos();
                d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
            }
            return;
        }

        // airborne drop outdoors: mostly airy pings around the listener;
        // sometimes a real ground splat from below
        d.surface = false;
        if n_slices > 0 && self.rng.next_f32() < 0.2 {
            d.bank_off = (self.rng.next_u64() as usize % n_slices) * BANK_SLOT;
            d.bank_pos = 0.0;
            d.bank_rate = 0.85 + self.rng.next_f32() * 0.3;
            d.env = 0.01 + 0.02 * self.rng.next_f32().powi(2);
            d.decay = 1.0;
            d.click = 0.0;
            d.modal = false;
            // ground splashes arrive from below-ish around you
            let el = -0.25 + 0.3 * self.rng.next_f32();
            let (se, ce) = el.sin_cos();
            d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
            return;
        }
        d.bank_off = usize::MAX;
        d.modal = false;
        let f = 1400.0 + self.rng.next_f32().powi(2) * 5200.0;
        d.step = core::f32::consts::TAU * f / self.sample_rate;
        d.env = 0.010 + 0.030 * self.rng.next_f32().powi(3);
        d.click = d.env * 0.35;
        let tau = 0.004 + 0.014 * self.rng.next_f32();
        d.decay = (-1.0 / (tau * self.sample_rate)).exp();
        let el = -0.15 + 0.6 * self.rng.next_f32(); // the splash plane
        let (se, ce) = el.sin_cos();
        d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
    }

    /// One sample of rain onto the world-anchored SH bus, from the last
    /// [`set_environment`] state.
    #[inline]
    pub fn process(&mut self, bus: &mut [f32; NCH]) {
        let inten = self.intensity.tick();
        let user = self.gain.tick();
        if inten < 1e-4 && self.hiss_lo.abs() < 1e-9 && self.drops.iter().all(|d| d.env < 1e-6) {
            return; // fully dry and settled — costs nothing
        }

        let enclosure = self.enclosure.tick();
        let roof = self.roof.tick();
        // keep every routed/seep gain converging even while unused, so
        // drop weighting and hiss shaping never read stale values
        let mut route_g = [[0.0f32; 3]; MAX_ENV_ROUTES];
        for (g, s) in route_g.iter_mut().zip(self.routes.slots.iter_mut()) {
            for b in 0..3 {
                g[b] = s.gains[b].tick();
            }
        }
        let mut seep_g = [0.0f32; 3];
        for (g, s) in seep_g.iter_mut().zip(self.seep.iter_mut()) {
            *g = s.tick();
        }

        // Poisson drop scheduling; a roof concentrates the whole surface's
        // impacts over your head, so enclosure raises the audible rate.
        // The curve is steep: drizzle is SPARSE (individual taps with real
        // gaps), and only real rain approaches the fused rate.
        let rate = (RATE_FULL_HZ * inten.powf(2.5) + 6.0 * inten) * (1.0 + 0.8 * enclosure);
        if rate > 0.01 {
            // advance normalized time by the CURRENT rate — the schedule
            // follows ramps instead of freezing a stale interval
            self.spawn_in -= rate / self.sample_rate;
            if self.spawn_in <= 0.0 {
                self.spawn_drop(enclosure, roof);
                let u = self.rng.next_f32().max(1e-6);
                self.spawn_in = -u.ln();
            }
        }

        // drops: airborne pings are shell-attenuated indoors; surface
        // impacts (roof knocks, window ticks) come through at full level
        let direct = 1.0 - 0.85 * enclosure;
        let mut drum_in = 0.0f32;
        for d in &mut self.drops {
            if d.env < 1e-6 {
                continue;
            }
            let tick = if d.click > 1e-6 {
                let w = self.rng.next_f32() * 2.0 - 1.0;
                let c = w * d.click;
                d.click *= 0.55; // ~0.1 ms burst
                c
            } else {
                0.0
            };
            let s = if d.modal {
                // excitation: soft noise burst (~1 ms) through the
                // impactor lowpass, driving the material's modes
                let x_in = if d.exc > 1e-3 {
                    let w = self.rng.next_f32() * 2.0 - 1.0;
                    d.exc *= 0.982; // ~1 ms burst
                    d.exc_lp += d.exc_coef * (w * d.exc - d.exc_lp);
                    d.exc_lp
                } else {
                    0.0
                };
                let mut out = 0.0f32;
                for (m, st) in d.modes.iter_mut().enumerate() {
                    let y = st[0] * st[2] + st[1] * st[3] + d.mode_gain[m] * x_in;
                    st[3] = st[2];
                    st[2] = y;
                    out += y;
                }
                d.env *= d.decay;
                out
            } else if d.bank_off != usize::MAX {
                // recorded hit: linear-interp read at the drop's rate
                let i = d.bank_pos as usize;
                if i + 1 >= BANK_SLOT {
                    d.env = 0.0;
                    continue;
                }
                let fr = d.bank_pos - i as f32;
                let a = self.bank[d.bank_off + i];
                let b = self.bank[d.bank_off + i + 1];
                d.bank_pos += d.bank_rate;
                (a + fr * (b - a)) * d.env + tick
            } else {
                let v = d.phase.sin() * d.env + tick;
                d.phase += d.step;
                d.env *= d.decay;
                v
            };
            drum_in += s;
            let g = s * if d.surface { 1.0 } else { direct } * MASTER * user;
            for k in 0..NCH {
                bus[k] += g * d.enc[k];
            }
        }

        // roof drumming: blurred structure-borne bed under the discrete
        // taps — only where the sky is genuinely overhead (a storey above
        // you silences it; outdoors there is no roof at all)
        self.drum_lp += 0.04 * (drum_in - self.drum_lp);
        let drum_g = self.drum_lp * roof * enclosure * 1.2 * MASTER * user;
        for k in 0..NCH {
            bus[k] += drum_g * self.up_enc[k];
        }

        // downpour noise: band-shaped noise — what the drops fuse into.
        // It has no business at drizzle: below ~0.3 intensity there is
        // none; it grows steeply after.
        let w = self.rng.next_f32() * 2.0 - 1.0;
        self.hiss_hi += 0.55 * (w - self.hiss_hi); // ~6 kHz pole
        self.hiss_lo += 0.05 * (self.hiss_hi - self.hiss_lo);
        let raw = self.hiss_hi - self.hiss_lo; // bandpassed
        let hiss_amt = ((inten - 0.28) / 0.72).clamp(0.0, 1.0).powf(1.6);
        if hiss_amt > 0.0 {
            let base = hiss_amt * MASTER * user;
            // outdoors: the noise IS the wet ground around you — a low
            // splash field, not an overhead wash
            let open_g = raw * base * 0.5 * (1.0 - enclosure);
            if open_g.abs() > 1e-9 {
                for k in 0..NCH {
                    bus[k] += open_g * self.ground_enc[k];
                }
            }
            // through each inlet: the outdoor splash field pours in at
            // the opening's direction, band-shaped by its geometry price
            if enclosure > 0.05 {
                for (i, s) in self.routes.slots.iter_mut().enumerate() {
                    let g = &route_g[i];
                    if g[0] + g[1] + g[2] < 1e-6 {
                        continue;
                    }
                    // alternate polarity per inlet: each opening passes a
                    // DIFFERENT slice of the splash field, so inlets must
                    // sum in power, not amplitude
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let shaped = s.split.process(
                        raw * sign,
                        &[
                            g[0] * base * ROUTE_HISS * enclosure,
                            g[1] * base * ROUTE_HISS * enclosure,
                            g[2] * base * ROUTE_HISS * enclosure,
                        ],
                    );
                    for k in 0..NCH {
                        bus[k] += shaped * s.enc[k];
                    }
                }
                // shell seep: the diffuse residual inside the room —
                // dark, directionless
                if seep_g[0] + seep_g[1] + seep_g[2] > 1e-6 {
                    let w2 = self.rng.next_f32() * 2.0 - 1.0;
                    self.seep_hi += 0.55 * (w2 - self.seep_hi);
                    self.seep_lo += 0.05 * (self.seep_hi - self.seep_lo);
                    let shaped = self.seep_split.process(
                        self.seep_hi - self.seep_lo,
                        &[
                            seep_g[0] * base * SEEP_HISS * enclosure,
                            seep_g[1] * base * SEEP_HISS * enclosure,
                            seep_g[2] * base * SEEP_HISS * enclosure,
                        ],
                    );
                    bus[0] += shaped;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{EnvRoute, EnvWindow};

    fn outdoor_env() -> Environment {
        let dirs = [[0.0f32, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]];
        Environment {
            seep: [0.0; 3],
            enclosure: 0.0,
            roof_gain: 0.0,
            routes: dirs
                .iter()
                .enumerate()
                .map(|(k, d)| EnvRoute {
                    id: 100 + k as u32,
                    dir: *d,
                    gains: [1.0, 1.0, 1.0],
                    dist: 35.0,
                })
                .collect(),
            windows: vec![],
        }
    }

    fn indoor_env() -> Environment {
        Environment {
            seep: [0.18, 0.09, 0.03],
            enclosure: 0.9,
            roof_gain: 1.0,
            routes: vec![],
            windows: vec![EnvWindow { dir: [1.0, 0.0, 0.0], gain: 0.5 }],
        }
    }

    fn energy_over(r: &mut Rain, n: usize) -> f32 {
        let mut e = 0.0f64;
        for _ in 0..n {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus);
            e += (bus[0] * bus[0]) as f64;
        }
        (e / n as f64) as f32
    }

    fn settle(r: &mut Rain, n: usize) {
        for _ in 0..n {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus);
        }
    }

    /// The material tables must produce genuinely distinct impacts:
    /// glass rings brighter than stone, metal rings longer than glass.
    #[test]
    fn materials_have_distinct_signatures() {
        let render = |table: &ModeTable| -> Vec<f32> {
            let sr = 48_000.0;
            // same recurrence as the modal path
            let mut modes = [[0.0f32; 4]; 4];
            let mut gains = [0.0f32; 4];
            for (m, &(f, tau, amp)) in table.iter().enumerate() {
                let r = (-1.0 / (tau * sr)).exp();
                let th = core::f32::consts::TAU * f / sr;
                modes[m] = [2.0 * r * th.cos(), -r * r, 0.0, 0.0];
                gains[m] = amp * (1.0 - r) * 0.05;
            }
            let mut exc = 1.0f32;
            let mut lp = 0.0f32;
            let mut rng = Rng::new(9);
            (0..9600)
                .map(|_| {
                    let x = if exc > 1e-3 {
                        let w = rng.next_f32() * 2.0 - 1.0;
                        exc *= 0.982;
                        lp += 0.3 * (w * exc - lp);
                        lp
                    } else {
                        0.0
                    };
                    let mut out = 0.0;
                    for (m, st) in modes.iter_mut().enumerate() {
                        let y = st[0] * st[2] + st[1] * st[3] + gains[m] * x;
                        st[3] = st[2];
                        st[2] = y;
                        out += y;
                    }
                    out
                })
                .collect()
        };
        let zcr = |x: &[f32]| -> usize {
            x.windows(2).filter(|w| w[0].signum() != w[1].signum()).count()
        };
        let late_e = |x: &[f32]| -> f32 {
            x[2880..].iter().map(|v| v * v).sum() // after 60 ms
        };
        let (g, m, st) = (render(&GLASS_MODES), render(&METAL_MODES), render(&STONE_MODES));
        assert!(zcr(&g) > 2 * zcr(&st), "glass must ring brighter than stone: {} vs {}", zcr(&g), zcr(&st));
        assert!(late_e(&m) > 4.0 * late_e(&g), "metal must ring longer than glass");
        assert!(late_e(&st) < 0.05 * late_e(&m), "stone must be dead by 60 ms");
    }

    #[test]
    fn dry_is_silent_and_free() {
        let mut r = Rain::new(48_000.0);
        r.set_environment(&outdoor_env());
        assert_eq!(energy_over(&mut r, 48_000), 0.0);
    }

    #[test]
    fn rain_ramps_in_and_back_out() {
        let mut r = Rain::new(48_000.0);
        r.set_environment(&outdoor_env());
        r.set_intensity(1.0);
        let early = energy_over(&mut r, 24_000); // first 0.5 s of ramp
        settle(&mut r, 48_000 * 12);
        let full = energy_over(&mut r, 48_000);
        assert!(full > 4.0 * early, "should swell in: early {early} full {full}");
        r.set_intensity(0.0);
        settle(&mut r, 48_000 * 30);
        let after = energy_over(&mut r, 48_000);
        assert!(after < full * 1e-3, "should die away: {after} vs {full}");
    }

    /// Indoors, rain is TAPPY: discrete surface impacts give a much
    /// higher crest factor than the fused outdoor wash.
    #[test]
    fn indoors_is_tappier() {
        let crest = |env: &Environment| -> f32 {
            let mut r = Rain::new(48_000.0);
            r.set_environment(env);
            r.set_intensity(0.5); // moderate rain: events stay discrete
            settle(&mut r, 48_000 * 12);
            let (mut peak, mut e) = (0.0f32, 0.0f64);
            for _ in 0..48_000 * 2 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus);
                peak = peak.max(bus[0].abs());
                e += (bus[0] * bus[0]) as f64;
            }
            peak / ((e / (48_000.0 * 2.0)) as f32).sqrt().max(1e-9)
        };
        let (out_c, in_c) = (crest(&outdoor_env()), crest(&indoor_env()));
        assert!(in_c > 1.3 * out_c, "indoor rain should be tappier: crest {in_c} vs {out_c}");
    }

    /// Drizzle must be taps with real gaps, not a hiss bed: high crest
    /// factor and a near-silent floor between events.
    #[test]
    fn drizzle_is_sparse_taps_not_hiss() {
        let mut r = Rain::new(48_000.0);
        r.set_environment(&outdoor_env());
        r.set_intensity(0.3);
        settle(&mut r, 48_000 * 12);
        let mut peak = 0.0f32;
        let mut e = 0.0f64;
        let mut quiet = 0u32;
        const N: usize = 48_000 * 2;
        for _ in 0..N {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus);
            peak = peak.max(bus[0].abs());
            e += (bus[0] * bus[0]) as f64;
            if bus[0].abs() < 1e-4 {
                quiet += 1;
            }
        }
        let crest = peak / ((e / N as f64) as f32).sqrt().max(1e-9);
        assert!(crest > 6.0, "drizzle should be spiky events: crest {crest}");
        assert!(
            quiet as f32 / N as f32 > 0.4,
            "drizzle needs real gaps between taps: quiet fraction {}",
            quiet as f32 / N as f32
        );
    }

    /// The shell claim: with no bright pane ticks in the mix (a
    /// windowless room), what remains indoors — knocks + seep — must be
    /// both quieter and darker than standing in the open rain. (Window
    /// ticks themselves are deliberately bright; they are tested by ear.)
    #[test]
    fn indoors_is_quieter_and_darker() {
        let probe = |env: &Environment| -> (f32, f32) {
            let mut r = Rain::new(48_000.0);
            r.set_environment(env);
            r.set_intensity(1.0);
            settle(&mut r, 48_000 * 12);
            // total energy + crude HF-ness (first difference energy)
            let (mut e, mut hf) = (0.0f64, 0.0f64);
            let mut prev = 0.0f32;
            for _ in 0..48_000 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus);
                e += (bus[0] * bus[0]) as f64;
                hf += ((bus[0] - prev) * (bus[0] - prev)) as f64;
                prev = bus[0];
            }
            (e as f32, (hf / e.max(1e-12)) as f32)
        };
        let (out_e, out_hf) = probe(&outdoor_env());
        let mut env = indoor_env();
        env.windows.clear();
        env.enclosure = 1.0; // fully under the shell
        let (in_e, in_hf) = probe(&env);
        assert!(in_e < 0.6 * out_e, "indoors should be quieter: {in_e} vs {out_e}");
        assert!(in_hf < 0.8 * out_hf, "indoors should be darker: {in_hf} vs {out_hf}");
    }

    /// The roof only drums when the sky is genuinely overhead: under
    /// another storey the same enclosure keeps its taps, but the
    /// overhead layer (roof knocks + drumming, both encoded upward)
    /// disappears. Measured on the Z channel, where "up" lives.
    #[test]
    fn no_sky_overhead_no_drumming() {
        let up_energy = |roof: f32| -> f32 {
            let mut env = indoor_env();
            env.roof_gain = roof;
            env.windows.clear(); // isolate the overhead layer
            env.enclosure = 1.0; // fully under the shell — no ring leak
            let mut r = Rain::new(48_000.0);
            r.set_environment(&env);
            r.set_intensity(0.45); // drop-dominated: below the hiss knee
            settle(&mut r, 48_000 * 12);
            let mut e = 0.0f64;
            for _ in 0..48_000 * 4 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus);
                e += (bus[2] * bus[2]) as f64; // ACN 2 = Z (up)
            }
            e as f32
        };
        let (roofed, storeyed) = (up_energy(1.0), up_energy(0.0));
        assert!(
            roofed > 2.0 * storeyed,
            "sky overhead must put the rain over your head: {roofed} vs {storeyed}"
        );
    }
}
