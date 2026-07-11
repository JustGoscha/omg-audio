//! Rain, synthesized from its statistics — the same philosophy as the late
//! reverb field: individually audible events rendered exactly, the dense
//! remainder as a texture with the right aggregate properties.
//!
//!  - Drops: a Poisson process whose rate follows intensity. Each drop is a
//!    short decaying ping (random pitch/level), spatialized on the
//!    world-anchored SH bus from an elevated random direction.
//!  - Downpour: band-shaped noise whose level grows super-linearly with
//!    intensity — at high rates the drops fuse into hiss, exactly like the
//!    reflection tail fuses into reverb.
//!  - Enclosure: indoors the direct hiss dulls through the building
//!    (shares the ambience lowpass) and a low roof-drumming layer fades in
//!    from straight above.
//!
//! Intensity is slew-limited (~6 s), so rain starts and stops like
//! weather, not like a fader.

use crate::ambi::{encode_gains, NCH};
use crate::smooth::Smoothed;
use omg_core::rng::Rng;

const MAX_DROPS: usize = 32;
/// Poisson rate at full intensity (drops audible as events; beyond this
/// they'd fuse anyway, so the hiss carries the rest).
const RATE_FULL_HZ: f32 = 260.0;
/// Overall level trim (calibrated against the night-city ambience bed).
const MASTER: f32 = 0.22;
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

#[derive(Clone, Copy)]
struct RainRoute {
    enc: [f32; NCH],
    gain: f32,
    lp_coef: f32,
    lp: f32,
}

impl Default for RainRoute {
    fn default() -> Self {
        Self { enc: [0.0; NCH], gain: 0.0, lp_coef: 1.0, lp: 0.0 }
    }
}

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
    // roof drumming (indoors): heavy lowpass on its own drop layer
    drum_lp: f32,
    // shell muffle state for the hiss (one-pole, coef from the room)
    muff_lp: f32,
    /// Bank of real recorded drop/splat hits (uniform BANK_SLOT slices).
    bank: Vec<f32>,
    /// Aperture streams: rain heard THROUGH openings of the listener's
    /// room (doorways, windows) — direction-encoded, gained and filtered
    /// by what fills the opening (nothing / glass / a closed door panel).
    routes: [RainRoute; 4],
    sample_rate: f32,
    up_enc: [f32; NCH],
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
            drum_lp: 0.0,
            muff_lp: 0.0,
            bank: Vec::new(),
            routes: [RainRoute::default(); 4],
            sample_rate,
            up_enc: encode_gains([0.0, 0.0, 1.0]),
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

    /// Aperture routes from the simulation: up to 4 × [dir_x, dir_y,
    /// gain, lp_coef]. gain 0 disables a slot. World-frame directions —
    /// the SH bus rotates with the head downstream.
    pub fn set_routes(&mut self, flat: &[f32]) {
        for (i, r) in self.routes.iter_mut().enumerate() {
            let o = i * 4;
            if o + 4 > flat.len() || flat[o + 2] <= 1e-4 {
                r.gain = 0.0;
                continue;
            }
            r.enc = encode_gains([flat[o], flat[o + 1], 0.0]);
            r.gain = flat[o + 2];
            r.lp_coef = flat[o + 3].clamp(0.001, 1.0);
        }
    }

    /// Configure drop `next` as a modal impact on the given material.
    fn setup_modal(d: &mut Drop, table: &ModeTable, level: f32, rng: &mut Rng, sr: f32) {
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
        let fc = 1200.0 + 3200.0 * rng.next_f32();
        d.exc_coef = 1.0 - (-core::f32::consts::TAU * fc / sr).exp();
    }

    fn spawn_drop(&mut self, enclosure: f32) {
        let d = &mut self.drops[self.next];
        self.next = (self.next + 1) % MAX_DROPS;
        d.phase = 0.0;
        let az = self.rng.next_f32() * core::f32::consts::TAU;

        // Enclosed: most drops become discrete surface impacts — roof
        // knocks from straight overhead, brighter glass ticks from the
        // sides. That's the tappy part of rain heard from inside.
        let n_slices = self.bank.len() / BANK_SLOT;
        // Through-the-opening drops: audible individual drops AT the
        // doorway/window direction, weighted by how open it is.
        let route_w: f32 = self.routes.iter().map(|r| r.gain).sum();
        if enclosure > 0.3 && self.rng.next_f32() < (route_w * 0.5).min(0.45) {
            let pick = self.rng.next_f32() * route_w;
            let mut acc = 0.0;
            let mut enc = self.routes[0].enc;
            let mut g = 0.0;
            for r in &self.routes {
                acc += r.gain;
                if pick <= acc {
                    enc = r.enc;
                    g = r.gain;
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
            d.surface = true;
            let glass = self.rng.next_f32() < 0.4;
            if glass && n_slices > 0 && self.rng.next_f32() < 0.3 {
                // window splat: a REAL recorded hit, hot and harsh, with
                // pitch/gain variation so no two reads alike
                d.bank_off = (self.rng.next_u64() as usize % n_slices) * BANK_SLOT;
                d.bank_pos = 0.0;
                // narrow spread, biased slightly down: dull, not bubbly
                d.bank_rate = 0.8 + self.rng.next_f32() * 0.3;
                d.env = 0.025 + 0.05 * self.rng.next_f32().powi(2);
                d.decay = 1.0;
                d.click = d.env * 0.25;
                let el = 0.15 + 0.35 * self.rng.next_f32();
                let (se, ce) = el.sin_cos();
                d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
                return;
            }
            // modal impact: the surface decides the sound. Windows are
            // glass; roofs here are stone/concrete with the occasional
            // metal ledge/gutter ping.
            let (table, level, el): (&ModeTable, f32, f32) = if glass {
                (&GLASS_MODES, 0.02 + 0.045 * self.rng.next_f32().powi(2), 0.15 + 0.35 * self.rng.next_f32())
            } else if self.rng.next_f32() < 0.15 {
                (&METAL_MODES, 0.012 + 0.03 * self.rng.next_f32().powi(2), 1.0 + 0.4 * self.rng.next_f32())
            } else {
                (&STONE_MODES, 0.025 + 0.05 * self.rng.next_f32().powi(2), 1.1 + 0.4 * self.rng.next_f32())
            };
            Self::setup_modal(d, table, level, &mut self.rng, self.sample_rate);
            let (se, ce) = el.min(1.5).sin_cos();
            d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
            return;
        }

        // airborne drop: mostly airy pings; sometimes a real ground splat
        d.surface = false;
        if n_slices > 0 && self.rng.next_f32() < 0.2 {
            d.bank_off = (self.rng.next_u64() as usize % n_slices) * BANK_SLOT;
            d.bank_pos = 0.0;
            d.bank_rate = 0.85 + self.rng.next_f32() * 0.3;
            d.env = 0.01 + 0.02 * self.rng.next_f32().powi(2);
            d.decay = 1.0;
            d.click = 0.0;
            // ground splashes arrive from below-ish around you
            let el = -0.25 + 0.3 * self.rng.next_f32();
            let (se, ce) = el.sin_cos();
            d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
            return;
        }
        d.bank_off = usize::MAX;
        let f = 1400.0 + self.rng.next_f32().powi(2) * 5200.0;
        d.step = core::f32::consts::TAU * f / self.sample_rate;
        d.env = 0.010 + 0.030 * self.rng.next_f32().powi(3);
        d.click = d.env * 0.35;
        let tau = 0.004 + 0.014 * self.rng.next_f32();
        d.decay = (-1.0 / (tau * self.sample_rate)).exp();
        let el = 0.35 + 0.5 * self.rng.next_f32() + 0.6 * enclosure;
        let (se, ce) = el.min(1.5).sin_cos();
        d.enc = encode_gains([az.cos() * ce, az.sin() * ce, se]);
    }

    /// One sample of rain onto the world-anchored SH bus.
    /// `enclosure`: 0 = open sky … 1 = deep inside a building.
    /// `muffle_coef`: one-pole coefficient of the room's air/shell filter
    /// (shared with the ambience bed).
    #[inline]
    pub fn process(&mut self, bus: &mut [f32; NCH], enclosure: f32, muffle_coef: f32) {
        let inten = self.intensity.tick();
        let user = self.gain.tick();
        if inten < 1e-4 && self.hiss_lo.abs() < 1e-9 && self.drops.iter().all(|d| d.env < 1e-6) {
            return; // fully dry and settled — costs nothing
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
                self.spawn_drop(enclosure);
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
        // taps — reduced now that impacts carry the presence
        self.drum_lp += 0.04 * (drum_in - self.drum_lp);
        let drum_g = self.drum_lp * enclosure * 1.2 * MASTER * user;
        for k in 0..NCH {
            bus[k] += drum_g * self.up_enc[k];
        }

        // downpour hiss: band-shaped noise, level ∝ intensity^1.8,
        // dulled indoors by the shared shell filter
        let w = self.rng.next_f32() * 2.0 - 1.0;
        self.hiss_hi += 0.55 * (w - self.hiss_hi); // ~6 kHz pole
        self.hiss_lo += 0.05 * (self.hiss_hi - self.hiss_lo);
        let raw = self.hiss_hi - self.hiss_lo; // bandpassed
        // shell muffle: the room's one-pole, blended in by enclosure —
        // outdoors dry, indoors properly darkened, not just quieter
        self.muff_lp += muffle_coef.clamp(0.0, 1.0) * (raw - self.muff_lp);
        let hiss = raw + enclosure * (self.muff_lp - raw);
        // Hiss is what drops fuse into — it has no business at drizzle.
        // Below ~0.3 intensity there is none; it grows steeply after.
        let hiss_amt = ((inten - 0.28) / 0.72).clamp(0.0, 1.0).powf(1.6);
        // Aperture streams: the outdoor rain heard THROUGH each opening,
        // localized at its direction, filtered by what fills it. This is
        // what makes an open door audibly pour and a closed one seal.
        if enclosure > 0.05 && hiss_amt > 0.0 {
            for r in &mut self.routes {
                if r.gain <= 1e-4 {
                    continue;
                }
                r.lp += r.lp_coef * (raw - r.lp);
                let g = r.lp * hiss_amt * 0.55 * MASTER * user * r.gain * enclosure;
                for k in 0..NCH {
                    bus[k] += g * r.enc[k];
                }
            }
        }
        let hg = hiss * hiss_amt * 0.5 * MASTER * user * (1.0 - 0.6 * enclosure);
        // diffuse: omni + a touch of overhead bias
        bus[0] += hg;
        for k in 0..NCH {
            bus[k] += 0.4 * hg * self.up_enc[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy_over(r: &mut Rain, n: usize, enclosure: f32) -> f32 {
        let mut e = 0.0f64;
        for _ in 0..n {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus, enclosure, 1.0);
            e += (bus[0] * bus[0]) as f64;
        }
        (e / n as f64) as f32
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
        assert_eq!(energy_over(&mut r, 48_000, 0.0), 0.0);
    }

    #[test]
    fn rain_ramps_in_and_back_out() {
        let mut r = Rain::new(48_000.0);
        r.set_intensity(1.0);
        let early = energy_over(&mut r, 24_000, 0.0); // first 0.5 s of ramp
        for _ in 0..48_000 * 12 {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus, 0.0, 1.0);
        }
        let full = energy_over(&mut r, 48_000, 0.0);
        assert!(full > 4.0 * early, "should swell in: early {early} full {full}");
        r.set_intensity(0.0);
        for _ in 0..48_000 * 30 {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus, 0.0, 1.0);
        }
        let after = energy_over(&mut r, 48_000, 0.0);
        assert!(after < full * 1e-3, "should die away: {after} vs {full}");
    }

    /// Indoors, rain is TAPPY: discrete surface impacts give a much
    /// higher crest factor than the fused outdoor wash.
    #[test]
    fn indoors_is_tappier() {
        let crest = |enclosure: f32| -> f32 {
            let mut r = Rain::new(48_000.0);
            r.set_intensity(0.5); // moderate rain: events stay discrete
            for _ in 0..48_000 * 12 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus, enclosure, 0.05);
            }
            let (mut peak, mut e) = (0.0f32, 0.0f64);
            for _ in 0..48_000 * 2 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus, enclosure, 0.05);
                peak = peak.max(bus[0].abs());
                e += (bus[0] * bus[0]) as f64;
            }
            peak / ((e / (48_000.0 * 2.0)) as f32).sqrt().max(1e-9)
        };
        let (out_c, in_c) = (crest(0.0), crest(1.0));
        assert!(in_c > 1.3 * out_c, "indoor rain should be tappier: crest {in_c} vs {out_c}");
    }

    /// Drizzle must be taps with real gaps, not a hiss bed: high crest
    /// factor and a near-silent floor between events.
    #[test]
    fn drizzle_is_sparse_taps_not_hiss() {
        let mut r = Rain::new(48_000.0);
        r.set_intensity(0.3);
        for _ in 0..48_000 * 12 {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus, 0.0, 1.0);
        }
        let mut peak = 0.0f32;
        let mut e = 0.0f64;
        let mut quiet = 0u32;
        const N: usize = 48_000 * 2;
        for _ in 0..N {
            let mut bus = [0.0f32; NCH];
            r.process(&mut bus, 0.0, 1.0);
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

    #[test]
    fn indoors_is_quieter_and_darker() {
        let settle = |enclosure: f32| -> (f32, f32) {
            let mut r = Rain::new(48_000.0);
            r.set_intensity(1.0);
            for _ in 0..48_000 * 12 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus, enclosure, 0.05);
            }
            // total energy + crude HF-ness (first difference energy)
            let (mut e, mut hf) = (0.0f64, 0.0f64);
            let mut prev = 0.0f32;
            for _ in 0..48_000 {
                let mut bus = [0.0f32; NCH];
                r.process(&mut bus, enclosure, 0.05);
                e += (bus[0] * bus[0]) as f64;
                hf += ((bus[0] - prev) * (bus[0] - prev)) as f64;
                prev = bus[0];
            }
            (e as f32, (hf / e.max(1e-12)) as f32)
        };
        let (out_e, out_hf) = settle(0.0);
        let (in_e, in_hf) = settle(1.0);
        assert!(in_e < 0.6 * out_e, "indoors should be quieter: {in_e} vs {out_e}");
        assert!(in_hf < 0.8 * out_hf, "indoors should be darker: {in_hf} vs {out_hf}");
    }
}
