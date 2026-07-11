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

const MAX_DROPS: usize = 24;
/// Poisson rate at full intensity (drops audible as events; beyond this
/// they'd fuse anyway, so the hiss carries the rest).
const RATE_FULL_HZ: f32 = 260.0;
/// Overall level trim (calibrated against the night-city ambience bed).
const MASTER: f32 = 0.22;

struct Drop {
    phase: f32,
    step: f32,   // radians/sample
    env: f32,    // exponential amplitude
    decay: f32,  // per-sample multiplier
    enc: [f32; NCH],
}

pub struct Rain {
    intensity: Smoothed,
    /// User mixer gain (smoothed), on top of the weather intensity.
    gain: Smoothed,
    rng: Rng,
    drops: [Drop; MAX_DROPS],
    next: usize,
    spawn_in: f32, // samples until next drop
    // downpour hiss shaping: noise → band between the two poles
    hiss_lo: f32,
    hiss_hi: f32,
    // roof drumming (indoors): heavy lowpass on its own drop layer
    drum_lp: f32,
    // shell muffle state for the hiss (one-pole, coef from the room)
    muff_lp: f32,
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
                enc: [0.0; NCH],
            }),
            next: 0,
            spawn_in: f32::MAX,
            hiss_lo: 0.0,
            hiss_hi: 0.0,
            drum_lp: 0.0,
            muff_lp: 0.0,
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

    fn spawn_drop(&mut self, enclosure: f32) {
        let d = &mut self.drops[self.next];
        self.next = (self.next + 1) % MAX_DROPS;
        // pitch: small drops ping high; occasional fat low splats
        let f = 1400.0 + self.rng.next_f32().powi(2) * 5200.0;
        d.step = core::f32::consts::TAU * f / self.sample_rate;
        d.phase = 0.0;
        d.env = 0.010 + 0.030 * self.rng.next_f32().powi(3);
        // 4–18 ms decay
        let tau = 0.004 + 0.014 * self.rng.next_f32();
        d.decay = (-1.0 / (tau * self.sample_rate)).exp();
        // direction: random azimuth, elevated — rain comes from above;
        // indoors the cone narrows toward straight overhead (the roof)
        let az = self.rng.next_f32() * core::f32::consts::TAU;
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
        if inten < 1e-4 && self.spawn_in == f32::MAX && self.hiss_lo.abs() < 1e-9 {
            return; // fully dry and settled — costs nothing
        }

        // Poisson drop scheduling
        let rate = RATE_FULL_HZ * inten * inten + 14.0 * inten;
        if rate > 0.01 {
            self.spawn_in -= 1.0;
            if self.spawn_in <= 0.0 || self.spawn_in == f32::MAX - 1.0 {
                self.spawn_drop(enclosure);
                // exponential inter-arrival times
                let u = self.rng.next_f32().max(1e-6);
                self.spawn_in = -u.ln() * self.sample_rate / rate;
            }
        } else {
            self.spawn_in = f32::MAX;
        }

        // drop pings (direct, dulled by the shell when indoors)
        let direct = 1.0 - 0.85 * enclosure;
        let mut drum_in = 0.0f32;
        for d in &mut self.drops {
            if d.env < 1e-6 {
                continue;
            }
            let s = d.phase.sin() * d.env;
            d.phase += d.step;
            d.env *= d.decay;
            drum_in += s;
            let g = s * direct * MASTER * user;
            for k in 0..NCH {
                bus[k] += g * d.enc[k];
            }
        }

        // roof drumming: the same drop energy, structure-borne — heavy
        // lowpass, from straight above, only meaningful indoors
        self.drum_lp += 0.04 * (drum_in - self.drum_lp);
        let drum_g = self.drum_lp * enclosure * 2.4 * MASTER * user;
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
        let hg = hiss * inten.powf(1.8) * 0.5 * MASTER * user * (1.0 - 0.6 * enclosure);
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
