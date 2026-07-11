//! Ambience: the outdoor loop (city hum, night nature) rendered through
//! the environment's inlets instead of as a listener-anchored bed.
//!
//!  - Each environment route (an aperture of the listener's room, or a
//!    horizon sector outdoors) plays a decorrelated read of the loop,
//!    band-shaped by the route's geometry-priced gains and encoded at its
//!    world direction on the SH bus.
//!  - The diffuse residual (the field that built up INSIDE the room via
//!    its shell) plays as four decorrelated world-anchored feeds, shaped
//!    by the power-balance seep spectrum.
//!
//! Room transitions, door swings and occlusion all arrive here already
//! priced by the simulation; per-slot smoothing only irons the 20 Hz
//! tick quantization.

use crate::ambi::{encode_gains, NCH};
use crate::bands::BandSplit;
use crate::env::{Environment, RouteSlots, MAX_ENV_ROUTES};
use crate::smooth::Smoothed;

/// Level of a unit-gain inlet. Calibrated so the open sky (four unit
/// horizon sectors) sits at the loudness the old bed had outdoors.
const ROUTE_LEVEL: f32 = 0.047;
/// The diffuse residual, spread over four feeds.
const SEEP_LEVEL: f32 = 0.047;

pub struct Ambience {
    data: Vec<f32>,
    stereo: bool,
    pos: usize,
    user: Smoothed,
    routes: RouteSlots,
    seep: [Smoothed; 3],
    seep_split: [BandSplit; 4],
    seep_enc: [[f32; NCH]; 4],
}

impl Ambience {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            data: Vec::new(),
            stereo: false,
            pos: 0,
            user: Smoothed::new(1.0, 0.02, sample_rate),
            routes: RouteSlots::new(sample_rate, 0.25),
            seep: core::array::from_fn(|_| Smoothed::new(0.0, 0.4, sample_rate)),
            seep_split: core::array::from_fn(|_| BandSplit::new(sample_rate)),
            // world-anchored diffuse feed directions (N/E/S/W): the SH bus
            // counter-rotates with head turns downstream
            seep_enc: [
                encode_gains([0.0, 1.0, 0.0]),
                encode_gains([1.0, 0.0, 0.0]),
                encode_gains([0.0, -1.0, 0.0]),
                encode_gains([-1.0, 0.0, 0.0]),
            ],
        }
    }

    /// Install the loop (mono, or interleaved stereo).
    pub fn set_loop(&mut self, data: Vec<f32>, stereo: bool) {
        self.data = data;
        self.stereo = stereo;
        self.pos = 0;
    }

    /// Mixer fader for the ambience channel.
    pub fn set_user(&mut self, g: f32) {
        self.user.set(g.clamp(0.0, 8.0));
    }

    pub fn set_environment(&mut self, env: &Environment) {
        self.routes.update(env);
        for (s, &v) in self.seep.iter_mut().zip(env.seep.iter()) {
            s.set(v);
        }
    }

    /// Decorrelated loop read: feed `k` reads its own offset into the
    /// loop; stereo material additionally alternates channels.
    #[inline]
    fn read(&self, k: usize, frames: usize) -> f32 {
        let f = (self.pos + (k + 1) * frames / (MAX_ENV_ROUTES + 6)) % frames;
        if self.stereo {
            self.data[f * 2 + (k % 2)]
        } else {
            self.data[f]
        }
    }

    /// One sample onto the world-anchored SH bus.
    #[inline]
    pub fn process(&mut self, bus: &mut [f32; NCH]) {
        if self.data.is_empty() {
            return;
        }
        let frames = if self.stereo { self.data.len() / 2 } else { self.data.len() };
        self.pos = (self.pos + 1) % frames;
        let user = self.user.tick();

        for i in 0..MAX_ENV_ROUTES {
            let live = {
                let s = &mut self.routes.slots[i];
                let mut gains = [0.0f32; 3];
                let mut any = false;
                for (g, sm) in gains.iter_mut().zip(s.gains.iter_mut()) {
                    *g = sm.tick() * ROUTE_LEVEL * user;
                    any |= *g > 1e-7;
                }
                if any { Some(gains) } else { None }
            };
            if let Some(gains) = live {
                let x = self.read(i, frames);
                let s = &mut self.routes.slots[i];
                let y = s.split.process(x, &gains);
                for k in 0..NCH {
                    bus[k] += y * s.enc[k];
                }
            }
        }

        let mut seep_g = [0.0f32; 3];
        let mut seep_any = false;
        for (g, sm) in seep_g.iter_mut().zip(self.seep.iter_mut()) {
            *g = sm.tick() * SEEP_LEVEL * user;
            seep_any |= *g > 1e-7;
        }
        if seep_any {
            for d in 0..4 {
                let x = self.read(MAX_ENV_ROUTES + d, frames);
                let y = self.seep_split[d].process(x, &seep_g);
                for k in 0..NCH {
                    bus[k] += y * self.seep_enc[d][k];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::EnvRoute;

    fn noise_loop(n: usize) -> Vec<f32> {
        let mut rng = omg_core::rng::Rng::new(7);
        (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    fn energy(a: &mut Ambience, n: usize) -> f32 {
        let mut e = 0.0f64;
        for _ in 0..n {
            let mut bus = [0.0f32; NCH];
            a.process(&mut bus);
            e += (bus[0] * bus[0]) as f64;
        }
        (e / n as f64) as f32
    }

    fn env_with(routes: Vec<EnvRoute>, seep: [f32; 3]) -> Environment {
        Environment { seep, enclosure: 0.0, roof_gain: 0.0, routes, windows: vec![] }
    }

    #[test]
    fn no_environment_is_silent() {
        let mut a = Ambience::new(48_000.0);
        a.set_loop(noise_loop(48_000), false);
        assert_eq!(energy(&mut a, 4800), 0.0);
    }

    /// Swapping the entire route set (a hard room change, worst case)
    /// must fade, not click: no sample step larger than the signal scale.
    #[test]
    fn route_swaps_never_click() {
        let mut a = Ambience::new(48_000.0);
        a.set_loop(noise_loop(48_000), false);
        let r = |id, gains| EnvRoute { id, dir: [1.0, 0.0, 0.0], gains, dist: 2.0 };
        a.set_environment(&env_with(vec![r(0, [1.0, 1.0, 1.0])], [0.0; 3]));
        for _ in 0..48_000 {
            let mut bus = [0.0f32; NCH];
            a.process(&mut bus);
        }
        // swap to a disjoint id set
        a.set_environment(&env_with(vec![r(5, [1.0, 1.0, 1.0])], [0.0; 3]));
        let mut prev = 0.0f32;
        let mut max_step = 0.0f32;
        for i in 0..24_000 {
            let mut bus = [0.0f32; NCH];
            a.process(&mut bus);
            if i > 0 {
                max_step = max_step.max((bus[0] - prev).abs());
            }
            prev = bus[0];
        }
        // white-noise sample deltas at this level are ~2·0.047; a click
        // from an unsmoothed swap would be a much larger discontinuity
        assert!(max_step < 0.35, "route swap clicked: step {max_step}");
        // and the new route must be audible after the fade
        assert!(energy(&mut a, 24_000) > 1e-5);
    }

    /// Seep gains shape the spectrum: a lows-only seep must be darker
    /// than a flat one at equal total gain.
    #[test]
    fn seep_spectrum_follows_the_bands() {
        let hf_ratio = |seep: [f32; 3]| -> f32 {
            let mut a = Ambience::new(48_000.0);
            a.set_loop(noise_loop(48_000), false);
            a.set_environment(&env_with(vec![], seep));
            for _ in 0..48_000 {
                let mut bus = [0.0f32; NCH];
                a.process(&mut bus);
            }
            let (mut e, mut hf) = (0.0f64, 0.0f64);
            let mut prev = 0.0f32;
            for _ in 0..48_000 {
                let mut bus = [0.0f32; NCH];
                a.process(&mut bus);
                e += (bus[0] * bus[0]) as f64;
                hf += ((bus[0] - prev) * (bus[0] - prev)) as f64;
                prev = bus[0];
            }
            (hf / e.max(1e-12)) as f32
        };
        assert!(hf_ratio([1.0, 0.3, 0.05]) < 0.5 * hf_ratio([1.0, 1.0, 1.0]));
    }
}
