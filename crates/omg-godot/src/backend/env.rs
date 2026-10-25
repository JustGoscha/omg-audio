//! The outdoor field — an ambience bed and rain — routed by the level.
//!
//! omg-audio's Mixer has no environment input yet, so this runs beside
//! it: omg-dsp's Ambience and Rain on their own SH bus, decoded through
//! the same speaker HRIRs, head-rotated with the listener, added to the
//! mixer's output. Their routing comes from each frame's SKY cells (the
//! listener's own rays that reached open sky: through doorways, panes,
//! off facades) — the v3 branch of the walkthrough scene's
//! WorldSim::environment, minus its room-graph terms, which are replaced
//! here by the frame's open-sky fraction (enclosure) and a host-provided
//! roof probe.
//!
//! GAP / to de-duplicate: this belongs in omg-audio's Mixer (an
//! environment bus before its loudness stage, so the ear adaptation rides
//! ambience and scene together).

use omg_audio::frame::SkyCell;
use omg_dsp::ambi::{HeadRotation, NCH};
use omg_dsp::env::{EnvRoute, EnvWindow, Environment, MAX_ENV_ROUTES, MAX_ENV_WINDOWS};
use omg_dsp::hrtf::BinauralDecoder;
use omg_dsp::smooth::Smoothed;

/// Escaped-ray energy fraction → route amplitude (WorldSim's DOME_GAIN:
/// Σ gains² ≈ 3 in the open field).
const SKY_GAIN: f32 = 4.5;
/// Open-sky fraction of a listener in an open field (sky above plus the
/// ground bounce): the 'not enclosed at all' reference.
const OPEN_SKY: f32 = 0.6;

/// What the host knows about the listener's surroundings.
#[derive(Clone, Default)]
pub struct EnvContext {
    /// Something overhead with open sky above it (rain drums on it).
    pub roof: bool,
    /// Directions (listener frame-free, world) and distances to nearby
    /// glass panes: rain patters on them.
    pub windows: Vec<([f32; 3], f32)>,
}

/// Sky bin of a direction: 8 azimuth sectors + the zenith cap.
fn sky_bin(d: [f32; 3]) -> usize {
    if d[2] > 0.87 {
        return 8;
    }
    let az = d[1].atan2(d[0]) + core::f32::consts::PI;
    ((az / (core::f32::consts::TAU / 8.0)) as usize).min(7)
}

/// Frame sky + host context → the Environment omg-dsp's beds consume.
pub fn environment(sky: &[SkyCell], ctx: &EnvContext) -> Environment {
    let mut e = [[0.0f32; 3]; 9];
    let mut dsum = [[0.0f32; 3]; 9];
    let mut total = 0.0f32;
    for c in sky {
        let k = sky_bin(c.dir);
        for b in 0..3 {
            e[k][b] += c.energy[b];
            dsum[k][b] += c.dir[b] * c.energy[1];
        }
        total += c.energy[1];
    }
    let mut routes: Vec<EnvRoute> = (0..9)
        .filter(|&k| e[k][1] * SKY_GAIN > 1e-8)
        .map(|k| {
            let l = (dsum[k][0] * dsum[k][0] + dsum[k][1] * dsum[k][1] + dsum[k][2] * dsum[k][2]).sqrt();
            let dir = if l > 1e-9 {
                [dsum[k][0] / l, dsum[k][1] / l, dsum[k][2] / l]
            } else if k == 8 {
                [0.0, 0.0, 1.0]
            } else {
                let az = (k as f32 + 0.5) * core::f32::consts::TAU / 8.0 - core::f32::consts::PI;
                [az.cos(), az.sin(), 0.0]
            };
            EnvRoute { id: 200 + k as u32, dir, gains: core::array::from_fn(|b| (e[k][b] * SKY_GAIN).sqrt()), dist: 10.0 }
        })
        .collect();
    routes.sort_by(|a, b| b.gains[1].total_cmp(&a.gains[1]));
    routes.truncate(MAX_ENV_ROUTES);
    let enclosure = (1.0 - total / OPEN_SKY).clamp(0.0, 1.0);
    let mut windows: Vec<EnvWindow> = ctx
        .windows
        .iter()
        .map(|(dir, dist)| EnvWindow { dir: *dir, gain: enclosure * 3.0 / (3.0 + dist.max(0.3)) })
        .collect();
    windows.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    windows.truncate(MAX_ENV_WINDOWS);
    // the shell's own seep: mass-law leakage of the outdoor bed
    let seep = [0.10 * enclosure, 0.03 * enclosure, 0.006 * enclosure];
    Environment {
        seep,
        enclosure,
        roof_gain: if ctx.roof { enclosure } else { 0.0 },
        // the ambience bed comes in through the same sky inlets (the
        // demo scene shapes its crickets as emitters in the grass —
        // omg_scene::ambient — which a generic host has no data for)
        amb_routes: routes.clone(),
        amb_seep: seep,
        routes,
        windows,
    }
}

pub struct EnvBed {
    ambience: omg_dsp::ambience::Ambience,
    rain: omg_dsp::rain::Rain,
    decode: Option<BinauralDecoder>,
    head: [Smoothed; 3],
    has_bed: bool,
    rain_on: bool,
    pub enclosure: f32,
}

impl EnvBed {
    pub fn new(sr: f32, speakers: &[u8]) -> Self {
        Self {
            ambience: omg_dsp::ambience::Ambience::new(sr),
            rain: omg_dsp::rain::Rain::new(sr),
            decode: Some(BinauralDecoder::from_bytes(speakers)),
            head: core::array::from_fn(|_| Smoothed::new(0.0, 0.03, sr)),
            has_bed: false,
            rain_on: false,
            enclosure: 0.0,
        }
    }

    /// Ambience loop (interleaved stereo when `stereo`), played at its
    /// authored loudness with slow swings flattened.
    pub fn set_ambience(&mut self, mut data: Vec<f32>, stereo: bool, sr: f32) {
        omg_dsp::level::flatten_slow_loudness(&mut data, if stereo { 2 } else { 1 }, sr);
        self.has_bed = !data.is_empty();
        self.ambience.set_loop(data, stereo);
    }

    pub fn set_ambience_gain(&mut self, g: f32) {
        self.ambience.set_user(g);
    }

    pub fn set_rain_bank(&mut self, bank: Vec<f32>) {
        self.rain.set_bank(bank);
    }

    pub fn set_rain(&mut self, intensity: f32) {
        self.rain_on |= intensity > 0.0;
        self.rain.set_intensity(intensity);
    }

    pub fn set_rain_gain(&mut self, g: f32) {
        self.rain.set_gain(g);
    }

    pub fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        for (s, v) in self.head.iter_mut().zip([yaw, pitch, roll]) {
            s.set(v);
        }
    }

    pub fn set_environment(&mut self, env: &Environment) {
        self.enclosure = env.enclosure;
        self.ambience.set_environment(env);
        self.rain.set_environment(env);
    }

    /// Add the beds into `l`/`r`.
    pub fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let Some(dec) = &mut self.decode else { return };
        // nothing to play: no bed loaded and the rain has stopped
        if !self.has_bed && !self.rain_on {
            return;
        }
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let mut bus = [0.0f32; NCH];
            self.rain.process(&mut bus);
            self.ambience.process(&mut bus);
            HeadRotation::new(self.head[0].tick(), self.head[1].tick(), self.head[2].tick()).apply(&mut bus);
            let (x, y) = dec.process(&bus);
            *a += x;
            *b += y;
        }
    }
}
