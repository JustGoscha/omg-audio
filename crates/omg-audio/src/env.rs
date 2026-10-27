//! The outdoor field — an ambience bed and rain — routed by the level.
//!
//! The simulation describes the listener's surroundings each tick: the
//! frame's SKY cells (the listener's own rays that reached open sky —
//! through doorways, panes, off facades), a roof probe and the glass
//! panes near by, folded into the `Environment` omg-dsp's ambience bed
//! and rain engine consume (`Frame::env`). The mixer plays both on its
//! SH bus before the loudness stage, so the ear adaptation rides the
//! scene and the outdoors together.

use crate::frame::SkyCell;
use omg_dsp::env::{EnvRoute, EnvWindow, Environment, MAX_ENV_ROUTES, MAX_ENV_WINDOWS};

/// Escaped-ray energy fraction → route amplitude (the walkthrough
/// scene's DOME_GAIN: Σ gains² ≈ 3 in the open field).
const SKY_GAIN: f32 = 4.5;
/// Open-sky fraction of a listener in an open field (sky above plus the
/// ground bounce): the 'not enclosed at all' reference.
const OPEN_SKY: f32 = 0.6;
/// Panes farther than this don't patter audibly (m).
pub const WINDOW_REACH: f32 = 12.0;

/// What the simulation knows about the listener's surroundings besides
/// the sky cells.
#[derive(Clone, Debug, Default)]
pub struct EnvContext {
    /// Something overhead with open sky above it (rain drums on it).
    pub roof: bool,
    /// Directions (world) and distances to nearby glass panes: rain
    /// patters on them.
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

/// Sky cells + context → the Environment omg-dsp's beds consume.
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
        // the ambience bed comes in through the same sky inlets
        amb_routes: routes.clone(),
        amb_seep: seep,
        routes,
        windows,
    }
}
