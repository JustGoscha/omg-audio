//! Environment state: how the OUTDOOR sound field (ambience, rain)
//! reaches the listener. Produced by the simulation from geometry alone —
//! a power balance over the room graph plus per-aperture routing — and
//! consumed by the ambience bed and the rain engine. This replaces
//! per-room hand constants and listener-anchored beds: room transitions,
//! door swings and occlusion all arrive here already priced.
//!
//! The struct crosses the worker→worklet boundary as a flat f32 block
//! (`write_flat` / `read_flat`), like ParamBlocks do.

pub const MAX_ENV_ROUTES: usize = 8;
pub const MAX_ENV_WINDOWS: usize = 4;

const ROUTE_STRIDE: usize = 8; // id, dir xyz, gains ×3, dist
const WINDOW_STRIDE: usize = 4; // dir xyz, gain

/// seep(3) + enclosure + roof + n_routes + routes + n_windows + windows.
pub const ENV_FLAT_LEN: usize =
    3 + 1 + 1 + 1 + MAX_ENV_ROUTES * ROUTE_STRIDE + 1 + MAX_ENV_WINDOWS * WINDOW_STRIDE;

/// One inlet for the outdoor field: an aperture of the listener's room
/// (door slit, pane, stairwell) or — outdoors — a horizon sector of the
/// open sky itself. Everything is world-frame; the SH bus rotates with
/// the head downstream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvRoute {
    /// Stable identity across ticks (aperture index / horizon sector id):
    /// the engine keys its per-route smoothers on this.
    pub id: u32,
    /// Unit direction the field arrives FROM.
    pub dir: [f32; 3],
    /// Per-band amplitude of the outdoor field through this inlet
    /// (field level × filler transmission × near-field reach × blend).
    pub gains: [f32; 3],
    /// Meters to the radiating point (drop sounds scale with this).
    pub dist: f32,
}

/// A glass pane of the listener's room: rain drops impact ON it, so the
/// drop synthesis anchors there instead of at a random direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvWindow {
    pub dir: [f32; 3],
    /// Blend-weighted level for impacts on this pane (near-field reach
    /// already applied).
    pub gain: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Environment {
    /// Diffuse residual: the outdoor field's level INSIDE the listener's
    /// room shell (amplitude per band, √ of the power-balance energy).
    /// Zero outdoors — there the directional routes carry the whole field.
    pub seep: [f32; 3],
    /// 0 = open sky … 1 = sealed room. Continuous across blend zones;
    /// drives rain drop statistics (roof concentration, airborne mix).
    pub enclosure: f32,
    /// Sky-exposed fraction of the ceiling over the listener's head
    /// (0 under another storey or outdoors) — scales roof drumming.
    pub roof_gain: f32,
    pub routes: Vec<EnvRoute>,
    pub windows: Vec<EnvWindow>,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            seep: [0.0; 3],
            enclosure: 0.0,
            roof_gain: 0.0,
            routes: Vec::new(),
            windows: Vec::new(),
        }
    }
}

impl Environment {
    pub fn write_flat(&self, out: &mut [f32]) {
        assert!(out.len() >= ENV_FLAT_LEN);
        out[..ENV_FLAT_LEN].fill(0.0);
        out[..3].copy_from_slice(&self.seep);
        out[3] = self.enclosure;
        out[4] = self.roof_gain;
        let nr = self.routes.len().min(MAX_ENV_ROUTES);
        out[5] = nr as f32;
        for (i, r) in self.routes.iter().take(nr).enumerate() {
            let o = 6 + i * ROUTE_STRIDE;
            out[o] = r.id as f32;
            out[o + 1..o + 4].copy_from_slice(&r.dir);
            out[o + 4..o + 7].copy_from_slice(&r.gains);
            out[o + 7] = r.dist;
        }
        let wo = 6 + MAX_ENV_ROUTES * ROUTE_STRIDE;
        let nw = self.windows.len().min(MAX_ENV_WINDOWS);
        out[wo] = nw as f32;
        for (i, w) in self.windows.iter().take(nw).enumerate() {
            let o = wo + 1 + i * WINDOW_STRIDE;
            out[o..o + 3].copy_from_slice(&w.dir);
            out[o + 3] = w.gain;
        }
    }

    pub fn read_flat(flat: &[f32]) -> Self {
        assert!(flat.len() >= ENV_FLAT_LEN);
        let nr = (flat[5] as usize).min(MAX_ENV_ROUTES);
        let routes = (0..nr)
            .map(|i| {
                let o = 6 + i * ROUTE_STRIDE;
                EnvRoute {
                    id: flat[o] as u32,
                    dir: [flat[o + 1], flat[o + 2], flat[o + 3]],
                    gains: [flat[o + 4], flat[o + 5], flat[o + 6]],
                    dist: flat[o + 7],
                }
            })
            .collect();
        let wo = 6 + MAX_ENV_ROUTES * ROUTE_STRIDE;
        let nw = (flat[wo] as usize).min(MAX_ENV_WINDOWS);
        let windows = (0..nw)
            .map(|i| {
                let o = wo + 1 + i * WINDOW_STRIDE;
                EnvWindow { dir: [flat[o], flat[o + 1], flat[o + 2]], gain: flat[o + 3] }
            })
            .collect();
        Self {
            seep: [flat[0], flat[1], flat[2]],
            enclosure: flat[3],
            roof_gain: flat[4],
            routes,
            windows,
        }
    }
}

// --------------------------------------------------------------- engine side

use crate::ambi::{encode_gains, NCH};
use crate::bands::BandSplit;
use crate::smooth::Smoothed;

/// Per-route engine state: a fixed bank of slots keyed by the route's
/// stable id, each with smoothed per-band gains and a band splitter.
/// Routes that disappear ramp to silence in place; new ones fade in from
/// zero — no set-membership change ever clicks.
pub struct RouteSlot {
    pub id: u32,
    pub enc: [f32; NCH],
    pub gains: [Smoothed; 3],
    pub split: BandSplit,
    pub dist: f32,
}

pub struct RouteSlots {
    pub slots: [RouteSlot; MAX_ENV_ROUTES],
}

impl RouteSlots {
    pub const FREE: u32 = u32::MAX;

    pub fn new(sample_rate: f32, tau_s: f32) -> Self {
        Self {
            slots: core::array::from_fn(|_| RouteSlot {
                id: Self::FREE,
                enc: [0.0; NCH],
                gains: core::array::from_fn(|_| Smoothed::new(0.0, tau_s, sample_rate)),
                split: BandSplit::new(sample_rate),
                dist: 1.0,
            }),
        }
    }

    /// Apply a fresh environment: update matching ids, retire missing
    /// ones toward zero, place new routes in silent slots.
    pub fn update(&mut self, env: &Environment) {
        for s in &mut self.slots {
            if s.id != Self::FREE && !env.routes.iter().any(|r| r.id == s.id) {
                for g in &mut s.gains {
                    g.set(0.0);
                }
            }
        }
        for r in &env.routes {
            let slot = if let Some(s) = self.slots.iter_mut().find(|s| s.id == r.id) {
                s
            } else if let Some(s) = self.slots.iter_mut().find(|s| {
                s.id == Self::FREE
                    || s.gains.iter().all(|g| g.current() < 1e-5 && g.target_val() < 1e-5)
            }) {
                s.id = r.id;
                for g in &mut s.gains {
                    g.snap(0.0);
                }
                s
            } else {
                continue; // bank full of live routes — sim caps at the same size
            };
            slot.enc = encode_gains(r.dir);
            slot.dist = r.dist;
            for (g, &v) in slot.gains.iter_mut().zip(r.gains.iter()) {
                g.set(v);
            }
        }
    }

    /// Sum of current mid-band gains (drop-spawn weighting).
    pub fn total_mid(&self) -> f32 {
        self.slots.iter().map(|s| s.gains[1].current()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_round_trip_is_lossless() {
        let env = Environment {
            seep: [0.2, 0.1, 0.05],
            enclosure: 0.87,
            roof_gain: 0.5,
            routes: vec![
                EnvRoute {
                    id: 3,
                    dir: [0.6, -0.8, 0.0],
                    gains: [0.5, 0.25, 0.125],
                    dist: 4.5,
                },
                EnvRoute { id: 101, dir: [0.0, 1.0, 0.0], gains: [1.0, 0.9, 0.8], dist: 35.0 },
            ],
            windows: vec![EnvWindow { dir: [0.0, 0.0, 1.0], gain: 0.4 }],
        };
        let mut flat = [0.0f32; ENV_FLAT_LEN];
        env.write_flat(&mut flat);
        assert_eq!(Environment::read_flat(&flat), env);
    }
}
