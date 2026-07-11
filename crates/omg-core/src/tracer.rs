//! Stochastic energy path tracer. Estimates the *statistics* of the late
//! field (per-band echogram → RT60 and level), not individual paths — the
//! deterministic early field comes from `ism`. This is the part that moves
//! to wgpu compute later; the algorithm is embarrassingly parallel per ray.

use crate::params::ReverbParams;
use crate::rng::Rng;
use crate::scene::Shoebox;
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

pub const BIN_DT: f32 = 0.010; // 10 ms echogram bins
pub const MAX_TIME: f32 = 3.0;
const NBINS: usize = (MAX_TIME / BIN_DT) as usize;
const RECEIVER_RADIUS: f32 = 0.5;
const WALL_EPS: f32 = 1e-3;

pub struct Echogram {
    pub bins: Vec<[f32; NBANDS]>,
}

impl Echogram {
    pub fn new() -> Self {
        Self { bins: vec![[0.0; NBANDS]; NBINS] }
    }

    pub fn clear(&mut self) {
        for b in &mut self.bins {
            *b = [0.0; NBANDS];
        }
    }

    /// Exponential moving average toward `other` — temporal accumulation
    /// across simulation updates, the fix for Monte Carlo flutter.
    pub fn ema(&mut self, other: &Echogram, alpha: f32) {
        for (a, o) in self.bins.iter_mut().zip(other.bins.iter()) {
            for band in 0..NBANDS {
                a[band] += alpha * (o[band] - a[band]);
            }
        }
    }
}

pub fn trace(
    room: &Shoebox,
    source: Vec3,
    listener: Vec3,
    n_rays: u32,
    source_energy: [f32; NBANDS],
    rng: &mut Rng,
    out: &mut Echogram,
) {
    out.clear();
    let per_ray = 1.0 / n_rays as f32;

    for _ in 0..n_rays {
        let mut pos = source;
        let mut dir = rng.unit_sphere();
        let mut energy: [f32; NBANDS] = core::array::from_fn(|b| per_ray * source_energy[b]);
        let mut dist_total = 0.0f32;

        for _bounce in 0..64 {
            let (t_hit, wall) = room.raycast(pos, dir);
            if !t_hit.is_finite() || t_hit <= 0.0 {
                break;
            }

            // Receiver sphere crossing along this segment.
            let to_l = listener - pos;
            let s = to_l.dot(dir);
            if s > 0.0 && s < t_hit {
                let closest = pos + dir * s;
                if (closest - listener).length() < RECEIVER_RADIUS {
                    let arrival = (dist_total + s) / SPEED_OF_SOUND;
                    let bin = (arrival / BIN_DT) as usize;
                    if bin < NBINS {
                        for b in 0..NBANDS {
                            out.bins[bin][b] += energy[b];
                        }
                    }
                }
            }

            // Advance to the wall, absorb, pick specular or diffuse bounce.
            pos = pos + dir * t_hit;
            dist_total += t_hit;
            if dist_total / SPEED_OF_SOUND > MAX_TIME {
                break;
            }
            let mat = &room.walls[wall];
            let mut alive = false;
            for b in 0..NBANDS {
                energy[b] *= 1.0 - mat.absorption[b];
                if energy[b] > 1e-7 * per_ray {
                    alive = true;
                }
            }
            if !alive {
                break;
            }

            let axis = wall / 2;
            if rng.next_f32() < mat.scattering {
                // Lambertian: normal + uniform sphere point, renormalized.
                let mut normal = Vec3::new(0.0, 0.0, 0.0);
                normal.set(axis, if wall % 2 == 0 { 1.0 } else { -1.0 });
                dir = (normal + rng.unit_sphere()).normalize();
                // Guard against grazing/degenerate directions.
                if dir.dot(normal) < 1e-3 {
                    dir = normal;
                }
            } else {
                dir.set(axis, -dir.get(axis));
            }
            // Nudge off the wall to avoid self-intersection.
            pos.set(
                axis,
                if wall % 2 == 0 { WALL_EPS } else { room.size.get(axis) - WALL_EPS },
            );
        }
    }
}

/// Schroeder backward integration + linear fit between -5 dB and -25 dB.
pub fn estimate_reverb(echo: &Echogram) -> ReverbParams {
    let mut rt60 = [0.5f32; NBANDS];

    for b in 0..NBANDS {
        // Backward-integrated energy decay curve.
        let mut total = 0.0f64;
        let mut edc = vec![0.0f64; NBINS];
        for i in (0..NBINS).rev() {
            total += echo.bins[i][b] as f64;
            edc[i] = total;
        }
        if total <= 0.0 {
            continue;
        }
        let db = |i: usize| 10.0 * (edc[i] / total).max(1e-12).log10();

        let mut t5 = None;
        let mut t25 = None;
        for i in 0..NBINS {
            let d = db(i);
            if t5.is_none() && d <= -5.0 {
                t5 = Some(i);
            }
            if t25.is_none() && d <= -25.0 {
                t25 = Some(i);
                break;
            }
        }
        if let (Some(i5), Some(i25)) = (t5, t25) {
            if i25 > i5 {
                let dt = (i25 - i5) as f32 * BIN_DT;
                let ddb = (db(i25) - db(i5)) as f32; // negative
                rt60[b] = (-60.0 * dt / ddb).clamp(0.1, 5.0);
            }
        }
    }

    // Late level per band: energy arriving after 80 ms. This is an absolute
    // measurement at the listener for a unit source — deliberately NOT
    // scaled by direct-path distance (the diffuse field doesn't get louder
    // when you walk up to the source; the direct sound does).
    let late_start = (0.080 / BIN_DT) as usize;
    let mut level = [0.0f32; NBANDS];
    for (b, lv) in level.iter_mut().enumerate() {
        let mut late_e = 0.0f32;
        for i in late_start..NBINS {
            late_e += echo.bins[i][b];
        }
        *lv = late_e.sqrt().min(0.7);
    }

    ReverbParams { rt60, level }
}
