//! Stochastic energy path tracer. Estimates the *statistics* of the late
//! field (per-band echogram → RT60 and level), not individual paths — the
//! deterministic early field comes from `ism`. This is the part that moves
//! to wgpu compute later; the algorithm is embarrassingly parallel per ray.

use crate::params::ReverbParams;
use crate::rng::Rng;
use crate::scene::AcousticGeometry;
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

pub const BIN_DT: f32 = 0.010; // 10 ms echogram bins
pub const MAX_TIME: f32 = 3.0;
const NBINS: usize = (MAX_TIME / BIN_DT) as usize;
const RECEIVER_RADIUS: f32 = 0.5;
const WALL_EPS: f32 = 1e-3;

pub struct Echogram {
    pub bins: Vec<[f32; NBANDS]>,
    /// Energy-weighted arrival direction per bin (mid band), pointing from
    /// the listener toward where the energy came from. Its length relative
    /// to the bin energy measures anisotropy: 1 = all energy from one
    /// direction (e.g. through a doorway), 0 = fully diffuse.
    pub dirs: Vec<[f32; 3]>,
}

impl Echogram {
    pub fn new() -> Self {
        Self {
            bins: vec![[0.0; NBANDS]; NBINS],
            dirs: vec![[0.0; 3]; NBINS],
        }
    }

    pub fn clear(&mut self) {
        for b in &mut self.bins {
            *b = [0.0; NBANDS];
        }
        for d in &mut self.dirs {
            *d = [0.0; 3];
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
        for (a, o) in self.dirs.iter_mut().zip(other.dirs.iter()) {
            for k in 0..3 {
                a[k] += alpha * (o[k] - a[k]);
            }
        }
    }

    /// Aggregate direction of arrivals from `from_s` on: (unit direction,
    /// anisotropy 0…1). A room heard through its doorway reports the
    /// doorway's direction with high anisotropy — no portal bookkeeping.
    pub fn late_direction(&self, from_s: f32) -> ([f32; 3], f32) {
        let start = (from_s / BIN_DT) as usize;
        let mut v = [0.0f32; 3];
        let mut e = 0.0f32;
        for i in start.min(NBINS)..NBINS {
            for k in 0..3 {
                v[k] += self.dirs[i][k];
            }
            e += self.bins[i][1];
        }
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len < 1e-12 || e < 1e-12 {
            return ([1.0, 0.0, 0.0], 0.0);
        }
        ([v[0] / len, v[1] / len, v[2] / len], (len / e).min(1.0))
    }
}

pub fn trace(
    room: &impl AcousticGeometry,
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

        // NOTE: no Russian roulette here. In low-absorption rooms the decay
        // curve IS the long tail — stochastic termination makes the EDC
        // spike-dominated and corrupts the RT60 fit (measured, not theory).
        // The honest budget lever is `n_rays`: fewer rays + EMA temporal
        // accumulation degrades variance, never bias.
        for _bounce in 0..64 {
            // A miss = the ray escapes open geometry; its final segment
            // still passes the receiver check (that segment IS how sound
            // leaves a room through a doorway and reaches you outside).
            let hit = room.raycast_hit(pos, dir);
            let t_hit = hit.as_ref().map_or(f32::MAX, |h| h.t);

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
                        // arrival direction: back along the ray
                        out.dirs[bin][0] -= dir.x * energy[1];
                        out.dirs[bin][1] -= dir.y * energy[1];
                        out.dirs[bin][2] -= dir.z * energy[1];
                    }
                }
            }

            let Some(hit) = hit else {
                break; // escaped
            };
            // Advance to the surface, absorb, pick specular or diffuse bounce.
            pos = pos + dir * t_hit;
            dist_total += t_hit;
            if dist_total / SPEED_OF_SOUND > MAX_TIME {
                break;
            }
            let mat = &hit.material;
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

            let normal = hit.normal; // oriented against the incoming ray
            if rng.next_f32() < mat.scattering {
                // Lambertian: normal + uniform sphere point, renormalized.
                dir = (normal + rng.unit_sphere()).normalize();
                // Guard against grazing/degenerate directions.
                if dir.dot(normal) < 1e-3 {
                    dir = normal;
                }
            } else {
                dir = dir - normal * (2.0 * dir.dot(normal));
            }
            // Nudge off the surface to avoid self-intersection.
            pos = pos + normal * WALL_EPS;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::mesh::MeshBuilder;

    /// The emergent-portal proof: a mesh room with a doorway hole in one
    /// wall, listener OUTSIDE, zero portal/door/room authoring. Rays can
    /// only reach the listener through the opening — so the late field's
    /// aggregate direction must point at the doorway, with high
    /// anisotropy. This is what replaces per-portal coupled reverb.
    #[test]
    fn reverb_through_a_hole_reports_the_hole_direction() {
        let mut mb = MeshBuilder::new();
        let m = mb.material(Material::CONCRETE);
        let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        // room shell [0,8]×[0,6]×[0,3]: five solid faces...
        mb.quad(v(0.0, 0.0, 0.0), v(0.0, 6.0, 0.0), v(0.0, 6.0, 3.0), v(0.0, 0.0, 3.0), m);
        mb.quad(v(0.0, 0.0, 0.0), v(8.0, 0.0, 0.0), v(8.0, 0.0, 3.0), v(0.0, 0.0, 3.0), m);
        mb.quad(v(0.0, 6.0, 0.0), v(8.0, 6.0, 0.0), v(8.0, 6.0, 3.0), v(0.0, 6.0, 3.0), m);
        mb.quad(v(0.0, 0.0, 0.0), v(8.0, 0.0, 0.0), v(8.0, 6.0, 0.0), v(0.0, 6.0, 0.0), m);
        mb.quad(v(0.0, 0.0, 3.0), v(8.0, 0.0, 3.0), v(8.0, 6.0, 3.0), v(0.0, 6.0, 3.0), m);
        // ...and the x=8 wall with a doorway hole y∈[2.5,3.5], z∈[0,2.1]
        mb.quad(v(8.0, 0.0, 0.0), v(8.0, 2.5, 0.0), v(8.0, 2.5, 3.0), v(8.0, 0.0, 3.0), m);
        mb.quad(v(8.0, 3.5, 0.0), v(8.0, 6.0, 0.0), v(8.0, 6.0, 3.0), v(8.0, 3.5, 3.0), m);
        mb.quad(v(8.0, 2.5, 2.1), v(8.0, 3.5, 2.1), v(8.0, 3.5, 3.0), v(8.0, 2.5, 3.0), m);
        let mesh = mb.build();

        let src = Vec3::new(2.0, 3.0, 1.5);
        let lis = Vec3::new(12.0, 3.0, 1.6);
        let mut rng = Rng::new(21);
        let mut echo = Echogram::new();
        trace(&mesh, src, lis, 16_384, [1.0; 3], &mut rng, &mut echo);

        let (dir, aniso) = echo.late_direction(0.05);
        // doorway center (8, 3, 1.05) seen from (12, 3, 1.6)
        let to_hole = (Vec3::new(8.0, 3.0, 1.05) - lis).normalize();
        let dot = dir[0] * to_hole.x + dir[1] * to_hole.y + dir[2] * to_hole.z;
        assert!(dot > 0.85, "late field should point at the doorway: dir {dir:?} dot {dot}");
        assert!(aniso > 0.5, "through-a-hole field should be anisotropic: {aniso}");
    }

    /// Inside a closed room the late field is diffuse: low anisotropy.
    #[test]
    fn closed_room_late_field_is_diffuse() {
        use crate::scene::Shoebox;
        let room = Shoebox::new(Vec3::new(8.0, 6.0, 3.0), [Material::DRYWALL; 6]);
        let mut rng = Rng::new(5);
        let mut echo = Echogram::new();
        trace(&room, Vec3::new(2.0, 3.0, 1.5), Vec3::new(6.0, 2.0, 1.6), 16_384, [1.0; 3], &mut rng, &mut echo);
        let (_, aniso) = echo.late_direction(0.15);
        assert!(aniso < 0.35, "closed-room tail should be diffuse: {aniso}");
    }
}
