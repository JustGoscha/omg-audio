//! Image source method for shoebox rooms (Allen & Berkley 1979).
//! Gives the deterministic early-reflection field: exact delays, directions
//! and per-band gains for every specular path up to `max_order`.
//! In a convex shoebox every image is valid, so no occlusion test is needed;
//! the general-geometry version of this module will validate paths by ray.

use crate::material::air_attenuation;
use crate::params::Tap;
use crate::scene::Shoebox;
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

const MIN_DIST: f32 = 0.3;

pub fn image_source_taps(
    room: &Shoebox,
    source: Vec3,
    listener: Vec3,
    max_order: u32,
    out: &mut Vec<Tap>,
) {
    out.clear();
    let n_max = (max_order as i32 + 1) / 2 + 1;
    // Amplitude reflection factors per wall.
    let refl: [[f32; NBANDS]; 6] = core::array::from_fn(|w| room.walls[w].reflection_amplitude());

    for px in 0..2i32 {
        for py in 0..2i32 {
            for pz in 0..2i32 {
                for nx in -n_max..=n_max {
                    for ny in -n_max..=n_max {
                        for nz in -n_max..=n_max {
                            // Reflection counts per wall along each axis:
                            // wall at coord 0 → |n - p|, wall at coord L → |n|.
                            let hits = [
                                (nx - px).unsigned_abs(),
                                nx.unsigned_abs(),
                                (ny - py).unsigned_abs(),
                                ny.unsigned_abs(),
                                (nz - pz).unsigned_abs(),
                                nz.unsigned_abs(),
                            ];
                            let order: u32 = hits.iter().sum();
                            if order > max_order {
                                continue;
                            }

                            let img = Vec3::new(
                                (1 - 2 * px) as f32 * source.x + 2.0 * nx as f32 * room.size.x,
                                (1 - 2 * py) as f32 * source.y + 2.0 * ny as f32 * room.size.y,
                                (1 - 2 * pz) as f32 * source.z + 2.0 * nz as f32 * room.size.z,
                            );
                            let to_img = img - listener;
                            let dist = to_img.length().max(MIN_DIST);
                            let dir = to_img.normalize();
                            let air = air_attenuation(dist);

                            let mut gains = [0.0f32; NBANDS];
                            for b in 0..NBANDS {
                                let mut g = air[b] / dist;
                                for w in 0..6 {
                                    for _ in 0..hits[w] {
                                        g *= refl[w][b];
                                    }
                                }
                                gains[b] = g;
                            }

                            out.push(Tap {
                                key: out.len() as u32,
                                delay_s: dist / SPEED_OF_SOUND,
                                dir: [dir.x, dir.y, dir.z],
                                gains,
                            });
                        }
                    }
                }
            }
        }
    }
    // Enumeration order is deterministic and lattice-stable, so tap index i
    // always refers to the same reflection path across updates — that is
    // what lets the renderer smooth per-tap parameters without re-matching.
}
