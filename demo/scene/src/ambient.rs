//! Ambient emitters: ambience beds with a SHAPE in the world.
//!
//! The classic bed is a dome — an audio skybox around the listener,
//! routed through whatever inlets the traced sky finds (`dome.rs`). That
//! is right for wind or far traffic, and wrong for crickets: they sit in
//! the grass. Fly up and they must come from BELOW, quieter and narrower
//! the higher you go; walk away and they collapse toward the meadow and
//! fade. So an ambience emitter is data with a shape:
//!
//!  - `Dome`: the skybox, exactly as before (level × the traced sky).
//!  - `GroundArea`: emitters spread over a ground rectangle (minus holes:
//!    building footprints, a road).
//!  - `Box`: emitters filling a volume (a forest, a crowd in a hall).
//!
//! A shaped emitter is a continuum of incoherent point emitters at
//! density σ: each area/volume element radiates σ·dA, arriving with
//! spherical spreading 1/ρ² and air absorption. Nothing else is assumed:
//! the field at the listener is the integral over the shape, so its
//! level and direction are the shape's projection as seen from the ear —
//! surround at ear height inside a meadow (equal energy per octave of
//! distance, mostly near-horizontal), a downward lobe from above that
//! narrows and fades as the area's solid angle shrinks, a compact
//! direction from afar.
//!
//! The integral is an adaptive cell quadrature (Barnes–Hut style): a
//! cell is split until it is small against its distance, so the cost is
//! logarithmic in the shape's extent and the same code works at 1.6 m
//! inside the meadow and 10 km away. Each leaf cell is a point emitter;
//! the caller prices each leaf's straight line to the ear (walls by mass
//! law, panes, door leaves — the dome's own geometry), which is the
//! DIRECT field. `world.rs` adds the share the geometry blocked back in
//! through the traced inlets (doorways, windows, shell seep), so indoor
//! listening keeps working exactly like the dome bed did.

use crate::dome::{bin_of, DOME_BINS};
use omg_core::material::air_attenuation;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

/// Axis-aligned ground rectangle ((x0, y0), (x1, y1)).
pub type Rect = ((f32, f32), (f32, f32));

#[derive(Clone, Debug, PartialEq)]
pub enum AmbientShape {
    /// The audio skybox: surrounds from above and the horizon wherever
    /// you are, routed by the traced sky (`dome.rs` / `Tracer::sky`).
    Dome,
    /// Emitters on the ground at height `z` over the rectangle
    /// `min..max`, minus `holes` (buildings, a road).
    GroundArea { min: (f32, f32), max: (f32, f32), z: f32, holes: Vec<Rect> },
    /// Emitters filling a volume.
    Box { min: Vec3, max: Vec3 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AmbientEmitter {
    pub shape: AmbientShape,
    /// Dome: energy multiplier on the traced sky (1 = the classic bed).
    /// Shaped: emitter density σ (energy per m² / m³), relative — the
    /// scene calibrates all shaped emitters together, see [`calibrate`].
    pub level: f32,
}

/// A cell is a leaf once its size is below this fraction of its
/// distance (the opening angle of the Barnes–Hut criterion).
const THETA: f32 = 0.5;
/// Smallest cell edge, m: finer than any wavelength that matters here.
const MIN_CELL: f32 = 0.5;
const MAX_DEPTH: u32 = 32;

/// One quadrature leaf: a point emitter standing for its cell.
#[derive(Clone, Copy, Debug)]
pub struct Leaf {
    /// Emission point (the cell's uncovered centroid).
    pub p: Vec3,
    /// Free-field energy at the ear per band (spreading + air, no
    /// occlusion), before calibration.
    pub e: [f32; NBANDS],
    /// The ear sits inside this cell: it surrounds, no single direction.
    pub omni: bool,
}

fn overlap(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

/// Split one cell into children along every axis that is at least half
/// as long as the longest (so elongated shapes split their long side
/// first and cells stay roughly square).
fn split(min: Vec3, max: Vec3) -> Vec<(Vec3, Vec3)> {
    let ext = [max.x - min.x, max.y - min.y, max.z - min.z];
    let m = ext[0].max(ext[1]).max(ext[2]);
    let mut out = vec![(min, max)];
    for axis in 0..3 {
        if ext[axis] < 0.5 * m || ext[axis] <= 0.0 {
            continue;
        }
        let mut next = Vec::with_capacity(out.len() * 2);
        for (a, b) in out {
            let mid = match axis {
                0 => 0.5 * (a.x + b.x),
                1 => 0.5 * (a.y + b.y),
                _ => 0.5 * (a.z + b.z),
            };
            let (mut b1, mut a2) = (b, a);
            match axis {
                0 => {
                    b1.x = mid;
                    a2.x = mid;
                }
                1 => {
                    b1.y = mid;
                    a2.y = mid;
                }
                _ => {
                    b1.z = mid;
                    a2.z = mid;
                }
            }
            next.push((a, b1));
            next.push((a2, b));
        }
        out = next;
    }
    out
}

fn recurse(min: Vec3, max: Vec3, holes: &[Rect], eye: Vec3, depth: u32, out: &mut Vec<Leaf>) {
    let ext = [max.x - min.x, max.y - min.y, max.z - min.z];
    let flat = ext[2] <= 0.0;
    // measure (area or volume) and centroid, holes carved out of ground
    // cells (holes are disjoint footprints; overlaps would double-carve)
    let c0 = (min + max) * 0.5;
    let (measure, centroid) = if flat {
        let a = ext[0] * ext[1];
        let (mut m, mut mx, mut my) = (a, a * c0.x, a * c0.y);
        for &((hx0, hy0), (hx1, hy1)) in holes {
            let ox = overlap(min.x, max.x, hx0, hx1);
            let oy = overlap(min.y, max.y, hy0, hy1);
            let ao = ox * oy;
            if ao > 0.0 {
                m -= ao;
                mx -= ao * 0.5 * (min.x.max(hx0) + max.x.min(hx1));
                my -= ao * 0.5 * (min.y.max(hy0) + max.y.min(hy1));
            }
        }
        if m <= 1e-4 * a.max(1e-9) {
            return; // all hole
        }
        (m, Vec3::new(mx / m, my / m, c0.z))
    } else {
        (ext[0] * ext[1] * ext[2], c0)
    };
    let size = ext[0].max(ext[1]).max(ext[2]);
    let d = (c0 - eye).length();
    if size < THETA * d || size <= MIN_CELL || depth >= MAX_DEPTH {
        let p = centroid;
        let rho = (p - eye).length();
        // the cell-averaged 1/ρ² stays finite when the ear is in or at
        // the cell: soften by the cell's own half-size
        let r2 = rho * rho + 0.25 * size * size;
        let air = air_attenuation(rho);
        out.push(Leaf {
            p,
            e: core::array::from_fn(|b| measure * air[b] * air[b] / r2),
            omni: rho < 0.5 * size,
        });
        return;
    }
    for (a, b) in split(min, max) {
        recurse(a, b, holes, eye, depth + 1, out);
    }
}

/// Quadrature leaves of one shaped emitter as seen from `eye` (free
/// field, density 1). `Dome` has none — it is routed by the sky.
pub fn leaves(shape: &AmbientShape, eye: Vec3, out: &mut Vec<Leaf>) {
    match shape {
        AmbientShape::Dome => {}
        AmbientShape::GroundArea { min, max, z, holes } => {
            if max.0 > min.0 && max.1 > min.1 {
                recurse(Vec3::new(min.0, min.1, *z), Vec3::new(max.0, max.1, *z), holes, eye, 0, out);
            }
        }
        AmbientShape::Box { min, max } => {
            if max.x > min.x && max.y > min.y && max.z > min.z {
                recurse(*min, *max, &[], eye, 0, out);
            }
        }
    }
}

/// Free-field energy per band of all shaped emitters at `eye` (density-
/// weighted, uncalibrated).
pub fn free_total(emitters: &[AmbientEmitter], eye: Vec3) -> [f32; NBANDS] {
    let mut buf = Vec::new();
    let mut t = [0.0f32; NBANDS];
    for em in emitters {
        buf.clear();
        leaves(&em.shape, eye, &mut buf);
        for l in &buf {
            for b in 0..NBANDS {
                t[b] += em.level * l.e[b];
            }
        }
    }
    t
}

/// Per-band scale that makes the shaped emitters' free field at the
/// reference ear `at` equal `target` (the recording IS the field heard
/// there). One number per band for the whole scene: positions then
/// differ from the recording only by geometry.
pub fn calibrate(emitters: &[AmbientEmitter], at: Vec3, target: [f32; NBANDS]) -> [f32; NBANDS] {
    let t = free_total(emitters, at);
    core::array::from_fn(|b| if t[b] > 1e-20 { target[b] / t[b] } else { 0.0 })
}

/// The field of the shaped emitters at an ear, binned like the dome
/// (8 azimuth sectors + zenith, by ARRIVAL direction).
#[derive(Clone, Copy, Debug)]
pub struct ShapedField {
    /// Per bin: direct energy per band (occlusion priced by `vis`).
    pub direct: [[f32; NBANDS]; DOME_BINS],
    /// Per bin: energy-weighted (mid band) sum of arrival directions.
    pub dir_sum: [Vec3; DOME_BINS],
    /// Total free-field energy per band (as if nothing stood in the way).
    pub free: [f32; NBANDS],
}

impl ShapedField {
    /// Total direct energy per band.
    pub fn direct_total(&self) -> [f32; NBANDS] {
        core::array::from_fn(|b| self.direct.iter().map(|k| k[b]).sum())
    }

    /// Energy-weighted mean arrival direction (mid band) and its length:
    /// ~0 for a surrounding field, 1 for a point.
    pub fn mean_dir(&self) -> (Vec3, f32) {
        let mut s = Vec3::new(0.0, 0.0, 0.0);
        let mut e = 0.0f32;
        for k in 0..DOME_BINS {
            s = s + self.dir_sum[k];
            e += self.direct[k][1];
        }
        let l = s.length();
        if l < 1e-20 || e < 1e-20 {
            return (Vec3::new(0.0, 0.0, 0.0), 0.0);
        }
        (s * (1.0 / l), l / e)
    }
}

/// Bin centre direction (where a surrounding leaf's share is anchored).
pub fn bin_center(k: usize) -> Vec3 {
    if k == 8 {
        return Vec3::new(0.0, 0.0, 1.0);
    }
    let az = (k as f32 + 0.5) * core::f32::consts::TAU / 8.0 - core::f32::consts::PI;
    Vec3::new(az.cos(), az.sin(), 0.0)
}

/// Solid-angle share of each dome bin (zenith cap above z = 0.87).
fn bin_share(k: usize) -> f32 {
    if k == 8 { 0.5 * (1.0 - 0.87) } else { 0.5 * 1.87 / 8.0 }
}

/// Field of the shaped emitters at `eye`. `cal` is the scene calibration
/// ([`calibrate`]); `vis(eye, p)` prices the straight line from the ear
/// to an emission point (per-band ENERGY transmission, 1 = clear).
/// Leaves too weak to matter skip the visibility query (counted clear).
pub fn field(
    emitters: &[AmbientEmitter],
    cal: [f32; NBANDS],
    eye: Vec3,
    mut vis: impl FnMut(Vec3, Vec3) -> [f32; NBANDS],
    buf: &mut Vec<Leaf>,
) -> ShapedField {
    let mut f = ShapedField {
        direct: [[0.0; NBANDS]; DOME_BINS],
        dir_sum: [Vec3::new(0.0, 0.0, 0.0); DOME_BINS],
        free: [0.0; NBANDS],
    };
    buf.clear();
    let mut lvl: Vec<f32> = Vec::new();
    for em in emitters {
        leaves(&em.shape, eye, buf);
        lvl.resize(buf.len(), em.level);
    }
    for (l, &s) in buf.iter_mut().zip(&lvl) {
        for b in 0..NBANDS {
            l.e[b] *= s * cal[b];
            f.free[b] += l.e[b];
        }
    }
    let floor = 1e-4 * f.free[1].max(f.free[2]).max(f.free[0]);
    for l in buf.iter() {
        if l.omni {
            for k in 0..DOME_BINS {
                let w = bin_share(k);
                for b in 0..NBANDS {
                    f.direct[k][b] += w * l.e[b];
                }
                f.dir_sum[k] = f.dir_sum[k] + bin_center(k) * (w * l.e[1]);
            }
            continue;
        }
        let d = l.p - eye;
        let dl = d.length().max(1e-6);
        let dir = d * (1.0 / dl);
        let t = if l.e[1].max(l.e[0]) > floor { vis(eye, l.p) } else { [1.0; NBANDS] };
        let k = bin_of(dir);
        for b in 0..NBANDS {
            f.direct[k][b] += t[b] * l.e[b];
        }
        f.dir_sum[k] = f.dir_sum[k] + dir * (t[1] * l.e[1]);
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quadrature converges on the analytic integral: a disc-free
    /// check — a 2a × 2a square under the ear at height h, no air
    /// (low band, short range) — against a fine brute-force grid.
    #[test]
    fn quadrature_matches_brute_force() {
        let shape = AmbientShape::GroundArea { min: (-20.0, -20.0), max: (20.0, 20.0), z: 0.0, holes: vec![] };
        for h in [1.6f32, 10.0, 60.0] {
            let eye = Vec3::new(3.0, -2.0, h);
            let mut buf = Vec::new();
            leaves(&shape, eye, &mut buf);
            let q: f32 = buf.iter().map(|l| l.e[0]).sum();
            let n = 800;
            let dx = 40.0 / n as f32;
            let mut brute = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let x = -20.0 + (i as f32 + 0.5) * dx;
                    let y = -20.0 + (j as f32 + 0.5) * dx;
                    let p = Vec3::new(x, y, 0.0);
                    let r = (p - eye).length();
                    let a = air_attenuation(r)[0];
                    brute += (dx * dx * a * a / (r * r)) as f64;
                }
            }
            let rel = (q as f64 - brute).abs() / brute;
            assert!(rel < 0.05, "h={h}: quadrature {q} vs brute {brute} ({rel})");
            assert!(buf.len() < 600, "leaf count stays small: {}", buf.len());
        }
    }

    /// Holes carve their area out exactly.
    #[test]
    fn holes_remove_their_emitters() {
        let full = AmbientShape::GroundArea { min: (0.0, 0.0), max: (10.0, 10.0), z: 0.0, holes: vec![] };
        let holed = AmbientShape::GroundArea {
            min: (0.0, 0.0),
            max: (10.0, 10.0),
            z: 0.0,
            holes: vec![((0.0, 0.0), (5.0, 10.0))],
        };
        let far = Vec3::new(5.0, 5.0, 2000.0);
        let e = |s: &AmbientShape| {
            let mut b = Vec::new();
            leaves(s, far, &mut b);
            b.iter().map(|l| l.e[0]).sum::<f32>()
        };
        let r = e(&holed) / e(&full);
        assert!((r - 0.5).abs() < 0.01, "half the area, half the energy: {r}");
    }
}
