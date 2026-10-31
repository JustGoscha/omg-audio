//! Portals (`Tracer::portals`): a sounding room heard through its
//! openings, estimated from the sources' side.
//!
//! A listener's rays find a loud room behind a door by chance — through
//! the opening, round the jamb, into the room and on to a speaker — and
//! the rare one that does carries thousands of times the typical tick's
//! energy: the room's field outside comes in bursts (the club's bass
//! from the field south of the town: a tick at −33 dB among ticks at
//! −75). From the source's side the same paths are common: its sound
//! fills its room and leaks out of every opening. So for a source
//! enclosed in a room, rays from the source record what leaves each
//! opening (`Aperture`: a door's hole, a pane's), per direction and
//! delay — and the listener's walk, where it first crosses an opening,
//! takes what comes through it from there from the record.
//!
//! Every path is counted once: a path crossing an opening whose part
//! from its first crossing (from the listener) to the source has
//! reflected and been rough — scattered, bent round an edge, through a
//! wall or a leaf — is the record's; every other path is the walk's, as
//! before (`walk_ray` leaves the record's out). The record holds every
//! crossing of such source paths (any crossing may be a listener path's
//! first); the walk adds, at a ray's first crossing, its throughput times
//! the radiance the record holds there, the leaf (if the ray passed one
//! there) the record's.

use super::*;

/// A record's delay bins (s): what leaves an opening, from its emission.
pub(super) const PORTAL_BIN_S: f32 = 0.02;
pub(super) const PORTAL_BINS: usize = (MAX_TIME / PORTAL_BIN_S) as usize;
/// A record's running mean holds at most this many ticks (a source and
/// the world standing still): ~0.5 s at 60 ticks a second.
const PORTAL_MEMORY: f32 = 32.0;
/// A source ray under this share of what it set out with goes on only as
/// often as it carries (Russian roulette: the survivor carries the
/// rest) — unbiased; a ringing room's rays no longer all walk to its
/// three seconds' end.
const ROULETTE: f32 = 0.25;
/// Air over a path, per band (energy), by its length in `AIR_STEP_S` of
/// travel: what a record's bin and a walk's crossing bring together.
const AIR_STEP_S: f32 = 0.01;
pub(super) const AIR_STEPS: usize = (2.0 * MAX_TIME / AIR_STEP_S) as usize + 2;

/// `AIR_STEPS` entries: the energy air leaves over each length.
pub(super) fn air_table() -> Vec<[f32; NBANDS]> {
    (0..AIR_STEPS)
        .map(|i| {
            let a = air_attenuation(i as f32 * AIR_STEP_S * SPEED_OF_SOUND);
            a.map(|x| x * x)
        })
        .collect()
}

/// What leaves one side of an opening: per direction cell (`cells`) and
/// delay bin, the radiance there — energy per area and steradian, in the
/// walk's units (a unit source: 1/r² at r) — over the opening, the air
/// on the way left out (paid on the whole path where it is deposited:
/// air over two lengths is not air over one times air over the other).
pub(super) type Record = Vec<[f32; NBANDS]>;

/// A record's slot: `N_CELLS × PORTAL_BINS`, cell-major.
pub(super) fn slot(cell: usize, bin: usize) -> usize {
    cell * PORTAL_BINS + bin
}

/// One source's records, averaged while it and the world stand.
#[derive(Default)]
pub(super) struct PortalState {
    at: Option<Vec3>,
    sig: u64,
    n: f32,
    /// By (aperture id, side it leaves toward: 0 +axis, 1 −axis).
    pub(super) recs: HashMap<(u32, u8), Record>,
}

impl PortalState {
    /// This tick's record into the running mean — over again where the
    /// source moved or the world changed.
    pub(super) fn fold(&mut self, fresh: HashMap<(u32, u8), Record>, pos: Vec3, sig: u64) {
        if self.at.is_none_or(|a| (a - pos).length() > STILL_M) || self.sig != sig {
            self.recs.clear();
            self.n = 0.0;
            self.at = Some(pos);
            self.sig = sig;
        }
        let a = 1.0 / (self.n + 1.0);
        for (k, rec) in self.recs.iter_mut() {
            match fresh.get(k) {
                Some(f) => {
                    for (r, x) in rec.iter_mut().zip(f) {
                        for b in 0..NBANDS {
                            r[b] += a * (x[b] - r[b]);
                        }
                    }
                }
                None => {
                    for r in rec.iter_mut() {
                        for v in r.iter_mut() {
                            *v *= 1.0 - a;
                        }
                    }
                }
            }
        }
        for (k, f) in fresh {
            self.recs.entry(k).or_insert_with(|| f.iter().map(|x| x.map(|v| v * a)).collect());
        }
        self.n = (self.n + 1.0).min(PORTAL_MEMORY);
    }
}

/// The side of an opening a direction leaves toward (0: +axis).
pub(super) fn side_of(ap: &Aperture, d: Vec3) -> u8 {
    if d.get(ap.ax) > 0.0 {
        0
    } else {
        1
    }
}

/// The opening's area (across its axis).
fn area(ap: &Aperture) -> f32 {
    let e = ap.max - ap.min;
    (0..3).filter(|&k| k != ap.ax).map(|k| e.get(k).max(1e-3)).product()
}

/// ∫ max(ω·n, 0) dω over each direction cell, for n = +x, −x, +y, −y,
/// +z, −z (index 2·axis + side): what a cell's radiance passes through a
/// unit area facing n — the record's radiance is its flux over the
/// opening's area and this.
pub(super) fn cos_tables(cells: &[Vec3]) -> Vec<Vec<f32>> {
    let n = 1 << 17;
    let dirs = fibonacci_sphere(n);
    let dw = 4.0 * core::f32::consts::PI / n as f32;
    let mut tabs = vec![vec![0.0f32; cells.len()]; 6];
    for d in dirs {
        let c = cell_index(cells, d);
        for (k, v) in [d.x, d.y, d.z].into_iter().enumerate() {
            if v > 0.0 {
                tabs[2 * k][c] += v * dw;
            } else {
                tabs[2 * k + 1][c] -= v * dw;
            }
        }
    }
    tabs
}

/// What a source's paths leave through the openings this tick: `rays`
/// from `pos` (each a 4π/rays share of a unit source), walked as the
/// listener's are — reflected, scattered, bent round edges, through walls
/// and leaves by their draws — every crossing of a path that has
/// reflected and been rough recorded (`Record`).
#[allow(clippy::too_many_arguments)]
pub(super) fn trace_source(
    world: &World,
    edges: Option<&EdgeSet>,
    marks: &mut EdgeMarks,
    hits: &mut Vec<SegHit>,
    cells: &[Vec3],
    ctab: &[Vec<f32>],
    pos: Vec3,
    rays: u32,
    diffraction: bool,
    rng: &mut Rng,
) -> HashMap<(u32, u8), Record> {
    let aps = world.apertures;
    let mut out: HashMap<(u32, u8), Record> = HashMap::default();
    if aps.is_empty() || rays == 0 {
        return out;
    }
    let flux = 4.0 * core::f32::consts::PI / rays as f32;
    let record = |out: &mut HashMap<(u32, u8), Record>, a: Vec3, b: Vec3, length: f32, thr: [f32; NBANDS], dir: Vec3| {
        for ap in aps {
            let Some(x) = ap.crossed_by(a, b) else { continue };
            let l = length + (x - a).length();
            let bin = (l / SPEED_OF_SOUND / PORTAL_BIN_S) as usize;
            if bin >= PORTAL_BINS {
                continue;
            }
            let side = side_of(ap, dir);
            let cell = cell_index(cells, dir);
            let c = ctab[2 * ap.ax + side as usize][cell];
            if c < 1e-6 {
                continue;
            }
            let k = 1.0 / (area(ap) * c);
            let rec = out.entry((ap.id, side)).or_insert_with(|| vec![[0.0; NBANDS]; cells.len() * PORTAL_BINS]);
            let r = &mut rec[slot(cell, bin)];
            for b in 0..NBANDS {
                r[b] += thr[b] * k;
            }
        }
    };
    for _ in 0..rays {
        let mut dir = rng.unit_sphere();
        let mut origin = pos;
        let mut thr = [flux; NBANDS];
        let mut length = 0.0f32;
        let mut refl = 0u32;
        let mut rough = false;
        let mut bends = 0u32;
        let mut last_edge: Option<usize> = None;
        for _ in 0..LATE_MAX_SEGMENTS + MAX_BENDS {
            if world.in_earth(origin) {
                break;
            }
            let hit = world.raycast(origin, dir);
            let t_hit = hit.as_ref().map_or(f32::MAX, |h| h.t);
            let edge = if diffraction && bends < MAX_BENDS {
                let reach = if t_hit.is_finite() { t_hit } else { 400.0 };
                edges.and_then(|es| es.nearest_in(&mut *marks, origin, dir, reach, last_edge))
            } else {
                None
            };
            let t_seg = edge.map_or(t_hit, |(te, _, _, _)| te);
            if refl >= 1 && rough {
                record(&mut out, origin, origin + dir * t_seg.min(1e3), length, thr, dir);
            }
            if let Some((te, a, u, ei)) = edge {
                last_edge = Some(ei);
                let (eps, w) = sample_bend(rng, a);
                origin = origin + dir * te;
                length += te;
                dir = rotate_about(dir, u, eps);
                for b in 0..NBANDS {
                    thr[b] *= w[b];
                }
                rough = true;
                bends += 1;
                continue;
            }
            let Some(hit) = hit else { break };
            let p = origin + dir * t_hit;
            length += t_hit;
            if length / SPEED_OF_SOUND > MAX_TIME {
                break;
            }
            let mat = hit.material;
            let normal = hit.normal;
            let keep: [f32; NBANDS] = core::array::from_fn(|b| (1.0 - mat.absorption[b] - mat.transmission[b] * mat.transmission[b]).max(0.0));
            // through (the walk's draw: importance-floored)
            let p_t = mat.transmission.iter().fold(0.0f32, |m, &t| m.max(t * t));
            let pure = p_t > 1e-5 && keep.iter().all(|&k| k < 1e-6);
            let pt = if pure {
                1.0
            } else if p_t > 1e-5 {
                p_t.max(0.02).min(0.5)
            } else {
                0.0
            };
            if pt > 0.0 && rng.next_f32() < pt {
                let (t2, beyond) = world.through(p, dir, &mat, &mut *hits);
                for b in 0..NBANDS {
                    thr[b] *= t2[b] / pt;
                }
                rough = true;
                // through a leaf in an opening: what leaves it, the leaf's
                // part paid
                if refl >= 1 {
                    record(&mut out, p, beyond, length, thr, dir);
                }
                origin = beyond;
                continue;
            }
            for b in 0..NBANDS {
                thr[b] *= keep[b] / (1.0 - pt);
            }
            let top = thr.iter().fold(0.0f32, |m, &e| m.max(e));
            if top < LATE_FLOOR * flux {
                break;
            }
            if top < ROULETTE * flux {
                let p = top / (ROULETTE * flux);
                if rng.next_f32() >= p {
                    break;
                }
                for t in &mut thr {
                    *t /= p;
                }
            }
            refl += 1;
            if rng.next_f32() < mat.scattering {
                dir = (normal + rng.unit_sphere()).normalize();
                if dir.dot(normal) < 1e-3 {
                    dir = normal;
                }
                rough = true;
            } else {
                dir = dir - normal * (2.0 * dir.dot(normal));
            }
            origin = p + normal * EPS;
        }
    }
    out
}

/// A walk's crossing (`walk_ray`, `ChunkOut::portals`) deposited for one
/// source: the record's cell `cell` of what leaves the opening toward the
/// walk's side, every delay bin, × `w` (the walk's throughput there, its
/// share 4π/n) × the air over the whole path (`air`: `air_table`),
/// arriving from `dir` `delay0` after emission plus the bin's.
#[allow(clippy::too_many_arguments)]
pub(super) fn deposit_portal(acc: &mut Acc, cells: &[Vec3], air: &[[f32; NBANDS]], rec: &Record, cell: usize, w: [f32; NBANDS], delay0: f32, dir: Vec3) {
    let half = 0.5 * PORTAL_BIN_S;
    for bin in 0..PORTAL_BINS {
        let r = rec[slot(cell, bin)];
        if r[1] <= 0.0 && r[0] <= 0.0 && r[2] <= 0.0 {
            continue;
        }
        let delay = delay0 + (bin as f32 + 0.5) * PORTAL_BIN_S;
        let a = air[((delay / AIR_STEP_S).round() as usize).min(AIR_STEPS - 1)];
        let e: [f32; NBANDS] = core::array::from_fn(|b| w[b] * r[b] * a[b]);
        deposit_spread(acc, cells, e, delay, dir, half);
    }
}
