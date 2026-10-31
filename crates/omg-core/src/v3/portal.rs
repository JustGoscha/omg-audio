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

/// An opening is a source's portal while its rays cross it this often a
/// tick (on average): its record well sampled. Where they seldom come (a
/// door across the town from the club), a record of a few rare crossings
/// is a heavy tail of its own — the walk keeps those paths.
pub(super) const PORTAL_MIN_CROSSINGS: f32 = 4.0;

/// One source's records, averaged while it and the world stand.
#[derive(Default)]
pub(super) struct PortalState {
    at: Option<Vec3>,
    sig: u64,
    n: f32,
    /// By (aperture id, side it leaves toward: 0 +axis, 1 −axis).
    pub(super) recs: HashMap<(u32, u8), Record>,
    /// Crossings a tick recorded there (the running mean's).
    counts: HashMap<(u32, u8), f32>,
}

/// A tick's trace of one source: the records and how many crossings
/// each holds.
pub(super) type Fresh = HashMap<(u32, u8), (Record, u32)>;

/// A source ray's diffuse bounce connected to the ears (light tracing):
/// what it brings per band, when, and from where (from the ears).
pub(super) type Conn = ([f32; NBANDS], f32, Vec3);

impl PortalState {
    /// This tick's record into the running mean — over again where the
    /// source moved or the world changed.
    pub(super) fn fold(&mut self, fresh: Fresh, pos: Vec3, sig: u64) {
        if self.at.is_none_or(|a| (a - pos).length() > STILL_M) || self.sig != sig {
            self.recs.clear();
            self.counts.clear();
            self.n = 0.0;
            self.at = Some(pos);
            self.sig = sig;
        }
        let a = 1.0 / (self.n + 1.0);
        for (k, c) in self.counts.iter_mut() {
            *c += a * (fresh.get(k).map_or(0.0, |f| f.1 as f32) - *c);
        }
        for (k, f) in &fresh {
            self.counts.entry(*k).or_insert(a * f.1 as f32);
        }
        for (k, rec) in self.recs.iter_mut() {
            match fresh.get(k).map(|f| &f.0) {
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
        for (k, (f, _)) in fresh {
            self.recs.entry(k).or_insert_with(|| f.iter().map(|x| x.map(|v| v * a)).collect());
        }
        self.n = (self.n + 1.0).min(PORTAL_MEMORY);
    }

    /// The openings (bits by index in `aps`, the first 64) its rays cross
    /// often enough to be its portals (`PORTAL_MIN_CROSSINGS`, either side).
    pub(super) fn mask(&self, aps: &[Aperture]) -> u64 {
        let mut m = 0u64;
        for (k, ap) in aps.iter().enumerate().take(64) {
            let c = self.counts.get(&(ap.id, 0)).copied().unwrap_or(0.0) + self.counts.get(&(ap.id, 1)).copied().unwrap_or(0.0);
            if c >= PORTAL_MIN_CROSSINGS {
                m |= 1 << k;
            }
        }
        m
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
    lis: Option<(Vec3, u64)>,
    rng: &mut Rng,
) -> (Fresh, Vec<Conn>) {
    let aps = world.apertures;
    let mut out: Fresh = HashMap::default();
    let mut conns: Vec<Conn> = Vec::new();
    if aps.is_empty() || rays == 0 {
        return (out, conns);
    }
    let flux = 4.0 * core::f32::consts::PI / rays as f32;
    // every crossing of a path that has reflected and been rough, into
    // the records (`own`: else the walk's); and whether the last opening
    // crossed (by index) makes what follows the source's to bring to the
    // ears — one of its portals (`lis`' mask), crossed with such a past
    let record = |out: &mut Fresh, a: Vec3, b: Vec3, length: f32, thr: [f32; NBANDS], dir: Vec3, own: bool, last: &mut Option<bool>| {
        let mut far = -1.0f32;
        for (k, ap) in aps.iter().enumerate() {
            let Some(x) = ap.crossed_by(a, b) else { continue };
            let dk = (x - a).length();
            if dk > far {
                far = dk;
                *last = Some(own && lis.is_some_and(|(_, m)| k < 64 && (m >> k) & 1 == 1));
            }
            if !own {
                continue;
            }
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
            let (rec, n) = out.entry((ap.id, side)).or_insert_with(|| (vec![[0.0; NBANDS]; cells.len() * PORTAL_BINS], 0));
            *n += 1;
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
        // the last opening crossed: the ears' share past it (`record`)
        let mut last: Option<bool> = None;
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
            record(&mut out, origin, origin + dir * t_seg.min(1e3), length, thr, dir, refl >= 1 && rough, &mut last);
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
                record(&mut out, p, beyond, length, thr, dir, refl >= 1, &mut last);
                origin = beyond;
                continue;
            }
            // light tracing: past one of its portals, crossed with a rough
            // past, this bounce's diffuse share straight to the ears where
            // they see it (nothing between, no opening crossed): a path
            // whose first touch from the ears is a diffuse bounce, the
            // walk leaves to this (`walk_ray`'s `lt`)
            if let (Some((ear, _)), Some(true)) = (lis, last) {
                if mat.scattering > 0.0 {
                    let off = p + normal * EPS;
                    let to = ear - off;
                    let r = to.length();
                    if r > 1e-3 {
                        let d = to * (1.0 / r);
                        let cos_out = d.dot(normal);
                        if cos_out > 0.0 && world.raycast(off, d).is_none_or(|h| h.t >= r - 1e-3) && super::first_crossing(aps, off, ear).is_none() {
                            let air = air_attenuation(length + r);
                            let g = mat.scattering * cos_out / (core::f32::consts::PI * r * r);
                            let e: [f32; NBANDS] = core::array::from_fn(|b| thr[b] * keep[b] * g * air[b] * air[b]);
                            conns.push((e, (length + r) / SPEED_OF_SOUND, d * -1.0));
                        }
                    }
                }
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
    (out, conns)
}

/// The openings a walk connects to (`walk_ray`'s next-event to openings):
/// per aperture (its index in `World::apertures`) what its records hold
/// leaving each side (mid band, summed over every portal source, cell
/// and bin) — the importance an opening is picked by.
pub(super) type PortalAps = Vec<(usize, [f32; 2])>;

/// The openings' records summed per side (`PortalAps`): each source's
/// where the opening is its portal (`masks`).
pub(super) fn portal_aps(aps: &[Aperture], sources: &[Source], masks: &[u64], states: &HashMap<u32, PortalState>) -> PortalAps {
    let mut out: PortalAps = Vec::new();
    for (k, ap) in aps.iter().enumerate().take(64) {
        let mut tot = [0.0f32; 2];
        for (s, &m) in sources.iter().zip(masks) {
            if (m >> k) & 1 == 0 {
                continue;
            }
            let Some(st) = states.get(&s.id) else { continue };
            for side in 0..2u8 {
                if let Some(rec) = st.recs.get(&(ap.id, side)) {
                    tot[side as usize] += rec.iter().map(|r| r[1]).sum::<f32>();
                }
            }
        }
        if tot[0] > 0.0 || tot[1] > 0.0 {
            out.push((k, tot));
        }
    }
    out
}

/// What each opening would bring a diffuse bounce at `p` (facing
/// `normal`), roughly — its record leaving toward `p` × its area × the
/// cosines at both ends / distance² (from its middle) — and their sum:
/// the odds it is picked by (the same at the bounce and wherever the
/// walk weighs a crossing against it).
pub(super) fn pick_weights(aps: &[Aperture], pa: &PortalAps, p: Vec3, normal: Vec3, out: &mut Vec<f32>) -> f32 {
    out.clear();
    let mut sum = 0.0f32;
    for &(k, tot) in pa {
        let ap = &aps[k];
        let mid = (ap.min + ap.max) * 0.5;
        let to = mid - p;
        let r2 = to.dot(to).max(1e-6);
        let d = to * (1.0 / r2.sqrt());
        let cos_p = d.dot(normal);
        let cos_q = d.get(ap.ax).abs();
        // the radiance toward p leaves toward p's side: along −d
        let side = side_of(ap, d * -1.0) as usize;
        let c = if cos_p > 0.0 { tot[side] * area(ap) * cos_p * cos_q / r2.max(1.0) } else { 0.0 };
        out.push(c);
        sum += c;
    }
    sum
}

/// A point on an opening's middle plane, uniform over it.
pub(super) fn sample_opening(ap: &Aperture, u1: f32, u2: f32) -> Vec3 {
    let (a, b) = ((ap.ax + 1) % 3, (ap.ax + 2) % 3);
    let mut q = [0.0f32; 3];
    q[ap.ax] = ap.mid;
    q[a] = ap.min.get(a) + u1 * (ap.max.get(a) - ap.min.get(a));
    q[b] = ap.min.get(b) + u2 * (ap.max.get(b) - ap.min.get(b));
    Vec3::new(q[0], q[1], q[2])
}

/// The solid-angle odds of reaching a point of `ap` `r` away along a
/// direction at `cos_q` to the opening's axis, the point drawn uniform
/// over its area: r²/(A·cos_q).
pub(super) fn opening_pdf(ap: &Aperture, r: f32, cos_q: f32) -> f32 {
    r * r / (area(ap) * cos_q.max(1e-6))
}

/// A tick's walk deposits at openings, gathered (`walk_rays`): per
/// aperture id, side and direction cell, the walk's weight by its delay
/// to the opening (`HIST_BIN_S` bins) and their weighted first
/// direction — then each portal source's record laid over it once
/// (`convolve`), not once per crossing.
pub(super) const HIST_BIN_S: f32 = 0.01;
pub(super) const HIST_BINS: usize = (MAX_TIME / HIST_BIN_S) as usize;

#[derive(Clone)]
pub(super) struct Hist {
    pub(super) w: Vec<[f32; NBANDS]>,
    pub(super) dir: Vec<Vec3>,
}

impl Hist {
    pub(super) fn new() -> Self {
        Self { w: vec![[0.0; NBANDS]; HIST_BINS], dir: vec![Vec3::new(0.0, 0.0, 0.0); HIST_BINS] }
    }
}

/// Every bin of `h` (the walk's weight by delay to the opening, cell
/// `cell`) × every bin of `rec` (what the opening's record holds in that
/// cell) × the air over the joined path, into `acc`: the echogram's bins
/// directly, the early part (`EARLY_S`) through `deposit`.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolve(acc: &mut Acc, cells: &[Vec3], air: &[[f32; NBANDS]], h: &Hist, rec: &Record, cell: usize) {
    let r0 = slot(cell, 0);
    let rbins = &rec[r0..r0 + PORTAL_BINS];
    let nz: Vec<usize> = (0..PORTAL_BINS).filter(|&j| rbins[j][1] > 0.0 || rbins[j][0] > 0.0 || rbins[j][2] > 0.0).collect();
    if nz.is_empty() {
        return;
    }
    let nbins = acc.echo.bins.len();
    for (i, w) in h.w.iter().enumerate() {
        if w[1] <= 0.0 && w[0] <= 0.0 && w[2] <= 0.0 {
            continue;
        }
        let dl = h.dir[i].length();
        let dir = if dl > 1e-12 { h.dir[i] * (1.0 / dl) } else { Vec3::new(0.0, 0.0, 1.0) };
        let t_i = (i as f32 + 0.5) * HIST_BIN_S;
        for &j in &nz {
            let delay = t_i + (j as f32 + 0.5) * PORTAL_BIN_S;
            let a = air[((delay / AIR_STEP_S).round() as usize).min(AIR_STEPS - 1)];
            let r = rbins[j];
            let e: [f32; NBANDS] = core::array::from_fn(|b| w[b] * r[b] * a[b]);
            let rel = (delay - acc.t_free).max(0.0);
            if rel < EARLY_S {
                deposit_spread(acc, cells, e, delay, dir, 0.5 * PORTAL_BIN_S);
                continue;
            }
            // the record's 20 ms bin over the echogram's two 10 ms ones
            for half in [-0.25f32, 0.25] {
                let bin = ((rel + half * PORTAL_BIN_S * 2.0) / BIN_DT) as usize;
                if bin >= nbins {
                    continue;
                }
                for b in 0..NBANDS {
                    acc.total[b] += 0.5 * e[b];
                    acc.echo.bins[bin][b] += 0.5 * e[b] * LEGACY_UNITS;
                }
                let m = 0.5 * e[1] * LEGACY_UNITS;
                acc.echo.dirs[bin][0] += dir.x * m;
                acc.echo.dirs[bin][1] += dir.y * m;
                acc.echo.dirs[bin][2] += dir.z * m;
            }
        }
    }
}
