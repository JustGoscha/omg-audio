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
/// A record is the mean of this many ticks' rays (a source and the world
/// standing still; then no more are traced): ~0.5 s at 60 ticks a second.
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

/// ... and once its record holds this many ticks' rays: a record started
/// over (a door moving starts every one over each tick) holds one tick's
/// few rays in its direction cells and delay bins, a heavy tail of its
/// own — the walk keeps those paths until it fills (a door's swing: the
/// treble clicked −32 dB under the tone, −45 without portals).
const PORTAL_MIN_TICKS: f32 = 8.0;

/// One source's records, averaged while it and the world stand.
#[derive(Default)]
pub(super) struct PortalState {
    at: Option<Vec3>,
    sig: u64,
    n: f32,
    /// By (aperture id, side it leaves toward: 0 +axis, 1 −axis): the sum
    /// of the `n` ticks' records (their mean: × `scale`).
    pub(super) recs: HashMap<(u32, u8), Record>,
    /// The crossings the `n` ticks recorded there.
    counts: HashMap<(u32, u8), f32>,
    /// The bounces past the openings of the `n` ticks traced, and the
    /// running sum of what they carry (`weight`), for drawing them.
    vpls: Vec<Vpl>,
    cum: Vec<f64>,
    /// Where the source was and the world's signature, last tick.
    seen: Option<(Vec3, u64)>,
}

/// A tick's trace of one source: per record, what its rays left in which
/// slots (a few hundred of a record's 9,600: a tick's records in full
/// were allocated, zeroed and averaged in whole, per part of the rays),
/// and how many crossings.
pub(super) type Fresh = HashMap<(u32, u8), (Vec<(u32, [f32; NBANDS])>, u32)>;

/// A source ray's diffuse bounce connected to the ears (light tracing):
/// what it brings per band, when, and from where (from the ears).
pub(super) type Conn = ([f32; NBANDS], f32, Vec3);

/// A source ray's diffuse bounce past an opening it crossed with a rough
/// past (light tracing's vertex), kept while the source and the world
/// stand (`PortalState::vpls`) and connected to the ears every tick
/// (`connect`): off its surface, its facing, what its diffuse share
/// carries on (its throughput × what the surface keeps × scattering), its
/// path from the source, and the opening the path crossed last (index).
#[derive(Clone, Copy)]
pub(super) struct Vpl {
    p: Vec3,
    normal: Vec3,
    e: [f32; NBANDS],
    length: f32,
    ap: u8,
}

impl PortalState {
    /// This tick's record into the sum — over again where the source moved
    /// or the world changed. (No more are traced once it holds
    /// `PORTAL_MEMORY` ticks: the sum over `n` is their mean.)
    pub(super) fn fold(&mut self, fresh: Fresh, vpls: Vec<Vpl>, pos: Vec3, sig: u64) {
        if self.resets(pos, sig) {
            self.recs.clear();
            self.counts.clear();
            self.vpls.clear();
            self.cum.clear();
            self.n = 0.0;
            self.at = Some(pos);
            self.sig = sig;
        }
        let mut acc = self.cum.last().copied().unwrap_or(0.0);
        for v in &vpls {
            acc += weight(v) as f64;
            self.cum.push(acc);
        }
        self.vpls.extend(vpls);
        for (k, (deps, crossings)) in fresh {
            *self.counts.entry(k).or_insert(0.0) += crossings as f32;
            let rec = self.recs.entry(k).or_insert_with(|| vec![[0.0; NBANDS]; N_CELLS * PORTAL_BINS]);
            for (at, e) in deps {
                let r = &mut rec[at as usize];
                for b in 0..NBANDS {
                    r[b] += e[b];
                }
            }
        }
        self.n += 1.0;
    }

    /// What the sums are scaled by to the ticks' mean.
    pub(super) fn scale(&self) -> f32 {
        if self.n > 0.0 {
            1.0 / self.n
        } else {
            0.0
        }
    }

    /// Does a tick traced with the source at `pos` in world `sig` start
    /// the records over (the source moved, the world changed)?
    pub(super) fn resets(&self, pos: Vec3, sig: u64) -> bool {
        self.at.is_none_or(|a| (a - pos).length() > STILL_M) || self.sig != sig
    }

    /// Are rays still wanted: the records start over, or don't yet hold
    /// `PORTAL_MEMORY` ticks? Full, they are the mean of that many and
    /// stay as they are while the source and the world stand — more rays
    /// would only make them a running mean that forgets.
    pub(super) fn wants_rays(&self, pos: Vec3, sig: u64) -> bool {
        self.resets(pos, sig) || self.n < PORTAL_MEMORY
    }

    /// The ticks its records hold for a tick traced with the source at
    /// `pos` in world `sig` (none where they start over).
    pub(super) fn held(&self, pos: Vec3, sig: u64) -> f32 {
        if self.resets(pos, sig) {
            0.0
        } else {
            self.n
        }
    }

    /// Are the source and the world as they were last tick? Records
    /// started over while either moves would be the next tick's to start
    /// over again (no portal till they hold `PORTAL_MIN_TICKS`): rays
    /// traced then are thrown away — a door or a curtain moving started
    /// every enclosed source's over each tick, 2–5 ms a tick for nothing.
    pub(super) fn steady(&self, pos: Vec3, sig: u64) -> bool {
        self.seen.is_some_and(|(p, s)| (p - pos).length() <= STILL_M && s == sig)
    }

    /// This tick's source and world, for `steady` next tick.
    pub(super) fn see(&mut self, pos: Vec3, sig: u64) {
        self.seen = Some((pos, sig));
    }

    /// The bounces this tick's light tracing draws from, and how many
    /// ticks' rays left them (none where the records start over).
    pub(super) fn pool(&self, pos: Vec3, sig: u64) -> (&[Vpl], &[f64], f32) {
        if self.resets(pos, sig) {
            (&[], &[], 0.0)
        } else {
            (&self.vpls, &self.cum, self.n)
        }
    }

    /// The openings (bits by index in `aps`, the first 64) its rays cross
    /// often enough to be its portals (`PORTAL_MIN_CROSSINGS`, either side),
    /// its records holding `PORTAL_MIN_TICKS` ticks.
    pub(super) fn mask(&self, aps: &[Aperture]) -> u64 {
        let mut m = 0u64;
        if self.n < PORTAL_MIN_TICKS {
            return m;
        }
        for (k, ap) in aps.iter().enumerate().take(64) {
            let c = (self.counts.get(&(ap.id, 0)).copied().unwrap_or(0.0) + self.counts.get(&(ap.id, 1)).copied().unwrap_or(0.0)) * self.scale();
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
/// from `pos` (each a 4π/`all` share of a unit source: a part of the
/// tick's `all` rays, the parts' records added up — `merge`), walked as the
/// listener's are — reflected, scattered, bent round edges, through walls
/// and leaves by their draws — every crossing of a path that has
/// reflected and been rough recorded (`Record`), and every diffuse bounce
/// past such a crossing kept (`Vpl`).
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
    all: u32,
    diffraction: bool,
    rng: &mut Rng,
) -> (Fresh, Vec<Vpl>) {
    let aps = world.apertures;
    let mut out: Fresh = HashMap::default();
    let mut vpls: Vec<Vpl> = Vec::new();
    if aps.is_empty() || rays == 0 {
        return (out, vpls);
    }
    let flux = 4.0 * core::f32::consts::PI / all.max(1) as f32;
    // every crossing of a path that has reflected and been rough, into
    // the records (`own`: else the walk's); and the last opening crossed
    // (by index), with such a past or not: what follows it is the
    // source's to bring to the ears where it is one of its portals
    let record = |out: &mut Fresh, a: Vec3, b: Vec3, length: f32, thr: [f32; NBANDS], dir: Vec3, own: bool, last: &mut Option<(bool, usize)>| {
        let mut far = -1.0f32;
        for (k, ap) in aps.iter().enumerate() {
            let Some(x) = ap.crossed_by(a, b) else { continue };
            let dk = (x - a).length();
            if dk > far {
                far = dk;
                *last = Some((own, k));
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
            let (deps, n) = out.entry((ap.id, side)).or_default();
            *n += 1;
            deps.push((slot(cell, bin) as u32, thr.map(|t| t * k)));
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
        let mut last: Option<(bool, usize)> = None;
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
            // light tracing's vertex: past an opening crossed with a rough
            // past, this bounce's diffuse share — kept whichever way the
            // ray goes on (the draw below), connected to the ears later
            if let Some((true, k)) = last {
                if mat.scattering > 0.0 && k < 64 {
                    let e: [f32; NBANDS] = core::array::from_fn(|b| thr[b] * keep[b] * mat.scattering);
                    if e.iter().any(|&v| v > 0.0) {
                        vpls.push(Vpl { p: p + normal * EPS, normal, e, length, ap: k as u8 });
                    }
                }
            }
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
    (out, vpls)
}

/// What a bounce is drawn by: the most it carries in a band.
fn weight(v: &Vpl) -> f32 {
    v.e.iter().fold(0.0f32, |m, &x| m.max(x))
}

/// Light tracing's draws from a source's pool a tick, per ray it traces a
/// tick. Any number is unbiased; fewer cost less and spread more (the
/// club's light-traced bass in the canyon, per tick: 1 a ray cv 2.41 and
/// the largest tick 48× the mean, 4: 1.07 and 17×, 8: 0.80 and 9×; as
/// many as a tick's rays leave bounces past an opening — a dozen a ray —
/// cost 1.37 ms a tick in Godot's rooms, 8 a ray 0.46).
const DRAWS_PER_RAY: usize = 8;

/// Light tracing from a source's pool — the bounces `kept` from the ticks
/// before (`ticks` of them with this one) and this tick's `new` —
/// `DRAWS_PER_RAY` draws for each of the tick's `rays`, each connected
/// straight to the ears where they see it (nothing between, no opening crossed) when the
/// opening its path crossed last is one of the source's portals (`mask`):
/// a path whose first touch from the ears is a diffuse bounce, the walk
/// leaves to this (`walk_ray`'s `lt`). Drawn by what each carries,
/// evenly along their running sum (one random offset): a bounce carrying
/// many times the others' (a ray through a wall by the draw, × 50) is
/// drawn as often as it carries and brings its share every tick — drawn
/// uniformly, it came as all of a tick's energy now and then (a burst).
#[allow(clippy::too_many_arguments)]
pub(super) fn connect(world: &World, kept: &[Vpl], kept_cum: &[f64], new: &[Vpl], ticks: f32, rays: usize, mask: u64, ear: Vec3, rng: &mut Rng) -> Vec<Conn> {
    let n = kept.len() + new.len();
    let mut out: Vec<Conn> = Vec::new();
    if n == 0 || ticks <= 0.0 {
        return out;
    }
    // the kept bounces' running sum is kept with them; this tick's after it
    let base = kept_cum.last().copied().unwrap_or(0.0);
    let mut sum = base;
    let new_cum: Vec<f64> = new
        .iter()
        .map(|v| {
            sum += weight(v) as f64;
            sum
        })
        .collect();
    if sum <= 0.0 {
        return out;
    }
    let draws = (rays * DRAWS_PER_RAY).max(1);
    let step = sum / draws as f64;
    let mut u = rng.next_f32() as f64 * step;
    for _ in 0..draws {
        // the first bounce whose running sum passes u
        let v = if (u < base || new.is_empty()) && !kept.is_empty() {
            &kept[kept_cum.partition_point(|&c| c <= u).min(kept.len() - 1)]
        } else {
            &new[new_cum.partition_point(|&c| c <= u).min(new.len() - 1)]
        };
        u += step;
        if (mask >> v.ap) & 1 == 0 {
            continue;
        }
        let to = ear - v.p;
        let r = to.length();
        if r <= 1e-3 {
            continue;
        }
        let d = to * (1.0 / r);
        let cos_out = d.dot(v.normal);
        // the openings first: from inside the source's room every bounce
        // outside is an opening away, and a cast is the dear test
        if cos_out <= 0.0 || super::first_crossing(world.apertures, v.p, ear).is_some() || world.raycast(v.p, d).is_some_and(|h| h.t < r - 1e-3) {
            continue;
        }
        let air = air_attenuation(v.length + r);
        // its odds weight(v) / sum a draw, of `draws`, over `ticks` ticks
        let w = (sum / (draws as f64 * weight(v) as f64 * ticks as f64)) as f32;
        let g = w * cos_out / (core::f32::consts::PI * r * r);
        out.push((core::array::from_fn(|b| v.e[b] * g * air[b] * air[b]), (v.length + r) / SPEED_OF_SOUND, d * -1.0));
    }
    out
}

/// A part of a tick's trace into the rest of it: the records and their
/// crossings added up.
pub(super) fn merge(into: &mut Fresh, part: Fresh) {
    for (k, (deps, n)) in part {
        let (d0, n0) = into.entry(k).or_default();
        d0.extend(deps);
        *n0 += n;
    }
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
                    tot[side as usize] += rec.iter().map(|r| r[1]).sum::<f32>() * st.scale();
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
pub(super) fn convolve(acc: &mut Acc, cells: &[Vec3], air: &[[f32; NBANDS]], h: &Hist, rec: &Record, scale: f32, cell: usize) {
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
            let e: [f32; NBANDS] = core::array::from_fn(|b| w[b] * r[b] * scale * a[b]);
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
