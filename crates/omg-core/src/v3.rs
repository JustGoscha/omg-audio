//! Engine v3: one listener tracer for everything (see ENGINE_V3.md).
//!
//! `n` rays leave the listener and bounce up to `x` times. At every
//! bounce the surface connects to the sources it can see (next-event
//! estimation), weighted by how diffuse it is, what it absorbs and
//! every wall the connection passes through. Specular chains also count
//! when they run into a source, whose sphere grows with path length so
//! distant sources are found as often as near ones — and every hit is
//! priced from the source's real position. The direct path is one
//! deterministic segment.
//!
//! Units: a unit source gives free-field energy 1/d² at distance d (tap
//! amplitude 1/d). All energies are per band. Taps come out in WORLD
//! frame, unit source gain; the caller rotates and scales.

use std::collections::HashMap;

use crate::material::{air_attenuation, Material};
use crate::mesh::{Mesh, SegHit};
use crate::params::{ReverbParams, Tap};
use crate::paths::segment_transmission;
use crate::rng::Rng;
use crate::scene::{AcousticGeometry, GeomHit, WithPanels};
use crate::tracer::{estimate_reverb, Echogram, BIN_DT, MAX_TIME};
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

/// Early window after the free-field arrival: binned into taps.
pub const EARLY_S: f32 = 0.080;
/// Early delay bin width.
const DELAY_BIN_S: f32 = 0.002;
/// Direction cells for early binning.
const N_CELLS: usize = 64;
/// Taps kept per source (strongest early bins).
pub const MAX_EARLY_TAPS: usize = 24;
/// Distances below this are clamped (a source on a wall is not infinite).
const MIN_DIST: f32 = 0.3;
const EPS: f32 = 1e-3;
/// v2 echogram units: energy per 0.5 m receiver sphere = true / 16.
const LEGACY_UNITS: f32 = 1.0 / 16.0;
/// Tap key of the direct path; early bins start above it.
pub const KEY_DIRECT: u32 = 1;
const KEY_EARLY_BASE: u32 = 16;

/// Geometry for v3: the world mesh plus overlay boxes (door leaves,
/// panes, furniture) with their own materials.
pub struct World<'a> {
    pub mesh: &'a Mesh,
    pub panels: &'a [(Vec3, Vec3, Material)],
}

impl World<'_> {
    fn raycast(&self, p: Vec3, d: Vec3) -> Option<GeomHit> {
        WithPanels { base: self.mesh, panels: self.panels }.raycast_hit(p, d)
    }

    /// Per-band amplitude of the segment a→b: mesh walls by their real
    /// thickness, overlay boxes once each.
    pub fn transmission(&self, a: Vec3, b: Vec3, hits: &mut Vec<SegHit>) -> [f32; NBANDS] {
        let mut g = segment_transmission(self.mesh, a, b, hits);
        for (mn, mx, m) in self.panels {
            if segment_crosses_box(a, b, *mn, *mx) {
                for band in 0..NBANDS {
                    g[band] *= m.transmission[band];
                }
            }
        }
        g
    }
}

fn segment_crosses_box(a: Vec3, b: Vec3, mn: Vec3, mx: Vec3) -> bool {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for axis in 0..3 {
        let (da, pa, lo, hi) = (d.get(axis), a.get(axis), mn.get(axis), mx.get(axis));
        if da.abs() < 1e-9 {
            if pa < lo || pa > hi {
                return false;
            }
        } else {
            let (mut ta, mut tb) = ((lo - pa) / da, (hi - pa) / da);
            if ta > tb {
                core::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return false;
            }
        }
    }
    // strictly inside the segment: an endpoint ON a box face is not a crossing
    t1 > 1e-4 && t0 < 1.0 - 1e-4
}

/// One source for a trace. `id` keys its persistent state across ticks.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub id: u32,
    pub pos: Vec3,
    /// Relative loudness (gain²) for importance-sampled next-event
    /// connections; 1 for unit sources.
    pub weight: f32,
}

/// Compute budget: `rays` from the listener, each up to `bounces`
/// reflections. `k`: source sphere radius per meter of path.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub rays: u32,
    pub bounces: u32,
    pub k: f32,
    /// EMA weight of a fresh tick (1 = no smoothing).
    pub ema: f32,
    /// Next-event connections per bounce. With more sources than this,
    /// each bounce samples this many by importance (loudness reaching
    /// the hit point) and reweights — cost per bounce stays flat as the
    /// source count grows. 0 = connect to every source.
    pub nee: usize,
    /// LATE family: rays that keep bouncing until the path is MAX_TIME
    /// long or has lost 60 dB, counting only paths with MORE than
    /// `bounces` reflections (the early family counts the rest — every
    /// path exactly once). This is the reverb tail, traced.
    pub late_rays: u32,
    /// EMA weight of a fresh tick for the late echogram (the tail is a
    /// slow statistic: few late rays per tick, averaged over many).
    pub late_ema: f32,
    /// Next-event connections per bounce for the late family (the tail
    /// is a sum of many weak bounces: one sampled source each is plenty).
    pub late_nee: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self { rays: 1024, bounces: 4, k: 0.06, ema: 0.35, nee: 4, late_rays: 256, late_ema: 0.15, late_nee: 1 }
    }
}

/// Hard cap on late-family segments per ray (the time and energy limits
/// end it first in any real room).
const LATE_MAX_SEGMENTS: u32 = 600;
/// Late rays stop once they have lost this much (60 dB).
const LATE_FLOOR: f32 = 1e-6;

/// What one source sounds like at the listener (world frame, unit gain).
#[derive(Clone, Debug)]
pub struct SourceField {
    pub taps: Vec<Tap>,
    pub reverb: ReverbParams,
    /// Direction the late energy arrives from, and how one-sided it is.
    pub late_dir: [f32; 3],
    pub aniso: f32,
    /// This tick's total reflected energy (true units, unsmoothed): all
    /// NEE and sphere contributions, early and late.
    pub reflected: [f32; NBANDS],
}

#[derive(Clone, Copy)]
struct Bin {
    e: [f32; NBANDS],
    /// Σ E_mid · delay, Σ E_mid · dir
    ed: f32,
    dir: Vec3,
}

impl Default for Bin {
    fn default() -> Self {
        Self { e: [0.0; NBANDS], ed: 0.0, dir: Vec3::new(0.0, 0.0, 0.0) }
    }
}

impl Bin {
    fn add(&mut self, e: [f32; NBANDS], delay: f32, dir: Vec3) {
        for b in 0..NBANDS {
            self.e[b] += e[b];
        }
        self.ed += e[1] * delay;
        self.dir = self.dir + dir * e[1];
    }
}

#[derive(Default)]
struct SourceState {
    early: HashMap<u32, Bin>,
    echo: Option<Echogram>,
}

/// Fresh per-tick accumulators for one source.
struct Acc {
    early: HashMap<u32, Bin>,
    echo: Echogram,
    t_free: f32,
    total: [f32; NBANDS],
}

/// The tracer with its persistent per-source state (EMA across ticks).
pub struct Tracer {
    rng: Rng,
    cells: Vec<Vec3>,
    state: HashMap<u32, SourceState>,
    hits: Vec<SegHit>,
}

impl Tracer {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            cells: fibonacci_sphere(N_CELLS),
            state: HashMap::new(),
            hits: Vec::new(),
        }
    }

    /// Forget a source's history (it moved discontinuously, or left).
    pub fn reset(&mut self, id: u32) {
        self.state.remove(&id);
    }

    /// Trace once from `listener` and return each source's field, in
    /// the order of `sources`.
    pub fn trace(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
    ) -> Vec<SourceField> {
        let n = budget.rays.max(1);
        let mut accs: Vec<Acc> = sources
            .iter()
            .map(|s| Acc {
                early: HashMap::new(),
                echo: Echogram::new(),
                t_free: (s.pos - listener).length() / SPEED_OF_SOUND,
                total: [0.0; NBANDS],
            })
            .collect();

        // 1. direct: one exact segment per source
        let mut direct: Vec<Option<Tap>> = Vec::with_capacity(sources.len());
        for s in sources {
            let to = s.pos - listener;
            let d = to.length().max(MIN_DIST);
            let t = world.transmission(listener, s.pos, &mut self.hits);
            let air = air_attenuation(d);
            let gains: [f32; NBANDS] = core::array::from_fn(|b| t[b] * air[b] / d);
            direct.push(if gains.iter().any(|&g| g > 1e-7) {
                let dir = to * (1.0 / to.length().max(1e-6));
                Some(Tap { key: KEY_DIRECT, delay_s: d / SPEED_OF_SOUND, dir: [dir.x, dir.y, dir.z], gains })
            } else {
                None
            });
        }

        // 2. rays: stratified (Fibonacci) and randomly rotated per tick
        let rot = random_rotation(&mut self.rng);
        let primary = fibonacci_sphere(n as usize);
        let mut picks: Vec<(usize, f32)> = Vec::new();
        let mut imp: Vec<f32> = Vec::new();
        // Two families, disjoint by reflection count, so every path is
        // counted exactly once: EARLY rays count paths with 1..=bounces
        // reflections; LATE rays (fewer, long) count everything beyond.
        let late_primary = fibonacci_sphere(budget.late_rays as usize);
        let fams: [(&[Vec3], u32, u32, u32, usize); 2] = [
            (&primary, 1, budget.bounces, budget.bounces + 1, budget.nee),
            (&late_primary, budget.bounces + 1, u32::MAX, LATE_MAX_SEGMENTS, budget.late_nee),
        ];
        for (dirs, lo, hi, segments, nee) in fams {
        if dirs.is_empty() || lo > hi {
            continue;
        }
        let inv_n = 1.0 / dirs.len() as f32;
        let counts = |order: u32| order >= lo && order <= hi;
        for &d0 in dirs {
            let first = rot_apply(&rot, d0);
            let mut origin = listener;
            let mut dir = first;
            let mut thr = [1.0f32; NBANDS];
            let mut length = 0.0f32;
            // reflections so far, and whether the last one was specular
            // (only then do sphere hits count — NEE covered diffuse ones)
            let mut refl = 0u32;
            let mut specular = false;
            for _seg in 0..segments {
                let hit = world.raycast(origin, dir);
                let t_hit = hit.as_ref().map_or(f32::MAX, |h| h.t);

                // 3. specular chains into a source sphere, priced from the
                // source's REAL position
                if specular && counts(refl) {
                    for (si, s) in sources.iter().enumerate() {
                        let to = s.pos - origin;
                        let along = to.dot(dir);
                        if along <= 0.0 || along >= t_hit {
                            continue;
                        }
                        let seg = to.length();
                        let l_s = (length + seg).max(MIN_DIST);
                        let r = budget.k * l_s;
                        let closest = (origin + dir * along - s.pos).length();
                        if closest > r {
                            continue;
                        }
                        let t = world.transmission(origin, s.pos, &mut self.hits);
                        let air = air_attenuation(l_s);
                        let w = 4.0 * inv_n / (budget.k * budget.k * l_s * l_s);
                        let e: [f32; NBANDS] =
                            core::array::from_fn(|b| thr[b] * t[b] * t[b] * air[b] * air[b] * w);
                        deposit(&mut accs[si], &self.cells, e, l_s / SPEED_OF_SOUND, first);
                    }
                }

                let Some(hit) = hit else { break };
                let p = origin + dir * t_hit;
                length += t_hit;
                if length / SPEED_OF_SOUND > MAX_TIME {
                    break;
                }
                let mat = hit.material;
                let normal = hit.normal; // against the incoming ray
                let keep: [f32; NBANDS] = core::array::from_fn(|b| {
                    (1.0 - mat.absorption[b] - mat.transmission[b] * mat.transmission[b]).max(0.0)
                });

                // 2. next-event estimation: the diffuse share of this
                // surface, lit by every source it can reach
                if mat.scattering > 0.0 && counts(refl + 1) {
                    let off = p + normal * EPS;
                    // which sources, and with what weight: all of them, or
                    // `nee` draws ∝ weight / distance² (unbiased: 1/(m·p))
                    picks.clear();
                    if nee == 0 || sources.len() <= nee {
                        picks.extend((0..sources.len()).map(|i| (i, 1.0f32)));
                    } else {
                        imp.clear();
                        let mut sum = 0.0f32;
                        for s in sources {
                            let d2 = (s.pos - off).dot(s.pos - off) + 1.0;
                            let w = s.weight.max(1e-12) / d2;
                            sum += w;
                            imp.push(sum);
                        }
                        let m = nee;
                        for _ in 0..m {
                            let u = self.rng.next_f32() * sum;
                            let i = imp.partition_point(|&c| c < u).min(sources.len() - 1);
                            let pi = (imp[i] - if i > 0 { imp[i - 1] } else { 0.0 }) / sum;
                            picks.push((i, 1.0 / (m as f32 * pi.max(1e-12))));
                        }
                    }
                    for &(si, pw) in picks.iter() {
                        let s = &sources[si];
                        let to = s.pos - off;
                        let r = to.length();
                        if r < 1e-4 {
                            continue;
                        }
                        let cos_in = to.dot(normal) / r;
                        if cos_in <= 0.0 {
                            continue; // source behind this surface
                        }
                        let t = world.transmission(off, s.pos, &mut self.hits);
                        let rr = r.max(MIN_DIST);
                        let air = air_attenuation(length + r);
                        let w = pw * mat.scattering * cos_in * 4.0 * inv_n / (rr * rr);
                        let e: [f32; NBANDS] = core::array::from_fn(|b| {
                            thr[b] * keep[b] * t[b] * t[b] * air[b] * air[b] * w
                        });
                        deposit(&mut accs[si], &self.cells, e, (length + r) / SPEED_OF_SOUND, first);
                    }
                }

                // 4. continue: transmit (importance-floored) or reflect
                let mut p_t = 0.0f32;
                for b in 0..NBANDS {
                    p_t = p_t.max(mat.transmission[b] * mat.transmission[b]);
                }
                let pt = if p_t > 1e-5 { p_t.max(0.02).min(0.5) } else { 0.0 };
                if pt > 0.0 && self.rng.next_f32() < pt {
                    for b in 0..NBANDS {
                        thr[b] *= mat.transmission[b] * mat.transmission[b] / pt;
                    }
                    origin = p + dir * (2.0 * EPS);
                    continue; // straight on; `specular` unchanged
                }
                for b in 0..NBANDS {
                    thr[b] *= keep[b] / (1.0 - pt);
                }
                if thr.iter().all(|&e| e < LATE_FLOOR) {
                    break;
                }
                refl += 1;
                if self.rng.next_f32() < mat.scattering {
                    dir = (normal + self.rng.unit_sphere()).normalize();
                    if dir.dot(normal) < 1e-3 {
                        dir = normal;
                    }
                    specular = false;
                } else {
                    dir = dir - normal * (2.0 * dir.dot(normal));
                    specular = true;
                }
                origin = p + normal * EPS;
            }
        }
        }

        // 5. smooth over ticks and build each source's field
        let mut out = Vec::with_capacity(sources.len());
        for (si, s) in sources.iter().enumerate() {
            let acc = core::mem::replace(
                &mut accs[si],
                Acc { early: HashMap::new(), echo: Echogram::new(), t_free: 0.0, total: [0.0; NBANDS] },
            );
            let st = self.state.entry(s.id).or_default();
            let a = budget.ema.clamp(0.0, 1.0);
            let fresh_start = st.echo.is_none();
            let alpha = if fresh_start { 1.0 } else { a };
            // early bins
            let mut keys: Vec<u32> = st.early.keys().copied().collect();
            for k in acc.early.keys() {
                if !st.early.contains_key(k) {
                    keys.push(*k);
                }
            }
            for k in keys {
                let f = acc.early.get(&k).copied().unwrap_or_default();
                let cur = st.early.entry(k).or_default();
                for b in 0..NBANDS {
                    cur.e[b] += alpha * (f.e[b] - cur.e[b]);
                }
                cur.ed += alpha * (f.ed - cur.ed);
                cur.dir = cur.dir + (f.dir - cur.dir) * alpha;
            }
            st.early.retain(|_, b| b.e.iter().any(|&e| e > 1e-12));
            // late echogram: a slower statistic, a slower average
            match &mut st.echo {
                Some(e) => e.ema(&acc.echo, budget.late_ema.clamp(0.0, 1.0)),
                None => st.echo = Some(acc.echo),
            }

            let mut taps: Vec<Tap> = Vec::new();
            if let Some(t) = direct[si] {
                taps.push(t);
            }
            let mut early: Vec<(&u32, &Bin)> = st.early.iter().collect();
            early.sort_by(|x, y| y.1.e[1].total_cmp(&x.1.e[1]));
            for (k, b) in early.into_iter().take(MAX_EARLY_TAPS) {
                if b.e[1] <= 0.0 {
                    continue;
                }
                let dl = b.dir.length();
                let dir = if dl > 1e-9 { b.dir * (1.0 / dl) } else { self.cells[cell_of(*k)] };
                taps.push(Tap {
                    key: *k,
                    delay_s: b.ed / b.e[1],
                    dir: [dir.x, dir.y, dir.z],
                    gains: core::array::from_fn(|band| b.e[band].max(0.0).sqrt()),
                });
            }
            let echo = st.echo.as_ref().unwrap();
            let reverb = estimate_reverb(echo);
            let (late_dir, aniso) = echo.late_direction(EARLY_S);
            out.push(SourceField { taps, reverb, late_dir, aniso, reflected: acc.total });
        }
        out
    }
}

fn cell_of(key: u32) -> usize {
    ((key - KEY_EARLY_BASE) / 256) as usize % N_CELLS
}

/// Route one contribution: early bins within the window after the
/// free-field arrival, the echogram (relative to that arrival) always.
fn deposit(acc: &mut Acc, cells: &[Vec3], e: [f32; NBANDS], delay: f32, dir: Vec3) {
    if !e.iter().any(|&x| x > 0.0) {
        return;
    }
    for b in 0..NBANDS {
        acc.total[b] += e[b];
    }
    let rel = (delay - acc.t_free).max(0.0);
    let bin = (rel / BIN_DT) as usize;
    if bin < acc.echo.bins.len() {
        for b in 0..NBANDS {
            acc.echo.bins[bin][b] += e[b] * LEGACY_UNITS;
        }
        let m = e[1] * LEGACY_UNITS;
        acc.echo.dirs[bin][0] += dir.x * m;
        acc.echo.dirs[bin][1] += dir.y * m;
        acc.echo.dirs[bin][2] += dir.z * m;
    }
    if rel < EARLY_S {
        let mut best = 0usize;
        let mut bd = f32::MIN;
        for (i, c) in cells.iter().enumerate() {
            let d = c.dot(dir);
            if d > bd {
                bd = d;
                best = i;
            }
        }
        let dbin = (rel / DELAY_BIN_S) as u32;
        let key = KEY_EARLY_BASE + best as u32 * 256 + dbin.min(255);
        acc.early.entry(key).or_default().add(e, delay, dir);
    }
}

/// `n` nearly uniform unit vectors (golden-angle spiral).
pub fn fibonacci_sphere(n: usize) -> Vec<Vec3> {
    let golden = core::f32::consts::PI * (3.0 - 5f32.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden * i as f32;
            Vec3::new(r * phi.cos(), r * phi.sin(), z)
        })
        .collect()
}

/// Uniformly random rotation (from a random unit quaternion).
fn random_rotation(rng: &mut Rng) -> [[f32; 3]; 3] {
    let (u1, u2, u3) = (rng.next_f32(), rng.next_f32(), rng.next_f32());
    let tau = 2.0 * core::f32::consts::PI;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    let (x, y, z, w) = (a * (tau * u2).sin(), a * (tau * u2).cos(), b * (tau * u3).sin(), b * (tau * u3).cos());
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn rot_apply(m: &[[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}
