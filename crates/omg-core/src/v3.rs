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

/// The tracer's maps hash with fixed keys: what is summed while walking
/// one comes out in the same order every run — same seed, same inputs,
/// the same field to the last bit.
type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;

use crate::material::{air_attenuation, Material};
use crate::mesh::{Mesh, SegHit};
use crate::params::{ReverbParams, Tap};
use crate::paths::segment_transmission_after;
use crate::rng::Rng;
use crate::scene::{AcousticGeometry, GeomHit, WithPanels};
use crate::tracer::{estimate_reverb, Echogram, BIN_DT, MAX_TIME};
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

/// A tick's wall-clock by part (ms; zeros in wasm, which has no clock
/// here): what each part of the trace costs, for the budgets that share
/// a tick out by what matters most.
#[derive(Clone, Copy, Debug, Default)]
pub struct TickCosts {
    /// Direct paths and their visibility, every source.
    pub direct: f32,
    /// Single-edge bends (the edges the ear sees, the search per hidden
    /// source).
    pub edges: f32,
    /// Two-edge strings (of `edges`).
    pub edges2: f32,
    /// The early and the late family's walk.
    pub early: f32,
    pub late: f32,
    /// Mirror chains: named and solved.
    pub chains: f32,
    /// The tick's estimates folded into each source's field.
    pub fold: f32,
    /// The world had just changed: this tick traced extra rays for the
    /// averages restarting (`Tracer::burst`) — its cost is the change's,
    /// not the quality's (a governor leaves it out).
    pub burst: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct Stopwatch(std::time::Instant);
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Stopwatch;

#[cfg(not(target_arch = "wasm32"))]
impl Stopwatch {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }
    fn ms(&self) -> f32 {
        self.0.elapsed().as_secs_f32() * 1e3
    }
}

#[cfg(target_arch = "wasm32")]
impl Stopwatch {
    fn start() -> Self {
        Self
    }
    fn ms(&self) -> f32 {
        0.0
    }
}

/// V3.3: the walk run elsewhere (the GPU kernel), folded back here.
pub mod fold;
/// V3.3: the seam an external walker (the GPU kernel) plugs into.
pub mod walker;

/// Early window after the free-field arrival: binned into taps.
pub const EARLY_S: f32 = 0.080;
/// Early delay bin width.
const DELAY_BIN_S: f32 = 0.002;
/// Direction cells for early binning.
const N_CELLS: usize = 64;
/// Taps kept per source (strongest early bins).
pub const MAX_EARLY_TAPS: usize = 24;
/// Discrete echoes. A COHERENT chain — every bounce a mirror, nothing
/// scattered or bent, still loud — that arrives after the early window,
/// or has more than `Budget::bounces` reflections, is one exact path:
/// the echo off a cliff across a canyon, the flutter between two bare
/// parallel walls. Such chains are not estimated by sphere hits (one
/// exact path per order; in a small room most rays around it hit the
/// floor or ceiling first) but named by the coherent rays that pass near
/// a source, cached, and solved exactly — image source, every leg
/// validated — each tick (`solve_chains`). Looked for up to this long
/// after the free-field arrival.
pub const ECHO_S: f32 = 1.0;
/// Discrete echo taps kept per source: enough for a small room's flutter
/// train (a ~12 ms period over a few hundred ms).
pub const MAX_ECHO_TAPS: usize = 32;
/// A chain is loud while its mirror throughput (absorption, scattering,
/// transmission along it) stays above this …
const COHERENT_FLOOR: f32 = 1e-2;
/// … up to this many reflections.
pub const MAX_COHERENT_REFL: u32 = 24;
/// A coherent ray passing within this × path length of a source names
/// its chain.
const CHAIN_FIND_K: f32 = 0.25;
/// Ticks a chain stays cached without being found again (every solve
/// re-validates it: this only forgets what the finders stopped seeing).
const CHAIN_TTL: u32 = 200;
/// A train is regrown from its chains this often (ticks): members that
/// failed to validate get another chance after the listener moved.
const TRAIN_REGROW_TICKS: u64 = 40;
/// A still chain (neither end moved) is re-solved this often all the same.
const CHAIN_RESOLVE_TICKS: u64 = 10;
/// A rejected chain (no exact path, or too weak to keep) is not tried
/// again for this many ticks — one with no exact path only while neither
/// end moves (`Tracer::held`).
const CHAIN_REJECT_TICKS: u64 = 40;
/// Chain-finder rays per tick: a sixteenth of the early budget (chains
/// stay cached; the finders only have to come across each one once).
fn chain_finders(budget: Budget) -> usize {
    (budget.rays as usize / 16).max(32)
}
/// Chains cached per source.
const MAX_CHAINS: usize = 128;
/// Salience: a solved chain plays as a tap when it is this many times
/// louder than the diffuse field in its slot — the incoherent energy
/// arriving from its side (±x, ±y, ±z) within ±`ECHO_SPAN`, per direction
/// cell of that side and scaled to `ECHO_SLOT_S` — and keeps playing down
/// to `ECHO_STAY` (hysteresis: a tap that drops and returns is a
/// crossfade gap). A canyon echo stands thousands of times above the
/// silence around it; flutter between bare walls a few times above a
/// live room's tail; anything else sinks into the reverb.
const ECHO_ENTER: f32 = 2.0;
const ECHO_STAY: f32 = 1.0;
const ECHO_SPAN: f32 = 0.02;
const ECHO_SLOT_S: f32 = 0.012;
/// Chains arriving within this of each other and within ~25° play as one
/// tap (the ear hears one arrival; the cap counts arrivals).
const ECHO_MERGE_S: f32 = 0.0015;
const ECHO_MERGE_COS: f32 = 0.9;
/// Distances below this are clamped (a source on a wall is not infinite).
const MIN_DIST: f32 = 0.3;
const EPS: f32 = 1e-3;
/// A segment from free space (`World::transmission_free`) counts every
/// crossing past this (m): a hundredth of the lift `EPS` off the surface
/// it left, above f32's rounding at the town's coordinates.
const FREE_START_M: f32 = 1e-5;
/// The thickest solid a ray goes through as one wall (`World::through`).
const MAX_WALL_M: f32 = 3.0;
/// v2 echogram units: energy per 0.5 m receiver sphere = true / 16.
const LEGACY_UNITS: f32 = 1.0 / 16.0;
/// Tap key of the direct path; early bins start above it.
pub const KEY_DIRECT: u32 = 1;

// ---- Engine v4 (ENGINE_V4.md): the traced response, rendered as traced.
// No early/late split in time: every arrival is kept, at any delay, and
// split by KIND — coherent arrivals (the direct path, edge bends, exact
// mirror chains) play as their own taps when they stand out; everything
// scattered is the dense response (time × band × energy vector), which
// the renderer convolves.
/// Longest path followed (s after emission): mountains answer late.
/// Past it a diffuse tail only decays; the renderer continues it along
/// its own measured slope (`omg_dsp::dense`).
pub const RESPONSE_MAX_S: f32 = 4.8;
/// The dense response's time resolution.
pub const DENSE_BIN_S: f32 = DELAY_BIN_S;
/// Coherent arrivals play as their own taps, the ones standing out most
/// first — energy over the dense energy arriving within
/// `DISTINCT_SPAN_S` of them (one thing louder than the crowd around
/// it) — up to the renderer's budget; the rest join the dense response,
/// still at their time. Only a rendering choice: the energy and the
/// delay are the same either way.
const DISTINCT_SPAN_S: f32 = 0.004;
/// Coherent taps per source at most (the renderer's cost).
pub const MAX_COHERENT_TAPS: usize = 48;
/// Moving, a tick's response is averaged in with at least this weight
/// (as much as the step leaves the arrivals in place: `fold_v4`); standing
/// still — nothing moved, the world as it was — it is a plain running
/// mean, the same truth sampled again.
const STILL_ALPHA_FLOOR: f32 = 0.005;
/// v6: moved less than this (m) since a still spell began, a listener
/// or a source stands still — paths change by less, nothing an energy
/// estimate in `DELAY_BIN_S` bins (69 cm of path) can tell.
const STILL_M: f32 = 0.01;
/// v4's history test: a window of the dense response restarts from the
/// fresh trace when one tick lands this many standard errors from its
/// mean (a door shut, a step into another space). Wide: the trace's
/// estimates are heavy-tailed (one strong late path lifts a tick), and a
/// smaller real change still shows as a drift (`HISTORY_DRIFT_Z`).
const HISTORY_JUMP_Z: f32 = 8.0;
/// …or when the ticks since the fresh trace last crossed the mean, taken
/// together, lie this many standard errors to one side of it (their
/// summed deviation over √ticks, two ticks at least: a CUSUM). A slow
/// drift is followed within a few ticks; noise sums to nothing, however
/// skewed (most ticks land below a mean the rare strong path lifts — a
/// count of ticks on one side would restart on that alone).
const HISTORY_DRIFT_Z: f32 = 4.0;
/// Weight of a tick in a window's measured scatter (`dense_var`).
const HISTORY_VAR_EMA: f32 = 0.1;
/// History windows: from the free-field arrival, the first 4 ms, then each
/// twice as long as the one before (4–8 ms, 8–16 ms, … past 4 s).
pub const HISTORY_WINDOWS: usize = 12;

/// History window `w`'s length (s): 4 ms, 4 ms, 8 ms, … doubling.
fn history_width(w: usize) -> f32 {
    0.004 * (1u32 << w.saturating_sub(1)) as f32
}

fn history_window(bin: usize) -> usize {
    // in 2 ms units
    let t = (bin as f32 * DENSE_BIN_S / 0.002) as usize;
    if t < 2 {
        0
    } else {
        ((usize::BITS - 1 - t.leading_zeros()) as usize).min(HISTORY_WINDOWS - 1)
    }
}

/// The dense response's target effective sample count per bin (its
/// relative noise ~ 1/√32 ≈ 18 %), `smooth_dense`.
const ESS_TARGET: f64 = 32.0;
const KEY_EARLY_BASE: u32 = 16;
/// Solved chains past the early window: one tap per chain (its identity
/// is the path's — exact delay, exact Doppler), keyed by the chain's hash
/// in [KEY_CHAIN_BASE, KEY_CHAIN_BASE + KEY_CHAIN_SPAN): far past the
/// edge keys (20 000 + an edge index), and below 2^24 — keys cross to the
/// web as f32.
const KEY_CHAIN_BASE: u32 = 9_000_000;
const KEY_CHAIN_SPAN: u32 = 7_000_000;

/// Geometry for v3: the world mesh plus overlay boxes (door leaves,
/// panes, furniture) with their own materials.
pub struct World<'a> {
    pub mesh: &'a Mesh,
    pub panels: &'a [(Vec3, Vec3, Material)],
    /// Ground level, when the world has solid earth beneath it. Below
    /// it, everything outside `pockets` is earth: rays that end up there
    /// die, and segments through it are blocked.
    pub grade: Option<f32>,
    /// Below-grade air volumes (a bunker): (min, max).
    pub pockets: &'a [(Vec3, Vec3)],
}

/// Amplitude through earth (a few meters of soil: nothing audible).
const EARTH: f32 = 1e-4;

impl<'a> World<'a> {
    /// A world with no earth (open below), no overlays added.
    pub fn new(mesh: &'a Mesh, panels: &'a [(Vec3, Vec3, Material)]) -> Self {
        Self { mesh, panels, grade: None, pockets: &[] }
    }

    /// Would a ray leaving `p` along `d` with nothing left to hit run
    /// into the earth instead of reaching open sky? Downward rays always
    /// do; from below grade, anything not climbing steeply does.
    fn escapes_into_earth(&self, p: Vec3, d: Vec3) -> bool {
        let Some(g) = self.grade else { return false };
        d.z < -1e-3 || (p.z < g - 0.05 && d.z <= 0.05)
    }

    /// Is `p` inside solid earth?
    fn in_earth(&self, p: Vec3) -> bool {
        let Some(g) = self.grade else { return false };
        p.z < g - 0.05
            && !self.pockets.iter().any(|(mn, mx)| {
                p.x >= mn.x && p.x <= mx.x && p.y >= mn.y && p.y <= mx.y && p.z >= mn.z && p.z <= mx.z
            })
    }

    /// Does segment a→b run through earth for more than a few cm? Its
    /// below-grade part, minus whatever lies inside the air pockets.
    fn crosses_earth(&self, a: Vec3, b: Vec3) -> bool {
        let Some(g) = self.grade else { return false };
        let g = g - 0.05;
        let (za, zb) = (a.z, b.z);
        if za >= g && zb >= g {
            return false;
        }
        // parameter range below grade
        let (t0, t1) = if (zb - za).abs() < 1e-9 {
            (0.0, 1.0)
        } else {
            let tc = ((g - za) / (zb - za)).clamp(0.0, 1.0);
            if za < g { (0.0, tc) } else { (tc, 1.0) }
        };
        let len = (b - a).length();
        // subtract pocket intervals
        let mut ivs: Vec<(f32, f32)> = Vec::new();
        for (mn, mx) in self.pockets {
            if let Some((p0, p1)) = clip_box(a, b, *mn, *mx) {
                ivs.push((p0.max(t0), p1.min(t1)));
            }
        }
        ivs.retain(|iv| iv.1 > iv.0);
        ivs.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut covered = 0.0f32;
        let mut end = t0;
        for (s0, s1) in ivs {
            let s0 = s0.max(end);
            if s1 > s0 {
                covered += s1 - s0;
                end = s1;
            }
        }
        (t1 - t0 - covered) * len > 0.1
    }

    fn raycast(&self, p: Vec3, d: Vec3) -> Option<GeomHit> {
        WithPanels { base: self.mesh, panels: self.panels }.raycast_hit(p, d)
    }

    /// A walking ray going on through the surface it hit at `p`: the far
    /// face of the solid behind it (the next hit of the same material
    /// within `MAX_WALL_M`; none: a thin shell, or an object box — crossed
    /// at once), priced exactly as a segment through it is
    /// (`transmission`: the wall at its own thickness, once). The energy
    /// per band, and where the ray goes on.
    pub fn through(&self, p: Vec3, dir: Vec3, mat: &Material, hits: &mut Vec<SegHit>) -> ([f32; NBANDS], Vec3) {
        let inside = p + dir * EPS;
        let far = match self.raycast(inside, dir) {
            Some(h) if h.material == *mat && h.t < MAX_WALL_M => inside + dir * h.t,
            _ => p,
        };
        let t = self.transmission(p - dir * EPS, far + dir * EPS, hits);
        (core::array::from_fn(|b| t[b] * t[b]), far + dir * (2.0 * EPS))
    }

    /// Per-band amplitude of the segment a→b: mesh walls by their real
    /// thickness, overlay boxes once each.
    pub fn transmission(&self, a: Vec3, b: Vec3, hits: &mut Vec<SegHit>) -> [f32; NBANDS] {
        self.transmission_after(a, b, 1e-4, hits)
    }

    /// `transmission` from a point in free space — the listener, or a
    /// walked bounce lifted `EPS` off its surface, the segment leaving
    /// it: every surface it crosses counts, however near the start. (The
    /// default margin, a share of the segment, left a bounce in a corner
    /// — on the floor or a door's edge, a fraction of a millimetre from
    /// the wall beside it — seeing through that wall: the club through
    /// its shut door was mostly such leaks.)
    pub fn transmission_free(&self, a: Vec3, b: Vec3, hits: &mut Vec<SegHit>) -> [f32; NBANDS] {
        let len = (b - a).length().max(1e-9);
        self.transmission_after(a, b, (FREE_START_M / len).min(1e-4), hits)
    }

    fn transmission_after(&self, a: Vec3, b: Vec3, t_lo: f32, hits: &mut Vec<SegHit>) -> [f32; NBANDS] {
        if self.crosses_earth(a, b) {
            return [EARTH; NBANDS];
        }
        let mut g = segment_transmission_after(self.mesh, a, b, t_lo, hits);
        for (mn, mx, m) in self.panels {
            if segment_crosses_box(a, b, *mn, *mx, t_lo) {
                for band in 0..NBANDS {
                    g[band] *= m.transmission[band];
                }
            }
        }
        g
    }
}

/// Parameter interval [t0, t1] ⊂ [0, 1] of segment a→b inside the box.
fn clip_box(a: Vec3, b: Vec3, mn: Vec3, mx: Vec3) -> Option<(f32, f32)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for axis in 0..3 {
        let (da, pa, lo, hi) = (d.get(axis), a.get(axis), mn.get(axis), mx.get(axis));
        if da.abs() < 1e-9 {
            if pa < lo || pa > hi {
                return None;
            }
        } else {
            let (mut ta, mut tb) = ((lo - pa) / da, (hi - pa) / da);
            if ta > tb {
                core::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((t0, t1))
}

fn segment_crosses_box(a: Vec3, b: Vec3, mn: Vec3, mx: Vec3, t_lo: f32) -> bool {
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
    t1 > t_lo && t0 < 1.0 - 1e-4
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

/// Where a moving source was: its positions over the simulation's clock
/// (seconds), oldest first, joined by straight lines. Sound reaching the
/// listener now along a path of length L left the source when it stood
/// exactly L/c away along that path — far sources and long echoes come
/// from where it was then (`Track::emit`).
#[derive(Clone, Debug)]
pub struct Track {
    pub knots: Vec<(f64, Vec3)>,
    /// What it emitted before `from` or after `until` is not this track's
    /// (a teleport ended one track and began the next): ±∞ otherwise.
    pub from: f64,
    pub until: f64,
}

impl Track {
    /// Where it was at `t` (held at its ends).
    pub fn at(&self, t: f64) -> Vec3 {
        let k = &self.knots;
        if k.is_empty() {
            return Vec3::new(0.0, 0.0, 0.0);
        }
        let i = k.partition_point(|(tk, _)| *tk <= t);
        if i == 0 {
            return k[0].1;
        }
        if i == k.len() {
            return k[i - 1].1;
        }
        let ((t0, a), (t1, b)) = (k[i - 1], k[i]);
        let f = if t1 > t0 { ((t - t0) / (t1 - t0)) as f32 } else { 1.0 };
        a + (b - a) * f
    }

    /// Where it emitted what reaches the end of a path now (`now`): the
    /// path has `before` m behind it from the listener to `p`, its last
    /// leg runs from `p` straight to the source. The light-cone condition
    /// c·(now − t) = before + |x(t) − p| has one solution while the source
    /// is slower than sound; fixed-point steps contract by v/c each.
    /// None: emitted outside the track's span.
    pub fn emit(&self, now: f64, p: Vec3, before: f32) -> Option<Vec3> {
        let c = SPEED_OF_SOUND as f64;
        let mut x = self.at(now);
        let mut t = now;
        for _ in 0..4 {
            t = now - (before + (x - p).length()) as f64 / c;
            x = self.at(t);
        }
        (t >= self.from && t <= self.until).then_some(x)
    }
}

/// The sources' motion for a trace (`Tracer::set_motion`): the time the
/// trace is for, and per source (in the trace's order) its track — None
/// for one standing still (it is where it is).
#[derive(Clone, Debug, Default)]
pub struct Motion {
    pub now: f64,
    pub tracks: Vec<Option<Track>>,
}

impl Motion {
    /// Where source `si` (now at `cur`) emitted what reaches `p`'s path,
    /// `before` m from the listener (see `Track::emit`).
    #[inline]
    pub fn emit(&self, si: usize, cur: Vec3, p: Vec3, before: f32) -> Option<Vec3> {
        match self.tracks.get(si) {
            Some(Some(tr)) => tr.emit(self.now, p, before),
            _ => Some(cur),
        }
    }

    fn moving(&self, si: usize) -> bool {
        matches!(self.tracks.get(si), Some(Some(_)))
    }
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
    /// Next-event connections per bounce. With more sources facing the
    /// surface than this, each bounce connects this many — each source at
    /// most once, by what it would bring there with nothing in the way,
    /// the nearest loud ones for sure (`pick_systematic`) — and reweights:
    /// cost per bounce stays flat as the source count grows. 0 = connect
    /// to every source.
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
    /// Rays passing close to a mesh edge bend around it (V3.4).
    pub diffraction: bool,
    /// Blocked sources searching for the edges their direct field bends
    /// over per tick, one bend / two bends: the rest keep the edges they
    /// found, solved exactly for where they are now (`Tracer::edge_paths`;
    /// the most needed search first — loud, and moved furthest since).
    pub edge_searches: usize,
    pub edge2_searches: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            rays: 1024,
            bounces: 4,
            k: 0.06,
            ema: 0.35,
            nee: 4,
            late_rays: 256,
            late_ema: 0.15,
            late_nee: 1,
            diffraction: true,
            edge_searches: 64,
            edge2_searches: 64,
        }
    }
}

/// A ray splits at the first surface passing at least this share of the
/// energy (its loudest band) — any partition: both branches go on with
/// their exact weights, so what leaks through a wall, a door or a pane is
/// estimated from every ray that meets it. (A random 2 % pass through
/// anything quieter than a light door left a closed room's field to a
/// handful of rays per tick: the club heard through its windows wandered
/// by ±2 dB and took ~15 s to settle.) Measured: +1 ms on a 48 ms town
/// tick — the branches through thick walls die young.
const SPLIT_T2: f32 = 1e-6;

/// One branch of a walked ray (`SPLIT_T2`): where it is, what it
/// carries, its path's state so far, and the segment it resumes at.
#[derive(Clone, Copy)]
struct Branch {
    origin: Vec3,
    dir: Vec3,
    thr: [f32; NBANDS],
    length: f32,
    // reflections so far, and whether the last one was specular (only
    // then do sphere hits count — NEE covered diffuse ones)
    refl: u32,
    specular: bool,
    // every bounce a mirror so far (nothing scattered or bent), and that
    // mirror chain's own throughput still above the floor (latched: once
    // quiet, quiet)
    coherent: bool,
    loud: bool,
    mirror_thr: f32,
    // bent around an edge since the last diffuse bounce, and where the ray
    // was when it first bent (gating: a bend only adds what the straight
    // line from there doesn't carry); a ray passes an edge once
    bent: bool,
    bends: u32,
    bend_from: Option<Vec3>,
    last_edge: Option<usize>,
    seg0: u32,
    split: bool,
}

impl Branch {
    fn start(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir,
            thr: [1.0; NBANDS],
            length: 0.0,
            refl: 0,
            specular: false,
            coherent: true,
            loud: true,
            mirror_thr: 1.0,
            bent: false,
            bends: 0,
            bend_from: None,
            last_edge: None,
            seg0: 0,
            split: false,
        }
    }
}

/// Hard cap on late-family segments per ray (the time and energy limits
/// end it first in any real room).
const LATE_MAX_SEGMENTS: u32 = 600;
/// Late rays stop once they have lost this much (80 dB: `DENSE_FLOOR`).
const LATE_FLOOR: f32 = 1e-8;
/// v4: the dense response ends this far under its loudest bin (80 dB): the
/// tail decays on (physically it never stops), but past this nothing of it
/// is heard under any adaptation — at 60 dB a long room's tail ended
/// audibly, still 56 dB under a clap's peak.
pub const DENSE_FLOOR: f32 = 1e-8;

/// v4: the scattered part of a source's response at the listener, from
/// its free-field arrival on: per `bin_s` bin, the energy per band and
/// the mid band's energy vector (Σ E·direction, world frame; its length
/// over E is how one-sided the arrivals are). Same units as tap gains².
#[derive(Clone, Debug, Default)]
pub struct DenseResponse {
    pub bin_s: f32,
    /// [E_low, E_mid, E_high, vx, vy, vz] per bin.
    pub bins: Vec<[f32; 6]>,
    /// Per history window (`HISTORY_WINDOWS`: the first 4 ms from the
    /// free-field arrival, then each twice as long), the estimate's
    /// relative standard error (0: exact, or not known).
    pub noise: [f32; HISTORY_WINDOWS],
    /// Coherent arrivals past the renderer's taps (solved mirror chains
    /// beyond `MAX_COHERENT_TAPS`: a flutter train's later echoes):
    /// [delay after the free field (s), energy low / mid / high, unit
    /// direction x, y, z]. Their energy is in `bins` too (every statistic
    /// counts them); the kernel plays them as the impulses they are, the
    /// noise from the rest — not folded into it.
    pub impulses: Vec<[f32; 7]>,
}

impl DenseResponse {
    /// How far `other`'s mid-band envelope is from this one's: L1 over
    /// the larger total (0 = the same, 2 = disjoint), per history window
    /// (the first 4 ms from the free-field arrival, then each twice as
    /// long), counting only the windows that differ beyond the two
    /// estimates' noise (`HISTORY_JUMP_Z` standard errors, as the history
    /// test's jump).
    pub fn change_from(&self, other: &DenseResponse) -> f32 {
        self.compare(other, true)
    }

    /// `change_from` counting every window's difference, noise and all
    /// (what a crossfaded rebuild follows: it costs work, never a jump).
    pub fn difference_from(&self, other: &DenseResponse) -> f32 {
        self.compare(other, false)
    }

    fn compare(&self, other: &DenseResponse, significant: bool) -> f32 {
        let mut a = [0.0f32; HISTORY_WINDOWS];
        let mut b = [0.0f32; HISTORY_WINDOWS];
        let at = |d: &DenseResponse, i: usize| history_window((i as f32 * d.bin_s / DENSE_BIN_S) as usize);
        for (i, v) in self.bins.iter().enumerate() {
            a[at(self, i)] += v[1];
        }
        for (i, v) in other.bins.iter().enumerate() {
            b[at(other, i)] += v[1];
        }
        let diff: f32 = (0..HISTORY_WINDOWS)
            .map(|w| {
                let (x, y) = (a[w], b[w]);
                if !significant {
                    return (x - y).abs();
                }
                if x <= 0.0 && y <= 0.0 {
                    return 0.0;
                }
                if x <= 0.0 || y <= 0.0 {
                    // there in one, not in the other: real if the one is sure
                    let (v, n) = if x > 0.0 { (x, self.noise[w]) } else { (y, other.noise[w]) };
                    return if n * HISTORY_JUMP_Z < 1.0 { v } else { 0.0 };
                }
                let se = (self.noise[w] * self.noise[w] + other.noise[w] * other.noise[w]).sqrt();
                if (x / y).ln().abs() > HISTORY_JUMP_Z * se {
                    (x - y).abs()
                } else {
                    0.0
                }
            })
            .sum();
        diff / a.iter().sum::<f32>().max(b.iter().sum::<f32>()).max(1e-30)
    }

    /// Reverberation time per band (s), as measured from an impulse
    /// response: Schroeder's backward-integrated decay, its slope from −5
    /// to −25 dB (T20) taken to 60 dB. A response still ringing at the
    /// window's end (`RESPONSE_MAX_S`) gets the energy past it along the
    /// decay of its last second (as the renderer continues it), so a
    /// long tail isn't cut short by the window. 0 where there is none.
    pub fn rt60(&self) -> [f32; NBANDS] {
        let n = self.bins.len();
        let dt = self.bin_s as f64;
        core::array::from_fn(|b| {
            if n < 8 || dt <= 0.0 {
                return 0.0;
            }
            let (fit, beyond) = self.tail(b);
            // the decay curve, in dB under the total
            let mut edc = vec![0.0f64; n];
            let mut acc = beyond;
            for i in (0..n).rev() {
                acc += self.bins[i][b].max(0.0) as f64;
                edc[i] = acc;
            }
            let total = edc[0];
            if total <= 0.0 {
                return 0.0;
            }
            let db = |i: usize| 10.0 * (edc[i] / total).max(1e-30).log10();
            let Some(i5) = (0..n).find(|&i| db(i) <= -5.0) else { return 0.0 };
            let Some(i25) = (i5..n).find(|&i| db(i) <= -25.0) else {
                // the window ends above −25 dB: the tail's own decay
                return fit.map_or(0.0, |k| (6.0 * core::f64::consts::LN_10 * dt / -k) as f32);
            };
            // least squares of the curve over −5 … −25 dB
            let (mut sx, mut sy, mut sxx, mut sxy, mut m) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for i in i5..=i25 {
                let (x, y) = (i as f64 * dt, db(i));
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
                m += 1.0;
            }
            let slope = if m >= 2.0 { (m * sxy - sx * sy) / (m * sxx - sx * sx) } else { -20.0 / ((i25 - i5).max(1) as f64 * dt) };
            if slope < 0.0 { (-60.0 / slope) as f32 } else { 0.0 }
        })
    }
}

impl DenseResponse {
    /// Band `b`'s decay over the window's last second — ln E per bin, its
    /// slope k (< 0) — and, for a response still ringing at the window's
    /// end (`RESPONSE_MAX_S`), the energy past it along that decay (as the
    /// renderer continues it).
    fn tail(&self, b: usize) -> (Option<f64>, f64) {
        let n = self.bins.len();
        let dt = self.bin_s as f64;
        let fit_from = n.saturating_sub((1.0 / dt) as usize);
        let (mut sx, mut sy, mut sxx, mut sxy, mut m) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, v) in self.bins.iter().enumerate().skip(fit_from) {
            if v[b] > 0.0 {
                let (x, y) = (i as f64, (v[b] as f64).ln());
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
                m += 1.0;
            }
        }
        let fit = (m >= 8.0).then(|| (m * sxy - sx * sy) / (m * sxx - sx * sx)).filter(|k| *k < 0.0);
        let ringing = (n as f32) * self.bin_s >= RESPONSE_MAX_S - 0.05;
        let beyond = match (ringing, fit) {
            (true, Some(k)) => {
                let end = ((sy - k * sx) / m + k * (n - 1) as f64).exp();
                end * k.exp() / (1.0 - k.exp())
            }
            _ => 0.0,
        };
        (fit, beyond)
    }

    /// Per band, the response's centre time: its energy's mean arrival
    /// after the free-field arrival (s), the continued tail included. 0
    /// where there is none.
    pub fn centre_time(&self) -> [f32; NBANDS] {
        let n = self.bins.len();
        let dt = self.bin_s as f64;
        core::array::from_fn(|b| {
            if n == 0 || dt <= 0.0 {
                return 0.0;
            }
            let (mut e, mut et) = (0.0f64, 0.0f64);
            for (i, v) in self.bins.iter().enumerate() {
                let x = v[b].max(0.0) as f64;
                e += x;
                et += x * (i as f64 + 0.5) * dt;
            }
            if n >= 8 {
                if let (Some(k), beyond) = self.tail(b) {
                    // past the window, bin by bin along e^{k·j}: its mean
                    // lies e^k / (1 − e^k) bins past the end
                    let q = k.exp();
                    e += beyond;
                    et += beyond * (n as f64 + 0.5 + q / (1.0 - q)) * dt;
                }
            }
            if e > 0.0 {
                (et / e) as f32
            } else {
                0.0
            }
        })
    }
}

/// What one source sounds like at the listener (world frame, unit gain).
#[derive(Clone, Debug)]
pub struct SourceField {
    pub taps: Vec<Tap>,
    /// v4 only: the dense (scattered) response; the taps are then the
    /// direct path, edge bends and the distinct mirror chains.
    pub dense: Option<DenseResponse>,
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
    /// A solved chain's arrival after the free field, this tick's.
    last: f32,
}

impl Default for Bin {
    fn default() -> Self {
        Self { e: [0.0; NBANDS], ed: 0.0, dir: Vec3::new(0.0, 0.0, 0.0), last: 0.0 }
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

#[derive(Default, Clone)]
struct SourceState {
    early: HashMap<u32, Bin>,
    echo: Option<Echogram>,
    /// Solved chains currently played as discrete taps: one group per
    /// tap, its key first (chains arriving together merge, `ECHO_MERGE_S`).
    echo_on: Vec<Vec<u32>>,
    /// v4: the averaged dense response (see `DenseResponse::bins`), with
    /// a 7th channel: Σ of the contributions' squared mid energies (their
    /// effective number, `ESS_TARGET`).
    dense: Vec<[f32; 7]>,
    /// v4: per history window (`history_window`), the ticks since the
    /// window last changed (its running mean's count) and how many ticks
    /// its average spans now (its effective sample count's multiplier).
    dense_age: [u32; HISTORY_WINDOWS],
    dense_ticks: [f32; HISTORY_WINDOWS],
    /// v4: per history window, the fresh trace's measured scatter around
    /// the mean (variance of ln fresh/mean, tick to tick): the noise the
    /// effective sample count misses (rare strong paths, one ray's many
    /// correlated contributions).
    dense_var: [f32; HISTORY_WINDOWS],
    /// v4: per history window, how many ticks in a row the fresh trace
    /// came out on the same side of the mean (+ above, − below) and their
    /// summed deviation in standard errors: a drift the per-tick test is
    /// too coarse to see (`HISTORY_DRIFT_Z`).
    dense_run: [i32; HISTORY_WINDOWS],
    dense_drift: [f32; HISTORY_WINDOWS],
    /// v4: the measured scatter of the whole response's energy (the
    /// first test: `fold_v4`).
    dense_var_all: f32,
    /// v4: where listener and source were last tick; v6: where they
    /// stood when the current still spell began.
    at: Option<(Vec3, Vec3)>,
    /// v4: the world changed since the last tick (a door moved): the
    /// response follows as when moving — the tests say what changed.
    /// v6: its averages start over (`fold`).
    restart: bool,
    /// v6: the world moved this tick (a door on its way): a moving tick
    /// for the averages, the late one as quick as the early (`fold`).
    world_moving: bool,
    /// v6: how many ticks' worth the early and the late average hold —
    /// 1/Σw² of their (normalized) weights: n after n ticks of a running
    /// mean, (2 − α)/α in an EMA's steady state, 0 empty (`fold`).
    held: [f32; 2],
}

/// The diffuse field's mid-band energy in one echo slot at `rel`: the
/// echogram within ±`ECHO_SPAN`, less the `coherent` (solved chain)
/// energy among it, per direction cell, scaled to `ECHO_SLOT_S`.
fn diffuse_slot(echo: &Echogram, rel: f32, coherent: f32) -> f32 {
    let lo = ((rel - ECHO_SPAN) / BIN_DT).max(0.0) as usize;
    let hi = (((rel + ECHO_SPAN) / BIN_DT) as usize).min(echo.bins.len().saturating_sub(1));
    if hi < lo {
        return 0.0;
    }
    let e: f32 = (lo..=hi).map(|i| echo.bins[i][1]).sum();
    (e - coherent).max(0.0) / N_CELLS as f32 * ECHO_SLOT_S / ((hi - lo + 1) as f32 * BIN_DT)
}

impl SourceState {
    /// Step 5a, shared by `trace` and the externally walked fold: smooth
    /// one tick's binned contributions into the running state.
    #[allow(clippy::too_many_arguments)]
    fn fold(
        &mut self,
        early: HashMap<u32, Bin>,
        echo: Echogram,
        budget: Budget,
        v6: bool,
        listener: Vec3,
        source: Vec3,
    ) {
        let first = self.echo.is_none();
        let (mut alpha_early, mut alpha_late) = (if first { 1.0 } else { budget.ema.clamp(0.0, 1.0) }, budget.late_ema.clamp(0.0, 1.0));
        if v6 {
            // the world changed after a rest (a door started moving:
            // `Tracer::prepare`) — nothing averaged before describes this
            // world: the averages start over, the sound follows at once
            // (and the change's burst traces more rays for the first ticks)
            if std::mem::take(&mut self.restart) {
                self.held = [0.0; 2];
            }
            // each tick weighs what the next tick of a running mean of
            // what the average holds would: 1/(n + 1). Moving — the
            // listener or the source, or the world on its way — at least
            // the EMA's weight, so it follows; the late average as quick
            // as the early one while the world moves (a door changes every
            // order of the field at once, not the slow drift of the tail
            // as a listener walks). Standing still — neither end moved
            // `STILL_M` from where the spell began, the world at rest —
            // the running mean goes on converging (down to
            // `STILL_ALPHA_FLOOR`: ~10 s of memory, should something
            // change that no test sees)
            let world_moved = std::mem::take(&mut self.world_moving);
            let moved = world_moved || self.at.is_none_or(|(l, s)| (l - listener).length() > STILL_M || (s - source).length() > STILL_M);
            if moved {
                self.at = Some((listener, source));
            }
            let emas = [budget.ema, if world_moved { budget.ema } else { budget.late_ema }];
            let mut alphas = [0.0f32; 2];
            for ((a, n), ema) in alphas.iter_mut().zip(self.held.iter_mut()).zip(emas) {
                let mean = 1.0 / (*n + 1.0);
                *a = if moved { mean.max(ema.clamp(0.0, 1.0)) } else { mean.max(STILL_ALPHA_FLOOR) };
                let w2 = if *n > 0.0 { 1.0 / *n } else { 0.0 };
                *n = 1.0 / ((1.0 - *a) * (1.0 - *a) * w2 + *a * *a);
            }
            [alpha_early, alpha_late] = alphas;
        }
        // solved chains are exact, not noisy: averaged like the early bins
        // only so a chain that just validated fades in rather than pops
        let alpha_echo = alpha_early;
        let mut keys: Vec<u32> = self.early.keys().copied().collect();
        for k in early.keys() {
            if !self.early.contains_key(k) {
                keys.push(*k);
            }
        }
        for k in keys {
            let f = early.get(&k).copied().unwrap_or_default();
            let alpha = if k >= KEY_CHAIN_BASE { alpha_echo } else { alpha_early };
            let cur = self.early.entry(k).or_default();
            for b in 0..NBANDS {
                cur.e[b] += alpha * (f.e[b] - cur.e[b]);
            }
            cur.ed += alpha * (f.ed - cur.ed);
            cur.dir = cur.dir + (f.dir - cur.dir) * alpha;
            if f.e[1] > 0.0 {
                cur.last = f.last;
            }
        }
        self.early.retain(|_, b| b.e.iter().any(|&e| e > 1e-12));
        // late echogram: a slower statistic, a slower average
        match &mut self.echo {
            Some(e) => e.ema(&echo, alpha_late),
            None => self.echo = Some(echo),
        }
        // discrete echoes: the salient solved chains (with hysteresis),
        // the strongest MAX_ECHO_TAPS; each slot's field is measured
        // without the chains arriving in it
        let echo = self.echo.as_ref().expect("set above");
        let chains: Vec<(u32, f32, f32)> = self
            .early
            .iter()
            .filter(|(k, b)| **k >= KEY_CHAIN_BASE && b.e[1] > 0.0)
            .map(|(k, b)| (*k, b.last, b.e[1]))
            .collect();
        let playing = |k: u32| self.echo_on.iter().any(|g| g.contains(&k));
        let leads = |k: u32| self.echo_on.iter().any(|g| g[0] == k);
        let mut on: Vec<(u32, f32, f32)> = Vec::new();
        for &(k, rel, e) in &chains {
            let mine: f32 =
                chains.iter().filter(|c| (c.1 - rel).abs() <= ECHO_SPAN + 0.5 * BIN_DT).map(|c| c.2).sum();
            let salience = e * LEGACY_UNITS / diffuse_slot(echo, rel, mine * LEGACY_UNITS).max(1e-30);
            if salience >= if playing(k) { ECHO_STAY } else { ECHO_ENTER } {
                on.push((k, rel, e));
            }
        }
        // chains arriving together from about the same way are one
        // arrival to the ear: they share a tap (led by the chain that led
        // it before, else the strongest), and the cap counts arrivals
        on.sort_by(|x, y| leads(y.0).cmp(&leads(x.0)).then(y.2.total_cmp(&x.2)));
        let dir_of = |k: u32| {
            let d = self.early[&k].dir;
            let l = d.length();
            if l > 1e-9 { d * (1.0 / l) } else { d }
        };
        let mut groups: Vec<(f32, Vec3, Vec<u32>)> = Vec::new();
        for (k, rel, _) in on {
            let d = dir_of(k);
            if let Some(g) = groups.iter_mut().find(|g| (g.0 - rel).abs() <= ECHO_MERGE_S && g.1.dot(d) >= ECHO_MERGE_COS) {
                g.2.push(k);
            } else if groups.len() < MAX_ECHO_TAPS {
                groups.push((rel, d, vec![k]));
            }
        }
        self.echo_on = groups.into_iter().map(|g| g.2).collect();
    }

    /// v4's step 5a: one tick in. The solved chains are exact: this
    /// tick's energies as they are. The dense response is a running mean
    /// per history window only while the fresh trace agrees with it — a
    /// window whose fresh energy lies `HISTORY_JUMP_Z` standard errors
    /// from its mean (both estimates' noise, from their effective sample
    /// counts, or as measured), or drifts `HISTORY_DRIFT_Z` to one side
    /// over a few ticks, has changed and restarts from the fresh trace: a
    /// door, a step into another space, a moved source are heard the tick
    /// they happen; a still scene's noise averages out. Moving (or the
    /// world changing: a door swinging), a window's mean spans fewer
    /// ticks: as many as the motion leaves its arrivals in place — a step
    /// of `d` moves every path by at most `d`, a share `d / (c · width)` of
    /// a window `width` long (the late windows, spanning hundreds of
    /// metres of path, keep averaging under a turning head's hands); at
    /// most the budget's late EMA.
    fn fold_v4(&mut self, acc: Acc, budget: Budget, listener: Vec3, source: Vec3) {
        let first = self.echo.is_none();
        let world_changed = std::mem::take(&mut self.restart);
        let step = match self.at {
            Some((l, s)) => (listener - l).length() + (source - s).length(),
            None => 0.0,
        };
        self.at = Some((listener, source));
        // coherent arrivals (solved chains): exact, no history
        self.early = acc.early.into_iter().filter(|(_, b)| b.e.iter().any(|&e| e > 1e-12)).collect();
        // the dense response, window by window
        let n = self.dense.len().max(acc.dense.len());
        self.dense.resize(n, [0.0; 7]);
        let mut sums = [[0.0f64; 4]; HISTORY_WINDOWS]; // fresh Σe, Σe², history Σe, Σe²
        for i in 0..n {
            let w = history_window(i);
            let f = acc.dense.get(i).copied().unwrap_or([0.0; 7]);
            let h = self.dense[i];
            sums[w][0] += f[1] as f64;
            sums[w][1] += f[6] as f64;
            sums[w][2] += h[1] as f64;
            sums[w][3] += h[6] as f64;
        }
        // the whole response first: a tick holds hundreds of contributions
        // where a window may hold one, so a door shut or a step into
        // another space shows in the total at once even where no window
        // alone can tell it from its noise — every window restarts. (The
        // world changing is the prior for it: the bar drops to the drift's.)
        let (ft, fqt, ht) = sums.iter().fold((0.0, 0.0, 0.0), |a, w| (a.0 + w[0], a.1 + w[1], a.2 + w[2]));
        let hqt: f64 = (0..HISTORY_WINDOWS).map(|w| sums[w][3] / self.dense_ticks[w].max(1.0) as f64).sum();
        let all = !first
            && if ft > 0.0 && ht > 0.0 {
                let var = (fqt / (ft * ft) + hqt / (ht * ht)).max(self.dense_var_all as f64);
                let dev = (ft / ht).ln();
                let bar = if world_changed { HISTORY_DRIFT_Z } else { HISTORY_JUMP_Z };
                let all = dev.abs() > bar as f64 * var.sqrt();
                if !all {
                    let d2 = (dev * dev).min(9.0 * var) as f32;
                    self.dense_var_all += HISTORY_VAR_EMA * (d2 - self.dense_var_all);
                }
                all
            } else {
                (ft > 0.0) != (ht > 0.0)
            };
        let moving = budget.late_ema.clamp(STILL_ALPHA_FLOOR, 1.0);
        let mut alpha = [1.0f32; HISTORY_WINDOWS];
        for w in 0..HISTORY_WINDOWS {
            let floor = if world_changed {
                moving
            } else if step == 0.0 {
                // nothing moved: every tick is the same truth sampled
                // again — a plain running mean, converging for as long as
                // it holds (a floor kept ~400 ticks and let rare strong
                // paths wander it by ±2 dB)
                0.0
            } else {
                (step / (SPEED_OF_SOUND * history_width(w))).clamp(STILL_ALPHA_FLOOR, moving)
            };
            let [fe, fq, he, hq] = sums[w];
            let n_f = if fq > 0.0 { fe * fe / fq } else { 0.0 };
            let n_h = if hq > 0.0 { he * he / hq * self.dense_ticks[w] as f64 } else { 0.0 };
            // the scatter expected of the fresh trace around the mean: its
            // effective sample counts', or as measured, whichever is wider
            let var = (1.0 / n_f.max(1.0) + 1.0 / n_h.max(1.0)).max(self.dense_var[w] as f64);
            let dev = if fe > 0.0 && he > 0.0 { (fe / he).ln() } else { 0.0 };
            let z = dev / var.sqrt().max(1e-6);
            let changed = first
                || all
                || if fe > 0.0 && he > 0.0 {
                    z.abs() > HISTORY_JUMP_Z as f64
                } else if he > 0.0 {
                    // nothing where the mean expects contributions
                    n_h / self.dense_ticks[w].max(1.0) as f64 > HISTORY_JUMP_Z as f64
                } else {
                    // contributions where the mean has none
                    n_f > HISTORY_JUMP_Z as f64
                };
            // the drift test: the run on one side of the mean, summed
            let (run, drift) = (&mut self.dense_run[w], &mut self.dense_drift[w]);
            if (z > 0.0 && *run < 0) || (z < 0.0 && *run > 0) {
                *run = 0;
                *drift = 0.0;
            }
            if z != 0.0 {
                *run += z.signum() as i32;
                *drift += z as f32;
            }
            let drifted = run.abs() >= 2 && drift.abs() / (run.abs() as f32).sqrt() > HISTORY_DRIFT_Z;
            // nothing moved and the world is as it was: the same truth,
            // sampled again — an outlier is the estimate's own noise
            let still = step == 0.0 && !world_changed;
            // the world changed (a door): what it averaged belongs to the
            // world before — every window starts over (the next ticks
            // trace with more rays, `Tracer::burst`)
            let changed = first || world_changed || (!still && (changed || drifted));
            if changed {
                *run = 0;
                *drift = 0.0;
            } else if !first && fe > 0.0 && he > 0.0 {
                // the scatter learns from ticks that agree only (a change
                // learnt as noise would hide the next one), an outlier
                // counting at most 3 standard errors
                let d2 = (dev * dev).min(9.0 * var) as f32;
                self.dense_var[w] += HISTORY_VAR_EMA * (d2 - self.dense_var[w]);
            }
            if changed {
                self.dense_age[w] = 0;
            }
            let run = 1.0 / (self.dense_age[w] + 1) as f32;
            alpha[w] = run.max(floor);
            self.dense_ticks[w] = if alpha[w] > run { (2.0 - alpha[w]) / alpha[w] } else { (self.dense_age[w] + 1) as f32 };
            self.dense_age[w] = self.dense_age[w].saturating_add(1);
        }
        for (i, cur) in self.dense.iter_mut().enumerate() {
            let a = alpha[history_window(i)];
            let f = acc.dense.get(i).copied().unwrap_or([0.0; 7]);
            for c in 0..7 {
                cur[c] += a * (f[c] - cur[c]);
            }
        }
        while self.dense.last().is_some_and(|b| b[1] <= 1e-15 && b[0] <= 1e-15 && b[2] <= 1e-15) {
            self.dense.pop();
        }
        let a_echo = alpha.iter().cloned().fold(0.0f32, f32::max);
        match &mut self.echo {
            Some(e) => e.ema(&acc.echo, a_echo),
            None => self.echo = Some(acc.echo),
        }
    }

    /// v4's step 5b: the direct path and edge bends are taps, and so are
    /// the solved mirror chains standing out most (`MAX_COHERENT_TAPS`);
    /// the rest join the dense response at their delay. The dense
    /// response ends `DENSE_FLOOR` (80 dB) below its loudest bin.
    fn field_v4(
        &self,
        t_free: f32,
        direct: Option<Tap>,
        edge_taps: &mut Vec<Tap>,
        reflected: [f32; NBANDS],
        shared: f32,
    ) -> SourceField {
        let mut taps: Vec<Tap> = Vec::new();
        if let Some(t) = direct {
            taps.push(t);
        }
        taps.append(edge_taps);
        let mut dense = smooth_dense(&self.dense, &self.dense_ticks);
        let span = (DISTINCT_SPAN_S / DENSE_BIN_S).ceil() as usize;
        let around = |dense: &[[f32; 6]], i: usize| -> f32 {
            dense.iter().skip(i.saturating_sub(span)).take(2 * span + 1).map(|d| d[1]).sum()
        };
        // rank by how far each stands out of the crowd around it
        let mut chains: Vec<(u32, &Bin, f32)> = self
            .early
            .iter()
            .filter(|(k, b)| **k >= KEY_CHAIN_BASE && b.e[1] > 0.0)
            .map(|(k, b)| {
                let i = (b.last.max(0.0) / DENSE_BIN_S) as usize;
                (*k, b, b.e[1] / around(&dense, i).max(1e-30))
            })
            .collect();
        chains.sort_by(|x, y| y.2.total_cmp(&x.2));
        let mut impulses = Vec::new();
        for (n, (k, b, _)) in chains.into_iter().enumerate() {
            let i = (b.last.max(0.0) / DENSE_BIN_S) as usize;
            let dl = b.dir.length();
            let dir = if dl > 1e-9 { b.dir * (1.0 / dl) } else { Vec3::new(0.0, 0.0, 1.0) };
            if n < MAX_COHERENT_TAPS {
                taps.push(Tap {
                    key: k,
                    delay_s: t_free + b.last,
                    dir: [dir.x, dir.y, dir.z],
                    gains: core::array::from_fn(|band| b.e[band].max(0.0).sqrt()),
                });
            } else {
                if dense.len() <= i {
                    dense.resize(i + 1, [0.0; 6]);
                }
                for band in 0..NBANDS {
                    dense[i][band] += b.e[band].max(0.0);
                }
                dense[i][3] += dir.x * b.e[1];
                dense[i][4] += dir.y * b.e[1];
                dense[i][5] += dir.z * b.e[1];
                impulses.push([b.last.max(0.0), b.e[0].max(0.0), b.e[1].max(0.0), b.e[2].max(0.0), dir.x, dir.y, dir.z]);
            }
        }
        let peak = dense.iter().map(|d| d[1]).fold(0.0f32, f32::max);
        while dense.last().is_some_and(|d| d[1] < peak * DENSE_FLOOR) {
            dense.pop();
        }
        // the old renderer's and the HUD's readouts, from the response
        let reverb = estimate_reverb(self.echo.as_ref().unwrap_or(&Echogram::new()));
        let (mut v, mut e) = (Vec3::new(0.0, 0.0, 0.0), 0.0f32);
        for d in &dense {
            v = v + Vec3::new(d[3], d[4], d[5]);
            e += d[1];
        }
        let vl = v.length();
        let late_dir = if vl > 1e-12 { [v.x / vl, v.y / vl, v.z / vl] } else { [0.0, 0.0, 1.0] };
        let aniso = if e > 0.0 { (vl / e).min(1.0) } else { 0.0 };
        // each window's standard error: the per-tick scatter (the effective
        // sample count's, or as measured) over the ticks averaged
        let mut sums = [[0.0f64; 2]; HISTORY_WINDOWS];
        for (i, d) in self.dense.iter().enumerate() {
            let w = history_window(i);
            sums[w][0] += d[1] as f64;
            sums[w][1] += d[6] as f64;
        }
        let noise: [f32; HISTORY_WINDOWS] = core::array::from_fn(|w| {
            let [e, q] = sums[w];
            if e <= 0.0 {
                return 0.0;
            }
            let per_tick = (q / (e * e)).max(self.dense_var[w] as f64);
            // v5: a tick's estimate holds the pool's paths too, each in
            // `shared` ticks — the ticks averaged hold that many fewer
            // distinct paths (counted as independent, the error would be
            // √shared too small: noise judged a change)
            let ticks = (self.dense_ticks[w] as f64 + shared as f64 - 1.0) / shared as f64;
            (per_tick / ticks.max(1.0)).sqrt() as f32
        });
        let end = dense.len() as f32 * DENSE_BIN_S;
        impulses.retain(|i| i[0] < end);
        SourceField { taps, dense: Some(DenseResponse { bin_s: DENSE_BIN_S, bins: dense, noise, impulses }), reverb, late_dir, aniso, reflected }
    }

    /// Step 5b: this source's field from the smoothed state.
    fn field(
        &self,
        cells: &[Vec3],
        t_free: f32,
        direct: Option<Tap>,
        edge_taps: &mut Vec<Tap>,
        reflected: [f32; NBANDS],
    ) -> SourceField {
        let mut taps: Vec<Tap> = Vec::new();
        if let Some(t) = direct {
            taps.push(t);
        }
        taps.append(edge_taps);
        let mut early: Vec<(&u32, &Bin)> = self.early.iter().filter(|(k, _)| **k < KEY_CHAIN_BASE).collect();
        early.sort_by(|x, y| y.1.e[1].total_cmp(&x.1.e[1]));
        for (k, b) in early.into_iter().take(MAX_EARLY_TAPS) {
            if b.e[1] <= 0.0 {
                continue;
            }
            let dl = b.dir.length();
            let dir = if dl > 1e-9 { b.dir * (1.0 / dl) } else { cells[cell_of(*k)] };
            // Delay = this tick's free-field arrival + the bin's fixed
            // offset. An energy-weighted mean of stochastic arrivals
            // jitters tick to tick, and the renderer glides every
            // same-key delay change as motion: audible pitch warble
            // ("tape delay"). Pinned offsets move only when the
            // listener or source moves — coherent, real Doppler.
            let dbin = (k - KEY_EARLY_BASE) % 256;
            taps.push(Tap {
                key: *k,
                delay_s: t_free + (dbin as f32 + 0.5) * DELAY_BIN_S,
                dir: [dir.x, dir.y, dir.z],
                gains: core::array::from_fn(|band| b.e[band].max(0.0).sqrt()),
            });
        }
        let empty;
        let mut echo = match self.echo.as_ref() {
            Some(e) => e,
            None => {
                empty = Echogram::new();
                &empty
            }
        };
        // the discrete echoes (chosen in `fold`), played as taps — and
        // taken out of the echogram, so the reverb doesn't carry them a
        // second time
        let owned;
        if !self.echo_on.is_empty() {
            let mut rest = echo.clone();
            for group in &self.echo_on {
                // the group's energy at its lead chain's exact arrival
                let Some(lead) = self.early.get(&group[0]) else { continue };
                let rel = lead.last;
                let mut e = [0.0f32; NBANDS];
                let mut d = Vec3::new(0.0, 0.0, 0.0);
                for k in group {
                    if let Some(b) = self.early.get(k) {
                        for band in 0..NBANDS {
                            e[band] += b.e[band].max(0.0);
                        }
                        d = d + b.dir;
                        if let Some(bin) = rest.bins.get_mut((b.last / BIN_DT) as usize) {
                            for band in 0..NBANDS {
                                bin[band] = (bin[band] - b.e[band] * LEGACY_UNITS).max(0.0);
                            }
                        }
                    }
                }
                let dl = d.length();
                if e[1] <= 0.0 || dl <= 1e-9 {
                    continue;
                }
                let dir = d * (1.0 / dl);
                taps.push(Tap {
                    key: group[0],
                    delay_s: t_free + rel,
                    dir: [dir.x, dir.y, dir.z],
                    gains: core::array::from_fn(|band| e[band].sqrt()),
                });
            }
            owned = rest;
            echo = &owned;
        }
        let reverb = estimate_reverb(echo);
        let (late_dir, aniso) = echo.late_direction(EARLY_S);
        SourceField { taps, dense: None, reverb, late_dir, aniso, reflected }
    }
}

/// Fresh per-tick accumulators for one source.
/// v5: a late path past its first bounce (diffuse there: what follows
/// doesn't depend on where the listener stood) — reconnected to the
/// listener's next positions (`Tracer::reuse_late`).
struct LateRecord {
    /// The first bounce: where, its normal (towards the listener's side),
    /// how far from the listener, at what incidence (cos).
    v1: Vec3,
    n1: Vec3,
    r1: f32,
    cos1: f32,
    /// Everything the path deposited past it: (source id, energy, delay,
    /// spread), as walked (× 1/rays of its tick).
    deps: Vec<(u32, [f32; NBANDS], f32, f32)>,
}

/// v5: the late paths of the last `LATE_POOL_TICKS` ticks.
struct PoolTick {
    records: Vec<LateRecord>,
    /// The late rays walked that tick (every one a sample, reusable or
    /// not), the listener, and the sources (id, position) then.
    rays: f32,
    listener: Vec3,
    sources: Vec<(u32, Vec3)>,
}

/// v5: ticks of late paths reused (each tick's estimate averages its own
/// rays and theirs: ×(1 + this) samples, standing still or moving).
const LATE_POOL_TICKS: usize = 7;
/// … from listener positions at most this far from where it is now (the
/// reconnection is exact at any distance; far, its weights scatter).
const LATE_POOL_REACH_M: f32 = 3.0;

struct Acc {
    early: HashMap<u32, Bin>,
    echo: Echogram,
    t_free: f32,
    total: [f32; NBANDS],
    /// v4: this tick's dense response, 7 channels (empty in v3).
    dense: Vec<[f32; 7]>,
    v4: bool,
    /// Squared-energy weight of the contribution being deposited: 1, or
    /// n for each of n parts of a spread one (so it still counts as ONE
    /// sample: n parts of (e/n)² · n = e²/n each, e² over the spread).
    sq_scale: f32,
}

impl Acc {
    fn new(t_free: f32, v4: bool) -> Self {
        Self { early: HashMap::default(), echo: Echogram::new(), t_free, total: [0.0; NBANDS], dense: Vec::new(), v4, sq_scale: 1.0 }
    }

    /// Every contribution × `f` (an estimate's weight: its samples' share
    /// of a pooled average); the squared-energy channel × f².
    fn scale(&mut self, f: f32) {
        for b in 0..NBANDS {
            self.total[b] *= f;
        }
        for d in &mut self.dense {
            for c in 0..6 {
                d[c] *= f;
            }
            d[6] *= f * f;
        }
        for bin in &mut self.echo.bins {
            for b in 0..NBANDS {
                bin[b] *= f;
            }
        }
        for d in &mut self.echo.dirs {
            for k in 0..3 {
                d[k] *= f;
            }
        }
        for b in self.early.values_mut() {
            for band in 0..NBANDS {
                b.e[band] *= f;
            }
            b.ed *= f;
            b.dir = b.dir * f;
        }
    }

    /// Add another walk's contributions — a disjoint set of paths (the
    /// CPU's late family onto a GPU walk's early one).
    fn merge(&mut self, o: Acc) {
        for b in 0..NBANDS {
            self.total[b] += o.total[b];
        }
        if self.dense.len() < o.dense.len() {
            self.dense.resize(o.dense.len(), [0.0; 7]);
        }
        for (d, x) in self.dense.iter_mut().zip(&o.dense) {
            for c in 0..7 {
                d[c] += x[c];
            }
        }
        for (bin, x) in self.echo.bins.iter_mut().zip(&o.echo.bins) {
            for b in 0..NBANDS {
                bin[b] += x[b];
            }
        }
        for (d, x) in self.echo.dirs.iter_mut().zip(&o.echo.dirs) {
            for k in 0..3 {
                d[k] += x[k];
            }
        }
        for (k, b) in o.early {
            let cur = self.early.entry(k).or_default();
            for band in 0..NBANDS {
                cur.e[band] += b.e[band];
            }
            cur.ed += b.ed;
            cur.dir = cur.dir + b.dir;
            cur.last = b.last;
        }
    }
}


/// The tracer with its persistent per-source state (EMA across ticks).
pub struct Tracer {
    rng: Rng,
    cells: Vec<Vec3>,
    state: HashMap<u32, SourceState>,
    hits: Vec<SegHit>,
    /// The outdoor field as the listener's own rays find it: per
    /// direction cell, the energy of rays that escaped to open sky
    /// (through doorways, panes, walls, off facades — whatever the walk
    /// did) and the energy-weighted sum of their starting directions.
    /// Averaged over ticks. This replaces the separate dome ray fan.
    sky: Vec<([f32; NBANDS], Vec3)>,
    sky_primed: bool,
    /// The sky's history test (`fold_sky`): ticks since it last changed,
    /// the measured scatter of its total, where the listener was.
    sky_age: u32,
    sky_var: f32,
    sky_at: Option<Vec3>,
    /// The sky's drift test (as the dense response's): the run of ticks
    /// on one side of its mean and their summed deviation.
    sky_run: i32,
    sky_drift: f32,
    /// The world changed since the last tick (a door moved): the sky
    /// follows as when moving.
    sky_world: bool,
    /// Where the listener was last tick: crossing a surface since then
    /// (flying through a wall) drops every average — the rays that built
    /// it can't reach the new side.
    last_listener: Option<Vec3>,
    /// Bumped on every such drop; an external walk launched before it
    /// (its result arrives a tick late) is discarded.
    pub(crate) epoch: u32,
    /// Diffraction edges of the last mesh traced (rebuilt if it changes).
    edges: Option<EdgeSet>,
    tick: u64,
    /// Which edges the listener sees: (listener, tick, mask).
    /// Which edges the listener sees, looked at as searches need them:
    /// (where it was, per edge 0 not yet looked at / 1 seen / 2 hidden).
    ear_vis: Option<(Vec3, Vec<u8>)>,
    /// Per source id: the edges its paths bend over, solved every tick
    /// (`edge_paths`).
    edge_cache: HashMap<u32, EdgePaths>,
    /// Per source id: the energy that reached the listener from it last
    /// tick (every path, its gain² included) — how much its paths matter
    /// to the ear (`edge_paths` searches by it).
    heard: HashMap<u32, f32>,
    /// Per source id: where it stood, and which edges it sees (0 not yet
    /// looked at, 1 seen, 2 hidden) — two-edge strings start from these.
    src_vis: HashMap<u32, (Vec3, Vec<u8>)>,
    /// Mirror chains past `bounces`, per (source id, plane-sequence
    /// hash): found by coherent rays, solved exactly every tick.
    chains: HashMap<(u32, u64), MirrorChain>,
    /// The mesh's flat faces as mirrors (`mirror_faces`), built once per
    /// mesh (keyed by it): first-order chains are named from them.
    faces: Option<(usize, Vec<MirrorFace>)>,
    /// Chains that failed to validate (or lost to stronger ones), by the
    /// tick they did — and, for one with no exact path, where the listener
    /// and its source were then: not named or grown again for
    /// `CHAIN_REJECT_TICKS` (see `held`).
    rejected: HashMap<(u32, u64), Rejection>,
    /// Engine v4: keep the whole traced response (see `RESPONSE_MAX_S`):
    /// every SourceField carries `dense`, coherent arrivals are taps.
    pub v4: bool,
    /// Engine v5 (on v4): the late family's paths are reused across ticks
    /// — past their first (diffuse) bounce, reconnected exactly to where
    /// the listener is now (`reuse_late`).
    pub v5: bool,
    /// Engine v6 (on v3: `v4` off): v3's estimate and reverb — steady in
    /// motion — with v6's refinements (ENGINE_V6.md).
    pub v6: bool,
    /// The last tick's cost by part.
    pub costs: TickCosts,
    /// Threads the ray walk is spread over (1: the tick's own; wasm walks
    /// on one whatever this says). The result doesn't depend on it.
    pub threads: usize,
    /// Each walking thread's buffers, kept across ticks.
    scratch: Vec<WalkScratch>,
    late_pool: std::collections::VecDeque<PoolTick>,
    /// The world signature the pool was walked in (a door moved: gone).
    late_pool_sig: u64,
    /// v4: a signature of the world's movable parts (panels) last tick: a
    /// door closing ends every source's settled average.
    world_sig: u64,
    /// Ticks since the world last changed (v6: under the early average's
    /// memory, a door is still on its way).
    rest: u32,
    /// Ticks the world changed in its current motion (v6: one that went
    /// on for more than a tick restarts again where it stops).
    moved_for: u32,
    /// Capture the strongest paths of each tick for the debug view.
    pub debug: bool,
    /// Last tick's strongest sampled paths (the walk's reflections):
    /// (source id, mid energy, polyline ears → bounces → source),
    /// strongest first.
    pub debug_paths: Vec<(u32, f32, Vec<Vec3>)>,
    /// The exactly solved paths (with `debug`): direct and edge bends of
    /// the last tick, mirror chains of the last solve. See `exact_paths`.
    exact: Vec<ExactPath>,
    exact_mirrors: Vec<ExactPath>,
    /// The sources' motion for the next trace (`set_motion`).
    motion: Motion,
    /// Ticks left that trace `BURST_RAYS`× the rays: after the world
    /// changed (a door), every response restarts from fresh traces, and a
    /// single tick's estimate misses its rare strong paths (it lands low).
    burst: u32,
}

/// After the world changes, this many ticks trace this many times the
/// rays (`Tracer::burst`).
const BURST_TICKS: u32 = 3;
const BURST_RAYS: u32 = 2;
/// v6: after the world changes, this many ticks trace this many times
/// the late rays (and the first `BURST_TICKS` twice the early ones).
const V6_BURST_TICKS: u32 = 5;
const V6_BURST_LATE: u32 = 4;

/// An exactly solved path, for a debug view: unlike the sampled ones it
/// is the same path from tick to tick (the same `key` while it exists)
/// and moves continuously with the listener and the source.
#[derive(Clone, Debug)]
pub struct ExactPath {
    pub source: u32,
    /// 0 direct, 1 edge bend, 2 mirror chain.
    pub kind: u8,
    /// Stable within (source, kind): the edge, or the chain's planes.
    pub key: u64,
    /// Mid-band energy arriving along it.
    pub energy: f32,
    /// The ears, every bend or reflection, the source.
    pub points: Vec<Vec3>,
}

/// A coherent chain past `bounces`: the planes its mirrors lie in, in
/// order from the listener.
#[derive(Clone)]
struct MirrorChain {
    planes: Vec<(Vec3, f32)>,
    ttl: u32,
    /// Mid-band energy at the last solve (what eviction ranks by).
    energy: f32,
    /// The tick its train was last grown (`grow_trains`; 0 = never).
    grown_at: u64,
    /// The last solve: listener, source, tick, (energy, delay, direction)
    /// — reused while neither end moves (re-solved every
    /// `CHAIN_RESOLVE_TICKS` all the same: doors open, panes close).
    solved: Option<(Vec3, Vec3, u64, ([f32; NBANDS], f32, Vec3))>,
}

/// The plane of a hit, canonical (largest normal component positive).
fn plane_of(normal: Vec3, p: Vec3) -> (Vec3, f32) {
    let a = [normal.x, normal.y, normal.z];
    let i = (0..3).max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs())).unwrap_or(0);
    let n = if a[i] < 0.0 { normal * -1.0 } else { normal };
    (n, n.dot(p))
}

fn planes_hash(planes: &[(Vec3, f32)]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (n, d) in planes {
        for q in [n.x * 1e3, n.y * 1e3, n.z * 1e3, d * 1e3] {
            h = (h ^ (q.round() as i64 as u64)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// When a chain was rejected, and — one with no exact path — where the
/// listener and its source were then (`Tracer::held`).
type Rejection = (u64, Option<(Vec3, Vec3)>);

/// A flat face of the world — its coplanar triangles together, or a
/// panel's side — as a mirror: its plane as chains key it, and the box
/// around it.
#[derive(Clone, Copy)]
struct MirrorFace {
    plane: (Vec3, f32),
    lo: Vec3,
    hi: Vec3,
}

/// The mesh's flat faces (`MirrorFace`): its triangles grouped by plane.
fn mirror_faces(mesh: &Mesh) -> Vec<MirrorFace> {
    let mut by_plane: HashMap<u64, usize> = HashMap::default();
    let mut faces: Vec<MirrorFace> = Vec::new();
    for t in &mesh.indices {
        let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
        let n = (b - a).cross(c - a);
        if n.length() < 1e-9 {
            continue;
        }
        let plane = plane_of(n.normalize(), a);
        let i = *by_plane.entry(planes_hash(&[plane])).or_insert_with(|| {
            faces.push(MirrorFace { plane, lo: a, hi: a });
            faces.len() - 1
        });
        let f = &mut faces[i];
        for p in [a, b, c] {
            f.lo = f.lo.min(p);
            f.hi = f.hi.max(p);
        }
    }
    faces
}

/// A panel's six sides as mirrors.
fn panel_faces((lo, hi): (Vec3, Vec3)) -> [MirrorFace; 6] {
    let axis = |k: usize| [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)][k];
    let get = |v: Vec3, k: usize| [v.x, v.y, v.z][k];
    let side = |k: usize, at: Vec3| {
        let c = get(at, k);
        let (mut a, mut b) = (lo, hi);
        match k {
            0 => (a.x, b.x) = (c, c),
            1 => (a.y, b.y) = (c, c),
            _ => (a.z, b.z) = (c, c),
        }
        MirrorFace { plane: plane_of(axis(k), at), lo: a, hi: b }
    };
    [side(0, lo), side(0, hi), side(1, lo), side(1, hi), side(2, lo), side(2, hi)]
}

fn planes_eq(a: (Vec3, f32), b: (Vec3, f32)) -> bool {
    a.0.dot(b.0) > 0.999 && (a.1 - b.1).abs() < 1e-3
}

fn mirror_point(p: Vec3, (n, d): (Vec3, f32)) -> Vec3 {
    p - n * (2.0 * (n.dot(p) - d))
}

/// Paths kept per tick for the debug view.
const DEBUG_PATHS: usize = 64;

/// Keep `path` if it is among the strongest DEBUG_PATHS so far. Once the
/// list is full its weakest entry sits at index 0, so a path that cannot
/// make the cut is turned away with one comparison (this runs for every
/// next-event connection of every ray while debugging).
/// The reflection points of a mirror chain, by its images (the chain is
/// known valid): ears, each reflection, the source.
fn chain_points(planes: &[(Vec3, f32)], lis: Vec3, src: Vec3) -> Vec<Vec3> {
    let n = planes.len();
    let mut images = vec![src; n + 1];
    for k in (0..n).rev() {
        images[k] = mirror_point(images[k + 1], planes[k]);
    }
    let mut pts = vec![lis];
    let mut p = lis;
    for (k, &(nrm, d)) in planes.iter().enumerate() {
        let to = images[k] - p;
        let denom = nrm.dot(to);
        if denom.abs() < 1e-9 {
            break;
        }
        p = p + to * ((d - nrm.dot(p)) / denom);
        pts.push(p);
    }
    pts.push(src);
    pts
}

fn keep_path(list: &mut Vec<(u32, f32, Vec<Vec3>)>, id: u32, e: f32, verts: &[Vec3], end: Vec3) {
    let full = list.len() >= DEBUG_PATHS;
    if full && e <= list[0].1 {
        return;
    }
    let mut v = verts.to_vec();
    v.push(end);
    if full {
        list[0] = (id, e, v);
    } else {
        list.push((id, e, v));
        if list.len() < DEBUG_PATHS {
            return;
        }
    }
    let weakest = list.iter().enumerate().min_by(|a, b| a.1 .1.total_cmp(&b.1 .1)).map_or(0, |(i, _)| i);
    list.swap(0, weakest);
}

impl Tracer {
    /// Between full ticks: the exact paths' taps for where the listener
    /// and the sources are now, keyed as a full tick keys them — the
    /// direct path computed afresh, the cached edge bends re-timed through
    /// their bend points (their gains as the last search found them), every
    /// cached mirror chain re-solved. Per source, in `sources` order. The
    /// stochastic walk (the dense response, the reverb) is not touched:
    /// it changes slowly and follows at the next full tick.
    pub fn exact_taps(&mut self, world: &World, listener: Vec3, sources: &[Source]) -> Vec<Vec<Tap>> {
        let mut out: Vec<Vec<Tap>> = Vec::with_capacity(sources.len());
        for (si, s) in sources.iter().enumerate() {
            let mut taps = Vec::new();
            // the direct path, from where the source emitted what arrives
            if let Some(sp) = self.direct_from(si, s, listener) {
                let to = sp - listener;
                let d = to.length().max(MIN_DIST);
                let t = world.transmission(listener, sp, &mut self.hits);
                let air = air_attenuation(d);
                let gains: [f32; NBANDS] = core::array::from_fn(|b| t[b] * air[b] / d);
                if gains.iter().any(|&g| g > 1e-7) {
                    let dir = to * (1.0 / to.length().max(1e-6));
                    taps.push(Tap { key: KEY_DIRECT, delay_s: d / SPEED_OF_SOUND, dir: [dir.x, dir.y, dir.z], gains });
                }
            }
            // the bends, through this tick's bend points
            if let Some(c) = self.edge_cache.get(&s.id) {
                for (tap, p) in c.taps.iter().zip(&c.points) {
                    let last = *p.last().expect("a bend has its points");
                    if let Some(bp) = self.motion.emit(si, s.pos, last, polyline_len(listener, p)) {
                        taps.push(retimed(tap, p, listener, bp));
                    }
                }
            }
            // the mirror chains, re-solved
            let chains: Vec<(u64, Vec<(Vec3, f32)>)> =
                self.chains.iter().filter(|(k, _)| k.0 == s.id).map(|(k, c)| (k.1, c.planes.clone())).collect();
            for (key, planes) in chains {
                if let Some((e, delay, dir)) = self.solve_chain_at(world, &planes, listener, si, s.pos) {
                    taps.push(Tap {
                        key: KEY_CHAIN_BASE + (key % KEY_CHAIN_SPAN as u64) as u32,
                        delay_s: delay,
                        dir: [dir.x, dir.y, dir.z],
                        gains: core::array::from_fn(|b| e[b].max(0.0).sqrt()),
                    });
                }
            }
            out.push(taps);
        }
        out
    }

    /// The exactly solved paths (with `debug`): the direct paths and edge
    /// bends of the last tick, the mirror chains of the last solve.
    pub fn exact_paths(&self) -> impl Iterator<Item = &ExactPath> {
        self.exact.iter().chain(&self.exact_mirrors)
    }

    /// Forget every recorded debug path.
    pub fn clear_debug_paths(&mut self) {
        self.debug_paths.clear();
        self.exact.clear();
        self.exact_mirrors.clear();
    }

    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            cells: fibonacci_sphere(N_CELLS),
            state: HashMap::default(),
            hits: Vec::new(),
            sky: vec![([0.0; NBANDS], Vec3::new(0.0, 0.0, 0.0)); N_CELLS],
            sky_primed: false,
            sky_age: 0,
            sky_var: 0.0,
            sky_run: 0,
            sky_drift: 0.0,
            sky_world: false,
            sky_at: None,
            last_listener: None,
            epoch: 0,
            edges: None,
            tick: 0,
            ear_vis: None,
            edge_cache: HashMap::default(),
            heard: HashMap::default(),
            threads: 1,
            scratch: Vec::new(),
            src_vis: HashMap::default(),
            chains: HashMap::default(),
            rejected: HashMap::default(),
            faces: None,
            v4: false,
            v5: false,
            v6: false,
            costs: TickCosts::default(),
            late_pool: std::collections::VecDeque::new(),
            late_pool_sig: 0,
            world_sig: 0,
            rest: u32::MAX,
            moved_for: 0,
            debug: false,
            debug_paths: Vec::new(),
            exact: Vec::new(),
            exact_mirrors: Vec::new(),
            motion: Motion::default(),
            burst: 0,
        }
    }

    /// The sources' motion for the next `trace` / `exact_taps` (tracks in
    /// the order of the `sources` they get): every connection then reaches
    /// each source where it was when it emitted what arrives now.
    pub fn set_motion(&mut self, motion: Motion) {
        self.motion = motion;
    }

    /// A source's free-field arrival distance now (to where it emitted
    /// what arrives straight), None outside its track's span.
    fn direct_from(&self, si: usize, s: &Source, listener: Vec3) -> Option<Vec3> {
        self.motion.emit(si, s.pos, listener, 0.0)
    }

    /// A mirror chain solved for a moving source: the image of where the
    /// source was when it emitted what arrives along the chain now.
    fn solve_chain_at(&mut self, world: &World, planes: &[(Vec3, f32)], listener: Vec3, si: usize, cur: Vec3) -> Option<([f32; NBANDS], f32, Vec3)> {
        let Some(Some(tr)) = self.motion.tracks.get(si) else {
            return solve_chain(world, planes, listener, cur, &mut self.hits);
        };
        let tr = tr.clone();
        let now = self.motion.now;
        // where it emitted what arrives along the chain now: the light
        // cone to the listener from the chain's image of the source
        // (fixed-point steps contract by v/c). Its length needs no valid
        // path; only the place found is solved, legs and all. (Each step
        // solved, starting from where the source is now, a facade's echo
        // of a passing car went 8 m before the facade's end: the car's
        // place now mirrored past it while the one it emitted from didn't.)
        let image_len = |x: Vec3| {
            let im = planes.iter().rev().fold(x, |im, &pl| mirror_point(im, pl));
            (im - listener).length()
        };
        let mut x = tr.at(now);
        for _ in 0..4 {
            x = tr.at(now - (image_len(x) / SPEED_OF_SOUND) as f64);
        }
        let r = solve_chain(world, planes, listener, x, &mut self.hits)?;
        let t = now - r.1 as f64;
        (t >= tr.from && t <= tr.until).then_some(r)
    }

    /// The traced outdoor field: (unit direction, per-band energy) per
    /// direction cell. Energy is the fraction of the listener's sphere of
    /// directions that reaches open sky, weighted by everything lost on
    /// the way (1.0 = in the middle of an empty plain with a perfect
    /// mirror floor).
    pub fn sky(&self) -> Vec<([f32; 3], [f32; NBANDS])> {
        self.sky
            .iter()
            .enumerate()
            .map(|(i, (e, d))| {
                let dl = d.length();
                let dir = if dl > 1e-9 { *d * (1.0 / dl) } else { self.cells[i] };
                ([dir.x, dir.y, dir.z], *e)
            })
            .collect()
    }

    /// v4: a source's measured per-tick scatter of the dense response's
    /// fresh trace around its mean, per history window (variance of ln
    /// fresh/mean) — the estimate's own noise.
    pub fn dense_scatter(&self, id: u32) -> Option<[f32; HISTORY_WINDOWS]> {
        self.state.get(&id).map(|s| s.dense_var)
    }

    /// Source `to` takes over what `from` has learnt — its averaged
    /// response, its echo chains, its edge paths — and `from` starts over:
    /// a source that jumped leaves a ghost at the place it left (`to`),
    /// whose response is the one already converged there.
    pub fn fork(&mut self, from: u32, to: u32) {
        if let Some(st) = self.state.remove(&from) {
            self.state.insert(to, st);
        }
        let moved: Vec<(u64, MirrorChain)> =
            self.chains.iter().filter(|(k, _)| k.0 == from).map(|(k, c)| (k.1, c.clone())).collect();
        self.chains.retain(|k, _| k.0 != from);
        for (h, c) in moved {
            self.chains.insert((to, h), c);
        }
        if let Some(e) = self.edge_cache.remove(&from) {
            self.edge_cache.insert(to, e);
        }
        if let Some(v) = self.src_vis.remove(&from) {
            self.src_vis.insert(to, v);
        }
        if let Some(h) = self.heard.remove(&from) {
            self.heard.insert(to, h);
        }
    }

    /// Forget a source's history (it moved discontinuously, or left).
    pub fn reset(&mut self, id: u32) {
        self.state.remove(&id);
        self.chains.retain(|k, _| k.0 != id);
        self.rejected.retain(|k, _| k.0 != id);
    }

    /// Forget every average (the listener crossed a surface).
    /// How many ticks' estimates share each walked path (v5: the pool's).
    pub(crate) fn shared_ticks(&self) -> f32 {
        if self.v5 && self.v4 {
            (1 + LATE_POOL_TICKS) as f32
        } else {
            1.0
        }
    }

    fn forget_all(&mut self) {
        self.late_pool.clear();
        self.state.clear();
        self.chains.clear();
        self.rejected.clear();
        self.sky_primed = false;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Step 4b, shared with the externally walked fold: the sky. A running
    /// mean (moving or the world changing, at most the budget's EMA) while
    /// the fresh escapes agree with it — the whole sky's energy within
    /// `HISTORY_JUMP_Z` of its measured scatter, and not drifting
    /// (`HISTORY_DRIFT_Z`); beyond it (a door shut, a step indoors) it
    /// restarts from this tick's escapes at once.
    fn fold_sky(&mut self, fresh: impl Iterator<Item = ([f32; NBANDS], Vec3)>, budget: Budget, listener: Vec3) {
        let fresh: Vec<([f32; NBANDS], Vec3)> = fresh.collect();
        let f_tot: f32 = fresh.iter().map(|c| c.0[1]).sum();
        let h_tot: f32 = self.sky.iter().map(|c| c.0[1]).sum();
        let moved = self.sky_at != Some(listener) || std::mem::take(&mut self.sky_world);
        self.sky_at = Some(listener);
        // (nothing moved, the world as it was: the same truth sampled
        // again — no outlier restarts it)
        let changed = !self.sky_primed
            || moved && match (f_tot > 0.0, h_tot > 0.0) {
                (true, true) => {
                    let d = (f_tot / h_tot).ln();
                    // (a 1 % floor: a sky just primed has no scatter yet)
                    let z = d / self.sky_var.max(1e-4).sqrt();
                    self.sky_var += HISTORY_VAR_EMA * (d * d - self.sky_var);
                    if (z > 0.0 && self.sky_run < 0) || (z < 0.0 && self.sky_run > 0) {
                        self.sky_run = 0;
                        self.sky_drift = 0.0;
                    }
                    if z != 0.0 {
                        self.sky_run += z.signum() as i32;
                        self.sky_drift += z;
                    }
                    z.abs() > HISTORY_JUMP_Z
                        || (self.sky_run.abs() >= 2 && self.sky_drift.abs() / (self.sky_run.abs() as f32).sqrt() > HISTORY_DRIFT_Z)
                }
                (false, false) => false,
                _ => true,
            };
        if changed {
            self.sky_age = 0;
            self.sky_run = 0;
            self.sky_drift = 0.0;
        }
        let floor = if moved { budget.ema.clamp(STILL_ALPHA_FLOOR, 1.0) } else { 0.0 };
        let a = (1.0 / (self.sky_age + 1) as f32).max(floor);
        self.sky_age = self.sky_age.saturating_add(1);
        self.sky_primed = true;
        for (cur, f) in self.sky.iter_mut().zip(fresh) {
            for b in 0..NBANDS {
                cur.0[b] += a * (f.0[b] - cur.0[b]);
            }
            cur.1 = cur.1 + (f.1 - cur.1) * a;
        }
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
        let mut accs: Vec<Acc> = sources
            .iter()
            .enumerate()
            .map(|(si, s)| Acc::new((self.direct_from(si, s, listener).unwrap_or(s.pos) - listener).length() / SPEED_OF_SOUND, self.v4))
            .collect();

        let debug = self.debug;
        let mut dbg: Vec<(u32, f32, Vec<Vec3>)> = Vec::new();
        self.costs = TickCosts::default();

        // 1. direct, visibility, edge next-event paths (shared with the
        // externally walked tick, fold.rs)
        let clock = Stopwatch::start();
        let (direct, vis, mut edge_taps) = self.prepare(world, listener, sources, budget);
        self.costs.direct = clock.ms() - self.costs.edges;
        // the world just changed: more rays for the responses restarting
        self.costs.burst = self.burst > 0;
        let budget = if self.burst > 0 {
            self.burst -= 1;
            if self.v6 {
                // v6's averages start over (`SourceState::fold`): the late
                // family's mean gets as many samples within
                // `V6_BURST_TICKS` as its EMA holds — (2 − α)/α ≈ 19 ticks'
                // at α 0.1 — the early one's (≈ 5 at α 0.35) in the first
                // `BURST_TICKS`. Fewer, and a mean of a few ticks of a
                // heavy-tailed estimate sits under the truth and steps up
                // as the rare large samples come (the club through its
                // shut door: −17 dB, then +3, +5 dB over two seconds)
                let early = if self.burst + BURST_TICKS >= V6_BURST_TICKS { BURST_RAYS } else { 1 };
                Budget { rays: budget.rays * early, late_rays: budget.late_rays * V6_BURST_LATE, ..budget }
            } else {
                Budget { rays: budget.rays * BURST_RAYS, late_rays: budget.late_rays * BURST_RAYS, ..budget }
            }
        } else {
            budget
        };

        // 2. the ray walk, both families
        let sky_acc = self.walk_rays(world, listener, sources, budget, &vis, true, &mut accs, &mut dbg);

        if debug {
            dbg.sort_by(|a, b| b.1.total_cmp(&a.1));
            self.debug_paths = dbg;
        }
        self.fold_sky(sky_acc.into_iter(), budget, listener);

        // 4b. the exact solver's chains: named by a few mirror rays,
        // solved every tick while cached
        let clock = Stopwatch::start();
        self.find_chains(world, listener, sources, budget, chain_finders(budget));
        self.solve_chains(world, listener, sources, budget, &mut accs);
        self.costs.chains = clock.ms();



        // 5. smooth over ticks and build each source's field
        let clock = Stopwatch::start();
        let mut out = Vec::with_capacity(sources.len());
        let shared = self.shared_ticks();
        for (si, s) in sources.iter().enumerate() {
            let t_free = accs[si].t_free;
            let acc = core::mem::replace(
                &mut accs[si],
                Acc::new(0.0, self.v4),
            );
            let st = self.state.entry(s.id).or_default();
            if self.v4 {
                let total = acc.total;
                st.fold_v4(acc, budget, listener, s.pos);
                out.push(st.field_v4(t_free, direct[si], &mut edge_taps[si], total, shared));
            } else {
                st.fold(acc.early, acc.echo, budget, self.v6, listener, s.pos);
                out.push(st.field(&self.cells, t_free, direct[si], &mut edge_taps[si], acc.total));
            }
        }
        self.costs.fold = clock.ms();
        self.note_heard(sources, &out);
        out
    }

    /// What reached the listener from each source this tick (`heard`).
    fn note_heard(&mut self, sources: &[Source], out: &[SourceField]) {
        for (s, f) in sources.iter().zip(out) {
            let e: f32 = f.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum::<f32>() + f.reflected[1];
            self.heard.insert(s.id, e * s.weight);
        }
    }

    /// Step 2 of a tick: the ray walk, deposited into `accs` (one per
    /// source) and `dbg`; returns the sky cells' escapes. The EARLY family
    /// (paths with 0..=bounces reflections, `budget.rays` rays) only when
    /// `early`, then the LATE family (everything beyond, `budget.late_rays`)
    /// — a live GPU walk leaves only the late family to the CPU
    /// (`trace_walked`): long serial walks are what a GPU is slowest at.
    #[allow(clippy::too_many_arguments)]
    fn walk_rays(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        vis: &[[f32; NBANDS]],
        early: bool,
        accs: &mut [Acc],
        dbg: &mut Vec<(u32, f32, Vec<Vec3>)>,
    ) -> Vec<([f32; NBANDS], Vec3)> {
        let n = budget.rays.max(1);
        // 2. rays: stratified (Fibonacci) and randomly rotated per tick;
        // each draws its own random numbers, seeded from the tick's, its
        // family and its index — the walk is the same however many threads
        // share it (`walk_chunks`)
        let rot = random_rotation(&mut self.rng);
        let seed = self.rng.next_u64();
        let primary = if early { fibonacci_sphere(n as usize) } else { Vec::new() };
        // Three estimators, disjoint per path, so every path is counted
        // exactly once: EARLY rays count paths with 1..=bounces
        // reflections, LATE rays (fewer, long) everything beyond — except
        // the exact chain solver's (`solver_owns`: coherent, loud chains
        // that are long or arrive late — flutter, canyon echoes), which
        // `find_chains` names and `solve_chains` counts.
        let late_primary = fibonacci_sphere(budget.late_rays as usize);
        let mut sky_acc: Vec<([f32; NBANDS], Vec3)> =
            vec![([0.0; NBANDS], Vec3::new(0.0, 0.0, 0.0)); N_CELLS];
        let fams: [(&[Vec3], u32, u32, u32, usize); 2] = [
            (&primary, 0, budget.bounces, budget.bounces + 1 + MAX_BENDS, budget.nee),
            (&late_primary, budget.bounces + 1, u32::MAX, LATE_MAX_SEGMENTS + MAX_BENDS, budget.late_nee),
        ];
        // v5: the late family walks into estimates of its own — the rays
        // that bounced diffusely first (A: what follows doesn't depend on
        // where the listener stood) and the rest (B) — and the pool's
        // paths join A (`reuse_late`)
        let pooling = self.v5 && self.v4;
        let (mut late_a, mut late_b): (Vec<Acc>, Vec<Acc>) = if pooling {
            (accs.iter().map(|a| Acc::new(a.t_free, a.v4)).collect(), accs.iter().map(|a| Acc::new(a.t_free, a.v4)).collect())
        } else {
            (Vec::new(), Vec::new())
        };
        let mut records: Vec<LateRecord> = Vec::new();
        let mut late_rays = 0usize;
        let t_free: Vec<f32> = accs.iter().map(|a| a.t_free).collect();
        let threads = self.threads.max(1);
        if self.scratch.len() < threads {
            self.scratch.resize_with(threads, WalkScratch::default);
        }
        let edges = self.edges.take();
        let cx = WalkCtx {
            world,
            listener,
            sources,
            budget,
            vis,
            rot,
            cells: &self.cells,
            motion: &self.motion,
            edges: edges.as_ref(),
            t_free: &t_free,
            debug: self.debug,
            v4: self.v4,
        };
        for (fi, (dirs, lo, hi, segments, nee)) in fams.into_iter().enumerate() {
            if dirs.is_empty() || lo > hi {
                continue;
            }
            let clock = Stopwatch::start();
            let pooled = pooling && lo > 0;
            if pooled {
                late_rays = dirs.len();
            }
            let fam = Fam { lo, hi, segments, nee, inv_n: 1.0 / dirs.len() as f32, pooled };
            let walked = walk_chunks(&cx, &fam, &mut self.scratch[..threads], dirs, ray_seed(seed.wrapping_add(fi as u64)));
            // what the rays left, in ray order
            for out in walked {
                for (id, e, verts, end) in &out.dbg {
                    keep_path(dbg, *id, *e, verts, *end);
                }
                for (cell, e, d) in out.sky {
                    let c = &mut sky_acc[cell as usize];
                    for b in 0..NBANDS {
                        c.0[b] += e[b];
                    }
                    c.1 = c.1 + d;
                }
                if pooled {
                    let mut deps = out.deps.into_iter();
                    for (a, k, record) in out.rays {
                        let into = if a { &mut late_a } else { &mut late_b };
                        for (si, e, delay, half, first) in deps.by_ref().take(k as usize) {
                            deposit_spread(&mut into[si as usize], cx.cells, e, delay, first, half);
                        }
                        records.extend(record);
                    }
                } else {
                    for (si, e, delay, half, first) in out.deps {
                        deposit_spread(&mut accs[si as usize], cx.cells, e, delay, first, half);
                    }
                }
            }
            *if lo > 0 { &mut self.costs.late } else { &mut self.costs.early } += clock.ms();
        }
        self.edges = edges;
        if pooling && late_rays > 0 {
            self.reuse_late(world, listener, sources, late_rays as f32, records, &mut late_a);
            for (acc, (a, b)) in accs.iter_mut().zip(late_a.into_iter().zip(late_b)) {
                acc.merge(a);
                acc.merge(b);
            }
        }
        sky_acc
    }
    /// v5: the late family's diffuse-first estimate (`fresh`: this tick's
    /// rays, × 1/rays) joined by the pool's paths, reconnected to where
    /// the listener is now — per source the average over this tick's rays
    /// and every pooled tick's in which the source stood where it stands
    /// (those rays all count, reusable or not: the rest added nothing
    /// to this estimate). A pooled path is weighted by how the solid
    /// angle of its first bounce changed (cos θ / r², the reconnection's
    /// Jacobian), priced through whatever is between, its delays moved by
    /// the new first leg, its air by the new length: exact, unbiased.
    /// Then this tick's paths join the pool.
    fn reuse_late(&mut self, world: &World, listener: Vec3, sources: &[Source], rays: f32, records: Vec<LateRecord>, fresh: &mut [Acc]) {
        if self.late_pool_sig != self.world_sig {
            self.late_pool.clear();
            self.late_pool_sig = self.world_sig;
        }
        let mut reused: Vec<Acc> = fresh.iter().map(|a| Acc::new(a.t_free, a.v4)).collect();
        let mut samples = vec![0.0f32; sources.len()];
        for tick in &self.late_pool {
            if (tick.listener - listener).length() > LATE_POOL_REACH_M {
                continue;
            }
            // the sources standing where they stood then
            let still: Vec<bool> =
                sources.iter().map(|s| tick.sources.iter().any(|(id, p)| *id == s.id && (*p - s.pos).length() < 1e-3)).collect();
            for (n, &ok) in samples.iter_mut().zip(&still) {
                if ok {
                    *n += tick.rays;
                }
            }
            for rec in &tick.records {
                let to = listener - rec.v1;
                let r1 = to.length();
                if r1 < 1e-3 {
                    continue;
                }
                let cos1 = to.dot(rec.n1) / r1;
                if cos1 <= 1e-3 {
                    continue; // the bounce faces away: it doesn't reach here
                }
                let jac = (cos1 / (r1 * r1)) / (rec.cos1 / (rec.r1 * rec.r1));
                let t = world.transmission(listener, rec.v1 + rec.n1 * (2.0 * EPS), &mut self.hits);
                let dir = (rec.v1 - listener) * (1.0 / r1);
                let shift = (r1 - rec.r1) / SPEED_OF_SOUND;
                for &(id, e, delay, half) in &rec.deps {
                    let Some(si) = sources.iter().position(|s| s.id == id) else { continue };
                    if !still[si] {
                        continue;
                    }
                    let (a0, a1) = (air_attenuation(delay * SPEED_OF_SOUND), air_attenuation((delay + shift) * SPEED_OF_SOUND));
                    let e2: [f32; NBANDS] = core::array::from_fn(|b| {
                        let air = if a0[b] > 0.0 { (a1[b] / a0[b]).powi(2) } else { 0.0 };
                        e[b] * tick.rays * jac * t[b] * t[b] * air
                    });
                    deposit_spread(&mut reused[si], &self.cells, e2, delay + shift, dir, half);
                }
            }
        }
        for ((f, r), &n) in fresh.iter_mut().zip(reused.iter_mut()).zip(&samples) {
            let total = rays + n;
            f.scale(rays / total);
            r.scale(1.0 / total);
            let r = std::mem::replace(r, Acc::new(0.0, false));
            f.merge(r);
        }
        let snapshot = sources.iter().map(|s| (s.id, s.pos)).collect();
        self.late_pool.push_front(PoolTick { records, rays, listener, sources: snapshot });
        self.late_pool.truncate(LATE_POOL_TICKS);
    }

    /// Is the chain `key` held out, rejected lately? One with no exact
    /// path only while the listener and its source (at `src`, `moving`
    /// or not) stay where they were: an end that moved may have given it
    /// one (a facade's echo of a passing car exists while the mirror
    /// point is on the facade — ~1.8 s for a car 30 m off at 100 km/h, all
    /// of it inside a hold begun as the car came up: the echo was never
    /// heard). One that lost to stronger chains: for the whole hold.
    fn held(&self, key: (u32, u64), listener: Vec3, src: Vec3, moving: bool) -> bool {
        match self.rejected.get(&key) {
            Some(&(t, at)) if self.tick < t + CHAIN_REJECT_TICKS => at.is_none_or(|(l, s)| !moving && l == listener && s == src),
            _ => false,
        }
    }

    /// Cache (or refresh) the chain `planes` for source `id` (at `src`,
    /// `moving` or not; the listener at `listener`).
    fn name_chain(&mut self, id: u32, planes: &[(Vec3, f32)], cached: &mut HashMap<u32, usize>, (listener, src, moving): (Vec3, Vec3, bool)) {
        let key = (id, planes_hash(planes));
        if self.held(key, listener, src, moving) {
            return;
        }
        if let Some(c) = self.chains.get_mut(&key) {
            c.ttl = CHAIN_TTL;
        } else if cached.get(&id).copied().unwrap_or(0) < 2 * MAX_CHAINS {
            self.chains.insert(key, MirrorChain { planes: planes.to_vec(), ttl: CHAIN_TTL, energy: 0.0, grown_at: 0, solved: None });
            *cached.entry(id).or_default() += 1;
        }
    }

    /// The chain finder (the CPU walk's and the GPU kernel's alike): `n`
    /// rays from the listener that only mirror — ending where a surface
    /// scatters, as often as it scatters — naming the chains the solver
    /// owns (`solver_owns`) as they pass near a source. Directions only:
    /// no energy is counted here.
    fn find_chains(&mut self, world: &World, listener: Vec3, sources: &[Source], budget: Budget, n: usize) {
        let mut cached: HashMap<u32, usize> = HashMap::default();
        for k in self.chains.keys() {
            *cached.entry(k.0).or_default() += 1;
        }
        let t_free: Vec<f32> = sources.iter().map(|s| (s.pos - listener).length() / SPEED_OF_SOUND).collect();
        // past the echo window of the farthest source, nothing is named
        let window = if self.v4 { RESPONSE_MAX_S } else { ECHO_S };
        let reach = (window + t_free.iter().copied().fold(0.0, f32::max)) * SPEED_OF_SOUND;
        self.name_mirrors(world, listener, sources, budget, reach, &t_free, &mut cached);
        let mut planes: Vec<(Vec3, f32)> = Vec::new();
        for _ in 0..n {
            let mut dir = self.rng.unit_sphere();
            let mut origin = listener;
            let mut length = 0.0f32;
            let mut mirror_thr = 1.0f32;
            planes.clear();
            for refl in 0..=MAX_COHERENT_REFL {
                let hit = world.raycast(origin, dir);
                let t_seg = hit.as_ref().map_or(f32::MAX, |h| h.t);
                if refl >= 1 {
                    for (si, s) in sources.iter().enumerate() {
                        let to = s.pos - origin;
                        let along = to.dot(dir);
                        if along <= 0.0 || along >= t_seg {
                            continue;
                        }
                        let l_s = (length + to.length()).max(MIN_DIST);
                        let rel = l_s / SPEED_OF_SOUND - t_free[si];
                        let owns = self.v4 || refl > budget.bounces || rel >= EARLY_S;
                        if owns && (origin + dir * along - s.pos).length() <= CHAIN_FIND_K * l_s {
                            let moving = self.motion.moving(si);
                            self.name_chain(s.id, &planes, &mut cached, (listener, s.pos, moving));
                        }
                    }
                }
                let Some(hit) = hit else { break };
                if length + hit.t > reach {
                    break;
                }
                let m = hit.material;
                mirror_thr *= (1.0 - m.absorption[1] - m.transmission[1] * m.transmission[1]).max(0.0) * (1.0 - m.scattering);
                // the walk's own odds: a scattering surface ends the
                // chain as often as it scatters (rough chains are rare)
                if mirror_thr < COHERENT_FLOOR || self.rng.next_f32() < m.scattering {
                    break;
                }
                let p = origin + dir * hit.t;
                length += hit.t;
                planes.push(plane_of(hit.normal, p));
                dir = dir - hit.normal * (2.0 * dir.dot(hit.normal));
                origin = p + hit.normal * EPS;
            }
        }
    }

    /// First-order chains are not left to the finders: every flat face
    /// that mirrors a source to the listener — the mirror point of where
    /// the source emitted what arrives that way inside the face's box —
    /// is named every tick (the solve validates it, legs and all). A
    /// facade 100 m off fills ~0.1 % of the finders' sphere, and its echo
    /// of a passing car lives a second or two: the finders came across it
    /// in some runs and not in others, and an echo unnamed is not heard at
    /// all (the walk leaves the solver's paths to it).
    #[allow(clippy::too_many_arguments)]
    fn name_mirrors(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        reach: f32,
        t_free: &[f32],
        cached: &mut HashMap<u32, usize>,
    ) {
        let mesh_id = world.mesh.positions.as_ptr() as usize ^ world.mesh.indices.len();
        if self.faces.as_ref().is_none_or(|f| f.0 != mesh_id) {
            self.faces = Some((mesh_id, mirror_faces(world.mesh)));
        }
        let (id, faces) = self.faces.take().expect("built above");
        let panels: Vec<MirrorFace> = world.panels.iter().flat_map(|&(lo, hi, _)| panel_faces((lo, hi))).collect();
        // (1 cm: a mirror point on the face's edge)
        let pad = Vec3::new(0.01, 0.01, 0.01);
        let within = |p: Vec3, f: &MirrorFace| {
            let (lo, hi) = (f.lo - pad, f.hi + pad);
            p.x >= lo.x && p.y >= lo.y && p.z >= lo.z && p.x <= hi.x && p.y <= hi.y && p.z <= hi.z
        };
        // the ground, under every outdoor echo
        let ground = world.grade.map(|g| plane_of(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, g)));
        for f in faces.iter().chain(&panels) {
            let (n, d) = f.plane;
            let sl = n.dot(listener) - d;
            if sl.abs() < 1e-3 {
                continue;
            }
            let image = listener - n * (2.0 * sl);
            for (si, s) in sources.iter().enumerate() {
                // where it emitted what arrives off this mirror now
                let Some(x) = self.motion.emit(si, s.pos, image, 0.0) else { continue };
                let sx = n.dot(x) - d;
                if sx * sl <= 0.0 || sx.abs() < 1e-3 {
                    continue;
                }
                let len = (x - image).length();
                let rel = len / SPEED_OF_SOUND - t_free[si];
                // the solver's (`solver_owns`), within the echo window
                if len > reach || !(self.v4 || 1 > budget.bounces || rel >= EARLY_S) {
                    continue;
                }
                let q = x + (image - x) * (sx / (sx + sl));
                if !within(q, f) {
                    continue;
                }
                let moving = self.motion.moving(si);
                self.name_chain(s.id, &[f.plane], cached, (listener, s.pos, moving));
                // and its twins off the ground, a few millimetres longer
                // when both ends stand low over it: the ground first or
                // last (the direct sound's own twin is the ground's
                // first-order chain) — the finders' rays come across them
                // as rarely
                let Some(g) = ground.filter(|&g| !planes_eq(f.plane, g)) else { continue };
                for (twin, at) in [([g, f.plane], 2), ([f.plane, g], 1)] {
                    if within(chain_points(&twin, listener, x)[at], f) {
                        self.name_chain(s.id, &twin, cached, (listener, s.pos, moving));
                    }
                }
            }
        }
        self.faces = Some((id, faces));
    }

    /// The CPU's share of an externally walked tick: name chains with a
    /// few mirror rays, solve every cached one, and return what they
    /// deposit (per source) — merged into the walk's bins by the fold.
    fn walked_chains(&mut self, world: &World, listener: Vec3, sources: &[Source], budget: Budget) -> Vec<Acc> {
        self.find_chains(world, listener, sources, budget, chain_finders(budget));
        let mut accs: Vec<Acc> = sources
            .iter()
            .enumerate()
            .map(|(si, s)| Acc::new((self.direct_from(si, s, listener).unwrap_or(s.pos) - listener).length() / SPEED_OF_SOUND, self.v4))
            .collect();
        self.solve_chains(world, listener, sources, budget, &mut accs);
        accs
    }

    /// Every cached mirror chain longer than `bounces`, solved exactly
    /// for its source: valid ones deposit their image-source energy
    /// (mirror losses × air / L²) at their exact delay; invalid or
    /// expired ones leave the cache.
    fn solve_chains(&mut self, world: &World, listener: Vec3, sources: &[Source], budget: Budget, accs: &mut [Acc]) {
        self.grow_trains();
        let mut drop: Vec<(u32, u64)> = Vec::new();
        // rejected: no exact path (where the listener and the source
        // were), or short and early (the sphere estimate's, held)
        let mut reject: Vec<((u32, u64), Option<(Vec3, Vec3)>)> = Vec::new();
        let mut mirrors: Vec<ExactPath> = Vec::new();
        let keys: Vec<(u32, u64)> = self.chains.keys().copied().collect();
        for key in keys {
            let Some(si) = sources.iter().position(|s| s.id == key.0) else {
                drop.push(key);
                continue;
            };
            let c = self.chains.get_mut(&key).expect("listed above");
            c.ttl = c.ttl.saturating_sub(1);
            if c.ttl == 0 {
                drop.push(key);
                continue;
            }
            let long = c.planes.len() as u32 > budget.bounces;
            let src = sources[si].pos;
            let tick = self.tick;
            let result = match c.solved {
                Some((l, s, t, r)) if l == listener && s == src && tick < t + CHAIN_RESOLVE_TICKS => Some(r),
                _ if self.motion.moving(si) => {
                    // a moving source: the image of where it emitted
                    let planes = c.planes.clone();
                    let r = self.solve_chain_at(world, &planes, listener, si, src);
                    self.chains.get_mut(&key).expect("listed above").solved = r.map(|r| (listener, src, tick, r));
                    r
                }
                _ => {
                    let r = solve_chain(world, &c.planes, listener, src, &mut self.hits);
                    c.solved = r.map(|r| (listener, src, tick, r));
                    r
                }
            };
            if let (true, Some((e, delay, _))) = (self.debug, result) {
                // (v3: short early chains are the sphere estimate's; a
                // last leg through a solid wall carries nothing)
                if e[1] > 0.0 && (self.v4 || long || delay - accs[si].t_free >= EARLY_S) {
                    let points = chain_points(&self.chains[&key].planes, listener, src);
                    mirrors.push(ExactPath { source: key.0, kind: 2, key: key.1, energy: e[1], points });
                }
            }
            match result {
                // v4: every valid chain is a coherent arrival — the field
                // decides whether it plays as its own tap or joins the
                // dense response (`SourceState::field_v4`)
                Some((e, delay, dir)) if self.v4 => {
                    // a chain that solves exists: it stays while it does
                    // (forgetting it because the finders happened not to
                    // pass by again took a real echo away, and the next
                    // find brought it back)
                    let c = self.chains.get_mut(&key).expect("listed above");
                    c.energy = e[1];
                    c.ttl = CHAIN_TTL;
                    let rel = delay - accs[si].t_free;
                    if rel < RESPONSE_MAX_S {
                        let k = KEY_CHAIN_BASE + (key.1 % KEY_CHAIN_SPAN as u64) as u32;
                        let b = accs[si].early.entry(k).or_default();
                        b.add(e, delay, dir);
                        b.last = rel;
                    }
                }
                // short and early: the sphere estimate's (see `solver_owns`)
                Some((_, delay, _)) if !long && delay - accs[si].t_free < EARLY_S => reject.push((key, None)),
                Some((e, delay, dir)) => {
                    self.chains.get_mut(&key).expect("listed above").energy = e[1];
                    // the echogram (and, inside the early window, the
                    // early bins); past it, the chain's own echo bin
                    deposit(&mut accs[si], &self.cells, e, delay, dir);
                    let rel = delay - accs[si].t_free;
                    if (EARLY_S..ECHO_S).contains(&rel) {
                        let k = KEY_CHAIN_BASE + (key.1 % KEY_CHAIN_SPAN as u64) as u32;
                        let b = accs[si].early.entry(k).or_default();
                        b.add(e, delay, dir);
                        b.last = rel;
                    }
                }
                None => reject.push((key, Some((listener, src)))),
            }
        }
        for k in drop {
            self.chains.remove(&k);
        }
        self.exact_mirrors = mirrors;
        let tick = self.tick;
        for (k, at) in reject {
            self.chains.remove(&k);
            self.rejected.insert(k, (tick, at));
        }
        self.rejected.retain(|_, r| tick < r.0 + CHAIN_REJECT_TICKS);
        // over the cap (newly named ones get one solve to prove
        // themselves): keep each source's MAX_CHAINS that count — v3 the
        // strongest; v4 those standing out most of the scattered field
        // arriving with them (`DISTINCT_SPAN_S`, its averaged dense
        // response): a flutter's late echoes over the early chains via
        // absorbing surfaces, louder but buried in their field — the
        // same measure the field picks its taps by
        let span = (DISTINCT_SPAN_S / DENSE_BIN_S).ceil() as usize;
        let rank = |k: &(u32, u64), c: &MirrorChain| -> f32 {
            let (true, Some((_, _, _, (e, delay, _)))) = (self.v4, c.solved) else { return c.energy };
            let Some(s) = sources.iter().find(|s| s.id == k.0) else { return 0.0 };
            let rel = delay - (s.pos - listener).length() / SPEED_OF_SOUND;
            let i = (rel.max(0.0) / DENSE_BIN_S) as usize;
            let around: f32 = self.state.get(&k.0).map_or(0.0, |st| {
                st.dense.iter().skip(i.saturating_sub(span)).take(2 * span + 1).map(|b| b[1]).sum()
            });
            e[1] / around.max(1e-30)
        };
        let mut by_source: HashMap<u32, Vec<((u32, u64), f32)>> = HashMap::default();
        for (k, c) in &self.chains {
            by_source.entry(k.0).or_default().push((*k, rank(k, c)));
        }
        for (_, mut v) in by_source {
            if v.len() > MAX_CHAINS {
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (k, _) in &v[MAX_CHAINS..] {
                    self.chains.remove(k);
                    self.rejected.insert(*k, (tick, None));
                }
            }
        }
    }

    /// A chain whose last mirrors alternate between two parallel planes
    /// is one member of a train — the flutter between them: every other
    /// number of round trips is a path too, as rare for a ray to find as
    /// this one was. Propose them all (the solve validates each).
    fn grow_trains(&mut self) {
        let mut per_source: HashMap<u32, usize> = HashMap::default();
        for k in self.chains.keys() {
            *per_source.entry(k.0).or_default() += 1;
        }
        let mut new: Vec<((u32, u64), MirrorChain)> = Vec::new();
        let mut grown: Vec<(u32, u64)> = Vec::new();
        let tick = self.tick;
        for (key, c) in &self.chains {
            let p = &c.planes;
            let n = p.len();
            if n < 2 || (c.grown_at != 0 && tick < c.grown_at + TRAIN_REGROW_TICKS) {
                continue;
            }
            grown.push(*key);
            let (a, b) = (p[n - 2], p[n - 1]);
            if a.0.dot(b.0) < 0.999 || (a.1 - b.1).abs() < 0.01 {
                continue; // not a parallel pair: no train
            }
            // where the alternation starts
            let mut start = n - 2;
            while start > 0 && planes_eq(p[start - 1], p[start + 1]) {
                start -= 1;
            }
            let phase = if (n - 2 - start) % 2 == 0 { (a, b) } else { (b, a) };
            // this chain's own family (its prefix, then the alternation),
            // and the pure alternation both ways round (off either wall
            // first): the train's halves
            let families = [(start, phase), (0, phase), (0, (phase.1, phase.0))];
            for &(pre, (first, second)) in &families {
                for len in pre + 2..=MAX_COHERENT_REFL as usize {
                    let planes: Vec<(Vec3, f32)> =
                        (0..len).map(|i| if i < pre { p[i] } else if (i - pre) % 2 == 0 { first } else { second }).collect();
                    let k = (key.0, planes_hash(&planes));
                    let count = per_source.entry(key.0).or_default();
                    let rejected = self.rejected.get(&k).is_some_and(|r| tick < r.0 + CHAIN_REJECT_TICKS);
                if *count >= 2 * MAX_CHAINS || rejected || self.chains.contains_key(&k) || new.iter().any(|(nk, _)| *nk == k) {
                        continue;
                    }
                    *count += 1;
                    new.push((k, MirrorChain { planes, ttl: c.ttl, energy: 0.0, grown_at: 0, solved: None }));
                }
            }
        }
        for k in grown {
            if let Some(c) = self.chains.get_mut(&k) {
                c.grown_at = tick.max(1);
            }
        }
        self.chains.extend(new);
    }

    /// Everything before the walk: the exact direct segment per source
    /// (step 1), each source's per-band direct visibility, and the
    /// edge next-event paths (diffraction); with `debug`, the exact
    /// direct and bent paths for the debug view.
    #[allow(clippy::type_complexity)]
    fn prepare(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
    ) -> (Vec<Option<Tap>>, Vec<[f32; NBANDS]>, Vec<Vec<Tap>>) {
        // 0. crossed a surface since last tick (flew through a wall)?
        // Then nothing averaged on the old side can reach this one.
        // (the segment reaches 1 cm past both ends: a listener exactly ON
        // a wall face last tick or this one still counts as crossing it)
        if let Some(prev) = self.last_listener {
            let step = listener - prev;
            let len = step.length();
            if len > 1e-4 {
                let pad = step * (0.01 / len);
                if world.transmission(prev - pad, listener + pad, &mut self.hits)[1] < 0.999 {
                    self.forget_all();
                }
            }
        }
        self.last_listener = Some(listener);
        // the world changed (an object moved, a door closed): every cached
        // solve is redone now, and v4's responses and the sky follow the
        // next traces as when moving (their history tests say what changed:
        // a door swinging shut must not make every tick a fresh, noisy
        // estimate)
        let sig = panels_sig(world.panels);
        let changed = sig != self.world_sig;
        if changed {
            self.world_sig = sig;
            for c in self.chains.values_mut() {
                c.solved = None;
            }
            self.rejected.clear();
            self.edge_cache.clear();
            self.src_vis.clear();
            self.ear_vis = None;
            self.sky_world = true;
        }
        // v6: the averages restart (and trace the burst) when the world
        // starts changing after a rest, and when it comes to rest after
        // changing on: a door slammed in one tick restarts once; a door
        // swinging shut over a second restarts as it starts, goes on as
        // moving ticks while on its way (`SourceState::fold`), and starts
        // over where it stops — nothing averaged on the way describes the
        // shut door (a leaf a few millimetres ajar leaks ~14 dB more).
        // A change after a pause shorter than the early average's memory
        // ((2 − α)/α ticks) is the same motion going on: a leaf easing
        // into its frame moves under the millimetre the signature sees
        // for a tick or two before it closes. Restarting at every tick of
        // the swing played each tick's own heavy-tailed estimate (55–110
        // dB of climbing back, clicks; `assess doorswing EAR=0`); not
        // restarting where it stops kept the swing's field in the averages
        // for seconds (Godot: +9…+11 dB 3 s after the shut command, now
        // 0…+5)
        let memory = ((2.0 - budget.ema.max(0.05)) / budget.ema.max(0.05)).ceil() as u32;
        let settled = !changed && self.rest == 0 && self.moved_for >= 2;
        let restart = if self.v6 { (changed && self.rest >= memory) || settled } else { changed };
        self.moved_for = if changed { if self.rest < memory { self.moved_for + 1 } else { 1 } } else if self.rest >= memory { 0 } else { self.moved_for };
        self.rest = if changed { 0 } else { self.rest.saturating_add(1) };
        if restart || (self.v6 && changed) {
            for st in self.state.values_mut() {
                st.restart |= restart;
                st.world_moving |= self.v6 && changed;
            }
        }
        if restart {
            self.burst = if self.v6 { V6_BURST_TICKS } else { BURST_TICKS };
        }
        self.exact.clear();
        let debug = self.debug;
        // 1. direct: one exact segment per source
        let mut direct: Vec<Option<Tap>> = Vec::with_capacity(sources.len());
        // where each source emitted what arrives straight now (moving
        // sources: back along their track; None: outside its span)
        let emitted: Vec<Option<Vec3>> = sources.iter().enumerate().map(|(si, s)| self.direct_from(si, s, listener)).collect();
        for (s, sp) in sources.iter().zip(&emitted) {
            let Some(sp) = *sp else {
                direct.push(None);
                continue;
            };
            let to = sp - listener;
            let d = to.length().max(MIN_DIST);
            let t = world.transmission(listener, sp, &mut self.hits);
            let air = air_attenuation(d);
            let gains: [f32; NBANDS] = core::array::from_fn(|b| t[b] * air[b] / d);
            if debug && gains[1] > 1e-7 {
                self.exact.push(ExactPath { source: s.id, kind: 0, key: 0, energy: gains[1] * gains[1], points: vec![listener, sp] });
            }
            direct.push(if gains.iter().any(|&g| g > 1e-7) {
                let dir = to * (1.0 / to.length().max(1e-6));
                Some(Tap { key: KEY_DIRECT, delay_s: d / SPEED_OF_SOUND, dir: [dir.x, dir.y, dir.z], gains })
            } else {
                None
            });
        }

        // per-band direct visibility: diffracted DIRECT paths only carry
        // what the straight ray does not (the exact direct already does)
        let vis: Vec<[f32; NBANDS]> = sources
            .iter()
            .zip(&direct)
            .map(|(_, d)| match d {
                Some(t) => {
                    let dd = (t.delay_s * SPEED_OF_SOUND).max(MIN_DIST);
                    let air = air_attenuation(dd);
                    core::array::from_fn(|b| (t.gains[b] * dd / air[b].max(1e-9)).clamp(0.0, 1.0))
                }
                None => [0.0; NBANDS],
            })
            .collect();
        if budget.diffraction {
            let key = (world.mesh.indices.len(), world.mesh.positions.len());
            if self.edges.as_ref().map_or(true, |e| e.key != key) {
                self.edges = Some(EdgeSet::build(world.mesh));
                // the paths kept name edges of the old set
                self.edge_cache.clear();
                self.src_vis.clear();
                self.ear_vis = None;
            }
            if let Some(es) = self.edges.as_mut() {
                es.mask_filled(world.panels, world.grade);
            }
        }

        self.tick += 1;
        let edges_clock = Stopwatch::start();
        let edge_taps = self.edge_paths(world, listener, sources, &emitted, &vis, budget);
        self.costs.edges = edges_clock.ms();
        (direct, vis, edge_taps)
    }

    /// Edge next-event estimation: the DIRECT field around one or two
    /// edges, connected deterministically for every blocked source (a
    /// random ray finding a distant source after bending is too rare to
    /// carry it). One edge: knife-edge (Kurze–Anderson) per band; two: the
    /// string over both, priced by its detour (ISO 9613-2). Legs are priced
    /// through whatever they cross, only the share the straight ray
    /// doesn't carry; distinct edges add (each is its own path).
    ///
    /// Each source keeps the edges its last search found and solves them
    /// again every tick for where both ends are now — bend points, legs,
    /// prices: exact and continuous however it moves. Searching which
    /// edges carry it (every exposed edge for one bend, pairs for two)
    /// costs far more, so searches go by perceptual importance: each tick
    /// at most `Budget::edge_searches` (one bend) and
    /// `Budget::edge2_searches` (two) sources search, those whose stale
    /// search costs the ear most first — how loud they arrive (`heard`, as
    /// loudness) × how far either end moved since their last search.
    fn edge_paths(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        emitted: &[Option<Vec3>],
        vis: &[[f32; NBANDS]],
        budget: Budget,
    ) -> Vec<Vec<Tap>> {
        let mut edge_taps: Vec<Vec<Tap>> = vec![Vec::new(); sources.len()];
        if !budget.diffraction {
            return edge_taps;
        }
        let Some(es) = self.edges.take() else { return edge_taps };
        let tick = self.tick;
        let debug = self.debug;

        // who searches: the blocked sources due a search (either end
        // moved `EDGE_REUSE_M` since; never searched: first), the most
        // needed first
        let mut due1: Vec<(bool, f32, usize)> = Vec::new();
        let mut due2: Vec<(bool, f32, usize)> = Vec::new();
        for (si, s) in sources.iter().enumerate() {
            if vis[si][1] > 0.999 {
                continue;
            }
            let Some(sp) = emitted[si] else { continue };
            let loud = self.heard.get(&s.id).map_or(0.0, |&e| e.powf(LOUDNESS_EXP));
            let need = |at: Option<(Vec3, Vec3, u64)>| match at {
                None => (true, loud),
                Some((l, p, _)) => {
                    let n = (l - listener).length().max((p - sp).length()) / EDGE_REUSE_M;
                    (false, if n >= 1.0 { loud * n } else { -1.0 })
                }
            };
            let c = self.edge_cache.get(&s.id);
            let (first, n) = need(c.and_then(|c| c.one_at));
            if first || n >= 0.0 {
                due1.push((first, n, si));
            }
            let (first, n) = need(c.and_then(|c| c.two_at));
            if first || n >= 0.0 {
                due2.push((first, n, si));
            }
        }
        let most_needed = |a: &(bool, f32, usize), b: &(bool, f32, usize)| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1));
        due1.sort_by(most_needed);
        due1.truncate(budget.edge_searches);
        due2.sort_by(most_needed);
        due2.truncate(budget.edge2_searches);

        // which edges the listener sees, as the searches ask (shared by
        // every search; while the listener stays put)
        if self.ear_vis.as_ref().is_none_or(|(p, m)| (*p - listener).length() > EDGE_REUSE_M || m.len() != es.edges.len()) {
            self.ear_vis = Some((listener, vec![0; es.edges.len()]));
        }
        // one bend: rank every exposed edge by its pure knife-edge bound,
        // price the best the listener sees, keep the strongest distinct
        for &(_, _, si) in &due1 {
            let s = &sources[si];
            let sp = emitted[si].expect("due sources emitted");
            let ear_seen = &mut self.ear_vis.as_mut().expect("set above").1;
            let direct_len = (sp - listener).length().max(MIN_DIST);
            let mut cand: Vec<(f32, usize)> = es
                .edges
                .iter()
                .enumerate()
                .filter(|(ei, _)| !es.filled[*ei])
                .filter_map(|(ei, (a, b))| {
                    let p = crate::paths::bend_point(*a, *b, sp, listener);
                    let (d1, d2) = ((p - listener).length(), (sp - p).length());
                    if d1 < 0.2 || d2 < 0.2 {
                        return None;
                    }
                    // a bend AT ground level is no silhouette for an
                    // above-ground path (it is where an edge meets the
                    // ground, not where it stands out)
                    if world.grade.is_some_and(|g| p.z < g + 0.3 && listener.z > g) {
                        return None;
                    }
                    let ka = crate::diffraction::knife_edge_bands((d1 + d2 - direct_len).max(1e-4));
                    Some((ka[0] / (d1 + d2), ei))
                })
                .collect();
            cand.sort_by(|x, y| y.0.total_cmp(&x.0));
            let mut found: Vec<(f32, Vec3, usize)> = cand
                .into_iter()
                .take(EDGE_LOOKS)
                .filter(|&(_, ei)| ear_sees(world, &es, ear_seen, ei, listener))
                .take(EDGE_CANDIDATES)
                .filter_map(|(_, ei)| price_one(world, &es, &mut self.hits, ei, listener, sp, vis[si]).map(|(p, tap)| (tap.gains[0] + tap.gains[1], p, ei)))
                .collect();
            found.sort_by(|x, y| y.0.total_cmp(&x.0));
            let mut kept: Vec<(Vec3, usize)> = Vec::new();
            for (_, p, ei) in found {
                if kept.len() >= EDGE_PATHS {
                    break;
                }
                // coincident edges (a parapet over its slab) are one path
                if kept.iter().any(|q| (q.0 - p).length() < 1.0) {
                    continue;
                }
                kept.push((p, ei));
            }
            let c = self.edge_cache.entry(s.id).or_default();
            c.one = kept.into_iter().map(|k| k.1).collect();
            c.one_at = Some((listener, sp, tick));
        }
        // two bends: the string over a thick building or around two
        // corners, where no single bend carries the field
        for &(_, _, si) in &due2 {
            let s = &sources[si];
            let sp = emitted[si].expect("due sources emitted");
            let ear_seen = &mut self.ear_vis.as_mut().expect("set above").1;
            let sv = self.src_vis.entry(s.id).or_insert_with(|| (sp, vec![0; es.edges.len()]));
            if (sv.0 - sp).length() > EDGE_REUSE_M || sv.1.len() != es.edges.len() {
                *sv = (sp, vec![0; es.edges.len()]);
            }
            let clock2 = Stopwatch::start();
            let two = two_edge_paths(world, &es, ear_seen, &mut sv.1, &mut self.hits, listener, sp, vis[si]);
            self.costs.edges2 += clock2.ms();
            // where the source's single bends are now: a string through
            // one of them is that bend's path
            let ones: Vec<Vec3> = self.edge_cache.get(&s.id).map_or(Vec::new(), |c| {
                c.one.iter().map(|&ei| crate::paths::bend_point(es.edges[ei].0, es.edges[ei].1, sp, listener)).collect()
            });
            let c = self.edge_cache.entry(s.id).or_default();
            c.two = two
                .into_iter()
                .filter(|(_, _, pts)| !ones.iter().any(|q| (*q - pts[0]).length() < 1.0 || (*q - pts[1]).length() < 1.0))
                .map(|(e1, e2, _)| (e1, e2))
                .collect();
            c.two_at = Some((listener, sp, tick));
        }

        // every blocked source's paths, solved for now
        for (si, s) in sources.iter().enumerate() {
            if vis[si][1] > 0.999 {
                continue;
            }
            let Some(sp) = emitted[si] else { continue };
            let Some(c) = self.edge_cache.get_mut(&s.id) else { continue };
            c.taps.clear();
            c.points.clear();
            for &ei in &c.one {
                if let Some((p, tap)) = price_one(world, &es, &mut self.hits, ei, listener, sp, vis[si]) {
                    c.taps.push(tap);
                    c.points.push(vec![p]);
                }
            }
            for &(e1, e2) in &c.two {
                let (l, p1, p2) = refine_two(&es, e1, e2, listener, sp);
                if let Some((tap, _)) = price_two(world, &es, &mut self.hits, e1, e2, l, p1, p2, listener, sp, vis[si]) {
                    c.taps.push(tap);
                    c.points.push(vec![p1, p2]);
                }
            }
            if debug {
                for (tap, p) in c.taps.iter().zip(&c.points) {
                    let mut points = vec![listener];
                    points.extend_from_slice(p);
                    points.push(sp);
                    self.exact.push(ExactPath { source: s.id, kind: 1, key: tap.key as u64, energy: tap.gains[1] * tap.gains[1], points });
                }
            }
            edge_taps[si] = c.taps.clone();
        }
        self.edges = Some(es);
        edge_taps
    }
}

/// What a source keeps of its edge paths between searches
/// (`Tracer::edge_paths`).
#[derive(Clone, Default)]
struct EdgePaths {
    /// The edges its single bends go over, and where both ends were
    /// (listener, source) and the tick when they were searched.
    one: Vec<usize>,
    one_at: Option<(Vec3, Vec3, u64)>,
    /// The edge pairs its two-edge strings go over, likewise.
    two: Vec<(usize, usize)>,
    two_at: Option<(Vec3, Vec3, u64)>,
    /// This tick's taps and each one's bend points (between ticks they
    /// are re-timed through them: `exact_taps`).
    taps: Vec<Tap>,
    points: Vec<Vec<Vec3>>,
}

/// Loudness grows as energy to this power (Stevens: twice as loud per
/// 10 dB): how much a source's paths matter to the ear.
const LOUDNESS_EXP: f32 = 0.3;

/// The single-edge path over edge `ei` for where both ends are now: its
/// bend point and tap — only the share the straight ray doesn't carry
/// (`vis`); None where nothing bends there. Both legs meet 3 cm off the
/// edge on the OUTSIDE of the bend (away from whatever the path bends
/// around): a true silhouette edge has open air there and neither leg
/// grazes it; an edge lying on a zero-thickness wall plane leaves one leg
/// crossing that wall.
#[allow(clippy::too_many_arguments)]
fn price_one(world: &World, es: &EdgeSet, hits: &mut Vec<SegHit>, ei: usize, listener: Vec3, sp: Vec3, vis: [f32; NBANDS]) -> Option<(Vec3, Tap)> {
    if es.filled[ei] {
        return None;
    }
    let (a, b) = es.edges[ei];
    let p = crate::paths::bend_point(a, b, sp, listener);
    let (d1, d2) = ((p - listener).length(), (sp - p).length());
    if d1 < 0.2 || d2 < 0.2 {
        return None;
    }
    if world.grade.is_some_and(|g| p.z < g + 0.3 && listener.z > g) {
        return None;
    }
    let n1 = (p - listener) * (1.0 / d1);
    let out = n1 - (sp - p) * (1.0 / d2);
    if out.length() < 1e-3 {
        return None; // no bend: the straight line itself
    }
    let q = p + out.normalize() * 0.03;
    let t1 = world.transmission(listener, q, hits);
    if t1[0] < 1e-6 {
        return None;
    }
    let t2 = world.transmission(q, sp, hits);
    if t1[0] * t2[0] < 1e-6 {
        return None;
    }
    let det = d1 + d2 - (sp - listener).length().max(MIN_DIST);
    let ka = crate::diffraction::knife_edge_bands(det.max(1e-4));
    let air = air_attenuation(d1 + d2);
    let gains: [f32; NBANDS] = core::array::from_fn(|b| ka[b] * t1[b] * t2[b] * air[b] / (d1 + d2) * (1.0 - vis[b]).clamp(0.0, 1.0).sqrt());
    if gains.iter().all(|&g| g < 1e-7) {
        return None;
    }
    Some((p, Tap { key: KEY_EDGE_BASE + ei as u32, delay_s: (d1 + d2) / SPEED_OF_SOUND, dir: [n1.x, n1.y, n1.z], gains }))
}

/// What a walk is walking: shared by its threads, read only.
#[derive(Clone, Copy)]
struct WalkCtx<'a> {
    world: &'a World<'a>,
    listener: Vec3,
    sources: &'a [Source],
    budget: Budget,
    vis: &'a [[f32; NBANDS]],
    /// This tick's rotation of the rays' Fibonacci directions.
    rot: [[f32; 3]; 3],
    cells: &'a [Vec3],
    motion: &'a Motion,
    edges: Option<&'a EdgeSet>,
    /// Each source's free-field arrival (s).
    t_free: &'a [f32],
    debug: bool,
    v4: bool,
}

/// One family of rays (`walk_rays`): the reflection orders it counts,
/// how many segments its rays go, next-event connections a bounce, each
/// ray's weight, and (v5) whether its paths are pooled across ticks.
#[derive(Clone, Copy)]
struct Fam {
    lo: u32,
    hi: u32,
    segments: u32,
    nee: usize,
    inv_n: f32,
    pooled: bool,
}

/// A walking thread's own buffers, kept across ticks.
#[derive(Default)]
struct WalkScratch {
    hits: Vec<SegHit>,
    marks: EdgeMarks,
    picks: Vec<Pick>,
    branches: Vec<Branch>,
    verts: Vec<Vec3>,
    ray_deps: Vec<(usize, [f32; NBANDS], f32, f32)>,
}

/// What a run of rays left, in the order they made it — replayed in ray
/// order by `walk_rays`, so the result is the same however many threads
/// walked them.
#[derive(Default)]
struct ChunkOut {
    /// Deposits: (source, energy, delay, spread each way (0: a point),
    /// the ray's first direction).
    deps: Vec<(u32, [f32; NBANDS], f32, f32, Vec3)>,
    /// Escapes to the sky: (direction cell, energy, its weighted
    /// direction).
    sky: Vec<(u32, [f32; NBANDS], Vec3)>,
    /// v5's pooled family, per ray: whether it is reusable (A), how many
    /// of `deps` are its, its record.
    rays: Vec<(bool, u32, Option<LateRecord>)>,
    /// Debug: the paths found (source id, mid energy, vertices, end).
    dbg: Vec<(u32, f32, Vec<Vec3>, Vec3)>,
}

/// Chunks per walking thread (`walk_chunks`): long late rays balance
/// out over a few each. How the rays are cut doesn't change the result.
const WALK_CHUNKS_PER_THREAD: usize = 4;

/// A ray's own random numbers: seeded from the tick's, its family and
/// its index (splitmix64), the same on any thread.
fn ray_seed(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Walk `dirs` in chunks on as many threads as `scratch` holds (one on
/// wasm): every chunk's output, in ray order.
fn walk_chunks(cx: &WalkCtx, fam: &Fam, scratch: &mut [WalkScratch], dirs: &[Vec3], seed: u64) -> Vec<ChunkOut> {
    let chunk = dirs.len().div_ceil(scratch.len() * WALK_CHUNKS_PER_THREAD).max(1);
    let n_chunks = dirs.len().div_ceil(chunk);
    let walk = |sc: &mut WalkScratch, c: usize| -> ChunkOut {
        let mut out = ChunkOut::default();
        let end = ((c + 1) * chunk).min(dirs.len());
        for (i, &d0) in dirs.iter().enumerate().take(end).skip(c * chunk) {
            let mut rng = Rng::new(ray_seed(seed ^ i as u64));
            walk_ray(cx, fam, sc, &mut rng, d0, &mut out);
        }
        out
    };
    #[cfg(not(target_arch = "wasm32"))]
    if scratch.len() > 1 && n_chunks > 1 {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (walk, next) = (&walk, &next);
        let mut done: Vec<(usize, ChunkOut)> = std::thread::scope(|s| {
            let workers: Vec<_> = scratch
                .iter_mut()
                .take(n_chunks)
                .map(|sc| {
                    s.spawn(move || {
                        let mut mine = Vec::new();
                        loop {
                            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if c >= n_chunks {
                                break;
                            }
                            mine.push((c, walk(sc, c)));
                        }
                        mine
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().expect("a walking thread panicked")).collect()
        });
        done.sort_by_key(|d| d.0);
        return done.into_iter().map(|d| d.1).collect();
    }
    let sc = &mut scratch[0];
    (0..n_chunks).map(|c| walk(sc, c)).collect()
}

/// One ray of a family, walked whole (`walk_rays` step 2): what it
/// deposits, what escapes to the sky and (debug) the paths it found go
/// to `out`, in the order it made them.
fn walk_ray(cx: &WalkCtx, fam: &Fam, sc: &mut WalkScratch, rng: &mut Rng, d0: Vec3, out: &mut ChunkOut) {
    let WalkCtx { world, listener, sources, budget, vis, debug, v4, .. } = *cx;
    let Fam { lo, hi, segments, nee, inv_n, pooled } = *fam;
    let WalkScratch { hits, marks, picks, branches, verts, ray_deps } = sc;
    let counts = |order: u32| order >= lo && order <= hi;
    // the exact solver's paths (both families evaluate it identically):
    // coherent, loud chains within the cap that have more than `bounces`
    // reflections or arrive after the early window — v4: every coherent
    // chain is the solver's, an exact image-source path, whenever it
    // arrives; v3: only the long and the late ones
    let bounces = budget.bounces;
    let solver_owns = |refl: u32, coherent: bool, loud: bool, rel: f32| {
        coherent && loud && refl >= 1 && refl <= MAX_COHERENT_REFL && (v4 || refl > bounces || rel >= EARLY_S)
    };
    let first = rot_apply(&cx.rot, d0);
    branches.clear();
    branches.push(Branch::start(listener, first));
    // v5: this ray's first bounce (diffuse, nothing on the way
    // there: A) and what it deposits
    let mut first_bounce: Option<(Vec3, Vec3, f32, f32)> = None;
    let mut reusable = pooled;
    ray_deps.clear();
    while let Some(br) = branches.pop() {
    let Branch {
        mut origin,
        mut dir,
        mut thr,
        mut length,
        mut refl,
        mut specular,
        mut coherent,
        mut loud,
        mut mirror_thr,
        mut bent,
        mut bends,
        mut bend_from,
        mut last_edge,
        seg0,
        mut split,
    } = br;
    if debug {
        verts.clear();
        verts.push(listener);
        if seg0 > 0 {
            verts.push(origin);
        }
    }
    for seg in seg0..segments {
        if world.in_earth(origin) {
            break; // transmitted into solid ground: gone
        }
        let hit = world.raycast(origin, dir);
        let t_hit = hit.as_ref().map_or(f32::MAX, |h| h.t);
        // diffraction: the edge this segment passes closest to
        let edge = if budget.diffraction && bends < MAX_BENDS {
            let reach = if t_hit.is_finite() { t_hit } else { 400.0 };
            cx.edges.and_then(|es| es.nearest_in(&mut *marks, origin, dir, reach, last_edge))
        } else {
            None
        };
        let t_seg = edge.map_or(t_hit, |(te, _, _, _)| te);

        // 3. specular chains into a source sphere, priced from the
        // source's REAL position — except the exact solver's own
        if (specular || bent) && counts(refl) {
            for (si, s) in sources.iter().enumerate() {
                // where it emitted what this path brings now
                let Some(sp) = cx.motion.emit(si, s.pos, origin, length) else { continue };
                let to = sp - origin;
                let along = to.dot(dir);
                if along <= 0.0 || along >= t_seg {
                    continue;
                }
                let seg = to.length();
                let l_s = (length + seg).max(MIN_DIST);
                let r = budget.k * l_s;
                let closest = (origin + dir * along - sp).length();
                if closest > r {
                    continue;
                }
                if solver_owns(refl, coherent && !bent, loud, l_s / SPEED_OF_SOUND - cx.t_free[si]) {
                    continue; // the exact solver counts this one
                }
                let t = world.transmission_free(origin, sp, &mut *hits);
                let air = air_attenuation(l_s);
                let w = 4.0 * inv_n / (budget.k * budget.k * l_s * l_s);
                // a bent path with no reflection is the DIRECT
                // field around an edge: only what the straight
                // ray doesn't already carry
                // one or two edges, no reflection: the edge
                // next-event paths, already counted exactly
                if refl == 0 && bends <= 2 && !specular {
                    continue;
                }
                // a bent path only carries what the straight
                // line from where it bent doesn't (lit side:
                // nothing — that wave is already counted)
                let gate: [f32; NBANDS] = match bend_from {
                    Some(_) if refl == 0 => core::array::from_fn(|b| 1.0 - vis[si][b]),
                    Some(f) => {
                        let tv = world.transmission(f, sp, &mut *hits);
                        core::array::from_fn(|b| (1.0 - tv[b]).clamp(0.0, 1.0))
                    }
                    None => [1.0; NBANDS],
                };
                let e: [f32; NBANDS] = core::array::from_fn(|b| {
                    thr[b] * t[b] * t[b] * air[b] * air[b] * w * gate[b]
                });
                if debug {
                    out.dbg.push((s.id, e[1], verts.clone(), sp));
                }
                // a sphere hit only knows the path to within the
                // sphere: its arrival is spread over r/c each way
                if pooled {
                    ray_deps.push((si, e, l_s / SPEED_OF_SOUND, r / SPEED_OF_SOUND));
                } else {
                    out.deps.push((si as u32, e, l_s / SPEED_OF_SOUND, r / SPEED_OF_SOUND, first));
                }
            }
        }

        if let Some((te, a, u, ei)) = edge {
            if seg == 0 {
                reusable = false;
            }
            last_edge = Some(ei);
            // bend around the edge at the closest approach
            if bend_from.is_none() {
                bend_from = Some(origin);
            }
            let (eps, w) = sample_bend(rng, a);
            origin = origin + dir * te;
            length += te;
            if debug {
                verts.push(origin);
            }
            dir = rotate_about(dir, u, eps);
            for b in 0..NBANDS {
                thr[b] *= w[b];
            }
            bent = true;
            bends += 1;
            coherent = false;
            continue;
        }

        let Some(hit) = hit else {
            // escaped to open sky: the outdoor field reaches the
            // listener along this ray's first direction. Escapes
            // split by reflection count like every other path
            // (early family ≤ bounces, late family beyond).
            let mine = if lo == 0 { refl <= hi } else { refl >= lo };
            if mine && !world.escapes_into_earth(origin, dir) {
                let mut best = 0usize;
                let mut bd = f32::MIN;
                for (ci, c) in cx.cells.iter().enumerate() {
                    let d = c.dot(first);
                    if d > bd {
                        bd = d;
                        best = ci;
                    }
                }
                out.sky.push((best as u32, core::array::from_fn(|b| thr[b] * inv_n), first * (thr[1] * inv_n)));
            }
            break;
        };
        let p = origin + dir * t_hit;
        if debug {
            verts.push(p);
        }
        length += t_hit;
        if length / SPEED_OF_SOUND > if v4 { RESPONSE_MAX_S } else { MAX_TIME } {
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
            // the sources this surface faces, each with what it
            // would bring here with nothing in the way — where it
            // emitted what this connection brings now
            picks.clear();
            for (si, s) in sources.iter().enumerate() {
                let Some(sp) = cx.motion.emit(si, s.pos, off, length) else { continue };
                let to = sp - off;
                let along = to.dot(normal);
                if along <= 0.0 {
                    continue; // source behind this surface
                }
                let r2 = to.dot(to);
                if r2 < 1e-8 {
                    continue;
                }
                // w · cos / max(r, MIN_DIST)²
                let c = s.weight.max(1e-12) * along / (r2.sqrt() * r2.max(MIN_DIST * MIN_DIST));
                picks.push(Pick { si: si as u32, c, pw: 1.0 });
            }
            // more than `nee`: each at most once, by importance
            if nee > 0 && picks.len() > nee {
                pick_systematic(picks, nee, rng.next_f32());
            }
            for &Pick { si, pw, .. } in picks.iter() {
                let si = si as usize;
                let s = &sources[si];
                let Some(sp) = cx.motion.emit(si, s.pos, off, length) else { continue };
                let to = sp - off;
                let r = to.length();
                let cos_in = to.dot(normal) / r;
                let t = world.transmission_free(off, sp, &mut *hits);
                let rr = r.max(MIN_DIST);
                let air = air_attenuation(length + r);
                let w = pw * mat.scattering * cos_in * 4.0 * inv_n / (rr * rr);
                let e: [f32; NBANDS] = core::array::from_fn(|b| {
                    thr[b] * keep[b] * t[b] * t[b] * air[b] * air[b] * w
                });
                if debug {
                    out.dbg.push((s.id, e[1], verts.clone(), sp));
                }
                if pooled {
                    ray_deps.push((si, e, (length + r) / SPEED_OF_SOUND, 0.0));
                } else {
                    out.deps.push((si as u32, e, (length + r) / SPEED_OF_SOUND, 0.0, first));
                }
            }
        }

        // 4. continue: transmit (importance-floored) or reflect
        let mut p_t = 0.0f32;
        for b in 0..NBANDS {
            p_t = p_t.max(mat.transmission[b] * mat.transmission[b]);
        }
        // v4: a partition passing real energy (a door, glass, a
        // light wall): the ray splits — the transmitted branch goes
        // on with its exact weight, this one reflects with its own
        // — once per ray; elsewhere the unbiased draw. (v3 keeps
        // the draw: its echo selection, a salience test against
        // the diffuse estimate, was calibrated on it. v6 too: from
        // the square its one split mostly went into some other
        // building — the club's late spread 5.1/4.2/3.5 → 6.7/6.1/
        // 7.3 dB, `assess still`.)
        let split_here = v4 && !split && p_t >= SPLIT_T2;
        if split_here && seg == 0 {
            reusable = false;
        }
        if split_here {
            split = true;
            // the wall as a whole: the same price a segment
            // through it pays (both faces, its thickness, once)
            let (t2, beyond) = world.through(p, dir, &mat, &mut *hits);
            let t_thr: [f32; NBANDS] = core::array::from_fn(|b| thr[b] * t2[b]);
            let t_mirror = mirror_thr * t2[1];
            branches.push(Branch {
                origin: beyond,
                dir,
                thr: t_thr,
                length,
                refl,
                specular,
                coherent,
                loud: loud && t_mirror >= COHERENT_FLOOR,
                mirror_thr: t_mirror,
                bent,
                bends,
                bend_from,
                last_edge,
                seg0: seg + 1,
                split: true,
            });
        }
        let pt = if split_here {
            0.0
        } else if p_t > 1e-5 {
            p_t.max(0.02).min(0.5)
        } else {
            0.0
        };
        if pt > 0.0 && rng.next_f32() < pt {
            if seg == 0 {
                reusable = false;
            }
            let (t2, beyond) = world.through(p, dir, &mat, &mut *hits);
            for b in 0..NBANDS {
                thr[b] *= t2[b] / pt;
            }
            mirror_thr *= t2[1];
            loud &= mirror_thr >= COHERENT_FLOOR;
            origin = beyond;
            continue; // straight on; `specular` unchanged
        }
        let before = thr.iter().fold(0.0f32, |a, &b| a.max(b));
        for b in 0..NBANDS {
            thr[b] *= keep[b] / (1.0 - pt);
        }
        if v4 && hi == u32::MAX && length > EARLY_S * SPEED_OF_SOUND {
            // v4's late rays past the early window: Russian roulette ∝
            // amplitude — survive with √(what this bounce kept), the
            // survivor carries 1 / that. Unbiased; the rays alive (and
            // the work) fall with the amplitude, not with time, so a
            // ringing stone room no longer walks every ray to the
            // window's end. Inside the early window every ray goes on:
            // what the taps are estimated from keeps its samples
            let after = thr.iter().fold(0.0f32, |a, &b| a.max(b));
            let p = if before > 0.0 { (after / before).sqrt().min(1.0) } else { 0.0 };
            if rng.next_f32() >= p {
                break;
            }
            for t in &mut thr {
                *t /= p;
            }
        }
        if thr.iter().all(|&e| e < LATE_FLOOR) {
            break;
        }
        refl += 1;
        bend_from = None;
        mirror_thr *= keep[1] * (1.0 - mat.scattering);
        loud &= mirror_thr >= COHERENT_FLOOR;
        if rng.next_f32() < mat.scattering {
            if seg == 0 && reusable {
                first_bounce = Some((p, normal, length, -dir.dot(normal)));
            }
            dir = (normal + rng.unit_sphere()).normalize();
            if dir.dot(normal) < 1e-3 {
                dir = normal;
            }
            specular = false;
            bent = false;
            coherent = false;
        } else {
            if seg == 0 {
                reusable = false;
            }
            dir = dir - normal * (2.0 * dir.dot(normal));
            specular = true;
        }
        origin = p + normal * EPS;
    }
    }
    if pooled {
        let a = reusable && first_bounce.is_some();
        let record = match (a, first_bounce) {
            (true, Some((v1, n1, r1, cos1))) if !ray_deps.is_empty() => {
                let deps = ray_deps.iter().map(|&(si, e, d, h)| (sources[si].id, e, d, h)).collect();
                Some(LateRecord { v1, n1, r1, cos1, deps })
            }
            _ => None,
        };
        for &(si, e, delay, half) in ray_deps.iter() {
            out.deps.push((si as u32, e, delay, half, first));
        }
        out.rays.push((a, ray_deps.len() as u32, record));
    }
}

/// An edge path's tap for where listener and source are now: its delay
/// and arrival direction through the same bend points (its gains as
/// found).
fn retimed(tap: &Tap, bends: &[Vec3], listener: Vec3, source: Vec3) -> Tap {
    let first = bends[0];
    let a = first - listener;
    let la = a.length().max(1e-6);
    let dir = a * (1.0 / la);
    let len = polyline_len(listener, bends) + (source - bends[bends.len() - 1]).length();
    Tap { delay_s: len / SPEED_OF_SOUND, dir: [dir.x, dir.y, dir.z], ..*tap }
}

/// Length from `from` through `points` (to the last of them).
fn polyline_len(from: Vec3, points: &[Vec3]) -> f32 {
    let mut p = from;
    let mut l = 0.0;
    for &q in points {
        l += (q - p).length();
        p = q;
    }
    l
}

/// Two-edge strings searched per source at most this much longer than the
/// straight line (m, and as a share of it): longer detours are other
/// paths' business (reflections), and the search stays bounded.
const TWO_EDGE_DETOUR_M: f32 = 40.0;
const TWO_EDGE_DETOUR_SHARE: f32 = 0.3;
/// Edges looked at per two-edge search (nearest the straight line first),
/// pairs refined, and strings validated through the geometry.
const TWO_EDGE_NEAR: usize = 256;
const TWO_EDGE_REFINE: usize = 1024;
const TWO_EDGE_VALIDATE: usize = 1024;
/// Two-edge paths kept per source.
const TWO_EDGE_PATHS: usize = 2;
const KEY_EDGE2_BASE: u32 = 5_000_000;
const KEY_EDGE2_SPAN: u32 = 3_000_000;

/// Does `p` see edge `(a, b)` (a few points along it, pulled 5 cm toward
/// `p`; not through the earth)?
fn sees_edge(world: &World, p: Vec3, a: Vec3, b: Vec3) -> bool {
    sees_edge_within(world, p, a, b, 2000.0)
}

/// Does the listener at `p` see edge `ei` (not filled; points within
/// 400 m)? Looked at once per `seen` entry (0 not yet, 1 yes, 2 no).
fn ear_sees(world: &World, es: &EdgeSet, seen: &mut [u8], ei: usize, p: Vec3) -> bool {
    if seen[ei] == 0 {
        let (a, b) = es.edges[ei];
        seen[ei] = if !es.filled[ei] && sees_edge_within(world, p, a, b, 400.0) { 1 } else { 2 };
    }
    seen[ei] == 1
}

/// `sees_edge`, only points nearer than `reach` (m).
fn sees_edge_within(world: &World, p: Vec3, a: Vec3, b: Vec3, reach: f32) -> bool {
    [0.1f32, 0.3, 0.5, 0.7, 0.9].iter().any(|f| {
        let m = a + (b - a) * *f;
        let to = m - p;
        let l = to.length();
        l > 0.2 && l < reach && {
            let end = m - to * (0.05 / l);
            world.mesh.raycast(p, to * (1.0 / l)).map_or(true, |(t, _)| t > l - 0.06) && !world.crosses_earth(p, end)
        }
    })
}

/// The exact paths over two edges between `listener` and `src` (whose
/// straight line is blocked): the shortest string through a pair — an
/// edge the ear sees, then one the source sees — each bend point where
/// the string is shortest (alternating until it settles), every leg
/// priced through whatever it crosses. Priced as a whole by its detour
/// (ISO 9613-2's multi-edge barrier: thick buildings cost a little more
/// than thin walls, not 5 dB per grazing corner), only the share the
/// straight line doesn't carry. `src_seen` caches which edges the source
/// sees (0 unknown, 1 yes, 2 no). Strongest first, at most
/// `TWO_EDGE_PATHS`, as (edge pair, bend points).
#[allow(clippy::too_many_arguments)]
fn two_edge_paths(
    world: &World,
    es: &EdgeSet,
    ear_seen: &mut [u8],
    src_seen: &mut [u8],
    hits: &mut Vec<SegHit>,
    listener: Vec3,
    src: Vec3,
    vis: [f32; NBANDS],
) -> Vec<(usize, usize, [Vec3; 2])> {
    let direct = (src - listener).length().max(MIN_DIST);
    let limit = TWO_EDGE_DETOUR_M.min(TWO_EDGE_DETOUR_SHARE * direct).max(4.0);
    // the edges near the straight line (an ellipse around it), each with
    // its own bend point for the straight pair
    let mut near: Vec<(f32, usize, Vec3)> = es
        .edges
        .iter()
        .enumerate()
        .filter(|(ei, _)| !es.filled[*ei])
        .filter_map(|(ei, (a, b))| {
            let c = crate::paths::bend_point(*a, *b, src, listener);
            let l = (c - listener).length() + (src - c).length();
            (l - direct <= limit && world.grade.is_none_or(|g| c.z > g + 0.3)).then_some((l, ei, c))
        })
        .collect();
    near.sort_by(|x, y| x.0.total_cmp(&y.0));
    // a string starts at an edge the ear sees and ends at one the source
    // sees: the nearest of each (an edge near the line that neither sees —
    // the pews inside a closed church — starts nothing)
    // (each side looks at no more than `EDGE_LOOKS` edges: from inside
    // a closed hall almost none of those near the line outside are seen)
    let first: Vec<&(f32, usize, Vec3)> =
        near.iter().take(EDGE_LOOKS).filter(|e| ear_sees(world, es, ear_seen, e.1, listener)).take(TWO_EDGE_NEAR).collect();
    let mut last: Vec<&(f32, usize, Vec3)> = Vec::new();
    for e in near.iter().take(EDGE_LOOKS) {
        if last.len() >= TWO_EDGE_NEAR {
            break;
        }
        if src_seen[e.1] == 0 {
            let (a, b) = es.edges[e.1];
            src_seen[e.1] = if sees_edge(world, src, a, b) { 1 } else { 2 };
        }
        if src_seen[e.1] == 1 {
            last.push(e);
        }
    }
    // pairs by their straight-pair estimate, the shortest refined
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for e1 in &first {
        for e2 in &last {
            if e1.1 == e2.1 {
                continue;
            }
            let l = (e1.2 - listener).length() + (e2.2 - e1.2).length() + (src - e2.2).length();
            if l - direct <= limit {
                pairs.push((l, e1.1, e2.1));
            }
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    pairs.truncate(TWO_EDGE_REFINE);
    let mut strings: Vec<(f32, usize, usize, Vec3, Vec3)> = pairs
        .into_iter()
        .map(|(_, e1, e2)| {
            let (l, p1, p2) = refine_two(es, e1, e2, listener, src);
            (l, e1, e2, p1, p2)
        })
        .collect();
    strings.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: Vec<(f32, usize, usize, [Vec3; 2])> = Vec::new();
    let mut clear = 0;
    for (l, e1, e2, p1, p2) in strings.into_iter().take(TWO_EDGE_VALIDATE) {
        // shortest first: two strings with open air all along end the search
        if clear >= TWO_EDGE_PATHS {
            break;
        }
        if l - direct > limit {
            continue;
        }
        if let Some((tap, open)) = price_two(world, es, hits, e1, e2, l, p1, p2, listener, src, vis) {
            clear += open as usize;
            out.push((tap.gains[0] + tap.gains[1], e1, e2, [p1, p2]));
        }
    }
    out.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut kept: Vec<(usize, usize, [Vec3; 2])> = Vec::new();
    for (_, e1, e2, pts) in out {
        if kept.len() >= TWO_EDGE_PATHS {
            break;
        }
        if kept.iter().any(|(_, _, q)| (q[0] - pts[0]).length() < 1.0 && (q[1] - pts[1]).length() < 1.0) {
            continue;
        }
        kept.push((e1, e2, pts));
    }
    kept
}

/// The shortest string over edges `e1` (the ear's side) then `e2`: each
/// bend point where the string is shortest, alternating until it settles.
/// (length, bend points)
fn refine_two(es: &EdgeSet, e1: usize, e2: usize, listener: Vec3, src: Vec3) -> (f32, Vec3, Vec3) {
    let ((a1, b1), (a2, b2)) = (es.edges[e1], es.edges[e2]);
    let mut p2 = (a2 + b2) * 0.5;
    let mut p1 = crate::paths::bend_point(a1, b1, p2, listener);
    for _ in 0..4 {
        p2 = crate::paths::bend_point(a2, b2, src, p1);
        p1 = crate::paths::bend_point(a1, b1, p2, listener);
    }
    ((p1 - listener).length() + (p2 - p1).length() + (src - p2).length(), p1, p2)
}

/// The two-edge string of length `l` through bend points `p1`, `p2`
/// (`refine_two`), validated and priced: every leg through whatever it
/// crosses, the whole by its detour (ISO 9613-2's multi-edge barrier),
/// only the share the straight line doesn't carry (`vis`). With whether
/// it has open air all along. None where no string goes over this pair.
#[allow(clippy::too_many_arguments)]
fn price_two(
    world: &World,
    es: &EdgeSet,
    hits: &mut Vec<SegHit>,
    e1: usize,
    e2: usize,
    l: f32,
    p1: Vec3,
    p2: Vec3,
    listener: Vec3,
    src: Vec3,
    vis: [f32; NBANDS],
) -> Option<(Tap, bool)> {
    if es.filled[e1] || es.filled[e2] {
        return None;
    }
    let direct = (src - listener).length().max(MIN_DIST);
    let z = l - direct;
    if z <= 0.0 || (p2 - p1).length() < 0.5 {
        return None;
    }
    let na = (p1 - listener).normalize();
    // legs meet 3 cm off each edge on the outside of the string — away
    // from the straight line it goes around (a grazing edge, the string
    // almost straight there, still has an outside)
    let line = src - listener;
    let off = |p: Vec3| -> Option<Vec3> {
        let k = ((p - listener).dot(line) / line.dot(line)).clamp(0.0, 1.0);
        let o = p - (listener + line * k);
        (o.length() > 1e-4).then(|| p + o.normalize() * 0.03)
    };
    let (q1, q2) = (off(p1)?, off(p2)?);
    // a leg through the fixed geometry (a string through a church, not
    // over it): one ray each, before pricing anything
    let open = |a: Vec3, b: Vec3| {
        let d = b - a;
        let l = d.length();
        l < 1e-3 || world.mesh.raycast(a, d * (1.0 / l)).is_none_or(|(t, _)| t >= l - 1e-3)
    };
    if !open(listener, q1) || !open(q1, q2) || !open(q2, src) {
        return None;
    }
    let t1 = world.transmission(listener, q1, hits);
    let t12 = world.transmission(q1, q2, hits);
    let t2 = world.transmission(q2, src, hits);
    if t1[0] * t12[0] * t2[0] < 1e-6 {
        return None;
    }
    let lateral = p1.z.max(p2.z) < listener.z.max(src.z) + 0.5;
    let iso = crate::diffraction::iso9613_barrier_bands(z, 2, (p2 - p1).length(), (src - p2).length(), (p1 - listener).length(), direct, lateral);
    let air = air_attenuation(l);
    let gains: [f32; NBANDS] = core::array::from_fn(|b| iso[b] * t1[b] * t12[b] * t2[b] * air[b] / l * (1.0 - vis[b]).clamp(0.0, 1.0).sqrt());
    if gains.iter().all(|&g| g < 1e-7) {
        return None;
    }
    let key = KEY_EDGE2_BASE + ((e1 as u64 * 7919 + e2 as u64) % KEY_EDGE2_SPAN as u64) as u32;
    Some((Tap { key, delay_s: l / SPEED_OF_SOUND, dir: [na.x, na.y, na.z], gains }, t1[1] * t12[1] * t2[1] > 0.5))
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
    if acc.v4 {
        // v4: everything the walk estimates (scattered, bent, quiet
        // mirror paths) is the dense response; the echogram only feeds
        // the RT60 readout
        let i = (rel / DENSE_BIN_S) as usize;
        if rel < RESPONSE_MAX_S {
            if acc.dense.len() <= i {
                acc.dense.resize(i + 1, [0.0; 7]);
            }
            let bin = &mut acc.dense[i];
            for b in 0..NBANDS {
                bin[b] += e[b];
            }
            bin[3] += dir.x * e[1];
            bin[4] += dir.y * e[1];
            bin[5] += dir.z * e[1];
            bin[6] += acc.sq_scale * e[1] * e[1];
        }
        let j = (rel / BIN_DT) as usize;
        if j < acc.echo.bins.len() {
            for b in 0..NBANDS {
                acc.echo.bins[j][b] += e[b] * LEGACY_UNITS;
            }
        }
        return;
    }
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
        let dbin = (rel / DELAY_BIN_S) as u32;
        let key = KEY_EARLY_BASE + cell_index(cells, dir) as u32 * 256 + dbin.min(255);
        acc.early.entry(key).or_default().add(e, delay, dir);
    }
}

/// The dense response at the resolution its samples support: each bin
/// averaged over the narrowest window around it holding `ESS_TARGET`
/// effective samples — (Σw)²/Σw², the count of equal-weight samples
/// with the same spread, times the ticks the average spans — so a rare
/// heavy sample (a ray bouncing right by the source) is diluted where
/// samples are sparse (late, far), while the well-sampled early part
/// keeps the full 2 ms. Windows reach back at most a quarter of a bin's
/// delay (the onset is never smeared).
fn smooth_dense(d: &[[f32; 7]], ticks: &[f32; HISTORY_WINDOWS]) -> Vec<[f32; 6]> {
    let n = d.len();
    let mut out = vec![[0.0f32; 6]; n];
    if n == 0 {
        return out;
    }
    // prefix sums of every channel
    let mut pre = vec![[0.0f64; 7]; n + 1];
    for i in 0..n {
        for c in 0..7 {
            pre[i + 1][c] = pre[i][c] + d[i][c] as f64;
        }
    }
    let sum = |a: usize, b: usize, c: usize| pre[b][c] - pre[a][c];
    for i in 0..n {
        let reach = (i / 4).max(1);
        let mut h = 0usize;
        loop {
            let (a, b) = (i.saturating_sub(h), (i + h + 1).min(n));
            let w = sum(a, b, 1);
            let q = sum(a, b, 6);
            let ess = if q > 0.0 { w * w / q * ticks[history_window(i)].max(1.0) as f64 } else { f64::INFINITY };
            if ess >= ESS_TARGET || h >= reach {
                let len = (b - a) as f32;
                for c in 0..6 {
                    out[i][c] = (sum(a, b, c) as f32) / len;
                }
                break;
            }
            h = (h * 2).max(1);
        }
    }
    out
}

/// A hash of the panels (boxes and their materials) for change detection.
fn panels_sig(panels: &[(Vec3, Vec3, Material)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: f32| {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    for (a, b, m) in panels {
        // to the millimetre: a leaf easing into its frame by less (a host
        // animating it asymptotically) changes nothing a sound can tell,
        // and would start every average over each tick
        for v in [a.x, a.y, a.z, b.x, b.y, b.z] {
            eat((v * 1000.0).round());
        }
        eat(m.transmission[1]);
        eat(m.absorption[1]);
    }
    h
}

/// `deposit`, the energy spread evenly over `delay ± half` (v4: an
/// estimate's own time uncertainty; v3: at `delay`).
fn deposit_spread(acc: &mut Acc, cells: &[Vec3], e: [f32; NBANDS], delay: f32, dir: Vec3, half: f32) {
    if !acc.v4 || half < 0.5 * DENSE_BIN_S {
        deposit(acc, cells, e, delay, dir);
        return;
    }
    let n = (2.0 * half / DENSE_BIN_S).ceil() as usize + 1;
    let part: [f32; NBANDS] = core::array::from_fn(|b| e[b] / n as f32);
    acc.sq_scale = n as f32;
    for i in 0..n {
        let t = delay - half + 2.0 * half * i as f32 / (n - 1) as f32;
        deposit(acc, cells, part, t.max(acc.t_free), dir);
    }
    acc.sq_scale = 1.0;
}

/// Exact image-source solve of a mirror chain between `lis` and `src`:
/// mirror the source across the planes (last first), unfold from the
/// listener, and check every leg reaches its plane first. Returns the
/// path's energy (per band: mirror losses × final-leg transmission ×
/// air / L²), its delay and its direction at the listener — or None
/// when the chain doesn't connect (or is quiet past the coherent floor).
fn solve_chain(
    world: &World,
    planes: &[(Vec3, f32)],
    lis: Vec3,
    src: Vec3,
    hits: &mut Vec<SegHit>,
) -> Option<([f32; NBANDS], f32, Vec3)> {
    let n = planes.len();
    if n == 0 || n as u32 > MAX_COHERENT_REFL {
        return None;
    }
    // images[k]: the source mirrored across planes k.. (images[n] = src)
    let mut images = vec![src; n + 1];
    for k in (0..n).rev() {
        images[k] = mirror_point(images[k + 1], planes[k]);
    }
    let mut p = lis;
    let mut first = None;
    let mut keep = [1.0f32; NBANDS];
    for k in 0..n {
        let (nrm, d) = planes[k];
        let to = images[k] - p;
        let denom = nrm.dot(to);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = (d - nrm.dot(p)) / denom;
        if !(t > 1e-5 && t < 1.0) {
            return None;
        }
        let q = p + to * t;
        let len = (q - p).length();
        if len < 1e-3 {
            return None;
        }
        let dir = (q - p) * (1.0 / len);
        let skip = if k == 0 { 0.0 } else { EPS };
        // the fixed geometry decides the path; movable boxes on a leg (a
        // door leaf, a pane) pass it with their transmission, as the walk
        // does — a closed door muffles a mirror path, it doesn't erase it
        let hit = world.mesh.raycast_hit(p + dir * skip, dir)?;
        if (hit.t + skip - len).abs() > 0.02 || hit.normal.dot(nrm).abs() < 0.999 {
            return None; // something else first, or not this plane
        }
        first.get_or_insert(dir);
        for (mn, mx, pm) in world.panels {
            if segment_crosses_box(p, q, *mn, *mx, 1e-4) {
                for b in 0..NBANDS {
                    keep[b] *= pm.transmission[b] * pm.transmission[b];
                }
            }
        }
        let m = hit.material;
        for b in 0..NBANDS {
            keep[b] *= (1.0 - m.absorption[b] - m.transmission[b] * m.transmission[b]).max(0.0) * (1.0 - m.scattering);
        }
        // keep[1] is the chain's mirror throughput: past the floor it is
        // no longer coherent-loud (the late family's, not ours)
        if keep[1] < COHERENT_FLOOR {
            return None;
        }
        p = q;
    }
    let last = src - p;
    let last_len = last.length().max(1e-4);
    let t = world.transmission(p + last * (EPS / last_len), src, hits);
    let l = (images[0] - lis).length();
    let lc = l.max(MIN_DIST);
    let air = air_attenuation(l);
    let e = core::array::from_fn(|b| keep[b] * t[b] * t[b] * air[b] * air[b] / (lc * lc));
    Some((e, l / SPEED_OF_SOUND, first?))
}

/// The direction cell nearest `dir`.
fn cell_index(cells: &[Vec3], dir: Vec3) -> usize {
    let mut best = 0usize;
    let mut bd = f32::MIN;
    for (i, c) in cells.iter().enumerate() {
        let d = c.dot(dir);
        if d > bd {
            bd = d;
            best = i;
        }
    }
    best
}

// ------------------------------------------------------- diffraction

/// Rays passing farther than this from an edge don't feel it.
pub const EDGE_REACH: f32 = 3.0;
/// Spread of the bend angle: σ = λ / (DIFF_C · a), a = closest distance
/// to the edge (uncertainty relation: the narrower the gap a ray squeezes
/// past, the wider its direction spreads — and longer waves spread more).
/// Calibrated against the Kurze–Anderson knife edge (fx_v3_tracer).
pub const DIFF_C: f32 = 4.0 * core::f32::consts::PI;
/// Bends per ray (multi-edge paths: over a roof, around a corner).
const MAX_BENDS: u32 = 3;
/// How far along one ray segment edges are searched (m).
const EDGE_SEARCH: f32 = 120.0;
/// A source's edge search, and which edges the listener and the source
/// see, hold until either moves this far (meanwhile the edges found are
/// solved for where both are: `Tracer::edge_paths`).
const EDGE_REUSE_M: f32 = 0.2;
/// Deterministic single-edge direct paths kept per source.
const EDGE_PATHS: usize = 3;
/// Edges whose legs are checked per blocked source (best knife-edge
/// bounds first — most near the blocked line are themselves blocked).
const EDGE_CANDIDATES: usize = 40;
/// Edges a search looks at, at most, for whether the listener (or the
/// source) sees them — the best bounds (or the shortest detours) first.
/// From inside a closed hall almost none of the edges near the line to a
/// source outside are seen: looking down the whole list cost 5–11 ms a
/// search in the Godot demo's cathedral, for the edges least likely to
/// carry anything. Enough for the town's bell bent around the cathedral
/// (its corners rank behind a few hundred hidden edges of the nave; 256
/// missed them, 512 finds them).
const EDGE_LOOKS: usize = 1024;
/// Tap keys of edge paths: base + edge index (early bins sit below).
const KEY_EDGE_BASE: u32 = 20_000;
const EDGE_CELL: f32 = 4.0;

/// Which edges one segment query has looked at already (`EdgeSet::
/// nearest_in`): the query's stamp per edge.
#[derive(Clone, Default)]
struct EdgeMarks {
    seen: Vec<u32>,
    stamp: u32,
}

impl EdgeMarks {
    /// One mark per edge of a set of `n`.
    fn fit(&mut self, n: usize) {
        if self.seen.len() != n {
            self.seen = vec![0; n];
            self.stamp = 0;
        }
    }
}

/// Diffracting edges of a mesh (sharp creases ≥ 30° and free borders),
/// in a uniform grid for segment queries.
pub struct EdgeSet {
    pub edges: Vec<(Vec3, Vec3)>,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    key: (usize, usize),
    /// `nearest`'s own marks (a walk's threads each bring theirs:
    /// `nearest_in`).
    marks: EdgeMarks,
    /// Bounds of all edges, padded by EDGE_REACH: rays outside it skip
    /// the search entirely.
    lo: Vec3,
    hi: Vec3,
    /// Edges whose opening is FILLED this tick (a pane, a closed or ajar
    /// door leaf against the frame): not exposed, no diffraction.
    filled: Vec<bool>,
}

/// Words before the edges in `EdgeSet::flat`.
pub const EDGE_FLAT_HEADER: usize = 16;

/// A grid cell's hash in `EdgeSet::flat`'s table (v3_walk.wgsl
/// `cell_slot`): wrapping u32 arithmetic, the same on both sides.
pub fn edge_cell_hash(k: (i32, i32, i32)) -> u32 {
    (k.0 as u32).wrapping_mul(73_856_093) ^ (k.1 as u32).wrapping_mul(19_349_663) ^ (k.2 as u32).wrapping_mul(83_492_791)
}

fn cell_key(p: Vec3) -> (i32, i32, i32) {
    (
        (p.x / EDGE_CELL).floor() as i32,
        (p.y / EDGE_CELL).floor() as i32,
        (p.z / EDGE_CELL).floor() as i32,
    )
}

impl EdgeSet {
    pub fn build(mesh: &Mesh) -> Self {
        let mut by_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::default();
        for (ti, idx) in mesh.indices.iter().enumerate() {
            for k in 0..3 {
                let (i, j) = (idx[k], idx[(k + 1) % 3]);
                by_edge.entry((i.min(j), i.max(j))).or_default().push(ti as u32);
            }
        }
        // in vertex order: two builds of one mesh agree edge for edge (the
        // GPU's copy, `flat`, is indexed like this one)
        let mut by_edge: Vec<_> = by_edge.into_iter().collect();
        by_edge.sort_unstable_by_key(|(k, _)| *k);
        let mut edges = Vec::new();
        for ((i, j), tris) in by_edge {
            let (a, b) = (mesh.positions[i as usize], mesh.positions[j as usize]);
            let candidate = match tris.as_slice() {
                // a free border — unless the surface simply continues in
                // the same plane past it (a seam between the strips a
                // wall with a window hole is built from)
                [t] => !seam(mesh, *t, a, b),
                [t0, t1] => mesh.tri_normal(*t0).dot(mesh.tri_normal(*t1)).abs() < 0.866,
                _ => false,
            };
            if candidate && (b - a).length() > 0.05 && exposed(mesh, a, b) {
                edges.push((a, b));
            }
        }
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::default();
        for (ei, (a, b)) in edges.iter().enumerate() {
            let lo = Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)) - Vec3::new(EDGE_REACH, EDGE_REACH, EDGE_REACH);
            let hi = Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)) + Vec3::new(EDGE_REACH, EDGE_REACH, EDGE_REACH);
            let (k0, k1) = (cell_key(lo), cell_key(hi));
            // very long edges (a ground plane's border) would flood the grid
            let cells = ((k1.0 - k0.0 + 1) as i64) * ((k1.1 - k0.1 + 1) as i64) * ((k1.2 - k0.2 + 1) as i64);
            if cells > 4096 {
                continue;
            }
            for x in k0.0..=k1.0 {
                for y in k0.1..=k1.1 {
                    for z in k0.2..=k1.2 {
                        grid.entry((x, y, z)).or_default().push(ei as u32);
                    }
                }
            }
        }
        let n = edges.len();
        let r = Vec3::new(EDGE_REACH, EDGE_REACH, EDGE_REACH);
        let (mut lo, mut hi) = (Vec3::new(f32::MAX, f32::MAX, f32::MAX), Vec3::new(f32::MIN, f32::MIN, f32::MIN));
        for (a, b) in &edges {
            for p in [*a, *b] {
                lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
                hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
            }
        }
        Self {
            edges,
            grid,
            key: (mesh.indices.len(), mesh.positions.len()),
            marks: EdgeMarks { seen: vec![0; n], stamp: 0 },
            lo: lo - r,
            hi: hi + r,
            filled: vec![false; n],
        }
    }

    /// The set flattened for the GPU walk (v3_walk.wgsl `nearest_edge`,
    /// binding 8), u32 words: a header (edges, slots, slots' offset, lists'
    /// offset, then 0 to `EDGE_FLAT_HEADER`), the edges (a xyz, 0, b xyz, 0
    /// as f32 bits), the grid as an open-addressed table of `slots` cells
    /// (kx, ky, kz, start, count; count 0 = empty; probed linearly from
    /// `edge_cell_hash`), and the cells' edge lists. Built from one mesh
    /// twice, the order is the same (`build` sorts).
    pub fn flat(&self) -> Vec<u32> {
        let f = |x: f32| x.to_bits();
        let slots = (self.grid.len() * 2).next_power_of_two().max(2);
        let mut w = vec![0u32; EDGE_FLAT_HEADER];
        for (a, b) in &self.edges {
            w.extend_from_slice(&[f(a.x), f(a.y), f(a.z), 0, f(b.x), f(b.y), f(b.z), 0]);
        }
        let off_slots = w.len();
        w.resize(off_slots + slots * 5, 0);
        let off_lists = w.len();
        let mut keys: Vec<_> = self.grid.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let list = &self.grid[&k];
            let mut h = edge_cell_hash(k) as usize & (slots - 1);
            while w[off_slots + h * 5 + 4] != 0 {
                h = (h + 1) & (slots - 1);
            }
            let (o, start) = (off_slots + h * 5, (w.len() - off_lists) as u32);
            w[o..o + 5].copy_from_slice(&[k.0 as u32, k.1 as u32, k.2 as u32, start, list.len() as u32]);
            w.extend_from_slice(list);
        }
        w[..4].copy_from_slice(&[self.edges.len() as u32, slots as u32, off_slots as u32, off_lists as u32]);
        w
    }

    /// Mark edges that bound an opening an overlay box currently fills,
    /// and edges lying on the ground (no shadow to bend into).
    fn mask_filled(&mut self, panels: &[(Vec3, Vec3, Material)], grade: Option<f32>) {
        for (ei, (a, b)) in self.edges.iter().enumerate() {
            // edges that never rise above the ground (wall bottoms, the
            // rim of a buried room): no shadow to bend into
            let on_ground = grade.is_some_and(|g| a.z.max(b.z) < g + 0.3);
            self.filled[ei] = on_ground || panels.iter().any(|(mn, mx, _)| {
                [0.25f32, 0.5, 0.75].iter().any(|f| {
                    let p = *a + (*b - *a) * *f;
                    let dx = (mn.x - p.x).max(p.x - mx.x).max(0.0);
                    let dy = (mn.y - p.y).max(p.y - mx.y).max(0.0);
                    let dz = (mn.z - p.z).max(p.z - mx.z).max(0.0);
                    dx * dx + dy * dy + dz * dz < 0.08 * 0.08
                })
            });
        }
    }

    /// The edge the segment o + d·t, t ∈ (0, t_end) passes closest to
    /// (within EDGE_REACH, alongside the edge's span): (t, distance, unit
    /// edge direction).
    fn nearest(&mut self, o: Vec3, d: Vec3, t_end: f32, skip: Option<usize>) -> Option<(f32, f32, Vec3, usize)> {
        let mut marks = core::mem::take(&mut self.marks);
        let found = self.nearest_in(&mut marks, o, d, t_end, skip);
        self.marks = marks;
        found
    }

    /// `nearest`, marking the edges looked at in `marks` (one per thread).
    fn nearest_in(&self, marks: &mut EdgeMarks, o: Vec3, d: Vec3, t_end: f32, skip: Option<usize>) -> Option<(f32, f32, Vec3, usize)> {
        marks.fit(self.edges.len());
        marks.stamp = marks.stamp.wrapping_add(1);
        if marks.stamp == 0 {
            marks.seen.iter_mut().for_each(|s| *s = 0);
            marks.stamp = 1;
        }
        // only where edges are: clip to their padded bounds, and look no
        // further than EDGE_SEARCH along one segment
        let Some((c0, c1)) = clip_box(o, o + d * t_end.min(EDGE_SEARCH), self.lo, self.hi) else {
            return None;
        };
        let seg = t_end.min(EDGE_SEARCH);
        let (t_start, t_end) = (c0 * seg, c1 * seg);
        let mut best: Option<(f32, f32, Vec3, usize)> = None;
        let mut last = None;
        let step = EDGE_CELL * 0.5;
        let mut t = t_start;
        loop {
            let k = cell_key(o + d * t.min(t_end));
            if last != Some(k) {
                last = Some(k);
                if let Some(list) = self.grid.get(&k) {
                    for &ei in list {
                        let ei = ei as usize;
                        if marks.seen[ei] == marks.stamp || self.filled[ei] || skip == Some(ei) {
                            continue;
                        }
                        marks.seen[ei] = marks.stamp;
                        let (a, b) = self.edges[ei];
                        let e = b - a;
                        let w0 = o - a;
                        let (bb, cc, dd, ee) = (d.dot(e), e.dot(e), d.dot(w0), e.dot(w0));
                        let den = cc - bb * bb;
                        if den < 1e-9 {
                            continue; // parallel to the edge
                        }
                        let tt = (bb * ee - cc * dd) / den;
                        let ss = (ee - bb * dd) / den;
                        if !(0.02..=0.98).contains(&ss) || tt < 0.05 || tt > t_end - 0.05 {
                            continue;
                        }
                        let dist = (o + d * tt - (a + e * ss)).length();
                        if dist < EDGE_REACH && best.map_or(true, |(_, bd, _, _)| dist < bd) {
                            best = Some((tt, dist, e * (1.0 / cc.sqrt()), ei));
                        }
                    }
                }
            }
            if t >= t_end {
                break;
            }
            t = (t + step).min(t_end);
        }
        best
    }
}

/// Is the free border a→b of triangle `t` just a seam — does a coplanar
/// surface continue right past it?
fn seam(mesh: &Mesh, t: u32, a: Vec3, b: Vec3) -> bool {
    let n = mesh.tri_normal(t);
    let idx = mesh.indices[t as usize];
    let c = (mesh.positions[idx[0] as usize] + mesh.positions[idx[1] as usize] + mesh.positions[idx[2] as usize])
        * (1.0 / 3.0);
    let u = (b - a).normalize();
    let mut out = u.cross(n).normalize();
    if (c - (a + b) * 0.5).dot(out) > 0.0 {
        out = out * -1.0; // point AWAY from the triangle, within its plane
    }
    let mut continued = 0;
    for f in [0.25f32, 0.5, 0.75] {
        let q = a + (b - a) * f + out * 0.03;
        let o = q + n * 0.05;
        if mesh.raycast(o, n * -1.0).is_some_and(|(t, _)| t < 0.1) {
            continued += 1;
        }
    }
    continued >= 2
}

/// Does open space wrap around this edge by more than ~200°? Only such
/// edges diffract into a shadow: a building's outside corner, a roof
/// line, a door jamb, a screen's top. A wall standing on the ground, or
/// a room's inside corner, has a quarter of space around it and casts no
/// shadow to bend into — but in a mesh without shared vertices those
/// show up as free face borders. Probed geometrically: 16 directions in
/// the plane across the edge, at three points along it.
pub fn exposed(mesh: &Mesh, a: Vec3, b: Vec3) -> bool {
    const N: usize = 32;
    const R: f32 = 0.5;
    let u = (b - a).normalize();
    let helper = if u.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let p1 = u.cross(helper).normalize();
    let p2 = u.cross(p1);
    let mut best = 0usize;
    for f in [0.25f32, 0.5, 0.75] {
        let m = a + (b - a) * f;
        let free: Vec<bool> = (0..N)
            .map(|k| {
                let th = k as f32 * core::f32::consts::TAU / N as f32;
                let d = p1 * th.cos() + p2 * th.sin();
                let o = m + d * 0.02;
                !mesh.raycast(o, d).is_some_and(|(t, _)| t < R)
            })
            .collect();
        // longest circular run of free directions
        let mut run = 0usize;
        let mut longest = 0usize;
        for k in 0..2 * N {
            if free[k % N] {
                run += 1;
                longest = longest.max(run.min(N));
            } else {
                run = 0;
            }
        }
        best = best.max(longest);
    }
    // > 205°: a wall standing on the ground measures 180–200° (rays along
    // its plane slip by), a gable's rake (wall under a 50° roof) ~220°, an
    // outside corner or roof ridge ~270°+ — the strings over a building
    // cross its gables, so they are edges too
    best * 360 > 205 * N
}

/// Bend angle for a ray passing an edge at distance `a`: sampled from
/// the three bands' (truncated Cauchy) spreads in equal parts; returns
/// the angle and each band's unbiased weight pdf_b / mixture.
fn sample_bend(rng: &mut Rng, a: f32) -> (f32, [f32; NBANDS]) {
    let half_pi = core::f32::consts::FRAC_PI_2;
    let sig: [f32; NBANDS] = core::array::from_fn(|b| {
        let lambda = SPEED_OF_SOUND / crate::diffraction::BAND_CENTER_HZ[b];
        (lambda / (DIFF_C * a.max(0.05))).max(1e-4)
    });
    let pdf = |b: usize, e: f32| -> f32 {
        let s = sig[b];
        let z = (2.0 / core::f32::consts::PI) * (half_pi / s).atan();
        (1.0 / (core::f32::consts::PI * z)) * s / (s * s + e * e)
    };
    let j = ((rng.next_f32() * NBANDS as f32) as usize).min(NBANDS - 1);
    let u = rng.next_f32() * 2.0 - 1.0;
    let eps = sig[j] * (u * (half_pi / sig[j]).atan()).tan();
    let mix = (0..NBANDS).map(|b| pdf(b, eps)).sum::<f32>() / NBANDS as f32;
    (eps, core::array::from_fn(|b| pdf(b, eps) / mix.max(1e-12)))
}

/// Rotate `d` about unit axis `u` by `eps` (Rodrigues): the bent ray
/// keeps its angle to the edge (Keller's cone).
fn rotate_about(d: Vec3, u: Vec3, eps: f32) -> Vec3 {
    let (s, c) = eps.sin_cos();
    (d * c + u.cross(d) * s + u * (u.dot(d) * (1.0 - c))).normalize()
}

/// A next-event candidate: a source a surface point faces, what it would
/// bring there with nothing in the way (`c`, up to the factors all
/// share), and the weight its connection carries (`pw`, 1/π when
/// sampled).
#[derive(Clone, Copy)]
struct Pick {
    si: u32,
    c: f32,
    pw: f32,
}

/// Keep `m` of `picks` on average, each at most once with probability
/// π = min(1, c/τ) — τ such that Σπ = m: the sources that would bring
/// the most here are connected for sure, the rest share what is left in
/// proportion to what they would bring — and weight each kept one 1/π
/// (unbiased). Systematic: one uniform `u`, thresholds u, u+1, … over
/// the running sum of π (⌊Σπ⌋ or ⌈Σπ⌉ kept).
///
/// Draws with replacement ∝ c (what this was) weight a rare pick by the
/// sum of every source's c: on a surface by a near source, a far one
/// seen through an open door came in hundreds of times over — one such
/// sample moved a club's level by 5 dB for seconds. Here no pick weighs
/// more than τ, the share of the sources not connected for sure.
fn pick_systematic(picks: &mut Vec<Pick>, m: usize, u: f32) {
    let total: f32 = picks.iter().map(|p| p.c).sum();
    let mut tau = total / m as f32;
    // the sure ones (c ≥ τ) take one connection each, the rest share the
    // others; τ only falls, settling within m rounds
    for _ in 0..m {
        let (mut sure, mut k) = (0.0f32, 0usize);
        for p in picks.iter() {
            if p.c >= tau {
                sure += p.c;
                k += 1;
            }
        }
        if k >= m {
            break;
        }
        let next = (total - sure) / (m - k) as f32;
        if !(next < tau) {
            break;
        }
        tau = next;
    }
    let (mut run, mut at) = (0.0f32, u);
    picks.retain_mut(|p| {
        let pi = (p.c / tau).min(1.0);
        run += pi;
        if at < run {
            at += 1.0;
            p.pw = 1.0 / pi;
            true
        } else {
            false
        }
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A bounce in a corner — on the floor a fraction of a millimetre
    /// from a wall — connects to a source behind that wall through the
    /// wall, not past it. The default margin (a share of the segment)
    /// let it through: the club heard through its shut door was mostly
    /// such leaks among its loudest samples.
    #[test]
    fn a_bounce_in_a_corner_does_not_see_through_the_wall_beside_it() {
        use crate::mesh::MeshBuilder;
        let mut mb = MeshBuilder::new();
        let floor = mb.material(Material::CONCRETE);
        let door = mb.material(Material::DOOR);
        let v = Vec3::new;
        // the floor of both rooms, and the wall between them at x = 2
        mb.tri(v(0.0, 0.0, 0.0), v(4.0, 0.0, 0.0), v(4.0, 4.0, 0.0), floor);
        mb.tri(v(0.0, 0.0, 0.0), v(4.0, 4.0, 0.0), v(0.0, 4.0, 0.0), floor);
        mb.tri(v(2.0, 0.0, 0.0), v(2.0, 4.0, 0.0), v(2.0, 4.0, 3.0), door);
        mb.tri(v(2.0, 0.0, 0.0), v(2.0, 4.0, 3.0), v(2.0, 0.0, 3.0), door);
        let mesh = mb.build();
        let world = World { mesh: &mesh, panels: &[], grade: None, pockets: &[] };
        let mut hits = Vec::new();
        let src = v(3.5, 1.0, 1.5);
        let wall = world.transmission(v(1.0, 2.0, 1.0), src, &mut hits)[1];
        assert!(wall < 0.5, "the wall stops most of it: {wall}");
        for gap in [1e-4f32, 5e-5, 2e-5] {
            // the bounce on the floor, lifted EPS off it
            let off = v(2.0 - gap, 2.0, 0.0) + v(0.0, 0.0, 1.0) * EPS;
            let t = world.transmission_free(off, src, &mut hits)[1];
            assert!((t / wall - 1.0).abs() < 1e-3, "{gap} m from the wall: {t} through it, the wall passes {wall}");
        }
        let leak = world.transmission(v(2.0 - 1e-4, 2.0, EPS), src, &mut hits)[1];
        assert_eq!(leak, 1.0, "the default margin's blind start (what the walk used)");
    }

    /// The next-event draw: each source kept with probability π, the near
    /// loud one always, `m` kept every time, the estimate unbiased for any
    /// transmission, and no kept source weighed above τ (the share of the
    /// ones not connected for sure).
    #[test]
    fn systematic_picks_are_unbiased_and_bounded() {
        let c = [100.0f32, 5.0, 3.0, 1.0, 0.5, 0.2];
        // what each connection would find (its transmission, any value)
        let t = [0.3f32, 1.0, 0.01, 0.7, 1.0, 0.05];
        let m = 2;
        let tau = (c.iter().sum::<f32>() - c[0]) / (m - 1) as f32;
        let n = 20_000;
        let (mut kept, mut est) = ([0usize; 6], 0.0f64);
        for k in 0..n {
            let mut picks: Vec<Pick> = c
                .iter()
                .enumerate()
                .map(|(si, &c)| Pick { si: si as u32, c, pw: 1.0 })
                .collect();
            pick_systematic(&mut picks, m, (k as f32 + 0.5) / n as f32);
            assert_eq!(picks.len(), m, "Σπ = m: m kept every time");
            for p in &picks {
                let i = p.si as usize;
                kept[i] += 1;
                assert!(p.pw * p.c <= c[i].max(tau) * 1.0001, "weight bounded by τ");
                est += (p.pw * p.c * t[i]) as f64;
            }
        }
        assert_eq!(kept[0], n, "the near loud source is connected for sure");
        for i in 1..6 {
            let pi = (c[i] / tau).min(1.0) as f64;
            let f = kept[i] as f64 / n as f64;
            assert!((f - pi).abs() < 1e-3, "source {i}: kept {f:.4}, π {pi:.4}");
        }
        let truth: f64 = c.iter().zip(&t).map(|(&c, &t)| (c * t) as f64).sum();
        let mean = est / n as f64;
        assert!((mean / truth - 1.0).abs() < 1e-3, "unbiased: {mean} against {truth}");
    }
}
