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
/// again for this many ticks.
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
/// v2 echogram units: energy per 0.5 m receiver sphere = true / 16.
const LEGACY_UNITS: f32 = 1.0 / 16.0;
/// Tap key of the direct path; early bins start above it.
pub const KEY_DIRECT: u32 = 1;
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

    /// Per-band amplitude of the segment a→b: mesh walls by their real
    /// thickness, overlay boxes once each.
    pub fn transmission(&self, a: Vec3, b: Vec3, hits: &mut Vec<SegHit>) -> [f32; NBANDS] {
        if self.crosses_earth(a, b) {
            return [EARTH; NBANDS];
        }
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
    /// Rays passing close to a mesh edge bend around it (V3.4).
    pub diffraction: bool,
}

impl Default for Budget {
    fn default() -> Self {
        Self { rays: 1024, bounces: 4, k: 0.06, ema: 0.35, nee: 4, late_rays: 256, late_ema: 0.15, late_nee: 1, diffraction: true }
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

#[derive(Default)]
struct SourceState {
    early: HashMap<u32, Bin>,
    echo: Option<Echogram>,
    /// Solved chains currently played as discrete taps: one group per
    /// tap, its key first (chains arriving together merge, `ECHO_MERGE_S`).
    echo_on: Vec<Vec<u32>>,
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
    fn fold(
        &mut self,
        early: HashMap<u32, Bin>,
        echo: Echogram,
        budget: Budget,
    ) {
        let first = self.echo.is_none();
        let alpha_early = if first { 1.0 } else { budget.ema.clamp(0.0, 1.0) };
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
            Some(e) => e.ema(&echo, budget.late_ema.clamp(0.0, 1.0)),
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
        SourceField { taps, reverb, late_dir, aniso, reflected }
    }
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
    /// The outdoor field as the listener's own rays find it: per
    /// direction cell, the energy of rays that escaped to open sky
    /// (through doorways, panes, walls, off facades — whatever the walk
    /// did) and the energy-weighted sum of their starting directions.
    /// Averaged over ticks. This replaces the separate dome ray fan.
    sky: Vec<([f32; NBANDS], Vec3)>,
    sky_primed: bool,
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
    ear_vis: Option<(Vec3, u64, Vec<bool>)>,
    /// Per source id: (listener, source, tick, edge-path taps).
    edge_cache: HashMap<u32, (Vec3, Vec3, u64, Vec<Tap>)>,
    /// Mirror chains past `bounces`, per (source id, plane-sequence
    /// hash): found by coherent rays, solved exactly every tick.
    chains: HashMap<(u32, u64), MirrorChain>,
    /// Chains that failed to validate (or lost to stronger ones), by the
    /// tick they did: not named or grown again for `CHAIN_REJECT_TICKS`.
    rejected: HashMap<(u32, u64), u64>,
    /// Capture the strongest paths of each tick for the debug view.
    pub debug: bool,
    /// Last tick's strongest paths: (source id, mid energy, polyline
    /// ears → bounces → source), strongest first.
    pub debug_paths: Vec<(u32, f32, Vec<Vec3>)>,
}

/// A coherent chain past `bounces`: the planes its mirrors lie in, in
/// order from the listener.
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

fn planes_eq(a: (Vec3, f32), b: (Vec3, f32)) -> bool {
    a.0.dot(b.0) > 0.999 && (a.1 - b.1).abs() < 1e-3
}

fn mirror_point(p: Vec3, (n, d): (Vec3, f32)) -> Vec3 {
    p - n * (2.0 * (n.dot(p) - d))
}

/// Paths kept per tick for the debug view.
const DEBUG_PATHS: usize = 64;

/// Keep `path` if it is among the strongest DEBUG_PATHS so far.
fn keep_path(list: &mut Vec<(u32, f32, Vec<Vec3>)>, id: u32, e: f32, verts: &[Vec3], end: Vec3) {
    if list.len() >= DEBUG_PATHS {
        let (mi, me) = list
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.1))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        if e <= me {
            return;
        }
        list.swap_remove(mi);
    }
    let mut v = verts.to_vec();
    v.push(end);
    list.push((id, e, v));
}

impl Tracer {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            cells: fibonacci_sphere(N_CELLS),
            state: HashMap::new(),
            hits: Vec::new(),
            sky: vec![([0.0; NBANDS], Vec3::new(0.0, 0.0, 0.0)); N_CELLS],
            sky_primed: false,
            last_listener: None,
            epoch: 0,
            edges: None,
            tick: 0,
            ear_vis: None,
            edge_cache: HashMap::new(),
            chains: HashMap::new(),
            rejected: HashMap::new(),
            debug: false,
            debug_paths: Vec::new(),
        }
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

    /// Forget a source's history (it moved discontinuously, or left).
    pub fn reset(&mut self, id: u32) {
        self.state.remove(&id);
        self.chains.retain(|k, _| k.0 != id);
        self.rejected.retain(|k, _| k.0 != id);
    }

    /// Forget every average (the listener crossed a surface).
    fn forget_all(&mut self) {
        self.state.clear();
        self.chains.clear();
        self.rejected.clear();
        self.sky_primed = false;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Step 4b, shared with the externally walked fold: average the sky.
    fn fold_sky(&mut self, fresh: impl Iterator<Item = ([f32; NBANDS], Vec3)>, budget: Budget) {
        let a = if self.sky_primed { budget.ema.clamp(0.0, 1.0) } else { 1.0 };
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

        let debug = self.debug;
        let mut dbg: Vec<(u32, f32, Vec<Vec3>)> = Vec::new();
        let mut verts: Vec<Vec3> = Vec::new();

        // 1. direct, visibility, edge next-event paths (shared with the
        // externally walked tick, fold.rs)
        let (direct, vis, mut edge_taps) = self.prepare(world, listener, sources, budget, &mut dbg);

        // 2. rays: stratified (Fibonacci) and randomly rotated per tick
        let rot = random_rotation(&mut self.rng);
        let primary = fibonacci_sphere(n as usize);
        let mut picks: Vec<(usize, f32)> = Vec::new();
        let mut imp: Vec<f32> = Vec::new();
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
        // the exact solver's paths (both families evaluate it
        // identically): coherent, loud chains within the cap that have
        // more than `bounces` reflections or arrive after the early window
        let bounces = budget.bounces;
        let solver_owns = |refl: u32, coherent: bool, loud: bool, rel: f32| {
            coherent && loud && refl >= 1 && refl <= MAX_COHERENT_REFL && (refl > bounces || rel >= EARLY_S)
        };
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
            // every bounce a mirror so far (nothing scattered or bent),
            // and that mirror chain's own throughput still above the
            // floor (latched: once quiet, quiet)
            let mut coherent = true;
            let mut loud = true;
            let mut mirror_thr = 1.0f32;
            // bent around an edge since the last diffuse bounce, and
            // where the ray was when it first bent (gating: a bend only
            // adds what the straight line from there doesn't carry)
            let mut bent = false;
            let mut bends = 0u32;
            let mut bend_from: Option<Vec3> = None;
            // a ray passes an edge once: never bend twice at the same one
            let mut last_edge: Option<usize> = None;
            if debug {
                verts.clear();
                verts.push(listener);
            }
            for _seg in 0..segments {
                if world.in_earth(origin) {
                    break; // transmitted into solid ground: gone
                }
                let hit = world.raycast(origin, dir);
                let t_hit = hit.as_ref().map_or(f32::MAX, |h| h.t);
                // diffraction: the edge this segment passes closest to
                let edge = if budget.diffraction && bends < MAX_BENDS {
                    let reach = if t_hit.is_finite() { t_hit } else { 400.0 };
                    self.edges.as_mut().and_then(|es| es.nearest(origin, dir, reach, last_edge))
                } else {
                    None
                };
                let t_seg = edge.map_or(t_hit, |(te, _, _, _)| te);

                // 3. specular chains into a source sphere, priced from the
                // source's REAL position — except the exact solver's own
                if (specular || bent) && counts(refl) {
                    for (si, s) in sources.iter().enumerate() {
                        let to = s.pos - origin;
                        let along = to.dot(dir);
                        if along <= 0.0 || along >= t_seg {
                            continue;
                        }
                        let seg = to.length();
                        let l_s = (length + seg).max(MIN_DIST);
                        let r = budget.k * l_s;
                        let closest = (origin + dir * along - s.pos).length();
                        if closest > r {
                            continue;
                        }
                        if solver_owns(refl, coherent && !bent, loud, l_s / SPEED_OF_SOUND - accs[si].t_free) {
                            continue; // the exact solver counts this one
                        }
                        let t = world.transmission(origin, s.pos, &mut self.hits);
                        let air = air_attenuation(l_s);
                        let w = 4.0 * inv_n / (budget.k * budget.k * l_s * l_s);
                        // a bent path with no reflection is the DIRECT
                        // field around an edge: only what the straight
                        // ray doesn't already carry
                        // one edge, no reflection: that is the edge
                        // next-event path, already counted exactly
                        if refl == 0 && bends == 1 && !specular {
                            continue;
                        }
                        // a bent path only carries what the straight
                        // line from where it bent doesn't (lit side:
                        // nothing — that wave is already counted)
                        let gate: [f32; NBANDS] = match bend_from {
                            Some(_) if refl == 0 => core::array::from_fn(|b| 1.0 - vis[si][b]),
                            Some(f) => {
                                let tv = world.transmission(f, s.pos, &mut self.hits);
                                core::array::from_fn(|b| (1.0 - tv[b]).clamp(0.0, 1.0))
                            }
                            None => [1.0; NBANDS],
                        };
                        let e: [f32; NBANDS] = core::array::from_fn(|b| {
                            thr[b] * t[b] * t[b] * air[b] * air[b] * w * gate[b]
                        });
                        if debug {
                            keep_path(&mut dbg, s.id, e[1], &verts, s.pos);
                        }
                        deposit(&mut accs[si], &self.cells, e, l_s / SPEED_OF_SOUND, first);
                    }
                }

                if let Some((te, a, u, ei)) = edge {
                    last_edge = Some(ei);
                    // bend around the edge at the closest approach
                    if bend_from.is_none() {
                        bend_from = Some(origin);
                    }
                    let (eps, w) = sample_bend(&mut self.rng, a);
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
                        for (ci, c) in self.cells.iter().enumerate() {
                            let d = c.dot(first);
                            if d > bd {
                                bd = d;
                                best = ci;
                            }
                        }
                        let cell = &mut sky_acc[best];
                        for b in 0..NBANDS {
                            cell.0[b] += thr[b] * inv_n;
                        }
                        cell.1 = cell.1 + first * (thr[1] * inv_n);
                    }
                    break;
                };
                let p = origin + dir * t_hit;
                if debug {
                    verts.push(p);
                }
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
                        if debug {
                            keep_path(&mut dbg, s.id, e[1], &verts, s.pos);
                        }
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
                    mirror_thr *= mat.transmission[1] * mat.transmission[1];
                    loud &= mirror_thr >= COHERENT_FLOOR;
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
                bend_from = None;
                mirror_thr *= keep[1] * (1.0 - mat.scattering);
                loud &= mirror_thr >= COHERENT_FLOOR;
                if self.rng.next_f32() < mat.scattering {
                    dir = (normal + self.rng.unit_sphere()).normalize();
                    if dir.dot(normal) < 1e-3 {
                        dir = normal;
                    }
                    specular = false;
                    bent = false;
                    coherent = false;
                } else {
                    dir = dir - normal * (2.0 * dir.dot(normal));
                    specular = true;
                }
                origin = p + normal * EPS;
            }
        }
        }

        if debug {
            dbg.sort_by(|a, b| b.1.total_cmp(&a.1));
            self.debug_paths = dbg;
        }
        self.fold_sky(sky_acc.into_iter(), budget);

        // 4b. the exact solver's chains: named by a few mirror rays,
        // solved every tick while cached
        self.find_chains(world, listener, sources, budget, chain_finders(budget));
        self.solve_chains(world, listener, sources, budget, &mut accs);

        // 5. smooth over ticks and build each source's field
        let mut out = Vec::with_capacity(sources.len());
        for (si, s) in sources.iter().enumerate() {
            let t_free = accs[si].t_free;
            let acc = core::mem::replace(
                &mut accs[si],
                Acc { early: HashMap::new(), echo: Echogram::new(), t_free: 0.0, total: [0.0; NBANDS] },
            );
            let st = self.state.entry(s.id).or_default();
            st.fold(acc.early, acc.echo, budget);
            out.push(st.field(&self.cells, t_free, direct[si], &mut edge_taps[si], acc.total));
        }
        out
    }

    /// Cache (or refresh) the chain `planes` for source `id`.
    fn name_chain(&mut self, id: u32, planes: &[(Vec3, f32)], cached: &mut HashMap<u32, usize>) {
        let key = (id, planes_hash(planes));
        if self.rejected.get(&key).is_some_and(|&t| self.tick < t + CHAIN_REJECT_TICKS) {
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
        let mut cached: HashMap<u32, usize> = HashMap::new();
        for k in self.chains.keys() {
            *cached.entry(k.0).or_default() += 1;
        }
        let t_free: Vec<f32> = sources.iter().map(|s| (s.pos - listener).length() / SPEED_OF_SOUND).collect();
        // past the echo window of the farthest source, nothing is named
        let reach = (ECHO_S + t_free.iter().copied().fold(0.0, f32::max)) * SPEED_OF_SOUND;
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
                        let owns = refl > budget.bounces || rel >= EARLY_S;
                        if owns && (origin + dir * along - s.pos).length() <= CHAIN_FIND_K * l_s {
                            self.name_chain(s.id, &planes, &mut cached);
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

    /// The CPU's share of an externally walked tick: name chains with a
    /// few mirror rays, solve every cached one, and return what they
    /// deposit (per source) — merged into the walk's bins by the fold.
    fn walked_chains(&mut self, world: &World, listener: Vec3, sources: &[Source], budget: Budget) -> Vec<Acc> {
        self.find_chains(world, listener, sources, budget, chain_finders(budget));
        let mut accs: Vec<Acc> = sources
            .iter()
            .map(|s| Acc {
                early: HashMap::new(),
                echo: Echogram::new(),
                t_free: (s.pos - listener).length() / SPEED_OF_SOUND,
                total: [0.0; NBANDS],
            })
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
        let mut reject: Vec<(u32, u64)> = Vec::new();
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
                _ => {
                    let r = solve_chain(world, &c.planes, listener, src, &mut self.hits);
                    c.solved = r.map(|r| (listener, src, tick, r));
                    r
                }
            };
            match result {
                // short and early: the sphere estimate's (see `solver_owns`)
                Some((_, delay, _)) if !long && delay - accs[si].t_free < EARLY_S => reject.push(key),
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
                None => reject.push(key),
            }
        }
        for k in drop {
            self.chains.remove(&k);
        }
        let tick = self.tick;
        for k in reject {
            self.chains.remove(&k);
            self.rejected.insert(k, tick);
        }
        self.rejected.retain(|_, t| tick < *t + CHAIN_REJECT_TICKS);
        // over the cap (newly named ones get one solve to prove
        // themselves): keep each source's strongest MAX_CHAINS
        let mut by_source: HashMap<u32, Vec<((u32, u64), f32)>> = HashMap::new();
        for (k, c) in &self.chains {
            by_source.entry(k.0).or_default().push((*k, c.energy));
        }
        for (_, mut v) in by_source {
            if v.len() > MAX_CHAINS {
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (k, _) in &v[MAX_CHAINS..] {
                    self.chains.remove(k);
                    self.rejected.insert(*k, tick);
                }
            }
        }
    }

    /// A chain whose last mirrors alternate between two parallel planes
    /// is one member of a train — the flutter between them: every other
    /// number of round trips is a path too, as rare for a ray to find as
    /// this one was. Propose them all (the solve validates each).
    fn grow_trains(&mut self) {
        let mut per_source: HashMap<u32, usize> = HashMap::new();
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
                    let rejected = self.rejected.get(&k).is_some_and(|&t| tick < t + CHAIN_REJECT_TICKS);
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
    /// edge next-event paths (diffraction). Debug paths go to `dbg`.
    #[allow(clippy::type_complexity)]
    fn prepare(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        dbg: &mut Vec<(u32, f32, Vec<Vec3>)>,
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
        let debug = self.debug;
        // 1. direct: one exact segment per source
        let mut direct: Vec<Option<Tap>> = Vec::with_capacity(sources.len());
        for s in sources {
            let to = s.pos - listener;
            let d = to.length().max(MIN_DIST);
            let t = world.transmission(listener, s.pos, &mut self.hits);
            let air = air_attenuation(d);
            let gains: [f32; NBANDS] = core::array::from_fn(|b| t[b] * air[b] / d);
            if debug && gains[1] > 1e-7 {
                // direct paths always show, ranked like the rest
                keep_path(dbg, s.id, gains[1] * gains[1], &[listener], s.pos);
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
            .map(|(s, d)| match d {
                Some(t) => {
                    let dd = (s.pos - listener).length().max(MIN_DIST);
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
            }
            if let Some(es) = self.edges.as_mut() {
                es.mask_filled(world.panels, world.grade);
            }
        }

        // Edge next-event estimation: the DIRECT field around one edge,
        // connected deterministically for every blocked source (a random
        // ray finding a distant source after bending is too rare to
        // carry it). Knife-edge (Kurze–Anderson) per band, legs priced
        // through whatever they cross, only the share the straight ray
        // doesn't carry. Distinct edges add (each is its own path).
        self.tick += 1;
        let tick = self.tick;
        let mut edge_taps: Vec<Vec<Tap>> = vec![Vec::new(); sources.len()];
        if let (true, Some(es)) = (budget.diffraction, self.edges.as_ref()) {
            // which edges can the listener see at all? (shared by every
            // source: a few points along each edge, pulled 5 cm toward the
            // listener; reused while the listener stays put)
            let fresh_vis = self.ear_vis.as_ref().map_or(true, |(p, t, m)| {
                (*p - listener).length() > EDGE_REUSE_M || tick - t >= EDGE_REFRESH_TICKS || m.len() != es.edges.len()
            });
            if fresh_vis {
            let seen_from_ear: Vec<bool> = es
                .edges
                .iter()
                .enumerate()
                .map(|(ei, (a, b))| {
                    !es.filled[ei]
                        && [0.1f32, 0.3, 0.5, 0.7, 0.9].iter().any(|f| {
                            let m = *a + (*b - *a) * *f;
                            let to = m - listener;
                            let l = to.length();
                            l > 0.2 && l < 400.0 && {
                                let end = m - to * (0.05 / l);
                                world.mesh.raycast(listener, to * (1.0 / l)).map_or(true, |(t, _)| t > l - 0.06)
                                    && !world.crosses_earth(listener, end)
                            }
                        })
                })
                .collect();
                self.ear_vis = Some((listener, tick, seen_from_ear));
            }
            let seen_from_ear = &self.ear_vis.as_ref().unwrap().2;
            for (si, s) in sources.iter().enumerate() {
                if vis[si][1] > 0.999 {
                    continue;
                }
                if let Some((pl, ps, t, taps)) = self.edge_cache.get(&s.id) {
                    if (*pl - listener).length() < EDGE_REUSE_M
                        && (*ps - s.pos).length() < EDGE_REUSE_M
                        && tick - t < EDGE_REFRESH_TICKS
                    {
                        edge_taps[si] = taps.clone();
                        continue;
                    }
                }
                let direct_len = (s.pos - listener).length().max(MIN_DIST);
                // rank every exposed edge by its pure knife-edge bound
                let mut cand: Vec<(f32, usize, Vec3)> = es
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(ei, _)| seen_from_ear[*ei])

                    .filter_map(|(ei, (a, b))| {
                        let p = crate::paths::bend_point(*a, *b, s.pos, listener);
                        let (d1, d2) = ((p - listener).length(), (s.pos - p).length());
                        if d1 < 0.2 || d2 < 0.2 {
                            return None;
                        }
                        // a bend AT ground level is no silhouette for an
                        // above-ground path (it is where an edge meets the
                        // ground, not where it stands out)
                        if world.grade.is_some_and(|g| p.z < g + 0.3 && listener.z > g) {
                            return None;
                        }
                        let det = d1 + d2 - direct_len;
                        let ka = crate::diffraction::knife_edge_bands(det.max(1e-4));
                        Some((ka[0] / (d1 + d2), ei, p))
                    })
                    .collect();
                cand.sort_by(|x, y| y.0.total_cmp(&x.0));
                // price every candidate, then keep the strongest distinct
                let mut found: Vec<(f32, Vec3, Tap)> = Vec::new();
                for (_, ei, p) in cand.into_iter().take(EDGE_CANDIDATES) {
                    let to_edge = p - listener;
                    let n1 = to_edge.normalize();
                    let (d1, d2) = (to_edge.length(), (s.pos - p).length());
                    // both legs meet 3 cm off the edge on the OUTSIDE of
                    // the bend (away from whatever the path bends around):
                    // a true silhouette edge has open air there and neither
                    // leg grazes it; an edge lying on a zero-thickness wall
                    // plane leaves one leg crossing that wall
                    let n2 = (s.pos - p).normalize();
                    let out = n1 - n2;
                    if out.length() < 1e-3 {
                        continue; // no bend: the straight line itself
                    }
                    let q = p + out.normalize() * 0.03;
                    let t1 = world.transmission(listener, q, &mut self.hits);
                    let t2 = world.transmission(q, s.pos, &mut self.hits);
                    if t1[0] * t2[0] < 1e-6 {
                        continue;
                    }
                    let det = d1 + d2 - direct_len;
                    let ka = crate::diffraction::knife_edge_bands(det.max(1e-4));
                    let air = air_attenuation(d1 + d2);
                    let gains: [f32; NBANDS] = core::array::from_fn(|b| {
                        ka[b] * t1[b] * t2[b] * air[b] / (d1 + d2) * (1.0 - vis[si][b]).sqrt()
                    });
                    if gains.iter().all(|&g| g < 1e-7) {
                        continue;
                    }
                    found.push((
                        gains[0] + gains[1],
                        p,
                        Tap {
                            key: KEY_EDGE_BASE + ei as u32,
                            delay_s: (d1 + d2) / SPEED_OF_SOUND,
                            dir: [n1.x, n1.y, n1.z],
                            gains,
                        },
                    ));
                }
                found.sort_by(|x, y| y.0.total_cmp(&x.0));
                let mut kept: Vec<Vec3> = Vec::new();
                for (_, p, tap) in found {
                    if kept.len() >= EDGE_PATHS {
                        break;
                    }
                    // coincident edges (a parapet over its slab) are one path
                    if kept.iter().any(|q| (*q - p).length() < 1.0) {
                        continue;
                    }
                    kept.push(p);
                    if debug {
                        keep_path(dbg, s.id, tap.gains[1] * tap.gains[1], &[listener, p], s.pos);
                    }
                    edge_taps[si].push(tap);
                }
                self.edge_cache.insert(s.id, (listener, s.pos, tick, edge_taps[si].clone()));
            }
        }

        (direct, vis, edge_taps)
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
        let dbin = (rel / DELAY_BIN_S) as u32;
        let key = KEY_EARLY_BASE + cell_index(cells, dir) as u32 * 256 + dbin.min(255);
        acc.early.entry(key).or_default().add(e, delay, dir);
    }
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
        let hit = world.raycast(p + dir * skip, dir)?;
        if (hit.t + skip - len).abs() > 0.02 || hit.normal.dot(nrm).abs() < 0.999 {
            return None; // something else first, or not this plane
        }
        first.get_or_insert(dir);
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
/// Edge paths and listener edge visibility are reused until the
/// listener (or the source) moves this far, refreshed every N ticks.
const EDGE_REUSE_M: f32 = 0.2;
const EDGE_REFRESH_TICKS: u64 = 8;
/// Deterministic single-edge direct paths kept per source.
const EDGE_PATHS: usize = 3;
/// Edges whose legs are checked per blocked source (best knife-edge
/// bounds first — most near the blocked line are themselves blocked).
const EDGE_CANDIDATES: usize = 40;
/// Tap keys of edge paths: base + edge index (early bins sit below).
const KEY_EDGE_BASE: u32 = 20_000;
const EDGE_CELL: f32 = 4.0;

/// Diffracting edges of a mesh (sharp creases ≥ 30° and free borders),
/// in a uniform grid for segment queries.
pub struct EdgeSet {
    pub edges: Vec<(Vec3, Vec3)>,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    key: (usize, usize),
    seen: Vec<u32>,
    stamp: u32,
    /// Bounds of all edges, padded by EDGE_REACH: rays outside it skip
    /// the search entirely.
    lo: Vec3,
    hi: Vec3,
    /// Edges whose opening is FILLED this tick (a pane, a closed or ajar
    /// door leaf against the frame): not exposed, no diffraction.
    filled: Vec<bool>,
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
        let mut by_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (ti, idx) in mesh.indices.iter().enumerate() {
            for k in 0..3 {
                let (i, j) = (idx[k], idx[(k + 1) % 3]);
                by_edge.entry((i.min(j), i.max(j))).or_default().push(ti as u32);
            }
        }
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
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
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
            seen: vec![0; n],
            stamp: 0,
            lo: lo - r,
            hi: hi + r,
            filled: vec![false; n],
        }
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
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.stamp = 1;
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
                        if self.seen[ei] == self.stamp || self.filled[ei] || skip == Some(ei) {
                            continue;
                        }
                        self.seen[ei] = self.stamp;
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
    const N: usize = 16;
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
    // > 225°: a wall standing on the ground measures ~200° (rays along
    // its plane slip by), an outside corner or roof line ~270°+
    best * 360 > 225 * N
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
