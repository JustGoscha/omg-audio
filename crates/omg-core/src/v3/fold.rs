//! Engine v3, walked elsewhere (ENGINE_V3.md, V3.3).
//!
//! The heavy half of a v3 tick — the ray walk of both families: raycasts,
//! next-event shadow segments, source-sphere tests, transmission sampling,
//! edge bends, sky escapes — can run outside this crate, in the WGSL
//! kernel `crates/omg-gpu/shaders/v3_walk.wgsl` (natively through wgpu,
//! in the browser through `web/gpu.js`). The walker hands back the walk's
//! contributions ALREADY BINNED exactly as `deposit` bins them (direction
//! cell × 2 ms early bins, 10 ms echogram bins relative to the free-field
//! arrival, sky cells); this module folds them into the same per-source
//! EMA state and builds the same `SourceField`s as `Tracer::trace`.
//! Everything that is not the walk — the exact direct segments, the
//! direct visibility, the edge next-event paths — runs on the CPU through
//! the very code `trace` uses (`Tracer::prepare`), every tick, even while
//! an asynchronous walk is still in flight.
//!
//! `Tracer::trace` (the CPU walk) stays the oracle and the fallback. The
//! fold below mirrors trace()'s step 5 — keep the two in step.
//!
//! Flat formats (the contract with the kernel; bump `WALK_LAYOUT_VERSION`
//! on ANY change, here and in v3_walk.wgsl):
//!
//! JOB — u32 words (f32 payloads as bits): the header, then the scene.
//! ```text
//! header (HEADER_WORDS = 40, the kernel's 160-byte uniform):
//!   0..3 rot row 0   3 n_early     4..7 rot row 1   7 n_late
//!   8..11 rot row 2  11 bounces    12..15 listener  15 seed
//!   16 k  17 nee  18 late_nee  19 n_sources
//!   20 n_panels  21 n_pockets  22 has_grade  23 grade
//!   24 debug  25 n_edges  26 off_panels  27 off_pockets  (scene words)
//!   28..31 edge bounds lo  31 off_edges   32..35 edge bounds hi
//!   35..40 reserved (0)
//! scene: sources (SRC_WORDS = 12 each): pos xyz, weight, t_free, id,
//!          direct visibility xyz, fixed-point norm, 0, 0
//!        panels (PANEL_WORDS = 16):  min xyz, scattering, max xyz, 0,
//!          absorption xyz, 0, transmission xyz, 0
//!        pockets (POCKET_WORDS = 8): min xyz, 0, max xyz, 0
//!        edges (EDGE_WORDS = 8): a xyz, 0, b xyz, 0 — the diffracting
//!          edges the CPU walk could bend at this tick (gridded, not
//!          filled by a panel, not on the ground); none = no diffraction
//! ```
//! OUTPUT — f32 words, true energy units (the fold applies the v2
//! echogram scale): SKY_WORDS of sky cells, then one SOURCE_WORDS block
//! per job source in job order. A slot is (e₀, e₁, e₂, Σe₁·dx, Σe₁·dy,
//! Σe₁·dz) with d the ray's first direction. A source block holds the
//! early slots [cell·DELAY_BINS + delay bin], then the echogram slots,
//! then the per-band total over the echogram (+1 pad word). Echogram
//! direction sums are only kept from EARLY_S on (all `late_direction`
//! reads); `reflected` counts what lands inside the 3 s echogram.
//!
//! DEBUG — DEBUG_WORDS per early ray: mid energy, source index, vertex
//! count, 0, then up to DEBUG_VERTS vertices (ears → bends/bounces →
//! source).

use super::*;

/// Bump on ANY change to the flat formats above or in v3_walk.wgsl.
pub const WALK_LAYOUT_VERSION: u32 = 6;
/// Direction cells for early binning (and the sky).
pub const CELLS: usize = N_CELLS;
/// Early delay bins: the early window over the delay bin width.
pub const DELAY_BINS: usize = 40;
/// Echogram bins (the v2 echogram: MAX_TIME / BIN_DT).
pub const ECHO_BINS: usize = (MAX_TIME / BIN_DT) as usize;
/// Words per slot: three band energies + the mid-weighted direction sum.
pub const SLOT: usize = 6;
pub const EARLY_WORDS: usize = CELLS * DELAY_BINS * SLOT;
pub const ECHO_WORDS: usize = ECHO_BINS * SLOT;
/// One source's output block: early slots, echogram, totals (+ pad).
pub const SOURCE_WORDS: usize = EARLY_WORDS + ECHO_WORDS + 4;
/// v4 (header word 35 set): the block is the dense response instead —
/// `DENSE_BINS` slots of (e₀, e₁, e₂, Σe₁·dx, Σe₁·dy, Σe₁·dz, Σe₁²) at
/// `DENSE_BIN_S` from the free-field arrival, then the totals.
pub const DENSE_BINS: usize = (RESPONSE_MAX_S / DENSE_BIN_S + 0.5) as usize;
pub const DENSE_SLOT: usize = 7;
pub const DENSE_WORDS: usize = DENSE_BINS * DENSE_SLOT;
const _: () = assert!(DENSE_WORDS + 4 <= SOURCE_WORDS);
pub const SKY_WORDS: usize = CELLS * SLOT;
pub const HEADER_WORDS: usize = 40;
pub const SRC_WORDS: usize = 12;
pub const PANEL_WORDS: usize = 16;
pub const POCKET_WORDS: usize = 8;
pub const EDGE_WORDS: usize = 8;
/// Per-dispatch caps. A world beyond them traces on the CPU instead — a
/// silently clamped input would change the physics.
pub const MAX_SOURCES: usize = 64;
/// Panels: door leaves and panes plus every large piece of furniture (the
/// demo town carries ~90).
pub const MAX_PANELS: usize = 256;
pub const MAX_POCKETS: usize = 8;
pub const MAX_EDGES: usize = 16384;
pub const DEBUG_VERTS: usize = 8;
pub const DEBUG_WORDS: usize = 4 + DEBUG_VERTS * 3;
/// Words per sliced ray's state in the kernel's `dbg` buffer: the ray's
/// and its split branch's (v3_walk.wgsl STATE_WORDS).
pub const STATE_WORDS: usize = 56;
/// Debug paths a walker needs to return at most (the view's cap).
pub const DEBUG_KEEP: usize = DEBUG_PATHS;

/// Output words for a job with `n_sources` sources.
pub fn output_words(n_sources: usize) -> usize {
    SKY_WORDS + n_sources * SOURCE_WORDS
}

/// Scene words a job can carry at most (host buffer sizing).
pub const MAX_SCENE_WORDS: usize =
    MAX_SOURCES * SRC_WORDS + MAX_PANELS * PANEL_WORDS + MAX_POCKETS * POCKET_WORDS + MAX_EDGES * EDGE_WORDS;

/// Was this edge put into the CPU's search grid? (`EdgeSet::build`
/// skips edges whose padded box spans more than 4096 cells.)
fn gridded(a: Vec3, b: Vec3) -> bool {
    let r = Vec3::new(EDGE_REACH, EDGE_REACH, EDGE_REACH);
    let lo = Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)) - r;
    let hi = Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)) + r;
    let (k0, k1) = (cell_key(lo), cell_key(hi));
    let cells = ((k1.0 - k0.0 + 1) as i64) * ((k1.1 - k0.1 + 1) as i64) * ((k1.2 - k0.2 + 1) as i64);
    cells <= 4096
}

/// One tick's walk, flattened for an external walker.
#[derive(Clone, Debug)]
pub struct WalkJob {
    pub words: Vec<u32>,
    /// Source ids in job order (the output blocks' order).
    pub ids: Vec<u32>,
    /// The tracer's epoch at launch; hand it back in `Walked::epoch`. A
    /// result from before the listener crossed a surface is discarded.
    pub epoch: u32,
}

impl WalkJob {
    /// Flatten one tick. `vis`: each source's per-band direct visibility
    /// (from `prepare`); `edges`: the diffraction edge set with this
    /// tick's fill mask, None when diffraction is off.
    #[allow(clippy::too_many_arguments)]
    /// Mark the job as an engine-v4 walk (the dense response is binned).
    pub fn v4(mut self, on: bool) -> Self {
        self.words[35] = on as u32;
        self
    }

    pub fn build(
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        rot: [[f32; 3]; 3],
        seed: u32,
        debug: bool,
        vis: &[[f32; NBANDS]],
        edges: Option<&EdgeSet>,
    ) -> Self {
        let f = |x: f32| x.to_bits();
        let mut w = vec![0u32; HEADER_WORDS];
        for r in 0..3 {
            for c in 0..3 {
                w[r * 4 + c] = f(rot[r][c]);
            }
        }
        w[3] = budget.rays.max(1);
        w[7] = budget.late_rays;
        w[11] = budget.bounces;
        w[12] = f(listener.x);
        w[13] = f(listener.y);
        w[14] = f(listener.z);
        w[15] = seed;
        w[16] = f(budget.k);
        w[17] = budget.nee as u32;
        w[18] = budget.late_nee as u32;
        w[19] = sources.len() as u32;
        w[20] = world.panels.len() as u32;
        w[21] = world.pockets.len() as u32;
        w[22] = world.grade.is_some() as u32;
        w[23] = f(world.grade.unwrap_or(0.0));
        w[24] = debug as u32;
        // w[35]: v4 (set by the tracer, `WalkJob::v4`)
        for (si, s) in sources.iter().enumerate() {
            let d = (s.pos - listener).length();
            let v = vis.get(si).copied().unwrap_or([0.0; NBANDS]);
            // fixed-point normalization: the free-field energy at the
            // source's distance sets the scale its bins accumulate at
            let norm = (d * d).max(1.0);
            w.extend_from_slice(&[f(s.pos.x), f(s.pos.y), f(s.pos.z), f(s.weight), f(d / SPEED_OF_SOUND), s.id]);
            w.extend_from_slice(&[f(v[0]), f(v[1]), f(v[2]), f(norm), 0, 0]);
        }
        w[26] = (w.len() - HEADER_WORDS) as u32;
        for (mn, mx, m) in world.panels {
            w.extend_from_slice(&[f(mn.x), f(mn.y), f(mn.z), f(m.scattering)]);
            w.extend_from_slice(&[f(mx.x), f(mx.y), f(mx.z), 0]);
            w.extend_from_slice(&[f(m.absorption[0]), f(m.absorption[1]), f(m.absorption[2]), 0]);
            w.extend_from_slice(&[f(m.transmission[0]), f(m.transmission[1]), f(m.transmission[2]), 0]);
        }
        w[27] = (w.len() - HEADER_WORDS) as u32;
        for (mn, mx) in world.pockets {
            w.extend_from_slice(&[f(mn.x), f(mn.y), f(mn.z), 0, f(mx.x), f(mx.y), f(mx.z), 0]);
        }
        w[31] = (w.len() - HEADER_WORDS) as u32;
        let mut n_edges = 0u32;
        if let Some(es) = edges {
            for (ei, (a, b)) in es.edges.iter().enumerate() {
                if es.filled[ei] || !gridded(*a, *b) || n_edges as usize >= MAX_EDGES {
                    continue;
                }
                w.extend_from_slice(&[f(a.x), f(a.y), f(a.z), 0, f(b.x), f(b.y), f(b.z), 0]);
                n_edges += 1;
            }
            for (k, x) in [es.lo.x, es.lo.y, es.lo.z].iter().enumerate() {
                w[28 + k] = f(*x);
            }
            for (k, x) in [es.hi.x, es.hi.y, es.hi.z].iter().enumerate() {
                w[32 + k] = f(*x);
            }
        }
        w[25] = n_edges;
        Self { words: w, ids: sources.iter().map(|s| s.id).collect(), epoch: 0 }
    }

    pub fn header(&self) -> &[u32] {
        &self.words[..HEADER_WORDS]
    }
    /// Everything after the header (sources, panels, pockets, edges).
    pub fn scene(&self) -> &[u32] {
        &self.words[HEADER_WORDS..]
    }
    pub fn n_early(&self) -> u32 {
        self.words[3]
    }
    pub fn n_late(&self) -> u32 {
        self.words[7]
    }
    /// Threads to launch: one per ray of either family.
    /// Dispatches a sliced walk takes at `steps` segments a ray each: the
    /// early rays' longest walk (v4: a ray and its split branch, each up
    /// to the early family's segments).
    pub fn passes(&self, steps: u32) -> u32 {
        if steps == 0 {
            return 1;
        }
        let segments = self.words[11] + 1 + super::MAX_BENDS;
        let walk = if self.words[35] != 0 { 2 * segments } else { segments };
        walk.div_ceil(steps).max(1)
    }

    pub fn n_rays(&self) -> u32 {
        self.n_early() + self.n_late()
    }
    pub fn n_sources(&self) -> usize {
        self.words[19] as usize
    }
    pub fn n_panels(&self) -> usize {
        self.words[20] as usize
    }
    pub fn n_pockets(&self) -> usize {
        self.words[21] as usize
    }
    pub fn n_edges(&self) -> usize {
        self.words[25] as usize
    }
    pub fn debug(&self) -> bool {
        self.words[24] != 0
    }
}

/// A finished walk: the binned output of one job.
#[derive(Clone, Debug, Default)]
pub struct Walked {
    /// The job's source ids, in output-block order (a late result keys
    /// its blocks by id: the source set may have changed since).
    pub ids: Vec<u32>,
    /// `output_words(ids.len())` f32 words.
    pub data: Vec<f32>,
    /// Strongest walked paths: (source id, mid energy, polyline).
    pub debug: Vec<(u32, f32, Vec<Vec3>)>,
    /// `WalkJob::epoch` of the job this answers.
    pub epoch: u32,
}

/// Decode debug records (DEBUG_WORDS each, any order) into the
/// strongest DEBUG_KEEP paths. `ids` maps a record's source index.
pub fn decode_debug(words: &[u32], ids: &[u32]) -> Vec<(u32, f32, Vec<Vec3>)> {
    let mut list = Vec::new();
    for r in words.chunks_exact(DEBUG_WORDS) {
        let e = f32::from_bits(r[0]);
        let (si, nv) = (r[1] as usize, (r[2] as usize).min(DEBUG_VERTS));
        if !(e > 0.0) || si >= ids.len() || nv < 2 {
            continue;
        }
        let verts: Vec<Vec3> = (0..nv)
            .map(|k| {
                let o = 4 + k * 3;
                Vec3::new(f32::from_bits(r[o]), f32::from_bits(r[o + 1]), f32::from_bits(r[o + 2]))
            })
            .collect();
        let (end, body) = verts.split_last().unwrap();
        keep_path(&mut list, ids[si], e, body, *end);
    }
    list.sort_by(|a, b| b.1.total_cmp(&a.1));
    list
}

/// What a walker returns for one job.
pub enum Walk {
    /// A finished walk — this job's, or an earlier one's (asynchronous
    /// walkers deliver a tick late; the EMA absorbs it).
    Fresh(Walked),
    /// Nothing finished yet: hold the smoothed state, refresh the
    /// direct and edge paths.
    Pending,
    /// Can't take this job (no device, caps): trace on the CPU instead.
    Refused,
}

/// Parity hooks for the GPU tests: the CPU walk's edge set with this
/// world's fill mask, and its nearest-edge query along a segment
/// (t, distance) — exactly what the walk bends at.
pub fn edge_set(world: &World) -> EdgeSet {
    let mut es = EdgeSet::build(world.mesh);
    es.mask_filled(world.panels, world.grade);
    es
}

pub fn edge_nearest(es: &mut EdgeSet, o: Vec3, d: Vec3, t_end: f32) -> Option<(f32, f32)> {
    es.nearest(o, d, t_end, None).map(|(t, a, _, _)| (t, a))
}

impl Tracer {
    /// `trace`, with the ray walk done by `walk` (the GPU kernel). The
    /// direct and edge paths, the EMA and the field construction run here
    /// exactly as in `trace`. Returns None when the walker refused (or the
    /// world exceeds the flat caps) — the caller traces on the CPU instead.
    pub fn trace_walked(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        walk: &mut dyn FnMut(&WalkJob) -> Walk,
    ) -> Option<Vec<SourceField>> {
        if sources.len() > MAX_SOURCES || world.panels.len() > MAX_PANELS || world.pockets.len() > MAX_POCKETS {
            return None;
        }
        let dbg: Vec<(u32, f32, Vec<Vec3>)> = Vec::new();
        let (direct, vis, edge_taps) = self.prepare(world, listener, sources, budget);
        let rot = random_rotation(&mut self.rng);
        let seed = (self.rng.next_u64() >> 32) as u32;
        let edges = if budget.diffraction { self.edges.as_ref() } else { None };
        let mut job = WalkJob::build(world, listener, sources, budget, rot, seed, self.debug, &vis, edges).v4(self.v4);
        job.epoch = self.epoch;
        let walked = match walk(&job) {
            Walk::Refused => return None,
            Walk::Pending => None,
            Walk::Fresh(w) => Some(w),
        };
        // the mirror chains the kernel leaves to the CPU (named by a few
        // mirror rays here, solved exactly), folded in with a fresh walk
        let chains = walked.as_ref().map(|_| self.walked_chains(world, listener, sources, budget));
        Some(self.fold_walked(listener, sources, budget, direct, edge_taps, dbg, walked, chains))
    }

    /// Fold a walk's binned contributions (or none, while one is in
    /// flight) into the smoothed state and build every source's field.
    #[allow(clippy::too_many_arguments)]
    fn fold_walked(
        &mut self,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        direct: Vec<Option<Tap>>,
        mut edge_taps: Vec<Vec<Tap>>,
        mut dbg: Vec<(u32, f32, Vec<Vec3>)>,
        walked: Option<Walked>,
        chains: Option<Vec<Acc>>,
    ) -> Vec<SourceField> {
        // a walk launched before the listener crossed a surface describes
        // the other side: drop it, the next one is from here
        let fresh = walked
            .as_ref()
            .is_some_and(|w| w.epoch == self.epoch && w.data.len() >= output_words(w.ids.len()));
        if fresh {
            let w = walked.as_ref().unwrap();
            if self.debug {
                for (id, e, v) in &w.debug {
                    let (end, body) = v.split_last().unwrap();
                    keep_path(&mut dbg, *id, *e, body, *end);
                }
            }
            // the sky field, averaged exactly like the CPU walk's
            let sky: Vec<([f32; NBANDS], Vec3)> = (0..self.sky.len())
                .map(|ci| {
                    let e = &w.data[ci * SLOT..ci * SLOT + SLOT];
                    ([e[0], e[1], e[2]], Vec3::new(e[3], e[4], e[5]))
                })
                .collect();
            self.fold_sky(sky.into_iter(), budget, listener);
        }
        if self.debug {
            dbg.sort_by(|a, b| b.1.total_cmp(&a.1));
            self.debug_paths = dbg;
        }

        // 5. smooth over ticks and build each source's field — the same
        // `SourceState::fold` / `field` as `trace` (a source this walk
        // didn't carry keeps its state)
        let mut out = Vec::with_capacity(sources.len());
        for (si, s) in sources.iter().enumerate() {
            let t_free = (s.pos - listener).length() / SPEED_OF_SOUND;
            let block = walked.as_ref().filter(|_| fresh).and_then(|w| {
                w.ids.iter().position(|&id| id == s.id).map(|k| {
                    let o = SKY_WORDS + k * SOURCE_WORDS;
                    &w.data[o..o + SOURCE_WORDS]
                })
            });
            let st = self.state.entry(s.id).or_default();
            let mut total = [0.0f32; NBANDS];
            if self.v4 {
                if let Some(blk) = block {
                    // the walk's dense response, as `deposit` bins it, and
                    // the echogram (the RT60 readout) summed from it
                    let mut acc = Acc::new(t_free, true);
                    let mut last = 0;
                    for bin in 0..DENSE_BINS {
                        let o = bin * DENSE_SLOT;
                        if blk[o..o + DENSE_SLOT].iter().any(|&x| x != 0.0) {
                            last = bin + 1;
                        }
                    }
                    acc.dense = (0..last)
                        .map(|bin| core::array::from_fn(|c| blk[bin * DENSE_SLOT + c]))
                        .collect();
                    for (bin, d) in acc.dense.iter().enumerate() {
                        let j = (bin as f32 * DENSE_BIN_S / BIN_DT) as usize;
                        if j < acc.echo.bins.len() {
                            for b in 0..NBANDS {
                                acc.echo.bins[j][b] += d[b] * LEGACY_UNITS;
                            }
                        }
                    }
                    let o = DENSE_WORDS;
                    acc.total = [blk[o], blk[o + 1], blk[o + 2]];
                    // the CPU-solved chains on top (the kernel skipped them)
                    if let Some(c) = chains.as_ref().and_then(|c| c.get(si)) {
                        for (k, b) in &c.early {
                            acc.early.insert(*k, *b);
                            for band in 0..NBANDS {
                                acc.total[band] += b.e[band];
                            }
                        }
                    }
                    total = acc.total;
                    st.fold_v4(acc, budget, listener, s.pos);
                }
                out.push(st.field_v4(t_free, direct[si], &mut edge_taps[si], total));
                continue;
            }
            if let Some(blk) = block {
                // the walk's accumulators, as `deposit` would have built them
                let mut early: HashMap<u32, Bin> = HashMap::new();
                for cell in 0..CELLS {
                    for db in 0..DELAY_BINS {
                        let o = (cell * DELAY_BINS + db) * SLOT;
                        let e = [blk[o], blk[o + 1], blk[o + 2]];
                        if !e.iter().any(|&x| x > 0.0) {
                            continue;
                        }
                        let key = KEY_EARLY_BASE + cell as u32 * 256 + db as u32;
                        early.insert(
                            key,
                            Bin {
                                e,
                                // (unused downstream: the tap delay is pinned
                                // to the bin; kept for the EMA's symmetry)
                                ed: e[1] * (t_free + (db as f32 + 0.5) * DELAY_BIN_S),
                                dir: Vec3::new(blk[o + 3], blk[o + 4], blk[o + 5]),
                                last: 0.0,
                            },
                        );
                    }
                }
                let mut echo = Echogram::new();
                for bin in 0..ECHO_BINS.min(echo.bins.len()) {
                    let o = EARLY_WORDS + bin * SLOT;
                    for b in 0..NBANDS {
                        echo.bins[bin][b] = blk[o + b] * LEGACY_UNITS;
                    }
                    for k in 0..3 {
                        echo.dirs[bin][k] = blk[o + 3 + k] * LEGACY_UNITS;
                    }
                }
                let o = EARLY_WORDS + ECHO_WORDS;
                total = [blk[o], blk[o + 1], blk[o + 2]];
                // the CPU-solved chains on top (the kernel skipped them)
                if let Some(acc) = chains.as_ref().and_then(|c| c.get(si)) {
                    for (k, b) in &acc.early {
                        let cur = early.entry(*k).or_default();
                        for band in 0..NBANDS {
                            cur.e[band] += b.e[band];
                            total[band] += b.e[band];
                        }
                        cur.ed += b.ed;
                        cur.dir = cur.dir + b.dir;
                        cur.last = b.last;
                    }
                    for (bin, add) in echo.bins.iter_mut().zip(acc.echo.bins.iter()) {
                        for band in 0..NBANDS {
                            bin[band] += add[band];
                        }
                    }
                    for (d, add) in echo.dirs.iter_mut().zip(acc.echo.dirs.iter()) {
                        for k in 0..3 {
                            d[k] += add[k];
                        }
                    }
                }

                st.fold(early, echo, budget);
            }
            out.push(st.field(&self.cells, t_free, direct[si], &mut edge_taps[si], total));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_constants() {
        assert_eq!(ECHO_BINS, 300);
        assert!((DELAY_BINS as f32 * DELAY_BIN_S - EARLY_S).abs() < 1e-6);
        assert_eq!(SOURCE_WORDS, 17164);
        assert_eq!(SKY_WORDS, 384);
        assert_eq!(DEBUG_WORDS, 28);
    }

    /// An asynchronous walker answers a tick late. Once the listener has
    /// crossed a surface, the answer to a job launched on the OTHER side
    /// must be dropped — folding it in would re-seed the cleared averages
    /// with the room just left (its reverb heard outside).
    #[test]
    fn a_walk_from_before_crossing_a_wall_is_discarded() {
        let mut mb = crate::mesh::MeshBuilder::new();
        let wall = mb.material(Material::BRICK);
        mb.abox(Vec3::new(-0.2, -50.0, -1.0), Vec3::new(0.2, 50.0, 10.0), wall);
        let mesh = mb.build();
        let world = World::new(&mesh, &[]);
        let src = [Source { id: 7, pos: Vec3::new(5.0, 0.0, 1.6), weight: 1.0 }];
        let budget = Budget::default();
        let mut tracer = Tracer::new(3);
        // every walk lands energy in every early bin; `stale` answers
        // with the previous job's epoch, as a one-tick-late GPU would
        let mut last_epoch = 0u32;
        let walk = |stale: bool| {
            move |job: &WalkJob| {
                let epoch = if stale { last_epoch } else { job.epoch };
                last_epoch = job.epoch;
                Walk::Fresh(Walked { ids: job.ids.clone(), data: vec![1e-3; output_words(job.ids.len())], debug: Vec::new(), epoch })
            }
        };
        let early = |f: &SourceField| f.taps.iter().filter(|t| t.key >= KEY_EARLY_BASE && t.key < 20_000).count();
        let near = Vec3::new(3.0, 0.0, 1.6);
        for _ in 0..3 {
            let f = tracer.trace_walked(&world, near, &src, budget, &mut walk(false)).unwrap();
            assert!(early(&f[0]) > 0, "a fresh walk on this side lands");
        }
        // fly through the wall: this tick's answer is the old side's job
        let far = Vec3::new(-3.0, 0.0, 1.6);
        let f = tracer.trace_walked(&world, far, &src, budget, &mut walk(true)).unwrap();
        assert_eq!(early(&f[0]), 0, "the walk from the other side of the wall was folded in");
        let f = tracer.trace_walked(&world, far, &src, budget, &mut walk(false)).unwrap();
        assert!(early(&f[0]) > 0, "the next walk, from this side, lands again");
    }

    #[test]
    fn job_tables_round_trip() {
        let mesh = crate::mesh::MeshBuilder::new().build();
        let panels = [(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0), Material::WOOD_PANEL)];
        let pockets = [(Vec3::new(-1.0, -1.0, -3.0), Vec3::new(1.0, 1.0, -0.5))];
        let world = World { grade: Some(0.0), pockets: &pockets, ..World::new(&mesh, &panels) };
        let srcs = [Source { id: 7, pos: Vec3::new(3.0, 4.0, 0.0), weight: 2.0 }];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = Budget { rays: 100, late_rays: 10, bounces: 3, ..Budget::default() };
        let job = WalkJob::build(&world, Vec3::new(0.0, 0.0, 0.0), &srcs, b, rot, 9, true, &[[0.5; 3]], None);
        assert_eq!(job.n_rays(), 110);
        assert_eq!((job.n_sources(), job.n_panels(), job.n_pockets(), job.n_edges()), (1, 1, 1, 0));
        let sc = job.scene();
        assert_eq!(f32::from_bits(sc[4]), 5.0 / SPEED_OF_SOUND);
        assert_eq!(sc[5], 7);
        assert_eq!(f32::from_bits(sc[7]), 0.5);
        assert_eq!(f32::from_bits(sc[9]), 25.0);
        let op = job.words[26] as usize;
        assert_eq!(op, SRC_WORDS);
        assert_eq!(f32::from_bits(sc[op + 3]), Material::WOOD_PANEL.scattering);
        let oq = job.words[27] as usize;
        assert_eq!(f32::from_bits(sc[oq + 6]), -0.5);
        assert_eq!(job.words[31] as usize, oq + POCKET_WORDS);
        assert!(job.debug());
    }
}
