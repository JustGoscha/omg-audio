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
//!   35 v4  36 a live walk's pass word (`slice_word`; 0 for a whole
//!   walk)  37 state_off  38..40 reserved (0)
//! scene: sources (SRC_WORDS = 12 each): pos xyz, weight, t_free, id,
//!          direct visibility xyz, fixed-point norm, 0, 0
//!        panels (PANEL_WORDS = 16):  min xyz, scattering, max xyz, 0,
//!          absorption xyz, 0, transmission xyz, 0
//!        pockets (POCKET_WORDS = 8): min xyz, 0, max xyz, 0
//!        edges: n_edges bits (32 a word) over the walker's own copy of
//!          the edge set (`EdgeSet::flat`, uploaded with the mesh): set =
//!          filled by a panel or on the ground this tick (no bend there);
//!          n_edges 0 = no diffraction
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
pub const WALK_LAYOUT_VERSION: u32 = 8;
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
/// Per-dispatch caps. A world beyond them traces on the CPU instead — a
/// silently clamped input would change the physics.
pub const MAX_SOURCES: usize = 64;
/// Panels: door leaves and panes plus every large piece of furniture (the
/// demo town carries ~90).
pub const MAX_PANELS: usize = 256;
pub const MAX_POCKETS: usize = 8;
pub const MAX_EDGES: usize = 65536;
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
    MAX_SOURCES * SRC_WORDS + MAX_PANELS * PANEL_WORDS + MAX_POCKETS * POCKET_WORDS + MAX_EDGES / 32;

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
        if let Some(es) = edges.filter(|es| !es.edges.is_empty()) {
            n_edges = es.edges.len() as u32;
            let mut mask = vec![0u32; es.edges.len().div_ceil(32)];
            for (ei, _) in es.filled.iter().enumerate().filter(|(_, f)| **f) {
                mask[ei / 32] |= 1 << (ei % 32);
            }
            w.extend_from_slice(&mask);
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
    /// Dispatches a sliced walk takes at `steps` segments a ray each: the
    /// early rays' longest walk (a sliced walk's early rays don't split:
    /// v3_walk.wgsl).
    pub fn passes(&self, steps: u32) -> u32 {
        if steps == 0 {
            return 1;
        }
        (self.words[11] + 1 + super::MAX_BENDS).div_ceil(steps).max(1)
    }

    /// Plan this job as a live (sliced) walk: the early rays
    /// `SLICE_STEPS` segments per dispatch, their states kept between the
    /// walk's dispatches (none while debug captures them whole: pass 0
    /// walks them); late rays, if the job carries any, walk whole in pass
    /// 0 — a live walker leaves them to the CPU (`V3Walker::walks_late`).
    /// `max_passes` caps the dispatches (0: none).
    pub fn live(&self, max_passes: u32) -> LiveWalk {
        let n_early = self.n_early();
        // the states after the debug records' room (kept for every early
        // ray, so toggling debug doesn't move them)
        let state_off = n_early * DEBUG_WORDS as u32;
        let states = if self.debug() { 0 } else { n_early as usize };
        let mut header = self.header().to_vec();
        header[37] = state_off;
        let passes = self.passes(SLICE_STEPS);
        LiveWalk {
            header,
            passes: if max_passes > 0 { passes.min(max_passes) } else { passes },
            state_words: state_off as usize + states * STATE_WORDS,
        }
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

/// Segments per early ray per dispatch in a live (sliced) walk; they
/// walk whole within the walk's passes.
pub const SLICE_STEPS: u32 = 2;

/// Header word 36 of a live walk's dispatch `pass`.
pub fn slice_word(pass: u32) -> u32 {
    SLICE_STEPS | (pass << 16)
}

/// One live walk, planned (`WalkJob::live`): the header (word 36 per pass
/// from `slice_word`), its dispatches, and the words the kernel's state
/// buffer (binding 7) needs.
#[derive(Clone, Debug)]
pub struct LiveWalk {
    pub header: Vec<u32>,
    pub passes: u32,
    pub state_words: usize,
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
    /// exactly as in `trace`. `cpu_late`: the walker leaves the late family
    /// to the CPU — the job carries none, and with each fresh walk that
    /// many late rays are walked here, whole, from where the listener is
    /// now (`V3Walker::walks_late`). Returns None when the walker refused
    /// (or the world exceeds the flat caps) — the caller traces on the
    /// CPU instead.
    pub fn trace_walked(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        cpu_late: Option<u32>,
        walk: &mut dyn FnMut(&WalkJob) -> Walk,
    ) -> Option<Vec<SourceField>> {
        if sources.len() > MAX_SOURCES || world.panels.len() > MAX_PANELS || world.pockets.len() > MAX_POCKETS {
            return None;
        }
        let mut dbg: Vec<(u32, f32, Vec<Vec3>)> = Vec::new();
        let (direct, vis, edge_taps) = self.prepare(world, listener, sources, budget);
        let rot = random_rotation(&mut self.rng);
        let seed = (self.rng.next_u64() >> 32) as u32;
        let edges = if budget.diffraction { self.edges.as_ref() } else { None };
        let job_budget = if cpu_late.is_some() { Budget { late_rays: 0, ..budget } } else { budget };
        // the GPU walk connects to where each source emitted its direct
        // sound (a moving source's echoes on the CPU reach further back)
        let at_emission: Vec<Source> =
            sources.iter().enumerate().map(|(si, s)| Source { pos: self.direct_from(si, s, listener).unwrap_or(s.pos), ..*s }).collect();
        let mut job = WalkJob::build(world, listener, &at_emission, job_budget, rot, seed, self.debug, &vis, edges).v4(self.v4);
        job.epoch = self.epoch;
        let walked = match walk(&job) {
            Walk::Refused => return None,
            Walk::Pending => None,
            Walk::Fresh(w) => Some(w),
        };
        let fresh = walked.as_ref().is_some_and(|w| w.epoch == self.epoch);
        // the mirror chains the kernel leaves to the CPU (named by a few
        // mirror rays here, solved exactly), folded in with a fresh walk
        let chains = walked.as_ref().map(|_| self.walked_chains(world, listener, sources, budget));
        // and the late family, when the walker leaves it to the CPU
        let late = match cpu_late {
            Some(n) if n > 0 && fresh => {
                let mut accs: Vec<Acc> = at_emission
                    .iter()
                    .map(|s| Acc::new((s.pos - listener).length() / SPEED_OF_SOUND, self.v4))
                    .collect();
                let late_budget = Budget { late_rays: n, ..budget };
                let sky = self.walk_rays(world, listener, sources, late_budget, &vis, false, &mut accs, &mut dbg);
                Some((accs, sky))
            }
            _ => None,
        };
        Some(self.fold_walked(listener, sources, budget, direct, edge_taps, dbg, walked, chains, late))
    }

    /// Fold a walk's binned contributions (or none, while one is in
    /// flight) — with the CPU-solved chains and the CPU's late family, if
    /// any, on top — into the smoothed state and build every source's
    /// field.
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
        late: Option<(Vec<Acc>, Vec<([f32; NBANDS], Vec3)>)>,
    ) -> Vec<SourceField> {
        // a walk launched before the listener crossed a surface describes
        // the other side: drop it, the next one is from here
        let fresh = walked
            .as_ref()
            .is_some_and(|w| w.epoch == self.epoch && w.data.len() >= output_words(w.ids.len()));
        let (late_accs, late_sky) = match late {
            Some((a, sky)) => (a, Some(sky)),
            None => (Vec::new(), None),
        };
        // per source: what goes on top of the walk's own bins
        let mut extra: Vec<Vec<Acc>> = sources.iter().map(|_| Vec::new()).collect();
        for list in [chains.unwrap_or_default(), late_accs] {
            for (x, acc) in extra.iter_mut().zip(list) {
                x.push(acc);
            }
        }
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
                    let (mut b, mut d) = ([e[0], e[1], e[2]], Vec3::new(e[3], e[4], e[5]));
                    if let Some(l) = late_sky.as_ref().and_then(|l| l.get(ci)) {
                        for k in 0..NBANDS {
                            b[k] += l.0[k];
                        }
                        d = d + l.1;
                    }
                    (b, d)
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
        let shared = self.shared_ticks();
        for (si, s) in sources.iter().enumerate() {
            let t_free = (self.direct_from(si, s, listener).unwrap_or(s.pos) - listener).length() / SPEED_OF_SOUND;
            let block = walked.as_ref().filter(|_| fresh).and_then(|w| {
                w.ids.iter().position(|&id| id == s.id).map(|k| {
                    let o = SKY_WORDS + k * SOURCE_WORDS;
                    &w.data[o..o + SOURCE_WORDS]
                })
            });
            let st = self.state.entry(s.id).or_default();
            let mut total = [0.0f32; NBANDS];
            if let Some(blk) = block {
                let mut acc = if self.v4 { walked_dense(blk, t_free) } else { walked_early(blk, t_free) };
                for x in extra[si].drain(..) {
                    acc.merge(x);
                }
                total = acc.total;
                if self.v4 {
                    st.fold_v4(acc, budget, listener, s.pos);
                } else {
                    st.fold(acc.early, acc.echo, budget, self.v6);
                }
            }
            out.push(if self.v4 {
                st.field_v4(t_free, direct[si], &mut edge_taps[si], total, shared)
            } else {
                st.field(&self.cells, t_free, direct[si], &mut edge_taps[si], total)
            });
        }
        self.note_heard(sources, &out);
        out
    }
}

/// A v4 walk's source block as `deposit` would have built it: the dense
/// response, and the echogram (the RT60 readout) summed from it.
fn walked_dense(blk: &[f32], t_free: f32) -> Acc {
    let mut acc = Acc::new(t_free, true);
    let mut last = 0;
    for bin in 0..DENSE_BINS {
        let o = bin * DENSE_SLOT;
        if blk[o..o + DENSE_SLOT].iter().any(|&x| x != 0.0) {
            last = bin + 1;
        }
    }
    acc.dense = (0..last).map(|bin| core::array::from_fn(|c| blk[bin * DENSE_SLOT + c])).collect();
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
    acc
}

/// A v3 walk's source block as `deposit` would have built it: the early
/// bins, the echogram, the totals.
fn walked_early(blk: &[f32], t_free: f32) -> Acc {
    let mut acc = Acc::new(t_free, false);
    for cell in 0..CELLS {
        for db in 0..DELAY_BINS {
            let o = (cell * DELAY_BINS + db) * SLOT;
            let e = [blk[o], blk[o + 1], blk[o + 2]];
            if !e.iter().any(|&x| x > 0.0) {
                continue;
            }
            let key = KEY_EARLY_BASE + cell as u32 * 256 + db as u32;
            acc.early.insert(
                key,
                Bin {
                    e,
                    // (unused downstream: the tap delay is pinned to the
                    // bin; kept for the EMA's symmetry)
                    ed: e[1] * (t_free + (db as f32 + 0.5) * DELAY_BIN_S),
                    dir: Vec3::new(blk[o + 3], blk[o + 4], blk[o + 5]),
                    last: 0.0,
                },
            );
        }
    }
    for bin in 0..ECHO_BINS.min(acc.echo.bins.len()) {
        let o = EARLY_WORDS + bin * SLOT;
        for b in 0..NBANDS {
            acc.echo.bins[bin][b] = blk[o + b] * LEGACY_UNITS;
        }
        for k in 0..3 {
            acc.echo.dirs[bin][k] = blk[o + 3 + k] * LEGACY_UNITS;
        }
    }
    let o = EARLY_WORDS + ECHO_WORDS;
    acc.total = [blk[o], blk[o + 1], blk[o + 2]];
    acc
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
            let f = tracer.trace_walked(&world, near, &src, budget, None, &mut walk(false)).unwrap();
            assert!(early(&f[0]) > 0, "a fresh walk on this side lands");
        }
        // fly through the wall: this tick's answer is the old side's job
        let far = Vec3::new(-3.0, 0.0, 1.6);
        let f = tracer.trace_walked(&world, far, &src, budget, None, &mut walk(true)).unwrap();
        assert_eq!(early(&f[0]), 0, "the walk from the other side of the wall was folded in");
        let f = tracer.trace_walked(&world, far, &src, budget, None, &mut walk(false)).unwrap();
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
