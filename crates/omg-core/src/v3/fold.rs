//! Engine v3, walked elsewhere (ENGINE_V3.md, V3.3).
//!
//! The heavy half of a v3 tick — the ray walk of both families: raycasts,
//! next-event shadow segments, source-sphere tests, transmission sampling,
//! sky escapes — can run outside this crate, in the WGSL kernel
//! `crates/omg-gpu/shaders/v3_walk.wgsl` (natively through wgpu, in the
//! browser through `web/gpu.js`). The walker hands back the walk's
//! contributions ALREADY BINNED exactly as `deposit` bins them (direction
//! cell × 2 ms early bins, 10 ms echogram bins relative to the free-field
//! arrival, per-band totals, sky cells); this module folds them into the
//! same per-source EMA state and builds the same `SourceField`s as
//! `Tracer::trace`. The direct path never leaves the CPU: one exact
//! segment per source, noise-free, recomputed every tick even while an
//! asynchronous walk is still in flight.
//!
//! Everything here is additive: `Tracer::trace` (the CPU walk) stays the
//! oracle and the fallback. The fold below mirrors trace()'s step 5 line
//! for line — keep the two in step when either changes.
//!
//! Flat formats (the contract with the kernel; bump `WALK_LAYOUT_VERSION`
//! on ANY change, here and in v3_walk.wgsl):
//!
//! JOB — u32 words (f32 payloads as bits), header then three tables:
//! ```text
//! header (HEADER_WORDS = 28, the kernel's 112-byte uniform):
//!   0..3 rot row 0   3 n_early     4..7 rot row 1   7 n_late
//!   8..11 rot row 2  11 bounces    12..15 listener  15 seed
//!   16 k  17 nee  18 late_nee  19 n_sources
//!   20 n_panels  21 n_pockets  22 has_grade  23 grade
//!   24 debug  25..28 reserved (0)
//! sources (SRC_WORDS = 8 each):   pos xyz, weight, t_free, id, 0, 0
//! panels (PANEL_WORDS = 16 each): min xyz, scattering, max xyz, 0,
//!                                 absorption xyz, 0, transmission xyz, 0
//! pockets (POCKET_WORDS = 8):     min xyz, 0, max xyz, 0
//! ```
//! OUTPUT — f32 words, true energy units (the fold applies the v2
//! echogram scale): SKY_WORDS of sky cells, then one SOURCE_WORDS block
//! per job source in job order. A slot is (e₀, e₁, e₂, Σe₁·dx, Σe₁·dy,
//! Σe₁·dz) with d the ray's first direction. A source block holds the
//! early slots [cell·DELAY_BINS + delay bin], then the echogram slots,
//! then the per-band total (+1 pad word).
//!
//! DEBUG — DEBUG_WORDS per early ray: mid energy, source index, vertex
//! count, 0, then up to DEBUG_VERTS vertices (ears → bounces → source).

use super::*;

/// Bump on ANY change to the flat formats above or in v3_walk.wgsl.
pub const WALK_LAYOUT_VERSION: u32 = 1;
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
pub const SKY_WORDS: usize = CELLS * SLOT;
pub const HEADER_WORDS: usize = 28;
pub const SRC_WORDS: usize = 8;
pub const PANEL_WORDS: usize = 16;
pub const POCKET_WORDS: usize = 8;
/// Per-dispatch caps. A world beyond them traces on the CPU instead — a
/// silently clamped input would change the physics.
pub const MAX_SOURCES: usize = 64;
pub const MAX_PANELS: usize = 64;
pub const MAX_POCKETS: usize = 8;
pub const DEBUG_VERTS: usize = 8;
pub const DEBUG_WORDS: usize = 4 + DEBUG_VERTS * 3;
/// Debug paths a walker needs to return at most (the view's cap).
pub const DEBUG_KEEP: usize = DEBUG_PATHS;

/// Output words for a job with `n_sources` sources.
pub fn output_words(n_sources: usize) -> usize {
    SKY_WORDS + n_sources * SOURCE_WORDS
}

/// One tick's walk, flattened for an external walker.
#[derive(Clone, Debug)]
pub struct WalkJob {
    pub words: Vec<u32>,
    /// Source ids in job order (the output blocks' order).
    pub ids: Vec<u32>,
}

impl WalkJob {
    fn build(
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        rot: [[f32; 3]; 3],
        seed: u32,
        debug: bool,
    ) -> Self {
        let f = |x: f32| x.to_bits();
        let n_early = budget.rays.max(1);
        let mut w = vec![0u32; HEADER_WORDS];
        for r in 0..3 {
            for c in 0..3 {
                w[r * 4 + c] = f(rot[r][c]);
            }
        }
        w[3] = n_early;
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
        for s in sources {
            let t_free = (s.pos - listener).length() / SPEED_OF_SOUND;
            w.extend_from_slice(&[f(s.pos.x), f(s.pos.y), f(s.pos.z), f(s.weight), f(t_free), s.id, 0, 0]);
        }
        for (mn, mx, m) in world.panels {
            w.extend_from_slice(&[f(mn.x), f(mn.y), f(mn.z), f(m.scattering)]);
            w.extend_from_slice(&[f(mx.x), f(mx.y), f(mx.z), 0]);
            w.extend_from_slice(&[f(m.absorption[0]), f(m.absorption[1]), f(m.absorption[2]), 0]);
            w.extend_from_slice(&[f(m.transmission[0]), f(m.transmission[1]), f(m.transmission[2]), 0]);
        }
        for (mn, mx) in world.pockets {
            w.extend_from_slice(&[f(mn.x), f(mn.y), f(mn.z), 0, f(mx.x), f(mx.y), f(mx.z), 0]);
        }
        Self { words: w, ids: sources.iter().map(|s| s.id).collect() }
    }

    pub fn header(&self) -> &[u32] {
        &self.words[..HEADER_WORDS]
    }
    pub fn n_early(&self) -> u32 {
        self.words[3]
    }
    pub fn n_late(&self) -> u32 {
        self.words[7]
    }
    /// Threads to launch: one per ray of either family.
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
    pub fn debug(&self) -> bool {
        self.words[24] != 0
    }
    pub fn sources(&self) -> &[u32] {
        let o = HEADER_WORDS;
        &self.words[o..o + self.n_sources() * SRC_WORDS]
    }
    pub fn panels(&self) -> &[u32] {
        let o = HEADER_WORDS + self.n_sources() * SRC_WORDS;
        &self.words[o..o + self.n_panels() * PANEL_WORDS]
    }
    pub fn pockets(&self) -> &[u32] {
        let o = HEADER_WORDS + self.n_sources() * SRC_WORDS + self.n_panels() * PANEL_WORDS;
        &self.words[o..o + self.n_pockets() * POCKET_WORDS]
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
    /// Nothing finished yet: hold the smoothed state, refresh the direct.
    Pending,
    /// Can't take this job (no device, caps): trace on the CPU instead.
    Refused,
}

impl Tracer {
    /// `trace`, with the ray walk done by `walk` (the GPU kernel). The
    /// direct paths, the EMA and the field construction run here exactly
    /// as in `trace`. Returns None when the walker refused (or the world
    /// exceeds the flat caps) — the caller traces on the CPU instead.
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
        let rot = random_rotation(&mut self.rng);
        let seed = (self.rng.next_u64() >> 32) as u32;
        let job = WalkJob::build(world, listener, sources, budget, rot, seed, self.debug);
        let walked = match walk(&job) {
            Walk::Refused => return None,
            Walk::Pending => None,
            Walk::Fresh(w) => Some(w),
        };
        Some(self.fold_walked(world, listener, sources, budget, walked))
    }

    /// Fold a walk's binned contributions (or none, while one is in
    /// flight) into the smoothed state and build every source's field.
    pub fn fold_walked(
        &mut self,
        world: &World,
        listener: Vec3,
        sources: &[Source],
        budget: Budget,
        walked: Option<Walked>,
    ) -> Vec<SourceField> {
        let debug = self.debug;
        let mut dbg: Vec<(u32, f32, Vec<Vec3>)> = Vec::new();

        // 1. direct: one exact segment per source (as in `trace`)
        let mut direct: Vec<Option<Tap>> = Vec::with_capacity(sources.len());
        for s in sources {
            let to = s.pos - listener;
            let d = to.length().max(MIN_DIST);
            let t = world.transmission(listener, s.pos, &mut self.hits);
            let air = air_attenuation(d);
            let gains: [f32; NBANDS] = core::array::from_fn(|b| t[b] * air[b] / d);
            if debug && gains[1] > 1e-7 {
                keep_path(&mut dbg, s.id, gains[1] * gains[1], &[listener], s.pos);
            }
            direct.push(if gains.iter().any(|&g| g > 1e-7) {
                let dir = to * (1.0 / to.length().max(1e-6));
                Some(Tap { key: KEY_DIRECT, delay_s: d / SPEED_OF_SOUND, dir: [dir.x, dir.y, dir.z], gains })
            } else {
                None
            });
        }

        let fresh = walked.as_ref().is_some_and(|w| w.data.len() >= output_words(w.ids.len()));
        if fresh {
            let w = walked.as_ref().unwrap();
            if debug {
                for (id, e, v) in &w.debug {
                    let (end, body) = v.split_last().unwrap();
                    keep_path(&mut dbg, *id, *e, body, *end);
                }
                dbg.sort_by(|a, b| b.1.total_cmp(&a.1));
                self.debug_paths = dbg;
            }
            // the sky field, averaged like the CPU walk's
            let a = if self.sky_primed { budget.ema.clamp(0.0, 1.0) } else { 1.0 };
            self.sky_primed = true;
            for (ci, cur) in self.sky.iter_mut().enumerate() {
                let o = ci * SLOT;
                let e = &w.data[o..o + SLOT];
                for b in 0..NBANDS {
                    cur.0[b] += a * (e[b] - cur.0[b]);
                }
                cur.1 = cur.1 + (Vec3::new(e[3], e[4], e[5]) - cur.1) * a;
            }
        } else if debug {
            dbg.sort_by(|a, b| b.1.total_cmp(&a.1));
            self.debug_paths = dbg;
        }

        // 5. smooth over ticks and build each source's field (mirrors
        // `trace` step 5; a source this walk didn't carry keeps its state)
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
                                ed: e[1] * (t_free + (db as f32 + 0.5) * DELAY_BIN_S),
                                dir: Vec3::new(blk[o + 3], blk[o + 4], blk[o + 5]),
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

                let a = budget.ema.clamp(0.0, 1.0);
                let alpha = if st.echo.is_none() { 1.0 } else { a };
                let mut keys: Vec<u32> = st.early.keys().copied().collect();
                for k in early.keys() {
                    if !st.early.contains_key(k) {
                        keys.push(*k);
                    }
                }
                for k in keys {
                    let f = early.get(&k).copied().unwrap_or_default();
                    let cur = st.early.entry(k).or_default();
                    for b in 0..NBANDS {
                        cur.e[b] += alpha * (f.e[b] - cur.e[b]);
                    }
                    cur.ed += alpha * (f.ed - cur.ed);
                    cur.dir = cur.dir + (f.dir - cur.dir) * alpha;
                }
                st.early.retain(|_, b| b.e.iter().any(|&e| e > 1e-12));
                match &mut st.echo {
                    Some(e) => e.ema(&echo, budget.late_ema.clamp(0.0, 1.0)),
                    None => st.echo = Some(echo),
                }
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
                let dbin = (k - KEY_EARLY_BASE) % 256;
                taps.push(Tap {
                    key: *k,
                    delay_s: t_free + (dbin as f32 + 0.5) * DELAY_BIN_S,
                    dir: [dir.x, dir.y, dir.z],
                    gains: core::array::from_fn(|band| b.e[band].max(0.0).sqrt()),
                });
            }
            let empty;
            let echo = match st.echo.as_ref() {
                Some(e) => e,
                None => {
                    empty = Echogram::new();
                    &empty
                }
            };
            let reverb = estimate_reverb(echo);
            let (late_dir, aniso) = echo.late_direction(EARLY_S);
            out.push(SourceField { taps, reverb, late_dir, aniso, reflected: total });
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

    #[test]
    fn job_tables_round_trip() {
        let mesh = crate::mesh::MeshBuilder::new().build();
        let panels = [(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0), Material::WOOD_PANEL)];
        let pockets = [(Vec3::new(-1.0, -1.0, -3.0), Vec3::new(1.0, 1.0, -0.5))];
        let world = World { grade: Some(0.0), pockets: &pockets, ..World::new(&mesh, &panels) };
        let srcs = [Source { id: 7, pos: Vec3::new(3.0, 4.0, 0.0), weight: 2.0 }];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = Budget { rays: 100, late_rays: 10, bounces: 3, ..Budget::default() };
        let job = WalkJob::build(&world, Vec3::new(0.0, 0.0, 0.0), &srcs, b, rot, 9, true);
        assert_eq!(job.n_rays(), 110);
        assert_eq!((job.n_sources(), job.n_panels(), job.n_pockets()), (1, 1, 1));
        assert_eq!(f32::from_bits(job.sources()[4]), 5.0 / SPEED_OF_SOUND);
        assert_eq!(job.sources()[5], 7);
        assert_eq!(f32::from_bits(job.panels()[3]), Material::WOOD_PANEL.scattering);
        assert_eq!(f32::from_bits(job.pockets()[6]), -0.5);
        assert!(job.debug());
    }
}
