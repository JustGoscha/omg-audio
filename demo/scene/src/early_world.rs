//! C6c (GPU_PLAN.md Track C endgame): ONE listener context for the whole
//! world. A single chain cache, discovered from the listener over the
//! world mesh, is shared by every source — discovery is listener-launched
//! and therefore source-independent, so the doorway costs the same as the
//! open square: there is no routing, no virtual source, no aperture
//! re-radiation and no crossing blend anywhere in this path. A doorway is
//! a hole the solved legs thread for free; a wall crossing pays mass law;
//! a closing door leaf is an `extras` box whose transmission fades the
//! same records continuously (keys never change, so the renderer glides).

use omg_core::mesh::{Mesh, SegHit};
use omg_core::pt::Aabb;
use omg_core::pt_mesh::{mesh_chains, mesh_record, mesh_vertices, MChain, MeshRecord, SurfaceTable};
use omg_core::vec3::Vec3;
use std::collections::HashMap;
use std::sync::Mutex;

/// A world-discovery provider (the GPU seam, kernel K3): returns chains
/// found for THIS call synchronously (native wgpu), or chains injected
/// from an earlier async dispatch (the web driver). `false` = nothing
/// this tick AND don't run the CPU fan (a job is pending); the direct
/// path is always solved regardless, and a stalled driver falls back to
/// the CPU fan after a grace window.
pub trait WorldDiscovery: Send {
    fn discover(&mut self, listener: Vec3, rot: u32, out: &mut Vec<MChain>) -> bool;
}

static DISCOVERY: Mutex<Option<Box<dyn WorldDiscovery>>> = Mutex::new(None);

pub fn set_world_discovery(d: Box<dyn WorldDiscovery>) {
    *DISCOVERY.lock().unwrap() = Some(d);
}

pub fn clear_world_discovery() {
    *DISCOVERY.lock().unwrap() = None;
}

/// C7a: a batched (source × chain) solve provider (the K4 kernel). One
/// call per tick solves EVERY pair against the same chain list;
/// `out[si * chains.len() + ci]` holds the pair's record or None. A
/// `false` return (or missing sources beyond a device cap) falls back
/// to the per-source CPU solve — physics never depends on the GPU.
pub trait WorldSolve: Send {
    fn solve_batch(
        &mut self,
        sources: &[(u16, Vec3)],
        chains: &[MChain],
        listener: Vec3,
        extras: &[Aabb],
        out: &mut Vec<Option<MeshRecord>>,
    ) -> bool;
}

static SOLVER: Mutex<Option<Box<dyn WorldSolve>>> = Mutex::new(None);

pub fn set_world_solver(s: Box<dyn WorldSolve>) {
    *SOLVER.lock().unwrap() = Some(s);
}

pub fn clear_world_solver() {
    *SOLVER.lock().unwrap() = None;
}

/// Ticks a pending async provider may stay silent before the CPU fan
/// backstops it (chains TTL over 200 ticks; this is far inside that).
const PROVIDER_GRACE: u32 = 20;

/// Ticks a discovered chain stays cached without re-discovery. The pool
/// is shared across sources, so a chain any source validates stays warm.
const CHAIN_TTL: u32 = 200;

/// Listener discovery rays per tick — ONCE per tick for all sources
/// (the per-room engine paid its fan per source per room).
const RAYS_PER_TICK: u32 = 768;

/// Live-chain cap: solving is per source per chain, so this bounds the
/// whole world's early workload. Discovery refresh keeps the set biased
/// toward what the listener can currently see.
const MAX_CHAINS: usize = 384;

/// Debug polyline budget (floats), matching the web bridge buffer.
const DEBUG_CAP: usize = 6000;

pub struct WorldEarly {
    pub table: SurfaceTable,
    /// Overlay boxes for discovery (parallel to the appended faces).
    disc_boxes: Vec<(Vec3, Vec3)>,
    chains: HashMap<MChain, u32>,
    rot: u32,
    scratch: Vec<MChain>,
    seg_buf: Vec<SegHit>,
    verts_buf: Vec<Vec3>,
    /// Consecutive silent ticks from an async provider (grace counter).
    pending: u32,
    /// True when the last discovery came from a registered provider —
    /// telemetry for the UI's early cell.
    pub gpu_discovery: bool,
    /// C7a: this tick's batched solve results, keyed by source id —
    /// `solve_source` replays from here instead of solving on the CPU.
    batch: HashMap<u16, Vec<MeshRecord>>,
    batch_out: Vec<Option<MeshRecord>>,
    /// True when the current tick's records came from the batch solver.
    pub gpu_solve: bool,
    /// Debug-ray capture: enabled by the first consumer call, filled
    /// during solves as [src_idx, n_verts, xyz × n]… in world coords.
    pub debug_on: bool,
    pub debug_buf: Vec<f32>,
}

impl WorldEarly {
    /// `furn`: overlay boxes (furniture, world coords) whose faces
    /// become rect-bounded reflective surfaces — chains may bounce off
    /// a table top like off a wall; the solve validates against the
    /// face rect. The live furniture switch gates their use per tick.
    pub fn new(mesh: &Mesh, furn: &[(Vec3, Vec3, omg_core::material::Material)]) -> Self {
        let mut table = SurfaceTable::build(mesh);
        for (mn, mx, m) in furn {
            table.append_box(*mn, *mx, m);
        }
        Self {
            disc_boxes: furn.iter().map(|(mn, mx, _)| (*mn, *mx)).collect(),
            table,
            chains: HashMap::new(),
            rot: 0,
            scratch: Vec::new(),
            seg_buf: Vec::new(),
            verts_buf: Vec::new(),
            pending: 0,
            gpu_discovery: false,
            batch: HashMap::new(),
            batch_out: Vec::new(),
            gpu_solve: false,
            debug_on: false,
            debug_buf: Vec::new(),
        }
    }

    /// The chain list a batched solve (and the CPU replay order) uses:
    /// the direct path first, then the cached chains in sorted order,
    /// overlay-face chains gated by the furniture switch — exactly the
    /// order `solve_source`'s CPU path walks.
    fn chain_list(&self, out: &mut Vec<MChain>) {
        out.clear();
        out.push(([omg_core::pt_mesh::NO_SURF; omg_core::pt_mesh::M_MAX_ORDER], 0));
        let furn_ok = crate::quality::furniture_on();
        let mut cs: Vec<MChain> = self.chains.keys().copied().collect();
        cs.sort();
        for (chain, order) in cs {
            if !furn_ok
                && chain[..order as usize]
                    .iter()
                    .any(|&sid| sid >= self.table.base_overlay)
            {
                continue;
            }
            out.push((chain, order));
        }
    }

    /// C7a: one batched solve for the whole tick's source set. Fills
    /// the per-source replay cache; sources the provider didn't cover
    /// (no provider, over its cap, failed dispatch) simply miss the
    /// cache and take the CPU path in `solve_source`.
    pub fn batch_solve(&mut self, sources: &[(u16, Vec3)], listener: Vec3, extras: &[Aabb]) {
        self.batch.clear();
        self.gpu_solve = false;
        if sources.is_empty() {
            return;
        }
        let mut guard = SOLVER.lock().unwrap();
        let Some(solver) = guard.as_mut() else { return };
        let mut list = Vec::new();
        self.chain_list(&mut list);
        self.batch_out.clear();
        if !solver.solve_batch(sources, &list, listener, extras, &mut self.batch_out) {
            return;
        }
        let nch = list.len();
        if self.batch_out.len() < sources.len() * nch {
            return;
        }
        for (si, &(id, _)) in sources.iter().enumerate() {
            let recs: Vec<MeshRecord> = self.batch_out[si * nch..(si + 1) * nch]
                .iter()
                .flatten()
                .copied()
                .collect();
            self.batch.insert(id, recs);
        }
        self.gpu_solve = true;
    }

    /// Once per tick: run the listener fan (the registered GPU provider
    /// when present, the CPU fan otherwise), refresh the shared TTL table.
    pub fn begin_tick(&mut self, mesh: &Mesh, listener: Vec3) {
        // last tick's batched records must never leak into this tick
        // (sources move) — batch_solve refills after discovery
        self.batch.clear();
        self.gpu_solve = false;
        self.scratch.clear();
        let (provided, have_provider) = {
            let mut guard = DISCOVERY.lock().unwrap();
            match guard.as_mut() {
                Some(d) => (d.discover(listener, self.rot, &mut self.scratch), true),
                None => (false, false),
            }
        };
        if provided || !self.scratch.is_empty() {
            self.pending = 0;
            self.gpu_discovery = true;
        } else if !have_provider || {
            self.pending += 1;
            self.pending > PROVIDER_GRACE
        } {
            let boxes: &[(Vec3, Vec3)] = if crate::quality::furniture_on() {
                &self.disc_boxes
            } else {
                &[]
            };
            mesh_chains(
                mesh,
                boxes,
                self.table.base_overlay,
                listener,
                RAYS_PER_TICK,
                self.rot,
                &mut self.scratch,
            );
            self.gpu_discovery = false;
        }
        self.rot = self.rot.wrapping_add(1);
        for &c in &self.scratch {
            match self.chains.get_mut(&c) {
                Some(ttl) => *ttl = CHAIN_TTL,
                None => {
                    if self.chains.len() < MAX_CHAINS {
                        self.chains.insert(c, CHAIN_TTL);
                    } else if let Some((&old, _)) =
                        self.chains.iter().min_by_key(|(_, ttl)| **ttl)
                    {
                        // full: what the listener sees NOW beats the
                        // stalest memory — walking must never starve
                        // the fresh set behind five-steps-ago chains
                        self.chains.remove(&old);
                        self.chains.insert(c, CHAIN_TTL);
                    }
                }
            }
        }
        self.chains.retain(|_, ttl| {
            *ttl -= 1;
            *ttl > 0
        });
        self.debug_buf.clear();
    }

    /// Exact-solve the direct path and every cached chain for one source
    /// (world coordinates throughout). Records are deduplicated by
    /// geometry: coincident wall planes (two rooms sharing a boundary)
    /// make two chains solve to the same image, which would double the
    /// energy and flap the key.
    pub fn solve_source(
        &mut self,
        mesh: &Mesh,
        source: u16,
        src: Vec3,
        listener: Vec3,
        extras: &[Aabb],
        out: &mut Vec<MeshRecord>,
    ) {
        out.clear();
        let dup_of = |out: &[MeshRecord], r: &MeshRecord| {
            out.iter().any(|q| {
                (q.delay_s - r.delay_s).abs() < 1e-4
                    && q.dir[0] * r.dir[0] + q.dir[1] * r.dir[1] + q.dir[2] * r.dir[2] > 0.999
            })
        };
        if let Some(recs) = self.batch.get(&source) {
            // C7a replay: the batch already solved this source against
            // the same chain list in the same order — only the
            // geometric dedupe (chain records only) remains CPU-side.
            for r in recs {
                if r.order == 0 || !dup_of(out, r) {
                    out.push(*r);
                }
            }
        } else {
            if let Some(r) =
                mesh_record(mesh, &self.table, &[], source, src, listener, extras, &mut self.seg_buf)
            {
                out.push(r);
            }
            // deterministic order (HashMap iteration is not): sort chains so
            // the dedupe winner is stable across ticks
            self.scratch.clear();
            self.scratch.extend(self.chains.keys().copied());
            self.scratch.sort();
            let furn_ok = crate::quality::furniture_on();
            for &(chain, order) in &self.scratch {
                let c = &chain[..order as usize];
                // furniture switch off: overlay-face chains stay cached but
                // emit nothing (identical to how occluded seeds behave)
                if !furn_ok && c.iter().any(|&sid| sid >= self.table.base_overlay) {
                    continue;
                }
                let Some(r) =
                    mesh_record(mesh, &self.table, c, source, src, listener, extras, &mut self.seg_buf)
                else {
                    continue;
                };
                if !dup_of(out, &r) {
                    out.push(r);
                }
            }
        }
        if self.debug_on && self.debug_buf.len() < DEBUG_CAP {
            for r in out.iter().take(12) {
                let c = &r.chain[..r.order as usize];
                if mesh_vertices(&self.table, c, src, listener, &mut self.verts_buf) {
                    let n = self.verts_buf.len();
                    if self.debug_buf.len() + 2 + n * 3 > DEBUG_CAP {
                        break;
                    }
                    self.debug_buf.push((source / 8) as f32);
                    self.debug_buf.push(n as f32);
                    for v in &self.verts_buf {
                        self.debug_buf.extend_from_slice(&[v.x, v.y, v.z]);
                    }
                }
            }
        }
    }
}
