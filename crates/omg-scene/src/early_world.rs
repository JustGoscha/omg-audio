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

/// Ticks a discovered chain stays cached without re-discovery. The pool
/// is shared across sources, so a chain any source validates stays warm.
const CHAIN_TTL: u32 = 200;

/// Listener discovery rays per tick — ONCE per tick for all sources
/// (the per-room engine paid its fan per source per room).
const RAYS_PER_TICK: u32 = 768;

/// Live-chain cap: solving is per source per chain, so this bounds the
/// whole world's early workload. Discovery refresh keeps the set biased
/// toward what the listener can currently see.
const MAX_CHAINS: usize = 320;

/// Debug polyline budget (floats), matching the web bridge buffer.
const DEBUG_CAP: usize = 6000;

pub struct WorldEarly {
    pub table: SurfaceTable,
    chains: HashMap<MChain, u32>,
    rot: u32,
    scratch: Vec<MChain>,
    seg_buf: Vec<SegHit>,
    verts_buf: Vec<Vec3>,
    /// Debug-ray capture: enabled by the first consumer call, filled
    /// during solves as [src_idx, n_verts, xyz × n]… in world coords.
    pub debug_on: bool,
    pub debug_buf: Vec<f32>,
}

impl WorldEarly {
    pub fn new(mesh: &Mesh) -> Self {
        Self {
            table: SurfaceTable::build(mesh),
            chains: HashMap::new(),
            rot: 0,
            scratch: Vec::new(),
            seg_buf: Vec::new(),
            verts_buf: Vec::new(),
            debug_on: false,
            debug_buf: Vec::new(),
        }
    }

    /// Once per tick: run the listener fan, refresh the shared TTL table.
    pub fn begin_tick(&mut self, mesh: &Mesh, listener: Vec3) {
        self.scratch.clear();
        mesh_chains(mesh, listener, RAYS_PER_TICK, self.rot, &mut self.scratch);
        self.rot = self.rot.wrapping_add(1);
        for &c in &self.scratch {
            match self.chains.get_mut(&c) {
                Some(ttl) => *ttl = CHAIN_TTL,
                None => {
                    if self.chains.len() < MAX_CHAINS {
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
        if let Some(r) = mesh_record(mesh, &self.table, &[], source, src, listener, extras, &mut self.seg_buf) {
            out.push(r);
        }
        // deterministic order (HashMap iteration is not): sort chains so
        // the dedupe winner is stable across ticks
        self.scratch.clear();
        self.scratch.extend(self.chains.keys().copied());
        self.scratch.sort();
        for &(chain, order) in &self.scratch {
            let c = &chain[..order as usize];
            let Some(r) = mesh_record(mesh, &self.table, c, source, src, listener, extras, &mut self.seg_buf)
            else {
                continue;
            };
            let dup = out.iter().any(|q| {
                (q.delay_s - r.delay_s).abs() < 1e-4
                    && q.dir[0] * r.dir[0] + q.dir[1] * r.dir[1] + q.dir[2] * r.dir[2] > 0.999
            });
            if !dup {
                out.push(r);
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
