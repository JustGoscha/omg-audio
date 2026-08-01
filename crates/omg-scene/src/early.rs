//! PT-early phases C2–C4 (GPU_PLAN.md Track C): the per-(source, room)
//! path cache between discovery and the tap stream, plus the seam that
//! lets a GPU (native wgpu or the web's JS driver) provide discovery.
//!
//! Chains are the durable thing — a chain's identity survives motion
//! while its geometry glides — so the cache stores chains with a TTL
//! and re-solves every cached chain EXACTLY each tick (a solve is a
//! handful of mirrors: cheaper than one traced ray). Discovery only
//! has to find a chain once per TTL window. The direct path and every
//! order ≤2 chain are permanent seeds (37 solves — validation prunes
//! them per tick anyway); discovery earns the order-3 tail, and from
//! C5 on, the occluder-face chains. A chain whose exact solve stops
//! validating drops immediately and the tap fades through the
//! renderer's normal slot release.

use omg_core::params::Tap;
use omg_core::pt::{chain_vertices, pt_chains, record_for_occ, seed_chains, Aabb, Chain, PT_MAX_ORDER};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use std::sync::Mutex;

/// Tap-key namespace for PT paths: far above ISM indices (≤ ~63),
/// multi-rig offsets (≤ 64×8) and the portal straight tap (9000).
pub const PT_KEY_BASE: u32 = 100_000;

/// Ticks a discovered chain stays cached without being re-discovered.
const CHAIN_TTL: u32 = 200;

/// CPU discovery rays per tick (the GPU kernel runs 4096 for the same
/// cost class; both only have to find each chain once per TTL).
const RAYS_PER_TICK: u32 = 512;

/// A discovery provider (the GPU seam). `discover` may return chains
/// found for THIS call synchronously (native wgpu) or chains injected
/// from an earlier asynchronous dispatch (the web driver) — the cache
/// merges either into its TTL table. Return false to signal "nothing
/// this tick, and don't run CPU discovery either" (a pending async
/// job); returning true with no chains is a valid empty result.
pub trait EarlyDiscovery: Send {
    fn discover(
        &mut self,
        id: u32,
        room: &Shoebox,
        listener: Vec3,
        rot: u32,
        out: &mut Vec<Chain>,
    ) -> bool;
}

static DISCOVERY: Mutex<Option<Box<dyn EarlyDiscovery>>> = Mutex::new(None);

pub fn set_early_discovery(d: Box<dyn EarlyDiscovery>) {
    *DISCOVERY.lock().unwrap() = Some(d);
}

pub fn clear_early_discovery() {
    *DISCOVERY.lock().unwrap() = None;
}

struct CachedChain {
    chain: [u8; PT_MAX_ORDER],
    order: u8,
    ttl: u32,
    permanent: bool,
}

pub struct PathCache {
    id: u32,
    chains: Vec<CachedChain>,
    rot: u32,
    scratch: Vec<Chain>,
}

impl PathCache {
    pub fn new(id: u32) -> Self {
        let mut seeds = Vec::new();
        seed_chains(&mut seeds);
        Self {
            id,
            chains: seeds
                .into_iter()
                .map(|(chain, order)| CachedChain { chain, order, ttl: CHAIN_TTL, permanent: true })
                .collect(),
            rot: 0,
            scratch: Vec::new(),
        }
    }

    /// One tick: discover (GPU provider or the CPU fan), refresh the
    /// TTL table, exact-solve every live chain into `out` as taps.
    pub fn update(
        &mut self,
        room: &Shoebox,
        src: Vec3,
        listener: Vec3,
        occluders: &[Aabb],
        out: &mut Vec<Tap>,
    ) -> usize {
        self.scratch.clear();
        let provided = {
            let mut guard = DISCOVERY.lock().unwrap();
            match guard.as_mut() {
                Some(d) => d.discover(self.id, room, listener, self.rot, &mut self.scratch),
                None => false,
            }
        };
        if !provided && self.scratch.is_empty() && DISCOVERY.lock().unwrap().is_none() {
            pt_chains(room, listener, RAYS_PER_TICK, self.rot, &mut self.scratch);
        }
        self.rot = self.rot.wrapping_add(1);

        for &(chain, order) in &self.scratch {
            match self
                .chains
                .iter_mut()
                .find(|c| c.chain == chain && c.order == order)
            {
                Some(c) => c.ttl = CHAIN_TTL,
                None => self.chains.push(CachedChain { chain, order, ttl: CHAIN_TTL, permanent: false }),
            }
        }

        out.clear();
        self.chains.retain_mut(|c| {
            if !c.permanent {
                c.ttl = c.ttl.saturating_sub(1);
                if c.ttl == 0 {
                    return false;
                }
            }
            let chain = &c.chain[..c.order as usize];
            // exact solve, fresh every tick: geometry glides, key holds
            match record_for_occ(room, chain, 0, src, listener, occluders) {
                Some(r) => {
                    out.push(Tap {
                        key: PT_KEY_BASE + r.key(),
                        delay_s: r.delay_s,
                        dir: r.dir,
                        gains: r.gains,
                    });
                    true
                }
                // stopped validating (an occluder moved in, C5): keep
                // permanent seeds cached but emit nothing for them
                None => c.permanent,
            }
        });
        out.len()
    }

    /// Debug: append every live chain's polyline as
    /// [n_verts, x,y,z × n] (room-local). Caps at `max_paths`.
    pub fn debug_rays(
        &self,
        room: &Shoebox,
        src: Vec3,
        listener: Vec3,
        max_paths: usize,
        out: &mut Vec<f32>,
    ) {
        let mut verts = Vec::new();
        for c in self.chains.iter().take(max_paths) {
            let chain = &c.chain[..c.order as usize];
            if chain_vertices(room, chain, src, listener, &mut verts) {
                out.push(verts.len() as f32);
                for v in &verts {
                    out.extend_from_slice(&[v.x, v.y, v.z]);
                }
            }
        }
    }
}
