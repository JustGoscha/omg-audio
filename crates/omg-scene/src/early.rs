//! PT-early phase C2 (GPU_PLAN.md Track C): the per-(source, room)
//! path cache between discovery and the tap stream.
//!
//! Chains are the durable thing — a chain's identity survives motion
//! while its geometry glides — so the cache stores chains with a TTL
//! and re-solves every cached chain EXACTLY each tick (a solve is a
//! handful of mirrors: cheaper than one traced ray). Discovery only
//! has to find a chain once per TTL window; the rotating fan
//! accumulates coverage across ticks. A chain whose exact solve stops
//! validating (C5: an occluder moved into a segment) drops immediately
//! and the tap fades through the renderer's normal slot release.

use omg_core::params::Tap;
use omg_core::pt::{pt_discover, record_for, PathRecord, NO_WALL, PT_MAX_ORDER};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;

/// Tap-key namespace for PT paths: far above ISM indices (≤ ~63),
/// multi-rig offsets (≤ 64×8) and the portal straight tap (9000).
pub const PT_KEY_BASE: u32 = 100_000;

/// Ticks a discovered chain stays cached without being re-discovered.
/// In an empty room chains never invalidate, so this only bounds how
/// long a C5-occluded-then-freed path may take to reappear.
const CHAIN_TTL: u32 = 200;

/// Discovery rays per tick. Order ≤2 is seeded exhaustively inside
/// pt_discover; rays only earn the order-3 tail, so a small rotating
/// fan converges within ~a second of ticks.
const RAYS_PER_TICK: u32 = 512;

#[derive(Clone, Copy)]
struct CachedChain {
    chain: [u8; PT_MAX_ORDER],
    order: u8,
    ttl: u32,
}

#[derive(Default)]
pub struct PathCache {
    chains: Vec<CachedChain>,
    rot: u32,
    scratch: Vec<PathRecord>,
}

impl PathCache {
    /// One tick: discover (rotating fan), refresh/insert chains, then
    /// exact-solve every live chain into `out` as taps. Returns the
    /// number of live paths.
    pub fn update(
        &mut self,
        room: &Shoebox,
        src: Vec3,
        listener: Vec3,
        out: &mut Vec<Tap>,
    ) -> usize {
        pt_discover(room, &[src], listener, RAYS_PER_TICK, self.rot, &mut self.scratch);
        self.rot = self.rot.wrapping_add(1);
        for r in &self.scratch {
            match self
                .chains
                .iter_mut()
                .find(|c| c.chain == r.chain && c.order == r.order)
            {
                Some(c) => c.ttl = CHAIN_TTL,
                None => self.chains.push(CachedChain {
                    chain: r.chain,
                    order: r.order,
                    ttl: CHAIN_TTL,
                }),
            }
        }

        out.clear();
        self.chains.retain_mut(|c| {
            c.ttl = c.ttl.saturating_sub(1);
            if c.ttl == 0 {
                return false;
            }
            let chain = &c.chain[..c.order as usize];
            // exact solve, fresh every tick: geometry glides, key holds
            match record_for(room, chain, 0, src, listener) {
                Some(r) => {
                    out.push(Tap {
                        key: PT_KEY_BASE + r.key(),
                        delay_s: r.delay_s,
                        dir: r.dir,
                        gains: r.gains,
                    });
                    true
                }
                // stopped validating (an occluder moved in): drop NOW —
                // the renderer fades the vanished key
                None => false,
            }
        });
        let _ = NO_WALL;
        out.len()
    }
}
