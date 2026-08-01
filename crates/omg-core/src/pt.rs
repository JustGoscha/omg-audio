//! PT-early phase C1 (GPU_PLAN.md Track C): path-traced early
//! reflections — the discovery half. Listener-launched rays find which
//! ordered wall CHAINS carry sound; each discovered chain is then
//! solved ANALYTICALLY by mirror reconstruction (the same mirrors the
//! image-source lattice composes), so delays, directions and gains are
//! machine-exact — rays never quantize the answer, they only discover
//! its existence. With in-room occluders (phase C5) the same chains
//! get segment-validated instead of trusted, which is exactly where
//! image-source enumeration stops scaling and discovery keeps working.
//!
//! Conventions mirror ism.rs exactly: gains are amplitude
//! `air(dist)/max(dist, 0.3) × Π reflection_amplitude(wall)`, dir is
//! the unit vector from listener toward the (image) source — where the
//! sound arrives from — and delay is `dist / c`.

use crate::material::air_attenuation;
use crate::rng::Rng;
use crate::scene::Shoebox;
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

/// Chain length cap. 3 matches the ISM order the engine ships with.
pub const PT_MAX_ORDER: usize = 3;
const MIN_DIST: f32 = 0.3;
/// Empty chain slot marker.
pub const NO_WALL: u8 = 0xFF;

/// One exact early path: `chain` is the ordered walls hit walking the
/// path FROM THE LISTENER (chain[0] = last bounce before the ear).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathRecord {
    pub source: u16,
    pub chain: [u8; PT_MAX_ORDER],
    pub order: u8,
    pub delay_s: f32,
    pub dir: [f32; 3],
    pub gains: [f32; NBANDS],
}

impl PathRecord {
    /// Stable identity: source id + packed chain. Base-7 digits (wall
    /// 0..5, 6 = empty) keep distinct chains collision-free.
    pub fn key(&self) -> u32 {
        let mut k = self.source as u32;
        for i in 0..PT_MAX_ORDER {
            let d = if self.chain[i] == NO_WALL { 6 } else { self.chain[i] as u32 };
            k = k * 7 + d;
        }
        k
    }
}

/// Mirror a point across wall plane `w` of the box.
fn mirror(p: Vec3, w: u8, size: Vec3) -> Vec3 {
    let mut m = p;
    match w {
        0 => m.x = -p.x,
        1 => m.x = 2.0 * size.x - p.x,
        2 => m.y = -p.y,
        3 => m.y = 2.0 * size.y - p.y,
        4 => m.z = -p.z,
        _ => m.z = 2.0 * size.z - p.z,
    }
    m
}

/// Solve a chain exactly: image of `source` seen by `listener` through
/// the ordered reflections, plus validation that the straight line to
/// the image actually unfolds into this chain inside the box (grazing
/// discoveries can name a chain the geometry doesn't support).
/// Returns None for invalid chains.
pub fn solve_chain(
    room: &Shoebox,
    chain: &[u8],
    source: Vec3,
    listener: Vec3,
) -> Option<(f32, Vec3, f32)> {
    // image = M_{c0}(M_{c1}(... M_{ck-1}(source)))  — c0 is the wall
    // nearest the listener, so it is applied LAST walking from source.
    let mut img = source;
    for &w in chain.iter().rev() {
        img = mirror(img, w, room.size);
    }
    let to_img = img - listener;
    let dist = to_img.length();
    if dist < 1e-4 {
        return None;
    }
    let dir = to_img * (1.0 / dist);

    // Unfold walk: trace the actual reflected path listener→…→source and
    // require it to hit exactly this chain, in order. Solved per plane
    // (not via first-hit raycast) so CORNER-COINCIDENT reflections pass:
    // a path through the very corner hits both walls at the same t (a
    // valid retroreflection — the golden rooms produce one exactly),
    // where a strict nearest-wall walk would see t = 0 and refuse. A
    // different wall strictly earlier still invalidates. This per-plane
    // walk is also the hook where C5 occluder segment tests slot in.
    let mut pos = listener;
    let mut d = dir;
    for &w in chain {
        let axis = (w / 2) as usize;
        let plane = if w % 2 == 0 { 0.0 } else { room.size.get(axis) };
        let di = d.get(axis);
        if di.abs() < 1e-9 {
            return None; // parallel to the plane: unreachable
        }
        let t = ((plane - pos.get(axis)) / di).max(0.0);
        if !t.is_finite() {
            return None;
        }
        // moving away from the plane never hits it
        if (plane - pos.get(axis)) * di < -1e-6 {
            return None;
        }
        // no OTHER wall strictly earlier (tolerance covers corner ties)
        let (t_first, _) = room.raycast(pos, d);
        if t_first < t - 1e-3 {
            return None;
        }
        pos = pos + d * t;
        pos.set(axis, plane); // exact, and keeps fp drift off the walls
        let mut n = Vec3::new(0.0, 0.0, 0.0);
        n.set(axis, if w % 2 == 0 { 1.0 } else { -1.0 });
        d = d - n * (2.0 * d.dot(n));
    }
    // final leg must reach the source before any wall
    let to_src = source - pos;
    let leg = to_src.length();
    if leg > 1e-4 {
        let ld = to_src * (1.0 / leg);
        let (t, _) = room.raycast(pos, ld);
        if t < leg - 1e-3 {
            return None;
        }
        // and the leg direction must be the reflected continuation
        if ld.dot(d) < 0.999 {
            return None;
        }
    }
    Some((dist, dir, leg))
}

/// Build the full record for a validated chain (gains per ism.rs).
pub fn record_for(
    room: &Shoebox,
    chain: &[u8],
    source: u16,
    src_pos: Vec3,
    listener: Vec3,
) -> Option<PathRecord> {
    let (dist, dir, _) = solve_chain(room, chain, src_pos, listener)?;
    let d = dist.max(MIN_DIST);
    let air = air_attenuation(d);
    let mut gains = [0.0f32; NBANDS];
    for b in 0..NBANDS {
        let mut g = air[b] / d;
        for &w in chain {
            g *= room.walls[w as usize].reflection_amplitude()[b];
        }
        gains[b] = g;
    }
    let mut c = [NO_WALL; PT_MAX_ORDER];
    c[..chain.len()].copy_from_slice(chain);
    Some(PathRecord {
        source,
        chain: c,
        order: chain.len() as u8,
        delay_s: dist / SPEED_OF_SOUND,
        dir: [dir.x, dir.y, dir.z],
        gains,
    })
}

/// A candidate chain: walls in listener-first order, plus length.
pub type Chain = ([u8; PT_MAX_ORDER], u8);

/// Seed set: the direct path and every order ≤2 chain. 37 exact solves
/// cost nothing, guarantee low-order completeness regardless of ray
/// luck (corner-adjacent paths live in mm-wide discovery corridors),
/// and leave the rays to earn the combinatorial tail — order 3 here,
/// occluder-face chains from C5 on.
pub fn seed_chains(out: &mut Vec<Chain>) {
    out.push(([NO_WALL; PT_MAX_ORDER], 0));
    for w1 in 0..6u8 {
        let mut c = [NO_WALL; PT_MAX_ORDER];
        c[0] = w1;
        out.push((c, 1));
        for w2 in 0..6u8 {
            if w2 != w1 {
                let mut c2 = c;
                c2[1] = w2;
                out.push((c2, 2));
            }
        }
    }
}

/// Ray discovery: a deterministic golden-spiral fan from the listener
/// (`rot` rotates the fan so coverage accumulates across ticks),
/// bounced specularly up to PT_MAX_ORDER. Every prefix of every ray's
/// wall sequence is a candidate chain. Chains are source-independent
/// in a convex box — this is exactly the part the GPU kernel
/// (pt_early.wgsl) replaces, emitting the same chain set as a bitmap.
pub fn pt_chains(room: &Shoebox, listener: Vec3, n_rays: u32, rot: u32, out: &mut Vec<Chain>) {
    let mut seen = std::collections::HashSet::new();
    let ga = core::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let mut jitter = Rng::new(0x9E37 ^ rot as u64 | 1);
    for i in 0..n_rays {
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / n_rays as f32;
        let r = (1.0 - z * z).max(0.0).sqrt();
        // golden-angle azimuth, rotated per tick + tiny jitter so the
        // same chains aren't the only ones ever sampled
        let phi = ga * i as f32
            + rot as f32 * 0.61803398875 * core::f32::consts::TAU
            + jitter.next_f32() * 0.02;
        let mut dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        let mut pos = listener;
        let mut chain = [NO_WALL; PT_MAX_ORDER];
        for k in 0..PT_MAX_ORDER {
            let (t, wall) = room.raycast(pos, dir);
            if !t.is_finite() || t <= 1e-5 {
                break;
            }
            pos = pos + dir * t;
            chain[k] = wall as u8;
            // leading-zero-proof key (a plain positional hash aliases
            // wall 0 with "no wall" — measured bug, not theory)
            let mut key = 1u64;
            for &w in &chain[..=k] {
                key = key * 8 + (w as u64 + 1);
            }
            if seen.insert(key) {
                out.push((chain, (k + 1) as u8));
            }
            let mut n = Vec3::new(0.0, 0.0, 0.0);
            n.set(wall / 2, if wall % 2 == 0 { 1.0 } else { -1.0 });
            dir = dir - n * (2.0 * dir.dot(n));
            pos = pos + n * 1e-5;
        }
    }
}

/// Discovery + seeding + exact solving in one call (the C1 shape; the
/// engine's cache drives pt_chains/seed_chains/record_for itself).
pub fn pt_discover(
    room: &Shoebox,
    sources: &[Vec3],
    listener: Vec3,
    n_rays: u32,
    rot: u32,
    out: &mut Vec<PathRecord>,
) {
    out.clear();
    let mut chains = Vec::new();
    seed_chains(&mut chains);
    pt_chains(room, listener, n_rays, rot, &mut chains);

    let mut seen = std::collections::HashSet::new();
    for (chain, order) in chains {
        let chain = &chain[..order as usize];
        for (si, &sp) in sources.iter().enumerate() {
            let mut k = si as u64 * 4096 + 1;
            for &w in chain {
                k = k * 8 + (w as u64 + 1);
            }
            if !seen.insert(k) {
                continue;
            }
            if let Some(r) = record_for(room, chain, si as u16, sp, listener) {
                // Corner degeneracy: at an exact edge both orderings of
                // the two walls validate as the SAME physical path —
                // emitting both would double its energy.
                let dup = out.iter().any(|q: &PathRecord| {
                    q.source == r.source
                        && (q.delay_s - r.delay_s).abs() < 2e-5
                        && q.dir[0] * r.dir[0] + q.dir[1] * r.dir[1] + q.dir[2] * r.dir[2]
                            > 0.99999
                });
                if !dup {
                    out.push(r);
                }
            }
        }
    }
}
