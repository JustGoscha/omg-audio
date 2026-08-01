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

use crate::diffraction::knife_edge_bands;
use crate::material::air_attenuation;
use crate::rng::Rng;
use crate::scene::Shoebox;
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

/// An axis-aligned occluder inside a room (C5: furniture, pillars,
/// bar counters). `transmission` is the per-band AMPLITUDE that
/// survives passing through the piece (mass-law flavored: a sofa lets
/// muffled bass through, a stone pillar next to nothing); blocked
/// paths carry it instead of vanishing, and the blocked direct path
/// takes whichever is louder per band — the through-body seep or the
/// knife-edge bend around the silhouette.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
    pub transmission: [f32; NBANDS],
}

/// Solid stone default for tests / plain blockers.
pub const OPAQUE: [f32; NBANDS] = [0.02, 0.002, 0.0];

impl Aabb {
    /// Clip the segment a→b against the box: Some((t0, t1)) of the
    /// interior crossing, None when it misses.
    pub fn clip(&self, a: Vec3, b: Vec3) -> Option<(f32, f32)> {
        let d = b - a;
        let mut t0 = 0.0f32;
        let mut t1 = 1.0f32;
        for axis in 0..3 {
            let (da, aa, mn, mx) =
                (d.get(axis), a.get(axis), self.min.get(axis), self.max.get(axis));
            if da.abs() < 1e-9 {
                if aa < mn || aa > mx {
                    return None;
                }
            } else {
                let (mut ta, mut tb) = ((mn - aa) / da, (mx - aa) / da);
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
        if t1 > 1e-4 && t0 < 1.0 - 1e-4 {
            Some((t0, t1))
        } else {
            None
        }
    }

    /// Does the open segment a→b pass through this box?
    pub fn blocks(&self, a: Vec3, b: Vec3) -> bool {
        let d = b - a;
        let mut t0 = 0.0f32;
        let mut t1 = 1.0f32;
        for axis in 0..3 {
            let (da, aa, mn, mx) =
                (d.get(axis), a.get(axis), self.min.get(axis), self.max.get(axis));
            if da.abs() < 1e-9 {
                if aa < mn || aa > mx {
                    return false;
                }
            } else {
                let (mut ta, mut tb) = ((mn - aa) / da, (mx - aa) / da);
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
        // interior crossing only (touching a face at the very ends of
        // the segment is not occlusion)
        t1 > 1e-4 && t0 < 1.0 - 1e-4
    }
}

fn segment_blocked(occluders: &[Aabb], a: Vec3, b: Vec3) -> Option<usize> {
    occluders.iter().position(|o| o.blocks(a, b))
}

/// Cheapest bend around a box between two points: every one of the 12
/// box edges is a knife-edge candidate — apex = the point of the edge
/// nearest the straight segment, nudged just off the box — and a
/// candidate only counts if both bent legs actually clear the box.
/// Minimum valid detour wins: sound wraps around the nearest
/// silhouette, whichever edge that is. None = the box seals the path
/// (wall-to-wall furniture): only the late field remains.
fn best_bend(o: &Aabb, a: Vec3, b: Vec3, room_size: Vec3) -> Option<(Vec3, f32)> {
    let straight = (b - a).length();
    let (mn, mx) = (o.min, o.max);
    let corner = |i: u32| {
        Vec3::new(
            if i & 1 == 0 { mn.x } else { mx.x },
            if i & 2 == 0 { mn.y } else { mx.y },
            if i & 4 == 0 { mn.z } else { mx.z },
        )
    };
    // the 12 edges as corner-index pairs
    const EDGES: [(u32, u32); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // x-aligned
        (0, 2), (1, 3), (4, 6), (5, 7), // y-aligned
        (0, 4), (1, 5), (2, 6), (3, 7), // z-aligned
    ];
    let center = (mn + mx) * 0.5;
    let ab = b - a;
    let mut best: Option<(Vec3, f32)> = None;
    for (i0, i1) in EDGES {
        let (e0, e1) = (corner(i0), corner(i1));
        let ev = e1 - e0;
        // closest point of the edge line to the straight line a→b
        // (standard line-line closest point, clamped to the edge)
        let w0 = e0 - a;
        let (aa, bb, cc) = (ab.dot(ab), ab.dot(ev), ev.dot(ev));
        let (dd, ee) = (ab.dot(w0), ev.dot(w0));
        let denom = aa * cc - bb * bb;
        let s = if denom.abs() < 1e-9 { 0.5 } else { ((bb * dd - aa * ee) / denom).clamp(0.0, 1.0) };
        let on_edge = e0 + ev * s;
        // nudge off the box so the legs don't graze its faces
        let out = (on_edge - center).normalize();
        let apex = on_edge + out * 1e-3;
        if o.blocks(a, apex) || o.blocks(apex, b) {
            continue;
        }
        let detour = ((apex - a).length() + (b - apex).length() - straight).max(0.0);
        if best.map_or(true, |(_, d)| detour < d) {
            best = Some((apex, detour));
        }
    }
    // Double bends hugging each face: a box wider than the path's
    // clearance needs TWO edges (up over the far rim, down at the near
    // rim — same for side wraps); a single apex always leaves one leg
    // inside the box. Apexes = the entry/exit points of the straight
    // segment lifted onto the face plane (just outside), which is the
    // taut-string path over that face. Total detour through one knife
    // edge is a mild over-estimate of the two-edge loss — conservative.
    if let Some((t0, t1)) = o.clip(a, b) {
        let d = b - a;
        let p_entry = a + d * t0;
        let p_exit = a + d * t1;
        for axis in 0..3usize {
            for (v, sign) in [(mn.get(axis), -1.0f32), (mx.get(axis), 1.0)] {
                let lift = |p: Vec3| {
                    let mut q = p;
                    q.set(axis, v + sign * 1e-3);
                    q
                };
                let (a1, a2) = (lift(p_entry), lift(p_exit));
                // stay inside the room (no bending under the floor or
                // through a wall the box touches)
                if a1.get(axis) < 1e-4 || a1.get(axis) > room_size.get(axis) - 1e-4 {
                    continue;
                }
                if o.blocks(a, a1) || o.blocks(a2, b) {
                    continue;
                }
                let detour = ((a1 - a).length() + (a2 - a1).length() + (b - a2).length()
                    - straight)
                    .max(0.0);
                if best.map_or(true, |(_, dd)| detour < dd) {
                    best = Some((a1, detour));
                }
            }
        }
    }
    best
}

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
    solve_chain_occ(room, chain, source, listener, &[])
}

/// `solve_chain` with in-room occluders: every unfolded segment must
/// clear every box, or the chain doesn't exist right now (C5 — the
/// cache drops it and the tap fades).
pub fn solve_chain_occ(
    room: &Shoebox,
    chain: &[u8],
    source: Vec3,
    listener: Vec3,
    occluders: &[Aabb],
) -> Option<(f32, Vec3, f32)> {
    solve_chain_trans(room, chain, source, listener, occluders)
        .and_then(|(d, dir, leg, t)| (t[0] >= 0.999).then_some((d, dir, leg)))
}

/// Like `solve_chain_occ` but blocked segments accumulate the pieces'
/// through-transmission instead of invalidating the chain. Returns the
/// per-band amplitude product (1.0 = unobstructed); geometric
/// invalidity is still None.
pub fn solve_chain_trans(
    room: &Shoebox,
    chain: &[u8],
    source: Vec3,
    listener: Vec3,
    occluders: &[Aabb],
) -> Option<(f32, Vec3, f32, [f32; NBANDS])> {
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

    let mut trans = [1.0f32; NBANDS];
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
        let next = {
            let mut p = pos + d * t;
            p.set(axis, plane); // exact, and keeps fp drift off the walls
            p
        };
        for o in occluders {
            if o.blocks(pos, next) {
                for b in 0..NBANDS {
                    trans[b] *= o.transmission[b];
                }
            }
        }
        pos = next;
        let mut n = Vec3::new(0.0, 0.0, 0.0);
        n.set(axis, if w % 2 == 0 { 1.0 } else { -1.0 });
        d = d - n * (2.0 * d.dot(n));
    }
    // final leg must reach the source before any wall or occluder
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
    for o in occluders {
        if o.blocks(pos, source) {
            for b in 0..NBANDS {
                trans[b] *= o.transmission[b];
            }
        }
    }
    Some((dist, dir, leg, trans))
}

/// Build the full record for a validated chain (gains per ism.rs).
pub fn record_for(
    room: &Shoebox,
    chain: &[u8],
    source: u16,
    src_pos: Vec3,
    listener: Vec3,
) -> Option<PathRecord> {
    record_for_occ(room, chain, source, src_pos, listener, &[])
}

/// `record_for` with occluders. Blocked reflections simply cease to
/// exist (the late field covers their energy); the blocked DIRECT path
/// instead hands off to a knife-edge bend over the blocking box's top
/// — same tap key, longer delay, Kurze–Anderson per-band loss — so
/// walking into a shadow sweeps the sound instead of cutting it.
pub fn record_for_occ(
    room: &Shoebox,
    chain: &[u8],
    source: u16,
    src_pos: Vec3,
    listener: Vec3,
    occluders: &[Aabb],
) -> Option<PathRecord> {
    let solved = solve_chain_trans(room, chain, src_pos, listener, occluders);
    let (dist, dir, gains) = match solved {
        Some((dist, dir, _, trans)) => {
            let d = dist.max(MIN_DIST);
            let air = air_attenuation(d);
            let mut gains = [0.0f32; NBANDS];
            for b in 0..NBANDS {
                let mut g = air[b] / d * trans[b];
                for &w in chain {
                    g *= room.walls[w as usize].reflection_amplitude()[b];
                }
                gains[b] = g;
            }
            let mut out_dist = dist;
            if chain.is_empty() {
                if trans[0] >= 0.999 {
                    // CLEAR direct path: Kurze–Anderson lit-side edge
                    // proximity takes the field to −5 dB at each shadow
                    // boundary so crossing into the bent branch below
                    // is continuous. Distant boxes contribute factor 1.
                    for o in occluders {
                        if let Some((_, detour)) = best_bend(o, listener, src_pos, room.size) {
                            let ke = knife_edge_bands(-detour);
                            for b in 0..NBANDS {
                                gains[b] *= ke[b];
                            }
                        }
                    }
                } else if let Some(bi) = segment_blocked(occluders, listener, src_pos) {
                    // BLOCKED direct: the through-body seep above vs
                    // the knife-edge bend around the silhouette — per
                    // band, whichever carries more survives (bass
                    // seeps through a sofa, treble bends past stone).
                    // The tap's single delay follows the low band's
                    // winning mechanism: straight for seep, bent for
                    // the wrap.
                    if let Some((_, detour)) =
                        best_bend(&occluders[bi], listener, src_pos, room.size)
                    {
                        let ke = knife_edge_bands(detour);
                        let bent = dist + detour;
                        let bair = air_attenuation(bent);
                        if bair[0] / bent.max(MIN_DIST) * ke[0] > gains[0] {
                            out_dist = bent;
                        }
                        for b in 0..NBANDS {
                            gains[b] = gains[b].max(bair[b] / bent.max(MIN_DIST) * ke[b]);
                        }
                    }
                }
            }
            if gains.iter().all(|&g| g < 1e-6) {
                return None; // fully swallowed: cease to exist
            }
            (out_dist, dir, gains)
        }
        None => return None,
    };
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

/// The actual polyline of a chain's path: listener, each reflection
/// point, source — for visualization. None if the chain is invalid.
pub fn chain_vertices(
    room: &Shoebox,
    chain: &[u8],
    source: Vec3,
    listener: Vec3,
    out: &mut Vec<Vec3>,
) -> bool {
    let Some((_, dir, _, _)) = solve_chain_trans(room, chain, source, listener, &[]) else {
        return false;
    };
    out.clear();
    out.push(listener);
    let mut pos = listener;
    let mut d = dir;
    for &w in chain {
        let axis = (w / 2) as usize;
        let plane = if w % 2 == 0 { 0.0 } else { room.size.get(axis) };
        let di = d.get(axis);
        if di.abs() < 1e-9 {
            return false;
        }
        let t = ((plane - pos.get(axis)) / di).max(0.0);
        pos = pos + d * t;
        pos.set(axis, plane);
        out.push(pos);
        let mut n = Vec3::new(0.0, 0.0, 0.0);
        n.set(axis, if w % 2 == 0 { 1.0 } else { -1.0 });
        d = d - n * (2.0 * d.dot(n));
    }
    out.push(source);
    true
}

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
