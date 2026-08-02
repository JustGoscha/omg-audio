//! PT-early phase C6b (GPU_PLAN.md Track C endgame): chains over the
//! WORLD MESH instead of a room's analytic walls. A chain is an ordered
//! list of authored SURFACE ids (C6a); discovery bounces listener rays
//! over the BVH, the solve mirrors the source across the surfaces'
//! planes (machine-exact, same construction as the box solver), and the
//! unfold validation additionally accumulates TRANSMISSION for every
//! surface a leg crosses — a doorway is just a hole a leg threads for
//! free, a wall crossing pays its mass-law loss. This is the engine
//! that makes portals unnecessary: sound reaches the listener because
//! geometry lets it, not because a room graph says so.
//!
//! Conventions mirror pt.rs/ism.rs: amplitude gains
//! `air(dist)/max(dist, 0.3) × Π reflection × Π transmission`, dir from
//! listener toward the (image) source, delay `dist / c`.

use crate::material::air_attenuation;
use crate::mesh::{Mesh, SegHit};
use crate::pt::Aabb;
use crate::rng::Rng;
use crate::vec3::Vec3;
use crate::{NBANDS, SPEED_OF_SOUND};

pub const M_MAX_ORDER: usize = 3;
const MIN_DIST: f32 = 0.3;
pub const NO_SURF: u16 = 0xFFFF;
/// Gains below this (all bands) drop the record.
const AUDIBLE_FLOOR: f32 = 2e-5;

/// One authored plane of the world: unit normal, offset (n·x = d),
/// and the per-band amplitudes of its material.
pub struct Surface {
    pub n: Vec3,
    pub d: f32,
    pub refl: [f32; NBANDS],
    pub trans: [f32; NBANDS],
    /// Rect-bounded OVERLAY face (a furniture box side): the AABB of
    /// the face itself. `None` = authored mesh plane, validated by
    /// raycast; `Some` = validated by the rect (there are no triangles
    /// to hit — the box is world state, not mesh).
    pub rect: Option<(Vec3, Vec3)>,
}

/// Per-surface plane/material table, built once per mesh.
pub struct SurfaceTable {
    pub surfaces: Vec<Surface>,
    /// First overlay-face id — everything below is authored mesh.
    pub base_overlay: u16,
}

impl SurfaceTable {
    pub fn build(mesh: &Mesh) -> Self {
        let mut surfaces: Vec<Surface> = Vec::new();
        for t in 0..mesh.tri_count() as u32 {
            let sid = mesh.tri_surface(t) as usize;
            if sid >= surfaces.len() {
                surfaces.resize_with(sid + 1, || Surface {
                    n: Vec3::new(0.0, 0.0, 1.0),
                    d: f32::MAX,
                    refl: [0.0; NBANDS],
                    trans: [0.0; NBANDS],
                    rect: None,
                });
            }
            let s = &mut surfaces[sid];
            if s.d == f32::MAX {
                let n = mesh.tri_normal(t);
                let p = mesh.positions[mesh.indices[t as usize][0] as usize];
                let m = &mesh.materials[mesh.tri_material[t as usize] as usize];
                *s = Surface {
                    n,
                    d: n.dot(p),
                    refl: m.reflection_amplitude(),
                    trans: m.transmission,
                    rect: None,
                };
            }
        }
        let base_overlay = surfaces.len() as u16;
        Self { surfaces, base_overlay }
    }

    /// Append the six faces of an overlay box (furniture) as
    /// rect-bounded reflective surfaces with the box's material —
    /// order: (x·min, x·max, y·min, y·max, z·min, z·max), matching the
    /// discovery kernels' face indexing. Chains may then reflect off a
    /// table top exactly like off a wall.
    pub fn append_box(&mut self, mn: Vec3, mx: Vec3, m: &crate::material::Material) {
        for axis in 0..3 {
            for side in 0..2 {
                let mut n = Vec3::new(0.0, 0.0, 0.0);
                n.set(axis, if side == 0 { -1.0 } else { 1.0 });
                let plane = if side == 0 { mn.get(axis) } else { mx.get(axis) };
                let mut fmin = mn;
                let mut fmax = mx;
                fmin.set(axis, plane);
                fmax.set(axis, plane);
                self.surfaces.push(Surface {
                    n,
                    d: n.dot(if side == 0 { mn } else { mx }),
                    refl: m.reflection_amplitude(),
                    trans: m.transmission,
                    rect: Some((fmin, fmax)),
                });
            }
        }
    }
}

pub type MChain = ([u16; M_MAX_ORDER], u8);

/// A solved world path.
#[derive(Clone, Copy, Debug)]
pub struct MeshRecord {
    pub source: u16,
    pub chain: [u16; M_MAX_ORDER],
    pub order: u8,
    pub delay_s: f32,
    pub dir: [f32; 3],
    pub gains: [f32; NBANDS],
}

impl MeshRecord {
    pub fn key(&self) -> u32 {
        // FNV over source + surface ids (u16 space is too wide for a
        // positional base — hash and accept the ~2^-32 collision odds)
        let mut h: u32 = 0x811C_9DC5 ^ self.source as u32;
        for i in 0..self.order as usize {
            h = (h ^ self.chain[i] as u32).wrapping_mul(16_777_619);
        }
        h = (h ^ 0xA5) .wrapping_mul(16_777_619);
        h
    }
}

fn mirror(p: Vec3, s: &Surface) -> Vec3 {
    p - s.n * (2.0 * (s.n.dot(p) - s.d))
}

/// Transmission of every surface the OPEN segment a→b crosses, skipping
/// crossings that belong to `skip` (the reflection surface itself).
/// `extras` are transient blockers the mesh doesn't carry — door leaves,
/// glass panes, furniture — as boxes with per-band transmission; each box
/// the leg passes through multiplies in once (a leaf across a door hole
/// is exactly what turns "free" back into "mass law").
fn leg_transmission(
    mesh: &Mesh,
    a: Vec3,
    b: Vec3,
    skip: Option<(u16, f32)>,
    extras: &[Aabb],
    buf: &mut Vec<SegHit>,
    trans: &mut [f32; NBANDS],
) -> bool {
    mesh.segment_hits(a, b, buf);
    let seg_len = (b - a).length().max(1e-6);
    for h in buf.iter() {
        if let Some((sid, t_expect)) = skip {
            if mesh.tri_surface(h.tri) == sid && (h.t - t_expect).abs() * seg_len < 0.05 {
                continue;
            }
        }
        let m = &mesh.materials[h.material as usize];
        for b in 0..NBANDS {
            trans[b] *= m.transmission[b];
        }
        if trans.iter().all(|&x| x < 1e-6) {
            return false;
        }
    }
    for x in extras {
        if x.clip(a, b).is_some() {
            for b in 0..NBANDS {
                trans[b] *= x.transmission[b];
            }
            if trans.iter().all(|&x| x < 1e-6) {
                return false;
            }
        }
    }
    true
}

/// Exact solve + validation + transmission for a surface chain between
/// world-space `source` and `listener`.
pub fn mesh_record(
    mesh: &Mesh,
    table: &SurfaceTable,
    chain: &[u16],
    source: u16,
    src_pos: Vec3,
    listener: Vec3,
    extras: &[Aabb],
    buf: &mut Vec<SegHit>,
) -> Option<MeshRecord> {
    // image: mirror source across the chain planes, listener-first
    // chain c0 applied last
    let mut img = src_pos;
    for &sid in chain.iter().rev() {
        img = mirror(img, table.surfaces.get(sid as usize)?);
    }
    let to_img = img - listener;
    let dist = to_img.length();
    if dist < 1e-4 {
        return None;
    }
    let dir = to_img * (1.0 / dist);

    let mut gains_mul = [1.0f32; NBANDS];
    let mut pos = listener;
    let mut d = dir;
    for &sid in chain {
        let s = table.surfaces.get(sid as usize)?;
        let denom = s.n.dot(d);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = (s.d - s.n.dot(pos)) / denom;
        if t <= 1e-4 {
            return None;
        }
        let hit = pos + d * t;
        if let Some((fmin, fmax)) = s.rect {
            // overlay face (furniture): the reflection point must land
            // ON the face rect — there is no mesh to raycast for it
            let eps = 1e-3;
            for a in 0..3 {
                if hit.get(a) < fmin.get(a) - eps || hit.get(a) > fmax.get(a) + eps {
                    return None;
                }
            }
            // a NEARER opaque mesh blocker still kills the specular claim
            if let Some((rt, rtri)) = mesh.raycast(pos + d * 1e-4, d) {
                if rt < t - 1e-3 {
                    let m = &mesh.materials[mesh.tri_material[rtri as usize] as usize];
                    if m.transmission.iter().all(|&x| x < 1e-6) {
                        return None;
                    }
                }
            }
        } else {
            // the reflection point must land on REAL mesh of this surface
            // (not in a door hole, and NOT in the empty air beyond the
            // wall's extent — the plane is infinite, the wall is not):
            // the nearest mesh hit along the ray must be this surface AT
            // this distance. A first hit BEYOND t means nothing stands
            // at the bounce point — the sealed-club ghost that reflected
            // off a neighbor's wall plane three meters past its corner.
            let (rt, rtri) = mesh.raycast(pos + d * 1e-4, d)?;
            let on_surface = mesh.tri_surface(rtri) == sid && (rt - t).abs() < 2e-2;
            if !on_surface {
                if rt < t - 1e-3 {
                    // a nearer blocker: opaque kills the specular claim;
                    // a transmissive one may stand in front, but the
                    // chain surface must still EXIST at the bounce point
                    let m = &mesh.materials[mesh.tri_material[rtri as usize] as usize];
                    if m.transmission.iter().all(|&x| x < 1e-4) {
                        return None;
                    }
                    mesh.segment_hits(pos, hit + d * 0.05, buf);
                    if !buf
                        .iter()
                        .any(|h| mesh.tri_surface(h.tri) == sid && (h.t * (t + 0.05) - t).abs() < 0.1)
                    {
                        return None;
                    }
                } else {
                    return None;
                }
            }
        }
        // transmission across everything this leg crosses (the
        // reflection surface itself excluded at its own t)
        if !leg_transmission(mesh, pos, hit, Some((sid, 1.0)), extras, buf, &mut gains_mul) {
            return None;
        }
        for b in 0..NBANDS {
            gains_mul[b] *= s.refl[b];
        }
        pos = hit + s.n * if s.n.dot(d) < 0.0 { 1e-4 } else { -1e-4 };
        d = d - s.n * (2.0 * s.n.dot(d));
    }
    // ISM validity: the source must lie on the SAME side of the last
    // mirror plane as the reflected ray — a source BEHIND the mirror is
    // a transmission ghost wearing a reflection costume (its re-crossing
    // of the plane starts exactly at the hit point and evades payment):
    // the sealed-club "sharp pocket" solve records.
    if let Some(&last) = chain.last() {
        let s = table.surfaces.get(last as usize)?;
        if (s.n.dot(pos) - s.d) * (s.n.dot(src_pos) - s.d) < 0.0 {
            return None;
        }
    }
    // final leg to the source, transmission for every crossing
    if !leg_transmission(mesh, pos, src_pos, None, extras, buf, &mut gains_mul) {
        return None;
    }
    if !chain.is_empty() {
        let to_src = src_pos - pos;
        let leg = to_src.length();
        if leg > 1e-4 && (to_src * (1.0 / leg)).dot(d) < 0.995 {
            return None; // reflected continuation must reach the source
        }
    }

    let dtot = dist.max(MIN_DIST);
    let air = air_attenuation(dtot);
    let mut gains = [0.0f32; NBANDS];
    for b in 0..NBANDS {
        gains[b] = air[b] / dtot * gains_mul[b];
    }
    if gains.iter().all(|&g| g < AUDIBLE_FLOOR) {
        return None;
    }
    let mut c = [NO_SURF; M_MAX_ORDER];
    c[..chain.len()].copy_from_slice(chain);
    Some(MeshRecord {
        source,
        chain: c,
        order: chain.len() as u8,
        delay_s: dist / SPEED_OF_SOUND,
        dir: [dir.x, dir.y, dir.z],
        gains,
    })
}

/// Debug: the world-space polyline of a chain (listener, hit points…,
/// source). Pure geometry — no validation, no transmission; callers only
/// pass chains whose records already solved this tick.
pub fn mesh_vertices(
    table: &SurfaceTable,
    chain: &[u16],
    src_pos: Vec3,
    listener: Vec3,
    out: &mut Vec<Vec3>,
) -> bool {
    out.clear();
    let mut img = src_pos;
    for &sid in chain.iter().rev() {
        let Some(s) = table.surfaces.get(sid as usize) else { return false };
        img = mirror(img, s);
    }
    let to_img = img - listener;
    let dist = to_img.length();
    if dist < 1e-4 {
        return false;
    }
    let mut d = to_img * (1.0 / dist);
    let mut pos = listener;
    out.push(pos);
    for &sid in chain {
        let Some(s) = table.surfaces.get(sid as usize) else { return false };
        let denom = s.n.dot(d);
        if denom.abs() < 1e-9 {
            return false;
        }
        let t = (s.d - s.n.dot(pos)) / denom;
        if t <= 1e-4 {
            return false;
        }
        pos = pos + d * t;
        out.push(pos);
        d = d - s.n * (2.0 * s.n.dot(d));
    }
    out.push(src_pos);
    true
}

/// Discovery over the mesh: listener-launched rotating golden fan,
/// specular bounces, chains of surface ids. Rays stop at mesh hits
/// (transmission chains behind walls are seeded by the direct path
/// and validated with crossings — specular discovery through masonry
/// is not attempted). `boxes` are overlay boxes (furniture) whose
/// faces count as reflectors too, ids `base + box·6 + face`
/// (face = axis·2 + min/max) — matching `SurfaceTable::append_box`.
pub fn mesh_chains(
    mesh: &Mesh,
    boxes: &[(Vec3, Vec3)],
    base: u16,
    listener: Vec3,
    n_rays: u32,
    rot: u32,
    out: &mut Vec<MChain>,
) {
    let mut seen = std::collections::HashSet::new();
    let ga = core::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let mut jitter = Rng::new(0xC6B ^ rot as u64 | 1);
    for i in 0..n_rays {
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / n_rays as f32;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = ga * i as f32
            + rot as f32 * 0.61803398875 * core::f32::consts::TAU
            + jitter.next_f32() * 0.02;
        let mut dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        let mut pos = listener;
        let mut chain = [NO_SURF; M_MAX_ORDER];
        for k in 0..M_MAX_ORDER {
            let mesh_hit = mesh.raycast(pos, dir);
            let mut t = mesh_hit.map_or(f32::MAX, |(t, _)| t);
            let mut sid = mesh_hit.map(|(_, tri)| mesh.tri_surface(tri));
            let mut normal =
                mesh_hit.map_or(Vec3::new(0.0, 0.0, 1.0), |(_, tri)| mesh.tri_normal(tri));
            // overlay boxes: slab-entry test, nearest wins
            for (bi, (mn, mx)) in boxes.iter().enumerate() {
                let (mut t0, mut t1) = (1e-4f32, t);
                let mut axis = 3usize;
                for a in 0..3 {
                    let da = dir.get(a);
                    if da.abs() < 1e-9 {
                        if pos.get(a) < mn.get(a) || pos.get(a) > mx.get(a) {
                            t0 = f32::MAX;
                            break;
                        }
                    } else {
                        let (mut ta, mut tb) =
                            ((mn.get(a) - pos.get(a)) / da, (mx.get(a) - pos.get(a)) / da);
                        if ta > tb {
                            core::mem::swap(&mut ta, &mut tb);
                        }
                        if ta > t0 {
                            t0 = ta;
                            axis = a;
                        }
                        t1 = t1.min(tb);
                        if t0 > t1 {
                            t0 = f32::MAX;
                            break;
                        }
                    }
                }
                if t0 < t && axis < 3 {
                    t = t0;
                    let face = axis * 2 + if dir.get(axis) > 0.0 { 0 } else { 1 };
                    sid = Some(base + (bi * 6 + face) as u16);
                    let mut n = Vec3::new(0.0, 0.0, 0.0);
                    n.set(axis, if dir.get(axis) > 0.0 { -1.0 } else { 1.0 });
                    normal = n;
                }
            }
            let Some(sid) = sid else { break };
            if t <= 1e-4 || t > 200.0 {
                break;
            }
            pos = pos + dir * t;
            chain[k] = sid;
            let mut key = 1u64;
            for &s in &chain[..=k] {
                key = key * 65536 + s as u64 + 1;
            }
            if seen.insert(key) {
                out.push((chain, (k + 1) as u8));
            }
            let n = if normal.dot(dir) > 0.0 { normal * -1.0 } else { normal };
            dir = dir - n * (2.0 * dir.dot(n));
            pos = pos + n * 1e-4;
        }
    }
}
