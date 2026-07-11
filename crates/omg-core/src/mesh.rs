//! Arbitrary triangle geometry for the acoustic simulation: an indexed
//! mesh with per-triangle materials, a BVH for ray queries, and
//! diffraction-edge extraction with an explicit importance budget.
//!
//! Design rules (see repo README):
//!  - The same tracer runs over `Mesh` and the analytic `Shoebox` via
//!    `AcousticGeometry` — equivalence is unit-tested on a box.
//!  - Everything that scales with scene complexity (edges, per-segment
//!    wall crossings) is budgeted: callers pass a cap and the least
//!    perceptually important entries are dropped first.

use crate::material::Material;
use crate::scene::{AcousticGeometry, GeomHit};
use crate::vec3::Vec3;

const LEAF_SIZE: usize = 4;
const LEAF_BIT: u32 = 1 << 31;

pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    /// Per-triangle index into `materials`.
    pub tri_material: Vec<u16>,
    pub materials: Vec<Material>,
    nodes: Vec<BvhNode>,
    /// Intersection primitives in BVH-leaf order. Oversized triangles are
    /// tessellated into patches here (coplanar, same original id) — one
    /// scene-spanning wall triangle would otherwise inflate every node's
    /// bounds and defeat traversal culling. `tri` refers back to the
    /// ORIGINAL triangle for materials/normals/edges.
    packed: Vec<PackedTri>,
}

/// BVH primitives larger than this get split (longest edge, meters).
const MAX_PRIM_EDGE: f32 = 4.0;

#[derive(Clone, Copy)]
struct PackedTri {
    a: Vec3,
    e1: Vec3,
    e2: Vec3,
    tri: u32,
    material: u16,
}

struct BvhNode {
    bmin: Vec3,
    bmax: Vec3,
    /// Leaf: LEAF_BIT | start (into `order`), `b` = count.
    /// Internal: `a` = left child, `b` = right child.
    a: u32,
    b: u32,
}

/// One crossing of a segment through the mesh (for transmission queries).
#[derive(Clone, Copy)]
pub struct SegHit {
    /// Position along the segment, 0..1.
    pub t: f32,
    pub tri: u32,
    pub material: u16,
}

/// Incremental mesh assembly with vertex dedup (quantized to 0.1 mm).
#[derive(Default)]
pub struct MeshBuilder {
    positions: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
    tri_material: Vec<u16>,
    materials: Vec<Material>,
    lookup: std::collections::HashMap<(i64, i64, i64), u32>,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn material(&mut self, m: Material) -> u16 {
        self.materials.push(m);
        (self.materials.len() - 1) as u16
    }

    pub fn vertex(&mut self, p: Vec3) -> u32 {
        let q = |v: f32| (v * 1e4).round() as i64;
        *self.lookup.entry((q(p.x), q(p.y), q(p.z))).or_insert_with(|| {
            self.positions.push(p);
            (self.positions.len() - 1) as u32
        })
    }

    pub fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, material: u16) {
        let (ia, ib, ic) = (self.vertex(a), self.vertex(b), self.vertex(c));
        self.indices.push([ia, ib, ic]);
        self.tri_material.push(material);
    }

    /// Quad a-b-c-d (planar, in winding order) as two triangles.
    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, material: u16) {
        self.tri(a, b, c, material);
        self.tri(a, c, d, material);
    }

    /// Axis-aligned box [min, max], all faces one material, normals
    /// wound outward (the tracer orients normals per-ray anyway).
    pub fn abox(&mut self, min: Vec3, max: Vec3, material: u16) {
        let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let (a, b) = (min, max);
        self.quad(v(a.x, a.y, a.z), v(a.x, b.y, a.z), v(a.x, b.y, b.z), v(a.x, a.y, b.z), material);
        self.quad(v(b.x, a.y, a.z), v(b.x, a.y, b.z), v(b.x, b.y, b.z), v(b.x, b.y, a.z), material);
        self.quad(v(a.x, a.y, a.z), v(a.x, a.y, b.z), v(b.x, a.y, b.z), v(b.x, a.y, a.z), material);
        self.quad(v(a.x, b.y, a.z), v(b.x, b.y, a.z), v(b.x, b.y, b.z), v(a.x, b.y, b.z), material);
        self.quad(v(a.x, a.y, a.z), v(b.x, a.y, a.z), v(b.x, b.y, a.z), v(a.x, b.y, a.z), material);
        self.quad(v(a.x, a.y, b.z), v(a.x, b.y, b.z), v(b.x, b.y, b.z), v(b.x, a.y, b.z), material);
    }

    pub fn build(self) -> Mesh {
        Mesh::new(self.positions, self.indices, self.tri_material, self.materials)
    }
}

impl Mesh {
    pub fn new(
        positions: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
        tri_material: Vec<u16>,
        materials: Vec<Material>,
    ) -> Self {
        assert_eq!(indices.len(), tri_material.len());
        let mut mesh = Self {
            positions,
            indices,
            tri_material,
            materials,
            nodes: Vec::new(),
            packed: Vec::new(),
        };
        mesh.build_bvh();
        mesh
    }

    pub fn tri_count(&self) -> usize {
        self.indices.len()
    }

    fn tri_verts(&self, tri: u32) -> (Vec3, Vec3, Vec3) {
        let [a, b, c] = self.indices[tri as usize];
        (
            self.positions[a as usize],
            self.positions[b as usize],
            self.positions[c as usize],
        )
    }

    fn build_bvh(&mut self) {
        self.nodes.clear();
        // Primitive list: original triangles, oversized ones tessellated.
        let mut prims: Vec<PackedTri> = Vec::with_capacity(self.indices.len());
        let mut work: Vec<(Vec3, Vec3, Vec3, u32)> = (0..self.indices.len() as u32)
            .map(|t| {
                let (a, b, c) = self.tri_verts(t);
                (a, b, c, t)
            })
            .collect();
        while let Some((a, b, c, t)) = work.pop() {
            let (lab, lbc, lca) =
                ((b - a).length(), (c - b).length(), (a - c).length());
            let longest = lab.max(lbc).max(lca);
            if longest > MAX_PRIM_EDGE {
                // split the longest edge at its midpoint
                if lab >= lbc && lab >= lca {
                    let m = (a + b) * 0.5;
                    work.push((a, m, c, t));
                    work.push((m, b, c, t));
                } else if lbc >= lca {
                    let m = (b + c) * 0.5;
                    work.push((a, b, m, t));
                    work.push((a, m, c, t));
                } else {
                    let m = (c + a) * 0.5;
                    work.push((a, b, m, t));
                    work.push((m, b, c, t));
                }
                continue;
            }
            prims.push(PackedTri {
                a,
                e1: b - a,
                e2: c - a,
                tri: t,
                material: self.tri_material[t as usize],
            });
        }
        if prims.is_empty() {
            self.packed = prims;
            return;
        }
        let n = prims.len();
        self.build_node(&mut prims, 0, n);
        self.packed = prims;
    }

    /// Binned SAH split (16 bins on the widest centroid axis); falls back
    /// to a median split when SAH finds no paying cut. Tree quality is what
    /// keeps traversal cheap — see mesh_bench.
    fn build_node(&mut self, prims: &mut [PackedTri], start: usize, end: usize) -> u32 {
        const NBIN: usize = 16;
        let pbounds = |p: &PackedTri| {
            let (a, b, c) = (p.a, p.a + p.e1, p.a + p.e2);
            (a.min(b).min(c), a.max(b).max(c))
        };
        let pcentroid = |p: &PackedTri| p.a + (p.e1 + p.e2) * (1.0 / 3.0);
        let area = |lo: Vec3, hi: Vec3| {
            let e = (hi - lo).max(Vec3::new(0.0, 0.0, 0.0));
            2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
        };

        let mut bmin = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut bmax = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        let mut cmin = bmin;
        let mut cmax = bmax;
        for t in &prims[start..end] {
            let (lo, hi) = pbounds(t);
            bmin = bmin.min(lo);
            bmax = bmax.max(hi);
            let c = pcentroid(t);
            cmin = cmin.min(c);
            cmax = cmax.max(c);
        }
        let idx = self.nodes.len() as u32;
        self.nodes.push(BvhNode { bmin, bmax, a: 0, b: 0 });

        let count = end - start;
        if count <= LEAF_SIZE {
            self.nodes[idx as usize].a = LEAF_BIT | start as u32;
            self.nodes[idx as usize].b = count as u32;
            return idx;
        }

        let cext = cmax - cmin;
        let axis = if cext.x >= cext.y && cext.x >= cext.z {
            0
        } else if cext.y >= cext.z {
            1
        } else {
            2
        };
        let cwidth = cext.get(axis);
        let mid = if cwidth < 1e-6 {
            start + count / 2 // all centroids coincide: arbitrary split
        } else {
            // bin by centroid, evaluate SAH cost at each of the 15 cuts
            let bin_of = |p: &PackedTri| {
                ((pcentroid(p).get(axis) - cmin.get(axis)) / cwidth * NBIN as f32)
                    .min(NBIN as f32 - 1.0) as usize
            };
            let mut bn = [(Vec3::new(f32::MAX, f32::MAX, f32::MAX), Vec3::new(f32::MIN, f32::MIN, f32::MIN), 0u32); NBIN];
            for t in &prims[start..end] {
                let k = bin_of(t);
                let (lo, hi) = pbounds(t);
                bn[k].0 = bn[k].0.min(lo);
                bn[k].1 = bn[k].1.max(hi);
                bn[k].2 += 1;
            }
            // sweep: prefix/suffix areas
            let mut right_cost = [0.0f32; NBIN];
            let mut acc_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            let mut acc_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
            let mut acc_n = 0u32;
            for k in (1..NBIN).rev() {
                acc_min = acc_min.min(bn[k].0);
                acc_max = acc_max.max(bn[k].1);
                acc_n += bn[k].2;
                right_cost[k - 1] = if acc_n > 0 { area(acc_min, acc_max) * acc_n as f32 } else { 0.0 };
            }
            let mut best_cut = usize::MAX;
            let mut best_cost = f32::MAX;
            acc_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            acc_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
            acc_n = 0;
            for k in 0..NBIN - 1 {
                acc_min = acc_min.min(bn[k].0);
                acc_max = acc_max.max(bn[k].1);
                acc_n += bn[k].2;
                if acc_n == 0 || acc_n == count as u32 {
                    continue;
                }
                let cost = area(acc_min, acc_max) * acc_n as f32 + right_cost[k];
                if cost < best_cost {
                    best_cost = cost;
                    best_cut = k;
                }
            }
            if best_cut == usize::MAX {
                start + count / 2
            } else {
                // stable two-pointer partition by bin
                let slice = &mut prims[start..end];
                let mut i = 0usize;
                let mut j = slice.len();
                while i < j {
                    if bin_of(&slice[i]) <= best_cut {
                        i += 1;
                    } else {
                        j -= 1;
                        slice.swap(i, j);
                    }
                }
                start + i
            }
        };
        let mid = mid.clamp(start + 1, end - 1);

        let left = self.build_node(prims, start, mid);
        let right = self.build_node(prims, mid, end);
        self.nodes[idx as usize].a = left;
        self.nodes[idx as usize].b = right;
        idx
    }

    /// Slab test returning the entry distance (0 if starting inside).
    #[inline(always)]
    fn ray_aabb(bmin: Vec3, bmax: Vec3, o: Vec3, inv_d: Vec3, tmax: f32) -> Option<f32> {
        let mut t0 = 0.0f32;
        let mut t1 = tmax;
        for axis in 0..3 {
            let inv = inv_d.get(axis);
            let mut ta = (bmin.get(axis) - o.get(axis)) * inv;
            let mut tb = (bmax.get(axis) - o.get(axis)) * inv;
            if ta > tb {
                core::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return None;
            }
        }
        Some(t0)
    }

    /// Möller–Trumbore, both faces. Returns t. (Brute-force reference for
    /// the BVH equivalence test.)
    #[cfg(test)]
    fn ray_tri(&self, tri: u32, o: Vec3, d: Vec3) -> Option<f32> {
        let (a, b, c) = self.tri_verts(tri);
        Self::ray_tri_packed(a, b - a, c - a, o, d)
    }

    #[inline(always)]
    fn ray_tri_packed(a: Vec3, e1: Vec3, e2: Vec3, o: Vec3, d: Vec3) -> Option<f32> {
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-9 {
            return None;
        }
        let inv = 1.0 / det;
        let s = o - a;
        let u = s.dot(p) * inv;
        if !(-1e-6..=1.0 + 1e-6).contains(&u) {
            return None;
        }
        let q = s.cross(e1);
        let v = d.dot(q) * inv;
        if v < -1e-6 || u + v > 1.0 + 1e-6 {
            return None;
        }
        let t = e2.dot(q) * inv;
        (t > 1e-5).then_some(t)
    }

    pub fn tri_normal(&self, tri: u32) -> Vec3 {
        let (a, b, c) = self.tri_verts(tri);
        (b - a).cross(c - a).normalize()
    }

    /// Nearest hit along a normalized ray. Also see `AcousticGeometry`.
    pub fn raycast(&self, o: Vec3, d: Vec3) -> Option<(f32, u32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_d = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        let mut best: Option<(f32, u32)> = None;
        // Ordered traversal: (node, entry distance), nearer child popped
        // first so `best` shrinks early and culls the far subtree.
        let mut stack = [(0u32, 0.0f32); 64];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let (ni, t_enter) = stack[sp];
            if best.is_some_and(|(bt, _)| t_enter >= bt) {
                continue;
            }
            let node = &self.nodes[ni as usize];
            if node.a & LEAF_BIT != 0 {
                let start = (node.a & !LEAF_BIT) as usize;
                for pt in &self.packed[start..start + node.b as usize] {
                    if let Some(t) = Self::ray_tri_packed(pt.a, pt.e1, pt.e2, o, d) {
                        if best.is_none_or(|(bt, _)| t < bt) {
                            best = Some((t, pt.tri));
                        }
                    }
                }
            } else {
                let tmax = best.map_or(f32::MAX, |(t, _)| t);
                let l = &self.nodes[node.a as usize];
                let r = &self.nodes[node.b as usize];
                let tl = Self::ray_aabb(l.bmin, l.bmax, o, inv_d, tmax);
                let tr = Self::ray_aabb(r.bmin, r.bmax, o, inv_d, tmax);
                match (tl, tr) {
                    (Some(a), Some(b)) => {
                        // push far first, near on top
                        let (near, nt, far, ft) = if a <= b {
                            (node.a, a, node.b, b)
                        } else {
                            (node.b, b, node.a, a)
                        };
                        stack[sp] = (far, ft);
                        stack[sp + 1] = (near, nt);
                        sp += 2;
                    }
                    (Some(a), None) => {
                        stack[sp] = (node.a, a);
                        sp += 1;
                    }
                    (None, Some(b)) => {
                        stack[sp] = (node.b, b);
                        sp += 1;
                    }
                    (None, None) => {}
                }
            }
        }
        best
    }

    /// Every crossing of segment a→b strictly between its endpoints,
    /// sorted by t — the transmission query (each hit is a surface the
    /// sound must pass through). `out` is reused by callers.
    pub fn segment_hits(&self, a: Vec3, b: Vec3, out: &mut Vec<SegHit>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let d = b - a;
        let len = d.length();
        if len < 1e-6 {
            return;
        }
        let dn = d * (1.0 / len);
        let inv_d = Vec3::new(1.0 / dn.x, 1.0 / dn.y, 1.0 / dn.z);
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if Self::ray_aabb(node.bmin, node.bmax, a, inv_d, len).is_none() {
                continue;
            }
            if node.a & LEAF_BIT != 0 {
                let start = (node.a & !LEAF_BIT) as usize;
                for pt in &self.packed[start..start + node.b as usize] {
                    if let Some(t) = Self::ray_tri_packed(pt.a, pt.e1, pt.e2, a, dn) {
                        let tt = t / len;
                        if tt > 1e-4 && tt < 1.0 - 1e-4 {
                            out.push(SegHit { t: tt, tri: pt.tri, material: pt.material });
                        }
                    }
                }
            } else {
                stack[sp] = node.a;
                stack[sp + 1] = node.b;
                sp += 2;
            }
        }
        out.sort_by(|x, y| x.t.total_cmp(&y.t));
        // Same-t dedup: a crossing on the shared edge/diagonal of coplanar
        // triangles is ONE physical surface, not two.
        out.dedup_by(|next, kept| (next.t - kept.t).abs() < 1e-4 / len.max(1.0));
    }
}

impl AcousticGeometry for Mesh {
    fn raycast_hit(&self, p: Vec3, d: Vec3) -> Option<GeomHit> {
        let (t, tri) = self.raycast(p, d)?;
        let mut normal = self.tri_normal(tri);
        if normal.dot(d) > 0.0 {
            normal = normal * -1.0; // orient against the ray
        }
        Some(GeomHit {
            t,
            normal,
            material: self.materials[self.tri_material[tri as usize] as usize],
        })
    }
}

// ------------------------------------------------- diffraction edges

/// A candidate diffraction edge (sharp crease or boundary), with an
/// importance score for budgeted selection.
#[derive(Clone, Copy)]
pub struct DiffractionEdge {
    pub a: Vec3,
    pub b: Vec3,
    /// Static perceptual importance: edge length × how sharp the crease
    /// is (a 90° building corner ranks above a 20° facade kink). Runtime
    /// ranking multiplies in source/listener proximity.
    pub importance: f32,
}

/// Extract diffraction edges: every mesh edge whose two faces meet at a
/// dihedral angle ≥ `min_dihedral_deg`, plus boundary edges (single
/// face). Vertex positions are matched exactly (MeshBuilder dedups).
pub fn extract_edges(mesh: &Mesh, min_dihedral_deg: f32) -> Vec<DiffractionEdge> {
    use std::collections::HashMap;
    let mut by_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (ti, idx) in mesh.indices.iter().enumerate() {
        for k in 0..3 {
            let (i, j) = (idx[k], idx[(k + 1) % 3]);
            let key = (i.min(j), i.max(j));
            by_edge.entry(key).or_default().push(ti as u32);
        }
    }
    let mut edges = Vec::new();
    for ((i, j), tris) in by_edge {
        let a = mesh.positions[i as usize];
        let b = mesh.positions[j as usize];
        let len = (b - a).length();
        if len < 1e-4 {
            continue;
        }
        let sharpness = match tris.as_slice() {
            [_] => 1.0, // boundary edge: maximally exposed
            [t0, t1] => {
                let n0 = mesh.tri_normal(*t0);
                let n1 = mesh.tri_normal(*t1);
                let angle = n0.dot(n1).clamp(-1.0, 1.0).acos().to_degrees();
                if angle < min_dihedral_deg {
                    continue; // near-coplanar: not a diffracting crease
                }
                angle / 180.0
            }
            _ => continue, // non-manifold junction: skip
        };
        edges.push(DiffractionEdge { a, b, importance: len * sharpness });
    }
    edges
}

/// The perceptual budget: keep the `max` most important edges. This is
/// what keeps arbitrary geometry real-time on any tier — dense meshes
/// degrade by dropping short/shallow creases first, never by breaking.
pub fn budget_edges(mut edges: Vec<DiffractionEdge>, max: usize) -> Vec<DiffractionEdge> {
    edges.sort_by(|x, y| y.importance.total_cmp(&x.importance));
    edges.truncate(max);
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn box_mesh(size: Vec3) -> Mesh {
        let mut mb = MeshBuilder::new();
        let m = mb.material(Material::CONCRETE);
        mb.abox(Vec3::new(0.0, 0.0, 0.0), size, m);
        mb.build()
    }

    #[test]
    fn bvh_raycast_matches_brute_force() {
        // random triangle soup
        let mut rng = Rng::new(7);
        let mut mb = MeshBuilder::new();
        let m = mb.material(Material::CONCRETE);
        for _ in 0..200 {
            let base = Vec3::new(
                rng.next_f32() * 20.0,
                rng.next_f32() * 20.0,
                rng.next_f32() * 20.0,
            );
            mb.tri(
                base,
                base + rng.unit_sphere() * 2.0,
                base + rng.unit_sphere() * 2.0,
                m,
            );
        }
        let mesh = mb.build();
        for _ in 0..500 {
            let o = Vec3::new(
                rng.next_f32() * 20.0,
                rng.next_f32() * 20.0,
                rng.next_f32() * 20.0,
            );
            let d = rng.unit_sphere();
            let bvh = mesh.raycast(o, d);
            let brute = (0..mesh.tri_count() as u32)
                .filter_map(|t| mesh.ray_tri(t, o, d).map(|tt| (tt, t)))
                .min_by(|x, y| x.0.total_cmp(&y.0));
            match (bvh, brute) {
                (Some((tb, _)), Some((tr, _))) => {
                    assert!((tb - tr).abs() < 1e-4, "bvh {tb} vs brute {tr}")
                }
                (None, None) => {}
                other => panic!("bvh/brute disagree: {other:?}"),
            }
        }
    }

    #[test]
    fn box_mesh_traces_like_shoebox() {
        use crate::scene::Shoebox;
        use crate::tracer::{estimate_reverb, trace, Echogram};
        let size = Vec3::new(8.0, 6.0, 3.0);
        let walls = [Material::CONCRETE; 6];
        let sbox = Shoebox::new(size, walls);
        let mesh = box_mesh(size);
        let src = Vec3::new(2.0, 3.0, 1.5);
        let lis = Vec3::new(6.0, 2.0, 1.6);

        let mut rng = Rng::new(42);
        let mut e_box = Echogram::new();
        trace(&sbox, src, lis, 8192, [1.0; 3], &mut rng, &mut e_box);
        let mut rng = Rng::new(42);
        let mut e_mesh = Echogram::new();
        trace(&mesh, src, lis, 8192, [1.0; 3], &mut rng, &mut e_mesh);

        let rb = estimate_reverb(&e_box);
        let rm = estimate_reverb(&e_mesh);
        for b in 0..3 {
            let ratio = rm.rt60[b] / rb.rt60[b];
            assert!(
                (0.8..1.25).contains(&ratio),
                "band {b}: mesh rt60 {} vs shoebox {}",
                rm.rt60[b],
                rb.rt60[b]
            );
            let lr = rm.level[b] / rb.level[b].max(1e-9);
            assert!((0.7..1.4).contains(&lr), "band {b}: level ratio {lr}");
        }
    }

    #[test]
    fn segment_hits_counts_wall_crossings() {
        let mesh = box_mesh(Vec3::new(4.0, 4.0, 3.0));
        let mut hits = Vec::new();
        // through the whole box: enter + exit
        mesh.segment_hits(Vec3::new(-2.0, 2.0, 1.5), Vec3::new(6.0, 2.0, 1.5), &mut hits);
        assert_eq!(hits.len(), 2, "expected enter+exit");
        assert!(hits[0].t < hits[1].t);
        // endpoint inside: only the entry wall
        mesh.segment_hits(Vec3::new(-2.0, 2.0, 1.5), Vec3::new(2.0, 2.0, 1.5), &mut hits);
        assert_eq!(hits.len(), 1);
        // fully inside: nothing
        mesh.segment_hits(Vec3::new(1.0, 2.0, 1.5), Vec3::new(3.0, 2.0, 1.5), &mut hits);
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn box_has_twelve_sharp_edges_and_budget_ranks() {
        let mesh = box_mesh(Vec3::new(8.0, 6.0, 3.0));
        let edges = extract_edges(&mesh, 40.0);
        assert_eq!(edges.len(), 12, "a box has 12 creases");
        let top = budget_edges(edges, 4);
        assert_eq!(top.len(), 4);
        // the 4 most important edges of an 8×6×3 box are the 8 m ones
        for e in &top {
            assert!((e.b - e.a).length() > 5.9, "budget kept a short edge");
        }
    }
}
