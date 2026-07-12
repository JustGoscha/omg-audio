//! Automatic propagation paths on a raw triangle mesh — zero authoring.
//! No rooms, no doors, no portals: the scene mesh is the acoustic model.
//!
//!  - Direct: one segment query. Wall thickness is EMERGENT — real walls
//!    are two parallel faces, so pairing entry/exit crossings yields the
//!    thickness the mass law needs.
//!  - Bent paths: a search over the auto-extracted diffraction edges
//!    (`mesh::extract_edges`), priced by the Kurze–Anderson kernel. A
//!    door jamb, a building corner and a roof line are the same thing
//!    here: an edge.
//!  - Budgets everywhere: edge candidates are ranked by static importance
//!    × proximity to the blocked corridor, and the caller caps the number
//!    of returned paths — perceptually unimportant paths drop first.

use crate::diffraction::knife_edge_bands;
use crate::mesh::{budget_edges, extract_edges, DiffractionEdge, Mesh, SegHit};
use crate::vec3::Vec3;
use crate::NBANDS;

/// Per-query knobs — the simulation-LOD surface. Scale these down with a
/// source's perceptual weight (distance-attenuated loudness).
#[derive(Clone, Copy)]
pub struct PathBudget {
    /// How many edges to try as single-bend candidates.
    pub edge_candidates: usize,
    /// Double-bend pairs: top `pair_edges` near the source × top
    /// `pair_edges` near the listener (deep obstacles — around a building
    /// or over its roof — need two bends; a single bend only clears thin
    /// ones).
    pub pair_edges: usize,
    /// How many bent paths to return (strongest first).
    pub max_paths: usize,
}

impl Default for PathBudget {
    fn default() -> Self {
        Self { edge_candidates: 24, pair_edges: 8, max_paths: 4 }
    }
}

/// One found propagation path from source to listener.
pub struct FoundPath {
    /// src, bend points…, listener.
    pub points: Vec<Vec3>,
    /// Total path length (meters).
    pub length: f32,
    /// Per-band amplitude of everything the geometry did to it
    /// (transmission through surfaces × knife-edge bends). Spreading, air
    /// absorption and delay are the caller's (they depend on rendering).
    pub gains: [f32; NBANDS],
    /// Stable identity for renderer tap keys: 0 = direct,
    /// 1 + edge index for single-bend paths.
    pub key: u32,
}

pub struct AutoPaths {
    edges: Vec<DiffractionEdge>,
    /// Scratch buffers (queries are &mut and alloc-free after warmup).
    hits: Vec<SegHit>,
}

impl AutoPaths {
    /// Extract and budget the static edge graph. `edge_budget` caps the
    /// graph size for dense meshes (importance-ranked).
    pub fn new(mesh: &Mesh, edge_budget: usize) -> Self {
        Self {
            edges: budget_edges(extract_edges(mesh, 40.0), edge_budget),
            hits: Vec::new(),
        }
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Per-band amplitude of the straight segment a→b through the mesh.
    /// Surface crossings are paired (entry, exit) so wall thickness — and
    /// with it the mass law — comes from the actual geometry.
    pub fn transmission(&mut self, mesh: &Mesh, a: Vec3, b: Vec3) -> [f32; NBANDS] {
        mesh.segment_hits(a, b, &mut self.hits);
        let len = (b - a).length();
        let mut g = [1.0f32; NBANDS];
        let mut i = 0;
        while i < self.hits.len() {
            let enter = self.hits[i];
            let mat = mesh.materials[enter.material as usize];
            // Exit face: next crossing of the SAME material — the far side
            // of this wall. An unpaired crossing (thin shell, segment ends
            // inside) falls back to a nominal thickness.
            let thickness = if i + 1 < self.hits.len()
                && self.hits[i + 1].material == enter.material
            {
                i += 1;
                ((self.hits[i].t - enter.t) * len).max(0.01)
            } else {
                0.1
            };
            let tr = mat.transmission_at(thickness);
            for band in 0..NBANDS {
                g[band] *= tr[band];
            }
            i += 1;
        }
        g
    }

    /// Find propagation paths src → lis: the direct segment plus, when it
    /// is meaningfully blocked, the strongest bent paths over the edge
    /// graph. Results are sorted strongest-first, capped by the budget.
    pub fn find(
        &mut self,
        mesh: &Mesh,
        src: Vec3,
        lis: Vec3,
        budget: PathBudget,
        out: &mut Vec<FoundPath>,
    ) {
        out.clear();
        let direct_g = self.transmission(mesh, src, lis);
        let blocked = direct_g[1] < 0.25; // mid band audibly obstructed
        out.push(FoundPath {
            points: vec![src, lis],
            length: (lis - src).length(),
            gains: direct_g,
            key: 0,
        });
        if !blocked {
            return;
        }

        // Rank edges by static importance over distance to the CORRIDOR
        // (the blocked segment itself): the edges that matter silhouette
        // it — endpoint-distance ranking let a facade's window-hole edges
        // crowd out the alley corners the sound actually rounds.
        let seg = lis - src;
        let seg_len2 = seg.dot(seg).max(1e-6);
        let rank_corridor = |edges: &[DiffractionEdge], lo_t: f32, hi_t: f32, n: usize| {
            let mut r: Vec<(f32, usize)> = edges
                .iter()
                .enumerate()
                .filter_map(|(i, e)| {
                    let c = (e.a + e.b) * 0.5;
                    let t = ((c - src).dot(seg) / seg_len2).clamp(0.0, 1.0);
                    if t < lo_t || t > hi_t {
                        return None;
                    }
                    let on = src + seg * t;
                    let d2 = (c - on).dot(c - on);
                    Some((e.importance / (d2 + 0.5), i))
                })
                .collect();
            r.sort_by(|x, y| y.0.total_cmp(&x.0));
            r.truncate(n);
            r.into_iter().map(|(_, i)| i).collect::<Vec<usize>>()
        };
        // Single bends, ranked by their TRUE knife-edge bound: the bend
        // point is closed-form, so we can price every edge's detour and
        // spend transmission queries on the best — a distance heuristic
        // kept ranking long roof edges over the short vertical corner
        // that actually carried the path. (For dense meshes the corridor
        // prefilter below caps the pool first.)
        let prefilter: Vec<usize> = if self.edges.len() > 1024 {
            rank_corridor(&self.edges, 0.0, 1.0, 1024)
        } else {
            (0..self.edges.len()).collect()
        };
        let mut ranked: Vec<(f32, usize)> = prefilter
            .into_iter()
            .filter_map(|ei| {
                let e = self.edges[ei];
                let p = bend_point(e.a, e.b, src, lis);
                let d1 = (p - src).length();
                let d2 = (lis - p).length();
                if d1 < 0.2 || d2 < 0.2 {
                    return None;
                }
                let detour = d1 + d2 - (lis - src).length();
                Some((knife_edge_bands(detour.max(1e-4))[1], ei))
            })
            .collect();
        ranked.sort_by(|x, y| y.0.total_cmp(&x.0));
        ranked.truncate(budget.edge_candidates);

        let mut bent: Vec<FoundPath> = Vec::new();
        for &(_, ei) in &ranked {
            let e = self.edges[ei];
            let p = bend_point(e.a, e.b, src, lis);
            let d1 = (p - src).length();
            let d2 = (lis - p).length();
            if d1 < 0.2 || d2 < 0.2 {
                continue;
            }
            // Legs PAY their transmission instead of hard-rejecting: a
            // leg that grazes a corner clips a thin sliver of wall and
            // degrades smoothly (mass law), so paths fade at clearance
            // boundaries instead of vanishing — hard thresholds made
            // shadow levels jump as candidate paths popped in and out.
            let l1 = self.transmission(mesh, src, p);
            let l2 = self.transmission(mesh, p, lis);
            if l1[0] * l2[0] < 1e-4 {
                continue;
            }
            let detour = d1 + d2 - (lis - src).length();
            let ke = knife_edge_bands(detour.max(1e-4));
            let mut gains = [0.0f32; NBANDS];
            for band in 0..NBANDS {
                gains[band] = ke[band] * l1[band] * l2[band];
            }
            bent.push(FoundPath {
                points: vec![src, p, lis],
                length: d1 + d2,
                gains,
                key: 1 + ei as u32,
            });
        }
        // Double bends: deep obstacles need an entry and an exit bend.
        // Candidates: edges nearest the source × edges nearest the
        // listener; bend points refined by alternating optimization.
        // entry bends live on the source half of the corridor, exit bends
        // on the listener half (overlapping so mid-corridor edges serve both)
        let near_src = rank_corridor(&self.edges, 0.0, 0.65, budget.pair_edges);
        let near_lis = rank_corridor(&self.edges, 0.35, 1.0, budget.pair_edges);
        // Branch and bound: geometric refinement is cheap (no mesh
        // queries), so refine EVERY pair, bound it by its pure knife-edge
        // product, and spend the expensive transmission queries on the
        // best bounds only. Finds the strong alley pairs a hard NxN
        // evaluation cap used to miss.
        struct PairCand {
            bound: f32,
            p1: Vec3,
            p2: Vec3,
            ei: usize,
            ej: usize,
        }
        let mut cands: Vec<PairCand> = Vec::new();
        for &ei in &near_src {
            for &ej in &near_lis {
                if ei == ej {
                    continue;
                }
                let (e1, e2) = (self.edges[ei], self.edges[ej]);
                // alternate bend-point refinement (converges in a few steps)
                let mut p2 = (e2.a + e2.b) * 0.5;
                let mut p1 = bend_point(e1.a, e1.b, src, p2);
                for _ in 0..3 {
                    p2 = bend_point(e2.a, e2.b, p1, lis);
                    p1 = bend_point(e1.a, e1.b, src, p2);
                }
                let (d1, dm, d2) =
                    ((p1 - src).length(), (p2 - p1).length(), (lis - p2).length());
                if d1 < 0.2 || dm < 0.2 || d2 < 0.2 {
                    continue;
                }
                let det1 = d1 + dm - (p2 - src).length();
                let det2 = dm + d2 - (lis - p1).length();
                let bound =
                    knife_edge_bands(det1.max(1e-4))[1] * knife_edge_bands(det2.max(1e-4))[1];
                cands.push(PairCand { bound, p1, p2, ei, ej });
            }
        }
        cands.sort_by(|x, y| y.bound.total_cmp(&x.bound));
        cands.truncate(64); // transmission-query budget
        for c in cands {
            let (p1, p2, ei, ej) = (c.p1, c.p2, c.ei, c.ej);
            {
                let (d1, dm, d2) =
                    ((p1 - src).length(), (p2 - p1).length(), (lis - p2).length());
                let l1 = self.transmission(mesh, src, p1);
                let lm = self.transmission(mesh, p1, p2);
                let l2 = self.transmission(mesh, p2, lis);
                if l1[0] * lm[0] * l2[0] < 1e-4 {
                    continue;
                }
                // rubber-band detours per vertex
                let det1 = d1 + dm - (p2 - src).length();
                let det2 = dm + d2 - (lis - p1).length();
                let ke1 = knife_edge_bands(det1.max(1e-4));
                let ke2 = knife_edge_bands(det2.max(1e-4));
                let mut gains = [0.0f32; NBANDS];
                for band in 0..NBANDS {
                    gains[band] = ke1[band] * ke2[band] * l1[band] * lm[band] * l2[band];
                }
                bent.push(FoundPath {
                    points: vec![src, p1, p2, lis],
                    length: d1 + dm + d2,
                    gains,
                    key: 100_000 + (ei * 4096 + ej) as u32,
                });
            }
        }

        // Over the silhouette: the taut string over EVERYTHING standing
        // in the segment's vertical plane — one rubber band regardless of
        // how many buildings block (edge pairs cap at two bends; a path
        // over club AND house needs more).
        if let Some(pth) = self.over_silhouette(mesh, src, lis) {
            bent.push(pth);
        }

        bent.sort_by(|x, y| y.gains[1].total_cmp(&x.gains[1]));
        bent.truncate(budget.max_paths);
        out.extend(bent);
    }

    /// Shortest path over the height profile between src and lis: sample
    /// the geometry's top by downward raycasts along the segment, take
    /// the upper convex hull in the vertical plane (the taut string), and
    /// price each hull vertex as a knife edge with its local detour.
    fn over_silhouette(&mut self, mesh: &Mesh, src: Vec3, lis: Vec3) -> Option<FoundPath> {
        const K: usize = 24;
        let flat = Vec3::new(lis.x - src.x, lis.y - src.y, 0.0);
        let run = flat.length();
        if run < 1.0 {
            return None;
        }
        let down = Vec3::new(0.0, 0.0, -1.0);
        // (s along the run, z) profile points that poke above the sight line
        let mut pts: Vec<(f32, f32)> = vec![(0.0, src.z)];
        for i in 1..K {
            let t = i as f32 / K as f32;
            let x = src.x + flat.x * t;
            let y = src.y + flat.y * t;
            let line_z = src.z + (lis.z - src.z) * t;
            if let Some((th, _)) = mesh.raycast(Vec3::new(x, y, 60.0), down) {
                let top = 60.0 - th;
                if top > line_z + 0.05 {
                    pts.push((run * t, top + 0.03));
                }
            }
        }
        if pts.len() == 1 {
            return None; // nothing pokes above the line
        }
        pts.push((run, lis.z));
        // upper convex hull (points are already sorted by s)
        let mut hull: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
        for p in pts {
            while hull.len() >= 2 {
                let (ax, az) = hull[hull.len() - 2];
                let (bx, bz) = hull[hull.len() - 1];
                // keep only left turns looking along +s (upper hull)
                if (bx - ax) * (p.1 - az) - (bz - az) * (p.0 - ax) >= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(p);
        }
        if hull.len() < 3 {
            return None;
        }
        let d2 = |a: (f32, f32), b: (f32, f32)| {
            ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
        };
        let mut gains = [1.0f32; NBANDS];
        let mut length = 0.0f32;
        for w in hull.windows(2) {
            length += d2(w[0], w[1]);
        }
        for i in 1..hull.len() - 1 {
            let detour = d2(hull[i - 1], hull[i]) + d2(hull[i], hull[i + 1])
                - d2(hull[i - 1], hull[i + 1]);
            let ke = knife_edge_bands(detour.max(1e-4));
            for band in 0..NBANDS {
                gains[band] *= ke[band];
            }
        }
        let points = hull
            .iter()
            .map(|&(sv, z)| {
                let t = sv / run;
                Vec3::new(src.x + flat.x * t, src.y + flat.y * t, z)
            })
            .collect();
        Some(FoundPath { points, length, gains, key: 90_000 })
    }
}

/// The point on segment [a, b] minimizing |src→p| + |p→lis| (the physical
/// bend point on a diffracting edge). Convex in the parameter — golden
/// section on t ∈ [0, 1].
fn bend_point(a: Vec3, b: Vec3, src: Vec3, lis: Vec3) -> Vec3 {
    let cost = |t: f32| {
        let p = a + (b - a) * t;
        (p - src).length() + (lis - p).length()
    };
    let phi = 0.618_034f32;
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let (mut m1, mut m2) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
    let (mut c1, mut c2) = (cost(m1), cost(m2));
    for _ in 0..24 {
        if c1 <= c2 {
            hi = m2;
            m2 = m1;
            c2 = c1;
            m1 = hi - phi * (hi - lo);
            c1 = cost(m1);
        } else {
            lo = m1;
            m1 = m2;
            c1 = c2;
            m2 = lo + phi * (hi - lo);
            c2 = cost(m2);
        }
    }
    a + (b - a) * (0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::mesh::MeshBuilder;

    /// A hollow-walled building: outer shell minus inner cavity — real
    /// walls with real thickness, like an imported game level.
    fn building(mb: &mut MeshBuilder, min: Vec3, max: Vec3, t: f32, m: u16) {
        mb.abox(min, max, m);
        mb.abox(
            Vec3::new(min.x + t, min.y + t, min.z + t),
            Vec3::new(max.x - t, max.y - t, max.z - t),
            m,
        );
    }

    #[test]
    fn wall_thickness_is_emergent() {
        let mut mb = MeshBuilder::new();
        let brick = mb.material(Material::BRICK);
        building(&mut mb, Vec3::new(0.0, 0.0, 0.0), Vec3::new(8.0, 6.0, 3.0), 0.24, brick);
        let mesh = mb.build();
        let mut ap = AutoPaths::new(&mesh, 64);

        // through one 0.24 m wall: matches the authored-thickness mass law
        let got = ap.transmission(&mesh, Vec3::new(-2.0, 3.0, 1.5), Vec3::new(2.0, 3.0, 1.5));
        let want = Material::BRICK.transmission_at(0.24);
        for b in 0..NBANDS {
            assert!(
                (got[b] - want[b]).abs() < 0.02,
                "band {b}: emergent {} vs authored {}",
                got[b],
                want[b]
            );
        }
        // through the whole building (two walls): squared
        let got2 = ap.transmission(&mesh, Vec3::new(-2.0, 3.0, 1.5), Vec3::new(10.0, 3.0, 1.5));
        for b in 0..NBANDS {
            assert!(
                (got2[b] - want[b] * want[b]).abs() < 0.02,
                "band {b}: two walls {} vs {}",
                got2[b],
                want[b] * want[b]
            );
        }
    }

    #[test]
    fn bent_paths_found_behind_a_building_no_authoring() {
        let mut mb = MeshBuilder::new();
        let brick = mb.material(Material::BRICK);
        // solid obstacle 10×8×4 between source and listener, on the ground
        // (without a ground slab the search rightly finds paths bending
        // UNDER the building through empty space)
        mb.abox(Vec3::new(10.0, -4.0, 0.0), Vec3::new(20.0, 4.0, 4.0), brick);
        mb.abox(Vec3::new(-40.0, -40.0, -0.5), Vec3::new(60.0, 40.0, 0.0), brick);
        let mesh = mb.build();
        let mut ap = AutoPaths::new(&mesh, 64);

        let src = Vec3::new(5.0, 0.0, 1.5);
        let lis = Vec3::new(25.0, 0.0, 1.6);
        let mut paths = Vec::new();
        let budget = PathBudget { max_paths: 6, ..Default::default() };
        ap.find(&mesh, src, lis, budget, &mut paths);

        // direct is heavily blocked (thick brick), bent paths exist
        assert!(paths[0].gains[1] < 0.1, "direct should be blocked: {:?}", paths[0].gains);
        assert!(paths.len() > 1, "no bent paths found");
        let best = &paths[1];
        assert!(best.gains[0] > 2.0 * best.gains[2], "bends must favor lows: {:?}", best.gains);
        // some path must go over the top (bend point near the roof plane)
        assert!(
            paths[1..].iter().any(|p| p.points[1].z > 3.5),
            "expected an over-the-roof path among {:?}",
            paths[1..].iter().map(|p| p.points[1]).collect::<Vec<_>>()
        );
        // and some path around the side (bend near a vertical edge)
        assert!(
            paths[1..].iter().any(|p| p.points[1].y.abs() > 3.5 && p.points[1].z < 3.5),
            "expected an around-the-side path"
        );
        // bent paths are longer than the straight line
        assert!(best.length > (lis - src).length() + 0.1);
    }

    #[test]
    fn clear_line_returns_only_direct() {
        let mut mb = MeshBuilder::new();
        let m = mb.material(Material::BRICK);
        mb.abox(Vec3::new(0.0, 10.0, 0.0), Vec3::new(4.0, 12.0, 3.0), m); // off to the side
        let mesh = mb.build();
        let mut ap = AutoPaths::new(&mesh, 64);
        let mut paths = Vec::new();
        ap.find(&mesh, Vec3::new(0.0, 0.0, 1.5), Vec3::new(10.0, 0.0, 1.5), PathBudget::default(), &mut paths);
        assert_eq!(paths.len(), 1);
        assert!((paths[0].gains[1] - 1.0).abs() < 1e-6);
    }
}
