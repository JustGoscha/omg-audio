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
    /// Strict TAP semantics: paths become ADDITIVE audio taps, so a
    /// near-straight corner "bend" (deflection < ~6°) is rejected and
    /// legs are priced extended past their apex (coincident-plane
    /// grazing pays its wall). The default (false) keeps the classic
    /// occlusion-FLOOR semantics, where grazing corner paths carry the
    /// designed −5 dB boundary continuity under a max().
    pub taps: bool,
}

impl Default for PathBudget {
    fn default() -> Self {
        Self { edge_candidates: 24, pair_edges: 8, max_paths: 4, taps: false }
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
        segment_transmission(mesh, a, b, &mut self.hits)
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
            // TAP semantics: legs price EXTENDED slightly past the apex
            // (an apex ON a coincident wall plane otherwise grazes that
            // wall at t = 1 and pays nothing — the sealed-club "sharp
            // pocket" bug); a true silhouette edge has air beyond it and
            // is unaffected. Floor semantics keep exact legs.
            let (e1, e2) = if budget.taps {
                // both legs meet 3 cm off the edge on the OUTSIDE of the
                // bend: a silhouette edge has air there, so neither leg
                // grazes it (a grazing leg read as hit or miss depending
                // on float detail in the BVH); an apex ON a coincident
                // wall plane still makes a leg cross that wall
                let q = outside_of_bend(p, src, lis, 0.03);
                (q, q)
            } else {
                (p, p)
            };
            let l1 = self.transmission(mesh, src, e1);
            let l2 = self.transmission(mesh, e2, lis);
            if l1[0] * l2[0] < 1e-4 {
                continue;
            }
            let detour = d1 + d2 - (lis - src).length();
            let ke = knife_edge_bands(detour.max(1e-4));
            let floor = diffraction_floor(d1 + d2);
            let mut gains = [0.0f32; NBANDS];
            for band in 0..NBANDS {
                gains[band] = ke[band].max(floor) * l1[band] * l2[band];
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
        // One pool over the WHOLE corridor, paired in order along it: a
        // fixed source-half × listener-half split cannot pair two
        // obstacles that both stand near one end — the normal case for a
        // distant source (a bell 300 m out, rounding a cathedral corner
        // and then a hall corner within the last 60 m).
        let pool = rank_corridor(&self.edges, 0.0, 1.0, 2 * budget.pair_edges);
        let along = |i: usize| {
            let c = (self.edges[i].a + self.edges[i].b) * 0.5;
            (c - src).dot(seg) / seg_len2
        };
        let mut near_src: Vec<usize> = Vec::new();
        let mut near_lis: Vec<usize> = Vec::new();
        for &i in &pool {
            near_src.push(i);
            near_lis.push(i);
        }
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
                // entry bend before exit bend along the corridor
                if ei == ej || along(ei) > along(ej) + 0.02 {
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
                // same endpoint-grazing rule as single bends (tap
                // semantics only)
                let (x1, m1, m2, x2) = if budget.taps {
                    let q1 = outside_of_bend(p1, src, p2, 0.03);
                    let q2 = outside_of_bend(p2, p1, lis, 0.03);
                    (q1, q1, q2, q2)
                } else {
                    (p1, p1, p2, p2)
                };
                let l1 = self.transmission(mesh, src, x1);
                let lm = self.transmission(mesh, m1, m2);
                let l2 = self.transmission(mesh, x2, lis);
                if l1[0] * lm[0] * l2[0] < 1e-4 {
                    continue;
                }
                // rubber-band detours per vertex
                let det1 = d1 + dm - (p2 - src).length();
                let det2 = dm + d2 - (lis - p1).length();
                let ke1 = knife_edge_bands(det1.max(1e-4));
                let ke2 = knife_edge_bands(det2.max(1e-4));
                let floor = diffraction_floor(d1 + dm + d2);
                let mut gains = [0.0f32; NBANDS];
                for band in 0..NBANDS {
                    gains[band] =
                        (ke1[band] * ke2[band]).max(floor) * l1[band] * lm[band] * l2[band];
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
        let line_z = |t: f32| src.z + (lis.z - src.z) * t;
        let top_at = |t: f32| -> Option<f32> {
            let x = src.x + flat.x * t;
            let y = src.y + flat.y * t;
            mesh.raycast(Vec3::new(x, y, 60.0), down).map(|(th, _)| 60.0 - th)
        };
        // Coarse pass: where does geometry poke above the sight line?
        let mut pokes: Vec<f32> = Vec::new();
        for i in 1..K {
            let t = i as f32 / K as f32;
            if let Some(top) = top_at(t) {
                if top > line_z(t) + 0.05 {
                    pokes.push(run * t);
                }
            }
        }
        if pokes.is_empty() {
            return None; // nothing pokes above the line
        }
        let mut pts: Vec<(f32, f32)> = vec![(0.0, src.z)];
        for &sp in &pokes {
            if let Some(top) = top_at(sp / run) {
                pts.push((sp, top + 0.03));
            }
        }
        pts.push((run, lis.z));
        // The coarse probes only prove an obstacle is NEAR: its true edges,
        // and thin features (a 0.4 m parapet) can fall between probes, so a
        // string built on them clips roof corners. Refine on violations:
        // wherever a leg passes through geometry, the taut string goes
        // OVER that column's top — probe the exact column and rebuild.
        // This converges: what survives is a crossing straight below a
        // string vertex, i.e. a real one (a drop through a roof into a
        // house, through the earth into a bunker), priced below.
        let mut hull: Vec<(f32, f32)> = Vec::new();
        for _round in 0..24 {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            hull = upper_hull(&pts);
            let mut added = false;
            for w in hull.windows(2) {
                let (a, b) = (w[0], w[1]);
                let pa = Vec3::new(src.x + flat.x * a.0 / run, src.y + flat.y * a.0 / run, a.1);
                let pb = Vec3::new(src.x + flat.x * b.0 / run, src.y + flat.y * b.0 / run, b.1);
                mesh.segment_hits(pa, pb, &mut self.hits);
                for h in self.hits.iter() {
                    let sv = a.0 + (b.0 - a.0) * h.t;
                    let zl = a.1 + (b.1 - a.1) * h.t;
                    if sv <= 1e-3 || sv >= run - 1e-3 {
                        continue;
                    }
                    // a hit on a vertical face sits exactly on a column
                    // edge: probe a hair to either side, take the higher
                    let mut top = [sv - 0.05, sv, sv + 0.05]
                        .iter()
                        .filter_map(|&q| top_at(q.clamp(0.0, run) / run))
                        .fold(None, |m: Option<f32>, t| Some(m.map_or(t, |m| m.max(t))));
                    // A thin VERTICAL surface (a parapet band, a wall with
                    // no modeled top) has no face for a downward probe to
                    // land on: the probe slips past to the slab below. Its
                    // top edge is the struck triangle's highest vertex.
                    if let Some(tri) = mesh.indices.get(h.tri as usize) {
                        let [v0, v1, v2] = tri.map(|i| mesh.positions[i as usize]);
                        let n = (v1 - v0).cross(v2 - v0);
                        if n.z.abs() < 0.3 * n.length() {
                            let edge = v0.z.max(v1.z).max(v2.z);
                            top = Some(top.map_or(edge, |t| t.max(edge)));
                        }
                    }
                    if let Some(top) = top {
                        let want = top + 0.03;
                        if want > zl {
                            match pts.iter_mut().find(|p| (p.0 - sv).abs() < 0.02) {
                                // a probe already stands here, too low
                                // (it saw the slab, not the parapet): raise it
                                Some(p) if p.1 < want - 1e-3 => {
                                    p.1 = want;
                                    added = true;
                                }
                                Some(_) => {}
                                None => {
                                    pts.push((sv, want));
                                    added = true;
                                }
                            }
                        }
                    }
                }
            }
            if !added {
                break;
            }
        }
        if hull.len() < 3 {
            return None;
        }
        let d2 = |a: (f32, f32), b: (f32, f32)| {
            ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
        };
        let mut length = 0.0f32;
        for w in hull.windows(2) {
            length += d2(w[0], w[1]);
        }
        // the whole string priced once, by its total detour (ISO 9613-2)
        let n = hull.len();
        let direct = d2(hull[0], hull[n - 1]);
        let mut gains = crate::diffraction::iso9613_barrier_bands(
            length - direct,
            n - 2,
            d2(hull[1], hull[n - 2]),
            d2(hull[0], hull[1]),
            d2(hull[n - 2], hull[n - 1]),
            direct,
            false,
        );
        let floor = diffraction_floor(length);
        for band in 0..NBANDS {
            gains[band] = gains[band].max(floor);
        }
        let points: Vec<Vec3> = hull
            .iter()
            .map(|&(sv, z)| {
                let t = sv / run;
                Vec3::new(src.x + flat.x * t, src.y + flat.y * t, z)
            })
            .collect();
        // The string clears every TOP it probed — but a leg can still pass
        // THROUGH geometry to reach an enclosed end: a listener inside a
        // house (down through the roof) or below grade (down through the
        // earth). Price every leg like the edge paths price theirs.
        for w in points.windows(2) {
            let tr = self.transmission(mesh, w[0], w[1]);
            for band in 0..NBANDS {
                gains[band] *= tr[band];
            }
        }
        if gains[0] < 1e-6 {
            return None;
        }
        Some(FoundPath { points, length, gains, key: 90_000 })
    }
}

/// Per-band amplitude of the straight segment a→b through the mesh.
/// Surface crossings are paired (entry, exit) so wall thickness — and
/// with it the mass law — comes from the actual geometry. `hits` is
/// scratch space.
pub fn segment_transmission(mesh: &Mesh, a: Vec3, b: Vec3, hits: &mut Vec<crate::mesh::SegHit>) -> [f32; NBANDS] {
    mesh.segment_hits(a, b, hits);
    let len = (b - a).length();
    let mut g = [1.0f32; NBANDS];
    let mut i = 0;
    while i < hits.len() {
        let enter = hits[i];
        let mat = mesh.materials[enter.material as usize];
        // Exit face: next crossing of the SAME material — the far side
        // of this wall. An unpaired crossing (thin shell, segment ends
        // inside) falls back to a nominal thickness.
        let thickness = if i + 1 < hits.len() && hits[i + 1].material == enter.material {
            i += 1;
            ((hits[i].t - enter.t) * len).max(0.01)
        } else {
            0.1
        };
        // an authored thickness wins over the geometry's (one wall either way)
        let thickness = mesh.material_thickness(enter.material).unwrap_or(thickness);
        let tr = mat.transmission_at(thickness);
        for band in 0..NBANDS {
            g[band] *= tr[band];
        }
        i += 1;
    }
    g
}

/// Deepest diffraction loss (amplitude) a bent path of total length
/// `length_m` may carry. Outdoors, turbulence and scattering pour energy
/// into every geometric shadow, so a barrier never shields as deeply as
/// the knife-edge law alone predicts — ISO 9613-2 caps barrier
/// attenuation at 20 dB (single) / 25 dB (multiple diffraction). The cap
/// fades in with distance (≈ 25 dB beyond ~100 m, ≈ 50 dB at 10 m), so
/// short indoor bends around a door jamb keep their full depth.
pub fn diffraction_floor(length_m: f32) -> f32 {
    let max_db = 25.0 + 40.0 * (-length_m.max(0.0) / 20.0).exp();
    10f32.powf(-max_db / 20.0)
}

/// Shortest path in PLAN from `a` to `b` around axis-aligned rectangles
/// `([min_x, min_y], [max_x, max_y])` — the taut string around building
/// footprints, the sideways twin of the over-the-silhouette string.
/// Visibility graph over the rectangle corners + Dijkstra. Returns the
/// polyline a → corners… → b, or None when the straight line is already
/// clear or no way around exists. Touching a rectangle's boundary does
/// not block (strings run along faces and wrap corners).
pub fn plan_detour(a: (f32, f32), b: (f32, f32), rects: &[([f32; 2], [f32; 2])]) -> Option<Vec<(f32, f32)>> {
    let blocked = |p: (f32, f32), q: (f32, f32)| {
        rects.iter().any(|(mn, mx)| seg_crosses_rect(p, q, *mn, *mx))
    };
    if !blocked(a, b) {
        return None;
    }
    let inside = |p: (f32, f32)| {
        rects.iter().any(|(mn, mx)| {
            p.0 > mn[0] + 1e-3 && p.0 < mx[0] - 1e-3 && p.1 > mn[1] + 1e-3 && p.1 < mx[1] - 1e-3
        })
    };
    let mut nodes: Vec<(f32, f32)> = vec![a, b];
    for (mn, mx) in rects {
        for c in [(mn[0], mn[1]), (mx[0], mn[1]), (mx[0], mx[1]), (mn[0], mx[1])] {
            if !inside(c) {
                nodes.push(c);
            }
        }
    }
    let n = nodes.len();
    let dist = |i: usize, j: usize| {
        ((nodes[i].0 - nodes[j].0).powi(2) + (nodes[i].1 - nodes[j].1).powi(2)).sqrt()
    };
    let mut best = vec![f32::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut done = vec![false; n];
    best[0] = 0.0;
    loop {
        let mut u = usize::MAX;
        for i in 0..n {
            if !done[i] && best[i].is_finite() && (u == usize::MAX || best[i] < best[u]) {
                u = i;
            }
        }
        if u == usize::MAX || u == 1 {
            break;
        }
        done[u] = true;
        for v in 0..n {
            if done[v] || v == u {
                continue;
            }
            let nd = best[u] + dist(u, v);
            if nd < best[v] && !blocked(nodes[u], nodes[v]) {
                best[v] = nd;
                prev[v] = u;
            }
        }
    }
    if !best[1].is_finite() {
        return None;
    }
    let mut path = vec![nodes[1]];
    let mut i = 1;
    while prev[i] != usize::MAX {
        i = prev[i];
        path.push(nodes[i]);
    }
    path.reverse();
    Some(path)
}

/// Does segment p→q pass through the OPEN interior of the rectangle?
fn seg_crosses_rect(p: (f32, f32), q: (f32, f32), mn: [f32; 2], mx: [f32; 2]) -> bool {
    const E: f32 = 1e-3;
    let (x0, y0, x1, y1) = (mn[0] + E, mn[1] + E, mx[0] - E, mx[1] - E);
    if x0 >= x1 || y0 >= y1 {
        return false;
    }
    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (pp, qq) in [(-dx, p.0 - x0), (dx, x1 - p.0), (-dy, p.1 - y0), (dy, y1 - p.1)] {
        if pp.abs() < 1e-9 {
            if qq < 0.0 {
                return false;
            }
        } else {
            let r = qq / pp;
            if pp < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return false;
            }
        }
    }
    t1 - t0 > 1e-6
}

/// `p` nudged `d` meters to the outside of the bend a → p → b (the side
/// away from whatever the path bends around).
fn outside_of_bend(p: Vec3, a: Vec3, b: Vec3, d: f32) -> Vec3 {
    let out = (p - a).normalize() - (b - p).normalize();
    if out.length() < 1e-4 {
        return p;
    }
    p + out.normalize() * d
}

/// Upper convex hull of (s, z) points sorted by s: the taut string.
fn upper_hull(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
    for &p in pts {
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
    hull
}

/// The point on segment [a, b] minimizing |src→p| + |p→lis| (the physical
/// bend point on a diffracting edge). Convex in the parameter — golden
/// section on t ∈ [0, 1].
pub fn bend_point(a: Vec3, b: Vec3, src: Vec3, lis: Vec3) -> Vec3 {
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
