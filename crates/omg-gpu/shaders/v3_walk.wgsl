// Engine v3 listener walk (ENGINE_V3.md V3.3, kernel K5) — the GPU port
// of the ray walk inside omg_core::v3::Tracer::trace, diffraction (V3.4)
// included. One thread = one ray of either family: threads [0, n_late)
// are LATE rays (paths beyond `bounces` reflections, walked until 3 s or
// −60 dB — scheduled first, they are the long ones), the rest EARLY rays
// (paths with 0..=bounces reflections). Per segment: the nearest exposed
// edge within 3 m bends the ray (uncertainty-relation spread, per-band
// unbiased weights, Keller's cone); per hit: next-event estimation to
// the sources (importance-sampled when there are more than `nee`),
// segment transmission priced through the real mesh (paired faces →
// thickness), panel boxes and the earth; specular (or bent) chains count
// hits on distance-scaled source spheres; transmission branch with
// importance floor; escapes to open sky land in the sky cells unless they
// escape into the earth. Contributions are binned HERE exactly as v3.rs
// `deposit` bins them; the CPU (omg_core::v3::fold) folds the bins into
// its EMA state. The direct paths and the edge next-event paths stay on
// the CPU (Tracer::prepare).
//
// Accumulation: 64-bit fixed point in pairs of u32 (lo, hi) — integer
// atomicAdd on lo, the carry (detected from the returned old value) and
// the high part added to hi. Float CAS loops were 5–25× slower: every
// deposit of one source hits the same few hot words, and CAS retries
// under contention go quadratic. Each source accumulates at its own scale
// (2^44 × its free-field normalization max(d², 1)), so a bell 300 m out
// keeps the same relative resolution as a voice at 1 m: 5.7e-14 of the
// free-field energy (−132 dB) per contribution, 5e5× it of signed
// headroom per word. The
// `resolve` entry turns the accumulators back into f32 true units.
//
// Mirror discipline: every constant and branch below matches v3.rs.
// Known, documented deviation: the echogram's direction sums start at
// EARLY_S (the only part read).
//
// WALK_LAYOUT_VERSION 7 — must match omg_core::v3::fold and layout.rs:
//   Job {rot0@0, n_early@12, rot1@16, n_late@28, rot2@32, bounces@44,
//        listener@48, seed@60, k@64, nee@68, late_nee@72, n_sources@76,
//        n_panels@80, n_pockets@84, has_grade@88, grade@92, debug@96,
//        n_edges@100, off_panels@104, off_pockets@108, edge_lo@112,
//        off_edges@124, edge_hi@128, v4@140}                      160 B
//   scene (u32 words): sources ×12 {pos, weight, t_free, id, vis, norm},
//        panels ×16 {pmin, scattering, pmax, -, absorption, -,
//        transmission, -}, pockets ×8 {min, -, max, -}, edges ×8 {a, -, b, -}
//   Node/Prim/Mat: MESH_LAYOUT_VERSION 3 (trace_mesh.wgsl)
// Output (resolve, f32): SKY_WORDS sky, then per source [early
// CELLS×DELAY_BINS slots | echo ECHO_BINS slots | total ×3, pad]; a slot
// = (e0, e1, e2, Σe1·dx, Σe1·dy, Σe1·dz), true energy units.
// v4 (job.v4): per source [dense DENSE_BINS slots of 7 (the six above +
// Σe1²) | total ×3, pad] — the whole response, no early/late split.

// The exact chain solver's paths (omg_core::v3 `solver_owns`): coherent
// — every bounce a mirror, nothing scattered or bent — and loud chains
// with more than `bounces` reflections or arriving after the early
// window. The CPU names and solves those; sphere hits skip them here.
const COHERENT_FLOOR: f32 = 1e-2;
const MAX_COHERENT_REFL: u32 = 24u;
const N_CELLS: u32 = 64u;
const DELAY_BINS: u32 = 40u;
const ECHO_BINS: u32 = 300u;
const ECHO_DIR_FROM: u32 = 8u; // EARLY_S / BIN_DT: late_direction's start
const SLOT: u32 = 6u;
const EARLY_WORDS: u32 = 15360u; // N_CELLS * DELAY_BINS * SLOT
const ECHO_WORDS: u32 = 1800u; // ECHO_BINS * SLOT
const SOURCE_WORDS: u32 = 17164u; // EARLY_WORDS + ECHO_WORDS + 4
// engine v4 (omg_core::v3 RESPONSE_MAX_S, DENSE_BIN_S; fold DENSE_*)
const DENSE_BINS: u32 = 2400u;
const DENSE_SLOT: u32 = 7u;
const DENSE_WORDS: u32 = 16800u;
const DENSE_BIN_S: f32 = 0.002;
const RESPONSE_MAX_S: f32 = 4.8;
const SKY_WORDS: u32 = 384u; // N_CELLS * SLOT
const MAX_SOURCES: u32 = 64u;
const SRC_WORDS: u32 = 12u;
const DEBUG_VERTS: u32 = 8u;
const DEBUG_WORDS: u32 = 28u;
const PROBE_WORDS: u32 = 20u;

const EARLY_S: f32 = 0.080;
const DELAY_BIN_S: f32 = 0.002;
const BIN_DT: f32 = 0.010;
const MAX_TIME: f32 = 3.0;
const SPEED_OF_SOUND: f32 = 343.0;
const AIR_ABS = vec3<f32>(1.0e-4, 5.0e-4, 3.0e-3);
const MIN_DIST: f32 = 0.3;
const EPS: f32 = 1e-3;
// v3.rs MAX_WALL_M: the thickest solid a ray goes through as one wall
const MAX_WALL_M: f32 = 3.0;
const LATE_FLOOR: f32 = 1e-8;
const LATE_MAX_SEGMENTS: u32 = 600u;
const EARTH: f32 = 1e-4;
const REF_THICKNESS_M: f32 = 0.15;
const LEAF_BIT: u32 = 0x80000000u;
const T_MISS: f32 = 3.4e38;
const SEG_CAP: u32 = 48u;
const MAX_POCKETS: u32 = 8u;
// diffraction (v3.rs V3.4)
const EDGE_REACH: f32 = 3.0;
const EDGE_SEARCH: f32 = 120.0;
const DIFF_C: f32 = 12.566370614; // 4π
const MAX_BENDS: u32 = 3u;
const BAND_HZ = vec3<f32>(125.0, 790.0, 5600.0);
const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;
// fixed point
const FX_SCALE: f32 = 17592186044416.0; // 2^44
const TWO32: f32 = 4294967296.0;
const NO_EDGE: u32 = 0xFFFFFFFFu;
// a sliced early ray's state (Job.steps_pass; omg-gpu v3.rs): origin 0..3,
// dir 3..6, thr 6..9, length 9, refl 10, flags 11, mirror_thr 12,
// bends 13, bend_from 14..17, last_edge 17, segment 18, rng 19, first
// direction 20..23, first cell 23, alive 27
// a ray's state, then its split branch's (v3.rs `Branch`: the same 24
// words at +28, word 55 = pending)
const STATE_WORDS: u32 = 56u;
// v4: a ray splits at the first surface passing this share of the energy
// (v3.rs SPLIT_T2)
const SPLIT_T2: f32 = 0.02;

struct Job {
    rot0: vec3<f32>,
    n_early: u32,
    rot1: vec3<f32>,
    n_late: u32,
    rot2: vec3<f32>,
    bounces: u32,
    listener: vec3<f32>,
    seed: u32,
    k: f32,
    nee: u32,
    late_nee: u32,
    n_sources: u32,
    n_panels: u32,
    n_pockets: u32,
    has_grade: u32,
    grade: f32,
    debug: u32,
    n_edges: u32,
    off_panels: u32,
    off_pockets: u32,
    edge_lo: vec3<f32>,
    off_edges: u32,
    edge_hi: vec3<f32>,
    v4: u32,
    // a sliced walk (0: every ray walked whole in this one dispatch — the
    // classic walk; omg_core::v3::fold::WalkJob::live): `steps` (low 16
    // bits) segments per early ray per dispatch, a tick's walk being
    // several dispatches (`pass`, high 16 bits; the early rays start in
    // pass 0 and are done after the last, their states in `dbg` from
    // `state_off`). Late rays walk whole in pass 0 (a live walk leaves them
    // to the CPU: V3Walker::walks_late). No slicing while debug captures
    // the early rays' polylines.
    steps_pass: u32,
    state_off: u32,
    _r0: u32,
    _r1: u32,
}

struct Node {
    bmin: vec3<f32>,
    a: u32,
    bmax: vec3<f32>,
    b: u32,
}

struct Prim {
    a: vec3<f32>,
    mat: u32,
    e1: vec3<f32>,
    surf: u32,
    e2: vec3<f32>,
    _p1: u32,
}

struct Mat {
    absorption: vec3<f32>,
    scattering: f32,
    transmission: vec3<f32>,
    // > 0: authored wall thickness (Mesh::material_thickness), else inferred
    thickness: f32,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> prims: array<Prim>;
@group(0) @binding(3) var<storage, read> mats: array<Mat>;
@group(0) @binding(4) var<storage, read> scene: array<u32>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> dbg: array<u32>;
@group(0) @binding(8) var<storage, read> edgebuf: array<u32>;

// ---------------------------------------------------------------- scene

fn sf(i: u32) -> f32 {
    return bitcast<f32>(scene[i]);
}

fn sv3(i: u32) -> vec3<f32> {
    return vec3<f32>(sf(i), sf(i + 1u), sf(i + 2u));
}

fn src_pos(si: u32) -> vec3<f32> { return sv3(si * SRC_WORDS); }
fn src_weight(si: u32) -> f32 { return sf(si * SRC_WORDS + 3u); }
fn src_t_free(si: u32) -> f32 { return sf(si * SRC_WORDS + 4u); }
fn src_vis(si: u32) -> vec3<f32> { return sv3(si * SRC_WORDS + 6u); }
fn src_norm(si: u32) -> f32 { return sf(si * SRC_WORDS + 9u); }

struct Panel {
    pmin: vec3<f32>,
    scattering: f32,
    pmax: vec3<f32>,
    absorption: vec3<f32>,
    transmission: vec3<f32>,
}

fn panel(i: u32) -> Panel {
    let o = job.off_panels + i * 16u;
    return Panel(sv3(o), sf(o + 3u), sv3(o + 4u), sv3(o + 8u), sv3(o + 12u));
}

fn pocket_min(i: u32) -> vec3<f32> { return sv3(job.off_pockets + i * 8u); }
fn pocket_max(i: u32) -> vec3<f32> { return sv3(job.off_pockets + i * 8u + 4u); }

// ------------------------------------------------------------------ rng

var<private> rng_state: u32;

fn pcg_hash(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn pcg_next() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32() -> f32 {
    return f32(pcg_next() >> 8u) / 16777216.0;
}

// Rng::unit_sphere
fn unit_sphere() -> vec3<f32> {
    let z = 1.0 - 2.0 * rand_f32();
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 6.28318530718 * rand_f32();
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// fibonacci_sphere(n)[i]; the golden angle is reduced mod 2π first
// (GPU cos/sin lose accuracy on large arguments): (3 − √5)/2 turns per
// index, i = lo + 4096·hi with frac(4096·g) precomputed
fn fib(i: u32, n: u32) -> vec3<f32> {
    let z = 1.0 - 2.0 * (f32(i) + 0.5) / f32(n);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let turns = fract(f32(i & 0xFFFu) * 0.38196601125) + fract(f32(i >> 12u) * 0.53278208);
    let phi = fract(turns) * 6.28318530718;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// --------------------------------------------------------- intersection

fn ray_tri(a: vec3<f32>, e1: vec3<f32>, e2: vec3<f32>, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) { return -1.0; }
    let inv = 1.0 / det;
    let s = o - a;
    let u = dot(s, p) * inv;
    if (u < -1e-6 || u > 1.0 + 1e-6) { return -1.0; }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if (v < -1e-6 || u + v > 1.0 + 1e-6) { return -1.0; }
    let t = dot(e2, q) * inv;
    if (t > 1e-5) { return t; }
    return -1.0;
}

fn safe_inv(d: vec3<f32>) -> vec3<f32> {
    let dd = select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    return vec3<f32>(1.0) / dd;
}

// Mesh::ray_aabb: entry distance, or < 0 on a miss.
fn ray_aabb(bmin: vec3<f32>, bmax: vec3<f32>, o: vec3<f32>, inv_d: vec3<f32>, tmax: f32) -> f32 {
    let t1 = (bmin - o) * inv_d;
    let t2 = (bmax - o) * inv_d;
    let lo = min(t1, t2);
    let hi = max(t1, t2);
    let t0 = max(max(lo.x, lo.y), max(lo.z, 0.0));
    let tb = min(min(hi.x, hi.y), min(hi.z, tmax));
    if (t0 > tb) { return -1.0; }
    return t0;
}

struct Hit {
    t: f32,
    normal: vec3<f32>,
    absorption: vec3<f32>,
    scattering: f32,
    transmission: vec3<f32>,
}

// World::raycast — the mesh (ordered BVH traversal, as Mesh::raycast)
// with the overlay panels (WithPanels: slab entry, nearest wins). The
// normal faces the incoming ray.
fn raycast_world(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h: Hit;
    h.t = T_MISS;
    let inv_d = safe_inv(d);
    var stack_i: array<u32, 64>;
    var stack_t: array<f32, 64>;
    var sp = 1u;
    stack_i[0] = 0u;
    stack_t[0] = 0.0;
    var best = 0u;
    var have = false;
    // the root's own box first (a dummy empty-mesh root never passes)
    let root = nodes[0];
    if (ray_aabb(root.bmin, root.bmax, o, inv_d, T_MISS) < 0.0) { sp = 0u; }
    while (sp > 0u) {
        sp -= 1u;
        if (stack_t[sp] >= h.t) { continue; }
        let node = nodes[stack_i[sp]];
        if ((node.a & LEAF_BIT) != 0u) {
            let start = node.a & ~LEAF_BIT;
            for (var i = 0u; i < node.b; i++) {
                let p = prims[start + i];
                let t = ray_tri(p.a, p.e1, p.e2, o, d);
                if (t > 0.0 && t < h.t) {
                    h.t = t;
                    best = start + i;
                    have = true;
                }
            }
        } else {
            let l = nodes[node.a];
            let r = nodes[node.b];
            let tl = ray_aabb(l.bmin, l.bmax, o, inv_d, h.t);
            let tr = ray_aabb(r.bmin, r.bmax, o, inv_d, h.t);
            if (tl >= 0.0 && tr >= 0.0 && sp + 2u <= 64u) {
                if (tl <= tr) {
                    stack_i[sp] = node.b; stack_t[sp] = tr;
                    stack_i[sp + 1u] = node.a; stack_t[sp + 1u] = tl;
                } else {
                    stack_i[sp] = node.a; stack_t[sp] = tl;
                    stack_i[sp + 1u] = node.b; stack_t[sp + 1u] = tr;
                }
                sp += 2u;
            } else if (tl >= 0.0 && sp < 64u) {
                stack_i[sp] = node.a; stack_t[sp] = tl;
                sp += 1u;
            } else if (tr >= 0.0 && sp < 64u) {
                stack_i[sp] = node.b; stack_t[sp] = tr;
                sp += 1u;
            }
        }
    }
    if (have) {
        let p = prims[best];
        var n = normalize(cross(p.e1, p.e2));
        if (dot(n, d) > 0.0) { n = -n; }
        h.normal = n;
        let m = mats[p.mat];
        h.absorption = m.absorption;
        h.scattering = m.scattering;
        h.transmission = m.transmission;
    }
    for (var i = 0u; i < job.n_panels; i++) {
        let pl = panel(i);
        var t0 = 0.0;
        var t1 = T_MISS;
        var axis = 3u;
        var miss = false;
        for (var a = 0u; a < 3u; a++) {
            if (abs(d[a]) < 1e-9) {
                if (o[a] < pl.pmin[a] || o[a] > pl.pmax[a]) { miss = true; break; }
            } else {
                var ta = (pl.pmin[a] - o[a]) / d[a];
                var tb = (pl.pmax[a] - o[a]) / d[a];
                if (ta > tb) { let tmp = ta; ta = tb; tb = tmp; }
                if (ta > t0) { t0 = ta; axis = a; }
                t1 = min(t1, tb);
                if (t0 > t1) { miss = true; break; }
            }
        }
        if (!miss && t0 > 1e-4 && axis < 3u && t0 < h.t) {
            h.t = t0;
            var n = vec3<f32>(0.0);
            n[axis] = select(1.0, -1.0, d[axis] > 0.0);
            h.normal = n;
            h.absorption = pl.absorption;
            h.scattering = pl.scattering;
            h.transmission = pl.transmission;
        }
    }
    return h;
}

// -------------------------------------------------------- transmission

var<private> seg_t: array<f32, SEG_CAP>;
var<private> seg_mat: array<u32, SEG_CAP>;
var<private> seg_surf: array<u32, SEG_CAP>;

// paths::segment_transmission over Mesh::segment_hits: every crossing
// of the open segment, sorted, same-t clusters deduplicated (LOWEST
// surface id wins, anchor t kept), consecutive same-material crossings
// paired into one wall of real thickness (mass law).
fn mesh_transmission(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let dvec = b - a;
    let len = length(dvec);
    if (len < 1e-6) { return vec3<f32>(1.0); }
    let dn = dvec / len;
    let inv_d = safe_inv(dn);
    var n = 0u;
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    while (sp > 0u) {
        sp -= 1u;
        let node = nodes[stack[sp]];
        if (ray_aabb(node.bmin, node.bmax, a, inv_d, len) < 0.0) { continue; }
        if ((node.a & LEAF_BIT) != 0u) {
            let start = node.a & ~LEAF_BIT;
            for (var i = 0u; i < node.b; i++) {
                let p = prims[start + i];
                let t = ray_tri(p.a, p.e1, p.e2, a, dn);
                if (t < 0.0) { continue; }
                let tt = t / len;
                if (tt <= 1e-4 || tt >= 1.0 - 1e-4) { continue; }
                // sorted insert; past the cap the farthest crossing drops
                if (n == SEG_CAP && tt >= seg_t[SEG_CAP - 1u]) { continue; }
                var j = min(n, SEG_CAP - 1u);
                while (j > 0u && seg_t[j - 1u] > tt) {
                    seg_t[j] = seg_t[j - 1u];
                    seg_mat[j] = seg_mat[j - 1u];
                    seg_surf[j] = seg_surf[j - 1u];
                    j -= 1u;
                }
                seg_t[j] = tt;
                seg_mat[j] = p.mat;
                seg_surf[j] = p.surf;
                n = min(n + 1u, SEG_CAP);
            }
        } else if (sp + 2u <= 64u) {
            stack[sp] = node.a;
            stack[sp + 1u] = node.b;
            sp += 2u;
        }
    }
    // same-t dedup
    let tol = 1e-4 / max(len, 1.0);
    var w = 0u;
    for (var i = 0u; i < n; i++) {
        if (w > 0u && abs(seg_t[i] - seg_t[w - 1u]) < tol) {
            if (seg_surf[i] < seg_surf[w - 1u]) {
                seg_mat[w - 1u] = seg_mat[i];
                seg_surf[w - 1u] = seg_surf[i];
            }
        } else {
            seg_t[w] = seg_t[i];
            seg_mat[w] = seg_mat[i];
            seg_surf[w] = seg_surf[i];
            w += 1u;
        }
    }
    // pair entry/exit faces of the same material
    var g = vec3<f32>(1.0);
    var i = 0u;
    while (i < w) {
        let m = seg_mat[i];
        var thickness = 0.1;
        if (i + 1u < w && seg_mat[i + 1u] == m) {
            thickness = max((seg_t[i + 1u] - seg_t[i]) * len, 0.01);
            i += 1u;
        }
        if (mats[m].thickness > 0.0) {
            thickness = mats[m].thickness;
        }
        let kk = min(REF_THICKNESS_M / max(thickness, 0.02), 2.0);
        g = g * min(mats[m].transmission * kk, vec3<f32>(0.9));
        i += 1u;
    }
    return g;
}

// clip_box: parameter interval of a→b inside the box; x > y = none.
fn clip_box(a: vec3<f32>, b: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>) -> vec2<f32> {
    let d = b - a;
    var t0 = 0.0;
    var t1 = 1.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let da = d[axis];
        if (abs(da) < 1e-9) {
            if (a[axis] < mn[axis] || a[axis] > mx[axis]) { return vec2<f32>(1.0, 0.0); }
        } else {
            var ta = (mn[axis] - a[axis]) / da;
            var tb = (mx[axis] - a[axis]) / da;
            if (ta > tb) { let tmp = ta; ta = tb; tb = tmp; }
            t0 = max(t0, ta);
            t1 = min(t1, tb);
            if (t0 > t1) { return vec2<f32>(1.0, 0.0); }
        }
    }
    return vec2<f32>(t0, t1);
}

fn segment_crosses_box(a: vec3<f32>, b: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>) -> bool {
    let iv = clip_box(a, b, mn, mx);
    if (iv.x > iv.y) { return false; }
    return iv.y > 1e-4 && iv.x < 1.0 - 1e-4;
}

fn in_earth(p: vec3<f32>) -> bool {
    if (job.has_grade == 0u || p.z >= job.grade - 0.05) { return false; }
    for (var i = 0u; i < job.n_pockets; i++) {
        if (all(p >= pocket_min(i)) && all(p <= pocket_max(i))) { return false; }
    }
    return true;
}

fn escapes_into_earth(p: vec3<f32>, d: vec3<f32>) -> bool {
    if (job.has_grade == 0u) { return false; }
    return d.z < -1e-3 || (p.z < job.grade - 0.05 && d.z <= 0.05);
}

// World::crosses_earth: more than 10 cm of the segment below grade and
// outside every air pocket.
fn crosses_earth(a: vec3<f32>, b: vec3<f32>) -> bool {
    if (job.has_grade == 0u) { return false; }
    let g = job.grade - 0.05;
    let za = a.z;
    let zb = b.z;
    if (za >= g && zb >= g) { return false; }
    var t0 = 0.0;
    var t1 = 1.0;
    if (abs(zb - za) >= 1e-9) {
        let tc = clamp((g - za) / (zb - za), 0.0, 1.0);
        if (za < g) { t1 = tc; } else { t0 = tc; }
    }
    let len = length(b - a);
    var s0: array<f32, MAX_POCKETS>;
    var s1: array<f32, MAX_POCKETS>;
    var n = 0u;
    for (var i = 0u; i < min(job.n_pockets, MAX_POCKETS); i++) {
        let iv = clip_box(a, b, pocket_min(i), pocket_max(i));
        if (iv.x > iv.y) { continue; }
        let lo = max(iv.x, t0);
        let hi = min(iv.y, t1);
        if (hi <= lo) { continue; }
        var j = n;
        while (j > 0u && s0[j - 1u] > lo) {
            s0[j] = s0[j - 1u];
            s1[j] = s1[j - 1u];
            j -= 1u;
        }
        s0[j] = lo;
        s1[j] = hi;
        n += 1u;
    }
    var covered = 0.0;
    var end = t0;
    for (var i = 0u; i < n; i++) {
        let lo = max(s0[i], end);
        if (s1[i] > lo) {
            covered += s1[i] - lo;
            end = s1[i];
        }
    }
    return (t1 - t0 - covered) * len > 0.1;
}

// World::through: a walking ray going on through the surface it hit at
// `p` — the far face of the solid behind it (the next hit of the same
// material within MAX_WALL_M; none: a thin shell or an object box,
// crossed at once), priced as a segment through it: the energy per band
// (xyz) and where the ray goes on (written to `beyond`).
var<private> beyond: vec3<f32>;
fn through(p: vec3<f32>, dir: vec3<f32>, hit: Hit) -> vec3<f32> {
    let inside = p + dir * EPS;
    let h = raycast_world(inside, dir);
    var far = p;
    if (h.t < MAX_WALL_M && all(h.absorption == hit.absorption) && h.scattering == hit.scattering
        && all(h.transmission == hit.transmission)) {
        far = inside + dir * h.t;
    }
    let t = seg_transmission(p - dir * EPS, far + dir * EPS);
    beyond = far + dir * (2.0 * EPS);
    return t * t;
}

// World::transmission: per-band amplitude of the segment a→b.
fn seg_transmission(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    if (crosses_earth(a, b)) { return vec3<f32>(EARTH); }
    var g = mesh_transmission(a, b);
    for (var i = 0u; i < job.n_panels; i++) {
        let pl = panel(i);
        if (segment_crosses_box(a, b, pl.pmin, pl.pmax)) {
            g = g * pl.transmission;
        }
    }
    return g;
}

fn air(dist: f32) -> vec3<f32> {
    return exp(-AIR_ABS * dist);
}

// --------------------------------------------------------- diffraction

struct EdgeHit {
    found: bool,
    t: f32,
    dist: f32,
    u: vec3<f32>,
    idx: u32,
}

// EdgeSet::nearest: the usable edge the segment o + d·t passes closest
// to (within EDGE_REACH, alongside the edge's span, clear of both ends of
// the segment), searched no further than EDGE_SEARCH and only inside the
// edges' padded bounds — through the same 4 m grid, sampled every 2 m
// along the segment (`edgebuf`, EdgeSet::flat), skipping the edges this
// tick's mask fills.
const EDGE_CELL: f32 = 4.0;
const EDGE_FLAT_HEADER: u32 = 16u;
const NO_SLOT: u32 = 0xFFFFFFFFu;

// the table slot of grid cell k (EdgeSet::flat: 5 words — kx, ky, kz,
// list start, count; count 0 = empty), or NO_SLOT
fn cell_slot(k: vec3<i32>) -> u32 {
    let mask = edgebuf[1] - 1u;
    let off = edgebuf[2];
    let u = bitcast<vec3<u32>>(k);
    var h = ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) & mask;
    // (the table is never full: an empty slot ends every probe)
    for (var n = 0u; n <= mask; n++) {
        let o = off + h * 5u;
        if (edgebuf[o + 4u] == 0u) { return NO_SLOT; }
        if (edgebuf[o] == u.x && edgebuf[o + 1u] == u.y && edgebuf[o + 2u] == u.z) { return o; }
        h = (h + 1u) & mask;
    }
    return NO_SLOT;
}

fn edge_filled(i: u32) -> bool {
    return ((scene[job.off_edges + (i >> 5u)] >> (i & 31u)) & 1u) != 0u;
}

fn nearest_edge(o: vec3<f32>, d: vec3<f32>, reach: f32, skip: u32) -> EdgeHit {
    var h: EdgeHit;
    h.found = false;
    let seg = min(reach, EDGE_SEARCH);
    let iv = clip_box(o, o + d * seg, job.edge_lo, job.edge_hi);
    if (iv.x > iv.y) { return h; }
    let t_start = iv.x * seg;
    let t_end = iv.y * seg;
    let lists = edgebuf[3];
    var best = EDGE_REACH;
    var last = vec3<i32>(0);
    var have_last = false;
    var t = t_start;
    loop {
        let k = vec3<i32>(floor((o + d * min(t, t_end)) / EDGE_CELL));
        if (!have_last || any(k != last)) {
            have_last = true;
            last = k;
            let so = cell_slot(k);
            if (so != NO_SLOT) {
                let start = lists + edgebuf[so + 3u];
                let count = edgebuf[so + 4u];
                for (var j = 0u; j < count; j++) {
                    let i = edgebuf[start + j];
                    if (i == skip || edge_filled(i)) { continue; }
                    let eo = EDGE_FLAT_HEADER + i * 8u;
                    let a = vec3<f32>(bitcast<f32>(edgebuf[eo]), bitcast<f32>(edgebuf[eo + 1u]), bitcast<f32>(edgebuf[eo + 2u]));
                    let e = vec3<f32>(bitcast<f32>(edgebuf[eo + 4u]), bitcast<f32>(edgebuf[eo + 5u]), bitcast<f32>(edgebuf[eo + 6u])) - a;
                    let w0 = o - a;
                    let bb = dot(d, e);
                    let cc = dot(e, e);
                    let dd = dot(d, w0);
                    let ee = dot(e, w0);
                    let den = cc - bb * bb;
                    if (den < 1e-9) { continue; } // parallel to the edge
                    let tt = (bb * ee - cc * dd) / den;
                    let ss = (ee - bb * dd) / den;
                    if (ss < 0.02 || ss > 0.98 || tt < 0.05 || tt > t_end - 0.05) { continue; }
                    let dist = length(o + d * tt - (a + e * ss));
                    if (dist < best) {
                        best = dist;
                        h.found = true;
                        h.t = tt;
                        h.dist = dist;
                        h.u = e / sqrt(cc);
                        h.idx = i;
                    }
                }
            }
        }
        if (t >= t_end) { break; }
        t = min(t + EDGE_CELL * 0.5, t_end);
    }
    return h;
}

// sample_bend: the bend angle from the three bands' truncated-Cauchy
// spreads σ_b = λ_b / (4π·a) in equal parts; per-band weight pdf_b / mix.
var<private> bend_w: vec3<f32>;

fn sample_bend(a: f32) -> f32 {
    let sig = max((SPEED_OF_SOUND / BAND_HZ) / (DIFF_C * max(a, 0.05)), vec3<f32>(1e-4));
    let j = min(u32(rand_f32() * 3.0), 2u);
    let u = rand_f32() * 2.0 - 1.0;
    let sj = sig[j];
    let eps = sj * tan(u * atan(HALF_PI / sj));
    let z = (2.0 / PI) * atan(HALF_PI / sig);
    let pdf = (1.0 / (PI * z)) * sig / (sig * sig + vec3<f32>(eps * eps));
    let mix = (pdf.x + pdf.y + pdf.z) / 3.0;
    bend_w = pdf / max(mix, 1e-12);
    return eps;
}

// rotate_about: Rodrigues — the bent ray keeps its angle to the edge.
fn rotate_about(d: vec3<f32>, u: vec3<f32>, eps: f32) -> vec3<f32> {
    let s = sin(eps);
    let c = cos(eps);
    return normalize(d * c + cross(u, d) * s + u * (dot(u, d) * (1.0 - c)));
}

// ------------------------------------------------------------ binning

// Add x (already at its fixed-point scale) to 64-bit accumulator v.
fn add_fx(v: u32, x: f32) {
    if (!(abs(x) > 0.0)) { return; } // zero (or NaN): nothing
    let m = min(abs(x), 9.2e18); // < 2^63: signed headroom
    var hi = u32(m * (1.0 / TWO32));
    var lo = 0u;
    if (m < 7.2e16) {
        // below 2^56, f32(hi) is exact and so is the remainder; above,
        // m's ulp exceeds 2^32 and its low word is zero anyway
        lo = u32(min(m - f32(hi) * TWO32 + 0.5, 4294967295.0));
    }
    if (x < 0.0) {
        // two's complement of the 64-bit magnitude
        lo = ~lo + 1u;
        hi = ~hi + select(0u, 1u, lo == 0u);
    }
    let old = atomicAdd(&acc[2u * v], lo);
    let h = hi + select(0u, 1u, old + lo < old);
    if (h != 0u) {
        atomicAdd(&acc[2u * v + 1u], h);
    }
}

fn read_fx(v: u32) -> f32 {
    let lo = atomicLoad(&acc[2u * v]);
    let hi = atomicLoad(&acc[2u * v + 1u]);
    if ((hi & 0x80000000u) != 0u) {
        let nlo = ~lo + 1u;
        let nhi = ~hi + select(0u, 1u, nlo == 0u);
        return -(f32(nhi) * TWO32 + f32(nlo));
    }
    return f32(hi) * TWO32 + f32(lo);
}

// the ray's constants for its deposits
var<private> ray_first: vec3<f32>;
var<private> ray_cell: u32;
// debug capture (early rays only): the walked polyline and the best
// contribution so far
var<private> dbg_on: bool;
var<private> dbg_slot: u32;
var<private> dbg_best: f32;
var<private> verts: array<vec3<f32>, DEBUG_VERTS>;
var<private> n_verts: u32;

fn push_vert(p: vec3<f32>) {
    if (n_verts < DEBUG_VERTS - 1u) {
        verts[n_verts] = p;
        n_verts += 1u;
    } else {
        n_verts = DEBUG_VERTS; // too deep to draw: stop capturing
    }
}

// v3.rs `deposit`: the echogram relative to the free-field arrival; the
// early bins (first-direction cell × 2 ms) inside 80 ms.
fn deposit(si: u32, e: vec3<f32>, delay: f32) {
    deposit_sq(si, e, delay, 1.0);
}

// v3.rs `deposit_spread`: v4 spreads a sphere hit over its own r/c
// time uncertainty (each part one n-th, its square counted n times so the
// spread hit stays ONE sample); v3 deposits at `delay`.
fn deposit_spread(si: u32, e: vec3<f32>, delay: f32, half: f32) {
    if (job.v4 == 0u || half < 0.5 * DENSE_BIN_S) {
        deposit(si, e, delay);
        return;
    }
    let n = u32(ceil(2.0 * half / DENSE_BIN_S)) + 1u;
    let part = e / f32(n);
    let t_free = src_t_free(si);
    for (var i = 0u; i < n; i++) {
        let t = delay - half + 2.0 * half * f32(i) / f32(n - 1u);
        deposit_sq(si, part, max(t, t_free), f32(n));
    }
}

fn deposit_sq(si: u32, e: vec3<f32>, delay: f32, sq_scale: f32) {
    if (!(e.x > 0.0 || e.y > 0.0 || e.z > 0.0)) { return; }
    let sc = src_norm(si) * FX_SCALE;
    let es = e * sc;
    let dm = ray_first * es.y;
    let base = SKY_WORDS + si * SOURCE_WORDS;
    let rel = max(delay - src_t_free(si), 0.0);
    if (job.v4 != 0u) {
        // v4: everything the walk estimates is the dense response
        let bin = u32(rel / DENSE_BIN_S);
        if (rel < RESPONSE_MAX_S && bin < DENSE_BINS) {
            let o = base + bin * DENSE_SLOT;
            add_fx(o, es.x);
            add_fx(o + 1u, es.y);
            add_fx(o + 2u, es.z);
            add_fx(o + 3u, dm.x);
            add_fx(o + 4u, dm.y);
            add_fx(o + 5u, dm.z);
            add_fx(o + 6u, sq_scale * es.y * e.y);
        }
        return;
    }
    if (rel < MAX_TIME) {
        let bin = u32(rel / BIN_DT);
        if (bin < ECHO_BINS) {
            let o = base + EARLY_WORDS + bin * SLOT;
            add_fx(o, es.x);
            add_fx(o + 1u, es.y);
            add_fx(o + 2u, es.z);
            if (bin >= ECHO_DIR_FROM) {
                add_fx(o + 3u, dm.x);
                add_fx(o + 4u, dm.y);
                add_fx(o + 5u, dm.z);
            }
        }
    }
    if (rel < EARLY_S) {
        let db = min(u32(rel / DELAY_BIN_S), DELAY_BINS - 1u);
        let o = base + (ray_cell * DELAY_BINS + db) * SLOT;
        add_fx(o, es.x);
        add_fx(o + 1u, es.y);
        add_fx(o + 2u, es.z);
        add_fx(o + 3u, dm.x);
        add_fx(o + 4u, dm.y);
        add_fx(o + 5u, dm.z);
    }
    if (dbg_on && e.y > dbg_best && n_verts < DEBUG_VERTS) {
        dbg_best = e.y;
        let o = dbg_slot;
        dbg[o] = bitcast<u32>(e.y);
        dbg[o + 1u] = si;
        dbg[o + 2u] = n_verts + 1u;
        for (var k = 0u; k < n_verts; k++) {
            dbg[o + 4u + k * 3u] = bitcast<u32>(verts[k].x);
            dbg[o + 5u + k * 3u] = bitcast<u32>(verts[k].y);
            dbg[o + 6u + k * 3u] = bitcast<u32>(verts[k].z);
        }
        let s = src_pos(si);
        dbg[o + 4u + n_verts * 3u] = bitcast<u32>(s.x);
        dbg[o + 5u + n_verts * 3u] = bitcast<u32>(s.y);
        dbg[o + 6u + n_verts * 3u] = bitcast<u32>(s.z);
    }
}

// the 64 direction cells (fibonacci_sphere(64)): nearest to `d`
fn cell_of(d: vec3<f32>) -> u32 {
    var best = 0u;
    var bd = -3.4e38;
    for (var c = 0u; c < N_CELLS; c++) {
        let v = dot(fib(c, N_CELLS), d);
        if (v > bd) {
            bd = v;
            best = c;
        }
    }
    return best;
}

// ---------------------------------------------------------------- walk

@compute @workgroup_size(64)
fn walk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    let n_early = job.n_early;
    let n_late = job.n_late;
    if (ray >= n_early + n_late) { return; }
    let late = ray < n_late;
    let idx = select(ray - n_late, ray, late);
    let n = select(n_early, n_late, late);
    // the family: EARLY counts reflections 0..=bounces (0: bent around
    // edges only), LATE everything beyond
    let lo = select(0u, job.bounces + 1u, late);
    let hi = select(job.bounces, 0xFFFFFFFFu, late);
    let segments = select(job.bounces + 1u, LATE_MAX_SEGMENTS, late) + MAX_BENDS;
    let nee = select(job.nee, job.late_nee, late);
    let steps = job.steps_pass & 0xFFFFu;
    let pass_idx = job.steps_pass >> 16u;
    let sliced = !late && steps > 0u && job.debug == 0u;
    // rays walked whole (the late family, debug capture) run in pass 0 only
    if (!sliced && pass_idx > 0u) { return; }
    let inv_n = 1.0 / f32(n);
    let n_src = min(job.n_sources, MAX_SOURCES);
    let diffraction = job.n_edges > 0u;

    rng_state = pcg_hash(job.seed ^ pcg_hash(ray * 0x9E3779B9u + 0x7F4A7C15u));

    let d0 = fib(idx, n);
    let first = vec3<f32>(dot(job.rot0, d0), dot(job.rot1, d0), dot(job.rot2, d0));
    ray_first = first;
    ray_cell = cell_of(first);
    dbg_on = job.debug != 0u && !late;
    dbg_slot = idx * DEBUG_WORDS;
    dbg_best = 0.0;
    verts[0] = job.listener;
    n_verts = 1u;

    var origin = job.listener;
    var dir = first;
    var thr = vec3<f32>(1.0);
    var length_m = 0.0;
    // reflections so far, and whether the last one was specular (only
    // then — or after a bend — do sphere hits count)
    var refl = 0u;
    var specular = false;
    // every bounce a mirror so far (nothing scattered or bent), and that
    // mirror chain's own throughput still above COHERENT_FLOOR (latched)
    var mirror_only = true;
    var loud = true;
    var mirror_thr = 1.0;
    // bent since the last diffuse bounce; where it first bent since the
    // last reflection; never twice at the same edge in a row
    var bent = false;
    var bends = 0u;
    var have_bend_from = false;
    var bend_from = vec3<f32>(0.0);
    var last_edge = NO_EDGE;
    var cum: array<f32, MAX_SOURCES>;
    // v4: split once (at a partition); the transmitted branch waits here
    // while the reflected one walks on (v3.rs `Branch`)
    var split = false;
    var has_pending = false;
    var p_origin = vec3<f32>(0.0);
    var p_dir = vec3<f32>(0.0);
    var p_thr = vec3<f32>(0.0);
    var p_length = 0.0;
    var p_refl = 0u;
    var p_flags = 0u;
    var p_mirror_thr = 0.0;
    var p_bends = 0u;
    var p_bend_from = vec3<f32>(0.0);
    var p_last_edge = NO_EDGE;
    var p_seg = 0u;

    // a sliced ray starts in pass 0, goes on where the last dispatch left
    // it and, once done, sits out the tick's later passes
    let so = job.state_off + idx * STATE_WORDS;
    var seg0 = 0u;
    if (sliced && pass_idx > 0u) {
        if (dbg[so + 27u] != 1u) { return; }
        origin = vec3<f32>(bitcast<f32>(dbg[so]), bitcast<f32>(dbg[so + 1u]), bitcast<f32>(dbg[so + 2u]));
        dir = vec3<f32>(bitcast<f32>(dbg[so + 3u]), bitcast<f32>(dbg[so + 4u]), bitcast<f32>(dbg[so + 5u]));
        thr = vec3<f32>(bitcast<f32>(dbg[so + 6u]), bitcast<f32>(dbg[so + 7u]), bitcast<f32>(dbg[so + 8u]));
        length_m = bitcast<f32>(dbg[so + 9u]);
        refl = dbg[so + 10u];
        let flags = dbg[so + 11u];
        specular = (flags & 1u) != 0u;
        mirror_only = (flags & 2u) != 0u;
        loud = (flags & 4u) != 0u;
        bent = (flags & 8u) != 0u;
        have_bend_from = (flags & 16u) != 0u;
        mirror_thr = bitcast<f32>(dbg[so + 12u]);
        bends = dbg[so + 13u];
        bend_from = vec3<f32>(bitcast<f32>(dbg[so + 14u]), bitcast<f32>(dbg[so + 15u]), bitcast<f32>(dbg[so + 16u]));
        last_edge = dbg[so + 17u];
        seg0 = dbg[so + 18u];
        rng_state = dbg[so + 19u];
        ray_first = vec3<f32>(bitcast<f32>(dbg[so + 20u]), bitcast<f32>(dbg[so + 21u]), bitcast<f32>(dbg[so + 22u]));
        ray_cell = dbg[so + 23u];
        split = (flags & 32u) != 0u;
        has_pending = dbg[so + 55u] == 1u;
        if (has_pending) {
            let q = so + 28u;
            p_origin = vec3<f32>(bitcast<f32>(dbg[q]), bitcast<f32>(dbg[q + 1u]), bitcast<f32>(dbg[q + 2u]));
            p_dir = vec3<f32>(bitcast<f32>(dbg[q + 3u]), bitcast<f32>(dbg[q + 4u]), bitcast<f32>(dbg[q + 5u]));
            p_thr = vec3<f32>(bitcast<f32>(dbg[q + 6u]), bitcast<f32>(dbg[q + 7u]), bitcast<f32>(dbg[q + 8u]));
            p_length = bitcast<f32>(dbg[q + 9u]);
            p_refl = dbg[q + 10u];
            p_flags = dbg[q + 11u];
            p_mirror_thr = bitcast<f32>(dbg[q + 12u]);
            p_bends = dbg[q + 13u];
            p_bend_from = vec3<f32>(bitcast<f32>(dbg[q + 14u]), bitcast<f32>(dbg[q + 15u]), bitcast<f32>(dbg[q + 16u]));
            p_last_edge = dbg[q + 17u];
            p_seg = dbg[q + 18u];
        }
    }

    var walked = 0u;
    loop {
    for (var seg = seg0; seg < segments; seg++) {
        if (sliced && walked >= steps) {
            // this dispatch's share walked: the lane keeps its ray
            dbg[so] = bitcast<u32>(origin.x);
            dbg[so + 1u] = bitcast<u32>(origin.y);
            dbg[so + 2u] = bitcast<u32>(origin.z);
            dbg[so + 3u] = bitcast<u32>(dir.x);
            dbg[so + 4u] = bitcast<u32>(dir.y);
            dbg[so + 5u] = bitcast<u32>(dir.z);
            dbg[so + 6u] = bitcast<u32>(thr.x);
            dbg[so + 7u] = bitcast<u32>(thr.y);
            dbg[so + 8u] = bitcast<u32>(thr.z);
            dbg[so + 9u] = bitcast<u32>(length_m);
            dbg[so + 10u] = refl;
            dbg[so + 11u] = select(0u, 1u, specular) | select(0u, 2u, mirror_only) | select(0u, 4u, loud)
                | select(0u, 8u, bent) | select(0u, 16u, have_bend_from) | select(0u, 32u, split);
            dbg[so + 12u] = bitcast<u32>(mirror_thr);
            dbg[so + 13u] = bends;
            dbg[so + 14u] = bitcast<u32>(bend_from.x);
            dbg[so + 15u] = bitcast<u32>(bend_from.y);
            dbg[so + 16u] = bitcast<u32>(bend_from.z);
            dbg[so + 17u] = last_edge;
            dbg[so + 18u] = seg;
            dbg[so + 19u] = rng_state;
            dbg[so + 20u] = bitcast<u32>(ray_first.x);
            dbg[so + 21u] = bitcast<u32>(ray_first.y);
            dbg[so + 22u] = bitcast<u32>(ray_first.z);
            dbg[so + 23u] = ray_cell;
            dbg[so + 27u] = 1u;
            dbg[so + 55u] = select(0u, 1u, has_pending);
            if (has_pending) {
                let q = so + 28u;
                dbg[q] = bitcast<u32>(p_origin.x);
                dbg[q + 1u] = bitcast<u32>(p_origin.y);
                dbg[q + 2u] = bitcast<u32>(p_origin.z);
                dbg[q + 3u] = bitcast<u32>(p_dir.x);
                dbg[q + 4u] = bitcast<u32>(p_dir.y);
                dbg[q + 5u] = bitcast<u32>(p_dir.z);
                dbg[q + 6u] = bitcast<u32>(p_thr.x);
                dbg[q + 7u] = bitcast<u32>(p_thr.y);
                dbg[q + 8u] = bitcast<u32>(p_thr.z);
                dbg[q + 9u] = bitcast<u32>(p_length);
                dbg[q + 10u] = p_refl;
                dbg[q + 11u] = p_flags;
                dbg[q + 12u] = bitcast<u32>(p_mirror_thr);
                dbg[q + 13u] = p_bends;
                dbg[q + 14u] = bitcast<u32>(p_bend_from.x);
                dbg[q + 15u] = bitcast<u32>(p_bend_from.y);
                dbg[q + 16u] = bitcast<u32>(p_bend_from.z);
                dbg[q + 17u] = p_last_edge;
                dbg[q + 18u] = p_seg;
            }
            return;
        }
        walked += 1u;
        if (in_earth(origin)) { break; }
        let hit = raycast_world(origin, dir);
        let t_hit = hit.t;
        // diffraction: the edge this segment passes closest to (a miss
        // searches EDGE_SEARCH — the CPU's `reach` is f32::MAX there)
        var edge: EdgeHit;
        edge.found = false;
        if (diffraction && bends < MAX_BENDS) {
            edge = nearest_edge(origin, dir, t_hit, last_edge);
        }
        let t_seg = select(t_hit, edge.t, edge.found);

        // 3. specular (or bent) chains into a source sphere, priced from
        // the source's REAL position
        if ((specular || bent) && refl >= lo && refl <= hi) {
            for (var si = 0u; si < n_src; si++) {
                let sp = src_pos(si);
                let to = sp - origin;
                let along = dot(to, dir);
                if (along <= 0.0 || along >= t_seg) { continue; }
                let seg_len = length(to);
                let l_s = max(length_m + seg_len, MIN_DIST);
                let r = job.k * l_s;
                let closest = length(origin + dir * along - sp);
                if (closest > r) { continue; }
                // one edge, no reflection: the edge next-event path,
                // already counted exactly on the CPU
                if (refl == 0u && bends == 1u && !specular) { continue; }
                // the exact chain solver's (on the CPU)
                let rel = l_s / SPEED_OF_SOUND - length(sp - job.listener) / SPEED_OF_SOUND;
                if (mirror_only && loud && refl >= 1u && refl <= MAX_COHERENT_REFL && (job.v4 != 0u || refl > job.bounces || rel >= EARLY_S)) {
                    continue;
                }
                let t = seg_transmission(origin, sp);
                let a = air(l_s);
                let w = 4.0 * inv_n / (job.k * job.k * l_s * l_s);
                // a bent path only carries what the straight line from
                // where it bent doesn't
                var gate = vec3<f32>(1.0);
                if (have_bend_from) {
                    if (refl == 0u) {
                        gate = vec3<f32>(1.0) - src_vis(si);
                    } else {
                        gate = clamp(vec3<f32>(1.0) - seg_transmission(bend_from, sp), vec3<f32>(0.0), vec3<f32>(1.0));
                    }
                }
                deposit_spread(si, thr * t * t * a * a * w * gate, l_s / SPEED_OF_SOUND, r / SPEED_OF_SOUND);
            }
        }

        if (edge.found) {
            // bend around the edge at the closest approach
            last_edge = edge.idx;
            if (!have_bend_from) {
                bend_from = origin;
                have_bend_from = true;
            }
            let eps = sample_bend(edge.dist);
            origin = origin + dir * edge.t;
            length_m += edge.t;
            push_vert(origin);
            dir = rotate_about(dir, edge.u, eps);
            thr = thr * bend_w;
            bent = true;
            bends += 1u;
            mirror_only = false;
            continue;
        }

        if (t_hit >= T_MISS) {
            // escaped to open sky, along the ray's first direction
            let mine = select(refl <= hi, refl >= lo, late);
            if (mine && !escapes_into_earth(origin, dir)) {
                let o = ray_cell * SLOT;
                let e = thr * inv_n * FX_SCALE;
                add_fx(o, e.x);
                add_fx(o + 1u, e.y);
                add_fx(o + 2u, e.z);
                let dm = ray_first * e.y;
                add_fx(o + 3u, dm.x);
                add_fx(o + 4u, dm.y);
                add_fx(o + 5u, dm.z);
            }
            break;
        }
        let p = origin + dir * t_hit;
        push_vert(p);
        length_m += t_hit;
        if (length_m / SPEED_OF_SOUND > select(MAX_TIME, RESPONSE_MAX_S, job.v4 != 0u)) { break; }
        let normal = hit.normal;
        let t2 = hit.transmission * hit.transmission;
        let keep = max(vec3<f32>(1.0) - hit.absorption - t2, vec3<f32>(0.0));

        // 2. next-event estimation: the diffuse share of this surface,
        // lit by the sources it can reach (all, or `nee` importance draws)
        if (hit.scattering > 0.0 && refl + 1u >= lo && refl + 1u <= hi) {
            let off = p + normal * EPS;
            let all_src = nee == 0u || n_src <= nee;
            var sum = 0.0;
            if (!all_src) {
                for (var si = 0u; si < n_src; si++) {
                    let dv = src_pos(si) - off;
                    sum += max(src_weight(si), 1e-12) / (dot(dv, dv) + 1.0);
                    cum[si] = sum;
                }
            }
            let picks = select(nee, n_src, all_src);
            for (var m = 0u; m < picks; m++) {
                var si = m;
                var pw = 1.0;
                if (!all_src) {
                    // partition_point(|c| c < u): the first cum ≥ u
                    let u = rand_f32() * sum;
                    var lo_i = 0u;
                    var hi_i = n_src;
                    while (lo_i < hi_i) {
                        let mid = (lo_i + hi_i) / 2u;
                        if (cum[mid] < u) { lo_i = mid + 1u; } else { hi_i = mid; }
                    }
                    si = min(lo_i, n_src - 1u);
                    var prev = 0.0;
                    if (si > 0u) { prev = cum[si - 1u]; }
                    let pi = (cum[si] - prev) / sum;
                    pw = 1.0 / (f32(nee) * max(pi, 1e-12));
                }
                let sp = src_pos(si);
                let to = sp - off;
                let r = length(to);
                if (r < 1e-4) { continue; }
                let cos_in = dot(to, normal) / r;
                if (cos_in <= 0.0) { continue; }
                let t = seg_transmission(off, sp);
                let rr = max(r, MIN_DIST);
                let a = air(length_m + r);
                let w = pw * hit.scattering * cos_in * 4.0 * inv_n / (rr * rr);
                deposit(si, thr * keep * t * t * a * a * w, (length_m + r) / SPEED_OF_SOUND);
            }
        }

        // 4. continue: v4 splits at the first partition passing real
        // energy (both branches, their exact weights) — except a sliced
        // walk's early rays, whose walk must end within the walk's passes
        // (the late pool carries its rays on); else transmit
        // (importance-floored) or reflect
        let p_t = max(max(t2.x, t2.y), t2.z);
        let split_here = job.v4 != 0u && !split && p_t >= SPLIT_T2 && (late || !sliced);
        if (split_here) {
            split = true;
            has_pending = true;
            // the wall as a whole (v3.rs World::through)
            let tw = through(p, dir, hit);
            p_origin = beyond;
            p_dir = dir;
            p_thr = thr * tw;
            p_length = length_m;
            p_refl = refl;
            p_mirror_thr = mirror_thr * tw.y;
            p_flags = select(0u, 1u, specular) | select(0u, 2u, mirror_only)
                | select(0u, 4u, loud && p_mirror_thr >= COHERENT_FLOOR) | select(0u, 8u, bent)
                | select(0u, 16u, have_bend_from);
            p_bends = bends;
            p_bend_from = bend_from;
            p_last_edge = last_edge;
            p_seg = seg + 1u;
        }
        let pt = select(select(0.0, min(max(p_t, 0.02), 0.5), p_t > 1e-5), 0.0, split_here);
        if (pt > 0.0 && rand_f32() < pt) {
            let tw = through(p, dir, hit);
            thr = thr * tw / pt;
            mirror_thr = mirror_thr * tw.y;
            loud = loud && mirror_thr >= COHERENT_FLOOR;
            origin = beyond;
            continue; // straight on; `specular` unchanged
        }
        let before = max(max(thr.x, thr.y), thr.z);
        thr = thr * keep / (1.0 - pt);
        if (job.v4 != 0u && late && length_m > EARLY_S * SPEED_OF_SOUND) {
            // v3.rs: v4's late rays' Russian roulette ∝ amplitude, past the
            // early window
            let after = max(max(thr.x, thr.y), thr.z);
            let p = select(0.0, min(sqrt(after / before), 1.0), before > 0.0);
            if (rand_f32() >= p) { break; }
            thr = thr / p;
        }
        if (all(thr < vec3<f32>(LATE_FLOOR))) { break; }
        refl += 1u;
        have_bend_from = false;
        mirror_thr = mirror_thr * keep.y * (1.0 - hit.scattering);
        loud = loud && mirror_thr >= COHERENT_FLOOR;
        if (rand_f32() < hit.scattering) {
            var nd = normalize(normal + unit_sphere());
            if (dot(nd, normal) < 1e-3) { nd = normal; }
            dir = nd;
            specular = false;
            bent = false;
            mirror_only = false;
        } else {
            dir = dir - normal * (2.0 * dot(dir, normal));
            specular = true;
        }
        origin = p + normal * EPS;
    }
    // the branch ended: the split's transmitted one walks next
    if (!has_pending) { break; }
    has_pending = false;
    origin = p_origin;
    dir = p_dir;
    thr = p_thr;
    length_m = p_length;
    refl = p_refl;
    specular = (p_flags & 1u) != 0u;
    mirror_only = (p_flags & 2u) != 0u;
    loud = (p_flags & 4u) != 0u;
    bent = (p_flags & 8u) != 0u;
    have_bend_from = (p_flags & 16u) != 0u;
    mirror_thr = p_mirror_thr;
    bends = p_bends;
    bend_from = p_bend_from;
    last_edge = p_last_edge;
    seg0 = p_seg;
    n_verts = 1u;
    push_vert(origin);
    }
    if (sliced) {
        dbg[so + 27u] = 0u; // ended
        dbg[so + 55u] = 0u;
    }
}

// ------------------------------------------------------------- resolve
// One thread per output word: the fixed-point accumulators back to f32
// true units; a source's per-band total is its echogram's sum.
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let n_out = SKY_WORDS + job.n_sources * SOURCE_WORDS;
    if (w >= n_out) { return; }
    if (w < SKY_WORDS) {
        out[w] = read_fx(w) / FX_SCALE;
        return;
    }
    let si = (w - SKY_WORDS) / SOURCE_WORDS;
    let local = (w - SKY_WORDS) % SOURCE_WORDS;
    let sc = src_norm(si) * FX_SCALE;
    if (job.v4 != 0u) {
        if (local < DENSE_WORDS) {
            out[w] = read_fx(w) / sc;
            return;
        }
        let b = local - DENSE_WORDS;
        var sum = 0.0;
        if (b < 3u) {
            let base = SKY_WORDS + si * SOURCE_WORDS;
            for (var bin = 0u; bin < DENSE_BINS; bin++) {
                sum += read_fx(base + bin * DENSE_SLOT + b);
            }
        }
        out[w] = sum / sc;
        return;
    }
    if (local < EARLY_WORDS + ECHO_WORDS) {
        out[w] = read_fx(w) / sc;
        return;
    }
    let b = local - EARLY_WORDS - ECHO_WORDS;
    var sum = 0.0;
    if (b < 3u) {
        let base = SKY_WORDS + si * SOURCE_WORDS + EARLY_WORDS;
        for (var bin = 0u; bin < ECHO_BINS; bin++) {
            sum += read_fx(base + bin * SLOT + b);
        }
    }
    out[w] = sum / sc;
}

// ---------------------------------------------------------------- probe
// Deterministic parity hook for the native tests (not used by the walk):
// for job.n_early records of PROBE_WORDS in `dbg` — segment a (xyz), b
// (xyz) — writes World::transmission(a, b) at +6, World::raycast from a
// toward b (t, T_MISS on a miss) at +9 and its normal at +10, the earth
// predicates in_earth(a), crosses_earth(a, b) at +13, +14, and the
// nearest usable edge along a→b (found, t, distance) at +15..+17.
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= job.n_early) { return; }
    let o = i * PROBE_WORDS;
    let a = vec3<f32>(bitcast<f32>(dbg[o]), bitcast<f32>(dbg[o + 1u]), bitcast<f32>(dbg[o + 2u]));
    let b = vec3<f32>(bitcast<f32>(dbg[o + 3u]), bitcast<f32>(dbg[o + 4u]), bitcast<f32>(dbg[o + 5u]));
    let t = seg_transmission(a, b);
    dbg[o + 6u] = bitcast<u32>(t.x);
    dbg[o + 7u] = bitcast<u32>(t.y);
    dbg[o + 8u] = bitcast<u32>(t.z);
    let d = normalize(b - a);
    let h = raycast_world(a, d);
    dbg[o + 9u] = bitcast<u32>(h.t);
    dbg[o + 10u] = bitcast<u32>(h.normal.x);
    dbg[o + 11u] = bitcast<u32>(h.normal.y);
    dbg[o + 12u] = bitcast<u32>(h.normal.z);
    dbg[o + 13u] = select(0u, 1u, in_earth(a));
    dbg[o + 14u] = select(0u, 1u, crosses_earth(a, b));
    let e = nearest_edge(a, d, length(b - a), NO_EDGE);
    dbg[o + 15u] = select(0u, 1u, e.found);
    dbg[o + 16u] = bitcast<u32>(e.t);
    dbg[o + 17u] = bitcast<u32>(e.dist);
}
