// C7a kernel K4 — the batched (source × chain) exact solve. One thread
// = one pair: mirror the source across the chain's planes (the same
// machine-exact construction as pt_mesh::mesh_record), validate every
// bounce against the real mesh (rect faces for furniture overlays),
// accumulate mass-law transmission over every surface each leg
// crosses, and emit a solved record or an invalid slot. The BVH,
// prims, materials and surface table are the ones the trace/discovery
// kernels already uploaded — per tick only the chain list, the source
// list and the extras boxes move.
//
// SOLVE_LAYOUT_VERSION 1 — must match crates/omg-gpu/src/layout.rs:
//   Surface {n@0, d@12, refl@16, has_rect@28, trans@32,
//            rmin@48, rmax@64}                            80 B
//   Job     {n_sources@0, n_chains@4, n_extras@8, listener@16} 32 B
//   Src     {pos@0, id@12}                                16 B
//   Chain   vec2<u32>: c0|c1<<16, c2|order<<16             8 B
//   Extra   {bmin@0, bmax@16, trans@32}                   48 B
//   Rec     {dir@0, delay@12, gains@16, valid@28}         32 B
//
// Numeric conventions mirror pt_mesh.rs exactly: epsilons, the
// AUDIBLE_FLOOR, MIN_DIST, air absorption, and the segment-hit dedup
// (coincident wall planes of adjacent rooms are ONE physical surface).

const M_MAX_ORDER: u32 = 3u;
const MIN_DIST: f32 = 0.3;
const AUDIBLE_FLOOR: f32 = 2e-5;
const SPEED_OF_SOUND: f32 = 343.0;
const AIR_ABS = vec3<f32>(1.0e-4, 5.0e-4, 3.0e-3);
const LEAF_BIT: u32 = 0x80000000u;
const T_MISS: f32 = 3.4e38;

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
    _p: u32,
}

struct Surface {
    n: vec3<f32>,
    d: f32,
    refl: vec3<f32>,
    has_rect: u32,
    trans: vec3<f32>,
    _p0: u32,
    rmin: vec3<f32>,
    _p1: u32,
    rmax: vec3<f32>,
    _p2: u32,
}

struct Job {
    n_sources: u32,
    n_chains: u32,
    n_extras: u32,
    _p0: u32,
    listener: vec3<f32>,
    _p1: u32,
}

struct SrcIn {
    pos: vec3<f32>,
    id: u32,
}

struct Extra {
    bmin: vec3<f32>,
    _p0: u32,
    bmax: vec3<f32>,
    _p1: u32,
    trans: vec3<f32>,
    _p2: u32,
}

struct Rec {
    dir: vec3<f32>,
    delay: f32,
    gains: vec3<f32>,
    valid: u32,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> prims: array<Prim>;
@group(0) @binding(3) var<storage, read> mats: array<Mat>;
@group(0) @binding(4) var<storage, read> surfs: array<Surface>;
@group(0) @binding(5) var<storage, read> chains: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> sources: array<SrcIn>;
@group(0) @binding(7) var<storage, read> extras: array<Extra>;
@group(0) @binding(8) var<storage, read_write> recs: array<Rec>;

// Möller–Trumbore, both faces — identical to Mesh::ray_tri_packed.
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

// Slab entry distance vs tmax; t < 0 means miss (Mesh::ray_aabb).
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

// Nearest mesh hit — the ORDERED traversal of Mesh::raycast, near child
// popped first, strict `t < best`, so coincident-plane ties resolve in
// the same traversal order as the CPU. Returns (t, prim); t==T_MISS on
// miss.
struct RayHit {
    t: f32,
    prim: u32,
}

fn raycast_prim(o: vec3<f32>, d: vec3<f32>) -> RayHit {
    var h: RayHit;
    h.t = T_MISS;
    h.prim = 0u;
    let inv_d = safe_inv(d);
    var stack_i: array<u32, 64>;
    var stack_t: array<f32, 64>;
    var sp = 1u;
    stack_i[0] = 0u;
    stack_t[0] = 0.0;
    while (sp > 0u) {
        sp -= 1u;
        let t_enter = stack_t[sp];
        if (t_enter >= h.t) { continue; }
        let node = nodes[stack_i[sp]];
        if ((node.a & LEAF_BIT) != 0u) {
            let start = node.a & ~LEAF_BIT;
            for (var i = 0u; i < node.b; i++) {
                let p = prims[start + i];
                let t = ray_tri(p.a, p.e1, p.e2, o, d);
                if (t > 0.0 && t < h.t) {
                    h.t = t;
                    h.prim = start + i;
                }
            }
        } else {
            let l = nodes[node.a];
            let r = nodes[node.b];
            let tl = ray_aabb(l.bmin, l.bmax, o, inv_d, h.t);
            let tr = ray_aabb(r.bmin, r.bmax, o, inv_d, h.t);
            if (tl >= 0.0 && tr >= 0.0 && sp + 2u <= 64u) {
                // far first, near on top — matches Mesh::raycast
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
    return h;
}

// Crossing list capacity for the same-t dedup (a crossing on the
// shared plane of coplanar triangles is ONE physical surface, exactly
// as Mesh::segment_hits dedups). Small and per-thread.
const SEG_CAP: u32 = 24u;

// Every distinct crossing of the OPEN segment a→b, deduplicated the
// way Mesh::segment_hits does: within a coincident cluster the LOWEST
// surface id wins. Adjacent rooms author a shared wall twice with
// different materials, and the two triangles' float t order flips with
// sub-ulp ray noise — an id rule is the only tie-break the CPU and the
// GPU can agree on, and which material is paid must match or every
// through-that-wall gain drifts by the material ratio.
struct SegHits {
    n: u32,
    t: array<f32, SEG_CAP>,
    mat: array<u32, SEG_CAP>,
    surf: array<u32, SEG_CAP>,
}

fn collect_seg_hits(a: vec3<f32>, b: vec3<f32>, out: ptr<function, SegHits>) {
    (*out).n = 0u;
    let dvec = b - a;
    let len = length(dvec);
    if (len < 1e-6) { return; }
    let dn = dvec / len;
    let inv_d = safe_inv(dn);
    let tol = 1e-4 / max(len, 1.0);
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
                var dup = false;
                for (var j = 0u; j < (*out).n; j++) {
                    if (abs(tt - (*out).t[j]) < tol) {
                        if (p.surf < (*out).surf[j]) {
                            // keep the cluster's anchor t, take the
                            // winner's identity — mirrors segment_hits
                            (*out).mat[j] = p.mat;
                            (*out).surf[j] = p.surf;
                        }
                        dup = true;
                        break;
                    }
                }
                if (dup) { continue; }
                if ((*out).n < SEG_CAP) {
                    (*out).t[(*out).n] = tt;
                    (*out).mat[(*out).n] = p.mat;
                    (*out).surf[(*out).n] = p.surf;
                    (*out).n += 1u;
                }
            }
        } else if (sp + 2u <= 64u) {
            stack[sp] = node.a;
            stack[sp + 1u] = node.b;
            sp += 2u;
        }
    }
}

// Transmission of every surface the OPEN segment a→b crosses — the
// leg_transmission mesh walk. `skip_sid >= 0` skips crossings of that
// surface within 5 cm of the leg END (the reflection surface itself at
// its own t). Multiplies into `gm`.
fn leg_mesh_trans(a: vec3<f32>, b: vec3<f32>, skip_sid: i32, gm: ptr<function, vec3<f32>>) {
    var hits: SegHits;
    collect_seg_hits(a, b, &hits);
    let len = length(b - a);
    for (var j = 0u; j < hits.n; j++) {
        if (skip_sid >= 0 && i32(hits.surf[j]) == skip_sid && abs(hits.t[j] - 1.0) * len < 0.05) {
            continue;
        }
        *gm = *gm * mats[hits.mat[j]].transmission;
    }
}

// Does the segment a→b cross surface `sid` at distance ~expect_t? The
// mesh_record fallback when a transmissive blocker stands in front of
// the bounce point — the chain surface must still EXIST there. The
// dedup applies first, so a coincident foreign plane can shadow the
// hit exactly as on the CPU.
fn seg_has_surface(a: vec3<f32>, b: vec3<f32>, sid: u32, expect_t: f32) -> bool {
    var hits: SegHits;
    collect_seg_hits(a, b, &hits);
    let len = length(b - a);
    for (var j = 0u; j < hits.n; j++) {
        if (hits.surf[j] == sid && abs(hits.t[j] * len - expect_t) < 0.1) { return true; }
    }
    return false;
}

// Aabb::clip — does the open segment cross the box interior?
fn extra_crosses(a: vec3<f32>, b: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    let d = b - a;
    var t0 = 0.0;
    var t1 = 1.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let da = d[axis];
        if (abs(da) < 1e-9) {
            if (a[axis] < bmin[axis] || a[axis] > bmax[axis]) { return false; }
        } else {
            var ta = (bmin[axis] - a[axis]) / da;
            var tb = (bmax[axis] - a[axis]) / da;
            if (ta > tb) { let tmp = ta; ta = tb; tb = tmp; }
            t0 = max(t0, ta);
            t1 = min(t1, tb);
            if (t0 > t1) { return false; }
        }
    }
    return t1 > 1e-4 && t0 < 1.0 - 1e-4;
}

// leg_transmission: mesh crossings (skip the reflection surface at its
// own t) then extras boxes.
fn leg_trans(a: vec3<f32>, b: vec3<f32>, skip_sid: i32, gm: ptr<function, vec3<f32>>) {
    leg_mesh_trans(a, b, skip_sid, gm);
    for (var i = 0u; i < job.n_extras; i++) {
        let x = extras[i];
        if (extra_crosses(a, b, x.bmin, x.bmax)) {
            *gm = *gm * x.trans;
        }
    }
}

fn mirror_p(p: vec3<f32>, si: u32) -> vec3<f32> {
    let s = surfs[si];
    return p - s.n * (2.0 * (dot(s.n, p) - s.d));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    let n_pairs = job.n_sources * job.n_chains;
    if (pair >= n_pairs) { return; }
    // invalid until proven — the host never clears the buffer
    recs[pair].valid = 0u;

    let si = pair / job.n_chains;
    let ci = pair % job.n_chains;
    let cw = chains[ci];
    var chain: array<u32, 3>;
    chain[0] = cw.x & 0xFFFFu;
    chain[1] = cw.x >> 16u;
    chain[2] = cw.y & 0xFFFFu;
    let order = min(cw.y >> 16u, M_MAX_ORDER);
    let src = sources[si].pos;
    let lis = job.listener;
    let n_surfs = arrayLength(&surfs);
    for (var k = 0u; k < order; k++) {
        if (chain[k] >= n_surfs) { return; }
    }

    // image: mirror the source across the chain planes, listener-first
    // chain c0 applied last
    var img = src;
    for (var k = order; k > 0u; k--) {
        img = mirror_p(img, chain[k - 1u]);
    }
    let to_img = img - lis;
    let dist = length(to_img);
    if (dist < 1e-4) { return; }
    let dir = to_img / dist;

    var gm = vec3<f32>(1.0);
    var pos = lis;
    var d = dir;
    for (var k = 0u; k < order; k++) {
        let sid = chain[k];
        let s = surfs[sid];
        let denom = dot(s.n, d);
        if (abs(denom) < 1e-9) { return; }
        let t = (s.d - dot(s.n, pos)) / denom;
        if (t <= 1e-4) { return; }
        let hit = pos + d * t;
        if (s.has_rect != 0u) {
            // overlay face (furniture): the reflection point must land
            // ON the face rect — there is no mesh to raycast for it
            if (any(hit < s.rmin - vec3<f32>(1e-3)) || any(hit > s.rmax + vec3<f32>(1e-3))) {
                return;
            }
            // a NEARER opaque mesh blocker still kills the specular claim
            let rh = raycast_prim(pos + d * 1e-4, d);
            if (rh.t < T_MISS && rh.t < t - 1e-3) {
                let m = mats[prims[rh.prim].mat];
                if (all(m.transmission < vec3<f32>(1e-6))) { return; }
            }
        } else {
            // the reflection point must land on REAL mesh of this
            // surface at its own distance (the plane is infinite, the
            // wall is not)
            let rh = raycast_prim(pos + d * 1e-4, d);
            if (rh.t >= T_MISS) { return; }
            let on_surface = prims[rh.prim].surf == sid && abs(rh.t - t) < 2e-2;
            if (!on_surface) {
                if (rh.t < t - 1e-3) {
                    // a nearer blocker: opaque kills the claim; a
                    // transmissive one may stand in front, but the
                    // chain surface must still EXIST at the bounce
                    let m = mats[prims[rh.prim].mat];
                    if (all(m.transmission < vec3<f32>(1e-4))) { return; }
                    if (!seg_has_surface(pos, hit + d * 0.05, sid, t)) { return; }
                } else {
                    return;
                }
            }
        }
        // transmission across everything this leg crosses (the
        // reflection surface itself excluded at its own t)
        leg_trans(pos, hit, i32(sid), &gm);
        gm = gm * s.refl;
        if (all(gm < vec3<f32>(1e-7))) { return; }
        pos = hit + s.n * select(-1e-4, 1e-4, dot(s.n, d) < 0.0);
        d = d - s.n * (2.0 * dot(s.n, d));
    }
    // ISM validity: the source must lie on the SAME side of the last
    // mirror plane as the reflected ray (a source BEHIND the mirror is
    // a transmission ghost wearing a reflection costume)
    if (order > 0u) {
        let s = surfs[chain[order - 1u]];
        if ((dot(s.n, pos) - s.d) * (dot(s.n, src) - s.d) < 0.0) { return; }
    }
    // final leg to the source, transmission for every crossing
    leg_trans(pos, src, -1, &gm);
    if (order > 0u) {
        let to_src = src - pos;
        let leg = length(to_src);
        if (leg > 1e-4 && dot(to_src / leg, d) < 0.995) { return; }
    }

    let dtot = max(dist, MIN_DIST);
    let air = exp(-AIR_ABS * dtot);
    let gains = air / dtot * gm;
    if (all(gains < vec3<f32>(AUDIBLE_FLOOR))) { return; }
    recs[pair] = Rec(dir, dist / SPEED_OF_SOUND, gains, 1u);
}
