// Stochastic WORLD-MESH energy tracer (C6d, kernel K2) — the same
// tracer as trace_box.wgsl over a flattened BVH: one thread = one ray,
// identical receiver/bounce/termination rules. Intersection is ordered
// BVH traversal + Möller–Trumbore, plus PANEL overlay boxes (door
// leaves, glass panes — world STATE, not authored mesh) that reflect
// like any surface. A ray that misses everything has escaped through a
// real opening; its final segment still passes the receiver check —
// that segment is how sound leaves a room through a doorway.
//
// MESH_LAYOUT_VERSION 1 — must match crates/omg-gpu/src/layout.rs:
//   Node  {bmin@0, a@12, bmax@16, b@28}                 32 B
//   Prim  {a@0, mat@12, e1@16, e2@32}                   48 B
//   Panel {pmin@0, scattering@12, pmax@16, absorption@32} 48 B
//   Job   {n_rays@0, seed@4, n_panels@8, source@16,
//          listener@32, energy@48}                      64 B
// Fixed point: energy u32 = e * 2^30; direction i32 = d·e * 2^28.

const NBANDS: u32 = 3u;
const NBINS: u32 = 300u;
const BIN_DT: f32 = 0.010;
const MAX_TIME: f32 = 3.0;
const SPEED_OF_SOUND: f32 = 343.0;
const RECEIVER_RADIUS: f32 = 0.5;
const WALL_EPS: f32 = 1e-3;
const MAX_BOUNCES: u32 = 64u;
const ENERGY_SCALE: f32 = 1073741824.0; // 2^30
const DIR_SCALE: f32 = 268435456.0; // 2^28
const LEAF_BIT: u32 = 0x80000000u;
const T_MISS: f32 = 3.4e38;

struct Node {
    bmin: vec3<f32>,
    a: u32, // leaf: LEAF_BIT | prim start; internal: left child
    bmax: vec3<f32>,
    b: u32, // leaf: prim count; internal: right child
}

struct Prim {
    a: vec3<f32>,
    mat: u32,
    e1: vec3<f32>,
    _p0: u32,
    e2: vec3<f32>,
    _p1: u32,
}

struct Panel {
    pmin: vec3<f32>,
    scattering: f32,
    pmax: vec3<f32>,
    _p0: u32,
    absorption: vec3<f32>,
    _p1: u32,
}

struct Mat {
    absorption: vec3<f32>,
    scattering: f32,
}

struct Job {
    n_rays: u32,
    seed: u32,
    n_panels: u32,
    _p0: u32,
    source: vec3<f32>,
    _p1: u32,
    listener: vec3<f32>,
    _p2: u32,
    energy: vec3<f32>,
    _p3: u32,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> prims: array<Prim>;
@group(0) @binding(3) var<storage, read> mats: array<Mat>;
@group(0) @binding(4) var<storage, read> panels: array<Panel>;
@group(0) @binding(5) var<storage, read_write> bins: array<atomic<u32>, 900>; // NBINS*NBANDS
@group(0) @binding(6) var<storage, read_write> dirs: array<atomic<i32>, 900>; // NBINS*3

var<private> rng_state: u32;

fn pcg_next() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32() -> f32 {
    return f32(pcg_next() >> 8u) / 16777216.0;
}

fn unit_sphere() -> vec3<f32> {
    let z = 1.0 - 2.0 * rand_f32();
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 6.28318530718 * rand_f32();
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

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

// Hit result, flattened for WGSL (no Option): t == T_MISS means miss.
struct Hit {
    t: f32,
    normal: vec3<f32>,
    absorption: vec3<f32>,
    scattering: f32,
}

fn raycast_world(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h: Hit;
    h.t = T_MISS;
    // robust slab: kill exact-zero components so 0*inf NaNs can't appear
    let dd = select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    let inv_d = vec3<f32>(1.0) / dd;

    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    var best_prim = 0u;
    var have_prim = false;
    while (sp > 0u) {
        sp -= 1u;
        let node = nodes[stack[sp]];
        // slab test against current best
        let t1 = (node.bmin - o) * inv_d;
        let t2 = (node.bmax - o) * inv_d;
        let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
        let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
        if (tmax < max(tmin, 0.0) || tmin >= h.t) { continue; }
        if ((node.a & LEAF_BIT) != 0u) {
            let start = node.a & ~LEAF_BIT;
            for (var i = 0u; i < node.b; i++) {
                let p = prims[start + i];
                let t = ray_tri(p.a, p.e1, p.e2, o, d);
                if (t > 0.0 && t < h.t) {
                    h.t = t;
                    best_prim = start + i;
                    have_prim = true;
                }
            }
        } else if (sp + 2u <= 64u) {
            stack[sp] = node.a;
            stack[sp + 1u] = node.b;
            sp += 2u;
        }
    }
    if (have_prim) {
        let p = prims[best_prim];
        var n = normalize(cross(p.e1, p.e2));
        if (dot(n, d) > 0.0) { n = -n; }
        h.normal = n;
        let m = mats[p.mat];
        h.absorption = m.absorption;
        h.scattering = m.scattering;
    }

    // panel overlays: slab ENTRY test, nearest wins
    for (var i = 0u; i < job.n_panels; i++) {
        let pl = panels[i];
        var t0 = 0.0;
        var t1p = T_MISS;
        var axis = 3u;
        for (var a = 0u; a < 3u; a++) {
            var ta = (pl.pmin[a] - o[a]) * inv_d[a];
            var tb = (pl.pmax[a] - o[a]) * inv_d[a];
            if (ta > tb) { let tmp = ta; ta = tb; tb = tmp; }
            if (ta > t0) { t0 = ta; axis = a; }
            t1p = min(t1p, tb);
        }
        if (t0 > 1e-4 && t0 <= t1p && axis < 3u && t0 < h.t) {
            h.t = t0;
            var n = vec3<f32>(0.0);
            n[axis] = select(1.0, -1.0, d[axis] > 0.0);
            h.normal = n;
            h.absorption = pl.absorption;
            h.scattering = pl.scattering;
        }
    }
    return h;
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    if (ray >= job.n_rays) { return; }
    rng_state = (job.seed ^ (ray * 0x9E3779B9u)) + 1u;
    pcg_next();
    pcg_next();

    let per_ray = 1.0 / f32(job.n_rays);
    var pos = job.source;
    var dir = unit_sphere();
    var energy = job.energy * per_ray;
    var dist_total = 0.0;

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let hit = raycast_world(pos, dir);
        let t_hit = hit.t;

        // Receiver sphere crossing along this segment — BEFORE advancing,
        // so the final escaping segment of a miss still counts (that IS
        // sound leaving through a doorway toward an outside listener).
        let to_l = job.listener - pos;
        let s = dot(to_l, dir);
        if (s > 0.0 && s < t_hit) {
            let closest = pos + dir * s;
            if (length(closest - job.listener) < RECEIVER_RADIUS) {
                let arrival = (dist_total + s) / SPEED_OF_SOUND;
                let bin = u32(arrival / BIN_DT);
                if (bin < NBINS) {
                    for (var b = 0u; b < NBANDS; b++) {
                        atomicAdd(&bins[bin * NBANDS + b], u32(round(energy[b] * ENERGY_SCALE)));
                    }
                    let de = dir * energy[1] * DIR_SCALE;
                    atomicAdd(&dirs[bin * 3u + 0u], i32(round(-de.x)));
                    atomicAdd(&dirs[bin * 3u + 1u], i32(round(-de.y)));
                    atomicAdd(&dirs[bin * 3u + 2u], i32(round(-de.z)));
                }
            }
        }

        if (t_hit >= T_MISS) { break; } // escaped through an opening

        pos = pos + dir * t_hit;
        dist_total = dist_total + t_hit;
        if (dist_total / SPEED_OF_SOUND > MAX_TIME) { break; }

        energy = energy * (vec3<f32>(1.0) - hit.absorption);
        let floor_e = 1e-7 * per_ray;
        if (energy.x <= floor_e && energy.y <= floor_e && energy.z <= floor_e) { break; }

        let normal = hit.normal;
        if (rand_f32() < hit.scattering) {
            var nd = normalize(normal + unit_sphere());
            if (dot(nd, normal) < 1e-3) { nd = normal; }
            dir = nd;
        } else {
            dir = dir - normal * (2.0 * dot(dir, normal));
        }
        pos = pos + normal * WALL_EPS;
    }
}
