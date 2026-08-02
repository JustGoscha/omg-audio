// World-mesh chain DISCOVERY (C6d, kernel K3) — the GPU port of
// omg-core's mesh_chains(): a listener-launched rotating golden fan
// bounces specularly over the same flattened BVH the trace kernel
// walks, and every bounce prefix becomes a candidate chain of
// authored-surface ids. Discovery only has to FIND a chain once per
// TTL window — the wasm side exact-solves and dedups — so the output
// is a raw append list the CPU merges; duplicates are free.
//
// One thread = one ray, up to 3 bounces, each prefix appended as two
// u32: (s0 | s1<<16), (s2 | order<<16), with 0xFFFF for "no surface".
// Shares MESH_LAYOUT_VERSION 1 buffers (Node/Prim in layout.rs).

const M_MAX_ORDER: u32 = 3u;
const LEAF_BIT: u32 = 0x80000000u;
const T_MISS: f32 = 3.4e38;
const CAP: u32 = 16384u; // chain slots in the output list

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

struct Job {
    n_rays: u32,
    rot: u32,
    _p0: u32,
    _p1: u32,
    listener: vec3<f32>,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> prims: array<Prim>;
@group(0) @binding(3) var<storage, read_write> chains: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> count: atomic<u32>;

var<private> rng_state: u32;

fn pcg_next() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32() -> f32 {
    return f32(pcg_next() >> 8u) / 16777216.0;
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

struct Hit {
    t: f32,
    prim: u32,
}

fn raycast(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h: Hit;
    h.t = T_MISS;
    let dd = select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    let inv_d = vec3<f32>(1.0) / dd;
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    while (sp > 0u) {
        sp -= 1u;
        let node = nodes[stack[sp]];
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
                    h.prim = start + i;
                }
            }
        } else if (sp + 2u <= 64u) {
            stack[sp] = node.a;
            stack[sp + 1u] = node.b;
            sp += 2u;
        }
    }
    return h;
}

@compute @workgroup_size(64)
fn discover(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    if (ray >= job.n_rays) { return; }
    rng_state = (0xC6B0u ^ job.rot ^ (ray * 0x9E3779B9u)) + 1u;
    pcg_next();

    // rotating golden fan, matching mesh_chains' stratification
    let z = 1.0 - 2.0 * (f32(ray) + 0.5) / f32(job.n_rays);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let ga = 2.39996322972; // pi * (3 - sqrt(5))
    let phi = ga * f32(ray) + f32(job.rot) * 0.61803398875 * 6.28318530718
        + rand_f32() * 0.02;
    var dir = vec3<f32>(r * cos(phi), r * sin(phi), z);
    var pos = job.listener;
    var c = vec3<u32>(0xFFFFu, 0xFFFFu, 0xFFFFu);

    for (var k = 0u; k < M_MAX_ORDER; k++) {
        let h = raycast(pos, dir);
        if (h.t >= T_MISS || h.t <= 1e-4 || h.t > 200.0) { break; }
        let p = prims[h.prim];
        c[k] = p.surf;
        let idx = atomicAdd(&count, 1u);
        if (idx < CAP) {
            chains[idx] = vec2<u32>(
                (c.x & 0xFFFFu) | ((c.y & 0xFFFFu) << 16u),
                (c.z & 0xFFFFu) | ((k + 1u) << 16u),
            );
        }
        pos = pos + dir * h.t;
        var n = normalize(cross(p.e1, p.e2));
        if (dot(n, dir) > 0.0) { n = -n; }
        dir = dir - n * (2.0 * dot(dir, n));
        pos = pos + n * 1e-4;
    }
}
