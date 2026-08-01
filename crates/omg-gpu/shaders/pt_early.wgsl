// PT-early chain discovery (GPU_PLAN.md Track C phase C3) — the GPU
// half of omg-core's pt_chains(). One thread = one ray of the rotating
// golden-spiral fan, bounced specularly ≤3 times over the analytic
// box; every prefix of the wall sequence marks one bit in a 258-slot
// chain-occupancy bitmap. Chains are source-independent in a convex
// box, so the entire dispatch output is 9 words — the host decodes
// bits back into chains and the CPU-side cache does the exact solving.
//
// Bitmap layout (base-6 chain index):
//   order 1: bit  w1                    (6 slots)
//   order 2: bit  6 + w1*6 + w2         (36 slots)
//   order 3: bit 42 + w1*36 + w2*6 + w3 (216 slots)
// LAYOUT_VERSION 1 — must match omg-gpu/src/lib.rs and web/gpu.js.

struct PtJob {
    size: vec3<f32>,
    n_rays: u32,
    listener: vec3<f32>,
    rot: u32,
}

@group(0) @binding(0) var<uniform> job: PtJob;
@group(0) @binding(1) var<storage, read_write> bitmap: array<atomic<u32>, 9>;

fn raycast_box(p: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    var best_t = 3.4e38;
    var best_w = 0u;
    for (var axis = 0u; axis < 3u; axis++) {
        let di = d[axis];
        if (di > 1e-9) {
            let t = (job.size[axis] - p[axis]) / di;
            if (t < best_t) { best_t = t; best_w = 2u * axis + 1u; }
        } else if (di < -1e-9) {
            let t = -p[axis] / di;
            if (t < best_t) { best_t = t; best_w = 2u * axis; }
        }
    }
    return vec2<f32>(best_t, f32(best_w));
}

fn mark(idx: u32) {
    atomicOr(&bitmap[idx >> 5u], 1u << (idx & 31u));
}

@compute @workgroup_size(64)
fn discover(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= job.n_rays) { return; }
    let ga = 2.39996322973; // golden angle
    let z = 1.0 - 2.0 * (f32(i) + 0.5) / f32(job.n_rays);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = ga * f32(i) + f32(job.rot) * 0.61803398875 * 6.28318530718;
    var dir = vec3<f32>(r * cos(phi), r * sin(phi), z);
    var pos = job.listener;

    var c1 = 0u; var c2 = 0u;
    for (var k = 0u; k < 3u; k++) {
        let hit = raycast_box(pos, dir);
        let t = hit.x;
        if (t <= 1e-5) { break; }
        let w = u32(hit.y);
        pos = pos + dir * t;
        if (k == 0u) { c1 = w; mark(c1); }
        else if (k == 1u) { c2 = w; mark(6u + c1 * 6u + c2); }
        else { mark(42u + c1 * 36u + c2 * 6u + w); }
        var n = vec3<f32>(0.0);
        n[w / 2u] = select(-1.0, 1.0, (w % 2u) == 0u);
        dir = dir - n * (2.0 * dot(dir, n));
        pos = pos + n * 1e-5;
    }
}
