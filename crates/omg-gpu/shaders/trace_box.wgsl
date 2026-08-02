// Stochastic shoebox energy tracer — GPU port of omg-core's trace()
// (crates/omg-core/src/tracer.rs). One thread = one ray. Preserves the
// CPU algorithm exactly:
//   · receiver-sphere check BEFORE advancing to the hit, including the
//     final escaping segment (that segment is how sound leaves a room
//     through a doorway and reaches a listener outside);
//   · NO Russian roulette — stochastic termination corrupts the RT60
//     fit in low-absorption rooms (measured, see tracer.rs comment);
//   · 64-bounce cap, 3.0 s time cutoff, 1e-7·per_ray energy floor,
//     1e-3 m surface nudge, Lambertian-vs-specular on mat.scattering
//     with the grazing-direction guard.
// RNG differs by design (PCG hash chain vs xorshift64*): parity is
// statistical, enforced by the Phase 0 goldens.
//
// LAYOUT_VERSION 2 — must match crates/omg-gpu/src/layout.rs.
// Job uniform offsets: size@0 n_rays@12 source@16 seed@28 listener@32
// energy@48 faces@64 (6×32 B) — total 256 B.
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

struct Face {
    absorption: vec3<f32>,
    scattering: f32,
    transmission: vec3<f32>,
    _p: u32,
}

struct Job {
    size: vec3<f32>,
    n_rays: u32,
    source: vec3<f32>,
    seed: u32,
    listener: vec3<f32>,
    _pad0: u32,
    energy: vec3<f32>,
    _pad1: u32,
    faces: array<Face, 6>,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 900>; // NBINS*NBANDS
@group(0) @binding(2) var<storage, read_write> dirs: array<atomic<i32>, 900>; // NBINS*3

// PCG output hash iterated as a stream. 32-bit state is enough here:
// each ray draws a few hundred samples and streams are decorrelated by
// seeding with hash(job.seed, ray_id).
var<private> rng_state: u32;

fn pcg_next() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32() -> f32 {
    // 24-bit mantissa fraction in [0,1), like Rng::next_f32
    return f32(pcg_next() >> 8u) / 16777216.0;
}

fn unit_sphere() -> vec3<f32> {
    let z = 1.0 - 2.0 * rand_f32();
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 6.28318530718 * rand_f32();
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// Shoebox::raycast for a ray starting inside the box: nearest wall,
// returns t in .w and wall index encoded via best_w.
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

fn wall_normal(w: u32) -> vec3<f32> {
    // inward-facing: +1 on a min wall, -1 on a max wall
    var n = vec3<f32>(0.0);
    n[w / 2u] = select(-1.0, 1.0, (w % 2u) == 0u);
    return n;
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    if (ray >= job.n_rays) { return; }
    rng_state = (job.seed ^ (ray * 0x9E3779B9u)) + 1u;
    // decorrelate the first draws from the seed pattern
    pcg_next();
    pcg_next();

    let per_ray = 1.0 / f32(job.n_rays);
    var pos = job.source;
    var dir = unit_sphere();
    var energy = job.energy * per_ray;
    var dist_total = 0.0;

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let hit = raycast_box(pos, dir);
        let t_hit = hit.x;
        let wall = u32(hit.y);
        // t_hit is always finite for a point inside the box; a
        // degenerate outside-the-box state ends below via the checks.

        // Receiver sphere crossing along this segment (BEFORE advancing;
        // covers the escaping segment on a miss — here the box always
        // hits, but the structure mirrors the CPU tracer exactly).
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
                    // arrival direction: back along the ray, mid band
                    let de = dir * energy[1] * DIR_SCALE;
                    atomicAdd(&dirs[bin * 3u + 0u], i32(round(-de.x)));
                    atomicAdd(&dirs[bin * 3u + 1u], i32(round(-de.y)));
                    atomicAdd(&dirs[bin * 3u + 2u], i32(round(-de.z)));
                }
            }
        }

        // Advance to the surface, absorb, pick specular or diffuse.
        pos = pos + dir * t_hit;
        dist_total = dist_total + t_hit;
        if (dist_total / SPEED_OF_SOUND > MAX_TIME) { break; }

        let mat = job.faces[wall];
        // through-the-wall branch (mass law) — mirrors tracer.rs
        let t2 = mat.transmission * mat.transmission;
        let p_raw = max(max(t2.x, t2.y), t2.z);
        // importance floor — mirrors tracer.rs: branch often, weight less
        let p_t = select(0.0, min(max(p_raw, 0.02), 0.5), p_raw > 1e-5);
        let floor_e = 1e-7 * per_ray;
        if (p_t > 0.0 && rand_f32() < p_t) {
            // transmitted rays leave the box's world: outside is void
            break;
        }
        energy = energy * max(vec3<f32>(1.0) - mat.absorption - t2, vec3<f32>(0.0)) / (1.0 - p_t);
        if (energy.x <= floor_e && energy.y <= floor_e && energy.z <= floor_e) { break; }

        let normal = wall_normal(wall);
        if (rand_f32() < mat.scattering) {
            // Lambertian: normal + uniform sphere point, renormalized,
            // with the grazing/degenerate guard.
            var nd = normalize(normal + unit_sphere());
            if (dot(nd, normal) < 1e-3) { nd = normal; }
            dir = nd;
        } else {
            dir = dir - normal * (2.0 * dot(dir, normal));
        }
        pos = pos + normal * WALL_EPS;
    }
}
