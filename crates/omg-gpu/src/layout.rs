//! GPU buffer layouts for the trace kernel — #[repr(C)] mirrors of what
//! `shaders/trace_box.wgsl` declares, plus the fixed-point scales shared
//! with the shader. Every struct has a size test (G5: layout drift is
//! the classic silent killer); the WGSL carries the same offsets in a
//! comment. Bump LAYOUT_VERSION on ANY change here or in the WGSL
//! structs — the web driver checks it and falls back to CPU on mismatch
//! instead of decoding garbage.

use omg_core::tracer::{BIN_DT, MAX_TIME};
use omg_core::NBANDS;

pub const LAYOUT_VERSION: u32 = 2;

pub const NBINS: usize = (MAX_TIME / BIN_DT) as usize; // 300
/// Output buffer lengths (u32 words / i32 words).
pub const BINS_LEN: usize = NBINS * NBANDS; // energy, fixed-point u32
pub const DIRS_LEN: usize = NBINS * 3; // direction·energy, fixed-point i32

/// Energy fixed point: per-band totals across all rays are ≤ 1 (each ray
/// starts at source_energy/n_rays and only decays), so 2^30 leaves 4×
/// headroom in a u32 accumulator. Decode: `u as f32 / ENERGY_SCALE`.
pub const ENERGY_SCALE: f32 = (1u32 << 30) as f32;
/// Direction fixed point: signed accumulators (atomic<i32>), components
/// bounded by the same ≤ 1 energy sum. Decode: `i as f32 / DIR_SCALE`.
pub const DIR_SCALE: f32 = (1u32 << 28) as f32;

/// One face's acoustics. WGSL: absorption @0, scattering @12,
/// transmission @16, thickness @28 — 32 B. Transmission drives the
/// tracer's through-the-wall branch (mass law); thickness > 0 overrides
/// the wall thickness the v3 walk infers (`Mesh::material_thickness`;
/// the other kernels ignore it).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFace {
    pub absorption: [f32; 3],
    pub scattering: f32,
    pub transmission: [f32; 3],
    pub thickness: f32,
}

/// The whole trace job as one uniform block.
/// WGSL offsets: size @0, n_rays @12, source @16, seed @28, listener
/// @32, _pad0 @44, energy @48, _pad1 @60, faces @64 (6 × 32 B) = 256 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTraceJob {
    pub size: [f32; 3],
    pub n_rays: u32,
    pub source: [f32; 3],
    pub seed: u32,
    pub listener: [f32; 3],
    pub _pad0: u32,
    pub energy: [f32; 3],
    pub _pad1: u32,
    /// Wall order matches `Shoebox::walls`: 0=x·min 1=x·max 2=y·min
    /// 3=y·max 4=z·min (floor) 5=z·max (ceiling).
    pub faces: [GpuFace; 6],
}

impl GpuTraceJob {
    pub fn new(
        room: &omg_core::scene::Shoebox,
        src: omg_core::vec3::Vec3,
        lis: omg_core::vec3::Vec3,
        n_rays: u32,
        energy: [f32; NBANDS],
        seed: u32,
    ) -> Self {
        let face = |i: usize| GpuFace {
            absorption: room.walls[i].absorption,
            scattering: room.walls[i].scattering,
            transmission: room.walls[i].transmission,
            thickness: 0.0,
        };
        Self {
            size: [room.size.x, room.size.y, room.size.z],
            n_rays,
            source: [src.x, src.y, src.z],
            seed,
            listener: [lis.x, lis.y, lis.z],
            _pad0: 0,
            energy,
            _pad1: 0,
            faces: [face(0), face(1), face(2), face(3), face(4), face(5)],
        }
    }
}

// ------------------------------------------------ C6d: world-mesh trace

/// Bump on ANY change to the mesh-kernel structs here or in
/// trace_mesh.wgsl — the web driver checks it like LAYOUT_VERSION.
pub const MESH_LAYOUT_VERSION: u32 = 3;
/// Panel slots in the panels buffer (door leaves + panes + the late
/// field's significant furniture).
pub const MAX_PANELS: usize = 64;

/// One BVH node, exactly as `Mesh` traverses it. WGSL: bmin @0, a @12,
/// bmax @16, b @28 — 32 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub bmin: [f32; 3],
    /// Leaf: `GPU_LEAF_BIT | prim_start`; internal: left child index.
    pub a: u32,
    pub bmax: [f32; 3],
    /// Leaf: prim count; internal: right child index.
    pub b: u32,
}

/// One packed primitive (vertex + two edges, Möller–Trumbore form).
/// WGSL: a @0, mat @12, e1 @16, surf @28, e2 @32 — 48 B. `surf` is the
/// authored-surface id (C6a) the discovery kernel builds chains from;
/// the trace kernel ignores it.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPrim {
    pub a: [f32; 3],
    pub mat: u32,
    pub e1: [f32; 3],
    pub surf: u32,
    pub e2: [f32; 3],
    pub _p1: u32,
}

/// A transient overlay box (door leaf, glass pane) with INLINE
/// acoustics — panel materials need not exist in the mesh's material
/// table. WGSL: pmin @0, scattering @12, pmax @16, absorption @32,
/// transmission @48 — 64 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPanel {
    pub pmin: [f32; 3],
    pub scattering: f32,
    pub pmax: [f32; 3],
    pub _p0: u32,
    pub absorption: [f32; 3],
    pub _p1: u32,
    pub transmission: [f32; 3],
    pub _p2: u32,
}

/// The world-trace job uniform. WGSL: n_rays @0, seed @4, n_panels @8,
/// source @16, listener @32, energy @48 — 64 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMeshJob {
    pub n_rays: u32,
    pub seed: u32,
    pub n_panels: u32,
    pub _p0: u32,
    pub source: [f32; 3],
    pub _p1: u32,
    pub listener: [f32; 3],
    pub _p2: u32,
    pub energy: [f32; 3],
    pub _p3: u32,
}

// ------------------------------------------ C7a: batched solve (K4)

/// Bump on ANY change to the solve-kernel structs here or in
/// solve_mesh.wgsl.
pub const SOLVE_LAYOUT_VERSION: u32 = 1;
/// Per-dispatch caps — sources beyond the cap fall back to the CPU
/// solve per source; the chain cap sits above the sim's MAX_CHAINS.
pub const MAX_SOLVE_SOURCES: usize = 128;
pub const MAX_SOLVE_CHAINS: usize = 400;
pub const MAX_SOLVE_EXTRAS: usize = 256;

/// One authored plane / overlay face of the SurfaceTable. WGSL: n @0,
/// d @12, refl @16, has_rect @28, trans @32, rmin @48, rmax @64 — 80 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSurface {
    pub n: [f32; 3],
    pub d: f32,
    pub refl: [f32; 3],
    pub has_rect: u32,
    pub trans: [f32; 3],
    pub _p0: u32,
    pub rmin: [f32; 3],
    pub _p1: u32,
    pub rmax: [f32; 3],
    pub _p2: u32,
}

/// The solve job uniform. WGSL: n_sources @0, n_chains @4, n_extras @8,
/// listener @16 — 32 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSolveJob {
    pub n_sources: u32,
    pub n_chains: u32,
    pub n_extras: u32,
    pub _p0: u32,
    pub listener: [f32; 3],
    pub _p1: u32,
}

/// One source in the batch: world position + the caller's source key.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSolveSrc {
    pub pos: [f32; 3],
    pub id: u32,
}

/// A transient blocker box with per-band transmission (door leaves,
/// panes, furniture) — pt::Aabb for the kernel. 48 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuExtra {
    pub bmin: [f32; 3],
    pub _p0: u32,
    pub bmax: [f32; 3],
    pub _p1: u32,
    pub trans: [f32; 3],
    pub _p2: u32,
}

/// One solved record slot, (source-major) pair order. 32 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSolveRec {
    pub dir: [f32; 3],
    pub delay: f32,
    pub gains: [f32; 3],
    pub valid: u32,
}

// ------------------------------------- Engine v3 listener walk (K5)
//
// The flat job and output formats are defined ONCE, by
// omg_core::v3::fold (the wasm side builds jobs without wgpu); these are
// their #[repr(C)] mirrors so the size/offset tests pin the kernel's
// struct layout (v3_walk.wgsl) to that contract.

pub use omg_core::v3::fold::WALK_LAYOUT_VERSION as V3_LAYOUT_VERSION;

/// The walk job uniform — the first `fold::HEADER_WORDS` words of a
/// WalkJob. WGSL: rot0 @0, n_early @12, rot1 @16, n_late @28, rot2 @32,
/// bounces @44, listener @48, seed @60, k @64, nee @68, late_nee @72,
/// n_sources @76, n_panels @80, n_pockets @84, has_grade @88, grade @92,
/// debug @96, n_edges @100, off_panels @104, off_pockets @108,
/// edge_lo @112, off_edges @124, edge_hi @128 — 160 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuV3Job {
    pub rot0: [f32; 3],
    pub n_early: u32,
    pub rot1: [f32; 3],
    pub n_late: u32,
    pub rot2: [f32; 3],
    pub bounces: u32,
    pub listener: [f32; 3],
    pub seed: u32,
    pub k: f32,
    pub nee: u32,
    pub late_nee: u32,
    pub n_sources: u32,
    pub n_panels: u32,
    pub n_pockets: u32,
    pub has_grade: u32,
    pub grade: f32,
    pub debug: u32,
    pub n_edges: u32,
    pub off_panels: u32,
    pub off_pockets: u32,
    pub edge_lo: [f32; 3],
    pub off_edges: u32,
    pub edge_hi: [f32; 3],
    pub _r: [u32; 5],
}

/// One walk source (scene words). WGSL accessors: pos @0, weight @12,
/// t_free @16, id @20, direct visibility @24, fixed-point norm @36 — 48 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuV3Src {
    pub pos: [f32; 3],
    pub weight: f32,
    pub t_free: f32,
    pub id: u32,
    pub vis: [f32; 3],
    pub norm: f32,
    pub _p: [u32; 2],
}

/// A below-grade air pocket. WGSL: bmin @0, bmax @16 — 32 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuV3Pocket {
    pub bmin: [f32; 3],
    pub _p0: u32,
    pub bmax: [f32; 3],
    pub _p1: u32,
}

/// Flatten a SurfaceTable (authored planes + rect-bounded overlay
/// faces) into the solve kernel's surfaces buffer.
pub fn flatten_surfaces(table: &omg_core::pt_mesh::SurfaceTable) -> Vec<GpuSurface> {
    table
        .surfaces
        .iter()
        .map(|s| {
            let (rmin, rmax) = s.rect.unwrap_or((
                omg_core::vec3::Vec3::new(0.0, 0.0, 0.0),
                omg_core::vec3::Vec3::new(0.0, 0.0, 0.0),
            ));
            GpuSurface {
                n: [s.n.x, s.n.y, s.n.z],
                d: s.d,
                refl: s.refl,
                has_rect: s.rect.is_some() as u32,
                trans: s.trans,
                _p0: 0,
                rmin: [rmin.x, rmin.y, rmin.z],
                _p1: 0,
                rmax: [rmax.x, rmax.y, rmax.z],
                _p2: 0,
            }
        })
        .collect()
}

/// Flatten a mesh's BVH + materials into the kernel's buffers.
pub fn flatten_mesh(
    mesh: &omg_core::mesh::Mesh,
) -> (Vec<GpuBvhNode>, Vec<GpuPrim>, Vec<GpuFace>) {
    let mut nodes = Vec::new();
    let mut prims = Vec::new();
    mesh.visit_bvh(
        &mut |bmin, bmax, a, b| {
            nodes.push(GpuBvhNode {
                bmin: [bmin.x, bmin.y, bmin.z],
                a,
                bmax: [bmax.x, bmax.y, bmax.z],
                b,
            });
        },
        &mut |a, e1, e2, m, surf| {
            prims.push(GpuPrim {
                a: [a.x, a.y, a.z],
                mat: m as u32,
                e1: [e1.x, e1.y, e1.z],
                surf: surf as u32,
                e2: [e2.x, e2.y, e2.z],
                _p1: 0,
            });
        },
    );
    let mats = mesh
        .materials
        .iter()
        .enumerate()
        .map(|(i, m)| GpuFace {
            absorption: m.absorption,
            scattering: m.scattering,
            transmission: m.transmission,
            thickness: mesh.material_thickness(i as u16).unwrap_or(0.0),
        })
        .collect();
    (nodes, prims, mats)
}

/// Decode the kernel's fixed-point output buffers into an Echogram.
pub fn decode_echogram(bins: &[u32], dirs: &[i32], out: &mut omg_core::tracer::Echogram) {
    out.clear();
    for i in 0..NBINS {
        for b in 0..NBANDS {
            out.bins[i][b] = bins[i * NBANDS + b] as f32 / ENERGY_SCALE;
        }
        for k in 0..3 {
            out.dirs[i][k] = dirs[i * 3 + k] as f32 / DIR_SCALE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes() {
        assert_eq!(core::mem::size_of::<GpuFace>(), 32);
        assert_eq!(core::mem::size_of::<GpuTraceJob>(), 256);
        assert_eq!(core::mem::offset_of!(GpuTraceJob, n_rays), 12);
        assert_eq!(core::mem::offset_of!(GpuTraceJob, source), 16);
        assert_eq!(core::mem::offset_of!(GpuTraceJob, seed), 28);
        assert_eq!(core::mem::offset_of!(GpuTraceJob, listener), 32);
        assert_eq!(core::mem::offset_of!(GpuTraceJob, energy), 48);
        assert_eq!(core::mem::offset_of!(GpuTraceJob, faces), 64);
        assert_eq!(NBINS, 300);
    }

    #[test]
    fn mesh_layout_sizes() {
        assert_eq!(core::mem::size_of::<GpuBvhNode>(), 32);
        assert_eq!(core::mem::offset_of!(GpuBvhNode, a), 12);
        assert_eq!(core::mem::offset_of!(GpuBvhNode, bmax), 16);
        assert_eq!(core::mem::size_of::<GpuPrim>(), 48);
        assert_eq!(core::mem::offset_of!(GpuPrim, e1), 16);
        assert_eq!(core::mem::offset_of!(GpuPrim, surf), 28);
        assert_eq!(core::mem::offset_of!(GpuPrim, e2), 32);
        assert_eq!(core::mem::size_of::<GpuPanel>(), 64);
        assert_eq!(core::mem::offset_of!(GpuPanel, absorption), 32);
        assert_eq!(core::mem::offset_of!(GpuPanel, transmission), 48);
        assert_eq!(core::mem::size_of::<GpuMeshJob>(), 64);
        assert_eq!(core::mem::offset_of!(GpuMeshJob, source), 16);
        assert_eq!(core::mem::offset_of!(GpuMeshJob, listener), 32);
        assert_eq!(core::mem::offset_of!(GpuMeshJob, energy), 48);
    }

    #[test]
    fn v3_layout_matches_the_fold_contract() {
        use omg_core::v3::fold::{HEADER_WORDS, PANEL_WORDS, POCKET_WORDS, SRC_WORDS, WalkJob};
        use omg_core::v3::{Budget, Source, World};
        use omg_core::vec3::Vec3;
        assert_eq!(core::mem::size_of::<GpuV3Job>(), HEADER_WORDS * 4);
        assert_eq!(core::mem::size_of::<GpuV3Job>(), 160);
        assert_eq!(core::mem::offset_of!(GpuV3Job, n_early), 12);
        assert_eq!(core::mem::offset_of!(GpuV3Job, n_late), 28);
        assert_eq!(core::mem::offset_of!(GpuV3Job, bounces), 44);
        assert_eq!(core::mem::offset_of!(GpuV3Job, listener), 48);
        assert_eq!(core::mem::offset_of!(GpuV3Job, seed), 60);
        assert_eq!(core::mem::offset_of!(GpuV3Job, k), 64);
        assert_eq!(core::mem::offset_of!(GpuV3Job, n_sources), 76);
        assert_eq!(core::mem::offset_of!(GpuV3Job, grade), 92);
        assert_eq!(core::mem::offset_of!(GpuV3Job, debug), 96);
        assert_eq!(core::mem::offset_of!(GpuV3Job, n_edges), 100);
        assert_eq!(core::mem::offset_of!(GpuV3Job, off_pockets), 108);
        assert_eq!(core::mem::offset_of!(GpuV3Job, edge_lo), 112);
        assert_eq!(core::mem::offset_of!(GpuV3Job, off_edges), 124);
        assert_eq!(core::mem::offset_of!(GpuV3Job, edge_hi), 128);
        assert_eq!(core::mem::size_of::<GpuV3Src>(), SRC_WORDS * 4);
        assert_eq!(core::mem::offset_of!(GpuV3Src, t_free), 16);
        assert_eq!(core::mem::offset_of!(GpuV3Src, vis), 24);
        assert_eq!(core::mem::offset_of!(GpuV3Src, norm), 36);
        assert_eq!(core::mem::size_of::<GpuPanel>(), PANEL_WORDS * 4);
        assert_eq!(core::mem::size_of::<GpuV3Pocket>(), POCKET_WORDS * 4);
        assert_eq!(core::mem::offset_of!(GpuV3Pocket, bmax), 16);
        // a real job decodes through the mirror structs
        let mesh = omg_core::mesh::MeshBuilder::new().build();
        let w = World::new(&mesh, &[]);
        let src = [Source { id: 3, pos: Vec3::new(0.0, 4.0, 3.0), weight: 1.0 }];
        let b = Budget { rays: 77, late_rays: 5, bounces: 2, ..Budget::default() };
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let job = WalkJob::build(&w, Vec3::new(0.0, 0.0, 0.0), &src, b, rot, 5, false, &[[0.25; 3]], None);
        let h: &GpuV3Job = bytemuck::from_bytes(bytemuck::cast_slice(job.header()));
        assert_eq!((h.n_early, h.n_late, h.bounces, h.n_sources, h.seed), (77, 5, 2, 1, 5));
        assert_eq!(h.off_panels as usize, SRC_WORDS);
        let s: &GpuV3Src = bytemuck::from_bytes(bytemuck::cast_slice(&job.scene()[..SRC_WORDS]));
        assert_eq!((s.id, s.vis, s.norm), (3, [0.25; 3], 25.0));
    }

    #[test]
    fn solve_layout_sizes() {
        assert_eq!(core::mem::size_of::<GpuSurface>(), 80);
        assert_eq!(core::mem::offset_of!(GpuSurface, refl), 16);
        assert_eq!(core::mem::offset_of!(GpuSurface, has_rect), 28);
        assert_eq!(core::mem::offset_of!(GpuSurface, trans), 32);
        assert_eq!(core::mem::offset_of!(GpuSurface, rmin), 48);
        assert_eq!(core::mem::offset_of!(GpuSurface, rmax), 64);
        assert_eq!(core::mem::size_of::<GpuSolveJob>(), 32);
        assert_eq!(core::mem::offset_of!(GpuSolveJob, listener), 16);
        assert_eq!(core::mem::size_of::<GpuSolveSrc>(), 16);
        assert_eq!(core::mem::size_of::<GpuExtra>(), 48);
        assert_eq!(core::mem::offset_of!(GpuExtra, bmax), 16);
        assert_eq!(core::mem::offset_of!(GpuExtra, trans), 32);
        assert_eq!(core::mem::size_of::<GpuSolveRec>(), 32);
        assert_eq!(core::mem::offset_of!(GpuSolveRec, gains), 16);
        assert_eq!(core::mem::offset_of!(GpuSolveRec, valid), 28);
    }
}
