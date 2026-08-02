//! GPU buffer layouts for the trace kernel — #[repr(C)] mirrors of what
//! `shaders/trace_box.wgsl` declares, plus the fixed-point scales shared
//! with the shader. Every struct has a size test (G5: layout drift is
//! the classic silent killer); the WGSL carries the same offsets in a
//! comment. Bump LAYOUT_VERSION on ANY change here or in the WGSL
//! structs — the web driver checks it and falls back to CPU on mismatch
//! instead of decoding garbage.

use omg_core::tracer::{BIN_DT, MAX_TIME};
use omg_core::NBANDS;

pub const LAYOUT_VERSION: u32 = 1;

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

/// One face's acoustics. WGSL: absorption vec3<f32> @0, scattering @12.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFace {
    pub absorption: [f32; 3],
    pub scattering: f32,
}

/// The whole trace job as one uniform block.
/// WGSL offsets: size @0, n_rays @12, source @16, seed @28, listener
/// @32, _pad0 @44, energy @48, _pad1 @60, faces @64 (6 × 16 B) = 160 B.
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
pub const MESH_LAYOUT_VERSION: u32 = 2;
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
/// table. WGSL: pmin @0, scattering @12, pmax @16, absorption @32 — 48 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPanel {
    pub pmin: [f32; 3],
    pub scattering: f32,
    pub pmax: [f32; 3],
    pub _p0: u32,
    pub absorption: [f32; 3],
    pub _p1: u32,
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
        .map(|m| GpuFace { absorption: m.absorption, scattering: m.scattering })
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
        assert_eq!(core::mem::size_of::<GpuFace>(), 16);
        assert_eq!(core::mem::size_of::<GpuTraceJob>(), 160);
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
        assert_eq!(core::mem::size_of::<GpuPanel>(), 48);
        assert_eq!(core::mem::offset_of!(GpuPanel, absorption), 32);
        assert_eq!(core::mem::size_of::<GpuMeshJob>(), 64);
        assert_eq!(core::mem::offset_of!(GpuMeshJob, source), 16);
        assert_eq!(core::mem::offset_of!(GpuMeshJob, listener), 32);
        assert_eq!(core::mem::offset_of!(GpuMeshJob, energy), 48);
    }
}
