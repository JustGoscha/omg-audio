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
}
