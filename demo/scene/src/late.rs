//! The late-field backend seam (GPU_PLAN.md phase 2): where `Sim` sends
//! its stochastic traces. Two implementations exist by design and both
//! stay forever — the inline CPU tracer (default: no registration, no
//! locks taken beyond one mutex, bit-identical to the pre-seam code)
//! and the wgpu kernel (`omg-gpu`, registered by the app at startup
//! when `OMG_GPU=1`). The CPU path is both the fallback for machines
//! without a usable adapter and the oracle GPU parity is tested
//! against.
//!
//! Divergence from the plan's submit/collect ring, on purpose: the
//! measured synchronous GPU dispatch (incl. readback) is ~1 ms — seven
//! times CHEAPER than the CPU trace it replaces — so a pipelined ring
//! would add a tick of staleness to save a millisecond. The trait keeps
//! the door open: batching + async collection become worthwhile in
//! phase 5 when one submission carries every source's trace.

use omg_core::material::Material;
use omg_core::rng::Rng;
use omg_core::scene::{AcousticGeometry, Shoebox};
use omg_core::tracer::{trace, Echogram};
use omg_core::vec3::Vec3;
use omg_core::NBANDS;
use std::sync::Mutex;

/// A late-field trace executor. `id` names the (source, room) Sim
/// instance — stable across ticks, usable as a slot key by pipelined
/// implementations. Returns true when `out` holds a fresh echogram.
pub trait LateBackend: Send {
    #[allow(clippy::too_many_arguments)]
    fn trace(
        &mut self,
        id: u32,
        room: &Shoebox,
        src: Vec3,
        lis: Vec3,
        n_rays: u32,
        energy: [f32; NBANDS],
        rng: &mut Rng,
        out: &mut Echogram,
    ) -> bool;

    /// Asynchronous backends (the web's JS-driven WebGPU dispatch)
    /// deliver a previously submitted trace here, one tick late — the
    /// staleness the two-clock design already absorbs. Synchronous
    /// backends keep the default.
    fn poll_into(&mut self, _id: u32, _out: &mut Echogram) -> bool {
        false
    }
}

static BACKEND: Mutex<Option<Box<dyn LateBackend>>> = Mutex::new(None);

/// Install a backend (the app's `OMG_GPU=1` path). Passing the result
/// of a failed GPU init is the caller's problem — register only a
/// working backend; unregistered means the inline CPU tracer.
pub fn set_late_backend(b: Box<dyn LateBackend>) {
    *BACKEND.lock().unwrap() = Some(b);
}

/// Drop the registered backend: traces return to the inline CPU tracer
/// on the next call (a live A/B toggle for the tuning panel).
pub fn clear_late_backend() {
    *BACKEND.lock().unwrap() = None;
}

/// Deliver an async backend's finished trace for `id`, if one arrived.
pub(crate) fn poll_into(id: u32, out: &mut Echogram) -> bool {
    let mut guard = BACKEND.lock().unwrap();
    match guard.as_mut() {
        Some(b) => b.poll_into(id, out),
        None => false,
    }
}

/// C6d: the WORLD late-field executor — the same seam contract as
/// `LateBackend`, over the world mesh instead of a shoebox. The mesh
/// itself lives with the backend (uploaded once at registration); per
/// call only the pose and the panel overlays (door leaves, panes)
/// travel. `rays` is the CPU budget — a GPU implementation is free to
/// multiply it (variance, never bias).
pub trait WorldLateBackend: Send {
    fn trace_world(
        &mut self,
        id: u32,
        src: Vec3,
        lis: Vec3,
        rays: u32,
        panels: &[(Vec3, Vec3, Material)],
        out: &mut Echogram,
    ) -> bool;
}

static WORLD_BACKEND: Mutex<Option<Box<dyn WorldLateBackend>>> = Mutex::new(None);

pub fn set_world_late_backend(b: Box<dyn WorldLateBackend>) {
    *WORLD_BACKEND.lock().unwrap() = Some(b);
}

pub fn clear_world_late_backend() {
    *WORLD_BACKEND.lock().unwrap() = None;
}

/// Route one WORLD trace: the registered backend if any, else the
/// inline CPU tracer over `world` (which already carries the panel
/// overlays via `WithPanels`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_trace_world(
    id: u32,
    world: &impl AcousticGeometry,
    src: Vec3,
    lis: Vec3,
    rays: u32,
    panels: &[(Vec3, Vec3, Material)],
    rng: &mut Rng,
    out: &mut Echogram,
) -> bool {
    let mut guard = WORLD_BACKEND.lock().unwrap();
    match guard.as_mut() {
        Some(b) => b.trace_world(id, src, lis, rays, panels, out),
        None => {
            trace(world, src, lis, rays, [1.0; NBANDS], rng, out);
            true
        }
    }
}

/// Route one trace: the registered backend if any, else the inline CPU
/// tracer (exactly the pre-seam behavior, same per-Sim RNG stream).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_trace(
    id: u32,
    room: &Shoebox,
    src: Vec3,
    lis: Vec3,
    n_rays: u32,
    energy: [f32; NBANDS],
    rng: &mut Rng,
    out: &mut Echogram,
) -> bool {
    let mut guard = BACKEND.lock().unwrap();
    match guard.as_mut() {
        Some(b) => b.trace(id, room, src, lis, n_rays, energy, rng, out),
        None => {
            trace(room, src, lis, n_rays, energy, rng, out);
            true
        }
    }
}
