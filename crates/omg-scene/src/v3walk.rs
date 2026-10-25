//! Engine v3's walk seam (ENGINE_V3.md V3.3): where `WorldSim` sends the
//! listener walk. Same pattern as `late.rs`: unregistered = the inline
//! CPU walk (`Tracer::trace`, the oracle and the fallback); registered
//! = an external walker — the native wgpu kernel (`omg-gpu`, OMG_GPU=1)
//! or the web's JS-driven WebGPU proxy, whose results land a tick late.
//! The walker is only used while the compute switch is on
//! (`quality::set_gpu_backend`), and a walker that refuses a job (caps,
//! lost device) hands that tick back to the CPU at the CPU budget.

use omg_core::v3::fold::{Walk, WalkJob};
use omg_core::v3::{Budget, Source, SourceField, Tracer, World};
use omg_core::vec3::Vec3;
use std::sync::Mutex;

/// Walks a flat v3 job: synchronously (native: `Walk::Fresh` with this
/// job's bins) or asynchronously (web: queue this job, return an
/// earlier one's result or `Walk::Pending`).
pub trait V3Walker: Send {
    fn walk(&mut self, job: &WalkJob) -> Walk;

    /// Wall-clock of the last finished walk, ms (stats only).
    fn last_ms(&self) -> f32 {
        0.0
    }
}

static WALKER: Mutex<Option<Box<dyn V3Walker>>> = Mutex::new(None);

/// Install a working walker (register only after its pipeline built).
/// Raises the v3 ray budgets to their GPU tier (quality.rs).
pub fn set_v3_walker(w: Box<dyn V3Walker>) {
    *WALKER.lock().unwrap() = Some(w);
    crate::quality::set_v3_walker_on(true);
}

/// Back to the inline CPU walk on the next tick.
pub fn clear_v3_walker() {
    *WALKER.lock().unwrap() = None;
    crate::quality::set_v3_walker_on(false);
}

/// Last walk's wall-clock (0 without a walker).
pub fn last_ms() -> f32 {
    WALKER.lock().unwrap().as_ref().map_or(0.0, |w| w.last_ms())
}

/// One v3 tick: the registered walker when the compute switch is on,
/// else (or when it refuses) the CPU walk.
pub fn trace(
    tracer: &mut Tracer,
    world: &World,
    listener: Vec3,
    sources: &[Source],
    budget: Budget,
) -> Vec<SourceField> {
    if crate::quality::v3_gpu() {
        let mut guard = WALKER.lock().unwrap();
        if let Some(w) = guard.as_mut() {
            if let Some(fields) = tracer.trace_walked(world, listener, sources, budget, &mut |job| w.walk(job)) {
                return fields;
            }
        }
        drop(guard);
        // refused: this tick walks on the CPU, at the CPU budget
        let tier = crate::quality::tier();
        let cpu = Budget { rays: tier.v3_rays_for(false), late_rays: tier.v3_late_rays_for(false), ..budget };
        return tracer.trace(world, listener, sources, cpu);
    }
    tracer.trace(world, listener, sources, budget)
}
