//! The v3 walk seam (ENGINE_V3.md V3.3): where a v3 tick's ray walk runs.
//!
//! `Tracer::trace` walks on the CPU (the oracle and the fallback). A
//! `V3Walker` runs the same walk elsewhere — the native wgpu kernel
//! (`omg-gpu`), or the web's JS-driven WebGPU proxy — synchronously
//! (`Walk::Fresh` with this job's bins) or asynchronously (queue this
//! job, return an earlier one's result or `Walk::Pending`). A walker that
//! refuses a job (caps, lost device) hands that tick back to the CPU.
//!
//! This trait lives in omg-core so a walker can be written against the
//! engine alone; `omg_scene::v3walk` re-exports it for the demo.

use super::fold::{Walk, WalkJob};
use super::{Budget, Source, SourceField, Tracer, World};
use crate::vec3::Vec3;

/// Walks a flat v3 job (see `fold.rs` for the format).
pub trait V3Walker: Send {
    fn walk(&mut self, job: &WalkJob) -> Walk;

    /// Wall-clock of the last finished walk, ms (stats only).
    fn last_ms(&self) -> f32 {
        0.0
    }

    /// The longest single dispatch of the last walk, ms (a sliced walk is
    /// several; stats only).
    fn last_max_ms(&self) -> f32 {
        self.last_ms()
    }

    /// Whether this walker walks the late family too. A live GPU walk
    /// doesn't: a late ray's hundreds of bounces run one after another,
    /// the one thing a GPU is slowest at — sliced across walks they would
    /// carry where the listener was seconds ago — so the CPU walks them,
    /// whole, from where the listener is (`Tracer::trace_walked`).
    fn walks_late(&self) -> bool {
        true
    }
}

/// One v3 tick: through `walker` when there is one, else (or when it
/// refuses the job) the CPU walk at `cpu_budget`.
pub fn trace_with(
    tracer: &mut Tracer,
    world: &World,
    listener: Vec3,
    sources: &[Source],
    budget: Budget,
    cpu_budget: Budget,
    walker: Option<&mut dyn V3Walker>,
) -> Vec<SourceField> {
    if let Some(w) = walker {
        // the late family at the CPU's own count when the walker leaves it
        let cpu_late = (!w.walks_late()).then_some(cpu_budget.late_rays);
        if let Some(fields) = tracer.trace_walked(world, listener, sources, budget, cpu_late, &mut |job| w.walk(job)) {
            return fields;
        }
    }
    tracer.trace(world, listener, sources, cpu_budget)
}
