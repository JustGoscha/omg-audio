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

/// One v3 tick: through `walker` when there is one, else the CPU walk at
/// `cpu_budget` — also while the walker's walks are stale or refused
/// (`Tracer::trace_walked`).
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
        if let Some(fields) = tracer.trace_walked(world, listener, sources, budget, cpu_budget, cpu_late, &mut |job| w.walk(job)) {
            return fields;
        }
    }
    tracer.trace(world, listener, sources, cpu_budget)
}

#[cfg(test)]
mod tests {
    use super::super::fold::{output_words, Walked};
    use super::*;
    use crate::material::Material;
    use crate::mesh::MeshBuilder;

    /// A walker that answers `late` ticks after a job (a large level's
    /// GPU walk, sliced to its share of the GPU): one walk in flight, the
    /// job of the tick it lands on launched next. Every walk lands energy
    /// in every bin.
    struct Late {
        late: u32,
        flight: Option<(u32, Vec<u32>, u32, Vec3)>,
    }

    impl V3Walker for Late {
        fn walk(&mut self, job: &WalkJob) -> Walk {
            match self.flight.take() {
                Some((epoch, ids, age, at)) if age + 1 >= self.late => {
                    self.flight = Some((job.epoch, job.ids.clone(), 0, job.listener));
                    Walk::Fresh(Walked { data: vec![1e-3; output_words(ids.len())], ids, debug: Vec::new(), epoch, listener: at })
                }
                Some((epoch, ids, age, at)) => {
                    self.flight = Some((epoch, ids, age + 1, at));
                    Walk::Pending
                }
                None => {
                    self.flight = Some((job.epoch, job.ids.clone(), 0, job.listener));
                    Walk::Pending
                }
            }
        }
    }

    /// Flying through a hall's pillars on a walker that answers four
    /// ticks late: each crossing drops every average and the walk in
    /// flight (launched on the other side). Walked by the walker alone
    /// the hall went silent but for the direct sound until a walk launched
    /// on this side landed — and with a pillar every few ticks none did
    /// (the Godot level's nave at 8 m/s: no reverb for 1.6 s, then all of
    /// it at once). The CPU walks until a walk launched since the crossing
    /// lands: every tick of the flight carries the hall's reflections and
    /// reverb, and once standing the walker's walks take over.
    #[test]
    fn after_a_crossing_the_cpu_walks_until_the_walker_catches_up() {
        let mut mb = MeshBuilder::new();
        let stone = mb.material(Material::BRICK);
        let t = 0.3;
        let (lo, hi) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(12.0, 10.0, 6.0));
        mb.abox(Vec3::new(lo.x - t, lo.y - t, lo.z - t), Vec3::new(hi.x + t, hi.y + t, lo.z), stone);
        mb.abox(Vec3::new(lo.x - t, lo.y - t, hi.z), Vec3::new(hi.x + t, hi.y + t, hi.z + t), stone);
        mb.abox(Vec3::new(lo.x - t, lo.y - t, lo.z), Vec3::new(lo.x, hi.y + t, hi.z), stone);
        mb.abox(Vec3::new(hi.x, lo.y - t, lo.z), Vec3::new(hi.x + t, hi.y + t, hi.z), stone);
        mb.abox(Vec3::new(lo.x, lo.y - t, lo.z), Vec3::new(hi.x, lo.y, hi.z), stone);
        mb.abox(Vec3::new(lo.x, hi.y, lo.z), Vec3::new(hi.x, hi.y + t, hi.z), stone);
        // two pillars on the flight's line (y = 4), the source off it
        for x in [4.85, 6.85] {
            mb.abox(Vec3::new(x, 3.8, 0.0), Vec3::new(x + 0.3, 4.2, 6.0), stone);
        }
        let mesh = mb.build();
        let world = World::new(&mesh, &[]);
        let src = [Source { id: 3, pos: Vec3::new(10.0, 7.0, 1.6), weight: 1.0 }];
        let budget = Budget::default();
        let mut tracer = Tracer::new(5);
        let mut walker = Late { late: 4, flight: None };
        let at = |x: f32| Vec3::new(x, 4.0, 1.6);
        let early = |f: &SourceField| f.taps.iter().filter(|t| t.key >= 16 && t.key < 20_000).count();
        // standing until the walker's walks land
        for _ in 0..12 {
            let f = trace_with(&mut tracer, &world, at(2.0), &src, budget, budget, Some(&mut walker));
            assert!(f[0].reverb.level[1] >= 0.0);
        }
        // 0.4 m a tick (8 m/s at 20 Hz) through both pillars
        let mut empty = Vec::new();
        for k in 1..=18 {
            let x = 2.0 + 0.4 * k as f32;
            let f = trace_with(&mut tracer, &world, at(x), &src, budget, budget, Some(&mut walker));
            if early(&f[0]) == 0 || !(f[0].reverb.level[1] > 0.0) {
                empty.push(x);
            }
        }
        assert!(empty.is_empty(), "ticks with no reflections or reverb, at x = {empty:?}");
        // standing: once a walk launched here lands, the walker's walks
        // take over (the CPU walks no rays: its walk's cost is nil) —
        // within the walker's lateness and a tick
        let mut cpu_ticks = Vec::new();
        for k in 0..10 {
            trace_with(&mut tracer, &world, at(9.2), &src, budget, budget, Some(&mut walker));
            if tracer.costs.early + tracer.costs.late > 0.0 {
                cpu_ticks.push(k);
            }
        }
        assert!(cpu_ticks.iter().all(|&k| k <= walker.late as usize), "the CPU still walked standing, ticks {cpu_ticks:?}");
    }
}
