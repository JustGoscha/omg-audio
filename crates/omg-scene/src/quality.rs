//! The CPU quality ladder: one setting that scales every budgeted
//! sim-side workload so a weak device sheds perceptually unimportant
//! work before anything audibly breaks (GPU_PLAN.md, Track B).
//!
//! Every lever here trades VARIANCE or refresh latency, never bias:
//! fewer trace rays flutter the RT60 estimate slightly more (the EMA in
//! `sim.rs` absorbs it), a staler gate reacts slower to room changes,
//! fewer dome rays add bin noise under the dome's own EMA. The bounce
//! cap is deliberately NOT a lever — truncating bounces biases the RT60
//! fit in low-absorption rooms (see the no-Russian-roulette comment in
//! `tracer.rs`).
//!
//! A process-wide atomic rather than a field threaded through `Sim`:
//! sims are created per (source, room) all over `world.rs`, and the
//! tier is a device property, not a per-source one.

use core::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    High,
    Med,
    Low,
}

static TIER: AtomicU32 = AtomicU32::new(0);

/// Per-lever manual overrides (a tuning UI's independent sliders):
/// 0 = "follow the tier", anything else wins over the tier value.
/// Ids: 0 trace rays, 1 gate age, 2 dome rays, 3 dome events, 4 ISM order,
/// 5 v3 early rays, 6 v3 reflections per ray, 7 v3 late rays.
static OVERRIDES: [AtomicU32; 8] = [
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
];

/// 0 = High, 1 = Med, 2 = Low (clamped). Callable from any thread; the
/// sim reads it at the next use of each budget.
pub fn set_tier(t: u32) {
    TIER.store(t.min(2), Ordering::Relaxed);
}

/// Whether a GPU late-field backend is active. A 32768-ray dispatch
/// costs ~1.5 ms where the CPU needs 7.3 ms for 4096 (omg-gpu speed
/// probe), so the tier ray budgets multiply 8× — pure variance
/// reduction, the estimate just gets steadier. Bounce depth stays at
/// the shared 64 cap: raising it changes the measured tail (a real
/// improvement, not noise) and therefore needs re-goldening first —
/// that's the phase 5 item, not a flag.
static GPU: AtomicU32 = AtomicU32::new(0);

pub fn set_gpu_backend(on: bool) {
    GPU.store(on as u32, Ordering::Relaxed);
}

const GPU_RAY_MULT: u32 = 8;
const GPU_RAY_CAP: u32 = 32768;

/// Whether an engine-v3 walker (the GPU listener walk, `v3walk.rs`) is
/// registered. The v3 budgets rise only when it is AND the compute
/// switch is on — a registered late backend alone must never push the
/// CPU walk to a GPU-sized budget.
static V3_WALKER: AtomicU32 = AtomicU32::new(0);

pub fn set_v3_walker_on(on: bool) {
    V3_WALKER.store(on as u32, Ordering::Relaxed);
}

/// Engine v3 walks on the GPU this tick.
pub fn v3_gpu() -> bool {
    GPU.load(Ordering::Relaxed) != 0 && V3_WALKER.load(Ordering::Relaxed) != 0
}

/// v3 budgets on the GPU: 8× rays of both families (variance only —
/// the estimate is the same, it just settles faster and flutters less).
const V3_GPU_MULT: u32 = 8;

/// Early-reflections backend (GPU_PLAN.md Track C): 0 = ism (the
/// classic image-source engine), 1 = traced (PT-early: discovery +
/// exact chain solve). Both are kept forever; `traced` currently also
/// covers what C5's `hybrid` will refine (low orders are already
/// seeded deterministically inside discovery).
static EARLY: AtomicU32 = AtomicU32::new(0);

/// Early-field engine: 0 = ism, 1 = traced (v2), 2 = hybrid (reserved,
/// runs traced), 3 = v3 — the one listener tracer (ENGINE_V3.md), which
/// replaces early AND late, routing, bends and blending for every source.
pub fn set_early(mode: u32) {
    EARLY.store(mode.min(3), Ordering::Relaxed);
}

/// The v2 world-mesh early pipeline (traced or hybrid).
pub fn early_traced() -> bool {
    matches!(EARLY.load(Ordering::Relaxed), 1 | 2)
}

/// Engine v3: one listener trace per tick serves every source.
pub fn early_v3() -> bool {
    EARLY.load(Ordering::Relaxed) == 3
}

pub fn early() -> u32 {
    EARLY.load(Ordering::Relaxed)
}

/// A/B module switches (all default ON) — not quality levers but whole
/// engine features, toggleable live for listening comparisons.
/// Ids: 0 = edge diffraction (knife-edge bend taps around jambs and
/// building corners + the outdoor occlusion floors), 1 = furniture
/// acoustics (occluder boxes shadowing/transmitting in the early solve).
static MODULES: [AtomicU32; 2] = [AtomicU32::new(1), AtomicU32::new(1)];

pub fn set_module(id: u32, on: bool) {
    if let Some(m) = MODULES.get(id as usize) {
        m.store(on as u32, Ordering::Relaxed);
    }
}

pub fn diffraction_on() -> bool {
    MODULES[0].load(Ordering::Relaxed) != 0
}

pub fn furniture_on() -> bool {
    MODULES[1].load(Ordering::Relaxed) != 0
}

/// Manually pin one lever (see `OVERRIDES` for ids); `value` 0 hands the
/// lever back to the tier. Out-of-range ids are ignored.
pub fn set_override(id: u32, value: u32) {
    if let Some(o) = OVERRIDES.get(id as usize) {
        o.store(value, Ordering::Relaxed);
    }
}

fn over(id: usize, tier_val: u32) -> u32 {
    match OVERRIDES[id].load(Ordering::Relaxed) {
        0 => tier_val,
        v => v,
    }
}

pub fn tier() -> Tier {
    match TIER.load(Ordering::Relaxed) {
        0 => Tier::High,
        1 => Tier::Med,
        _ => Tier::Low,
    }
}

impl Tier {
    fn trace_rays_tier(self) -> u32 {
        match self {
            Tier::High => 4096,
            Tier::Med => 2048,
            Tier::Low => 1024,
        }
    }

    fn gate_max_age_tier(self) -> u32 {
        match self {
            Tier::High => 8,
            Tier::Med => 12,
            Tier::Low => 16,
        }
    }

    fn dome_rays_tier(self) -> usize {
        match self {
            Tier::High => 384,
            Tier::Med => 256,
            Tier::Low => 160,
        }
    }

    fn dome_events_tier(self) -> usize {
        match self {
            Tier::High => 6,
            Tier::Med => 5,
            Tier::Low => 4,
        }
    }

    fn ism_order_tier(self) -> u32 {
        match self {
            Tier::High => 3,
            Tier::Med => 3,
            Tier::Low => 2,
        }
    }

    /// Stochastic tracer rays per trace (High is the historical 4096;
    /// ×8 on a GPU backend). A manual override is absolute either way.
    pub fn trace_rays(self) -> u32 {
        let tier = if GPU.load(Ordering::Relaxed) != 0 {
            (self.trace_rays_tier() * GPU_RAY_MULT).min(GPU_RAY_CAP)
        } else {
            self.trace_rays_tier()
        };
        over(0, tier).clamp(64, GPU_RAY_CAP)
    }

    /// TraceGate refresh age (ticks at 20 Hz) for an idle scene.
    pub fn gate_max_age(self) -> u32 {
        over(1, self.gate_max_age_tier()).clamp(1, 64)
    }

    /// World-mesh late-trace rays (C6d). BVH queries cost ~10× the
    /// analytic box, and the world.rs budget is ONE trace per tick, so
    /// the ray count is small and the EMA carries the variance. CPU
    /// only for now — the GPU multiplier deliberately does not apply.
    /// Engine v3: rays from the listener per tick, and bounces per ray
    /// (ENGINE_V3.md). The EMA across ticks carries the variance.
    /// ×8 while the GPU walker runs (`v3_gpu`).
    pub fn v3_rays(self) -> u32 {
        self.v3_rays_for(v3_gpu())
    }

    /// The early-ray budget for a CPU (`gpu = false`) or GPU walk.
    pub fn v3_rays_for(self, gpu: bool) -> u32 {
        let t = match self {
            Tier::High => 1024,
            Tier::Med => 768,
            Tier::Low => 512,
        };
        let (t, cap) = if gpu { (t * V3_GPU_MULT, 65536) } else { (t, 16384) };
        over(5, t).clamp(32, cap)
    }

    /// Engine v3 late family: long rays per tick carrying the reverb
    /// tail (averaged over ~10 ticks). ×8 on the GPU walker.
    pub fn v3_late_rays(self) -> u32 {
        self.v3_late_rays_for(v3_gpu())
    }

    pub fn v3_late_rays_for(self, gpu: bool) -> u32 {
        let t = match self {
            Tier::High => 128,
            Tier::Med => 96,
            Tier::Low => 48,
        };
        let (t, cap) = if gpu { (t * V3_GPU_MULT, 8192) } else { (t, 4096) };
        over(7, t).clamp(8, cap)
    }

    /// Reflections per EARLY ray (the late rays continue beyond it).
    pub fn v3_bounces(self) -> u32 {
        let t = match self {
            Tier::High => 6,
            Tier::Med => 4,
            Tier::Low => 3,
        };
        over(6, t).clamp(1, 32)
    }

    pub fn world_rays(self) -> u32 {
        match self {
            Tier::High => 512,
            Tier::Med => 384,
            Tier::Low => 256,
        }
    }

    /// Ambient-dome rays per tick (High is the historical 384).
    pub fn dome_rays(self) -> usize {
        over(2, self.dome_rays_tier() as u32).clamp(16, 384) as usize
    }

    /// Ambient-dome bounce/pane events per ray.
    pub fn dome_events(self) -> usize {
        over(3, self.dome_events_tier() as u32).clamp(1, 6) as usize
    }

    /// Image-source order for single-source rooms. Never below 2:
    /// order-2 carries the audible early-reflection pattern; order-3
    /// images are the quietest taps and the late field covers the gap.
    pub fn ism_order(self) -> u32 {
        over(4, self.ism_order_tier()).clamp(1, 3)
    }
}
