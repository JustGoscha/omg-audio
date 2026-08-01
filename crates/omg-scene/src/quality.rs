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
/// Ids: 0 trace rays, 1 gate age, 2 dome rays, 3 dome events, 4 ISM order.
static OVERRIDES: [AtomicU32; 5] = [
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

/// Early-reflections backend (GPU_PLAN.md Track C): 0 = ism (the
/// classic image-source engine), 1 = traced (PT-early: discovery +
/// exact chain solve). Both are kept forever; `traced` currently also
/// covers what C5's `hybrid` will refine (low orders are already
/// seeded deterministically inside discovery).
static EARLY: AtomicU32 = AtomicU32::new(0);

pub fn set_early(mode: u32) {
    EARLY.store(mode.min(2), Ordering::Relaxed);
}

pub fn early_traced() -> bool {
    EARLY.load(Ordering::Relaxed) != 0
}

pub fn early() -> u32 {
    EARLY.load(Ordering::Relaxed)
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
