# Engine v4 — the traced response, rendered as traced

v3 traces rays from the listener and finds every path a sound takes — and
then throws most of that away: arrivals inside 80 ms become up to 24
taps, everything after is folded into 10 ms bins and boiled down to one
decay time and one level per band, which a generic feedback-delay reverb
turns back into a smooth tail. Whatever the trace found in the timing of
the late part — an echo off a cliff 20 m away, the flutter between two
bare walls, a cathedral's 15 s ring — is lost at that step.

v4 keeps the trace. There is no early/late split in time: every arrival
is kept, at any delay, and only split by **kind**.

## What is traced

The v3 walk is unchanged (rays from the listener, next-event connections
to every source, sphere hits, edge bends, transmission, the sky). What
changes is what it keeps (`omg_core::v3`, `Tracer::v4`):

- **Coherent arrivals** — paths that only mirror (no scattering, no
  bend): exact image-source paths, found by the walk's mirror rays and
  solved exactly every tick (`solve_chain`, every leg validated against
  the fixed geometry, movable boxes like a closed door priced with their
  transmission). The direct path and edge bends are coherent too. Each
  has an exact delay and direction.
- **The dense response** — everything the walk estimates: every path
  with a diffuse bounce, every bent path, mirror paths too quiet for the
  solver. Physically diffuse. Kept per source as energy per band and the
  energy vector (Σ E·direction), per 2 ms, from the free-field arrival to
  `RESPONSE_MAX_S` (4.8 s).

Every path is counted exactly once (`solver_owns`).

### The dense estimate's resolution follows its samples

Stochastic estimates are noisy, and a 2 ms bin late in a tail sees few
samples. Instead of tuned smoothing windows:

- a sphere hit only knows its path to within the sphere: its energy is
  spread over its own time uncertainty, ±r/c;
- each bin keeps Σw² next to Σw, so its effective sample count
  (Σw)²/Σw² is known (a single heavy sample — a ray bouncing right by the
  source — counts as one); each bin is then averaged over the narrowest
  window holding 32 effective samples (~18 % noise). The well-sampled
  early part keeps 2 ms; a sparse late tail smooths itself. Coherent
  arrivals are exact and never smoothed;
- standing still (listener and source unmoved, the world unchanged),
  ticks average as a running mean and converge instead of flickering;
  any move, a door closing, restores the fast average.

## What plays as what

The direct path and edge bends are taps. Mirror chains are taps, ranked
by how much they **stand out from the dense energy arriving around them**
(±4 ms) — one thing clearly louder than the crowd — up to the renderer's
budget (48 per source); the rest join the dense response at their delay.
This is only a rendering choice: the energy and the delay are the same
either way. A tap keeps its own Doppler and exact direction filter; the
dense response has neither, which is right for a diffuse field.

## Rendering the dense response (omg-dsp `dense`)

- **Impulse response**: many incoherent arrivals sum like noise, so each
  bin becomes noise with exactly the traced energy per band (bands from
  one white noise split at the tracer's crossovers, so a flat response
  is white). In first-order ambisonics: the one-sided share of the energy
  vector as a coherent plane wave from its direction, the rest as
  decorrelated diffuse noise. One shared noise bank, a fixed realization
  per source: a new kernel only reshapes the envelope, so kernel swaps
  don't click.
- **The tail beyond the window**: past the mixing time a diffuse field
  only decays, exponentially. A response still ringing at the window's
  end (above −60 dB) is continued along its own decay, fitted per band
  over its last second, until −60 dB (cap 10 s).
- **Convolution**: two-stage uniformly partitioned (128-sample head for
  the first 85 ms, 4096-sample tail after it), fed with the source as of
  its first arrival, adding into the renderer's SH bus — decoded with
  everything else. Equal to direct convolution (test).

## The engine (omg-audio)

`Config::v4` (default on). The simulation hands the mixer taps per source
and, per source, a kernel (`Frame::kernels`, shared Arcs: a frame
carries every current kernel, the mixer swaps one in when it changes, so
a skipped frame loses nothing). Kernels are built on a thread of their
own when a source's dense envelope changed by ~0.6 dB (or after a while
by less). The mixer keeps a source rendering while its convolution rings
(it remembers when the source last sounded, past the play log forgetting
a one-shot).

The GPU walk bins the same dense response (walk layout 5); the CPU keeps
solving the mirror chains for both.

## Measured

- A clap in a canyon (rock cliffs 40 m apart) comes back at the round
  trips, through the whole engine (`tests/fx_v4.rs`); the offline
  comparison (`omg-core/examples/v4_compare.rs`) keeps the echo train
  past 1.5 s, where v3 had 9 echoes from its echo special case.
- A bare 4 m room: two interleaved flutter trains, 23 ms period, 6–279 ms,
  as exact taps.
- Decay closer to Sabine than v3 (living room 0.34 s vs ~0.31 s; v3
  0.44 s); a stone hall rings past v3's 5 s cap.
- A closed door costs 7.9 / 10.8 / 11.4 dB (low / mid / high).
- CPU/GPU parity: dense totals within 0.1 dB, the same decay, the same
  coherent taps.
- Cost: town bench (64 sources) CPU tick 18.3 ms (v3 16.5), mixer 3.1×
  real time (v3 4.5×); Godot demo GPU tick 0.7 ms, render 4.9× real time,
  a 66-source stress 2.6×.

## What is left

- The web demo (wasm, no threads in the worklet): kernels built in the
  worker and transferred, a budget of convolved sources.
- Kernel memory for very long tails (a 10 s kernel is ~15 MB).
- The v3 echo special case (salience gates, merge rules) stays for v3
  mode only.
