# Engine v5.5 — heard as it is, when it is, from where it was

v4 renders the traced response as traced; v5 reuses the reverb's late
paths across ticks. v5.5 (the default, `Config { v4: true, v5: true }`,
"v5.5" in the Godot settings) keeps both and fixes what was heard wrong
in them — measured with impulses through the whole engine
(`demo/web/wasm/examples/assess.rs`; `docs/runs/2026-10-29/`).

## Sound lives on a timeline of where it was emitted

Every source keeps a **track** — where it was, on the host's clock
(`Simulation::set_time`; the threaded engine stamps every position, the
listener pose carries its own time). Every connection reaches the source
where it was when it emitted what arrives now along that path:

    c · (now − t) = path so far + |x(t) − p|

solved by fixed-point steps along the track (`Track::emit`, contracting
by v/c; one solution while the source is slower than sound). The direct
path, the next-event connections at every bounce, sphere hits, edge
bends, solved mirror chains, exact frames — and the GPU walk, through
the direct sound's emission point. A car 1 km off at 100 km/h is heard
from ~81 m back; an echo 2 km long from ~160 m back; Doppler per path
follows from how those points move. Static sources skip the solve.

- **Jumps** (faster than sound: a teleport, a reused slot) leave a
  **ghost** at the old place: its own traced source and mixer channel,
  with the source's renderer as it was, its play log ending at the jump
  (`PlayLog::ghost`), and what the tracer learnt there
  (`Tracer::fork`); the source plays only what it emits from then on
  (`PlayLog::emitted_from`). A click in flight arrives from where it
  left, and rings.
- **Switched off**, a source is traced until what it emitted has arrived
  and rung out (its travel time + the longest kernel): nothing is cut
  because it stopped.
- A one-shot read within a sample before its start ramps into its first
  sample (a fractional delay used to lose a click's onset).

## Changes heard at once, without holes

- Tap parameters glide on 8 ms; a path appearing or vanishing fades on
  20 ms; only a same-key delay jump keeps a 50 ms crossfade (two copies
  of one sound at different delays: faster, the phase sweep is a blip).
- Exact frames also fire when an object moves (a door) and carry the
  kernels the thread finished since the last frame.
- No more kernel cuts. A stale kernel (another space) is withdrawn and
  the source's parametric reverb takes over — fed all along, crossfaded
  against the convolution (`Renderer::set_convolving`) — while the
  convolver keeps its input (`DenseConv::set_warm` / `prepare`), so the
  rebuilt kernel crossfades back in over the true recent input. The same
  for sources moving in and out of the convolution budget.
- A channel that isn't rendering (silent, masked, virtualized) settles:
  its glides and fades jump to their targets, so a door shut between two
  sounds is shut for the next one.
- The audible window times a pathless channel's scattered field from its
  free-field arrival (a ghost, a source behind walls rings on).

## Steady when nothing changes, fresh when something does

Changes are known exactly — the listener or a source moved, the world
changed — so the dense estimate needs no statistical guesses about them:

- Nothing moved, the world as it was: a plain running mean (the same
  truth sampled again; no statistical restarts, no floor).
- Moving: step-sized forgetting (a step d moves every arrival by at most
  d) and the history tests.
- The world changed (a door): every window restarts, and the next three
  ticks trace twice the rays (a single tick misses its rare strong paths).
- Every ray splits at its first partition — both branches with their
  exact weights — so a closed room's field is estimated from every ray
  that meets its walls (+1 ms a town tick).
- A mirror chain that solves stays cached while it does.

## Diffraction around buildings

Exact two-edge next-event paths: a string through an edge the ear sees
and one the source sees (the nearest of each within a detour ellipse),
bend points where the string is shortest (alternating), every leg
checked through the geometry, priced as a whole by its detour (ISO
9613-2 multi-edge). Edges with more than 205° of open air are exposed —
a gable's rake under a 50° roof is one. The walk no longer counts the
paths these carry. The bell on the square — the cathedral's nave in the
way — arrives at −39…−46 dB, steady, where it was −72 dB of noise.

## The ear

A comfortable band (adapted to 50–80 dB SPL) where nothing adapts: a
change between ordinary places plays at its full size. Above it loud
places duck, below it quiet ones come up (half the difference).
`Mixer::set_ear(false)` (Godot: settings → engine → ear) plays the scene
at its own level.

## Measured

`docs/runs/2026-10-29/README.md`: the club door shut — the response
within ±2.6 dB of settled from 50 ms on (was +12…+27 dB early); ten
identical clicks standing still — 0–0.9 dB sd (was 2–7 dB, spans of
10–21 dB); a car at 100 km/h — on time within 0.4 ms (was −42…+21 ms);
a click in flight while its source jumps or is switched off — arrives
and rings. Tests: `demo/web/wasm/tests/town_system.rs`.
