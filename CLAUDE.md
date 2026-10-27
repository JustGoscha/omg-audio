# omg-audio — manifesto

omg-audio is a real-time, path-traced, binaural sound engine. Everything
heard is simulated from the geometry the host passes in. Every change is
developed against this document: when a shortcut conflicts with it, the
shortcut loses. (Collected from the author's own words, Oct 25–27 2026.)

## 1. Physics, not assumptions

- **Maximum realism and physical accuracy.** No perceptual guesses, no
  hand-tuned constants, no fudge factors, no arbitrary cutoffs (no 80 ms
  early/late split, no reverb cap at 5 s, no bounce cap that truncates a
  tail). *"The hand tuned constants make no sense to me. It should be
  physically simulated."* *"A lot of your optimizations seem like cheap
  assumptions that are going against realism."*
- **One simulation for everything.** Inside and outside, near and far,
  early and late are the same model. *"I don't understand why inside is
  modeled differently than outside at all. The simulation itself should
  account for everything."*
- **The bitter lesson.** Path trace everything — rays from the listener,
  reflections, connections to sources with transmission through walls,
  diffraction — and echo, flutter, reverb, occlusion and coupled rooms come
  out of it. An echo in the mountains can arrive 3 s later; path tracing
  must carry it, not cut it.
- **Rays leave from the listener** (cost per listener, not per source).
  A far source grows with distance so the rays still find it; the only
  reason to drop a source is that it is too quiet to matter.
- **Physical materials.** Surfaces sound like what they are (pavement vs
  grass, wood, glass, brick, trees). Walls and doors attenuate as their
  materials do: a closed door sounds closed.
- **Sound takes time to travel.** A bell 3–10 km away is heard seconds
  later, if nothing is in the way. Doppler is real motion only: no
  warble, no tape-delay, no flanging from jittering delays.

## 2. Sound lives on a timeline

- Every source is a timeline of what it emitted when; each path reads it
  at *now − its travel time*. *"We still hear the past."* A finished sound
  still arrives, echoes and rings until the paths carry no more energy.
  **Nothing is ever cut because the source stopped emitting.**
- Sample and loop sounds are a record of what played when (no audio
  buffer); live-generated sounds record into a buffer read back later.
- Reverb dies because the rays can no longer reach you — not by a timer
  or a fade — and it does not linger where it cannot reach (walk out of a
  room: its reverb goes as fast as its paths do).

## 3. Real time over estimation and averaging

- **Realism in real time — good enough for a first-person shooter.** Fast
  moves, gunshots, doors slammed while running: the sound keeps up.
- *"We should aim for real time over estimations and averaging. It is
  possible."* Compute what is true now, every frame, instead of smoothing
  statistics over seconds. A door closing, walking out of a room, flying
  through walls: the sound reacts at once — no seconds of lag, no cracks.
- **Fast ticks.** Paths update at least every frame (close sources
  faster); a 50–100 ms tick is far too slow for moving sounds.
- **Modest ray counts are fine.** *"I usually barely notice differences
  between 1000 rays and 8000 rays."* Spend compute on correctness and
  responsiveness, not on more samples of the same estimate.
- **The response is the sum of arrivals.** Traced arrivals land at their
  exact delays in direction buckets; HRTF per direction. Prefer that over
  separate parametric reverbs; convolve only where it *is* that sum. An
  arrival many dB above its neighbours is its own distinct echo; many at
  similar loudness are a dense field.

## 4. Dynamic range like an ear

- A loudness stage protects ears and keeps the dynamic range: when it is
  loud everything ducks; when it is quiet more becomes audible.
- A sliding audible window may skip sounds too quiet to hear under the
  current adaptation — judged by what **arrives at the ear** (the timeline
  through the paths, tails included), never by what a source emits now.

## 5. Performance is a design constraint

- Fast on the web (CPU and GPU) and natively (Godot at least as fast as
  the web). 64–256 sources; aim for cost that grows sub-linearly.
- **The GPU never blocks graphics**: a configurable share (default 20 %),
  an implementer's choice.
- **Auto-scaling in every host**: governors walk quality by measured cost
  — shed at once, climb when calm.
- Performance tests as you go; measure before and after every change.

## 6. One engine, apart from its demos

- One core (omg-audio) used identically by Godot, the web (three.js or
  any web engine) and native. No per-platform engines; consolidate
  duplicated code. Rebuild every host after an engine change.
- The engine ships on its own: hosts pass geometry and tag materials with
  simple primitives. Levels, layouts and sounds are demo content in
  `demo/`, never in engine crates.
- **Easy to drop in.** Ideally the engine takes a Godot or three.js
  scene's own geometry and tags: drop it into a scene, tag what needs
  tagging (with hints so an AI agent can tag materials automatically),
  and the engine simplifies the geometry itself when it must.
- Works on headphones and on speakers (directional cues still help on 2D
  speakers).

## 7. Tests

- Every effect has an isolated, headless test of the core with just
  geometry passed in; the demo is where they come together.
- Claim an improvement only after measuring it headless against a
  baseline (A/B). Don't revert to escape a regression — use the earlier
  state as the baseline and improve on it.

## 8. How we work

- Sound-engine work is sequential, one step at a time, in the main
  session; parallel agents only for purely visual changes.
- Commit and push finished work.
- Explain in simple terms when asked.

## Build and test

- `export PATH="$HOME/.cargo/bin:$PATH"`
- Tests: `cargo test --workspace --exclude omg-godot --release --no-fail-fast`
- Web: `sh tools/build_web.sh`, then `node tools/web_smoke.mjs`; serve
  with `python3 tools/serve.py` → http://localhost:8000/demo/web/
- Godot: `sh tools/build_godot.sh`; test with
  `/Applications/Godot.app/Contents/MacOS/Godot --headless --path demo/godot -s test.gd`
