# WINDOWS_PLAN — why a strong Windows box "runs like shit", and the fix ladder

Field report: Windows desktop, good CPU, good discrete GPU, Chrome —
poor frame rate and/or broken audio. The Mac this engine was tuned on
is the *smaller* machine, so raw compute is not the problem. Every
plausible cause below is Windows-environment-specific, ordered by
likelihood × cheapness to fix. Each phase is small and independently
shippable; W0/W1 are already implemented.

## The mental model

Three loops share the machine and any ONE of them saturating drags all
three down, because they are coupled:

1. **Render loop** (main thread + GPU): three.js at the display refresh.
2. **Sim loop** (worker, 20 Hz): wasm tick + WebGPU compute dispatches
   (late-field trace + PT discovery) **on the same GPU** as rendering.
3. **Audio loop** (AudioWorklet, per-quantum): binaural render; misses
   its deadline → audible gaps (the governors shed quality to protect it).

Windows-specific multipliers this plan attacks:

- Desktop displays are commonly **4K and/or 144 Hz** — up to ~8 MP per
  frame at 2.4× the Hz the engine was tuned at. The render loop eats
  the whole GPU; the sim's compute dispatches queue behind it; `gpuMs`
  balloons; the governor sheds; audio still starves on bad frames.
- Chrome-on-Windows renders WebGL through **ANGLE→D3D11** and WebGPU
  through **D3D12**; buffer-map readbacks (our `mapAsync` results) have
  higher and spikier latency than on Metal.
- Dual-adapter machines (iGPU + dGPU) can hand Chrome the **iGPU**
  unless `powerPreference: 'high-performance'` is requested — and even
  then Windows "Graphics settings" per-app preference can override.
- The **audio device period** differs (WASAPI, often 10 ms shared-mode)
  and Bluetooth headsets on Windows add large, jittery device buffers.

## W0 — instrument first *(SHIPPED)*

One paste-able line each, so a report from any machine is diagnosable:

- `[sys] {plat, dpr, screen, gpu, webgpu, hz, px}` at startup — `gpu`
  is the ANGLE renderer string (names the actual adapter: if it says
  "Intel(R) UHD" on a 4080 box, that's the whole bug), `hz` is the
  measured rAF rate, `px` the real canvas resolution.
- `[sys-audio] {rate, base, hint}` when audio starts.
- Already there: blackbox ring (peak/agc/gaps/load/rafGap/outputLatency),
  stats panel (sim tick, render load, gaps, gpu dispatch ms).

**Ask a Windows tester for: the `[sys]` line, the `[sys-audio]` line,
and 30 s of the stats panel while walking through the club door.**

## W1 — GPU contention quick wins *(SHIPPED)*

- `powerPreference: 'high-performance'` on the WebGL renderer (the
  WebGPU adapter already requested it) — takes the dGPU on dual-adapter
  machines.
- **Physical-pixel budget** (3.4 MP, ≈ the Mac reference) instead of a
  bare `dpr ≤ 2` cap: on a 4K canvas the pixel ratio scales *down*
  (floor 0.6) so the render loop can never claim 8 MP × 144 Hz.
  `?px=8` lifts it (megapixels) for A/B.

## W2 — decouple the render loop from the audio path

The remaining structural risk: at 144 Hz even a within-budget render
loop doubles GPU queue pressure vs 60 Hz for zero perceptual gain in
this demo.

- **Frame limiter**: render at most every N rAF ticks so the effective
  rate is ≤ ~72 fps on high-refresh displays (sim/audio unaffected —
  they're on their own clocks; the camera interpolates anyway).
  `?fps=144` disables. Implementation: in the rAF callback, skip
  `renderer.render` (and only that — meters/HUD still update) when
  `now - lastRender < 1000 / cap`.
- **Governor hook**: if `gpuMs` (dispatch round-trip) stays above ~1.5×
  its floor for seconds while render load is high, drop the frame cap
  a notch before dropping AUDIO quality — shed photons before sound.

## W3 — audio robustness on WASAPI

- `?latency=playback` *(SHIPPED)* — bigger device buffers; the correct
  trade on any machine whose gaps counter climbs while `load` is fine
  (device-side starvation, not engine overload).
- **Auto-fallback**: persist a `localStorage` flag when a session ends
  with heavy `gaps`; next session starts with `latencyHint: 'balanced'`
  and a HUD note. (Recreating the context mid-session is audible —
  prefer next-session.)
- **Bluetooth detection**: `outputLatency` > ~150 ms or jittering →
  show the existing device-meter hint; Windows BT stacks are the top
  gap generator in the field.

## W4 — WebGPU dispatch hygiene on D3D12

Only if W1/W2 telemetry still shows `gpuMs` spikes at constant load:

- Deepen the trace buffer pool (4 → 8 slots) so a slow `mapAsync`
  round-trip never blocks the next dispatch (the seam already tolerates
  async results landing late — `poll_into`).
- Batch the PT job and trace job into one submit per tick (one fence
  instead of two).
- If map latency stays pathological: keep compute on GPU but move
  readback to a staging ring sized 3 ticks deep — accept 150 ms of
  sim-side staleness in exchange for zero stalls (late field is a
  slowly-varying statistic; the EMA doesn't care).

## W5 — verify, then lock in

- A/B on the Windows box: default vs `?px=8`, `?fps=144`,
  `?latency=playback`, gpu on/off, ism/traced — the stats panel numbers
  name the culprit in minutes.
- Whatever proves out becomes the default; the flags stay as overrides.
- Add the `[sys]` line to the blackbox dump header so field reports
  carry the environment automatically.
