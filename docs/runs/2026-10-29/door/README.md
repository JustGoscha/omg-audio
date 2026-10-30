# A door heard where it is: v6.1 against v6.2 (Oct 29–30, night)

The club's inner door swinging shut while it plays, the ears 1 m before
it. v6.1 averages each tick's traced field over the last few ticks: the
reverb and the early sound behind a moving door trail the leaf. v6.2
prices the averaged paths through the door with its leaf where it is
now, every tick and frame (ENGINE_V6.md, "v6.2").

## Godot (the real host)

`record.gd door`: the door opens at 2 s and shuts at 6 s (eased over
about a second), quality auto, the walk on the GPU (Godot's default),
recorded from the audio thread.
`door_<engine>.m4a` to listen (the club's own music),
`door_<engine>_tone.m4a` the club playing a 500 Hz tone alone; `.txt`
the analyses, `.frames.csv` per game frame.

Level of the tone, every 50 ms from 5.9 s (dB):

    v6.1  -7 -7 -7 -7 -7 -9 -10 -9 -9 -8 -8 -9 -10 -10 -11 -11 -12 -13 -15 -17 -18 -20 -21 -22 -23 -38 -44 -48 -42 -34
    v6.2  -7 -7 -7 -7 -7 -7 -7 -8 -10 -11 -10 -13 -16 -18 -20 -22 -24 -26 -28 -29 -31 -33 -32 -33 -34 -36 -39 -36 -34 -34

0.7 s after the shut command v6.2 is at −26 dB, falling with the leaf;
v6.1 holds at −11 … −13 dB, then plunges 25 dB under where it settles and
climbs back (the restart where the leaf comes to rest). Level jumps: v6.2
none, v6.1 two over 2 dB shut. Above 3 kHz, worst −48.8 dB (v6.2)
against −35.2 dB (v6.1, four windows over −40 dB). The tick, 4.8–19.2 ms
(v6.1) and 6.3–30.0 ms (v6.2: the CPU walks while the door moves).

(Earlier recordings of the night — before the CPU took over a moving
door's ticks, and before the GPU's rays passed a leaf half the time —
showed v6.2 ahead less cleanly.)

## Headless, against the door standing still

`assess doorswing EAR=0` (the door's openness e^(−7 t) from 4 s, as
Godot eases it) against `assess doortruth DUR=8` — the door standing at
each openness the swing passes, 8 s averaged: what the swing should
sound like at each instant. rms difference over the swing's first 1.25
s, three seeds (`headless_doorswing_<engine>_seed<n>.txt`, truths
`headless_doortruth_<engine>.txt`, their mean the reference):

| engine | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| v6.1 | 5.3 dB | 6.0 dB | 6.2 dB | 5.85 dB |
| v6.2 | 3.2 dB | 3.7 dB | 5.3 dB | 4.08 dB |

`headless_doorswing_<engine>.m4a`: the default seed, to listen.

On the GPU walk, as Godot runs it (`COMPUTE=gpu`, built with `--features
gpu`; `headless_gpu_doorswing_<engine>_seed<n>.txt`): v6.1 13.1 / 11.7 /
11.2 dB rms (its walk arrives a tick late on top of the averaging: +10
dB on average, the door heard where it was), v6.2 3.7 / 3.5 / 3.5 dB —
while a door moves the tick walks on the CPU, which prices the door.

Standing doors on the GPU (`headless_gpu_still_*`, `headless_gpu_door_*`):
the club from the square, click to click, early 1.6 / 1.7 / 2.2 → 0.7 /
0.2 / 0.3 dB, late 5.6 / 2.1 / 2.6 → 1.0 / 0.5 / 1.0 dB; a slammed door
0.1 s on, early +12.4 / +13.5 / +13.4 → −0.3 / −1.2 / −0.2 dB from where it
settles; and settled, the shut door against open early −15 / −16 / −16
dB (v6.1 on the GPU: a shut door nearly open in the treble) → −22 / −36
/ −45 dB (as the CPU's walk reads it) — the GPU's early rays pass a
door's leaf half the time (the draw's floor was one in thirty).

The isolated test (`fx_simulation::a_door_swinging_shut_is_heard_where_it_is`,
a room with a doorway, the ears at four places before and beside it):
v6.1 4.4 dB, v6.2 2.9 dB rms (three seeds: 4.80 → 3.40).

## Standing doors, and the rest (headless)

`headless_<scenario>_<engine>.txt` (`assess <scenario> EAR=0`): `still`
— the club heard from the square through its walls and shut door, a
click every 3 s: late spread 3.0 / 2.4 / 2.6 → 1.1 / 0.9 / 1.2 dB (v6.1's
drifting 7 dB down over 30 s, v6.2's steady: the split at the leaf);
`door` — the club's door slammed shut, 0.1 s on: the highs +8 / +11 dB
over where they settle (v6.1) → −5 / −3 dB (v6.2); `walkout`, `fly` —
unchanged.

## Reproduce

    sh tools/build_godot.sh
    sh tools/record_runs.sh $PWD/docs/runs/2026-10-29/door "v6.1 v6.2" door
    for s in 1 2 3; do SEED=$s EAR=0 MODE=v6.2 cargo run --release -p omg-web --example assess -- doorswing; done
    DUR=8 EAR=0 MODE=v6.2 cargo run --release -p omg-web --example assess -- doortruth
    cargo run --release -p omg-web --example door_ms -- v6.1 v6.2    # the cost
    SEED=1 COMPUTE=gpu EAR=0 MODE=v6.2 cargo run --release -p omg-web --features gpu --example assess -- doorswing
    COMPUTE=gpu cargo run --release -p omg-web --features gpu --example door_ms -- v6.1 v6.2
