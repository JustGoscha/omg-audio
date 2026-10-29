# A door heard where it is: v6.1 against v6.2 (Oct 29–30, night)

The club's inner door swinging shut while it plays, the ears 1 m before
it. v6.1 averages each tick's traced field over the last few ticks: the
reverb and the early sound behind a moving door trail the leaf. v6.2
prices the averaged paths through the door with its leaf where it is
now, every tick and frame (ENGINE_V6.md, "v6.2").

## Godot (the real host)

`record.gd door`: the door opens at 2 s and shuts at 6 s (eased over
about a second), quality auto, recorded from the audio thread.
`door_<engine>.m4a` to listen (the club's own music),
`door_<engine>_tone.m4a` the club playing a 500 Hz tone alone; `.txt`
the analyses, `.frames.csv` per game frame.

Level of the tone, every 50 ms from 5.9 s (dB):

    v6.1  -5 -5 -5 -5 -7 -8 -7 -8 -8 -10 -11 -11 -11 -11 -11 -12 -12 -12 -13 -14 -16 -18 -19 -20 -28 -39 -37 -30 -32
    v6.2  -6 -6 -6 -6 -7 -7 -9 -10 -11 -12 -13 -9 -11 -15 -17 -19 -22 -24 -24 -24 -25 -29 -31 -32 -35 -35 -30 -29 -29

0.7 s after the shut command v6.2 is 22 dB down; v6.1 holds at −11 …
−12 dB until the leaf settles and drops 20 dB at once. Within 3 dB of the
shut level at 6.94 s (v6.2) against 7.11 s (v6.1). Both: no level jumps
over 2 dB, nothing over −40 dB above 3 kHz in any phase.

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
