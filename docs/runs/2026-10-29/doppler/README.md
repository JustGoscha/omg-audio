# Doppler in the hosts: one instant per frame (Oct 29, evening)

A 500 Hz tone on a car passing 30 m off at 100 km/h (`assess cartone`'s
pass: the car on x = −130 from y = −50, the ears at (−100, 100)), and the
car standing while the ears walk past it at 6 m/s (from (−100, 65)
north). Each recording is demodulated at the tone (`tools/demod_tone.mjs`,
the same arithmetic as `assess cartone`): pitch and level per ms against
their own 25 ms course. What moves faster than the motion can is jitter.

| host | case | before | after | headless, same quality |
|---|---|---|---|---|
| web (Chrome) | car | 6.78 Hz rms, 0.68 dB, 154 ms off by >2 dB (19.15 Hz, 2.89 dB without the page's time stamps) | **0.58 Hz, 0.15 dB, 0** | 0.26 Hz, 0.04 dB |
| web (Chrome) | walk | 4.83 Hz, 0.25 dB | **0.04 Hz, 0.01 dB** | 0.03 Hz, 0.01 dB |
| Godot | car | 9.02 Hz, 0.97 dB, 488 ms | **0.27 Hz, 0.04 dB, 0** | 0.26 Hz, 0.04 dB |
| Godot | walk | 0.74 Hz, 0.02 dB | **0.04 Hz, 0.02 dB** | 0.03 Hz, 0.01 dB |

The tone runs at the top quality rung (Godot `--quality=3`, the web's
governor at level 10, `assess LEVEL=10`): lower rungs add the estimate's
own wobble (`assess cartone` governed: 4.3 Hz — OPEN.md), the same in
every host. 9.0 Hz rms on 500 Hz is ±1.8 % — about 30 cents of warble.

## What was wrong

A frame is the world at one instant; the mixer moves every path's delay
over the time between two frames' instants. Three things broke that:

- **The frame's time was the listener's.** Frames carried the time the
  listener's pose was stamped with, while the paths reached the moving
  sources at the simulation's clock. In Godot a still listener kept the
  stamp of its first send — no time passed between any two frames, and
  the mixer divided each frame's motion by the average interval between
  arrivals while the motion per frame varied (8 or 17 ms of it): the car
  warbled. Now a frame carries the clock it was traced at
  (`Frame::time`); an exact frame of an instant already traced is not
  made, and a whole tick of that instant (nothing new since) is that
  moment traced again — its paths glide on as they were.
- **Hosts stamped with the wrong time.** The web page posted its pose 20
  times a second, stamped when posted, and the worker's clock ran on
  ahead; Godot sent the ears every rendered frame (120 Hz) while they
  move on physics steps (60 Hz), each with the frame's own time — the
  ears stood still a frame and moved twice as fast the next. Now the page
  posts every rendered frame stamped with that frame's time and the
  worker makes one frame of each (a whole tick every 50 ms of it); Godot
  stamps every pose and place of a frame with the last physics step's
  time (game time: steps exactly a step apart).
- **The mixer's update windows tore at unequal steps.** Each update
  glides from the previous target to the new one over its motion; its
  window began the *new* motion after the previous window began instead
  of where the previous one ended. Equal steps hid it; frames 8 and 17 ms
  apart tore the timeline by the difference every update. Fixed: the
  running listener under game ticks (3 or 4 physics steps a tick) went
  0.106 → 0.068 % worst 20 ms deviation at 27 m/s.

`crates/omg-audio/tests/fx_doppler_host.rs` `doppler_stays_steady_under_
the_threaded_engines_timing` replays the threaded engine's timing (120 Hz
frames, 60 Hz physics, exact frames at 120 Hz on their own phase, 20 Hz
ticks, a 512-sample audio callback taking the newest frame): worst 20 ms
deviation 0.009–0.026 %; the ears stamped with each frame's time 2.0–5.0
%; on the engine before, the running ears 1.17 % (limit 0.46 %).

## The second car recording

`godot_car_after_echo` (1.75 Hz) beats at ~30 Hz between 1.5 and 2 s:
the mirror chain finder found the town's west facade (x ≈ 0) — an echo
13 dB under the car, ~0.47 s late, its Doppler ~26 Hz lower. A real
echo, found in some runs and not in others: OPEN.md, "A distinct echo
found by chance".

## Files

`<host>_<case>_<before|after>.m4a` (listen) and `.txt` (the
demodulation). Before = 670de1f; web_car_unstamped = the page without
its time stamps (as before Oct 29). The web recordings are the page's
own output with its ear model on (quiet: −48 dB). Reproduce:

    python3 tools/serve.py &
    node tools/web_doppler.mjs car.wav            # WALK=6 LIS=-100,65 Y0=100: the walk
    node tools/demod_tone.mjs car.wav
    Godot --path demo/godot -- --record=<dir> --scenario=carfar|walk --engine=v6 --tone --quality=3
    LEVEL=10 EAR=0 cargo run --release -p omg-web --example assess -- cartone   # WALK=6 LIS=-100,65,1.6 Y0=100
