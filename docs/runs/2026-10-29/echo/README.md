# A facade's echo of a passing car (Oct 29, night)

A car passes 30 m off at 100 km/h (`assess cartone`'s pass). The town's
west facade (x = 0) mirrors it to the ears for ~1.8 s, ~0.45 s late and
10–13 dB under it (`examples/echo_ref`: every triangle a mirror, both
legs checked), and two eastern faces earlier, ~7 dB under. The engine
played the west echo at two quality rungs of eleven; elsewhere it was
missing. Now it plays at every rung, with its twin off the ground
(`ENGINE_V6.md`, "Echoes found, not chanced upon").

| file | what |
|---|---|
| `headless_car_before.m4a` / `.txt` | rung 4, the tracer before (aec9e83^): no echo — the tone glides cleanly (0.27 Hz rms against its 25 ms course) |
| `headless_car_after.m4a` / `.txt` | rung 4 now: the echoes beat with the car (9.1 Hz rms; where in the pass, per half second) |
| `godot_carfar_v6.m4a` | the Godot game at 1f824db, the car's own motor: the slapback off the facade |
| `godot_carfar_v6_tone.m4a` / `.txt` | the same with the 500 Hz tone |

The beat is two arrivals with different Doppler shifts (the echo's path
lengthens more slowly than the direct one): physics, not jitter.
