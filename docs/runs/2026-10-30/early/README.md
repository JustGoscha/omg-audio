# The dense early field (Oct 30)

`assess still EAR=0 RUNG=3 PACE=0 SEED=1 N=4 SRC=<club|hands> SPOT=<spot>`:
a click every 3 s, nothing moving; per click the energy of the first
80 ms after its arrival (`early`) and after (`late`), low / mid / high,
dB. `v5_*`: the traced response convolved (the sum of arrivals, the
reference); `v6.2_*`: the engine before the dense early field
(df90f4f); `dense_v6.2_*`: with it (ENGINE_V6.md, "The dense early
field"); `tried_dense_into_reverb_*`: the early bins past the strongest
24 added to the late reverb's level instead — reverted.

| place | v5 early | v6.2 early | v5 late | v6.2 late |
|---|---|---|---|---|
| inside the club | 5.7 / 8.9 / 16.8 | −0.7 / 4.2 / 12.7 | 2.8 / 5.9 / 9.9 | 1.3 / 5.3 / 9.8 |
| a clap in the hall | −11.9 / −5.9 / 2.2 | −12.6 / −6.2 / 1.9 | −18.0 / −14.7 / −9.8 | −17.7 / −13.8 / −9.9 |
| a clap in the cathedral | −12.5 / −6.1 / 1.9 | −12.7 / −6.3 / 1.7 | −25.7 / −21.7 / −16.3 | −22.0 / −18.8 / −15.3 |
| the club from the square | −46.9 / −55.7 / −53.3 | −42.1 / −52.1 / −53.2 | −42.5 / −52.3 / −49.4 | −44.8 / −52.4 / −59.7 |

(click 3 of 4 in each run.)

With the dense early field (`dense_v6.2_*`), against v5:

| place | early | late | whole |
|---|---|---|---|
| inside the club | −3.3 / −1.1 / −0.9 (was −6.4 / −4.7 / −4.1) | −1.5 / −0.1 / +0.7 | −2.6 / −0.7 / −0.6 (was −4.1 / −2.9 / −3.1) |
| the club's entrance | +1.1 / −0.5 / −0.3 | +0.1 / +0.7 / −0.6 | +0.4 / +0.3 / −0.4 |
| a clap in the hall | −0.5 / +0.1 / +0.1 | +0.4 / +1.0 / +0.2 | −0.3 / +0.2 / +0.1 |
| a clap in the cathedral | 0.0 / +0.1 / +0.2 | +3.8 / +3.1 / +2.0 | +0.3 / +0.2 / +0.2 |

In motion (`motion/`, `before_*` without it, `after_*` with it, three
seeds, `EAR=0 RUNG=3 PACE=0`): `fly` (clicks) the same pops per seed,
`fly SIGNAL=tone` (`flytone`) clean in both, `cartone` the same pitch
and level jitter, `walkout` the same tail, `doorswing` against
`doortruth DUR=8` (each build its own) rms 3.16 / 6.85 / 5.60 → 3.23 /
5.18 / 5.59 dB.
