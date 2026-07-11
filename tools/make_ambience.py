#!/usr/bin/env python3
"""Synthesize a loopable outdoor nature bed (CC0 by construction):
wind through a wandering filter + sparse birdsong. Pure stdlib.

Usage: make_ambience.py assets/ambience48.wav [seconds]
"""
import math
import random
import struct
import sys
import wave

FS = 48000
random.seed(11)


def main():
    path = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    n = int(seconds * FS)
    out = [0.0] * n

    # --- wind: noise → two cascaded one-poles with slowly wandering cutoff
    lp1 = lp2 = 0.0
    wander = 0.0
    for i in range(n):
        t = i / FS
        wander += 0.00003 * ((random.random() * 2 - 1) - wander * 0.01)
        # gusts: two incommensurate slow LFOs, integer cycles over the loop
        gust = 0.55 + 0.3 * math.sin(2 * math.pi * 3 * t / seconds) \
                    + 0.15 * math.sin(2 * math.pi * 7 * t / seconds + 1.3)
        c = 0.02 + 0.05 * max(0.0, gust + 40.0 * wander)
        c = min(max(c, 0.008), 0.12)
        w = random.random() * 2 - 1
        lp1 += c * (w - lp1)
        lp2 += c * (lp1 - lp2)
        out[i] += 2.6 * gust * lp2

    # --- birds: sparse FM chirps, 2–6 s apart, varied species-ish patterns
    t_next = 1.0
    while t_next < seconds - 1.5:
        base = random.uniform(2200, 4400)
        n_notes = random.randint(2, 5)
        t0 = t_next
        for k in range(n_notes):
            dur = random.uniform(0.06, 0.16)
            sweep = random.uniform(-900, 1200)
            amp = random.uniform(0.05, 0.12)
            s0 = int(t0 * FS)
            phase = 0.0
            for i in range(int(dur * FS)):
                tt = i / (dur * FS)
                f = base + sweep * tt + 300 * math.sin(2 * math.pi * 28 * tt)
                phase += 2 * math.pi * f / FS
                env = math.sin(math.pi * tt) ** 2
                idx = s0 + i
                if idx < n:
                    out[idx] += amp * env * math.sin(phase)
            t0 += dur + random.uniform(0.04, 0.18)
        t_next += random.uniform(2.0, 6.0)

    # loop seam: crossfade the last second into the first
    xf = FS
    for i in range(xf):
        a = i / xf
        out[i] = out[i] * a + out[n - xf + i] * (1 - a)
    data = out[: n - xf]

    peak = max(abs(x) for x in data) or 1.0
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(b"".join(
            struct.pack("<h", int(x / peak * 0.7 * 32767)) for x in data))
    print(f"wrote {path}: {len(data)/FS:.1f}s loopable nature bed")


if __name__ == "__main__":
    main()
