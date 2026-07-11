#!/usr/bin/env python3
"""Synthesize a loopable SERENE outdoor bed (CC0 by construction):
barely-there dark wind + sparse melodic birdsong with long silences.

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

    # --- air: very soft, very dark breath — three cascaded one-poles,
    # gentle swells, no gusts
    lp1 = lp2 = lp3 = 0.0
    for i in range(n):
        t = i / FS
        swell = 0.75 + 0.25 * math.sin(2 * math.pi * 2 * t / seconds) \
                     + 0.10 * math.sin(2 * math.pi * 5 * t / seconds + 0.7)
        w = random.random() * 2 - 1
        lp1 += 0.015 * (w - lp1)
        lp2 += 0.015 * (lp1 - lp2)
        lp3 += 0.03 * (lp2 - lp3)
        out[i] += 1.1 * swell * lp3

    # --- birds: sparse, melodic, soft — pure tones with slow vibrato,
    # falling two-note calls and the occasional gentle trill, long silences
    t_next = 2.0
    while t_next < seconds - 3.0:
        kind = random.random()
        t0 = t_next
        if kind < 0.6:
            # two-note falling call (soft, blackbird-ish)
            base = random.uniform(1700, 2600)
            for f0 in (base, base * random.uniform(0.78, 0.86)):
                dur = random.uniform(0.22, 0.38)
                amp = random.uniform(0.025, 0.05)
                s0 = int(t0 * FS)
                phase = 0.0
                for i in range(int(dur * FS)):
                    tt = i / (dur * FS)
                    f = f0 * (1.0 - 0.06 * tt) + 18 * math.sin(2 * math.pi * 5.5 * tt)
                    phase += 2 * math.pi * f / FS
                    env = math.sin(math.pi * tt) ** 1.5
                    idx = s0 + i
                    if idx < n:
                        out[idx] += amp * env * math.sin(phase)
                t0 += dur + random.uniform(0.25, 0.5)
        else:
            # gentle trill, far away
            base = random.uniform(2400, 3300)
            amp = random.uniform(0.015, 0.03)
            dur = random.uniform(0.5, 0.9)
            s0 = int(t0 * FS)
            phase = 0.0
            for i in range(int(dur * FS)):
                tt = i / (dur * FS)
                f = base + 140 * math.sin(2 * math.pi * 11 * tt)
                phase += 2 * math.pi * f / FS
                env = math.sin(math.pi * tt) ** 2 * 0.9
                idx = s0 + i
                if idx < n:
                    out[idx] += amp * env * math.sin(phase)
        t_next += random.uniform(4.0, 10.0)

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
