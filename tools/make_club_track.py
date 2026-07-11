#!/usr/bin/env python3
"""Synthesize a house loop (CC0 by construction) with heavy sub content —
the club-transmission demo needs guaranteed bass. Pure stdlib.

Usage: make_club_track.py assets/club48.wav [seconds]
"""
import math
import random
import struct
import sys
import wave

FS = 48000
BPM = 124.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT

random.seed(7)


def render(seconds):
    n = int(seconds * FS)
    out = [0.0] * n

    # A-minor-ish bass pattern, one note per beat (root, root, b3, b7 …)
    bass_seq = [55.0, 55.0, 65.41, 49.0, 55.0, 55.0, 73.42, 65.41]

    t_beat = 0.0
    beat_idx = 0
    while t_beat < seconds:
        s0 = int(t_beat * FS)

        # Kick: 150→48 Hz exponential pitch sweep + click, every beat.
        phase = 0.0
        for i in range(int(0.30 * FS)):
            t = i / FS
            f = 48.0 + 102.0 * math.exp(-t * 28.0)
            phase += 2 * math.pi * f / FS
            env = math.exp(-t * 11.0)
            k = s0 + i
            if k < n:
                out[k] += 0.95 * env * math.sin(phase)
        for i in range(int(0.004 * FS)):  # click
            k = s0 + i
            if k < n:
                out[k] += 0.25 * (random.random() * 2 - 1) * (1 - i / (0.004 * FS))

        # Sub bass note: starts after the kick transient, ducked (sidechain).
        f = bass_seq[beat_idx % len(bass_seq)]
        for i in range(int(0.42 * FS)):
            t = i / FS
            duck = min(1.0, t / 0.12)  # sidechain pump
            env = duck * min(1.0, (0.42 - t) / 0.06)
            k = s0 + int(0.02 * FS) + i
            if k < n:
                out[k] += 0.5 * env * math.sin(2 * math.pi * f * t)
                out[k] += 0.12 * env * math.sin(2 * math.pi * 2 * f * t)

        # Offbeat open hat.
        h0 = s0 + int(0.5 * BEAT * FS)
        hp = 0.0
        for i in range(int(0.09 * FS)):
            k = h0 + i
            if k < n:
                w = random.random() * 2 - 1
                hp = 0.6 * hp + w - 0.94 * w  # crude highpass flavor
                out[k] += 0.16 * (w - hp) * math.exp(-i / (0.03 * FS))

        # Clap on beats 2 and 4.
        if beat_idx % 2 == 1:
            for burst in (0.0, 0.012, 0.026):
                c0 = s0 + int(burst * FS)
                for i in range(int(0.05 * FS)):
                    k = c0 + i
                    if k < n:
                        out[k] += 0.2 * (random.random() * 2 - 1) * math.exp(-i / (0.012 * FS))

        # Sparse stab every 2 bars.
        if beat_idx % 8 == 6:
            for i in range(int(0.22 * FS)):
                t = i / FS
                env = math.exp(-t * 9.0)
                k = s0 + i
                if k < n:
                    for fmul, amp in ((220.0, 0.10), (277.18, 0.08), (329.63, 0.07)):
                        out[k] += amp * env * math.sin(2 * math.pi * fmul * t)

        t_beat += BEAT
        beat_idx += 1

    # Gentle soft clip + normalize.
    peak = max(abs(x) for x in out) or 1.0
    return [math.tanh(x / peak * 1.4) * 0.72 for x in out]


def main():
    path = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 70.0
    data = render(seconds)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(b"".join(struct.pack("<h", int(x * 32767)) for x in data))
    print(f"wrote {path}: {seconds}s house loop at {BPM} BPM (CC0 by construction)")


if __name__ == "__main__":
    main()
