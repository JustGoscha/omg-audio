#!/usr/bin/env python3
"""Synthesize projectile FX (CC0 by construction): whistle, thump, boom.
Pure stdlib. Usage: make_fx.py assets/
"""
import math
import random
import struct
import sys
import wave

FS = 48000
random.seed(3)


def write(path, data):
    peak = max(abs(x) for x in data) or 1.0
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1, min(1, x / peak * 0.8)) * 32767)) for x in data))
    print("wrote", path, f"{len(data)/FS:.1f}s")


def whistle(seconds=4.0):
    # constant wind-whistle: steady tone with slow drift, breathy band noise
    n = int(seconds * FS)
    out = [0.0] * n
    phase = 0.0
    lp = 0.0
    bp = 0.0
    drift = 0.0
    for i in range(n):
        t = i / FS
        drift += 0.0007 * ((random.random() * 2 - 1) - drift)  # slow wander
        f = 2100.0 * math.exp(-t * 0.10) * (1.0 + 0.15 * drift)
        phase += 2 * math.pi * f / FS
        w = random.random() * 2 - 1
        lp += 0.10 * (w - lp)     # wide noise floor
        bp += 0.30 * (lp - bp)    # breathy band around the tone
        env = min(1.0, t / 0.15) * 0.9
        out[i] = env * (0.45 * math.sin(phase) + 0.65 * bp + 0.2 * lp)
    return out


def thump(seconds=0.25):
    n = int(seconds * FS)
    out = [0.0] * n
    phase = 0.0
    for i in range(n):
        t = i / FS
        f = 55.0 + 120.0 * math.exp(-t * 40.0)
        phase += 2 * math.pi * f / FS
        env = math.exp(-t * 22.0)
        out[i] = env * math.sin(phase)
        if i < int(0.006 * FS):
            out[i] += 0.5 * (random.random() * 2 - 1) * (1 - i / (0.006 * FS))
    return out


def boom(seconds=2.5):
    n = int(seconds * FS)
    out = [0.0] * n
    phase = 0.0
    lp = 0.0
    for i in range(n):
        t = i / FS
        f = 26.0 + 90.0 * math.exp(-t * 6.0)
        phase += 2 * math.pi * f / FS
        env = math.exp(-t * 2.2)
        w = random.random() * 2 - 1
        lp += 0.04 * (w - lp)  # rumbly filtered noise
        out[i] = env * (1.1 * math.sin(phase) + 2.4 * lp)
        # crackle tail
        if random.random() < 0.0004 * math.exp(-t * 1.5):
            for k in range(i, min(i + 200, n)):
                out[k] += 0.25 * math.exp(-(k - i) / 60.0) * (random.random() * 2 - 1)
    return out


d = sys.argv[1].rstrip("/")
write(f"{d}/fx_whistle.wav", whistle())
write(f"{d}/fx_thump.wav", thump())
write(f"{d}/fx_boom.wav", boom())
