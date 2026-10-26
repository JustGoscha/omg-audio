#!/usr/bin/env python3
"""Build assets/fountain48.ogg: a small garden fountain, synthesized in-repo
(no recording, no license to track). A steady loop — the ENGINE places it,
so this is the water as heard from close by, dry.

The sound of a jet falling into a basin, from four layers:
  - the jet's hiss: white noise band-passed around 2.5 kHz, its level
    wandering slowly (the jet is never quite steady);
  - the splash bed: noise band-passed lower (~700 Hz) with a fast random
    flutter — sheets of water landing;
  - droplets: many short bubble 'plinks' (decaying sines whose pitch rises
    a little as the bubble rings — Minnaert resonance), log-uniform
    800–4000 Hz, ~30 a second, random levels;
  - a low burble: fewer, bigger bubbles at 150–400 Hz.
Seed-fixed (the asset is reproducible); the tail is crossfaded into the
head so it loops without a seam.

Usage: make_fountain.py assets/
"""
import math
import random
import struct
import subprocess
import sys

FS = 48000
SECONDS = 12.0
XFADE = 0.5


def biquad_bp(x, f0, q):
    """RBJ band-pass (constant 0 dB peak gain)."""
    w = 2 * math.pi * f0 / FS
    alpha = math.sin(w) / (2 * q)
    b0, b1, b2 = alpha, 0.0, -alpha
    a0, a1, a2 = 1 + alpha, -2 * math.cos(w), 1 - alpha
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    y = [0.0] * len(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):
        o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, v, y1, o
        y[i] = o
    return y


def wander(n, rate_hz, rng):
    """A slow random level curve around 1 (smoothed noise, ±~35 %)."""
    k = 1 - math.exp(-2 * math.pi * rate_hz / FS)
    a = b = 0.0
    out = [0.0] * n
    for i in range(n):
        a += k * (rng.gauss(0, 1) - a)
        b += k * (a - b)
        out[i] = b
    s = max(1e-9, max(abs(v) for v in out))
    return [1.0 + 0.35 * v / s for v in out]


def bubbles(buf, rng, rate, f_lo, f_hi, decay_ms, level):
    n = len(buf)
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        start = int(t * FS)
        if start >= n:
            break
        f0 = math.exp(rng.uniform(math.log(f_lo), math.log(f_hi)))
        tau = rng.uniform(*decay_ms) / 1000.0
        amp = level * math.exp(rng.gauss(0, 0.6))
        ph = 0.0
        for j in range(int(tau * 6 * FS)):
            if start + j >= n:
                break
            tt = j / FS
            f = f0 * (1 + 0.6 * tt / (tau * 6))  # the ring rises as the bubble shrinks
            ph += 2 * math.pi * f / FS
            env = math.exp(-tt / tau) * min(1.0, j / 24)  # 0.5 ms attack: no click
            buf[start + j] += amp * env * math.sin(ph)


def main():
    out_dir = sys.argv[1].rstrip("/")
    rng = random.Random(0xF0_0D)
    n = int((SECONDS + XFADE) * FS)
    noise = [rng.gauss(0, 1) for _ in range(n)]
    noise2 = [rng.gauss(0, 1) for _ in range(n)]
    hiss = biquad_bp(noise, 2500.0, 0.6)
    lvl = wander(n, 0.8, rng)
    splash = biquad_bp(noise2, 700.0, 0.9)
    flutter = wander(n, 14.0, rng)
    buf = [0.30 * h * l + 0.55 * s * f * f for h, l, s, f in zip(hiss, lvl, splash, flutter)]
    bubbles(buf, rng, 30.0, 800.0, 4000.0, (4.0, 18.0), 0.18)
    bubbles(buf, rng, 4.0, 150.0, 400.0, (25.0, 60.0), 0.35)
    # loop: crossfade the tail into the head (equal power)
    m = int(SECONDS * FS)
    xf = int(XFADE * FS)
    out = buf[:m]
    for k in range(xf):
        a = k / xf
        out[k] = buf[m + k] * math.cos(a * math.pi / 2) + out[k] * math.sin(a * math.pi / 2)
    peak = max(abs(v) for v in out) or 1.0
    out = [v * 0.7 / peak for v in out]
    raw = struct.pack(f"<{len(out)}f", *out)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "f32le", "-ar", str(FS), "-ac", "1", "-i", "-",
         "-c:a", "libopus", "-b:a", "96k", f"{out_dir}/fountain48.ogg"],
        input=raw, check=True)
    print(f"fountain48.ogg ({len(out) / FS:.1f}s, synthesized)")


if __name__ == "__main__":
    main()
