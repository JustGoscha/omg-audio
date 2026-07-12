#!/usr/bin/env python3
"""Build assets/drops48.ogg: a bank of individual water-drop/splat hits,
sliced from CC0 freesound previews, peak-normalized, packed into uniform
150 ms slots (48 kHz mono). The engine triggers random slots with pitch
and gain variation for rain-on-window/roof impacts.

All sources are Creative Commons 0 (verified on their pages):
    https://freesound.org/people/<user>/sounds/<id>/
Usage: make_drops.py assets/
"""
import struct
import subprocess
import sys
import tempfile
import urllib.request

# Drop-on-SURFACE recordings only (can, metal ledge, umbrella, window):
# water-into-water drips read as bubbly plinks, not rain hitting a
# building — that distinction is the whole character of the bank.
SOURCES = [
    ("felix.blume/414101 drop on empty can", "https://cdn.freesound.org/previews/414/414101_1661766-hq.ogg"),
    ("Erbsland-Music/428605 rain on metal ledge", "https://cdn.freesound.org/previews/428/428605_522747-hq.ogg"),
    ("launemax/274765 raindrops under umbrella", "https://cdn.freesound.org/previews/274/274765_389377-hq.ogg"),
    ("petewyer2/438967 rain on umbrella", "https://cdn.freesound.org/previews/438/438967_1942716-hq.ogg"),
    ("xkeril/669486 rain on window interior", "https://cdn.freesound.org/previews/669/669486_13504080-hq.ogg"),
    ("ivolipa/329112 rain falling on umbrella", "https://cdn.freesound.org/previews/329/329112_3474310-hq.ogg"),
]

FS = 48000
SLOT = 7200  # 150 ms at the 48 kHz authoring rate (rain.rs BANK_SLOT_S)
MAX_PER_FILE = 6


def decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(FS), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return list(struct.unpack(f"<{len(raw)//4}f", raw))


def slice_hits(x):
    """Strictly curated: a usable hit must be ISOLATED (near-silence
    before the onset), SHORT (decays fast — no reverb tails or smears)
    and SPIKY (high crest — a tap, not a swell)."""
    peak = max((abs(v) for v in x), default=0.0)
    if peak < 1e-4:
        return []
    on = peak * 0.25
    off = peak * 0.02
    pre = int(0.040 * FS)
    hits = []
    i = pre
    n = len(x)
    while i < n and len(hits) < MAX_PER_FILE:
        if abs(x[i]) < on:
            i += 1
            continue
        # isolation: the 40 ms before the onset must be quiet
        if max(abs(v) for v in x[i - pre:i - int(0.002 * FS)] or [1.0]) > off * 2:
            i += int(0.010 * FS)
            continue
        start = max(0, i - int(0.002 * FS))
        end = i
        quiet = 0
        while end < n and end - start < SLOT:
            if abs(x[end]) < off:
                quiet += 1
                if quiet > int(0.020 * FS):
                    break
            else:
                quiet = 0
            end += 1
        seg = x[start:end]
        i = end + int(0.050 * FS)
        if len(seg) < int(0.012 * FS) or len(seg) > int(0.110 * FS):
            continue  # too tiny to read, or a smear
        p = max(abs(v) for v in seg)
        rms = (sum(v * v for v in seg) / len(seg)) ** 0.5
        if p / max(rms, 1e-9) < 4.5:
            continue  # not a tap
        # energy must be front-loaded (fast decay)
        half = len(seg) // 2
        e1 = sum(v * v for v in seg[:half])
        e2 = sum(v * v for v in seg[half:])
        if e2 > 0.35 * e1:
            continue
        seg = [v * 0.9 / p for v in seg]
        fade = min(int(0.004 * FS), len(seg))
        for k in range(fade):
            seg[len(seg) - fade + k] *= 1.0 - k / fade
        hits.append(seg)
    return hits


def main():
    outdir = sys.argv[1].rstrip("/")
    bank = []
    for name, url in SOURCES:
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg") as f:
                req = urllib.request.Request(url, headers={"User-Agent": "omg-audio-tools"})
                f.write(urllib.request.urlopen(req, timeout=30).read())
                f.flush()
                hits = slice_hits(decode(f.name))
                print(f"{name}: {len(hits)} hit(s)")
                bank.extend(hits)
        except Exception as e:  # noqa: BLE001 — a missing preview just shrinks the bank
            print(f"{name}: skipped ({e})")
    assert len(bank) >= 8, "bank too small"

    flat = []
    for seg in bank:
        flat.extend(seg)
        flat.extend([0.0] * (SLOT - len(seg)))
    raw = struct.pack(f"<{len(flat)}f", *flat)
    wav = f"{outdir}/drops48.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "f32le", "-ar", str(FS), "-ac", "1", "-i", "-",
         "-c:a", "libopus", "-b:a", "128k", f"{outdir}/drops48.ogg"],
        input=raw, check=True)
    print(f"wrote {outdir}/drops48.ogg: {len(bank)} slots × {SLOT / FS * 1000:.0f} ms")
    _ = wav


if __name__ == "__main__":
    main()
