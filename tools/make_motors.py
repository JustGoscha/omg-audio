#!/usr/bin/env python3
"""Build assets/motor{0..3}48.ogg: steady vehicle-motor loops for the
demo's passing cars, cut from CC0 freesound recordings and seam-
crossfaded so they loop without a click. The ENGINE renders the pass-by
(distance, air, Doppler) — these must be constant-RPM beds, never
pass-by recordings.

All sources are Creative Commons 0 (verified on their pages):
    https://freesound.org/people/<user>/sounds/<id>/
Usage: make_motors.py assets/
"""
import struct
import subprocess
import sys
import tempfile
import urllib.request

SOURCES = [
    ("AndrewAlexander/369053 Honda CRV idle", "https://cdn.freesound.org/previews/369/369053_1535801-hq.ogg"),
    ("qubodup/54909 bus motor", "https://cdn.freesound.org/previews/54/54909_71257-hq.ogg"),
    ("KVV_Audio/748274 diesel", "https://cdn.freesound.org/previews/748/748274_12846320-hq.ogg"),
    ("soundjoao/325808 motor loop", "https://cdn.freesound.org/previews/325/325808_5606411-hq.ogg"),
]

FS = 48000
SECONDS = 6.0
XFADE = 0.35


def decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(FS), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return list(struct.unpack(f"<{len(raw)//4}f", raw))


def loopify(x):
    """Steady middle segment, tail crossfaded into the head."""
    n = min(len(x), int(SECONDS * FS))
    start = max(0, (len(x) - n) // 2)
    seg = x[start:start + n]
    xf = int(XFADE * FS)
    out = seg[: n - xf]
    for k in range(xf):
        a = k / xf
        out[k] = seg[n - xf + k] * (1.0 - a) + out[k] * a
    peak = max(abs(v) for v in out) or 1.0
    return [v * 0.7 / peak for v in out]


def main():
    outdir = sys.argv[1].rstrip("/")
    for i, (name, url) in enumerate(SOURCES):
        with tempfile.NamedTemporaryFile(suffix=".ogg") as f:
            req = urllib.request.Request(url, headers={"User-Agent": "omg-audio-tools"})
            f.write(urllib.request.urlopen(req, timeout=30).read())
            f.flush()
            loop = loopify(decode(f.name))
            raw = struct.pack(f"<{len(loop)}f", *loop)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "f32le", "-ar", str(FS), "-ac", "1", "-i", "-",
                 "-c:a", "libopus", "-b:a", "96k", f"{outdir}/motor{i}48.ogg"],
                input=raw, check=True)
            print(f"motor{i}48.ogg ← {name} ({len(loop)/FS:.1f}s)")


if __name__ == "__main__":
    main()
