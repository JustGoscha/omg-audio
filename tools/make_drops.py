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

SOURCES = [
    ("sandome/68330", "https://cdn.freesound.org/previews/68/68330_973966-hq.ogg"),
    ("deleted_user/166325", "https://cdn.freesound.org/previews/166/166325_2104797-hq.ogg"),
    ("mattfinarelli/533146", "https://cdn.freesound.org/previews/533/533146_7566729-hq.ogg"),
    ("AardsReal/842164", "https://cdn.freesound.org/previews/842/842164_13307919-hq.ogg"),
    ("DigPro120/558850", "https://cdn.freesound.org/previews/558/558850_7414526-hq.ogg"),
    ("ianslattery/168861", "https://cdn.freesound.org/previews/168/168861_1993921-hq.ogg"),
    ("MasterSuite/667386", "https://cdn.freesound.org/previews/667/667386_14357477-hq.ogg"),
    ("Lunardrive/22438", "https://cdn.freesound.org/previews/22/22438_120830-hq.ogg"),
    ("Mega-X-stream/546279", "https://cdn.freesound.org/previews/546/546279_4937681-hq.ogg"),
    ("Panska_Sand/498999", "https://cdn.freesound.org/previews/498/498999_10821817-hq.ogg"),
    ("paespedro/174718", "https://cdn.freesound.org/previews/174/174718_1850811-hq.ogg"),
    ("nebulasnails/495118", "https://cdn.freesound.org/previews/495/495118_2723982-hq.ogg"),
]

FS = 48000
SLOT = 7200  # 150 ms — keep in sync with rain.rs BANK_SLOT
MAX_PER_FILE = 3


def decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(FS), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return list(struct.unpack(f"<{len(raw)//4}f", raw))


def slice_hits(x):
    peak = max((abs(v) for v in x), default=0.0)
    if peak < 1e-4:
        return []
    on = peak * 0.12
    off = peak * 0.015
    hits = []
    i = 0
    n = len(x)
    while i < n and len(hits) < MAX_PER_FILE:
        if abs(x[i]) < on:
            i += 1
            continue
        start = max(0, i - int(0.003 * FS))
        end = i
        quiet = 0
        while end < n and end - start < SLOT:
            if abs(x[end]) < off:
                quiet += 1
                if quiet > int(0.030 * FS):
                    break
            else:
                quiet = 0
            end += 1
        seg = x[start:end]
        if len(seg) >= int(0.015 * FS):
            p = max(abs(v) for v in seg)
            seg = [v * 0.9 / p for v in seg]
            fade = min(int(0.005 * FS), len(seg))
            for k in range(fade):
                seg[len(seg) - fade + k] *= 1.0 - k / fade
            hits.append(seg)
        i = end + int(0.050 * FS)
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
