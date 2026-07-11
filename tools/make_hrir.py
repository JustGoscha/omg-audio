#!/usr/bin/env python3
"""Build assets/hrir_ico12.bin from the MIT KEMAR compact HRIR set.

Picks the nearest KEMAR measurement for each of 12 icosahedron-vertex
directions, resamples 44.1 kHz → 48 kHz, and writes a flat little-endian
binary the engine can read without any parser dependencies:

    u32 count, u32 taps
    then per speaker: 3×f32 dir (x fwd, y left, z up), taps×f32 L, taps×f32 R

KEMAR convention: files H{elev}e{az:03}a.wav, azimuth clockwise from front,
0–180° measured; stereo = (left ear, right ear). Left hemisphere via
symmetry: swap ears at mirrored azimuth.

Usage: make_hrir.py <kemar_dir> <out_dir>
Writes: hrir_ico12.bin, hrir_dodeca20.bin (virtual-speaker decode grids)
        hrir_grid.bin (all measurements, both hemispheres — nearest-HRIR
        point rendering of direct paths)
"""
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TAPS_OUT = 128
PHI = (1 + 5**0.5) / 2


def ico_vertices():
    raw = []
    for a in (-1, 1):
        for b in (-PHI, PHI):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    out = []
    for v in raw:
        n = math.sqrt(sum(c * c for c in v))
        out.append(tuple(c / n for c in v))
    return out


def load_pair(kemar: Path, elev: int, az_deg: float):
    """HRIR (L, R) for our-frame direction at KEMAR elev/azimuth."""
    mirror = False
    az = az_deg % 360.0
    if az > 180.0:
        az = 360.0 - az
        mirror = True
    # available azimuths for this elevation
    files = sorted((kemar / f"elev{elev}").glob(f"H{elev}e*a.wav"))
    assert files, f"no files for elev {elev}"
    best = min(files, key=lambda f: abs(int(f.name.split("e")[1][:3]) - az))
    with wave.open(str(best)) as w:
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    left, right = data[0::2], data[1::2]
    if mirror:
        left, right = right, left
    return left, right, best.name


def dodeca_vertices():
    raw = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                raw.append((a, b, c))
    for a in (-1 / PHI, 1 / PHI):
        for b in (-PHI, PHI):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    out = []
    for v in raw:
        n = math.sqrt(sum(c * c for c in v))
        out.append(tuple(c / n for c in v))
    return out


def resample48(x44):
    y = resample_poly(x44, 160, 147)[:TAPS_OUT].astype(np.float32)
    return np.pad(y, (0, TAPS_OUT - len(y)))


def write_bin(path, records):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(records), TAPS_OUT))
        for (d, l, r) in records:
            f.write(struct.pack("<3f", *d))
            f.write(l.tobytes())
            f.write(r.tobytes())
    print(f"wrote {path} ({len(records)} HRIRs × {TAPS_OUT} taps)")


def main():
    kemar = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    elevs = sorted(
        int(d.name.replace("elev", "")) for d in kemar.iterdir() if d.name.startswith("elev")
    )

    # Virtual-speaker grids for the ambisonic bus decode.
    for name, verts in (("hrir_ico12.bin", ico_vertices()), ("hrir_dodeca20.bin", dodeca_vertices())):
        records = []
        for (x, y, z) in verts:
            elev_deg = math.degrees(math.asin(z))
            az_deg = math.degrees(math.atan2(-y, x)) % 360.0  # KEMAR az is clockwise
            elev = min(elevs, key=lambda e: abs(e - elev_deg))
            l44, r44, _ = load_pair(kemar, elev, az_deg)
            records.append(((x, y, z), resample48(l44), resample48(r44)))
        write_bin(out_dir / name, records)

    # Dense grid: every measurement, both hemispheres — for nearest-HRIR
    # point rendering of direct paths.
    records = []
    for elev in elevs:
        for f in sorted((kemar / f"elev{elev}").glob(f"H{elev}e*a.wav")):
            az = int(f.name.split("e")[1][:3])
            with wave.open(str(f)) as w:
                data = (
                    np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
                    / 32768.0
                )
            left, right = resample48(data[0::2]), resample48(data[1::2])
            el, azr = math.radians(elev), math.radians(az)
            # our frame: x fwd, y left, z up; KEMAR az clockwise from front
            x, y, z = math.cos(el) * math.cos(azr), -math.cos(el) * math.sin(azr), math.sin(el)
            records.append(((x, y, z), left, right))
            if 0 < az < 180:  # mirror to the other hemisphere, ears swapped
                records.append(((x, -y, z), right, left))
    write_bin(out_dir / "hrir_grid.bin", records)


if __name__ == "__main__":
    main()
