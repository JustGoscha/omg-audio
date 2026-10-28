#!/usr/bin/env python3
"""Build the engine's HRIR files from a full-sphere SOFA set (SADIE II).

    make_hrir_sofa.py <SimpleFreeFieldHRIR.sofa @48 kHz> <out_dir> [old_grid.bin]

Default set: SADIE II D2 (KEMAR, head and torso), 8802 directions over
the whole sphere (−90…+90°), diffuse-field equalized — University of
York, Apache-2.0, Armstrong et al., "A Perceptual Spectral Difference
Model for Binaural Signals", Applied Sciences 8(11):2029 (2018),
https://doi.org/10.3390/app8112029. (The MIT KEMAR set the engine used
before stops at −40°: everything below played as −40°.)

The SADIE HRIRs are (approximately) linear phase: their energy sits in
the middle of 256 taps, behind pre-ringing. Each ear becomes its
minimum-phase version (same magnitude), and the interaural time
difference — measured by cross-correlating the ears below 1.5 kHz — is
put back exactly, as a fractional delay on the later ear; 128 taps
(2.7 ms) hold it all, with no added latency. (Minimum phase plus ITD is
a perceptual equivalence, not physics: the ear hears the magnitude and
the interaural delay, not the per-ear phase.)

Written in the engine's flat format (see make_hrir.py): u32 count, u32
taps, then per direction 3×f32 dir (x forward, y left, z up), taps×f32
left, taps×f32 right.
  hrir_grid.bin        the dense grid, thinned to ~3° along each
                       measured ring (every ring kept)
  hrir_dodeca20.bin,   the virtual speakers of the ambisonic decode:
  hrir_ico12.bin       the measured direction nearest each vertex
With `old_grid.bin`, the set is scaled to its diffuse-field energy (the
mean of L² + R² over directions), so the engine's loudness calibration
holds.
"""
import math
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

TAPS = 128
NFFT = 1024
ARC_DEG = 3.0
PHI = (1 + 5**0.5) / 2


def min_phase(H):
    """The minimum-phase impulse response with magnitude spectrum H
    (rfft bins of NFFT; real cepstrum)."""
    logm = np.log(np.maximum(H, 1e-9))
    c = np.fft.irfft(logm, NFFT)
    w = np.zeros(NFFT)
    w[0] = 1.0
    w[1 : NFFT // 2] = 2.0
    w[NFFT // 2] = 1.0
    return np.fft.irfft(np.exp(np.fft.rfft(c * w, NFFT)), NFFT)


def lowpass(h, fs, fc=1500.0):
    X = np.fft.rfft(h, NFFT)
    f = np.fft.rfftfreq(NFFT, 1 / fs)
    X[f > fc] = 0.0
    return np.fft.irfft(X, NFFT)


def itd_samples(l, r, fs):
    """Right ear's delay behind the left (samples, fractional): the lag of
    the low-passed cross-correlation's peak, refined by a parabola."""
    a, b = lowpass(l, fs), lowpass(r, fs)
    xc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * NFFT)) * np.fft.rfft(b, 2 * NFFT), 2 * NFFT)
    lags = np.concatenate([np.arange(0, NFFT), np.arange(-NFFT, 0)])
    lim = int(0.0012 * fs)  # a head: |ITD| < ~1 ms
    ok = np.abs(lags) <= lim
    i = np.argmax(np.where(ok, xc, -np.inf))
    y0, y1, y2 = xc[i - 1], xc[i], xc[(i + 1) % len(xc)]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return lags[i] + frac


def delayed(h, d):
    """h delayed by d samples (fractional: a linear phase in frequency)."""
    X = np.fft.rfft(h, NFFT)
    k = np.arange(len(X))
    return np.fft.irfft(X * np.exp(-2j * np.pi * k * d / NFFT), NFFT)


def fade_to(h, n):
    out = h[:n].copy()
    m = 16
    out[n - m :] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(m) + 1) / m)
    return out.astype(np.float32)


def pair(l, r, fs, ml_ir=None, mr_ir=None):
    """Minimum-phase ears, the later one delayed by the measured ITD. With
    the mirrored direction's ears (`ml_ir`, `mr_ir`), symmetric: each
    ear's magnitude the geometric mean with the other ear's at the mirror,
    the ITD the mean of the two (the measured head is ~1 dB lopsided; a
    source straight ahead must not lean)."""
    mag = lambda h: np.abs(np.fft.rfft(h, NFFT))
    d = itd_samples(l, r, fs)
    hl, hr = mag(l), mag(r)
    if ml_ir is not None:
        hl, hr = np.sqrt(hl * mag(mr_ir)), np.sqrt(hr * mag(ml_ir))
        d = 0.5 * (d - itd_samples(ml_ir, mr_ir, fs))
    ml, mr = min_phase(hl), min_phase(hr)
    if d >= 0:
        mr = delayed(mr, d)
    else:
        ml = delayed(ml, -d)
    return fade_to(ml, TAPS), fade_to(mr, TAPS)


def vec(az, el):
    a, e = math.radians(az), math.radians(el)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def write_bin(path, records):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(records), TAPS))
        for d, l, r in records:
            f.write(struct.pack("<3f", *d))
            f.write(l.astype("<f4").tobytes())
            f.write(r.astype("<f4").tobytes())
    print(f"wrote {path} ({len(records)} HRIRs × {TAPS} taps)")


def read_bin(path):
    b = Path(path).read_bytes()
    n, taps = struct.unpack("<II", b[:8])
    off, recs = 8, []
    for _ in range(n):
        d = struct.unpack("<3f", b[off : off + 12])
        off += 12
        l = np.frombuffer(b[off : off + 4 * taps], "<f4")
        off += 4 * taps
        r = np.frombuffer(b[off : off + 4 * taps], "<f4")
        off += 4 * taps
        recs.append((d, l, r))
    return recs


def ico_vertices():
    raw = []
    for a in (-1, 1):
        for b in (-PHI, PHI):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    return [tuple(c / math.sqrt(sum(x * x for x in v)) for c in v) for v in raw]


def dodeca_vertices():
    raw = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    for a in (-1 / PHI, 1 / PHI):
        for b in (-PHI, PHI):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    return [tuple(c / math.sqrt(sum(x * x for x in v)) for c in v) for v in raw]


def main():
    sofa, out = Path(sys.argv[1]), Path(sys.argv[2])
    f = h5py.File(sofa, "r")
    fs = float(f["Data.SamplingRate"][0])
    assert abs(fs - 48000.0) < 1, f"want 48 kHz, got {fs}"
    pos = f["SourcePosition"][:]
    ir = f["Data.IR"][:]
    dirs = np.array([vec(az, el) for az, el, _ in pos])

    def sym_pair(i):
        # the measured direction mirrored left↔right (y → −y)
        m = dirs[i] * np.array([1.0, -1.0, 1.0])
        j = int(np.argmax(dirs @ m))
        return pair(ir[i, 0], ir[i, 1], fs, ir[j, 0], ir[j, 1])

    # the grid: every ring, ~ARC_DEG apart along it
    keep = []
    for el in np.unique(np.round(pos[:, 1], 2)):
        ring = np.where(np.abs(pos[:, 1] - el) < 0.01)[0]
        n = max(1, round(360.0 * math.cos(math.radians(el)) / ARC_DEG))
        want = np.arange(n) * 360.0 / n
        for az in want:
            da = np.abs(((pos[ring, 0] - az) + 180.0) % 360.0 - 180.0)
            keep.append(ring[int(np.argmin(da))])
    keep = sorted(set(keep))
    grid = []
    for i in keep:
        l, r = sym_pair(i)
        grid.append((tuple(dirs[i]), l, r))

    # the old set's diffuse-field energy: the engine's calibration
    scale = 1.0
    if len(sys.argv) > 3:
        old = read_bin(sys.argv[3])
        e_old = np.mean([np.sum(l * l) + np.sum(r * r) for _, l, r in old])
        e_new = np.mean([np.sum(l * l) + np.sum(r * r) for _, l, r in grid])
        scale = math.sqrt(e_old / e_new)
        print(f"diffuse-field energy: old {e_old:.4f}, new {e_new:.4f} → ×{scale:.3f}")
    grid = [(d, l * scale, r * scale) for d, l, r in grid]
    write_bin(out / "hrir_grid.bin", grid)

    for name, verts in (("hrir_ico12.bin", ico_vertices()), ("hrir_dodeca20.bin", dodeca_vertices())):
        recs = []
        for v in verts:
            i = int(np.argmax(dirs @ np.array(v)))
            l, r = sym_pair(i)
            recs.append((v, l * scale, r * scale))
        write_bin(out / name, recs)


if __name__ == "__main__":
    main()
