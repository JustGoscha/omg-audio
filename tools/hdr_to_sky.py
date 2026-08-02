#!/usr/bin/env python3
"""Poly Haven night HDRI -> minimal equirect sky PNG.

Keeps the star hemisphere, dissolves the photographic horizon band and
everything below it into a clean fog-colored gradient (the demo's own
geometry + fog owns the ground), and exposes from SKY-only samples so
the Milky Way reads. Pure stdlib (RGBE decode + PNG encode).

usage: hdr_to_sky.py in.hdr out.png
"""
import math
import struct
import sys
import zlib

SRC, DST = sys.argv[1], sys.argv[2]
FOG = (0x1a / 255, 0x22 / 255, 0x33 / 255)  # match scene fog (linear-ish)
HORIZON = 0.455   # v where the photo content starts to dissolve
BLEND_END = 0.52  # fully fog below this
data = open(SRC, 'rb').read()

pos = 0
while True:
    nl = data.index(b'\n', pos)
    if nl == pos:
        pos = nl + 1
        break
    pos = nl + 1
nl = data.index(b'\n', pos)
parts = data[pos:nl].decode().split()
pos = nl + 1
assert parts[0] == '-Y' and parts[2] == '+X'
H, W = int(parts[1]), int(parts[3])

rgbe = bytearray(W * H * 4)
o = pos
for y in range(H):
    row = y * W * 4
    assert data[o] == 2 and data[o + 1] == 2
    o += 4
    for c in range(4):
        x = 0
        while x < W:
            cnt = data[o]; o += 1
            if cnt > 128:
                v = data[o]; o += 1
                for _ in range(cnt - 128):
                    rgbe[row + x * 4 + c] = v
                    x += 1
            else:
                for _ in range(cnt):
                    rgbe[row + x * 4 + c] = data[o]; o += 1
                    x += 1

# exposure from the SKY only (top 45%), 98th percentile -> bright star
ldexp = math.ldexp
lums = []
for i in range(0, W * int(H * 0.45), 397):
    e = rgbe[i * 4 + 3]
    if e:
        s = ldexp(1.0, e - 136)
        lums.append(0.2126 * rgbe[i * 4] * s + 0.7152 * rgbe[i * 4 + 1] * s
                    + 0.0722 * rgbe[i * 4 + 2] * s)
lums.sort()
# expose the sky BACKGROUND (median) to a gentle glow; stars saturate
# on their own. A percentile-of-star target just amplifies sensor noise.
k = 0.042 / max(lums[len(lums) // 2], 1e-6)
print('sky exposure k =', round(k, 2))
# subtle night grade: cool the warm sensor gray toward blue
TINT = (0.88, 0.95, 1.12)

inv_g = 1 / 2.05
lut = {}
raw = bytearray()
for y in range(H):
    raw.append(0)
    v = y / H
    if v >= BLEND_END:
        # below the horizon: fog fading toward near-black at the nadir
        t = min(1.0, (v - BLEND_END) / (1.0 - BLEND_END))
        f = 1.0 - 0.65 * t
        px = bytes(min(255, int(255 * ((c * f) ** inv_g) + 0.5)) for c in FOG)
        raw += px * W
        continue
    blend = 0.0 if v < HORIZON else (v - HORIZON) / (BLEND_END - HORIZON)
    base = y * W * 4
    for x in range(W):
        i = base + x * 4
        e = rgbe[i + 3]
        if e == 0:
            r = g = b = 0.0
        else:
            s = ldexp(1.0, e - 136) * k
            r = rgbe[i] * s * TINT[0]
            g = rgbe[i + 1] * s * TINT[1]
            b = rgbe[i + 2] * s * TINT[2]
        out = bytearray(3)
        for c, vv in enumerate((r, g, b)):
            vv = vv * (1 - blend) + FOG[c] * blend
            key = int(vv * 4096)
            u = lut.get(key)
            if u is None:
                u = min(255, int(255.0 * ((1.0 - math.exp(-vv)) ** inv_g) + 0.5))
                lut[key] = u
            out[c] = u
        raw += out


def chunk(tag, payload):
    return (struct.pack('>I', len(payload)) + tag + payload
            + struct.pack('>I', zlib.crc32(tag + payload) & 0xFFFFFFFF))


png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(bytes(raw), 7))
png += chunk(b'IEND', b'')
open(DST, 'wb').write(png)
print(DST, round(len(png) / 1e6, 2), 'MB')
