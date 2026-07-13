#!/usr/bin/env python3
"""Render top-down schematic frames for a walkthrough JSON.

Usage: render_viz.py walk.json frames_dir [fps]
Then:  ffmpeg -framerate 30 -i frames_dir/%05d.png -i audio.wav \
         -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -shortest out.mp4
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

BG = (14, 17, 22)
ROOM_LINE = (90, 105, 125)
ROOM_LINE_ACTIVE = (120, 200, 255)
ROOM_FILL_ACTIVE = (24, 34, 46)
PATH_DIM = (55, 65, 80)
PATH_LIT = (110, 170, 220)
LISTENER = (80, 220, 255)
SOURCE_COLORS = [(255, 170, 60), (110, 224, 160), (255, 90, 158)]  # music, voice, club
TEXT = (200, 210, 225)
TEXT_DIM = (120, 130, 150)

W, H = 800, 1200
MARGIN = 70


def main():
    data = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    os.makedirs(out_dir, exist_ok=True)

    rooms = data["rooms"]
    ticks = data["ticks"]
    duration = data["duration"]

    # Fit the view to the actual architecture plus the walked path: the
    # open-air "room" spans kilometers of street and would zoom every
    # building into a sliver. Oversized rooms neither bound nor draw.
    def compact(r):
        return (r["max"][0] - r["min"][0] < 120
                and r["max"][1] - r["min"][1] < 120)

    PAD = 3.0
    xs = [c for r in rooms if compact(r) for c in (r["min"][0], r["max"][0])]
    ys = [c for r in rooms if compact(r) for c in (r["min"][1], r["max"][1])]
    xs += [tk["pos"][0] for tk in ticks]
    ys += [tk["pos"][1] for tk in ticks]
    x0, x1, y0, y1 = min(xs) - PAD, max(xs) + PAD, min(ys) - PAD, max(ys) + PAD
    scale = min((W - 2 * MARGIN) / (x1 - x0), (H - 2 * MARGIN) / (y1 - y0))
    ox = (W - (x1 - x0) * scale) / 2
    oy = (H - (y1 - y0) * scale) / 2

    def sxy(wx, wy):
        return (ox + (wx - x0) * scale, H - (oy + (wy - y0) * scale))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        font = font_small = font_big = ImageFont.load_default()

    def tick_at(t):
        """Linear interpolation between sim ticks at time t."""
        idx = min(int(t / duration * (len(ticks) - 1)), len(ticks) - 2)
        while idx > 0 and ticks[idx]["t"] > t:
            idx -= 1
        while idx < len(ticks) - 2 and ticks[idx + 1]["t"] <= t:
            idx += 1
        a, b = ticks[idx], ticks[idx + 1]
        f = (t - a["t"]) / max(b["t"] - a["t"], 1e-6)
        f = max(0.0, min(1.0, f))

        def lerp(ka, kb):
            return ka + f * (kb - ka)

        yaw_a, yaw_b = a["yaw"], b["yaw"]
        d = math.atan2(math.sin(yaw_b - yaw_a), math.cos(yaw_b - yaw_a))
        return {
            "pos": (lerp(a["pos"][0], b["pos"][0]), lerp(a["pos"][1], b["pos"][1])),
            "yaw": yaw_a + f * d,
            "room": a["room"],
            "rt60": lerp(a["rt60"], b["rt60"]),
            "routes": a["routes"],
        }

    n_frames = int(duration * fps)
    for frame in range(n_frames):
        t = frame / fps
        st = tick_at(t)
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)

        # Rooms ("Outside" is open air: thin border, no walls implied;
        # oversized open-air rects stay invisible — the HUD names them)
        for i, r in enumerate(rooms):
            if not compact(r):
                continue
            p0 = sxy(r["min"][0], r["max"][1])
            p1 = sxy(r["max"][0], r["min"][1])
            active = i == st["room"]
            outdoor = r["name"] == "Outside"
            if active:
                dr.rectangle([p0, p1], fill=ROOM_FILL_ACTIVE)
            dr.rectangle([p0, p1],
                         outline=ROOM_LINE_ACTIVE if active else ROOM_LINE,
                         width=1 if outdoor else (4 if active else 3))
            cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            # stacked storeys share a footprint: stagger their labels
            stack = sum(1 for q in rooms[:i]
                        if q["min"] == r["min"] and q["max"] == r["max"])
            label = r["name"] + (" (open air)" if outdoor else "")
            dr.text((cx, cy + stack * 32), label, font=font, anchor="mm",
                    fill=TEXT if active else TEXT_DIM)

        # Doorways: erase a gap in the shared wall
        for door in data["doors"]:
            (dx, dy), axis = door["pos"], door["axis"]
            gx, gy = sxy(dx, dy)
            if axis == 1:
                seg = [(gx - 0.55 * scale, gy), (gx + 0.55 * scale, gy)]
            else:
                seg = [(gx, gy - 0.55 * scale), (gx, gy + 0.55 * scale)]
            dr.line(seg, fill=BG, width=8)
            dr.line(seg, fill=(60, 70, 85), width=2)

        # Path: full (dim, dotted) + traveled (lit)
        pts = [sxy(*tk["pos"]) for tk in ticks]
        for i in range(0, len(pts) - 1, 2):
            dr.line([pts[i], pts[i + 1]], fill=PATH_DIM, width=2)
        done = [sxy(*tk["pos"]) for tk in ticks if tk["t"] <= t]
        if len(done) > 1:
            dr.line(done, fill=PATH_LIT, width=3)

        # Fixed sources: pulsing dot + routed path (through doorways) to
        # listener. Dynamic slots (thrown balls, cars) never play in the
        # scripted walkthrough — skip their parked placeholders.
        for si, sdef in enumerate(data["sources"]):
            if sdef["name"].startswith(("ball", "car")):
                continue
            col = SOURCE_COLORS[si % len(SOURCE_COLORS)]
            route = [(p[0], p[1]) for p in st["routes"][si]]
            rpts = [sxy(*p) for p in route[:-1]] + [sxy(*st["pos"])]
            for a_, b_ in zip(rpts, rpts[1:]):
                seglen = math.hypot(b_[0] - a_[0], b_[1] - a_[1])
                ndash = max(int(seglen / 14), 1)
                for k in range(0, ndash, 2):
                    f0, f1 = k / ndash, min((k + 1) / ndash, 1.0)
                    dr.line([(a_[0] + (b_[0] - a_[0]) * f0, a_[1] + (b_[1] - a_[1]) * f0),
                             (a_[0] + (b_[0] - a_[0]) * f1, a_[1] + (b_[1] - a_[1]) * f1)],
                            fill=tuple(c // 2 for c in col), width=2)
            sx_, sy_ = sxy(*sdef["pos"])
            pulse = 0.35 + 0.35 * (0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.6 + si))
            rr = scale * pulse
            dr.ellipse([sx_ - rr, sy_ - rr, sx_ + rr, sy_ + rr], outline=col, width=2)
            dr.ellipse([sx_ - 8, sy_ - 8, sx_ + 8, sy_ + 8], fill=col)
            dr.text((sx_, sy_ - 26), sdef["name"], font=font_small, anchor="mm", fill=col)

        # Listener: oriented triangle (screen y is flipped → negate yaw)
        lx_, ly_ = sxy(*st["pos"])
        yaw = -st["yaw"]
        size = 0.42 * scale
        tri = []
        for ang, rad in [(0, 1.5 * size), (2.5, size), (-2.5, size)]:
            a = yaw + ang
            tri.append((lx_ + rad * math.cos(a), ly_ + rad * math.sin(a)))
        dr.polygon(tri, fill=LISTENER)
        dr.text((lx_, ly_ + 30), "you", font=font_small, anchor="mm", fill=LISTENER)

        # HUD
        dr.text((30, 26), "omg-audio · walkthrough", font=font_big, fill=TEXT)
        dr.text((30, 72), f"room: {rooms[st['room']]['name']}", font=font, fill=TEXT)
        dr.text((30, 106), f"measured RT60 (mid): {st['rt60']:.2f} s", font=font, fill=TEXT_DIM)
        dr.text((W - 30, 26), f"{t:5.1f} s", font=font_big, anchor="ra", fill=TEXT_DIM)

        # RT60 bar
        bar_w = (W - 60) * min(st["rt60"] / 3.0, 1.0)
        dr.rectangle([30, 140, 30 + bar_w, 148], fill=ROOM_LINE_ACTIVE)
        dr.rectangle([30, 140, W - 30, 148], outline=ROOM_LINE, width=1)

        img.save(f"{out_dir}/{frame:05d}.png")

    print(f"wrote {n_frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
