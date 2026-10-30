#!/usr/bin/env python3
"""Build demo/assets/radio48.ogg as a seamless loop: the bunker radio's
clip (florianreichelt/423446, CC0) as first shipped faded in over its
first 0.3 s and out over its last 0.4 s, so every loop dropped to
silence for a moment. This drops both fades and crossfades the music
just before the fade-out into the music just after the fade-in
(equal power, 0.9 s), and cuts the loop to whole Opus frames (with the
encoder's 312-sample pre-skip) so no decoder pads its end.

Usage: make_radio_loop.py <first shipped clip> demo/assets/radio48.ogg
    (git show 7662a81:demo/assets/radio48.ogg > radio_orig.ogg)
"""
import subprocess
import sys

FS = 48000
HEAD = 14400      # after the fade-in (0.3 s)
XFADE = 43200     # 0.9 s
PRE_SKIP = 312    # libopus' encoder delay at 48 kHz
FRAME = 960       # 20 ms Opus frames
FRAMES = 707      # the loop: FRAMES * FRAME - PRE_SKIP samples (14.13 s)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = FRAMES * FRAME - PRE_SKIP
    mid0 = HEAD + XFADE
    end = HEAD + n  # where the tail starts: the mid runs up to it
    graph = (
        f"[0]aresample={FS},asplit=3[a][b][c];"
        f"[a]atrim=start_sample={HEAD}:end_sample={mid0},asetpts=PTS-STARTPTS[head];"
        f"[b]atrim=start_sample={end}:end_sample={end + XFADE},asetpts=PTS-STARTPTS[tail];"
        f"[c]atrim=start_sample={mid0}:end_sample={end},asetpts=PTS-STARTPTS[mid];"
        f"[tail][head]acrossfade=d={XFADE / FS}:c1=qsin:c2=qsin[xf];"
        "[xf][mid]concat=n=2:v=0:a=1[out]"
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
         "-filter_complex", graph, "-map", "[out]",
         "-c:a", "libopus", "-b:a", "96k", "-ar", str(FS), dst],
        check=True)
    print(f"{dst}: {n / FS:.2f} s loop ({FRAMES} Opus frames)")


if __name__ == "__main__":
    main()
