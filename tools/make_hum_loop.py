#!/usr/bin/env python3
"""Build demo/assets/hum48.ogg: the dome's stage voice, a woman humming a
folk tune, quietly — "Woman humming a nondescript tune" by PSsoundproject
(CC0; Wikimedia Commons, File:Woman_humming_a_nondescript_tune.wav, the
same take as freesound.org/people/PSsoundproject/sounds/183565/). Dry —
recorded close on an iPod touch, a phrase's end falls 20 dB in ~60 ms
(the sung CC0 takes found all carried their room, 140–390 ms) — which
the engine needs: the dome's echo is its own.

The take hums from 0.2 s to 18.5 s, its breaths at 5, 9.5 and 14 s, then
a second of quiet: the loop is the whole take, its seam in that quiet.
The iPod's hiss (−50 dBFS, 36 dB under the humming) is turned down by
an FFT denoise and, in the pauses, a gentle gate (−20 dB at most), and
kept out of the bass and the top (80 Hz, 12 kHz); 30 ms fades either side of the
seam; 48 kHz Opus as the other assets.

Usage: make_hum_loop.py <Woman_humming_a_nondescript_tune.wav> demo/assets/hum48.ogg
"""
import subprocess
import sys

START, END = 0.12, 19.6  # s: from the quiet before the first phrase into the quiet after the last


def main():
    src, dst = sys.argv[1], sys.argv[2]
    chain = (
        f"atrim=start={START}:end={END},asetpts=PTS-STARTPTS,"
        "highpass=f=80,lowpass=f=12000,afftdn=nr=20:nf=-46:tn=1,"
        "agate=threshold=0.01:ratio=3:range=0.1:attack=5:release=200,"
        f"afade=t=in:d=0.03,afade=t=out:st={END - START - 0.03}:d=0.03,"
        "aresample=48000"
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-ac", "1",
         "-af", chain, "-c:a", "libopus", "-b:a", "96k", "-ar", "48000", dst],
        check=True,
    )


if __name__ == "__main__":
    main()
