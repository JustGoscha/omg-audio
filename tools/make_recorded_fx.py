#!/usr/bin/env python3
"""Build the listener's recorded one-shots (both demos load them):

- demo/assets/fx_clap.ogg (K): one real palm clap — from "Palmas
  sevillanas (flamenco clapping), 160 BPM" by Santiago Sánchez Cifuentes
  (CC0; Wikimedia Commons,
  File:Palmas_sevillanas_(flamenco_clapping),_160_BPM.ogg). The clap at
  14.127 s: bright (~2.1 kHz), down 20 dB in 8 ms and 40 dB in 37 ms —
  dry, the room's answer the engine's own. 2 ms before its onset to 87
  ms after, a 20 ms fade out: the recording's next clap follows 90 ms
  after it (a 250 ms cut held both — heard as a double clap). (It was a
  synthesized clap, tools/make_fx.py: it sounded like a rim hit.)
- demo/assets/fx_call_whistle.ogg (J): a whistled call, three notes —
  "Human whistling" by TwoWings (public domain; Wikimedia Commons,
  File:Human_whistling.ogg),
  0.15–1.47 s, 10 / 60 ms fades.

Mono, 48 kHz, FLAC in Ogg as the other fx.

Usage: make_recorded_fx.py <palmas .ogg> <Human_whistling.ogg> demo/assets
"""
import subprocess
import sys


def cut(src, dst, start, dur, fade_in, fade_out):
    chain = (
        f"atrim=start={start}:duration={dur},asetpts=PTS-STARTPTS,"
        f"afade=t=in:d={fade_in},afade=t=out:st={dur - fade_out}:d={fade_out},"
        "aresample=48000"
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-ac", "1",
         "-af", chain, "-c:a", "flac", "-ar", "48000", dst],
        check=True,
    )


def main():
    palmas, whistle, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cut(palmas, f"{out}/fx_clap.ogg", 14.131, 0.089, 0.001, 0.02)
    cut(whistle, f"{out}/fx_call_whistle.ogg", 0.15, 1.32, 0.01, 0.06)


if __name__ == "__main__":
    main()
