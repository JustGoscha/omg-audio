#!/bin/sh
# The final render of an engine change: the Godot demo's tour of the whole
# level (demo/godot/demo/record.gd, scenario `tour`: seeded, every event at
# a fixed point of the walk), recorded as Godot plays it, read back
# (analyze_rec: level jumps and treble per phase), and put together as a
# video with its sound and a spectrogram of the whole walk.
#
#   sh tools/record_tour.sh <out dir> [engine: v6.2] [video fps: 10]
#
# Writes <out>/tour_<engine>.wav (+ .wav.csv, .frames.csv, .tour.csv: when
# each step began), .txt (the analysis), .m4a, .mp4 (the window and the
# sound) and _spectrogram.png.
set -e
cd "$(dirname "$0")/.."
mkdir -p "$1"
out="$(cd "$1" && pwd)"
engine="${2:-v6.2}"
fps="${3:-10}"
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
cargo build -q --release -p omg-web --example analyze_rec
GODOT=/Applications/Godot.app/Contents/MacOS/Godot
name="tour_${engine}"
rm -rf "$out/${name}_frames"
"$GODOT" --path demo/godot -- --record="$out" --scenario=tour --engine="$engine" --video="$fps" 2>&1 | grep -E "recorded|ERROR" || true
./target/release/examples/analyze_rec "$out/$name" > "$out/$name.txt"
if command -v ffmpeg >/dev/null; then
  ffmpeg -loglevel error -y -framerate "$fps" -i "$out/${name}_frames/%06d.jpg" -i "$out/$name.wav" \
    -c:v libx264 -pix_fmt yuv420p -crf 23 -c:a aac -b:a 192k -shortest "$out/$name.mp4"
  ffmpeg -loglevel error -y -i "$out/$name.wav" -lavfi "showspectrumpic=s=2400x800:legend=1:fscale=log:color=intensity" "$out/${name}_spectrogram.png"
  ffmpeg -loglevel error -y -i "$out/$name.wav" -c:a aac -b:a 192k "$out/$name.m4a"
  rm -rf "$out/${name}_frames"
fi
ls -la "$out" | grep "$name"
