#!/bin/sh
# Record the Godot demo's scenarios as Godot plays them (demo/godot/demo/
# record.gd) and read each back (analyze_rec): fly, car and door, per
# engine, with the steady tone (artifacts are measurable) and with the
# demo's own sounds (to listen to). Writes <out>/<scenario>_<engine>[_tone]
# .wav/.wav.csv/.frames.csv, .txt (the analysis) and .m4a (to listen).
#
#   sh tools/record_runs.sh <out dir> [engines: v3 v6] [scenarios: fly car door]
set -e
cd "$(dirname "$0")/.."
out="$1"
engines="${2:-v3 v6}"
scenarios="${3:-fly car door}"
mkdir -p "$out"
# absolute: Godot runs with demo/godot as its working directory
out="$(cd "$out" && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cargo build -q --release -p omg-web --example analyze_rec
GODOT=/Applications/Godot.app/Contents/MacOS/Godot
for s in $scenarios; do
  for e in $engines; do
    for tone in "--tone" ""; do
      name="${s}_${e}${tone:+_tone}"
      "$GODOT" --path demo/godot -- --record="$out" --scenario="$s" --engine="$e" $tone 2>&1 | grep -E "recorded|ERROR" || true
      ./target/release/examples/analyze_rec "$out/$name" > "$out/$name.txt"
      if command -v afconvert >/dev/null; then
        afconvert -f m4af -d aac -b 192000 "$out/$name.wav" "$out/$name.m4a" 2>/dev/null || true
      fi
    done
  done
done
