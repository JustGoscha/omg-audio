#!/bin/sh
# Build the Godot demo as a downloadable macOS app (universal: Apple
# silicon + Intel), ad-hoc signed, zipped into dist/ for a release.
#
#   sh tools/build_godot_mac.sh
#
# Needs: rustup targets aarch64-apple-darwin and x86_64-apple-darwin, Godot
# at $GODOT (default: the macOS app). The macOS export template is fetched
# on first use (that part of the official template archive only).
set -e
cd "$(dirname "$0")/.."
GODOT=${GODOT:-/Applications/Godot.app/Contents/MacOS/Godot}
export PATH="$HOME/.cargo/bin:$PATH"
version=$("$GODOT" --version | sed -E 's/^([0-9]+\.[0-9]+(\.[0-9]+)?)\.stable.*/\1/')
tdir="$HOME/Library/Application Support/Godot/export_templates/$version.stable"
if [ ! -f "$tdir/macos.zip" ]; then
  echo "fetching Godot $version macOS export template → $tdir"
  python3 -I tools/fetch_godot_templates.py "$version" "$tdir" macos.zip
fi
# 1. the GDExtension for both architectures, glued into one dylib
for t in aarch64-apple-darwin x86_64-apple-darwin; do
  GODOT4_BIN="$GODOT" cargo build -p omg-godot --release --target $t
done
mkdir -p demo/godot/bin
rm -f demo/godot/bin/libomg_godot.dylib # macOS kills a process whose dylib is overwritten in place
lipo -create target/aarch64-apple-darwin/release/libomg_godot.dylib \
  target/x86_64-apple-darwin/release/libomg_godot.dylib -output demo/godot/bin/libomg_godot.dylib
# 2. the assets, into the project for the export (as build_godot_web.sh)
sh tools/pack_godot_assets.sh
# 3. import, export (Godot zips the .app)
mkdir -p dist
rm -f dist/omg-audio-godot-macos.zip
"$GODOT" --headless --path demo/godot --import >/dev/null 2>&1 || true
"$GODOT" --headless --path demo/godot --export-release "macOS" ../../dist/omg-audio-godot-macos.zip
rm -rf demo/godot/assets
ls -la dist/omg-audio-godot-macos.zip
