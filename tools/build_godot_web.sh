#!/bin/sh
# Build the Godot demo for the web: the GDExtension as an Emscripten side
# module (wasm32-unknown-emscripten, threads), then Godot's Web export into
# godot-web/ (served next to web/ by the Pages site).
#
#   sh tools/build_godot_web.sh
#
# Needs: rustup nightly with rust-src and the wasm32-unknown-emscripten
# target; emcc on PATH — ideally the Emscripten version Godot's own web
# templates were built with (4.0.20 for Godot 4.7: `emsdk install 4.0.20`;
# set EMSDK to an activated emsdk); Godot at $GODOT (default: the macOS app).
# The Web export templates are fetched on first use (web parts only, ~90 MB).
set -e
cd "$(dirname "$0")/.."
GODOT=${GODOT:-/Applications/Godot.app/Contents/MacOS/Godot}
export PATH="$HOME/.cargo/bin:$PATH"
[ -n "$EMSDK" ] && . "$EMSDK/emsdk_env.sh" >/dev/null 2>&1

version=$("$GODOT" --version | sed -E 's/^([0-9]+\.[0-9]+(\.[0-9]+)?)\.stable.*/\1/')
echo "godot $version · $(emcc --version | head -1)"

# 1. export templates (web only) where Godot looks for them
case "$(uname -s)" in
  Darwin) tdir="$HOME/Library/Application Support/Godot/export_templates/$version.stable" ;;
  *)      tdir="${XDG_DATA_HOME:-$HOME/.local/share}/godot/export_templates/$version.stable" ;;
esac
if [ ! -f "$tdir/web_dlink_release.zip" ]; then
  echo "fetching Godot $version web export templates → $tdir"
  python3 -I tools/fetch_godot_web_templates.py "$version" "$tdir"
fi

# 2. the GDExtension: threads (matches the export's thread support),
# panic=abort (gdext cannot catch panics on wasm; no unwinding tables to
# mismatch Godot's exception model), wasm SIMD (omg-dsp's simd128 HRIR
# kernels), 32-bit bindings from this Godot
GODOT4_BIN="$GODOT" \
RUSTFLAGS="-C link-args=-pthread -C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-args=-sSIDE_MODULE=2 -C llvm-args=-enable-emscripten-cxx-exceptions=0 -C panic=abort \
-Z default-visibility=hidden -Z link-native-libraries=no" \
cargo +nightly build -Zbuild-std=std,panic_abort --target wasm32-unknown-emscripten \
  -p omg-godot --release
mkdir -p godot/bin
cp target/wasm32-unknown-emscripten/release/omg_godot.wasm godot/bin/

# 3. the assets, into the project for the export (natively the demo reads
# them from ../assets): kept as plain files ("keep" importer), not imported
rm -rf godot/assets
mkdir -p godot/assets
(cd assets && find . -type f ! -name '*.bin' ! -name '.*' | while read -r f; do
  mkdir -p "../godot/assets/$(dirname "$f")"
  cp "$f" "../godot/assets/$f"
  printf '[remap]\n\nimporter="keep"\n' > "../godot/assets/$f.import"
done)

# 4. import, export
rm -rf godot-web
mkdir -p godot-web
"$GODOT" --headless --path godot --import >/dev/null 2>&1 || true
"$GODOT" --headless --path godot --export-release "Web" ../godot-web/index.html
rm -rf godot/assets
du -sh godot-web
ls -la godot-web
