#!/bin/sh
# Build the wasm module and stage it into web/.
set -e
cd "$(dirname "$0")/.."
# simd128: baseline in Chrome 91+/Firefox 89+/Safari 16.4+ — the HRIR
# convolutions (point taps + speaker decode) are written to vectorize.
RUSTFLAGS="-C target-feature=+simd128" \
    cargo build --release -p omg-web --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/omg_web.wasm web/omg_web.wasm
ls -la web/omg_web.wasm
