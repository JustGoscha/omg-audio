#!/bin/sh
# Build the wasm module and stage it into web/.
set -e
cd "$(dirname "$0")/.."
cargo build --release -p omg-web --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/omg_web.wasm web/omg_web.wasm
ls -la web/omg_web.wasm
