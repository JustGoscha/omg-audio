#!/bin/sh
# Build the GDExtension and stage it into the Godot project.
set -e
cd "$(dirname "$0")/.."
cargo build --release -p omg-godot
mkdir -p godot/bin
case "$(uname -s)" in
  Darwin) cp target/release/libomg_godot.dylib godot/bin/ ;;
  Linux)  cp target/release/libomg_godot.so godot/bin/ ;;
  *)      cp target/release/omg_godot.dll godot/bin/ ;;
esac
ls -la godot/bin/
