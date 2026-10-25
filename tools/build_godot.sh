#!/bin/sh
# Build the GDExtension and stage it into the Godot project.
set -e
cd "$(dirname "$0")/.."
cargo build --release -p omg-godot
mkdir -p godot/bin
case "$(uname -s)" in
  # a fresh file, not cp over the old one: macOS caches a loaded
  # library's code signature per file and SIGKILLs (exit 137) a process
  # that loads a dylib overwritten in place
  Darwin) rm -f godot/bin/libomg_godot.dylib && cp target/release/libomg_godot.dylib godot/bin/ ;;
  Linux)  cp target/release/libomg_godot.so godot/bin/ ;;
  *)      cp target/release/omg_godot.dll godot/bin/ ;;
esac
ls -la godot/bin/
