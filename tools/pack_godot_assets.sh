#!/bin/sh
# Copy assets/ into the Godot project as godot/assets/ for an export
# (natively the demo reads them from ../assets): plain files ("keep"
# importer), not imported. The export scripts remove it afterwards.
set -e
cd "$(dirname "$0")/.."
rm -rf godot/assets
mkdir -p godot/assets
(cd assets && find . -type f ! -name '*.bin' ! -name '.*' | while read -r f; do
  mkdir -p "../godot/assets/$(dirname "$f")"
  cp "$f" "../godot/assets/$f"
  printf '[remap]\n\nimporter="keep"\n' > "../godot/assets/$f.import"
done)
