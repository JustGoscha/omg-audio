#!/bin/sh
# Fetch the CC0 Poly Haven textures the Godot demo dresses its level with
# (1k JPG: diffuse, OpenGL normal, AO/rough/metal packed). They are
# committed under assets/textures/ (re-encoded smaller, ~5 MB); this
# script documents where they came from and refreshes them. Licence: CC0 (https://polyhaven.com/license).
set -e
cd "$(dirname "$0")/../assets/textures"
new=""
for n in castle_brick_02_red white_sandstone_bricks_03 plastered_wall_04 \
         brushed_concrete concrete_block_wall roof_slates_02 cobblestone_floor_08 \
         leafy_grass asphalt_02 brown_planks_05 herringbone_parquet wood_floor \
         marble_tiles; do
  for m in diff nor_gl arm; do
    f="${n}_${m}_1k.jpg"
    [ -f "$f" ] && continue
    url="https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/$n/$f"
    # Poly Haven names the diffuse map "diff" or "diffuse" per asset
    curl -sfL -o "$f" "$url" || \
      curl -sfL -o "$f" "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/$n/${n}_diffuse_1k.jpg"
    new="$new $f"
  done
done
# keep the repo light: re-encode what was just fetched (diffuse q5,
# normal q6, ARM downscaled to 512 px)
for f in $new; do
  case "$f" in
    *_arm_1k.jpg) opt="-vf scale=512:512 -q:v 6" ;;
    *_nor_gl_1k.jpg) opt="-q:v 6" ;;
    *) opt="-q:v 5" ;;
  esac
  ffmpeg -loglevel error -y -i "$f" $opt "t_$f" && mv "t_$f" "$f"
done
du -sh .
