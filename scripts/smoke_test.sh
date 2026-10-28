#!/usr/bin/env bash
# Every command at toy size on one GPU: 8 training pairs, 20 optimization steps, 4 images with the adapter and 4 from
# the released student. A few minutes plus the model downloads. The adapter is not useful at this size; the run checks
# that the installation and the models work.
#
#   scripts/smoke_test.sh [flux2-klein|sd-turbo|z-image-turbo] [output directory]
set -euo pipefail
MODEL=${1:-sd-turbo}
OUT=${2:-runs/smoke-$MODEL}
mkdir -p "$OUT"
printf 'red panda\nhare\n' > "$OUT/classes.txt"

fast-and-useful targets --model "$MODEL" --classes "$OUT/classes.txt" --per-class 4 --part-size 4 --out "$OUT/targets"
fast-and-useful train --targets "$OUT/targets" --out "$OUT/adapter" --steps 20 --eval-every 10
fast-and-useful generate --adapter "$OUT/adapter" --classes "$OUT/classes.txt" --per-class 2 --out "$OUT/adapted"
fast-and-useful generate --model "$MODEL" --classes "$OUT/classes.txt" --per-class 2 --out "$OUT/released"
echo "smoke test passed: images in $OUT/adapted and $OUT/released"
