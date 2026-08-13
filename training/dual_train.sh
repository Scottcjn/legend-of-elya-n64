#!/bin/sh
# SPDX-License-Identifier: MIT
#
# dual_train.sh — train the members of the N64 dual-processor consensus.
#
# Three configurations, one recipe.  The recipe is the shipped model's
# (training/README.md): v7 corpus, reps 800/800/400, 9000 steps, paraphrase
# holdout on.  Only --quant and --n-layers vary, so a ternary member and the
# int8 member of the same seed differ ONLY in the resolution of a weight --
# which is the whole point of the experiment.
#
#   tern8   ternary_bn, 8 layers   what the ROM ships, on the MIPS core
#   int88   int8,       8 layers   the reference int8 pair.  DOES NOT FIT:
#                                  2,031,628 + 6,750,220 = 8,781,848 B of
#                                  weights alone, against 8,388,608 B of RDRAM.
#   int84   int8,       4 layers   the int8 pair that fits, for the RSP
#
# --micro-batch 16 is memory, not arithmetic (training/test_microbatch_equiv.py).
#
#   sh training/dual_train.sh 1337 7 23 101
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-mix_out}
STEPS=${STEPS:-9000}
export PYTORCH_ALLOC_CONF=expandable_segments:True

for seed in "$@"; do
    for cfg in "ternary_bn 8 tern8" "int8 8 int88" "int8 4 int84"; do
        set -- $cfg
        quant=$1; nl=$2; name=$3
        tag="${name}_s${seed}"
        if [ -f "$OUT/${tag}_meta.json" ]; then
            echo "== $tag already done, skipping"
            continue
        fi
        echo "== $tag  quant=$quant layers=$nl seed=$seed steps=$STEPS"
        python3 training/train_mix.py \
            --quant "$quant" --n-layers "$nl" --seed "$seed" \
            --reps 800,800,400 --steps "$STEPS" --micro-batch 16 \
            --tag "$tag" --outdir "$OUT" \
            --eval-interval 1500 --log-interval 1500
        python3 training/qat_npz_to_seq.py "$OUT/$tag.npz" "$OUT/$tag.seq"
    done
done
echo "ALL TRAINING DONE"
