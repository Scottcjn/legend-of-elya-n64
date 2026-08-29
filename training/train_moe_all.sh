#!/usr/bin/env bash
# Train one ternary expert per shard for the N64 streaming-MoE bank.
# 4 layers x 256d each = 3,211,264 params -> a 1,048,588 B SEQ2 blob, which is
# the uniform stride src/expert_cache.c wants (F-R027 measured a 160KB slice
# hiding; a 1MB expert is the real unit).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/scott/legend-of-elya-n64
for shard in identity dungeon hardware rustchain lore; do
  echo "=== $shard ==="
  python3 train_sophia_v9_qat.py --corpus v7 --shard "$shard" \
      --quant ternary --n-layers 4 --steps 12000 \
      --eval-interval 2000 --log-interval 2000 \
      --outdir moe_out --tag "$shard" 2>&1 | tail -4
done
echo "ALL EXPERTS DONE"
