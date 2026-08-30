#!/usr/bin/env bash
# One ternary expert per NPC (moe_shards.NPC_EXPERTS), 4 layers x 256d each,
# which is the 1,048,588 B blob src/expert_cache.c streams at a uniform stride.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/scott/legend-of-elya-n64
mkdir -p npc_out
for npc in sophia aldric brunhild; do
  echo "=== $npc ==="
  python3 train_sophia_v9_qat.py --corpus v7 --shard "npc:$npc" \
      --quant ternary --n-layers 4 --steps 12000 \
      --eval-interval 3000 --log-interval 3000 \
      --outdir npc_out --tag "$npc" 2>&1 | tail -3
done
echo "ALL NPC EXPERTS DONE"
