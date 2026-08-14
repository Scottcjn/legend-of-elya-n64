#!/bin/sh
# SPDX-License-Identifier: MIT
#
# dual_sweep.sh — score every pairing that answers a different question.
#
#   A  ternary8 + int8-4, SAME seed      what the ROM actually runs
#   B  ternary8 + int8-8, SAME seed      the pair that does not fit in RDRAM,
#                                        so the cost of shrinking the RSP arm
#                                        to 4 layers is a measurement
#   C  ternary8 + int8-4, DIFFERENT seed is the diversity the FORMAT or the
#                                        seed?  If A and C score the same it
#                                        is the format.
#   D  ternary8 + ternary8, diff seed    the same-format control.  If D scores
#                                        as well as A, the format claim is
#                                        wrong and this is just an ensemble.
#
#   sh training/dual_sweep.sh
set -e
cd "$(dirname "$0")/.."
OUT=docs/dual_logs
M=mix_out
mkdir -p "$OUT"

run() {   # tag  tern  other
    if [ -f "$OUT/sweep_$1.json" ]; then echo "== $1 done, skipping"; return; fi
    echo "== $1  $2  $3"
    python3 -u training/dual_eval.py --tern "$M/$2" --int8 "$M/$3" \
        --skip-keys --json "$OUT/sweep_$1.json" > "$OUT/sweep_$1.log" 2>&1
    grep -E "^  (single|single1|raw|shift1|norm) " "$OUT/sweep_$1.log" || true
}

for s in 1337 7 23 101; do
    run "A_int84_s$s"  "tern8_s$s.seq" "int84_s$s.seq"
done
for s in 1337 7 23 101; do
    run "B_int88_s$s"  "tern8_s$s.seq" "int88_s$s.seq"
done
# C: rotate the int8 partner one seed along
run "C_cross_1337_7"   "tern8_s1337.seq" "int84_s7.seq"
run "C_cross_7_23"     "tern8_s7.seq"    "int84_s23.seq"
run "C_cross_23_101"   "tern8_s23.seq"   "int84_s101.seq"
run "C_cross_101_1337" "tern8_s101.seq"  "int84_s1337.seq"
# D: same format, different seed
run "D_terntern_1337_7"   "tern8_s1337.seq" "tern8_s7.seq"
run "D_terntern_7_23"     "tern8_s7.seq"    "tern8_s23.seq"
run "D_terntern_23_101"   "tern8_s23.seq"   "tern8_s101.seq"
run "D_terntern_101_1337" "tern8_s101.seq"  "tern8_s1337.seq"
echo "SWEEP DONE"
