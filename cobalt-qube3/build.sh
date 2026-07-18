#!/bin/sh
# Build Sophia's transformer for the Cobalt Qube 3 (AMD K6-2, gcc 2.95).
# On the Qube, fetch this dir + ../weights/sophia_weights.bin, then:
gcc -O2 -I. oracle_main.c nano_gpt.c -o oracle -lm
[ -x oracle ] && echo "built: ./oracle" || echo "build failed"
