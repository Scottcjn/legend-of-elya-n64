#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
rdram_budget.py — how large a model fits in 8 MB of N64 RDRAM, per weight width.

Everything here is derived from MEASURED numbers, not estimates:
  * text+data and non-model .bss come from the real `make base` link (FINDINGS T9)
  * framebuffer is the ROM's actual display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2)
  * the KV cache formula is the one implemented in nano_gpt.h under SGAI_KV_INT8

The brief's budget held the KV cache fixed at the CURRENT model's size.  It is
not fixed: it grows as layers*ctx*embed, so a bigger model pays for it twice.
That coupling is modelled here.

Per-layer parameters (head_dim pinned to 32, so n_heads = embed/32):
    attention  4 * E^2
    ffn        2 * E * 4E  = 8 * E^2
    total     12 * E^2
plus a tied embedding table of vocab*E, which stays int8 in every format.
"""

RDRAM = 8 * 1024 * 1024          # 8,388,608 — Expansion Pak maximum
VOCAB = 256
CTX = 128
HEAD_DIM = 32
Q_BLOCK = 32

# --- measured fixed costs, from the real ROM link (T9) ---------------------
TEXT_DATA = 148_120 + 34_668     # make base, int8 KV
BSS_OTHER = 19_844               # bss minus wbuf minus G.kv
FRAMEBUF = 320 * 240 * 2 * 2     # 307,200
HEAP_STACK = 128 * 1024          # reserve; libdragon stack sits at the top

FIXED_MEASURED = TEXT_DATA + BSS_OTHER + FRAMEBUF + HEAP_STACK
FIXED_CONSERVATIVE = 1024 * 1024  # 1.0 MB, leaves room for a bigger game


def weight_bytes(L, E, bits):
    """Packed weights + f16 block scales + int8 embedding + header."""
    p = L * 12 * E * E
    return int(p * bits / 8) + (p // Q_BLOCK) * 2 + VOCAB * E + 12


def kv_bytes(L, E, int8=True):
    """sizeof(SGAIKVCache), including the trailing pad.

    The struct is __attribute__((aligned(8))) and ends in an `int pos`, so it
    rounds up to a multiple of 8 — 589,832 rather than 589,828 for the current
    shape. Verified against a real ROM: the ares probe prints kvbytes=589832.
    """
    if int8:
        # int8 k,v  +  one float32 scale per (layer, pos, head)
        n = 2 * L * CTX * E + 2 * L * CTX * (E // HEAD_DIM) * 4 + 4
    else:
        n = 2 * L * CTX * E * 4 + 4
    return (n + 7) & ~7


def params(L, E):
    return L * 12 * E * E + VOCAB * E


def best_shape(bits, fixed, int8_kv=True, min_L=6, max_E=768):
    """Largest parameter count over SANE shapes.

    Unconstrained, this always answers "1 layer, 1152 embed", because params
    grow as L*E^2 while the KV cache grows as L*E — so the optimum is always
    maximally shallow and wide.  A 1-layer transformer is not a useful model,
    so depth is floored at min_L and width capped at max_E.  Both bounds are
    stated in the output; they are judgement, not measurement.
    """
    avail = RDRAM - fixed
    best = None
    for E in range(128, max_E + 1, 32):     # embed must be a multiple of head_dim
        for L in range(min_L, 33):
            need = weight_bytes(L, E, bits) + kv_bytes(L, E, int8_kv)
            if need > avail:
                break
            p = params(L, E)
            if best is None or p > best[0]:
                best = (p, L, E, need)
    return best, avail


CUR = params(8, 256)

for label, fixed in (("measured fixed cost %d B" % FIXED_MEASURED, FIXED_MEASURED),
                     ("conservative 1.0 MB reserve", FIXED_CONSERVATIVE)):
    print("=" * 78)
    print("RESERVE: %s   ->  %d B available for weights+KV"
          % (label, RDRAM - fixed))
    print("  (shapes constrained to L >= 6 and embed <= 768; see docstring)")
    print("%5s %12s %6s %6s %12s %12s %8s" %
          ("bits", "params", "L", "embed", "weights B", "KV B", "vs now"))
    for bits in (8, 6, 5, 4, 3, 2):
        (p, L, E, need), avail = best_shape(bits, fixed)
        print("%5d %12s %6d %6d %12s %12s %7.2fx"
              % (bits, "{:,}".format(p), L, E,
                 "{:,}".format(weight_bytes(L, E, bits)),
                 "{:,}".format(kv_bytes(L, E)), p / CUR))
    print("  -- same, but depth pinned to the current 8 layers --")
    for bits in (8, 6, 5, 4, 3, 2):
        (p, L, E, need), avail = best_shape(bits, fixed, min_L=8, max_E=768)
        best = None
        for E in range(128, 769, 32):
            need = weight_bytes(8, E, bits) + kv_bytes(8, E)
            if need <= avail and (best is None or params(8, E) > best[0]):
                best = (params(8, E), E, need)
        if best:
            print("%5d %12s %6d %6d %12s %12s %7.2fx"
                  % (bits, "{:,}".format(best[0]), 8, best[1],
                     "{:,}".format(weight_bytes(8, best[1], bits)),
                     "{:,}".format(kv_bytes(8, best[1])), best[0] / CUR))

print()
print("current shipped model: L=8 E=256 -> %s params" % "{:,}".format(CUR))
print("  int8 weights + f16 scales : %s B" % "{:,}".format(weight_bytes(8, 256, 8)))
print("  float32 KV                : %s B" % "{:,}".format(kv_bytes(8, 256, False)))
print("  int8 KV                   : %s B" % "{:,}".format(kv_bytes(8, 256, True)))
