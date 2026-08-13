#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""blob_layout.py — walk a SEAI/SEQn weight blob and print what it actually is.

Written to settle one question by arithmetic rather than by reading the README:
the shipped blob is 2,031,628 bytes, which factors as

    12 + 65,536 + 8 * 196,608 + 8 * 49,152

and the "8" in that expression is the number of TRANSFORMER LAYERS, not a
number of mixture-of-experts experts.  The blob's own 12-byte header carries
n_layers in byte 4, the per-layer figure decomposes exactly into the six
tensors the engine reads (wq wk wv wo wff1 wff2), and the walk consumes the
file to the last byte.  A mixture would have to put the expert count
somewhere; there is nowhere for it to be.

Usage:  blob_layout.py filesystem/sophia_weights.bin
"""
import struct
import sys

# (name, out_mult, in_mult) in units of n_embed — the layout the ROM reads,
# matching TENSORS in tools/quantize_n64.py and sgai_layer_ptrs() in nano_gpt.c.
TENSORS = [("wq", 1, 1), ("wk", 1, 1), ("wv", 1, 1), ("wo", 1, 1),
           ("wff1", 4, 1), ("wff2", 1, 4)]
Q_BLOCK = 32


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "filesystem/sophia_weights.bin"
    data = open(path, "rb").read()
    magic = data[:4]
    if magic[:3] != b"SEQ" and magic != b"SEAI":
        print("not a SEAI/SEQn blob: magic = %r" % magic)
        return 2
    bits = 8 if magic == b"SEAI" else (magic[3] - ord("0"))

    (_m, n_layers, n_embed, n_heads, vocab, ctx,
     em16) = struct.unpack("<IBHBHBB", data[:12])

    print("file            %s" % path)
    print("size            %d B" % len(data))
    print("magic           %s   -> %d-bit weights" % (magic.decode(), bits))
    print("n_layers        %d      <- byte 4 of the header" % n_layers)
    print("n_embed         %d" % n_embed)
    print("n_heads         %d" % n_heads)
    print("vocab           %d" % vocab)
    print("ctx             %d" % ctx)
    print("em_scale_x16    %d" % em16)
    print()

    pos = 12
    print("%-8s %10s  %s" % ("region", "bytes", "at"))
    print("%-8s %10d  %d" % ("header", 12, 0))
    emb = vocab * n_embed
    print("%-8s %10d  %d   (int8, %d x %d)" % ("embed", emb, pos, vocab, n_embed))
    pos += emb

    lay_w = lay_s = 0
    for nm, om, im in TENSORS:
        n = (om * n_embed) * (im * n_embed)
        lay_w += (n * bits) // 8
        lay_s += (n // Q_BLOCK) * 2
    for li in range(n_layers):
        if li == 0:
            for nm, om, im in TENSORS:
                n = (om * n_embed) * (im * n_embed)
                print("  %-6s %10d  %s   (%d x %d weights @ %d bit)"
                      % (nm, (n * bits) // 8, "-", om * n_embed, im * n_embed, bits))
            print("  %-6s %10d  %s   (float16, one per %d-weight block)"
                  % ("scales", lay_s, "-", Q_BLOCK))
        print("%-8s %10d  %d" % ("layer %d" % li, lay_w + lay_s, pos))
        pos += lay_w + lay_s

    params = emb + n_layers * sum((om * n_embed) * (im * n_embed)
                                  for _, om, im in TENSORS)
    print()
    print("walk ended at   %d" % pos)
    print("file size       %d" % len(data))
    print("VERDICT         %s" % ("EXACT — the layout accounts for every byte"
                                  if pos == len(data) else
                                  "MISMATCH — %+d B" % (pos - len(data))))
    print("parameters      %d  (embedding %d + %d layers x %d)"
          % (params, emb, n_layers,
             sum((om * n_embed) * (im * n_embed) for _, om, im in TENSORS)))
    print("decomposition   %d = 12 + %d + %d*%d + %d*%d"
          % (len(data), emb, n_layers, lay_w, n_layers, lay_s))
    return 0 if pos == len(data) else 1


if __name__ == "__main__":
    sys.exit(main())
