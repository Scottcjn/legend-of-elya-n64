#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
quantize_ternary_n64.py — 2-bit packed ternary ("SEQ2") export for the N64 LLM.

This is a thin front end for `quantize_n64.py --bits 2 --method twn`; the whole
2..8-bit family shares one implementation so there is only one packer to get
right.  It is kept as its own entry point because ternary is the case with a
specific design argument attached, recorded here:

WHY 2-BIT PACKED AND NOT AN INDEX STREAM
----------------------------------------
The sibling Game Boy and Genesis ports use an index-stream ternary format: one
byte of column index per non-zero weight.  That only pays on a sparse tensor.
The shipped N64 Q8 weights are 98.1 % non-zero (measured), so an index stream
would make the file BIGGER, not smaller.  It also cannot work here at all:
wff2 has an input dimension of 1024 and the index byte tops out at 255.
A fixed 2-bit packing needs no indices and is unconditionally 4x smaller than
int8: 6,750,220 -> 2,031,628 bytes, 3.3226x including the unchanged float16
block scales and the int8 embedding table.

WHY IT IS NOT SHIPPED
---------------------
Post-training ternarization destroys this model.  Measured with the ROM's own
C engine over 3 prompts and 512 teacher-forced tokens, at TWN thresholds 0.5,
0.6, 0.7 and 0.8: teacher-forced top-1 agreement 30-36 %, free-running text
degenerating to "mamamamamayayayayay".  No threshold rescues it; the weight
reconstruction error is ~47 % of the weight magnitude.
The failure is a property of POST-TRAINING conversion, not of ternary weights.
With the quantizer inside the training loop (`train_sophia_v8.py --qat-bits 2`,
or `tools/qat_finetune.py`) fluent English comes back.  See FINDINGS T3/T7/T14.

Usage:
    quantize_ternary_n64.py IN.bin OUT.bin [--tau 0.7]
"""

import argparse
import sys

from quantize_n64 import convert


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--tau", type=float, default=0.7,
                    help="TWN threshold multiplier: delta = TAU * mean(|w|) per "
                         "32-weight block. 0.7 is Li & Liu's standard value.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    data = open(a.infile, "rb").read()
    blob, _ = convert(data, bits=2, method="twn", tau=a.tau, verbose=not a.quiet)
    open(a.outfile, "wb").write(blob)
    if not a.quiet:
        print("in  %-40s %9d B" % (a.infile, len(data)))
        print("out %-40s %9d B   %.4fx smaller"
              % (a.outfile, len(blob), len(data) / float(len(blob))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
