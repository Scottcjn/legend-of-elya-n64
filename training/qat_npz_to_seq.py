#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""qat_npz_to_seq.py — convert a QAT export (npz + _meta.json) to a ROM-loadable
SEQn blob.

WHY THIS EXISTS
---------------
train_sophia_v9_qat.py's own `.bin` writer is INCOMPATIBLE with the ROM on three
counts, all silent:

  1. magic       it writes "SEAI" for every scheme, so `sgai_init` selects
                 bits=8 and decodes a 2-bit blob as int8 (and `sgai_layer_ptrs`
                 then walks 4x past the buffer).
  2. bit order   `pack_codes` is LSB-first within the byte; `matmul_t2` /
                 `matmul_qn` read MSB-first.
  3. code space  `pack_codes` stores OFFSET BINARY (`u = v - lo`, so ternary is
                 -1->01, 0->10, +1->11).  The runtime decodes TWO'S COMPLEMENT
                 (00->0, 01->+1, 11->-1).

This tool re-emits from the .npz — the same array the numpy oracle
(eval_qat.py `load_qat`) reads — so the ROM blob and the oracle are provably the
same numbers, not two independent quantizations that happen to agree.

Output is byte-for-byte the format of tools/quantize_n64.py ("SEQn").
"""
import argparse
import json
import os
import struct

import numpy as np

QB = 32
TENSORS = ("wq", "wk", "wv", "wo", "wff1", "wff2")


def pack_bits_msb(codes, bits):
    """codes: flat uint8, low `bits` bits significant.  MSB-first bit stream,
    identical to tools/quantize_n64.py:pack_bits."""
    if bits == 8:
        return codes.astype(np.uint8)
    b = ((codes[:, None].astype(np.uint16) >> np.arange(bits - 1, -1, -1)) & 1)
    return np.packbits(b.astype(np.uint8).reshape(-1))


def unpack_bits_msb(packed, n, bits):
    if bits == 8:
        return packed[:n].astype(np.uint8)
    b = np.unpackbits(packed)[: n * bits].reshape(n, bits).astype(np.uint16)
    w = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint16)
    return (b * w).sum(axis=1).astype(np.uint8)


def sign_extend(codes, bits):
    m = 1 << (bits - 1)
    return ((codes.astype(np.int32) ^ m) - m)


def convert(npz_path, meta_path, out_path):
    meta = json.load(open(meta_path))
    z = np.load(npz_path)
    bits = int(meta["bits"])
    if not (2 <= bits <= 8):
        raise SystemExit("bits=%r is not a SEQn width" % bits)
    nL = int(meta["n_layers"])
    E = int(meta["n_embed"])
    H = int(meta["n_heads"])
    ctx = int(meta["ctx"])
    x16 = int(meta["em_scale_x16"])
    vocab = 256

    emb = z["emb_q"].astype(np.int8)
    assert emb.shape == (vocab, E), emb.shape

    blob = bytearray()
    blob += b"SEQ" + bytes([ord("0") + bits])
    blob += struct.pack("<BHBHBB", nL, E, H, vocab, ctx, x16)
    blob += emb.tobytes()

    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for li in range(nL):
        codes_all, scales_all = [], []
        for nm in TENSORS:
            q = z["L%d_%s_q" % (li, nm)]
            q = q.reshape(q.shape[0], -1).astype(np.int32)   # (out, nb, 32)->(out,in)
            s = z["L%d_%s_s" % (li, nm)].astype(np.float16)  # (out, nb)
            if q.min() < lo or q.max() > hi:
                raise SystemExit("%s L%d codes out of range [%d,%d]: [%d,%d]"
                                 % (nm, li, lo, hi, q.min(), q.max()))
            assert s.shape == (q.shape[0], q.shape[1] // QB), (s.shape, q.shape)
            flat = q.reshape(-1)
            u = (flat & ((1 << bits) - 1)).astype(np.uint8)  # two's complement
            pk = pack_bits_msb(u, bits)
            back = sign_extend(unpack_bits_msb(pk, flat.size, bits), bits)
            if not np.array_equal(back, flat):
                raise SystemExit("pack/unpack self-check FAILED on L%d %s" % (li, nm))
            codes_all.append(pk)
            scales_all.append(s)
        for pk in codes_all:
            blob += pk.tobytes()
        for s in scales_all:
            blob += s.tobytes()

    with open(out_path, "wb") as fh:
        fh.write(blob)
    return len(blob), bits, dict(n_layers=nL, n_embed=E, n_heads=H, ctx=ctx, x16=x16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--meta", default=None)
    a = ap.parse_args()
    meta = a.meta or a.npz.replace(".npz", "_meta.json")
    n, bits, g = convert(a.npz, meta, a.out)
    print("wrote %s  %d B  bits=%d  %s" % (a.out, n, bits, g))


if __name__ == "__main__":
    main()
