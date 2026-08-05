#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
quantize_n64.py — re-quantize the Nintendo 64 LLM's SEAI Q8 weight blob to any
bit width from 2 to 8, bit-packed.

FORMATS
-------
Input  "SEAI"  : the shipped blob.  int8 weights + one float16 scale per
                 32-weight block.  See FINDINGS T0/T1.
Output "SEQn"  : identical 12-byte packed header, magic is the four ASCII bytes
                 'S','E','Q', '0'+bits.  n == 8 reproduces the SEAI content
                 with a different magic; n == 2 is ternary.

    u32 magic  'S','E','Q','0'+bits      (0x5345516E read big-endian on N64)
    u8  n_layers
    u16 n_embed          LITTLE-endian (as in SEAI)
    u8  n_heads
    u16 vocab            LITTLE-endian
    u8  ctx
    u8  em_scale_x16
    int8 embedding[vocab*n_embed]     <- carried over VERBATIM, never re-quantized
    per layer:
        packed wq, wk, wv, wo, wff1, wff2      (bits/8 bytes per weight)
        f16 LE sq, sk, sv, so, sff1, sff2      (one per 32-weight block)

BIT PACKING
-----------
Weights are packed MSB-first as a big-endian bit stream, in the same flat
row-major order as the SEAI int8 array.  Every quantization block is 32
weights, so a block is always exactly 4*bits bytes and never straddles the
block boundary.  Convenient group sizes fall out:

    bits=2  4 weights / 1 byte      bits=5  8 weights / 5 bytes
    bits=3  8 weights / 3 bytes     bits=6  4 weights / 3 bytes
    bits=4  2 weights / 1 byte      bits=8  1 weight  / 1 byte

Every width stores a plain two's-complement signed integer, so the runtime
decode is one sign-extension for all of them:  v = (int8)(code << (8-b)) >> (8-b).

TERNARY (bits=2) is the same two's-complement encoding restricted to
{-1, 0, +1}: codes 00 -> 0, 01 -> +1, 11 -> -1.  Code 10 (= -2) is never
emitted, so a ternary-specialised kernel can branch on the code instead of
sign-extending, while the generic kernel still decodes it correctly.

METHODS
-------
  maxabs (default for bits>=3)
        s = max|w| / (2^(b-1)-1);  q = clip(round(w/s)).  This is what
        train_sophia_v8.py did for Q8, applied at a narrower width.
  mse   Same form, but the scale is searched over 24 multipliers in
        [0.35, 1.0] x max|w| and the one with the lowest squared reconstruction
        error is kept.  Costs nothing at run time (the scale is stored anyway)
        and matters at 3-4 bits where clipping the tail beats keeping it.
  twn   (default for bits=2) Ternary Weight Networks, Li & Liu 2016, per block:
            delta = TAU * mean|w| ; t = sign(w) if |w|>delta else 0
            alpha = mean|w| over the survivors            <- stored as the scale

All of it is done in int8 space: the source is w = q*s with s constant across a
block, so quantizing q and multiplying the resulting scale by s is exactly
equivalent and avoids a float round trip.

Usage:
    quantize_n64.py IN.bin OUT.bin --bits 5 [--method maxabs|mse|twn] [--tau 0.7]
"""

import argparse
import struct
import sys

import numpy as np

Q_BLOCK = 32
MAGIC_SEAI = b"SEAI"

# (name, out_mult, in_mult) relative to n_embed
TENSORS = [("wq", 1, 1), ("wk", 1, 1), ("wv", 1, 1), ("wo", 1, 1),
           ("wff1", 4, 1), ("wff2", 1, 4)]


# --------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------
def pack_bits(codes, bits):
    """codes: flat uint8, each holding `bits` significant bits. MSB-first."""
    if bits == 8:
        return codes.astype(np.uint8)
    b = ((codes[:, None].astype(np.uint16) >> np.arange(bits - 1, -1, -1)) & 1)
    return np.packbits(b.astype(np.uint8).reshape(-1))


def unpack_bits(packed, n, bits):
    """inverse of pack_bits, used by the self-check."""
    if bits == 8:
        return packed[:n].astype(np.uint8)
    b = np.unpackbits(packed)[: n * bits].reshape(n, bits).astype(np.uint16)
    w = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint16)
    return (b * w).sum(axis=1).astype(np.uint8)


def sign_extend(codes, bits):
    m = 1 << (bits - 1)
    return ((codes.astype(np.int32) ^ m) - m)


# --------------------------------------------------------------------------
# quantizers.  All take q:(nblk,32) int8 and return (levels int32, alpha f32)
# where the reconstruction is levels * alpha.
# --------------------------------------------------------------------------
def q_maxabs(q, bits, _tau):
    qmax = (1 << (bits - 1)) - 1
    a = np.abs(q.astype(np.float32))
    s = a.max(axis=1, keepdims=True) / qmax
    s = np.where(s <= 0, 1e-12, s)
    lv = np.clip(np.round(q.astype(np.float32) / s), -qmax - 1, qmax).astype(np.int32)
    return lv, s[:, 0].astype(np.float32)


def q_mse(q, bits, _tau):
    qmax = (1 << (bits - 1)) - 1
    f = q.astype(np.float32)
    mx = np.abs(f).max(axis=1, keepdims=True)
    mx = np.where(mx <= 0, 1e-12, mx)
    best_err = None
    best_lv = None
    best_s = None
    for mult in np.linspace(0.35, 1.0, 24):
        s = mx * mult / qmax
        lv = np.clip(np.round(f / s), -qmax - 1, qmax)
        err = ((lv * s - f) ** 2).sum(axis=1, keepdims=True)
        if best_err is None:
            best_err, best_lv, best_s = err, lv, s
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best_lv = np.where(take, lv, best_lv)
            best_s = np.where(take, s, best_s)
    return best_lv.astype(np.int32), best_s[:, 0].astype(np.float32)


def q_twn(q, bits, tau):
    if bits != 2:
        raise SystemExit("--method twn is only defined for --bits 2")
    a = np.abs(q.astype(np.float32))
    delta = tau * a.mean(axis=1, keepdims=True)
    keep = a > delta
    lv = np.where(keep, np.sign(q.astype(np.float32)), 0.0).astype(np.int32)
    n = keep.sum(axis=1)
    alpha = np.where(n > 0, np.where(keep, a, 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)
    return lv, alpha.astype(np.float32)


METHODS = {"maxabs": q_maxabs, "mse": q_mse, "twn": q_twn}


# --------------------------------------------------------------------------
def parse_header(buf):
    if buf[:4] != MAGIC_SEAI:
        raise SystemExit("input is not a SEAI blob (magic = %r)" % buf[:4])
    _, n_layers, n_embed, n_heads, vocab, ctx, em16 = struct.unpack("<IBHBHBB", buf[:12])
    return dict(n_layers=n_layers, n_embed=n_embed, n_heads=n_heads,
                vocab=vocab, ctx=ctx, em_scale_x16=em16)


def convert(data, bits, method, tau, verbose=True, selfcheck=True):
    h = parse_header(data)
    n_embed, vocab, n_layers = h["n_embed"], h["vocab"], h["n_layers"]
    qfn = METHODS[method]

    out = bytearray()
    out += b"SEQ" + bytes([ord('0') + bits])
    out += struct.pack("<BHBHBB", n_layers, n_embed, h["n_heads"],
                       vocab, h["ctx"], h["em_scale_x16"])
    emb_bytes = vocab * n_embed
    out += data[12:12 + emb_bytes]

    pos = 12 + emb_bytes
    stats = {}
    for _li in range(n_layers):
        sizes = [(nm, (om * n_embed) * (im * n_embed)) for nm, om, im in TENSORS]
        woff = pos
        soff = pos + sum(sz for _, sz in sizes)
        packed, scales = [], []
        for nm, nelem in sizes:
            q = np.frombuffer(data, dtype=np.int8, count=nelem, offset=woff)
            nblk = nelem // Q_BLOCK
            s = np.frombuffer(data, dtype="<f2", count=nblk, offset=soff).astype(np.float32)
            woff += nelem
            soff += nblk * 2

            lv, alpha_q = qfn(q.reshape(nblk, Q_BLOCK), bits, tau)
            codes = (lv.reshape(-1).astype(np.int32) & ((1 << bits) - 1)).astype(np.uint8)
            pk = pack_bits(codes, bits)
            if selfcheck:
                back = sign_extend(unpack_bits(pk, nelem, bits), bits)
                if not np.array_equal(back, lv.reshape(-1)):
                    raise SystemExit("pack/unpack self-check FAILED on %s" % nm)
            alpha = (alpha_q * s).astype(np.float16)     # what actually ships
            packed.append(pk)
            scales.append(alpha)

            st = stats.setdefault(nm, dict(n=0, zero=0, sqerr=0.0, sqw=0.0))
            st["n"] += nelem
            st["zero"] += int((lv == 0).sum())
            # error is measured against the float weights the ROM sees today,
            # using the float16-rounded scale that will actually be stored.
            rec = lv * alpha.astype(np.float32)[:, None]
            orig = q.reshape(nblk, Q_BLOCK).astype(np.float32) * s[:, None]
            st["sqerr"] += float(((rec - orig) ** 2).sum())
            st["sqw"] += float((orig ** 2).sum())

        for p in packed:
            out += p.tobytes()
        for sc in scales:
            out += sc.astype("<f2").tobytes()
        pos = soff

    if pos != len(data):
        raise SystemExit("layout walk ended at %d, input is %d bytes" % (pos, len(data)))

    if verbose:
        hdr = "bits=%d method=%s" % (bits, method)
        if method == "twn":
            hdr += " tau=%.2f" % tau
        print(hdr)
        print("%-6s %12s %10s %8s %10s" % ("tensor", "weights", "zeros", "zero%", "NRMSE"))
        tn = tz = 0
        te = tw = 0.0
        for nm, _, _ in TENSORS:
            st = stats[nm]
            tn += st["n"]; tz += st["zero"]; te += st["sqerr"]; tw += st["sqw"]
            print("%-6s %12d %10d %7.2f%% %10.4f"
                  % (nm, st["n"], st["zero"], 100.0 * st["zero"] / st["n"],
                     (st["sqerr"] / st["sqw"]) ** 0.5))
        print("%-6s %12d %10d %7.2f%% %10.4f"
              % ("TOTAL", tn, tz, 100.0 * tz / tn, (te / tw) ** 0.5))
    return bytes(out), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--bits", type=int, default=5, choices=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--method", default=None, choices=sorted(METHODS))
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.method is None:
        a.method = "twn" if a.bits == 2 else "maxabs"

    data = open(a.infile, "rb").read()
    blob, _ = convert(data, a.bits, a.method, a.tau, verbose=not a.quiet)
    open(a.outfile, "wb").write(blob)
    if not a.quiet:
        print("in  %-40s %9d B" % (a.infile, len(data)))
        print("out %-40s %9d B   %.4fx smaller" % (a.outfile, len(blob),
                                                   len(data) / float(len(blob))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
