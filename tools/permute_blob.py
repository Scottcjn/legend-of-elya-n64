#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""permute_blob.py — pre-permute a SEQn expert blob into the RSP's lane order.

WHY. sgai_init_ex() permutes every weight tensor into the RSP's 8-blocks-per-
lane order at load time. Measured (F-R030/F-R031): **161 ms per expert swap**,
paid on every NPC change, and prefetch cannot hide it because it is CPU work
rather than transfer. Doing it offline deletes it.

THE TRAP THIS TOOL EXISTS TO DEFUSE. The permutation is
    new[i*8 + b] = old[b*G + i],   G = 32 (int8) or 8 (ternary)
For ternary G = 8, so it is an 8x8 transpose — an INVOLUTION. Permuting an
already-permuted ternary blob returns it to row-major; the RSP then computes on
the wrong layout **and still emits fluent text**, while the host numpy oracle
and the CPU kernels (which want row-major) keep agreeing with each other.
Verified empirically for in_dim 256/1024 x out_dim 256/1024: double-permute is
byte-identical to the original.

So this tool does not rely on anyone remembering the rule:
  1. Output carries a DIFFERENT MAGIC, "SEP"+bits instead of "SEQ"+bits. A ROM
     that does not understand pre-permuted blobs fails the magic test and
     refuses to load, instead of computing on a layout it cannot handle.
  2. It REFUSES to permute a blob that is already "SEP".
  3. It appends a 4-byte big-endian FINGERPRINT over the first superblock of
     the first weight tensor — bytes that the permutation demonstrably moves.
     The loader recomputes it. A row-major blob wearing an SEP magic (the
     double-permute case, where the magic alone cannot help) fails that check.

Usage:  python3 tools/permute_blob.py in.seq2.bin out.sep2.bin
        python3 tools/permute_blob.py --verify out.sep2.bin
"""
import argparse
import struct
import sys

N_EMBED, N_TENSORS, VOCAB = 256, 6, 256
T_IN  = (N_EMBED, N_EMBED, N_EMBED, N_EMBED, N_EMBED, N_EMBED * 4)
T_OUT = (N_EMBED, N_EMBED, N_EMBED, N_EMBED, N_EMBED * 4, N_EMBED)
HDR = 12
QB = 32
FP_BYTES = 256          # one ternary superblock; the unit the permutation moves


def permute_tensor(buf, off, in_dim, out_dim, bits):
    """Byte-for-byte port of matmul_rsp2.c:rsp2_permute_tensor()."""
    G = 32 if bits == 8 else 8
    sbb = 8 * G
    row_bytes = in_dim if bits == 8 else in_dim // 4
    nsb = row_bytes // sbb
    if nsb * sbb != row_bytes:
        raise SystemExit("row_bytes %d is not a whole number of %d-byte "
                         "superblocks; the C kernel would not tile this either"
                         % (row_bytes, sbb))
    for r in range(out_dim):
        p = off + r * row_bytes
        for s in range(nsb):
            q = p + s * sbb
            blk = bytes(buf[q:q + sbb])
            for i in range(G):
                base = i * 8
                for b in range(8):
                    buf[q + base + b] = blk[b * G + i]


def layer_layout(bits, n_layers):
    """Offsets of each tensor's codes, mirroring sgai_layer_ptrs()."""
    attn_elems = N_EMBED * N_EMBED
    ff_elems = N_EMBED * (N_EMBED * 4)
    elems = (attn_elems,) * 4 + (ff_elems, ff_elems)
    layer_elems = sum(elems)
    wbytes = layer_elems * bits // 8
    scale_bytes = layer_elems // QB * 2
    base = HDR + VOCAB * N_EMBED
    out = []
    for li in range(n_layers):
        p = base + li * (wbytes + scale_bytes)
        for t in range(N_TENSORS):
            out.append((p, T_IN[t], T_OUT[t]))
            p += elems[t] * bits // 8
    return out, wbytes, scale_bytes, base


def fingerprint(buf, base):
    """FNV-1a over the first superblock of the first weight tensor."""
    h = 0x811C9DC5
    for i in range(base, base + FP_BYTES):
        h = ((h ^ buf[i]) * 0x01000193) & 0xFFFFFFFF
    return h


def read_blob(path):
    data = bytearray(open(path, "rb").read())
    magic = bytes(data[:4])
    if magic[:3] == b"SEP":
        kind, bits = "SEP", magic[3] - 0x30
    elif magic[:3] == b"SEQ":
        kind, bits = "SEQ", magic[3] - 0x30
    else:
        raise SystemExit("%s: not a SEQn/SEPn blob (magic %r)" % (path, magic))
    if not (2 <= bits <= 8):
        raise SystemExit("%s: bit width %d out of range" % (path, bits))
    n_layers, embed, heads, vocab, ctx, x16 = struct.unpack("<BHBHBB", data[4:12])
    if embed != N_EMBED or vocab != VOCAB:
        raise SystemExit("%s: embed=%d vocab=%d, this tool assumes %d/%d"
                         % (path, embed, vocab, N_EMBED, VOCAB))
    return data, kind, bits, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing SEP blob's fingerprint and exit")
    a = ap.parse_args()

    data, kind, bits, n_layers = read_blob(a.src)
    tensors, wbytes, scale_bytes, base = layer_layout(bits, n_layers)
    payload = base + n_layers * (wbytes + scale_bytes)

    if a.verify:
        if kind != "SEP":
            raise SystemExit("%s is %s, not a pre-permuted SEP blob" % (a.src, kind))
        if len(data) < payload + 4:
            raise SystemExit("%s: no fingerprint trailer" % a.src)
        want = struct.unpack(">I", bytes(data[payload:payload + 4]))[0]
        got = fingerprint(data, base)
        print("%s: fingerprint stored=%08x computed=%08x %s"
              % (a.src, want, got, "OK" if want == got else "MISMATCH"))
        return 0 if want == got else 2

    if kind == "SEP":
        raise SystemExit(
            "%s is ALREADY pre-permuted (magic SEP%d). Refusing: for ternary the\n"
            "permutation is an involution, so permuting again would silently\n"
            "restore row-major order and the RSP would compute on the wrong\n"
            "layout while still emitting fluent text." % (a.src, bits))
    if not a.dst:
        raise SystemExit("need an output path")

    for off, in_dim, out_dim in tensors:
        permute_tensor(data, off, in_dim, out_dim, bits)

    data[0:4] = b"SEP" + bytes([0x30 + bits])
    fp = fingerprint(data, base)
    out = bytes(data[:payload]) + struct.pack(">I", fp)
    open(a.dst, "wb").write(out)
    print("wrote %s  %d B  bits=%d layers=%d  fingerprint=%08x  (was %d B)"
          % (a.dst, len(out), bits, n_layers, fp, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
