#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""make_moe_bank.py — assemble per-shard SEQ2 blobs into one N64 expert bank.

Layout, chosen to match what `src/expert_cache.c` already implements rather
than inventing a format for it:

    header (32 B)      "SGMB" | version | n_experts | expert_len | reserved
    name table         n_experts x 16 B, NUL-padded shard names
    padding            to `expert_len` alignment
    expert 0           a COMPLETE SEQ2 blob, padded to expert_len
    expert 1           ...

Every expert is a standalone SEQ2 blob -- its own embedding included -- so the
ROM does not need a new parser: `ec_acquire()` returns a pointer that
`sgai_init_ex()` can load directly. The cost is a duplicated 64KB embedding per
expert; the benefit is that the streaming path is a pointer swap, which is the
whole reason the Genesis Lock-On design works and the reason F-R027 could
measure the DMA hiding before any of this existed.

Uniform stride is not cosmetic: `ec_init()` takes ONE `expert_len` and derives
every offset from it.
"""
import argparse, json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_shards

HDR_MAGIC = b"SGMB"
HDR_SIZE  = 32
NAME_LEN  = 16

def build(blobs, names, out_path):
    assert len(blobs) == len(names)
    n = len(blobs)
    raw = [open(b, "rb").read() for b in blobs]
    layouts = set()
    for r, b in zip(raw, blobs):
        if r[:3] not in (b"SEQ", b"SEP"):
            raise SystemExit("%s is not a SEQn/SEPn blob (magic %r)" % (b, r[:4]))
        layouts.add(bytes(r[:3]))
    if len(layouts) > 1:
        # A mixed bank would load correctly for some experts and compute on the
        # wrong layout for others -- and because the permutation only moves
        # bytes, the wrong ones would still emit fluent text. Refuse.
        raise SystemExit("MIXED LAYOUTS in one bank: %s. Every expert must be "
                         "either all row-major (SEQ) or all pre-permuted (SEP)."
                         % sorted(x.decode() for x in layouts))
    layout = layouts.pop().decode()
    expert_len = max(len(r) for r in raw)
    expert_len = (expert_len + 7) & ~7          # PI DMA wants 8-byte alignment
    table = HDR_SIZE + n * NAME_LEN
    # 16-byte alignment, NOT expert_len alignment: PI DMA needs 8 (16 is free
    # and keeps the slot buffers quadword-aligned).  Rounding the base up to a
    # whole expert wasted a full megabyte of cartridge on padding.
    base  = (table + 15) & ~15

    out = bytearray()
    out += HDR_MAGIC
    # BIG-endian: the N64 is big-endian and this bank header is a NEW format,
    # so it is written in the target's byte order and the ROM needs no swap.
    # (The SEQ2 blobs INSIDE keep their own little-endian header, which
    # nano_gpt.c already byte-swaps at load -- that format predates this one.)
    out += struct.pack(">HHII", 1, n, expert_len, base)
    out += b"\0" * (HDR_SIZE - len(out))
    for nm in names:
        out += nm.encode()[:NAME_LEN].ljust(NAME_LEN, b"\0")
    out += b"\0" * (base - len(out))
    sizes = []
    for r in raw:
        out += r + b"\0" * (expert_len - len(r))
        sizes.append(len(r))

    with open(out_path, "wb") as fh:
        fh.write(out)
    return dict(path=out_path, total=len(out), n_experts=n, layout=layout,
                expert_len=expert_len, base=base, names=names, sizes=sizes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="moe_out")
    # The default ORDER is moe_shards.SHARDS, not a hand-typed list: the bank
    # index IS the expert id the generated router returns, so a different order
    # here would route every prompt into the wrong expert while everything
    # still "worked".  Overriding --shards is for partial banks only.
    ap.add_argument("--shards",
                    default=",".join(n for n, _ in moe_shards.SHARDS))
    ap.add_argument("--out", default="filesystem_moe/sophia_moe.bin")
    a = ap.parse_args()
    names = a.shards.split(",")
    blobs = [os.path.join(a.dir, "%s.seq2.bin" % s) for s in names]
    missing = [b for b in blobs if not os.path.exists(b)]
    if missing:
        raise SystemExit("missing: %s (run qat_npz_to_seq.py first)" % missing)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    router_order = [n for n, _ in moe_shards.SHARDS]
    if names != router_order:
        print("WARNING: bank order %s != router order %s -- the ROM will route "
              "prompts into the wrong experts unless this is a deliberate "
              "partial bank" % (names, router_order), file=sys.stderr)
    info = build(blobs, names, a.out)
    print(json.dumps(info, indent=2))
    print("\nec_init(ec, rom_base, expert_len=%d, n_experts=%d, ...)"
          " with experts starting at +%d" % (info["expert_len"], info["n_experts"], info["base"]))

if __name__ == "__main__":
    main()
