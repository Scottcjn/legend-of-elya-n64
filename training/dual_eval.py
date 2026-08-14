#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""dual_eval.py — everything the ROM's dual arm has to be checked against.

Four things, in one pass, so the ROM's transcript and the vote's score come
off the SAME loaded blobs:

  1. logit RMS of each member, and the shift that makes them commensurate.
     The sibling port measured 54.7x and used shift 5; that is a property of
     ITS pair of models, not of the scheme, and it is re-measured here.
  2. ROM CROSS-CHECK: the exact decode the ROM does -- greedy argmax over
     32..126, prompt ingested at temperature 0, N tokens out -- for each arm.
     These strings must match dual_probe.c's TEXT lines character for
     character or the ROM's arithmetic is wrong.
  3. whole-key score, the eval the shipped model is quoted on.
  4. held-out next-character accuracy, which is where a vote can actually show
     something: on the whole-key eval both members are at 98.4% and the
     ceiling is the measurement.

  python3 training/dual_eval.py --tern A.seq --int8 B.seq
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consensus as C
import eval12 as E12

ROM_PROMPT = "sage says: Who are you?: "


def rom_decode(models, prompt, n_new, rule, scales=None, lo=32, hi=126):
    """Exactly dual_probe.c: ingest the prompt at temperature 0, then generate.

    The ROM feeds every prompt byte through the forward pass and throws the
    prediction away, then feeds back its own last output.  Reproduced here
    step for step rather than approximated, because the point of this function
    is to be byte-comparable with a machine."""
    rs = [C.Runner(m) for m in models]
    b = prompt.encode()
    tok = 32
    for c in b:
        lg = [r.step(c) for r in rs]
        tok = C.combine(lg, rule, scales, lo, hi)
    out = []
    for _ in range(n_new):
        out.append(tok)
        lg = [r.step(tok) for r in rs]
        tok = C.combine(lg, rule, scales, lo, hi)
    # dual_probe stores the token RETURNED by each call, i.e. the prediction
    # made from the token it was fed -- so the first stored character is the
    # one predicted from the last prompt byte.
    return bytes(out).decode("ascii", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tern", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--ngen", type=int, default=16)
    ap.add_argument("--rules", default="single,single1,raw,shift1,shift5,norm,agree")
    ap.add_argument("--skip-keys", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    mt, mi = E12.load_seq(a.tern), E12.load_seq(a.int8)
    ms = [mt, mi]
    print("tern %s layers=%d   int8 %s layers=%d"
          % (os.path.basename(a.tern), len(mt.layers),
             os.path.basename(a.int8), len(mi.layers)), flush=True)

    bank, _ = E12.answer_bank()
    idk, qak = C.key_sets()
    keys = idk + qak

    sc = C.logit_scales(ms, keys)
    ratio = sc[1] / sc[0]
    print("logit RMS  ternary %.4f  int8 %.4f  ratio %.3fx  -> shift%d"
          % (sc[0], sc[1], ratio, int(round(np.log2(ratio)))), flush=True)

    print("\n-- ROM cross-check, prompt %r, %d tokens --" % (ROM_PROMPT, a.ngen),
          flush=True)
    romtxt = {}
    romtxt["TERN"] = rom_decode([mt], ROM_PROMPT, a.ngen, "single")
    romtxt["INT8"] = rom_decode([mi], ROM_PROMPT, a.ngen, "single")
    for sh in (0, 1, 2, 5):
        romtxt["VOTE_shift%d" % sh] = rom_decode(ms, ROM_PROMPT, a.ngen,
                                                 "shift%d" % sh, sc)
    for k, v in romtxt.items():
        print("  %-12s %r" % (k, v), flush=True)

    held, stats = C.held_lines()
    print("\n-- held-out next-char accuracy, %d lines, %d chars of text --"
          % (len(held), sum(len(h) for h in held)), flush=True)
    rules = a.rules.split(",")
    # ONE teacher-forced pass, both members, logits kept -- then every rule is
    # scored off the same stored logits.  Re-running the forward pass per rule
    # would be 7x the numpy for arithmetically identical inputs, and would also
    # leave open the question of whether the arms saw the same text.
    LT, LI, NX = [], [], []
    ctx = min(m.ctx for m in ms)
    for line in held:
        rt, ri = C.Runner(mt), C.Runner(mi)
        b = line.encode("ascii", "replace")
        for i in range(min(len(b) - 1, ctx - 1)):
            lt, li = rt.step(b[i]), ri.step(b[i])
            nxt = b[i + 1]
            if not 32 <= nxt <= 126:
                continue
            LT.append(lt); LI.append(li); NX.append(nxt)
    LT = np.array(LT, np.float32); LI = np.array(LI, np.float32)
    NX = np.array(NX)
    n = len(NX)
    nc = {}
    for r in rules:
        if r == "single":
            pick = LT[:, 32:127].argmax(1) + 32
        elif r == "single1":
            pick = LI[:, 32:127].argmax(1) + 32
        elif r == "raw":
            pick = (LT + LI)[:, 32:127].argmax(1) + 32
        elif r.startswith("shift"):
            pick = (LT + LI / float(1 << int(r[5:])))[:, 32:127].argmax(1) + 32
        elif r == "norm":
            pick = (LT / sc[0] + LI / sc[1])[:, 32:127].argmax(1) + 32
        elif r == "agree":
            pt = LT[:, 32:127].argmax(1) + 32
            pi = LI[:, 32:127].argmax(1) + 32
            ps = (LT + LI)[:, 32:127].argmax(1) + 32
            pick = np.where(pt == pi, pt, ps)
        else:
            raise SystemExit("unknown rule %r" % r)
        h = int((pick == NX).sum())
        nc[r] = (h, n)
        print("  %-8s %6d/%-6d  %6.2f%%" % (r, h, n, 100.0 * h / n), flush=True)
    # Significance of the vote against the better member, paired.
    #
    # The two arms see the SAME 1716 characters, so an unpaired proportions
    # test throws away the pairing and is the wrong test.  McNemar's exact
    # test conditions on the positions where exactly one of them is right,
    # which is the only place the difference can come from.
    def mcnemar(pick_a, pick_b):
        ra, rb = (pick_a == NX), (pick_b == NX)
        b = int((ra & ~rb).sum())   # a right, b wrong
        c = int((rb & ~ra).sum())   # b right, a wrong
        nn = b + c
        if nn == 0:
            return b, c, 1.0
        # exact two-sided binomial against p = 0.5
        k = min(b, c)
        from math import comb
        tail = sum(comb(nn, i) for i in range(k + 1)) / float(2 ** nn)
        return b, c, min(1.0, 2.0 * tail)

    print("\n-- paired significance, McNemar exact, vs each member --", flush=True)
    picks = {}
    picks["single"] = LT[:, 32:127].argmax(1) + 32
    picks["single1"] = LI[:, 32:127].argmax(1) + 32
    for r in rules:
        if r in ("single", "single1"):
            continue
        if r == "raw":
            picks[r] = (LT + LI)[:, 32:127].argmax(1) + 32
        elif r.startswith("shift"):
            picks[r] = (LT + LI / float(1 << int(r[5:])))[:, 32:127].argmax(1) + 32
        elif r == "norm":
            picks[r] = (LT / sc[0] + LI / sc[1])[:, 32:127].argmax(1) + 32
        elif r == "agree":
            pa = picks["single"]; pb = picks["single1"]
            picks[r] = np.where(pa == pb, pa, (LT + LI)[:, 32:127].argmax(1) + 32)
    sig = {}
    for r in rules:
        if r in ("single", "single1"):
            continue
        bt, ct, pt_ = mcnemar(picks["single"], picks[r])
        bi, ci, pi_ = mcnemar(picks["single1"], picks[r])
        sig[r] = {"vs_tern": [bt, ct, pt_], "vs_int8": [bi, ci, pi_]}
        print("  %-8s vs ternary: +%d/-%d p=%.4g   vs int8: +%d/-%d p=%.4g"
              % (r, ct, bt, pt_, ci, bi, pi_), flush=True)

    # How often the two members disagree at all -- the vote's whole budget.
    pt = LT[:, 32:127].argmax(1) + 32
    pi = LI[:, 32:127].argmax(1) + 32
    dis = int((pt != pi).sum())
    both = int(((pt == NX) & (pi == NX)).sum())
    only_t = int(((pt == NX) & (pi != NX)).sum())
    only_i = int(((pi == NX) & (pt != NX)).sum())
    neither = n - both - only_t - only_i
    print("  members disagree %d/%d = %.1f%%   both-right %d  tern-only %d  "
          "int8-only %d  neither %d" % (dis, n, 100.0 * dis / n,
                                        both, only_t, only_i, neither), flush=True)
    nc["_diag"] = {"disagree": dis, "both": both, "tern_only": only_t,
                   "int8_only": only_i, "neither": neither, "n": n}

    keyres = {}
    if not a.skip_keys:
        print("\n-- whole-key score, %d keys --" % len(keys), flush=True)
        for r in rules:
            if r == "single1":
                h, _ = C.score([mi], keys, bank, "single", None)
            elif r == "single":
                h, _ = C.score([mt], keys, bank, "single", None)
            else:
                h, _ = C.score(ms, keys, bank, r, sc)
            keyres[r] = h
            print("  %-8s %3d/%d  %5.1f%%" % (r, h, len(keys),
                                              100.0 * h / len(keys)), flush=True)

    if a.json:
        json.dump({"logit_rms": sc, "ratio": ratio, "rom": romtxt,
                   "nextchar": nc, "mcnemar": sig, "keys": keyres,
                   "held_lines": len(held), "split": stats},
                  open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
