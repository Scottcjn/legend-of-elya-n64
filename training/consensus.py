#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""consensus.py — the host specification for the N64 dual-processor vote.

The N64 has two processors that prefer OPPOSITE weight formats: measured on
this port, ternary is 57.0% cheaper than int8 on the MIPS core and 11.9% more
expensive than int8 on the RSP.  So run each format where it wins and let them
vote.  This file is the arithmetic of that vote, on the host, so the ROM has
something exact to be checked against.

It is a specification, not an estimate: `generate()` here is the same numpy
forward pass `training/eval12.py` uses (the oracle the shipped blob is already
verified against), extended to step several models over one token stream and
combine their logits before the argmax.

COMBINE RULES
  single      one model alone
  raw         Lt + Li, untouched.  The scales differ, so this is the louder
              model wearing a two-model hat -- included because it is the
              mistake the whole scheme fails on if nobody measures it.
  shiftN      Lt + Li / 2**N.  One instruction on either processor.
  norm        Lt/rms(Lt) + Li/rms(Li), rms measured over the same rows.
              The expensive control the cheap rule has to match.
  agree       argmax over Lt + Li only where the members' own argmaxes
              disagree; otherwise the (unanimous) argmax.

USAGE
  python3 training/consensus.py --tern A.seq --int8 B.seq --rule shift5
  python3 training/consensus.py --sweep mix_out            # every seed, every arm
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval12 as E12


# ---------------------------------------------------------------- a stepper
class Runner:
    """One model, one KV cache, fed one token at a time.

    Split out of eval12.generate() so several models can be driven over the
    SAME token stream -- which is what a vote is.  Verified against
    eval12.generate() by test_consensus_equiv(): identical logits, every step.
    """

    def __init__(self, m):
        self.m = m
        E, CTX = m.n_embed, m.ctx
        nL = len(m.layers)
        self.k = np.zeros((nL, CTX, E), np.float32)
        self.v = np.zeros((nL, CTX, E), np.float32)
        self.pos = 0

    def step(self, tok):
        m = self.m
        E, H, HD = m.n_embed, m.n_heads, m.head_dim
        pos = self.pos
        x = m.emb[tok].copy()
        for li, L in enumerate(m.layers):
            res = x
            xn = E12.rms(x)
            q = L["wq"] @ xn
            self.k[li, pos] = L["wk"] @ xn
            self.v[li, pos] = L["wv"] @ xn
            n_ctx = pos + 1
            ao = np.zeros(E, np.float32)
            for h in range(H):
                sl = slice(h * HD, (h + 1) * HD)
                s = (self.k[li, :n_ctx, sl] @ q[sl]) * (HD ** -0.5)
                ao[sl] = E12.softmax(s) @ self.v[li, :n_ctx, sl]
            x = res + L["wo"] @ ao
            res = x
            xn = E12.rms(x)
            x = res + L["wff2"] @ np.maximum(L["wff1"] @ xn, 0.0)
        self.pos += 1
        return m.emb @ E12.rms(x)


# ---------------------------------------------------------------- the vote
def parse_rule(rule):
    if rule.startswith("shift"):
        return "shift", int(rule[5:])
    return rule, 0


def combine(logits, rule, scales=None, lo=32, hi=126):
    """logits: list of (vocab,) arrays, member 0 first.  Returns a token id."""
    kind, n = parse_rule(rule)
    if kind == "single":
        return int(np.argmax(logits[0][lo:hi + 1])) + lo
    if kind == "raw":
        s = sum(logits)
    elif kind == "shift":
        s = logits[0] + logits[1] / float(1 << n)
    elif kind == "norm":
        s = sum(l / sc for l, sc in zip(logits, scales))
    elif kind == "agree":
        picks = [int(np.argmax(l[lo:hi + 1])) + lo for l in logits]
        if all(p == picks[0] for p in picks):
            return picks[0]
        s = sum(logits)
    else:
        raise SystemExit("unknown rule %r" % rule)
    return int(np.argmax(s[lo:hi + 1])) + lo


def generate(models, prompt, n_new, rule="shift5", scales=None, lo=32, hi=126):
    """Greedy decode with the ROM's printable-ASCII band, several models voting.

    Every member is fed EVERY token, including the ones it would not have
    chosen: the vote picks the token, and both KV caches then carry it.  That
    is what the ROM does and it is the only version that is implementable --
    skipping a member on a token leaves a hole in its cache (see the sibling
    port's entry 11 on why a per-token gate between independent models is an
    oracle only)."""
    runners = [Runner(m) for m in models]
    seq = list(prompt if isinstance(prompt, (bytes, bytearray)) else prompt.encode())
    out = []
    ctx = min(m.ctx for m in models)
    for pos in range(len(seq) + n_new - 1):
        if pos >= ctx:
            break
        tok = seq[pos] if pos < len(seq) else out[-1]
        lg = [r.step(tok) for r in runners]
        if pos >= len(seq) - 1:
            out.append(combine(lg, rule, scales, lo, hi))
    return bytes(out).decode("ascii", "replace")


# ------------------------------------------------------- logit scale, measured
def logit_scales(models, keys, n_new=8):
    """RMS of each member's logits over the same decoded positions.

    The scale ratio is the whole risk in a naive sum, so it is measured on the
    real key set rather than reasoned about from the weight format."""
    runners = [Runner(m) for m in models]
    acc = [[] for _ in models]
    for key in keys[:16]:
        rs = [Runner(m) for m in models]
        seq = list(key.encode())
        cur = None
        for pos in range(len(seq) + n_new - 1):
            if pos >= min(m.ctx for m in models):
                break
            tok = seq[pos] if pos < len(seq) else cur
            lg = [r.step(tok) for r in rs]
            if pos >= len(seq) - 1:
                for i, l in enumerate(lg):
                    acc[i].append(float(np.sqrt((l[32:127] ** 2).mean())))
                cur = int(np.argmax(lg[0][32:127])) + 32
    del runners
    return [float(np.mean(a)) for a in acc]


# ---------------------------------------------------------------- scoring
def key_sets():
    L = E12.v7_lists()
    def ks(name):
        out = []
        for line in L[name]:
            k = line.partition(": ")[0] + ": "
            if k not in out:
                out.append(k)
        return out
    return ks("IDENTITY_PAIRS"), ks("QA_PAIRS")


def score(models, keys, bank, rule, scales=None, n_new=48):
    hits, rows = 0, []
    for k in keys:
        g = generate(models, k, n_new, rule, scales)
        h = E12.score_prompt(k, g, bank)[0]
        hits += bool(h)
        rows.append((k, g, bool(h)))
    return hits, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tern", help="SEQ2 blob (the CPU member)")
    ap.add_argument("--int8", help="SEQ8 blob (the RSP member)")
    ap.add_argument("--rule", default="shift5")
    ap.add_argument("--keys", default="all", choices=["all", "game12", "identity", "qa"])
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    bank, _origin = E12.answer_bank()
    idk, qak = key_sets()
    keys = {"all": idk + qak, "identity": idk, "qa": qak,
            "game12": [k for _n, k in E12.GAME_12]}[a.keys]

    ms = [E12.load_seq(a.tern), E12.load_seq(a.int8)]
    sc = logit_scales(ms, keys)
    print("logit RMS  ternary %.4f   int8 %.4f   ratio %.2fx"
          % (sc[0], sc[1], sc[1] / sc[0]))
    print("nearest power of two: shift%d" % int(round(np.log2(sc[1] / sc[0]))))

    res = {}
    for rule in ("single", "single1", "raw", "shift5", "norm", "agree"):
        if rule == "single1":
            h, rows = score([ms[1]], keys, bank, "single", None, a.gen_tokens)
        else:
            h, rows = score(ms, keys, bank, rule, sc, a.gen_tokens)
        res[rule] = h
        print("%-8s %3d/%d  %5.1f%%" % (rule, h, len(keys), 100.0 * h / len(keys)))
    if a.json:
        json.dump({"scores": res, "n": len(keys), "logit_rms": sc},
                  open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
