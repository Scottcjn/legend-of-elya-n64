#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""eval_moe.py — score the routed MoE bank on the SAME 32 game prompts and the
SAME metric as the dense baseline (docs/N64_COHERENCE_FINDINGS.md C027).

For each prompt in legend_of_elya.c's PROMPTS[]: route it with the ROM's own
rule (training/moe_shards.py, which tools/gen_moe_router.py compiles into
moe_router.h), generate greedily from THAT expert's blob using build/host_eval
-- the ROM's own nano_gpt.c compiled natively -- and score:

  * invented words in the game-cut answer, against the 828-word v7 vocabulary
  * whether the answer starts with a trained answer for that key (eval12 rule)

The dense baseline to beat, measured 2026-08-28: 32/32 answers with zero
invented words, 28/28 trained-answer hits, 2.56 invented words per 64-token
free run.
"""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval12 as E
import moe_shards as M

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HOST_EVAL = os.path.join(ROOT, "build", "host_eval")

def prompts():
    src = open(os.path.join(ROOT, "legend_of_elya.c")).read()
    blk = src[src.index("static const char *PROMPTS[] = {"):]
    blk = blk[:blk.index("};")]
    return re.findall(r'"([^"]+)"', blk)

def vocab():
    L = E.v7_lists()
    v = set()
    for name in ("IDENTITY_PAIRS", "QA_PAIRS", "CORPUS_LINES"):
        for line in L[name]:
            v.update(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*", line))
    return v

def gen(blob, ps, n=64):
    out = {}
    for i in range(0, len(ps), 8):
        args = [HOST_EVAL, blob]
        for p in ps[i:i + 8]:
            args += ["--prompt", p]
        args += ["-n", str(n)]
        r = subprocess.run(args, capture_output=True, text=True)
        for ln in (r.stdout + r.stderr).splitlines():
            m = re.match(r'\s+(.+?:)\s+->\s+"(.*)"\s*$', ln)
            if m:
                out[m.group(1) + " "] = m.group(2)
    return out

def main():
    moedir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "training", "moe_out")
    P, V, bank = prompts(), vocab(), E.answer_bank()[0]
    by_expert = {}
    for p in P:
        by_expert.setdefault(M.shard_of(p), []).append(p)

    gens, routed = {}, {}
    for shard, ps in by_expert.items():
        blob = os.path.join(moedir, "%s.seq2.bin" % shard)
        if not os.path.exists(blob):
            raise SystemExit("missing expert blob %s" % blob)
        for k, v in gen(blob, ps).items():
            gens[k] = v
            routed[k] = shard

    clean = hits = scored = 0
    inv_full = words_full = 0
    for k in sorted(gens, key=lambda x: P.index(x.strip() + " ") if (x.strip() + " ") in P else 0):
        g = gens[k]
        cut = E.game_cut(g)
        bad = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", cut) if w.lower() not in V]
        fb = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", g) if w.lower() not in V]
        inv_full += len(fb); words_full += len(re.findall(r"[A-Za-z][A-Za-z']*", g))
        ok = None
        if k in bank:
            ok, _ = E.score_prompt(k, g, bank); scored += 1; hits += ok
        clean += (not bad)
        print("  [%s] %-14s %-30s %r inv=%s" %
              ("HIT " if ok else ("MISS" if ok is not None else " -- "),
               routed[k], k, cut, bad))
    n = len(gens)
    print("\nMoE ROUTED: %d/%d answers with 0 invented words; trained-answer hits %d/%d" %
          (clean, n, hits, scored))
    print("free-run 64 tok: %d invented / %d words = %.2f per line" %
          (inv_full, words_full, inv_full / max(n, 1)))
    print("dense baseline (C027): 32/32 clean, 28/28 hits, 2.56 invented/line")

if __name__ == "__main__":
    main()
