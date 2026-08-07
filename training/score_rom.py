#!/usr/bin/env python3
"""Score an ares GAME12_PROBE log with the same rule eval12.py uses."""
import re, sys
import eval12 as E
bank = E.answer_bank()[0]
log = open(sys.argv[1], errors="replace").read().splitlines()
arms = {"G12": {}, "G12T": {}}
hdr = ""
for ln in log:
    if ln.startswith("G12 rdy="):
        hdr = ln.strip(); continue
    m = re.match(r"^(G12T?) (\d+)\|([^|]*)\|(.*)$", ln)
    if m:
        arms[m.group(1)][int(m.group(2))] = (m.group(3), m.group(4))
print(hdr)
for arm, label in (("G12", "temperature 0 (greedy)"), ("G12T", "temperature_q8=64 (real game path)")):
    d = arms[arm]
    if not d: continue
    hits = 0
    print(f"\n-- {label} --")
    for i in sorted(d):
        key, gen = d[i]
        ok, _ = E.score_prompt(key, gen, bank)
        hits += ok
        print(f"  [{'HIT ' if ok else 'MISS'}] {key:32s} -> {E.game_cut(gen)!r}   (raw {gen!r})")
    print(f"  ROM SCORE ({arm}): {hits}/{len(d)}")
