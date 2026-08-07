#!/usr/bin/env python3
"""Aggregate mix_out/*_meta.json into the sweep table."""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "mix_out"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*_meta.json"))):
    m = json.load(open(f))
    sc = m.get("scores", {})
    for which in ("final", "best"):
        if which not in sc:
            continue
        s = sc[which]
        rows.append(dict(tag=m["tag"], reps=tuple(m.get("reps", ())), which=which,
                         quant=m["quant"], steps=m["steps"],
                         g12=s["game12"], ident=s["identity"], identT=s["identity_total"],
                         qa=s["qa"], qaT=s["qa_total"],
                         val=m["final_val"] if which == "final" else m["best_val"],
                         fit=m["final_fit"],
                         mass=m.get("split", {}).get("mass", {})))
print(f"{'tag':14s} {'reps':18s} {'sel':6s} {'quant':11s} {'GAME':6s} {'IDENT':8s} "
      f"{'QA':9s} {'val':7s} {'fit':7s}")
for r in rows:
    print(f"{r['tag']:14s} {str(r['reps']):18s} {r['which']:6s} {r['quant']:11s} "
          f"{r['g12']:2d}/12  {r['ident']:2d}/{r['identT']:<3d}  "
          f"{r['qa']:2d}/{r['qaT']:<4d}  {r['val']:7.3f} {r['fit']:7.3f}")
