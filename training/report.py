#!/usr/bin/env python3
"""Turn eval_all*.json into the tables for the write-up."""
import json
import sys

RD = 8_388_608
f = sys.argv[1] if len(sys.argv) > 1 else "eval_all.json"
d = json.load(open(f))
rows = d["rows"]

order = {"none": 0, "int8": 1, "int6": 2, "int5": 3, "int4": 4, "int3": 5,
         "ternary": 6, "ternary_bn": 7}


def key(r):
    return (r["n_embed"], r["n_layers"], r["mode"] != "QAT",
            order.get(r["quant"], 9), r["tag"])


P = "Who are you?"

print("\n### Main table (prompt 'Who are you?', 24 greedy tokens)\n")
print("| mode | scheme | shape | val | fit | blob B | +int8 KV | fits 8,388,608 | top1 vs my float control | text |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=key):
    shape = f"{r['n_layers']}L x {r['n_embed']}d"
    t1 = r["top1_vs_control_all"]
    t1s = "-" if t1 is None else f"{100*t1:.0f}%"
    fit = "-" if r["final_fit"] is None else f"{r['final_fit']:.3f}"
    tot = r["blob"] + r["kv_int8_perhead"]
    print(f"| {r['mode']} | {r['quant']}{'' if r['quant']!='ternary' else ' t='+str(r['tau'])} "
          f"| {shape} | {r['final_val']:.3f} | {fit} | {r['blob']:,} | {tot:,} "
          f"| {'YES' if tot <= RD else 'no'} | {t1s} | `{r['gens'][P]}` |")

print("\n### QAT vs post-training, same corpus / seed / steps / init\n")
print("| scheme | PTQ text | QAT text |")
print("|---|---|---|")
qat = {r["quant"]: r for r in rows if r["mode"] == "QAT" and r["n_embed"] == 256 and r["n_layers"] == 8}
ptq = {r["quant"]: r for r in rows if r["mode"] == "PTQ"}
for s in ["int8", "int6", "int5", "int4", "int3", "ternary", "ternary_bn"]:
    a = ptq.get(s)
    b = qat.get(s)
    if a and b:
        print(f"| {s} | `{a['gens'][P]}` | `{b['gens'][P]}` |")

print("\n### All prompts, ternary only\n")
for s in ["ternary", "ternary_bn"]:
    for lbl, r in (("PTQ", ptq.get(s)), ("QAT", qat.get(s))):
        if not r:
            continue
        print(f"\n**{lbl} {s}** (val {r['final_val']:.3f})")
        for k, v in r["gens"].items():
            print(f"  {k[:34]!r:<40} -> `{v}`")

print("\n### int8 KV cost, per model (self-comparison, float KV vs int8 KV)\n")
print("| model | top1 float-KV vs int8-KV |")
print("|---|---|")
for r in sorted(rows, key=key):
    print(f"| {r['mode']} {r['tag']} | {100*r['top1_self_kv8_all']:.0f}% |")
