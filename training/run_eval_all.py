#!/usr/bin/env python3
"""Master evaluation: QAT vs post-training quantization, side by side.

Three families, all judged by the same numpy reference port of nano_gpt.c:
  A. the SHIPPED 8.4M Q8 model, post-training quantized  (run_eval_ptq_shipped.py)
  B. my float control (`--quant none`), post-training quantized
  C. my QAT models
B and C share seed, data, steps, init and hyperparameters, so B-vs-C isolates
exactly one variable: whether the quantizer was on during training.
"""
import argparse
import glob
import json
import os

import numpy as np

import eval_qat as E

N = 24
RAW = E.RAW_PROMPTS
PERS = E.PERSONA_PROMPTS


def gen_all(m, kv_int8=False, kv_per_head=True):
    return {p: E.generate(m, list(p), N, kv_int8=kv_int8, kv_per_head=kv_per_head)
            for p in RAW + PERS}


def agree(a, b, keys):
    return float(np.mean([np.mean([x == y for x, y in zip(a[p], b[p])]) for p in keys]))


def kv_bytes(nL, ctx, E_, H):
    f = nL * ctx * E_ * 4 * 2 + 4
    ph = nL * ctx * E_ * 2 + nL * ctx * H * 2 * 4 + 4
    pv = nL * ctx * E_ * 2 + nL * ctx * 2 * 4 + 4
    return f, ph, pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["qat_out", "ptq_out"])
    ap.add_argument("--out", default="eval_all.json")
    ap.add_argument("--rdram", type=int, default=8_388_608)
    args = ap.parse_args()

    shipped = E.load_shipped()
    ship_gen = gen_all(shipped)
    print("shipped Q8 reference:")
    for p in RAW:
        print(f"   {p.decode():<14} -> {E.show(ship_gen[p])!r}", flush=True)

    rows = []
    metas = []
    for d in args.dirs:
        metas += sorted(glob.glob(os.path.join(d, "*_meta.json")))

    models = {}
    for mp in metas:
        npz = mp.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        m, meta = E.load_qat(npz)
        models[(meta.get("mode", "QAT"), meta["tag"])] = (m, meta)

    ctrl_key = ("QAT", "none")
    ctrl_gen = gen_all(models[ctrl_key][0]) if ctrl_key in models else None

    for (mode, tag), (m, meta) in sorted(models.items()):
        g = gen_all(m)
        nq = meta["quantized_weights"]
        bits = meta["bits"]
        blob = os.path.getsize(meta["bin"]) if os.path.exists(meta["bin"]) else meta["blob_bytes"]
        f_kv, ph_kv, pv_kv = kv_bytes(meta["n_layers"], meta["ctx"], meta["n_embed"], meta["n_heads"])
        g_kv8 = gen_all(m, kv_int8=True, kv_per_head=True)
        row = {
            "mode": mode, "tag": tag, "quant": meta["quant"], "tau": meta.get("tau"),
            "bits": bits, "params": meta["params"], "quantized_weights": nq,
            "n_layers": meta["n_layers"], "n_embed": meta["n_embed"],
            "ffn_mult": meta["ffn_mult"], "ctx": meta["ctx"],
            "final_val": meta.get("final_val", meta.get("best_val")),
            "final_fit": meta.get("final_fit"),
            "blob": blob,
            "kv_float": f_kv, "kv_int8_perhead": ph_kv, "kv_int8_pervec": pv_kv,
            "total_float_kv": blob + f_kv, "total_int8_kv": blob + ph_kv,
            "fits_rdram_int8kv": (blob + ph_kv) <= args.rdram,
            "fits_rdram_float_kv": (blob + f_kv) <= args.rdram,
            "top1_vs_shipped_raw": agree(g, ship_gen, RAW),
            "top1_vs_shipped_all": agree(g, ship_gen, RAW + PERS),
            "top1_vs_control_raw": agree(g, ctrl_gen, RAW) if ctrl_gen else None,
            "top1_vs_control_all": agree(g, ctrl_gen, RAW + PERS) if ctrl_gen else None,
            "top1_self_kv8_all": agree(g_kv8, g, RAW + PERS),
            "gens": {p.decode(): E.show(g[p]) for p in RAW + PERS},
            "gens_kv_int8": {p.decode(): E.show(g_kv8[p]) for p in RAW + PERS},
        }
        rows.append(row)
        print(f"{mode:4s} {tag:14s} bits={bits:2d} val={row['final_val']:.3f} "
              f"fit={row['final_fit'] if row['final_fit'] is None else round(row['final_fit'],3)} "
              f"blob={blob:,} fits={row['fits_rdram_int8kv']} "
              f"kv8_top1={row['top1_self_kv8_all']*100:.0f}% "
              f"| {row['gens']['Who are you?']!r}", flush=True)

    # int8 KV cost on the shipped model too (never trained with a quantized KV)
    sh = {}
    for name, ph in (("perhead", True), ("pervec", False)):
        g = gen_all(shipped, kv_int8=True, kv_per_head=ph)
        sh[name] = {"top1_all": agree(g, ship_gen, RAW + PERS),
                    "top1_raw": agree(g, ship_gen, RAW),
                    "gens": {p.decode(): E.show(g[p]) for p in RAW + PERS}}
        print(f"shipped Q8 + int8 KV ({name}): top1 raw={sh[name]['top1_raw']*100:.1f}% "
              f"all={sh[name]['top1_all']*100:.1f}% | {sh[name]['gens']['Who are you?']!r}", flush=True)

    json.dump({"shipped": {p.decode(): E.show(ship_gen[p]) for p in RAW + PERS},
               "shipped_kv_int8": sh, "rows": rows},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
