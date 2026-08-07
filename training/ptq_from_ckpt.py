#!/usr/bin/env python3
"""Post-training quantization of a trained checkpoint, for the QAT-vs-PTQ control.

Because QuantLinear applies fake-quant at forward time from the latent float
weights, loading the float-control checkpoint into a model configured with
`--quant int5` (say) IS exactly post-training quantization of that model. Same
data, same seed, same steps, same init as the QAT arm -- the only difference is
when the quantizer was switched on. That is the controlled experiment.

Also reports the val loss under PTQ and writes the same npz/bin artifacts.
"""
import argparse
import json
import os

import numpy as np
import torch

import train_sophia_v9_qat as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--quant", required=True)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--kv-quant", default="int8")
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default="ptq_out")
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-bs", type=int, default=64)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    cfg = T.ModelCfg(n_layers=a["n_layers"], n_embed=a["n_embed"], n_heads=a["n_heads"],
                     ffn_mult=a["ffn_mult"], ctx=a["ctx"], quant=args.quant, tau=args.tau,
                     kv_quant=args.kv_quant, act_quant=a["act_quant"])
    T.set_seed(a["seed"])
    if a.get("corpus", "v8") == "v7":
        tc, ec = T.build_v7_corpus(a["seed"])
    else:
        _ds, _pi, _o, tc, ec = T.build_data(a["seed"], a["train_repeats"], a["eval_repeats"], cfg.ctx)
    val, trn = T.Data(ec, cfg.ctx), T.Data(tc, cfg.ctx)

    model = T.NanoGPTQ(cfg).to(T.device)
    model.load_state_dict(ck.get("final_state", ck["state"]))
    model.eval()
    vl = T.est_loss(model, val, args.val_bs, args.val_batches,
                    np.random.default_rng(a["seed"] + 999))
    fit = T.est_loss(model, trn, args.val_bs, args.val_batches,
                     np.random.default_rng(a["seed"] + 555))

    meta = T.export_blob(model, args.tag, args.outdir)
    meta.update({"tag": args.tag, "quant": args.quant, "tau": args.tau,
                 "kv_quant": args.kv_quant, "act_quant": cfg.act_quant,
                 "best_val": vl, "final_val": vl, "final_fit": fit, "mode": "PTQ", "source_ckpt": args.ckpt, "corpus": a.get("corpus","v8"),
                 "n_layers": cfg.n_layers, "n_embed": cfg.n_embed,
                 "n_heads": cfg.n_heads, "ffn_mult": cfg.ffn_mult, "ctx": cfg.ctx,
                 "params": sum(x.numel() for x in model.parameters()),
                 "quantized_weights": sum(T._tensor_of(b, n).numel()
                                          for b in model.blocks for n in T.TENSORS)})
    with open(os.path.join(args.outdir, f"{args.tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[PTQ {args.tag}] quant={args.quant} fit={fit:.4f} val={vl:.4f} blob={meta['blob_bytes']:,}B")


if __name__ == "__main__":
    main()
