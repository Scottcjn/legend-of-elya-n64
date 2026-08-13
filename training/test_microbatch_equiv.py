#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove --micro-batch changes nothing but peak memory.

The dual-model work needed two more trained models on a laptop 4070 whose VRAM
was already 6.8 GB occupied by other jobs, and a batch-64 step of this model
wants about 1 GB.  Gradient accumulation over equal-sized micro-batches is the
standard fix, and it is only a fix if it computes the same gradient.

Equal sizes matter: the step loss is a mean over batch_size*ctx tokens, so
"mean of the means" equals the overall mean only when every micro-batch has the
same number of rows.  train_mix.py rejects a --micro-batch that does not divide
--batch-size for exactly that reason.

Reports max|g_full - g_accum| over every parameter, absolutely and relative to
max|g_full|.  Anything above float32 summation noise is a bug, not a rounding.

    python3 training/test_microbatch_equiv.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mix as T


def grads_of(model):
    return {k: p.grad.detach().clone() for k, p in model.named_parameters()
            if p.grad is not None}


def main():
    bs, micro = 16, 4
    T.set_seed(1337)
    cfg = T.ModelCfg(n_layers=8, n_embed=256, n_heads=8, ffn_mult=4, ctx=128,
                     quant="int8", kv_quant="int8")
    tr, _va, _st = T.build_v7_corpus(1337, (800, 800, 400))
    data = T.Data(tr, cfg.ctx)
    model = T.NanoGPTQ(cfg).to(T.device)

    x, y = data.batch(bs, np.random.default_rng(4242))

    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x).reshape(-1, T.VOCAB), y.reshape(-1)).backward()
    g_full = grads_of(model)

    def accum(m):
        n = bs // m
        model.zero_grad(set_to_none=True)
        for mb in range(n):
            xs, ys = x[mb * m:(mb + 1) * m], y[mb * m:(mb + 1) * m]
            (F.cross_entropy(model(xs).reshape(-1, T.VOCAB), ys.reshape(-1)) / n).backward()
        return grads_of(model)

    g_a = accum(micro)     # 4 x 4
    g_b = accum(bs // 2)   # 2 x 8

    def worst_of(ga, gb):
        w, nm = 0.0, ""
        for k in ga:
            d = float((ga[k] - gb[k]).abs().max())
            if d > w:
                w, nm = d, k
        return w, nm

    ref = max(float(v.abs().max()) for v in g_full.values())
    d_test, n_test = worst_of(g_full, g_a)      # the change under test
    d_ctrl, n_ctrl = worst_of(g_a, g_b)         # pure reduction-order control

    print(f"params compared      {len(g_full)}")
    print(f"max|grad| overall    {ref:.6e}")
    print(f"TEST    1x{bs} vs {bs//micro}x{micro}   worst {d_test:.4e}"
          f"  rel {d_test/ref:.3e}  ({n_test})")
    print(f"CONTROL {bs//micro}x{micro} vs 2x{bs//2}   worst {d_ctrl:.4e}"
          f"  rel {d_ctrl/ref:.3e}  ({n_ctrl})")
    print("Both are float32 reduction-order noise if they are the same size.")
    # The control is two accumulations that MUST be mathematically identical, so
    # it measures the floor.  The test passes if it does not exceed that floor
    # by more than 4x -- i.e. the split is noise, not a different quantity.
    ok = d_test <= max(4.0 * d_ctrl, 1e-9)
    print("RESULT:", "EQUIVALENT (within reduction-order noise)" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
