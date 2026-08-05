#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
qat_finetune.py — quantization-aware FINE-TUNE of the shipped N64 model.

Post-training ternarization destroys this model (FINDINGS T7).  The standard
answer is quantization-aware training.  Training from scratch is 180,000 steps;
fine-tuning from the weights that already exist is far cheaper and is what
anyone would actually do, so that is what this does:

  1. rebuild the NanoGPT from train_sophia_v8.py
  2. warm-start it by DEQUANTIZING the shipped SEAI blob back to float
     (int8 * f16 block scale — the exact inverse of the exporter)
  3. turn on QuantLinear's fake quantizer at the target width (2 = ternary)
  4. fine-tune on the same corpus the model was trained on
  5. export a SEQn blob directly from the float weights

Step 5 exports straight from float rather than going float -> int8 -> ternary,
so the shipped bits are exactly what the training loop simulated.

Usage:
  qat_finetune.py --bits 2 --steps 4000 --out out/w_b2_qat.bin
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import train_sophia_v8 as T                      # noqa: E402
from quantize_n64 import pack_bits               # noqa: E402

Q_BLOCK = T.Q_BLOCK
TENSORS = ["wq", "wk", "wv", "wo", "wff1", "wff2"]


# --------------------------------------------------------------------------
def load_seai(path):
    """Dequantize the shipped blob back to float tensors."""
    d = open(path, "rb").read()
    assert d[:4] == b"SEAI", d[:4]
    _, n_layers, n_embed, n_heads, vocab, ctx, em16 = struct.unpack("<IBHBHBB", d[:12])
    assert (n_layers, n_embed, vocab) == (T.N_LAYERS, T.N_EMBED, T.VOCAB)
    off = 12
    emb_q = np.frombuffer(d, np.int8, vocab * n_embed, off).reshape(vocab, n_embed)
    off += vocab * n_embed
    em_scale = (em16 / 16.0) if em16 else 3.5      # nano_gpt.c's fallback
    emb = emb_q.astype(np.float32) * (em_scale / 127.0)

    shapes = [(n_embed, n_embed)] * 4 + [(n_embed * 4, n_embed), (n_embed, n_embed * 4)]
    layers = []
    for _ in range(n_layers):
        counts = [a * b for a, b in shapes]
        w0, s0 = off, off + sum(counts)
        L = {}
        for (nm, shp, cnt) in zip(TENSORS, shapes, counts):
            q = np.frombuffer(d, np.int8, cnt, w0).astype(np.float32)
            s = np.frombuffer(d, "<f2", cnt // Q_BLOCK, s0).astype(np.float32)
            L[nm] = (q.reshape(-1, Q_BLOCK) * s[:, None]).reshape(shp)
            w0 += cnt
            s0 += (cnt // Q_BLOCK) * 2
        off = s0
        layers.append(L)
    assert off == len(d), (off, len(d))
    return emb, layers, em_scale


def warm_start(model, emb, layers):
    with torch.no_grad():
        model.emb.weight.copy_(torch.from_numpy(emb))
        for blk, L in zip(model.blocks, layers):
            for nm in TENSORS:
                mod = getattr(blk.attn, nm, None) or getattr(blk, nm)
                mod.weight.copy_(torch.from_numpy(L[nm]))


# --------------------------------------------------------------------------
def export(model, path, bits, tau, em_scale):
    """Write a SEQn blob straight from the float weights."""
    out = bytearray()
    out += b"SEQ" + bytes([ord('0') + bits])
    em16 = int(round(em_scale * 16.0)) & 0xFF
    out += struct.pack("<BHBHBB", T.N_LAYERS, T.N_EMBED, T.N_HEADS, T.VOCAB, T.CTX, em16)

    emb = model.emb.weight.detach().cpu().float().numpy()
    emb_q = np.clip(np.round(emb / em_scale * 127.0), -128, 127).astype(np.int8)
    out += emb_q.tobytes()

    for blk in model.blocks:
        packed, scales = [], []
        for nm in TENSORS:
            mod = getattr(blk.attn, nm, None) or getattr(blk, nm)
            w = mod.weight.detach().cpu().float().reshape(-1, Q_BLOCK)
            if bits == 2:
                a = w.abs()
                delta = tau * a.mean(dim=1, keepdim=True)
                keep = a > delta
                lv = (torch.sign(w) * keep)
                n = keep.sum(dim=1, keepdim=True).clamp(min=1)
                alpha = ((a * keep).sum(dim=1, keepdim=True) / n)
            else:
                qmax = (1 << (bits - 1)) - 1
                alpha = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)
                lv = torch.clamp(torch.round(w / alpha.half().float()), -qmax - 1, qmax)
            lv = lv.to(torch.int32).reshape(-1).numpy()
            codes = (lv & ((1 << bits) - 1)).astype(np.uint8)
            packed.append(pack_bits(codes, bits))
            scales.append(alpha.reshape(-1).numpy().astype(np.float16))
        for p in packed:
            out += p.tobytes()
        for s in scales:
            out += s.astype("<f2").tobytes()

    open(path, "wb").write(bytes(out))
    return len(out)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seai", default=os.path.join(ROOT, "filesystem", "sophia_weights.bin"))
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-train", action="store_true",
                    help="Export the warm-started weights without fine-tuning. "
                         "This is the POST-TRAINING control arm, produced through "
                         "exactly the same export path as the QAT arm.")
    ap.add_argument("--em-scale", type=float, default=0.4,
                    help="Embedding scale to train and export at. The shipped blob "
                         "stores 0 for this field, so nano_gpt.c falls back to 3.5; "
                         "at 3.5 the logits are so large that cross-entropy is "
                         "dominated by temperature and stops tracking quality "
                         "(float val 19.47 vs ternary 16.02 — anti-correlated). "
                         "0.4 minimises float val loss and makes CE usable as a "
                         "training signal. Greedy output is unchanged by the "
                         "choice (verified: identical text at 3.5 and 0.4).")
    a = ap.parse_args()

    T.set_seed(1337)
    dataset = json.loads(T.PERSONA_DATASET_JSON)
    T.validate_dataset(dataset)
    info, order = T.persona_maps(dataset)
    train_samples, _, eval_by = T.build_formatted_sets(dataset, info, order)
    train_data = T.CharDataset(T.build_corpus(train_samples, 220, 1337))
    eval_samples = [s for pid in order for s in eval_by[pid]]
    val_data = T.CharDataset(T.build_corpus(eval_samples, 12, 1338))

    emb, layers, em_scale_file = load_seai(a.seai)
    em_scale = a.em_scale
    emb = emb / em_scale_file * em_scale        # re-express at the chosen scale
    model = T.NanoGPT().to(T.device)
    warm_start(model, emb, layers)

    def val():
        return T.estimate_loss(model, val_data, a.batch_size, 8)

    T.QAT_BITS, T.QAT_TAU = 0, a.tau
    v_float = val()
    T.QAT_BITS = a.bits
    v_quant = val()
    print(f"warm start   val(float weights) = {v_float:.4f}"
          f"   val({a.bits}-bit fake-quant) = {v_quant:.4f}")

    if a.eval_only:
        return 0

    if a.no_train:
        n = export(model, a.out, a.bits, a.tau, em_scale)
        print(f"wrote CONTROL (no fine-tune) {a.out}  {n} bytes")
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.05)
    t0 = time.time()
    best = float("inf")
    best_state = None
    for step in range(1, a.steps + 1):
        xb, yb = train_data.batch(a.batch_size, T.device)
        loss = F.cross_entropy(model(xb).reshape(-1, T.VOCAB), yb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 250 == 0 or step == 1:
            v = val()
            if v < best:
                best = v
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            print(f"step {step:6d}/{a.steps}  train={loss.item():.4f}  val={v:.4f}"
                  f"  best={best:.4f}  {time.time()-t0:.0f}s", flush=True)
    if best_state:
        model.load_state_dict(best_state)
        model.to(T.device)
    print(f"best QAT val = {best:.4f}   (warm-start fake-quant val was {v_quant:.4f},"
          f" float val was {v_float:.4f})")
    n = export(model, a.out, a.bits, a.tau, em_scale)
    print(f"wrote {a.out}  {n} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
