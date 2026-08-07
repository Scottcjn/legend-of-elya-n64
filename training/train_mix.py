#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
train_sophia_v9_qat.py - Quantization-aware trainer for the Legend of Elya N64 LLM.

Derived from train_sophia_v8.py (same dataset, same architecture, same SEAI
weight layout) with one addition: every quantized Linear runs a straight-through
estimator, so the forward pass sees the quantized weights the ROM will actually
execute while the backward pass updates the latent float weights.

Quantization is per 32-weight block ALONG THE INPUT DIM, which is exactly the
layout `export_q8()` writes and `nano_gpt.c` reads (a flat row-major reshape of
an (out,in) weight into 32-element groups puts each group inside one output
row's input span). Scales are rounded to float16 in training too, because that
is how they are stored.

The STE, the activation fake-quant and the KV fake-quant follow the proven
sibling trainer ~/legend-of-elya-genesis/train/train_elya_moe.py.

Schemes:
  --quant none                 float control
  --quant int3|int4|int5|int6|int8
  --quant ternary              TWN: delta = tau*mean|w| per block,
                               alpha = mean|w| over the survivors
  --quant ternary_bn           BitNet b1.58 absmean per block:
                               s = mean|w|, q = clamp(round(w/s), -1, 1)

Embedding stays int8 (and it is the tied head), matching BitNet practice and
the SEAI format's `em_scale_x16` global scale.
"""

import argparse
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval12 as E12

device = "cuda" if torch.cuda.is_available() else "cpu"
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

VOCAB = 256
Q_BLOCK = 32
MAGIC_LE = 0x49414553


# --------------------------------------------------------------------------
# fake quantization
# --------------------------------------------------------------------------

def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward uses q, backward flows into x."""
    return x + (q - x).detach()


def _f16(t: torch.Tensor) -> torch.Tensor:
    """Round a scale tensor to float16, which is how the blob stores it."""
    return t.half().float()


def q_codes_intn(w: torch.Tensor, bits: int):
    """Per-32-block symmetric int-N. Returns (codes, scales) as float tensors."""
    out_f, in_f = w.shape
    b = w.reshape(out_f, in_f // Q_BLOCK, Q_BLOCK)
    qmax = (1 << (bits - 1)) - 1
    s = _f16((b.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8))
    q = (b / s).round().clamp(-qmax - 1, qmax)
    return q, s


def q_codes_ternary(w: torch.Tensor, tau: float):
    """TWN per 32-block. delta = tau*mean|w|; alpha = mean|w| over survivors."""
    out_f, in_f = w.shape
    b = w.reshape(out_f, in_f // Q_BLOCK, Q_BLOCK)
    ab = b.abs()
    delta = tau * ab.mean(-1, keepdim=True)
    m = (ab > delta).to(b.dtype)
    alpha = _f16(((ab * m).sum(-1, keepdim=True) /
                  m.sum(-1, keepdim=True).clamp(min=1.0)).clamp(min=1e-8))
    return torch.sign(b) * m, alpha


def q_codes_ternary_bn(w: torch.Tensor, _tau: float = 0.0):
    """BitNet b1.58 absmean, applied per 32-block (the sibling model's rule)."""
    out_f, in_f = w.shape
    b = w.reshape(out_f, in_f // Q_BLOCK, Q_BLOCK)
    s = _f16(b.abs().mean(-1, keepdim=True).clamp(min=1e-8))
    return (b / s).round().clamp(-1, 1), s


def fake_quant_weight(w: torch.Tensor, scheme: str, tau: float) -> torch.Tensor:
    if scheme == "none":
        return w
    if scheme == "ternary":
        q, s = q_codes_ternary(w, tau)
    elif scheme == "ternary_bn":
        q, s = q_codes_ternary_bn(w)
    else:
        q, s = q_codes_intn(w, int(scheme[3:]))
    return ste(w, (q * s).reshape(w.shape))


def fq_emb(w: torch.Tensor) -> torch.Tensor:
    """int8 embedding with the SEAI global em_scale_x16 scale. Exactly matches
    quantize_embedding_table() in v8 / embed_lookup() in nano_gpt.c."""
    mx = w.detach().abs().max().clamp(min=1e-6)
    x16 = torch.round(torch.clamp(mx, max=255.0 / 16.0) * 16.0).clamp(min=1.0)
    stored = x16 / 16.0
    q = torch.round(w / stored * 127.0).clamp(-128, 127)
    return ste(w, q * stored / 127.0)


def fq_act_q12(x: torch.Tensor) -> torch.Tensor:
    """Q12 fixed point, as the sibling GBC runtime uses. The N64 runtime is
    float32, so this is OPTIONAL here and off by default."""
    return ste(x, (x * 4096.0).round().clamp(-32768, 32767) / 4096.0)


def fq_kv_int8(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-vector absmax int8 KV. x is (B, H, T, HD); one scale per
    (batch, head, position) vector, which is how an int8 KV cache would be
    stored on the N64 (1 byte/element + one float scale per head-vector)."""
    s = (x.detach().abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
    return ste(x, (x / s).round().clamp(-127, 127) * s)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

@dataclass
class ModelCfg:
    n_layers: int = 8
    n_embed: int = 256
    n_heads: int = 8
    ffn_mult: int = 4
    ctx: int = 128
    quant: str = "int8"
    tau: float = 0.7
    kv_quant: str = "int8"      # none | int8
    act_quant: str = "none"     # none | q12


class QuantLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, cfg: ModelCfg):
        super().__init__()
        if in_f % Q_BLOCK:
            raise ValueError("input dim must be a multiple of the 32-weight block")
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.5 / math.sqrt(in_f))
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quant_weight(self.weight, self.cfg.quant, self.cfg.tau)
        y = F.linear(x, w)
        return fq_act_q12(y) if self.cfg.act_quant == "q12" else y


class RMSNorm(nn.Module):
    """No learned parameters, matching nano_gpt.c's rms_norm()."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x / x.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt()
        return fq_act_q12(y) if self.cfg.act_quant == "q12" else y


class Attn(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        e = cfg.n_embed
        self.cfg = cfg
        self.head_dim = e // cfg.n_heads
        self.wq = QuantLinear(e, e, cfg)
        self.wk = QuantLinear(e, e, cfg)
        self.wv = QuantLinear(e, e, cfg)
        self.wo = QuantLinear(e, e, cfg)
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.ctx, cfg.ctx, dtype=torch.bool)).view(1, 1, cfg.ctx, cfg.ctx),
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h, hd = self.cfg.n_heads, self.head_dim

        def pr(layer):
            return layer(x).view(b, t, h, hd).transpose(1, 2)

        q, k, v = pr(self.wq), pr(self.wk), pr(self.wv)
        if self.cfg.kv_quant == "int8":
            k, v = fq_kv_int8(k), fq_kv_int8(v)
        a = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        a = a.masked_fill(~self.mask[:, :, :t, :t], float("-inf"))
        o = (F.softmax(a, dim=-1) @ v).transpose(1, 2).contiguous().view(b, t, c)
        if self.cfg.act_quant == "q12":
            o = fq_act_q12(o)
        return self.wo(o)


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        e, f = cfg.n_embed, cfg.n_embed * cfg.ffn_mult
        self.cfg = cfg
        self.ln1, self.ln2 = RMSNorm(cfg), RMSNorm(cfg)
        self.attn = Attn(cfg)
        self.wff1 = QuantLinear(e, f, cfg)
        self.wff2 = QuantLinear(f, e, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.cfg.act_quant == "q12":
            x = fq_act_q12(x)
        x = x + self.wff2(F.relu(self.wff1(self.ln2(x))))
        return fq_act_q12(x) if self.cfg.act_quant == "q12" else x


class NanoGPTQ(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(VOCAB, cfg.n_embed)
        nn.init.normal_(self.emb.weight, std=0.3)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        ew = fq_emb(self.emb.weight)
        x = ew[idx]
        if self.cfg.act_quant == "q12":
            x = fq_act_q12(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ ew.T


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

TENSORS = ("wq", "wk", "wv", "wo", "wff1", "wff2")


def _tensor_of(block: Block, name: str) -> torch.Tensor:
    if name in ("wq", "wk", "wv", "wo"):
        return getattr(block.attn, name).weight
    return getattr(block, name).weight


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    """Bit-pack signed codes LSB-first. Ternary uses 2 bits (4/byte)."""
    flat = codes.reshape(-1).astype(np.int64)
    lo = -(1 << (bits - 1))
    u = (flat - lo).astype(np.uint64)          # shift to unsigned
    if (u >> np.uint64(bits)).any():
        raise ValueError("code out of range for the requested bit width")
    if bits == 8:
        return u.astype(np.uint8).tobytes()
    per = 8 // bits
    if 8 % bits:                                # 3, 5, 6 bits -> pack into a bitstream
        nbits = flat.size * bits
        out = np.zeros((nbits + 7) // 8, dtype=np.uint8)
        bitpos = np.arange(flat.size, dtype=np.int64) * bits
        for k in range(bits):
            p = bitpos + k
            np.bitwise_or.at(out, p >> 3,
                             (((u >> np.uint64(k)) & np.uint64(1)).astype(np.uint8) << (p & 7).astype(np.uint8)))
        return out.tobytes()
    pad = (-flat.size) % per
    if pad:
        u = np.concatenate([u, np.zeros(pad, np.uint64)])
    g = u.reshape(-1, per).astype(np.uint8)
    out = np.zeros(g.shape[0], np.uint8)
    for k in range(per):
        out |= (g[:, k] << (k * bits)).astype(np.uint8)
    return out.tobytes()


def export_blob(model: NanoGPTQ, tag: str, outdir: str) -> Dict[str, object]:
    """Write (a) an npz of int codes + fp16 scales for the numpy reference and
    (b) a bit-packed binary blob whose length is MEASURED, not estimated."""
    cfg = model.cfg
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    scheme = cfg.quant
    bits = {"ternary": 2, "ternary_bn": 2, "none": 32}.get(scheme, 0) or int(scheme[3:])

    with torch.no_grad():
        ew = model.emb.weight.detach().cpu().float()
        mx = float(ew.abs().max().clamp(min=1e-6))
        x16 = int(max(round(min(mx, 255.0 / 16.0) * 16.0), 1))
        stored = x16 / 16.0
        emb_q = np.clip(np.round(ew.numpy() / stored * 127.0), -128, 127).astype(np.int8)

        store: Dict[str, np.ndarray] = {"emb_q": emb_q}
        blob = bytearray()
        blob += struct.pack("<IBHBHBB", MAGIC_LE, cfg.n_layers, cfg.n_embed,
                            cfg.n_heads, VOCAB, cfg.ctx, x16)
        blob += emb_q.tobytes()

        for li, blk in enumerate(model.blocks):
            per_layer_codes, per_layer_scales = [], []
            for name in TENSORS:
                w = _tensor_of(blk, name).detach().float()
                if scheme == "none":
                    q = w.cpu().numpy()
                    s = np.ones((w.shape[0], w.shape[1] // Q_BLOCK, 1), np.float32)
                elif scheme == "ternary":
                    qc, sc = q_codes_ternary(w, cfg.tau)
                    q, s = qc.cpu().numpy(), sc.cpu().numpy()
                elif scheme == "ternary_bn":
                    qc, sc = q_codes_ternary_bn(w)
                    q, s = qc.cpu().numpy(), sc.cpu().numpy()
                else:
                    qc, sc = q_codes_intn(w, bits)
                    q, s = qc.cpu().numpy(), sc.cpu().numpy()
                store[f"L{li}_{name}_q"] = q.astype(np.float32) if scheme == "none" else q.astype(np.int8)
                store[f"L{li}_{name}_s"] = s.astype(np.float16).reshape(s.shape[0], -1)
                per_layer_codes.append(q)
                per_layer_scales.append(s)
            if scheme != "none":
                for q in per_layer_codes:
                    blob += pack_codes(q.astype(np.int64), bits)
                for s in per_layer_scales:
                    blob += s.astype(np.float16).tobytes()
            else:
                for q in per_layer_codes:
                    blob += q.astype(np.float32).tobytes()

    npz_path = os.path.join(outdir, f"{tag}.npz")
    bin_path = os.path.join(outdir, f"{tag}.bin")
    np.savez(npz_path, **store)
    with open(bin_path, "wb") as fh:
        fh.write(blob)
    return {"npz": npz_path, "bin": bin_path, "blob_bytes": len(blob),
            "em_scale_x16": x16, "em_scale": stored, "bits": bits}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_v7(min_answers_to_hold: int = 3, corpus_hold_every: int = 10):
    """Build a REAL held-out split of the v7 corpus.

    v7 ships no split at all; `train_sophia_v9_qat.build_v7_corpus` made "val" an
    independent reshuffle of the SAME lines, so v7 val == v7 fit by construction
    and val loss carried no generalization signal whatsoever.

    A conventional random holdout is not available here: the corpus is 400
    distinct lines and the game asks 12 specific questions whose answers must
    stay in training.  So the split is a PARAPHRASE holdout:

      * QA/IDENTITY lines are grouped by their key ("What is the G4?: ").  For a
        key with >= `min_answers_to_hold` distinct answers, the LAST answer (list
        order -- deterministic, no RNG) is moved to validation.  Every key keeps
        at least two answers in training, so no game prompt loses its answer.
      * CORPUS_LINES have no key, so every `corpus_hold_every`-th line is held
        out.

    The validation corpus is therefore made ONLY of strings the model has never
    seen, while the task the game exercises is fully covered by training.  Val
    loss then measures "did it learn the shape of an answer for this key" rather
    than "did it memorize this exact string".
    """
    lists = E12.v7_lists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "train_sophia_v7.py"))
    train, held = {"IDENTITY_PAIRS": [], "QA_PAIRS": [], "CORPUS_LINES": []}, []
    stats = {}
    for name in ("IDENTITY_PAIRS", "QA_PAIRS"):
        by_key = {}
        for line in lists[name]:
            by_key.setdefault(line.partition(": ")[0] + ": ", []).append(line)
        nheld = 0
        for _key, lines in by_key.items():
            if len(lines) >= min_answers_to_hold:
                train[name].extend(lines[:-1])
                held.append(lines[-1])
                nheld += 1
            else:
                train[name].extend(lines)
        stats[name] = {"lines": len(lists[name]), "keys": len(by_key),
                       "held": nheld, "train_lines": len(train[name])}
    for i, line in enumerate(lists["CORPUS_LINES"]):
        (held if (i % corpus_hold_every == corpus_hold_every - 1) else train["CORPUS_LINES"]).append(line)
    stats["CORPUS_LINES"] = {"lines": len(lists["CORPUS_LINES"]), "keys": 0,
                             "held": len(lists["CORPUS_LINES"]) - len(train["CORPUS_LINES"]),
                             "train_lines": len(train["CORPUS_LINES"])}
    stats["held_total"] = len(held)
    return train, held, stats


def _weave(lines_by_group, reps, seed):
    rng = random.Random(seed)
    out = []
    for name, r in reps.items():
        for _ in range(r):
            ls = list(lines_by_group[name])
            rng.shuffle(ls)
            out.extend(ls)
    rng.shuffle(out)
    return ("\n".join(out) + "\n").encode("ascii", "replace")


def build_v7_corpus(seed: int, reps=(2800, 800, 400), holdout=True,
                    min_answers_to_hold=3):
    """(train_bytes, val_bytes, stats). `reps` = (IDENTITY, QA, CORPUS)."""
    if holdout:
        train, held, stats = split_v7(min_answers_to_hold)
    else:
        L = E12.v7_lists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "train_sophia_v7.py"))
        train, held, stats = L, [], {"held_total": 0}
    rmap = {"IDENTITY_PAIRS": reps[0], "QA_PAIRS": reps[1], "CORPUS_LINES": reps[2]}
    tr = _weave(train, rmap, seed)
    # the val corpus is unseen strings only; repeat enough to sample ctx windows
    if held:
        va = _weave({"HELD": held}, {"HELD": 400}, seed + 7777)
    else:
        va = tr
    stats["mass"] = {k: len(train[k]) * rmap[k] for k in rmap}
    stats["train_bytes"] = len(tr)
    stats["val_bytes"] = len(va)
    return tr, va, stats


class Data:
    def __init__(self, payload: bytes, ctx: int):
        self.arr = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
        self.ctx = ctx

    def batch(self, bs: int, rng: np.random.Generator):
        idx = rng.integers(0, self.arr.size - self.ctx - 1, size=bs)
        x = np.stack([self.arr[i:i + self.ctx] for i in idx])
        y = np.stack([self.arr[i + 1:i + self.ctx + 1] for i in idx])
        return (torch.from_numpy(x).long().to(device),
                torch.from_numpy(y).long().to(device))


@torch.no_grad()
def est_loss(model, data: Data, bs: int, batches: int, rng) -> float:
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = data.batch(bs, rng)
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    model.train()
    return tot / max(batches, 1)


@torch.no_grad()
def gen_greedy(model, prompts: Sequence[str], n_new: int, lo=32, hi=126) -> List[str]:
    """Batched greedy decode with the ROM's printable-ASCII-only argmax.

    This is the fake-quantized forward pass, so what is scored is what the
    exported blob computes (the STE makes forward == quantized weights)."""
    model.eval()
    ctx = model.cfg.ctx
    enc = [list(p.encode("ascii", "replace")) for p in prompts]
    L0 = max(len(e) for e in enc)
    # left-pad is wrong for a causal char model; decode each length group apart
    outs = [""] * len(prompts)
    by_len: Dict[int, List[int]] = {}
    for i, e in enumerate(enc):
        by_len.setdefault(len(e), []).append(i)
    for ln, idxs in by_len.items():
        seq = torch.tensor([enc[i] for i in idxs], dtype=torch.long, device=device)
        gen = []
        for _ in range(n_new):
            logits = model(seq[:, -ctx:])[:, -1, :]
            nxt = logits[:, lo:hi + 1].argmax(-1) + lo
            gen.append(nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            if seq.shape[1] > ctx:
                break
        g = torch.stack(gen, 1).cpu().numpy()
        for j, i in enumerate(idxs):
            outs[i] = bytes(int(c) for c in g[j]).decode("ascii", "replace")
    model.train()
    del L0
    return outs


def score_keys(model, keys: Sequence[str], bank, n_new: int):
    gens = gen_greedy(model, list(keys), n_new)
    hits = [E12.score_prompt(k, g, bank)[0] for k, g in zip(keys, gens)]
    return sum(hits), gens, hits


def all_keys():
    """Every distinct QA/IDENTITY key in v7, split by category."""
    L = E12.v7_lists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "train_sophia_v7.py"))
    def ks(name):
        out = []
        for line in L[name]:
            k = line.partition(": ")[0] + ": "
            if k not in out:
                out.append(k)
        return out
    return ks("IDENTITY_PAIRS"), ks("QA_PAIRS")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quant", default="int8",
                   choices=["none", "int3", "int4", "int5", "int6", "int8", "ternary", "ternary_bn"])
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--kv-quant", default="int8", choices=["none", "int8"])
    p.add_argument("--act-quant", default="none", choices=["none", "q12"])
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-embed", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--train-repeats", type=int, default=220)
    p.add_argument("--eval-repeats", type=int, default=12)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-bs", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--select", default="final", choices=["final", "best"],
                   help="which weights to export; 'final' keeps every scheme at an "
                        "identical step budget, which is the fairer comparison")
    p.add_argument("--reps", default="2800,800,400",
                   help="replication mix IDENTITY,QA,CORPUS (v7 default 2800,800,400)")
    p.add_argument("--no-holdout", action="store_true",
                   help="train on every line (val then equals fit; dishonest, for ablation only)")
    p.add_argument("--min-answers-to-hold", type=int, default=3)
    p.add_argument("--gen-tokens", type=int, default=48)
    p.add_argument("--tag", default=None)
    p.add_argument("--outdir", default="mix_out")
    args = p.parse_args()

    tag = args.tag or args.quant
    set_seed(args.seed)

    cfg = ModelCfg(n_layers=args.n_layers, n_embed=args.n_embed, n_heads=args.n_heads,
                   ffn_mult=args.ffn_mult, ctx=args.ctx, quant=args.quant, tau=args.tau,
                   kv_quant=args.kv_quant, act_quant=args.act_quant)

    reps = tuple(int(v) for v in args.reps.split(","))
    assert len(reps) == 3, "--reps needs IDENTITY,QA,CORPUS"
    train_corpus, eval_corpus, split_stats = build_v7_corpus(
        args.seed, reps, holdout=not args.no_holdout,
        min_answers_to_hold=args.min_answers_to_hold)
    train_data, val_data = Data(train_corpus, cfg.ctx), Data(eval_corpus, cfg.ctx)
    print(f"[{tag}] reps={reps} split={json.dumps(split_stats)}", flush=True)

    bank = E12.answer_bank(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "train_sophia_v7.py"))[0]
    game12 = [k for _npc, k in E12.GAME_12]
    id_keys, qa_keys = all_keys()

    model = NanoGPTQ(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    n_quant = sum(_tensor_of(b, n).numel() for b in model.blocks for n in TENSORS)
    print(f"[{tag}] device={device} params={n_params:,} quantized_weights={n_quant:,} "
          f"cfg={asdict(cfg)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    def lr_lambda(i: int) -> float:
        if i < args.warmup:
            return (i + 1) / max(args.warmup, 1)
        pr = min(1.0, max(0.0, (i - args.warmup) / max(args.steps - args.warmup, 1)))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * pr))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng_t = np.random.default_rng(args.seed)
    rng_v = np.random.default_rng(args.seed + 999)

    best_val, best_step, best_state = float("inf"), 0, None
    final_val, final_fit = float("nan"), float("nan")
    hist: List[Dict[str, float]] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = train_data.batch(args.batch_size, rng_t)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            # identical fixed batches every time, for both corpora
            vl = est_loss(model, val_data, args.val_bs, args.val_batches,
                          np.random.default_rng(args.seed + 999))
            fit = est_loss(model, train_data, args.val_bs, args.val_batches,
                           np.random.default_rng(args.seed + 555))
            final_val, final_fit = vl, fit
            if vl < best_val:
                best_val, best_step = vl, step
                best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
            g12, _gg, _hh = score_keys(model, game12, bank, args.gen_tokens)
            hist.append({"step": step, "train": float(loss.item()), "val": float(vl),
                         "fit": float(fit), "game12": g12})
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                el = time.time() - t0
                print(f"[{tag}] step {step:6d}/{args.steps} train={loss.item():.4f} "
                      f"fit={fit:.4f} val={vl:.4f} best={best_val:.4f}@{best_step} "
                      f"GAME12={g12}/12 "
                      f"lr={opt.param_groups[0]['lr']:.2e} gn={float(gn):.2f} "
                      f"{el:.0f}s eta={(args.steps-step)/max(step/el,1e-9)/60:.1f}m", flush=True)

    final_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}

    # ---- score BOTH candidate checkpoints on the metric that matters --------
    scores = {}
    for which, st in (("final", final_state), ("best", best_state)):
        if st is None:
            continue
        model.load_state_dict(st)
        g12, gens12, hits12 = score_keys(model, game12, bank, args.gen_tokens)
        gi, gensi, hitsi = score_keys(model, id_keys, bank, args.gen_tokens)
        gq, gensq, hitsq = score_keys(model, qa_keys, bank, args.gen_tokens)
        scores[which] = {
            "game12": g12, "identity": gi, "identity_total": len(id_keys),
            "qa": gq, "qa_total": len(qa_keys),
            "game12_rows": [{"prompt": k, "gen": g, "hit": bool(h)}
                            for k, g, h in zip(game12, gens12, hits12)],
            "identity_rows": [{"prompt": k, "gen": g, "hit": bool(h)}
                              for k, g, h in zip(id_keys, gensi, hitsi)],
            "qa_rows": [{"prompt": k, "gen": g, "hit": bool(h)}
                        for k, g, h in zip(qa_keys, gensq, hitsq)],
        }
        print(f"[{tag}] select={which}: GAME12={g12}/12  IDENTITY={gi}/{len(id_keys)}  "
              f"QA={gq}/{len(qa_keys)}", flush=True)

    model.load_state_dict(best_state if args.select == "best" else final_state)
    model.eval()
    meta = export_blob(model, tag, args.outdir)
    torch.save({"state": final_state if args.select == "final" else best_state,
                "final_state": final_state, "best_state": best_state,
                "cfg": asdict(cfg), "args": vars(args)},
               os.path.join(args.outdir, f"{tag}.pt"))
    meta.update({"tag": tag, "quant": args.quant, "tau": args.tau,
                 "kv_quant": args.kv_quant, "act_quant": args.act_quant,
                 "select": args.select, "final_val": final_val, "final_fit": final_fit,
                 "best_val": best_val, "best_step": best_step, "steps": args.steps,
                 "params": n_params, "quantized_weights": n_quant,
                 "n_layers": cfg.n_layers, "n_embed": cfg.n_embed,
                 "n_heads": cfg.n_heads, "ffn_mult": cfg.ffn_mult, "ctx": cfg.ctx,
                 "seed": args.seed, "corpus": "v7", "reps": list(reps),
                 "split": split_stats, "holdout": not args.no_holdout,
                 "scores": scores,
                 "lr": args.lr, "batch_size": args.batch_size,
                 "weight_decay": args.weight_decay,
                 "train_seconds": round(time.time() - t0, 1), "history": hist})
    with open(os.path.join(args.outdir, f"{tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{tag}] DONE final_val={final_val:.4f} final_fit={final_fit:.4f} "
          f"best_val={best_val:.4f}@{best_step} "
          f"blob={meta['blob_bytes']:,}B in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()


def unpack_codes(buf: bytes, n: int, bits: int) -> np.ndarray:
    """Inverse of pack_codes(); used to prove the blob is really decodable."""
    lo = -(1 << (bits - 1))
    if bits == 8:
        return np.frombuffer(buf, np.uint8, n).astype(np.int64) + lo
    bitsarr = np.unpackbits(np.frombuffer(buf, np.uint8), bitorder="little")
    idx = np.arange(n)[:, None] * bits + np.arange(bits)[None, :]
    vals = (bitsarr[idx].astype(np.int64) * (1 << np.arange(bits))).sum(1)
    return vals + lo
