#!/usr/bin/env python3
"""Numpy reference evaluation for the N64 SEAI model: QAT vs post-training quant.

Extends $S/n64_ternary_probe.py. Two corrections against nano_gpt.c were made
after reading the C:

  1. nano_gpt.c:600 applies a final rms_norm before project_to_logits. The probe
     skipped it. For greedy decoding this is a no-op (positive scalar scale
     cannot move an argmax) but it is applied here for fidelity.
  2. nano_gpt.c:459-461 takes the greedy argmax over printable ASCII 32..126
     ONLY. The probe took argmax over all 256 bytes, which CAN change the token
     stream. This file matches the ROM (and the v8 trainer's own generation
     mask). Post-training numbers are therefore re-derived here rather than
     quoted, so PTQ and QAT are compared under identical conditions.

The float16 scale LUT is copied verbatim: do NOT byte-swap in Python.
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

QB = 32
TENSORS = ("wq", "wk", "wv", "wo", "wff1", "wff2")
SHIPPED = os.path.expanduser("~/legend-of-elya-n64/filesystem/sophia_weights.bin")


def _build_f16_lut():
    u = np.arange(65536, dtype=np.uint32)
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    frac = (u & 0x3FF).astype(np.float32)
    val = np.empty(65536, np.float32)
    sub = exp == 0
    val[sub] = (frac[sub] / 1024.0) * (1.0 / 16384.0)
    inf = exp == 31
    val[inf] = 65504.0
    nrm = ~(sub | inf)
    val[nrm] = (1.0 + frac[nrm] / 1024.0) * np.exp2((exp[nrm].astype(np.float32) - 15.0))
    return np.where(sign == 1, -val, val).astype(np.float32)


F16 = _build_f16_lut()


class Model:
    """Dequantized weights + geometry. `emb` is already scaled by em_scale/127."""

    def __init__(self, emb, layers, n_heads, ctx, name=""):
        self.emb = emb.astype(np.float32)
        self.layers = layers
        self.n_heads = n_heads
        self.ctx = ctx
        self.n_embed = emb.shape[1]
        self.head_dim = self.n_embed // n_heads
        self.name = name


def load_shipped(path=SHIPPED):
    d = open(path, "rb").read()
    magic, n_layers, n_embed, n_heads, vocab, ctx, x16 = struct.unpack("<IBHBHBB", d[:12])
    e, off = n_embed, 12
    emb = np.frombuffer(d, np.int8, vocab * e, off).reshape(vocab, e).astype(np.float32)
    off += vocab * e
    # nano_gpt.c:538-540 -- the shipped blob stores em_scale_x16 = 0, and the C
    # falls back to 3.5 for "old weight files". Reproduce that or the embedding
    # (and therefore every logit) comes out zero and greedy emits only spaces.
    em_scale = x16 / 16.0
    if em_scale < 0.01:
        em_scale = 3.5
    scale = em_scale / 127.0
    shapes = [("wq", e, e), ("wk", e, e), ("wv", e, e), ("wo", e, e),
              ("wff1", e * 4, e), ("wff2", e, e * 4)]
    layers = []
    for _ in range(n_layers):
        raw = {}
        for nm, od, idim in shapes:
            raw[nm] = np.frombuffer(d, np.int8, od * idim, off).reshape(od, idim).astype(np.float32)
            off += od * idim
        t = {}
        for nm, od, idim in shapes:
            s = F16[np.frombuffer(d, "<u2", od * idim // QB, off).reshape(od, idim // QB)]
            off += (od * idim // QB) * 2
            t[nm] = raw[nm] * np.repeat(s, QB, axis=1)
        layers.append(t)
    assert off == len(d), (off, len(d))
    return Model(emb * scale, layers, n_heads, ctx, name="shipped-Q8")


def load_qat(npz_path, meta_path=None):
    meta_path = meta_path or npz_path.replace(".npz", "_meta.json")
    meta = json.load(open(meta_path))
    z = np.load(npz_path)
    scale = meta["em_scale"] / 127.0
    emb = z["emb_q"].astype(np.float32) * scale
    layers = []
    for li in range(meta["n_layers"]):
        t = {}
        for nm in TENSORS:
            q = z[f"L{li}_{nm}_q"].astype(np.float32)
            q = q.reshape(q.shape[0], -1)            # (out, nb, 32) -> (out, in)
            s = z[f"L{li}_{nm}_s"].astype(np.float32)
            t[nm] = q * np.repeat(s, QB, axis=1)
        layers.append(t)
    return Model(emb, layers, meta["n_heads"], meta["ctx"], name=meta["tag"]), meta


# ---------------------------------------------------------------- PTQ schemes

def ptq_intn(wf, bits):
    od, idim = wf.shape
    b = wf.reshape(od, idim // QB, QB)
    qmax = (1 << (bits - 1)) - 1
    s = np.maximum(np.abs(b).max(axis=2, keepdims=True) / qmax, 1e-12).astype(np.float16).astype(np.float32)
    q = np.clip(np.round(b / s), -qmax - 1, qmax)
    return (q * s).reshape(od, idim)


def ptq_ternary(wf, tau):
    od, idim = wf.shape
    b = wf.reshape(od, idim // QB, QB)
    ab = np.abs(b)
    delta = tau * ab.mean(axis=2, keepdims=True)
    m = (ab > delta).astype(np.float32)
    alpha = ((ab * m).sum(2, keepdims=True) / np.maximum(m.sum(2, keepdims=True), 1)).astype(np.float16).astype(np.float32)
    return (np.sign(b) * m * np.maximum(alpha, 1e-12)).reshape(od, idim)


def ptq_ternary_bn(wf, _tau=0.0):
    od, idim = wf.shape
    b = wf.reshape(od, idim // QB, QB)
    s = np.maximum(np.abs(b).mean(axis=2, keepdims=True), 1e-12).astype(np.float16).astype(np.float32)
    return (np.clip(np.round(b / s), -1, 1) * s).reshape(od, idim)


def apply_ptq(model, scheme, tau=0.7):
    if scheme == "none":
        return model
    fn = {"ternary": lambda w: ptq_ternary(w, tau),
          "ternary_bn": ptq_ternary_bn}.get(scheme)
    if fn is None:
        bits = int(scheme[3:])
        fn = lambda w, b=bits: ptq_intn(w, b)
    layers = [{nm: fn(L[nm]) for nm in TENSORS} for L in model.layers]
    return Model(model.emb, layers, model.n_heads, model.ctx, name=f"{model.name}+PTQ-{scheme}")


# -------------------------------------------------------------- forward pass

def rms(v):
    return v / np.sqrt((v * v).mean() + 1e-6)


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def q_int8_vec(v):
    """Dynamic absmax int8 for one KV vector (per head slice)."""
    s = np.abs(v).max() / 127.0
    if s <= 0:
        return v
    return np.round(v / s).clip(-127, 127) * s


def generate(m: Model, prompt_bytes, n_new, kv_int8=False, kv_per_head=True,
             greedy_lo=32, greedy_hi=126):
    E, H, HD, CTX = m.n_embed, m.n_heads, m.head_dim, m.ctx
    nL = len(m.layers)
    k = np.zeros((nL, CTX, E), np.float32)
    v = np.zeros((nL, CTX, E), np.float32)
    out, seq = [], list(prompt_bytes)
    for pos in range(len(seq) + n_new - 1):
        if pos >= CTX:
            break
        tok = seq[pos] if pos < len(seq) else out[-1]
        x = m.emb[tok].copy()
        for li, L in enumerate(m.layers):
            res = x
            xn = rms(x)
            q = L["wq"] @ xn
            kv_k = L["wk"] @ xn
            kv_v = L["wv"] @ xn
            if kv_int8:
                if kv_per_head:
                    for h in range(H):
                        sl = slice(h * HD, (h + 1) * HD)
                        kv_k[sl] = q_int8_vec(kv_k[sl])
                        kv_v[sl] = q_int8_vec(kv_v[sl])
                else:
                    kv_k = q_int8_vec(kv_k)
                    kv_v = q_int8_vec(kv_v)
            k[li, pos] = kv_k
            v[li, pos] = kv_v
            n_ctx = pos + 1
            ao = np.zeros(E, np.float32)
            for h in range(H):
                sl = slice(h * HD, (h + 1) * HD)
                s = (k[li, :n_ctx, sl] @ q[sl]) * (HD ** -0.5)
                ao[sl] = softmax(s) @ v[li, :n_ctx, sl]     # PSE conductance pinned to 1.0
            x = res + L["wo"] @ ao
            res = x
            xn = rms(x)
            x = res + L["wff2"] @ np.maximum(L["wff1"] @ xn, 0.0)
        logits = m.emb @ rms(x)                              # nano_gpt.c:600-607
        if pos >= len(seq) - 1:
            band = logits[greedy_lo:greedy_hi + 1]
            out.append(int(np.argmax(band)) + greedy_lo)
    return out


def show(b):
    return "".join(chr(c) if 32 <= c < 127 else f"\\x{c:02x}" for c in b)


def blob_bytes(n_quant_weights, bits, vocab, n_embed, ctx=None):
    return int(np.ceil(n_quant_weights * bits / 8) + (n_quant_weights // QB) * 2
               + vocab * n_embed + 12)


RAW_PROMPTS = [b"Elya", b"Who are you?", b"The dungeon"]
PERSONA_PROMPTS = [
    b"<|sophia|>\nPlayer: Who are you?\nNPC:",
    b"<|sophia|>\nPlayer: The dungeon is dark.\nNPC:",
    b"<|guard|>\nPlayer: Who are you?\nNPC:",
]
