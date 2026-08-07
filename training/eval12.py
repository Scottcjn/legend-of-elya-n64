#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""eval12.py — score a Sophia blob/checkpoint on THE 12 GAME PROMPTS.

The metric this project is actually optimizing.  `legend_of_elya.c`'s
NPC_DIALOG_OPTIONS[3][4] is the ground truth list; every one of the 12 is a
verbatim key in train_sophia_v7.py, so the "correct" answers are the trained
completions for that key.

Decoding matches the ROM (nano_gpt.c sample_logits with temperature_q8 == 0):
greedy argmax over printable ASCII 32..126 only.  Newline (10) is NOT in that
band, so the model cannot emit a line break; a generation therefore runs on into
whatever line came next in the corpus.  Correctness is judged as
"the generation STARTS WITH one of the trained answers for this key".

Sources:
  --shipped PATH        SEAI int8 blob (the shipped format)
  --seq PATH            SEQn blob (tools/qat_npz_to_seq.py output)
  --npz PATH            QAT npz + _meta.json (the numpy oracle's own view)
  --pt PATH             a torch checkpoint from train_mix.py (float, unquantized)
"""
import argparse
import ast
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QB = 32
TENSORS = ("wq", "wk", "wv", "wo", "wff1", "wff2")

# ---------------------------------------------------------------- the 12
# legend_of_elya.c NPC_DIALOG_OPTIONS[NPC_COUNT=3][DIALOG_OPTIONS=4]
GAME_12 = [
    ("Sophia",   "Who are you?: "),
    ("Sophia",   "What lurks here?: "),
    ("Sophia",   "Tell me a secret.: "),
    ("Sophia",   "What is RustChain?: "),
    ("Aldric",   "What is your name?: "),
    ("Aldric",   "What is proof of antiquity?: "),
    ("Aldric",   "What is MIPS?: "),
    ("Aldric",   "How big is your model?: "),
    ("Brunhild", "What do I need here?: "),
    ("Brunhild", "What is the G4?: "),
    ("Brunhild", "What is AltiVec?: "),
    ("Brunhild", "What is vec_perm?: "),
]


def v7_lists(path=None):
    path = path or os.path.join(HERE, "train_sophia_v7.py")
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm in ("IDENTITY_PAIRS", "QA_PAIRS", "CORPUS_LINES"):
                out[nm] = ast.literal_eval(node.value)
    return out


def answer_bank(path=None):
    """key ('Who are you?: ') -> [trained answers], and which list it came from."""
    L = v7_lists(path)
    bank, origin = {}, {}
    for name in ("IDENTITY_PAIRS", "QA_PAIRS"):
        for line in L[name]:
            k, _, a = line.partition(": ")
            key = k + ": "
            bank.setdefault(key, []).append(a)
            origin.setdefault(key, name)
    return bank, origin


# ---------------------------------------------------------------- f16 LUT
def _f16lut():
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


F16 = _f16lut()


class Model:
    def __init__(self, emb, layers, n_heads, ctx, name="", em_scale_x16=None):
        self.emb = emb.astype(np.float32)
        self.layers = layers
        self.n_heads = n_heads
        self.ctx = ctx
        self.n_embed = emb.shape[1]
        self.head_dim = self.n_embed // n_heads
        self.name = name
        self.em_scale_x16 = em_scale_x16


def _shapes(e):
    return [("wq", e, e), ("wk", e, e), ("wv", e, e), ("wo", e, e),
            ("wff1", e * 4, e), ("wff2", e, e * 4)]


def load_seai(path):
    """int8 SEAI blob, decoded exactly as nano_gpt.c does (incl. the 3.5 fallback)."""
    d = open(path, "rb").read()
    magic, nL, e, H, vocab, ctx, x16 = struct.unpack("<IBHBHBB", d[:12])
    assert magic == 0x49414553, hex(magic)
    off = 12
    emb = np.frombuffer(d, np.int8, vocab * e, off).reshape(vocab, e).astype(np.float32)
    off += vocab * e
    em_scale = x16 / 16.0
    fallback = em_scale < 0.01
    if fallback:
        em_scale = 3.5
    layers = []
    for _ in range(nL):
        raw = {}
        for nm, od, idim in _shapes(e):
            raw[nm] = np.frombuffer(d, np.int8, od * idim, off).reshape(od, idim).astype(np.float32)
            off += od * idim
        t = {}
        for nm, od, idim in _shapes(e):
            s = F16[np.frombuffer(d, "<u2", od * idim // QB, off).reshape(od, idim // QB)]
            off += (od * idim // QB) * 2
            t[nm] = raw[nm] * np.repeat(s, QB, axis=1)
        layers.append(t)
    assert off == len(d), (off, len(d))
    m = Model(emb * (em_scale / 127.0), layers, H, ctx,
              name=os.path.basename(path) + (" [x16=0 -> 3.5 fallback]" if fallback else ""),
              em_scale_x16=x16)
    return m


def load_seq(path):
    """SEQn blob, decoded with the ROM's rules: MSB-first, two's complement."""
    d = open(path, "rb").read()
    if d[:3] != b"SEQ":
        raise SystemExit("not SEQn: %r" % d[:4])
    bits = d[3] - ord("0")
    nL, e, H, vocab, ctx, x16 = struct.unpack("<BHBHBB", d[4:12])
    off = 12
    emb = np.frombuffer(d, np.int8, vocab * e, off).reshape(vocab, e).astype(np.float32)
    off += vocab * e
    em_scale = x16 / 16.0
    if em_scale < 0.01:
        em_scale = 3.5
    layers = []
    for _ in range(nL):
        raw = {}
        for nm, od, idim in _shapes(e):
            n = od * idim
            nb = n * bits // 8
            pk = np.frombuffer(d, np.uint8, nb, off)
            off += nb
            b = np.unpackbits(pk)[: n * bits].reshape(n, bits).astype(np.int32)
            w = (1 << np.arange(bits - 1, -1, -1)).astype(np.int32)
            code = (b * w).sum(axis=1)
            msb = 1 << (bits - 1)
            raw[nm] = ((code ^ msb) - msb).reshape(od, idim).astype(np.float32)
        t = {}
        for nm, od, idim in _shapes(e):
            s = F16[np.frombuffer(d, "<u2", od * idim // QB, off).reshape(od, idim // QB)]
            off += (od * idim // QB) * 2
            t[nm] = raw[nm] * np.repeat(s, QB, axis=1)
        layers.append(t)
    assert off == len(d), (off, len(d))
    return Model(emb * (em_scale / 127.0), layers, H, ctx,
                 name=os.path.basename(path), em_scale_x16=x16)


def load_npz(npz_path, meta_path=None):
    meta = json.load(open(meta_path or npz_path.replace(".npz", "_meta.json")))
    z = np.load(npz_path)
    emb = z["emb_q"].astype(np.float32) * (meta["em_scale"] / 127.0)
    layers = []
    for li in range(meta["n_layers"]):
        t = {}
        for nm in TENSORS:
            q = z[f"L{li}_{nm}_q"].astype(np.float32)
            q = q.reshape(q.shape[0], -1)
            s = z[f"L{li}_{nm}_s"].astype(np.float32)
            t[nm] = q * np.repeat(s, QB, axis=1)
        layers.append(t)
    return Model(emb, layers, meta["n_heads"], meta["ctx"],
                 name=meta["tag"], em_scale_x16=meta["em_scale_x16"]), meta


def load_pt(path, which="final"):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    st = ck.get(f"{which}_state") or ck["state"]
    emb = st["emb.weight"].float().numpy()
    layers = []
    for li in range(cfg["n_layers"]):
        t = {}
        for nm in TENSORS:
            pre = f"blocks.{li}.attn.{nm}.weight" if nm in ("wq", "wk", "wv", "wo") \
                else f"blocks.{li}.{nm}.weight"
            t[nm] = st[pre].float().numpy()
        layers.append(t)
    return Model(emb, layers, cfg["n_heads"], cfg["ctx"],
                 name=os.path.basename(path) + f"[{which},float]"), cfg


# ---------------------------------------------------------------- forward
def rms(v):
    return v / np.sqrt((v * v).mean() + 1e-6)


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def generate(m, prompt_bytes, n_new, lo=32, hi=126):
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
            k[li, pos] = L["wk"] @ xn
            v[li, pos] = L["wv"] @ xn
            n_ctx = pos + 1
            ao = np.zeros(E, np.float32)
            for h in range(H):
                sl = slice(h * HD, (h + 1) * HD)
                s = (k[li, :n_ctx, sl] @ q[sl]) * (HD ** -0.5)
                ao[sl] = softmax(s) @ v[li, :n_ctx, sl]
            x = res + L["wo"] @ ao
            res = x
            xn = rms(x)
            x = res + L["wff2"] @ np.maximum(L["wff1"] @ xn, 0.0)
        logits = m.emb @ rms(x)
        if pos >= len(seq) - 1:
            out.append(int(np.argmax(logits[lo:hi + 1])) + lo)
    return bytes(out).decode("ascii", "replace")


# ---------------------------------------------------------------- scoring
def game_cut(gen):
    """What the PLAYER actually sees.  legend_of_elya.c:update_generating_step
    terminates the line on the first '.' once >= 8 output characters have been
    emitted (and on '\\n', which the sampler can never produce), and does NOT
    append that '.'.  So the trailing greedy degeneration is never displayed."""
    for i, c in enumerate(gen):
        if c == "." and i + 1 >= 8:
            return gen[:i]
    return gen


def score_prompt(key, gen, bank):
    """Return (hit, matched_answer). hit iff gen starts with a trained answer."""
    best = None
    for a in bank.get(key, []):
        if gen.startswith(a):
            if best is None or len(a) > len(best):
                best = a
    return (best is not None), best


def run(m, n_new=48, bank=None, quiet=False):
    bank = bank or answer_bank()[0]
    rows, hits = [], 0
    for npc, key in GAME_12:
        g = generate(m, key.encode("ascii"), n_new)
        ok, matched = score_prompt(key, g, bank)
        hits += ok
        rows.append({"npc": npc, "prompt": key, "gen": g, "shown": game_cut(g),
                     "hit": bool(ok), "matched": matched})
        if not quiet:
            print(f"  [{'HIT ' if ok else 'MISS'}] {npc:9s} {key:32s} -> {game_cut(g)!r}"
                  f"   (raw {g!r})")
    return hits, rows


def run_identity(m, n_new=40, bank=None, quiet=False):
    """Every distinct IDENTITY_PAIRS key, scored the same way."""
    bank = bank or answer_bank()[0]
    L = v7_lists()
    keys = []
    for line in L["IDENTITY_PAIRS"]:
        k = line.partition(": ")[0] + ": "
        if k not in keys:
            keys.append(k)
    rows, hits = [], 0
    for key in keys:
        g = generate(m, key.encode("ascii"), n_new)
        ok, matched = score_prompt(key, g, bank)
        hits += ok
        rows.append({"prompt": key, "gen": g, "hit": bool(ok), "matched": matched})
        if not quiet:
            print(f"  [{'HIT ' if ok else 'MISS'}] {key:28s} -> {g!r}")
    return hits, len(keys), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped")
    ap.add_argument("--seq")
    ap.add_argument("--npz")
    ap.add_argument("--pt")
    ap.add_argument("--which", default="final", choices=["final", "best"])
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    if a.shipped:
        m = load_seai(a.shipped)
    elif a.seq:
        m = load_seq(a.seq)
    elif a.npz:
        m, _ = load_npz(a.npz)
    elif a.pt:
        m, _ = load_pt(a.pt, a.which)
    else:
        raise SystemExit("need one of --shipped/--seq/--npz/--pt")

    bank = answer_bank()[0]
    print(f"== {m.name}  nL={len(m.layers)} E={m.n_embed} H={m.n_heads} "
          f"ctx={m.ctx} em_scale_x16={m.em_scale_x16}")
    print("-- 12 GAME PROMPTS --")
    hits, rows = run(m, a.n_new, bank)
    print(f"SCORE: {hits}/12")
    res = {"name": m.name, "game12": hits, "rows": rows, "em_scale_x16": m.em_scale_x16}
    if a.identity:
        print("-- IDENTITY KEYS --")
        ih, itot, irows = run_identity(m, 40, bank)
        print(f"IDENTITY: {ih}/{itot}")
        res.update({"identity": ih, "identity_total": itot, "identity_rows": irows})
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
