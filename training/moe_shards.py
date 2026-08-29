#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""moe_shards.py — split the v7 corpus into topic experts for the N64 MoE.

The split is NOT invented: it is the grouping `legend_of_elya.c`'s own
`PROMPTS[]` array already uses (identity / dungeon / RustChain / hardware /
N64 lore / Elya lore), collapsed to four shards of comparable size.

Matching is by plain lowercase SUBSTRING, never regex, for the same reason the
Genesis Lock-On router does it that way (`legend-of-elya-genesis/train/
experts.py`): the ROM router has to perform the identical match, and a regex
the ROM cannot run would let the training labels and the runtime routing
disagree silently. Order matters -- first shard whose keyword hits wins, so
specific topics precede general ones.

A line matching nothing lands in `identity`, the general shard; CORPUS_LINES
(prose with no question key) go to every shard, because they are what teaches
the model English rather than facts.
"""
SHARDS = [
    ("rustchain", [
        "rustchain", "rtc", "proof of antiquity", "proof of work", "epoch",
        "node", "mining", "mine ", "blockchain", "token", "wallet", "ledger",
        "antiquity", "attestation", "settle",
    ]),
    ("hardware", [
        "g4", "g5", "power8", "altivec", "vec_perm", "vr4300", "mips",
        "n64", "console", "cartridge", "cpu", "silicon", "chip", "ram",
        "expansion pak", "model", "weights", "kilobytes", "parameters",
        "language runs you", "runs this rom", "hardware", "simd", "threads",
        "powerpc", "amiga", "68k", "sparc", "vintage", "rsp", "rdram",
        "processor", "transistor", "clock speed", "megahertz", "mhz",
    ]),
    ("dungeon", [
        "lurks", "proceed", "need here", "secret", "dungeon", "wall", "crystal",
        "key", "door", "treasure", "ghost", "goblin", "spirit", "haunt",
        "triforce", "guard", "realm", "hero", "quest", "boss", "trap", "bomb",
        "lens", "gold", "tile", "shadow", "danger", "monster",
    ]),
    ("lore", [
        "zelda", "link", "ganon", "hyrule", "kokiri", "lon lon", "ocarina",
        "deku", "triforce", "epona", "navi", "saria", "sheik", "termina",
        "nintendo", "nes", "snes", "n64 game", "mario", "samus", "metroid",
        "sega", "genesis", "amiga game", "arcade", "cartridge game",
    ]),
    ("identity", []),          # the catch-all, deliberately last
]

def shard_of(line: str) -> str:
    low = line.lower()
    for name, keys in SHARDS:
        if not keys:
            continue
        if any(k in low for k in keys):
            return name
    return "identity"

def split_lists(lists: dict) -> dict:
    """{'IDENTITY_PAIRS':[...], 'QA_PAIRS':[...], 'CORPUS_LINES':[...]}
    -> {shard: {same three keys}}.  CORPUS_LINES are shared by every shard."""
    names = [n for n, _ in SHARDS]
    out = {n: {"IDENTITY_PAIRS": [], "QA_PAIRS": [], "CORPUS_LINES": list(lists["CORPUS_LINES"])}
           for n in names}
    for key in ("IDENTITY_PAIRS", "QA_PAIRS"):
        for line in lists[key]:
            out[shard_of(line)][key].append(line)
    return out

if __name__ == "__main__":
    import eval12 as E
    L = E.v7_lists()
    s = split_lists(L)
    tot = 0
    for n, d in s.items():
        q = len(d["IDENTITY_PAIRS"]) + len(d["QA_PAIRS"])
        tot += q
        print(f"{n:10s} {q:4d} QA lines  (+{len(d['CORPUS_LINES'])} shared prose)")
    print(f"{'total':10s} {tot:4d} of {len(L['IDENTITY_PAIRS'])+len(L['QA_PAIRS'])}")
