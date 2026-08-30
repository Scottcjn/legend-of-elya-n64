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


# ── Per-NPC experts ─────────────────────────────────────────────────────────
# One expert per CHARACTER, not per topic. Each NPC's corpus is the shared
# prose (which teaches English rather than facts) plus the topic shards that
# character actually knows about, so an expert is never starved down to one
# NPC's handful of QA lines -- which is the failure the design panel warned
# about and the reason each entry below lists more than one topic.
NPC_EXPERTS = [
    ("sophia",   ["identity", "dungeon"]),          # the guide: herself, and the realm
    ("aldric",   ["lore", "identity"]),             # the keeper: Zelda/Nintendo lore
    ("brunhild", ["hardware", "rustchain"]),        # the smith: silicon and the chain
]

def npc_lists(lists: dict, npc: str) -> dict:
    """The corpus for ONE named NPC: shared prose + their topics' QA lines."""
    topics = dict(NPC_EXPERTS).get(npc)
    if topics is None:
        raise SystemExit("unknown npc %r; have %s"
                         % (npc, [n for n, _ in NPC_EXPERTS]))
    parts = split_lists(lists)
    out = {"IDENTITY_PAIRS": [], "QA_PAIRS": [],
           "CORPUS_LINES": list(lists["CORPUS_LINES"])}
    for t in topics:
        out["IDENTITY_PAIRS"] += parts[t]["IDENTITY_PAIRS"]
        out["QA_PAIRS"]       += parts[t]["QA_PAIRS"]
    # Teach this NPC the commands the world actually grants them. Generated
    # from training/world_defs.py, the same file tools/gen_cmd_trie.py builds
    # the ROM's decoder from, so the model is never taught a command the trie
    # forbids -- and never learns one another NPC owns.
    try:
        import gen_cmd_corpus
        out["QA_PAIRS"] += gen_cmd_corpus.lines_for(npc, reps=2)
    except Exception as e:
        print("  (no command lines: %s)" % e)
    return out
