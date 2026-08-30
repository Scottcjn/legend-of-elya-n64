#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""world_defs.py — the single source of truth for what NPCs can DO.

Everything downstream is generated from this file: the ROM's command trie
(tools/gen_cmd_trie.py -> cmd_trie.h), the training lines that teach the model
to emit commands (training/gen_cmd_corpus.py), and the game-side validator.
That is the same discipline moe_router.h follows, for the same reason -- two
hand-maintained copies of the world is not a style problem, it is the bug.

A command is emitted IN BAND with the dialogue, wrapped in two control bytes
that the display band (ASCII 32..126) already discards:

    0x01 verb ' ' arg 0x02          e.g.  \\x01give lamp\\x02

The bytes are invisible to the textbox by construction, exactly as the
newline end-of-answer was (C028) -- the game routes them to the command
parser instead of the dialogue buffer.
"""

# Items an NPC can hand over. Keep the words inside the v7 vocabulary where
# possible so the model's existing English carries the command.
ITEMS = [
    "lamp",      # lights the dungeon
    "key",       # opens the locked door
    "map",       # reveals the room graph
    "gem",       # the forge's payment
    "book",      # Aldric's lore volume
]

# Rooms, matching legend_of_elya.c's RoomID order exactly.
ROOMS = ["dungeon", "library", "forge"]

# Quest ids and the states a command may set.
QUESTS = ["lantern", "ledger", "relic"]
QUEST_STATES = ["start", "done"]

# verb -> ordered list of argument classes
VERBS = {
    "give":  ["item"],       # NPC hands the player an item
    "open":  ["room"],       # NPC opens the way to a room
    "quest": ["quest", "state"],
    "recall":["item"],       # NPC remembers the player has something
}

CLASSES = {"item": ITEMS, "room": ROOMS, "quest": QUESTS, "state": QUEST_STATES}

# Which NPC may issue which verbs, and with which arguments. A command the
# world does not grant is never trained and never reachable in the trie.
NPC_GRANTS = {
    "sophia":   {"give": ["lamp", "map"],  "open": ["library", "forge"],
                 "quest": (["lantern"], QUEST_STATES), "recall": ITEMS},
    "aldric":   {"give": ["book", "key"],  "open": ["dungeon", "forge"],
                 "quest": (["ledger"],  QUEST_STATES), "recall": ITEMS},
    "brunhild": {"give": ["gem"],          "open": ["dungeon", "library"],
                 "quest": (["relic"],   QUEST_STATES), "recall": ITEMS},
}

CMD_START, CMD_END = 0x01, 0x02


def commands_for(npc):
    """Every legal command string for one NPC, as (verb, args) tuples."""
    out = []
    g = NPC_GRANTS[npc]
    for verb, spec in g.items():
        if verb == "quest":
            ids, states = spec
            for q in ids:
                for st in states:
                    out.append((verb, [q, st]))
        else:
            for a in spec:
                out.append((verb, [a]))
    return out


def all_commands():
    seen, out = set(), []
    for npc in NPC_GRANTS:
        for verb, args in commands_for(npc):
            k = (verb, tuple(args))
            if k not in seen:
                seen.add(k); out.append((verb, args))
    return out


def render(verb, args):
    return "\x01" + verb + " " + " ".join(args) + "\x02"


if __name__ == "__main__":
    total = all_commands()
    print(f"{len(total)} distinct commands across {len(NPC_GRANTS)} NPCs")
    for npc in NPC_GRANTS:
        cs = commands_for(npc)
        print(f"  {npc:9s} {len(cs):2d}: " +
              ", ".join(f"{v} {' '.join(a)}" for v, a in cs[:4]) + " ...")
