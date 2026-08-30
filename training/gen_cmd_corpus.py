#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_cmd_corpus.py — teach an NPC to ACT, not just answer.

Emits QA lines in the v7 format with a command appended to the answer:

    "Can I have the lamp?: Take it, traveller.\\x01give lamp\\x02"

Only commands world_defs grants that NPC are generated, so the model is never
taught something the ROM's trie forbids -- the corpus and the decoder come
from one file by construction.

The likeliest failure of this whole lane, stated up front: whether a
3.2M-parameter byte-level model reliably learns to emit 0x01 at the right
moment is UNTESTED. These lines are the experiment, not the proof.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import world_defs as W

# Player phrasings -> the answer prose that precedes the command.
ASK = {
    "give":   ["Can I have the {a}?", "May I take the {a}?", "I need the {a}.",
               "Give me the {a}.", "Do you have a {a}?"],
    "open":   ["Open the way to the {a}.", "Let me into the {a}.",
               "How do I reach the {a}?", "Take me to the {a}."],
    "quest":  ["What is my task?", "Give me a quest.", "What should I do?"],
    "recall": ["Remember I carry the {a}.", "I have the {a}.",
               "Note that I hold the {a}."],
}
SAY = {
    "give":   ["Take it, traveller.", "It is yours.", "Carry it well.",
               "Here, you have earned this."],
    "open":   ["The way is open.", "Go through, I will wait.",
               "Step where I point."],
    "quest":  ["Seek it and return.", "Begin, and do not stray.",
               "The task is set."],
    "recall": ["I will remember.", "Noted, traveller.", "It is written."],
}

def lines_for(npc, reps=1):
    out = []
    for verb, args in W.commands_for(npc):
        cmd = W.render(verb, args)
        for i, q in enumerate(ASK[verb]):
            for j, a in enumerate(SAY[verb]):
                if (i + j) % 2:                      # thin the cross product
                    continue
                key = q.format(a=args[0]) if "{a}" in q else q
                out.append(f"{key}: {a}{cmd}")
    return out * reps

if __name__ == "__main__":
    for npc in W.NPC_GRANTS:
        ls = lines_for(npc)
        print(f"{npc:9s} {len(ls):3d} command lines")
        for l in ls[:2]:
            print("   " + l.replace("\x01", "<CMD>").replace("\x02", "<END>"))
