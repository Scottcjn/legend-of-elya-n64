Design document written (prose ≈880 words; `wc -w` 920 including code-block and bullet tokens). Path: `/tmp/claude-1000/-home-scott/f8759638-0914-4711-b8f5-df3ccea844f2/scratchpad/lane3.md` — commit as `docs/N64_TOOL_CALL_DESIGN.md`.

---

# Lane 3 — NPC tool calls: an in-band command grammar with a decode-time trie

**Summary.** An NPC should be able to give an item, open a door, move, set a quest flag or remember a fact — actions, not just prose. This lane emits those as a bracketed command *inside the same byte stream as the dialogue*, by the mechanism C028/C029 proved for newline: the vocabulary is 256 bytes (`SGAI_VOCAB`) and the display band is 32..126, so control bytes are already invisible and already discarded unless a sampler flag lets them compete. Inside a command the candidate set is not the ASCII band but the legal-next-byte set of a generated trie, so a malformed command — or one naming an item or room that does not exist in this room, this save — is never emitted rather than rejected afterwards. We criticised a lexicon mask for prose because it fakes coherence the model lacks (C027); for command syntax the mask is not a cheat, it is the parser. Nothing here has been built or measured.

## Grammar

```
<cmd>  ::= 0x01 <verb> ' ' <arg> [ ' ' <arg> ] 0x02
<verb> ::= give | take | open | lock | goto | quest | flag | recall
```
Verbs and args are lowercase words already in the 828-word v7 vocabulary where possible, so the model's English carries the command. Arity and argument class live in the trie: after `give ` only item names may follow, after `goto ` only rooms, after `quest ` a quest id then `start|done`. `0x02` is legal only at an accept node.

## In-band emission

`sample_logits()` gains `-DSGAI_CMD_BAND`, built like C029's `SAMP_LO`/`SAMP_MAP`: byte 1 is parked in slot 30 of the prose softmax (newline owns 31), so the contiguous softmax, total and cumulative-draw loops cover it as one more entry and no dead compare enters the hot loop when the flag is off. The default ROM must stay byte-identical, tested C029's way — `mips64-elf-objdump -d build/nano_gpt_rsp_ovl.o` before and after, not md5 (DWARF line numbers move).

Drawing slot 30 sets `state->cmd_mode = 1` and switches the sampler to `sample_masked(logits, mask, n)`, `mask` coming from the trie. `update_generating_step()` routes command bytes to `cmd_feed()`, not `G.dialog_buf`, so a command is never printed nor truncated by the `'.'` rule. Dialogue resumes after `0x02`.

## The trie, and why malformed commands are unreachable

`cmd_trie.h` is **generated** by `tools/gen_cmd_trie.py` from `training/world_defs.py` — the discipline of `moe_router.h`, for its reason: two hand-maintained copies of the world is the bug.

```c
typedef struct { uint8_t byte; uint16_t child; uint32_t slots; } CmdEdge;
typedef struct { uint16_t edge0; uint8_t n_edge, accept; }        CmdNode;
```
`slots` is the union of world-object ids under that edge (≤32 per verb class). Each token `cmd_mask(node, live)` clears every edge whose `slots & live` is zero, `live` rebuilt that frame from game state: items the NPC holds, exits present in `ROOM_MAP[G.current_room]`, quests defined. An edge with no live leaf is not a candidate, so **`goto forge` from a room with no west exit, or `give lantern` when no such item exists, cannot be sampled at all.** Cost is O(edges at this node) per token — trivial beside ~330M cycles/token (F-R028), but unmeasured.

## Parse-time rejection, and how this lane fails loudly

`cmd_parse()` re-validates the finished command against `world_defs` independently of the trie: verb known, arity right, every referent live. If the decoder is correct this can never fire — so **a nonzero `g_cmd.rejected` is not a handled error, it is the alarm that trie and world desynced**, drawn on the HUD beside the tok/s readout and logged to IS-Viewer with the offending bytes. Counters `emitted / parsed / executed` are shown together, because `emitted > 0, executed == 0` is the shape of every bug this repo keeps hitting (`expert_cache` unlinked for three weeks, `ec_init` leaving `expert_off[]` zero, `dma_read_async` reading open bus at the right size and duration).

The guard also gets an **ablation control**: delete one item from `world_defs.py`, rebuild, re-run the same prompts. If that item's emission count does not go to zero, the mask is decorative and the lane is false-green.

## Training data

`training/gen_cmd_corpus.py` crosses templates with the world table and emits QA lines in the existing format, command appended before the trainer's `"\n"`:

```
"Can I have the lamp?: Take it, traveller.\x01give lamp\x02"
```
Only live referents are generated, so the model is never taught a command the trie forbids. The build then runs every line through a host build of the ROM's own `cmd_parse()` (`tools/cmd_lint`); one rejected line fails the build. Routing is unchanged — these lines fall through `moe_shards.shard_of()` like any other, mostly into `dungeon` and `identity`.

## What I did NOT verify

- Nothing here was compiled, trained, booted, or measured. No number below F-R028/C029 is mine.
- That the model can *learn* to emit 0x01 at the right moment at 3.2M params — untested, and the likeliest failure of the lane.
- Trie mask cost per token; the claim "trivial" is arithmetic, not a probe.
- Whether two extra control bytes in the softmax band perturb prose quality, or command lines displace enough corpus to move the 28/28 hit rate (C027's metric must be re-run with the flag on).
- RDRAM headroom for the trie and `live` state beside two 1MB expert slots.
- Expert switching: a command begun under one expert with a route change mid-answer is undefined here.
- EverDrive/SD and real silicon: every claim cited is ares, and this lane adds none of its own.
