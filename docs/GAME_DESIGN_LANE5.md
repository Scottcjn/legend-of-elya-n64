# Lane 5 — The game that only works because every NPC is its own model

**Summary.** Every NPC is bound to one ternary expert (4L x 256d, 3,211,264
params, a complete 1,048,588 B SEQ2 blob) streamed off the cartridge or the
EverDrive SD card. That is the game, not a footnote: an NPC can only answer out
of its own weights, so **knowledge is geography**. Ask a miner about Ganon and
you do not get "I don't know", you get a fluent wrong sentence — C027 measured
exactly that (4/32 prompts whose key is absent still produce a real, wrong
sentence). So the player's job is not collecting keys, it is **corroborating**: a
claim from one expert is a *rumour*; the same claim emitted independently by a
second expert is a *settled fact*, and settled facts open doors. Gossip
propagates because the player carries words, and words are what `moe_route()`
matches on. Budget: ~5.87 tok/s, self-terminating answers median 27 chars
(C028) — one line is ~4.6 s of token-by-token typing. Pace for speech.

## The world — hub plus five areas
- **The Study (hub).** Elyan Labs' Victorian study: Sophia, the fireplace, the
  Ledger on the desk, one door per area, each sealed by a settled fact.
- **The Attic of Silicon.** G4s, an Amiga, a POWER8 under a dust sheet.
  *hardware*.
- **The Node Yard.** Rain, cabling, an epoch clock. *rustchain*.
- **The Cartridge Dungeon.** Zelda-shaped: 6 rooms, a compass, a boss. *dungeon*.
- **The Arcade Reliquary.** Dead cabinets that still talk. *lore*.
- **The Router's Gate.** Endgame — where keywordless prompts fall through.
  *identity*, the catch-all: canon, not a bug.

## Cast (12) — NPC : one-line identity : expert
1. **Sophia Elya** — guide who admits what she cannot know — *identity*
2. **The Flameholder** — player-side lantern naming the speaking model —
   *no expert; it is the debug HUD*
3. **Grandfather G4** — AltiVec bore, right about silicon — *hardware*
4. **Nine-Seventy** — a G5 resenting the G4's antiquity multiplier — *hardware*
5. **Vee-Arr** — the VR4300 you are running on, embarrassed — *hardware*
6. **The Assayer** — weighs hardware for proof-of-antiquity — *rustchain*
7. **Epoch, the Clock-Keeper** — settles rewards, hates unsettled claims —
   *rustchain*
8. **Pale Ledger** — ghost of a rolled-back block — *rustchain*
9. **The Warden** — dungeon guard who speaks only warnings — *dungeon*
10. **Tilebreaker** — thief who knows what is under the floor — *dungeon*
11. **The Reliquarian** — remembers other people's cartridges — *lore*
12. **The Router** — final boss; answers *anything*, trusted by *nothing* —
    *identity*

NPCs share the five shipped experts today (see limits). Personality comes from
the persona prefix in `NPC_PROFILES[]`; **facts** come only from the binding.

## Quest graph — gossip between models
Words are keys because `moe_route()` is substring matching. Hearing a keyword
adds it to the **Satchel of Words**; speaking it forms the prompt that expert
actually sees.

- The Warden says *something* guards the realm, not what. → keyword
  **"antiquity"** to The Assayer → a claim about old hardware. *Rumour.*
- Carry that sentence to **Grandfather G4** (independent expert). If his answer
  carries it too → **SETTLED**, and Study door 1 opens.
- Epoch refuses to act on anything unsettled — his character *is* the rule.
- **Pale Ledger** confabulates on purpose (off-shard keyword in, fluent nonsense
  out): the player learns to distrust one witness.
- Endgame: The Router answers everything, because it is the catch-all. Beating
  it means refusing its answers and settling them elsewhere.

## The anti-silent-failure control, as a game item
The **Flameholder's Lantern** (Z toggles) draws live over the textbox:
`E=hardware slot=1 off=0x00300020 head=3f9a swap=7.5M`. That is
`ledger_draw(display_context_t)` reading `ExpertCache` plus the first 8 bytes of
the resident blob. If the wrong expert answers, or a slot holds open bus
(F-R028's bug), the player sees it inside the fiction. The NPC↔expert table is
`src/game/npc_bank.h`, **generated** by `tools/gen_npc_bank.py` from
`training/moe_shards.py` exactly as `moe_router.h` is; a hand-written second copy
is the bug class this repo keeps re-learning.

Signatures: `uint16_t npc_expert(NpcId)` (generated); `int quest_settle(FactId,
uint16_t expert, const char *sentence)` → 1 when a second *distinct* expert
emits a matching sentence; `void ledger_draw(display_context_t)`.

## The 20-minute vertical slice, in build order
1. **Study + Attic + Node Yard.** Four NPCs — Sophia, Grandfather G4, The
   Assayer, Epoch — on three shipped experts.
2. **Token-by-token textbox**, `-DSGAI_NEWLINE_STOP` (C029) so lines end
   themselves. This is the feel of the whole game.
3. **Approach-swap.** `ec_prefetch()` on entering an NPC's trigger radius, so
   the 7.53M-cycle re-permute lands under the walk-up, not the textbox.
4. **Satchel + one settle chain** (antiquity: Assayer → G4) opening one door.
5. **Lantern HUD**, on by default until this has run on a television.

## What I did NOT verify
- Nothing here was built or measured; it is design only.
- Two 1MB slots + KV + framebuffer fitting in 8 MB is the brief's claim; I did
  not re-run `tools/rdram_budget.py` against a build with RDP assets.
- SD streaming (libcart/FatFs, `sd:/`) comes from the brief; I measured no SD
  latency, so "prefetch hides it" holds for **cart PI DMA only**
  (F-R027/F-R028).
- Per-NPC experts beyond the five trained shards do not exist; splitting
  *hardware* into three NPC-sized experts is untested and may cost quality.
- Corroboration assumes two experts can emit the *same* sentence for a shared
  fact. Shared prose (`CORPUS_LINES`) reaches every shard, which makes it
  plausible, but I ran no cross-expert agreement eval.
- `quest_settle`'s string match (exact? normalised?) is unspecified and
  unmeasured — the likeliest place for a silent false positive.
- Approach-prefetch hiding the 80 ms swap is an assumption; `moe_probe.c`
  measured the swap cost, not its concealment.
