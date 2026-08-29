# Router v2 — NPC identity as the primary routing key

**Summary.** `moe_route()` (generated into `moe_router.h` from
`training/moe_shards.py`) picks one of 5 experts by lowercase substring on the
prompt alone. In a game the expert is decided by *who you walked up to*: the
Forge Master should answer out of `hardware` whatever you type at him. Router v2
makes the NPC the primary key, keeps prompt keywords as a *gated* secondary
signal, adds a fallback when an NPC's expert is missing from the loaded bank,
and — the point of the design — never returns a bare expert id. It returns a
`MoeRoute` that names the reason it chose, so a router that silently answers out
of expert 0 (the `ec_init`/`expert_off[]` failure mode) cannot look healthy. The
source of truth stays `training/moe_shards.py`; the NPC table lands *in that same
file*, `tools/gen_moe_router.py` compiles it to the header, and a generated hash
is checked at bank load so a stale ROM/bank pair fails loudly instead of routing
to the wrong shard. **Nothing here has been run yet** — see the last section.

## Source of truth (one file, still)

`training/moe_shards.py` gains one table beside `SHARDS`:

```python
NPCS = [  # id_tag, display, home, allow(override), fallback chain
  ("sophia","Sophia","identity",{"lore","rustchain","hardware","dungeon"},["lore","identity"]),
  ("forge_master","Forge Master","hardware",{"rustchain"},["identity"]),
  ("librarian","Librarian","lore",{"dungeon","identity"},["identity"]),
  ("none","",None,set(),["identity"]),
]
OVERRIDE_MIN, OVERRIDE_MARGIN = 2, 1
```
plus one function `route2(npc, text, resident_mask) -> (expert, reason)`, the
*only* definition of the rule. `shard_of()` becomes
`route2(NPC_NONE, line, ALL).expert`, so existing labels are unchanged by
construction.

`tools/gen_moe_router.py` emits into `moe_router.h`: the keyword tables as today;
`MOE_NPC_*` ids and `moe_npc_name[]`; `moe_npc_home[]`, `moe_npc_allow[]` (a
`uint32_t` expert bitmask), `moe_npc_fallback[][]`; the two constants;
`MOE_ROUTER_HASH` (FNV-1a over `repr(SHARDS)+repr(NPCS)`+constants); and
`MOE_ROUTER_CASES[]`, a generated (npc, prompt, expert, reason) golden table
drawn from the trainer's own corpus.

## ROM API

```c
#define MOE_R_NPC_HOME 0  /* home expert; no topic signal strong enough */
#define MOE_R_TOPIC    1  /* topic override won the gate                */
#define MOE_R_FALLBACK 2  /* home not resident in this bank             */
#define MOE_R_UNSURE   3  /* topic pointed outside allow[]; stayed home */
#define MOE_R_NONE     4  /* nothing resident — do not generate         */

typedef struct { uint16_t expert, npc; uint8_t reason, best, hits_home, hits_best; } MoeRoute;
MoeRoute moe_route2(uint16_t npc, const char *prompt, uint32_t resident_mask);
static inline uint16_t moe_route(const char *p);   /* v1 shim, unchanged callers */
```

Algorithm: lowercase into the same 96-byte scratch; count *distinct* keyword
hits per expert (`uint8_t hits[MOE_N_EXPERTS]`) instead of first-hit-wins;
`home = moe_npc_home[npc]`. Override to `best` only if
`best != home && (moe_npc_allow[npc] >> best) & 1 && hits[best] >= 2 &&
hits[best] - hits[home] >= 1` → `MOE_R_TOPIC`. If `best != home` but the gate
fails → `MOE_R_UNSURE`, expert = home; the game shows the NPC's deflection line
("that is not my trade") rather than answering out of a shard nobody trained in
that voice. Then residency: if the chosen expert is not in `resident_mask`, walk
`moe_npc_fallback[npc]` → `MOE_R_FALLBACK`; if none is resident, `MOE_R_NONE`
with `expert = MOE_NO_EXPERT` and the caller must refuse to generate. Cost is a
few hundred cycles of `strstr` — noise beside the 7.53M-cycle switch.

## Residency by name, not by index

`resident_mask` is built at bank load by comparing the SGMB 16-byte name table
to `moe_expert_name[]`. A bank name absent from the router is a hard fail (as
`h.n != MOE_N_EXPERTS` already is); a router expert absent from the bank clears
its mask bit. That is what lets an SD-card (`sd:/`) library ship banks holding
*subsets* of experts without "bank index IS the id" quietly mis-slicing.
`make_moe_bank.py` bumps SGMB `ver` to 2 and writes `MOE_ROUTER_HASH` at header
offset 16 (BE `u32`, reserved today); the ROM refuses a mismatch.

## Making drift impossible to hide

1. `gen_moe_router.py --check` regenerates to a temp file and `exit(1)`s on any
   diff — a `make check` prerequisite and a CI job.
2. `tools/router_parity.c` compiles `moe_route2` natively and replays
   `MOE_ROUTER_CASES[]`; `training/router_parity.py` replays the same cases
   through `route2()`. Both assert `(expert, reason)` equality per case. The
   cases are generated, so they cannot go stale against the table.
3. `eval_moe.py` must reproduce C027's numbers exactly under `NPC_NONE`.
4. `moe_probe.c` prints `reason` and `hits_home/hits_best` per prompt, so the
   console states its own routing rationale (the F-R024 XCHK pattern).

## Game-loop consequence

The NPC is known when the player is *near*, before any text exists: fire
`ec_prefetch(home)` then, if a slot is free, the `sgai_init_ex` swap on dialogue
open, so the measured ~80 ms re-permutation hides under the textbox animation.
Proposed policy (unmeasured): allow `MOE_R_TOPIC` only on a conversation's first
turn, or on `hits_best - hits_home >= 2`, so a chatty player cannot thrash an
80 ms switch every turn.

## What I did NOT verify

- **Nothing here was compiled, run, or measured** — no parity test, no ROM, no
  timing.
- That the ~80 ms switch hides under a textbox animation — that animation does
  not exist and the overlap was never measured.
- That distinct-hit counting reproduces v1's first-hit-wins on the current
  corpus. The `NPC_NONE` path is designed to, but the tie-break is a behaviour
  change I never diffed against the existing labels.
- `OVERRIDE_MIN=2`, `MARGIN=1`, and the `allow` sets are guesses; tuning them
  needs a labelled prompt set that does not exist.
- The `home`/`fallback`/`allow` rows assume `multi_npc.c`'s three personas
  (Sophia / Forge Master / Librarian) are the shipping cast.
- That SGMB offset 16 is unused by every reader — I read the writer only.
- Whether a subset bank plus fallback routing degrades answers; the 32-prompt
  metric has never been run with an expert missing.
- The new BE hash field has never been read on hardware.
