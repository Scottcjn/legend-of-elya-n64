# Lane 4 — the per-NPC expert pipeline at scale

**Summary.** Today one expert = one hand-written keyword list in
`training/moe_shards.py` + one `train_sophia_v9_qat.py --shard` run (12,000
steps, 491.6 s for `hardware`, `training/moe_out/hardware_meta.json`) + a manual
`qat_npz_to_seq.py` + `make_moe_bank.py`. Five experts, driven by
`training/train_moe_all.sh`: a `for` loop with no state. It does not survive
dozens of NPCs for four reasons visible in this repo: the corpus lives
inside Python (`train_sophia_v7.py`'s literal lists), the shard table is
hand-authored so adding an NPC is a code edit, `*_meta.json` records `seed` and
`steps` but **no corpus hash**, and `make_moe_bank.py` reads whatever
`*.seq2.bin` is on disk — a stale expert banks silently and the ROM answers
from it. This lane makes the NPC the unit of authorship, the manifest the unit
of truth, and adds one content check F-R028's bug class cannot survive.

## On-disk layout

```
npcs/<id>/npc.toml        id, display_name, expert index, route keys, arch overrides
npcs/<id>/corpus/*.qa     one record per stanza: "Q: ...\nA: ...\n\n"; A-only = prose
npcs/<id>/eval.qa         held-out prompts for THIS npc; never trained on
build/experts/<id>/       <id>.npz  <id>_meta.json  <id>.seq2.bin  <id>.card.json
build/bank/sophia_moe.bin + bank.manifest.json
training/moe_shards.py    GENERATED from npcs/*/npc.toml (keeps gen_moe_router.py intact)
```

`npc.toml` route keys are the same plain lowercase substrings `moe_shards.py`
uses today, with an explicit `order = N`; `moe_shards.SHARDS` and therefore
`moe_router.h` become second-order generated artifacts. Two hand-maintained
copies of a route key is the exact bug the router generator exists to prevent,
so the fix is to move the source of truth up, not to add a table.

## `tools/elyaforge.py` — the CLI

```
elyaforge lint                     # toml/qa parse, duplicate route keys, order collisions
elyaforge plan [--all|<id>...]     # prints per-npc: corpus_sha, want_sha, have_sha, action
elyaforge train <id> [--force]     # skipped when have_sha == want_sha
elyaforge export <id>              # npz -> SEQ2 via qat_npz_to_seq.py, then max|dW| assert
elyaforge bank                     # SGMB v2 + bank.manifest.json; refuses on any stale expert
elyaforge eval <id>|--bank         # eval_moe.py metric, writes <id>.card.json
elyaforge gate --baseline <sha>    # CI verdict, exit 1 on regression
elyaforge verify <bank.bin>        # re-hash every expert against the manifest
```

`want_sha = sha256(corpus bytes ‖ npc.toml train-affecting keys ‖ trainer sha
‖ seed ‖ steps ‖ arch tuple ‖ schema version)`; `have_sha` is read from
`<id>_meta.json`. **Incremental rebuild is that comparison and nothing else**:
editing one NPC's `.qa` moves exactly one `want_sha`, `plan` shows one
`RETRAIN`, `bank` relays the file. Nothing else retrains — nothing else's
inputs moved.

## Bank format change (SGMB v1 → v2)

`make_moe_bank.py` gains an `n_experts × 8 B` table between the name table and
`base`: `crc32(expert bytes)` + `len`. `base` is already a header field the ROM
reads (`moe_probe.c:121` passes `h.base` to `ec_init`), so v2 adds content
verification without a reader rewrite. The ROM CRCs each expert after
`ec_acquire()` and prints `MOE CRC ok=<id>` or halts. F-R028's open-bus read — right size,
right duration, wrong bytes, no error — fails this check in one frame.
`elyaforge verify` does the same off-target. Uniform stride stays: `bank`
**fails** if any blob exceeds the manifest's pinned `expert_len` rather than
silently re-laying every offset under a ROM built for the old one.

## The CI gate

`elyaforge gate` runs `training/eval_moe.py` (`build/host_eval` — the ROM's own
`nano_gpt.c`) and refuses the bank if, versus the baseline manifest:

1. any NPC's trained-answer hits drop (bank-wide baseline: 28/28, F-R028);
2. bank-wide clean answers drop below 31/32, or invented-words/line rises above
   1.84 + 0.15;
3. `moe_route()` sends any prompt in `eval.qa` to an expert other than its own —
   run against the **generated** `moe_router.h`, not the toml, so a stale
   generated file is a test failure, not a mystery;
4. any manifest CRC mismatches the banked bytes;
5. `git diff --exit-code moe_router.h training/moe_shards.py` after regenerating.

Checks 3–5 are the anti-silence checks. This repo's real failures
(`expert_cache.c` unlinked for three weeks under two green suites; `ec_init`
leaving `expert_off[]` zero) were all cases where the suite could not observe
the wrong thing, so every gate item is written against an artifact the ROM
itself consumes.

## What I did NOT verify

- I ran nothing. No training run, no `elyaforge` (it does not exist), no bank
  build, no ares boot. Every number quoted above is read from this repo's
  existing logs and `*_meta.json`, not re-measured.
- I did not check that `train_sophia_v9_qat.py` is deterministic given
  `--seed 1337` on the same GPU, let alone across machines. `set_seed()` sets
  python/numpy/torch seeds but I saw no `torch.use_deterministic_algorithms`
  or cuDNN flag; per-expert repro may need a `torch/cuda/driver` version tuple
  in the manifest rather than a promised bit-exact rebuild.
- I did not measure per-NPC training cost beyond the recorded
  `train_seconds: 491.6`, so "dozens of NPCs" wall-clock is extrapolation.
- I did not recompute the SGMB layout for large `n_experts`, nor raise
  `EC_MAX_EXPERTS` — **it is 16 today and "dozens of NPCs" exceeds it.**
- I did not test whether more experts than slots changes the ~100-cycle
  steady-state acquire; F-R028 measured 5 experts, 2 slots, one DMA per token.
- I did not check whether SD-card (`sd:/`) reads can feed `ec_request`'s async
  path at all — libcart/FatFs is a blocking file API, so a bank larger than the
  64 MB EverDrive ROM image likely needs a different acquire path than PI DMA,
  and I have no measurement of its rate.
- The metric assumes one shared 828-word vocabulary; per-NPC corpora grow it,
  so gates 1–2 only compare if the baseline is re-measured on every vocabulary
  change. I did not design that re-baselining.
