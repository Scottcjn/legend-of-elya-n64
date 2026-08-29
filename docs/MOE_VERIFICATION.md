# MoE verification: making the silent failures loud

**Summary.** The MoE path (F-R028) works today with five experts and a human
reading six lines off a TV. That does not scale to dozens, and every bug this
repo has shipped had the same shape: `expert_cache.c` unlinked for three weeks
behind two green host suites; `ec_init()` leaving `expert_off[]` zeroed so every
id read expert 0; `dma_read_async()` handed a KSEG1 pointer, filling slots with
open bus at the right size and duration; `ec_prefetch()` switching itself off in
the steady state. None raised an error. This specifies per-expert golden vectors
carried **inside the bank**, a boot self-test that is O(1) in expert count, an
on-console `MOEXCHK` verdict, CI gates, an EverDrive checklist, and one new
check for a failure nobody has hit. Nothing here has been run.

## 1. Per-expert golden vectors, carried in the bank

Extend `make_moe_bank.py`'s name entry from 16 B to 64 B:

```
name[16]   crc32_blob  u32   router_src_hash u32   prompt_id u16
ngen u8    flags u8    golden[24]  (bytes, ASCII transcript)  pad
```

`crc32_blob` covers the expert's `expert_len` bytes as they sit on the cart, so
a wrong offset, wrong stride, or open bus all fail it. `golden[]` is 24 greedy
tokens from a fixed prompt for that shard.

**Anti-tautology rule:** goldens come from a *different implementation* than
the one under test — `training/moe_golden.py` against the numpy oracle behind
the dense 176/176 claim, **never** `build/host_eval`, which is `nano_gpt.c`
compiled natively and cannot ratify itself. `router_src_hash` is the SHA-256 of
`training/moe_shards.py::SHARDS`, emitted by both `gen_moe_router.py` (as
`MOE_ROUTER_SRC_HASH`) and `make_moe_bank.py`. A bank and a ROM built from
different shard tables must refuse to run, not answer from the wrong expert.

## 2. Boot self-test that scales

Cost must not grow with expert count:

1. **O(1):** header magic/`n`/`len`/`base`; `h.n == MOE_N_EXPERTS`;
   `h.router_src_hash == MOE_ROUTER_SRC_HASH`. `moe_probe.c` already does the
   first two — add the third.
2. **O(N) but tiny:** DMA each expert's 32-byte SEQ2 header, check
   magic/`n_layers`/`d_model`. Dozens of 32-byte reads, not megabytes.
3. **Rotating deep check, one expert per boot:** a saved boot counter selects
   expert `b mod N`; stream it, CRC the resident megabyte, replay `golden[]`.
   Cost ≈ one switch + 24 tokens. The screen states coverage honestly:
   `EXPERT 7/40 DEEP OK — 40/40 within last 40 boots`.

Plus **verify-on-install**: `ec_acquire()` CRCs every freshly DMA'd slot before
returning it (`SGAI_MOE_VERIFY`, default on); a mismatch halts with the expert
id and both CRCs. No fallback, no silent retry.

## 3. `MOEXCHK` — the on-screen verdict

A ROM shaped like `xchk_probe.c`: for all 32 game prompts, route → stream →
swap → 24 tokens, compared byte-for-byte against `golden[]`, drawing

```
MOEXCHK PASS 32/32 prompts  40/40 experts  crc 40/40
```

mirrored to IS-Viewer, ending `MOEXCHK_DONE` so `~/aresroms/ares_run.sh` gates
on it. It also prints per-expert acquire/swap/gen cycles, so a prefetch
regression shows up as timing rather than silence.

## 4. CI gates (`make check` — no `.github/workflows` exists yet)

- `gen_moe_router.py -o -` diffed against `moe_router.h`; any delta fails.
- `ec_test` + `ec_regress` rebuilt on a **content-serving** DMA stub: it copies
  from a synthetic ROM whose expert *i* is filled with byte *i*, asserts the PI
  address is `base + i*len` and `< 0x2000_0000`, and the tests assert the
  *bytes*. This is the gate the old stubs lacked.
- **Link gate:** `grep expert_cache.o build/legend_of_elya_moeprobe.map`.
- Headless ares run of `MOEXCHK`, gated on `PASS`, on absence of `FAIL`, **and**
  on the log holding exactly 32 prompt lines. A run that prints nothing must
  not pass.

## 5. Real-hardware (EverDrive X7) checklist

ROM ≤ 64 MB; a larger bank moves to `sd:/elya/bank.sgmb` via libcart/FatFs.
On hardware: halt loudly if no Expansion Pak; print bank source (`cart` vs
`sd:`); re-measure acquire cycles — SD rate is *not* PI cart rate, so
prefetch-hides-under-a-token must be re-earned, not assumed; run the rotating
deep check across N power cycles; photograph `MOEXCHK PASS`.

## 6. What each check catches

- `expert_cache.c` unlinked, suites green → link gate; `MOEXCHK` on console.
- `expert_off[]` all zero → per-expert CRC (N−1 mismatch); stub address assert.
- KSEG1 → open bus → CRC on install; stub `< 0x2000_0000` assert.
- prefetch stalled by `inflight_slot` → `MOEXCHK` per-expert acquire cycles.

## 7. New check: slot-reuse aliasing under the generator

Untested failure: an expert is evicted and re-DMA'd into a slot while
`SGAIState` still points into it. The pointer stays valid, the weights change
mid-sentence, nothing crashes — the NPC changes personality mid-word.

Add `uint32_t gen` per slot, bumped in `ec_start_load()`. `sgai_init_ex()`
records `(slot, gen)`; `ec_assert_live()` — two loads and a compare — runs each
token and traps on mismatch. Add 16 guard bytes past `expert_len`, stamped with
the slot id, which also catches a DMA overrun. Prove it fires: a host thrash
test with 1 slot and 2 experts, and a `MOEXCHK` arm that forces the eviction and
**fails if the trap does not fire**. A check that has never fired is not a check.

## What I did NOT verify

- I ran nothing here: no code was compiled, executed, or timed.
- CRC-32 cost over 1 MB on the VR4300, against the 7.53M-cycle swap.
- That libcart/FatFs `sd:/` reads work here, their throughput, or whether
  `ec_request()`'s async model applies to SD (probably not).
- Save-memory for a boot counter on this cart config.
- Whether a numpy oracle usable for goldens exists in `training/` today;
  `eval_moe.py` uses `build/host_eval`, violating §1's rule.
- Size or timing of a 40-expert bank, or whether 2 slots still suffice.
- Whether the 64-byte name entry breaks an existing `SGMB` reader.
