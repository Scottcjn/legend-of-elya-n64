# Legend of Elya as a streaming-MoE adventure game — the plan, and what the review changed

**Status: DESIGN. Nothing in this document has been built or run.** It is the
synthesis of a 2026-08-29 review: ten design lanes (each adversarially verified
by a second reviewer) plus four external models — Grok CLI reading the repo and
libcart source, qwen3-max, qwen3-235b-a22b, and Codex. GLM-5.3 returns 403 on
this account and GLM-5.2 returns an EMPTY completion, which is a failed call,
not a clean result; both are recorded as unavailable rather than dropped.

The measured floor this builds on: F-R026 (epilogue overlap, 3.03 tok/s dense),
F-R028 (5 ternary experts, 5.87 tok/s, equal accuracy, 28% less garble),
C027-C029 (coherence and the trained newline EOS).

## The review changed three things. Take these before the design.

**1. One expert per NPC is wrong. One expert per DOMAIN, with per-NPC prefix
conditioning, is right.** Unanimous across the external panel. 3.2M parameters
is a *domain shard* — that is what bought equal accuracy and −28% garble — not a
character; a single NPC's lines cannot train 3.2M parameters without memorising
them. With two RDRAM slots and a 161 ms switch, three NPCs of different domains
standing in one room thrash. Identity is a few hundred bytes of prefix, held in
RDRAM, costing nothing. Pay the 1 MB and the switch when the *domain* changes,
not when the *name* does.

**2. The SD card is a promotion source, not a live read path.** Grok read
libcart: `edx_card_rd_dram()` is CMD18 plus a CPU loop of 2048 iterations
(busy-wait DAT, a blocking 512 B PI DMA, 8 B of discarded CRC) — **the CPU is
the transfer**, and cart access is held exclusive throughout. It is not
`dma_read_async` and it cannot hide behind generation. The alternative,
`edx_card_rd_cart()`, hands the EverDrive's FPGA a real SD -> cart-SDRAM DMA —
but stock libcart spins on `DMA_STA_BUSY`, and that SDRAM *is* the 64 MB ROM
image, so an unreserved write punches the running game. The viable shape:
reserve a tail window in cart SDRAM, issue the FPGA DMA, poll it in vblank
instead of spinning, then PI-DMA cart -> RDRAM — and that last hop is the one
already measured as hideable. Raw LBA, not `fopen("sd:/…")`: `debug_init_sdfs`
is a debug feature and disappears under `NDEBUG`.

**3. The 161 ms switch can be deleted, and the obvious way to do it fails
silently.** Storing experts pre-permuted removes `sgai_init_ex()`'s RSP lane
re-permutation entirely — it is a byte transpose, values unchanged. The trap:
for ternary the permutation is an 8x8 transpose, therefore an *involution*, so
permuting an already-permuted blob returns it to row-major, the RSP computes on
the wrong layout, **and it still emits fluent text**. The host oracle and the
CPU kernels read row-major, so host goldens keep passing. Mitigation is
structural, not procedural: a `layout_id` in the blob header, and a runtime that
refuses a layout it does not recognise.

## Corrections to our own record, made the same day

- A 1 MB expert load is **401 ms**, which is **2.4 tokens** at 169 ms/token. It
  did not "hide under a token" in F-R028 — `ec_prefetch()` had a whole 24-token
  turn of lead. F-R027's per-token hiding was a **160 KB** slice, 6.4x smaller.
  Prefetch depth is a design constraint: request an expert at least one
  room/turn before it is needed.
- The switch is **161 ms**, not the ~80 ms first written: CP0 Count ticks at
  46.875 MHz (half the CPU clock), the divisor this repo's own tok/s code uses.

## The game (lane 5), because the constraint is the mechanic

Every NPC answers only out of its own domain's weights. Ask a miner about Ganon
and you do not get "I don't know" — you get a fluent, confident, wrong sentence.
C027 measured exactly this: 4 of 32 prompts whose key is absent from the corpus
still produced a real, well-formed, incorrect answer. So the player's job is not
collecting keys, it is **corroboration**: one expert's claim is a rumour; the
same claim emitted independently by a second domain is a fact. Confabulation
stops being a defect and becomes the puzzle. Full design: `GAME_DESIGN_LANE5.md`.

## Build order

1. **`sd_probe.c`** — the arms and the pre-registered thresholds are already
   specified in `SD_EXPERT_STREAMING.md`. Every arm CRC-checks its bytes against
   the ROM copy; a missing card must print ABSENT, never a fast time. This
   decides whether the library can exceed 64 MB at all.
2. **Pre-permuted blobs + `layout_id`**, deleting the 161 ms switch, gated on a
   host test that proves a double-permute is *rejected* rather than merely wrong.
3. **Router v2** (`ROUTER_V2.md`): NPC identity as the primary key, still
   generated from one source of truth, returning a reason rather than a bare id.
4. **Domain experts + per-NPC prefix conditioning**, replacing one-per-NPC.
5. **The vertical slice**: one hub, three NPCs across two domains, one
   corroboration puzzle, on the EverDrive.

## What nobody has verified

Whether ares emulates an EverDrive SD card at all (if not, `sd_probe` is a
hardware-only test). Actual X7 SD throughput. Whether the cart-SDRAM tail window
is safely writable while a game runs. Whether the FPGA DMA can be polled
cooperatively rather than spun on. And every number in this repo is still ares,
never silicon — the EverDrive run remains the outstanding check on all of it.
