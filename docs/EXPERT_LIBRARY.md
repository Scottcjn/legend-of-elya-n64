# Expert Library at Scale — manifest, two-level cache, and how a wrong expert is caught

**Design only. Nothing here was built, compiled, or run.** The measured floor:
F-R028 (5 experts, 1 MB each, uniform stride, ~100-cycle steady-state acquire,
7.53M-cycle swap, 5.87 tok/s) and F-R027 (a 160 KB cart DMA hides under ONE token; a 1 MB expert needs ~2.4 tokens of lead —
see the 2026-08-29 correction in F-R028). This scales that to hundreds of experts across cartridge ROM and the
EverDrive's SD card. Its center of gravity is the failure mode this repo keeps
hitting — `ec_init` never filling `expert_off[]`, `dma_read_async` reading open
bus at the right size and duration — machinery reporting success while serving
the wrong bytes. At scale that does not crash: the blacksmith answers as the
librarian. So every expert carries a tag DMA'd *with its own weights*, and the
load path refuses to hand `sgai_init_ex()` a blob whose tag disagrees.

## 1. Manifest (`SGMX` v2) — generated, never hand-written

`training/make_moe_bank.py` gains `--manifest`, emitting the library manifest
from `training/moe_shards.py` — the source `tools/gen_moe_router.py` already
reads. Both carry a `router_build_id` (FNV-1a64 over the ordered shard table);
the ROM compares `MOE_ROUTER_BUILD_ID` against the manifest's and halts on
mismatch, so a bank rebuilt against a stale router is a boot error, not 400
subtly wrong NPCs.

```
manifest header, 32 B, big-endian
  0 u32 'SGMX'   4 u16 ver=2   6 u16 n_experts   8 u32 entry_size=64
 12 u32 entry_off  16 u32 router_build_id  20 u32 manifest_digest
 24 u32 flags   28 u32 reserved

entry, 64 B; index == the id moe_route() returns
  0 u16 id    2 u16 res_class (0 PINNED,1 CART,2 SD)   4 u32 src
  8 u32 loc (cart offset from bank base | sd path index)   12 u32 len
 16 u32 digest_full (CRC-32, whole blob)   20 u32 digest_stripe (16x4KiB)
 24 u16 n_layers  26 u16 n_embed  28 u32 corpus_epoch
 32 char name[16]  48 u32 room_hint  52.. reserved
```

Offsets come from the manifest, so `ec_init()` stops deriving them from a
uniform stride and expert sizes may differ — a 6-layer boss beside a 2-layer
shopkeeper.

## 2. The tag: what makes a wrong expert loud

Each blob is prefixed by a 32 B `SGXT` tag written by the same assembler pass:
`'SGXT' | u16 ver | u16 id | u32 len | u32 digest_stripe | char name[16]`.
Deliberate redundancy with the manifest, generated from one source; the
*disagreement* is the detector. After the transfer completes and the D-cache is
invalidated, `ec_verify()` checks, in order:

1. tag magic is `SGXT` **and `tag.id == requested id`** — this alone catches the
   whole `expert_off[]` bug class, because expert 0's bytes say 0;
2. `tag.len`/`tag.digest_stripe` equal the manifest's;
3. the SEQ2 header after the tag: magic `SEQ2`, `n_layers`/`n_embed` as declared;
4. recomputed CRC-32 over the 16 stripes equals `digest_stripe`.

Only then does the slot become `EC_READY`. On any failure the slot is poisoned
(first 8 bytes overwritten), `verify_fail[id]++`, and acquire returns
`EC_E_VERIFY`; the caller falls back to the PINNED identity expert and the HUD
prints the failing id and reason. Fail closed, never silently.

Full-blob CRC stays off the hot path (cost unmeasured, plausibly worse than the
7.53M swap). `-DMOE_VERIFY_FULL` verifies everything; the default build scrubs
one full digest per idle-frame budget, so a rotted SD sector surfaces within
minutes of play rather than never.

## 3. Two-level cache

L1 is RDRAM slots (2 x 1 MB today, sized from the manifest's max `len`). L2 is
the media: cart ROM for the hot core set, `sd:/elya/experts/*.exp` via
libcart+FatFs for the tail. `ec_pump()` runs once per frame and owns a per-slot
state machine: `FREE -> DMA (cart, async) | SDREAD (blocking, chunked <=32 KiB
per call) -> VERIFY -> READY`. Chunking keeps an SD read off the frame budget;
because it is not async, this design never claims SD hides under a token — that
is a cart-only property (F-R027/F-R028).

Prefetch is game-state driven: `ec_hint_room(room_id)` walks the manifest's
`room_hint` field and queues that room's experts at the door trigger, one room
ahead, so an SD expert has a room traversal to arrive. `ec_pin(id)` pins the NPC
in dialog. Eviction is LRU with two overrides: PINNED is never a victim, and
among candidates whose stamps fall in one window the cheaper-to-refetch (cart
before SD) goes first — refetch cost, not recency alone.

## 4. Header (`src/expert_cache.h` v2)

```c
typedef struct { uint16_t slot, gen; } EcHandle;  /* gen invalidates stale refs */

int  ec_init(ExpertCache*, const void *manifest_rom, uint32_t bank_rom_base,
             uint32_t router_build_id, uint8_t **slots, uint16_t n_slots);
void ec_pump(ExpertCache*);                       /* once per frame */
void ec_hint_room(ExpertCache*, uint16_t room_id);
void ec_pin(ExpertCache*, uint16_t id);
void ec_unpin(ExpertCache*, uint16_t id);

__attribute__((warn_unused_result))
int  ec_acquire(ExpertCache*, uint16_t id, const uint8_t **out, EcHandle *h);
int  ec_deref(ExpertCache*, EcHandle, const uint8_t **out);   /* gen-checked */

enum { EC_OK=0, EC_E_PENDING=-1, EC_E_VERIFY=-2, EC_E_MANIFEST=-3,
       EC_E_MEDIA=-4, EC_E_RANGE=-5 };
```

`ec_acquire()` returns status and writes the pointer out, so "I forgot to check"
is a compiler warning. `EcHandle.gen` bumps on eviction, so a pointer held across
a room change fails in `ec_deref()` rather than reading another NPC's weights.
Host suites must assert *contents*: a fake medium returning expert *j* for a
request for *i* must fail. That is the test the two green suites lacked.

## What I did NOT verify

- Nothing here was built, compiled, or run — no probe, no ROM, no host test.
- SD throughput/latency on an EverDrive-64 X7 under libcart+FatFs. The "one room
  ahead is enough" claim rests entirely on it.
- Whether libdragon's FatFs path chunks without holding a lock across frames,
  and whether SD reads are safe to interleave with PI DMA to cart.
- CRC-32 cost per MB on the VR4300 — no estimate was measured.
- That a manifest read at boot survives `dfs_rom_addr()` returning KSEG1 — the
  masking bug that bit `dma_read_async` applies here too.
- Whether non-uniform expert lengths break anything in `sgai_init_ex()` beyond
  the existing `n_layers` clamp.
- RDRAM budget for >2 slots alongside KV cache, framebuffer, and game state.
- That substring routing survives hundreds of experts; it is first-match-wins
  over ~120 keywords, unmeasured beyond 5 shards.
