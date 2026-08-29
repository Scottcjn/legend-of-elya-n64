# Lane 7 — Memory and Persistence

**Summary.** NPCs remember the player across power-off through a versioned, CRC-checked save that stores *enumerated* state only — per-NPC trust, fact bits, quest flags, world state — never free text. That constraint is forced by the model, not by storage: the shipped engine is char-level with `ctx_len = 128`, an 828-word vocabulary, and every trained line is `key: answer\n` (C027/C028). Arbitrary remembered prose fed as a prefix is out-of-distribution and would spend the same context the answer needs. So a remembered fact is an *id* whose surface string is one line of the training corpus, emitted from a single source of truth (`training/memory_facts.py`) into both the trainer and `memory_facts.h` by `tools/gen_memory_facts.py` — the same discipline that makes `moe_router.h` generated and never hand-edited. Canonical storage is the EverDrive SD card; EEPROM holds a lossy digest for cart-only play.

## Media tiers

| Tier | Path | Holds |
|---|---|---|
| SD (canonical) | `sd:/elya/save0.bin`, `save1.bin` | full schema, A/B double-buffered |
| EEPROM 16k (fallback) | 2 KB | 16 B header + 32×8 B NPC digest + 64 quest bits + CRC |
| Controller Pak | not used | `legend_of_elya.pak` in-tree is unexamined |

A/B slots plus a monotonic `save_counter` are the power-off story: write the *other* slot, fsync, then accept. Load picks the highest `save_counter` whose CRC validates. Never write in place.

## Byte layout (big-endian, like `SGMB` in `make_moe_bank.py`)

```
header 64 B
  0  u32 magic          'ELSV'
  4  u16 fmt_major      2 = incompatible layout change
  6  u16 fmt_minor      additive fields only
  8  u32 save_counter   monotonic; A/B arbitration
 12  u32 body_len
 16  u32 body_crc32     over bytes [64, 64+body_len)
 20  u32 fact_table_crc FACT_TABLE_CRC baked into the ROM
 24  u16 npc_rec_len    STRIDE for forward compat (never sizeof)
 26  u16 n_npc
 28  u32 quest_bits_lo / 32 u32 quest_bits_hi
 36  u32 world_seed
 40  u8[8] player_name  restricted to the trained band 32..126
 48  u8[12] reserved    0xA5 fill, NOT zero
 60  u32 hdr_crc32      header with body_crc32/hdr_crc32 zeroed
body: n_npc records of npc_rec_len (>= 24 B)
  0  u8 npc_id   1 u8 expert_id (MOE_E_*)   2 i8 trust (-64..63)
  3  u8 mood     4 u16 met_count   6 u16 last_topic
  8  u32 fact_bits      bit i = FACT_i known by THIS npc
 12  u32 topic_bits     which of the 32 PROMPTS[] were asked
 16  u32 first_met_save / 20 u32 rec_crc32
```

`0xA5` reserved fill and a per-record CRC exist so an all-zero record reads as *never written* rather than as valid neutral state — the `ec_init`/`dma_read_async` failure class was right size, right duration, wrong bytes, and only content checking catches it.

## Version skew

- `fmt_major` mismatch → refuse, offer New Game. No partial parse, ever.
- `fmt_minor` newer than the ROM → parse known fields by `npc_rec_len` stride, **preserve the unknown tail bytes verbatim** through read-modify-write so a downgrade does not amputate the newer save, and set `downgraded=1`.
- `fact_table_crc` mismatch → drop **all** `fact_bits`, keep trust/quest/topic. Fact ids are positional; a corpus edit renumbers them, and silently reinterpreting bit 7 as a different sentence is exactly the class of bug this repo keeps producing.
- `fact_bits` above the ROM's `FACT_COUNT` are preserved but never used to build a prefix.

## Prefix conditioning — the exact string

Facts reach the model only as whole trained lines, verbatim from the generated table:

```
"The hero found the crystal.\nThe hero is a friend.\nWhat lurks here?: "
```

`mem_build_prefix(const SaveNpc *n, char *out, int cap)` returns the byte count written, or `-1`. Rules, all enforced by the generator so they cannot drift:

1. `gen_memory_facts.py` **fails the build** if any fact string exceeds 26 bytes including its `\n`, or is not present in the shard corpus for that expert.
2. At most two fact lines, highest priority first, ordered by fact id for determinism.
3. Hard budget: `prefix_len + prompt_len <= 96`, leaving ≥32 for the answer (measured stop max was 35 chars, C028). Over budget → drop the lowest-priority line, then drop to zero facts; never truncate mid-line.
4. Trust maps to one of three trained tone lines by band (`< -8`, `-8..8`, `> 8`), also generated.

Because prefix bytes are part of the same 128 context, the on-screen HUD prints `MEM e=%d n=%d fx=%08lx len=%d` next to the token counter — a photograph of the TV is the evidence, per `xchk_probe.c`.

## Verification (`mem_probe.c`, `make memprobe`)

Write a save with a known pattern, drop the in-RAM copy, re-read from the medium, compare **every byte and every CRC** — not the return code of the write. Then corrupt one byte of body, one byte of header, and the whole of slot B, and assert the loader rejects/falls back in all three. Add `tools/gen_memory_facts.py --check` to CI so a stale `memory_facts.h` fails the build.

## What I did NOT verify

- **Nothing here was compiled or run.** No ROM built, no ares boot, no host test.
- SD **write** support via libcart/FatFs on the EverDrive X7, and its latency vs. the vblank budget. Read support is the measured fact; writes are assumed.
- Whether this cart declares EEPROM at all, or which size; the 2 KB digest is contingent on 16k EEPROM.
- That prefix conditioning preserves quality. The 28/28 trained-answer hit rate and 1.84 invented words/64 were measured **without** a prefix. Adding prefix lines requires retraining the shards, which produces a new model and therefore a new oracle — the 176/176 bit-exactness record does not transfer.
- RDRAM budget with two 1 MB expert slots + KV + framebuffer + save buffers; not recomputed.
- CRC32 implementation choice and its cost on the VR4300.
- `legend_of_elya.pak` (32 KB, in-tree) — not opened, not accounted for.
