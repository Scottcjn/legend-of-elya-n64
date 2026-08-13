# Streaming MoE on the N64 — the richer half of the Lock-On idea

> **STATUS: DESIGN ONLY — NOT SHIPPED, NOT BUILT.** `src/expert_cache.c` is real
> and compiles, but it appears in no Makefile target and is `#include`d by
> nothing. The ROM ships a **dense 8-layer transformer** that runs every layer
> on every token; there is no router, no expert selection, and no expert-count
> field in the weight format. If you are here because the blob's size factors as
> `12 + 65,536 + 8*196,608 + 8*49,152`, the 8 is the **layer count** — byte 4 of
> the blob header — not an expert count. Check it with
> `python3 scripts/blob_layout.py filesystem/sophia_weights.bin`, and see
> `docs/N64_RATE_FINDINGS.md` F-RT009.
>
> The "819K params (Q8)" and "858 KB" expert sizes below describe the v5 model
> that was current when this design was written. The shipped model is
> 6,356,992 ternary parameters in 2,031,628 bytes, so the capacity arithmetic
> here would need redoing before anyone builds it.

The Genesis version (`legend-of-elya-genesis/docs/LOCKON_MOE.md`) shards
experts across cartridge ROM and activates one with a pointer repoint.
Correcting an earlier claim of mine: **the N64 can do this too, and it can
do it better.** The current code loads the whole weight blob into a 1MB
RAM buffer, but that is an implementation choice, not a constraint.

## Why the N64 version is stronger

Verified against the installed libdragon headers
(`/home/scott/n64-toolchain/libdragon/include/`):

| Capability | Genesis | N64 |
|---|---|---|
| Cartridge budget | 4MB plain / 32MB w/ SEGA mapper | **64MB** (EverDrive 64) |
| Activation cost | pointer repoint (free) | PI DMA (~858KB) |
| **Activation can overlap compute** | no — nothing to overlap | **YES — `dma_read_async()`** |
| Resident expert cache | only what's bank-mapped (7 regions) | **RDRAM LRU, several experts** |
| Expert size | 114K params (ternary) | **819K params (Q8)** — 7x |
| Blend two experts (true top-2 MoE) | unaffordable | **possible** |
| Router could run on a coprocessor | Z80 (no — bus contention) | **RSP** |

Three of those are decisive.

**1. Asynchronous activation.** `dma_read_async(ram, pi_addr, len)` starts a
cartridge-to-RDRAM transfer and returns immediately; `dma_wait()` blocks
only when you actually need the data. The Genesis bank switch is
instantaneous, but there is nothing to overlap — it is all-or-nothing at
the moment of the switch. The N64 can **prefetch the next likely expert
while Elya is still speaking the current answer**. The transfer hides
completely behind generation.

**2. A real expert cache.** RDRAM is 4MB (8MB with the Expansion Pak).
After the shared embedding and KV cache, that leaves room for 3-4 resident
858KB experts (or 8+ with the Pak). Repeat questions on the same topic cost
nothing at all. The Genesis is limited to whatever the 7 remappable bank
regions hold at that instant.

**3. `dfs_rom_addr()` gives raw cartridge addresses.** libdragon's
DragonFS can hand back the ROM address of a file, so experts do not need
to be read through the filesystem layer at all — DMA straight from the
cartridge address to wherever we want them.

## Capacity

| | |
|---|---|
| Expert (4L/128d, Q8) | 858 KB |
| Shared (embedding + router + LUT) | ~35 KB |
| **Experts in a 64MB cart** | **~74** |
| Total parameters at 74 experts | **~60M** |
| Active parameters per token | **819K** — unchanged |
| RDRAM working set | shared + KV (256KB) + cache slots |

60M parameters on a 1996 console, with the working set of the model that
already runs there.

## Design

```
boot:      DMA shared block (embedding, router, exp LUT) -> RDRAM, resident
prompt:    e = route(prompt)                     [router is tiny, always RAM]
           if e is cache-resident      -> use it, zero cost
           else                        -> dma_read_async(slot, rom+off, len)
                                          show "Elya considers..." while the
                                          dungeon animation keeps running
                                          dma_wait() only at first token
generate:  ... and while generating, speculatively async-DMA the expert the
           router ranks second, into the LRU victim slot. Free if unused.
```

The speculative prefetch is the part the Genesis cannot do. It costs
nothing when wrong (the slot was going to be evicted anyway) and removes
the entire load latency when right.

## Cache policy

LRU over N slots, N = 3 without the Expansion Pak, 8 with it (detect at
boot via `get_memory_size()`). Pin nothing: the shared block is separate
and never evicted. On a miss, evict the least-recently-used slot that is
not the currently-generating expert.

## Blob format

Same SGTM container as the Genesis, with two differences:
- tensors stay in the proven **Q8 + float16 block scales** layout (the N64
  has an FPU; ternary would throw away quality it can afford to keep)
- the expert offset table is used as **cartridge offsets** for DMA rather
  than as pointers

Keeping one container format across both consoles means one trainer, one
exporter, and one set of test vectors.

## Honest risks

1. **DMA latency is real.** 858KB over the PI bus is roughly 30-170ms
   depending on cart hardware. Hidden by prefetch when the router is right;
   visible as a brief pause when it is wrong. Measure on real hardware
   before tuning the cache size — EverDrive 64 timings differ from a
   mask ROM.
2. **The N64 model is 7x larger per expert**, so a wrong route wastes far
   more than on the Genesis. Router accuracy matters more here.
3. **Expansion Pak detection must be honest** — if absent, 3 slots, and
   the code must not assume 8.
4. Everything about the `dma_read_async` overlap depends on the CPU not
   touching the destination buffer mid-transfer. The victim slot must be
   genuinely unused, not merely "probably not needed."
