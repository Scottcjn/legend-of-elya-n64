# N64 Measurement Instrument + expert_cache Audit — FINDINGS

Working copy: $S/n64 (copy of ~/legend-of-elya-n64, NOT the genesis repo).
Never modified: ~/legend-of-elya-genesis, ~/legend-of-elya-n64.

## F0 — Repo location correction (2026-08-04)
Task brief said the N64 variant lives in ~/legend-of-elya-genesis. THAT IS STALE.
~/legend-of-elya-genesis is a Sega Genesis / m68k / SGDK project (Makefile exports
GDK := /home/scott/marsdev/mars/m68k-elf). It has train/elya_n64.pt and
train/champions/n64_best.bin (weights only) but ZERO N64 engine source and NO
expert_cache anywhere (grep -rn expert_cache -> 0 hits).

Actual N64 engine is ~/legend-of-elya-n64:
- src/expert_cache.c, src/expert_cache.h, src/expert_cache_test.c
- src/nano_gpt.c, src/nano_gpt.h, src/legend_of_elya.c
- HEAD = 28e1ffe "N64 Streaming MoE: expert cache with async prefetch"

Toolchain: /home/scott/n64-toolchain/mips64-toolchain/bin/mips64-elf-gcc
Emulator:  /usr/games/mupen64plus 2.6.0 (--emumode, dummy plugins)

## Status log
- [x] Located real sources
- [ ] Read engine
- [ ] Build instrument

## F1 — expert_cache defect (a) "wired to nothing": **REAL, and worse than reported**
Evidence:
- `grep -rn "ec_init|ec_acquire|ec_request|ec_prefetch|ec_resident|ExpertCache|expert_cache"`
  over all *.c/*.h/Makefile returns hits ONLY inside `src/expert_cache.{c,h}` and
  `src/expert_cache_test.c`. No engine file references it.
- `grep -n "src/" Makefile` -> ZERO hits. The Makefile compiles `legend_of_elya.c`
  and `nano_gpt.c` from the REPO ROOT. The entire `src/` directory is dead code
  w.r.t. every ROM target.
- `build/` contains only `legend_of_elya.o` and `nano_gpt.o`. No `expert_cache.o`.
- HEAD commit 28e1ffe ("N64 Streaming MoE: expert cache with async prefetch") added
  `src/` as a *parallel copy* of the engine; root and src copies have since diverged:
  `diff -q nano_gpt.c src/nano_gpt.c` -> differ (641 vs 467 lines)
  `diff -q legend_of_elya.c src/legend_of_elya.c` -> differ (1844 vs 1173 lines)
So it is not merely "allocated but unused" — it is not compiled at all, and it sits
next to a stale fork of the engine. Any claim about its runtime behaviour to date is
unfounded because it has never executed on hardware or in an emulator.

## F2 — expert_cache defect (b) D-cache coherency: **REAL**
`src/expert_cache.c` uses `dma_read_async()` + `dma_wait()` and performs **no cache
maintenance whatsoever** (grep for `cache`/`Inval`/`Writeback` in expert_cache.c -> 0).
libdragon does not do it for you:
- Disassembled `dma.o` from `/home/scott/n64-toolchain/mips64-toolchain/mips64-elf/lib/libdragon.a`.
  `dma_read_raw_async` is: disable_interrupts -> spin on PI status -> `sw` the three PI
  registers (ram_addr, cart_addr, wr_len) -> enable_interrupts. **Zero `cache` instructions
  in the whole object.**
PI DMA writes straight into RDRAM behind the CPU's back. The R4300i has an 8 KB
write-back D-cache, so:
  1. stale VALID lines over the destination -> CPU reads pre-DMA garbage;
  2. stale DIRTY lines over the destination -> a later eviction WRITES OVER the
     freshly DMA'd weights.
Both produce intermittently wrong tokens, not a crash. Correct fix is
`data_cache_hit_writeback_invalidate(slot->mem, expert_len)` immediately BEFORE
starting the transfer (libdragon's spelling of `osWritebackDCache`+`osInvalDCache`).

## F3 — expert_cache defect (c) prefetch self-disables: **REAL**
`ec_prefetch()` line 145: `if (ec->inflight_slot != EC_NO_EXPERT) return;`
`inflight_slot` is cleared ONLY by `ec_settle()`. `ec_settle()` is called only from
`ec_start_load()` and from the two miss/prefetch-hit paths of `ec_acquire()`.
The `ec_acquire()` HIT path (line 119-123, `s>=0 && !loading`) returns WITHOUT settling.
Therefore the steady state is: prefetch expert B while generating from resident expert
A -> `inflight_slot = slot_B`, `slot_B.loading = 1`. Every subsequent token calls
`ec_acquire(A)` which takes the hit path and never settles. From then on:
  - every `ec_prefetch(...)` returns immediately at line 145 -> prefetching is OFF forever
  - `ec_resident(B)` stays false forever even though the DMA finished long ago
  - `ec_victim()` skips slot_B forever (`if (loading) continue;`), so a slot is leaked
The DMA has physically completed; only the bookkeeping is stuck. Fix: poll `dma_busy()`
in a non-blocking `ec_poll()` and clear the flag, and call it from the acquire hit path.

## F4 — **The engine is nondeterministic by construction** (blocks the naive correctness gate)
`nano_gpt.c:218` `pse_entropy()` does `asm volatile("mfc0 %0, $9")` — it reads the very
CP0 Count register we want to use as the instrument — and xorshift-mixes it.
`nano_gpt.c:229` `pse_burst_inject()` uses that value to add +/-2% RMS noise to 8 of the
256 activation dims, every 8th token, and it is called UNCONDITIONALLY from
`sgai_next_token()` (line 582), *before* `project_to_logits`. There is no #ifdef.

Consequences that dominate this whole task:
1. Even with `temperature_q8 == 0` (greedy argmax, `sample_logits` line 433) the token
   stream depends on elapsed CPU cycles. On real hardware output is NOT reproducible.
2. **Any optimization that changes timing changes the injected noise, which can change
   the tokens.** "Byte-identical output" and "make it faster" are therefore in direct
   conflict in the shipped configuration.
3. It also makes cycle numbers incomparable between runs: different noise -> different
   argmax -> different token -> different downstream work.
DECISION: the bench ROM gets a compile-time `BENCH_DET_PSE` switch that replaces the
`mfc0 $9` read with a pure counter-seeded LCG. Both the correctness gate and the cycle
measurements run in that mode, so the ROM does identical work every run. The stock
CP0-entropy path is measured separately, and its run-to-run token variance is reported
as a property of the shipped engine rather than hidden.

## F5 — libdragon ROMs DO NOT BOOT under mupen64plus 2.6 (blocker, worked around)
Symptom, on BOTH a hello-world libdragon ROM and the repo's own prebuilt
`legend_of_elya.z64`:
```
Core: Initializing 4 RDRAM modules for a total of 8 MB
Core: Starting R4300 emulator: Pure Interpreter
Core Warning: IPL3 detected 64 MB of RDRAM != 8 MB
<nothing ever again>
```
libdragon's own IPL3 ("Libdragon IPL3 Coded by Rasky", embedded in `n64tool` as
`default_ipl3`) mis-detects mupen64plus's RDRAM as 64 MB. libdragon then places the
stack/heap at the top of "64 MB" (~0x84000000), which mupen64plus maps to
read_nothing/write_nothing, so the ROM wedges before `main()` ever runs.
Disassembled mupen64plus's handler for that warning (`libmupen64plus.so.2` @0x40ac7):
it compares, prints, and falls through to the same continuation — it does NOT correct
`osMemSize`. `--set Core[DisableExtraMem]=1` does not help (still "64 MB != 8 MB").
=> Every previously reported N64 performance number obtained via mupen64plus is
   suspect, because a stock libdragon ROM does not reach main() on this emulator.

## F6 — WORKING OUTPUT CHANNEL: ISViewer, protocol reverse-engineered from the core
mupen64plus 2.6 DOES emulate the IS-Viewer. Exact protocol, read out of
`libmupen64plus.so.2` `write_is_viewer` @0x1e620:
```
and $0xfff,%esi        ; address mask is 0xFFF  (4 KB window, NOT 0xFFFF)
cmp $0x14,%esi         ; offset 0x14 == "flush length" register
test %edx,%edx         ; length 0 -> ignored
lea 0x20(%rbx),%rsi    ; payload buffer starts at offset 0x20
memchr(buf, '\n', len) ; ** only emits a line when a '\n' is present **
```
So: base `0xB3FF0000` (phys 0x13FF0000), payload at `+0x20`, write byte-length to
`+0x14`, and the payload MUST contain `\n`. Output appears on stdout as `Core: IS64: <line>`.
libdragon's `debug_init_isviewer()` handshake is irrelevant — raw `io_write`s work.

## F7 — VALIDATED: bare-metal custom IPL3 boots and prints
40-byte hand-written IPL3 (`$S/bare/ipl3.S`), assembled to ROM offset 0x40, ROM header
copied from a libdragon ROM. mupen64plus PIF-HLE copies ROM[0x40..0x1000] into SP DMEM
and executes at 0xA4000040 as expected.
```
Core: Starting R4300 emulator: Pure Interpreter
Core: IS64: HI64            <-- our string
```
This is the instrument's transport layer, proven end-to-end. It bypasses libdragon's
IPL3 entirely, so F5 no longer blocks measurement.

## F8 — **The shipped base ROM never loads its weights.** The LLM is inert.
`legend_of_elya.c:1742-1755`:
```c
int fd = dfs_open("/sophia_weights.bin");
if (fd >= 0) {
    static uint8_t wbuf[3 * 1024 * 1024] ...;  /* "3MB for v8 Q8 6-layer 192-embed" */
    int sz = dfs_size(fd);
    if (sz > 0 && sz <= (int)sizeof(wbuf)) {   /* <-- guard */
        dfs_read(wbuf, 1, sz, fd);
        sgai_init(&G.ai, wbuf);
        G.ai_ready = 1;
    } else { dfs_close(fd); }                  /* <-- silently taken */
}
```
Measured facts:
- `filesystem/sophia_weights.bin` = **6,750,220 bytes**  (git-tracked, and it is the
  Makefile prerequisite for the DFS)
- `sizeof(wbuf)` = 3,145,728 bytes
- 6,750,220 > 3,145,728  ->  the else branch is taken, `sgai_init` is never called,
  `G.ai_ready` stays 0.
- The blob really is in the ROM: prebuilt `legend_of_elya.z64` is 6,946,816 bytes and
  contains the `SEAI` magic at ROM offset 0x2c800 with 6,764,544 bytes following;
  `build/legend_of_elya.dfs` is 6,752,000 bytes.
- The blob size matches the *current* model exactly: 8 layers x 835,584 B/layer +
  65,536 B embedding + 12 B header = 6,750,220. So weights and `nano_gpt.h`
  (8 layers / 256 embed / 8 heads / ctx 128) agree; only `wbuf` is stale — its comment
  still describes the old "6-layer 192-embed" model.
Consequence: with weights unloaded, `sgai_next_token()` skips the whole transformer
(`if (state->is_loaded && state->weights != NULL)`, line 573) and only runs
embedding + rms_norm + PSE + logit projection. **Any previously reported token
throughput for the base ROM was measuring an empty shell.**

## F9 — The model does not fit in RDRAM at all; streaming is mandatory, not optional
weights 6,750,220 B + `SGAIKVCache` (8 layers x 128 ctx x 256 embed x 4 B x 2 (k,v))
= 2,097,156 B  ->  8,847,376 B **> 8,388,608 B (8 MB RDRAM, Expansion Pak maximum)**.
Even after fixing F8 the ROM cannot hold weights and KV cache in RDRAM simultaneously.
This is exactly the problem `expert_cache` was written to solve — and per F1 it is not
compiled into anything. The nano_gpt.c banner comment ("Weights in ROM (DMA'd on
demand): ~848KB ... Total RDRAM: ~263KB") describes a 4-layer/128-embed model that no
longer exists; the header says 8/256.

## F10 — INSTRUMENT BUILT: bare-metal CP0-Count bench harness
Lives in `$S/n64/bench/`. Nothing in it touches `~/legend-of-elya-n64`.
```
ipl3.S      88-byte custom IPL3: enables CU1/CU0/FR, PI-DMAs 256 KB of payload
            from cart 0x10001000 -> RDRAM 0x00020000, jumps to 0x80020000.
            It deliberately does NOT probe RDRAM size (that probe is what wedges
            libdragon's IPL3 under mupen64plus — F5).
start.S     sets sp=0x807FFFF0, zeroes .bss, calls bench_main().
shim/       freestanding memcpy/memset/strlen/memalign + the ISViewer transport.
            nano_gpt.c is compiled UNMODIFIED apart from the BENCH_DET_PSE guard.
bench.c     reads CP0 Count ($9) immediately before and after each
            sgai_next_token(), wrap-safe 32-bit delta accumulated into 64-bit.
build.sh    uses the SAME codegen flags as the real libdragon build (n64.mk):
            -march=vr4300 -mtune=vr4300 -falign-functions=32 -ffast-math
            -ftrapping-math -fno-associative-math -O2 -std=gnu99
mkrom.py    header | IPL3 | payload@0x1000 | weights@cart 0x00400000
run.sh      mupen64plus --emumode 0 --nospeedlimit, all-dummy plugins, scrapes
            "Core: IS64: " off stdout.
```
RDRAM map: payload 0x80020000, arena/KV 0x80070000, weights (RDRAM cfg) 0x80100000
(6,750,224 B, ends 0x8077000C), stack 0x807FFFF0.

First end-to-end run (weights dereferenced straight out of cart KSEG1):
```
BENCH_START
CFG weights=CART_UNCACHED ctx=128
MAGIC 1397047625          <- 0x53454149 == "SEAI", header really is being read
LOADED 1 KV 0
FATAL_INIT                <- memalign(2,097,156) failed against a 589,824 B arena
```
The harness correctly detected and REPORTED its own failure instead of silently
producing numbers. Arena raised to 3 MB for the cart config.

## F11 — **INSTRUMENT VALIDATED. And it is NOT a cycle counter.** (calibration ROM)
`bench/calib.c`, 100,000 iterations of hand-written asm loops with exactly known
instruction counts:
```
A_empty3      3 instr/iter  (addiu,bne,nop)              count 600,010
B_empty11    11 instr/iter  (A + 8x nop)                 count 2,200,010
C_rdram_ld11 11 instr/iter  (A + 8x lw from KSEG0 RDRAM) count 2,200,012
D_cart_ld11  11 instr/iter  (A + 8x lw from KSEG1 CART)  count 2,200,012
E_stride_ld  strided 4 MB walk, 8 loads/iter             count 5,600,016
```
- A: 300,000 instructions -> 600,010 counts. B: 1,100,000 -> 2,200,010.
  **CP0 Count advances by exactly 2 per retired instruction.** (+10 = the mfc0
  bracket itself.) mupen64plus's `CountPerOp` default is 2 and it is applied
  per *op*, not per cycle.
- C vs B: 8 cached RDRAM loads cost **+2 counts total** vs 8 nops.
- D vs B: 8 **uncached cart-ROM** loads cost **+2 counts total** vs 8 nops.
  On real R4300i hardware one uncached PI read is on the order of hundreds of
  cycles. mupen64plus prices it the same as a `nop`.
=> **The instrument measures instructions retired, not cycles.** It is exact,
   deterministic and perfectly usable for measuring anything that changes the
   instruction count. It is *completely blind* to D-cache misses, RDRAM latency
   and PI latency — i.e. blind to the exact thing the N64 is bottlenecked on.
   Any RDRAM-latency optimization (Step 4) CANNOT be validated on this emulator.
   Independent confirmation from the model itself: see F13.

## F12 — RUN-TO-RUN VARIANCE OF THE INSTRUMENT
Same ROM (`bench_ram.z64`), two consecutive headless runs, 16 tokens:
- **Token IDs: identical in all 16 positions.**
- Per-token counts: 14/16 bit-identical. TOK 7 differed by 2 counts, TOK 15 by 4.
- TOTAL_COUNT 2,356,661,418 vs 2,356,661,424 -> delta 6 on 2.36e9
  = **2.5 parts per billion (0.00000025 %)**.
So the instrument is reproducible to ~1e-8. Anything above ~0.001 % is signal.
Suspected source of the residual jitter: mupen64plus `Core[RandomizeInterrupt]`
defaults to True (it randomizes PI/SI interrupt timing). Tested below.

## F13 — Cross-config validation: identical tokens, and proof the timing model is flat
Two *different* builds — cart-uncached weights at ctx=128, vs RDRAM weights at
ctx=32 — produced the **same 16 tokens**:
`108 121 97 110 32 76 97 98 115 46 58 32 65 32 119 111`  = "lya Labs.: A wo"
(input was the prompt "Elya", so the model is really running and really coherent).
This is a genuine correctness cross-check: it validates that reducing SGAI_CTX from
128 to 32 does not perturb the first 16 tokens (all attention loops are bounded by
`n_ctx = min(pos+1, SGAI_CTX)`, so only the KV stride changes), and it validates
that the cart-resident and RDRAM-resident weight paths read the same bytes.
It also independently confirms F11: pulling ~6.3 M weight bytes per token over the
PI bus instead of out of RDRAM changed TOTAL_COUNT by 774 counts out of 2.36e9
(0.00003 %). On real hardware that difference would be one to two orders of
magnitude in wall time.

## F14 — BASELINE (the honest number)
Config: RDRAM-resident weights, ctx=32, greedy (temperature_q8 = 0),
deterministic PSE (`BENCH_DET_PSE`), prompt "Elya", 16 forward passes.
```
TOK  0  in 69('E')  out 108   count 146,820,206
TOK  1  in 108      out 121   count 146,886,276
TOK  2  in 121      out  97   count 146,949,084
TOK  3  in  97      out 110   count 147,011,816
TOK  4  in 110      out  32   count 147,074,374
TOK  5  in  32      out  76   count 147,136,174
TOK  6  in  76      out  97   count 147,198,624
TOK  7  in  97      out  98   count 147,263,526
TOK  8  in  98      out 115   count 147,322,728
TOK  9  in 115      out  46   count 147,385,076
TOK 10  in  46      out  58   count 147,447,116
TOK 11  in  58      out  32   count 147,509,356
TOK 12  in  32      out  65   count 147,571,674
TOK 13  in  65      out  32   count 147,632,198
TOK 14  in  32      out 119   count 147,695,240
TOK 15  in 119      out 111   count 147,757,952
TOTAL_COUNT 2,356,661,420
```
Interpretation: Count = 2 x instructions => **~73.4 M instructions per token**,
~1.18 G instructions for 16 tokens. The monotone rise across tokens (146.82 M ->
147.76 M, +0.64 %) is the attention loop growing with context — a good sanity signal
that the measurement tracks real work.
On a real 93.75 MHz R4300i, if it retired 1 instruction/cycle with zero stalls,
73.4 M instructions is **~0.78 s/token floor**. Real hardware would be far worse
because of the memory stalls this emulator does not model. Weight traffic is
~6.3 MB per token (every weight is touched once per forward pass), so RDRAM
bandwidth alone puts a hard floor on it.

## F15 — Residual jitter eliminated
`--set Core[RandomizeInterrupt]=0` is now in `bench/run.sh`. mupen64plus defaults it
to True (it randomizes PI/SI interrupt timing), which was the source of the +-2..4
counts in F12. With it disabled the calibration ROM is bit-identical run to run.
Runner note: mupen64plus must be stopped with **SIGINT, not SIGTERM** — on SIGTERM it
does not flush stdout and every IS-Viewer line is lost. That cost an hour early on and
is exactly the kind of thing that would silently produce "no output => it crashed".

## F16 — expert_cache defect (c) REPRODUCED EXECUTABLY, then FIXED
The existing suite `src/expert_cache_test.c` passes 16/16 while the bug is live,
because every `ec_prefetch()` in it is immediately followed by `ec_acquire()` of the
SAME expert — and that acquire is the only thing that ever clears `inflight_slot`.
New file `src/expert_cache_regress.c` models the actual streaming pattern (prefetch B,
keep generating from A) with a stub PI engine that reports busy/idle honestly.

BEFORE the fix:
```
  prefetch of B started a DMA                              ok
  B is recognised as resident once its DMA finished        FAIL
  a SECOND prefetch is still able to start a DMA           FAIL
  C also becomes resident                                  FAIL
  a THIRD prefetch can still find a victim slot            FAIL
FAILED (4 failures)
```
AFTER: both suites 100% green, and `mips64-elf-gcc -Wall -Wextra -Werror` clean.

Fix (`src/expert_cache.{c,h}`):
1. Added `ec_poll()` — the non-blocking counterpart to `ec_settle()`. It reads
   PI_STATUS (0xA4600010, bits 0-1) directly rather than libdragon's `dma_busy()`,
   which is `__attribute__((deprecated))` and would trip n64.mk's `-Werror`.
   Called at the top of `ec_resident`, `ec_request`, `ec_acquire` (**including the
   hit path**, which is the one that was missing) and `ec_prefetch`.
2. Defect (b): `ec_start_load()` now calls
   `data_cache_hit_writeback_invalidate(slot->mem, expert_len)` immediately before
   `dma_read_async()`.
3. A semantic bug I INTRODUCED and then fixed: once `ec_poll()` clears `loading`,
   `ec_acquire()` took the ordinary hit path, so `prefetch_hits` would have read zero
   forever — the fix would have silently destroyed the statistic that proves the
   feature works. Added `EcSlot.prefetched`, set by `ec_start_load(..., speculative)`
   and consumed on first acquire, so a successful speculation is credited whether or
   not its DMA had already landed. `ec_test_busy()` stub added to the original suite.

Honest limitation: this is verified on the HOST policy model. It cannot be verified
on hardware timing here, because (i) expert_cache is still not compiled into any ROM
(F1/F17) and (ii) mupen64plus prices memory access at zero (F11) and does not emulate
the D-cache at all, so defect (b) is *unobservable* on this emulator by construction.
The D-cache fix is justified from the R4300i manual + the libdragon disassembly, not
from a measurement, and I am not claiming otherwise.

## F17 — expert_cache defect (a): the wiring is a bigger job than a patch, and I did NOT fake it
To actually use `expert_cache` the ROM would need, at minimum:
1. the weight blob resharded into N uniform per-expert slabs with a table of offsets
   (`ec->expert_off[]`), which the current single monolithic `SEAI` blob is not;
2. `nano_gpt.c` to stop treating `state->weights` as one contiguous `SGAIHeader*`
   and instead resolve each `SGAILayer*` through `ec_acquire()`;
3. a router that picks the expert per token, plus a prefetch call site;
4. `expert_cache.c` added to the `legend_of_elya.elf` prerequisites in the Makefile
   (and moved out of the stale `src/` fork — see F1).
That is a redesign of the weight path, not a patch, and I could not do it and verify
it honestly in this budget. So: defects (b) and (c) are FIXED and tested; defect (a)
is documented, and `expert_cache.c` is confirmed to still compile clean for the N64
target (`mips64-elf-gcc -Wall -Wextra -Werror`) so it has not bit-rotted.

## F18 — A/B experiment: hoist the per-block scale out of the matmul inner loop
`matmul_q8()` inner loop is `acc += (float)row_w[j] * scale * input[j];` — two FP
multiplies per weight, even though `scale` is constant across each 32-weight block.
Hoisting it makes 32 `mul.s` into 1. GCC will not do this itself: n64.mk passes
`-fno-associative-math`. It IS a reassociation, so bit-identity is a question to be
measured, not assumed.

**Static instruction count of `matmul_q8` (objdump):**
```
base   110 instructions
hoist  114 instructions   <-- static counting says the optimization is WORSE
```
This is precisely why Hard Rule 1 exists. The dynamic measurement follows.

## F19 — Hard Rule 3 check: the REAL ROM builds, not just the bench
After every source change in this session, both real libdragon targets were rebuilt
with the local toolchain and link clean:
```
make base     N64_INST=/home/scott/n64-toolchain/mips64-toolchain  -> legend_of_elya.z64      6,946,816 B
make base-rsp N64_INST=/home/scott/n64-toolchain/mips64-toolchain  -> legend_of_elya_rsp.z64
```
`base-rsp` matters because it compiles `nano_gpt.c` a SECOND way (`-DUSE_RSP_MATMUL`),
which takes the other side of the `#ifdef` around `matmul_q8` — the bench ROM only ever
links the CPU path, so this is the build that would have caught a one-sided edit.
(The repo's `N64_INST` default `/home/sophia5070node/n64dev/mips64-toolchain` does not
exist on this machine; the local toolchain is `/home/scott/n64-toolchain/mips64-toolchain`.)
Real-ROM `.bss` is 5,262,200 B = the 3 MB `wbuf` + the 2 MB static `G.kv` + change,
which independently corroborates F9: raising `wbuf` to the real 6.75 MB would put
`.bss` at ~9 MB, i.e. over the 8 MB console maximum.

## F20 — Recommended fix for F8, NOT applied (unverifiable here)
The viable fix is to stop copying the blob into RDRAM and point the engine straight at
the cart, which libdragon supports directly:
```c
uint32_t rom = dfs_rom_addr("/sophia_weights.bin");   /* declared dragonfs.h:292 */
if (rom) { sgai_init(&G.ai, (const void *)(0xA0000000u | rom)); G.ai_ready = 1; }
```
That removes the 3 MB `wbuf` entirely and makes the model fit (F9).
**Evidence that the mechanism works**: my bench Config A does exactly this — weights
dereferenced through KSEG1 out of the cart at ctx=128 — and it produced 16 correct
tokens identical to the RDRAM path (F13).
**Why I did not apply it to `legend_of_elya.c`:** libdragon ROMs do not boot on the only
emulator available here (F5), so I would be shipping an unverified change to the real
game path. Per the brief's own standard, an unverifiable edit to the shipping ROM is
worse than a documented recommendation. It is also a performance *cliff* on real
hardware — every weight byte becomes a PI read — which is the argument for finally
wiring up `expert_cache` (F17) rather than either extreme.

## F21 — MY OWN MISTAKE, caught before it produced a claim
I wrote the `BENCH_DET_PSE` guard in `nano_gpt.c` (F4) and then **never added
`-DBENCH_DET_PSE` to `bench/build.sh`**. Every measurement up to and including the F14
baseline therefore ran the SHIPPED CP0-entropy path, not the deterministic one.
Why it did not invalidate F11/F12/F13/F14: under mupen64plus, CP0 Count is itself a
deterministic function of instructions executed (F11), so with RandomizeInterrupt=0 the
stock path is still reproducible *on this emulator*. Those numbers stand.
Why it DOES invalidate the F18 A/B as run: the hoist changes the instruction count ->
changes CP0 Count -> changes the injected entropy -> can change tokens for a reason
that has nothing to do with the arithmetic reassociation being tested. A token
difference would have been unattributable. Re-running the A/B with `-DBENCH_DET_PSE`
on both arms, which is the only way the byte-identity question is answerable.
Recorded rather than quietly fixed, because "the harness silently wasn't testing what
I said it was testing" is the single most valuable kind of finding in this exercise.

## F22 — CORRECTION to F15: RandomizeInterrupt=0 does NOT fully remove the jitter
F15 was validated only on the *calibration* ROM, which runs a few milliseconds of
emulated time and so never takes a VI interrupt. On the full 16-token bench, with
`RandomizeInterrupt=0`, four independent runs of the identical `base` ROM gave:
```
TOK 7    147,263,526 / 147,263,528 / 147,263,530
TOK 15   147,757,950 / 147,757,952 / 147,757,954
TOTAL  2,356,661,418 / ...420 / ...424 / ...426
```
Token IDs identical in every run. Spread on TOTAL_COUNT = 8 counts on 2.356e9
= **~3.4 parts per billion**. Two tokens out of sixteen carry all of it, which is
consistent with a periodic interrupt (VI) landing on a different instruction boundary
rather than with anything in the model.
**Reported instrument variance: +-8 counts (~3.4 ppb) on a 16-token run.**
Anything above ~0.001 % is signal by a margin of ~4 orders of magnitude.
So F15's "bit-identical" claim holds for short runs only, and is hereby narrowed.

## F23 — Files produced by this session (all inside the working copy, nothing upstream touched)
NEW:
  FINDINGS.md                     this journal
  bench/ipl3.S                    88-byte custom IPL3
  bench/start.S                   payload entry (sp, bss)
  bench/link.ld                   payload at 0x80020000
  bench/bench.c                   CP0-Count harness around sgai_next_token()
  bench/calib.c                   instrument calibration ROM
  bench/shim/libdragon.h          freestanding stand-in
  bench/shim/rt.c                 memcpy/memset/memalign + ISViewer transport
  bench/build.sh bench/build_calib.sh bench/mkrom.py bench/run.sh bench/romhdr.bin
  src/expert_cache_regress.c      failing-then-passing regression test for defect (c)
MODIFIED (working copy only):
  nano_gpt.h        2 lines   SGAI_CTX made overridable (#ifndef guard)
  nano_gpt.c       22 lines   BENCH_DET_PSE guard; OPT_HOIST_SCALE A/B arm
  src/expert_cache.c 59 lines  ec_poll(); D-cache writeback+invalidate; prefetched flag
  src/expert_cache.h  1 line   EcSlot.prefetched
  src/expert_cache_test.c 1 line  ec_test_busy() stub
`~/legend-of-elya-genesis` and `~/legend-of-elya-n64` were never written to; no commit,
no push, no PR.

## F24 — Known limitation of the bench ROM: emulator-only, by construction
The bench ROM carries a hand-written IPL3, so mupen64plus reports
`Unknown CIC type (000000844C074690)! using CIC 6102`. On a real console the PIF and
the CIC exchange a checksum computed over ROM[0x40..0x1000] during boot, and a custom
IPL3 will not match any stock CIC — the machine would lock up. Running the bench on
real hardware would need either a flashcart that patches the CIC handshake
(64drive / EverDrive do this) or a correctly-checksummed stock IPL3.
What this does and does not cost:
- It does NOT affect the validity of the measurements *as instruction counts*: the
  engine translation unit is compiled with the identical flags as the shipping ROM
  (`build.sh` mirrors n64.mk), and only the boot stub differs.
- It DOES mean the harness cannot currently be pointed at real silicon, which is the
  one place the RDRAM-latency question (Step 4) could actually be answered.

## F25 — Secondary defect found while reading the weight-load path (not fixed)
`legend_of_elya.c` keeps its KV cache as a static member `G.kv` (line 178) and assigns
`G.ai.kv = &G.kv` at line 1723 and again at 1750. But `sgai_init()` (nano_gpt.c:524)
*also* does `state->kv = memalign(8, sizeof(SGAIKVCache))` and memsets it.
So on the path where the weights DO load, the engine allocates a 2 MB KV cache from
the heap and the caller then immediately overwrites the pointer with `&G.kv` — the
2 MB allocation is leaked, on a machine with 8 MB of RAM total. Today this is masked
because that branch is dead (F8), which is the only reason it has not been noticed.
Any fix to F8 must also resolve the double ownership, or the ROM will lose 2 MB at
startup on top of the 3 MB `wbuf` and the 2 MB `G.kv`.
Not fixed here for the same reason as F20: I cannot boot the real ROM to verify it.

## F26 — MEASURED RESULT of the scale-hoist (stock-PSE arm)
```
                     TOTAL_COUNT        per token (TOK 0)
base                 2,356,661,426      146,820,206
hoist                2,180,504,240      135,810,382
delta                 -176,157,186       -11,009,824
                          -7.47 %            -7.50 %
```
**Output: BYTE-IDENTICAL across all 16 tokens.**
```
base   108 121 97 110 32 76 97 98 115 46 58 32 65 32 119 111
hoist  108 121 97 110 32 76 97 98 115 46 58 32 65 32 119 111
```
Effect size 7.47 % vs instrument variance ~3.4e-7 % (F22) — signal by seven orders
of magnitude.

**This is the Hard-Rule-1 case study.** Static instruction counting of `matmul_q8`
said the change made the function BIGGER (110 -> 114 instructions) and would have
rejected it. The dynamic measurement says it is a 7.5 % win. The static count is
larger because hoisting adds a block-level accumulator and epilogue (executed once per
32 weights) while removing one `mul.s` from the inner loop (executed 32 times per
block). Static size and dynamic cost move in opposite directions here.

Sanity check that the number is the right order of magnitude: the baseline is ~11.5
instructions per MAC (73.4 M instructions / 6.36 M MACs per token). Removing one of
those instructions predicts ~8.7 %; measured 7.5 %. Close enough to believe, not so
exact as to look fabricated.

Caveat pending: this arm ran the stock CP0-entropy PSE path (F21), so identical
tokens here is slightly lucky rather than proven — a changed instruction count changes
the injected noise. The `BENCH_DET_PSE` arm removes that confound; results below.

## F27 — DETERMINISTIC-PSE ARM: zero variance, and the A/B confirmed
Both arms rebuilt with `-DBENCH_DET_PSE` (F21 fix), same config (RDRAM weights,
ctx=32, greedy, prompt "Elya", 16 forward passes).

**Instrument variance is now EXACTLY ZERO.** Two independent runs of `det_base`:
`diff res_det_base.txt res_det_base2.txt` -> no differences at all. Every one of the
16 per-token counts and TOTAL_COUNT are bit-identical.

This *reinterprets* F12/F22. The +-8-count jitter was never instrument noise. It was
the ENGINE: `pse_entropy()` reads CP0 Count and injects it into the activations
(F4), so a VI interrupt landing one instruction earlier feeds back into the model's
own arithmetic and perturbs the next measurement. Remove that feedback loop and the
whole harness is deterministic to the last count.
**Corrected instrument variance: 0 counts with BENCH_DET_PSE; ~8 counts (3.4 ppb)
with the shipped CP0-entropy path.**

**A/B result, confound removed:**
```
                 TOTAL_COUNT        tokens
det_base       2,356,669,652        "lyan Labs.: A wo"
det_hoist      2,180,512,468        "lyan Labs.: A wo"
delta           -176,157,184        IDENTICAL, all 16
                    -7.4749 %
```
Against zero variance, a 7.47 % delta is unambiguous. And it agrees with the stock-PSE
arm (F26: -176,157,186 / -7.47 %) to within 2 counts, which is itself a nice
cross-check that the two arms measured the same change.

VERDICT on the scale hoist: **7.47 % fewer instructions, byte-identical output over
16 tokens, verified twice with two different entropy configurations.** Static
instruction counting would have rejected it (F18: 110 -> 114 instructions).
It is currently behind `-DOPT_HOIST_SCALE` and is NOT enabled in the shipping ROM
build, because the shipping ROM cannot be booted here to confirm end-to-end (F5).

# ===== ares detour (coordinator unblock) =====

## F28 — ares CAN boot libdragon ROMs. F5 was mupen64plus-specific, not universal.
`flatpak run dev.ares.ares --system "Nintendo 64" --no-file-prompt --setting Video/Driver=None
 --setting Audio/Driver=None <rom>`, under `xvfb-run -a`.

Two gotchas that cost the first three attempts and produced *zero output each time*:
1. **flatpak cannot see `/tmp`.** Its context is `filesystems=home;host;` but flatpak
   always gives the app a private `/tmp`. `flatpak run --command=ls dev.ares.ares
   /tmp/claude-1000` -> "No such file or directory", while `/home/scott` is visible.
   All ROMs must be staged under `$HOME` (I use `~/.cache/ares-n64-bench/`).
   With the ROM under /tmp, ares printed nothing at all and looked like a boot failure.
2. ares, like mupen64plus, does not flush stdout on SIGTERM — needs SIGINT.

Result with the ROM staged correctly, on the libdragon probe ROM:
```
Loaded probe
Vulkan Enabled: using paraLLEl-RDP
RAWISV_HELLO            <-- our raw IS-Viewer write, from a stock libdragon ROM
```
So: **ares boots stock libdragon ROMs and implements the IS-Viewer at the same
addresses mupen64plus does** (base 0xB3FF0000, payload +0x20, length +0x14).

## F29 — ares rejects my custom IPL3, exactly as predicted in F24
```
[unusual] [PIF::main] invalid IPL2 checksum: CIC-NUS-6102:a536c0f1d859 != cpu:2d640b38e9ff
```
and the bare-metal calib ROM produced no IS-Viewer output under ares. F24 predicted
this ("a custom IPL3 will not match any stock CIC"). mupen64plus does not check;
ares does. Consequence: the bare-metal harness is mupen64plus-only, and every ROM I
want to run under ares must be a **stock libdragon build**. Rebuilding the harness
that way, which also removes the F24 caveat entirely.

## F30 — probe ROM hangs inside libdragon's own `debug_init_isviewer()` under ares
The probe printed `RAWISV_HELLO` and then nothing — `LD_ISV_INIT=` never appeared.
So execution stops in `debug_init_isviewer()` (or the SRAM `dma_write_raw_async` that
follows it, with no save type configured). Not chased further: the raw IS-Viewer
writes work and are all the harness needs. Noted so it is not mistaken for a boot
failure later.

## F31 — **ares PRICES MEMORY REALISTICALLY. mupen64plus does not.** (the headline)
Same calibration experiment as F11, rebuilt as a stock libdragon ROM (`$S/ld/calib_ld.c`)
so it boots on both emulators. 100,000 iterations per case.

```
                    ares count      mupen64plus count
A_empty3   (3 i)       150,026            600,010
B_empty11 (11 i)       550,050          2,200,010
C_rdram_hit(11 i)      950,089          2,200,012
D_cart_uncached        108,550,050      2,200,012      <-- 49x apart
E_rdram_miss           17,400,105       5,600,016
F_rdram_hit2           1,300,094            n/a
```
Derived, per access (ares; CP0 Count ticks once per 2 CPU cycles on real R4300i):
```
A: 300,000 instructions -> 150,026 counts = 300,052 cycles  => 1.00 cycles/instruction
B: 1,100,000 instr      -> 550,050 counts                   => 1.00 cycles/instruction
cached RDRAM load   :     1.00 cycles each
D-cache MISS        :    40.25 cycles each   (E minus F, identical instruction mix)
uncached CART read  :   268.62 cycles each
```
**Every one of those is the documented real R4300i/N64 figure**: Count at half the CPU
clock, ~1 cycle load-use on a D-cache hit, ~40 cycles for a D-cache miss to RDRAM,
~250-300 cycles for a single uncached PI/cartridge read.

Contrast with mupen64plus (F11), where CP0 Count is exactly 2 per retired instruction
and **8 uncached cart reads cost the same as 8 nops**. The two emulators disagree by
**269x on the single most important cost on this platform**.

CONSEQUENCES — this invalidates the framing of several of my own earlier conclusions:
- F11's "Step 4 cannot be validated" is now WRONG. It is validatable, on ares.
- F14's baseline is not a cycle count. It is an instruction count. Still true, still
  useful, but it is not the number that matters on hardware.
- F13's "cart-resident and RDRAM-resident weights cost the same" is a mupen64plus
  artifact. On ares that difference is ~269x per access.
- **F20's recommended fix (point the engine at the cart via `dfs_rom_addr`) is now
  suspect**: it converts every weight byte read into a ~269-cycle PI transaction.
  It makes the ROM *correct* but potentially catastrophically slow. Measuring it
  rather than recommending it blind.

## F32 — REAL CYCLE-ACCURATE BASELINE on ares, and **F8 CONFIRMED ON-EMULATOR**
`$S/ld/bench_ld.c` — the same engine, built as a stock libdragon ROM with a real DFS
containing `sophia_weights.bin`, so this is the actual shipping weight-load path.
4-token run, RDRAM-resident weights, ctx=32, greedy, deterministic PSE:
```
MEMSIZE 8388608                                  <- ares gives 8 MB (Expansion Pak)
dfs_open 668472
dfs_size 6750220   wbuf_guard_3MB 0              <- ***F8 PROVEN, not inferred***
dfs_rom_addr 2952916880  (0xB001EF10)
rdram_buf 2148152144  bytes 6750224
dma_load_count 60953642                          <- PI DMA of the whole blob
CFG weights=RDRAM
MAGIC 1397047625   LOADED 1
TOK 0 in 69  out 108  count 103,069,327
TOK 1 in 108 out 121  count 103,120,569
TOK 2 in 121 out 97   count 103,184,991
TOK 3 in 97  out 110  count 103,241,316
```
**`wbuf_guard_3MB 0`** is the shipping ROM's own guard expression
(`sz <= 3*1024*1024`) evaluated on-emulator against the real DFS. It is FALSE.
F8 is no longer static analysis — the weight load is measurably skipped.

**Real timings (Count ticks at half the 93.75 MHz CPU clock):**
```
per token   103.1 M counts = 206.2 M cycles = 2.20 SECONDS per token
blob DMA     61.0 M counts = 121.9 M cycles = 1.30 s  (6,750,224 B -> 5.2 MB/s,
                                                       which is the real N64 PI rate)
```
Cross-check against mupen64plus: ~73.4 M instructions/token (F14) vs 206.2 M cycles
here => **2.81 cycles per instruction**. Roughly two thirds of the real cost is stall,
and mupen64plus was blind to all of it.

**Cross-emulator correctness agreement:** ares tokens 0-3 are `108 121 97 110`,
byte-identical to the first four of the mupen64plus 16-token run (F14/F27). Two
independent emulators, two different boot paths (custom IPL3 vs stock libdragon IPL3),
two different weight-load paths (raw PI DMA vs DFS) — same output.

## F33 — Second self-inflicted harness bug, caught by reading the output
I built three ares ROMs in a row with `make bench EXTRA="..."`. make does not track
changes to CFLAGS, and I only `rm`'d the objects *between* the 2nd and 3rd builds, not
before the 1st. So `ares_base16.z64` silently reused the previous 4-token
`bench_ld.o` and ran 4 tokens while claiming 16 — which would have produced an
A/B comparing a 4-token base against a 16-token hoist, i.e. a ~4x "regression" that
was pure build residue.
Caught because the log stopped at `TOK 3 ... BENCH_DONE`. Rebuilt from clean as
`ares_base16b`. Same class of error as F21: the harness silently not testing what it
claimed. Two in one session — flag-driven builds need a forced clean, every time.

## F34 — **A/B RE-MEASURED ON ares: the hoist is nearly TWICE as good as mupen64plus said**
16 tokens, RDRAM-resident weights, ctx=32, greedy, deterministic PSE, stock libdragon ROM.
```
                 TOTAL_COUNT       cycles          wall @93.75MHz   tokens
ares base16    1,657,760,950   3,315,521,900        35.37 s        "lyan Labs.: A wo"
ares hoist16   1,421,831,502   2,843,663,004        30.31 s        "lyan Labs.: A wo"
delta           -235,929,448                        -5.03 s
                    -14.23 %
```
**Output byte-identical, all 16 tokens** — so Hard Rule 2 is now satisfied on BOTH
emulators, with two different boot paths and two different weight-load paths.

mupen64plus measured this same change at **-7.47 %** (F26/F27). ares measures
**-14.23 %**. Both are correct measurements of different things: mupen64plus counts a
`mul.s` as one instruction like any other, while on a real VR4300 single-precision
multiply has multi-cycle latency. **Instruction counting UNDERSTATED the value of
removing floating-point multiplies by ~2x.** The lesson generalises: on this platform
an instruction-count instrument is not merely blind to memory, it also systematically
mis-weights the FPU.

Real cost of the baseline: **2.21 seconds per token** (103.1 M counts = 206.2 M cycles).

## F35 — **F20's recommended fix is 9.4x SLOWER *and* it produces WRONG OUTPUT.** Retracted.
Measured the `dfs_rom_addr()` + KSEG1 cart-pointer approach I recommended in F20,
1 token, everything else identical:
```
weights = RDRAM  : TOK 0 -> out 108,   103,065,411 counts  ( 2.20 s)
weights = CART   : TOK 0 -> out  71,   973,474,122 counts  (20.77 s)   9.44x slower
```
Two independent problems, both invisible on mupen64plus:
1. **9.44x slower** — every weight byte becomes a ~270-cycle PI transaction (F31).
   20.8 seconds per token is not a usable configuration.
2. **The output is WRONG.** `out 71` vs `out 108` on the very first token, from the
   same weights. The 32-bit read of the `SEAI` magic came back correct
   (`MAGIC 1397047625`), so the pointer is right — it is the *sub-word* reads that are
   not. `nano_gpt.c` reads weights as `int8_t` and `hdr->em_scale_x16` as `uint8_t`.
F13 claimed cart and RDRAM paths agreed; that was a mupen64plus artifact, because
mupen64plus models cart reads as plain array indexing with no PI bus semantics.
**F20 is retracted.** Verifying the byte-read hypothesis directly before asserting it.

## F36 — ROOT CAUSE: CPU **byte** reads from cartridge PI space return wrong data
`$S/ld/cartprobe.c` reads the same 16 bytes of the weight blob three ways on ares:
```
byteload   83 69  8  0  8  0  0  1  0  1 254 250 254 250 248 246   <- int8_t* through KSEG1
io_read32  83 69 65 73  8  0  1  8  0  1 128  0 254 250 245  24    <- 32-bit io_read
dma_rdram  83 69 65 73  8  0  1  8  0  1 128  0 254 250 245  24    <- PI DMA (ground truth)
mismatched_of_4096  2024        (49.4 % of bytes wrong)
em_scale_x16  cart 250   dma 0
```
32-bit reads and DMA agree exactly. Byte reads do not, and the corruption is
structured — the returned stream aliases in 2-byte groups, each cart halfword being
delivered twice:
```
byteload[0,1]   = real[0,1]      byteload[2,3]  = real[4,5]
byteload[4,5]   = real[4,5]      byteload[6,7]  = real[8,9]
byteload[8,9]   = real[8,9]      byteload[10,11]= real[12,13] ...
```
i.e. the PI bus serves the CPU at 16-bit minimum granularity and sub-word addressing
does not decode the way a naive pointer dereference assumes. `nano_gpt.c` reads every
weight as `int8_t` and `em_scale_x16` as `uint8_t`, so on the cart path **half the
weights and the embedding scale are garbage** — `em_scale_x16` alone reads 250 instead
of 0. That is why F35 saw `out 71` instead of `out 108`.

mupen64plus models cart reads as plain array indexing, so it showed byte-for-byte
agreement (F13) and gave no hint any of this existed.

**Conclusion: weights MUST reach the CPU via PI DMA into RDRAM. Pointing the engine at
the cart is not a slow-but-correct option; it is simply incorrect.**

## F37 — **F8 FIXED AND VERIFIED IN THE REAL SHIPPING ROM, on ares**
Iterated three times, each failure measured rather than guessed:

1. `memalign(16, 6750224)` from the libdragon heap -> **NULL**. Probe:
   `BOOT rdy=0 ... rom=2952974752 buf=0`. `dfs_rom_addr` was fine; the heap simply
   could not serve 6.75 MB.
2. Switched to a static buffer sized FROM THE MODEL, not a literal:
   ```c
   #define SGAI_BLOB_BYTES (((16u + SGAI_VOCAB*SGAI_N_EMBED
                           + SGAI_N_LAYERS*sizeof(SGAILayer)) + 15u) & ~15u)
   static uint8_t wbuf[SGAI_BLOB_BYTES] __attribute__((aligned(16)));
   ```
   This removes the root cause of F8: the old `3 * 1024 * 1024` literal (whose comment
   still described a "6-layer 192-embed" model) could go stale silently. A model-derived
   size cannot.
   At **SGAI_CTX=64** this gives bss 7,817,612 / total 8,001,704 and **ares refuses to
   run the ROM at all** — no output, not even ares's own "Loaded" line (a known-good ROM
   run back to back as a control printed normally, so this is the ROM, not the harness).
3. At **SGAI_CTX=32**: bss 7,293,068 / total 7,477,160 — **boots and works.**

```
BOOT rdy=1 w=6750220 ctx=32 mem=8388608 rom=2952975168 buf=2147669200 sz=6750224 heap=250000
TOKS 108 121 97 110 32 76 97 98 115 46 58 32 65 32 119 111
```
- `rdy=1` — `G.ai_ready` is set. **The shipping ROM now actually loads its weights.**
- `TOKS` is byte-identical to the bench reference sequence on both emulators, all 16.
- `heap=250000` — with the blob in bss, libdragon's heap has under 500 KB left. That
  is also why step 1 failed, and why ctx=64 does not boot: `display_init` takes its
  320x240x16bpp double buffer (~307 KB) from that same heap.

**Honest consequence: with resident weights, SGAI_CTX must drop from 128 to 32.**
The context window shrinks 4x. That is a product decision, not a free win, and it is
forced: 6,750,224 B of weights + a ctx-128 KV cache is 8.85 MB against an 8.39 MB
console. The only way to keep ctx=128 is to stream weights — i.e. finally wire up
`expert_cache` (F17). This is now a measured constraint, not a projection.

## F38 — The optimization is now VERIFIED IN THE SHIPPING ROM, and enabled by default
Built the real `legend_of_elya.z64` with the scale hoist active and the boot probe on:
```
BOOT rdy=1 w=6750220 ctx=32 mem=8388608 buf=2147669200 sz=6750224 heap=250000
TOKS 108 121 97 110 32 76 97 98 115 46 58 32 65 32 119 111
```
Byte-identical to the un-hoisted shipping ROM (F37) and to both bench baselines.
So the change is now verified byte-identical on:
- mupen64plus, bench harness, 16 tokens (F27)
- ares, bench harness, 16 tokens (F34)
- ares, **the real game ROM**, 16 tokens (this finding)
Given that, `matmul_q8` now takes the hoisted form by default; `-DOPT_NO_HOIST_SCALE`
restores the original. Measured value: **-14.2 % cycles** on the accurate emulator.

## F39 — Limit of the headless ares harness: the render loop cannot be exercised
The shipping ROM (probe compiled out) produces **no output at all** under
`xvfb-run` + ares, not even ares's own "Loaded" line — ares dies before flushing.
CONTROL: the **pristine, unmodified `~/legend-of-elya-n64/legend_of_elya.z64`** behaves
identically (0 bytes). So this is the harness, not my change: ares reports
"Vulkan Enabled: using paraLLEl-RDP" even with `Video/Driver=None`, and there is no GPU
under Xvfb. The probe build survives only because it runs all 16 forward passes inside
`game_init()`, before `main()` ever renders a frame.
(Also note `stdbuf` is useless here — it works by LD_PRELOAD, which does not cross the
flatpak/bwrap boundary, so ares's stdout stays block-buffered.)
**What is therefore verified:** boot, DFS, weight DMA, `sgai_init`, and 16 correct
forward passes in the real ROM. **What is NOT verified:** the interactive game loop,
rendering, and controller-driven generation. Those need a GPU-capable display or real
hardware.

## F40 — Corrections this detour forced on my own earlier conclusions
Recording these together because several of my confident earlier statements were
artifacts of a single unrepresentative tool.
| Earlier claim | Status after ares |
|---|---|
| F5 "libdragon ROMs cannot boot here" | **Wrong as stated** — true only of mupen64plus. ares boots them. |
| F11 "Step 4 cannot be validated on this platform" | **Wrong** — validatable on ares; memory is priced correctly. |
| F13 "cart and RDRAM weight paths agree" | **Wrong** — a mupen64plus artifact; 49 % of cart bytes are wrong (F36). |
| F14 baseline "~73.4 M instructions/token" | Still true, but it is not the cost. Real cost is 206 M cycles = 2.21 s/token. |
| F20 "recommend dfs_rom_addr cart pointer" | **RETRACTED** — 9.4x slower AND produces wrong output. |
| F24 "bench ROM is emulator-only" | Confirmed, and superseded: the ares harness uses stock libdragon ROMs. |
| F26/F27 hoist "-7.47 %" | Understated. Real figure -14.23 % cycles. |
The general lesson: an instrument that is *precise* (mupen64plus was deterministic to
the last count) can still be *inaccurate* about the thing that matters. Precision was
never the problem; representativeness was.
