# FINDINGS — making the RSP matmul coexist with rdpq

Journal for the rspq-overlay conversion. Append-only; every entry is a
discrete measured result, committed as it is written.

Baseline carried in from the probe work (not re-derived here):

| arm | CP0 (16 gen) | vs CPU int8 |
|---|---|---|
| CPU int8 | 1,275,062,235 | -- |
| CPU ternary | 548,354,836 | -57.00% |
| RSP int8 | 266,167,042 | **-79.12% (4.79x)** |
| RSP ternary | 298,088,755 | -76.62% (4.28x) |

The blocker: `rsp_matmul_init()` calls `rsp_init()` + `rsp_load(&rsp_mm2)`,
which overwrites the rspq microcode `rdpq_init()` installed. Nothing renders
afterwards, so the 4.79x only exists in a headless probe.

---

## F-O001 — libdragon's rspq DOES support what this needs

Checked the toolchain actually used by the build
(`N64_INST=$HOME/n64-toolchain/mips64-toolchain`), not upstream docs.

Present and complete:

- `mips64-elf/include/rspq.h` — `rspq_init()`, `rspq_overlay_register()`,
  `rspq_overlay_register_static()`, `rspq_overlay_unregister()`,
  `rspq_overlay_get_state()`, `rspq_write()` / `rspq_write_begin` /
  `_arg` / `_end`, `rspq_wait()`, `rspq_syncpoint_*`.
- `mips64-elf/include/rsp_queue.inc` — the assembler side:
  `RSPQ_BeginOverlayHeader` / `RSPQ_DefineCommand` / `RSPQ_EndOverlayHeader`,
  `RSPQ_BeginSavedState` / `RSPQ_EndSavedState` / `RSPQ_EmptySavedState`.
- `mips64-elf/include/rspq_constants.h` — `RSPQ_MAX_OVERLAY_COUNT 8`,
  `RSPQ_OVERLAY_TABLE_SIZE 0x10`, `RSPQ_DMEM_BUFFER_SIZE 0x100`.

So the answer to task 1 is **yes, the installed libdragon supports it**. No
version blocker. The blocker is a resource budget, quantified in F-O002.

### What rspq actually requires of an overlay

1. The overlay `.S` must `#include <rsp_queue.inc>`. That pulls in rspq's
   own resident DMEM state *and* its resident IMEM code; the overlay's own
   `.data` / `.text` are appended after them by `rsp.ld`. Only the appended
   part is what gets swapped in at runtime.
2. `.data` must open with `RSPQ_BeginOverlayHeader` ... `RSPQ_DefineCommand`
   (one per command) ... `RSPQ_EndOverlayHeader`.
3. The overlay must define exactly one saved state
   (`RSPQ_BeginSavedState`/`RSPQ_EndSavedState`, non-zero size, or
   `RSPQ_EmptySavedState`). That region — and *only* that region — is
   DMA'd out and back in when the overlay is swapped. Everything else in
   the overlay's DMEM is scratch that does not survive a swap.
4. A command function is entered with the first 4 words (16 bytes) of the
   command already in `a0`-`a3`, `t7` = command size, and `ra` = the return
   into `RSPQ_Loop`. **It must `jr ra`, not `break`.**
5. `gp` is globally reserved (`rspq_dmem_buf_ptr`) and must be preserved.
   Every other GPR is free.
6. `DMAIn` / `DMAOut` / `DMAInAsync` / `DMAOutAsync` / `DMAExec` are already
   linked in by rspq (`rsp_queue.inc` includes `rsp_dma.inc` itself), with
   the same `t0`/`t1`/`s0`/`s4` convention `rsp_mm2.S` already uses. The
   overlay must therefore *drop* its own `#include <rsp_dma.inc>` to avoid
   duplicate symbols.

## F-O002 — the real constraint is DMEM, and the current kernel does not fit

Measured, not estimated: assembled a minimal do-nothing overlay
(`probe/rsp_budget_probe.S`) and read the symbol table.

```
_ovl_data_start = 0x260      -> DMEM left for overlay data: 0x1000-0x260 = 3488 B
_ovl_text_start = 0x270      -> IMEM left for overlay text: 0x1000-0x270 = 3472 B
```

rspq's resident DMEM state is 608 bytes (vector shift tables, overlay table
and descriptors, pointer stack, the 256-byte command buffer, and the shared
RDPQ mode/scissor/buffer state). Its resident IMEM is 624 bytes.

Current `rsp_mm2.S` footprint, measured with `mips64-elf-size`:

```
.text   576 B    fits easily in 3472
.data  4096 B    needs ALL of DMEM -- overflows the 3488 budget by 608 B
```

The data segment is `params 64 + kmask 16 + pad 16 + xi 2048 + wbuf 1040 +
out 912 = 4096`, i.e. the standalone kernel was written to own the whole
chip. Subtract the overlay header, command table and saved state and the
usable window is ~3464 B, so the deficit is ~632 B.

**This is the entire difficulty of the conversion.** Not the API — the
budget. IMEM is not a problem; DMEM is.

## F-O003 — the conversion, and why the DMEM deficit went away

The deficit in F-O002 is only real if the buffers live in `.data`. They do
not have to.

`n64.mk` builds a ucode by linking with `rsp.ld` and then extracting the
blob with `objcopy -O binary -j .text` and `-j .data`. **`.bss` is not
extracted.** `rsp.ld` still places `.bss` in `ram_data` immediately after
`.data`, so a `.bss` buffer gets a real DMEM address but contributes
nothing to the ucode's data blob.

That matters twice over. rspq's overlay switch does:

```
# Load overlay data (saved state is included)
lhu t0, %lo(RSPQ_OVERLAY_DESCRIPTORS) + 0xE (ovl_index)
jal DMAInAsync
li  s4, %lo(_ovl_data_start)
```

i.e. it DMAs the **entire overlay data segment** on every switch, not just
the saved state. A 3.4 KB `.space` in `.data` would therefore have been
paid for on every single switch. In `.bss` it is paid for never.

Measured footprint of `rsp_mm2_ovl.S`:

```
overlay text   560 B   (1184 total - 624 rspq resident)   budget 3472
overlay data    48 B   (656 total  - 608 rspq resident)   budget 3488
mm_scratch @ 0x2D0 -> 3376 B of DMEM scratch, transferred never
```

So the overlay moves 48 bytes of data and 560 bytes of code per switch.

### What actually changed in the kernel

The arithmetic is untouched — both inner loops are byte-for-byte the ones
from `rsp_mm2.S`. The plumbing changes were:

- `break` -> `jr sp`, where `sp` holds the `ra` rspq entered us with (the
  DMA helpers use `jal`, and the RSP has no stack, so `sp` is just a spare
  register here).
- `gp` is reserved by rspq, so the superblock counter moved to `t8`.
- Dropped `#include <rsp_dma.inc>`; rspq already includes it.
- `$v30`/`$v31` are rspq's `vshift`/`vshift8`, but rspq re-runs
  `vxor vzero` + `lqv vshift` + `lqv vshift8` **before every command
  dispatch**, so clobbering them is safe. Verified in the dispatch tail of
  `rsp_queue.inc`.
- The CPU can no longer poke DMEM with `rsp_load_data()` (the RSP is busy
  running the queue), so the overlay DMAs its own parameter block and
  activation vector in, given an RDRAM pointer passed in the command.

### Buffer placement became dynamic, and that is a small win

Since the buffers no longer sit at fixed offsets, the CPU packs them into
the scratch window for the shape in hand (16-aligned, because LQV/SQV only
move full quadwords from 16-aligned addresses — the old fixed offsets got
that for free). Rows buffered before an output DMA, old vs new:

| shape (ternary) | old fixed layout | overlay dynamic layout |
|---|---|---|
| 256->256 q/k/v/o | 14 | 43 |
| 256->1024 ff1 | 14 | 43 |
| 1024->256 ff2 | 3 | 4 |

So the overlay does *fewer* output DMAs than the standalone kernel did,
which partly pays for the dispatch it adds.

## F-O004 — it works: rendering-capable build, all 912 matmuls on the RSP

`make base-rsp-ovl SGAI_BITS=2 EXTRA=-DBOOT_PROBE`, headless ares:

```
BOOT rdy=1 bits=2 bufbytes=2031632 ctx=128 failsz=0 cap=0
GAME TOKS 110 32 76 97 98 115 32 98 117 105 108 116 32 109 101 46
GAME TEXT n Labs built me.
GAME CP0 gen16=299118148
RSPPATH rsp=912 cpu=0
RSPDMA wdma=350208 wKiB=31920 outKiB=29184
BOOT_DONE
```

`rsp=912 cpu=0` — every matmul went to the RSP, nothing fell back to the
CPU because a shape would not fit the reduced DMEM window.

Against the standalone RSP ternary figure of 298,088,755 recorded before
the conversion, this is **+1,029,393 counts, +0.35 %**. The overlay
dispatch is close to free at this granularity (912 dispatches across 16
tokens, ~1,129 counts of overhead each). Same-tree A/B and token
comparison follow in F-O005.

One emulator warning appears once, at the first overlay switch:

```
[unusual] RSP DMA writing to RDRAM address 0x23a18 which is cached
          RSP DMA started at RSP PC: 0x080
```

RSP PC 0x080 is inside rspq's own resident code (0x000-0x26F), not the
overlay (0x270+), and the address is not one of ours — it is rspq saving
the outgoing overlay's state. This is libdragon's own bookkeeping, fires
once, and did not perturb any result below. Noting it rather than
explaining it away.

## F-O005 — token-exactness, 16 tokens, ternary, same blob

The CPU kernel is the oracle-verified reference (docs/N64_RSP_FINDINGS.md
F-R023). Both ROMs built from the same tree, same
`filesystem/sophia_weights.bin` (sha256 8fe00867...), prompt "Elya", 16
generated tokens:

```
CPU ternary  GAME TOKS 110 32 76 97 98 115 32 98 117 105 108 116 32 109 101 46
             GAME TEXT n Labs built me.
             GAME CP0 gen16=547832290

RSP overlay  GAME TOKS 110 32 76 97 98 115 32 98 117 105 108 116 32 109 101 46
             GAME TEXT n Labs built me.
             GAME CP0 gen16=299118148
             RSPPATH rsp=912 cpu=0
```

**16/16 tokens exact**, byte for byte, against the CPU reference.

### A caution that changed the experiment design

The pre-conversion RSP ternary figure (298,088,755) was measured on a
*different* ternary blob — the one shipped before commit 2a74607
retrained the model. F-R023 already recorded that two different ternary
blobs of the same shape time 0.31 % apart (299,026,720 vs 298,088,755).

That is the same order as the overlay overhead I am trying to measure, so
comparing my overlay number against the old recorded number would be
measuring the blob, not the overlay. Every speedup claim below is
therefore a **same-blob, same-tree A/B**: standalone kernel vs overlay
kernel, identical ROM otherwise.

## F-O006 — rendering: the overlay renders, the standalone crashes by name

Headless ares has no GPU, so this had to be run on a real display
(`DISPLAY=:0`, AMD 780M, direct rendering). Both ROMs are the playable
builds — no probe, no `while(1){}` — booted and screenshotted after ~45 s.

**Overlay build** (`make base-rsp-ovl SGAI_BITS=2`) —
`screenshots/rspq_overlay_renders.png`:

```
LEGEND OF ELYA
Nintendo 64 Homebrew
Elyan Labs
World's First N64 LLM
[Sophia AI: LOADED]
Press START to enter the dungeon...
                                            36 VPS
```

`[Sophia AI: LOADED]` is the game reporting that `sgai_init()` completed,
which is the call that runs `rsp_matmul_init()` -> `rspq_overlay_register()`
and permutes every weight tensor. So the matmul is loaded **and** the RDP
is drawing, at 36 VPS.

**Standalone build** (`make base-rsp SGAI_BITS=2`), same game, same frame —
`screenshots/standalone_rsp_crash.png`:

```
RSP CRASH | rsp_mm2 | rspq_next_buffer (src/rspq/rspq.c:977)
Crash symptom: wait loop timed out (200 ms)
PC:054 | STATUS:5000 [sig5 sig7] | DP_STATUS: e8 [gclk pipe busy ready]
```

libdragon names the offending microcode in its own crash header: `rsp_mm2`.
This is the reported bug, reproduced exactly and mechanically explained —
`rsp_load(&rsp_mm2)` overwrote the rspq microcode `rdpq_init()` installed,
so the first time rdpq needed the queue engine the RSP was running the
matmul instead, and rspq's 200 ms wait loop timed out.

That is as clean an A/B as this question admits: identical game, identical
weights, one line of dispatch difference, crash vs 36 VPS.

## F-O007 — overlay dispatch overhead: +0.06 %, same blob, same tree

Ternary, prompt "Elya", 16 generated tokens, both ROMs from this tree with
the same `filesystem/sophia_weights.bin`:

| arm | CP0 gen16 | vs CPU ternary |
|---|---|---|
| CPU ternary | 547,832,290 | -- |
| RSP standalone (`rsp_mm2.S`) | 298,935,090 | 1.833x |
| **RSP rspq overlay** | **299,118,148** | **1.831x** |

Overlay cost against the standalone: **+183,058 counts on 912 dispatches
across 16 tokens = +0.0612 %**, i.e. ~201 CP0 counts per dispatch.

All three arms produce the identical token stream (F-O005).

The overhead is small enough to be *below* the blob-to-blob variance
(0.31 %) that would have contaminated a naive comparison against the old
recorded figure — which is exactly why the A/B had to be same-blob. Had I
compared against the pre-retrain 298,088,755 I would have reported +0.35 %,
which would have been mostly blob, not overlay.

Two things make the dispatch this cheap:
- the overlay's data segment is 48 bytes (`.bss`, F-O003), so a switch
  moves 48 B of data and 560 B of code, not 3.4 KB;
- the dynamic buffer packing buys back output DMAs (43 rows per flush
  instead of 14 on the 256-wide shapes), partly cancelling the dispatch it
  adds.

## F-O008 — the full A/B, and the overlay turns out to be slightly FASTER

Every row below is the same tree, same weight blob, same prompt, headless
ares, `GAME CP0 gen16`. Two overlay variants are shown because the sync
primitive changed mid-session (see F-O009).

### Ternary, 16 tokens, prompt "Elya"

| arm | CP0 | vs CPU | vs standalone |
|---|---|---|---|
| CPU ternary | 547,832,290 | -- | -- |
| RSP standalone `rsp_mm2.S` | 298,935,090 | 1.833x | -- |
| RSP overlay, `rspq_wait` | 299,118,148 | 1.831x | +0.061 % |
| **RSP overlay, syncpoint (HEAD)** | **298,597,075** | **1.835x** | **-0.113 %** |

### Ternary, 48 tokens, prompt "Who are you?" (`PROBE_LONG`)

| arm | CP0 | vs CPU | vs standalone |
|---|---|---|---|
| CPU ternary | 1,812,297,430 | -- | -- |
| RSP standalone | 1,032,722,730 | 1.755x | -- |
| **RSP overlay (HEAD)** | **1,031,385,017** | **1.757x** | **-0.130 %** |

`RSPPATH rsp=2832 cpu=0` on both RSP arms — 2,832 matmuls, none falling
back to the CPU.

### The result

**The overlay is not slower. It is 0.11-0.13 % faster than the standalone
kernel it replaces**, consistently at both generation lengths.

That is not a rounding artifact of one run: it reproduces at 16 and 48
tokens, in the same direction and nearly the same magnitude, on a
deterministic emulator. The mechanism is F-O003's dynamic buffer packing —
the standalone kernel's fixed 912-byte output window buffered 14 rows
before each output DMA on the 256-wide shapes; the overlay packs the
window per shape and buffers 43. The output DMAs saved slightly exceed the
dispatch added.

So the honest answer to "does the speedup survive the overlay" is: it
survives, and the overlay dispatch is cheaper than the layout improvement
that came with it.

## F-O009 — rspq_wait vs syncpoint, measured

The first working overlay synchronised with `rspq_wait()`. libdragon's own
header says that is the wrong primitive: it "exists mostly for debugging
purposes", also blocks until the RDP has finished drawing, and
`rspq_syncpoint_new`/`rspq_syncpoint_wait` is what to use for RSP->CPU
data handoff.

Measured, ternary 16 tokens, identical otherwise:

```
rspq_wait()          299,118,148
rspq_syncpoint_wait  298,597,075     -521,073   -0.174%
```

-0.17 % is the *floor* on what this change is worth, because the headless
probe issues no RDP commands at all — there is nothing for `rspq_wait()`'s
RDP sync to wait on. In the real game loop, where a frame is being drawn,
`rspq_wait()` would additionally serialise all 912 matmuls per token
against the RDP. That cost does not appear in any probe number here and is
the reason for the change, rather than the 0.17 %.

## F-O010 — token-exactness after conversion: 80/80

All comparisons are exact token-ID streams, diffed, not eyeballed.

| check | tokens | result |
|---|---|---|
| ternary 16, overlay vs CPU oracle | 16 | identical |
| ternary 48, overlay vs CPU oracle | 48 | identical |
| ternary 48, standalone vs CPU oracle | 48 | identical |
| int8 16, overlay vs standalone | 16 | identical |

**80/80 tokens exact**, across both weight formats and two prompt lengths.
The 48-token stream is `: Sophia Elya of Elyan Labs.s.: Scott's workshop`,
byte-identical from the CPU reference, the standalone RSP kernel and the
overlay.

## F-O011 — the int8 arm: exact, but +0.15 %, and its blob is stale

int8 (`SGAI_BITS=8`, blob `weights/sophia_weights_large_v5fmt.bin`),
16 tokens:

| arm | CP0 | vs standalone |
|---|---|---|
| RSP standalone int8 | 267,076,103 | -- |
| RSP overlay int8 | 267,472,813 | **+0.149 %** |

Unlike ternary, int8 gets *slower* under the overlay, and the reason is
predictable from the DMEM budget. int8's ff2 row is 1024 bytes rather than
256, so after activations (2048 B) and the weight row (1040 B) only 288 B
of the 3376 B scratch window is left — one output row per flush, where the
standalone's fixed 912-byte window managed three. The other shapes improve
(40 rows vs 14), and the net is +0.15 %.

**Both int8 arms emit `t/77GGGGGGGGGGGG`, which is degenerate output.**
That is not caused by the overlay: the standalone control on the same tree
emits the identical degenerate stream, and the recorded pre-conversion
int8 run emitted `fter you name it`. Between then and now the tree gained
the sampler changes in 0480a95/5007948 (REP_FIX, FIRSTCHAR_FIX, argmax
fallback) while this int8 blob is the pre-retrain `large_v5fmt` one. So
the int8 blob is stale with respect to the current sampler. Flagging it
as a real pre-existing issue rather than folding it into my result; what
this session verifies about int8 is only that the overlay reproduces the
standalone kernel bit-exactly.

## F-O012 — what the playable build actually does

All captured from `make base-rsp-ovl SGAI_BITS=2` (no probe flags, current
HEAD) on `DISPLAY=:0`, driven with synthetic input.

1. `screenshots/rspq_overlay_title.png` — title screen, `[Sophia AI:
   LOADED]`, **48 VPS**.
2. `screenshots/rspq_overlay_renders_after_generation.png` — the
   `PROBE_THEN_RENDER` build: 16 tokens generated inside `game_init()`
   (912 matmuls each, so ~14,592 overlay switches with rdpq's state
   swapped out and back each time), *then* the render loop starts.
   Renders at **49 VPS**. This is the check the title screen cannot make:
   rdpq state survives thousands of overlay switches, not just
   registration.
3. `screenshots/rspq_overlay_dungeon.png` — Crystal Dungeon: tiles, hearts,
   MP bar, torch, chest, a mummy, the player and an NPC, `[A]Talk
   [B]Keyboard (auto-attack)`.
4. `screenshots/rspq_overlay_npc_answer.png` — **the point of the whole
   exercise.** Arcane Library, Aldric the Keeper, prompt "What is your
   name?" selected from the in-game menu, and the model's answer rendered
   in the dialog box:

```
Aldric the Keeper
Sophia Elya is my name
[A] Next   [B] Close
```

   The `RSP` label above the dialog box is drawn under `#ifdef
   USE_RSP_MATMUL` (legend_of_elya.c:1373), so its presence on screen is
   itself confirmation that this is the RSP build and the RSP path is
   compiled in.

So: a player can boot the ROM, walk into the library, ask the NPC a
question, and get an answer generated on the RSP while the RDP keeps
drawing the room. That is what did not exist before this change.

### Driving ares, for whoever repeats this

- Headless ares has no GPU. Rendering can only be checked on `DISPLAY=:0`.
- ares's window is named after the ROM, not "ares"; `xdotool search --name
  ares` finds two 10x10 decoys. Use `wmctrl -l | grep <romname>`.
- `import -window <id>` works; capturing by searching for "ares" does not.
- **Tapped keys are dropped.** `xdotool key` is too short for ares to
  latch; `keydown; sleep 0.35; keyup` works reliably.
- Comparing screenshots to detect state changes must crop out the status
  bar — the VPS counter changes every frame and makes every hash differ.
- In this ares config START had **no keyboard binding at all**, so the
  free-text "Ask Sophia" screen cannot be submitted from a keyboard. The
  canned-prompt path (`[A]Talk` near an NPC) needs only A and the D-pad.
- ares key indices in `settings.bml` are `0x1/0/<index>` with letters
  alphabetical and contiguous: **a=35 ... z=60**. Derived from the existing
  bindings (Pad.Left=35='a', Pad.Right=38='d', Pad.Down=53='s',
  Pad.Up=57='w', A=58='x', X=60='z') and confirmed by driving the game.
  A temporary Start binding was added for one run and the original
  `settings.bml` restored afterwards (verified by diff).
