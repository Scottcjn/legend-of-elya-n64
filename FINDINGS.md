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
