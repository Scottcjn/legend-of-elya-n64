# Which N64 emulator you measure on decides what you conclude

Two emulators were used to measure this project on 2026-08-04. They disagree by
269x on the single most important cost on this platform, and that disagreement
silently reversed several conclusions. This document exists so nobody repeats
the mistake.

## The calibration

Measured with hand-counted assembly loops, reading the CP0 Count register
(register 9) around each sequence.

| operation | ares | mupen64plus 2.6 |
|---|---|---|
| cycles per instruction | 1.00 | 1.00 (reported as 2 counts/instr) |
| cached RDRAM load | 1.00 cyc | ~0 |
| D-cache miss | 40.25 cyc | ~0 |
| **uncached cart read** | **268.6 cyc** | **~0, same as a `nop`** |

Every ares figure matches the documented real R4300i number. mupen64plus prices
memory access at approximately zero: eight uncached cartridge loads cost the
same as eight `nop` instructions, and streaming 6.3 MB per token over the PI bus
moved its total by 0.00003%.

**mupen64plus is precise and reproducible.** Repeated runs return identical
counts to the last digit. It is simply not representative. Precision was never
the problem; representativeness was.

## What that changed

**The real baseline.** 103.1M counts/token = 206.2M cycles = **2.21 seconds per
token**, at 2.81 cycles per instruction. Roughly two thirds of the true cost is
memory stall, all of which mupen64plus was blind to.

**Optimizations are worth about twice what instruction counting says.** The
`matmul_q8` scale hoist measures **-14.2%** on ares against -7.47% on
mupen64plus. Static instruction counting was worse than either: it said the
function got *bigger* (110 -> 114 instructions) and would have rejected the
change outright.

**Narrow weights are slower here unless the kernel is hand written.** Measured
on ares, which models memory:

| weight width | memory traffic | measured speed |
|---|---|---|
| 5-bit | -35% | **+25.2% slower** |
| ternary | -75% | **-57.5% faster** |

A generic bit-cursor unpacker costs more than the traffic it saves. Ternary wins
because its kernel needs neither a cursor nor a multiply. Per-width hand-written
kernels win; generic bit readers lose.

**A recommended fix was retracted as outright wrong.** Reading the weight blob
through a CPU cart pointer (`dfs_rom_addr()`) is not merely slow, it returns
corrupt data: CPU *byte* reads from cart PI space are aliased, delivering each
cart halfword twice. Measured 49.4% of bytes wrong (2024 of 4096);
`em_scale_x16` read 250 instead of 0. mupen64plus models cart reads as plain
array indexing, so its "both paths agree" result was an artifact of the
emulator, not a property of the hardware. **Use PI DMA, never CPU byte reads.**

## Booting

- **mupen64plus 2.6 cannot boot libdragon ROMs here at all.** Its IPL3
  mis-detects RDRAM as 64 MB, the stack lands at ~0x84000000 (unmapped), and the
  ROM wedges before `main()`. It warns and continues rather than correcting.
- **ares boots them** and reports the correct 8,388,608 bytes.
- ares validates the IPL2 checksum, so a hand-rolled IPL3 is rejected
  (`invalid IPL2 checksum: CIC-NUS-6102`). mupen64plus does not check. Build
  stock libdragon ROMs and the problem disappears.

Two practical traps when driving ares:

- **flatpak cannot see `/tmp`.** Its filesystem context is `home;host` but it
  always receives a private `/tmp`, so ROMs must be staged under `$HOME`.
- **ares loses stdout on SIGTERM.** Send SIGINT.

## Still unverified

The interactive game loop. Under headless ares the shipping ROM produces no
output, but so does the pristine unmodified repository ROM, so this is a harness
limitation rather than a regression: ares uses paraLLEl-RDP/Vulkan even with
`Video/Driver=None`, and there is no GPU under Xvfb.

Verified: boot, DFS, weight DMA, `sgai_init`, and 16 correct forward passes
emitting the reference token sequence. Not verified: rendering and
controller-driven generation. Real hardware remains the outstanding check.
