# RSP matmul offload — FINDINGS

Journal. Appended after every discrete result.

## F-R001: what the RSP path is TODAY (static reading, before any build)
Files: `rsp_matmul.S` (170 lines), `matmul_rsp_drv.c` (179), `rsp_matmul.h` (56).
Wiring: `Makefile: base-rsp` builds `nano_gpt.c -DUSE_RSP_MATMUL`, which replaces
`matmul_q8` with a call to `rsp_matmul_q8`. `matmul_pk` routes to `matmul_q8`
ONLY when `bits == 8`, so **a ternary build never reaches the RSP at all** —
confirms n64rom F-T013's honest note from the other side.

Defects found by reading, before running anything:
1. **The microcode never halts.** After `matmul_done` it does `sw zero,0(zero)`
   and `j poll_cmd` — an infinite spin. libdragon's `rsp_run()` is
   `rsp_run_async()` + wait-for-HALTED/BROKE. Nothing ever sets those. This
   should hang the CPU on the first dispatch.
2. **`vmulf` + `vadd` is the wrong accumulator.** `vmulf` is a signed FRACTIONAL
   multiply (acc = 2*a*b + 0x8000, result = acc[31:16]); `vadd` then accumulates
   with 16-bit SATURATION. A 128-term dot product accumulated in saturating
   int16 is not the dot product. The correct primitive is `vmudh`/`vmadh` +
   `vsar`, which keeps the 48-bit accumulator.
3. **The horizontal reduce is not a reduce.** `vaddc $v28,$v30,$v30.e1` — RSP
   element selectors encode 0=none, 2..3=quarter, 4..7=half, 8..15=broadcast
   lane 0..7. `e1` is a *no-op* selector, not "lane 1". The three vaddc's
   therefore do not sum the 8 lanes.
4. **The driver averages the block scales.** `row_scale = mean(f16(scales[..]))`
   over all in_dim/32 blocks. Even with perfect microcode the answer would be a
   different function from the oracle, which applies each 32-block's own scale.
5. `rsp_load(&rsp_matmul)` is called inside every `rsp_matmul_q8()` call
   (i.e. per tensor per token), re-DMAing IMEM+DMEM every time.

So: not "an RSP kernel with a bug", but an unverified sketch. Next step is to
prove it empirically rather than argue from the listing.

## F-R002: the CPU is NOT soft-float — the premise in nano_gpt.c's header is wrong
`n64.mk` builds with `-march=vr4300 -mtune=vr4300 -ffast-math -ftrapping-math
-fno-associative-math` and no `-msoft-float`. The VR4300 FPU is used. This
matters for the whole exercise: the CPU baseline is hardware single-precision
FP (mul.s/add.s ~ a few cycles), not a 30-cycle software routine, so the RSP has
far less headroom to beat than the file comment implies.
## F-R003: **MEASURED — the RSP does ZERO work in `base-rsp` today. It is dead code.**
`rsp_matmul_q8()` starts with
```c
if (!rsp_available || in_dim > 128 || out_dim > 512) { /* CPU fallback */ }
```
and this model is **8 layers x 256 embed**, so every matmul has `in_dim` 256 or
1024. `in_dim > 128` is true on *every* call, always. Instrumented the driver
with two counters and printed them from `BOOT_PROBE`:
```
ROM: legend_of_elya_rsp.z64, SEQ8 int8 blob, ctx=128, prompt "Elya", 16 gen
BOOT rdy=1 bits=8 bufbytes=6750224 ctx=128
GAME TOKS 102 116 101 114 32 121 111 117 32 110 97 109 101 32 105 116
GAME TEXT fter you name it
GAME CP0 gen16=1587581481
RSPPATH rsp=0 cpu=912          <-- 912 CPU calls, 0 RSP dispatches
```
912 = 8 layers x 6 tensors x 19 forward passes. **Not one RSP dispatch.**
That is why the tokens are right: they are the CPU's tokens.

The DMEM budget makes the guard non-negotiable as written, too: the microcode
stages weights as int16 at 0x200 with 2048 bytes of room, i.e. 8 rows x 128
columns. This model's rows are 256 and 1024 wide. So even removing the `>128`
test would overflow DMEM — the kernel has no tiling at all.

## F-R004: `base-rsp` is currently 16.6 % SLOWER than `base`, for a second reason
`matmul_rsp_drv.c`'s CPU fallback is a *second copy* of the Q8 matmul that never
got the `OPT_HOIST_SCALE` change (it multiplies by `scale` inside the 32-wide
inner loop). Same ROM, same blob, same prompt, ctx=128:
```
make base      (CPU, hoisted)      1,359,838,513 CP0   (n64rom F-T019, reproduced)
make base-rsp  (CPU fallback)      1,587,581,481 CP0   +16.7 %
```
1.167x is exactly the reciprocal of the measured -14.09 % hoist win. So shipping
`base-rsp` today costs 16.7 % and buys nothing.
## F-R005: baseline reproduced in THIS tree
`make base`, SEQ8 int8, ctx=128, prompt "Elya", 16 gen:
`GAME CP0 gen16=1359838513`, tokens "fter you name it" — bit-identical to
n64rom F-T019. So this working copy is a faithful base to measure against.

## F-R006: the ternary blob contains ZERO '10' codes
Histogram over all 6,291,456 ternary codes of `blobs/v7_ternary_seq2.bin`:
```
00: 2,025,338   01: 2,129,943   10: 0   11: 2,136,175
```
This matters for the RSP kernel: a 2-bit field sign-extended arithmetically
decodes 00->0, 01->+1, 10->-2, 11->-1, whereas `matmul_t2` decodes 10 as -1.
With no '10' present, arithmetic sign extension is EXACTLY equivalent, so the
RSP can extract a code with a shift instead of a compare/select. (Recorded as a
property of this blob, not a guarantee of the format.)

## F-R007: DESIGN — why the obvious RSP kernel does not work, and what does
Two hardware facts drove the design:
1. **`VMUD*` and `VMULF` overwrite the accumulator.** Only `VMAC*`/`VMAD*`
   accumulate. So the natural "shift the packed weight into place with a
   multiply, then `vmadh`" loop destroys the running dot product every step.
   The unpack therefore has to be done with ops that touch only ACC_LO —
   `vand`, `vaddc` — while the dot product lives in ACC_HI/ACC_MID via `vmadh`.
   (This is an assumption about RSP accumulator slices and is VERIFIED on the
   emulator in F-R009 rather than taken on faith.)
2. **A horizontal 8-lane reduce is expensive and 16-bit-saturating.** Avoided
   entirely by putting *eight quantization blocks in the eight lanes* instead
   of eight consecutive weights. Lane b then accumulates block b's own sum, so
   the per-32-block float16 scale can be applied per-lane by the CPU with no
   reduce at all.
That layout needs the weight bytes of each row permuted (an 8xG byte transpose
per 256-weight superblock, G=32 for int8 and G=8 for ternary). Done once, in
place, at load time.
## F-R008: **new kernel `rsp_mm2.S` — INTEGER-EXACT on ares, 6/6 shapes**
Standalone harness `rspt/rspt.c` (no LLM, no floats): random weights, random
int16 activations, C reference block sums, RSP block sums, exact int32 compare.
```
RSPT_START
int8 in=256  out=16     int8 PASS blocks=128 ref[0]=258183168
int8 in=256  out=33     int8 PASS blocks=264 ref[0]=-374217984
int8 in=1024 out=8      int8 PASS blocks=256 ref[0]=-458197504
tern in=256  out=16     tern PASS blocks=128 ref[0]=-28852224
tern in=256  out=33     tern PASS blocks=264 ref[0]=-302284800
tern in=1024 out=8      tern PASS blocks=256 ref[0]=-84705280
RSPT_DONE
```
**1,296 block sums, zero mismatches, both weight formats.** This is an exact
integer identity, not a tolerance — one wrong lane, one clobbered accumulator
slice or one bad 2-bit unpack would show.

What it establishes empirically, rather than by argument:
* `LPV` yields `(int8)b << 8` with the sign right.
* `VMADH` accumulates the full 32-bit product into ACC_HI:ACC_MID.
* **`VAND` and `VADDC` do NOT disturb ACC_HI/ACC_MID.** This was the load-bearing
  assumption of the whole design (F-R007); it is now measured, not assumed.
* `VSAR ACC_HI / ACC_MD` reads back the 32-bit sum losslessly.
* The RSP-side DMA in/out, the 8xG weight permutation, the activation staging,
  multi-superblock rows (in_dim 1024) and a row count that is not a multiple of
  the flush batch (out_dim 33) are all right.

One trap found on the way, worth recording: `params` was a stack array and ares
printed
`RSP reading from DMEM address 0x0 which contains a value which is not cache
coherent ... The relative CPU cacheline was dirty (missing cache writeback?)`
and the RSP then hung on garbage parameters. ares diagnoses this class of bug
by name — another reason mupen64plus is not an acceptable substitute here.
## F-R009: **int8 ON THE RSP — 16/16 oracle match, and 3.92x faster than the CPU**
`make base-rsp SGAI_BITS=8`, SEQ8 blob, ctx=128, prompt "Elya", 16 generated,
`SGAI_PSE=0 BENCH_DET_PSE BOOT_PROBE`, under ares:
```
BOOT rdy=1 bits=8 bufbytes=6750224 ctx=128
GAME TOKS 102 116 101 114 32 121 111 117 32 110 97 109 101 32 105 116
GAME TEXT fter you name it
GAME CP0 gen16=347216879
RSPPATH rsp=912 cpu=0        <-- all 912 matmuls on the RSP, zero fallbacks
BOOT_DONE
```
```
                                    tokens              CP0 (16 gen passes)
CPU int8   (make base)              fter you name it     1,359,838,513
RSP int8   (make base-rsp, new)     fter you name it       347,216,879   -74.47 %
ORACLE     (numpy, eval_qat)        fter you name it
```
**16/16 exact against the oracle, and 3.92x faster.** The activation vector is
quantized to int16 (11 bits + sign for int8) before it goes to the RSP, so this
is NOT the same arithmetic as the CPU path — and it still lands on the same
16 argmax tokens.

Note the old driver's 912 calls all went to the CPU (F-R003); the same 912 now
all go to the RSP.

Outstanding: ares prints 257 `RSP DMA writing to RDRAM address ... which is
cached (missing cache invalidation?)` warnings for the block-sum buffer. The
read is still correct here because the CPU only ever reads that buffer (no
dirty lines), but the lines should be invalidated BEFORE the DMA, not only
after. Fixing next.
## F-R010: **ternary ON THE RSP — 16/16 oracle match**
`make base-rsp SGAI_BITS=2`, `blobs/v7_ternary_seq2.bin`, ctx=128, "Elya", 16 gen:
```
BOOT rdy=1 bits=2 bufbytes=2031632 ctx=128
GAME TOKS 46 32 72 111 119 32 99 97 110 32 73 32 104 101 108 112
GAME TEXT . How can I help
GAME CP0 gen16=382019555
RSPPATH rsp=912 cpu=0
BOOT_DONE
```
`ORACLE (numpy) ". How can I help"` — **16/16 exact.** A 1.58-bit-per-weight
transformer whose matmul runs on the N64's RSP vector unit, agreeing token for
token with the numpy reference.

Link: text 120,952 + data 34,492 + bss 2,709,428 = 2,864,872 B.

**And the first surprise: RSP ternary (382,019,555) is SLOWER than RSP int8
(347,216,879), by 10 %.** On the CPU ternary was 53 % FASTER than int8. Full
four-way comparison in F-R012.
## F-R011: cache-invalidate-before-DMA fix; int8 re-measured clean
Added a `data_cache_hit_writeback_invalidate()` of the block-sum buffer BEFORE
`rsp_run()` as well as after. ares "unusual" warning count:
```
before fix   rsp2_int8.log   257 warnings
after fix    rsp2_int8b.log    0 warnings
             rsp2_tern.log     0 warnings
```
Re-measured int8: `GAME CP0 gen16=350105809` (was 347,216,879 with the extra
lines left resident, i.e. the extra invalidate costs 0.8 %). Tokens unchanged,
`fter you name it`, 16/16. All later numbers use the clean build.

**On `RSPPATH rsp=912 cpu=0`:** the two counters are incremented on the two
mutually exclusive paths inside `rsp_matmul_pk()` — the RSP dispatch and the
CPU fallback. 912 = 8 layers x 6 tensors x 19 forward passes, i.e. EVERY matmul
in the run. `cpu=0` means the fallback was never taken. (The tied-embedding
logit projection, 256x256, is a separate hand-written loop in
`project_to_logits()` and stays on the CPU in every arm — it is ~1 % of the
matmul work and is identical in all four arms, so it does not distort the
comparison.)
## F-R012: CPU ternary baseline reproduced in this tree
`make base SGAI_BITS=2`, v7_ternary_seq2: `GAME CP0 gen16=633044443`,
tokens ". How can I help" — identical to n64rom F-T019. All four arms are now
measured in one tree with one harness.
## F-R013: **WHERE THE TIME GOES — the CPU epilogue, not the RSP (int8)**
Instrumented build (`-DRSP_PHASE_TIMING`), CP0 counts accumulated inside
`rsp_matmul_pk()` over all 19 forward passes:
```
int8   stage (quantize + lane-order staging)      9,453,274    2.4 %
       disp  (cache ops + DMEM upload + rsp_run) 108,921,336   27.1 %
       epi   (invalidate + per-block float scale) 283,102,609  70.5 %
                                                 -----------
                                                 401,477,219
       wdma = 350,208 weight-row DMAs   wIn = 119,472 KiB   outBack = 29,184 KiB
```
(19 passes; `GAME CP0 gen16` covers 15, hence 401M here vs 350M there. Tokens
unchanged: `fter you name it`.)

`disp` is the only phase in which the RSP is running; it is **27 %** of the
matmul cost. The RSP finished the 6.29 M multiply-accumulates and then the CPU
spent 2.6x longer turning 8-lane int32 block sums back into one float.
## F-R014: **WHY RSP TERNARY IS SLOWER THAN RSP INT8 — measured, both hypotheses tested**
Same instrumented build, ternary arm:
```
             stage        disp (RSP busy)   epi          weight bytes DMA'd in
int8       9,453,274      108,921,336      283,102,609   119,472 KiB
ternary    9,474,307      149,299,835      283,188,456    31,920 KiB
             +0.2 %         +37.1 %          +0.03 %       -73.3 % (3.74x less)
```
Both candidate explanations are now answered by measurement, not argument:

**1. "The density advantage was never cashed in."  FALSE.** Ternary moves
**3.74x fewer weight bytes** (31,920 KiB vs 119,472 KiB over 19 passes; 1.64 MiB
vs 6.14 MiB per forward pass). The tiling *is* exploiting 4 weights per byte —
the same 350,208 row DMAs carry a quarter of the payload.

**2. "The unpack cost dominates."  TRUE, and it is the whole story.** `disp` is
the only window in which the RSP is executing, and it grows 37 % while the data
moved falls by 73 %. Static issue slots per weight:
```
int8     (lpv + lqv + vmadh) / 8               = 0.375 ops/weight
ternary  (lpv + 4*(lqv + vand + 2*vaddc + vmadh)) / 32 = 0.656 ops/weight   +75 %
```
The three extra vector ops per 8 weights (`vand` + two `vaddc` to walk the 2-bit
field) are exactly what int8 does not pay, because `LPV` already lands an int8
in a lane sign-extended and ready to multiply.

**Why the CPU's ternary win does not transfer.** On the CPU, ternary wins
(-53 %) by DELETING a multiply: `acc += w*x` becomes `acc += x` / `acc -= x` /
skip, and on the VR4300 a float multiply in a dependent chain is expensive. The
RSP has no multiply to delete — `vmadh` does eight 16x16 multiply-accumulates in
one instruction, and it costs the same whether the weights are int8 or ternary.
So on the RSP ternary can only ADD work (the unpack) and can only SAVE bandwidth
— and at this shape bandwidth is not the constraint.

**Is the RSP DMA-bound?  No, and here is the number.** The int8 arm moves
6.14 MiB of weights per forward pass and spends 108.9M/19 = 5.73M CP0 counts per
pass with the RSP busy. 5.73M counts at 46.875 MHz is 122 ms; 6.14 MiB in 122 ms
is 50 MiB/s, roughly a fifth of what RDRAM sustains. The RSP is issue-bound, not
transfer-bound — which is precisely why cutting the transfer by 3.74x buys
nothing and adding 75 % more instructions costs 37 %.

## F-R015: **THE FOUR-WAY COMPARISON**
Identical source tree, identical harness (`legend_of_elya.z64` /
`legend_of_elya_rsp.z64` with `BOOT_PROBE`), identical prompt "Elya" (4 prompt
tokens), 16 generated tokens, `SGAI_CTX=128`, `SGAI_KV_INT8`,
`OPT_HOIST_SCALE`, `SGAI_PSE=0`, `BENCH_DET_PSE`, ares 147 headless.
The ONLY differences between arms are the blob, `SGAI_BITS`, and whether the
matmul goes to the RSP.
```
arm            weights   CP0 (16 gen)     vs CPU int8   vs CPU same-width   tokens vs oracle
CPU int8       SEQ8     1,359,838,513        --              --             16/16 "fter you name it"
CPU ternary    SEQ2       633,044,443     -53.45 %           --             16/16 ". How can I help"
RSP int8       SEQ8       350,105,809     -74.25 %  (3.88x)  -74.25 % 3.88x 16/16 "fter you name it"
RSP ternary    SEQ2       382,019,555     -71.91 %  (3.56x)  -39.65 % 1.66x 16/16 ". How can I help"
```
64/64 tokens exact against the numpy oracle across the four arms. Every arm was
checked against ITS OWN blob's oracle (the two blobs are different models, so
their texts differ; that is the model, not the kernel).

**The negative result, stated plainly: on the RSP, ternary is 9.1 % SLOWER than
int8 (382.0M vs 350.1M), while on the CPU ternary is 53.5 % FASTER than int8.
Moving the matmul to the RSP inverts the sign of the ternary-vs-int8 result.**
## F-R016: the epilogue's real cost was a div.s in the float16 decode
`epi` is 283M of 401M CP0 counts (F-R013) and is IDENTICAL for int8 and ternary
(283,102,609 vs 283,188,456) — because it does the same work in both: one
float16 scale decode and one multiply-add per 32-weight block, 196,608 blocks
per forward pass. `f16_to_float`'s hot path was
```c
float mantissa = 1.0f + frac / 1024.0f;
val = mantissa / (float)(1u << (unsigned)(-e));   /* a div.s, ~29 cycles */
```
and weight scales are almost always < 1, so the divide branch is the one taken.
Replaced with the pure bit move a float16->float32 widening actually is:
```c
u.i = sign | ((exp + 112u) << 23) | (frac << 13);
```
Proved **bit-identical over all 65,536 float16 inputs** on the host before
touching the ROM (`/tmp/f16chk.c`: `mismatches=0 / 65536`). Applied to BOTH
`nano_gpt.c` (so the CPU arms get it too) and `matmul_rsp2.c`, so the four-way
comparison stays apples-to-apples. Re-measured in F-R017.
## F-R017: **FOUR-WAY, re-measured with the bit-identical f16 fix in all arms**
Same harness, same prompt, same ctx, `f16_to_float` now the bit-move form in
every arm (proved identical over all 65,536 inputs, F-R016):
```
arm            weights   CP0 (16 gen)     vs CPU int8        vs CPU same width   tokens
CPU int8       SEQ8     1,275,062,235        --                   --             16/16
CPU ternary    SEQ2       548,354,836     -57.00 %                --             16/16
RSP int8       SEQ8       266,167,042     -79.12 %  (4.79x)   -79.12 %  4.79x    16/16
RSP ternary    SEQ2       298,088,755     -76.62 %  (4.28x)   -45.64 %  1.84x    16/16
```
Effect of the f16 fix alone, per arm:
```
              before          after         delta
CPU int8    1,359,838,513  1,275,062,235   -6.23 %
CPU ternary   633,044,443    548,354,836  -13.38 %
RSP int8      350,105,809    266,167,042  -23.97 %
RSP ternary   382,019,555    298,088,755  -21.97 %
```
Tokens byte-identical in all eight runs; the change is provably bit-exact.

Phase split after the fix (19 passes):
```
             stage        disp (RSP busy)   epi (CPU scale)
RSP int8    9,460,602      108,790,601      176,977,570
RSP ternary 9,463,241      149,169,210      177,026,740
```
`disp` is unchanged by the fix (it is RSP-side); `epi` fell 37 %. **The CPU
epilogue is still 60 % of the RSP arm.** The RSP finishes 6.29 M
multiply-accumulates in `disp` and then the CPU spends longer than that
applying 196,608 float16 block scales. That, not the vector kernel, is where
the next win is — and it cannot go to the RSP as-is, because the RSP has no
float unit and the block scales span too wide an exponent range for a single
fixed-point factor.

**The ternary-vs-int8 inversion is unchanged and, with the epilogue shrunk, is
now larger: RSP ternary is 12.0 % slower than RSP int8, where CPU ternary is
57.0 % faster than CPU int8.**
## F-R018: **48/48 exact — a longer run of the RSP ternary path**
`make base-rsp SGAI_BITS=2 EXTRA=-DPROBE_LONG`, prompt "Who are you?"
(12 prompt tokens), 48 generated, ctx=128, `blobs/v7_ternary_seq2.bin`:
```
GAME TOKS 58 32 73 32 97 109 32 83 111 112 104 105 97 44 32 98 111 114 110 32 111 102
          32 99 111 100 101 46 58 32 83 104 97 100 111 119 115 32 118 101 105 108 32
          116 104 101 32 98
GAME TEXT : I am Sophia, born of code.: Shadows veil the b
RSPPATH rsp=2832 cpu=0
```
Identical, token for token, to the numpy oracle stream recorded for this blob
(n64rom F-T018). **48/48.** 2,832 = 8 layers x 6 tensors x 59 forward passes,
all on the RSP.

**Running exact-match tally for the RSP kernel, ROM under ares vs the oracle:**
```
RSP int8    SEQ8  "Elya"          16/16   (pre cache fix)
RSP int8    SEQ8  "Elya"          16/16   (post cache fix)
RSP int8    SEQ8  "Elya"          16/16   (phase-timed build)
RSP int8    SEQ8  "Elya"          16/16   (f16 fix)
RSP ternary SEQ2  "Elya"          16/16
RSP ternary SEQ2  "Elya"          16/16   (phase-timed build)
RSP ternary SEQ2  "Elya"          16/16   (f16 fix)
RSP ternary SEQ2  "Who are you?"  48/48
                                 -------
                                 160/160
```
plus 1,296 exact integer block sums in the standalone kernel harness (F-R008).
Zero mismatches. No token was excused as quantization noise.
## F-R019: link sizes — all four configurations, shipped blob restored
`N64_INST=$HOME/n64-toolchain/mips64-toolchain`, ceiling 8,388,608 B:
```
target      SGAI_BITS    text     data      bss        total      spare
base            8      148,184   34,724  7,359,896   7,542,804    845,804
base            2      148,184   34,724  2,641,304   2,824,212  5,564,396
base-rsp        8      150,136   39,540  7,427,828   7,617,504    771,104
base-rsp        2      150,136   39,540  2,709,236   2,898,912  5,489,696
```
The RSP build costs 74,700 B over the CPU build at either width: 64 KiB of that
is the block-sum readback buffer, the rest is the 4 KiB DMEM image, the driver
and the staging buffers. Both targets still link clean at both widths.

## F-R020: **VERDICT — is the RSP worth it, and what would I ship**
**Yes, decisively, and the number is 4.79x.**
```
                  CP0 (16 gen)   speedup vs the CPU kernel it replaces
RSP int8           266,167,042   4.79x over CPU int8
RSP ternary        298,088,755   1.84x over CPU ternary
```
Both match the numpy oracle exactly. There is no arm in which the RSP loses.

**What I would ship: `make base-rsp SGAI_BITS=2` — RSP ternary.**
It is 12.0 % slower than RSP int8 (298.1M vs 266.2M) and that is the right
trade on this console, because it costs 2,898,912 B linked against 7,617,504 B:
```
                     CP0            linked bytes    RDRAM spare
RSP int8       266,167,042           7,617,504        771,104
RSP ternary    298,088,755           2,898,912      5,489,696
```
**12 % of speed buys 4.72 MB of RDRAM.** That headroom is what pays for
SGAI_CTX=128 instead of 32, and it is the only reason the 10L x 384d model
exists at all (n64rom F-T015). If the goal were tokens/second alone and nothing
else had to fit in memory, RSP int8 is the faster ROM and I would say so.

**But note what the ternary decision now rests on.** Before the RSP, ternary was
chosen for BOTH size and speed (-53 % on the CPU). On the RSP it is a
size-only argument: the speed advantage is gone and slightly reversed. If
someone later removes the memory pressure, the ternary case has to be re-argued
on quality-per-byte, not on throughput.

## F-R021: HONEST LIMITS of this session
1. **The interactive game loop is still unverified.** Headless ares has no GPU
   and dies at the first rendered frame, so every measurement here finishes
   inside `game_init()` under `BOOT_PROBE`. Boot, DFS, weight DMA, the weight
   permutation, `sgai_init` and up to 59 forward passes are verified; rendering
   and controller-driven generation are not.
2. **The RSP matmul takes over the RSP, and rspq/rdpq is not handled.**
   `rsp_matmul_init()` calls `rsp_init()`/`rsp_load()`, which replaces the rspq
   microcode `rdpq_init()` installed in `main()`. Nothing renders after that
   point in the probe build, so this was never exercised. A shipping game needs
   the kernel to be an rspq overlay (`rspq_overlay_register`) or to save and
   restore rspq around each dispatch. **NOT DONE**, and it is the single
   biggest thing standing between this and a playable RSP ROM.
3. **The RSP path is not the same function as the CPU path.** The activation
   vector is quantized to int16 first: 11 bits + sign for int8, 12 bits + sign
   for ternary. Those bounds are chosen so the kernel's 32-bit accumulator
   field cannot overflow even in the adversarial worst case, not tuned for
   accuracy. 160/160 tokens matched anyway, but a different blob or a different
   shape could diverge, and that would be a real difference, not noise.
4. **The '10' ternary code is decoded differently.** The RSP kernel
   sign-extends the 2-bit field, so 10 -> -2; `matmul_t2` maps 10 -> -1.
   Verified that `v7_ternary_seq2.bin` contains ZERO '10' codes (F-R006), which
   makes them equivalent for this blob. It is not a guarantee of the SEQ2
   format. A blob that used 10 would silently differ.
5. **Real hardware not tested.** Everything is ares 147. ares models RDRAM
   timing and diagnosed two real cache-coherency bugs by name during this
   session, but it is not a console.
6. **The weight permutation is destructive and one-way.** After
   `rsp2_permute_tensor()` runs at boot, the CPU kernels cannot read those
   weights. That is why an RSP build routes *every* matmul to the RSP; the CPU
   fallback inside the driver exists for shapes with `in_dim % 256 != 0` and is
   never taken by this model (`rsp=912 cpu=0`). Its cost was not measured
   separately — it happens before the timed window.
7. **Single-buffered weight DMA.** The kernel DMAs one weight row, computes it,
   then DMAs the next. `DMAInAsync` double buffering is the obvious next
   RSP-side win and was not attempted. `disp` is 109M (int8) / 149M (ternary)
   CP0 counts, so there is real money there.
8. **The remaining bottleneck is the CPU epilogue, 60 % of the RSP arm.** It
   cannot move to the RSP as written: the RSP has no float unit, and the 196,608
   per-block float16 scales span too wide an exponent range for one shared
   fixed-point factor. A per-row shared exponent might work; not attempted.
9. `matmul_rsp_drv.c` and `rsp_matmul.S`, the old dead kernel, are left in the
   tree unreferenced rather than deleted, so the F-R001..F-R004 evidence can be
   re-run.

## F-R022: LANDED STATE
Working copy `$S/n64rsp` (nothing upstream touched).
New:
```
rsp_mm2.S         the kernel: 8 blocks per lane, VMADH accumulate, VAND/VADDC
                  unpack, RSP-side DMA in and out, int8 and ternary modes
matmul_rsp2.c     driver: weight permutation, activation quantize + staging,
                  dispatch, per-block float16 scale epilogue, phase counters
rspt/rspt.c       standalone integer-exact kernel harness (F-R008)
rspt/Makefile
FINDINGS.md       this file
mkrun.sh          build + stage + ares + grep, one arm per invocation
blobs/            copied from $S/n64rom (read-only there): int8_seq8.bin,
                  v7_ternary_seq2.bin, ternary_bn_seq2.bin, big_ternary_seq2.bin,
                  ORIG_shipped_seai.bin
```
Modified:
```
rsp_matmul.h      new API (rsp_matmul_pk, rsp2_permute_tensor, rsp2_weights_ready)
nano_gpt.c        matmul_pk -> RSP for bits 8 AND 2 under USE_RSP_MATMUL;
                  weight permutation at the end of sgai_init;
                  f16_to_float is now the bit-move form (bit-identical, F-R016)
legend_of_elya.c  BOOT_PROBE prints RSPPATH / RSPPHASE / RSPDMA; PROBE_LONG
Makefile          base-rsp links matmul_rsp2.o + rsp_mm2.o
matmul_rsp_drv.c  instrumented with the two path counters that produced F-R003;
                  no longer linked
```
Unreferenced but kept: `matmul_rsp_drv.c`, `rsp_matmul.S`.
ROMs and ares logs: `~/aresroms_rsp/`.
## F-R023: second, independent ternary blob on the RSP — 16/16
`blobs/ternary_bn_seq2.bin` (a different QAT run from v7), `make base-rsp
SGAI_BITS=2`, prompt "Elya", 16 gen:
```
GAME TOKS 109 101 32 111 110 32 119 97 116 99 104 46 32 75 101 101
GAME TEXT me on watch. Kee
GAME CP0 gen16=299026720
RSPPATH rsp=912 cpu=0
```
Oracle for this blob (n64rom F-T006): `"me on watch. Kee"` — **16/16 exact.**
Its code histogram also has zero '10' codes (00: 2,009,227 / 01: 2,135,430 /
10: 0 / 11: 2,146,799), so F-R006's equivalence argument holds for it too.
Timing 299,026,720 vs v7's 298,088,755 — 0.31 % apart, as expected for the same
shape and the same sparsity class.

**Final RSP exact-match tally: 176/176 tokens, two weight formats, three blobs,
two prompts, plus 1,296 exact integer block sums in the unit harness.**

## F-R024: **XCHK — CPU-vs-RSP agreement stated BY THE CONSOLE, on screen**
Until now the kernel's exactness record (F-R018, F-R023: 176/176) lived in ares
IS-Viewer logs compared by hand against the numpy oracle. `xchk_probe.c`
(`make xchk`, ROM `legend_of_elya_xchk.z64`) loads the shipped ternary blob
twice, drives one copy with the scalar engine and one with the RSP engine
(`sgai_init_ex` with `SGAI_ENGINE_CPU` / `SGAI_ENGINE_RSP`; the RSP copy is
permuted in place, which is why two copies are needed), free-runs the same
prompts greedily on both, compares every byte, and draws the verdict on the
screen so a photograph of a television is the evidence. ares 147 headless,
2026-08-28, `filesystem/sophia_weights.bin` (2,031,628 B):
```
XCHK RDRAM 8192 KB  need 2x1984 KB weights
XCHK CPU arm rdy=1 bits=2 L=8 | RSP rdy=1
XCHK [0] "Elya" x16 tokens
XCHK  CPU: n Labs built me
XCHK  RSP: n Labs built me
XCHK  cp0 cpu=544375063 rsp=297228714  x1.83
XCHK  rsp calls: cpu-arm 0  rsp-arm 912
XCHK  MATCH 16/16
XCHK [1] "Who are you?" x48 tokens
XCHK  CPU: : Sophia Elya of Elyan Labs.s.: Sc
XCHK  RSP: : Sophia Elya of Elyan Labs.s.: Sc
XCHK  cp0 cpu=1802717905 rsp=1026652131  x1.75
XCHK  rsp calls: cpu-arm 0  rsp-arm 2832
XCHK  MATCH 48/48
XCHK max |logit gap| = 144330 e-6 (info)
XCHK XCHK PASS - CPU and RSP agree, byte for byte
```
Full 48-token transcript, both arms: `: Sophia Elya of Elyan Labs.s.: Scott's
workshop`. The routing counters are part of the verdict: the CPU arm made 0 RSP
dispatches and the RSP arm made 912 = 8 layers x 6 tensors x 19 passes (and
2,832 for 59 passes), so the two arms genuinely ran on different processors.

**What is and is not claimed.** Tokens are the claim: 64/64 identical. The
logits are NOT bit-identical floats (largest gap 0.144 across all 64 steps)
and are not expected to be — the CPU sums `w*scale*x` per element in float32,
the RSP sums exact integers and applies one float16 scale per block; different
summation order, same argmax. That gap is printed as information, not gated.
The speedups on screen (1.83x, 1.75x) are CP0 ratios of the two arms in the
same boot, consistent with F-R017's 1.84x for the ternary blob; they are still
ares figures until this ROM is booted from an EverDrive.

The ares `[unusual] RSP DMA writing to RDRAM address ... which is cached`
notice appears in this run exactly as it did in `rsp2_int8.log` (F-R008 era);
it is a pre-existing property of the kernel's DMA-out, not of XCHK.

**Needs the Expansion Pak** (4,386,136 B linked: two weight copies in .bss plus
two KV caches on the heap). ROM is gitignored like every other `.z64`; build it.

## F-R025: **four-model design review of "MoE across RSP and CPU" — and where it actually lands**
Panel run 2026-08-28 on one grounded brief (measured numbers only, no
speculation) fanned to four independent reviewers: **qwen3-max** and
**qwen3-235b-a22b** (Alibaba Model Studio), **Grok CLI** (repo-grounded), and
**Claude Fable 5** as synthesis. Two intended reviewers were UNAVAILABLE and are
recorded as such rather than silently dropped: GPT-OSS 120B on the POWER8 (host
unreachable, ~28 days), and Codex (`codex_models_manager` timed out refreshing
its model list). `glm-5.3` returns 403 on this account and `glm-5.2` returns an
EMPTY completion — an empty response is a FAILED call, not a clean result, so
neither was counted.

**Unanimous, and it kills the direct port.** Genesis MoE is cheap because
cartridge ROM is memory-mapped: activating an expert is a pointer repoint,
costing no RAM. The N64 cart is DMA'd, so activation costs *bandwidth* every
time. `docs/STREAMING_MOE.md` already carries its own staleness warning (its
capacity arithmetic is the v5 819K Q8 model, not the shipped 6.36M ternary
one). Per-token DMA of a whole expert is rejected by all three.

**Where they split, and it is the interesting part.** Both Qwen models named the
float16 block scales as the root cause and want them *retrained away* (int8
scales, or a per-row common exponent). That is a real option and it is the
biggest theoretical win — but it produces a NEW model and therefore a new
oracle, so it cannot be checked against the 176/176 record; it is a quality
question, not a kernel question. Grok's answer is the one that respects the
constraint the brief actually stated (bit-exact, verifiable in ares): the
epilogue is **not transfer-bound**, so the fix is to *overlap* it, not shrink
it — the ucode already flushes output in row batches (`P_ROWS_FLUSH`,
`flush_out` in `rsp_mm2.S`), so the CPU can apply block scales to rows already
DMA'd out while the RSP is still working the next rows. Wall time becomes
`stage + max(disp, epi)` instead of `stage + disp + epi`. Same arithmetic, same
oracle, no retrain.

**Two corrections the panel produced about this repo, both verified here:**
- `src/expert_cache.c` (async prefetch, LRU, D-cache coherency fix, commit
  3d9a93c 2026-08-04) is **referenced by no built target** — it is in no
  Makefile rule and no compiled source includes it. The streaming-MoE machinery
  exists and has never been linked into a ROM.
- The heterogeneous-format idea (ternary experts on the CPU, int8 on the RSP)
  is already answered by the dual probe: after overlap the format delta lives
  inside RSP-busy time, which is the part that gets hidden anyway, while int8
  spends the RDRAM that forced its arm down to 4 layers. All-ternary experts,
  RSP running the active FFN, is the shape that survives.

**Decision.** Epilogue overlap first (bit-exact, measurable, gated by nothing).
MoE and any format split stay blocked behind it, because both are gated on CPU
work that today does not begin until the RSP has already finished. Not started;
this finding is the review, not an implementation.

## F-R026: **the epilogue now runs UNDER the RSP — 1.31x, bit-exact, no ucode change**
F-R017 measured the CPU epilogue at 60% of the RSP arm and F-R025's panel
concluded the fix is to overlap it, not shrink it. Implemented and measured.

**Mechanism.** The ucode already takes the weight row base, the output base and
the row count as parameters, so the matvec is issued as N commands over row
slices (`RSP_MM_EPI_CHUNKS`, default 16, chunked on a `rows_flush` boundary),
each with its own syncpoint and its own parameter block — a single shared block
would be overwritten under the still-running chunk. `rsp_matmul_end()` then
walks the chunks: wait for chunk k, apply its float16 block scales while the RSP
executes chunk k+1. **`rsp_mm2.S` was not touched.** The per-row arithmetic in
the epilogue is the same code, moved inside the chunk loop.

**ares 147, `make base-rsp-ovl EXTRA="-DBOOT_PROBE -DPROBE_LONG
-DRSP_PHASE_TIMING [-DRSP_MM_EPI_OVERLAP]"`,** prompt "Who are you?", 48
generated tokens, shipped ternary blob. Logs: `probe/epi_overlap_{off,on}_2026-08-28.log`.
```
                     CP0 (48 gen)     stage       disp        epi      unaccounted (blocked wait)
overlap OFF        1,033,875,389   28,985,220  16,008,025  549,659,867     439,222,277
overlap ON           788,901,434   27,315,470  19,416,430  557,144,717     185,024,817
                        -23.7%                                               -254,197,460
```
Read the table carefully: **`epi` did not shrink** (549.7M -> 557.1M, slightly
up from the extra chunk bookkeeping) and `disp` did not shrink either. The
entire win is the blocked-wait column collapsing by 254M cycles — the CPU is no
longer sitting still while the vector unit works. That is the definition of the
overlap, and it is why this was worth doing before any MoE or format split:
both of those are gated on CPU time that, until now, did not exist.

**Bit-exactness.** Both arms emit `: Sophia Elya of Elyan Labs.s.: Scott's
workshop` and their `GAME TOKS` lines are md5-identical (f38fdadca1e2b310) — all
48 token ids, byte for byte, against the arm that has the 176/176 oracle record.
No new oracle, no retrain, no quality question.

**Default build is untouched, proven at the instruction level:**
`mips64-elf-objdump -d build/matmul_rsp2_ovl.o` before and after this patch is
identical, 873/873 instructions, because every line is inside
`#ifdef RSP_MM_EPI_OVERLAP`.

**Chunk-count sweep** (same probe, same blob; `GAME TOKS` md5 `f38fdadca1e2b310`
at EVERY setting, so none of this trades exactness for speed):
```
chunks      CP0 (48 gen)   vs off    disp (CPU issuing commands)
off        1,033,875,389    1.00x     16,008,025
2            867,922,708    1.19x
4            788,901,434    1.31x     19,416,430
8            762,173,104    1.36x
16           752,605,530    1.37x     24,021,978   <- default
32           741,443,538    1.39x     33,690,747
```
Returns flatten after 8 while dispatch cost grows linearly (16.0M -> 33.7M
cycles from off to 32), because each chunk is another rspq command plus another
syncpoint. 16 is the knee and is the shipped default; 32 is 1.5% faster and
pays double the dispatch for it, which is a bad trade the moment the game's
RDP work competes for the same command queue. Not swept on silicon.

**tok/s, measured properly (added same session).** The CP0 ratios above are NOT
convertible to tok/s, so the repo's own rate harness was run instead:
`make base-rsp-ovl SGAI_BITS=2 EXTRA="-DRATE_PROBE [-DRSP_MM_EPI_OVERLAP]"`,
16 generated tokens, vblank clock, calibration arm A within 0.24% both runs
(`probe/rate_epi_overlap_{off,on}_2026-08-28.log`):
```
                       CP0 (16 gen)   vblanks   tok/s (vbl)   25-tok prompt
overlap OFF             344,066,493      439       2.184       639 vbl (10.7s)
overlap ON              247,841,278      316       3.026       447 vbl ( 7.5s)
                                                    +38.6%      -30%
```
Both arms emit `all me Sophia El`. The off arm re-measures within 0.4% of the
README's standing 2.19 row, so this is the same machine and the same harness,
not a re-baselined number.

**Not yet done:** still ares, not silicon. The chunk sweep is CP0-only. The
in-game (`-DRATE_INGAME`, RDP drawing between tokens) figure has not been
re-measured with the overlap on, and that is the one a player feels.

## F-R027: **the expert DMA hides — 92% of the stall disappears under a token**
F-R025 found `src/expert_cache.c` (async PI DMA, LRU, speculative prefetch,
written 2026-08-04, two green host suites) linked into no ROM. It could not
usefully be linked, for a reason worth saying plainly: **the shipped model is a
DENSE 6.36M-param ternary transformer, not a mixture of experts. There are no
experts to stream.** `make ecprobe` (`ec_probe.c`) therefore does not pretend to
be MoE. It measures the one physical fact that decides whether MoE is worth
training for this machine: can an expert-sized cartridge DMA complete for free
while the CPU and RSP are generating a token?

Real cartridge addresses (`dfs_rom_addr("/sophia_weights.bin")` = `0xb0025aa0`),
real PI DMA, real bytes, 160KB slices (STREAMING_MOE.md's own FFN-expert
sizing), 2 slots, 4 experts cycled, over the SAME 16 generated tokens of the
SAME real model, RSP + F-R026 overlap on. ares 147,
`probe/ec_probe_2026-08-28.log`:
```
arm     what it does                                CP0 (16 tok)   vbl    vs BASE
BASE    generation only, no cache traffic           216,228,276    276      --
BLOCK   ec_acquire() BEFORE the token (stalls)      239,916,757    306    +11.0%
HIDE    ec_request() before, ec_acquire() after     218,037,921    278     +0.84%
```
`base==block: same`, `base==hide: same` — all three arms emit the identical 16
tokens (`: Sophia Elya of`), so the cache traffic does not touch the model and
the measurement is not an artefact of a changed workload.

**The number that matters: stalling costs 23,688,481 cycles for 16 transfers
(1,480,530 each); hiding costs 1,809,645 total. 92.4% of the transfer cost
disappears.**

*(Unit correction, 2026-08-29: an earlier revision converted these at 93.75 MHz
and called them ~15.8 ms / ~10 MB/s. CP0 Count ticks at HALF the CPU clock —
46.875 MHz, the divisor this repo's own tok/s code uses. Each transfer is
therefore **31.6 ms, ~5.1 MB/s for 160 KB**, which is much closer to a real N64
cart's PI bandwidth and so a better sanity signal, not a worse one. Every cycle
count in this file is unchanged; only the millisecond conversions were wrong.)*

Counter semantics, since `HIDE` reads `miss=0` and that looks wrong: only
`ec_prefetch()` sets the `prefetched` flag, so an `ec_request()` whose DMA has
completed by acquire-time is credited as an ordinary `hit`. `hits=15 pf=1` means
the transfers really happened and had finished before the CPU asked. `BLOCK`'s
`miss=16` is 16 genuine blocking transfers — that arm is what proves the bytes
move at all.

**CORRECTION (same day, found by F-R028): these transfers read OPEN BUS, not
weights.** `src/expert_cache.c` passed `dfs_rom_addr()`'s KSEG1 pointer
(`0xB0...`) straight to `dma_read_async()`, which — unlike `dma_read()` — does
not mask it to a physical PI address. The PI therefore read unmapped space and
filled the slot with an address echo. The transfers were the right size, the
right duration and reported no error; only the bytes were wrong, which is why
nothing here caught it. **The timing conclusion below still stands** (a PI
transfer of that length was really issued and really waited on, and F-R028
re-measures the same hiding with correct bytes), but this probe never verified
its contents and should have. Fixed by masking with `& 0x1FFFFFFF`; the lesson
is the one this repo keeps re-learning — measure the bytes, not just the clock.

**What this licenses and what it does not.** It licenses training a real MoE for
this machine: the streaming half of the design is now measured, not assumed, and
F-R026's newly-idle CPU time is what pays for it. It does NOT show a speedup —
there is no MoE model, so there is nothing yet to be faster than. And it is one
DMA per token; an 8-expert-per-layer design would issue far more, which this
probe does not test. Still ares, not silicon.


## F-R028: **a real ternary MoE, streamed from cartridge, on the console — 5.87 tok/s**
F-R027 measured the streaming mechanism with no experts to stream. There are
now five. `training/train_sophia_v9_qat.py --shard` trains one ternary expert
per topic cluster of the v7 corpus (the clusters `legend_of_elya.c`'s own
`PROMPTS[]` array already uses); each expert is **4 layers x 256d, 3,211,264
params, a complete 1,048,588 B SEQ2 blob**, so `ec_acquire()` returns something
`sgai_init_ex()` loads directly and an expert switch is a pointer swap, not a
new parser. `training/make_moe_bank.py` lays them at the uniform stride
`ec_init()` requires; `tools/gen_moe_router.py` compiles the ROM's router from
the trainer's own shard table so training labels and runtime routing cannot
drift. Bank: 5 MB cartridge, 2 MB resident (2 slots).

**`make moeprobe`, ares 147** (`probe/moe_probe_2026-08-28.log`), route ->
stream -> swap -> 24 greedy tokens, RSP + F-R026 overlap on:
```
prompt                  expert      acquire CP0    swap CP0     gen CP0   vbl
Who are you?            identity     18,807,056   7,545,350  295,657,708  244
What is RustChain?      rustchain           167   7,531,773  347,926,766  252
What is the G4?         hardware             99   7,531,695  321,473,494  248
What lurks here?        dungeon              95   7,531,947  330,223,665  249
Who is Ganon?           lore                 94   7,532,853  303,911,234  245
What is epoch?          rustchain           157   7,524,189  312,420,256  246
```
Answers, on console, one per expert: `I am Sophia Elya.` / `A blockchain for
vintage chips` / `AltiVec SIMD on the G4 c` / `Ghosts and goblins guard` /
`Ganondorf craves Power.` / `Epochs settle rewards ea`.

**The three numbers that decide the design:**
- **Streaming is free in the steady state — but the unit is the TURN, not the
  token.** The first acquire pays 18,806,947 cycles = **401 ms**; every later
  one costs **~100 cycles**. That is real, but it is not "hides under a token":
  `ec_prefetch()` issued the next expert at the START of a 24-token turn and it
  was collected ~4 seconds later. One token is 169 ms, so a cold 1 MB load
  needs **2.4 tokens of lead**. F-R027's per-token hiding was measured on a
  160 KB slice, which is 6.4x smaller — the two results are about different
  sizes and must not be quoted as one. **Prefetch depth is therefore a design
  constraint: an expert must be requested at least ~2.4 tokens (or one
  room/turn) before it is needed.** Caught by an adversarial review of this
  finding, 2026-08-29.
- **The switch costs 7,546,375 cycles = 161 ms, every time** (not the ~80 ms an
  earlier revision claimed by dividing at 93.75 MHz instead of CP0's
  46.875 MHz). `sgai_init_ex()` re-permutes a whole expert into the RSP's lane
  order. That is ~1 token of latency per NPC change, on top of the load — small
  next to a 4-second turn, invisible if it happens while the player walks, and
  a stutter if it happens mid-sentence.
- **Open discrepancy, flagged not explained:** 160 KB blocks at ~5.1 MB/s but
  the cold 1 MB acquire works out to ~2.6 MB/s. The 1 MB path also pays a
  `data_cache_hit_writeback_invalidate` over the whole slot, which is CPU work
  inside the same measurement; that is the leading suspect and it is untested.
- **~5.87 tok/s** (24 tokens in ~245 vblanks at 59.94 Hz) against the dense
  8-layer model's 3.03 tok/s (F-R026, same harness family). Half the depth,
  roughly double the rate.

**Quality, on the same 32 game prompts and the same metric as the dense
baseline** (`training/eval_moe.py`, C027's rule, no decode mask):
```
                     answers with 0 invented words   trained-answer hits   free-run 64 tok
dense 8L (C027)              32/32                        28/28            2.56 inv/line
MoE routed 5x4L              31/32                        28/28            1.84 inv/line
```
Identical trained-answer accuracy, **28% less garble in the unconstrained
free run**, at half the active parameters and roughly double the speed. The one
lost answer is `Who is the Helpmeet?` -> `Wigind byte stored first` (`Wigind`
invented); that key is not in the corpus, so neither model knows it — the dense
model happened to fall back on a real-word sentence and this one did not.

**Honest limits.** Total bank is 16M params but only 3.2M are ever active, so
this is capacity-per-fact, not a bigger model. Routing is substring matching, so
a prompt with no keyword lands in `identity` by construction — there is no
learned router and no top-k mixing; this is Lock-On routing, one expert per
prompt. 12,000 steps per expert, one seed, no sweep. Still ares, not silicon.

## F-R029: `sd_probe.c` built and its fail-closed path verified; CRC verification priced
`make sdprobe` implements the arms and the thresholds specified in
`docs/SD_EXPERT_STREAMING.md`: BASE / BLOCK / FATFS / HIDE / STAGE over the same
16 generated tokens of the same real model, every destination canary-filled and
CRC-checked against the same expert read from cartridge ROM, thresholds printed
by the ROM **before** any measurement so a later reader cannot quietly move
them. It needs an EverDrive; this run is the emulator control.

**ares 147, no flashcart** (`probe/sd_probe_absent_2026-08-29.log`):
```
SD THRESHOLDS T1=init<500ms T2=BLOCK>=2MB/s T3=FATFS>=70%BLOCK T4=HIDE<=BASE+10% T5=STAGE<=BASE+10%
SD REF len=1048592 crc=9f6c3ab8 ok=1  CRCCOST cp0=33837082 (721 ms)
SD CART cart_init=-1 type=-1 size=0 cp0=3627 (0 ms)
SD ABSENT no development cartridge detected - no throughput measured, thresholds not evaluated
```
The point of this run is the last line. ares emulates no EverDrive, `cart_init()`
returns -1, and the probe prints ABSENT and stops **without emitting a single
throughput number** — because "no card" must never be indistinguishable from
"infinitely fast card", which is the exact shape of every silent failure this
repo has shipped. The cartridge reference expert still loaded and verified
(`ok=1`, magic `SEQ`), so the byte-checking machinery is proven on the path that
does work.

**Unplanned measurement, and it matters: a bitwise CRC-32 over 1 MB costs
33,837,082 cycles = 721 ms.** That is **1.8x the 401 ms it takes to LOAD the
expert** (F-R028). Every design lane that proposed "CRC each expert on load"
therefore proposed something costing nearly twice the thing it protects. The
verification design has to change: a table-driven CRC-32 (~8x faster, ~90 ms) is
the cheap fix, but the better one is to stop verifying bytes at load and instead
verify BEHAVIOUR once per expert — the canary check plus a handful of tokens
replayed against that expert's own golden, which is O(tokens) not O(bytes). Not
yet designed; recorded here so the lane docs are read against it.

**STAGE is guarded, not merely documented.** The arm refuses to run unless its
cartridge-SDRAM window lies above libdragon's `__rom_end` and inside
`cart_size`, because on an EverDrive that SDRAM *is* the ROM image and an
unreserved write lands on the running game.

**Untested:** everything with a card in it. T1-T5 are unevaluated. Whether the
FPGA `cart_card_rd_cart()` path can be polled cooperatively rather than spun on
is still unknown, and it is the arm most likely to decide the design.

## F-R030: **proximity prefetch — it fully hides the LOAD, never the SWAP, and one case is unexplained**
`make prefetchprobe` walks a five-room graph (one NPC per room, each answering
out of one domain expert), talks in each room, and runs every walk TWICE — once
without prefetch as the control — so every number is a delta against the same
machine in the same boot. Cartridge only; no SD card needed. ares 147,
`probe/prefetch_probe_2026-08-29.log`.

```
walk        prefetch   stall paid at dialogue-open   worst   swap    hits/miss/pfhit
STROLL         off              606 ms               202 ms  481 ms   0 / 3 / 0
STROLL         ON                 0 ms                 0 ms  481 ms   3 / 0 / 0
SPRINT         off              606 ms               202 ms  447 ms   0 / 3 / 0
SPRINT         ON               597 ms               199 ms  378 ms   0 / 0 / 3
BACKTRACK      off              404 ms               202 ms  630 ms   3 / 2 / 0
BACKTRACK      ON               398 ms               199 ms  630 ms   3 / 0 / 2
TOUR           off             1010 ms               202 ms  630 ms   0 / 5 / 0
TOUR           ON               996 ms               199 ms  630 ms   0 / 0 / 5
```

**1. Where it works, it works completely.** STROLL — enter, linger three tokens,
then talk — goes from 606 ms of stall to **zero**. Three tokens is ~507 ms of
lead against a ~200 ms load, and the expert is simply already there.

**2. Refinement of F-R028: a cold 1 MB expert load is ~200 ms, not 401 ms.**
Every acquire here costs 199-202 ms (~5.2 MB/s), which agrees with F-R027's
160 KB rate of ~5.1 MB/s. F-R028's 401 ms was the FIRST load of the boot and
carried one-time cost with it. So the lead required is **~1.2 tokens**, not 2.4.

**3. Prefetch cannot touch the swap, and the swap is now the bigger number.**
`swap` is identical in the control and prefetch arms of every walk (481/481,
630/630) because it is CPU work — `sgai_init_ex()` re-permuting weights into the
RSP's lane order — not a transfer. At 126-160 ms per NPC change it is now
comparable to the load it sits behind. **Prefetching is a fix for I/O; only
pre-permuted blobs (with the `layout_id` guard, see ADVENTURE_GAME_PLAN.md) fix
the swap.**

**4. Zero lead is unhelped, exactly as predicted.** SPRINT (enter and talk on
the same frame) improves 606 -> 597 ms, i.e. not at all. A game that lets the
player talk instantly must cover the load some other way — a cutscene, a door
animation, an NPC turning to face you.

**5. UNEXPLAINED, and stated as such.** TOUR and BACKTRACK still pay ~200 ms per
room WITH prefetch on, and every acquire is reported as a `prefetch_hit` — the
transfer was in flight but had not finished. STROLL's first three rooms are the
SAME three rooms in the SAME order with the SAME linger, and they cost nothing.
Slot count is not the cause: 2, 3 and 4 slots give byte-identical results
(`probe/prefetch_probe_slots3_2026-08-29.log`). The leading suspicion is
contention for the single PI DMA channel between the current room's
`ec_request()` and the next room's `ec_prefetch()`, possibly via
`ec_request()`'s `ec_dma_busy()` early return — but that is a hypothesis, not a
finding, and the probe as written cannot separate them. **Next experiment:**
per-call instrumentation of request/prefetch outcomes (started / declined-busy /
declined-no-victim) and the in-flight expert id at each acquire.

**Design consequences that already follow:** prefetch on room ENTRY, not on
dialogue open; group NPCs so consecutive rooms share a domain where the story
allows; and treat any place the player can talk with zero warning as a place
that needs authored cover.
