# FINDINGS — the tok/s counter

Journal for the rate audit. Append-only; every entry is a discrete measured
result. Raw emulator output for every number below is in `docs/rate_logs/`.

The question: this repo published **~60 tok/s** (README, video description,
elyanlabs.ai homepage, launch article) while `README.md:45` simultaneously said
"~60 tok/s in emulator, ~1-3 tok/s on real hardware". Two published figures,
mutually contradictory, propagated to four places before anyone checked either.

Harness, identical for every run below:

```
ROM     make base SGAI_BITS=2 EXTRA="-DRATE_PROBE"          (scalar)
        make base-rsp-ovl SGAI_BITS=2 EXTRA="-DRATE_PROBE"  (RSP overlay)
        add -DRATE_INGAME to fall through into the real game loop
weights filesystem/sophia_weights.bin  sha256 8fe00867736a8e50...  2,031,628 B
model   SEQ2 ternary, 8 layers, d=256, 8 heads, ctx 128, KV int8
emu     ares 147 (flatpak), headless under xvfb, Video/Audio Driver=None
runner  scripts/ares_rate_run.py   (pty, so lines are timestamped as they land)
```

---

## F-RT001 — the old counter was a constant, not a measurement

`src/legend_of_elya.c` at `bf97959` (2026-02-22) — the build whose on-screen
readout was filmed:

```c
int elapsed = G.frame - G.gen_start_frame;
if (elapsed > 0)
    G.gen_toks_sec = (float)G.gen_out_count * 60.0f / (float)elapsed;
```

`G.frame` is incremented once per iteration of `main()`'s `while (1)`
(bf97959 line 878). `update_generating_step()` is called once per iteration
(line 891) and makes exactly one `sgai_next_token()` call per invocation
(line 733). So `gen_out_count` and `elapsed` advance by exactly 1 each time
that expression runs. Their ratio is pinned at ~1 by construction and the
readout is pinned at the literal `60.0f` in the source.

**Nothing on the right-hand side depends on how long anything took.** The
counter reads ~60 on any machine at any speed. The `60.0f` encodes an
assumption that one loop iteration takes 1/60 s. It takes 0.81 s.

Reproduced rather than argued: the old expression is evaluated alongside the
new one, on the same tokens, in the same run
(`make base EXTRA="-DRATE_PROBE -DRATE_INGAME"`, `docs/rate_logs/scalar_ingame.log`):

```
RATE HUD n=1  vbl=48  toks_vbl=1.24 toks_cp0=1.24 toks_legacyframe=60.00
RATE HUD n=10 vbl=485 toks_vbl=1.23 toks_cp0=1.23 toks_legacyframe=60.00
RATE HUD n=19 vbl=934 toks_vbl=1.21 toks_cp0=1.22 toks_legacyframe=60.00
```

`toks_legacyframe = 60.00` on every token of every run, on both the scalar and
the RSP build, while the real rate differs between them by 1.8x. It is a
constant wearing the costume of a measurement.

## F-RT002 — where the published 61.8 came from

The video description's **61.8 tok/s** is the same expression's off-by-one.
The first-token fix at bf97959:728 sets `gen_out_count = 1` in the same
iteration that sets `gen_start_frame = G.frame`, so from then on the numerator
leads the denominator by exactly one:

```
gen_toks_sec = 60 * (k + 1) / k        k = output tokens since the reset
```

which starts at 120 and converges on 60 from above. At k = 33,
`60 * 34 / 33 = 61.818` -> displayed `61.8`. To the digit.

The current tree computes exactly `60.00` instead, because the first-token
append moved behind `SGAI_EMIT_FIRST_TOKEN` and no longer pre-loads the
numerator. Same constant, different off-by-one.

## F-RT003 — does ares advance CP0 COUNT against wall clock? YES

This was the live suspicion: the comment near `legend_of_elya.c:236` says
"emulators run TICKS_READ() deterministically", and if CP0 COUNT does not
advance against real time under emulation then every CP0-derived number in
`docs/` is measuring a fiction.

The experiment (`RATE_PROBE` phase A): burn a known number of CP0 counts in a
spin loop and count the VI vblanks that elapse. Vblank is 59.94 Hz by the NTSC
video standard, not by anyone's opinion about CPU speed, so it is the
independent clock.

```
RATE A cp0=234375000 vbl=299 cp0_sec_x1000=5000 vbl_sec_x1000=4988
```

| clock | reading | -> seconds |
|---|---:|---:|
| CP0 COUNT | 234,375,000 counts @ 46.875 MHz | 5.0000 |
| VI vblank | 299 fields @ 59.94 Hz | 4.9883 |

**Disagreement 0.23 %**, at a duration where one vblank is 0.33 % — smaller
than the smallest quantity the vblank clock can express. Identical in all four
runs (two builds x headless/in-game); the RSP build reads `cp0=234375001` for
the same 299 vblanks.

**So the emulator is not the culprit.** ares's CP0 tracks its VI. That matters
more than the counter bug does, because it means every CP0-derived measurement
in `docs/` and `FINDINGS.md` stands.

The 0.23 % does not entirely vanish at longer durations, though, and what is
left of it is a different (much smaller) thing — F-RT010.

### The third clock

Both clocks above are inside the emulated machine, so they could in principle
be wrong together. `scripts/ares_rate_run.py` timestamps every line with the
**host** clock. 5.000 emulated seconds took 9.4-14.5 host seconds across runs
— headless ares runs this workload at 0.35-0.63x real time, and the ratio
varies with load on the host GPU. That is the emulator being slow, which is
honest and harmless: emulated time is self-consistent, so a rate derived from
it is the rate the console would show. It is the *host* wall clock that must
never be quoted, and no number in this document is derived from it.

## F-RT004 — the denominator covered the wrong interval

Separate from F-RT001, and it survives into the vblank rewrite if you are not
careful: the numerator counts **output** tokens, so the denominator must cover
the **output** interval. Timing from `start_dialog_from_prompt()` puts prompt
ingestion — 25 forward passes whose output is discarded — into the denominator
of a count that excludes them.

Measured on the same run, both denominators printed
(`RATE HUD FINAL` vs `RATE HUD PREFIX`, `docs/rate_logs/scalar_ingame.log`):

| build | output-phase denominator | prompt-inclusive denominator | error |
|---|---:|---:|---:|
| scalar | 934 vbl -> **1.219 tok/s** | 2,111 vbl -> 0.539 tok/s | -2.26x |
| RSP overlay | 528 vbl -> **2.156 tok/s** | 1,173 vbl -> 0.970 tok/s | -2.22x |

Under-reporting by 2.3x is still misreporting. Prompt ingestion is a real cost
and is quoted separately (F-RT006), not smuggled into a number labelled tok/s.

## F-RT005 — honest tok/s

### Headless probe — pure inference, 16 tokens, prompt `sage says: Who are you?: `

| build | CP0 counts | vblanks | tok/s (CP0) | **tok/s (vblank)** | s/token |
|---|---:|---:|---:|---:|---:|
| scalar CPU (`make base`) | 609,379,304 | 778 | 1.230 | **1.232** | 0.812 |
| RSP overlay (`make base-rsp-ovl`) | 342,795,521 | 438 | 2.187 | **2.189** | 0.457 |

Both arms emit the identical 16 characters, `all me Sophia El`. `RSPPATH
rsp=1968 cpu=0` — every one of the 1,968 matmuls went to the vector unit, none
fell back.

### In-game — the real vsync-locked loop, the game's own prompt path

19 output tokens, `Call me Sophia Elya`, RDP rendering a frame between tokens
exactly as it does for a player:

| build | vblanks | **tok/s (vblank)** | tok/s (CP0) | vs headless |
|---|---:|---:|---:|---:|
| scalar CPU | 934 | **1.219** | 1.220 | -1.1 % |
| RSP overlay | 528 | **2.156** | 2.156 | -1.5 % |

**Scalar 1.22 tok/s. RSP overlay 2.16 tok/s.** Both under ares; neither has
ever run on silicon.

### RSP speedup on the shipped blob: 1.78x

778/438 = 1.776x on vblanks, 609,379,304/342,795,521 = 1.778x on CP0,
934/528 = 1.769x in-game. Three independent denominators, same answer.

This is **not the 4.769x** in `FINDINGS.md` F-O013. That figure is a ratio of
CP0 cycle counts on the **int8** blob, which is not what ships, and the int8
scalar path is 2.3x slower than the ternary scalar path to begin with
(`FINDINGS.md` baseline: CPU int8 1,275,062,235 vs CPU ternary 548,354,836).
The RSP wins less on ternary because ternary already took most of the win on
the CPU. Both numbers are true; only one of them is about the shipped ROM.

## F-RT006 — prompt ingestion, quoted separately

One forward pass per prompt byte, output discarded. The 25-token
persona-prefixed prompt, before the first output character exists:

| build | vblanks | seconds | CP0 counts | CP0 seconds | s/token |
|---|---:|---:|---:|---:|---:|
| scalar | 1,169 | 19.50 | 915,661,383 | 19.53 | 0.780 |
| RSP overlay | 637 | 10.63 | 499,085,116 | 10.65 | 0.425 |

In-game the same prompt cost 1,177 and 645 vblanks — 0.7 % and 1.3 % more than
headless, which is the rendering.

Ingestion is slightly *cheaper* per token than generation (0.780 vs 0.812 s on
the scalar build) because it runs at temperature 0, taking the argmax path and
skipping the softmax and repetition penalty. That the two agree to 4 % is a
consistency check on the measurement: the same forward pass, timed twice by
different code paths.

## F-RT007 — the vblank counter quantises, and that is the point

Each token takes 48-50 vblanks on the scalar build, so the displayed rate can
only take the values 59.94/k: 1.25, 1.22, 1.20 and nothing between. The HUD
series walks down that ladder as the average lengthens. Arithmetic that is not
measuring time does not quantise like that — the legacy counter printed
`60.00` on all 19 tokens, dead flat.

A second tell, unplanned: **the vblank count is more reproducible than CP0.**
Across four scalar runs of the same 16 tokens, vblanks were 778 every time
while CP0 read 609,575,279 / 609,861,593 / 609,379,304 / 609,771,066 — a
0.08 % spread.
The vblank interrupt fires asynchronously against the workload, so it perturbs
the cycle count it is not itself measured by. The clock that cannot be faked
also turns out to be the steadier of the two here.

CP0 is still computed every token into `perf_toks_cp0` and never displayed. It
is kept so the two clocks can be compared on any target — including silicon —
without a rebuild. On every run above they agree to within 0.2 %.

## F-RT008 — was 60 tok/s ever physically possible? No

Worth settling independently of where the number came from, because a plausible
figure and an impossible one are different kinds of error.

The shipped model is dense: every one of its 6,356,992 parameters is read and
multiplied once per token (F-RT009). At 93.75 MHz:

| claim | cycles/token | cycles per parameter |
|---|---:|---:|
| 60 tok/s | 1,562,500 | **0.246** |
| measured scalar, 1.23 tok/s | 76,172,413 | 11.98 |
| measured RSP overlay, 2.19 tok/s | 42,849,440 | 6.74 |

0.246 cycles per parameter means **four multiply-accumulates per clock** on a
single-issue in-order scalar CPU that must also extract and sign-extend a
2-bit weight for each one. It is not a hard number to miss; it is not
reachable in principle.

Nor could the RSP have reached it, and the filmed build did not even use the
RSP: 8 lanes at 62.5 MHz is 500 M MAC/s, so 60 tok/s is 76 % of the vector
unit's theoretical peak with zero overhead, zero DMA and zero CPU work — for a
path where the CPU still does attention, both norms, the sampler and all
control flow.

### Why the fleet cross-check pointed both ways

The cross-check that opened this job — 60 tok/s implying 1.91 cycles per
parameter, 1-3 tok/s implying 38-114 — was computed against the **819,200**
parameters the README documented. The ROM ships 6,356,992 (F-RT009), 7.76x
more. Rescaled to the model that actually ships, 60 tok/s is 0.246 cycles per
parameter (impossible) and the measured 1.23 tok/s is 11.98 (merely slow).
The paradox was an artifact of the same documentation error, one table away.

11.98 cycles per parameter is a statement about the scalar dequantise-and-
accumulate loop, not about the measurement: unpack 2 bits, sign-extend, convert
to float, multiply, accumulate, with FP latency exposed on an in-order pipeline.
The RSP path does the same work in 6.74. Cache misses are not the story —
streaming all 2,031,628 blob bytes through 16-byte cachelines at ares's
measured 40.25 cycles per D-cache miss is 5.1 M cycles, 6.7 % of the token.

## F-RT009 — the shipped blob is 8 LAYERS, not 8 experts

Last session's open thread: `2,031,628 = 12 + 65,536 + 8*196,608 + 8*49,152`
was read as evidence of an **8-expert mixture**. It is not. The 8 is the layer
count.

`scripts/blob_layout.py filesystem/sophia_weights.bin` walks it:

```
magic           SEQ2   -> 2-bit weights
n_layers        8      <- byte 4 of the header
n_embed 256   n_heads 8   vocab 256   ctx 128
  wq 16384  wk 16384  wv 16384  wo 16384  wff1 65536  wff2 65536   = 196,608
  scales 49,152   (float16, one per 32-weight block)
layer 0..7      245,760 B each
walk ended at   2031628      file size 2031628      EXACT
parameters      6356992  (embedding 65536 + 8 layers x 786432)
```

Four independent confirmations, no interpretation required:

1. **The header says so.** Byte 4 of the 12-byte header is `n_layers`
   (`<IBHBHBB` in `tools/quantize_n64.py:parse_header`, `SGAIHeader` in
   `nano_gpt.h:110`). It reads 8. There is no expert-count field anywhere in
   the format for a mixture to have been recorded in.
2. **The per-layer figure decomposes exactly** into the six tensors the engine
   reads — wq/wk/wv/wo at 256x256 and wff1/wff2 at 1024x256 — 786,432 weights
   at 2 bits = 196,608 B, plus 786,432/32 float16 scales = 49,152 B. Nothing
   is left over, and the walk consumes the file to its last byte.
3. **The engine runs them sequentially.** `SGAI_N_LAYERS 8` (`nano_gpt.h:22`),
   and the forward pass is `for (int l = 0; l < SGAI_N_LAYERS; l++)` applying
   every layer to every token (`nano_gpt.c:1090`). There is no router and no
   selection. The one thing in the file called a "router" is the PSE
   per-head attention conductance, which redistributes attention between the
   8 heads of a layer and selects no weights.
4. **The MoE code is not built.** `src/expert_cache.c` exists and is real, but
   it appears in no Makefile target, is `#include`d by nothing, and
   `legend_of_elya.c:1962` says so in the comment explaining why the large
   model cannot be loaded: "which is what `src/expert_cache.c` exists for and
   it is not yet wired to the weight path."

What the arithmetic *did* correctly detect is that **the shipped model is not
the one the README documented**: 8 layers not 4, d=256 not 128, 6.36 M
parameters not 819 K, ternary not Q8, 1,984 KB not 458 KB, ctx 128 not 64. The
README table had been describing the v5 model in `weights/sophia_weights.bin`
while the ROM loaded `filesystem/sophia_weights.bin`. Two different files, one
table. Corrected in README.

## F-RT010 — what is left of the 0.23 %: ares's field rate is 59.83 Hz, not 59.94

F-RT003 answered the question that was asked (does CP0 advance against wall
clock — yes). It left a 0.23 % residual attributed to +-1-vblank quantisation.
Most of it is that. Not all of it, and a number I cannot explain is not a
result, so the spin was repeated at five durations.

| CP0 target | CP0 counts | -> s | vblanks | expected @ 59.94 | residual | implied Hz |
|---:|---:|---:|---:|---:|---:|---:|
| 1 s | 46,875,001 | 1.0000 | 60 | 59.94 | +0.10 % | 60.00 |
| 5 s | 234,375,000 | 5.0000 | 299 | 299.70 | -0.234 % | 59.80 |
| 20 s | 937,500,000 | 20.0000 | 1,197 | 1,198.80 | -0.150 % | 59.85 |
| 28.4 s | 1,330,032,704 | 28.3740 | 1,698 | 1,700.9 | -0.170 % | 59.844 |
| **90 s** | **4,218,750,002** | **90.0000** | **5,385** | **5,394.6** | **-0.178 %** | **59.833** |

The residual **settles** at -0.18 % instead of shrinking to zero, and it does
not grow with duration. That is the signature of a constant rate offset, not of
drift: a clock running fast would accumulate error linearly, and this does not.
The +0.10 % at 1 s and -0.23 % at 5 s are the +-1-vblank floor (+-1.7 % and
+-0.33 % respectively) swamping a -0.18 % signal.

**ares emits 5,385 fields in 90.0000 s of CP0 time = 59.833 +- 0.011 Hz.**
That excludes 59.94 by ten times the uncertainty. It sits on the ~59.83 Hz
figure for a real N64's NTSC progressive output, which is *not* the 60/1.001
of the broadcast standard the ROM's constant assumes.

Consequences, stated rather than tuned away:

- Every vblank-derived rate in this document is **0.18 % high**, because
  `g_vi_hz` is 59.94. 1.232 becomes 1.230; 2.189 becomes 2.185. Nothing
  quoted here to three significant figures moves.
- The CP0 and vblank columns of F-RT005 agree to 0.2 %, and this is why: it is
  not noise, it is this constant.
- Which of ares's two clocks is off *relative to silicon* cannot be settled
  without silicon. The ROM keeps the standard 59.94 and prints both clocks, so
  an EverDrive 64 run resolves it in one line of output.
- Nothing anywhere in this repo should be quoted past three significant
  figures until that run happens.

## F-RT011 — the ROM in the repository was a probe build

Found while restoring the tree, not while looking for it. `legend_of_elya.z64`
as committed at `ad93f79` — the file `README.md` links as **Download ROM** —
contains the strings `A_START` and `_INGAME_DONE`. It was built with
`-DRATE_PROBE -DRATE_INGAME`. Anyone who downloaded it got the measurement
harness: a five-second spin loop, sixteen tokens generated headless, then a
dialog auto-started in the render loop. Not the game.

```
$ strings -n 4 legend_of_elya.z64 | grep -E "A_START|_INGAME_DONE"
A_START
_INGAME_DONE
```

Replaced with a clean `make base SGAI_BITS=2`. Verified three ways: no probe
strings in the binary, the build is byte-reproducible across two runs
(sha256 `94cb7772...` both times), and it boots and runs 90 s under ares with
no crash output.

Instrumentation belongs in a probe build. The rule that made this session's
numbers trustworthy — the tokens must not change — has a companion: the
artifact must not change either.

---

# Summary

| question | answer |
|---|---|
| What did the old counter report? | 60.00 tok/s, on every token, on every build, at any speed. It is `60.0f` in the source. |
| Where did 60/61.8 come from? | `gen_out_count * 60.0f / elapsed` where both operands advance by 1 per token. 61.8 = 60x34/33, the first-token off-by-one. |
| Does ares advance CP0 against wall clock? | **Yes.** 234,375,000 counts (5.0000 s) across 299 vblanks (4.9883 s) — 0.23 %, inside the +-1-vblank floor. The emulator was not the culprit. |
| Honest scalar tok/s | **1.23** headless, **1.22** in-game |
| Honest RSP overlay tok/s | **2.19** headless, **2.16** in-game |
| RSP speedup on the shipped blob | **1.78x** (not 4.769x — that is CP0 cycles on the int8 blob) |
| Prompt ingestion | 19.5 s scalar / 10.6 s RSP for 25 tokens, quoted separately |
| Is the blob an 8-expert mixture? | **No.** 8 layers, dense, sequential. The MoE cache is unwired. |
| Do the two clocks agree exactly? | To 0.18 %. ares's field rate measures 59.833 +- 0.011 Hz, not the 59.94 the ROM assumes (F-RT010). Quote nothing past 3 significant figures. |

## What could not be done

- **Nothing here has been run on real N64 silicon.** Every number is ares 147.
  ares's cycle model is the reason to trust it — `docs/EMULATOR_ACCURACY.md`
  calibrates it at 1.00 cycle/instruction, 40.25 cycles per D-cache miss and
  268.62 cycles per uncached cart read, all the documented R4300i figures —
  but a cycle model is not a console. An EverDrive 64 run is the outstanding
  check. The new counter is built to survive it: on silicon vblank and CP0
  must agree, and the ROM prints both if they do not.
- **The RSP overlay has never run on silicon either**, and ares emits a
  cache-coherency warning at the first overlay switch (`FINDINGS.md` F-O004)
  that could not be confirmed benign off-hardware.
- **The 60 Hz assumption in the old counter is not the only 60 in the tree.**
  `FRAME_CYCLES 781250` feeds the on-screen CPU% readout the same way. It is
  not a tok/s figure and was left alone, but it is wrong for the same reason:
  a generating frame is not 1/60 s, it is 0.81 s.
- **PAL was not tested.** `g_vi_hz` is set from `get_tv_type()` and the probe
  reports `tv=1` (NTSC) in every run. The PAL path is correct by construction
  and unexercised.
- **The RPI engine's "~100x faster, ~1000+ tok/s" claim was not measured.** No
  RPI build has been through this counter. It is marked as an estimate in the
  README rather than left reading like a measurement.
