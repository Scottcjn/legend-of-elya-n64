# N64 Dual-Processor Consensus — findings journal

Two models, two processors, one token stream. A ternary model on the MIPS
core, an int8 model on the RSP vector unit, their logits summed, the sum
sampled. Every number below is a count read off the machine — CP0 COUNT for
cycles, the VI vblank interrupt for wall time — or a host measurement that the
machine has been checked against. Where something is an assumption it says so.

Reproduce:

```sh
export N64_INST=$HOME/n64-toolchain/mips64-toolchain
sh training/dual_train.sh 1337 7 23 101          # 12 blobs, ~2 h on a 4070
make dual EXTRA="-DDUAL_SHIFT=0 -DRSP_PHASE_TIMING"
python3 scripts/ares_rate_run.py $HOME/dual.z64 run.log DUAL_DONE 1100
python3 training/dual_eval.py --tern mix_out/tern8_s1337.seq \
                              --int8 mix_out/int84_s1337.seq --verify-shift 0
sh training/dual_sweep.sh                        # every pairing, 4 seeds
```

Raw logs: `docs/dual_logs/`.

---

## The machine, and why this shape

Measured earlier on this port, 16 tokens, CP0 (`docs/N64_RSP_FINDINGS.md`):

| kernel | CP0 | |
|---|---:|---|
| CPU int8 | 1,275,545,400 | |
| CPU ternary | 548,354,836 | **−57.0 %** — ternary wins on the MIPS core |
| RSP int8 | 267,472,813 | |
| RSP ternary | — | **+11.9 %** slower than RSP int8 |

The RSP fetches eight operands per transaction, so ternary's operand-traffic
advantage vanishes there and only its unpack cost remains: 0.656 issue slots
per weight against int8's 0.375. The two processors prefer opposite formats.
That is the premise the rest of this document tests.

## Configuration

- Ternary arm: `tern8`, 8 layers, 256 embed, SEQ2 blob, 2,031,628 B, MIPS core.
- int8 arm: `int84`, **4** layers, SEAI/SEQ8 blob, 3,407,884 B, RSP.
  8 layers of int8 (6,750,220 B) plus the ternary blob is 8,781,848 B against
  an Expansion Pak's 8,388,608 B of RDRAM, before either KV cache, the code,
  the stack or a framebuffer. The 4-layer arm is what fits; F-DU005 prices it.
- Both arms `pse = 0`. The Physarum router is one global and two interleaved
  models would condition each other through it; it is also the configuration
  the numpy oracle implements, so `pse = 0` is the arm that can be checked
  against the host at all.
- Sampling: greedy, argmax over printable ASCII 32..126, which is exactly what
  `training/eval12.py` decodes.

---

## F-DU001 — the dual ROM runs, and shift 5 is wrong for these blobs

First three-arm run (`dual_logs/dual_run1.log`). The vote's transcript came
back as the ternary transcript, character for character — the failure the
shift exists to prevent, arriving from the other direction.

The sibling port measured its int8 head **54.7x louder** than its ternary head
and used a right-shift of 5 to make them commensurate. On these blobs:

| | logit RMS |
|---|---:|
| ternary (ROM) | 6.608 |
| int8 (ROM) | 7.530 |
| **ratio** | **1.14x** |
| ternary (host, over the key set) | 7.8194 |
| int8 (host) | 8.2921 |
| **ratio** | **1.06x → shift 0** |

`>>5` divides the int8 arm by 32 when it needed to be divided by 1, so the
int8 member contributed nothing. **54.7x was never a property of the scheme.
It was a property of that port's pair of models.** Re-measure it per pair;
`training/dual_eval.py` prints it and the nearest power of two.

## F-DU002 — the driver dispatched, but never started the RSP

`rsp_matmul_begin()` as committed could not overlap anything. `rspq_write()`
and `rspq_syncpoint_new()` both only append to the command buffer in RDRAM;
nothing tells the RSP there is work until `rspq_flush()`, and the only flush
on the path was the `rspq_flush_internal()` inside `rspq_syncpoint_wait()` —
which runs in `end()`, after the CPU has already done the other model's matmul
and come back. The RSP would have started when the CPU stopped.

`rsp_matmul_pk()` (begin immediately followed by end) hides this completely,
which is why the split-driver regression run was clean. Fixed by flushing in
`begin()`. This is the entire mechanism; without it every number below is the
serial number.

Also fixed: `rsp_matmul_init()` was not idempotent, and
`rspq_overlay_register()` asserts on a repeat registration — the second model's
`sgai_init_ex()` would have halted the machine at boot.

## F-DU003 — 100.0 % of the available overlap, which is worth 5.7 %

One build, one ROM, one run (`dual_logs/dual_run_serial_vs_overlap.log`),
16 tokens, greedy, prompt `sage says: Who are you?: `, vblank-measured.
`SERI` and `VOTE` are the *same vote* — same two models, same shift-0 sum,
same sampler — and differ only in whether the CPU waits for the RSP before
starting its own matmul or after (`sgai_dual_serial`).

| arm | vbl | tok/s | CP0 (gen) | stage | disp | **wait** | epi | text |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TERN | 773 | **1.240** | 605,590,390 | — | — | — | — | `Call me Sophia E` |
| INT8 | 206 | **4.655** | 161,679,944 | 3.41 M | 2.18 M | 43.80 M | 74.25 M | `Calobe corad for` |
| SERI | 981 | **0.977** | 768,547,819 | 3.68 M | 2.18 M | 43.78 M | 74.67 M | `Call me Sophia E` |
| VOTE | 925 | **1.036** | 724,753,859 | 3.61 M | 2.18 M | **0.071 M** | 73.65 M | `Call me Sophia E` |

`rsp_t_wait` is CP0 spent inside `end()` blocked on the vector unit — the
overlap *not* achieved. It falls 43,779,767 → 70,662, **99.84 % of the
blocking removed**, and free/blocked goes 0/384 → 384/0: the RSP had finished
every one of the 384 dispatches before the CPU came back for it.

**The overlap, checked two ways and agreeing.**
43,779,767 CP0 ÷ 46,875,000 Hz = 0.9339 s = **55.98 vblanks** of RSP time
there to be hidden. Measured wall clock fell 981 → 925 = **56 vblanks**.
56.0 available, 56.0 taken.

**And it is worth 5.7 %.** Those are not in tension. Only the matmul inner
loop runs on the RSP, and it is 56 of the 208 vblanks the int8 arm costs. The
int8 arm's marginal cost inside the vote is 119,163,469 CP0, of which

| | CP0 | share of the arm |
|---|---:|---:|
| driver float epilogue (one multiply per 32-weight block, CPU) | 73,654,565 | 61.8 % |
| activation quantize + lane-order staging (CPU) | 3,614,632 | 3.0 % |
| cache ops + dispatch (CPU) | 2,176,674 | 1.8 % |
| blocked on the RSP | 70,662 | 0.06 % |
| the int8 model's attention, norms, embedding, logits (CPU) | ≈ 39.6 M | 33.2 % |

Against the frontier: **vote / ternary = 925/773 = 1.197x**, serial would be
1.269x. **1.00x is not reachable by overlapping**, because 73 % of the second
arm is not the vector unit's work. We are on the achievable frontier; the
achievable frontier is 1.197x, not 1.00x.

If anyone wants the next 10 %, it is the epilogue, not the dispatch.

### Output correctness

Every arm matches the host reference character for character
(`training/dual_eval.py`, `dual_logs/dual_eval_s1337.log`):

| arm | ROM | host |
|---|---|---|
| TERN | `Call me Sophia E` | `Call me Sophia E` |
| INT8 | `Calobe corad for` | `Calobe corad for` |
| VOTE (shift0) | `Call me Sophia E` | `Call me Sophia E` |
| SERI (shift0) | `Call me Sophia E` | — (must equal VOTE, and does) |

### The caveat, stated plainly

**ares does not arbitrate the RDRAM bus.** A previous agent's instrument
self-test found the CPU running 0.09 % *faster* while the RSP pulled 444 MB/s.
The 925 vblanks is a directly measured wall clock, not `max(CPU, RSP)`, so
the overlap itself is not an assumption — but on silicon the RSP's DMA traffic
would steal cycles from the CPU, and **925 is a lower bound for real
hardware.** Nothing here has run on a console.

## F-DU004 — the vote beats both members on held-out text, p < 0.001, by 1.6 points

The whole-key eval the shipped model is quoted on cannot see any of this: both
members score **125/127 = 98.4 %** there and the ceiling is the measurement.
So the vote is scored on the **paraphrase holdout** — the last answer of every
key with three or more answers, plus every tenth corpus line, i.e. the only
strings in this project a trained model has genuinely not seen —
teacher-forced, next character, 1716 trials.

Seed 1337, ternary8 + int8-4 (`dual_logs/dual_eval_s1337.log`):

| rule | accuracy | vs ternary (McNemar exact) |
|---|---:|---|
| ternary alone | 57.81 % | — |
| int8 alone | 57.05 % | — |
| sum (shift 0) | 59.32 % | +59/−33, **p = 0.0088** |
| shift 1 | **59.44 %** | +48/−20, **p = 0.00091** |
| RMS normalise | 59.32 % | +58/−32, p = 0.0080 |
| shift 5 | 58.16 % | +8/−2, p = 0.11 — **not significant** |

The two arms see the same 1716 characters, so the test is paired; McNemar
conditions on the positions where exactly one of them is right, which is the
only place the difference can come from.

**The cheap rule matches the expensive control**: shift 1 and full RMS
normalisation agree to within a fifth of a point. That part of the sibling
port's claim does reproduce — with a shift of *one*.

**It is +1.6 points, not +7.1.** The effect is real and significant and a
quarter the size. The budget says why: the members disagree on 27.2 % of
characters, and an oracle picking the right one every time would score 62.8 %
against ternary's 57.8, so the whole prize is 5.0 points and the sum takes a
third of it.

## F-DU005 — the diversity is not the format, and not the seed either

Four seeds, every pairing that answers a different question
(`training/dual_sweep.sh`, `dual_logs/sweep_*.json`). Sum-of-logits accuracy
on the same 1716-character holdout:

| pairing | s1337 | s7 | s23 | s101 | mean |
|---|---:|---:|---:|---:|---:|
| ternary alone | 57.81 | 58.45 | 57.40 | 57.87 | 57.88 |
| **A** ternary8 + int8-4, same seed | 59.32 | 59.62 | 59.32 | 59.56 | **59.46** |
| **B** ternary8 + int8-8, same seed | 59.56 | 60.31 | 59.09 | 60.31 | **59.82** |
| **C** ternary8 + int8-4, cross seed | 59.67 | 60.08 | 60.02 | 59.27 | **59.76** |
| **D** ternary8 + ternary8, two seeds | 60.49 | 60.84 | 58.51 | — | **59.95** |

- **C ≈ A**: pairing models that share no seed scores the same as same-seed.
  The seed is not where the diversity comes from. (This much agrees with the
  sibling port: it measured cross-seed 27.1 % against same-seed 27.3 %.)
- **D ≈ A**: pairing the *same format* twice scores the same as mixing
  formats. **The format is not where the diversity comes from either.** The
  sibling port's "the diversity is the FORMAT not the seed" **does not
  reproduce here.** On this port the gain is what you get from two members of
  this family, however they differ.
- **B ≈ A**: doubling the RSP arm's depth from 4 layers to 8 buys +0.36
  points. The 4-layer arm is what fits in RDRAM and it costs almost nothing.

That is an accuracy finding, and it does not settle the design, because the
pairings are not equally affordable. F-DU006 measures the cost side.

## F-DU006 — once the RSP is overlapped, the format stops mattering

Same ternary 8-layer model on the MIPS core, same **four** layers on the RSP,
same seed, same shift, same sampler — only the RSP arm's weight format differs
(`dual_logs/dual_run_tern8_tern4.log` against `dual_logs/dual_run_verify_shift0.log`).

| | int8-4 on RSP | ternary-4 on RSP | delta |
|---|---:|---:|---:|
| B arm alone | 206 vbl | 228 vbl | **+10.7 %** |
| SERI, no overlap | 981 vbl | 1003 vbl | +2.2 % |
| **VOTE, overlapped** | **925 vbl** | **926 vbl** | **+0.11 %** |
| tok/s (vote) | 1.036 | 1.035 | |
| `rsp_t_wait`, SERI | 43,838,878 | 60,393,154 | |
| `rsp_t_wait`, VOTE | 95,251 | 95,097 | |
| epilogue CP0 | 73,663,603 | 73,663,603 | identical |
| staging CP0 | 3,423,380 | 3,411,019 | identical |
| blob | 3,407,884 B | 1,048,588 B | **3.25x smaller** |
| bss | 5,555,696 B | 3,196,400 B | **2,359,296 B less RDRAM** |
| held-out accuracy (sum) | 59.32 % | 59.27 % | the same |

int8 **is** cheaper on the vector unit — +10.7 % standalone, reproducing the
+11.9 % in `docs/N64_RSP_FINDINGS.md` at a different depth on a different
blob. **The premise is correct.**

It is also, once overlapped, worth **one vblank in 926.** The entire int8
advantage lives in RSP-busy time, and RSP-busy time is exactly what the CPU
stops waiting for. The 16.5 M CP0 the ternary arm spends extra on the vector
unit is spent while the MIPS core is doing the ternary model's own matmul, so
it costs nothing. What survives the overlap — the float epilogue, the
staging, the model's own attention and norms — is byte-for-byte identical
between the two formats.

**So the design conclusion inverts.** The two processors do prefer opposite
formats; it does not matter, because the preference is expressed entirely in
the part that gets hidden. What is left is memory, and there ternary wins by
2.36 MB on an 8 MB machine — enough to run the RSP arm at 8 layers instead of
4, or to leave the Expansion Pak's second half for something else.

Overlap, third independent configuration, still complete: 60,393,154 CP0 =
77.2 vblanks available, 1003 − 926 = 77 taken.

### F-DU006a — the same-format pair at full depth

`DUALFS=filesystem_dual_tt`, ternary on both processors, 8 layers each,
seeds 1337 and 7:

| arm | mixed (tern8 CPU + int8-4 RSP) | same format (tern8 CPU + tern8 RSP) |
|---|---:|---:|
| TERN | 774 vbl / 1.239 tok/s | 774 vbl / 1.239 tok/s |
| B arm | 206 vbl / 4.655 | 435 vbl / 2.204 |
| SERI | 981 vbl / 0.977 | 1211 vbl / 0.791 |
| VOTE | 925 vbl / **1.036** | 1057 vbl / **0.907** |
| `rsp_t_wait` | 43,838,878 → 95,251 | 120,794,547 → 189,911 |
| bss | 5,555,696 B | 4,179,440 B |
| held-out accuracy (sum) | 59.32 % | 60.49 % |

154.5 vblanks of RSP time available to hide, 154 taken. The mechanism does
not care which format the RSP is running.

**Harness cross-check.** The B arm here is the shipped configuration — the
ternary model, on the RSP, through the rspq overlay — and it measures
2.204 tok/s against the scalar arm's 1.239, i.e. **1.78x**. That is exactly
the honest baseline in `docs/N64_RATE_FINDINGS.md` (1.22 → 2.16, 1.78x),
reproduced by a completely separate probe. The instrument is sound.

## F-DU007 — ROM against host, character by character: CPU exact, RSP 98.84 %

The VERIFY arm runs 6 held-out lines teacher-forced, 258 predictions, and
prints each arm's pick string; `training/dual_eval.py --verify-shift 0`
prints the host's. Compared position by position:

| arm | exact vs host | ROM hits | host hits |
|---|---:|---:|---:|
| TERN — ternary on the CPU | **258/258** | 189 | 189 |
| INT8 — int8 on the RSP | 255/258 | 193 | 191 |
| VOTE — the sum, shift 0 | **258/258** | 197 | 197 |

The CPU arm and the vote are byte-identical to the numpy oracle. The RSP arm
is not, and should not be expected to be: the driver **quantizes the
activation vector** before handing it to the vector unit, so the RSP path
computes an int16 approximation of a float32 dot product, while the CPU
kernels multiply float activations by dequantized weights and carry no such
error. 3 of 258 positions flip, all three in one line, all three where two
logits were near-tied.

It cost the vote nothing here — at all three positions the ternary member
decided the sum — and that is what makes F-DU004's host accuracy number
transferable to the console without generating 1716 tokens on a 1.04 tok/s
machine: **the arithmetic is the host's arithmetic.**

---

## Summary — the four questions

**Does the vote run, and is it output-correct?** Yes. Three configurations of
the dual ROM boot under ares and generate. Against the numpy oracle on 258
teacher-forced held-out predictions the CPU arm is **258/258 exact** and the
**vote is 258/258 exact** in two of three configurations and 257/258 in the
third; the RSP arm is 255–257/258, because its driver quantizes activations
and the CPU's does not (F-DU007).

**tok/s, vblank-measured, 16 tokens.**

| arm | vbl | tok/s |
|---|---:|---:|
| ternary 8L, MIPS core, alone | 773–774 | **1.240** |
| int8 4L, RSP, alone | 206 | **4.655** |
| ternary 8L, RSP, alone | 435 | **2.204** (= 1.78x scalar, the published baseline) |
| the vote, serial | 981 | **0.977** |
| **the vote, overlapped** | **925** | **1.036** |

**How much overlap?** All of it, three times over. `rsp_t_wait` 43.84 M →
0.095 M CP0 (99.8 % of the blocking gone); 56.06 vblanks of RSP time
available and 56 taken; 77.2 available and 77 taken; 154.5 available and 154
taken. In the brief's terms the vote costs **1.197x** the ternary model
against a serial **1.269x** — and **1.00x is not reachable**, because only
the matmul inner loop is the vector unit's and that is 27 % of the second
arm. The other 73 % is the driver's float epilogue (61.8 % of the arm) and
the model's own attention, norms and projections.

**Does the +7.1 reproduce?** No. The vote beats both members by **+1.6
points** on held-out next-character prediction, p = 0.00091 (McNemar exact,
n = 1716), consistently across four seeds. And the diversity is not the
format: pairing two *ternary* models scores as well (F-DU005), and at equal
depth the format is worth one vblank in 926 (F-DU006).

---

## What has not been done

- **Nothing here has run on silicon.** Every cycle count is ares's.
- **RDRAM contention is unmeasured.** ares does not arbitrate the bus, so the
  concurrent arm's wall clock is a lower bound for real hardware.
- The accuracy sweep is four seeds on one corpus of 400 distinct lines. It is
  enough for the sign and roughly the size of the effect; it is not enough to
  separate 59.46 from 59.95.
