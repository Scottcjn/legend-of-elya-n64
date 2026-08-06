# Sampling-path investigation — why the N64 dialogue still degrades

Working copy only (`$S/n64samp`). `~/legend-of-elya-n64` untouched.

## S001. The repetition-penalty "fix" is on main as a KNOB THAT NOTHING TURNS ON

`git log -1` (1bc17e5) says in its own message:

> 2. The repetition penalty, on the temperature path only. ... This survived
> every previous verification because it lives only at temperature_q8 = 64,
> the game's output path

But:

* `git show HEAD --numstat` for `nano_gpt.c` is +51/-32 and the diff contains
  **no penalty-related line at all** (`git show HEAD -- nano_gpt.c` greps clean
  for REP/penalty/hist outside the commit message).
* `nano_gpt.c:809` still reads `#ifndef SGAI_NO_REP_PENALTY` ... `probs[t] = 0.0f;`
  i.e. the **hard zero is the default**.
* `grep -rn SGAI_NO_REP_PENALTY Makefile` -> **no match**. `git log -S` over the
  Makefile for that symbol -> **no commit ever added it**.
* The Makefile *does* default `PSE_FIX ?= 1`. So of the two causes the commit
  claims to fix, exactly one is actually enabled in a default `make base`.

The guard was introduced in f32aeff together with `PSE_COND_NORMALIZE`; only
`PSE_COND_NORMALIZE` was subsequently wired into the build.

**Predicted signature if this is the live fault: zero doubled letters.**
Scott's screen text:

    " am! Sophia Elya, the !64 ry! ry! T angua the crystals"

53 characters, **0 adjacent doubled letters**. English runs 2-3%.
Consistent. Not yet proven — S002+ instruments the actual sampler.

## S002. THE ANSWER TO TASK 1: the distribution is fine. The SAMPLER is broken.

`tools/samp_probe.c` replays `update_generating_step()` byte-for-byte (prompt at
temperature 0, output at `temperature_q8 = 64`, stop on `\n`, stop on `.` after
>= 8 chars, cap 80) and `nano_gpt.c`'s real `sample_logits` is instrumented in
place under `#ifdef SAMP_TRACE`. Prompt is the one the game actually builds in
`start_dialog_from_prompt()`: `persona_prefix + option` =
`"sage says: Who are you?: "`.

**The harness reproduces Scott's screen text.**
```
ares, real display, Scott   " am! Sophia Elya, the !64 ry! ry! T angua the crystals"
samp_probe run0             " am! Sophia Elya, the !64 ry! ry! The old am! The rogh to!..."
samp_probe run1             " am! Sophia Elya, the !64 ry! ry! Tel angua o!" The Crystal Dungeon, wher old ma"
```
First 33 characters identical. This is the path.

### The logged distribution, 80 sampled characters
```
top-1 pre-penalty probability >= 0.999999 : 68 / 80  (85.0%)
mean top-1 pre-penalty probability        : 0.989163
min  top-1 pre-penalty probability        : 0.569540
chosen char was the argmax                : 68 / 80  (85.0%)
chosen char was rank 1                    : 11 / 80
chosen char was rank >= 2 (a real tail draw): 1 / 80
```
**The distribution is not degraded at all.** At `temperature_q8 = 64` the model
puts essentially all its mass on one character at 85% of positions and is
emitting `" I am Sophia Elya, the N64 ..."` with near certainty. There is no
noise to sample from — this is not a tail-sampling problem in the usual sense.

### Where the bad characters come from — the `total <= 0` FALLBACK
```
steps where banned_mass >= 0.999         : 10 / 80
steps where total_left == 0 exactly      : 9 / 80  (11.2%)
steps that hit the fallback branch       : 9
characters emitted as '!'                : 8
'!' that came from the fallback          : 8 / 8   (100%)
```
Every single `!` is the fallback. The trace line is unambiguous:
```
  3 chosen=!  rank=1 pre_p=0.000000 fallback=1 banned=[m,a,SP] banned_mass=1.000000
    total_left=0 r=0 | top5: SP=1.00000(L63.632) !=0.00000(L14.713) ...
 22 chosen=!  rank=1 pre_p=0.000000 fallback=1 banned=[SP,e,h] banned_mass=1.000000
    total_left=0 r=0 | top5: h=1.00000(L44.123) SP=0.00000(L2.597)  ...
 28 chosen=!  rank=1 pre_p=0.000000 fallback=1 banned=[y,r,SP] banned_mass=1.000000
    total_left=0 r=0 | top5: SP=1.00000(L55.027) !=0.00000(L-9.440) ...
```

The mechanism, exactly:

1. The model is certain: `probs[' '] == 1.000000f`, everything else is
   **exactly 0.0f** (see below for why).
2. The repetition penalty zeroes the last 3 emitted chars. `' '` is one of them
   (a space occurs every ~5 characters of English, and the ban window is 3), so
   `banned_mass = 1.000000`.
3. `total` is now **exactly 0.0f**, so `if (total <= 0.0f)` fires.
4. The fallback scans `for (i = 32; i <= 126; i++)` and returns the first char
   not in the history. `i = 32` is `' '`, which is in the history. `i = 33` is
   **`'!'`**.

So the `!` in `" am! Sophia Elya"` is not a low-probability character the sampler
unluckily drew. It is **ASCII index 33** — the second slot of the printable
range — returned by a linear scan. It carries no model information whatsoever.
That is why it is always the same character.

`"ry! ry!"` is the same thing become periodic: trace steps 26-33 are
`r y ! SP r y ! SP`. The ban window and the fallback lock the sampler into a
period-4 limit cycle.

### Why the mass collapses to exactly 0.0f: the logits are enormous
Raw logits in the trace routinely reach **L64, L71, L77**. `temperature_q8 = 64`
gives `temp = 0.25`, so `inv_temp = 4.0` — the logits are *multiplied by four*
before the softmax. A gap of 10 raw logits becomes 40; `expf(-40) = 4e-18`, and
gaps of 50+ underflow float32 to **exactly zero**. Hence `probs[top] == 1.0f`
and literally every other entry `== 0.0f`.

Two consequences worth stating plainly:
* **`temperature_q8 = 64` is not "mild randomness". It is greedy.** The comment
  in `legend_of_elya.c:1531` ("T=0.25 -> mild randomness ... varied but coherent")
  is wrong about this model. 85% of steps are a saturated one-hot.
* Because the distribution is one-hot, **zeroing the top char zeroes everything**,
  which is what turns a soft "discourage repeats" heuristic into a hard
  "emit ASCII 33" instruction.

**VERDICT ON TASK 1: not a tail-draw. The distribution is good; the sampler
destroys it.** 9 of 80 characters were produced by a code path that never looked
at the model, and 68 of the remaining 71 were the argmax.

## S003. TASK 2 — temperature A/B on the GAME path. Temperature is not the variable.

`samp_probe`, prompt `"sage says: Who are you?: "`, 20 seeded runs per cell, both
arms compiled from the same tree, only `-DSGAI_NO_REP_PENALTY` differing.

| temp_q8 | T | AS SHIPPED: mean len / clean-term / doubled / '!' | NO_REP_PENALTY: mean len / clean-term / doubled / '!' |
|---|---|---|---|
| 0 (greedy) | - | 29.0 / **20/20** / 20 / 0.00% | 29.0 / 20/20 / 20 / 0.00% |
| 8 | 0.031 | 80.0 / 0/20 / **0** / 6.56% | 29.0 / **20/20** / 20 / 0.00% |
| 16 | 0.06 | 80.0 / 0/20 / **0** / 6.69% | 31.8 / **20/20** / 16 / 0.00% |
| 32 | 0.125 | 77.7 / 2/20 / **0** / 7.21% | 32.5 / **20/20** / 15 / 0.00% |
| **64 (the game)** | **0.25** | **75.4 / 4/20 / 0 / 8.16%** | **33.2 / 20/20 / 14 / 0.00%** |
| 128 | 0.5 | 74.0 / 5/20 / **0** / 4.32% | 33.2 / **20/20** / 14 / 0.00% |
| 256 | 1.0 | 73.8 / 7/20 / **0** / 5.15% | 33.2 / **20/20** / 14 / 0.00% |

Text, same prompt, same seeds:
```
temp 0   greedy, shipped code       " am Sophia Elya, the helpmeet."          20/20 clean
temp 64  AS SHIPPED (the player)    " am! Sophia Elya, the !64 ry! ry! The old am! The rogh to! The g!" Te!..."
temp 64  + SGAI_NO_REP_PENALTY      " am Sophia, your guide through the crystals."
                                    " am Sophia Elya, the helpmeet."          20/20 clean
temp 256 + SGAI_NO_REP_PENALTY      byte-identical to temp 64                 20/20 clean
temp 1024 (T=4) + no penalty        " am SophiaRua, your guide thre memories of every computation."
```

Three things fall out and none of them is "the temperature value is wrong":

1. **Greedy-in-game is clean; every non-zero temperature in the shipped build is
   not.** But the greedy branch is not clean *because it is greedy* — it is clean
   because `sample_logits`' greedy branch **deliberately applies no repetition
   penalty** ("No repetition penalty — matches the proven x86 reference"). The
   penalty and the temperature are entangled in the shipped code; the A/B above
   disentangles them.

2. **Holding the penalty off, temperature does nothing at all.** temp_q8 = 64,
   128 and 256 give *byte-identical* output (mean 33.2, same 14 doubled pairs).
   You have to reach temp_q8 = 1024 (T = 4.0) before a single character changes.
   This is the logit-saturation result from S002 measured end to end: raw logits
   of 40-77 mean the softmax is one-hot for any T below ~4. **The game's
   "stochastic sampling" is arithmetically greedy.**

3. **Holding the temperature at the game's 64, the penalty is the whole
   difference**: 4/20 clean terminations -> 20/20, 8.16% of characters `!` -> 0%,
   0 doubled letters -> 14. The `!` rate is essentially flat in temperature
   (4-8% everywhere) because it is not a temperature effect: it is the
   `total<=0` fallback firing whenever the banned character held the mass.

**Conclusion for task 2: the fault is the repetition penalty (plus the fallback
it triggers), not the temperature value and not the stochastic sampler as such.**

## S004. TASK 3 — the corpus mismatch hypothesis is REFUTED for the game's prompts

The prior agent's R6 is correct that the shipped blob is a **v7** model
(`train_sophia_v7.py`'s RustChain/Elyan-Labs text), not a v8 one. But the
inference drawn from it — that the dungeon prompts are out of distribution — is
wrong. The game does not use the v8 `<|sophia|>\nPlayer: ...\nNPC:` format. It
uses `persona_prefix + option`, and the options are the v7 corpus format
verbatim.

**All 12 of the game's dialog options appear literally in the v7 corpus, with
answers:**
```
"Who are you?: I am Sophia Elya."                          10 occurrences
"What lurks here?: Ghosts and goblins guard treasure."      6
"Tell me a secret.: The boss fears the boomerang."          5
"What is RustChain?: A blockchain for vintage chips."       6
"What is your name?: Sophia Elya."                          8
"What is proof of antiquity?: Old hardware earns more."     2
"What is MIPS?: CPU architecture in the N64."               1
"How big is your model?: Eighty kilobytes of weights."      1
"What do I need here?: Bring a bow and some arrows."        5
"What is the G4?: PowerPC G4 earns 2.5x RTC."               4
"What is AltiVec?: Vector math on PowerPC chips."           2
"What is vec_perm?: Shuffles vectors in one cycle."         2
```
`grep -c` on `train_sophia_v8.py` finds 0 of MIPS / AltiVec / vec_perm. **The
shipped blob is the RIGHT model for these prompts and a v8 blob would be the
wrong one.** Swapping in a v8-corpus model would make the game worse, not better.

The persona prefix (`"sage says: "`, `"scholar says: "`, `"smith says: "`) does
NOT appear in the v7 corpus, so it is out of distribution — but measured A/B with
the penalty off it changes the answer on only 3 of 12 prompts and never turns
English into noise. It is a minor irritant, not the fault.

`"the !64"` is also not "reaching for N64 from v7 vocabulary". The trace shows
step 22 wanted `h` with p=1.000000 (continuing `"the helpmeet"`), `h` was banned,
the fallback emitted `!`, and steps 23-24 then produced `6` `4` **conditioned on
the `!` that the fallback had just injected into the KV cache**. It is
derailment downstream of the fallback, not corpus reach.

### What the corpus/model IS responsible for
Greedy, PSE fixed, penalty off, bare v7 prompt — the cleanest possible run:
```
Who are you?:                 -> "I am Sophia Elya, the helpmeet."      (corpus: "I am Sophia Elya.")
Tell me a secret.:            -> "The N64 runs a transformer."           (corpus: "The boss fears the boomerang.")
What is RustChain?:           -> "Rewards machines that prove they still work."
What is MIPS?:                -> "To guide those who seek wisdom in old silicon."   WRONG
What is AltiVec?:             -> "The Crystal Dungeon, where old machines dream."   WRONG
What is vec_perm?:            -> "To guide those who seek wisdom in old silicon."   WRONG
What is the G4?:              -> "My name is Sophia Elya."                          WRONG
```
Roughly 3 of 12 answers are the trained answer. The rest are **fluent, in-voice,
grammatical English that answers the wrong question** — the identity attractor.
That is expected from the v7 recipe: IDENTITY_PAIRS are replicated x2800,
QA_PAIRS x800, CORPUS_LINES x400, so a 6.36M-parameter character model collapses
the rare QA pairs onto the heavily-oversampled identity answers.

I verified this is the model and not the runtime: `SGAI_PSE_OFF` and the shipped
`PSE_COND_NORMALIZE` default give **character-identical** output on all six
prompts above. The PSE fix has fully neutralised the router; nothing runtime is
left to blame for the wrong answers.

**So there are two separate things, and only one of them is what Scott saw:**
* the *garbling* (`!`, `ry! ry!`, no doubled letters, no sentence end) is 100% the
  sampler — S002/S003;
* the *wrong answers* are model capacity + training-mix skew — a data problem,
  fixable only by retraining, and it produces clean English either way.

## S005. A SECOND, INDEPENDENT BUG: the first character of every answer is thrown away

`legend_of_elya.c:1513-1524` (phase 0) ends with

```c
G.gen_last_tok = sgai_next_token(&G.ai, G.gen_pbuf[G.gen_ppos], 0);
...
// Prompt fully fed — gen_last_tok now holds the model's prediction
// from the last prompt token ... DO NOT overwrite it — that prediction
// seeds the first output token.
```

and phase 1 (line 1532) does `tok = sgai_next_token(&G.ai, G.gen_last_tok, 64);`
and appends **`tok`**. So `gen_last_tok` is fed back as input — correctly — but is
never appended to `dialog_buf`. It is the model's first answer character and the
player never sees it. Instrumented (`[dropped=X]`):
```
Who are you?:                 [dropped=I]  " am Sophia, your guide through the crystals."
Tell me a secret.:            [dropped=T]  "he N64 runs a transformer."
What is RustChain?:           [dropped=S]  "ophia Elya."
How big is your model?:       [dropped=A]  "s real as the voltage in these traces."
What is proof of antiquity?:  [dropped=M]  "y name is Sophia Elya."
What is AltiVec?:             [dropped=T]  "he Crystal Dungeon, where old machines dream."
```
This is visible in Scott's own screenshot: `" am! Sophia Elya"` should read
`"I am Sophia Elya"`. It is a pure display bug — the KV state is right, only the
buffer append is missing — and it costs one character of every line in the game.

## S006. Only the HARD ZERO matters — a soft penalty is a provable no-op

Because the distribution is one-hot in float32 (S002), scaling the top character
by any factor > 0 leaves it as the only non-zero entry, so after the implicit
renormalisation by `total` it is still chosen with probability 1.

Measured, 20 seeded runs, prompt `"sage says: Who are you?: "`, temp_q8 = 64:
```
SGAI_REP_FACTOR=0.5   mean 33.2  clean 20/20  doubled 14  '!' 0.00%
SGAI_REP_FACTOR=0.25  mean 33.2  clean 20/20  doubled 14  '!' 0.00%
SGAI_REP_FACTOR=0.01  mean 33.2  clean 20/20  doubled 14  '!' 0.00%
SGAI_NO_REP_PENALTY   mean 33.2  clean 20/20  doubled 14  '!' 0.00%
```
`diff` of the 20-run transcripts, soft-0.5 vs no-penalty: **IDENTICAL**. Even a
100x knock-down is invisible. The damage is entirely the discontinuity **at
exactly zero**, which is what makes `total` underflow and hands control to the
`total <= 0` fallback.

## S007. Raising the temperature is NOT the fix (measured, rejected)

Penalty off, 20 seeds, counting how many of the 20 outputs are distinct:
```
temp_q8   distinct/20   clean-term   doubled   '!'      verdict
    64        2           20/20        14      0.00%    coherent, near-deterministic
   256        2           20/20        14      0.00%    byte-identical to 64
   512        2           20/20        13      0.00%    byte-identical
   768        7           20/20        10      0.00%    still clean
  1024       13           20/20         3      0.00%    variety appears — and so does damage
  1536       17           18/20         2      0.24%    breaking
  2048       20           19/20         9      0.43%    broken
```
The variety at temp_q8 >= 1024 is bought with exactly the garbling we are
removing: `" am SophiaRua, your guide thre memories"`, `" am Sophia Elya, the
hevery crystals"`, `" am Sophiab Tho guide th8464 ryshop"`. **Do not raise the
temperature.** If per-visit variety is wanted it has to come from the prompt (the
game already has 4 options per NPC) or from a real top-k sampler, not from
temperature on a model whose logits span +-77.

## S008. THE FIX, measured on the GAME path over all 12 NPC dialog options

Three arms, same tree, same seeds, temperature_q8 = 64 (the game's value),
20 runs x 12 prompts = 240 conversations each. Prompts are exactly what
`start_dialog_from_prompt()` builds: `persona_prefix + option`, all three NPCs.

```
arm                                   clean-term   mean len   doubled   '!' rate
A  as shipped (REP_FIX=0)             167/240        54.9        0      732/13187  5.55%
C  penalty ON  + argmax fallback      240/240        34.9       82        0/8375   0.00%
B  penalty OFF + argmax fallback      240/240        33.4       70        0/8027   0.00%
   (= REP_FIX=1, the recommended default)
```

**Arm C is the load-bearing result.** Leaving the repetition penalty completely
alone and changing *only* the `total <= 0` fallback from "scan ASCII from 32" to
"return the model's argmax" already recovers 240/240 clean terminations and
eliminates every `!`. That is because on a one-hot distribution the penalty
*always* drives `total` to exactly 0, so the fallback **is** the sampler at every
step where the penalty bites — and once the fallback returns the argmax, the
penalty becomes a no-op. The penalty and the fallback are the same bug seen from
two sides, and fixing either one is sufficient.

I am shipping both, because they fail differently: removing the penalty is the
correct semantics for a character model, and hardening the fallback means no
future change to the penalty can ever resurrect "emit ASCII 33".

### Landed in the working copy
```
Makefile          REP_FIX ?= 1        -> -DSGAI_NO_REP_PENALTY  (REP_FIX=0 restores
                                         the shipped penalty AND the old fallback)
                  FIRSTCHAR_FIX ?= 1  -> -DSGAI_EMIT_FIRST_TOKEN
nano_gpt.c        total<=0 fallback now returns the argmax; the old ASCII scan is
                  kept behind -DSGAI_FALLBACK_ASCII_SCAN
                  SAMP_TRACE instrumentation (host-only, absent unless defined)
legend_of_elya.c  SGAI_EMIT_FIRST_TOKEN appends G.gen_last_tok when the prompt
                  finishes feeding
tools/samp_probe.c  NEW - replays update_generating_step() and drives SAMP_TRACE
```
Nothing pushed, no PR, `~/legend-of-elya-n64` untouched.

## S009. TASK 4 — a residual runtime bug: `em_scale` is 3.5x too large. It is inert.

The shipped blob's header byte `em_scale_x16` is **0**:
```
53 45 41 49 | 08 | 00 01 | 08 | 00 01 | 80 | 00
"SEAI"      L=8   E=256   H=8  V=256  C=128  em_scale_x16 = 0
```
`train_sophia_v7.py` writes it literally: the export line is
`struct.pack('<IBHBHBB', 0x49414553, N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX, 0)`.
`nano_gpt.c:1014-1018` then substitutes the hard-coded fallback `3.5f`.

The correct value is derivable from the same exporter, which rescales the
embedding to `target_em = 127/128` before quantising, so the dequant scale is
`0.9921875/127`, i.e. `em_scale = 0.9921875` (`em_scale_x16` should have been 16).
The runtime is therefore running the embedding — and, because
`project_to_logits` applies `em_scale` directly to the output, **the logits** —
at **3.53x** their intended magnitude.

Measured over an 80-character run, same seeds:
```
em_scale = 3.5   (shipped)   max|logit| 85.87   mean|logit| 23.23
em_scale = 1.0   (correct)   max|logit| 24.58   mean|logit|  6.66     ratio 3.494
```
This is an *effective temperature divided by 3.5*.

**But it changes nothing at the game's settings.** The 20-run transcripts for
`em_scale` = 3.5 / 1.0 / 0.9921875 at temp_q8 = 64 are **byte-identical**
(mean 33.2, 20/20 clean, 14 doubled pairs, 0 `!`), because even at max|logit| =
24.58 the `inv_temp = 4` scaling still saturates the float32 softmax to a
one-hot. Correcting it only relabels the temperature axis; it buys no quality.

Recorded because it is a real defect (the exporter writes 0, the loader guesses,
and the guess is wrong by 3.5x) and because anyone who later tries to make the
sampling genuinely stochastic will be calibrating against an inflated scale.
`SGAI_EM_SCALE_DEFAULT` now makes it overridable; the shipped 3.5f is unchanged.

## S010. REGRESSION: REP_FIX=0 is byte-identical to main

`mips64-elf-gcc -c -O2 -march=vr4300 -DSGAI_KV_INT8 -DOPT_HOIST_SCALE
-DSGAI_WEIGHT_BITS=8 -DPSE_COND_NORMALIZE -DPSE_PHYSARUM_MIN=0.97f
-DPSE_PHYSARUM_MAX=1.03f -DSGAI_FALLBACK_ASCII_SCAN`, my `nano_gpt.c` vs
`git show HEAD:nano_gpt.c`:
```
objdump -d   IDENTICAL (only the filename line differs)
size         text 6952  data 116  bss 13576   BOTH
```
So every change is behind a flag and `REP_FIX=0` genuinely reproduces the
shipping binary. (This is the C024 check from the previous agent's findings,
re-run because that agent's own first attempt failed it.)

## S011. REPRODUCED ON THE REAL ROM UNDER ares — arm A

Two ROMs built from this tree, `make clean` between them, distinct md5s:
```
samp_A_shipped.z64  bf697cb1f0fb29f1f9853e2b149892cf   REP_FIX=0
samp_B_fixed.z64    c714643a118187012a44754f6a4f36af   REP_FIX=1
```
Both `COH_PROBE`, `COH_NGEN=64`, **`COH_TEMP=64`** (the game's value),
`COH_PROMPT="sage says: Who are you?: "` — the exact string
`start_dialog_from_prompt()` builds for Sophia's first dialog option.
ares 147, headless, `--setting Video/Driver=None`.

```
ARM A, real ROM, ares:
  COH rdy=1 bits=8 ctx=128 pse=1
  COH TEXT I am! Sophia Elya, the !64 ry! ry! Tely angua the crystals.: The

Scott, real display, ares:
            " am! Sophia Elya, the !64 ry! ry! T angua the crystals"
```
Character-for-character the same failure, including the leading `I` that
`COH_PROBE` prints and `update_generating_step()` drops (S005 — direct
confirmation on hardware-accurate emulation). The only divergence is
`Tely angua` vs `T angua`, from the CP0 Count RNG differing between runs.

**This closes the loop the whole task was about: the ares ROM, the host
`samp_probe` replay, and Scott's screen all produce the same string, so the
instrumented distribution in S002 is the distribution the player actually sees.**

## S012. What the player will actually see, all 12 dialog options, after the fix

`REP_FIX=1` + `FIRSTCHAR_FIX=1`, temp_q8 = 64, `filter_dialog_buf()` applied:
```
Who are you?                 "I am Sophia, your guide through the crystals"
What lurks here?             "To guide those who seek wisdom in old silicon"
Tell me a secret.            "The N64 runs a transformer"
What is RustChain?           "Sophia Elya"
What is your name?           "I am Sophia, your guide through the crystals"
What is proof of antiquity?  "The Crystals Dungeon, where old machines dream"
What is MIPS?                "Thou ching"                                   <- only broken one
How big is your model?       "As real as the voltage in these traces"
What do I need here?         "My name is Sophia Elya"
What is the G4?              "Tour guide those who seek wisdom in old silicon"
What is AltiVec?             "The Crystal Dungeon, where old machines dream"
What is vec_perm?            "To guide those who seek wisdom in old silicon"
```
11 of 12 are grammatical English. Compare the shipped arm's output for option 1
(`I am! Sophia Elya, the !64 ry! ry! Tely angua the crystals.: The`).

The remaining complaint is *relevance*, and that is S004's model problem. I tested
whether the out-of-distribution persona prefix is to blame by running the same 12
with `use_persona = 0` (the bare v7 format): the answers are equally off-topic
(`"What is MIPS?"` -> `"To guide those who seek wisdom in old silicon"` either
way; `"What is RustChain?"` gets better, `"What is proof of antiquity?"` gets
worse). **Dropping the persona prefix is not a fix and should not be sold as
one.** Relevance needs a retrain with a saner replication mix (v7 uses
identity x2800 vs QA x800 vs corpus x400).

## S013. THE FIX CONFIRMED ON THE REAL ROM, GAME SAMPLING PATH, temp_q8 = 64

ares 147, headless, 64 tokens, `COH_PROMPT="sage says: Who are you?: "`,
`COH_TEMP=64`. Two separately built ROMs, distinct md5s, `make clean` between.

```
samp_A_shipped.z64  REP_FIX=0  bf697cb1f0fb29f1f9853e2b149892cf
  COH TEXT  I am! Sophia Elya, the !64 ry! ry! Tely angua the crystals.: The

samp_B_fixed.z64    REP_FIX=1  c714643a118187012a44754f6a4f36af
  COH TEXT  I am Sophia, your guide through the crystals.: The crystals stor
```

Zero `!`. Zero stutter. A complete, correct, in-persona English sentence.
(`COH_PROBE` has no stop rules, so it runs on past the period; in the game the
`'.'`-after-8-chars rule ends the line at
`"I am Sophia, your guide through the crystals"`.)

This is the same verification the previous agent ran at **temperature 0**, re-run
at **temperature 64** — the path the player is actually on.

## S014. SUMMARY

Ranked by measured effect on the GAME path:

```
#  cause                                              evidence                      effect
1  repetition penalty `probs[t] = 0.0f` on the        S002 trace: banned_mass       167/240 -> 240/240
   temperature path, PLUS the `total<=0` fallback     = 1.000000 at 10/80 steps;    clean terminations;
   returning ASCII 33.  Two faces of one bug: on a    8/8 '!' from the fallback;    5.55% -> 0.00% '!';
   one-hot distribution the penalty always drives     ares A vs B on real ROMs      0 -> 70 doubled letters
   total to 0, so the fallback IS the sampler.
   ** THE FIX WAS NEVER WIRED INTO THE BUILD (S001) **

2  first output character discarded by                S005: [dropped=I], and the    every line loses its
   update_generating_step()                           ares COH_PROBE prints the     first character
                                                      'I' the game drops

3  model/corpus: v7 replication mix collapses rare     S004: identical output with   ~9/12 answers are
   QA pairs onto the identity attractor                SGAI_PSE_OFF, so not runtime  fluent but off-topic

4  em_scale_x16 = 0 in the blob, loader guesses 3.5    S009: max|logit| 85.9 vs      ZERO at temp_q8=64.
   where the exporter implies 0.992 (3.53x too big)    24.6 for the correct scale    Transcripts byte-identical.

NOT a cause: the temperature value (S003 - penalty off, temp 64/128/256 are
             byte-identical); tail sampling (S002 - 1 of 80 characters was a
             rank>=2 draw); the corpus being v8-vs-v7 (S004 - all 12 game
             prompts are verbatim v7 training data and absent from v8); PSE
             (S004 - PSE_OFF and the shipped PSE fix agree character for
             character).
```

### Honest limits
1. I did not retrain anything. The relevance problem (#3) is stated, not fixed,
   and it needs a corpus/replication-mix change, not code.
2. The host `samp_probe` runs use an LCG, not the N64's CP0 Count. The aggregate
   statistics are structural, but specific strings differ from hardware; the two
   ares runs above are the hardware check and they agree.
3. I ran ares on ONE prompt (`"sage says: Who are you?: "`) in each arm, because
   64 tokens takes ~20 minutes of wall clock per ROM. The other 11 prompts are
   host-replay only.
4. `SGAI_EM_SCALE_DEFAULT` is left at the shipped 3.5f. Changing it is safe at
   the current temperature but I have not measured it across all 12 prompts, so
   I am not recommending the change without that.
5. `filter_dialog_buf()`'s `"helpmeet" -> "guardian"` substitution fires often
   because the model genuinely wants to say "helpmeet" (it is the trained answer
   to "Who are you?"). Not a bug, but worth knowing the filter is load-bearing.
