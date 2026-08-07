# N64 Sophia LM retrain — FINDINGS

Journal. Appended after every discrete result.

## 0. Setup / ground truth (established by reading, not running)

- Workspace: `$S/n64retrain` = `cp -r ~/legend-of-elya-n64`. Original untouched.
- GPU: RTX 4070 Laptop, 8188 MiB total, **3453 MiB already held by a llama-server**
  (PID 3694). ~4.7 GB usable. Budget batch accordingly.

### The 12 game prompts (located, not guessed)
`legend_of_elya.c` (repo ROOT — this is the build target, `src/` is a stale
4-layer copy) defines `NPC_DIALOG_OPTIONS[NPC_COUNT=3][DIALOG_OPTIONS=4]` at
line ~92. 3 NPCs x 4 options = **12**. All 12 are verbatim v7 keys:

| # | NPC | prompt | v7 category |
|---|-----|--------|-------------|
| 1 | Sophia | `Who are you?: ` | IDENTITY |
| 2 | Sophia | `What lurks here?: ` | QA |
| 3 | Sophia | `Tell me a secret.: ` | QA |
| 4 | Sophia | `What is RustChain?: ` | QA |
| 5 | Aldric | `What is your name?: ` | IDENTITY |
| 6 | Aldric | `What is proof of antiquity?: ` | QA |
| 7 | Aldric | `What is MIPS?: ` | QA |
| 8 | Aldric | `How big is your model?: ` | QA |
| 9 | Brunhild | `What do I need here?: ` | QA |
| 10 | Brunhild | `What is the G4?: ` | QA |
| 11 | Brunhild | `What is AltiVec?: ` | QA |
| 12 | Brunhild | `What is vec_perm?: ` | QA |

So the metric is **2 identity + 10 QA**. A collapse onto the identity attractor
scores at most 2/12 from the identity prompts plus luck.

### Architecture confusion resolved
- ROOT `nano_gpt.h`: 8 layers, 256 embed, 8 heads, ctx 128 -> 6,356,992 params.
  That matches the shipped blob header and the "6,356,992 params" in the brief.
- `src/nano_gpt.h`: 4/128/4/ctx64 — STALE, not the build target.
- `train_sophia_v7.py` itself declares 4/128/4/ctx64 and would export a small
  blob; the SHIPPED blob is the v7 *corpus* trained at the 8/256 shape (this is
  what `train_sophia_v9_qat.py --corpus v7` does at its default shape).
- Shipped `filesystem/sophia_weights.bin`: 6,750,220 B, magic `SEAI`,
  nL=8 E=256 H=8 vocab=256 ctx=128, **em_scale_x16 = 0** (the bug in task 5,
  confirmed by hexdump: byte 11 is 0x00).

### v7 replication mix, as actually counted (comments in v7 are wrong)
`IDENTITY_PAIRS` = 64 lines (header comment says 70), `QA_PAIRS` = 169 (says
180), `CORPUS_LINES` = 167 (says 220). Mass: 64x2800=179,200 vs 169x800=135,200
vs 167x400=66,800. Identity outweighs QA. Confirmed.

## 1. BASELINE — the shipped blob. And a correction to the premise.

`eval12.py` (written here) decodes a blob exactly as `nano_gpt.c` does and greedy-
decodes (argmax over printable ASCII 32..126, matching `sample_logits` at
`temperature_q8 == 0`). Newline is NOT in that band, so a generation runs on past
the answer into the next corpus line; correctness is therefore judged as
**"the generation starts with a trained v7 answer for that key"** (strict), and I
also report the looser human "does it answer the question" (lenient).

### Shipped `filesystem/sophia_weights.bin` — all 12, verbatim

```
Who are you?:                -> "I am Sophia Elya, the helpmeet. Who are you?: So"
What lurks here?:            -> "To guide those who seek wisdom in old silicon.: "
Tell me a secret.:           -> "The N64 runs a transformer. No one expected that"
What is RustChain?:          -> "rewards machines that prove they still work.: Te"
What is your name?:          -> "I am Sophia, your guide through the crystals.: T"
What is proof of antiquity?: -> "My name is Sophia Elya. Who are you?: I am Sophi"
What is MIPS?:               -> "To guide those who seek wisdom in old silicon.: "
How big is your model?:      -> "As real as the voltage in these traces.: The N64"
What do I need here?:        -> "I am Sophia Elya, the hese traces.: The N64 runs"
What is the G4?:             -> "My name is Sophia Elya.: The crystals store memo"
What is AltiVec?:            -> "The Crystal Dungeon, where old machines dream.: "
What is vec_perm?:           -> "To guide those who seek wisdom in old silicon.: "
```
**BASELINE = 0/12 strict, 4/12 lenient** (#1 identity, #3 a secret, #4 RustChain,
#5 a name). Identity keys: 0/33 strict. This reproduces
`docs/N64_SAMPLING_FINDINGS.md` line 216-230 ("roughly 3 of 12") exactly.

### 🔴 THE PREMISE IS WRONG: the shipped blob was never trained on v7.
Not one of those answers exists in `train_sophia_v7.py`. Nor in v5, v8, or
`train_sophia.py`. `"The Crystal Dungeon, where old machines dream."`,
`"To guide those who seek wisdom in old silicon."`, `"As real as the voltage in
these traces."` appear **nowhere in the repo except the doc that quotes the model
saying them**. Grep over the whole project:

    grep -rl "old machines dream" ~/legend-of-elya-n64  ->  docs/N64_SAMPLING_FINDINGS.md only

The shipped blob comes from a **lost corpus** that is v7-adjacent (right voice,
right topics, right cadence) but has different answer strings and, evidently, a
much stronger identity attractor. So the observed collapse is a property of THAT
blob, not of v7's 2800/800/400 replication mix.

### Proof: v7's UNMODIFIED mix already scores 12/12 at the shipped shape.
Two checkpoints already exist in the read-only `$S/n64qat/qat_out` that were
trained on the v7 corpus with the **untouched** 2800/800/400 weighting, at the
shipped 8L/256E/ctx128 shape, 9000 steps:

`v7_none.npz` (float) -> **12/12**, and `v7_ternary.npz` (ternary_bn QAT) ->
**12/12**. Every answer is the right answer to the right question, and they are
twelve DIFFERENT answers, not one attractor:

```
v7_none (float, mix 2800/800/400, i.e. the "buggy" mix)
Who are you?:                -> "I am Sophia, born of code."          HIT
What lurks here?:            -> "Skeletons patrol the catacombs."     HIT
Tell me a secret.:           -> "Bomb every wall, miss nothing."      HIT
What is RustChain?:          -> "Old silicon earns extra rewards."    HIT
What is your name?:          -> "Sophia Elya."                        HIT
What is proof of antiquity?: -> "Vintage silicon bonus."              HIT
What is MIPS?:               -> "CPU architecture in the N64."        HIT
How big is your model?:      -> "Eighty kilobytes of weights."        HIT
What do I need here?:        -> "Bombs crack crumbling walls."        HIT
What is the G4?:             -> "PowerPC G4 earns 2.5x RTC."          HIT
What is AltiVec?:            -> "SIMD instructions for G4 and G5."    HIT
What is vec_perm?:           -> "Shuffles vectors in one cycle."      HIT
```

**Conclusion so far: the fix is to retrain on v7 at all. The replication mix is
not the bug.** The sweep still runs (it is asked for and it is cheap) but the
12/12 metric is already saturated by the control, so the sweep will be judged on
a much wider all-keys metric plus a real held-out loss.

Residual quality issue, visible above: after the correct answer the greedy stream
degenerates (`".....: tter.: osss"`). The corpus separator is `\n` (byte 10) and
byte 10 is outside the sampler's printable band, so the model can never emit the
token it wants at end-of-answer. That is a stop-condition problem, tracked
separately from answer collapse.

## 2. The held-out split I built, and why

`train_sophia_v9_qat.build_v7_corpus` made "val" an independent reshuffle of the
SAME 400 lines. Train and val therefore draw ctx-128 windows from two
permutations of one string set: **v7 val == v7 fit by construction** and val loss
on v7 is a memorization readout, not a generalization signal. (Visible in the
existing runs: `v7_none` and `v7_ternary` both report final_val = 0.173.)

A random line holdout is not available: 400 distinct lines, and the game asks 12
specific questions whose answers must stay in training. So I built a
**paraphrase holdout** (`train_mix.split_v7`):

* QA/IDENTITY lines are grouped by key (`"What is the G4?: "`). For any key with
  **>= 3** distinct answers, the LAST answer (list order — deterministic, no RNG)
  moves to validation. Every key keeps >= 2 answers in training, so no game
  prompt loses its answer, and singleton keys (`What is MIPS?`, `How big is your
  model?`) are never touched.
* CORPUS_LINES have no key, so every 10th line is held out.

Result: **44 of 400 lines (11%) are strings the model never sees.**

| group | lines | keys | held out | in train |
|---|---|---|---|---|
| IDENTITY_PAIRS | 64 | 33 | 8 | 56 |
| QA_PAIRS | 169 | 94 | 20 | 149 |
| CORPUS_LINES | 167 | – | 16 | 151 |

The val corpus is built from the held-out lines ONLY. It is now a real
generalization probe, and it separates immediately: at step 300 of a smoke run,
fit = 1.178 while val = 2.266. Under the old v7 "split" those two numbers are the
same quantity.

Per the brief, per-prompt accuracy on the 12 game prompts is reported separately
from loss, and identity keys (33) and QA keys (94) are scored separately so a
gain on QA that costs identity is visible.

## 3. The ROM already has a clean stop — so trailing degeneration is invisible

`legend_of_elya.c:update_generating_step()` phase 1 sets `tok = 0` on `'\n'` and
on the first `'.'` once `gen_out_count >= 8`, and does not append it. So the
player sees the answer up to (not including) its terminating period, and the
greedy run-on garbage after it (`".....: tter.: osss"`) is never displayed.
`eval12.game_cut()` now reproduces that cut, and both the raw stream and the
displayed string are reported. This removes "post-answer degeneration" from the
list of things the retrain has to fix.

## 4. ROM harness works, and host == ROM on the shipped blob

Built from current main (`REP_FIX`/`FIRSTCHAR_FIX`/`PSE_FIX` all default on) with
a new `GAME12_PROBE` added to `legend_of_elya.c` in the workspace copy: it walks
`NPC_DIALOG_OPTIONS[3][4]` and prints each answer twice, once at temperature 0
(comparable to the numpy oracle) and once at temperature_q8 = 64 (the real
`update_generating_step` path). It runs inside `game_init()` like the existing
probes, because headless ares has no GPU and dies at the first rendered frame.

    make base EXTRA="-DGAME12_PROBE -DG12_NGEN=32"      N64_INST=$HOME/n64-toolchain/mips64-toolchain
    xvfb-run -a flatpak run dev.ares.ares --system "Nintendo 64" --no-file-prompt \
        --setting Video/Driver=None --setting Audio/Driver=None rom.z64   # ROM staged under $HOME

First lines off the real hardware model, shipped blob:
```
G12 rdy=1 bits=8 ctx=128 emx16=0
G12 0|Who are you?: |I am Sophia Elya, the helpmeet.
```
`emx16=0` is the em_scale bug, read off the loaded header on the target.
The answer is **character-identical to the host oracle**, so the host numbers in
this document are ROM-faithful.

## 5. `em_scale_x16` — the bug, quantified, and the fix

`train_sophia_v7.py:634` writes the field as a literal `0`:

    buf += struct.pack('<IBHBHBB', 0x49414553, N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX, 0)

...yet the very next block rescales the embedding to `target_em = 127.0/128.0 =
0.9921875` before quantizing. So the file implies `em_scale = 0.9922`
(`em_scale_x16 = 16`) and stores 0. `nano_gpt.c:1014-1019` reads
`em_scale = em_scale_x16 / 16.0f`, sees `< 0.01f`, and substitutes
`SGAI_EM_SCALE_DEFAULT 3.5f`. Every embedding lookup and every logit is
therefore **3.528x too large**. Confirmed on the target: `G12 rdy=1 bits=8
ctx=128 emx16=0`.

Measured on the host, shipped blob, greedy, 40 tokens, em_scale 3.5 vs 0.9922:
**11 of 12 generations are character-identical** — the softmax is saturated, as
the brief says — but it is **not fully inert**:

```
What is the G4?:
  em_scale 3.5    -> "My name is Sophia Elya.: The crystals st"
  em_scale 0.9922 -> "Are old machines learn new tricks.: The "
```
Same for 1.375 and 1.5: 11/12 identical, same single prompt flipping. So the
wrong scale is changing the answer on 1 of 12 prompts even today.

Also of note: the shipped embedding codes span [-110, 127] with exactly 1 of
65,536 entries at saturation, i.e. the int8 range is used properly — the header
field is simply not written.

**Fix**: `train_mix.export_blob` computes
`x16 = round(min(max|emb|, 255/16) * 16)` and writes it; `tools/qat_npz_to_seq.py`
copies it into the SEQn header. Nonzero, so the 3.5 fallback never fires. The
value produced for this shape has been 22-24 in every run so far.

## 6. Separating the TWO causes in the baseline: header bug vs. corpus

The coordinator is right that the baseline mixes two independent defects, so
here are the three arms, all greedy, all 40 tokens, all rendered through
`game_cut()` (what the player sees).

**Arm A — shipped blob, `em_scale_x16 = 0` -> loader substitutes 3.5f (today's ROM)**
```
Who are you?:                -> "I am Sophia Elya, the helpmeet"
What lurks here?:            -> "To guide those who seek wisdom in old si"
Tell me a secret.:           -> "The N64 runs a transformer"
What is RustChain?:          -> "rewards machines that prove they still w"
What is your name?:          -> "I am Sophia, your guide through the crys"
What is proof of antiquity?: -> "My name is Sophia Elya"
What is MIPS?:               -> "To guide those who seek wisdom in old si"
How big is your model?:      -> "As real as the voltage in these traces"
What do I need here?:        -> "I am Sophia Elya, the hese traces"
What is the G4?:             -> "My name is Sophia Elya"
What is AltiVec?:            -> "The Crystal Dungeon, where old machines "
What is vec_perm?:           -> "To guide those who seek wisdom in old si"
```
**STRICT 0/12. LENIENT 4/12** (#1, #3, #4, #5).

**Arm B — same blob, header bug fixed (`em_scale = 0.9922`), NO retrain**
Eleven of twelve are character-identical to Arm A. Exactly one flips:
```
What is the G4?:  3.5    -> "My name is Sophia Elya"           (identity attractor)
                  0.9922 -> "Are old machines learn new tricks"
```
**STRICT 0/12. LENIENT 4/12 — unchanged.**

I have to report this against the framing I was given. The flip *does* move that
prompt off the identity attractor and onto a line about old machines, which is
topically nearer "What is the G4?". But `"Are old machines learn new tricks"` is
ungrammatical and is **not an answer to the question**, so it does not convert to
a hit under either criterion. Fixing the header alone buys **zero answers**.

That is the useful result, and it is the one the decomposition was asked for:
**the header bug is not a material contributor to the baseline score.** Whatever
the retrain gains, it is not quietly collecting credit for the `em_scale` fix.
The `em_scale` fix is still worth shipping — it is a real correctness defect,
it is measurably not a no-op (1/12 outputs change), and it is fixed here — but it
is not a cause of the answer collapse.

**Arm C — retrained on v7, correct `em_scale`**: see §1 and §7. 12/12 strict.

Confirmed on the target, not just the host:
```
shipped int8 ROM : G12 rdy=1 bits=8 ctx=128 emx16=0    <- fallback fires
v7 ternary ROM   : G12 rdy=1 bits=2 ctx=128 emx16=22   <- correct value, no fallback
```

## 7. M0 — the UNCHANGED v7 mix, retrained. And the checkpoint trap, measured.

`M0_v7base`, reps 2800/800/400 (v7 exactly as written), float, 9000 steps, with
the §2 holdout:

| select | GAME12 | IDENTITY | QA | val |
|---|---|---|---|---|
| **final** | **12/12** | **32/33** | **94/94** | 5.33 |
| best_val@1500 | 11/12 | 31/33 | 90/94 | 3.77 |

The single identity "miss" at `final` is `"Sophia Elya?: "` -> `"That is me, your
guide."`, which is the trained answer for the neighbouring key `"Who is Sophia
Elya?: "`. It is a correct, in-voice answer; it just is not that key's own
string. Nothing is collapsed.

### The checkpoint trap is real and it is BACKWARDS from loss
Held-out val and the metric that matters move in **opposite directions** for the
whole run:

| step | val (real held-out) | GAME12 |
|---|---|---|
| 1500 | **3.77 (best)** | 11 |
| 3000 | 4.01 | 12 |
| 4500 | 4.29 | 12 |
| 6000 | 4.74 | 12 |
| 7500 | 5.06 | 12 |
| 9000 | 5.33 (worst) | **12** |

Picking `best_val` would have shipped the step-1500 weights, which emit
`"Shufles vectors in one cycle."` for `What is vec_perm?` — a misspelling of the
trained answer — plus five more misses. **`select: "final"` is correct here, and
it is chosen on the 12/12 metric, not on loss.** This is a corpus the game wants
memorized; rising held-out loss is the price of that and is not a defect.

## 8. THE SWEEP CANNOT IMPROVE ON THIS

M0 already scores 94/94 on every distinct QA key and 32/33 on every identity key.
There is no headroom. The remaining mixes are run because they were asked for and
they are cheap, but the conclusion is already fixed: **the 2800/800/400
replication imbalance is not the cause of the shipped ROM's answer collapse.**

## 9. REAL ROM UNDER ARES — 12/12, both sampling arms

ROM: current main + `GAME12_PROBE`, `make base SGAI_BITS=2`, blob =
`v7_ternary_baselinemix.seq2` (2,031,628 B, ternary/2-bit, SEQ2, x16 = 22).
Run headless under `flatpak run dev.ares.ares`, staged under `$HOME`, SIGINT.

```
G12 rdy=1 bits=2 ctx=128 emx16=22

temperature 0 (greedy)                    temperature_q8 = 64 (real game path)
Who are you?:                -> "I am Sophia, born of code"          HIT   (identical)
What lurks here?:            -> "Iron knuckles guard the way"        HIT   (identical)
Tell me a secret.:           -> "Bomb every wall, miss nothing"      HIT   (identical)
What is RustChain?:          -> "Old silicon earns extra rewards"    HIT   (identical)
What is your name?:          -> "Sophia Elya"                        HIT   (identical)
What is proof of antiquity?: -> "Vintage silicon bonus"              HIT   (identical)
What is MIPS?:               -> "CPU architecture in the N64"        HIT   (identical)
How big is your model?:      -> "Eighty kilobytes of weights"        HIT   (identical)
What do I need here?:        -> "Bombs crack crumbling walls"        HIT   (identical)
What is the G4?:             -> "PowerPC G4 earns 2"                 HIT   (identical)
What is AltiVec?:            -> "SIMD instructions for G4 and G5"    HIT   (identical)
What is vec_perm?:           -> "Shuffles vectors in one cycle"      HIT   (identical)

ROM SCORE (greedy)     12/12
ROM SCORE (temp 64)    12/12
```
The two arms are **character-identical on all 12** — the T = 0.25 softmax is
saturated, exactly as `docs/N64_SAMPLING_FINDINGS.md` S002 measured. And against
the numpy oracle for the same blob: **12/12 character-identical**, so the ROM and
the host are computing the same function on the real target.

Compare the same twelve on the shipped ROM: `G12 rdy=1 bits=8 ctx=128 emx16=0`,
0/12 strict.

### One residual blemish, and it is NOT the model
`What is the G4?` is answered correctly — the raw stream is
`"PowerPC G4 earns 2.5x RTC."` — but the player sees `"PowerPC G4 earns 2"`.
`update_generating_step()` terminates on the first `'.'` after 8 output chars, and
the `.` in `"2.5x"` is at offset 18. The stop rule cannot tell a decimal point
from a full stop. This is a runtime bug, not answer collapse, and it predates
this work. (A data-side workaround exists — v7 has two decimal-free answers for
that key, `"AltiVec SIMD on the G4 chip."` and `"Classic PowerPC from Apple."` —
but the honest fix is in the stop rule.)

## 10. Shipped ROM under ares — 0/12, and a ROM/host divergence worth naming

Same probe, same current-main build, shipped `SEAI` int8 blob:
```
G12 rdy=1 bits=8 ctx=128 emx16=0
Who are you?:                -> "I am Sophia Elya, the helpmeet"
What lurks here?:            -> "To guide those who seek wisdom i"
Tell me a secret.:           -> "The N64 runs a transformer"
What is RustChain?:          -> "rewards machines that prove they"
What is your name?:          -> "I am Sophia Elya, the helpmeet"     <- same line as Q1
What is proof of antiquity?: -> "My name is Sophia Elya"
What is MIPS?:               -> "To guide those who seek wisdom i"
How big is your model?:      -> "As real as the voltage in these "
What do I need here?:        -> "I am Sophia Elya, the hese trace"
What is the G4?:             -> "My name is Sophia Elya"
What is AltiVec?:            -> "The Crystal Dungeon, where old m"
What is vec_perm?:           -> "To guide those who seek wisdom i"
ROM SCORE greedy 0/12   ROM SCORE temp-64 0/12   (arms character-identical)
```
On the real target the collapse is if anything worse than on the host: `What is
your name?` returns the *same string* as `Who are you?`, and three prompts share
`"To guide those who seek wisdom in old silicon."`.

**ROM vs numpy oracle, shipped blob: 10/12 character-identical.** The two that
diverge (`What is your name?`, `What is proof of antiquity?`) are explained: the
ROM build has `PSE_FIX=1` (`PSE_COND_NORMALIZE`, band +-0.03) while the oracle
pins Physarum conductance at exactly 1.0, i.e. models `SGAI_PSE=0`. That residual
+-3% per-head gain is enough to flip an argmax on this blob.
The retrained ternary blob shows **12/12** ROM-vs-oracle agreement under the same
build, so the divergence is a property of the shipped weights' near-ties, not of
the kernel.

## 11. The replication-mix sweep (float control, 9000 steps, §2 holdout)

Line counts after the holdout: IDENTITY 56, QA 149, CORPUS 151. "mass" = lines x
reps = training line-instances per category.

| tag | reps (ID,QA,CORP) | mass ID / QA / CORP | sel | GAME12 | IDENTITY | QA |
|---|---|---|---|---|---|---|
| M0_v7base | 2800, 800, 400 | 156,800 / 119,200 / 60,400 | final | **12/12** | **32/33** | **94/94** |
| M0_v7base | " | " | best_val | 11/12 | 31/33 | 90/94 |
| M1_flat | 800, 800, 400 | 44,800 / 119,200 / 60,400 | final | **12/12** | 31/33 | **94/94** |
| M1_flat | " | " | best_val | 12/12 | 31/33 | 92/94 |
| M2_catmass | 2129, 800, 789 | 119,224 / 119,200 / 119,131 | final | **12/12** | 31/33 | **94/94** |
| M2_catmass | " | " | best_val | 12/12 | 30/33 | 85/94 |
| M3_qaheavy | 400, 1600, 400 | 22,400 / 238,400 / 60,400 | (running) | | | |
| M4_perline | 800, 800, 800 | 44,800 / 119,200 / 120,800 | (running) | | | |

M1 brings identity down to the QA per-line rate (3.5x -> 1x). M2 equalises
CATEGORY mass rather than per-line rate, as the brief suggested. M3 inverts the
imbalance. M4 makes every line equally frequent.

**Every mix scores 12/12 and 94/94.** Rebalancing changes nothing, because
nothing was broken: a 6.36M-parameter model trained on 356 distinct lines
memorizes all of them at any of these ratios. The only differentiator is
IDENTITY, where the ORIGINAL v7 weighting is (marginally) best at 32/33 — which
is what over-weighting identity is for.

Across every arm, `select: "final"` >= `select: "best_val"`. Choosing on loss
never wins and sometimes costs 9 QA keys (M2).

### Sweep complete — winner: M4_perline (800, 800, 800)

| tag | reps | sel | GAME12 | IDENTITY | QA | val | fit |
|---|---|---|---|---|---|---|---|
| M0_v7base | 2800, 800, 400 | final | 12/12 | 32/33 | 94/94 | 5.326 | 0.170 |
| M0_v7base | " | best | 11/12 | 31/33 | 90/94 | 3.767 | 0.170 |
| M1_flat | 800, 800, 400 | final | 12/12 | 31/33 | 94/94 | 5.369 | 0.179 |
| M1_flat | " | best | 12/12 | 31/33 | 92/94 | 3.832 | 0.179 |
| M2_catmass | 2129, 800, 789 | final | 12/12 | 31/33 | 94/94 | 5.228 | 0.181 |
| M2_catmass | " | best | 12/12 | 30/33 | 85/94 | 3.921 | 0.181 |
| M3_qaheavy | 400, 1600, 400 | final | 12/12 | 31/33 | 94/94 | 5.375 | 0.170 |
| M3_qaheavy | " | best | 12/12 | **28/33** | 90/94 | 4.016 | 0.170 |
| **M4_perline** | **800, 800, 800** | **final** | **12/12** | **33/33** | **94/94** | 5.477 | 0.188 |
| M4_perline | " | best | 12/12 | 31/33 | 93/94 | 3.722 | 0.188 |

**Winner: M4_perline, reps 800/800/800 — the only arm that is perfect on all 139
keys (12/12 game, 33/33 identity, 94/94 QA).** "800/800/800" means every distinct
line, whatever category, appears equally often; it removes the imbalance outright
rather than re-tuning it, which is the cleanest version of the intervention the
brief proposed.

The honest caveat: the margin over doing nothing is **one identity key out of
33**, and every mix including v7's own already scores 12/12 on the metric that
matters. The rebalance is a tidy-up, not the fix. The fix is retraining on v7 at
all.

The identity-vs-QA trade the brief asked me to watch for **is visible, at the
`best_val` checkpoint**: M3_qaheavy (QA mass 238,400 vs identity 22,400) drops to
28/33 identity there, the worst in the sweep, while holding 90/94 QA. At `final`
it recovers to 31/33. So the trade is real but training through it dissolves it.
Reporting identity separately was worth doing.

Every arm: `final` >= `best_val`. Loss-based checkpoint selection never wins.

## 12. FINAL QAT — ternary (target), winning mix 800/800/800, 12000 steps

```
[final_ternary_bn] select=final: GAME12=12/12  IDENTITY=32/33  QA=94/94
[final_ternary_bn] select=best : GAME12=12/12  IDENTITY=32/33  QA=87/94
[final_ternary_bn] DONE final_val=5.1202 final_fit=0.1883 best_val=3.7606@1500
                        blob=2,031,628 B in 1040s
```
`select: "final"` chosen — it ties on GAME12 and IDENTITY and is +7 QA keys over
`best_val`. Chosen on the key metric, not on loss (best_val is 3.76 vs final 5.12).

Ternary QAT costs exactly **one identity key** vs the float control (32/33 vs
M4_perline's 33/33) and nothing at all on GAME12 or QA.

### Export — the FIXED exporter, and the max|dW| proof
```
$ python3 tools/qat_npz_to_seq.py qat_out/final_ternary_bn.npz \
      blobs/final_ternary_bn.seq2 --meta qat_out/final_ternary_bn_meta.json
wrote blobs/final_ternary_bn.seq2  2031628 B  bits=2
      {'n_layers': 8, 'n_embed': 256, 'n_heads': 8, 'ctx': 128, 'x16': 22}

$ python3 verify_export.py qat_out/final_ternary_bn.npz blobs/final_ternary_bn.seq2
  geometry npz: nL=8 E=256 H=8 ctx=128 x16=22
  geometry seq: nL=8 E=256 H=8 ctx=128 x16=22
  emb                     max|dW| = 0
  all 8 layers x 6 tensors + emb: WORST max|dW| = 0
VERDICT: IDENTICAL (max|dW| = 0)
```
`verify_export.py` decodes the blob with the ROM's rules (MSB-first, two's
complement, one f16 scale per 32-weight block) and the npz with the oracle's, and
compares all 49 arrays. **max|dW| = 0 exactly, everywhere.**

The QAT `.bin` writer was NOT used: it emits "SEAI" for every scheme (so the
loader would decode a 2-bit blob as int8), packs LSB-first while the runtime reads
MSB-first, and stores offset binary while the runtime decodes two's complement.

* Blob format: **SEQ2**, 2,031,628 B, `em_scale_x16 = 22` (nonzero -> the 3.5f
  fallback in `nano_gpt.c:1018` never fires).
* Build: `make base SGAI_BITS=2` -> bss 2,641,564 (vs 7,360,156 for int8), ROM
  2,195,456 B.

### Host generations, final ternary blob, all 12 (as the player sees them)
```
Who are you?:                -> "Sophia Elya of Elyan Labs"        HIT
What lurks here?:            -> "Dark spirits haunt these halls"   HIT
Tell me a secret.:           -> "Gold hides beneath the tiles"     HIT
What is RustChain?:          -> "A blockchain for vintage chips"   HIT
What is your name?:          -> "Sophia Elya is my name"           HIT
What is proof of antiquity?: -> "Old hardware earns more"          HIT
What is MIPS?:               -> "CPU architecture in the N64"      HIT
How big is your model?:      -> "Eighty kilobytes of weights"      HIT
What do I need here?:        -> "The lens of truth reveals"        HIT
What is the G4?:             -> "AltiVec SIMD on the G4 chip"      HIT
What is AltiVec?:            -> "SIMD instructions for G4 and G5"  HIT
What is vec_perm?:           -> "POWER8 secret weapon for AI"      HIT
SCORE 12/12
```
Bonus: this model picks `"AltiVec SIMD on the G4 chip."` for the G4 key instead of
`"PowerPC G4 earns 2.5x RTC."`, so the decimal-point truncation blemish from §9
does not occur here.

## 13. int8 control arm — same mix, same steps

```
[final_int8] select=final: GAME12=12/12  IDENTITY=32/33  QA=94/94
[final_int8] select=best : GAME12=12/12  IDENTITY=31/33  QA=91/94
[final_int8] DONE final_val=5.6899 final_fit=0.1874 best_val=3.7750@1500
                  blob=6,750,220 B in 1057s
verify_export.py -> WORST max|dW| = 0   VERDICT: IDENTICAL
blobs/final_int8.seq8   6,750,220 B  bits=8  em_scale_x16 = 24
host eval: 12/12 game, 32/33 identity
```
**Ternary is indistinguishable from int8 on every metric measured** (12/12,
32/33, 94/94 both), at **30.1% of the size** (2,031,628 vs 6,750,220 B) and with
the -53.6% CPU / -76.6% RSP the Makefile already documents. Ternary is the ship
candidate; int8 exists only as the control.

| arm | blob | bytes | x16 | max\|dW\| | GAME12 | IDENTITY | QA |
|---|---|---|---|---|---|---|---|
| **ternary (ship)** | SEQ2 | 2,031,628 | 22 | **0** | **12/12** | 32/33 | 94/94 |
| int8 (control) | SEQ8 | 6,750,220 | 24 | **0** | 12/12 | 32/33 | 94/94 |
| float control (M4) | – | – | – | – | 12/12 | 33/33 | 94/94 |
| **shipped (baseline)** | SEAI | 6,750,220 | **0** | – | **0/12** | 0/33 | – |

## 14. FINAL — the shipped-candidate ternary blob on the REAL ROM under ares

ROM: current main (`REP_FIX`/`FIRSTCHAR_FIX`/`PSE_FIX` on) + `GAME12_PROBE`,
`make base SGAI_BITS=2`, blob `blobs/final_ternary_bn.seq2`.

```
G12 rdy=1 bits=2 ctx=128 emx16=22

                                greedy (temp 0)                    temp_q8=64 (real game path)
Who are you?:                -> "Sophia Elya of Elyan Labs"        same
What lurks here?:            -> "Dark spirits haunt these halls"   same
Tell me a secret.:           -> "Gold hides beneath the tiles"     same
What is RustChain?:          -> "A blockchain for vintage chips"   same
What is your name?:          -> "Sophia Elya is my name"           "Sophia Elya"
What is proof of antiquity?: -> "Old hardware earns more"          same
What is MIPS?:               -> "CPU architecture in the N64"      same
How big is your model?:      -> "Eighty kilobytes of weights"      same
What do I need here?:        -> "The lens of truth reveals"        same
What is the G4?:             -> "AltiVec SIMD on the G4 chip"      same
What is AltiVec?:            -> "SIMD instructions for G4 and G5"  same
What is vec_perm?:           -> "POWER8 secret weapon for AI"      same

ROM SCORE (greedy)  12/12       ROM SCORE (temp 64)  12/12
ROM vs numpy oracle: 12/12 character-identical
```
The one temp-64 difference (`"Sophia Elya is my name"` -> `"Sophia Elya"`) is a
real tail draw that lands on a *different correct answer* for the same key. Still
a hit.

## 15. Summary

| | GAME12 | IDENTITY | blob | em_scale_x16 |
|---|---|---|---|---|
| shipped, on the ROM | **0/12** | 0/33 | SEAI 6,750,220 | **0** (3.5 fallback) |
| shipped + em_scale fix, no retrain | 0/12 strict, 4/12 lenient (unchanged) | – | – | – |
| **retrained ternary, on the ROM** | **12/12** | 32/33 | **SEQ2 2,031,628** | **22** |

Cause of the collapse: **the shipped blob was trained on a corpus that is not in
the repository.** Retraining on `train_sophia_v7.py` fixes it completely. The
2800/800/400 replication imbalance was not the cause — v7's own mix scores 12/12
— though rebalancing to 800/800/800 is a small, free improvement (33/33 identity
in float) and is what the shipped blob uses.

Deliverables in this workspace (original repo untouched, verified by md5):
```
blobs/final_ternary_bn.seq2   2,031,628 B  SHIP CANDIDATE (also in filesystem/)
blobs/final_int8.seq8         6,750,220 B  control
blobs/ORIG_shipped_seai.bin   6,750,220 B  the baseline, preserved
qat_out/final_*.{npz,pt,_meta.json}        checkpoints + full per-key transcripts
mix_out/M*_meta.json                       the 5-mix sweep, per-key transcripts
eval12.py verify_export.py score_rom.py compare_rom_host.py train_mix.py
legend_of_elya.c: +GAME12_PROBE
```
