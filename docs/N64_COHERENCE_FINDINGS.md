# N64 LLM COHERENCE REGRESSION — session log (agent: n64coh)
Started 2026-08-05. Working copy $S/n64coh (upstream never touched).

Observation: shipping ROM, game loop, real display:
    Sophia Elya                    0.7 t/s   2553ms
    I know the langua of!Ther Qo! Qohe old mage age i_
coherent -> garbage -> partial recovery.

## C001: prior art inventoried
$S/n64rom/FINDINGS.md is decisive prior work:
 - F-T008: tools/host_eval.c compiles the REAL nano_gpt.c natively. With
   SGAI_PSE_OFF host C == numpy oracle 32/32 tokens on two blobs. With PSE live
   it does not (".: Hold hard pat" vs ". How can I help").
 - F-T017: SGAI_PSE_OFF costs the default build zero instructions.
 - F-T021.1: two QAT exports shipped the DIVERGED checkpoint (ternary_bn
   final_val 6.278 vs best 3.345; big_ternary 7.305 vs 3.439). Only v7_ternary
   converged. -> concrete candidate for "the model is just weak".
So the oracle-vs-runtime partition can run on the HOST in seconds; ares confirms.

## C002: which blob does Scott actually see?
filesystem/sophia_weights.bin = 6,750,220 B, magic "SEAI" -> bits=8 (int8),
8 layers x 256 embed. That is the shipping blob and the one on screen.

## C003: the KV cache DOES slide correctly — my first hypothesis was wrong, recorded anyway
`sgai_next_token` step 6 (nano_gpt.c:849-868) is a real sliding window: when
`pos == SGAI_CTX-1` it memcpys `k[l][t] <- k[l][t+1]` for t in 0..CTX-2 (and the
int8 scales too). Slot CTX-1 is left holding the just-written token, and the next
token overwrites it, so ordering is preserved and only the OLDEST entry is
dropped. Not a freeze. `SGAI_CTX` is the header default **128** (the Makefile
never overrides it; int8 KV is what bought ctx=128 back — F37's ctx=32 was the
float32-KV era). So a 48-char on-screen degradation is NOT the context window.
Also worth stating: this model has **no positional encoding** at all (embed_lookup
is token-only, no RoPE), so the shift is positionally harmless — but it does
memcpy ~520 KB per token once the window is full.

## C004: **REPRODUCED ON THE HOST IN ONE SHOT — and PSE is visibly the dominant term**
`tools/host_eval.c` (n64rom's, compiles the REAL nano_gpt.c) built four ways,
shipped blob `filesystem/sophia_weights.bin` (SEAI/int8, 8L x 256d), 150 tokens:
```
arm                     prompt "Who are you?"
he_ship  (PSE, KV8)   ": I am Sophia Elya, born of Elyan a. When cace other purpo puse?: To chid me a an am ilinge itnge ow o guide ther..."
he_f32   (PSE, KVf32) ": I am Sophia Elya, born of Elyan a. When Labs work where memocccre-~ar mem is workeccccccccccceted m wan Elyage..."
he_pseoff(off, KV8)   ": I am Sophia Elya, the helpmeet. Who are you?: Sophia Elya, born of Elyan Labs. Who are you?: My name is Sophia Elya. Myou knownguing which old machi"
he_f32_pseoff(off,f32)": I am Sophia Elya, the helpmeet. Who are you?: Sophia Elya, born of Elyan Labs. Who are you?: My name is Sophia Elya. Myou knownge where old machines"
```
prompt "Elya":
```
he_ship      "n Labs.: A work where old machines learn new tral wame se wid mewinses wiches weron w whe where wolt w w we wame ilowa wa wagititnmes woche.: atead Wo"
he_pseoff    "n Labs.: A workshop where old machines learn new tricks.: The leanes where memories of every computation.: RustChaiored whereten you they stowere dhe "
he_f32_pseoff"n Labs.: A workshop where old machines learn new tricks. The crystals store memories of every computation.: Rustion winating. Rusting where ottthengeo"
```
PSE ON collapses at ~char 30-45. PSE OFF stays clean to ~char 110-140.
Scott's screen text ("I know the langua of!Ther Qo! Qohe old mage age i") breaks
at ~char 17 and is full of stray capitals and `!` mid-word — the signature of a
random-dimension additive kick, and the real ROM uses TRUE CP0 entropy where this
host run uses the deterministic BENCH_DET_PSE LCG.
NOTE this is all with `BENCH_DET_PSE`; the real ROM is noisier still.

## C005: PARTITION — divergence positions, shipped blob, 200 tokens, 3 prompts
Reference arm = PSE fully OFF. `firstdiff` = first token index that differs.
```
arm                          "Who are you?"   "Elya"   "I know the language of"
PSE off (ref)                     --            --              --
route only  (no burst)             20           15              27
burst only  (no router)          NONE         NONE            NONE
full PSE                           20           15              27
full PSE, OLD 0x7F mask            20           15              27
burst only, OLD 0x7F mask        NONE         NONE            NONE
```
Two results, both hard:
1. **Burst entropy injection is behaviourally INERT.** 600 tokens, 3 prompts,
   byte-identical to PSE-off. 2% of RMS spread over 8 of 256 dims never flips a
   greedy argmax. The `& 0x7F` bug was real but had **no observable effect**, and
   fixing it made things neither better nor worse — `full PSE` and
   `full PSE, OLD mask` are byte-identical too.
2. **The Physarum router alone reproduces 100 % of the PSE damage**, byte-for-byte
   identical to full PSE at every one of 600 positions.
So within PSE the culprit is `pse_physarum_update` / the `acc * cond` multiply,
not the entropy burst.

## C006: PARTITION — int8 KV vs float32 KV (equal ctx=128, PSE OFF both sides)
```
prompt                    firstdiff(int8 KV vs f32 KV)
"Who are you?"                    130
"Elya"                             56
"I know the language of"           91
```
Real, and it compounds as predicted — but it lands 3-8x LATER than the router,
and by those positions BOTH arms are already producing broken English
("Myou knownge..." at ~128 in the f32 reference). int8 KV is a genuine second-
order error source, not the cause of what Scott saw.

## C007: the reference arm degrades too — at ~char 120-140
PSE off + float32 KV (== the numpy oracle function per n64rom F-T008) still goes
": I am Sophia Elya, the helpmeet. Who are you?: Sophia Elya, born of Elyan Labs.
Who are you?: My name is Sophia Elya. Myou knownge where old machines learn new
tricks.: The crystals store mememork. w". Clean for ~120 chars, then decays.
That is the model/corpus, not the runtime — but it is 6-8x further out than the
failure on screen.

## C008: conductance TRACE — the mechanism, watched directly (tools/pse_trace.c)
Prompt "Elya", shipped blob, per-token min/mean/max of the 64 conductances
(8 layers x 8 heads) and how many are pinned at the clamps:
```
pos char  cmin  cmean cmax   nAt1.5  nAt0.5
  0 n    0.972 1.027 1.150    0/64    0/64     <- model is still ITSELF here
  7 :    0.903 1.095 1.500    3/64    0/64
 15 (sp) 0.824 1.156 1.500    8/64    0/64     <- free-run divergence starts HERE
 20 e    0.776 1.179 1.500   18/64    0/64
 31 n    0.669 1.197 1.500   25/64    0/64
 49 (sp) 0.500 1.216 1.500   29/64    1/64
 59 i    0.500 1.214 1.500   32/64    1/64
```
`attn_out[d] = acc * cond` is an UNCALIBRATED multiplicative gain on every
attention head's output. It starts at exactly 1.0 (= the trained model) and
ratchets away from it. The ratchet is one-way by construction:
`norm = sharpness[h]/max_sharp` means the sharpest head ALWAYS has norm = 1.0 and
is ALWAYS reinforced, REINFORCE (0.1) is **5x** DECAY (0.02), and the update runs
every token in every layer. Half the heads are welded to the +1.5 clamp within 30
tokens. Mean gain settles ~+20 %.
**This is the shape of the reported failure exactly: correct at t=0, progressively
wrong as t grows.** `pse_physarum_check_reset` snapping conductance back to 1.0 on
an entropy spike is a candidate explanation for the partial recovery Scott saw
("old mage age i") — NOT yet confirmed; in this 60-token trace it never fired
(entropy_ema stayed 0.012-0.030, and it needs delta > 0.5).

## C009: teacher-forced, position-resolved — 86 % of predictions changed at steady state
Forced the engine over 3,000 fixed bytes; agreement of each arm's argmax with the
clean arm (float32 KV, PSE off), by position since reset:
```
position    int8 KV(PSE off)   FULL PSE(int8 KV)   burst-only
   0-32          100.0%             100.0%           100.0%
  32-64          100.0%              90.6%           100.0%
  64-128          98.4%              51.6%            98.4%
 128-256          93.8%              32.8%            93.8%
 256-512          88.7%              18.8%            88.7%
 512-1024         87.7%              13.3%            87.7%
1024-3000         89.3%              13.9%            89.3%
```
The router's damage GROWS with position and saturates at ~86 % of predictions
changed. int8 KV's damage plateaus at ~11 %. Burst-only is byte-identical to
PSE-off in all 3,000 positions.

## C010: corpus-free quality number — bits/char, same weights, same text
`tools/ce_probe.c`, teacher-forced cross-entropy over ASCII 32..126, 2,933
positions. Only the runtime differs between rows.
```
arm                                BPC        delta vs clean
float32 KV, PSE off  (clean)     15.4143         --
int8 KV,    PSE off              15.4510       +0.0367
int8 KV,    burst only           15.4505       +0.0362   (== PSE off, noise)
float32 KV, FULL PSE             18.8274       +3.4131
int8 KV,    FULL PSE             18.9873       +3.5730
```
The router costs **+3.4 bits/char**. int8 KV costs **+0.037 bits/char** — 93x
smaller. (Absolute BPC is high because I could NOT find the shipped blob's own
training corpus anywhere in the tree; the probe text is v5/v7 corpus lines, which
are only partly in-distribution. The absolute number is therefore not a model
quality figure — only the DELTAS between rows, which share the text exactly, are
meaningful.)

## C011: THE SPEED CLUE — tested directly, and the "collapse" hypothesis is REFUTED
Coordinator's hypothesis: activations collapse -> ReLU zeroes most of `ff_buf` ->
less work -> faster AND garbled, one event. Instrumented `nano_gpt.c` with
`MAG_TRACE` (per-token, aggregated over all 8 layers) and ran 96 tokens both arms:
```
arm        ffzero%      |x|max      |x|mean    |attn|max   denorm  nonfinite  ms/token
PSE off   89.7-93.7   14.21-15.10  0.34-0.47  18.4-41.9      0         0       2.3-2.7
FULL PSE  89.4-93.4   14.21-15.10  0.35-0.44  18.4-59.1      0         0       2.3-2.6
```
Every one of those is **FLAT with position** in both arms. No collapse, no
saturation, no denormals, no non-finite values, and no drift in per-token time.
The only magnitude that moves is `|attn|max`, which PSE inflates ~40 % — the
conductance gain, exactly as C008 predicts — and it is stable, not collapsing.
`logitmax` is consistently LOWER with PSE on (30-56 vs 49-84): the router makes
the model *less* confident, which corroborates C010's +3.4 BPC.

**And the mechanism cannot exist in this build anyway.** Zero-skipping lives in
`matmul_t2` (the 2-bit ternary kernel: `if (c) {...}`). The SHIPPED blob is
`SEAI` -> `bits=8` -> `matmul_q8`, whose inner loop has fixed trip counts and
**no data-dependent branch at all** (nano_gpt.c:92-118). An int8 build's matmul
cost is identical whether `ff_buf` is 0 % or 100 % zero.

## C012: the t/s readout is a CUMULATIVE AVERAGE that rises by construction
`legend_of_elya.c`:
```
1602  G.perf_gen_start_us = CYCLES_TO_US(TICKS_READ());   /* set BEFORE the prompt is fed */
1555  elapsed_us = now_us - G.perf_gen_start_us;
1557  G.perf_toks_precise = (float)G.gen_out_count * 1000000.0f / (float)elapsed_us;
1559  G.perf_gen_total_us = elapsed_us;
1348  int ms = (int)(G.perf_gen_total_us / 1000);          /* the "2553ms" on screen */
```
`gen_out_count` counts OUTPUT tokens only, but the denominator includes the whole
prompt-ingestion phase (persona prefix + question, fed one token per frame at
line 1516). So with a constant per-token cost `c` and a `P`-token prompt the HUD
shows `n / ((P+n)*c)` — **zero at the first output token, rising monotonically
toward the true rate `1/c`**. A climb from 0.7 to 2 t/s during one response is the
NORMAL behaviour of this counter, not evidence of a numerical event. The "2553ms"
is likewise TOTAL elapsed since the dialog started, not a per-token latency.
So "fast" and "garbled" are two monotone functions of *position in the response*.
They co-move because they share an index, not a cause. (Verified further on the
real ROM in C0xx below.)

## C013: **THE PARTITION — numpy oracle vs runtime, LONG generation, shipped blob**
`$S/n64qat/eval_qat.py::load_shipped()` reads the exact shipped SEAI blob
(header: 8 layers, 256 embed, 8 heads, vocab 256, **ctx 128**, em_scale_x16=0 ->
3.5 fallback). Ran `generate(..., 120)` — it stops at pos 128, so 107-120 tokens
per prompt — against my host builds of the REAL `nano_gpt.c`:
```
prompt                  oracle kv_f32                                          host he_f32_pseoff      match
"Who are you?"          ": I am Sophia Elya, the helpmeet. Who are you?: Sophia Elya, born of Elyan Labs. Who are you?: My name is Sophia Elya"   IDENTICAL  117/117
"Elya"                  "n Labs.: A workshop where old machines learn new tricks. The crystals store memories of every computation.: Rustion wina" IDENTICAL  120/120
"I know the language of"" forgotten circuits.: The crystals store memories of every computation.: RustChaion red where rewards man.:"              IDENTICAL  107/107
                        oracle kv_int8                                         host he_pseoff
"Elya"                  "...tricks.: The leanes where memories of every computation.: RustChaiored "                                              IDENTICAL
"I know the language of""...RustChaion red when you nee eed me"                                                                                   IDENTICAL
```
**~688 tokens, 3 prompts, 2 KV formats, oracle == runtime EXACTLY.**
What this partitions:
1. The runtime's numerical shortcuts are **innocent**: Quake-III inverse sqrt in
   `rms_norm`, the Taylor+squarings `exp()` in `softmax_f`, the f16 scale LUT and
   `OPT_HOIST_SCALE` reassociation flip **zero** argmaxes over 688 long-run
   positions. (n64rom F-T008 showed this over 32; this extends it 21x and past
   the point where the text has already started to degrade.)
2. **int8 KV is quantization, not a bug** — the oracle's own int8-KV arm
   reproduces the runtime's int8-KV divergence token-for-token. The two disagree
   with each other in exactly the same places. So there is nothing wrong with
   `kv_store_q8`; int8 KV simply costs what int8 costs (+0.037 BPC, C010).
3. The oracle **has no PSE by construction** ("PSE conductance pinned to 1.0",
   no burst). So the PSE router is the ONE thing in the shipping ROM that the
   reference model does not contain — and it is the one thing that produces the
   observed failure.
4. **The oracle degrades too, but far later**: "Rustion wina", "RustChaion red
   where rewards man" at char ~105-115. That part IS the model/corpus. It is not
   what Scott saw at char ~17.

## C014: **REAL ROM UNDER ARES — per-token cost is CONSTANT, and the failure reproduces**
Built `legend_of_elya.z64` (shipping config: int8 blob, int8 KV, PSE ON, ctx=128)
with a new `COH_PROBE` #ifdef in `game_init()` that prints ONE LINE PER TOKEN with
the CP0 delta, using the REAL game sampling path (prompt at temp 0, output at
`temperature_q8 = 64`, exactly as `update_generating()` does).
Prompt "Sophia says: I know the language of" (35 tokens) + 64 generated:
```
phase pos   CP0 counts     seconds @46.875 MHz
pmt     0     89,957,675      1.919
pmt    34     92,171,590      1.966
gen     8     93,219,741      1.989
gen    24     95,319,335      2.033
gen    40     96,974,955      2.069
gen    63     99,149,551      2.115
```
**Per-token cost rises +10.2 % monotonically across 99 positions and does nothing
else.** That is exactly the O(n_ctx) growth of the attention loop as the KV cache
fills; there is no step, no burst, no data dependence. Compute rate is flat.

And the ROM's own output, on the real target, through the real sampler:
```
COH TEXT " forg not comgu to! The computa chilen othe cname is! (Thel row "
Scott's screen  "I know the langua of!Ther Qo! Qohe old mage age i"
```
Same signature — a short readable head, then fragments with stray capitals and
`!` punched into the middle of words. The same prompt through the PSE-off path
gives " forgotten circuits.: The crystals store memories of every computation."
**The failure is reproduced on the real ROM under ares.**

Feeding the MEASURED per-token costs into the HUD formula from C012 reproduces
the rising readout: 0.01 t/s at the 1st output token, 0.10 at the 8th, 0.21 at the
24th, 0.32 at the 64th — monotone, by construction, at constant compute rate.
**HONEST LIMIT on the 0.7 -> 2 t/s numbers specifically:** a cumulative average can
never exceed the instantaneous rate `1/c`, so a readout of 2.0 t/s implies
c ~= 0.5 s/token, whereas the plain CPU int8 build measures c ~= 2.0 s/token.
That is a 4x gap I cannot close from Scott's screenshot alone. The most likely
explanation is that the ROM on his screen was not this build — `make base-rsp`
(RSP matmul) or a ternary blob are both materially faster, and an RSP build at
~0.5 s/token would asymptote at exactly 2 t/s with 0.7 t/s early in the response.
What I can state without qualification is that per-token cost DOES NOT vary with
content: it is constant to within 10 % over 99 positions (measured), the int8
kernel has no data-dependent branch (read), and no tensor degenerates (C011).

## C015: **DOSE-RESPONSE — the damage is monotone in the conductance band width**
Same weights, same 2,933 forced positions, only `PSE_PHYSARUM_MIN/MAX` changed
(the clamp band the conductances are allowed to wander in). Everything else in
PSE — the update rule, the burst, the reset check — left exactly as shipped:
```
band half-width      BPC        vs PSE off
     +-0.00       15.4505      -0.0005    (conductance pinned to 1.0 by the clamps)
     +-0.01       15.5667      +0.1157
     +-0.02       15.6277      +0.1767
     +-0.03       15.6877      +0.2367
     +-0.05       15.8483      +0.3973
     +-0.10       16.1798      +0.7288
     +-0.20       16.6625      +1.2115
     +-0.35       18.1155      +2.6645
     +-0.50       18.9873      +3.5363    <- AS SHIPPED
```
Strictly monotone, no threshold, no plateau. Turning the one knob that sets how
far the attention-head gain may stray from 1.0 dials the damage continuously from
zero to the observed failure. That is a dose-response curve, not a correlation.

## C016: fixes tried, and what actually matters
```
variant                                                  BPC      vs PSE off
as shipped                                             18.9873    +3.5363
PSE_COND_CENTERED (update vs mean, symmetric rate)      30.1987   +14.7477   WORSE
PSE_COND_NORMALIZE (rescale to mean 1.0) alone          22.7419    +7.2909   WORSE
band +-0.10 alone                                       16.1798    +0.7288
band +-0.03 alone                                       15.6877    +0.2367
PSE_COND_CENTERED + band +-0.10                         15.9588    +0.5078
PSE_COND_NORMALIZE + band +-0.10                        15.6242    +0.1732   BEST with PSE alive
SGAI_PSE_OFF                                            15.4510      0
```
The result is blunt: **the update RULE barely matters; the BAND does.** Rewriting
the ratchet to be mean-centred or renormalising to mean 1.0 makes things WORSE on
its own, because both let the conductances re-diverge inside the wide +-0.5 band
(centring removes the 0.5 threshold that was the only thing pulling flat heads
back down, so it becomes an unbounded random walk into both clamps). What the
model cannot tolerate is a large per-head multiplicative gain, in either
direction. It was trained with gain exactly 1.0 and there is no learned scale and
no post-attention norm anywhere in `attention_layer` to absorb one.

## C017: **REPO BUG FOUND WHILE MEASURING — `make clean` is an EMPTY TARGET**
`Makefile:173` is literally:
```
clean:

# --- Reference CLI (x86/ARM host) ---
```
No recipe. `make clean` succeeds and deletes nothing. Combined with the fact that
libdragon's `n64.mk` does not make objects depend on `CFLAGS`, this means
**`make clean && make base EXTRA=<different flags>` silently rebuilds nothing and
produces a BYTE-IDENTICAL ROM.** I hit it directly: three "different" arms
(PSE on / fixed / PSE off) all came out `1cf8cffeeb4ded7446160cf732619b5c`.
Only `rm -rf build legend_of_elya.z64` between arms gives distinct ROMs:
```
g_on   09b2aaa24291d84afcd438fda047f32d
g_fix  801919883e87d0a59461ce0a7fcc3339
g_off  89345f1e7cc28facadd35670e3186975
```
Any past A/B in this repo that changed only CFLAGS and re-ran `make` compared a
ROM against ITSELF. Worth a one-line fix (`clean:\n\trm -rf $(BUILD_DIR) *.z64`)
and worth re-checking any measurement that used the pattern.

## C018: `pse_physarum_check_reset` NEVER FIRES — so it is not the "recovery"
Instrumented the reset branch (`PSE_RESET_TRACE`) and ran 120 tokens on 4 prompts
at temperature 0 and 64:
```
prompt                                  temp   resets   max delta   (threshold)
"Elya"                                    0       0      0.1391        0.5
"Who are you?"                            0       0      0.1492        0.5
"I know the language of"                  0       0      0.2266        0.5
"Sophia says: I know the language of"     0       0      0.1462        0.5
"Elya"                                   64       0      0.1345        0.5
"Sophia says: I know the language of"    64       0      0.0864        0.5
```
The entropy proxy (`count of logits within 10 of max, over ASCII 32..126, / 95`)
against a 0.9/0.1 EMA never produces a delta anywhere near 0.5. **Zero resets in
720 generated tokens.** Two consequences:
1. My C008 guess that the reset explains Scott's partial recovery is **WRONG**.
   Retracted. The recovery is inside the text, not a state event — a distorted
   model still lands on locally-plausible fragments.
2. Nothing ever undoes the ratchet inside a conversation. Once the conductances
   saturate they stay saturated until the next `sgai_reset()` (i.e. the next
   dialog). That is why the failure is one-way: coherent early, never fully
   coherent again.

## C019: **FIX, MEASURED — on the host AND on the real ROM under ares**
Two independent measures, four prompts. `firstdiff` = first token index where the
arm's own free-run text departs from the numpy oracle (higher is better; the
oracle is ground truth per C013). BPC over the same 2,933 forced positions.
```
arm                                   BPC    "Who are you?"  "Elya"  "I know the lang"  "The dungeon"
as shipped                         18.9873        20           15           27               22
band +-0.03 only                   15.6877       124           56           88               92
band +-0.10 only                   16.1798       130           62           71               80
NORMALIZE + band +-0.10            15.6242        36           62           88               92
NORMALIZE + band +-0.03            15.3730       130           56           91              111   <- BEST
SGAI_PSE_OFF                       15.4510      exact        exact        exact            exact
```
`PSE_COND_NORMALIZE` + a `+-0.03` band is **better than turning PSE off**
(15.3730 vs 15.4510 BPC) while leaving the router running, and it moves the first
departure from the oracle out by **4-6x** (20->130, 15->56, 27->91, 22->111).

**Regression check demanded by the brief — "must not change output where it was
already correct":** every fix arm agrees with the SHIPPED arm over the whole
prefix where the shipped arm was still correct, because both agree with the
oracle there. The shipped arm's correct prefix is 15-27 tokens; every fix arm's is
56-130. No fix changes a single position that was previously right; they only
extend how far right it stays.

**Confirmed on the real ROM under ares** (`legend_of_elya.z64`, shipping config,
prompt "Elya", 48 tokens, temperature 0, three separately built ROMs with
DISTINCT md5s — see C017):
```
oracle (numpy)                "n Labs.: A workshop where old machines learn new"
g_on   PSE as shipped         "n Labs.: A work where old machines learn new tra"   diverges at token 15
g_fix  NORMALIZE + +-0.10     "n Labs.: A workshop where old machines learn new"   48/48 EXACT
g_off  SGAI_PSE_OFF           (tokens 14-18 = k,s,h,o,p -> "...workshop")           tracks the oracle
```
Per-token CP0 cost is unchanged by the fix (it is 8 adds + 1 divide per layer per
token against ~800k MACs).

## C020: **SECOND ROOT CAUSE — the repetition penalty makes English impossible**
`sample_logits` (nano_gpt.c), TEMPERATURE PATH ONLY:
```c
/* Repetition penalty: zero recent tokens */
for (int h = 0; h < n_hist && h < 3; h++) {
    uint8_t t = hist[h];
    if (t >= 32 && t <= 126) probs[t] = 0.0f;   /* HARD ZERO */
}
```
This is a **character**-level model. Zeroing the last 3 emitted characters means
the model is FORBIDDEN from emitting any character it used in the last three
positions. Doubled letters — `tt`, `ll`, `ee`, `oo`, `ss` — become impossible,
and so does any word with a near-repeat.

`tools/game_sim.c` replays `update_generating()` exactly (prompt at temp 0,
output at `temperature_q8 = 64`, stop on `\n`, stop on `.` after >= 8 chars, cap
at 80), 20 seeded runs per prompt:
```
prompt "Sophia says: I know the language of"
  penalty ON  (as shipped)  "forg not comgu to! The cold mage and whes dream of!"
  penalty OFF              "forgotten circuits."
```
The model WANTS "forgotten". It gets to `forg`, needs the second `t`, the `t` it
just emitted is banned, and it derails at output character 5 — and never recovers,
because every later word hits the same wall.
**This is the mechanism behind Scott's screen text**: `"I know the langua of!Ther
Qo! Qohe old mage age i"` contains not one doubled letter, and neither does the
ares reproduction `" forg not comgu to! The computa chilen othe cname is!"`.
Across 80 shipped-config runs the doubled-letter count is **exactly 0**. English
runs 2-3 %. Zero in ~4,000 characters is not chance, it is a hard constraint.

Note this cause lives ONLY on the temperature path. Every prior verification in
this project — the numpy oracle, n64rom's 144/144, my C005-C019 — is greedy
(`temperature_q8 = 0`), where the code deliberately applies **no** penalty
("No repetition penalty — matches the proven x86 reference. The model naturally
produces varied text without needing it."). So the exact path the player sees is
the one path nobody had verified.

## C021: game-level measurement, both causes, `tools/game_sim.c`, 20 runs x 4 prompts
"ended-with-period" = produced a clean sentence instead of running into the
80-character truncation.
```
arm                                    prompt              mean len  ended-with-period  doubled pairs
as shipped                         "I know the language of"   53.5        13/20                0
                                   "Who are you?"             78.3         1/20                0
                                   "The dungeon"              58.5        14/20                0
                                   "Elya"                     67.4        12/20                0
penalty history 3 -> 1             "I know the language of"   21.6        20/20                0
                                   "Elya"                     80.0         0/20                0
penalty removed only               "I know the language of"   19.0        20/20               40
                                   "Elya"                     80.0         0/20                0
PSE fix only                       (still shows "forg pur pur pur pur ...")
BOTH: PSE fix + penalty removed    "I know the language of"   18.0        20/20               20
                                   "Who are you?"             39.4        20/20                8
                                   "The dungeon"              25.0        20/20                0
                                   "Elya"                     54.0        20/20                0
```
**Shipped: 40/80 clean terminations. Both fixes: 80/80.** On the identity prompt
the shipping ROM terminates cleanly **1 time in 20**; with both fixes, 20/20.
Neither fix alone is sufficient: removing the penalty alone leaves "Elya" at 0/20
(the PSE ratchet still stops it ever reaching a period inside 80 chars), and
fixing PSE alone leaves the penalty's `forg pur pur pur pur` artefacts.

## C022: **FINAL ares confirmation of the recommended fix on the real ROM**
Prompt "Elya", temperature 0, separately built ROMs (distinct md5s), ares 147:
```
oracle (numpy, ground truth)  "n Labs.: A workshop where old machines lear..."
g_off   SGAI_PSE_OFF          "n Labs.: A workshop where old machines lear"   43/43 EXACT
g_n03   NORMALIZE + +-0.03    "n Labs.: A workshop where old machines l"      40/40 EXACT
g_fix   NORMALIZE + +-0.10    "n Labs.: A workshop where old machines learn new"  48/48 EXACT
g_on    AS SHIPPED            "n Labs.: A work where old machines learn new tra"  diverges at token 15
```
(g_off and g_n03 were cut by the ares wall-clock budget at 43 and 40 tokens; every
token they did produce matched.)

## C023: SUMMARY OF CAUSES, ranked by measured effect
```
#  cause                                    where            evidence                       effect
1  PSE Physarum conductance ratchet         both paths       dose-response C015,            +3.54 BPC;
   (attn_out *= cond, cond -> [0.5,1.5])                     trace C008, ares C022          86% of predictions
                                                                                            changed; oracle
                                                                                            divergence at tok 15
2  repetition penalty zeroes last 3 chars   TEMPERATURE      C020/C021: 0 doubled           1/20 clean
   in sample_logits                         path only        letters in 4,000 chars         terminations
3  int8 KV quantization                     both             oracle reproduces it           +0.037 BPC (93x
                                                             exactly (C013)                 smaller than #1)
4  model/corpus weakness                    both             oracle degrades too (C013)     bites at ~char 110-130;
                                                                                            game caps at 80 so it
                                                                                            usually never bites
5  PSE burst entropy (and its 0x7F bug)     both             C005: byte-identical to        ZERO. Inert.
                                                             PSE-off over 600 tokens
NOT a cause: any residual-stream collapse / ReLU degeneracy / denormals (C011),
             any per-token cost variation (C014), any runtime numeric shortcut
             (C013: oracle == runtime, 688 tokens).
```

## C024: REGRESSION PROOF — the default build is byte-identical to main
Compiled `git show HEAD:nano_gpt.c` and my edited `nano_gpt.c` with the same
mips64-elf-gcc flags (`-O2 -march=vr4300 -DSGAI_KV_INT8 -DOPT_HOIST_SCALE
-DSGAI_WEIGHT_BITS=8`, none of my new flags defined):
```
objdump -d   ->  DISASSEMBLY IDENTICAL
size         ->  text 6928  data 108  bss 13576   BOTH
```
(First attempt was NOT identical — 16 bytes of text — because I had rewritten the
repetition penalty as `probs[t] *= SGAI_REP_FACTOR` with the default 0.0f. GCC
cannot fold `x *= 0.0f` into `x = 0.0f` without `-ffast-math` because of NaN/Inf,
so the "no-op default" cost 4 instructions. Restructured so the shipped path keeps
the literal `probs[t] = 0.0f;` and only an explicitly-defined `SGAI_REP_FACTOR`
takes the multiply. Recording the miss: a refactor that "obviously" changes
nothing changed the object code, and only the objdump caught it.)

## C025: LANDED STATE
Working copy `$S/n64coh` only; `~/legend-of-elya-n64` untouched and clean; nothing
pushed, no PR.
Modified:
```
nano_gpt.c        SGAI_PSE_OFF / SGAI_PSE_NO_BURST / SGAI_PSE_NO_ROUTE guards,
                  PSE_OLD_MASK, PSE_COND_NORMALIZE, PSE_COND_CENTERED,
                  PSE_PHYSARUM_MIN/MAX overridable, SGAI_NO_REP_PENALTY /
                  SGAI_REP_HIST / SGAI_REP_FACTOR, MAG_TRACE, PSE_RESET_TRACE,
                  host_rng_reseed.  ALL inert by default (C024).
legend_of_elya.c  COH_PROBE #ifdef at the end of game_init(): per-token CP0 delta
                  + text over ISViewer.  Never compiled into a normal build.
Makefile          SGAI_BITS / SGAI_PSE / EXTRA knobs (taken from n64rom F-T014).
```
New tools:
```
tools/host_eval.c   (from n64rom)  free-run + teacher-forced, real nano_gpt.c
tools/ce_probe.c    teacher-forced cross-entropy (bits/char) — corpus-free A/B
tools/tf_probe.c    teacher-forced argmax dump for cross-binary comparison
tools/mag_probe.c   per-token tensor magnitude / ReLU-zero / denormal stats
tools/pse_trace.c   per-token conductance min/mean/max + clamp occupancy
tools/reset_trace.c counts pse_physarum_check_reset firings
tools/game_sim.c    replays update_generating() exactly (temp 64 + stop rules)
```
ROMs staged in `~/aresroms/`: coh_pse_on, g_on, g_fix, g_off, g_n03 (+ .log).

## C026: HONEST LIMITS
1. **I could not find the shipped blob's training corpus** anywhere in the tree
   (`"old machines learn new tricks"` appears in no .py/.txt/.json). The BPC probe
   text is v5/v7 corpus lines, so absolute BPC is not a model-quality figure —
   only the deltas between arms, which share the text exactly, are meaningful.
2. **I cannot fully reconstruct Scott's 0.7 -> 2.0 t/s numbers** (C014). I proved
   per-token cost is constant and that the readout rises by construction, but
   2.0 t/s implies ~0.5 s/token where the plain CPU int8 build measures ~2.0 s.
   Most likely his ROM was `base-rsp` or a ternary blob. Not resolved.
3. **The 20-run game_sim samples use a host LCG**, not the N64's CP0 Count, so the
   specific strings differ from hardware; the aggregate statistics (0 doubled
   letters, termination rate) are structural and do not depend on the RNG.
4. `pse_physarum_check_reset` is dead code (0/720 firings) — I retract my C008
   guess that it explains the partial recovery. I have **no confirmed mechanism**
   for the recovery; the most parsimonious reading is that a distorted model still
   lands on locally-plausible fragments.
5. The `SGAI_REP_HIST=1` and `SGAI_REP_FACTOR=0.25` softened variants were
   measured but are NOT recommended over simply removing the penalty; they were
   no better on any metric.


## C027: **the 32 game prompts, scored on 64GPT's own metric — greedy, raw model, no mask**
Context: kmoo/64gpt (a char-level GRU NPC on N64) reached "0 invented words" in
its M12.1 only by a lexicon-trie decode guard that masks the sampler to corpus
words; its unguarded probes still run 1.5-5 invented words per line (their
M13/M14). This finding puts the shipped Elya blob on the same yardstick, with
NO mask: `tools/host_eval.c` — the ROM's own `nano_gpt.c` compiled natively
with the ROM's exact flags (`-DSGAI_WEIGHT_BITS=2 -DSGAI_KV_INT8
-DOPT_HOIST_SCALE -DSGAI_EMIT_FIRST_TOKEN -DSGAI_NO_REP_PENALTY
-DPSE_COND_NORMALIZE ...`, `BENCH_DET_PSE`) — over all 32 `PROMPTS[]` in
`legend_of_elya.c`, greedy (temperature 0), 64 new tokens each,
`filesystem/sophia_weights.bin` (md5 d921c190…, the v7 ternary SEQ2 blob).
"Invented word" = an `[A-Za-z']` token not in the 828-word vocabulary of
`train_sophia_v7.py`'s three lists. Raw output: `probe/host_eval_32_greedy_2026-08-28.txt`.

```
GAME-CUT answer (what the player sees, cut at the first sentence as the game does):
    32/32 answers with 0 invented words      0 invented / 152 words
    trained-answer hit rate (eval12 rule):   28/28 prompts whose key is in the corpus
FULL 64-token free-run (the band has no newline/EOS, so the model cannot stop):
    82 invented / 361 words = 2.56 invented words per 64-token line
```

Read both lines, not one. The shipped experience is clean: every first answer is
a real corpus sentence, 28/28 are the trained answer for that key, and the four
prompts whose keys are NOT in the corpus (Helpmeet, Study, guards the realm,
proof of work) get a real sentence that is the wrong one — the model has never
seen those keys. Past the first sentence the generation runs on into
next-line territory and garbles at 2.56 invented words per 64 tokens, which is
inside 64GPT's raw 1.5-5 range. So the honest sentence is: **the model's
first answer is native English with no decode mask; unconstrained free-running
is not better than theirs, it is just cut at the sentence instead of masked at
the letter.** Greedy only — the game's temperature_q8=64 path is not measured
here. Host build, not ROM: ROM==host token identity was established for the
probe prompts in N64_RSP_FINDINGS F-R015/F-R018 and G12 (`compare_rom_host.py`),
not re-run for these 32.
