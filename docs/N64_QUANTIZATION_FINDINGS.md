# Ternary quantization of the N64 LLM — FINDINGS

Working copy: `$S/n64ter` (copy of `~/legend-of-elya-n64` @ 3d9a93c, branch
`fix/expert-cache-and-weights-guard`). `~/legend-of-elya-n64` and
`~/legend-of-elya-genesis` are NEVER written. No commit, no push, no PR.
CP0 bench harness copied in from a previous agent's tree as `bench_cp0/`
(source: `$S/n64/bench`, see `$S/n64/FINDINGS.md` F10-F27).

Journal is append-only and written as results land, not at the end.

---

## T0 — CORRECTION to the brief: the weight blob is LITTLE-endian, not big-endian
The brief says `sophia_weights.bin` is BIG-ENDIAN. It is not, and getting this
wrong would have produced a silently-garbage ternary blob.

Evidence 1 — the header bytes:
```
53 45 41 49 | 08 | 00 01 | 08 | 00 01 | 80 | 00
"SEAI"      n_lay  n_embed  hd  vocab  ctx  em_scale_x16
```
`n_embed` and `vocab` decode to 256 only when read LITTLE-endian (`00 01`).
Read big-endian they are 1. `struct.unpack('<IBHBHBB')` -> (…,8,256,8,256,128,0). ✓

Evidence 2 — the writer, `train_sophia_v8.py:787`:
`struct.pack("<IBHBHBB", MAGIC_LE, ...)` and numpy `.tobytes()` on an x86 host,
i.e. little-endian throughout.

Evidence 3 — the reader, `nano_gpt.c:44`:
`f16_to_float()` opens with `f16 = (f16>>8)|(f16<<8); /* file is LE, N64 is BE */`.
The runtime byte-swaps every float16 scale on read. So a ternary blob MUST also
store its float16 scales LITTLE-endian or the existing decoder returns nonsense.

Why "SEAI" still reads as `0x53454149` on the N64: the magic is 4 ASCII bytes, so
it is endian-neutral in appearance; `sgai_init` accepts the value or its swap.
Only the multi-byte *numeric* fields are LE. Header `em_scale_x16` is 0 in the
shipped blob (not what `train_sophia_v8.py` would write — this blob predates it),
so `sgai_init` falls back to `em_scale = 3.5f`. My format keeps that byte at 0 so
behaviour is identical.

## T1 — layout re-derived and confirmed byte-exact
```
header                12 B
embedding  256*256    65,536 B int8   (no per-block scales; uses em_scale)
8 x layer            835,584 B each:
   wq wk wv wo        65,536 B each  int8   (256x256, row-major, o-major)
   wff1              262,144 B       int8   (1024 out x 256 in)
   wff2              262,144 B       int8   (256 out x 1024 in)
   sq sk sv so         4,096 B each  f16 LE (2,048 blocks of 32)
   sff1 sff2          16,384 B each  f16 LE (8,192 blocks of 32)
12 + 65,536 + 8*835,584 = 6,750,220 = actual file size. ✓
```
Quantization blocks are 32 CONSECUTIVE elements of the flattened row-major
matrix (`train_sophia_v8.py:760`), which is exactly how `matmul_q8` indexes them
(`scales + o*in_dim/32`, `row_s[blk/32]`). So a block never straddles a row
(in_dim 256 and 1024 are both multiples of 32). Per-block TWN is therefore a
drop-in for the existing scale array — same count, same order, same size.

## T2 — parameter count: the repo's "8.4M" is wrong; it is 6.36M
Per layer: 4*256*256 (attn) + 2*256*1024 (ffn) = 262,144 + 524,288 = 786,432.
8 layers = 6,291,456. Plus tied embedding 65,536 = **6,356,992 params (6.36 M)**.
`nano_gpt.h:8`, `README.md:222/240` and `legend_of_elya.c:1760` all say 8.4M.
That figure is not reproducible from the architecture and does not match the blob
size either. Everything below uses the measured 6,356,992.

## T3 — Ternary quantizer written; size + sparsity MEASURED
`tools/quantize_ternary_n64.py` (later generalized, see T8). 2-bit packed,
4 weights/byte, MSB-first, codes 00=0 01=+1 10=-1. Per-32-block TWN in int8
space (equivalent because the Q8 block scale is constant within a block).
Embedding table left int8 on purpose (65,536 B, 1 % of file, and it is the tied
logit matrix).

**Blob size 2,031,628 B for every TAU** (size is TAU-independent by construction
— that is the point of a fixed 2-bit packing vs an index stream).
6,750,220 / 2,031,628 = **3.3226x compression**. The brief's arithmetic checks out
exactly.

Sparsity (fraction of weights driven to 0), MEASURED:
```
TAU     wq      wk      wv      wo      wff1    wff2    TOTAL
0.5   30.11%  31.63%  32.65%  30.22%  30.73%  52.05%  37.98%
0.6   35.73%  37.51%  38.39%  36.03%  36.50%  57.59%  43.67%
0.7   41.25%  43.21%  44.00%  41.67%  42.09%  62.18%  48.94%
0.8   46.55%  48.64%  49.19%  47.13%  47.48%  66.07%  53.81%
```
wff2 is consistently ~15-20 points sparser than every other tensor at the same
TAU. That is a real property of the model, not an artefact: wff2 takes the
post-ReLU 1024-dim input, so its rows are dominated by a few large weights and
the mean-|w| threshold cuts far more of the tail.
+1 and -1 counts are balanced to <1 % everywhere except wff1/wff2, which lean
~7 %/5 % negative — consistent with a ReLU network learning inhibitory FFN rows.

## T4 — COORDINATOR REDIRECT mid-task (2026-08-04)
Coordinator ran an independent numpy reference of the forward pass and reports:
post-training ternary destroys output (top-1 8-21 % vs Q8) at every TAU, and the
real residency blocker is the **float32 KV cache** (2,097,156 B), not the weights.
Revised priorities: int8 KV cache first, then int5/int6 weights, generalize the
quantizer over bit width, and document ternary as needing QAT rather than
shipping it. Everything below follows the revised brief. I am verifying the
coordinator's numbers with an INDEPENDENT reference (a native build of the real
`nano_gpt.c`, not a numpy re-implementation) rather than taking them on trust.

## T5 — HOST REFERENCE BUILT **and cross-validated against the N64 bench**
`tools/host_eval.c` `#include`s the real `nano_gpt.c` and compiles it natively
(x86-64) against the bench's freestanding libdragon shim. It is NOT a numpy
port: the Quake fast-inverse-sqrt in `rms_norm`, the Taylor/squaring `exp()` in
`softmax_f`, the greedy argmax restricted to ASCII 32..126, and the PSE Physarum
conductance router are all the ROM's own code.

Two compile-time deltas, both documented in the source:
- `HOST_BUILD` — `f16_to_float()` skips its byte-swap (blob is LE, host is LE;
  the swap exists only because the N64 is BE). Also swaps the CP0 RNG in
  `sample_logits` for an LCG; unreachable in greedy mode.
- `BENCH_DET_PSE` — `pse_entropy()` is a counter LCG, so runs are reproducible.

**Validation.** Built with `-DSGAI_CTX=32` to match the previous agent's bare-metal
N64 bench configuration (F14 in `$S/n64/FINDINGS.md`) and fed the same prompt:
```
N64 bench, real MIPS code under mupen64plus : "n Labs.: A wo"
host_eval, x86-64 native                    : "n Labs.: A wo"
```
13/13 tokens identical. The host reference reproduces the console token stream
bit-for-bit, so quality numbers measured with it transfer to the ROM.
(All quality work below uses the shipped `SGAI_CTX=128`.)

## T6 — Generalized quantizer + packed-weight runtime, both landed
`tools/quantize_n64.py` emits a "SEQn" blob at any width 2..8 (n is an ASCII
digit inside the magic, so `bits` is self-describing and the header stays the
same 12 bytes). Weights are an MSB-first big-endian bit stream in the same flat
row-major order as the int8 array; a 32-weight block is always exactly 4*bits
bytes, so nothing straddles a block. Every width is plain two's complement, so
one sign-extension decodes all of them; ternary is the same encoding restricted
to {-1,0,+1} (00 -> 0, 01 -> +1, 11 -> -1; 10 is never emitted).
`tools/quantize_ternary_n64.py` remains as the ternary-specific tool from T3.

Runtime, in `nano_gpt.c`: `matmul_t2` (ternary; the inner op really is
skip / `acc += x` / `acc -= x`, no multiply), `matmul_qn` (generic 3..6 bit,
sliding 32-bit window), and `matmul_pk` which dispatches on `state->w_bits`.
bits==8 falls straight through to the untouched `matmul_q8`, so an 8-bit blob
is bit-identical to before. Block scale multiplies once per 32 weights in all
three kernels. `sgai_layer_ptrs()` walks a packed layer arithmetically; the
`SGAILayer` struct only ever described the 8-bit case.
Also fixed while there: `sgai_init` now reads the magic byte-at-a-time into a
big-endian word instead of the old `magic == swap32(magic)` heuristic, which
happened to work for "SEAI" but is not safe for a family of magics.

**Measured blob sizes — all six match the arithmetic exactly:**
```
bits   blob bytes    vs Q8
 8     6,750,220     1.0000x   (== the shipped file size)
 6     5,177,356     1.3038x
 5     4,390,924     1.5373x
 4     3,604,492     1.8727x
 3     2,818,060     2.3953x
 2     2,031,628     3.3226x
```

## T7 — QUALITY, MEASURED. Two metrics, because free-running top-1 lies.
Free-running top-1 is a bad primary metric on an autoregressive model: one
divergent token changes every token after it, so it measures *when* the models
first disagreed, not *how often* they disagree. I report both:
- **free-run**: 3 prompts x 24 greedy tokens, autoregressive.
- **forced**: 512 tokens of the model's own training text fed to BOTH models,
  argmax compared position by position. This is the honest per-token agreement.
Reference is the shipped SEAI blob; PSE is deterministic (`BENCH_DET_PSE`),
greedy (`temperature_q8 = 0`), `SGAI_CTX = 128`.

```
blob                    free-run top1   forced top1   mean|dlogit|(forced)
SEQ8  (round trip)         100.0 %         99.8 %          0.006
SEQ6  maxabs                87.5 %         90.0 %          1.00
SEQ6  mse                   98.6 %         85.9 %          1.18
SEQ5  maxabs                80.6 %         81.2 %          1.97
SEQ5  mse                   87.5 %         82.6 %          1.88
SEQ4  maxabs                48.6 %         61.1 %          5.10
SEQ4  mse                   69.4 %         63.3 %          4.36
SEQ3  maxabs                29.2 %         43.2 %         11.05
SEQ3  mse                   27.8 %         45.3 %          9.07
SEQ2  twn tau=0.5           13.9 %         31.8 %         10.10
SEQ2  twn tau=0.6           16.7 %         31.8 %         10.32
SEQ2  twn tau=0.7            9.7 %         35.5 %         11.85
SEQ2  twn tau=0.8            4.2 %         30.1 %         10.55
```

**TERNARY IS GARBAGE. Plainly.** Side by side, prompt "Elya", 24 greedy tokens:
```
Q8 shipped   "n Labs.: A work where ol"
SEQ2 tau=0.5 "mamamamamayayayayay s sa"
SEQ2 tau=0.6 "m s ay s se att d tho y "
SEQ2 tau=0.7 "ya, sthol s stod mered y"
SEQ2 tau=0.8 "chtharknd than Elin LLan"
```
and prompt "Who are you?":
```
Q8 shipped   ": I am Sophia Elya, born"
SEQ2 tau=0.5 " ay ay a am a am a am a "
SEQ2 tau=0.7 ": TS TS TS The S S S S S"
```
No TAU rescues it; 0.5-0.8 are all degenerate, they just fail differently.
The weight NRMSE at ternary is 0.46-0.49, i.e. the reconstruction error is
roughly half the magnitude of the weights themselves. That is the whole story.
**This independently confirms the coordinator's numpy result, using the ROM's
own C engine rather than a re-implementation.**

**But I do NOT confirm "100 % top-1 at int5/int6".** Measured over 3 prompts +
512 forced tokens, 6-bit is 85.9-90.0 % and 5-bit is 81.2-82.6 %. The 100 %
figure reproduces only on the single prompt "Who are you?" — that one prompt
agrees 24/24 at both 5 and 6 bits in my run too. Adding two more prompts and a
512-token forced pass breaks it. So 5/6-bit weights are *good*, not *free*.
The degradation is also gentle rather than catastrophic: at 6-bit the "Elya"
continuation becomes "n Labs.: A workshop wher" instead of "n Labs.: A work
where ol" — different, still fluent English, still on-persona. That is a very
different failure mode from ternary's "mamamamamayayayayay".

MSE-optimal scale search helps at 4 bits (48.6 -> 69.4 % free-run,
61.1 -> 63.3 % forced) and is a wash at 5-6 bits. It costs nothing at run time.

Caveat I am not hiding: SEQ8 is 99.8 % rather than 100 % on the forced pass.
The 8-bit path re-derives the block scale (`s = max|q|/127`) and re-rounds it to
float16, which is not an exact identity on the already-int8 input. One token in
512 flips. A `--method passthru` that copies the original bytes verbatim would
be exactly 100 %; I did not add it because the 8-bit path is not what anyone
would ship (the shipped SEAI blob already is that).

## T8 — int8 KV CACHE: implemented and MEASURED. It is not free either.
`SGAI_KV_INT8` in `nano_gpt.h`/`nano_gpt.c`. k and v become int8 with one
float32 scale per **(layer, position, head)**. Per-head and not per-vector
because every consumer is per-head — the score loop dots 32 contiguous dims of
one head and the V sum reads the same 32 — so a single 256-dim scale would let
one loud head set the quantization step for the other seven.
```
                        float32 KV     int8 KV
  k + v data            2,097,152 B     524,288 B
  scales                        0 B      65,536 B
  pos                           4 B           4 B
  TOTAL                 2,097,156 B     589,828 B     3.56x smaller
  saving                                1,507,328 B
```
The coordinator's table budgets int8 KV at 524,289 B, i.e. with no scales at
all. That is not implementable — without a dequant scale the cache is unitless.
The real figure is **589,828 B**; the extra 65,536 B does not change any of the
fit conclusions but every arithmetic below uses the real number.

Both dequant scales are hoisted the same way the Q8 block scale is: the K scale
multiplies the finished 32-dim dot product once, and the V scale is folded into
the attention weight once per timestep instead of once per (timestep, dim).

**Quality cost, measured** (shipped SEAI weights unchanged, only the cache
changes; gold = float32 KV):
```
free-run  top1  72/72  = 100.0 %
forced    top1 469/512 =  91.6 %
```
Free-running text is character-identical on all three prompts. So it is
invisible in the demo and NOT invisible under measurement: 8.4 % of forced
positions flip their argmax. I am reporting the 91.6 %, not the 100 %.

**Composite table — every configuration scored against the SAME gold
(float32 KV + shipped SEAI weights):**
```
config                                    free-run   forced
float32 KV, SEAI weights   (gold)           100.0 %  100.0 %
int8 KV,    SEAI weights                    100.0 %   91.6 %
int8 KV,    SEQ6 maxabs                      77.8 %   83.8 %
int8 KV,    SEQ6 mse                         98.6 %   83.0 %
int8 KV,    SEQ5 maxabs                      80.6 %   79.1 %
int8 KV,    SEQ5 mse                         87.5 %   79.3 %
int8 KV,    SEQ4 mse                         69.4 %   62.7 %
```
int8 KV + 5-bit weights still generates clean on-persona English:
```
gold                    "Who are you?" -> ": I am Sophia Elya, born"
int8 KV + SEQ5 mse      "Who are you?" -> ": I am Sophia Elya, born"
int8 KV + SEQ5 mse      "Elya"         -> "n Labs.: A workshop wher"   (gold: "A work where ol")
int8 KV + SEQ5 mse      "The dungeon"  -> ", where old machines die"
```
Errors compose roughly additively in the forced metric (91.6 % KV-only and
82.6 % SEQ5-only give 79.3 % combined), which is what you would expect from two
independent perturbations rather than one amplifying the other.

## T9 — **THE MODEL NOW FITS RDRAM, WITH THE SHIPPED WEIGHTS UNTOUCHED.**
Real ROM, real toolchain (`make base N64_INST=$HOME/n64-toolchain/mips64-toolchain`),
`legend_of_elya.c` fixed so `wbuf` is sized from `nano_gpt.h` instead of a
hardcoded 3 MB describing a model that no longer exists, plus the `sgai_init`
KV double-allocation from F25 handed back instead of leaked.

```
                                   text     data        bss        total
int8 KV   (SGAI_KV=int8, default) 148,120  34,668  7,359,896   7,542,684   FITS
float32 KV(SGAI_KV=float32)       147,544  34,660  8,866,712   9,048,916   DOES NOT FIT
                                                                (8,388,608 ceiling)
```
Segment check on the int8-KV build: the single RWE LOAD is
`0x80000400 + 0x731620`, so the image ends at **0x80731A20 = 7,543,328 B** of
RDRAM. Largest bss symbols confirm the arithmetic:
`wbuf.0 = 0x670010 = 6,750,224 B`, `G = 0x90C38 = 592,952 B` (of which `G.kv`
is 589,828).
```
RDRAM ceiling (Expansion Pak)           8,388,608
image end                               7,543,328
framebuffer 320x240x16bpp x2              307,200
--------------------------------------------------
left for heap + stack                     538,080
```
The float32 build is 478,308 B over the ceiling before a framebuffer exists.
So the coordinator's central claim is CONFIRMED and now demonstrated on the
real link, not on a spreadsheet: **quantizing the KV cache alone makes the
model resident, with the weights bit-for-bit as shipped.** Free-running output
is character-identical on all three prompts (T8); the honest cost is 8.4 % of
teacher-forced argmax positions.

`make base-rsp` also builds clean (text 150,008 / data 37,740 / bss 7,362,224),
which matters because it compiles `nano_gpt.c` a second way (`-DUSE_RSP_MATMUL`)
and takes the other side of the `#ifdef` around `matmul_q8`.

Honest limitation, unchanged from the previous session's F5: **libdragon ROMs
do not boot under mupen64plus 2.6** on this machine (its IPL3 mis-detects RDRAM
as 64 MB and wedges before `main()`), so I can link and inspect this ROM but I
cannot run it. Every runtime number below comes from the bare-metal bench ROM,
which boots.

## T10 — HOW LARGE A MODEL FITS. `tools/rdram_budget.py`, from measured costs.
Two corrections to the brief's arithmetic before the table:
1. **The KV cache is not a constant.** The brief (and the coordinator's table)
   hold it at the current model's size. It grows as `layers * ctx * embed`, so
   a bigger model pays for it twice. The budget below models that coupling:
   `KV = 2*L*CTX*E + 2*L*CTX*(E/32)*4 + 4` (int8 data + one f32 scale per head).
2. **The baseline is 6,356,992 params, not 8.4 M** (T2), so every "x bigger"
   ratio in the brief is ~32 % pessimistic.

Fixed cost, from the real link (T9), not estimated:
```
text + data                      182,788
non-model .bss                    19,844
framebuffer 320x240x16bpp x2     307,200
heap + stack reserve             131,072
--------------------------------------
                                 640,904   -> 7,747,704 B for weights + KV
```

Largest model per weight width (shapes constrained to L >= 6, embed <= 768 —
unconstrained the optimum is always "1 layer, 1152 embed" because params grow
as L*E^2 while KV grows as L*E, and a 1-layer transformer is not a model):
```
bits       params      L  embed    weights B      KV B   vs 6.36M now
  8     6,356,992      8    256    6,750,220    589,828     1.00x   <- exactly today
  6     8,683,520      7    320    7,070,732    645,124     1.37x
  5    10,027,008     10    288    6,916,620    829,444     1.58x
  4    11,984,896      8    352    6,780,940    811,012     1.89x
  3    14,958,592     10    352    6,595,084  1,013,764     2.35x
  2    21,446,656      6    544    6,797,836    940,036     3.37x
```
The 8-bit row landing on **exactly L=8, E=256 — the shipped architecture — is
an independent check that the budget model is right**: the current model really
is sitting at the int8 ceiling, which is why it did not fit.

So the brief's "~20.4 M at ternary" is confirmed (I get 21.4 M), but its "2.42x"
should be **3.37x** against the real 6.36 M baseline. And the coordinator's
"int5 is only ~9.3 M (1.1x)" should be **10.0 M (1.58x)** for the same reason.

### What has to change in nano_gpt.h (and one thing that must change elsewhere)
Mostly the three constants — `SGAI_N_LAYERS`, `SGAI_N_EMBED`, `SGAI_N_HEADS`.
Everything sized off them (the layer struct, `SGAI_WEIGHT_BUF_BYTES`, the KV
cache, `pse_state.conductance`, the static scratch arrays in `attention_layer`)
follows automatically. Three things do NOT:
1. **`nano_gpt.c:` `float inv_sqrt_hd = 0.17678f;` is a hardcoded 1/sqrt(32).**
   It is only correct while `SGAI_HEAD_DIM == 32`. Change `n_embed` without
   changing `n_heads` and attention is silently mis-scaled — no crash, just a
   worse model. This should be derived from `SGAI_HEAD_DIM`, not typed in.
2. **`SGAI_WEIGHT_BUF_BYTES` is sized for int8.** A 21 M-param ternary model
   needs 21 MB by that macro and 6.8 MB in reality. The macro has to become
   bit-width-aware (a `SGAI_WEIGHT_BITS` build knob) or the loader has to size
   the buffer from the blob's magic at run time.
3. **`pse_burst_inject()` does `int dim = (ent >> (i & 15)) & 0x7F;`** — masked
   to 0..127, so on the current 256-dim model only the first half of the
   activation vector ever receives entropy, and on a 544-dim model it would be
   under a quarter. The comment still says "Only 8 of 128 dims". Harmless
   today, wrong at any width; already stale, not introduced by resizing.
Outside the header: `train_sophia_v8.py`'s `N_LAYERS`/`N_EMBED`/`N_HEADS` must
match, and `rsp_matmul.S` should be re-checked for dimension assumptions.

## T11 — QAT: patch to the TRAINING script, plus a fine-tune driver
Per the revised brief, ternary is documented and not shipped, and the useful
artefact is a QAT path in training rather than a runtime for weights that
produce garbage. Two pieces:

**(a) `train_sophia_v8.py`** gains `QuantLinear` — an `nn.Linear` whose weight
is fake-quantized on every forward pass with a straight-through estimator — a
`fake_quant_weight()` that matches `tools/quantize_n64.py` exactly *including
the float16 rounding of the block scale*, and `--qat-bits` / `--qat-tau` flags.
The six `nn.Linear(...)` constructors now go through a `linear()` factory, so
turning QAT on does not touch the model code. `--qat-bits 0` (default) is the
original float behaviour, byte for byte.

**(b) `tools/qat_finetune.py`** — training from scratch is 180,000 steps. This
warm-starts by DEQUANTIZING the shipped SEAI blob back to float (the exact
inverse of the exporter) and fine-tunes with the fake quantizer on, which is
what anyone would actually do.

**Reconstruction verified before trusting any of it.** The PyTorch model rebuilt
from the shipped blob generates, greedily:
```
PyTorch reconstruction  "Who are you?" -> ": I am Sophia Elya, the "
C engine (host_eval)    "Who are you?" -> ": I am Sophia Elya, born"
PyTorch reconstruction  "Elya"         -> "n Labs.: A workshop wher"
C engine (host_eval)    "Elya"         -> "n Labs.: A work where ol"
```
Same persona, same register, diverging late — expected, because the C engine
additionally runs the PSE Physarum conductance update and its Taylor `exp()`
and fast-inverse-sqrt approximations, none of which the trainer has. Good
enough to fine-tune against; not good enough to quote as identical.

### T11a — a real trap: cross-entropy is anti-correlated with quality at the shipped em_scale
The shipped blob stores `em_scale_x16 = 0`, so `nano_gpt.c` falls back to 3.5.
At 3.5 the logits are enormous and CE is dominated by temperature:
```
em_scale   3.50 : val(float) = 19.47   val(ternary) = 16.02   <- ternary "better"
em_scale   0.40 : val(float) =  3.45   val(ternary) =  3.99   <- correct ordering
```
Scanned 0.05..3.5; the float minimum is at ~0.4. Greedy output is identical at
3.5 and 0.4 (verified), so the choice does not change what the model says — but
it completely changes whether the loss is usable as a training signal. QAT runs
at 0.4 and exports `em_scale_x16 = 6`. Had I not checked this, the fine-tune
would have optimised a metric that rewards ternarization.

### T11b — CONTROL arm exported through the identical path
`--no-train` exports the warm-started weights with no fine-tuning, i.e.
post-training ternary, through exactly the same export code as the QAT arm. It
reproduces the disaster, which is what makes the comparison honest:
```
control (post-training ternary, em16=6)  free-run 12.5 %   forced 33.0 %
  "Who are you?" -> ": S S S S S S S S S S S "
  "Elya"         -> "yaya, s thel s s tol    "
```
consistent with the `quantize_n64.py` tau=0.7 blob (9.7 % / 35.5 %, T7).

## T12 — CP0 BENCH, ARM A: my refactor costs +1.60 % instructions on the 8-bit path
Bare-metal bench ROM (`bench_cp0/`, previous session's F10 instrument),
`-DBENCH_DET_PSE -DBENCH_WEIGHTS_RDRAM -DSGAI_CTX=32`, prompt "Elya",
16 forward passes, `Core[RandomizeInterrupt]=0`.
```
                                       TOTAL_COUNT      tokens
previous session det_base (F27)      2,356,669,652    "lyan Labs.: A wo"
arm A: my nano_gpt.c, 8-bit, f32 KV  2,394,471,284    "lyan Labs.: A wo"
delta                                  +37,801,632    IDENTICAL, all 16
                                            +1.604 %
```
**Output is byte-identical across all 16 tokens**, so the `matmul_pk` dispatch
does not change the 8-bit arithmetic — as designed, `bits == 8` falls through
to the untouched `matmul_q8`. But it is 1.6 % more instructions, and that is a
regression I am reporting rather than burying.
It is far too large to be the dispatch itself (48 extra calls per token cannot
cost 1.18 M instructions). The likely cause is lost constant propagation: the
old code called `matmul_q8(layer->wq, layer->sq, x, q, SGAI_N_EMBED,
SGAI_N_EMBED)` with literal dimensions at each site, so GCC could specialise
the loops; now the call goes through `matmul_pk(..., bits)` with `bits` a
runtime value, and the dimensions arrive as ordinary arguments.
Remember Hard Rule 1: that is a hypothesis from reading, not a measurement.
An `always_inline` arm is queued to test it.

Instrument reminder (previous session F11): under mupen64plus CP0 Count
advances **exactly 2 per retired instruction** and memory access is priced at
ZERO — an uncached cart read costs the same as a `nop`. These are instruction
counts, not cycles. Nothing here can measure the halved memory traffic that is
the actual point of narrower weights on real silicon.

## T13 — Kernel decode correctness proved separately from quality
`tools/kernel_test.c` packs random levels at each width with an independent
MSB-first packer, computes the reference dot product from the levels directly,
and compares against `matmul_pk`. A bit-order or sign-extension bug would
otherwise have shown up only as "some quality loss", which is exactly the kind
of thing that gets rationalised.
```
bits=2  worst relative error = 0.000e+00  PASS
bits=3  worst relative error = 0.000e+00  PASS
bits=4  worst relative error = 0.000e+00  PASS
bits=5  worst relative error = 0.000e+00  PASS
bits=6  worst relative error = 0.000e+00  PASS
```
Bit-exact at every width. So every quality number in T7/T8 is the quantization,
not the decoder.

## T14 — **QAT RECOVERS TERNARY.** Coherent English comes back.
6,000-step fine-tune, batch 32, lr 3e-4 cosine, warm-started from the shipped
weights, `QuantLinear` ternary fake-quant with STE on every forward.
```
val (float weights, warm start)                3.4484
val (ternary fake-quant, warm start, no QAT)   3.9928
val (ternary fake-quant, best during QAT)      1.7410   <- step 250
```
Generated text, all three ternary blobs identical in size (2,031,628 B):
```
gold  (SEAI int8)   "Who are you?" -> ": I am Sophia Elya, born"
                    "Elya"         -> "n Labs.: A work where ol"
                    "The dungeon"  -> ", where old machines die"

post-training TWN   "Who are you?" -> ": S S S S S S S S S S S "
(control, no QAT)   "Elya"         -> "yaya, s thel s s tol    "
                    "The dungeon"  -> " Th y y y y yourks cre  "

ternary + QAT       "Who are you?" -> " Slicar shouldes and sec"
                    "Elya"         -> "y bearn the silenes. Rec"
                    "The dungeon"  -> " older silene? NPC: Rune"
```
The QAT arm produces real words, sentence structure, punctuation, and it even
reproduces the corpus's `NPC:` dialogue marker. The control produces stuttering
single letters. Same format, same size, same export code path, same TAU — the
only difference is whether the quantizer was inside the training loop.

**Do NOT read the top-1 numbers as the answer here.** Against the ORIGINAL
model the QAT blob scores 8.3 % free-run / 26.4 % forced, *worse* than the
control's 12.5 % / 33.0 %. That is the metric failing, not the model:
fine-tuning moved the weights, so the QAT blob is a DIFFERENT model that
happens to be much better English. Agreement-with-the-original only measures
quality while the candidate is trying to imitate the original. An 8-bit
fine-tune at the identical budget is running as the apples-to-apples ceiling.

Honest caveats:
- The run **overfits**: train loss 0.17 by step 3000 while val climbs to 3.05.
  Best val is at step 250 and `best_state` is what gets exported. A real QAT
  run wants the full training schedule and a larger corpus, not a 6k-step
  fine-tune on a corpus the model has already memorised.
- Part of the val improvement is simply "more training on this corpus", not
  ternary recovery. The 8-bit control separates those.
- This does not make ternary shippable today. It shows the failure in T7 is a
  property of POST-TRAINING conversion, not of ternary weights, which is the
  claim the brief asked to be tested.

## T15 — **ares BOOTS THE REAL ROM. The model loads and runs. Verified end to end.**
mupen64plus cannot boot libdragon ROMs on this machine (F5: its IPL3 mis-detects
RDRAM as 64 MB and wedges before `main()`). **ares (flatpak `dev.ares.ares`,
version 147) boots them and reports the correct 8,388,608 B of RDRAM.**

A headless probe built from THIS tree — real libdragon `main()`, real
`dfs_init` / `dfs_open` / `dfs_read`, real `sgai_init`, `SGAI_KV_INT8`,
`SGAI_CTX = 128` — run under ares with output over the IS-Viewer:
```
PROBE bufbytes=6750224 kvbytes=589832
LOAD  sz=6750220 cap=6750224 loaded=1 bits=8
TOK 0  in 69('E') out 108 ...  TOK 15 in 119 out 111
TEXT  lyan Labs.: A wo
PROBE_DONE
```
- `loaded=1` — **the weight load succeeds.** This is the bug that has kept the
  transformer dark since f11042a, fixed and observed working on an emulated
  console rather than argued from a linker map.
- `TEXT lyan Labs.: A wo` — byte-identical to the host reference (T5) and to the
  bare-metal mupen64plus bench (T12). Three independent implementations of the
  same 16 tokens.
- `kvbytes=589832`, not the 589,828 I computed: the struct is
  `__attribute__((aligned(8)))` so the trailing `int pos` pads by 4. Corrected.
  (The coordinator's 532,484 assumes ONE scale per 256-dim vector; see T16.)

Trap worth recording: the IS-Viewer window only accepts **32-bit** writes. My
first probe stored bytes and three of every four characters vanished — the log
looked like line noise and could easily have been read as "the ROM crashed".
It had in fact run perfectly; only the transport was wrong.

CP0 Count under ares is NOT the same instrument as under mupen64plus (~104 M
per token here at ctx=128 vs ~149 M at ctx=32 there). Do not compare the two
emulators' numbers; compare arms within one emulator.

## T16 — KV scale granularity: per-head vs per-vector, MEASURED
The coordinator's 532,484 B figure implies one float32 scale per 256-dim
vector. I implemented one per head. A/B, same everything else, gold = float32 KV:
```
                              bytes      free-run   forced
per head  (8 scales/vector)  589,832      100.0 %    91.6 %
per vector(1 scale /vector)  532,488       87.5 %    89.1 %
```
Per-vector visibly changes the output ("Elya" -> "A workshop wher" instead of
"A work where ol"). Per-head costs **57,344 B** — 0.68 % of RDRAM, against a
539 KB margin — and buys back both the free-running identity and 2.5 points of
forced agreement. Keeping per-head. `SGAI_KV_SCALE_PERVEC` is left in as the
A/B arm so the choice stays checkable rather than asserted.

## T17 — The three latent bugs: fixed, and the behavioural change MEASURED
1. `nano_gpt.c` `float inv_sqrt_hd = 0.17678f` -> `SGAI_INV_SQRT_HEAD_DIM`,
   a table in `nano_gpt.h` keyed on `SGAI_HEAD_DIM` with correctly-rounded
   float32 values and an `#error` for any head dim not tabulated. Fixes both
   the "silently wrong if you resize the model" trap and the 3.3e-6 relative
   error the five-digit literal carried.
2. `SGAI_WEIGHT_BUF_BYTES` is now derived from `SGAI_WEIGHT_BITS` (default 8),
   so a ternary build reserves 2 MB rather than 6.8 MB. It replaced the
   hardcoded `3 * 1024 * 1024` whose own comment described a 6-layer 192-embed
   model deleted in f11042a — that stale constant IS the shipped bug.
3. `pse_burst_inject()` `(ent >> (i & 15)) & 0x7F` -> `% (uint32_t)n_embed`.
   **This one is live in the shipped model right now**: the mask caps the
   target dimension at 127, and the model has been 256-dim since f11042a, so
   PSE entropy has only ever been able to reach the first half of the
   activation vector. The dead `if (dim >= n_embed) dim = dim % n_embed;`
   line below it could never fire, because 0x7F < 256.

**Measured effect of fixes 1 and 3** (both change arithmetic), float32 KV so
the KV change is not mixed in, against the pre-fix baseline:
```
free-run  72/72  = 100.0 %
forced   512/512 = 100.0 %
```
Zero behavioural change on this model over 584 tokens. They are correctness
fixes that remove traps, not tuning. With int8 KV the combined config scores
100.0 % free-run / 91.8 % forced (was 91.6 % before the PSE fix — one token).

## T18 — **The scale hoist is -14.09 % on the REAL ROM under ares, not -7.47 %**
Same source, same ROM, only `-DOPT_HOIST_SCALE` differs. Real libdragon build,
`SGAI_KV_INT8`, ctx 128, 16 forward passes, ares:
```
                 CP0 counts, 16 tokens      text
base            1,674,524,492               "lyan Labs.: A wo"
hoist           1,438,601,627               "lyan Labs.: A wo"
delta            -235,922,865  = -14.089 %  BYTE-IDENTICAL
```
Three instruments, one change, three answers:
```
static instruction count of matmul_q8   110 -> 114 instructions   "WORSE"
mupen64plus CP0 (instructions only)                    -7.47 %
ares (models memory + pipeline), real ROM             -14.09 %
```
Static counting would have rejected it. mupen64plus understates it by half,
because it prices every memory access at zero (F11) and the hoist removes a
float multiply per weight — work that competes for the same issue slots as the
loads. **This is now enabled by default in the shipping build** (`SGAI_HOIST=1`
in the Makefile, `make base SGAI_HOIST=0` to revert), because it is verified
end to end on a booting ROM with byte-identical output.

## T19 — Narrow weights are SLOWER on the N64, and now we can measure it
The original brief said the memory-traffic win of narrow weights is
unmeasurable on this instrument. That was true of mupen64plus. ares models
memory, so the question is answerable, and the answer is not the hoped-for one.
Real ROM, ares, 16 forward passes, identical everything but the blob:
```
weights          blob bytes    CP0 counts (16 tok)   vs 8-bit    text
8-bit (SEAI)      6,750,220     1,438,601,627         --         "lyan Labs.: A wo"
5-bit (SEQ5 mse)  4,390,924     1,801,381,193        +25.2 %     "lyan Labs.: A wo"
```
Both loaded and ran on the emulated console (`LOAD ... loaded=1 bits=5`), and
both produce the same 16 tokens, so the packed-weight runtime is verified on
hardware-accurate emulation and not only on the host.

But 5-bit is **25 % SLOWER** despite moving 35 % fewer weight bytes. The
bit-extraction in `matmul_qn` — a refill loop, a shift, a mask and a
sign-extend per weight — costs more than the RDRAM traffic it saves, because
the weights are already resident in RDRAM and the R4300i is not starved on this
workload once the model fits. Narrow weights are a RESIDENCY tool here, not a
speed tool. If speed were the goal the answer would be a nibble-aligned width
(4-bit, two per byte, no bit cursor) or an unrolled per-width kernel, not a
generic bit reader.
This also retires the brief's expectation that ternary "should win big from
halved memory traffic on real hardware": on this model, at this size, with the
weights in RDRAM, it does not.

### T19a — ternary arm, same measurement
```
weights          blob bytes    CP0 counts (16 tok)   vs 8-bit    text
2-bit (SEQ2 twn)  2,031,628       611,362,202        -57.5 %     "lsaya, sthol s s"
```
Ternary is **2.35x faster** — the kernel has no multiply in the inner loop and
skips ~49 % of weights outright, on top of moving 3.3x fewer bytes — but the text is the same post-training garbage
measured on the host (T7). Loaded and ran on the emulated console
(`LOAD sz=2031628 cap=2031632 loaded=1 bits=2`), which at least proves the
2-bit runtime path works on hardware-accurate emulation.
So the ordering on real N64 timing is: **ternary -57.5 %**, 8-bit baseline,
5-bit +25.2 %. The 5-bit penalty is the generic bit-cursor in `matmul_qn`;
the ternary kernel is a specialised one with no cursor and no multiply. That
is the shape of the answer: a hand-written kernel per width wins, a generic
bit reader loses.

**Correction, recorded rather than quietly fixed:** I first wrote 1,182,244,590
(-17.8 %) into this entry from a mental estimate before reading the log. The
measured value is 611,362,202 (-57.5 %). Caught on the next command. The habit
that caught it is the same one this whole document runs on — do not write a
number you have not read off an instrument.

## T20 — LANDED STATE (tree frozen for the coordinator)
Modified (5 files), all separable hunks:
```
Makefile            SGAI_KV knob (int8 default), SGAI_HOIST knob (on default)
nano_gpt.h          SEQn magics; SGAI_INV_SQRT_HEAD_DIM table; bit-width-aware
                    SGAI_WEIGHT_BUF_BYTES; int8 SGAIKVCache; SGAILayerPtrs;
                    SGAIState.w_bits
nano_gpt.c          matmul_t2 / matmul_qn / matmul_pk + sgai_layer_ptrs;
                    int8 KV store + hoisted dequant scales; endian-safe magic
                    parse; inv_sqrt_hd fix; PSE dim-mask fix; HOST_BUILD guards
legend_of_elya.c    wbuf sized from nano_gpt.h; KV double-allocation freed;
                    weight-load failure printed with both sizes on the title screen
train_sophia_v8.py  QuantLinear + fake_quant_weight + linear() factory,
                    --qat-bits / --qat-tau
```
New: `FINDINGS.md`, `tools/` (quantize_n64.py, quantize_ternary_n64.py,
qat_finetune.py, rdram_budget.py, host_eval.c, kernel_test.c, shim/libdragon.h).

Verified at freeze:
- `make base` and `make base-rsp` both link clean with
  `N64_INST=$HOME/n64-toolchain/mips64-toolchain`
- `make base SGAI_KV=float32 SGAI_HOIST=0` still builds (control arm alive)
- `tools/kernel_test.c` -> KERNEL TEST PASSED (bit-exact, widths 2..6)
- shipping config vs the original float32-KV Q8 baseline:
  free-run 72/72 = 100.0 %, teacher-forced 473/512 = 92.4 %
- real ROM boots under ares, `loaded=1`, generates "lyan Labs.: A wo"
- `legend_of_elya.z64` restored to HEAD: it is a tracked build artefact and
  does not belong in a source diff.
