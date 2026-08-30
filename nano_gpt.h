// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <libdragon.h>

// Sophia Elya AI - World's First N64 LLM
// nano-GPT v9 LARGE: 8 layers, 256 embedding, 8 heads, vocab=256, ctx=128
// 6,356,992 parameters — requires 8MB Expansion Pak.
// (Was documented as "8.4M" here and in the README. That figure is 32% high:
//  8 layers x (4*256*256 attention + 2*256*1024 FFN) = 6,291,456, plus the
//  256*256 tied embedding = 6,356,992. Counted from the weight blob, whose
//  size 6,750,220 B matches this layout exactly.)
// RSP-accelerated matrix multiply via DMA tiling

#define SGAI_MAGIC      0x53454149  // "SEAI" — shipped int8 blob
/* "SEQn" — bit-packed weights, n = ASCII bit width (2..8).  Same 12-byte
 * header as SEAI; only the magic differs, so the loader picks the matmul
 * kernel off the magic alone.  See tools/quantize_n64.py for the format. */
#define SGAI_MAGIC_SEQ0 0x53455130  // "SEQ0" — add the bit width to get the magic
/* "SEP0" + bits: the SAME payload as SEQn but with every weight tensor ALREADY
 * permuted into the RSP's lane order, so sgai_init_ex() can skip the 161 ms
 * re-permutation it otherwise pays on every expert swap (F-R030/F-R031).
 *
 * It is a distinct MAGIC rather than a flag byte on purpose. For ternary the
 * permutation is an 8x8 transpose and therefore an INVOLUTION: a pre-permuted
 * blob permuted a second time returns to row-major, and the RSP then computes
 * on the wrong layout while still emitting fluent text. A separate magic makes
 * every wrong pairing fail to LOAD instead: an older ROM rejects an SEP blob
 * outright, and the CPU engine (which needs row-major) rejects it too.
 *
 * A 4-byte big-endian FNV-1a fingerprint over the first 256 bytes of the first
 * weight tensor is appended after the payload and re-checked at load, which
 * catches the one case a magic cannot: a row-major blob wearing an SEP label. */
#define SGAI_MAGIC_SEP0 0x53455030  // "SEP0"
#define SGAI_FP_BYTES   256
#define SGAI_MIN_BITS   2
#define SGAI_MAX_BITS   8
#define SGAI_N_LAYERS   8
#define SGAI_N_EMBED    256
#define SGAI_N_HEADS    8
#define SGAI_HEAD_DIM   (SGAI_N_EMBED / SGAI_N_HEADS)  // 32
#define SGAI_VOCAB      256
#ifndef SGAI_CTX
#define SGAI_CTX        128
#endif
#define SGAI_Q_BLOCK    32  // weight quantization block size

/* Attention score scale, 1/sqrt(head_dim).
 *
 * BUG FIX: nano_gpt.c hardcoded `0.17678f`, which is 1/sqrt(32) rounded to
 * five digits and is correct ONLY while SGAI_HEAD_DIM == 32.  Changing
 * SGAI_N_EMBED without changing SGAI_N_HEADS in the same proportion silently
 * mis-scales every attention score — no crash, no warning, just a worse model.
 * A table keyed on the actual head dim removes that trap: a head dim we have
 * not tabulated is a compile error rather than a wrong number.  These are the
 * correctly-rounded float32 values, so this also fixes the 3.3e-6 relative
 * error the five-digit literal carried. */
#if   SGAI_HEAD_DIM == 16
#define SGAI_INV_SQRT_HEAD_DIM 0.25f
#elif SGAI_HEAD_DIM == 32
#define SGAI_INV_SQRT_HEAD_DIM 0.17677669529663687f
#elif SGAI_HEAD_DIM == 64
#define SGAI_INV_SQRT_HEAD_DIM 0.125f
#elif SGAI_HEAD_DIM == 128
#define SGAI_INV_SQRT_HEAD_DIM 0.08838834764831843f
#else
#error "Add 1/sqrt(SGAI_HEAD_DIM) to the table in nano_gpt.h"
#endif

/* Bytes the loader must reserve for the weight blob.
 *
 * BUG FIX: this used to be a hardcoded 3 MB in legend_of_elya.c, described in
 * its own comment as "3MB for v8 Q8 6-layer 192-embed" — a model that has not
 * existed since commit f11042a.  The blob became 6,750,220 B, the `sz <=
 * sizeof(wbuf)` guard started failing, and the transformer has not run since.
 * Deriving the size from the architecture means that cannot recur.
 *
 * SGAI_WEIGHT_BITS is the widest weight format this build will accept, so a
 * ternary ("SEQ2") build does not reserve 4x the RAM it needs.  A blob narrower
 * than SGAI_WEIGHT_BITS simply uses less of the buffer; a wider one is rejected
 * by the size guard, loudly. */
#ifndef SGAI_WEIGHT_BITS
#define SGAI_WEIGHT_BITS 8
#endif
#define SGAI_LAYER_ELEMS_RAW  (4 * SGAI_N_EMBED * SGAI_N_EMBED + 2 * SGAI_N_EMBED * SGAI_N_EMBED * 4)
#define SGAI_LAYER_SCALES_RAW ((SGAI_LAYER_ELEMS_RAW / SGAI_Q_BLOCK) * 2)
#define SGAI_LAYER_BYTES(bits) \
    (((SGAI_LAYER_ELEMS_RAW * (bits)) / 8) + SGAI_LAYER_SCALES_RAW)
#define SGAI_WEIGHT_BUF_BYTES  \
    ((12 + SGAI_VOCAB * SGAI_N_EMBED                                       \
        + SGAI_N_LAYERS * SGAI_LAYER_BYTES(SGAI_WEIGHT_BITS) + 7) & ~7)

/* Same, for a blob whose width and depth are not the build's defaults.  The
 * dual-processor consensus holds a 2-bit 8-layer blob and an 8-bit N-layer one
 * in RDRAM at the same time, so one macro cannot size both. */
#define SGAI_WEIGHT_BUF_N(bits, nl) \
    ((12 + SGAI_VOCAB * SGAI_N_EMBED + (nl) * SGAI_LAYER_BYTES(bits) + 7) & ~7)

// Weight layout for one attention layer (Q8: int8 weights, float16 scales)
typedef struct {
    // Q8 packed weights (1 weight per byte, signed int8) + float16 scales (per 32-block)
    int8_t wq[SGAI_N_EMBED * SGAI_N_EMBED];        // Query projection
    int8_t wk[SGAI_N_EMBED * SGAI_N_EMBED];        // Key
    int8_t wv[SGAI_N_EMBED * SGAI_N_EMBED];        // Value
    int8_t wo[SGAI_N_EMBED * SGAI_N_EMBED];        // Output
    int8_t wff1[SGAI_N_EMBED * SGAI_N_EMBED * 4]; // FFN expand (192->768)
    int8_t wff2[SGAI_N_EMBED * SGAI_N_EMBED * 4]; // FFN contract (768->192)
    // Q8 scale factors (float16, one per 32-weight block)
    uint16_t sq[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sk[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sv[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t so[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sff1[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q_BLOCK];
    uint16_t sff2[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q_BLOCK];
} __attribute__((aligned(8))) SGAILayer;

/* Element counts, used to walk a bit-packed ("SEQn") layer arithmetically.
 * The SGAILayer struct above only describes the 8-bit case. */
#define SGAI_ATTN_ELEMS   (SGAI_N_EMBED * SGAI_N_EMBED)
#define SGAI_FF_ELEMS     (SGAI_N_EMBED * SGAI_N_EMBED * 4)
#define SGAI_LAYER_ELEMS  (4 * SGAI_ATTN_ELEMS + 2 * SGAI_FF_ELEMS)
#define SGAI_LAYER_SCALE_BYTES ((SGAI_LAYER_ELEMS / SGAI_Q_BLOCK) * 2)
#define SGAI_N_TENSORS    6

/* Resolved pointers to one layer's six weight tensors and their scale arrays,
 * in file order: wq wk wv wo wff1 wff2. */
typedef struct {
    const uint8_t  *w[SGAI_N_TENSORS];
    const uint16_t *s[SGAI_N_TENSORS];
} SGAILayerPtrs;

typedef struct {
    uint32_t magic;   // SGAI_MAGIC
    uint8_t n_layers;
    uint16_t n_embed;
    uint8_t n_heads;
    uint16_t vocab_size;
    uint8_t ctx_len;
    uint8_t em_scale_x16;  // embedding scale * 16 (e.g., 56 = 3.5)
    // After header: embedding table (vocab * embed bytes, Q8 int8)
    // Then n_layers SGAILayer structs
} __attribute__((packed)) SGAIHeader;

/* KV cache.
 *
 * The float32 cache is 8*128*256*4*2 = 2,097,152 B.  Together with the 6.75 MB
 * int8 weight blob that is 8,847,376 B, over the 8,388,608 B an Expansion Pak
 * console has, which is why the shipped ROM's weight load is guarded off and
 * the transformer never runs (see FINDINGS F8/F9 of the previous session).
 * The cache, not the weights, is where the cheap 1.5 MB is.
 *
 * SGAI_KV_INT8 stores k and v as int8 with one float32 scale per (layer,
 * position, HEAD).  Per-head and not per-vector because every consumer of the
 * cache is per-head: the score loop dots 32 contiguous dims of one head, and
 * the V-weighted sum reads the same 32.  A single scale over all 256 dims
 * would let one loud head set the quantization step for the other seven.
 *   int8 data   8*128*256*2   =   524,288 B
 *   scales      8*128*8*4*2   =    65,536 B
 *   total                         589,828 B   (3.56x smaller than float32)
 */
#if defined(SGAI_KV_INT8)
/* SGAI_KV_SCALE_PERVEC swaps the per-head scale for one scale per 256-dim
 * vector.  It saves 57,344 B and is measurably worse; kept as the A/B arm that
 * justifies the choice rather than asserting it. */
#ifdef SGAI_KV_SCALE_PERVEC
#define SGAI_KV_NSCALE 1
#else
#define SGAI_KV_NSCALE SGAI_N_HEADS
#endif
typedef struct {
    int8_t k[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    int8_t v[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    float  ks[SGAI_N_LAYERS][SGAI_CTX][SGAI_KV_NSCALE];
    float  vs[SGAI_N_LAYERS][SGAI_CTX][SGAI_KV_NSCALE];
    int pos;
} __attribute__((aligned(8))) SGAIKVCache;
#else
typedef struct {
    float k[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    float v[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    int pos;
} __attribute__((aligned(8))) SGAIKVCache;
#endif

/* Per-model activation scratch.
 *
 * These were `static` locals inside attention_layer(), which is correct for
 * ONE model called sequentially and silently wrong for two models whose
 * forward passes are interleaved -- which is exactly what the dual-processor
 * consensus does, so that model's q would land in this model's q.  One
 * scratch block per SGAIState makes the interleave safe by construction.
 * 12,288 B each. */
typedef struct {
    float q[SGAI_N_EMBED];
    float k_cur[SGAI_N_EMBED];
    float v_cur[SGAI_N_EMBED];
    float attn_out[SGAI_N_EMBED];
    float proj_out[SGAI_N_EMBED];
    float ff_out[SGAI_N_EMBED];
    float residual[SGAI_N_EMBED];
    float ff_buf[SGAI_N_EMBED * 4];
    float attn_scores[SGAI_CTX];
    float vw[SGAI_CTX];
} __attribute__((aligned(8))) SGAIScratch;

/* Which processor runs this model's matmuls. */
#define SGAI_ENGINE_CPU 0
#define SGAI_ENGINE_RSP 1

// Main inference state
typedef struct {
    const SGAIHeader *weights;  // Points into ROM
    SGAIKVCache *kv;            // In RDRAM (heap-allocated)
    SGAIScratch *sc;            // Activation scratch (NULL = use the shared one)
    int engine;                 // SGAI_ENGINE_CPU / SGAI_ENGINE_RSP
    int n_layers;               // From the blob header, <= SGAI_N_LAYERS
    int pse;                    // 1 = Physarum router + burst entropy active
    float x[SGAI_N_EMBED];     // Current token embedding
    float logits[SGAI_VOCAB];  // Output logits
    float em_scale;             // Embedding scale factor
    int w_bits;                 // Weight bit width: 8 for SEAI, 2..8 for SEQn
    uint8_t pre_permuted;       // 1 = SEPn, weights already in RSP lane order
    uint32_t tokens[SGAI_CTX]; // Generated token sequence
    int seq_len;
    int is_loaded;
    /* Hard-exclusion window: last 16 output tokens. */
    uint8_t penalty_hist[16];
    uint8_t penalty_n;
} SGAIState;

// API
void sgai_init(SGAIState *state, const void *rom_weights);
/* Same, but says which processor runs the matmuls and where the scratch is.
 * engine == SGAI_ENGINE_RSP permutes the weight tensors in place into the RSP
 * kernel's lane order, after which the CPU kernels can no longer read them. */
void sgai_init_ex(SGAIState *state, const void *rom_weights,
                  int engine, SGAIScratch *scratch, int pse);

/* One token through TWO models at once, their logits summed.
 *
 * a runs on the CPU, b on the RSP, and b's matmuls are dispatched so that a's
 * CPU work runs underneath them rather than after them.  b's logits are scaled
 * by 2^-shift before the sum: the two heads' logit magnitudes differ by a
 * factor that has to be measured, and a naive sum is the louder model alone
 * (measured on the sibling port at 54.7x; measured HERE in
 * docs/N64_DUAL_FINDINGS.md).  Returns the token the VOTE chose, which is then
 * fed to both models.
 *
 * Both models must have been sgai_init_ex'd with their own scratch. */
uint8_t sgai_dual_next_token(SGAIState *a, SGAIState *b, int shift,
                             uint8_t input_token, uint32_t temperature_q8);
/* The combined logits from the last sgai_dual_next_token(), for the
 * host-against-ROM check. */
extern float sgai_dual_logits[SGAI_VOCAB];
/* 1 = the no-overlap control: b's matmul is waited on before a's starts. */
extern int sgai_dual_serial;
/* Called once per transformer layer during sgai_next_token(), so a caller can
 * service something time-critical (the audio mixer) inside a 330 ms token.
 * NULL by default. */
extern void (*sgai_tick)(void);

/* In-band NPC commands (cmd_trie.h). All NULL/0 by default. */
extern int  (*sgai_cmd_allowed)(void);
extern int  (*sgai_cmd_mask)(uint8_t *mask);
extern void (*sgai_cmd_byte)(uint8_t b);
extern int  sgai_cmd_active;

void sgai_reset(SGAIState *state);
uint8_t sgai_next_token(SGAIState *state, uint8_t input_token, uint32_t temperature_q8);
void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8);

// RSP helpers (internal)
void sgai_rsp_matmul_q8(const int8_t *weights, const uint16_t *scales,
                         const int16_t *input, int16_t *output,
                         int in_dim, int out_dim);
void sgai_softmax_inplace(int16_t *vec, int len);
int16_t sgai_relu(int16_t x);
