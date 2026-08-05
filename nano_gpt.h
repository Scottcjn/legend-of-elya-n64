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

// Main inference state
typedef struct {
    const SGAIHeader *weights;  // Points into ROM
    SGAIKVCache *kv;            // In RDRAM (heap-allocated)
    float x[SGAI_N_EMBED];     // Current token embedding
    float logits[SGAI_VOCAB];  // Output logits
    float em_scale;             // Embedding scale factor
    int w_bits;                 // Weight bit width: 8 for SEAI, 2..8 for SEQn
    uint32_t tokens[SGAI_CTX]; // Generated token sequence
    int seq_len;
    int is_loaded;
    /* Hard-exclusion window: last 16 output tokens. */
    uint8_t penalty_hist[16];
    uint8_t penalty_n;
} SGAIState;

// API
void sgai_init(SGAIState *state, const void *rom_weights);
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
