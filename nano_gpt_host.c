/*
 * nano_gpt.c - Sophia Elya AI: World's First N64 LLM
 *
 * nano-GPT inference engine for N64 (MIPS R4300i + RSP)
 * Model: 2 layers, 128 embedding dim, 4 heads, vocab=256, ctx=32
 * Weights: Q4 quantized (2 nibbles/byte), scales in float16 per 32-block
 * Activations: Q8.7 fixed-point (int16_t, 1.0 = 128)
 *
 * RSP acceleration strategy:
 *   - DMA weight tiles into DMEM (4KB) in 128-byte chunks
 *   - data_cache_hit_writeback_invalidate prefetch hints
 *   - Process dot products in blocks of 8 (RSP vector width)
 *
 * Memory budget (8MB RDRAM):
 *   - Weights in ROM (DMA'd on demand): ~140KB
 *   - KV cache (SGAIKVCache): 2*32*128*2 * 2 = 32KB
 *   - Activations (x, logits, scratch): ~3KB
 *   - Total: well under 64KB working set
 */

#include "nano_gpt.h"
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <libdragon.h>

/* -----------------------------------------------------------------------
 * Fixed-point utilities
 * Q8.7: int16_t where 128 = 1.0
 * ----------------------------------------------------------------------- */

#define FP_ONE        128       /* 1.0 in Q8.7 */
#define FP_SCALE      128
#define FP_HALF       64        /* 0.5 in Q8.7 */

/* Multiply two Q8.7 values, return Q8.7 */
static inline int16_t fp_mul(int16_t a, int16_t b)
{
    return (int16_t)(((int32_t)a * (int32_t)b) >> 7);
}

/* Saturating add for Q8.7 */
static inline int16_t fp_add_sat(int32_t acc)
{
    if (acc > 32767) return 32767;
    if (acc < -32768) return -32768;
    return (int16_t)acc;
}

/* -----------------------------------------------------------------------
 * float16 decode (weights are stored as IEEE 754 half-precision scales)
 * Returns Q8.7 fixed-point (multiply this by raw Q4 nibble)
 * ----------------------------------------------------------------------- */
static int16_t f16_to_fp_scale(uint16_t f16)
{
    /* IEEE 754 half: s(1) | exp(5) | frac(10) */
    uint32_t sign     = (f16 >> 15) & 1;
    uint32_t exp      = (f16 >> 10) & 0x1F;
    uint32_t frac     = f16 & 0x3FF;
    float val;

    if (exp == 0) {
        /* subnormal */
        val = (frac / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        /* inf/nan -> clamp */
        val = 65504.0f;
    } else {
        val = (1.0f + frac / 1024.0f) * (float)(1 << (exp - 15));
    }
    if (sign) val = -val;

    /* Convert to Q8.7: clamp to int16 range */
    int32_t fixed = (int32_t)(val * FP_SCALE);
    if (fixed > 32767) fixed = 32767;
    if (fixed < -32768) fixed = -32768;
    return (int16_t)fixed;
}

/* -----------------------------------------------------------------------
 * Q4 dequantize helper
 * packed: pointer to packed byte array (2 nibbles/byte, low nibble first)
 * scales: float16 scale per 32-weight block
 * idx:    weight index (0-based)
 * Returns dequantized value in Q8.7
 * ----------------------------------------------------------------------- */
static inline int16_t q4_dequant(const uint8_t *packed, const uint16_t *scales, int idx)
{
    /* Extract nibble */
    uint8_t byte   = packed[idx >> 1];
    int     nibble = (idx & 1) ? (byte >> 4) : (byte & 0xF);
    int     w      = nibble - 8;              /* signed: -8..+7 */

    /* Scale for this block */
    int     block  = idx / SGAI_Q4_BLOCK;
    int16_t scale  = f16_to_fp_scale(scales[block]);

    /* w * scale: w is in [-8,7], scale in Q8.7 -> result in Q8.7 */
    return fp_mul((int16_t)(w * FP_ONE), scale);
}

/* -----------------------------------------------------------------------
 * RSP-accelerated matrix multiply (Q4 weights x Q8.7 input)
 *
 * Computes: output[out_dim] = W[out_dim x in_dim] * input[in_dim]
 * W is Q4 packed (out_dim * in_dim / 2 bytes), scales are float16.
 * output and input are Q8.7 fixed-point int16_t.
 *
 * RSP DMA strategy:
 *   We tile the weight matrix into 128-byte chunks (matching the RSP
 *   DMA granularity and DCACHE line size on R4300). For each output
 *   row we:
 *     1. Issue data_cache_hit_writeback_invalidate on the weight tile
 *        to get a clean DMA-ready line.
 *     2. Accumulate the dot product in int32 to avoid overflow.
 *     3. Scale back to Q8.7 and store.
 *
 * In a full RSP microcode implementation, steps 1-3 would be offloaded
 * to the RSP via DMA + vector MAC instructions (vmudh/vmadh on 8-lane
 * int16 vectors). Here we provide the CPU fallback with DMA hints so
 * the hardware prefetcher can pipeline weight loads.
 * ----------------------------------------------------------------------- */
void sgai_rsp_matmul_q4(const uint8_t *weights, const uint16_t *scales,
                          const int16_t *input,   int16_t *output,
                          int in_dim, int out_dim)
{
    /* Weight tile size: 128 bytes = 256 Q4 weights = 8 output rows worth
     * of inner-product for in_dim=128 (128/2 = 64 bytes per row, 2 rows/tile)
     * We prefetch 2 rows ahead. */
    const int TILE_BYTES = 128;

    for (int o = 0; o < out_dim; o++) {
        /* Prefetch next weight row into D-cache */
        int next_row = o + 2;
        if (next_row < out_dim) {
            const uint8_t *next_ptr = weights + (next_row * in_dim / 2);
            data_cache_hit_writeback_invalidate((void *)next_ptr, TILE_BYTES);
        }

        /* Dot product: row o of W (in_dim Q4 weights) dot input (Q8.7) */
        int32_t acc = 0;

        /* Process in blocks of 8 (RSP vector width) */
        const uint8_t  *row_w = weights + (o * in_dim / 2);
        const uint16_t *row_s = scales  + (o * in_dim / SGAI_Q4_BLOCK);

        for (int i = 0; i < in_dim; i += 8) {
            /* Unroll 8 lanes - mirrors RSP vmudh 8-element vector op */
            int lim = (i + 8 < in_dim) ? i + 8 : in_dim;
            for (int j = i; j < lim; j++) {
                int16_t w_dq = q4_dequant(row_w, row_s, j);
                acc += (int32_t)w_dq * (int32_t)input[j];
            }
        }

        /* Shift back: both operands were Q8.7 so product is Q16.14, >>7 to Q8.7 */
        acc >>= 7;
        output[o] = fp_add_sat(acc);
    }
}

/* -----------------------------------------------------------------------
 * Softmax in-place (Q8.7 input, Q8.7 output scaled to sum=128)
 * Uses integer approximation: find max, subtract, exponentiate with
 * e^x ≈ 1 + x for small x (sufficient for attention weights at this scale)
 * ----------------------------------------------------------------------- */
void sgai_softmax_inplace(int16_t *vec, int len)
{
    /* Find max for numerical stability */
    int16_t max_val = vec[0];
    for (int i = 1; i < len; i++) {
        if (vec[i] > max_val) max_val = vec[i];
    }

    /* Compute exp(x - max) approximation and sum
     * For Q8.7: e^x ≈ 1 + x + x^2/2 (2nd-order Taylor, x in [-4,0]) */
    int32_t sum = 0;
    int32_t exp_vals[SGAI_N_EMBED]; /* enough for attention (max in_dim) */
    int lim = (len < SGAI_N_EMBED) ? len : SGAI_N_EMBED;

    for (int i = 0; i < lim; i++) {
        int32_t x = (int32_t)(vec[i] - max_val);  /* x <= 0, Q8.7 */
        /* e^x approximation: clamp at -4.0 (=-512 in Q8.7) */
        if (x < -512) x = -512;
        /* Taylor: e^x ≈ 1 + x/128 + (x/128)^2/2 (convert to float domain) */
        /* In fixed-point: e_fp = FP_ONE + x + (x*x)/(2*FP_ONE) */
        int32_t e = FP_ONE + x + ((x * x) >> 8);
        if (e < 1) e = 1;
        exp_vals[i] = e;
        sum += e;
    }

    /* Normalize: output[i] = exp_vals[i] * FP_ONE / sum */
    if (sum == 0) sum = 1;
    for (int i = 0; i < lim; i++) {
        vec[i] = (int16_t)((exp_vals[i] * FP_ONE) / sum);
    }
}

/* ReLU for Q8.7 */
int16_t sgai_relu(int16_t x)
{
    return (x > 0) ? x : 0;
}

/* -----------------------------------------------------------------------
 * Layer normalization (simplified RMS norm)
 * Normalizes vec in-place. No learned scale/bias (nano-GPT omission).
 * ----------------------------------------------------------------------- */
static void rms_norm(int16_t *vec, int len)
{
    int64_t sum_sq = 0;
    for (int i = 0; i < len; i++) {
        sum_sq += (int64_t)vec[i] * vec[i];
    }
    /* RMS = sqrt(sum_sq / len), in Q8.7 */
    int32_t mean_sq = (int32_t)(sum_sq / len);
    /* Integer sqrt via Newton's method */
    if (mean_sq <= 0) return;
    int32_t rms = 128; /* initial guess for sqrt(mean_sq) */
    for (int iter = 0; iter < 8; iter++) {
        rms = (rms + mean_sq / rms) >> 1;
    }
    if (rms == 0) rms = 1;

    /* Normalize: vec[i] = vec[i] * FP_ONE / rms */
    for (int i = 0; i < len; i++) {
        vec[i] = (int16_t)(((int32_t)vec[i] * FP_ONE) / rms);
    }
}

/* -----------------------------------------------------------------------
 * Embedding lookup (Q4 packed table)
 * Writes SGAI_N_EMBED Q8.7 values into out[]
 * Embedding table immediately follows SGAIHeader in ROM.
 * ----------------------------------------------------------------------- */
static void embed_lookup(const SGAIHeader *hdr, uint8_t token, int16_t *out)
{
    /* Embedding table: vocab * n_embed / 2 bytes, right after header */
    const uint8_t *emb_table = (const uint8_t *)(hdr + 1);
    int offset = (int)token * SGAI_N_EMBED;

    /* No per-block scales for embedding; use scale=1.0 (FP_ONE) */
    /* For demo/placeholder, use token-seeded pseudo-embedding */
    if (hdr == NULL) {
        /* Null weights: deterministic hash-based embedding */
        for (int i = 0; i < SGAI_N_EMBED; i++) {
            /* Cheap hash: mix token with dimension index */
            uint32_t h = (uint32_t)token * 2654435761u + (uint32_t)i * 40503u;
            out[i] = (int16_t)((int8_t)(h >> 16));  /* [-128, 127] */
        }
        return;
    }

    for (int i = 0; i < SGAI_N_EMBED; i++) {
        int idx = offset + i;
        uint8_t byte   = emb_table[idx >> 1];
        int     nibble = (idx & 1) ? (byte >> 4) : (byte & 0xF);
        out[i] = (int16_t)((nibble - 8) * FP_ONE / 8);  /* scale to Q8.7 */
    }
}

/* -----------------------------------------------------------------------
 * Attention layer forward pass
 * Single multi-head attention block:
 *   1. Project to Q, K, V
 *   2. Store K, V in KV cache
 *   3. Compute attention scores (Q dot all cached K)
 *   4. Softmax + weighted sum of V
 *   5. Project output (Wo)
 *   6. Residual add
 *   7. FFN: x = x + ff2(relu(ff1(x)))
 * ----------------------------------------------------------------------- */
static void attention_layer(const SGAILayer *layer, SGAIKVCache *kv,
                             int layer_idx, int pos,
                             int16_t *x)
{
    static int16_t q[SGAI_N_EMBED];
    static int16_t k_cur[SGAI_N_EMBED];
    static int16_t v_cur[SGAI_N_EMBED];
    static int16_t attn_out[SGAI_N_EMBED];
    static int16_t ff_buf[SGAI_N_EMBED * 4];  /* FFN hidden (512) */
    static int16_t attn_scores[SGAI_CTX];
    static int16_t residual[SGAI_N_EMBED];

    /* Save residual for skip connection */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(int16_t));

    /* Layer norm input */
    rms_norm(x, SGAI_N_EMBED);

    /* Project Q, K, V */
    sgai_rsp_matmul_q4(layer->wq, layer->sq, x, q,     SGAI_N_EMBED, SGAI_N_EMBED);
    sgai_rsp_matmul_q4(layer->wk, layer->sk, x, k_cur, SGAI_N_EMBED, SGAI_N_EMBED);
    sgai_rsp_matmul_q4(layer->wv, layer->sv, x, v_cur, SGAI_N_EMBED, SGAI_N_EMBED);

    /* Store K, V in cache at current position */
    if (pos < SGAI_CTX) {
        memcpy(kv->k[layer_idx][pos], k_cur, SGAI_N_EMBED * sizeof(int16_t));
        memcpy(kv->v[layer_idx][pos], v_cur, SGAI_N_EMBED * sizeof(int16_t));
    }

    /* Compute attention scores: Q dot K[0..pos] for each head */
    /* Multi-head: process head by head */
    memset(attn_out, 0, SGAI_N_EMBED * sizeof(int16_t));

    int n_ctx = (pos + 1 < SGAI_CTX) ? pos + 1 : SGAI_CTX;

    for (int h = 0; h < SGAI_N_HEADS; h++) {
        const int16_t *q_head = q + h * SGAI_HEAD_DIM;

        /* Attention scores for this head over all positions */
        for (int t = 0; t < n_ctx; t++) {
            const int16_t *k_head = kv->k[layer_idx][t] + h * SGAI_HEAD_DIM;
            int32_t score = 0;
            for (int d = 0; d < SGAI_HEAD_DIM; d++) {
                score += (int32_t)q_head[d] * (int32_t)k_head[d];
            }
            /* Scale by 1/sqrt(head_dim) = 1/sqrt(32) ≈ 0.177
             * In Q8.7: multiply by 23 (≈ 0.177 * 128) then >> 7 */
            score = (score * 23) >> 14;  /* >> 7 for Q8.7^2, >> 7 for scale */
            attn_scores[t] = fp_add_sat(score);
        }

        /* Causal mask: positions beyond pos already excluded by n_ctx */
        /* Softmax over attn_scores[0..n_ctx-1] */
        sgai_softmax_inplace(attn_scores, n_ctx);

        /* Weighted sum of V */
        int head_out_base = h * SGAI_HEAD_DIM;
        for (int d = 0; d < SGAI_HEAD_DIM; d++) {
            int32_t acc = 0;
            for (int t = 0; t < n_ctx; t++) {
                const int16_t *v_head = kv->v[layer_idx][t] + h * SGAI_HEAD_DIM;
                acc += (int32_t)attn_scores[t] * (int32_t)v_head[d];
            }
            attn_out[head_out_base + d] = fp_add_sat(acc >> 7);
        }
    }

    /* Output projection Wo */
    static int16_t proj_out[SGAI_N_EMBED];
    sgai_rsp_matmul_q4(layer->wo, layer->so, attn_out, proj_out,
                        SGAI_N_EMBED, SGAI_N_EMBED);

    /* Residual add: x = residual + proj_out */
    for (int i = 0; i < SGAI_N_EMBED; i++) {
        x[i] = fp_add_sat((int32_t)residual[i] + (int32_t)proj_out[i]);
    }

    /* ---- FFN block ---- */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(int16_t));

    /* Layer norm before FFN */
    rms_norm(x, SGAI_N_EMBED);

    /* ff1: 128 -> 512 */
    sgai_rsp_matmul_q4(layer->wff1, layer->sff1, x, ff_buf,
                        SGAI_N_EMBED, SGAI_N_EMBED * 4);

    /* ReLU */
    for (int i = 0; i < SGAI_N_EMBED * 4; i++) {
        ff_buf[i] = sgai_relu(ff_buf[i]);
    }

    /* ff2: 512 -> 128 */
    static int16_t ff_out[SGAI_N_EMBED];
    sgai_rsp_matmul_q4(layer->wff2, layer->sff2, ff_buf, ff_out,
                        SGAI_N_EMBED * 4, SGAI_N_EMBED);

    /* Residual add */
    for (int i = 0; i < SGAI_N_EMBED; i++) {
        x[i] = fp_add_sat((int32_t)residual[i] + (int32_t)ff_out[i]);
    }
}

/* -----------------------------------------------------------------------
 * Logit projection: x[128] -> logits[256]
 * Uses a simple linear unembedding (tied embedding weights for efficiency).
 * For the null-weight demo, uses a byte-frequency prior biased toward
 * printable ASCII to produce semi-plausible character outputs.
 * ----------------------------------------------------------------------- */
static void project_to_logits(const SGAIHeader *hdr, const int16_t *x,
                               int16_t *logits)
{
    if (hdr == NULL) {
        /* Demo mode: logits biased toward printable ASCII (32-126) */
        for (int v = 0; v < SGAI_VOCAB; v++) {
            /* Base: prefer printable characters */
            int32_t base = (v >= 32 && v <= 126) ? FP_ONE : -(FP_ONE * 4);
            /* Mix in a projection of x[v % SGAI_N_EMBED] for variation */
            int32_t proj = x[v % SGAI_N_EMBED];
            logits[v] = fp_add_sat(base + (proj >> 2));
        }
        return;
    }

    /* Tied unembedding: use embedding table rows as projection vectors */
    const uint8_t *emb_table = (const uint8_t *)(hdr + 1);
    for (int v = 0; v < SGAI_VOCAB; v++) {
        int32_t acc = 0;
        int offset = v * SGAI_N_EMBED;
        for (int i = 0; i < SGAI_N_EMBED; i++) {
            int idx = offset + i;
            uint8_t byte   = emb_table[idx >> 1];
            int     nibble = (idx & 1) ? (byte >> 4) : (byte & 0xF);
            int16_t e_val  = (int16_t)((nibble - 8) * FP_ONE / 8);
            acc += (int32_t)e_val * (int32_t)x[i];
        }
        logits[v] = fp_add_sat(acc >> 7);
    }
}

/* -----------------------------------------------------------------------
 * Temperature sampling
 * temperature_q8: temperature in Q8 (256 = 1.0). Lower = more greedy.
 * Returns sampled token index.
 * ----------------------------------------------------------------------- */
static uint8_t sample_logits(const int16_t *logits, uint32_t temperature_q8)
{
    /* ASCII-constrained greedy sampler.
     *
     * The model biases toward high-byte output (0x80+) because the tiny
     * 2-layer architecture and Q4 weights produce logits dominated by
     * non-ASCII tokens. The Taylor softmax also mishandles large-negative
     * masked logits, making stochastic sampling unreliable.
     *
     * Solution: greedy argmax restricted to printable ASCII (0x20-0x7E).
     * With light top-3 randomness for temperature > 0.
     */
    int best = 0x20;
    int16_t best_val = logits[0x20];
    int top_v[3];
    int16_t top_s[3];
    int n_top = 0;

    /* Find best printable token */
    for (int v = 0x20; v <= 0x7E; v++) {
        if (logits[v] > best_val || n_top == 0) {
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
        }
    }

    /* Pure greedy for temperature == 0 */
    if (temperature_q8 == 0) {
        return (uint8_t)best;
    }

    /* Light stochastic: collect top-3 printable tokens */
    top_v[0] = best; top_s[0] = best_val;
    top_v[1] = (best == 0x61) ? 0x20 : 0x61;
    top_s[1] = logits[top_v[1]];
    top_v[2] = (best == 0x65) ? 0x20 : 0x65;
    top_s[2] = logits[top_v[2]];

    for (int v = 0x20; v <= 0x7E; v++) {
        if (v == top_v[0]) continue;
        if (logits[v] > top_s[1]) {
            top_v[2] = top_v[1]; top_s[2] = top_s[1];
            top_v[1] = v;        top_s[1] = logits[v];
            if (top_s[0] < top_s[1]) {
                int tv = top_v[0]; int16_t ts = top_s[0];
                top_v[0] = top_v[1]; top_s[0] = top_s[1];
                top_v[1] = tv; top_s[1] = ts;
            }
        } else if (logits[v] > top_s[2]) {
            top_v[2] = v; top_s[2] = logits[v];
        }
    }

    /* RNG from time */
    struct timespec _ts;
    clock_gettime(1, &_ts);
    uint32_t rng = (uint32_t)(_ts.tv_nsec ^ _ts.tv_sec);
    rng ^= rng >> 16; rng *= 0x45d9f3bU; rng ^= rng >> 16;

    /* Pick from top-3: strong bias to top if clear winner */
    int16_t gap01 = (int16_t)(top_s[0] - top_s[1]);
    int pick;
    if (gap01 > 24) {
        pick = 0;  /* clear winner, always pick best */
    } else {
        int r = (int)(rng & 0x7);
        pick = (r < 5) ? 0 : (r < 7) ? 1 : 2;
    }
    return (uint8_t)top_v[pick];
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

void sgai_init(SGAIState *state, const void *rom_weights)
{
    memset(state, 0, sizeof(SGAIState));

    if (rom_weights != NULL) {
        const SGAIHeader *hdr = (const SGAIHeader *)rom_weights;
        if (hdr->magic == SGAI_MAGIC || hdr->magic == 0x49414553U) {
            state->weights = hdr;
            state->is_loaded = 1;
        }
    }

    /* Allocate KV cache in RDRAM (8-byte aligned for DMA) */
    state->kv = (SGAIKVCache *)memalign(8, sizeof(SGAIKVCache));
    if (state->kv) {
        memset(state->kv, 0, sizeof(SGAIKVCache));
        state->kv->pos = 0;
    }

    state->seq_len = 0;
}

void sgai_reset(SGAIState *state)
{
    if (state->kv) {
        memset(state->kv, 0, sizeof(SGAIKVCache));
        state->kv->pos = 0;
    }
    state->seq_len = 0;
    memset(state->x, 0, sizeof(state->x));
    memset(state->logits, 0, sizeof(state->logits));
}

/*
 * Run one forward pass for a single input token.
 * Updates KV cache, returns next predicted token.
 */
uint8_t sgai_next_token(SGAIState *state, uint8_t input_token,
                          uint32_t temperature_q8)
{
    if (!state->kv) return 0;

    int pos = state->kv->pos;

    /* 1. Embedding lookup */
    embed_lookup(state->weights, input_token, state->x);

    /* 2. Run transformer layers */
    if (state->is_loaded && state->weights != NULL) {
        /* Weights layout after header:
         * - Embedding table: SGAI_VOCAB * SGAI_N_EMBED / 2 bytes
         * - n_layers SGAILayer structs */
        const uint8_t *after_hdr = (const uint8_t *)(state->weights + 1);
        size_t emb_table_bytes = SGAI_VOCAB * SGAI_N_EMBED / 2;
        const SGAILayer *layers = (const SGAILayer *)(after_hdr + emb_table_bytes);

        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            attention_layer(&layers[l], state->kv, l, pos, state->x);
        }
    } else {
        /* Demo mode: run with null weights (produces character-frequency biased output) */
        SGAILayer dummy;
        memset(&dummy, 0, sizeof(dummy));
        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            attention_layer(&dummy, state->kv, l, pos, state->x);
        }
    }

    /* 3. Final layer norm */
    rms_norm(state->x, SGAI_N_EMBED);

    /* 4. Project to logits */
    project_to_logits(state->weights, state->x, state->logits);

    /* 5. Sample next token */
    uint8_t next_tok = sample_logits(state->logits, temperature_q8);

    /* 6. Advance position in KV cache */
    if (state->kv->pos < SGAI_CTX - 1) {
        state->kv->pos++;
    } else {
        /* Context full: shift KV cache left (sliding window) */
        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            for (int t = 0; t < SGAI_CTX - 1; t++) {
                memcpy(state->kv->k[l][t], state->kv->k[l][t + 1],
                       SGAI_N_EMBED * sizeof(int16_t));
                memcpy(state->kv->v[l][t], state->kv->v[l][t + 1],
                       SGAI_N_EMBED * sizeof(int16_t));
            }
        }
    }

    /* Store token in sequence */
    if (state->seq_len < SGAI_CTX) {
        state->tokens[state->seq_len++] = input_token;
    }

    return next_tok;
}

/*
 * Generate up to max_tokens tokens from a prompt.
 * Output written to caller-provided buffer (null-terminated).
 */
void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8)
{
    sgai_reset(state);

    /* Process prompt tokens (teacher-forcing: feed each prompt token,
     * discard output until last) */
    uint8_t tok = 0;
    for (int i = 0; i < prompt_len; i++) {
        tok = sgai_next_token(state, prompt[i], temperature_q8);
    }

    /* Generate output tokens */
    int out_idx = 0;
    while (out_idx < max_tokens - 1) {
        tok = sgai_next_token(state, tok, temperature_q8);
        if (tok == 0) break;  /* null terminator */
        output[out_idx++] = tok;
    }
    output[out_idx] = 0;
}
