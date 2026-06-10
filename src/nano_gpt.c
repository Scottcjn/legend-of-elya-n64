// SPDX-License-Identifier: MIT
/*
 * nano_gpt.c - Sophia Elya AI: World's First N64 LLM
 *
 * Float32 inference engine for N64 (MIPS R4300i, -msoft-float)
 * Model: 4 layers, 128 embedding dim, 4 heads, vocab=256, ctx=64
 * Weights: Q8 (int8 per weight) + float16 scales, dequantized on-the-fly
 * Activations: float32 (software FPU emulation)
 *
 * Memory budget (8MB RDRAM):
 *   - Weights in ROM (DMA'd on demand): ~848KB
 *   - KV cache (float32): ~256KB
 *   - Activations scratch: ~7KB
 *   - Total RDRAM: ~263KB
 */

#include "nano_gpt.h"
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
/* NOTE: Do NOT include <math.h> — libm functions use hard-float FPU
 * instructions which crash when called from -msoft-float code.
 * All math implemented below using only integer ops + bit tricks. */
#include <libdragon.h>

/* Byte-swap helpers for LE weight file on BE N64 */
static inline uint16_t swap16(uint16_t x) { return (x >> 8) | (x << 8); }
static inline uint32_t swap32(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00)
         | ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000u);
}

/* -----------------------------------------------------------------------
 * float16 decode
 * Weights stored as IEEE 754 half-precision scales (little-endian in file).
 * N64 is big-endian: byte-swap before decoding.
 * ----------------------------------------------------------------------- */
static float f16_to_float(uint16_t f16)
{
    /* Byte-swap: file is LE, N64 is BE */
    f16 = (uint16_t)((f16 >> 8) | (f16 << 8));

    uint32_t sign = (f16 >> 15) & 1;
    uint32_t exp  = (f16 >> 10) & 0x1F;
    uint32_t frac = f16 & 0x3FF;
    float val;

    if (exp == 0) {
        val = (frac / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        val = 65504.0f;
    } else {
        float mantissa = 1.0f + frac / 1024.0f;
        int e = (int)exp - 15;
        if (e >= 0)
            val = mantissa * (float)(1u << (unsigned)e);
        else
            val = mantissa / (float)(1u << (unsigned)(-e));
    }
    return sign ? -val : val;
}

/* -----------------------------------------------------------------------
 * On-the-fly Q8 dequantization + matmul (float32)
 *
 * Computes: output[od] = W[od x id] * input[id]
 * W is Q8 int8 with float16 scales per 32-weight block.
 * Dequantized weight = int8_val * float16_scale
 * ----------------------------------------------------------------------- */
static void matmul_q8(const int8_t *weights, const uint16_t *scales,
                      const float *input, float *output,
                      int in_dim, int out_dim)
{
    for (int o = 0; o < out_dim; o++) {
        float acc = 0.0f;
        const int8_t   *row_w = weights + o * in_dim;
        const uint16_t *row_s = scales  + o * in_dim / SGAI_Q_BLOCK;

        for (int blk = 0; blk < in_dim; blk += SGAI_Q_BLOCK) {
            float scale = f16_to_float(row_s[blk / SGAI_Q_BLOCK]);
            int lim = (blk + SGAI_Q_BLOCK < in_dim) ? blk + SGAI_Q_BLOCK : in_dim;
            for (int j = blk; j < lim; j++) {
                acc += (float)row_w[j] * scale * input[j];
            }
        }
        output[o] = acc;
    }
}

/* -----------------------------------------------------------------------
 * RMS normalization (no learned parameters)
 * ----------------------------------------------------------------------- */
static void rms_norm(float *vec, int len)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < len; i++)
        sum_sq += vec[i] * vec[i];

    /* Fast inverse sqrt (Quake III trick) — uses only integer ops.
     * No FPU instructions needed: the union reinterprets float bits
     * as integer, does integer math, then reinterprets back.
     * With -msoft-float, float multiply/divide are software calls. */
    float mean_sq = sum_sq / (float)len + 1e-6f;

    union { float f; uint32_t i; } u;
    u.f = mean_sq;
    u.i = 0x5f3759df - (u.i >> 1);  /* Initial guess ≈ 1/sqrt(mean_sq) */
    float inv_rms = u.f;
    /* Two Newton-Raphson iterations: y = y * (1.5 - 0.5*x*y*y) */
    inv_rms = inv_rms * (1.5f - 0.5f * mean_sq * inv_rms * inv_rms);
    inv_rms = inv_rms * (1.5f - 0.5f * mean_sq * inv_rms * inv_rms);

    for (int i = 0; i < len; i++)
        vec[i] *= inv_rms;
}

/* -----------------------------------------------------------------------
 * Softmax (float32, numerically stable)
 * ----------------------------------------------------------------------- */
static void softmax_f(float *vec, int len)
{
    if (len <= 0) return;
    float mx = vec[0];
    for (int i = 1; i < len; i++)
        if (vec[i] > mx) mx = vec[i];

    /* exp() via range reduction + Taylor series.
     * ZERO float-to-int casts — avoids R4300i's missing trunc.w.s.
     * exp(x) = exp(x/128)^128. For |x|<20, |x/128| < 0.156.
     * Taylor degree 4 at |z|<0.16 gives <0.001% error.
     * 7 squarings give exp(x) with <0.1% error. */
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        float x = vec[i] - mx;
        if (x < -20.0f) { vec[i] = 0.0f; continue; }

        float z = x * (1.0f / 128.0f);
        /* Taylor: e^z ≈ 1 + z + z²/2 + z³/6 + z⁴/24 */
        float e = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666667f + z * 0.04166667f)));
        /* Square 7 times: e^128 */
        e = e * e;  e = e * e;  e = e * e;  e = e * e;
        e = e * e;  e = e * e;  e = e * e;
        vec[i] = e;
        sum += e;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < len; i++)
            vec[i] *= inv_sum;
    }
}

/* -----------------------------------------------------------------------
 * Embedding lookup with em scale restoration
 * ----------------------------------------------------------------------- */
static void embed_lookup(const SGAIHeader *hdr, float em_scale,
                         uint8_t token, float *out)
{
    const int8_t *emb_table = (const int8_t *)(hdr + 1);
    int offset = (int)token * SGAI_N_EMBED;
    float scale = em_scale / 127.0f;

    for (int i = 0; i < SGAI_N_EMBED; i++)
        out[i] = (float)emb_table[offset + i] * scale;
}

/* -----------------------------------------------------------------------
 * Attention + FFN layer forward pass (float32)
 * ----------------------------------------------------------------------- */
static void attention_layer(const SGAILayer *layer, SGAIKVCache *kv,
                            int layer_idx, int pos, float *x)
{
    static float q[SGAI_N_EMBED];
    static float k_cur[SGAI_N_EMBED];
    static float v_cur[SGAI_N_EMBED];
    static float attn_out[SGAI_N_EMBED];
    static float ff_buf[SGAI_N_EMBED * 4];
    static float attn_scores[SGAI_CTX];
    static float residual[SGAI_N_EMBED];

    /* Save residual */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(float));

    /* Layer norm */
    rms_norm(x, SGAI_N_EMBED);

    /* Q, K, V projections */
    matmul_q8(layer->wq, layer->sq, x, q,     SGAI_N_EMBED, SGAI_N_EMBED);
    matmul_q8(layer->wk, layer->sk, x, k_cur, SGAI_N_EMBED, SGAI_N_EMBED);
    matmul_q8(layer->wv, layer->sv, x, v_cur, SGAI_N_EMBED, SGAI_N_EMBED);

    /* Store K, V in cache */
    if (pos < SGAI_CTX) {
        memcpy(kv->k[layer_idx][pos], k_cur, SGAI_N_EMBED * sizeof(float));
        memcpy(kv->v[layer_idx][pos], v_cur, SGAI_N_EMBED * sizeof(float));
    }

    /* Multi-head attention */
    memset(attn_out, 0, SGAI_N_EMBED * sizeof(float));
    int n_ctx = (pos + 1 < SGAI_CTX) ? pos + 1 : SGAI_CTX;

    /* 1/sqrt(head_dim) = 1/sqrt(32) ≈ 0.17678 */
    float inv_sqrt_hd = 0.17678f;

    for (int h = 0; h < SGAI_N_HEADS; h++) {
        const float *q_head = q + h * SGAI_HEAD_DIM;

        /* Attention scores */
        for (int t = 0; t < n_ctx; t++) {
            const float *k_head = kv->k[layer_idx][t] + h * SGAI_HEAD_DIM;
            float score = 0.0f;
            for (int d = 0; d < SGAI_HEAD_DIM; d++)
                score += q_head[d] * k_head[d];
            attn_scores[t] = score * inv_sqrt_hd;
        }

        /* Softmax */
        softmax_f(attn_scores, n_ctx);

        /* Weighted sum of V */
        for (int d = 0; d < SGAI_HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < n_ctx; t++)
                acc += attn_scores[t] * kv->v[layer_idx][t][h * SGAI_HEAD_DIM + d];
            attn_out[h * SGAI_HEAD_DIM + d] = acc;
        }
    }

    /* Output projection */
    static float proj_out[SGAI_N_EMBED];
    matmul_q8(layer->wo, layer->so, attn_out, proj_out, SGAI_N_EMBED, SGAI_N_EMBED);

    /* Residual add */
    for (int i = 0; i < SGAI_N_EMBED; i++)
        x[i] = residual[i] + proj_out[i];

    /* FFN block */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(float));
    rms_norm(x, SGAI_N_EMBED);

    /* ff1: 128 -> 512 + ReLU */
    matmul_q8(layer->wff1, layer->sff1, x, ff_buf, SGAI_N_EMBED, SGAI_N_EMBED * 4);
    for (int i = 0; i < SGAI_N_EMBED * 4; i++)
        if (ff_buf[i] < 0.0f) ff_buf[i] = 0.0f;

    /* ff2: 512 -> 128 */
    static float ff_out[SGAI_N_EMBED];
    matmul_q8(layer->wff2, layer->sff2, ff_buf, ff_out, SGAI_N_EMBED * 4, SGAI_N_EMBED);

    /* Residual add */
    for (int i = 0; i < SGAI_N_EMBED; i++)
        x[i] = residual[i] + ff_out[i];
}

/* -----------------------------------------------------------------------
 * Logit projection (tied embedding)
 * ----------------------------------------------------------------------- */
static void project_to_logits(const SGAIHeader *hdr, float em_scale,
                              const float *x, float *logits)
{
    const int8_t *emb_table = (const int8_t *)(hdr + 1);
    float scale = em_scale / 127.0f;

    for (int v = 0; v < SGAI_VOCAB; v++) {
        float acc = 0.0f;
        int offset = v * SGAI_N_EMBED;
        for (int i = 0; i < SGAI_N_EMBED; i++)
            acc += (float)emb_table[offset + i] * scale * x[i];
        logits[v] = acc;
    }
}

/* -----------------------------------------------------------------------
 * Sampling with temperature and repetition penalty
 * ----------------------------------------------------------------------- */
static uint8_t sample_logits(const float *logits, uint32_t temperature_q8,
                             const uint8_t *hist, int n_hist)
{
    if (temperature_q8 == 0) {
        /* Greedy: pure argmax over printable ASCII 32-126.
         * No repetition penalty — matches the proven x86 reference.
         * The model naturally produces varied text without needing it. */
        int best = 32;
        for (int i = 33; i <= 126; i++)
            if (logits[i] > logits[best]) best = i;
        return (uint8_t)best;
    }

    /* Temperature sampling */
    static float probs[SGAI_VOCAB];
    float temp = (float)temperature_q8 / 256.0f;
    if (temp < 0.01f) temp = 0.01f;
    float inv_temp = 1.0f / temp;

    /* Apply temperature, restrict to printable ASCII */
    for (int i = 0; i < SGAI_VOCAB; i++) {
        if (i >= 32 && i <= 126)
            probs[i] = logits[i] * inv_temp;
        else
            probs[i] = -1e9f;
    }

    /* Softmax over printable range */
    softmax_f(probs + 32, 95);

    /* Zero non-printable */
    for (int i = 0; i < 32; i++) probs[i] = 0.0f;
    for (int i = 127; i < SGAI_VOCAB; i++) probs[i] = 0.0f;

    /* Repetition penalty: zero recent tokens */
    for (int h = 0; h < n_hist && h < 3; h++) {
        uint8_t t = hist[h];
        if (t >= 32 && t <= 126) probs[t] = 0.0f;
    }

    /* RNG from MIPS CP0 Count register */
    uint32_t rng;
    asm volatile("mfc0 %0, $9" : "=r"(rng));
    rng ^= rng >> 16;
    rng *= 0x45d9f3b;
    rng ^= rng >> 16;

    /* Multinomial sampling */
    float total = 0.0f;
    for (int i = 32; i <= 126; i++) total += probs[i];
    if (total <= 0.0f) {
        /* Fallback: first non-penalized printable char */
        for (int i = 32; i <= 126; i++) {
            int in_hist = 0;
            for (int h = 0; h < n_hist && h < 3; h++)
                if (hist[h] == (uint8_t)i) { in_hist = 1; break; }
            if (!in_hist) return (uint8_t)i;
        }
        return ' ';
    }

    float r = (float)(rng & 0xFFFF) / 65536.0f * total;
    float csum = 0.0f;
    for (int i = 32; i <= 126; i++) {
        csum += probs[i];
        if (r < csum) return (uint8_t)i;
    }
    return ' ';
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

void sgai_init(SGAIState *state, const void *rom_weights)
{
    memset(state, 0, sizeof(SGAIState));

    if (rom_weights != NULL) {
        const SGAIHeader *hdr = (const SGAIHeader *)rom_weights;
        /* Weight file is written LE by Python. On BE N64, magic reads byte-swapped. */
        uint32_t magic_raw = hdr->magic;
        if (magic_raw == SGAI_MAGIC || magic_raw == swap32(SGAI_MAGIC)) {
            state->weights = hdr;
            state->is_loaded = 1;
            /* em_scale_x16 is uint8_t — no byte swap needed */
            state->em_scale = (float)hdr->em_scale_x16 / 16.0f;
            if (state->em_scale < 0.01f)
                state->em_scale = 3.5f;  /* default for old weight files */
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
    memset(state->penalty_hist, 0, sizeof(state->penalty_hist));
    state->penalty_n = 0;
}

uint8_t sgai_next_token(SGAIState *state, uint8_t input_token,
                        uint32_t temperature_q8)
{
    if (!state->kv) return 0;
    int pos = state->kv->pos;

    /* 1. Embedding lookup */
    embed_lookup(state->weights, state->em_scale, input_token, state->x);

    /* 2. Run transformer layers */
    if (state->is_loaded && state->weights != NULL) {
        const uint8_t *after_hdr = (const uint8_t *)(state->weights + 1);
        size_t emb_table_bytes = SGAI_VOCAB * SGAI_N_EMBED;
        const SGAILayer *layers = (const SGAILayer *)(after_hdr + emb_table_bytes);

        for (int l = 0; l < SGAI_N_LAYERS; l++)
            attention_layer(&layers[l], state->kv, l, pos, state->x);
    }

    /* 3. Final layer norm */
    rms_norm(state->x, SGAI_N_EMBED);

    /* 4. Project to logits */
    project_to_logits(state->weights, state->em_scale, state->x, state->logits);

    /* 5. Sample */
    uint8_t next_tok = sample_logits(state->logits, temperature_q8,
                                     state->penalty_hist, (int)state->penalty_n);

    /* Update penalty history */
    if (temperature_q8 > 0) {
        int new_n = ((int)state->penalty_n < 3) ? (int)state->penalty_n + 1 : 3;
        for (int i = new_n - 1; i > 0; i--)
            state->penalty_hist[i] = state->penalty_hist[i - 1];
        state->penalty_hist[0] = next_tok;
        state->penalty_n = (uint8_t)new_n;
    }

    /* 6. Advance KV cache position */
    if (state->kv->pos < SGAI_CTX - 1) {
        state->kv->pos++;
    } else {
        /* Sliding window: shift KV cache left */
        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            for (int t = 0; t < SGAI_CTX - 1; t++) {
                memcpy(state->kv->k[l][t], state->kv->k[l][t + 1], SGAI_N_EMBED * sizeof(float));
                memcpy(state->kv->v[l][t], state->kv->v[l][t + 1], SGAI_N_EMBED * sizeof(float));
            }
        }
    }

    /* Store token in sequence */
    if (state->seq_len < SGAI_CTX)
        state->tokens[state->seq_len++] = input_token;

    return next_tok;
}

void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8)
{
    sgai_reset(state);

    /* Process prompt tokens */
    uint8_t tok = 0;
    for (int i = 0; i < prompt_len; i++)
        tok = sgai_next_token(state, prompt[i], temperature_q8);

    /* Generate output tokens */
    int out_idx = 0;
    while (out_idx < max_tokens - 1) {
        tok = sgai_next_token(state, tok, temperature_q8);
        if (tok == 0) break;
        output[out_idx++] = tok;
    }
    output[out_idx] = 0;
}
