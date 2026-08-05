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

#ifdef USE_RSP_MATMUL
#include "rsp_matmul.h"
#endif

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
#ifndef HOST_BUILD
    /* Byte-swap: file is LE, N64 is BE.
     * HOST_BUILD compiles this file natively on a little-endian x86 host as a
     * bit-accurate reference (tools/host_eval.c); there the file's LE half is
     * already in the right order and swapping would corrupt every scale. */
    f16 = (uint16_t)((f16 >> 8) | (f16 << 8));
#endif

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
#ifdef USE_RSP_MATMUL
/* When RSP is available, use the RSP-accelerated version */
static void matmul_q8(const int8_t *weights, const uint16_t *scales,
                      const float *input, float *output,
                      int in_dim, int out_dim)
{
    rsp_matmul_q8(weights, scales, input, output, in_dim, out_dim);
}
#else
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
#ifdef OPT_HOIST_SCALE
            /* `scale` is constant across the whole 32-weight block, so it does
             * not belong in the inner loop: 32 mul.s become 1. GCC will not do
             * this itself — n64.mk passes -fno-associative-math. NOTE this is a
             * reassociation, so it is NOT guaranteed bit-identical; that has to
             * be measured, not assumed. */
            float blk_acc = 0.0f;
            for (int j = blk; j < lim; j++) {
                blk_acc += (float)row_w[j] * input[j];
            }
            acc += blk_acc * scale;
#else
            for (int j = blk; j < lim; j++) {
                acc += (float)row_w[j] * scale * input[j];
            }
#endif
        }
        output[o] = acc;
    }
}
#endif /* USE_RSP_MATMUL */

/* -----------------------------------------------------------------------
 * Bit-packed weight matmuls ("SEQn" blobs)
 *
 * Weights are a big-endian bit stream, MSB first, `bits` bits per weight, in
 * the same flat row-major order as the int8 array they replace.  A
 * quantization block is 32 weights, so it is always exactly 4*bits bytes and
 * never straddles a byte the next block needs.
 *
 * The block scale multiplies ONCE per 32 weights, never inside the inner loop
 * — the same hoist that measured -7.47 % instructions for the Q8 path
 * (OPT_HOIST_SCALE / FINDINGS F26-F27 in the previous session).
 * ----------------------------------------------------------------------- */

/* Ternary specialisation: codes 00 -> 0, 01 -> +1, 11 -> -1 (10 is never
 * emitted by the quantizer and is treated as -1).  No multiply at all in the
 * inner loop: the op is skip / acc += x / acc -= x. */
static void matmul_t2(const uint8_t *w, const uint16_t *scales,
                      const float *input, float *output,
                      int in_dim, int out_dim)
{
    const int row_bytes = in_dim >> 2;
    const int nblk      = in_dim / SGAI_Q_BLOCK;

    for (int o = 0; o < out_dim; o++) {
        const uint8_t  *p     = w      + (size_t)o * row_bytes;
        const uint16_t *row_s = scales + (size_t)o * nblk;
        const float    *xi    = input;
        float acc = 0.0f;

        for (int b = 0; b < nblk; b++) {
            float blk = 0.0f;
            for (int g = 0; g < SGAI_Q_BLOCK / 4; g++) {
                uint32_t byte = *p++;
                uint32_t c;
                c = (byte >> 6) & 3u; if (c) { if (c == 1u) blk += xi[0]; else blk -= xi[0]; }
                c = (byte >> 4) & 3u; if (c) { if (c == 1u) blk += xi[1]; else blk -= xi[1]; }
                c = (byte >> 2) & 3u; if (c) { if (c == 1u) blk += xi[2]; else blk -= xi[2]; }
                c =  byte       & 3u; if (c) { if (c == 1u) blk += xi[3]; else blk -= xi[3]; }
                xi += 4;
            }
            acc += blk * f16_to_float(row_s[b]);
        }
        output[o] = acc;
    }
}

/* Generic width 3..6 (and 7, though the quantizer never emits it).
 * A sliding 32-bit window holds at most 13 bits, so no refill can overflow. */
static void matmul_qn(const uint8_t *w, const uint16_t *scales,
                      const float *input, float *output,
                      int in_dim, int out_dim, int bits)
{
    const int row_bytes = (in_dim * bits) >> 3;
    const int nblk      = in_dim / SGAI_Q_BLOCK;
    const int shift     = 8 - bits;              /* sign-extend via int8_t */
    const uint32_t mask = (1u << bits) - 1u;

    for (int o = 0; o < out_dim; o++) {
        const uint8_t  *p     = w      + (size_t)o * row_bytes;
        const uint16_t *row_s = scales + (size_t)o * nblk;
        const float    *xi    = input;
        float acc = 0.0f;

        for (int b = 0; b < nblk; b++) {
            uint32_t bitbuf = 0;
            int nbits = 0;
            float blk = 0.0f;
            for (int j = 0; j < SGAI_Q_BLOCK; j++) {
                while (nbits < bits) { bitbuf = (bitbuf << 8) | *p++; nbits += 8; }
                nbits -= bits;
                uint32_t code = (bitbuf >> nbits) & mask;
                int v = (int)((int8_t)(code << shift)) >> shift;
                blk += (float)v * xi[j];
            }
            acc += blk * f16_to_float(row_s[b]);
            xi += SGAI_Q_BLOCK;
        }
        output[o] = acc;
    }
}

/* Dispatch on the blob's declared bit width.  bits == 8 falls straight through
 * to the existing matmul_q8 (which is the RSP path when USE_RSP_MATMUL is set),
 * so an 8-bit blob computes bit-identically to before this change. */
#ifdef OPT_INLINE_DISPATCH
__attribute__((always_inline))
static inline
#else
static
#endif
void matmul_pk(const uint8_t *w, const uint16_t *scales,
                      const float *input, float *output,
                      int in_dim, int out_dim, int bits)
{
    if (bits == 8)
        matmul_q8((const int8_t *)w, scales, input, output, in_dim, out_dim);
    else if (bits == 2)
        matmul_t2(w, scales, input, output, in_dim, out_dim);
    else
        matmul_qn(w, scales, input, output, in_dim, out_dim, bits);
}

/* Resolve one layer's six tensors inside a bit-packed blob. */
static void sgai_layer_ptrs(const SGAIState *st, int li, SGAILayerPtrs *out)
{
    static const int elems[SGAI_N_TENSORS] = {
        SGAI_ATTN_ELEMS, SGAI_ATTN_ELEMS, SGAI_ATTN_ELEMS,
        SGAI_ATTN_ELEMS, SGAI_FF_ELEMS,   SGAI_FF_ELEMS
    };
    const int bits = st->w_bits;
    const uint8_t *base = (const uint8_t *)(st->weights + 1)
                        + (size_t)SGAI_VOCAB * SGAI_N_EMBED;
    const size_t wbytes = (size_t)SGAI_LAYER_ELEMS * (size_t)bits / 8u;
    const uint8_t *p  = base + (size_t)li * (wbytes + SGAI_LAYER_SCALE_BYTES);
    const uint8_t *sp = p + wbytes;

    for (int i = 0; i < SGAI_N_TENSORS; i++) {
        out->w[i] = p;
        p += (size_t)elems[i] * (size_t)bits / 8u;
        out->s[i] = (const uint16_t *)(const void *)sp;
        sp += ((size_t)elems[i] / SGAI_Q_BLOCK) * 2u;
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
 * PSE: Physarum Slime Mold Attention Router
 *
 * Maintains per-head "tube conductance" that evolves across tokens.
 * Heads that produce sharp attention grow; flat heads wither.
 * On entropy spike (topic change), conductances reset to explore.
 *
 * Inspired by Physarum polycephalum network optimization.
 * Equivalent to POWER8 PSE vec_perm collapse, adapted for MIPS.
 * ----------------------------------------------------------------------- */

#define PSE_PHYSARUM_REINFORCE  0.1f   /* Conductance growth rate (gentle) */
#define PSE_PHYSARUM_DECAY      0.02f  /* Conductance decay rate (slow) */
#ifndef PSE_PHYSARUM_MIN
#define PSE_PHYSARUM_MIN        0.5f
#endif   /* Never drop below half — tiny model needs all heads */
#ifndef PSE_PHYSARUM_MAX
#define PSE_PHYSARUM_MAX        1.5f
#endif   /* Mild amplification cap */
#define PSE_PHYSARUM_PRUNE      0.0f   /* DISABLED — don't skip heads on 4-head model */
#define PSE_BURST_INTERVAL      8      /* Entropy every 8th token (gentler) */
#define PSE_BURST_STRENGTH      0.02f  /* 2% — subtle seasoning, not a firehose */
#define PSE_BURST_DIMS          8      /* Only 8 of 128 dims */

static struct {
    float conductance[SGAI_N_LAYERS][SGAI_N_HEADS];
    float entropy_ema;
    uint32_t token_counter;
} pse_state;

static void pse_init(void)
{
    for (int l = 0; l < SGAI_N_LAYERS; l++)
        for (int h = 0; h < SGAI_N_HEADS; h++)
            pse_state.conductance[l][h] = 1.0f;
    pse_state.entropy_ema = 0.0f;
    pse_state.token_counter = 0;
}

/* Read CP0 COUNT with xorshift mixing — N64's equivalent of POWER8 mftb */
static inline uint32_t pse_entropy(void)
{
    uint32_t c;
#ifdef BENCH_DET_PSE
    /* Bench builds only: replace the CP0 COUNT read with a pure counter so the
     * engine does identical work every run. See FINDINGS F4 — the shipped path
     * makes both the token stream AND the cycle count irreproducible. */
    static uint32_t det_c = 0x9E3779B9u;
    det_c = det_c * 1664525u + 1013904223u;
    c = det_c;
#else
    asm volatile("mfc0 %0, $9" : "=r"(c));
#endif
    c ^= c << 13;
    c ^= c >> 17;
    c ^= c << 5;
    return c;
}

/* SGAI_PSE_OFF: verification builds only.
 *
 * The numpy oracle (n64qat/eval_qat.py) models the transformer WITHOUT the PSE
 * router — its comment says "PSE conductance pinned to 1.0" and it has no burst
 * entropy.  With PSE live the ROM and the oracle are not computing the same
 * function, so a token mismatch could not be attributed to the weight kernel,
 * which is the thing under test.  This flag pins conductance at 1.0 and drops
 * the burst injection so the two are the same function and the ternary kernel
 * is the only variable.  It changes NOTHING in the shipping build (undefined
 * by default) and is not a "make the test pass" fudge: it makes the test
 * meaningful, and both arms are reported. */

/* Burst entropy injection into activation vector (before logit projection) */
static void pse_burst_inject(float *x, int n_embed)
{
#if defined(SGAI_PSE_OFF) || defined(SGAI_PSE_NO_BURST)
    (void)x; (void)n_embed;
    return;
#else
    pse_state.token_counter++;
    if ((pse_state.token_counter % PSE_BURST_INTERVAL) != 0)
        return;

    uint32_t ent = pse_entropy();

    /* Compute activation RMS for scaling */
    float sum_sq = 0.0f;
    for (int i = 0; i < n_embed; i++)
        sum_sq += x[i] * x[i];

    /* Fast inverse sqrt (Quake III trick) */
    union { float f; uint32_t i; } u;
    u.f = sum_sq / (float)n_embed + 1e-8f;
    u.i = 0x5f3759df - (u.i >> 1);
    float inv_rms = u.f;
    inv_rms = inv_rms * (1.5f - 0.5f * (sum_sq / (float)n_embed) * inv_rms * inv_rms);
    float rms = 1.0f / (inv_rms + 1e-8f);

    float strength = rms * PSE_BURST_STRENGTH;
    for (int i = 0; i < PSE_BURST_DIMS; i++) {
        /* BUG FIX (live in the shipped model): this was `& 0x7F`, which masks
         * the target dimension to 0..127.  The model has been 256-dim since
         * f11042a, so entropy could only ever reach the FIRST HALF of the
         * activation vector — and the comment on PSE_BURST_DIMS still says
         * "Only 8 of 128 dims".  At the 544-dim widths a ternary model would
         * allow it would reach under a quarter.  The `dim %= n_embed` line
         * below could never fire because 0x7F < n_embed. */
#ifdef PSE_OLD_MASK
        int dim = (ent >> (i & 15)) & 0x7F;
        if (dim >= n_embed) dim = dim % n_embed;
#else
        int dim = (int)((ent >> (i & 15)) % (uint32_t)n_embed);
#endif
        float noise = ((ent >> (i + 16)) & 1) ? strength : -strength;
        x[dim] += noise;
        ent = ent * 1664525u + 1013904223u;  /* LCG step */
    }
#endif /* SGAI_PSE_OFF */
}

/* Update Physarum conductances after one attention layer */
static void pse_physarum_update(int layer_idx, const float *sharpness)
{
#if defined(SGAI_PSE_OFF) || defined(SGAI_PSE_NO_ROUTE)
    (void)layer_idx; (void)sharpness;
    return;
#else
    float max_sharp = 0.0f;
    for (int h = 0; h < SGAI_N_HEADS; h++)
        if (sharpness[h] > max_sharp) max_sharp = sharpness[h];
    if (max_sharp < 1e-6f) return;

#ifdef PSE_COND_CENTERED
    /* FIX (candidate B): make the update a true REDISTRIBUTION.  The shipped
     * rule compares each head's sharpness to the FIXED threshold 0.5 with a 5:1
     * reinforce:decay asymmetry, so the sum of the deltas is almost always
     * positive and the conductances ratchet upward without bound until they
     * hit the +1.5 clamp.  Comparing to the MEAN with a single symmetric rate
     * makes the deltas sum to exactly zero: heads still trade influence, but the
     * attention branch keeps unit net gain, which is what the model was
     * trained with. */
    float mean_norm = 0.0f;
    for (int h = 0; h < SGAI_N_HEADS; h++)
        mean_norm += sharpness[h] / max_sharp;
    mean_norm /= (float)SGAI_N_HEADS;
#endif
    for (int h = 0; h < SGAI_N_HEADS; h++) {
        float norm = sharpness[h] / max_sharp;
        float *cond = &pse_state.conductance[layer_idx][h];
#ifdef PSE_COND_CENTERED
        *cond += PSE_PHYSARUM_REINFORCE * (norm - mean_norm);
#else
        if (norm > 0.5f)
            *cond += PSE_PHYSARUM_REINFORCE * (norm - 0.5f);
        else
            *cond -= PSE_PHYSARUM_DECAY * (0.5f - norm);
#endif
        if (*cond < PSE_PHYSARUM_MIN) *cond = PSE_PHYSARUM_MIN;
        if (*cond > PSE_PHYSARUM_MAX) *cond = PSE_PHYSARUM_MAX;
    }

#ifdef PSE_COND_NORMALIZE
    /* FIX: the router is meant to REDISTRIBUTE attention between heads, not to
     * apply a net gain to the attention branch.  As written the update is a
     * one-way ratchet -- the sharpest head always has norm == 1.0 so it is
     * always reinforced, and REINFORCE (0.1) is 5x DECAY (0.02) -- so the mean
     * conductance walks from 1.0 (the trained model) to ~1.2 and half the heads
     * weld to the +1.5 clamp within 30 tokens.  Renormalising to mean 1.0 keeps
     * the relative routing and removes the net drift. */
    {
        float sum = 0.0f;
        for (int h = 0; h < SGAI_N_HEADS; h++)
            sum += pse_state.conductance[layer_idx][h];
        if (sum > 1e-6f) {
            float inv = (float)SGAI_N_HEADS / sum;
            for (int h = 0; h < SGAI_N_HEADS; h++)
                pse_state.conductance[layer_idx][h] *= inv;
        }
    }
#endif
#endif /* SGAI_PSE_OFF */
}

/* Check for entropy spike = topic change → reset to exploration */
static void pse_physarum_check_reset(const float *logits)
{
#if defined(SGAI_PSE_OFF) || defined(SGAI_PSE_NO_ROUTE)
    (void)logits;
    return;
#else
    /* Count active candidates as entropy proxy */
    float mx = logits[32];
    for (int i = 33; i <= 126; i++)
        if (logits[i] > mx) mx = logits[i];
    float count = 0.0f;
    for (int i = 32; i <= 126; i++)
        if (logits[i] > mx - 10.0f) count += 1.0f;
    float ent = count / 95.0f;

    float delta = ent - pse_state.entropy_ema;
    pse_state.entropy_ema = pse_state.entropy_ema * 0.9f + ent * 0.1f;

#ifdef PSE_RESET_TRACE
    extern long pse_reset_count; extern float pse_last_delta;
    pse_last_delta = delta;
    if (delta > 0.5f) pse_reset_count++;
#endif
    if (delta > 0.5f) {
        /* Topic change — reset all tubes to exploration */
        for (int l = 0; l < SGAI_N_LAYERS; l++)
            for (int h = 0; h < SGAI_N_HEADS; h++)
                pse_state.conductance[l][h] = 1.0f;
    }
#endif /* SGAI_PSE_OFF */
}


#ifdef MAG_TRACE
/* MAG_TRACE: per-token tensor magnitude / degeneracy statistics.  Host-only. */
struct mag_stats {
    double ff_zero, ff_n;          /* ReLU zero fraction over all layers */
    double x_absmax, attn_absmax;  /* peak magnitudes */
    double x_absmean, x_n;
    long   denorm, nonfinite;      /* degenerate float counts in x */
} mag;
static void mag_scan(const float *v, int n)
{
    for (int i = 0; i < n; i++) {
        float f = v[i]; float a = f < 0 ? -f : f;
        if (a > mag.x_absmax) mag.x_absmax = a;
        mag.x_absmean += a; mag.x_n += 1;
        union { float f; unsigned int u; } b; b.f = f;
        unsigned int e = (b.u >> 23) & 0xFF;
        if (e == 0   && (b.u & 0x7FFFFF)) mag.denorm++;
        if (e == 0xFF) mag.nonfinite++;
    }
}
#endif

/* -----------------------------------------------------------------------
 * Attention + FFN layer forward pass (float32)
 * With PSE: Physarum head routing + sparse FFN + weight skip
 * ----------------------------------------------------------------------- */
#ifdef SGAI_KV_INT8
/* Quantize one K and one V vector into the int8 cache, per head.
 * s = max|x| / 127 over the head's 32 dims; a head that is entirely zero gets
 * scale 0 and stores zeros, which dequantizes back to exactly zero. */
static void kv_store_q8(SGAIKVCache *kv, int layer_idx, int pos,
                        const float *k_cur, const float *v_cur)
{
    const int span = SGAI_N_EMBED / SGAI_KV_NSCALE;   /* 32 per head, or 256 */
    for (int h = 0; h < SGAI_KV_NSCALE; h++) {
        const int base = h * span;
        float kmax = 0.0f, vmax = 0.0f;
        for (int d = 0; d < span; d++) {
            float a = k_cur[base + d]; if (a < 0.0f) a = -a; if (a > kmax) kmax = a;
            float b = v_cur[base + d]; if (b < 0.0f) b = -b; if (b > vmax) vmax = b;
        }
        float ks = kmax * (1.0f / 127.0f);
        float vs = vmax * (1.0f / 127.0f);
        float kinv = (ks > 0.0f) ? (127.0f / kmax) : 0.0f;
        float vinv = (vs > 0.0f) ? (127.0f / vmax) : 0.0f;
        kv->ks[layer_idx][pos][h] = ks;
        kv->vs[layer_idx][pos][h] = vs;
        for (int d = 0; d < span; d++) {
            /* round-to-nearest without a float->int cast helper: the R4300i
             * path in this file deliberately avoids trunc.w.s, but a plain
             * (int) cast on a value already clamped to +-127.5 is safe. */
            float kq = k_cur[base + d] * kinv;
            float vq = v_cur[base + d] * vinv;
            kq += (kq >= 0.0f) ? 0.5f : -0.5f;
            vq += (vq >= 0.0f) ? 0.5f : -0.5f;
            int ki = (int)kq, vi = (int)vq;
            if (ki >  127) ki =  127;
            if (ki < -127) ki = -127;
            if (vi >  127) vi =  127;
            if (vi < -127) vi = -127;
            kv->k[layer_idx][pos][base + d] = (int8_t)ki;
            kv->v[layer_idx][pos][base + d] = (int8_t)vi;
        }
    }
}
#endif /* SGAI_KV_INT8 */

static void attention_layer(const SGAILayerPtrs *L, int bits, SGAIKVCache *kv,
                            int layer_idx, int pos, float *x)
{
    static float q[SGAI_N_EMBED];
    static float k_cur[SGAI_N_EMBED];
    static float v_cur[SGAI_N_EMBED];
    static float attn_out[SGAI_N_EMBED];
    static float ff_buf[SGAI_N_EMBED * 4];
    static float attn_scores[SGAI_CTX];
    static float residual[SGAI_N_EMBED];
    float head_sharpness[SGAI_N_HEADS];

    /* Save residual */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(float));

    /* Layer norm */
    rms_norm(x, SGAI_N_EMBED);

    /* Q, K, V projections */
    matmul_pk(L->w[0], L->s[0], x, q,     SGAI_N_EMBED, SGAI_N_EMBED, bits);
    matmul_pk(L->w[1], L->s[1], x, k_cur, SGAI_N_EMBED, SGAI_N_EMBED, bits);
    matmul_pk(L->w[2], L->s[2], x, v_cur, SGAI_N_EMBED, SGAI_N_EMBED, bits);

    /* Store K, V in cache */
    if (pos < SGAI_CTX) {
#ifdef SGAI_KV_INT8
        kv_store_q8(kv, layer_idx, pos, k_cur, v_cur);
#else
        memcpy(kv->k[layer_idx][pos], k_cur, SGAI_N_EMBED * sizeof(float));
        memcpy(kv->v[layer_idx][pos], v_cur, SGAI_N_EMBED * sizeof(float));
#endif
    }

    /* Multi-head attention with PSE Physarum routing */
    memset(attn_out, 0, SGAI_N_EMBED * sizeof(float));
    int n_ctx = (pos + 1 < SGAI_CTX) ? pos + 1 : SGAI_CTX;
    const float inv_sqrt_hd = SGAI_INV_SQRT_HEAD_DIM;

    for (int h = 0; h < SGAI_N_HEADS; h++) {
        float cond = pse_state.conductance[layer_idx][h];

        /* PSE: Physarum prune — skip retracted tubes entirely */
        if (cond < PSE_PHYSARUM_PRUNE) {
            head_sharpness[h] = 0.0f;
            continue;  /* Saves softmax + V-weighted-sum for this head */
        }

        const float *q_head = q + h * SGAI_HEAD_DIM;

        /* Attention scores + measure sharpness */
        float max_score = -1e9f;
        float sum_score = 0.0f;

        for (int t = 0; t < n_ctx; t++) {
#ifdef SGAI_KV_INT8
            /* One dequant scale per (layer, t, head): hoisted out of the dot
             * product entirely, exactly like the Q8 block scale. */
            const int8_t *k_head = kv->k[layer_idx][t] + h * SGAI_HEAD_DIM;
            float score = 0.0f;
            for (int d = 0; d < SGAI_HEAD_DIM; d++)
                score += q_head[d] * (float)k_head[d];
            score *= kv->ks[layer_idx][t][h / (SGAI_N_HEADS / SGAI_KV_NSCALE)];
#else
            const float *k_head = kv->k[layer_idx][t] + h * SGAI_HEAD_DIM;
            float score = 0.0f;
            for (int d = 0; d < SGAI_HEAD_DIM; d++)
                score += q_head[d] * k_head[d];
#endif
            attn_scores[t] = score * inv_sqrt_hd;
            if (attn_scores[t] > max_score) max_score = attn_scores[t];
            sum_score += attn_scores[t];
        }

        head_sharpness[h] = max_score - (sum_score / (float)n_ctx);

        /* Softmax */
        softmax_f(attn_scores, n_ctx);

        /* V-weighted-sum scaled by Physarum tube conductance */
#ifdef SGAI_KV_INT8
        /* Fold the per-(t,head) V dequant scale into the attention weight once,
         * instead of once per (t,d).  32 multiplies become 1 per timestep. */
        static float vw[SGAI_CTX];
        for (int t = 0; t < n_ctx; t++)
            vw[t] = attn_scores[t] * kv->vs[layer_idx][t][h / (SGAI_N_HEADS / SGAI_KV_NSCALE)];
        for (int d = 0; d < SGAI_HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < n_ctx; t++)
                acc += vw[t] * (float)kv->v[layer_idx][t][h * SGAI_HEAD_DIM + d];
            attn_out[h * SGAI_HEAD_DIM + d] = acc * cond;  /* Hebbian amplify */
        }
#else
        for (int d = 0; d < SGAI_HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < n_ctx; t++)
                acc += attn_scores[t] * kv->v[layer_idx][t][h * SGAI_HEAD_DIM + d];
            attn_out[h * SGAI_HEAD_DIM + d] = acc * cond;  /* Hebbian amplify */
        }
#endif
    }

    /* Update Physarum conductances */
    pse_physarum_update(layer_idx, head_sharpness);

    /* Output projection */
    static float proj_out[SGAI_N_EMBED];
    matmul_pk(L->w[3], L->s[3], attn_out, proj_out, SGAI_N_EMBED, SGAI_N_EMBED, bits);

    /* Residual add */
    for (int i = 0; i < SGAI_N_EMBED; i++)
        x[i] = residual[i] + proj_out[i];

    /* FFN block */
    memcpy(residual, x, SGAI_N_EMBED * sizeof(float));
    rms_norm(x, SGAI_N_EMBED);

    /* ff1: 128 -> 512 + ReLU */
    matmul_pk(L->w[4], L->s[4], x, ff_buf, SGAI_N_EMBED, SGAI_N_EMBED * 4, bits);
    for (int i = 0; i < SGAI_N_EMBED * 4; i++)
        if (ff_buf[i] < 0.0f) ff_buf[i] = 0.0f;
#ifdef MAG_TRACE
    for (int i = 0; i < SGAI_N_EMBED * 4; i++) {
        if (ff_buf[i] == 0.0f) mag.ff_zero += 1;
        mag.ff_n += 1;
    }
    for (int i = 0; i < SGAI_N_EMBED; i++) {
        float a = attn_out[i] < 0 ? -attn_out[i] : attn_out[i];
        if (a > mag.attn_absmax) mag.attn_absmax = a;
    }
    mag_scan(x, SGAI_N_EMBED);
#endif

    /* ff2: 512 -> 128 (dense — sparse column-wise was slower due to cache thrash) */
    static float ff_out[SGAI_N_EMBED];
    matmul_pk(L->w[5], L->s[5], ff_buf, ff_out, SGAI_N_EMBED * 4, SGAI_N_EMBED, bits);

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
#ifdef HOST_BUILD
static uint32_t host_rng_seed_pending = 0;
static void host_rng_reseed(uint32_t v){ host_rng_seed_pending = v ? v : 1u; }
#endif
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
#ifndef SGAI_NO_REP_PENALTY
#ifndef SGAI_REP_HIST
#define SGAI_REP_HIST 3
#endif
    for (int h = 0; h < n_hist && h < SGAI_REP_HIST; h++) {
        uint8_t t = hist[h];
#ifdef SGAI_REP_FACTOR
        /* softened penalty (experiment): scale instead of hard zero */
        if (t >= 32 && t <= 126) probs[t] *= (float)(SGAI_REP_FACTOR);
#else
        if (t >= 32 && t <= 126) probs[t] = 0.0f;
#endif
    }
#else
    (void)hist; (void)n_hist;
#endif

    /* RNG from MIPS CP0 Count register */
    uint32_t rng;
#ifdef HOST_BUILD
    static uint32_t host_rng = 0x12345678u;
    if (host_rng_seed_pending) { host_rng = host_rng_seed_pending; host_rng_seed_pending = 0; }
    host_rng = host_rng * 1664525u + 1013904223u;
    rng = host_rng;
#else
    asm volatile("mfc0 %0, $9" : "=r"(rng));
#endif
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
        /* The magic is four ASCII bytes ("SEAI" or "SEQn"), so read it a byte
         * at a time into a big-endian word.  That is endian-independent and
         * needs no swap heuristics — the old code's `magic == swap32(magic)`
         * dance only worked because "SEAI" happens not to collide with its own
         * byte-reversal, which "SEQn" does not guarantee. */
        const uint8_t *mb = (const uint8_t *)rom_weights;
        uint32_t magic_be = ((uint32_t)mb[0] << 24) | ((uint32_t)mb[1] << 16)
                          | ((uint32_t)mb[2] << 8)  |  (uint32_t)mb[3];
        int bits = 0;
        if (magic_be == SGAI_MAGIC) {
            bits = 8;                                   /* shipped int8 blob */
        } else if ((magic_be & 0xFFFFFF00u) == (uint32_t)(SGAI_MAGIC_SEQ0 & 0xFFFFFF00u)) {
            int n = (int)(magic_be - SGAI_MAGIC_SEQ0);  /* ASCII digit - '0' */
            if (n >= SGAI_MIN_BITS && n <= SGAI_MAX_BITS)
                bits = n;
        }
        if (bits) {
            state->w_bits = bits;
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

    /* PSE: Initialize Physarum slime mold attention router */
    pse_init();

#ifdef USE_RSP_MATMUL
    /* Initialize RSP matmul subsystem */
    if (rsp_matmul_init()) {
        debugf("RSP matmul initialized - SIMD acceleration active\n");
    } else {
        debugf("RSP matmul unavailable - using CPU fallback\n");
    }
#endif
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

    /* PSE: Reset Physarum to exploration mode for new conversation */
    pse_init();
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
        SGAILayerPtrs lp;
        for (int l = 0; l < SGAI_N_LAYERS; l++) {
            sgai_layer_ptrs(state, l, &lp);
            attention_layer(&lp, state->w_bits, state->kv, l, pos, state->x);
        }
    }

    /* 3. Final layer norm */
    rms_norm(state->x, SGAI_N_EMBED);

    /* PSE: Burst entropy injection (N64 CP0 COUNT = POWER8 mftb equivalent) */
    pse_burst_inject(state->x, SGAI_N_EMBED);

    /* 4. Project to logits */
    project_to_logits(state->weights, state->em_scale, state->x, state->logits);

    /* PSE: Check for topic change → Physarum exploration reset */
    pse_physarum_check_reset(state->logits);

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
                memcpy(state->kv->k[l][t], state->kv->k[l][t + 1],
                       SGAI_N_EMBED * sizeof(state->kv->k[0][0][0]));
                memcpy(state->kv->v[l][t], state->kv->v[l][t + 1],
                       SGAI_N_EMBED * sizeof(state->kv->v[0][0][0]));
#ifdef SGAI_KV_INT8
                memcpy(state->kv->ks[l][t], state->kv->ks[l][t + 1],
                       SGAI_KV_NSCALE * sizeof(float));
                memcpy(state->kv->vs[l][t], state->kv->vs[l][t + 1],
                       SGAI_KV_NSCALE * sizeof(float));
#endif
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
