// SPDX-License-Identifier: MIT
/*
 * matmul_rsp2.c — CPU-side driver for rsp_mm2.S.
 *
 * The RSP does the O(in_dim * out_dim) integer work and returns, per output
 * row, one int32 sum per 32-weight quantization block.  The CPU then applies
 * that block's float16 scale — exactly one float multiply per 32 weights,
 * which is the same arithmetic shape as the CPU kernel's OPT_HOIST_SCALE.
 *
 * The only numerical difference from the CPU kernel is that the activation
 * vector is quantized to int16 first; everything downstream is exact integer
 * arithmetic on the RSP.
 *
 * Weights must be permuted into the RSP's 8-blocks-per-lane order first; see
 * rsp2_permute_tensor(), called once at load time from sgai_init().
 */

#include "rsp_matmul.h"
#include "nano_gpt.h"
#include <rsp.h>
#include <string.h>
#include <libdragon.h>

/* RSP_MM_OVERLAY selects the rspq-overlay kernel (rsp_mm2_ovl.S), which
 * coexists with rdpq.  Without it we get the original standalone kernel
 * (rsp_mm2.S), which calls rsp_load() and therefore evicts the rspq
 * microcode rdpq_init() installed -- correct arithmetic, but nothing
 * renders afterwards.  Both are kept so the overlay's dispatch overhead can
 * be measured against the standalone rather than assumed.  See FINDINGS.md. */
#ifdef RSP_MM_OVERLAY
#include <rspq.h>
DEFINE_RSP_UCODE(rsp_mm2_ovl);
#else
DEFINE_RSP_UCODE(rsp_mm2);
#endif

#define P_MODE        0x00
#define P_OUT_DIM     0x04
#define P_W_BASE      0x08
#define P_ROW_STRIDE  0x0C
#define P_ROW_DMA     0x10
#define P_NSB         0x14
#define P_OUT_BASE    0x18
#define P_ROWS_FLUSH  0x1C
#define P_OUT_ROW_B   0x20
/* Overlay-only: the buffers are no longer at fixed DMEM offsets, because
 * rspq's resident state owns the bottom of DMEM and what is left has to be
 * packed per shape. */
#define P_XI_DMEM     0x24
#define P_WBUF_DMEM   0x28
#define P_OUT_DMEM    0x2C
#define P_XI_RDRAM    0x30
#define P_XI_BYTES    0x34

#define DM_XI  0x060
#define DMEM_END 0x1000u

/* Constant factor the kernel bakes into each product:
 *   int8    : LPV yields w<<8            -> 256
 *   ternary : VAND 0xC000 leaves code<<14 -> 16384                            */
#define K_INT8   256.0f
#define K_TERN   16384.0f

/* Activation clamp.  Chosen so the 32-bit accumulator field CANNOT overflow
 * even in the adversarial worst case (16 terms per extraction):
 *   int8    : 256   * 127 * 2047 * 16 = 1.065e9  < 2^31
 *   ternary : 16384 *   2 * 4095 * 16 = 2.146e9  < 2^31                       */
#define XMAX_INT8   2047.0f
#define XMAX_TERN   4095.0f

#define RSP2_MAX_IN    1024
#define RSP2_MAX_OUT   1024
/* readback: out_dim * nsb * 32 int16.  Worst cases are (256->1024, nsb 1) and
 * (1024->256, nsb 4), both 32768 int16 = 64 KiB. */
#define RSP2_RB_HALVES 32768

static int16_t  rsp2_xi[RSP2_MAX_IN]     __attribute__((aligned(16)));
static int16_t  rsp2_rb[RSP2_RB_HALVES]  __attribute__((aligned(16)));
static uint32_t rsp2_params[16]          __attribute__((aligned(16)));

static int rsp2_ready = 0;

uint32_t rsp_mm_calls_rsp = 0;
uint32_t rsp_mm_calls_cpu = 0;

/* Phase accounting, CP0 counts.  Compiled in only for the measurement build. */
uint32_t rsp_t_stage = 0;   /* quantize + lane-order staging of the input   */
uint32_t rsp_t_disp  = 0;   /* cache ops + DMEM upload + rsp_run (RSP busy) */
uint32_t rsp_t_epi   = 0;   /* invalidate + per-block float scale epilogue  */
uint32_t rsp_n_wdma  = 0;   /* weight-row DMAs the RSP issues              */
uint32_t rsp_b_w     = 0;   /* weight bytes DMA'd in  (KiB)                */
uint32_t rsp_b_out   = 0;   /* block-sum bytes DMA'd out (KiB)             */
#ifdef RSP_PHASE_TIMING
#define CP0_RD(v) asm volatile("mfc0 %0, $9" : "=r"(v))
#else
#define CP0_RD(v) do { (v) = 0; } while (0)
#endif

/* f16 decode — must agree bit-for-bit with nano_gpt.c's f16_to_float. */
static float f16d(uint16_t f16)
{
    f16 = (uint16_t)((f16 >> 8) | (f16 << 8));
    uint32_t sign = (uint32_t)(f16 & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(f16 >> 10) & 0x1Fu;
    uint32_t frac = (uint32_t)f16 & 0x3FFu;
    union { float f; uint32_t i; } u;

    /* OPT: a float16 -> float32 widening is a pure bit move for every normal
     * value, and every weight scale in these blobs is normal.  The arithmetic
     * form below it used to have a `mantissa / (float)(1u << -e)` on the hot
     * path -- a div.s, ~29 cycles on the VR4300 -- executed once per 32-weight
     * block, which is 196,608 times per forward pass.  Bit-identical for
     * normals: (1 + frac/1024) * 2^(exp-15) is exactly the float32 with
     * exponent field exp-15+127 and mantissa frac<<13. */
    if (exp == 0) {                       /* subnormal / zero */
        u.f = (float)frac * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        u.i |= sign;
        return u.f;
    }
    if (exp == 31) {                      /* inf/nan -> 65504.0f, as before */
        u.i = sign | 0x477FE000u;
        return u.f;
    }
    u.i = sign | ((exp + 112u) << 23) | (frac << 13);
    return u.f;
}

#ifdef RSP_MM_OVERLAY

static uint32_t mm_ovl_id;
static uint32_t mm_scratch_base;          /* DMEM offset, 0 = not probed  */
static uint32_t mm_layout_rb[4] __attribute__((aligned(16)));

/* Where the three buffers land inside the overlay's DMEM scratch window. */
typedef struct { uint32_t xi, wbuf, out; int rows_flush; } mm_plan_t;

/* Pack activations, one weight row, and the block-sum window into whatever
 * DMEM is left, keeping every base 16-aligned (LQV/SQV load and store full
 * quadwords only from 16-aligned addresses; the standalone kernel got this
 * for free from its hand-picked fixed offsets).  Returns 0 if the shape
 * cannot fit, in which case the caller falls back to the CPU kernel. */
static int rsp2_plan(int in_dim, int row_bytes, int out_row_b, int out_dim,
                     mm_plan_t *p)
{
    if (!mm_scratch_base) return 0;

    uint32_t xi   = (mm_scratch_base + 15u) & ~15u;
    uint32_t wbuf = (xi + (uint32_t)in_dim * 2u + 15u) & ~15u;
    /* +8 because DMAIn transfers row_bytes+8 to absorb RDRAM misalignment,
     * +8 more of slack, matching the standalone kernel's 1040-for-1024. */
    uint32_t out  = (wbuf + (uint32_t)row_bytes + 16u + 15u) & ~15u;

    if (out + (uint32_t)out_row_b > DMEM_END) return 0;

    int rows = (int)((DMEM_END - out) / (uint32_t)out_row_b);
    if (rows > out_dim) rows = out_dim;

    p->xi = xi; p->wbuf = wbuf; p->out = out; p->rows_flush = rows;
    return 1;
}

int rsp_matmul_init(void)
{
    /* rdpq_init() has already brought rspq up; registering an overlay adds
     * the matmul alongside it instead of replacing it. */
    mm_ovl_id = rspq_overlay_register(&rsp_mm2_ovl);

    /* Ask the overlay where its own scratch window starts rather than
     * hardcoding a boundary that moves whenever libdragon's resident state
     * changes size.  MMCmd_Layout (0x1) DMAs it back to us. */
    data_cache_hit_writeback_invalidate(mm_layout_rb, sizeof(mm_layout_rb));
    rspq_write(mm_ovl_id, 0x1, 0, (uint32_t)(uintptr_t)mm_layout_rb);
    rspq_wait();
    data_cache_hit_invalidate(mm_layout_rb, sizeof(mm_layout_rb));

    mm_scratch_base = mm_layout_rb[0];
    uint32_t reported_size = mm_layout_rb[1];

    /* Sanity-check what came back before trusting it with DMA addresses. */
    rsp2_ready = (mm_scratch_base >= 0x100u && mm_scratch_base < DMEM_END &&
                  reported_size == DMEM_END - mm_scratch_base);
    if (!rsp2_ready) mm_scratch_base = 0;
    return rsp2_ready;
}

#else  /* standalone kernel: owns the whole RSP, evicts rspq */

int rsp_matmul_init(void)
{
    rsp_init();
    rsp_load(&rsp_mm2);
    rsp2_ready = 1;
    return 1;
}

#endif

int rsp_matmul_available(void) { return rsp2_ready; }

/* ---------------------------------------------------------------------------
 * Weight permutation, applied once in place at load time.
 *
 * The kernel wants the eight blocks of a 256-weight superblock interleaved
 * byte-wise:  new[i*8 + b] = old[b*G + i],  G = 32 (int8) or 8 (ternary).
 * It is a pure permutation of the same bytes — no value changes.
 * ------------------------------------------------------------------------- */
void rsp2_permute_tensor(uint8_t *w, int in_dim, int out_dim, int bits)
{
    static uint8_t tmp[256];
    const int G        = (bits == 8) ? 32 : 8;
    const int sbb      = 8 * G;                       /* bytes per superblock */
    const int row_bytes = (bits == 8) ? in_dim : (in_dim / 4);
    const int nsb      = row_bytes / sbb;

    for (int r = 0; r < out_dim; r++) {
        uint8_t *p = w + (size_t)r * row_bytes;
        for (int s = 0; s < nsb; s++) {
            uint8_t *q = p + s * sbb;
            for (int i = 0; i < G; i++)
                for (int b = 0; b < 8; b++)
                    tmp[i * 8 + b] = q[b * G + i];
            memcpy(q, tmp, (size_t)sbb);
        }
    }
}

void rsp2_weights_ready(void *base, unsigned long bytes)
{
    data_cache_hit_writeback_invalidate(base, bytes);
}

/* ---------------------------------------------------------------------------
 * The matmul itself.
 * ------------------------------------------------------------------------- */
void rsp_matmul_pk(const uint8_t *weights, const uint16_t *scales,
                   const float *input, float *output,
                   int in_dim, int out_dim, int bits)
{
    const int nsb = in_dim >> 8;

    int shape_ok = (rsp2_ready && (bits == 8 || bits == 2) &&
                    !(in_dim & 255) && in_dim <= RSP2_MAX_IN &&
                    out_dim <= RSP2_MAX_OUT);
#ifdef RSP_MM_OVERLAY
    /* The overlay only gets the DMEM rspq does not need, so a shape can be
     * arithmetically fine and still not fit.  Decide that here, not after
     * we have already staged the activations. */
    mm_plan_t plan = { 0, 0, 0, 0 };
    if (shape_ok)
        shape_ok = rsp2_plan(in_dim, (bits == 8) ? in_dim : (in_dim / 4),
                             nsb * 64, out_dim, &plan);
#endif

    if (!shape_ok) {
        rsp_mm_calls_cpu++;
        /* Exact CPU fallback, same block-hoisted form as nano_gpt.c. */
        const int nblk = in_dim / SGAI_Q_BLOCK;
        const int row_bytes = (bits == 8) ? in_dim : (in_dim / 4);
        for (int o = 0; o < out_dim; o++) {
            const uint8_t  *rw = weights + (size_t)o * row_bytes;
            const uint16_t *rs = scales  + (size_t)o * nblk;
            float acc = 0.0f;
            for (int b = 0; b < nblk; b++) {
                float blk = 0.0f;
                if (bits == 8) {
                    const int8_t *w8 = (const int8_t *)rw + b * SGAI_Q_BLOCK;
                    for (int j = 0; j < SGAI_Q_BLOCK; j++)
                        blk += (float)w8[j] * input[b * SGAI_Q_BLOCK + j];
                } else {
                    const uint8_t *wp = rw + b * 8;
                    for (int g = 0; g < 8; g++) {
                        uint32_t by = wp[g];
                        for (int k = 0; k < 4; k++) {
                            uint32_t c = (by >> (6 - 2 * k)) & 3u;
                            if (c) {
                                float x = input[b * SGAI_Q_BLOCK + g * 4 + k];
                                if (c == 1u) blk += x; else blk -= x;
                            }
                        }
                    }
                }
                acc += blk * f16d(rs[b]);
            }
            output[o] = acc;
        }
        return;
    }

    rsp_mm_calls_rsp++;
    uint32_t _c0, _c1, _c2, _c3;
    CP0_RD(_c0);

    /* ---- quantize the activation vector -------------------------------- */
    float amax = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        float a = input[i];
        if (a < 0.0f) a = -a;
        if (a > amax) amax = a;
    }
    const float xmax = (bits == 8) ? XMAX_INT8 : XMAX_TERN;
    if (amax < 1e-20f) {
        for (int o = 0; o < out_dim; o++) output[o] = 0.0f;
        return;
    }
    const float sx = xmax / amax;

    /* ---- stage it in the kernel's lane order ---------------------------- */
    /* xi[sb*256 + k*8 + b] = xq[sb*256 + b*32 + k] */
    for (int sb = 0; sb < nsb; sb++) {
        const float *src = input + sb * 256;
        int16_t *dst = rsp2_xi + sb * 256;
        for (int b = 0; b < 8; b++) {
            const float *sblk = src + b * 32;
            for (int k = 0; k < 32; k++) {
                float v = sblk[k] * sx;
                v += (v >= 0.0f) ? 0.5f : -0.5f;
                int q = (int)v;
                if (q >  (int)xmax) q =  (int)xmax;
                if (q < -(int)xmax) q = -(int)xmax;
                dst[k * 8 + b] = (int16_t)q;
            }
        }
    }

    CP0_RD(_c1);

    /* ---- dispatch -------------------------------------------------------- */
    const int row_bytes = (bits == 8) ? in_dim : (in_dim / 4);
    const int out_row_b = nsb * 64;
#ifdef RSP_MM_OVERLAY
    const int rows_flush = plan.rows_flush;
#else
    int rows_flush = 912 / out_row_b;      /* DMEM output window is 912 bytes */
    if (rows_flush < 1) rows_flush = 1;
    if (rows_flush > out_dim) rows_flush = out_dim;
#endif

    rsp2_params[P_MODE / 4]       = (bits == 8) ? 0u : 1u;
    rsp2_params[P_OUT_DIM / 4]    = (uint32_t)out_dim;
    rsp2_params[P_W_BASE / 4]     = (uint32_t)(uintptr_t)weights;
    rsp2_params[P_ROW_STRIDE / 4] = (uint32_t)row_bytes;
    rsp2_params[P_ROW_DMA / 4]    = (uint32_t)(row_bytes + 8);
    rsp2_params[P_NSB / 4]        = (uint32_t)nsb;
    rsp2_params[P_OUT_BASE / 4]   = (uint32_t)(uintptr_t)rsp2_rb;
    rsp2_params[P_ROWS_FLUSH / 4] = (uint32_t)rows_flush;
    rsp2_params[P_OUT_ROW_B / 4]  = (uint32_t)out_row_b;
#ifdef RSP_MM_OVERLAY
    rsp2_params[P_XI_DMEM / 4]    = plan.xi;
    rsp2_params[P_WBUF_DMEM / 4]  = plan.wbuf;
    rsp2_params[P_OUT_DMEM / 4]   = plan.out;
    rsp2_params[P_XI_RDRAM / 4]   = (uint32_t)(uintptr_t)rsp2_xi;
    rsp2_params[P_XI_BYTES / 4]   = (uint32_t)(in_dim * 2);
#endif

    data_cache_hit_writeback_invalidate(rsp2_xi, (unsigned long)(in_dim * 2));
    data_cache_hit_writeback_invalidate(rsp2_params, sizeof(rsp2_params));
    /* Drop the previous call's block sums BEFORE the RSP DMAs new ones in, so
     * no cache line is resident over the transfer.  ares flags the omission by
     * name: "RSP DMA writing to RDRAM address ... which is cached". */
    data_cache_hit_writeback_invalidate(rsp2_rb,
        (unsigned long)((size_t)out_dim * out_row_b));

#ifdef RSP_MM_OVERLAY
    /* The RSP is running the rspq loop, so DMEM cannot be poked directly any
     * more.  The overlay pulls the parameter block in itself, and the
     * parameter block tells it where to find the activations. */
    rspq_write(mm_ovl_id, 0x0, 0, (uint32_t)(uintptr_t)rsp2_params);
    /* A syncpoint, not rspq_wait().  rspq_wait() additionally blocks until
     * the RDP has finished drawing, which is harmless in the headless probe
     * (no RDP work) but would serialise every matmul against the frame in
     * the actual game.  A syncpoint waits for exactly our command. */
    rspq_syncpoint_wait(rspq_syncpoint_new());
#else
    rsp_load_data(rsp2_xi, (unsigned long)(in_dim * 2), DM_XI);
    rsp_load_data(rsp2_params, 64, 0);
    rsp_run();
#endif

    CP0_RD(_c2);
    rsp_n_wdma += (uint32_t)out_dim;
    rsp_b_w    += (uint32_t)(((size_t)out_dim * (row_bytes + 8)) >> 10);
    rsp_b_out  += (uint32_t)(((size_t)out_dim * out_row_b) >> 10);

    data_cache_hit_writeback_invalidate(rsp2_rb,
        (unsigned long)((size_t)out_dim * out_row_b));

    /* ---- epilogue: one float multiply per 32-weight block ---------------- */
    const int nblk = in_dim / SGAI_Q_BLOCK;
    const float inv = 1.0f / (((bits == 8) ? K_INT8 : K_TERN) * sx);
    for (int o = 0; o < out_dim; o++) {
        const int16_t  *base = rsp2_rb + (size_t)o * nsb * 32;
        const uint16_t *rs   = scales  + (size_t)o * nblk;
        float acc = 0.0f;
        for (int sb = 0; sb < nsb; sb++) {
            const int16_t *h0 = base + sb * 32;
            const int16_t *h1 = h0 + 16;
            for (int b = 0; b < 8; b++) {
                int32_t p0 = (int32_t)(((uint32_t)(uint16_t)h0[b] << 16)
                                      | (uint32_t)(uint16_t)h0[8 + b]);
                int32_t p1 = (int32_t)(((uint32_t)(uint16_t)h1[b] << 16)
                                      | (uint32_t)(uint16_t)h1[8 + b]);
                acc += ((float)p0 + (float)p1) * f16d(rs[sb * 8 + b]);
            }
        }
        output[o] = acc * inv;
    }
    CP0_RD(_c3);
    rsp_t_stage += _c1 - _c0;
    rsp_t_disp  += _c2 - _c1;
    rsp_t_epi   += _c3 - _c2;
}
