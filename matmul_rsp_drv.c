// SPDX-License-Identifier: MIT
/*
 * RSP-accelerated matmul — CPU-side driver
 *
 * Converts float32 → int16 fixed-point, DMA's to RSP DMEM,
 * dispatches vector matmul microcode, reads back results.
 *
 * Processes 8 output rows per RSP dispatch (8-wide SIMD).
 */

#include "rsp_matmul.h"
#include <rsp.h>
#include <string.h>
#include <malloc.h>
#include <libdragon.h>

/* RSP ucode definition — rsp_matmul.S compiled by n64.mk build system */
DEFINE_RSP_UCODE(rsp_matmul);

/* DMEM offsets (must match rsp_matmul.S) */
#define DMEM_CMD        0x000
#define DMEM_OUT_DIM    0x004
#define DMEM_IN_DIM     0x008
#define DMEM_READY      0x010
#define DMEM_INPUT      0x020
#define DMEM_WEIGHTS    0x200
#define DMEM_OUTPUT     0x800

#define CMD_IDLE    0
#define CMD_MATMUL  1
#define CMD_HALT    0xFF

#define ROWS_PER_BLOCK  8
#define Q_BLOCK         32

/* Fixed-point scale: int16 range [-32768, 32767].
 * Scale float inputs so max maps to ~8192, leaving headroom. */
#define FP_SCALE   8192.0f

static int rsp_available = 0;

/* 8-byte aligned DMA buffers */
static int16_t input_i16[128] __attribute__((aligned(8)));
static int16_t weight_i16[ROWS_PER_BLOCK * 128] __attribute__((aligned(8)));
static int16_t output_buf[ROWS_PER_BLOCK] __attribute__((aligned(8)));

int rsp_matmul_init(void)
{
    rsp_init();
    rsp_available = 1;
    return 1;
}

int rsp_matmul_available(void)
{
    return rsp_available;
}

/* f16 decode (same as nano_gpt.c) */
static float f16_decode(uint16_t f16)
{
    f16 = (uint16_t)((f16 >> 8) | (f16 << 8));  /* LE→BE swap */
    uint32_t sign = (f16 >> 15) & 1;
    uint32_t exp  = (f16 >> 10) & 0x1F;
    uint32_t frac = f16 & 0x3FF;
    float val;
    if (exp == 0)
        val = (frac / 1024.0f) * (1.0f / 16384.0f);
    else if (exp == 31)
        val = 65504.0f;
    else {
        float mantissa = 1.0f + frac / 1024.0f;
        int e = (int)exp - 15;
        if (e >= 0) val = mantissa * (float)(1u << (unsigned)e);
        else        val = mantissa / (float)(1u << (unsigned)(-e));
    }
    return sign ? -val : val;
}

void rsp_matmul_q8(const int8_t *weights, const uint16_t *scales,
                   const float *input, float *output,
                   int in_dim, int out_dim)
{
    if (!rsp_available || in_dim > 128 || out_dim > 512) {
        /* Fallback: CPU matmul (same as nano_gpt.c matmul_q8) */
        for (int o = 0; o < out_dim; o++) {
            float acc = 0.0f;
            const int8_t   *row_w = weights + o * in_dim;
            const uint16_t *row_s = scales  + o * in_dim / Q_BLOCK;
            for (int blk = 0; blk < in_dim; blk += Q_BLOCK) {
                float scale = f16_decode(row_s[blk / Q_BLOCK]);
                int lim = (blk + Q_BLOCK < in_dim) ? blk + Q_BLOCK : in_dim;
                for (int j = blk; j < lim; j++)
                    acc += (float)row_w[j] * scale * input[j];
            }
            output[o] = acc;
        }
        return;
    }

    /* Convert float32 input to int16 fixed-point */
    float max_abs = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        float a = input[i];
        if (a < 0) a = -a;
        if (a > max_abs) max_abs = a;
    }
    float inp_scale = (max_abs > 1e-6f) ? (FP_SCALE / max_abs) : 1.0f;
    float inp_inv_scale = (max_abs > 1e-6f) ? (max_abs / FP_SCALE) : 1.0f;

    for (int i = 0; i < in_dim; i++) {
        float v = input[i] * inp_scale;
        if (v > 16000.0f) v = 16000.0f;
        if (v < -16000.0f) v = -16000.0f;
        input_i16[i] = (int16_t)v;
    }

    /* Load RSP ucode */
    rsp_load(&rsp_matmul);

    /* DMA input vector to RSP DMEM */
    int input_bytes = in_dim * 2;
    if (input_bytes & 7) input_bytes = (input_bytes + 7) & ~7;
    rsp_load_data(input_i16, input_bytes, DMEM_INPUT);

    /* Process output rows in blocks of ROWS_PER_BLOCK */
    for (int row = 0; row < out_dim; row += ROWS_PER_BLOCK) {
        int block_rows = out_dim - row;
        if (block_rows > ROWS_PER_BLOCK) block_rows = ROWS_PER_BLOCK;

        /* Prepare weight block: sign-extend int8 → int16 and pre-scale */
        for (int r = 0; r < block_rows; r++) {
            /* Compute average scale for this row */
            float row_scale = 0.0f;
            int n_blocks = in_dim / Q_BLOCK;
            for (int b = 0; b < n_blocks; b++)
                row_scale += f16_decode(scales[(row + r) * n_blocks + b]);
            row_scale /= n_blocks;

            /* Sign-extend int8 weights to int16, scaled by row_scale */
            const int8_t *src = weights + (row + r) * in_dim;
            int16_t *dst = weight_i16 + r * in_dim;
            float w_scale = row_scale * 256.0f;  /* Q8 fixed point */
            for (int j = 0; j < in_dim; j++) {
                float wv = (float)src[j] * w_scale;
                if (wv > 16000.0f) wv = 16000.0f;
                if (wv < -16000.0f) wv = -16000.0f;
                dst[j] = (int16_t)wv;
            }
        }

        int wbytes = block_rows * in_dim * 2;
        if (wbytes & 7) wbytes = (wbytes + 7) & ~7;
        rsp_load_data(weight_i16, wbytes, DMEM_WEIGHTS);

        /* Set command parameters */
        uint32_t params[5] __attribute__((aligned(8)));
        params[0] = CMD_MATMUL;
        params[1] = block_rows;
        params[2] = in_dim;
        params[3] = 0;
        params[4] = 0;  /* result_ready = 0 */
        rsp_load_data(params, 24, DMEM_CMD);

        /* Run RSP and wait */
        rsp_run();

        /* Read results back */
        int obytes = block_rows * 2;
        if (obytes & 7) obytes = (obytes + 7) & ~7;
        rsp_read_data(output_buf, obytes, DMEM_OUTPUT);

        /* Convert int16 results back to float32 */
        for (int r = 0; r < block_rows; r++) {
            /* vmulf produces fractional result (÷32768), undo that */
            output[row + r] = (float)output_buf[r] * inp_inv_scale / 256.0f;
        }
    }
}
