// SPDX-License-Identifier: MIT
/* Direct unit test of the bit-packed matmul kernels against a plain reference.
 * The end-to-end quality numbers would hide a bit-order bug as "some quality
 * loss", so decode correctness is checked on its own. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"

#define IN  1024
#define OUT 16

static uint8_t packed[IN * OUT];      /* enough for 8 bits */
static uint16_t scales[IN * OUT / SGAI_Q_BLOCK];
static float in_v[IN], got[OUT], want[OUT];
static int lev[IN * OUT];

static uint16_t f32_to_f16_le(float f)
{
    /* build the 16 bits, then store LE so f16_to_float's HOST_BUILD path reads it */
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000;
    int e = (int)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3FF;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | (31 << 10));
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

int main(void)
{
    int fails = 0;
    for (int bits = 2; bits <= 6; bits++) {
        srand(1234 + bits);
        int qmax = (1 << (bits - 1)) - 1;
        int qmin = -qmax - 1;
        if (bits == 2) { qmax = 1; qmin = -1; }   /* ternary alphabet */

        int nblk = IN / SGAI_Q_BLOCK;
        for (int i = 0; i < IN; i++) in_v[i] = (float)(rand() % 2001 - 1000) / 250.0f;
        for (int i = 0; i < OUT * nblk; i++)
            scales[i] = f32_to_f16_le((float)(rand() % 900 + 100) / 100000.0f);
        for (int i = 0; i < IN * OUT; i++)
            lev[i] = qmin + rand() % (qmax - qmin + 1);

        /* pack MSB-first, exactly as tools/quantize_n64.py does */
        memset(packed, 0, sizeof(packed));
        {
            long bitpos = 0;
            for (int i = 0; i < IN * OUT; i++) {
                unsigned code = (unsigned)lev[i] & ((1u << bits) - 1u);
                for (int b = bits - 1; b >= 0; b--) {
                    if ((code >> b) & 1u) packed[bitpos >> 3] |= (uint8_t)(0x80u >> (bitpos & 7));
                    bitpos++;
                }
            }
        }

        /* reference */
        for (int o = 0; o < OUT; o++) {
            float acc = 0.0f;
            for (int b = 0; b < nblk; b++) {
                float blk = 0.0f;
                for (int j = 0; j < SGAI_Q_BLOCK; j++) {
                    int idx = o * IN + b * SGAI_Q_BLOCK + j;
                    blk += (float)lev[idx] * in_v[b * SGAI_Q_BLOCK + j];
                }
                acc += blk * f16_to_float(scales[o * nblk + b]);
            }
            want[o] = acc;
        }

        matmul_pk(packed, scales, in_v, got, IN, OUT, bits);

        double worst = 0.0;
        for (int o = 0; o < OUT; o++) {
            double d = fabs((double)got[o] - (double)want[o]);
            double r = d / (fabs((double)want[o]) + 1e-9);
            if (r > worst) worst = r;
        }
        printf("bits=%d  worst relative error = %.3e  %s\n",
               bits, worst, worst < 1e-5 ? "PASS" : "FAIL");
        if (!(worst < 1e-5)) fails++;
    }
    printf(fails ? "KERNEL TEST FAILED\n" : "KERNEL TEST PASSED\n");
    return fails != 0;
}
