// SPDX-License-Identifier: MIT
#ifndef RSP_MATMUL_H
#define RSP_MATMUL_H

/*
 * RSP-accelerated matrix-vector multiply for the Sophia LLM.
 *
 * Handles BOTH weight formats:
 *   bits == 8 : one signed int8 per byte
 *   bits == 2 : four ternary codes per byte, MSB first
 *
 * The RSP does the integer O(in_dim*out_dim) work and returns one int32 sum
 * per 32-weight quantization block; the CPU applies the block's float16 scale.
 * See rsp_mm2.S for the lane layout and matmul_rsp2.c for the driver.
 *
 * Weights must first be permuted in place with rsp2_permute_tensor(), then the
 * D-cache written back with rsp2_weights_ready(), or the RSP will DMA stale
 * bytes.
 */

#include <stdint.h>

int  rsp_matmul_init(void);
int  rsp_matmul_available(void);

void rsp2_permute_tensor(uint8_t *w, int in_dim, int out_dim, int bits);
void rsp2_weights_ready(void *base, unsigned long bytes);

void rsp_matmul_pk(const uint8_t *weights, const uint16_t *scales,
                   const float *input, float *output,
                   int in_dim, int out_dim, int bits);

#endif
