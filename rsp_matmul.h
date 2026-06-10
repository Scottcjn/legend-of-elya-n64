// SPDX-License-Identifier: MIT
#ifndef RSP_MATMUL_H
#define RSP_MATMUL_H

/*
 * RSP-accelerated matrix-vector multiply for Sophia LLM
 *
 * Uses the RSP's 8-wide SIMD vector unit to accelerate Q8 dequant + matmul.
 * Processes 8 output rows at a time, each computing a 128-element dot product
 * in ~16 vector instructions instead of ~128 scalar multiplies.
 *
 * Estimated speedup: ~8-16x over CPU soft-float for the matmul hot loop.
 * Combined with RPC fallback, this gives:
 *   - RSP local: ~15-30 tok/s (from ~2 tok/s)
 *   - RPC bridge: ~100+ tok/s (limited by serial bandwidth)
 *
 * IMPORTANT: Audio also uses the RSP. This code uses libdragon's rspq
 * overlay system to safely interleave with audio processing.
 * When RSP is busy with audio, matmul waits. This adds ~1ms jitter
 * per audio buffer but doesn't affect audio quality.
 */

#include <stdint.h>

/*
 * Initialize the RSP matmul subsystem.
 * Call once at startup after rspq_init() and audio_init().
 * Returns 1 on success, 0 if RSP is not available.
 */
int rsp_matmul_init(void);

/*
 * RSP-accelerated Q8 matmul: output = W * input
 *
 * weights:  Q8 int8 matrix, row-major [out_dim × in_dim]
 * scales:   float16 scales, one per Q_BLOCK (32) weights
 * input:    float32 input vector [in_dim]
 * output:   float32 output vector [out_dim]
 * in_dim:   input dimension (e.g., 128)
 * out_dim:  output dimension (e.g., 128 or 512)
 *
 * Internally converts float32 input to int16 fixed-point,
 * runs RSP SIMD dot products, converts results back to float32.
 *
 * Falls back to CPU matmul if RSP is unavailable.
 */
void rsp_matmul_q8(const int8_t *weights, const uint16_t *scales,
                   const float *input, float *output,
                   int in_dim, int out_dim);

/*
 * Check if RSP matmul is active and working.
 */
int rsp_matmul_available(void);

#endif
