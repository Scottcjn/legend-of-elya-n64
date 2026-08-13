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

/* The CPU fallback the driver takes for a shape the RSP cannot tile. */
void rsp_matmul_cpu(const uint8_t *weights, const uint16_t *scales,
                    const float *input, float *output,
                    int in_dim, int out_dim, int bits);

/* ---------------------------------------------------------------------------
 * Split dispatch, for running the RSP CONCURRENTLY with the CPU.
 *
 * rsp_matmul_pk() stages, dispatches and then immediately blocks on a
 * syncpoint, so the CPU spends the whole vector-unit window doing nothing.
 * That is the right shape when there is only one model: there is nothing else
 * for the CPU to do.  With two models there is.
 *
 *   rsp_matmul_begin()   stage + dispatch, returns while the RSP is running
 *   rsp_matmul_pending() 1 if a dispatch is outstanding
 *   rsp_matmul_done()    non-blocking: has the RSP finished it?
 *   rsp_matmul_end()     block until it has, then apply the float scales
 *
 * begin() returns 0 if the shape cannot go to the RSP, in which case NOTHING
 * was dispatched and the caller must do the matmul itself; end() must not be
 * called.  Exactly one dispatch may be outstanding: the staging buffer, the
 * parameter block and the readback buffer are single and static.
 * ------------------------------------------------------------------------- */
int  rsp_matmul_begin(const uint8_t *weights, const float *input,
                      int in_dim, int out_dim, int bits);
int  rsp_matmul_pending(void);
int  rsp_matmul_done(void);
void rsp_matmul_end(const uint16_t *scales, float *output,
                    int in_dim, int out_dim, int bits);

/* CP0 counts spent inside rsp_matmul_end() actually WAITING for the RSP, i.e.
 * the overlap that was NOT achieved.  Zero would be perfect overlap. */
extern uint32_t rsp_t_wait;
/* Dispatches that were already complete when end() was called. */
extern uint32_t rsp_n_free;
extern uint32_t rsp_n_blocked;

#endif
