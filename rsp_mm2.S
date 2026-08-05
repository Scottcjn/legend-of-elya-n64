## SPDX-License-Identifier: MIT
##
## rsp_mm2.S — RSP matrix-vector kernel for the Sophia LLM, int8 and ternary.
##
## Structure (see FINDINGS F-R007):
##   The eight vector lanes hold eight DIFFERENT 32-weight quantization blocks,
##   not eight consecutive weights.  Lane b therefore accumulates block b's own
##   integer sum, and there is never a horizontal reduce.  The per-block float16
##   scale is applied by the CPU afterwards, one multiply per block.
##
##   Weights must be pre-permuted so that the eight blocks of a 256-weight
##   "superblock" are interleaved byte-wise (an 8xG transpose, G=32 int8 /
##   G=8 ternary).  The activation vector is staged in the matching order.
##
##   The dot product lives in the accumulator's HI/MID slices via VMADH.  All
##   unpacking uses VAND / VADDC, which write only ACC_LO, so the running sum
##   survives.  VMUD*/VMULF would clobber it and are never used inside the loop.
##
## Accumulated value per lane, per half-block (16 elements):
##   int8    : sum of 256 * w * x        (LPV gives w<<8)
##   ternary : sum of 16384 * code * x   (VAND 0xC000 leaves code at bits 15:14)
## The constant factor is divided out by the CPU.

#include <rsp.inc>

.set noreorder
.set at

#define P_MODE        0x00      /* 0 = int8, 1 = ternary                     */
#define P_OUT_DIM     0x04
#define P_W_BASE      0x08      /* RDRAM address of row 0                    */
#define P_ROW_STRIDE  0x0C      /* bytes between rows in RDRAM               */
#define P_ROW_DMA     0x10      /* bytes to DMA per row (multiple of 8)      */
#define P_NSB         0x14      /* superblocks per row = in_dim/256          */
#define P_OUT_BASE    0x18      /* RDRAM address for the block sums          */
#define P_ROWS_FLUSH  0x1C      /* rows buffered in DMEM before a DMA out    */
#define P_OUT_ROW_B   0x20      /* nsb * 64                                  */

#define DM_CONST   0x040
#define DM_XI      0x060
#define DM_WBUF    0x860
#define DM_OUT     0xC70

## ── Data Segment (this IS the DMEM image) ────────────────────────────────
.data
.align 3
params:      .space 0x40                             ## 0x000 .. 0x03F
kmask:       .half 0xC000, 0xC000, 0xC000, 0xC000
             .half 0xC000, 0xC000, 0xC000, 0xC000    ## 0x040 .. 0x04F
pad_a:       .space 0x10                             ## 0x050 .. 0x05F
xi_area:     .space 2048                             ## 0x060 .. 0x85F
wbuf_area:   .space 1040                             ## 0x860 .. 0xC6F
out_area:    .space 912                              ## 0xC70 .. 0xFFF

## ── Text Segment ─────────────────────────────────────────────────────────
.text 1

.globl _start
_start:
    vxor    $v00, $v00, $v00
    lqv     $v20, DM_CONST, zero         ## v20 = 0xC000 x8

    lw      s7, P_MODE(zero)
    lw      a1, P_OUT_DIM(zero)
    lw      a0, P_W_BASE(zero)
    lw      a2, P_ROW_STRIDE(zero)
    lw      a3, P_NSB(zero)
    lw      s1, P_OUT_BASE(zero)
    lw      s2, P_ROWS_FLUSH(zero)

    li      s3, 0                        ## rows currently buffered
    li      s5, DM_OUT                   ## DMEM write pointer

row_loop:
    ## ---- DMA one weight row into WBUF -------------------------------------
    lw      t0, P_ROW_DMA(zero)
    move    t1, t0
    addi    t0, t0, -1                   ## DMA_SIZE(width, 1)
    move    s0, a0
    jal     DMAIn
    li      s4, DM_WBUF
    move    k0, s4                       ## k0 = weight pointer (misalign-adj)
    li      k1, DM_XI                    ## k1 = activation pointer

    move    gp, a3                       ## gp = superblocks left
sb_loop:
    li      fp, 2                        ## two half-blocks per superblock
half_loop:
    vmudh   $v31, $v00, $v00             ## acc = 0 (all slices)

    bnez    s7, tern_half
    nop

    ## ---- int8: 16 elements x 8 blocks = 128 weights, 128 bytes ------------
    li      t3, 4
i8_loop:
    lpv     $v01, 0,  k0
    lqv     $v02, 0,  k1
    lpv     $v03, 8,  k0
    lqv     $v04, 16, k1
    vmadh   $v30, $v01, $v02
    lpv     $v05, 16, k0
    lqv     $v06, 32, k1
    vmadh   $v30, $v03, $v04
    lpv     $v07, 24, k0
    lqv     $v08, 48, k1
    vmadh   $v30, $v05, $v06
    addi    k0, k0, 32
    addi    t3, t3, -1
    vmadh   $v30, $v07, $v08
    bnez    t3, i8_loop
    addi    k1, k1, 64
    j       half_store
    nop

    ## ---- ternary: 16 elements x 8 blocks = 128 weights, 32 bytes ----------
tern_half:
    li      t3, 4
t2_loop:
    lpv     $v01, 0,  k0                 ## blocks b0..b7, 4 codes per byte
    lqv     $v02, 0,  k1
    vand    $v11, $v01, $v20
    vaddc   $v01, $v01, $v01
    vaddc   $v01, $v01, $v01
    vmadh   $v30, $v11, $v02
    lqv     $v04, 16, k1
    vand    $v12, $v01, $v20
    vaddc   $v01, $v01, $v01
    vaddc   $v01, $v01, $v01
    vmadh   $v30, $v12, $v04
    lqv     $v06, 32, k1
    vand    $v13, $v01, $v20
    vaddc   $v01, $v01, $v01
    vaddc   $v01, $v01, $v01
    vmadh   $v30, $v13, $v06
    lqv     $v08, 48, k1
    vand    $v14, $v01, $v20
    addi    k0, k0, 8
    addi    t3, t3, -1
    vmadh   $v30, $v14, $v08
    bnez    t3, t2_loop
    addi    k1, k1, 64

half_store:
    vsar    $v21, COP2_ACC_HI
    vsar    $v22, COP2_ACC_MD
    sqv     $v21, 0,  s5
    sqv     $v22, 16, s5
    addi    s5, s5, 32

    addi    fp, fp, -1
    bnez    fp, half_loop
    nop

    addi    gp, gp, -1
    bnez    gp, sb_loop
    nop

    ## ---- row done ---------------------------------------------------------
    add     a0, a0, a2                   ## next weight row
    addi    s3, s3, 1
    bne     s3, s2, row_next
    nop
    jal     flush_out
    nop
row_next:
    addi    a1, a1, -1
    bnez    a1, row_loop
    nop

    beqz    s3, all_done                 ## trailing partial flush
    nop
    jal     flush_out
    nop

all_done:
    break
    nop

## ---------------------------------------------------------------------------
## flush_out: DMA the DMEM output buffer to RDRAM, reset the pointer.
## Destroys t0,t1,t2,t4,t5,at,s0,s4.  Preserves ra via t6.
## ---------------------------------------------------------------------------
flush_out:
    move    t6, ra
    lw      t0, P_OUT_ROW_B(zero)
    move    t4, s3
    li      t5, 0
fo_mul:
    add     t5, t5, t0                   ## RSP has no multiply
    addi    t4, t4, -1
    bnez    t4, fo_mul
    nop
    move    t1, t5
    addi    t0, t5, -1
    move    s0, s1
    jal     DMAOut
    li      s4, DM_OUT
    add     s1, s1, t5
    li      s5, DM_OUT
    li      s3, 0
    jr      t6
    nop

#include <rsp_dma.inc>
