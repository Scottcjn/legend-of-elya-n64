## SPDX-License-Identifier: MIT
##
## rsp_mm2_ovl.S — rsp_mm2.S rebuilt as an rspq OVERLAY, so the matmul can
## coexist with rdpq instead of displacing it.
##
## The arithmetic is byte-for-byte the kernel from rsp_mm2.S (verified
## 176/176 tokens exact).  Only the plumbing changed:
##
##   1. Entry is an rspq command, not _start.  It returns with `jr ra` into
##      RSPQ_Loop instead of `break`.
##   2. rspq reserves gp (rspq_dmem_buf_ptr).  The superblock counter moved
##      from gp to t8.  The incoming ra is parked in sp (the RSP has no
##      stack, so sp is free) because the DMA helpers use jal.
##   3. rspq's resident state owns DMEM 0x000-0x25F, so the kernel can no
##      longer own the chip.  The big buffers moved to .bss, which n64.mk
##      does NOT extract into the ucode data blob -- that keeps the data
##      DMA'd on every overlay switch down to 48 bytes instead of 3.4 KB.
##   4. The three buffers (activations, weight row, block sums) are no
##      longer at fixed offsets.  The CPU packs them into the scratch window
##      for the shape at hand and passes the offsets in the parameter block.
##      That recovers more rows-per-flush than the old fixed layout had, in
##      every shape this model actually uses except int8 ff2.
##   5. rspq reloads vzero/vshift/vshift8 before every command dispatch, so
##      clobbering $v30/$v31 here is safe.
##
## See FINDINGS.md F-O001..F-O003.

#include <rsp_queue.inc>

.set noreorder
.set at

#define P_MODE        0x00      /* 0 = int8, 1 = ternary                     */
#define P_OUT_DIM     0x04
#define P_W_BASE      0x08      /* RDRAM address of row 0                    */
#define P_ROW_STRIDE  0x0C      /* bytes between rows in RDRAM               */
#define P_ROW_DMA     0x10      /* bytes to DMA per row                      */
#define P_NSB         0x14      /* superblocks per row = in_dim/256          */
#define P_OUT_BASE    0x18      /* RDRAM address for the block sums          */
#define P_ROWS_FLUSH  0x1C      /* rows buffered in DMEM before a DMA out    */
#define P_OUT_ROW_B   0x20      /* nsb * 64                                  */
#define P_XI_DMEM     0x24      /* DMEM offset of the activation buffer      */
#define P_WBUF_DMEM   0x28      /* DMEM offset of the weight-row buffer      */
#define P_OUT_DMEM    0x2C      /* DMEM offset of the block-sum buffer       */
#define P_XI_RDRAM    0x30      /* RDRAM address of the staged activations   */
#define P_XI_BYTES    0x34      /* bytes to DMA for the activations          */

#define PB(off) %lo(mm_params) + (off) (zero)

## ── Data segment ─────────────────────────────────────────────────────────
## Everything here is DMA'd in on every overlay switch, so it stays tiny.
    .data

    RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand MMCmd_Matmul, 8      /* 0x0: a1 = RDRAM param block  */
    RSPQ_DefineCommand MMCmd_Layout, 8      /* 0x1: a1 = RDRAM 8-byte out   */
    RSPQ_EndOverlayHeader

    ## The overlay keeps no state across calls: every call is handed a fresh
    ## parameter block.  rspq still requires a non-empty saved state.
    RSPQ_EmptySavedState

    .align 4
mm_kmask:   .half 0xC000, 0xC000, 0xC000, 0xC000
            .half 0xC000, 0xC000, 0xC000, 0xC000

## ── BSS ──────────────────────────────────────────────────────────────────
## Not part of the ucode data blob: never transferred on an overlay switch,
## it only reserves DMEM address space.
    .bss
    .align 3
mm_params:  .space 64
mm_scratch:                                  /* runs to the end of DMEM      */

## ── Text ─────────────────────────────────────────────────────────────────
    .text

## ---------------------------------------------------------------------------
## MMCmd_Layout — report the overlay's own DMEM layout to the CPU.
##
## The CPU has to pack three buffers into whatever DMEM is left after rspq's
## resident state and this overlay's own data.  Rather than hardcode that
## boundary (it moves whenever libdragon's resident state changes size), the
## overlay reports it once at init.
##
##   a1 = RDRAM address of an 8-byte, 8-aligned buffer.
##   [0] = DMEM offset of the scratch window
##   [4] = size of the scratch window in bytes
## ---------------------------------------------------------------------------
MMCmd_Layout:
    move    sp, ra
    li      t2, %lo(mm_scratch)
    sw      t2, PB(0)
    li      t4, 0x1000
    sub     t4, t4, t2
    sw      t4, PB(4)
    li      t0, 7                            ## DMA_SIZE(8, 1)
    li      t1, 8
    move    s0, a1
    jal     DMAOut
    li      s4, %lo(mm_params)
    jr      sp
    nop

## ---------------------------------------------------------------------------
## MMCmd_Matmul — the kernel.  a1 = RDRAM address of the 64-byte param block.
## ---------------------------------------------------------------------------
MMCmd_Matmul:
    move    sp, ra                           ## park the rspq return address

    ## ---- fetch the parameter block ---------------------------------------
    li      t0, 63                           ## DMA_SIZE(64, 1)
    li      t1, 64
    move    s0, a1
    jal     DMAIn
    li      s4, %lo(mm_params)

    ## ---- fetch the staged activation vector ------------------------------
    lw      t1, PB(P_XI_BYTES)
    lw      s0, PB(P_XI_RDRAM)
    addi    t0, t1, -1
    jal     DMAIn
    lw      s4, PB(P_XI_DMEM)

    vxor    $v00, $v00, $v00
    li      t2, %lo(mm_kmask)
    lqv     $v20, 0, t2                      ## v20 = 0xC000 x8

    lw      s7, PB(P_MODE)
    lw      a1, PB(P_OUT_DIM)
    lw      a0, PB(P_W_BASE)
    lw      a2, PB(P_ROW_STRIDE)
    lw      a3, PB(P_NSB)
    lw      s1, PB(P_OUT_BASE)
    lw      s2, PB(P_ROWS_FLUSH)
    lw      s6, PB(P_XI_DMEM)
    lw      v0, PB(P_WBUF_DMEM)
    lw      v1, PB(P_OUT_DMEM)

    li      s3, 0                            ## rows currently buffered
    move    s5, v1                           ## DMEM write pointer

row_loop:
    ## ---- DMA one weight row into WBUF -------------------------------------
    lw      t0, PB(P_ROW_DMA)
    move    t1, t0
    addi    t0, t0, -1                       ## DMA_SIZE(width, 1)
    move    s0, a0
    jal     DMAIn
    move    s4, v0
    move    k0, s4                           ## k0 = weight pointer (misalign-adj)
    move    k1, s6                           ## k1 = activation pointer

    move    t8, a3                           ## t8 = superblocks left
sb_loop:
    li      fp, 2                            ## two half-blocks per superblock
half_loop:
    vmudh   $v31, $v00, $v00                 ## acc = 0 (all slices)

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
    lpv     $v01, 0,  k0                     ## blocks b0..b7, 4 codes per byte
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

    addi    t8, t8, -1
    bnez    t8, sb_loop
    nop

    ## ---- row done ---------------------------------------------------------
    add     a0, a0, a2                       ## next weight row
    addi    s3, s3, 1
    bne     s3, s2, row_next
    nop
    jal     flush_out
    nop
row_next:
    addi    a1, a1, -1
    bnez    a1, row_loop
    nop

    beqz    s3, all_done                     ## trailing partial flush
    nop
    jal     flush_out
    nop

all_done:
    jr      sp                               ## back into RSPQ_Loop
    nop

## ---------------------------------------------------------------------------
## flush_out: DMA the DMEM output buffer to RDRAM, reset the pointer.
## Destroys t0,t1,t2,t4,t5,at,s0,s4.  Preserves ra via t6.
## ---------------------------------------------------------------------------
flush_out:
    move    t6, ra
    lw      t0, PB(P_OUT_ROW_B)
    move    t4, s3
    li      t5, 0
fo_mul:
    add     t5, t5, t0                       ## RSP has no multiply
    addi    t4, t4, -1
    bnez    t4, fo_mul
    nop
    move    t1, t5
    addi    t0, t5, -1
    move    s0, s1
    jal     DMAOut
    move    s4, v1
    add     s1, s1, t5
    move    s5, v1
    li      s3, 0
    jr      t6
    nop
