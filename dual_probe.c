// SPDX-License-Identifier: MIT
/*
 * dual_probe.c — the N64 dual-processor consensus, measured.
 *
 * Three arms over the SAME prompt and the SAME greedy sampler:
 *
 *   TERN   the ternary model on the MIPS core alone
 *   INT8   the int8 model on the RSP alone
 *   VOTE   both at once, the RSP dispatched so the CPU runs underneath it,
 *          logits combined as  Lt + Li >> shift
 *
 * Why this shape.  Measured on this port over 16 tokens (docs/N64_RSP_FINDINGS.md):
 *   CPU int8    1,275,545,400 CP0
 *   CPU ternary   548,354,836 CP0   -57.0 %   ternary wins on the MIPS core
 *   RSP int8      267,472,813 CP0             (RSP ternary is 11.9 % SLOWER)
 * The two processors prefer OPPOSITE formats, so each format is run where it
 * wins and the disagreement between them is the diversity the vote spends.
 *
 * Every number this prints is a count read off the machine: CP0 COUNT for
 * cycles, the VI vblank interrupt for wall time.  The vblank figure is the one
 * that may be quoted as tok/s -- CP0 is synthetic under an emulator and the
 * rate audit (docs/N64_RATE_FINDINGS.md) is what settled that.
 *
 * It is a standalone main() and not another #ifdef inside legend_of_elya.c
 * because it needs TWO models resident, and the game's globals are built
 * around one.
 */
#include <libdragon.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include "nano_gpt.h"
#ifdef USE_RSP_MATMUL
#include "rsp_matmul.h"
#endif

/* ------------------------------------------------------------------ clocks */
static volatile uint32_t g_vbl = 0;
static void vbl_counter(void) { g_vbl++; }
static float g_vi_hz = 59.94f;

#define CP0(x) asm volatile("mfc0 %0, $9" : "=r"(x))

/* IS-Viewer text out.  32-BIT WRITES ONLY: byte stores silently drop three of
 * every four characters (F37). */
static char pl[256];
#define ISV(s) do {                                                        \
    const char *_s = (s);                                                  \
    uint32_t _n = (uint32_t)strlen(_s), _w = (_n + 3) & ~3u;               \
    for (uint32_t _i = 0; _i < _w; _i += 4) {                              \
        uint32_t _v = 0;                                                   \
        for (int _b = 0; _b < 4; _b++) {                                   \
            uint32_t _c = (_i + _b < _n) ? (uint8_t)_s[_i + _b] : 0;       \
            _v |= _c << (24 - 8 * _b);                                     \
        }                                                                  \
        *(volatile uint32_t *)(uintptr_t)(0xB3FF0020ul + _i) = _v;         \
    }                                                                      \
    *(volatile uint32_t *)(uintptr_t)(0xB3FF0014ul) = _n;                  \
} while (0)

/* ------------------------------------------------------------------ config */
#ifndef DUAL_NGEN
#define DUAL_NGEN 16
#endif
#ifndef DUAL_SHIFT
#define DUAL_SHIFT 5
#endif
#ifndef DUAL_PROMPT
#define DUAL_PROMPT "sage says: Who are you?: "
#endif
/* Depth of the int8 arm's blob.  8 layers of int8 (6,750,220 B) plus 8 layers
 * of ternary (2,031,628 B) is 8,781,848 B of weights against 8,388,608 B of
 * RDRAM, before either KV cache, the code, the stack or a framebuffer.  The
 * int8 arm is therefore 4 layers deep and the cost of that is measured, not
 * assumed -- see the host score for the 8-layer pair it cannot ship. */
#ifndef DUAL_INT8_LAYERS
#define DUAL_INT8_LAYERS 4
#endif

/* ------------------------------------------------------------------- state */
static uint8_t wt_buf[SGAI_WEIGHT_BUF_N(2, SGAI_N_LAYERS)] __attribute__((aligned(16)));
static uint8_t wi_buf[SGAI_WEIGHT_BUF_N(8, DUAL_INT8_LAYERS)] __attribute__((aligned(16)));
static SGAIState   ST_T, ST_I;
static SGAIScratch SC_T, SC_I;
static char outbuf[DUAL_NGEN + 1];

static int load_blob(const char *path, uint8_t *dst, int cap, int *out_sz)
{
    int fd = dfs_open(path);
    if (fd < 0) { *out_sz = -1; return 0; }
    int sz = dfs_size(fd);
    *out_sz = sz;
    if (sz <= 0 || sz > cap) { dfs_close(fd); return 0; }
    dfs_read(dst, 1, sz, fd);
    dfs_close(fd);
    return 1;
}

/* RMS of a logit vector over the sampler's own band, x1000 as an integer so
 * no float formatting is involved.  The ratio between the two arms' RMS is
 * what the shift has to cancel; quoting it from the ROM means the shift is
 * justified by THIS build's blobs and not by the host's. */
static uint32_t logit_rms_x1000(const float *lg)
{
    float acc = 0.0f;
    for (int i = 32; i <= 126; i++) acc += lg[i] * lg[i];
    acc = acc / 95.0f;
    /* integer sqrt of the x1e6 fixed-point value, no libm */
    float r = acc, g = 1.0f;
    for (int i = 0; i < 40; i++) { g = 0.5f * (g + r / g); if (g <= 0.0f) break; }
    return (uint32_t)(g * 1000.0f + 0.5f);
}

#ifdef USE_RSP_MATMUL
struct rspsnap { uint32_t wait, freec, blocked, stage, disp, epi, calls_rsp, calls_cpu; };
static void snap(struct rspsnap *s)
{
    extern uint32_t rsp_t_stage, rsp_t_disp, rsp_t_epi;
    extern uint32_t rsp_mm_calls_rsp, rsp_mm_calls_cpu;
    s->wait = rsp_t_wait; s->freec = rsp_n_free; s->blocked = rsp_n_blocked;
    s->stage = rsp_t_stage; s->disp = rsp_t_disp; s->epi = rsp_t_epi;
    s->calls_rsp = rsp_mm_calls_rsp; s->calls_cpu = rsp_mm_calls_cpu;
}
static void snap_diff(const struct rspsnap *a, const struct rspsnap *b,
                      struct rspsnap *d)
{
    d->wait = b->wait - a->wait; d->freec = b->freec - a->freec;
    d->blocked = b->blocked - a->blocked; d->stage = b->stage - a->stage;
    d->disp = b->disp - a->disp; d->epi = b->epi - a->epi;
    d->calls_rsp = b->calls_rsp - a->calls_rsp;
    d->calls_cpu = b->calls_cpu - a->calls_cpu;
}
#endif

/* mode: 0 = ternary alone, 1 = int8 alone, 2 = the vote (overlapped),
 *       3 = the vote with the RSP NOT overlapped (the control) */
static void run_arm(const char *name, int mode)
{
    const char *p0 = DUAL_PROMPT;
    const int P = (int)strlen(p0);
    uint64_t cp0_gen = 0;
    uint32_t t0, t1, vb0, vb1, vp0, vp1;
    uint64_t cp0_prompt = 0;
    uint8_t tok = 0;
#ifdef USE_RSP_MATMUL
    struct rspsnap s0, s1, sd;
#endif

    sgai_reset(&ST_T);
    sgai_reset(&ST_I);
#ifdef USE_RSP_MATMUL
    sgai_dual_serial = (mode == 3) ? 1 : 0;
#endif

    /* Prompt ingestion, timed separately and never folded into tok/s: it is
     * one forward pass per prompt byte with the output discarded, and folding
     * it in halves the reported rate (F-RT004). */
    vp0 = g_vbl; CP0(t0);
    for (int i = 0; i < P; i++) {
        uint8_t c = (uint8_t)p0[i];
        if (mode == 0)      tok = sgai_next_token(&ST_T, c, 0);
        else if (mode == 1) tok = sgai_next_token(&ST_I, c, 0);
        else                tok = sgai_dual_next_token(&ST_T, &ST_I, DUAL_SHIFT, c, 0);
    }
    CP0(t1); vp1 = g_vbl;
    cp0_prompt = (uint32_t)(t1 - t0);

    sprintf(pl, "DUAL %s PROMPT toks=%d cp0=%llu vbl=%u\n", name, P,
            (unsigned long long)cp0_prompt, (unsigned)(vp1 - vp0));
    ISV(pl);

    /* The RSP counters are snapshotted HERE, after prompt ingestion, so the
     * phase breakdown covers exactly the DUAL_NGEN generated tokens the tok/s
     * figure covers.  Taking it before the prompt folded 25 extra forward
     * passes into every phase (the first run did; F-DU002's phase table is
     * over 41 passes, not 16). */
#ifdef USE_RSP_MATMUL
    snap(&s0);
#endif
    sprintf(pl, "DUAL %s GEN_START\n", name); ISV(pl);
    vb0 = g_vbl;
    for (int i = 0; i < DUAL_NGEN; i++) {
        /* Store the prediction carried IN, then step -- which is exactly what
         * the host reference does, so the two transcripts are directly
         * comparable instead of off by one (F-DU002 had to correct for it by
         * hand). */
        outbuf[i] = (char)tok;
        CP0(t0);
        if (mode == 0)      tok = sgai_next_token(&ST_T, tok, 0);
        else if (mode == 1) tok = sgai_next_token(&ST_I, tok, 0);
        else                tok = sgai_dual_next_token(&ST_T, &ST_I, DUAL_SHIFT, tok, 0);
        CP0(t1);
        cp0_gen += (uint32_t)(t1 - t0);
    }
    vb1 = g_vbl;
    outbuf[DUAL_NGEN] = 0;

#ifdef USE_RSP_MATMUL
    snap(&s1); snap_diff(&s0, &s1, &sd);
#endif

    sprintf(pl, "DUAL %s GEN toks=%d cp0=%llu vbl=%u\n", name, DUAL_NGEN,
            (unsigned long long)cp0_gen, (unsigned)(vb1 - vb0));
    ISV(pl);
    sprintf(pl, "DUAL %s tps_vbl_x1000=%u tps_cp0_x1000=%u\n", name,
            (unsigned)((vb1 - vb0) ? (uint64_t)DUAL_NGEN
                                     * (uint32_t)(g_vi_hz * 1000.0f)
                                     / (vb1 - vb0) : 0u),
            (unsigned)(cp0_gen ? (uint64_t)DUAL_NGEN * 46875000ull * 1000ull
                                 / cp0_gen : 0u));
    ISV(pl);

    /* The logit scales, from this run's final step. */
    sprintf(pl, "DUAL %s RMS tern_x1000=%u int8_x1000=%u\n", name,
            (unsigned)logit_rms_x1000(ST_T.logits),
            (unsigned)logit_rms_x1000(ST_I.logits));
    ISV(pl);

#ifdef USE_RSP_MATMUL
    /* The overlap accounting.  rsp_t_wait is the CP0 the CPU spent inside
     * end() actually waiting for the vector unit, i.e. the overlap NOT
     * achieved; free/blocked count the dispatches that were already finished
     * when the CPU came back for them. */
    sprintf(pl, "DUAL %s RSP calls=%u cpu=%u free=%u blocked=%u\n", name,
            (unsigned)sd.calls_rsp, (unsigned)sd.calls_cpu,
            (unsigned)sd.freec, (unsigned)sd.blocked);
    ISV(pl);
    sprintf(pl, "DUAL %s RSPT stage=%u disp=%u wait=%u epi=%u\n", name,
            (unsigned)sd.stage, (unsigned)sd.disp,
            (unsigned)sd.wait, (unsigned)sd.epi);
    ISV(pl);
#endif

    sprintf(pl, "DUAL %s TEXT %s\n", name, outbuf); ISV(pl);
    sprintf(pl, "DUAL %s GEN_END\n", name); ISV(pl);
}

int main(void)
{
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    timer_init();
    g_vi_hz = (get_tv_type() == TV_PAL) ? 50.00f : 59.94f;
    register_VI_handler(vbl_counter);
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();          /* brings rspq up; the matmul registers as an overlay */

    int szt = 0, szi = 0;
    int okt = load_blob("/sophia_weights.bin", wt_buf, (int)sizeof(wt_buf), &szt);
    int oki = load_blob("/sophia_int8.bin",    wi_buf, (int)sizeof(wi_buf), &szi);

    sprintf(pl, "DUAL INIT mem=%u tern_sz=%d cap=%d int8_sz=%d cap=%d ok=%d%d\n",
            (unsigned)get_memory_size(), szt, (int)sizeof(wt_buf),
            szi, (int)sizeof(wi_buf), okt, oki);
    ISV(pl);

    if (!okt || !oki) { ISV("DUAL FAIL blob\n"); ISV("DUAL_DONE\n"); while (1) {} }

    /* pse = 0 on BOTH arms.  The Physarum router is one global, so two models
     * running interleaved would condition each other's attention through it;
     * it is also the configuration the numpy oracle implements, so pse = 0 is
     * the arm that can be checked against the host at all. */
    sgai_init_ex(&ST_T, wt_buf, SGAI_ENGINE_CPU, &SC_T, 0);
    sgai_init_ex(&ST_I, wi_buf, SGAI_ENGINE_RSP, &SC_I, 0);

    sprintf(pl, "DUAL MODELS tern rdy=%d bits=%d nl=%d | int8 rdy=%d bits=%d nl=%d\n",
            ST_T.is_loaded, ST_T.w_bits, ST_T.n_layers,
            ST_I.is_loaded, ST_I.w_bits, ST_I.n_layers);
    ISV(pl);
    sprintf(pl, "DUAL KV tern=%p int8=%p shift=%d ngen=%d ctx=%d\n",
            (void *)ST_T.kv, (void *)ST_I.kv, DUAL_SHIFT, DUAL_NGEN, (int)SGAI_CTX);
    ISV(pl);
    if (!ST_T.is_loaded || !ST_I.is_loaded || !ST_T.kv || !ST_I.kv) {
        ISV("DUAL FAIL init\n"); ISV("DUAL_DONE\n"); while (1) {}
    }

    run_arm("TERN", 0);
    run_arm("INT8", 1);
    run_arm("SERI", 3);   /* the vote with the RSP NOT overlapped */
    run_arm("VOTE", 2);   /* the vote with it overlapped          */

    ISV("DUAL_DONE\n");
    while (1) { }
}
