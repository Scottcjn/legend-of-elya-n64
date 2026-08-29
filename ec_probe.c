// SPDX-License-Identifier: MIT
/*
 * ec_probe.c — does a streaming-MoE expert load actually HIDE behind a token?
 *
 * `src/expert_cache.c` (async PI DMA, LRU, speculative prefetch) has existed
 * since 2026-08-04 with two green host suites and has never been linked into a
 * ROM.  It could not usefully be, for a reason worth stating plainly: the
 * shipped model is a DENSE 6.36M-param ternary transformer, not a mixture of
 * experts.  There are no experts to stream.
 *
 * So this harness does not pretend to be MoE.  It measures the one physical
 * question that decides whether MoE is worth training for this machine:
 *
 *     while the CPU+RSP are generating a token, can a ~160KB expert-sized
 *     PI DMA from the cartridge complete for free, or does it cost time?
 *
 * Method: the ROM's own filesystem region is treated as a bank of
 * expert-sized slices (real cartridge addresses, real PI DMA, real bytes —
 * nothing is faked), and three arms are timed over the SAME 16 generated
 * tokens of the SAME real model:
 *
 *   BASE  generation only, no cache traffic          -> the floor
 *   BLOCK ec_acquire() before each token             -> the cost if it stalls
 *   HIDE  ec_request() before, ec_acquire() after    -> the cost if it hides
 *
 * If HIDE ~= BASE and BLOCK > BASE, the DMA is genuinely free while the
 * processors work, and F-R026's newly-idle CPU time is what pays for it.
 * If HIDE ~= BLOCK, streaming MoE costs a stall per expert and the Genesis
 * design does not transfer, whatever the design doc says.
 *
 * Build: make ecprobe          (ROM: legend_of_elya_ecprobe.z64)
 */
#include <libdragon.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "nano_gpt.h"
#include "src/expert_cache.h"

static volatile uint32_t g_vbl = 0;
static void vbl_counter(void) { g_vbl++; }
#define CP0(x) asm volatile("mfc0 %0, $9" : "=r"(x))

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

/* Expert-sized slice.  One FFN tensor of this model at ternary is
 * 256*1024/4 = 65,536 B; STREAMING_MOE.md's own sizing is ~160KB per FFN
 * expert.  160KB is the number this probe uses. */
#ifndef EC_SLICE
#define EC_SLICE (160 * 1024)
#endif
#ifndef EC_SLOTS
#define EC_SLOTS 2
#endif
#ifndef EC_NGEN
#define EC_NGEN 16
#endif
#define EC_NEXP 4

static uint8_t wbuf[SGAI_WEIGHT_BUF_N(2, SGAI_N_LAYERS)] __attribute__((aligned(16)));
static uint8_t ecmem[EC_SLOTS][EC_SLICE] __attribute__((aligned(16)));
static SGAIState  ST;
static SGAIScratch SC;
static ExpertCache EC;

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

/* mode: 0 = BASE, 1 = BLOCK (acquire before the token), 2 = HIDE
 * (request before, acquire after -- the DMA runs under the token) */
static void run_arm(const char *name, int mode, const char *prompt, uint8_t *out)
{
    uint32_t t0, t1, vb0, vb1, cp0 = 0;
    uint8_t tok = 0;
    sgai_reset(&ST);
    for (int i = 0; prompt[i]; i++) tok = sgai_next_token(&ST, (uint8_t)prompt[i], 0);

    EC.hits = EC.misses = EC.prefetch_hits = 0;
    vb0 = g_vbl;
    for (int i = 0; i < EC_NGEN; i++) {
        uint16_t e = (uint16_t)(i % EC_NEXP);
        CP0(t0);
        if (mode == 1) (void)ec_acquire(&EC, e);      /* stall, then compute */
        else if (mode == 2) ec_request(&EC, e);       /* start, then compute */
        if (i) tok = sgai_next_token(&ST, tok, 0);
        if (mode == 2) (void)ec_acquire(&EC, e);      /* collect afterwards  */
        CP0(t1);
        cp0 += (t1 - t0);
        out[i] = tok;
    }
    vb1 = g_vbl;
    sprintf(pl, "EC %s cp0=%u vbl=%u hits=%u miss=%u pf=%u\n", name,
            (unsigned)cp0, (unsigned)(vb1 - vb0),
            (unsigned)EC.hits, (unsigned)EC.misses, (unsigned)EC.prefetch_hits);
    ISV(pl);
}

int main(void)
{
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    timer_init();
    register_VI_handler(vbl_counter);
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();

    int sz = 0;
    if (!load_blob("/sophia_weights.bin", wbuf, (int)sizeof wbuf, &sz)) {
        ISV("EC FAIL blob\n"); ISV("EC_DONE\n"); while (1) {}
    }
    sgai_init_ex(&ST, wbuf, SGAI_ENGINE_RSP, &SC, 0);
    if (!ST.is_loaded || !ST.kv) { ISV("EC FAIL init\n"); ISV("EC_DONE\n"); while (1) {} }

    /* Real cartridge addresses: the DFS region of this very ROM.  The bytes
     * are the model blob itself, which is what an expert bank would be. */
    uint32_t rom_base = (uint32_t)dfs_rom_addr("/sophia_weights.bin");
    if (!rom_base) { ISV("EC FAIL rom_addr\n"); ISV("EC_DONE\n"); while (1) {} }
    uint8_t *slots[EC_SLOTS];
    for (int i = 0; i < EC_SLOTS; i++) slots[i] = ecmem[i];
    if (ec_init(&EC, rom_base, EC_SLICE, EC_NEXP, slots, EC_SLOTS) != 0) {
        ISV("EC FAIL ec_init\n"); ISV("EC_DONE\n"); while (1) {}
    }
    sprintf(pl, "EC INIT rom=%08x slice=%d slots=%d experts=%d ngen=%d\n",
            (unsigned)rom_base, (int)EC_SLICE, EC_SLOTS, EC_NEXP, EC_NGEN);
    ISV(pl);

    static uint8_t o0[EC_NGEN], o1[EC_NGEN], o2[EC_NGEN];
    const char *p = "Who are you?";
    run_arm("BASE ", 0, p, o0);
    run_arm("BLOCK", 1, p, o1);
    run_arm("HIDE ", 2, p, o2);

    /* Every arm must emit the same tokens: the cache traffic must not touch
     * the model.  If this says DIFF the measurement is void. */
    sprintf(pl, "EC TOKENS base==block:%s base==hide:%s\n",
            memcmp(o0, o1, EC_NGEN) ? "DIFF" : "same",
            memcmp(o0, o2, EC_NGEN) ? "DIFF" : "same");
    ISV(pl);
    { char t[EC_NGEN + 1]; memcpy(t, o0, EC_NGEN); t[EC_NGEN] = 0;
      for (int i = 0; i < EC_NGEN; i++) if (t[i] < 32 || t[i] > 126) t[i] = '?';
      sprintf(pl, "EC TEXT %s\n", t); ISV(pl); }
    ISV("EC_DONE\n");
    while (1) { }
}
