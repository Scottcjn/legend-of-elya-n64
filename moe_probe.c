// SPDX-License-Identifier: MIT
/*
 * moe_probe.c — the whole streaming-MoE chain, on the console.
 *
 *   route a prompt (moe_router.h, generated from the trainer's own shard table)
 *     -> ec_acquire() the expert (src/expert_cache.c, real PI DMA from cart)
 *       -> sgai_init_ex() on the streamed blob (each expert is a COMPLETE SEQ2
 *          blob, so this is a pointer swap, not a new parser)
 *         -> generate greedily and print what it said
 *
 * What this measures that F-R027 could not: F-R027 proved a 160KB DMA hides
 * behind a token, but it had no experts -- the shipped model is dense. This
 * runs REAL experts, so it also pays the two costs a pointer-swap design
 * actually has and that no design doc priced:
 *
 *   SWAP   sgai_init_ex() re-permutes every weight tensor into the RSP's
 *          lane order (nano_gpt.c does this once per model at init).  On an
 *          expert switch that permutation is paid again, over 1MB.
 *   FIRST  the first token after a switch cannot hide the DMA behind
 *          anything -- there is no previous token to hide under.
 *
 * Both are printed per prompt, next to the steady-state token cost, so the
 * design is judged on measured numbers instead of the hope that streaming is
 * free.  Build: make moeprobe
 */
#include <libdragon.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "nano_gpt.h"
#include "src/expert_cache.h"
#include "moe_router.h"

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

/* The bank header written by training/make_moe_bank.py. */
#define MOE_MAGIC 0x53474D42u          /* "SGMB" */
typedef struct { uint32_t magic; uint16_t ver, n; uint32_t len, base; } MoeHdr;

#ifndef MOE_SLOTS
#define MOE_SLOTS 2
#endif
/* Generation stops at the model's own end-of-answer, not at a fixed count.
 * The corpus is newline-joined, so byte 10 is a TRAINED end-of-answer (C028);
 * the greedy band 32..126 simply never let it win, which is why answers came
 * out truncated mid-word. -DSGAI_NEWLINE_STOP lets it compete, exactly as the
 * Genesis port already did (legend-of-elya-genesis/host/harness.c:43,
 * "if (tok == '\n') break;"). MOE_NGEN is now a CAP, not a target. */
#ifndef MOE_NGEN
#define MOE_NGEN 48
#endif
#ifndef MOE_EXPERT_BYTES
#define MOE_EXPERT_BYTES 1048592        /* 1,048,588 rounded to 8 */
#endif

static uint8_t ecmem[MOE_SLOTS][MOE_EXPERT_BYTES] __attribute__((aligned(16)));
static SGAIState   ST;
static SGAIScratch SC;
static ExpertCache EC;
static uint16_t cur_expert = 0xFFFF;

static const char *const PROBE[] = {
    "Who are you?: ",
    "What is RustChain?: ",
    "What is the G4?: ",
    "What lurks here?: ",
    "Who is Ganon?: ",
    "What is epoch?: ",
};
#define NPROBE ((int)(sizeof PROBE / sizeof PROBE[0]))

int main(void)
{
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    timer_init();
    register_VI_handler(vbl_counter);
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();

    /* Read the bank header out of the cartridge, then hand ec_init() the
     * stride the assembler actually used -- never a constant compiled in here,
     * which would silently mis-slice a rebuilt bank. */
    uint32_t rom = (uint32_t)dfs_rom_addr("/sophia_moe.bin");
    if (!rom) { ISV("MOE FAIL rom_addr\n"); ISV("MOE_DONE\n"); while (1) {} }
    static MoeHdr h __attribute__((aligned(16)));
    data_cache_hit_writeback_invalidate(&h, sizeof h);
    dma_read(&h, rom, sizeof h);
    if (h.magic != MOE_MAGIC) {
        sprintf(pl, "MOE FAIL magic=%08x\n", (unsigned)h.magic); ISV(pl);
        ISV("MOE_DONE\n"); while (1) {}
    }
    sprintf(pl, "MOE BANK n=%u len=%u base=%u rom=%08x slots=%d\n",
            (unsigned)h.n, (unsigned)h.len, (unsigned)h.base,
            (unsigned)rom, MOE_SLOTS);
    ISV(pl);
    /* The bank index IS the id moe_route() returns, so a bank with a
     * different expert count than the router was generated for cannot be
     * routed at all -- fail loudly instead of answering out of whichever
     * expert happens to sit at that index. */
    if (h.len > MOE_EXPERT_BYTES || h.n != MOE_N_EXPERTS) {
        sprintf(pl, "MOE FAIL bank n=%u router n=%u len=%u slot=%u\n",
                (unsigned)h.n, MOE_N_EXPERTS, (unsigned)h.len,
                (unsigned)MOE_EXPERT_BYTES);
        ISV(pl);
        ISV("MOE_DONE\n"); while (1) {}
    }

    uint8_t *slots[MOE_SLOTS];
    for (int i = 0; i < MOE_SLOTS; i++) slots[i] = ecmem[i];
    if (ec_init(&EC, rom + h.base, h.len, (uint16_t)h.n, slots, MOE_SLOTS) != 0) {
        ISV("MOE FAIL ec_init\n"); ISV("MOE_DONE\n"); while (1) {}
    }

    for (int i = 0; i < NPROBE; i++) {
        const char *p = PROBE[i];
        uint16_t e = moe_route(p);
        uint32_t t0, t1, t2, t3, vb0, vb1;

        /* Speculatively start the NEXT prompt's expert now, so the steady
         * state this prints is the one a game would actually see. */
        if (i + 1 < NPROBE) {
            uint16_t nxt = moe_route(PROBE[i + 1]);
            if (nxt != e) ec_prefetch(&EC, nxt, e);
        }

        CP0(t0);
        const uint8_t *w = ec_acquire(&EC, e);
        CP0(t1);
        if (!w) { ISV("MOE FAIL acquire\n"); continue; }

        if (e != cur_expert) {
            sgai_init_ex(&ST, w, SGAI_ENGINE_RSP, &SC, 0);
            cur_expert = e;
        }
        CP0(t2);
        if (!ST.is_loaded) { ISV("MOE FAIL load\n"); continue; }

        sgai_reset(&ST);
        uint8_t tok = 0;
        for (int k = 0; p[k]; k++) tok = sgai_next_token(&ST, (uint8_t)p[k], 0);
        static char out[MOE_NGEN + 1];
        vb0 = g_vbl;
        int n = 0;
        for (; n < MOE_NGEN; n++) {
            if (tok == '\n') break;          /* the trained end of answer */
            out[n] = (char)tok;
            tok = sgai_next_token(&ST, tok, 0);
        }
        vb1 = g_vbl;
        int hit_eos = (tok == '\n');
        out[n] = 0;
        for (int k = 0; k < n; k++)
            if (out[k] < 32 || out[k] > 126) out[k] = '?';
        CP0(t3);

        sprintf(pl, "MOE[%d] %s -> expert %u (%s)\n", i, p, (unsigned)e,
                moe_expert_name[e]);
        ISV(pl);
        sprintf(pl, "MOE[%d] acquire=%u swap=%u gen=%u vbl=%u hits=%u miss=%u pf=%u\n",
                i, (unsigned)(t1 - t0), (unsigned)(t2 - t1), (unsigned)(t3 - t2),
                (unsigned)(vb1 - vb0), (unsigned)EC.hits, (unsigned)EC.misses,
                (unsigned)EC.prefetch_hits);
        ISV(pl);
        sprintf(pl, "MOE[%d] TEXT %s\n", i, out); ISV(pl);
        sprintf(pl, "MOE[%d] %d tokens, %s\n", i, n,
                hit_eos ? "ended on its own (newline)" : "HIT THE CAP (truncated)");
        ISV(pl);
    }
    ISV("MOE_DONE\n");
    while (1) { }
}
