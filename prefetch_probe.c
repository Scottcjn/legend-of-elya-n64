// SPDX-License-Identifier: MIT
/*
 * prefetch_probe.c — does proximity prefetch hide the expert load in a
 * REALISTIC walk, and what happens when the player refuses to cooperate?
 *
 * THE CONSTRAINT (measured, F-R028/F-R029): a 1 MB expert takes 401 ms to load
 * from cartridge = 2.4 generated tokens of lead, and swapping it in costs
 * another 161 ms. F-R028's ~100-cycle steady-state acquire was real, but it
 * had a whole 24-token turn of warning. A game gets only the warning the
 * PLAYER gives it by walking somewhere.
 *
 * So the question is not "does prefetch work" but "does the player's own
 * movement supply enough lead, and what is the worst case when it does not."
 * Four scripted walks over a room graph, each ending in conversation, on the
 * real model and the real expert bank:
 *
 *   STROLL     enter, linger a beat, then talk        -> the happy path
 *   SPRINT     enter and talk immediately             -> zero lead
 *   BACKTRACK  oscillate between two domains          -> 2 slots, thrash
 *   TOUR       walk the whole graph, talking in each  -> steady state
 *
 * Each reports the stall actually paid at the moment dialogue opened, which is
 * the only latency a player can feel. Every walk runs TWICE — without prefetch
 * first as the control — so each figure is a delta against this same machine.
 *
 * Build: make prefetchprobe        (cartridge only; no SD card needed)
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
#define CP0_HZ 46875000u
#define MS(c)  ((unsigned)((uint64_t)(c) * 1000u / CP0_HZ))

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

#define MOE_MAGIC 0x53474D42u
typedef struct { uint32_t magic; uint16_t ver, n; uint32_t len, base; } MoeHdr;

#ifndef PF_SLOTS
#define PF_SLOTS 2
#endif
#ifndef PF_NGEN
#define PF_NGEN 8
#endif
#ifndef PF_EXPERT_BYTES
#define PF_EXPERT_BYTES 1048592
#endif
/* "Lingering" is measured in generated tokens, not wall clock: a token is the
 * only unit of time this machine actually spends, and it is what a walk has to
 * buy lead against. One token = 169 ms (F-R028). */
#ifndef PF_LINGER_TOKENS
#define PF_LINGER_TOKENS 3
#endif

static uint8_t ecmem[PF_SLOTS][PF_EXPERT_BYTES] __attribute__((aligned(16)));
static SGAIState   ST;
static SGAIScratch SC;
static ExpertCache EC;
static uint32_t EC_ROM, EC_LEN; static uint16_t EC_N;
static uint16_t resident = 0xFFFF;

typedef struct { const char *room, *npc, *line; uint16_t expert; } Room;
static const Room MAP[] = {
    { "Forge",   "Bram the Smith", "What is the G4?: ",    MOE_E_HARDWARE  },
    { "Vault",   "Ledger Keeper",  "What is RustChain?: ", MOE_E_RUSTCHAIN },
    { "Study",   "Sophia Elya",    "Who are you?: ",       MOE_E_IDENTITY  },
    { "Depths",  "Kragan",         "What lurks here?: ",   MOE_E_DUNGEON   },
    { "Archive", "The Chronicler", "Who is Ganon?: ",      MOE_E_LORE      },
};

static const int W_STROLL[]    = { 0, 1, 2 };
static const int W_SPRINT[]    = { 3, 4, 0 };
static const int W_BACKTRACK[] = { 0, 1, 0, 1, 0 };
static const int W_TOUR[]      = { 0, 1, 2, 3, 4 };

typedef struct { const char *name; const int *path; int n; int linger; } Walk;
static const Walk WALKS[] = {
    { "STROLL   ", W_STROLL,    3, PF_LINGER_TOKENS },
    { "SPRINT   ", W_SPRINT,    3, 0 },
    { "BACKTRACK", W_BACKTRACK, 5, PF_LINGER_TOKENS },
    { "TOUR     ", W_TOUR,      5, PF_LINGER_TOKENS },
};
#define NWALK ((int)(sizeof WALKS / sizeof WALKS[0]))

static uint8_t idle_tokens(uint8_t tok, int n)
{
    for (int i = 0; i < n; i++) tok = sgai_next_token(&ST, tok, 0);
    return tok;
}

static void run_walk(const Walk *w, int prefetch)
{
    uint32_t stall_total = 0, swap_total = 0, worst = 0;
    /* F-R030 follow-up: ec_start_load() opens with ec_settle(), a BLOCKING
     * dma_wait().  So a request or a speculative prefetch can pay for the
     * PREVIOUS transfer, and that time is spent inside those calls -- not
     * inside ec_acquire(), which is all the first version of this probe
     * timed.  Time all three. */
    uint32_t req_total = 0, pre_total = 0, ling_total = 0;
    int cold = 0;

    uint8_t *slots[PF_SLOTS];
    for (int i = 0; i < PF_SLOTS; i++) slots[i] = ecmem[i];
    ec_init(&EC, EC_ROM, EC_LEN, EC_N, slots, PF_SLOTS);
    resident = 0xFFFF;
#ifdef PF_VERBOSE
    sprintf(pl, "PF  walk=%s pf=%d n=%d linger_field=%d (macro %d)\n",
            w->name, prefetch, w->n, w->linger, PF_LINGER_TOKENS);
    ISV(pl);
#endif

    for (int step = 0; step < w->n; step++) {
        const Room *r = &MAP[w->path[step]];

        /* Entering a room is when the game knows what is coming: ask for this
         * room's expert, and for the NEXT room's while the player is still
         * here. That second request is the whole thesis. */
        uint32_t q0, q1, q2;
        CP0(q0);
        if (prefetch) ec_request(&EC, r->expert);
        CP0(q1);
        if (prefetch && step + 1 < w->n) {
            uint16_t nxt = MAP[w->path[step + 1]].expert;
            if (nxt != r->expert) ec_prefetch(&EC, nxt, r->expert);
        }
        CP0(q2);
        req_total += q1 - q0; pre_total += q2 - q1;

        /* The linger is the ONLY thing that supplies lead, so it must never
         * be skipped silently -- time it and report is_loaded alongside. */
        uint8_t tok = 0;
        uint32_t l0, l1;
        CP0(l0);
        if (w->linger) tok = idle_tokens(tok, w->linger);
        CP0(l1);
        ling_total += l1 - l0;

        uint32_t t0, t1, t2;
        CP0(t0);
        const uint8_t *wgt = ec_acquire(&EC, r->expert);
        CP0(t1);
        if (!wgt) { ISV("PF FAIL acquire\n"); return; }
        if (r->expert != resident) {
            sgai_init_ex(&ST, wgt, SGAI_ENGINE_RSP, &SC, 0);
            resident = r->expert;
        }
        CP0(t2);

        uint32_t stall = t1 - t0, swap = t2 - t1;
#ifdef PF_VERBOSE
        sprintf(pl, "PF   step%d %-8s e=%u req=%u ms pre=%u ms stall=%u ms "
                    "linger=%u ms loaded=%d inflight=%u\n",
                step, r->room, (unsigned)r->expert, MS(q1 - q0), MS(q2 - q1),
                MS(stall), MS(l1 - l0), ST.is_loaded,
                (unsigned)EC.inflight_slot);
        ISV(pl);
#endif
        stall_total += stall; swap_total += swap;
        if (stall > worst) worst = stall;
        if (MS(stall) > 50) cold++;

        sgai_reset(&ST);
        tok = 0;
        for (int i = 0; r->line[i]; i++)
            tok = sgai_next_token(&ST, (uint8_t)r->line[i], 0);
        tok = idle_tokens(tok, PF_NGEN);
    }

    sprintf(pl, "PF %s pf=%d req=%u ms pre=%u ms ling=%u ms stall=%u ms (worst %u ms, "
                "%d felt) swap=%u ms hits=%u miss=%u pfhit=%u\n",
            w->name, prefetch, MS(req_total), MS(pre_total), MS(ling_total),
            MS(stall_total), MS(worst), cold, MS(swap_total),
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

    uint32_t rom = (uint32_t)dfs_rom_addr("/sophia_moe.bin");
    if (!rom) { ISV("PF FAIL rom_addr\n"); ISV("PF_DONE\n"); while (1) {} }
    static MoeHdr h __attribute__((aligned(16)));
    data_cache_hit_writeback_invalidate(&h, sizeof h);
    dma_read(&h, rom, sizeof h);
    if (h.magic != MOE_MAGIC || h.n != MOE_N_EXPERTS || h.len > PF_EXPERT_BYTES) {
        sprintf(pl, "PF FAIL bank n=%u router=%u len=%u\n",
                (unsigned)h.n, MOE_N_EXPERTS, (unsigned)h.len); ISV(pl);
        ISV("PF_DONE\n"); while (1) {}
    }
    EC_ROM = rom + h.base; EC_LEN = h.len; EC_N = (uint16_t)h.n;
    uint8_t *slots[PF_SLOTS];
    for (int i = 0; i < PF_SLOTS; i++) slots[i] = ecmem[i];
    if (ec_init(&EC, EC_ROM, EC_LEN, EC_N, slots, PF_SLOTS) != 0) {
        ISV("PF FAIL ec_init\n"); ISV("PF_DONE\n"); while (1) {}
    }
    sprintf(pl, "PF BANK n=%u len=%u slots=%d linger=%d tok ngen=%d\n",
            (unsigned)h.n, (unsigned)h.len, PF_SLOTS, PF_LINGER_TOKENS, PF_NGEN);
    ISV(pl);

    const uint8_t *w0 = ec_acquire(&EC, MAP[0].expert);
    if (w0) { sgai_init_ex(&ST, w0, SGAI_ENGINE_RSP, &SC, 0); resident = MAP[0].expert; }
    if (!ST.is_loaded) { ISV("PF FAIL init\n"); ISV("PF_DONE\n"); while (1) {} }

    for (int i = 0; i < NWALK; i++) {
        run_walk(&WALKS[i], 0);
        run_walk(&WALKS[i], 1);
    }

    ISV("PF_DONE\n");
    while (1) { }
}
