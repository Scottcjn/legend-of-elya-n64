/* SPDX-License-Identifier: MIT
 * expert_cache.c — Streaming MoE expert cache (N64 / libdragon).
 * See docs/STREAMING_MOE.md.
 */
#include "expert_cache.h"
#include <string.h>
#ifndef EC_HOST_TEST
#include <libdragon.h>
/* PI_STATUS bits 0-1: DMA busy / IO busy.  Read directly rather than calling
 * libdragon's dma_busy(), which is marked deprecated and would trip -Werror. */
#define EC_PI_STATUS (*(volatile uint32_t *)0xA4600010u)
static inline int ec_dma_busy(void) { return (EC_PI_STATUS & 3u) != 0; }
/* PI DMA writes straight into RDRAM behind the CPU's back and libdragon does
 * NO cache maintenance for you (verified: zero `cache` instructions in
 * libdragon.a's dma.o).  Without this, stale VALID lines make the CPU read
 * pre-DMA garbage, and stale DIRTY lines get evicted ON TOP of the freshly
 * transferred weights.  Both give intermittently wrong tokens, not a crash. */
#define ec_dcache_prep(p, n) data_cache_hit_writeback_invalidate((p), (n))
#else
/* host build: stub the PI DMA so the cache policy can be unit-tested */
extern void  ec_test_dma(void *ram, unsigned long pi, unsigned long len);
extern void  ec_test_wait(void);
extern int   ec_test_busy(void);
#define dma_read_async(r, p, l) ec_test_dma((r), (p), (l))
#define dma_wait()              ec_test_wait()
#define ec_dma_busy()           ec_test_busy()
#define ec_dcache_prep(p, n)    ((void)(p), (void)(n))
#endif

static void ec_touch(ExpertCache *ec, uint16_t s)
{
    ec->slot[s].last_used = ++ec->clock;
}

int ec_init(ExpertCache *ec, uint32_t rom_base, uint32_t expert_len,
            uint16_t n_experts, uint8_t **slots_mem, uint16_t n_slots)
{
    if (n_slots == 0 || n_slots > EC_MAX_SLOTS)   return -1;
    if (n_experts == 0 || n_experts > EC_MAX_EXPERTS) return -2;
    if (expert_len == 0 || (expert_len & 7))      return -3;  /* DMA align */

    memset(ec, 0, sizeof(*ec));
    ec->rom_base   = rom_base;
    ec->expert_len = expert_len;
    ec->n_experts  = n_experts;
    ec->n_slots    = n_slots;
    ec->inflight_slot = EC_NO_EXPERT;
    ec->pending       = EC_NO_EXPERT;
    ec->pending_keep  = EC_NO_EXPERT;
    for (uint16_t s = 0; s < n_slots; s++) {
        ec->slot[s].mem    = slots_mem[s];
        ec->slot[s].expert = EC_NO_EXPERT;
    }
    /* Uniform stride.  This was MISSING: memset() left every expert_off[] at
     * zero, so ec_request() DMA'd expert 0's bytes for every expert id and
     * nothing complained -- the transfers were the right size and the right
     * duration, just from the wrong address.  The host suites stub the DMA and
     * assert on slot bookkeeping, so they could not see it, and F-R027 timed
     * the transfers without ever checking their contents.  A caller wanting a
     * non-uniform bank can still overwrite expert_off[] after ec_init(). */
    for (uint16_t e = 0; e < n_experts; e++)
        ec->expert_off[e] = (uint32_t)e * expert_len;
    return 0;
}

static int ec_find(ExpertCache *ec, uint16_t expert)
{
    for (uint16_t s = 0; s < ec->n_slots; s++)
        if (ec->slot[s].expert == expert) return (int)s;
    return -1;
}

/* Non-blocking counterpart to ec_settle().  Without this, inflight_slot is only
 * ever cleared by an ec_acquire() of the very expert being prefetched — so in
 * the steady streaming state (prefetch B, keep generating from A) the flag
 * sticks forever, ec_prefetch() returns at its "don't queue behind" guard on
 * every subsequent call, prefetching silently switches itself off, and the
 * slot is leaked because ec_victim() skips slots marked `loading`.
 * Must be called on every entry point, including the acquire HIT path. */
static int  ec_find(ExpertCache *ec, uint16_t expert);
static int  ec_victim(ExpertCache *ec, uint16_t keep);
static void ec_start_load(ExpertCache *ec, uint16_t expert, int slot,
                          int speculative);

static void ec_poll(ExpertCache *ec)
{
    if (ec->inflight_slot != EC_NO_EXPERT) {
        if (ec_dma_busy()) return;             /* still moving; do not block */
        ec->slot[ec->inflight_slot].loading = 0;
        ec->inflight_slot = EC_NO_EXPERT;
    }
    /* A speculative request made while the bus was busy is remembered rather
     * than dropped, and issued here the moment the demand load retires.
     * Without this, ec_prefetch() is dead on any walk through NEW rooms: the
     * caller's own ec_request() starts a DMA one line earlier, so the prefetch
     * that follows always hits the busy guard and silently does nothing — the
     * prefetch counters still read plausibly, which is why F-R030 measured
     * "prefetch on" and "prefetch off" as the same number. */
    if (ec->pending != EC_NO_EXPERT && ec->inflight_slot == EC_NO_EXPERT) {
        uint16_t want = ec->pending, keep = ec->pending_keep;
        ec->pending = EC_NO_EXPERT;
        if (want < ec->n_experts && ec_find(ec, want) < 0) {
            int v = ec_victim(ec, keep);
            if (v >= 0) ec_start_load(ec, want, v, 1);
        }
    }
}

int ec_resident(ExpertCache *ec, uint16_t expert)
{
    ec_poll(ec);
    int s = ec_find(ec, expert);
    return (s >= 0 && !ec->slot[s].loading);
}

/* Least-recently-used slot, never returning `keep`'s slot or one with a
 * DMA in flight. Returns -1 if every slot is protected. */
static int ec_victim(ExpertCache *ec, uint16_t keep)
{
    int best = -1;
    uint32_t oldest = 0xFFFFFFFFu;
    for (uint16_t s = 0; s < ec->n_slots; s++) {
        if (ec->slot[s].loading) continue;
        /* An EMPTY slot is always the best victim, and must be tested
         * before `keep` — EC_NO_EXPERT means both "empty" and "protect
         * nothing", so checking keep first would protect every free
         * slot and make a cold cache permanently unfillable. */
        if (ec->slot[s].expert == EC_NO_EXPERT) return (int)s;  /* free */
        if (ec->slot[s].expert == keep) continue;
        if (ec->slot[s].last_used < oldest) {
            oldest = ec->slot[s].last_used;
            best = (int)s;
        }
    }
    return best;
}

/* Complete any finished transfer. libdragon's dma_wait() blocks, so we
 * only call it when we genuinely need the data; this just records that
 * the slot is usable once we have waited. */
static void ec_settle(ExpertCache *ec)
{
    if (ec->inflight_slot == EC_NO_EXPERT) return;
    dma_wait();
    ec->slot[ec->inflight_slot].loading = 0;
    ec->inflight_slot = EC_NO_EXPERT;
}

static void ec_start_load(ExpertCache *ec, uint16_t expert, int slot,
                          int speculative)
{
    /* Only one PI transfer may be outstanding; finish the previous one. */
    ec_settle(ec);

    ec->slot[slot].expert     = expert;
    ec->slot[slot].loading    = 1;
    ec->slot[slot].prefetched = (uint8_t)speculative;
    ec->inflight_slot      = (uint16_t)slot;
    ec_touch(ec, (uint16_t)slot);

    /* Flush+invalidate the destination BEFORE the PI engine writes it. */
    ec_dcache_prep(ec->slot[slot].mem, ec->expert_len);

    /* PHYSICAL address.  dma_read() masks KSEG1 for you; the raw async path
     * does not, so handing it a 0xB0... pointer (which is exactly what
     * dfs_rom_addr() returns) makes the PI read unmapped space and fill the
     * slot with address-echo instead of weights -- same duration, same size,
     * wrong bytes, no error anywhere. */
    dma_read_async(ec->slot[slot].mem,
                   (ec->rom_base + ec->expert_off[expert]) & 0x1FFFFFFFul,
                   ec->expert_len);
}

void ec_request(ExpertCache *ec, uint16_t expert)
{
    if (expert >= ec->n_experts) return;
    ec_poll(ec);
    int s = ec_find(ec, expert);
    if (s >= 0) { ec_touch(ec, (uint16_t)s); return; }   /* resident or loading */

    int v = ec_victim(ec, EC_NO_EXPERT);
    if (v < 0) return;                                   /* all busy; try later */
    ec_start_load(ec, expert, v, 0);
}

const uint8_t *ec_acquire(ExpertCache *ec, uint16_t expert)
{
    if (expert >= ec->n_experts) return 0;
    ec_poll(ec);

    int s = ec_find(ec, expert);
    if (s >= 0 && !ec->slot[s].loading) {
        /* Credit the speculation that produced this slot, whether or not its
         * DMA had already completed by the time we got here — otherwise
         * ec_poll() would silently reclassify every successful prefetch as an
         * ordinary hit and the stat would always read zero. */
        if (ec->slot[s].prefetched) { ec->prefetch_hits++; ec->slot[s].prefetched = 0; }
        else                          ec->hits++;
        ec_touch(ec, (uint16_t)s);
        return ec->slot[s].mem;
    }
    if (s >= 0) {
        /* A transfer for exactly this expert is still in flight, so this call
         * is about to BLOCK.  Credit it to speculation only if speculation is
         * what started it; a demand load that has not landed is a miss, not a
         * win.  Counting both as prefetch_hits made the stat read as success
         * on every stalled walk (F-R030's unexplained TOUR result). */
        if (ec->slot[s].prefetched) ec->prefetch_hits++;
        else                        ec->misses++;
        ec->slot[s].prefetched = 0;
        ec_settle(ec);
        ec_touch(ec, (uint16_t)s);
        return ec->slot[s].mem;
    }

    ec->misses++;
    ec_request(ec, expert);
    s = ec_find(ec, expert);
    if (s < 0) return 0;
    ec_settle(ec);
    return ec->slot[s].mem;
}

void ec_prefetch(ExpertCache *ec, uint16_t expert, uint16_t keep)
{
    if (expert >= ec->n_experts || expert == keep) return;
    ec_poll(ec);
    if (ec_find(ec, expert) >= 0) return;         /* already here/coming */
    if (ec->inflight_slot != EC_NO_EXPERT) {
        /* Do not queue behind the bus -- but do not FORGET either.  ec_poll()
         * issues this the moment the current transfer retires, which is what
         * makes prefetch work on a walk through rooms the cache has never
         * seen: the caller's own ec_request() is always in flight one line
         * earlier.  Newest intent wins; a stale one is simply overwritten. */
        ec->pending      = expert;
        ec->pending_keep = keep;
        return;
    }

    int v = ec_victim(ec, keep);
    if (v < 0) { ec->pending = expert; ec->pending_keep = keep; return; }
    ec_start_load(ec, expert, v, 1);
}
