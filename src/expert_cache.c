/* SPDX-License-Identifier: MIT
 * expert_cache.c — Streaming MoE expert cache (N64 / libdragon).
 * See docs/STREAMING_MOE.md.
 */
#include "expert_cache.h"
#include <string.h>
#ifndef EC_HOST_TEST
#include <libdragon.h>
#else
/* host build: stub the PI DMA so the cache policy can be unit-tested */
extern void  ec_test_dma(void *ram, unsigned long pi, unsigned long len);
extern void  ec_test_wait(void);
#define dma_read_async(r, p, l) ec_test_dma((r), (p), (l))
#define dma_wait()              ec_test_wait()
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
    for (uint16_t s = 0; s < n_slots; s++) {
        ec->slot[s].mem    = slots_mem[s];
        ec->slot[s].expert = EC_NO_EXPERT;
    }
    return 0;
}

static int ec_find(ExpertCache *ec, uint16_t expert)
{
    for (uint16_t s = 0; s < ec->n_slots; s++)
        if (ec->slot[s].expert == expert) return (int)s;
    return -1;
}

int ec_resident(ExpertCache *ec, uint16_t expert)
{
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

static void ec_start_load(ExpertCache *ec, uint16_t expert, int slot)
{
    /* Only one PI transfer may be outstanding; finish the previous one. */
    ec_settle(ec);

    ec->slot[slot].expert  = expert;
    ec->slot[slot].loading = 1;
    ec->inflight_slot      = (uint16_t)slot;
    ec_touch(ec, (uint16_t)slot);

    dma_read_async(ec->slot[slot].mem,
                   ec->rom_base + ec->expert_off[expert],
                   ec->expert_len);
}

void ec_request(ExpertCache *ec, uint16_t expert)
{
    if (expert >= ec->n_experts) return;
    int s = ec_find(ec, expert);
    if (s >= 0) { ec_touch(ec, (uint16_t)s); return; }   /* resident or loading */

    int v = ec_victim(ec, EC_NO_EXPERT);
    if (v < 0) return;                                   /* all busy; try later */
    ec_start_load(ec, expert, v);
}

const uint8_t *ec_acquire(ExpertCache *ec, uint16_t expert)
{
    if (expert >= ec->n_experts) return 0;

    int s = ec_find(ec, expert);
    if (s >= 0 && !ec->slot[s].loading) {
        ec->hits++;
        ec_touch(ec, (uint16_t)s);
        return ec->slot[s].mem;
    }
    if (s >= 0) {
        /* a prefetch was already in flight for exactly this expert —
         * this is the case the whole design exists to produce */
        ec->prefetch_hits++;
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
    if (ec_find(ec, expert) >= 0) return;         /* already here/coming */
    if (ec->inflight_slot != EC_NO_EXPERT) return; /* don't queue behind */

    int v = ec_victim(ec, keep);
    if (v < 0) return;                            /* nothing safe to evict */
    ec_start_load(ec, expert, v);
}
