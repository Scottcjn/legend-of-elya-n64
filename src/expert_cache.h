/* SPDX-License-Identifier: MIT
 * expert_cache.h — Streaming MoE expert cache for the N64.
 *
 * See docs/STREAMING_MOE.md. Experts live in cartridge ROM; only the
 * active one (plus a speculative prefetch) occupies RDRAM. The N64's
 * advantage over the Genesis Lock-On design is that PI DMA is
 * ASYNCHRONOUS: the next expert can stream in while the CPU is still
 * generating tokens from the current one, so the load hides completely.
 */
#ifndef EXPERT_CACHE_H
#define EXPERT_CACHE_H

#include <stdint.h>

#define EC_MAX_SLOTS   8
#define EC_MAX_EXPERTS 16
#define EC_NO_EXPERT   0xFFFF

typedef struct {
    uint8_t  *mem;              /* RDRAM buffer for this slot        */
    uint16_t  expert;           /* which expert lives here, or NONE  */
    uint32_t  last_used;        /* LRU stamp                          */
    uint8_t   loading;          /* an async DMA is in flight          */
    uint8_t   prefetched;       /* filled speculatively, not yet used  */
} EcSlot;

typedef struct {
    uint32_t rom_base;          /* cart address of the SGTM blob      */
    uint32_t expert_off[EC_MAX_EXPERTS];   /* offsets within the blob */
    uint32_t expert_len;        /* bytes per expert (uniform)         */
    uint16_t n_experts;
    uint16_t n_slots;
    EcSlot   slot[EC_MAX_SLOTS];
    uint32_t clock;             /* monotonically increasing LRU stamp */
    uint16_t inflight_slot;     /* slot with a pending DMA, or NONE   */
    uint16_t pending;           /* prefetch deferred behind a busy bus */
    uint16_t pending_keep;      /* its `keep` argument                 */
    /* stats — worth showing on screen, this is the whole trick */
    uint32_t hits, misses, prefetch_hits;
} ExpertCache;

/* Set up the cache. slots_mem must be n_slots buffers of expert_len
 * bytes, 8-byte aligned (PI DMA requirement). Returns 0 on success. */
int  ec_init(ExpertCache *ec, uint32_t rom_base, uint32_t expert_len,
             uint16_t n_experts, uint8_t **slots_mem, uint16_t n_slots);

/* Is this expert already resident and ready to use? */
int  ec_resident(ExpertCache *ec, uint16_t expert);

/* Start an async load of `expert` into an LRU victim slot if it is not
 * already resident. Returns immediately; safe to call every frame.
 * Used both for the real request and for speculative prefetch. */
void ec_request(ExpertCache *ec, uint16_t expert);

/* Block until `expert` is resident and return a pointer to its weights.
 * Call this only when the first token actually needs them. */
const uint8_t *ec_acquire(ExpertCache *ec, uint16_t expert);

/* Speculative prefetch: load `expert` only if a slot is free or the LRU
 * victim is not `keep`. Costs nothing when wrong. */
void ec_prefetch(ExpertCache *ec, uint16_t expert, uint16_t keep);

#endif
