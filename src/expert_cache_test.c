/* Host test for the N64 streaming-MoE expert cache policy.
 * Builds with: gcc -DEC_HOST_TEST -o ec_test expert_cache_test.c expert_cache.c
 * Verifies LRU, prefetch-hit accounting, and that a live expert is never
 * evicted out from under the generator. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "expert_cache.h"

static int dma_calls = 0;
void ec_test_dma(void *ram, unsigned long pi, unsigned long len)
{ (void)ram; (void)pi; (void)len; dma_calls++; }
void ec_test_wait(void) { }

#define SLOTS 3
#define EXPERTS 4
#define ELEN 1024

static int fails = 0;
static void check(const char *what, int cond)
{
    printf("  %-52s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond) fails++;
}

int main(void)
{
    static uint8_t buf[SLOTS][ELEN];
    uint8_t *mem[SLOTS];
    for (int i = 0; i < SLOTS; i++) mem[i] = buf[i];

    ExpertCache ec;
    check("init", ec_init(&ec, 0x10000000, ELEN, EXPERTS, mem, SLOTS) == 0);
    for (int e = 0; e < EXPERTS; e++) ec.expert_off[e] = e * ELEN;

    check("cold expert not resident", !ec_resident(&ec, 0));

    /* cold acquire = miss */
    check("acquire 0 returns memory", ec_acquire(&ec, 0) != NULL);
    check("  counted as miss", ec.misses == 1);
    check("  now resident", ec_resident(&ec, 0));

    /* second acquire = hit, no new DMA */
    int before = dma_calls;
    ec_acquire(&ec, 0);
    check("re-acquire 0 is a hit", ec.hits == 1);
    check("  issued no new DMA", dma_calls == before);

    /* prefetch then acquire = prefetch hit, the whole point */
    ec_prefetch(&ec, 1, /*keep=*/0);
    check("prefetch 1 started a DMA", dma_calls == before + 1);
    ec_acquire(&ec, 1);
    check("acquire 1 counted as PREFETCH HIT", ec.prefetch_hits == 1);
    check("  no extra DMA for it", dma_calls == before + 1);

    /* fill the cache, then force eviction and confirm LRU order */
    ec_acquire(&ec, 2);                    /* slots now hold 0,1,2 */
    check("three experts resident", ec_resident(&ec, 0) &&
                                    ec_resident(&ec, 1) &&
                                    ec_resident(&ec, 2));
    ec_acquire(&ec, 1);                    /* touch 1: 0 is now LRU */
    ec_acquire(&ec, 3);                    /* must evict 0 */
    check("evicted the LRU expert (0)", !ec_resident(&ec, 0));
    check("kept recently-used 1", ec_resident(&ec, 1));
    check("kept 2", ec_resident(&ec, 2));
    check("loaded 3", ec_resident(&ec, 3));

    /* the safety property: prefetch must never evict the live expert */
    ec_acquire(&ec, 1);                    /* 1 is live */
    ec_acquire(&ec, 2);
    ec_acquire(&ec, 3);                    /* 1,2,3 resident, 1 is LRU */
    ec_prefetch(&ec, 0, /*keep=*/1);       /* wants a slot; 1 is protected */
    check("prefetch did not evict the live expert", ec_resident(&ec, 1));

    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
