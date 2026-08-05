/* Regression tests for the streaming-MoE expert cache.
 *
 * The existing suite (expert_cache_test.c) passes while a real bug is live,
 * because every prefetch in it is immediately followed by an ec_acquire() of
 * the SAME expert — and that acquire is the only thing that ever clears
 * inflight_slot.  The real streaming pattern is the opposite: you prefetch the
 * NEXT expert and keep generating from the CURRENT one for many tokens.
 *
 * Build: gcc -DEC_HOST_TEST -o ec_regress expert_cache_regress.c expert_cache.c
 */
#include <stdio.h>
#include <string.h>
#include "expert_cache.h"

static int dma_calls = 0;
static int dma_inflight = 0;          /* models the PI engine */
void ec_test_dma(void *ram, unsigned long pi, unsigned long len)
{ (void)ram; (void)pi; (void)len; dma_calls++; dma_inflight = 1; }
void ec_test_wait(void) { dma_inflight = 0; }
int  ec_test_busy(void) { return dma_inflight; }
/* the DMA physically completes while the CPU is off generating tokens */
static void dma_completes(void) { dma_inflight = 0; }

#define SLOTS 3
#define EXPERTS 4
#define ELEN 1024

static int fails = 0;
static void check(const char *what, int cond)
{
    printf("  %-56s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond) fails++;
}

int main(void)
{
    static uint8_t buf[SLOTS][ELEN];
    uint8_t *mem[SLOTS];
    for (int i = 0; i < SLOTS; i++) mem[i] = buf[i];

    ExpertCache ec;
    ec_init(&ec, 0x10000000, ELEN, EXPERTS, mem, SLOTS);
    for (int e = 0; e < EXPERTS; e++) ec.expert_off[e] = e * ELEN;

    printf("STEADY-STATE STREAMING (prefetch B, keep generating from A)\n");

    ec_acquire(&ec, 0);                 /* A resident */
    dma_completes();

    int before = dma_calls;
    ec_prefetch(&ec, 1, /*keep=*/0);    /* speculatively pull B */
    check("prefetch of B started a DMA", dma_calls == before + 1);

    dma_completes();                    /* the transfer really does finish */

    /* keep generating from A for a while — the acquire HIT path */
    for (int k = 0; k < 8; k++) ec_acquire(&ec, 0);

    check("B is recognised as resident once its DMA finished",
          ec_resident(&ec, 1));

    before = dma_calls;
    ec_prefetch(&ec, 2, /*keep=*/0);    /* now speculate on C */
    check("a SECOND prefetch is still able to start a DMA",
          dma_calls == before + 1);

    dma_completes();
    for (int k = 0; k < 8; k++) ec_acquire(&ec, 0);
    check("C also becomes resident", ec_resident(&ec, 2));

    /* and the slot must not be leaked: a victim search has to be able to
     * consider a slot whose DMA has long since finished */
    before = dma_calls;
    ec_prefetch(&ec, 3, /*keep=*/0);
    check("a THIRD prefetch can still find a victim slot",
          dma_calls == before + 1);

    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
