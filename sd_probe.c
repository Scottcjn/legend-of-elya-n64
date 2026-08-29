// SPDX-License-Identifier: MIT
/*
 * sd_probe.c — can an expert be streamed from the EverDrive's SD card, and can
 * that read hide behind generation the way the cartridge DMA does?
 *
 * WHY THIS EXISTS.  The cartridge ROM window is 252 MiB and an EverDrive-64
 * caps the ROM image at 64 MB, so a large expert library has to live on the SD
 * card.  Whether that is viable is one measurement, and this ROM is it.
 *
 * WHAT IS ALREADY KNOWN (docs/N64_RSP_FINDINGS.md):
 *   F-R027  a 160 KB cartridge DMA hides under ONE token (+0.84%); blocking on
 *           it instead costs +11.0%.
 *   F-R028  a 1 MB expert load is 401 ms = 2.4 tokens at 169 ms/token, so it
 *           hides under a TURN, not a token.  The expert switch is 161 ms.
 * Both are cartridge PI DMA, which the PI performs while the CPU works.  The
 * SD path may not be: libcart's edx_card_rd_dram() is a CPU loop over 512-byte
 * sectors, so the CPU may BE the transfer.  That is the thing to measure.
 *
 * THE ARMS, over the same 16 generated tokens of the same real model:
 *   BASE    generation only                                   -> the floor
 *   BLOCK   read a whole expert before each token              -> full stall
 *   FATFS   the same read through fopen/fread ("sd:/...")      -> stdio cost
 *   HIDE    the read sliced between tokens                     -> can it hide?
 *   STAGE   SD -> cart SDRAM (FPGA DMA), then the ordinary cart -> RDRAM DMA
 *           around the token                                   -> the two-hop
 *
 * THRESHOLDS, fixed HERE, before any run, and printed by the ROM so a later
 * reader cannot quietly move them (one token = 169 ms, F-R028):
 *   T1  init  (cart_init + card_init + mount)  < 500 ms  => boot-time only
 *   T2  BLOCK >= 2 MB/s                                  => usable at all
 *   T3  FATFS >= 70% of BLOCK                            => stdio is fine
 *   T4  HIDE  <= BASE + 10%                              => chunked SD hides
 *   T5  STAGE <= BASE + 10%                              => the two-hop wins
 *
 * HOW THIS REFUSES TO LIE.  Every bug this repo has shipped reported success
 * while serving wrong bytes (F-R027's open-bus read; ec_init's zeroed offsets).
 * So: every destination is pre-filled with a canary, every arm CRC-32s what it
 * read against the SAME expert read out of cartridge ROM, and a mismatch VOIDS
 * the run and suppresses the timings.  A missing SD card prints ABSENT and no
 * throughput line at all -- an absent card must never read as infinitely fast.
 *
 * Build: make sdprobe        (needs a card with sd:/elya/e00.sgai to time)
 */
#include <libdragon.h>
#include <libcart/cart.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "nano_gpt.h"
#include "src/expert_cache.h"

static volatile uint32_t g_vbl = 0;
static void vbl_counter(void) { g_vbl++; }
#define CP0(x) asm volatile("mfc0 %0, $9" : "=r"(x))
#define CP0_HZ 46875000u                  /* CP0 Count is HALF the CPU clock */
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

#define MOE_MAGIC 0x53474D42u                        /* "SGMB" */
typedef struct { uint32_t magic; uint16_t ver, n; uint32_t len, base; } MoeHdr;

#ifndef SD_NGEN
#define SD_NGEN 16
#endif
#ifndef SD_EXPERT_BYTES
#define SD_EXPERT_BYTES 1048592
#endif
#ifndef SD_CHUNK_SECTORS                 /* HIDE arm: sectors per slice */
#define SD_CHUNK_SECTORS 128             /* 64 KB */
#endif
#define SD_SECTOR 512
/* STAGE writes into cartridge SDRAM.  On an EverDrive that SDRAM IS the ROM
 * image, so an unreserved write lands on top of the running game.  Default the
 * staging window to the tail of the reported cart_size, and refuse the arm
 * unless the window sits ABOVE this ROM's own end. */
#ifndef SD_STAGE_CART
#define SD_STAGE_CART 0
#endif
#define CANARY 0xA5

static uint8_t  dst[SD_EXPERT_BYTES] __attribute__((aligned(16)));
static uint8_t  ref[SD_EXPERT_BYTES] __attribute__((aligned(16)));
static SGAIState   ST;
static SGAIScratch SC;
static uint8_t  wbuf[SGAI_WEIGHT_BUF_N(2, SGAI_N_LAYERS)] __attribute__((aligned(16)));

/* CRC-32, table-free.  Its own cost is measured and printed, so verification
 * is priced rather than assumed free. */
static uint32_t crc32(const uint8_t *p, uint32_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1)));
    }
    return ~c;
}

static void canary_fill(uint8_t *p, uint32_t n)
{
    memset(p, CANARY, n);
    data_cache_hit_writeback_invalidate(p, n);
}

/* One generated token of the real model, so every arm pays identical compute. */
static uint8_t step(uint8_t tok) { return sgai_next_token(&ST, tok, 0); }

int main(void)
{
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    timer_init();
    register_VI_handler(vbl_counter);
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();

    ISV("SD THRESHOLDS T1=init<500ms T2=BLOCK>=2MB/s T3=FATFS>=70%BLOCK "
        "T4=HIDE<=BASE+10% T5=STAGE<=BASE+10%\n");

    /* ---- the model, from the cartridge, identical in every arm ---------- */
    int fd = dfs_open("/sophia_weights.bin");
    if (fd < 0) { ISV("SD FAIL blob\n"); ISV("SD_DONE\n"); while (1) {} }
    int bsz = dfs_size(fd);
    if (bsz <= 0 || bsz > (int)sizeof wbuf) { ISV("SD FAIL blob size\n"); ISV("SD_DONE\n"); while (1) {} }
    dfs_read(wbuf, 1, bsz, fd); dfs_close(fd);
    sgai_init_ex(&ST, wbuf, SGAI_ENGINE_RSP, &SC, 0);
    if (!ST.is_loaded || !ST.kv) { ISV("SD FAIL init\n"); ISV("SD_DONE\n"); while (1) {} }

    /* ---- the reference expert, read from cartridge ROM ------------------ */
    uint32_t rom = (uint32_t)dfs_rom_addr("/sophia_moe.bin");
    uint32_t ref_crc = 0; uint32_t elen = SD_EXPERT_BYTES; int have_ref = 0;
    if (rom) {
        static MoeHdr h __attribute__((aligned(16)));
        data_cache_hit_writeback_invalidate(&h, sizeof h);
        dma_read(&h, rom, sizeof h);
        if (h.magic == MOE_MAGIC && h.len <= SD_EXPERT_BYTES) {
            elen = h.len;
            canary_fill(ref, elen);
            /* PHYSICAL address: the raw async path does not mask KSEG1 (F-R028) */
            dma_read(ref, (rom + h.base) & 0x1FFFFFFFu, elen);
            data_cache_hit_writeback_invalidate(ref, elen);
            uint32_t c0, c1; CP0(c0);
            ref_crc = crc32(ref, elen);
            CP0(c1);
            have_ref = (ref[0] == 'S' && ref[1] == 'E' && ref[2] == 'Q');
            sprintf(pl, "SD REF len=%u crc=%08x ok=%d  CRCCOST cp0=%u (%u ms)\n",
                    (unsigned)elen, (unsigned)ref_crc, have_ref,
                    (unsigned)(c1 - c0), MS(c1 - c0));
            ISV(pl);
        }
    }
    if (!have_ref)
        ISV("SD WARN no cartridge reference expert: byte checks CANNOT run, "
            "so any timing below is UNVERIFIED and must not be quoted\n");

    /* ---- the card ------------------------------------------------------- */
    uint32_t t0, t1;
    CP0(t0);
    int ct = cart_init();
    CP0(t1);
    sprintf(pl, "SD CART cart_init=%d type=%d size=%u cp0=%u (%u ms)\n",
            ct, cart_type, (unsigned)cart_size, (unsigned)(t1 - t0), MS(t1 - t0));
    ISV(pl);
    if (ct < 0 || cart_type == CART_NULL) {
        /* An absent or unrecognised cart is the EXPECTED result under an
         * emulator with no flashcart.  It must print ABSENT and stop, never a
         * throughput number: "no card" must not be indistinguishable from
         * "infinitely fast card". */
        ISV("SD ABSENT no development cartridge detected - no throughput "
            "measured, thresholds not evaluated\n");
        ISV("SD_DONE\n");
        while (1) { }
    }

    CP0(t0);
    int ci = cart_card_init();
    CP0(t1);
    sprintf(pl, "SD CARD card_init=%d cp0=%u (%u ms) byteswap=%d\n",
            ci, (unsigned)(t1 - t0), MS(t1 - t0), (int)cart_card_byteswap);
    ISV(pl);
    if (ci != 0) {
        ISV("SD ABSENT card_init failed - no SD card inserted or unsupported\n");
        ISV("SD_DONE\n");
        while (1) { }
    }

    /* Resolve the expert file through FatFs once; its LBA extent is what the
     * raw-sector arms read, so every arm reads THE SAME BYTES. */
    uint32_t lba = 0; int have_file = 0;
    CP0(t0);
    int mnt = debug_init_sdfs("sd:/", -1);
    CP0(t1);
    sprintf(pl, "SD MOUNT sdfs=%d cp0=%u (%u ms)\n", mnt, (unsigned)(t1 - t0), MS(t1 - t0));
    ISV(pl);
    if (mnt) {
        FILE *f = fopen("sd:/elya/e00.sgai", "rb");
        if (f) { have_file = 1; fclose(f); }
    }
    if (!have_file)
        ISV("SD WARN sd:/elya/e00.sgai missing - FATFS arm skipped; raw-sector "
            "arms need SD_LBA compiled in\n");
#ifdef SD_LBA
    lba = (uint32_t)(SD_LBA);
#endif

    /* STAGE safety: the staging window must lie past the end of this ROM
     * image, or the FPGA DMA overwrites the code that is running. */
    int stage_ok = 0;
    {
        /* __rom_end is libdragon's own symbol (n64sys.h): the end of THIS
         * ROM image inside cartridge space.  Staging must start above it. */
        uint32_t rom_end = (uint32_t)(uintptr_t)__rom_end & 0x1FFFFFFFu;
        uint32_t stage = (uint32_t)(SD_STAGE_CART);
        stage_ok = stage && cart_size &&
                   (stage & 0x1FFFFFFFu) >= rom_end &&
                   ((stage & 0x1FFFFFFFu) - 0x10000000u) + elen <= cart_size;
        sprintf(pl, "SD STAGEWIN addr=%08x rom_end=%08x cart_size=%u usable=%d%s\n",
                (unsigned)stage, (unsigned)rom_end, (unsigned)cart_size, stage_ok,
                stage_ok ? "" : "  (STAGE arm skipped - would overwrite the running ROM)");
        ISV(pl);
    }

    const uint32_t sectors = (elen + SD_SECTOR - 1) / SD_SECTOR;
    uint32_t base_cp0 = 0;

    /* ------------------------------------------------------------------ arms */
    /* mode: 0 BASE, 1 BLOCK, 2 FATFS, 3 HIDE, 4 STAGE */
    for (int mode = 0; mode <= 4; mode++) {
        static const char *NAME[] = { "BASE ", "BLOCK", "FATFS", "HIDE ", "STAGE" };
        if (mode == 2 && !have_file) continue;
        if ((mode == 1 || mode == 3 || mode == 4) && !lba) continue;
        if (mode == 4 && !stage_ok) continue;

        canary_fill(dst, elen);
        sgai_reset(&ST);
        uint8_t tok = 0;
        const char *p = "Who are you?";
        for (int i = 0; p[i]; i++) tok = step((uint8_t)p[i]);

        uint32_t vb0 = g_vbl, c0, c1, io = 0, done = 0;
        FILE *f = NULL;
        if (mode == 2) f = fopen("sd:/elya/e00.sgai", "rb");
        CP0(c0);
        for (int i = 0; i < SD_NGEN; i++) {
            uint32_t i0, i1;
            CP0(i0);
            switch (mode) {
            case 1: cart_card_rd_dram(dst, lba, sectors); break;
            case 2: if (f) { fseek(f, 0, SEEK_SET); fread(dst, 1, elen, f); } break;
            case 3: {                       /* slice the read between tokens */
                uint32_t left = sectors > done ? sectors - done : 0;
                uint32_t take = left > SD_CHUNK_SECTORS ? SD_CHUNK_SECTORS : left;
                if (take) { cart_card_rd_dram(dst + done * SD_SECTOR, lba + done, take);
                            done += take; }
                break; }
            case 4:                          /* SD -> cart SDRAM (FPGA DMA) */
                cart_card_rd_cart(SD_STAGE_CART, lba, sectors);
                break;
            default: break;
            }
            CP0(i1); io += (i1 - i0);
            tok = step(tok);
        }
        if (mode == 3) {                     /* drain whatever is left */
            uint32_t i0, i1; CP0(i0);
            if (done < sectors)
                cart_card_rd_dram(dst + done * SD_SECTOR, lba + done, sectors - done);
            CP0(i1); io += (i1 - i0);
        }
        if (mode == 4) {                     /* cart SDRAM -> RDRAM, the hop that hides */
            uint32_t i0, i1; CP0(i0);
            data_cache_hit_writeback_invalidate(dst, elen);
            dma_read(dst, SD_STAGE_CART & 0x1FFFFFFFu, elen);
            CP0(i1); io += (i1 - i0);
        }
        CP0(c1);
        if (f) fclose(f);
        uint32_t vb1 = g_vbl;

        data_cache_hit_writeback_invalidate(dst, elen);
        uint32_t got = (mode == 0) ? 0 : crc32(dst, elen);
        const char *verdict = "n/a";
        if (mode != 0) {
            int canary = (dst[0] == CANARY && dst[elen - 1] == CANARY);
            verdict = !have_ref ? "UNVERIFIED"
                    : canary    ? "CANARY(no bytes moved)"
                    : (got == ref_crc) ? "ok" : "DIFF";
        }
        if (mode == 0) base_cp0 = c1 - c0;

        sprintf(pl, "SD %s cp0=%u (%u ms) vbl=%u io=%u (%u ms) crc=%08x %s\n",
                NAME[mode], (unsigned)(c1 - c0), MS(c1 - c0),
                (unsigned)(vb1 - vb0), (unsigned)io, MS(io),
                (unsigned)got, verdict);
        ISV(pl);

        if (mode != 0 && base_cp0) {
            uint32_t over = (c1 - c0) > base_cp0 ? (c1 - c0) - base_cp0 : 0;
            unsigned pct = (unsigned)((uint64_t)over * 1000u / base_cp0);
            unsigned kbs = io ? (unsigned)(((uint64_t)elen * SD_NGEN * CP0_HZ)
                                           / ((uint64_t)io * 1024u)) : 0;
            sprintf(pl, "SD %s over_base=%u.%u%% rate=%u KB/s%s\n", NAME[mode],
                    pct / 10, pct % 10, kbs,
                    (mode == 3 || mode == 4) ? (pct <= 100 ? "  T4/T5 PASS" : "  T4/T5 FAIL")
                                             : "");
            ISV(pl);
        }
    }

    ISV("SD_DONE\n");
    while (1) { }
}
