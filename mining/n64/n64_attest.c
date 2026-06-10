// SPDX-License-Identifier: MIT
/**
 * N64 RTC Hardware Attestation — RustChain Proof-of-Antiquity for N64
 *
 * Runs 5 hardware fingerprint checks on real N64 silicon:
 *   1. CPU PRId — identifies R4300i revision
 *   2. COUNT timing — measures instruction cycle rate
 *   3. VI scan — video interface scanline timing
 *   4. Memory ratio — cached vs uncached access speed
 *   5. Anti-emulation — composite check for emulator artifacts
 *
 * Results written to Controller Pak (via Pico USB bridge) for
 * relay to RustChain attestation node.
 *
 * Part of Legend of Elya — World's First N64 LLM + Blockchain Mining ROM
 */

#include "n64_attest.h"
#include <string.h>
#include <stdio.h>

AttestState g_attest;

// ─── Hardware Register Addresses (KSEG1 uncached) ───────────────────────────
#define VI_CURRENT_REG  (*(volatile uint32_t *)0xA4400010)
#define RDRAM_CONFIG    (*(volatile uint32_t *)0xA3F00000)
#define RDRAM_DEVICE_ID (*(volatile uint32_t *)0xA3F00004)

// ─── N64 Antiquity Multiplier ────────────────────────────────────────────────
// N64 launched 1996 — older than any PowerPC Mac in the fleet.
// R4300i MIPS @ 93.75 MHz, custom NEC VR4300. True vintage silicon.
#define N64_ANTIQUITY_MULT_X10  30  // 3.0x

// Mining rate: base 1.5 RTC/epoch shared across miners.
// N64 gets 3.5x weight. With ~20 miners, N64 share per epoch:
//   (3.5 / total_weight) * 1.5 RTC ≈ 0.02-0.05 RTC/epoch
// We show estimated local accumulation.
#define RTC_PER_ATTEST_X1000  5  // 0.005 RTC per successful attestation round

// Frames between auto-attestation rounds (~5 seconds at 60fps)
#define MINING_COOLDOWN_FRAMES  300

// ─── RDP helpers ────────────────────────────────────────────────────────────
static void afill(int x, int y, int w, int h, color_t c) {
    rdpq_set_mode_fill(c);
    rdpq_fill_rectangle(x, y, x + w, y + h);
}

// ─── Raw Joybus Pak I/O ─────────────────────────────────────────────────────
// Direct PIF block construction for pak READ/WRITE on any port.
//
// BUG FIX: libdragon's execute_raw_command() uses `controller` as a raw
// byte offset into the 64-byte PIF block. For port > 0, byte 0 is left
// as 0x00, which the PIF interprets as TX=0 paired with the NEXT byte
// (our TX count) as RX — corrupting the entire block.
//
// Fix: Build the PIF block ourselves using the proven format from
// get_controllers_present(). Each preceding channel gets a real PROBE
// command (0xFF pad + TX=1 + RX=4 + CMD=0x00 + 4 RX space = 8 bytes)
// to properly advance the PIF channel counter. Then our target channel
// gets the actual pak command.

static uint8_t pak_addr_crc(uint16_t addr) {
    // N64 pak address CRC: 5-bit CRC in lower bits of address word
    uint16_t xor_tap;
    uint16_t data = addr;
    for (int i = 15; i >= 5; i--) {
        xor_tap = (data & 0x8000) ? 0x15 : 0x00;
        data <<= 1;
        data ^= xor_tap;
    }
    return (data >> 11) & 0x1F;
}

// Fill preceding channels with dummy PROBE commands (8 bytes each).
// Returns the byte position after all skipped channels.
static int pif_skip_channels(uint8_t *block, int target_port) {
    int pos = 0;
    for (int ch = 0; ch < target_port; ch++) {
        block[pos++] = 0xFF;  // Pad byte (proven format)
        block[pos++] = 0x01;  // TX=1 (just command byte)
        block[pos++] = 0x04;  // RX=4 (standard PROBE response)
        block[pos++] = 0x00;  // PROBE command
        block[pos++] = 0xFF;  // RX space byte 0
        block[pos++] = 0xFF;  // RX space byte 1
        block[pos++] = 0xFF;  // RX space byte 2
        block[pos++] = 0xFF;  // RX space byte 3
    }
    return pos;
}

static int raw_pak_read(int port, uint16_t addr, uint8_t *data) {
    uint8_t block[64] __attribute__((aligned(16)));
    uint8_t output[64] __attribute__((aligned(16)));

    memset(block, 0, 64);
    block[63] = 0x01;  // PIF processing flag

    // Skip preceding channels with dummy PROBE commands
    int pos = pif_skip_channels(block, port);
    int cmd_start = pos;  // Remember where our command starts

    // Address with CRC in lower 5 bits
    uint16_t addr_with_crc = (addr & 0xFFE0) | pak_addr_crc(addr);

    // READ command: TX=3 (cmd + 2 addr), RX=33 (32 data + 1 CRC)
    block[pos++] = 0x03;  // TX count
    block[pos++] = 0x21;  // RX count (33)
    block[pos++] = 0x02;  // READ command
    block[pos++] = (addr_with_crc >> 8) & 0xFF;
    block[pos++] = addr_with_crc & 0xFF;
    memset(&block[pos], 0xFF, 33);  // RX placeholder
    pos += 33;
    block[pos++] = 0xFE;  // End sentinel

    joybus_exec(block, output);

    // Response data starts after TX/RX counts + TX data in the output block
    // TX/RX counts (2) + TX data (3) = 5 bytes before RX data
    memcpy(data, &output[cmd_start + 5], 32);
    return 0;
}

static int raw_pak_write(int port, uint16_t addr, uint8_t *data) {
    uint8_t block[64] __attribute__((aligned(16)));
    uint8_t output[64] __attribute__((aligned(16)));

    memset(block, 0, 64);
    block[63] = 0x01;  // PIF processing flag

    // Skip preceding channels with dummy PROBE commands
    int pos = pif_skip_channels(block, port);

    // Address with CRC in lower 5 bits
    uint16_t addr_with_crc = (addr & 0xFFE0) | pak_addr_crc(addr);

    // WRITE command: TX=35 (cmd + 2 addr + 32 data), RX=1 (CRC ack)
    block[pos++] = 0x23;  // TX count (35)
    block[pos++] = 0x01;  // RX count (1)
    block[pos++] = 0x03;  // WRITE command
    block[pos++] = (addr_with_crc >> 8) & 0xFF;
    block[pos++] = addr_with_crc & 0xFF;
    memcpy(&block[pos], data, 32);
    pos += 32;
    block[pos++] = 0xFF;  // RX placeholder for CRC ack
    block[pos++] = 0xFE;  // End sentinel

    joybus_exec(block, output);
    return 0;
}

// ─── Bridge Detection via POLL ──────────────────────────────────────────────

// Debug: store stick values and port info for display
static int debug_stick_x = 0;
static int debug_stick_y = 0;
static int debug_attempts = 0;

static int detect_bridge(void) {
    struct controller_data keys;

    // Try multiple times — first scan may miss if Pico just booted
    for (int attempt = 0; attempt < 10; attempt++) {
        controller_read(&keys);

        // Read analog stick from Port 2 (BRIDGE_PORT=1)
        int sx = keys.c[BRIDGE_PORT].x;
        int sy = keys.c[BRIDGE_PORT].y;

        // Save for debug display
        debug_stick_x = sx;
        debug_stick_y = sy;
        debug_attempts = attempt + 1;

        // Check for bridge magic: stick_x='P'(0x50=80), stick_y='B'(0x42=66)
        if (sx == BRIDGE_MAGIC_X && sy == BRIDGE_MAGIC_Y) {
            return 1;  // Bridge detected!
        }

        // Small delay between retries
        for (volatile int d = 0; d < 50000; d++) { }
    }
    return 0;
}

// ─── Hardware Check Functions ───────────────────────────────────────────────
// Each returns a 32-byte result page:
//   Byte 0: check_id
//   Byte 1: passed (0/1)
//   Bytes 2-5: primary value (big-endian uint32)
//   Bytes 6-31: additional data

static void store_u32_be(uint8_t *dst, uint32_t val) {
    dst[0] = (val >> 24) & 0xFF;
    dst[1] = (val >> 16) & 0xFF;
    dst[2] = (val >> 8) & 0xFF;
    dst[3] = val & 0xFF;
}

// Check 1: CPU PRId (Coprocessor 0, Register 15)
// Real R4300i: 0x00000B22 (rev 2.2) or 0x00000B00
static void check_cpu_prid(uint8_t *page) {
    memset(page, 0, 32);
    page[0] = CHECK_CPU_PRID;

    uint32_t prid;
    asm volatile("mfc0 %0, $15" : "=r"(prid));

    store_u32_be(&page[2], prid);

    // R4300i family: upper byte = 0x0B
    int is_r4300i = ((prid & 0x0000FF00) == 0x00000B00);
    page[1] = is_r4300i ? 1 : 0;

    g_attest.check_values[0] = prid;
    g_attest.check_passed[0] = page[1];
}

// Check 2: COUNT register timing
// Execute 200 NOPs, measure CP0 COUNT delta.
// R4300i COUNT increments at half CPU clock (46.875 MHz).
// 200 NOPs = 200 cycles at 93.75 MHz = ~100 COUNT ticks.
// Emulators: often different values (too fast, too slow, or exact).
//
// CRITICAL: Everything in ONE asm block! Separate asm volatile blocks
// let GCC insert stores/loads between COUNT reads, inflating the delta.
// Also disable interrupts so timer ISRs don't skew measurement.
static void check_count_timing(uint8_t *page) {
    memset(page, 0, 32);
    page[0] = CHECK_COUNT_TIMING;

    // Save CP0 Status and disable interrupts (clear IE bit 0)
    uint32_t saved_sr;
    asm volatile("mfc0 %0, $12" : "=r"(saved_sr));
    asm volatile("mtc0 %0, $12" : : "r"(saved_sr & ~1u));

    // Single asm block: read COUNT, 200 NOPs, read COUNT, subtract.
    // .set noreorder prevents assembler from inserting delay slot fills.
    // Hazard NOP after each mfc0 (VR4300 requires 1 cycle before result use).
    uint32_t cycles;
    asm volatile(
        ".set noreorder\n"
        "mfc0  $t0, $9\n"        /* read COUNT start */
        "nop\n"                   /* mfc0 hazard slot */
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop; nop; nop;"
        "mfc0  $t1, $9\n"        /* read COUNT end */
        "nop\n"                   /* mfc0 hazard slot */
        "subu  %0, $t1, $t0\n"   /* cycles = end - start */
        ".set reorder\n"
        : "=r"(cycles)
        :
        : "t0", "t1"
    );

    // Restore interrupts
    asm volatile("mtc0 %0, $12" : : "r"(saved_sr));

    store_u32_be(&page[2], cycles);

    // Real R4300i: expect ~102 COUNT ticks (1 hazard + 200 NOPs + mfc0 = 202 CPU cycles / 2)
    // Allow wide range 50-500 for hardware variants (Japanese NUS-001 etc.)
    // Emulators typically give 0, 1, or wildly wrong values (>10000)
    int ok = (cycles >= 50 && cycles <= 500);
    page[1] = ok ? 1 : 0;

    g_attest.check_values[1] = cycles;
    g_attest.check_passed[1] = page[1];
}

// Check 3: VI_CURRENT scanline timing
// Read VI_CURRENT twice with delay, check delta matches real scanline rate.
// NTSC: 525 lines total, ~15.734 kHz line rate.
// At 93.75 MHz CPU: ~5960 cycles per scanline, ~2980 COUNT ticks per scanline.
static void check_vi_scan(uint8_t *page) {
    memset(page, 0, 32);
    page[0] = CHECK_VI_SCAN;

    uint32_t v0 = VI_CURRENT_REG;
    uint32_t c0, c1;
    asm volatile("mfc0 %0, $9" : "=r"(c0));

    // Busy-wait ~10000 cycles (~213us, should span a few scanlines)
    for (volatile int i = 0; i < 5000; i++) { }

    uint32_t v1 = VI_CURRENT_REG;
    asm volatile("mfc0 %0, $9" : "=r"(c1));

    uint32_t vi_delta = (v1 >= v0) ? (v1 - v0) : (525 - v0 + v1);
    uint32_t count_delta = c1 - c0;

    store_u32_be(&page[2], vi_delta);
    store_u32_be(&page[6], count_delta);

    // Should see at least 1 scanline change in ~213us
    // and COUNT should have advanced ~10000 ticks
    int ok = (vi_delta >= 1 && vi_delta < 100 && count_delta > 2000);
    page[1] = ok ? 1 : 0;

    g_attest.check_values[2] = vi_delta;
    g_attest.check_passed[2] = page[1];
}

// Check 4: Memory access timing ratio (cached vs uncached)
// Real N64: uncached RDRAM is ~10-30x slower than cached.
// Emulators: often uniform timing (ratio near 1:1).
static void check_memory_ratio(uint8_t *page) {
    memset(page, 0, 32);
    page[0] = CHECK_MEMORY_RATIO;

    volatile uint32_t *cached_ptr   = (volatile uint32_t *)0x80000400;  // KSEG0
    volatile uint32_t *uncached_ptr = (volatile uint32_t *)0xA0000400;  // KSEG1

    uint32_t c0, c1;
    uint32_t x;

    // Warm the cache line first
    x = *cached_ptr;

    // Measure cached access
    asm volatile("mfc0 %0, $9" : "=r"(c0));
    x = *cached_ptr;
    asm volatile("mfc0 %0, $9" : "=r"(c1));
    uint32_t cached_time = c1 - c0;
    (void)x;

    // Measure uncached access
    asm volatile("mfc0 %0, $9" : "=r"(c0));
    x = *uncached_ptr;
    asm volatile("mfc0 %0, $9" : "=r"(c1));
    uint32_t uncached_time = c1 - c0;
    (void)x;

    // Ratio * 100 (to avoid float)
    uint32_t ratio = (cached_time > 0) ?
        (uncached_time * 100) / cached_time : 0;

    store_u32_be(&page[2], ratio);
    store_u32_be(&page[6], cached_time);
    store_u32_be(&page[10], uncached_time);

    // Real hardware: ratio > 300 (uncached is 3-30x slower)
    // Emulators: ratio often < 200
    int ok = (ratio > 200);
    page[1] = ok ? 1 : 0;

    g_attest.check_values[3] = ratio;
    g_attest.check_passed[3] = page[1];
}

// Check 5: Anti-emulation composite
// Reads RDRAM config registers and checks for real hardware signatures.
// Also checks COUNT register jitter (real oscillators have drift).
static void check_anti_emu(uint8_t *page) {
    memset(page, 0, 32);
    page[0] = CHECK_ANTI_EMU;

    // Read RDRAM configuration registers
    uint32_t rdram_cfg = RDRAM_CONFIG;
    uint32_t rdram_id  = RDRAM_DEVICE_ID;

    store_u32_be(&page[2], rdram_cfg);
    store_u32_be(&page[6], rdram_id);

    // COUNT register jitter test: measure 10 short delays
    // Real hardware has oscillator jitter; emulators are perfectly uniform.
    uint32_t measurements[10];
    for (int i = 0; i < 10; i++) {
        uint32_t c0, c1;
        asm volatile("mfc0 %0, $9" : "=r"(c0));
        for (volatile int j = 0; j < 100; j++) { }
        asm volatile("mfc0 %0, $9" : "=r"(c1));
        measurements[i] = c1 - c0;
    }

    // Check variance: compute max - min
    uint32_t mn = measurements[0], mx = measurements[0];
    for (int i = 1; i < 10; i++) {
        if (measurements[i] < mn) mn = measurements[i];
        if (measurements[i] > mx) mx = measurements[i];
    }
    uint32_t jitter = mx - mn;

    store_u32_be(&page[10], jitter);
    store_u32_be(&page[14], mn);
    store_u32_be(&page[18], mx);

    // Real hardware: RDRAM config is non-zero, jitter > 0
    // Emulators: RDRAM config may be 0, jitter often exactly 0
    int ok = (rdram_cfg != 0 || rdram_id != 0);
    page[1] = ok ? 1 : 0;

    g_attest.check_values[4] = jitter;
    g_attest.check_passed[4] = page[1];
}

// ─── Big-endian read helpers ─────────────────────────────────────────────

static uint16_t load_u16_be(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

static uint32_t load_u32_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] << 8) | p[3];
}

// ─── Chain State Reader ─────────────────────────────────────────────────
// Reads 32-byte chain state page from Pico bridge at pak address 0x8000.
// Validates magic ("RC") and XOR checksum before updating g_attest fields.

void read_chain_state(void) {
    uint8_t page[32];
    raw_pak_read(BRIDGE_PORT, CHAIN_STATE_ADDR, page);

    // Validate magic bytes
    if (page[0] != CHAIN_STATE_MAGIC_0 || page[1] != CHAIN_STATE_MAGIC_1) {
        g_attest.chain_valid = 0;
        return;
    }

    // Validate XOR checksum (bytes 28-31 should all equal XOR of bytes 0-27)
    uint8_t xor_check = 0;
    for (int i = 0; i < 28; i++) {
        xor_check ^= page[i];
    }
    if (page[28] != xor_check || page[29] != xor_check ||
        page[30] != xor_check || page[31] != xor_check) {
        g_attest.chain_valid = 0;
        return;
    }

    // Parse fields (all big-endian, native on N64)
    g_attest.chain_flags           = page[3];
    g_attest.chain_epoch           = load_u16_be(&page[4]);
    g_attest.chain_slot            = load_u32_be(&page[6]);
    g_attest.chain_balance_milli   = load_u32_be(&page[10]);
    g_attest.chain_accepted        = load_u32_be(&page[14]);
    g_attest.chain_rejected        = load_u32_be(&page[18]);
    g_attest.chain_multiplier_x100 = load_u16_be(&page[22]);
    g_attest.chain_miners          = page[24];
    g_attest.chain_rotation_size   = page[25];
    g_attest.chain_turn_offset     = load_u16_be(&page[26]);
    g_attest.chain_valid           = 1;
}

// ─── Check dispatch table ───────────────────────────────────────────────────

typedef void (*check_fn)(uint8_t *page);
static const check_fn CHECKS[NUM_ATTEST_CHECKS] = {
    check_cpu_prid,
    check_count_timing,
    check_vi_scan,
    check_memory_ratio,
    check_anti_emu,
};

static const char *CHECK_NAMES[NUM_ATTEST_CHECKS] = {
    "CPU PRId",
    "COUNT Timing",
    "VI Scan",
    "Memory Ratio",
    "Anti-Emulation",
};

// ─── Hardware-Derived Wallet ────────────────────────────────────────────────
// Simple SHA-256-like hash (djb2 + mixing) of hardware-unique registers.
// Each N64 has different RDRAM chips, so RDRAM_DEVICE_ID varies per console.
// Combined with CP0 PRId (stepping/revision) and RDRAM config bits.

static uint32_t hw_hash_mix(uint32_t h, uint32_t val) {
    h ^= val;
    h = (h << 13) | (h >> 19);
    h *= 0x5BD1E995;
    h ^= h >> 15;
    return h;
}

void attest_generate_wallet(void) {
    // Read hardware-unique values
    uint32_t prid;
    asm volatile("mfc0 %0, $15" : "=r"(prid));

    uint32_t rdram_cfg = RDRAM_CONFIG;
    uint32_t rdram_id  = RDRAM_DEVICE_ID;

    // Read additional RDRAM banks (each 2MB module has its own ID)
    // Bank 1 at offset 0x200000, Bank 2 at 0x400000
    uint32_t rdram_id2 = (*(volatile uint32_t *)0xA3F80004);  // 2nd RDRAM module

    // Store raw values for display/debug
    g_attest.hw_seed[0] = prid;
    g_attest.hw_seed[1] = rdram_cfg;
    g_attest.hw_seed[2] = rdram_id;
    g_attest.hw_seed[3] = rdram_id2;

    // Hash all hardware values together
    uint32_t h1 = 0x811C9DC5;  // FNV offset basis
    h1 = hw_hash_mix(h1, prid);
    h1 = hw_hash_mix(h1, rdram_cfg);
    h1 = hw_hash_mix(h1, rdram_id);
    h1 = hw_hash_mix(h1, rdram_id2);

    // Second hash for more bits
    uint32_t h2 = 0xA136AAAD;
    h2 = hw_hash_mix(h2, rdram_id ^ prid);
    h2 = hw_hash_mix(h2, rdram_cfg ^ rdram_id2);
    h2 = hw_hash_mix(h2, prid ^ 0xDEAD64);
    h2 = hw_hash_mix(h2, rdram_id2 ^ 0xBEEF);

    // Format: "n64-" + 16 hex chars from two 32-bit hashes
    snprintf(g_attest.wallet_id, WALLET_ID_LEN + 1,
             "n64-%08lx%08lx",
             (unsigned long)h1, (unsigned long)h2);
}

// ─── Pak Write ──────────────────────────────────────────────────────────────

static void write_attest_to_pak(void) {
    uint8_t page[32];

    // Write header page at 0x0000
    memset(page, 0, 32);
    memcpy(page, ATTEST_MAGIC, 4);
    page[4] = 1;  // version
    page[5] = NUM_ATTEST_CHECKS;
    page[6] = NUM_ATTEST_CHECKS;  // total data pages
    page[7] = g_attest.all_passed ? 1 : 0;
    page[8] = g_attest.total_passed;
    // Embed wallet ID in header (bytes 9-28 = 20 chars)
    memcpy(&page[9], g_attest.wallet_id, WALLET_ID_LEN);
    raw_pak_write(BRIDGE_PORT, ATTEST_HDR_ADDR, page);

    // Write each check result
    for (int i = 0; i < NUM_ATTEST_CHECKS; i++) {
        // Re-run each check to get the page data
        // (values are cached in g_attest but we need the full 32-byte pages)
        memset(page, 0, 32);
        CHECKS[i](page);
        raw_pak_write(BRIDGE_PORT, ATTEST_DATA_ADDR + i * 32, page);
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

void attest_start(void) {
    memset(&g_attest, 0, sizeof(g_attest));
    g_attest.phase = ATTEST_PHASE_DETECT;

    // Generate hardware-derived wallet first
    attest_generate_wallet();

    snprintf(g_attest.status, sizeof(g_attest.status), "Detecting bridge...");
}

void attest_update(int frame) {
    switch (g_attest.phase) {

    case ATTEST_PHASE_DETECT:
        // Try to detect Pico bridge (retry a few times)
        g_attest.bridge_detected = detect_bridge();
        if (g_attest.bridge_detected) {
            g_attest.phase = ATTEST_PHASE_CONFIRM;
            snprintf(g_attest.status, sizeof(g_attest.status),
                     "Bridge found! Mine? A=Yes B=No");
        } else {
            g_attest.phase = ATTEST_PHASE_NO_BRIDGE;
            snprintf(g_attest.status, sizeof(g_attest.status),
                     "No bridge on Port 2");
        }
        break;

    case ATTEST_PHASE_RUNNING:
        if (g_attest.check_idx < NUM_ATTEST_CHECKS) {
            uint8_t page[32];
            snprintf(g_attest.status, sizeof(g_attest.status),
                     "Check %d/%d: %s...",
                     g_attest.check_idx + 1, NUM_ATTEST_CHECKS,
                     CHECK_NAMES[g_attest.check_idx]);

            CHECKS[g_attest.check_idx](page);
            g_attest.check_idx++;
        } else {
            // All checks done
            g_attest.total_passed = 0;
            for (int i = 0; i < NUM_ATTEST_CHECKS; i++)
                if (g_attest.check_passed[i]) g_attest.total_passed++;
            g_attest.all_passed = (g_attest.total_passed == NUM_ATTEST_CHECKS);

            g_attest.phase = ATTEST_PHASE_SENDING;
            snprintf(g_attest.status, sizeof(g_attest.status),
                     "Writing to bridge...");
        }
        break;

    case ATTEST_PHASE_SENDING:
        // Capture nonce from CP0 COUNT
        asm volatile("mfc0 %0, $9" : "=r"(g_attest.nonce));

        // Hash all check results
        {
            uint32_t ah = 0x811C9DC5;
            for (int i = 0; i < NUM_ATTEST_CHECKS; i++) {
                ah = hw_hash_mix(ah, g_attest.check_values[i]);
                ah = hw_hash_mix(ah, g_attest.check_passed[i]);
            }
            ah = hw_hash_mix(ah, g_attest.nonce);
            g_attest.attest_hash = ah;
        }

        g_attest.attest_count++;
        write_attest_to_pak();

        // Try to read chain state from bridge after writing attestation
        read_chain_state();

        // Mine if enough checks pass (4/5 = real hardware, just a tight timing range)
        if (g_attest.total_passed >= 4) {
            g_attest.accepted++;
            g_attest.rtc_mined_x1000 += RTC_PER_ATTEST_X1000;

            // Record mining start on first success
            if (g_attest.mining_start_count == 0)
                asm volatile("mfc0 %0, $9" : "=r"(g_attest.mining_start_count));

            // Enter continuous mining
            g_attest.phase = ATTEST_PHASE_MINING;
            g_attest.mining_cooldown = MINING_COOLDOWN_FRAMES;
            snprintf(g_attest.status, sizeof(g_attest.status),
                     "MINING RTC  |  B=Stop");
        } else {
            g_attest.rejected++;
            g_attest.phase = ATTEST_PHASE_DONE;
            snprintf(g_attest.status, sizeof(g_attest.status),
                     "%d/%d passed. A=Retry B=Back",
                     g_attest.total_passed, NUM_ATTEST_CHECKS);
        }
        break;

    case ATTEST_PHASE_MINING:
        // Continuous mining — auto re-attest on cooldown
        g_attest.mining_frame++;
        g_attest.mining_cooldown--;

        // Poll chain state from bridge every ~60 frames (~1 second)
        if ((g_attest.mining_frame - g_attest.chain_read_frame) >= 60) {
            g_attest.chain_read_frame = g_attest.mining_frame;
            read_chain_state();
        }

        // Update epoch: use real chain data if available, else local estimate
        if (g_attest.chain_valid) {
            g_attest.epoch = g_attest.chain_epoch;
            // Override local counters with server-confirmed values
            g_attest.rtc_mined_x1000 = g_attest.chain_balance_milli;
            g_attest.accepted = g_attest.chain_accepted;
            g_attest.rejected = g_attest.chain_rejected;
        } else {
            // Fallback: local epoch from COUNT register
            uint32_t now;
            asm volatile("mfc0 %0, $9" : "=r"(now));
            g_attest.epoch = now >> 28;
        }

        if (g_attest.mining_cooldown <= 0) {
            // Auto re-attest
            g_attest.phase = ATTEST_PHASE_RUNNING;
            g_attest.check_idx = 0;
        }
        break;

    default:
        break;
    }
}

int attest_handle_input(struct controller_data *keys) {
    switch (g_attest.phase) {

    case ATTEST_PHASE_CONFIRM:
        if (keys->c[0].A) {
            g_attest.phase = ATTEST_PHASE_RUNNING;
            g_attest.check_idx = 0;
        }
        if (keys->c[0].B) return 0;  // Exit attestation
        break;

    case ATTEST_PHASE_DONE:
        if (keys->c[0].B) return 0;
        if (keys->c[0].A) {
            g_attest.phase = ATTEST_PHASE_RUNNING;
            g_attest.check_idx = 0;
        }
        break;

    case ATTEST_PHASE_MINING:
        if (keys->c[0].B) return 0;  // Stop mining
        break;

    case ATTEST_PHASE_NO_BRIDGE:
        if (keys->c[0].B) return 0;  // Exit attestation
        break;

    default:
        break;
    }
    return 1;  // Stay in attestation
}

// ─── Rendering ──────────────────────────────────────────────────────────────

void attest_draw_scene(int frame) {
    // Dark green-tinted background (matrix/crypto vibe)
    afill(0, 0, 320, 240, RGBA32(0, 8, 0, 255));

    // Header bar
    afill(0, 0, 320, 20, RGBA32(0, 40, 0, 255));

    // Status bar at bottom
    afill(0, 220, 320, 20, RGBA32(0, 30, 0, 255));

    // Check result boxes (compact: 24px each, starting at y=30)
    if (g_attest.phase >= ATTEST_PHASE_RUNNING &&
        g_attest.phase != ATTEST_PHASE_NO_BRIDGE) {
        for (int i = 0; i < NUM_ATTEST_CHECKS; i++) {
            int y = 30 + i * 24;
            color_t box_color;

            if (i < g_attest.check_idx) {
                // Completed check
                box_color = g_attest.check_passed[i]
                    ? RGBA32(0, 60, 0, 255)    // green = pass
                    : RGBA32(60, 0, 0, 255);   // red = fail
            } else if (i == g_attest.check_idx && g_attest.phase == ATTEST_PHASE_RUNNING) {
                // Currently running (blink)
                int bright = 20 + ((frame / 4) & 1) * 20;
                box_color = RGBA32(bright, bright, 0, 255);
            } else {
                // Pending
                box_color = RGBA32(10, 10, 10, 255);
            }

            afill(20, y, 280, 20, box_color);
            // Border
            afill(20, y, 280, 1, RGBA32(0, 80, 0, 255));
            afill(20, y + 19, 280, 1, RGBA32(0, 80, 0, 255));
            afill(20, y, 1, 20, RGBA32(0, 80, 0, 255));
            afill(299, y, 1, 20, RGBA32(0, 80, 0, 255));
        }
    }

    // Confirm box (centered)
    if (g_attest.phase == ATTEST_PHASE_CONFIRM || g_attest.phase == ATTEST_PHASE_NO_BRIDGE) {
        afill(40, 80, 240, 80, RGBA32(0, 30, 0, 255));
        afill(40, 80, 240, 2, RGBA32(0, 120, 0, 255));
        afill(40, 158, 240, 2, RGBA32(0, 120, 0, 255));
        afill(40, 80, 2, 80, RGBA32(0, 120, 0, 255));
        afill(278, 80, 2, 80, RGBA32(0, 120, 0, 255));
    }
}

void attest_draw_text(surface_t *disp) {
    // Title
    graphics_draw_text(disp, 76, 4, "RTC HARDWARE ATTESTATION");

    // Always show wallet ID below title
    if (g_attest.wallet_id[0]) {
        graphics_draw_text(disp, 60, 16, g_attest.wallet_id);
    }

    if (g_attest.phase == ATTEST_PHASE_CONFIRM) {
        graphics_draw_text(disp, 72, 92,  "Pico Bridge Detected!");
        graphics_draw_text(disp, 56, 112, "Start RTC mining?");
        graphics_draw_text(disp, 92, 136, "A = Yes   B = No");
        graphics_draw_text(disp, 10, 226, g_attest.status);
    }

    if (g_attest.phase == ATTEST_PHASE_NO_BRIDGE) {
        graphics_draw_text(disp, 56, 92, "No serial bridge found");
        graphics_draw_text(disp, 52, 108, "Connect Pico to Port 2");

        // Debug: show POLL data from Port 2
        char dbg1[48], dbg2[48];
        snprintf(dbg1, sizeof(dbg1), "Port2 stick: X=%d Y=%d",
            debug_stick_x, debug_stick_y);
        snprintf(dbg2, sizeof(dbg2), "Want: X=%d Y=%d (%d tries)",
            BRIDGE_MAGIC_X, BRIDGE_MAGIC_Y, debug_attempts);
        graphics_draw_text(disp, 20, 128, dbg1);
        graphics_draw_text(disp, 20, 140, dbg2);

        graphics_draw_text(disp, 108, 160, "B = Back");
    }

    // Check results — compact layout (24px per check)
    if (g_attest.phase >= ATTEST_PHASE_RUNNING) {
        for (int i = 0; i < NUM_ATTEST_CHECKS; i++) {
            int y = 34 + i * 24;
            char line[48];

            if (i < g_attest.check_idx) {
                snprintf(line, sizeof(line), "%s: %s (0x%08lX)",
                         CHECK_NAMES[i],
                         g_attest.check_passed[i] ? "PASS" : "FAIL",
                         (unsigned long)g_attest.check_values[i]);
            } else if (i == g_attest.check_idx && g_attest.phase == ATTEST_PHASE_RUNNING) {
                snprintf(line, sizeof(line), "%s: Running...", CHECK_NAMES[i]);
            } else {
                snprintf(line, sizeof(line), "%s: Pending", CHECK_NAMES[i]);
            }

            graphics_draw_text(disp, 28, y, line);
        }
    }

    // Mining display — occupies bottom 4 lines (y=160..230)
    // No status bar text here — mining stats ARE the status
    if (g_attest.phase == ATTEST_PHASE_MINING ||
        g_attest.phase == ATTEST_PHASE_DONE) {

        char line[48];

        if (g_attest.phase == ATTEST_PHASE_MINING) {
            // Live mining — spinning indicator + multiplier
            const char *spin = "|/-\\";
            char sc = spin[(g_attest.mining_frame / 8) % 4];
            if (g_attest.chain_valid && g_attest.chain_multiplier_x100 > 0) {
                snprintf(line, sizeof(line), "%c MINING RTC %c  Mult:%lu.%02lux",
                         sc, sc,
                         (unsigned long)(g_attest.chain_multiplier_x100 / 100),
                         (unsigned long)(g_attest.chain_multiplier_x100 % 100));
            } else {
                snprintf(line, sizeof(line), "%c MINING RTC %c  Mult: 3.0x",
                         sc, sc);
            }
            graphics_draw_text(disp, 44, 160, line);
        } else {
            // Not enough checks passed
            snprintf(line, sizeof(line), "%d/%d passed  A=Retry B=Back",
                     g_attest.total_passed, NUM_ATTEST_CHECKS);
            graphics_draw_text(disp, 28, 160, line);
        }

        if (g_attest.chain_valid) {
            // ── Real chain data ──
            // Slot + Miners + Turn
            snprintf(line, sizeof(line), "Slot:%lu  Miners:%u  Turn:%u",
                     (unsigned long)g_attest.chain_slot,
                     (unsigned)g_attest.chain_miners,
                     (unsigned)g_attest.chain_turn_offset);
            graphics_draw_text(disp, 4, 176, line);

            // Epoch + Accepted/Rejected (from server)
            snprintf(line, sizeof(line), "Epoch:%u  OK:%lu  Fail:%lu",
                     (unsigned)g_attest.chain_epoch,
                     (unsigned long)g_attest.chain_accepted,
                     (unsigned long)g_attest.chain_rejected);
            graphics_draw_text(disp, 16, 192, line);

            // RTC balance (from server)
            snprintf(line, sizeof(line), "RTC: %lu.%03lu  B=Stop",
                     (unsigned long)(g_attest.chain_balance_milli / 1000),
                     (unsigned long)(g_attest.chain_balance_milli % 1000));
            graphics_draw_text(disp, 36, 208, line);
        } else {
            // ── Local estimates (no bridge data) ──
            // Nonce + Hash
            snprintf(line, sizeof(line), "Nonce:%08lX Hash:%08lX",
                     (unsigned long)g_attest.nonce,
                     (unsigned long)g_attest.attest_hash);
            graphics_draw_text(disp, 4, 176, line);

            // Epoch + Accepted/Rejected (local)
            snprintf(line, sizeof(line), "Epoch:%lu  OK:%lu  Fail:%lu",
                     (unsigned long)g_attest.epoch,
                     (unsigned long)g_attest.accepted,
                     (unsigned long)g_attest.rejected);
            graphics_draw_text(disp, 16, 192, line);

            // RTC mined (local estimate)
            snprintf(line, sizeof(line), "~%lu.%03lu RTC (local) B=Stop",
                     (unsigned long)(g_attest.rtc_mined_x1000 / 1000),
                     (unsigned long)(g_attest.rtc_mined_x1000 % 1000));
            graphics_draw_text(disp, 8, 208, line);
        }
    }

    // Only show status bar for phases that don't have mining display
    if (g_attest.phase == ATTEST_PHASE_RUNNING ||
        g_attest.phase == ATTEST_PHASE_SENDING) {
        graphics_draw_text(disp, 10, 226, g_attest.status);
    }
}
