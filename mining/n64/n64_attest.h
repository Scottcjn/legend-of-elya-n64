// SPDX-License-Identifier: MIT
#ifndef N64_ATTEST_H
#define N64_ATTEST_H

#include <libdragon.h>

// ─── Hardware Check IDs ─────────────────────────────────────────────────────
#define CHECK_CPU_PRID      1   // CP0 PRId register (R4300i identifier)
#define CHECK_COUNT_TIMING  2   // CP0 COUNT cycles for known instruction sequence
#define CHECK_VI_SCAN       3   // VI_CURRENT scanline timing
#define CHECK_MEMORY_RATIO  4   // Cached vs uncached memory access ratio
#define CHECK_ANTI_EMU      5   // Anti-emulation composite check
#define NUM_ATTEST_CHECKS   5

// ─── Bridge Detection via POLL ──────────────────────────────────────────────
// Pico reports stick_x=0x50 ('P'), stick_y=0x42 ('B') in POLL response.
// N64 reads these from controller_read() — uses the PROVEN working 4-byte path.
// (Pak READ returns all 0xFF due to timing issue — POLL always works.)
// Controller port for Pico bridge (0=Port1, 1=Port2, 2=Port3, 3=Port4)
// Port 1 reserved for real controller + EverDrive cart OS
#define BRIDGE_PORT         1

#define BRIDGE_MAGIC_X      0x50  // 'P' = 80 decimal (signed 8-bit safe)
#define BRIDGE_MAGIC_Y      0x42  // 'B' = 66 decimal (signed 8-bit safe)

// Pak addresses (still used for attestation data WRITE — N64→Pico direction works)
#define BRIDGE_DETECT_ADDR  0x7FE0
#define BRIDGE_MAGIC        "PICOBRIDGE"

// Attestation data written to pak starting at 0x0000
// Page 0x0000: header [N64A, ver, num_checks, total_pages, ...]
// Page 0x0020: check 1 result (32 bytes)
// Page 0x0040: check 2 result (32 bytes)
// etc.
#define ATTEST_HDR_ADDR     0x0000
#define ATTEST_DATA_ADDR    0x0020
#define ATTEST_MAGIC        "N64A"

// ─── Attestation Phases ─────────────────────────────────────────────────────
#define ATTEST_PHASE_DETECT     0   // Detecting bridge
#define ATTEST_PHASE_CONFIRM    1   // "Bridge found! Mine? A=Yes B=No"
#define ATTEST_PHASE_RUNNING    2   // Running hardware checks
#define ATTEST_PHASE_SENDING    3   // Writing results to pak
#define ATTEST_PHASE_DONE       4   // All done, show results
#define ATTEST_PHASE_NO_BRIDGE  5   // No bridge detected
#define ATTEST_PHASE_MINING     6   // Continuous mining loop

// ─── Hardware-Derived Wallet ──────────────────────────────────────────────
// Each N64 gets a unique miner ID from its silicon:
//   SHA256(RDRAM_CONFIG | RDRAM_DEVICE_ID | CP0_PRId | RDRAM banks)
// Same ROM works for everyone. Wallet is hardware-bound.
#define WALLET_ID_LEN  20   // "n64-" + 16 hex chars

// ─── Chain State (Host → Pico → N64) ────────────────────────────────────
// 32-byte page at pak address 0x8000, populated by host bridge.
// Carries real epoch, slot, balance, accepted/rejected from live node.
#define CHAIN_STATE_ADDR    0x8000
#define CHAIN_STATE_MAGIC_0 0x52  // 'R'
#define CHAIN_STATE_MAGIC_1 0x43  // 'C'

// ─── Mining State ─────────────────────────────────────────────────────────
// RustChain epoch = (time - genesis) / 600. N64 has no RTC, so we derive
// a nonce from CP0 COUNT and track attestation rounds locally.
#define RTC_EPOCH_SECS      600             // 10 minutes per epoch

// ─── Attestation State ──────────────────────────────────────────────────────
typedef struct {
    int phase;
    int check_idx;                          // Current check being run (0-4)
    uint8_t check_passed[NUM_ATTEST_CHECKS];
    uint32_t check_values[NUM_ATTEST_CHECKS];
    int bridge_detected;
    int all_passed;
    int total_passed;
    char status[48];
    char wallet_id[WALLET_ID_LEN + 1];     // Hardware-derived miner ID
    uint32_t hw_seed[4];                    // Raw hardware values used for wallet
    // Mining state — continuous auto-attest
    uint32_t nonce;                         // CP0 COUNT at attestation time
    uint32_t attest_hash;                   // Hash of check results
    uint32_t attest_count;                  // Attestation rounds completed
    uint32_t accepted;                      // Accepted attestations
    uint32_t rejected;                      // Rejected attestations
    uint32_t mining_frame;                  // Frame counter for animation
    uint32_t epoch;                         // Current epoch (derived from COUNT)
    uint32_t epoch_start_count;             // COUNT value at mining start
    uint32_t mining_start_count;            // COUNT at first attestation
    uint32_t rtc_mined_x1000;              // milli-RTC earned (display as /1000)
    int mining_cooldown;                    // Frames between auto-attestations
    // Chain state — real data from RustChain node (via Pico bridge)
    int chain_valid;                        // 1 if chain state page validated
    uint16_t chain_epoch;                   // Real epoch from node
    uint32_t chain_slot;                    // Real slot from node
    uint32_t chain_balance_milli;           // Balance in milli-RTC (5250 = 5.250 RTC)
    uint32_t chain_accepted;               // Server-confirmed accepted
    uint32_t chain_rejected;               // Server-confirmed rejected
    uint16_t chain_multiplier_x100;        // Antiquity multiplier * 100 (300 = 3.0x)
    uint8_t chain_miners;                  // Active miners enrolled
    uint8_t chain_rotation_size;           // Lottery rotation size
    uint16_t chain_turn_offset;            // Slots until your turn
    uint8_t chain_flags;                   // Bit0=last_accepted, Bit1=eligible, Bit2=data_valid
    uint32_t chain_read_frame;             // Last frame we read chain state
} AttestState;

extern AttestState g_attest;

// ─── API ────────────────────────────────────────────────────────────────────
void attest_start(void);                    // Begin attestation flow
void attest_update(int frame);              // Per-frame update (call when STATE_ATTEST)
void attest_draw_scene(int frame);          // RDP scene (dark background, status boxes)
void attest_draw_text(surface_t *disp);     // CPU text overlay
int  attest_handle_input(struct controller_data *keys);  // Returns 0 to exit attestation
void attest_generate_wallet(void);          // Generate hardware-derived wallet ID
void read_chain_state(void);               // Read chain state from Pico bridge (pak 0x8000)

#endif
