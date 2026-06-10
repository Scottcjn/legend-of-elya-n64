// SPDX-License-Identifier: MIT
// N64 Pico Bridge - Full Attestation Relay
// Handles POLL (bridge magic), READ, and WRITE (pak data relay)
// Pak WRITE data forwarded over USB serial as PAK_W:ADDR:HEXDATA
// for host bridge (n64_bridge.py) to submit to RustChain node.
//
// Pin scanner confirmed: Green wire = GP1 = joybus data

#include "joybus.h"
#include "n64_definitions.h"

#include <hardware/clocks.h>
#include <pico/stdlib.h>
#include <stdio.h>
#include <string.h>

#define DATA_PIN  1   // GP1 = Green wire = joybus data (confirmed by scanner)

// Joybus timing constants (matching N64Console.hpp values)
static const uint REPLY_DELAY_US = 4;    // Wait before responding
static const uint BYTE_TIMEOUT_US = 50;  // Timeout between bytes (each byte ~36us)
static const uint RESET_WAIT_US = 100;   // Wait after bad command before reset

// N64 controller pak data CRC-8
// The N64 validates this on READ responses and WRITE acks.
static uint8_t pak_data_crc(const uint8_t *data, int len) {
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) {
        for (int bit = 7; bit >= 0; bit--) {
            uint8_t xor_tap = (crc & 0x80) ? 0x85 : 0x00;
            crc <<= 1;
            if (data[i] & (1 << bit))
                crc |= 1;
            crc ^= xor_tap;
        }
    }
    return crc;
}

// Hex char helper for fast serial output (avoid printf overhead in hot path)
static const char HEX[] = "0123456789ABCDEF";

// ─── Chain State Buffer (Host → Pico → N64) ─────────────────────────────
// 32-byte page populated by host bridge via "CHAIN:HEXDATA\n" serial command.
// Served to N64 on READ at address 0x8000.
static uint8_t chain_state[32];
static bool chain_state_valid = false;

// Hex character to nibble value
static inline int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

// Non-blocking USB serial reader — parses "CHAIN:64hexchars\n" commands
// from host bridge without stalling joybus processing.
static char serial_buf[80];
static int serial_pos = 0;

static void check_serial_input() {
    while (true) {
        int c = getchar_timeout_us(0);  // Non-blocking
        if (c == PICO_ERROR_TIMEOUT) break;

        if (c == '\n' || c == '\r') {
            serial_buf[serial_pos] = '\0';

            // Parse CHAIN: command
            if (serial_pos >= 70 && serial_buf[0] == 'C' && serial_buf[1] == 'H'
                && serial_buf[2] == 'A' && serial_buf[3] == 'I'
                && serial_buf[4] == 'N' && serial_buf[5] == ':') {
                // Decode 64 hex chars → 32 bytes
                bool ok = true;
                uint8_t temp[32];
                for (int i = 0; i < 32; i++) {
                    int hi = hex_val(serial_buf[6 + i * 2]);
                    int lo = hex_val(serial_buf[6 + i * 2 + 1]);
                    if (hi < 0 || lo < 0) { ok = false; break; }
                    temp[i] = (hi << 4) | lo;
                }
                if (ok) {
                    // Validate magic bytes (0x52='R', 0x43='C')
                    if (temp[0] == 0x52 && temp[1] == 0x43) {
                        memcpy(chain_state, temp, 32);
                        chain_state_valid = true;
                        printf("CHAIN_OK\n");
                    } else {
                        printf("CHAIN_ERR:bad_magic\n");
                    }
                } else {
                    printf("CHAIN_ERR:bad_hex\n");
                }
            }

            serial_pos = 0;
        } else if (serial_pos < (int)(sizeof(serial_buf) - 1)) {
            serial_buf[serial_pos++] = (char)c;
        }
    }
}

int main(void) {
    set_sys_clock_khz(130000, true);
    stdio_init_all();

    // Wait for USB CDC
    sleep_ms(2000);
    printf("\n\n=== N64 PICO BRIDGE v2 ===\n");
    printf("SYSCLK=%lu\n", clock_get_hz(clk_sys));
    printf("Data pin: GP%d\n", DATA_PIN);
    printf("Mode: Full attestation relay (POLL + READ + WRITE)\n");

    // Controller report with bridge magic signature
    n64_report_t report = default_n64_report;
    report.stick_x = 0x50;  // 'P'
    report.stick_y = 0x42;  // 'B'

    // Probe status: device=0x0500 (standard controller), status=0x01 (pak present)
    // status=0x01 tells N64 that a controller pak is connected,
    // enabling pak READ/WRITE commands from the console.
    n64_status_t status = default_n64_status;
    status.status = 0x01;  // Pak present flag

    // Configure data pin for joybus
    gpio_pull_up(DATA_PIN);
    gpio_set_drive_strength(DATA_PIN, GPIO_DRIVE_STRENGTH_12MA);
    gpio_set_slew_rate(DATA_PIN, GPIO_SLEW_RATE_FAST);

    // Initialize low-level joybus port (bypass N64Console class)
    joybus_port_t port;
    joybus_port_init(&port, DATA_PIN, pio0, -1, -1);

    // LED indicator
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 1);

    printf("Listening for N64 commands (POLL/READ/WRITE)...\n");

    bool led = true;
    uint32_t poll_count = 0;
    uint32_t write_count = 0;
    uint32_t read_count = 0;

    while (true) {
        uint8_t cmd[1];

        // Wait for command byte (1ms timeout, first byte CAN timeout)
        // Short timeout lets us check USB serial for CHAIN: commands between N64 polls.
        // Safe: PIO captures joybus bits into FIFO regardless of CPU state.
        if (joybus_receive_bytes(&port, cmd, 1, 1000, true) != 1) {
            check_serial_input();
            continue;
        }

        // NOTE: No debug printf here! Any delay between receiving the
        // command byte and reading payload bytes causes PIO FIFO overflow
        // (N64 sends remaining bytes immediately after cmd byte).

        switch ((N64Command)cmd[0]) {

        case N64Command::PROBE:
        case N64Command::RESET:
            busy_wait_us(REPLY_DELAY_US);
            joybus_send_bytes(&port, (uint8_t *)&status, sizeof(n64_status_t));
            break;

        case N64Command::POLL:
            busy_wait_us(REPLY_DELAY_US);
            joybus_send_bytes(&port, (uint8_t *)&report, sizeof(n64_report_t));

            led = !led;
            gpio_put(PICO_DEFAULT_LED_PIN, led);
            poll_count++;

            if (poll_count <= 5 || (poll_count % 1000) == 0) {
                printf("POLL #%lu\n", poll_count);
            }
            break;

        case N64Command::WRITE_EXPANSION_BUS: {
            // N64 sends: [0x03] [addr_hi] [addr_lo] [32 bytes data] = 35 total
            // Pico responds: [1 byte CRC ack]
            // Then we relay data over USB serial for host bridge.
            uint8_t payload[34];  // 2 addr + 32 data
            uint received = joybus_receive_bytes(&port, payload, 34, BYTE_TIMEOUT_US, false);

            if (received >= 34) {
                uint16_t addr = ((uint16_t)payload[0] << 8) | payload[1];
                addr &= 0xFFE0;  // Mask off CRC bits from address word

                // Respond with data CRC immediately (timing-critical)
                uint8_t crc = pak_data_crc(&payload[2], 32);
                busy_wait_us(REPLY_DELAY_US);
                joybus_send_bytes(&port, &crc, 1);

                // Now relay over USB serial (after response sent)
                // Format: PAK_W:ADDR:HEXDATA (matches n64_bridge.py parser)
                char line[80];  // "PAK_W:" + 4 hex + ":" + 64 hex + "\n" = 76 chars
                int pos = 0;
                line[pos++] = 'P'; line[pos++] = 'A'; line[pos++] = 'K';
                line[pos++] = '_'; line[pos++] = 'W'; line[pos++] = ':';
                line[pos++] = HEX[(addr >> 12) & 0xF];
                line[pos++] = HEX[(addr >> 8) & 0xF];
                line[pos++] = HEX[(addr >> 4) & 0xF];
                line[pos++] = HEX[addr & 0xF];
                line[pos++] = ':';
                for (int i = 0; i < 32; i++) {
                    line[pos++] = HEX[(payload[2+i] >> 4) & 0xF];
                    line[pos++] = HEX[payload[2+i] & 0xF];
                }
                line[pos++] = '\n';
                line[pos] = '\0';

                // Non-blocking write to USB CDC
                fputs(line, stdout);
                fflush(stdout);

                write_count++;
            } else {
                // Incomplete WRITE — reset PIO
                busy_wait_us(RESET_WAIT_US);
                joybus_port_reset(&port);
            }
            break;
        }

        case N64Command::READ_EXPANSION_BUS: {
            // N64 sends: [0x02] [addr_hi] [addr_lo] = 3 total
            // Pico responds: [32 bytes data] [1 byte CRC]
            uint8_t addr_buf[2];
            uint received = joybus_receive_bytes(&port, addr_buf, 2, BYTE_TIMEOUT_US, false);

            if (received >= 2) {
                uint16_t addr = ((uint16_t)addr_buf[0] << 8) | addr_buf[1];
                addr &= 0xFFE0;

                uint8_t response[33];  // 32 data + 1 CRC

                if (addr == 0x8000 && chain_state_valid) {
                    // Serve chain state data from host bridge
                    memcpy(response, chain_state, 32);
                } else {
                    // All other addresses: return zeros
                    memset(response, 0x00, 32);
                }
                response[32] = pak_data_crc(response, 32);

                busy_wait_us(REPLY_DELAY_US);
                joybus_send_bytes(&port, response, 33);

                read_count++;
                if (read_count <= 5 || (read_count % 100) == 0) {
                    printf("PAK_R:%04X%s\n", addr,
                           (addr == 0x8000 && chain_state_valid) ? ":CHAIN" : "");
                }
            } else {
                busy_wait_us(RESET_WAIT_US);
                joybus_port_reset(&port);
            }
            break;
        }

        default:
            // Unknown command — wait for wire to settle, reset PIO
            busy_wait_us(RESET_WAIT_US);
            joybus_port_reset(&port);
            break;
        }
    }
}
