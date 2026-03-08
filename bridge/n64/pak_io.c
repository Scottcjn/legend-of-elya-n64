/*
 * Minimal joybus pak I/O for N64 ↔ Pico bridge
 *
 * Raw 32-byte READ/WRITE to Controller Pak addresses via PIF.
 * The Pico intercepts these at specific address ranges to act as bridge.
 */

#include "pak_io.h"
#include <string.h>

/* ─── Pak address CRC (5-bit) ─────────────────────────────────────────── */

static uint8_t pak_addr_crc(uint16_t addr)
{
    uint8_t crc = 0;
    addr &= 0xFFE0;
    for (int i = 15; i >= 5; i--) {
        int bit = (addr >> i) & 1;
        int fb = ((crc >> 4) ^ bit) & 1;
        crc <<= 1;
        if (fb) crc ^= 0x15;
        crc &= 0x1F;
    }
    return crc;
}

/* ─── PIF channel skip (position to correct controller port) ──────────── */

static int pif_skip_channels(uint8_t *block, int port)
{
    int pos = 0;
    for (int i = 0; i < port; i++) {
        /* Dummy PROBE: TX=1, RX=3, cmd=0x00 (status/probe) */
        block[pos++] = 0x01; /* TX */
        block[pos++] = 0x03; /* RX */
        block[pos++] = 0x00; /* PROBE */
        block[pos++] = 0xFF; /* RX placeholder */
        block[pos++] = 0xFF;
        block[pos++] = 0xFF;
    }
    return pos;
}

/* ─── Raw pak I/O ─────────────────────────────────────────────────────── */

int pak_read(int port, uint16_t addr, uint8_t *data)
{
    uint8_t block[64] __attribute__((aligned(16)));
    uint8_t output[64] __attribute__((aligned(16)));

    memset(block, 0, 64);
    block[63] = 0x01;  /* PIF processing flag */

    int pos = pif_skip_channels(block, port);
    int cmd_start = pos;

    uint16_t addr_crc = (addr & 0xFFE0) | pak_addr_crc(addr);

    /* READ: TX=3 (cmd + 2 addr), RX=33 (32 data + 1 CRC) */
    block[pos++] = 0x03;
    block[pos++] = 0x21;
    block[pos++] = 0x02;  /* READ command */
    block[pos++] = (addr_crc >> 8) & 0xFF;
    block[pos++] = addr_crc & 0xFF;
    memset(&block[pos], 0xFF, 33);
    pos += 33;
    block[pos++] = 0xFE;

    joybus_exec(block, output);
    memcpy(data, &output[cmd_start + 5], 32);
    return 0;
}

int pak_write(int port, uint16_t addr, const uint8_t *data)
{
    uint8_t block[64] __attribute__((aligned(16)));
    uint8_t output[64] __attribute__((aligned(16)));

    memset(block, 0, 64);
    block[63] = 0x01;

    int pos = pif_skip_channels(block, port);

    uint16_t addr_crc = (addr & 0xFFE0) | pak_addr_crc(addr);

    /* WRITE: TX=35 (cmd + 2 addr + 32 data), RX=1 (CRC ack) */
    block[pos++] = 0x23;
    block[pos++] = 0x01;
    block[pos++] = 0x03;  /* WRITE command */
    block[pos++] = (addr_crc >> 8) & 0xFF;
    block[pos++] = addr_crc & 0xFF;
    memcpy(&block[pos], data, 32);
    pos += 32;
    block[pos++] = 0xFF;
    block[pos++] = 0xFE;

    joybus_exec(block, output);
    return 0;
}

int pak_detect_bridge(int port)
{
    /* Write a known pattern to a test address, read it back.
     * Real Controller Pak mirrors writes; Pico bridge may respond differently.
     * We probe by writing to 0xF000 (unused range) and checking if
     * we get a valid response vs silence. */
    uint8_t probe[32], readback[32];
    memset(probe, 0, 32);
    probe[0] = 'P'; probe[1] = 'I'; probe[2] = 'C'; probe[3] = 'O';

    pak_write(port, 0xF000, probe);

    /* Small delay for Pico to process */
    for (volatile int i = 0; i < 10000; i++) {}

    memset(readback, 0xFF, 32);
    pak_read(port, 0xF000, readback);

    /* If we got our pattern back or any valid response, bridge is present */
    return (readback[0] == 'P' && readback[1] == 'I');
}
