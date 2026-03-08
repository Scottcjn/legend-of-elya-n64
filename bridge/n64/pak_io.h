#ifndef PAK_IO_H
#define PAK_IO_H

/*
 * Minimal joybus pak I/O for N64 ↔ Pico bridge
 *
 * Provides raw 32-byte READ/WRITE to pak addresses via PIF/joybus.
 * Used by LLM RPC (0x9000/0xA000) and attestation (0x0000/0x8000).
 */

#include <libdragon.h>

#define BRIDGE_PORT  1  /* Controller port 2 = index 1 */

/* Write 32 bytes to pak address. Returns 0 on success. */
int pak_write(int port, uint16_t addr, const uint8_t *data);

/* Read 32 bytes from pak address. Returns 0 on success. */
int pak_read(int port, uint16_t addr, uint8_t *data);

/* Detect if Pico bridge is connected on given port.
 * Returns 1 if bridge detected, 0 otherwise. */
int pak_detect_bridge(int port);

#endif
