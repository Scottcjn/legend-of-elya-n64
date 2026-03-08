#ifndef N64_LLM_RPC_H
#define N64_LLM_RPC_H

/*
 * N64 LLM RPC — Remote inference over Pico bridge
 *
 * Instead of running nano_gpt on 93MHz soft-float MIPS (~2 tok/s),
 * send the prompt through the existing Pico USB bridge to a host
 * running inference on real hardware (POWER8, GPU, or SophiaCore).
 *
 * Protocol:
 *   N64 → Pico: WRITE to 0x9000+ (prompt pages, 32 bytes each)
 *   Pico → N64: READ from 0xA000+ (response pages, 32 bytes each)
 *
 * Page format:
 *   0x9000 (request header):
 *     [0-3] magic "LLM\x01"
 *     [4]   prompt_len (total bytes)
 *     [5]   max_tokens (max response length)
 *     [6]   temperature_q8 (temp * 256)
 *     [7]   sequence_id (wraps 0-255, for matching req/resp)
 *     [8-31] first 24 bytes of prompt
 *
 *   0x9020+ (prompt continuation, 32 bytes each):
 *     [0-31] prompt bytes (ASCII)
 *
 *   0xA000 (response header):
 *     [0-3] magic "LLR\x01" (LLM Response)
 *     [4]   status: 0=pending, 1=ready, 2=error, 3=streaming
 *     [5]   response_len (total bytes available)
 *     [6]   sequence_id (matches request)
 *     [7]   tokens_generated
 *     [8-31] first 24 bytes of response
 *
 *   0xA020+ (response continuation, 32 bytes each):
 *     [0-31] response bytes (ASCII)
 *
 * The N64 writes the prompt, then polls 0xA000 until status != 0.
 * Host bridge receives LLM_REQ via serial, runs inference, sends back
 * LLM_RESP via serial. Pico stores response for N64 to READ.
 *
 * Total latency: ~200-500ms (network) vs ~30s (on-cartridge)
 * That's 60-150x speedup for a typical 80-char response.
 */

#include <libdragon.h>

/* Pak address ranges */
#define LLM_REQ_ADDR    0x9000  /* Request header */
#define LLM_REQ_DATA    0x9020  /* Request prompt continuation pages */
#define LLM_RESP_ADDR   0xA000  /* Response header */
#define LLM_RESP_DATA   0xA020  /* Response data continuation pages */

/* Magic bytes */
#define LLM_REQ_MAGIC   "LLM\x01"
#define LLM_RESP_MAGIC  "LLR\x01"

/* Response status */
#define LLM_STATUS_PENDING   0
#define LLM_STATUS_READY     1
#define LLM_STATUS_ERROR     2
#define LLM_STATUS_STREAMING 3

/* Max prompt/response size (limited by pak address space) */
#define LLM_MAX_PROMPT   128  /* 24 + 4*32 = 152 bytes max, we use 128 */
#define LLM_MAX_RESPONSE 128  /* 24 + 4*32 = 152 bytes max, we use 128 */

/* RPC state */
typedef struct {
    int rpc_available;           /* 1 if bridge supports LLM RPC */
    int request_pending;         /* 1 if waiting for response */
    uint8_t sequence_id;         /* Current sequence number */
    uint8_t response_status;     /* Last polled status */
    uint8_t response_buf[LLM_MAX_RESPONSE + 1]; /* Null-terminated response */
    int response_len;
    int poll_frame;              /* Frame of last poll */
    int poll_interval;           /* Frames between polls (start at 6 = 100ms) */
} LLMRpcState;

extern LLMRpcState g_llm_rpc;

/*
 * Send an LLM inference request through the bridge.
 * Returns 1 if request was sent, 0 if bridge not available.
 * prompt: ASCII prompt string (null-terminated)
 * max_tokens: max response length (1-128)
 * temperature_q8: temperature * 256 (64 = 0.25, 128 = 0.5)
 */
int llm_rpc_request(const char *prompt, int max_tokens, uint8_t temperature_q8);

/*
 * Poll for LLM response. Call once per frame during STATE_GENERATING.
 * Returns:
 *   LLM_STATUS_PENDING   - still waiting
 *   LLM_STATUS_READY     - response in g_llm_rpc.response_buf
 *   LLM_STATUS_STREAMING - partial response available
 *   LLM_STATUS_ERROR     - inference failed
 */
int llm_rpc_poll(void);

/*
 * Check if the bridge supports LLM RPC.
 * Call once after bridge detection succeeds.
 * Sends a probe to 0x9000 and checks if Pico acknowledges.
 */
int llm_rpc_detect(void);

#endif
