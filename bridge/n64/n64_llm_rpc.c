/*
 * N64 LLM RPC — Remote inference over Pico bridge
 *
 * Uses the pak READ/WRITE mechanism to send prompts
 * and receive responses from a host running real inference.
 *
 * 60-150x faster than on-cartridge nano_gpt (200ms vs 30s).
 */

#include "n64_llm_rpc.h"
#include "pak_io.h"
#include <string.h>

LLMRpcState g_llm_rpc;

/* ─── LLM RPC Implementation ─────────────────────────────────────────── */

int llm_rpc_detect(void)
{
    /* Detect bridge on controller port 2 */
    g_llm_rpc.rpc_available = pak_detect_bridge(BRIDGE_PORT);
    g_llm_rpc.sequence_id = 0;
    g_llm_rpc.request_pending = 0;
    g_llm_rpc.poll_interval = 6;  /* 100ms at 60fps */
    return g_llm_rpc.rpc_available;
}

int llm_rpc_request(const char *prompt, int max_tokens, uint8_t temperature_q8)
{
    if (!g_llm_rpc.rpc_available) return 0;

    int prompt_len = (int)strlen(prompt);
    if (prompt_len > LLM_MAX_PROMPT)
        prompt_len = LLM_MAX_PROMPT;
    if (max_tokens > LLM_MAX_RESPONSE)
        max_tokens = LLM_MAX_RESPONSE;

    g_llm_rpc.sequence_id++;

    /* ── Page 0: Request header at 0x9000 ── */
    uint8_t page[32];
    memset(page, 0, 32);

    page[0] = 'L';
    page[1] = 'L';
    page[2] = 'M';
    page[3] = 0x01;
    page[4] = (uint8_t)prompt_len;
    page[5] = (uint8_t)max_tokens;
    page[6] = temperature_q8;
    page[7] = g_llm_rpc.sequence_id;

    int first_chunk = (prompt_len < 24) ? prompt_len : 24;
    memcpy(&page[8], prompt, first_chunk);

    pak_write(BRIDGE_PORT, LLM_REQ_ADDR, page);

    /* ── Continuation pages ── */
    int remaining = prompt_len - first_chunk;
    int src_offset = first_chunk;
    uint16_t addr = LLM_REQ_DATA;

    while (remaining > 0) {
        memset(page, 0, 32);
        int chunk = (remaining < 32) ? remaining : 32;
        memcpy(page, prompt + src_offset, chunk);
        pak_write(BRIDGE_PORT, addr, page);

        src_offset += chunk;
        remaining -= chunk;
        addr += 0x0020;
    }

    g_llm_rpc.request_pending = 1;
    g_llm_rpc.response_status = LLM_STATUS_PENDING;
    g_llm_rpc.response_len = 0;
    memset(g_llm_rpc.response_buf, 0, sizeof(g_llm_rpc.response_buf));

    return 1;
}

int llm_rpc_poll(void)
{
    if (!g_llm_rpc.request_pending) return LLM_STATUS_ERROR;

    uint8_t page[32];
    memset(page, 0, 32);

    if (pak_read(BRIDGE_PORT, LLM_RESP_ADDR, page) != 0) {
        return LLM_STATUS_PENDING;
    }

    if (page[0] != 'L' || page[1] != 'L' || page[2] != 'R' || page[3] != 0x01) {
        return LLM_STATUS_PENDING;
    }

    uint8_t status = page[4];
    uint8_t resp_len = page[5];
    uint8_t seq_id = page[6];

    if (seq_id != g_llm_rpc.sequence_id) {
        return LLM_STATUS_PENDING;
    }

    g_llm_rpc.response_status = status;

    if (status == LLM_STATUS_READY || status == LLM_STATUS_STREAMING) {
        int total = (resp_len < LLM_MAX_RESPONSE) ? resp_len : LLM_MAX_RESPONSE;
        int first_chunk = (total < 24) ? total : 24;
        memcpy(g_llm_rpc.response_buf, &page[8], first_chunk);

        int remaining_r = total - first_chunk;
        int dst_offset = first_chunk;
        uint16_t raddr = LLM_RESP_DATA;

        while (remaining_r > 0) {
            memset(page, 0, 32);
            if (pak_read(BRIDGE_PORT, raddr, page) != 0) break;

            int chunk = (remaining_r < 32) ? remaining_r : 32;
            memcpy(g_llm_rpc.response_buf + dst_offset, page, chunk);

            dst_offset += chunk;
            remaining_r -= chunk;
            raddr += 0x0020;
        }

        g_llm_rpc.response_buf[dst_offset] = '\0';
        g_llm_rpc.response_len = dst_offset;

        if (status == LLM_STATUS_READY) {
            g_llm_rpc.request_pending = 0;
        }
    }

    return status;
}
