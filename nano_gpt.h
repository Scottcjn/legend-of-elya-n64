// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <libdragon.h>

// Sophia Elya AI - World's First N64 LLM
// nano-GPT v9 LARGE: 8 layers, 256 embedding, 8 heads, vocab=256, ctx=128
// 8.4M parameters — requires 8MB Expansion Pak
// RSP-accelerated matrix multiply via DMA tiling

#define SGAI_MAGIC      0x53454149  // "SEAI"
#define SGAI_N_LAYERS   8
#define SGAI_N_EMBED    256
#define SGAI_N_HEADS    8
#define SGAI_HEAD_DIM   (SGAI_N_EMBED / SGAI_N_HEADS)  // 32
#define SGAI_VOCAB      256
#ifndef SGAI_CTX
#define SGAI_CTX        128
#endif
#define SGAI_Q_BLOCK    32  // weight quantization block size

// Weight layout for one attention layer (Q8: int8 weights, float16 scales)
typedef struct {
    // Q8 packed weights (1 weight per byte, signed int8) + float16 scales (per 32-block)
    int8_t wq[SGAI_N_EMBED * SGAI_N_EMBED];        // Query projection
    int8_t wk[SGAI_N_EMBED * SGAI_N_EMBED];        // Key
    int8_t wv[SGAI_N_EMBED * SGAI_N_EMBED];        // Value
    int8_t wo[SGAI_N_EMBED * SGAI_N_EMBED];        // Output
    int8_t wff1[SGAI_N_EMBED * SGAI_N_EMBED * 4]; // FFN expand (192->768)
    int8_t wff2[SGAI_N_EMBED * SGAI_N_EMBED * 4]; // FFN contract (768->192)
    // Q8 scale factors (float16, one per 32-weight block)
    uint16_t sq[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sk[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sv[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t so[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sff1[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q_BLOCK];
    uint16_t sff2[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q_BLOCK];
} __attribute__((aligned(8))) SGAILayer;

typedef struct {
    uint32_t magic;   // SGAI_MAGIC
    uint8_t n_layers;
    uint16_t n_embed;
    uint8_t n_heads;
    uint16_t vocab_size;
    uint8_t ctx_len;
    uint8_t em_scale_x16;  // embedding scale * 16 (e.g., 56 = 3.5)
    // After header: embedding table (vocab * embed bytes, Q8 int8)
    // Then n_layers SGAILayer structs
} __attribute__((packed)) SGAIHeader;

// KV cache for inference (in RDRAM — ~1.15MB with 8MB Expansion Pak)
typedef struct {
    float k[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    float v[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    int pos;
} __attribute__((aligned(8))) SGAIKVCache;

// Main inference state
typedef struct {
    const SGAIHeader *weights;  // Points into ROM
    SGAIKVCache *kv;            // In RDRAM (heap-allocated)
    float x[SGAI_N_EMBED];     // Current token embedding
    float logits[SGAI_VOCAB];  // Output logits
    float em_scale;             // Embedding scale factor
    uint32_t tokens[SGAI_CTX]; // Generated token sequence
    int seq_len;
    int is_loaded;
    /* Hard-exclusion window: last 16 output tokens. */
    uint8_t penalty_hist[16];
    uint8_t penalty_n;
} SGAIState;

// API
void sgai_init(SGAIState *state, const void *rom_weights);
void sgai_reset(SGAIState *state);
uint8_t sgai_next_token(SGAIState *state, uint8_t input_token, uint32_t temperature_q8);
void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8);

// RSP helpers (internal)
void sgai_rsp_matmul_q8(const int8_t *weights, const uint16_t *scales,
                         const int16_t *input, int16_t *output,
                         int in_dim, int out_dim);
void sgai_softmax_inplace(int16_t *vec, int len);
int16_t sgai_relu(int16_t x);
