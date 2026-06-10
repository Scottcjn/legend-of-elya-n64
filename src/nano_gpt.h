// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <libdragon.h>

// Sophia Elya AI - World's First N64 LLM
// nano-GPT: 4 layers, 128 embedding, 4 heads, vocab=256, ctx=64 (v5 weights)
// Float32 inference (software FPU via -msoft-float on MIPS R4300i)

#define SGAI_MAGIC      0x53454149  // "SEAI"
#define SGAI_N_LAYERS   4
#define SGAI_N_EMBED    128
#define SGAI_N_HEADS    4
#define SGAI_HEAD_DIM   (SGAI_N_EMBED / SGAI_N_HEADS)  // 32
#define SGAI_VOCAB      256
#define SGAI_CTX        64
#define SGAI_Q_BLOCK    32  // weight quantization block size

// Weight layout for one attention layer (Q8: int8 weights, float16 scales)
// Weights stay compressed in ROM; dequantized to float32 on-the-fly during matmul
typedef struct {
    int8_t wq[SGAI_N_EMBED * SGAI_N_EMBED];
    int8_t wk[SGAI_N_EMBED * SGAI_N_EMBED];
    int8_t wv[SGAI_N_EMBED * SGAI_N_EMBED];
    int8_t wo[SGAI_N_EMBED * SGAI_N_EMBED];
    int8_t wff1[SGAI_N_EMBED * SGAI_N_EMBED * 4];
    int8_t wff2[SGAI_N_EMBED * SGAI_N_EMBED * 4];
    uint16_t sq[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sk[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sv[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t so[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q_BLOCK];
    uint16_t sff1[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q_BLOCK];
    uint16_t sff2[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q_BLOCK];
} __attribute__((aligned(8))) SGAILayer;

typedef struct {
    uint32_t magic;
    uint8_t n_layers;
    uint16_t n_embed;
    uint8_t n_heads;
    uint16_t vocab_size;
    uint8_t ctx_len;
    uint8_t em_scale_x16; // Embedding scale * 16 (e.g., em=3.5 -> 56)
} __attribute__((packed)) SGAIHeader;

// KV cache (float32, ~256KB in RDRAM)
typedef struct {
    float k[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    float v[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    int pos;
} __attribute__((aligned(8))) SGAIKVCache;

// Main inference state
typedef struct {
    const SGAIHeader *weights;
    SGAIKVCache *kv;
    float x[SGAI_N_EMBED];         // Current activation vector
    float logits[SGAI_VOCAB];      // Output logits
    uint32_t tokens[SGAI_CTX];
    int seq_len;
    int is_loaded;
    float em_scale;                 // Restored embedding scale factor
    uint8_t penalty_hist[16];
    uint8_t penalty_n;
} SGAIState;

// API
void sgai_init(SGAIState *state, const void *rom_weights);
void sgai_reset(SGAIState *state);
uint8_t sgai_next_token(SGAIState *state, uint8_t input_token, uint32_t temperature_q8);
void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8);
