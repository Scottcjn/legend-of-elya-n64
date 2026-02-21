#pragma once
#include <stdint.h>
#include <libdragon.h>

// Sophia Elya AI - World's First N64 LLM
// nano-GPT: 2 layers, 128 embedding, 4 heads, vocab=256, ctx=32
// RSP-accelerated matrix multiply via DMA tiling

#define SGAI_MAGIC      0x53454149  // "SEAI"
#define SGAI_N_LAYERS   2
#define SGAI_N_EMBED    128
#define SGAI_N_HEADS    4
#define SGAI_HEAD_DIM   (SGAI_N_EMBED / SGAI_N_HEADS)  // 32
#define SGAI_VOCAB      256
#define SGAI_CTX        32
#define SGAI_Q4_BLOCK   32  // weight quantization block size

// Weight layout for one attention layer
typedef struct {
    // Q4 packed weights (2 weights per byte) + float16 scales (per 32-block)
    uint8_t wq[SGAI_N_EMBED * SGAI_N_EMBED / 2];   // Query projection
    uint8_t wk[SGAI_N_EMBED * SGAI_N_EMBED / 2];   // Key
    uint8_t wv[SGAI_N_EMBED * SGAI_N_EMBED / 2];   // Value
    uint8_t wo[SGAI_N_EMBED * SGAI_N_EMBED / 2];   // Output
    uint8_t wff1[SGAI_N_EMBED * SGAI_N_EMBED * 4 / 2];  // FFN expand (128->512)
    uint8_t wff2[SGAI_N_EMBED * SGAI_N_EMBED * 4 / 2];  // FFN contract (512->128)
    // Q4 scale factors (float16, one per 32-weight block)
    uint16_t sq[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q4_BLOCK];
    uint16_t sk[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q4_BLOCK];
    uint16_t sv[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q4_BLOCK];
    uint16_t so[SGAI_N_EMBED * SGAI_N_EMBED / SGAI_Q4_BLOCK];
    uint16_t sff1[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q4_BLOCK];
    uint16_t sff2[(SGAI_N_EMBED * SGAI_N_EMBED * 4) / SGAI_Q4_BLOCK];
} __attribute__((aligned(8))) SGAILayer;

typedef struct {
    uint32_t magic;   // SGAI_MAGIC
    uint8_t n_layers;
    uint16_t n_embed;
    uint8_t n_heads;
    uint16_t vocab_size;
    uint8_t ctx_len;
    uint8_t pad[1];
    // After header: embedding table (vocab * embed, Q4 packed)
    // Then n_layers SGAILayer structs
} __attribute__((packed)) SGAIHeader;

// KV cache for inference (in RDRAM)
typedef struct {
    int16_t k[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];  // Fixed-point Q8
    int16_t v[SGAI_N_LAYERS][SGAI_CTX][SGAI_N_EMBED];
    int pos;
} __attribute__((aligned(8))) SGAIKVCache;

// Main inference state
typedef struct {
    const SGAIHeader *weights;  // Points into ROM
    SGAIKVCache *kv;            // In RDRAM
    int16_t x[SGAI_N_EMBED];   // Current token embedding (Q8.7 fixed-point)
    int16_t logits[SGAI_VOCAB]; // Output logits
    uint32_t tokens[SGAI_CTX]; // Generated token sequence
    int seq_len;
    int is_loaded;
} SGAIState;

// API
void sgai_init(SGAIState *state, const void *rom_weights);
void sgai_reset(SGAIState *state);
uint8_t sgai_next_token(SGAIState *state, uint8_t input_token, uint32_t temperature_q8);
void sgai_generate(SGAIState *state, const uint8_t *prompt, int prompt_len,
                   uint8_t *output, int max_tokens, uint32_t temperature_q8);

// RSP helpers (internal)
void sgai_rsp_matmul_q4(const uint8_t *weights, const uint16_t *scales,
                         const int16_t *input, int16_t *output,
                         int in_dim, int out_dim);
void sgai_softmax_inplace(int16_t *vec, int len);
int16_t sgai_relu(int16_t x);
