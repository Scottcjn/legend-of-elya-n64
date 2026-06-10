// SPDX-License-Identifier: MIT
/*
 * reference_cli.c - Host-side reference runner for Legend of Elya N64.
 *
 * This file mirrors the current top-level nano_gpt.c float/Q8 inference path:
 *   - Q8 int8 weights with float16 scales
 *   - tied embedding projection
 *   - RMS norm + approximate softmax
 *   - Physarum conductance routing and entropy burst injection
 *   - greedy printable-ASCII sampling
 *
 * It intentionally has no external dependencies beyond the C standard library.
 * The host entropy source uses clock() as a stand-in for the N64 CP0 COUNT
 * register, which keeps the control flow aligned with the ROM implementation.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_WEIGHTS "filesystem/sophia_weights.bin"
#define DEFAULT_MAX_TOKENS 96
#define PRINTABLE_START 32
#define PRINTABLE_END 126
#define Q_BLOCK 32

#define PSE_PHYSARUM_REINFORCE 0.1f
#define PSE_PHYSARUM_DECAY 0.02f
#define PSE_PHYSARUM_MIN 0.5f
#define PSE_PHYSARUM_MAX 1.5f
#define PSE_PHYSARUM_PRUNE 0.0f
#define PSE_BURST_INTERVAL 8
#define PSE_BURST_STRENGTH 0.02f
#define PSE_BURST_DIMS 8

typedef struct {
    const int8_t *wq;
    const int8_t *wk;
    const int8_t *wv;
    const int8_t *wo;
    const int8_t *wff1;
    const int8_t *wff2;
    const unsigned char *sq;
    const unsigned char *sk;
    const unsigned char *sv;
    const unsigned char *so;
    const unsigned char *sff1;
    const unsigned char *sff2;
} LayerView;

typedef struct {
    unsigned char *blob;
    size_t blob_size;
    unsigned int n_layers;
    unsigned int n_embed;
    unsigned int n_heads;
    unsigned int head_dim;
    unsigned int vocab;
    unsigned int ctx;
    float em_scale;
    const int8_t *emb_table;
    LayerView *layers;
} Model;

typedef struct {
    float *k;
    float *v;
    int pos;
} KVCache;

typedef struct {
    Model *model;
    KVCache kv;
    float *x;
    float *logits;
    unsigned int *tokens;
    int seq_len;

    float *q;
    float *k_cur;
    float *v_cur;
    float *attn_out;
    float *ff_buf;
    float *attn_scores;
    float *residual;
    float *proj_out;
    float *ff_out;
    float *head_sharpness;

    float *conductance;
    float entropy_ema;
    unsigned int token_counter;
} Runner;


static uint16_t read_u16_le(const unsigned char *ptr)
{
    return (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
}


static uint32_t read_u32_le(const unsigned char *ptr)
{
    return (uint32_t)ptr[0]
         | ((uint32_t)ptr[1] << 8)
         | ((uint32_t)ptr[2] << 16)
         | ((uint32_t)ptr[3] << 24);
}


static size_t kv_offset(const Runner *runner, int layer, int pos, int index)
{
    const Model *model = runner->model;
    return (((size_t)layer * model->ctx + (size_t)pos) * model->n_embed) + (size_t)index;
}


static size_t conductance_offset(const Runner *runner, int layer, int head)
{
    return (size_t)layer * runner->model->n_heads + (size_t)head;
}


static float *kv_k_slot(Runner *runner, int layer, int pos)
{
    return runner->kv.k + kv_offset(runner, layer, pos, 0);
}


static float *kv_v_slot(Runner *runner, int layer, int pos)
{
    return runner->kv.v + kv_offset(runner, layer, pos, 0);
}


static unsigned char *load_file(const char *path, size_t *size_out)
{
    FILE *handle = fopen(path, "rb");
    unsigned char *buf;
    long length;

    if (!handle) {
        fprintf(stderr, "Failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    if (fseek(handle, 0, SEEK_END) != 0) {
        fprintf(stderr, "Failed to seek %s\n", path);
        fclose(handle);
        return NULL;
    }

    length = ftell(handle);
    if (length < 0) {
        fprintf(stderr, "Failed to tell %s\n", path);
        fclose(handle);
        return NULL;
    }

    if (fseek(handle, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to rewind %s\n", path);
        fclose(handle);
        return NULL;
    }

    buf = (unsigned char *)malloc((size_t)length);
    if (!buf) {
        fprintf(stderr, "Out of memory reading %s\n", path);
        fclose(handle);
        return NULL;
    }

    if (fread(buf, 1, (size_t)length, handle) != (size_t)length) {
        fprintf(stderr, "Failed to read %s\n", path);
        free(buf);
        fclose(handle);
        return NULL;
    }

    fclose(handle);
    *size_out = (size_t)length;
    return buf;
}


static int load_model(Model *model, const char *path)
{
    size_t offset;
    size_t emb_bytes;
    size_t attn_bytes;
    size_t ffn_bytes;
    size_t attn_scales;
    size_t ffn_scales;
    unsigned int i;

    memset(model, 0, sizeof(*model));
    model->blob = load_file(path, &model->blob_size);
    if (!model->blob) {
        return -1;
    }

    if (model->blob_size < 12) {
        fprintf(stderr, "Weight file too small: %s\n", path);
        return -1;
    }

    if (read_u32_le(model->blob) != 0x53454149u && read_u32_le(model->blob) != 0x49414553u) {
        fprintf(stderr, "Unexpected weight magic in %s\n", path);
        return -1;
    }

    model->n_layers = model->blob[4];
    model->n_embed = read_u16_le(model->blob + 5);
    model->n_heads = model->blob[7];
    model->vocab = read_u16_le(model->blob + 8);
    model->ctx = model->blob[10];
    model->em_scale = model->blob[11] ? ((float)model->blob[11] / 16.0f) : 3.5f;

    if (!model->n_layers || !model->n_embed || !model->n_heads || !model->vocab || !model->ctx) {
        fprintf(stderr, "Invalid model header values in %s\n", path);
        return -1;
    }
    if (model->n_embed % model->n_heads != 0) {
        fprintf(stderr, "n_embed must divide cleanly by n_heads\n");
        return -1;
    }
    if (model->n_embed % Q_BLOCK != 0) {
        fprintf(stderr, "n_embed must be divisible by the Q8 block size\n");
        return -1;
    }

    model->head_dim = model->n_embed / model->n_heads;
    model->layers = (LayerView *)calloc(model->n_layers, sizeof(LayerView));
    if (!model->layers) {
        fprintf(stderr, "Out of memory allocating layer views\n");
        return -1;
    }

    offset = 12;
    emb_bytes = (size_t)model->vocab * model->n_embed;
    if (offset + emb_bytes > model->blob_size) {
        fprintf(stderr, "Weight file truncated before embedding table\n");
        return -1;
    }
    model->emb_table = (const int8_t *)(model->blob + offset);
    offset += emb_bytes;

    attn_bytes = (size_t)model->n_embed * model->n_embed;
    ffn_bytes = (size_t)model->n_embed * model->n_embed * 4u;
    attn_scales = (attn_bytes / Q_BLOCK) * 2u;
    ffn_scales = (ffn_bytes / Q_BLOCK) * 2u;

    for (i = 0; i < model->n_layers; ++i) {
        LayerView *layer = &model->layers[i];
        if (offset + attn_bytes * 4u + ffn_bytes * 2u + attn_scales * 4u + ffn_scales * 2u > model->blob_size) {
            fprintf(stderr, "Weight file truncated inside layer %u\n", i);
            return -1;
        }

        layer->wq = (const int8_t *)(model->blob + offset); offset += attn_bytes;
        layer->wk = (const int8_t *)(model->blob + offset); offset += attn_bytes;
        layer->wv = (const int8_t *)(model->blob + offset); offset += attn_bytes;
        layer->wo = (const int8_t *)(model->blob + offset); offset += attn_bytes;
        layer->wff1 = (const int8_t *)(model->blob + offset); offset += ffn_bytes;
        layer->wff2 = (const int8_t *)(model->blob + offset); offset += ffn_bytes;

        layer->sq = model->blob + offset; offset += attn_scales;
        layer->sk = model->blob + offset; offset += attn_scales;
        layer->sv = model->blob + offset; offset += attn_scales;
        layer->so = model->blob + offset; offset += attn_scales;
        layer->sff1 = model->blob + offset; offset += ffn_scales;
        layer->sff2 = model->blob + offset; offset += ffn_scales;
    }

    if (offset != model->blob_size) {
        fprintf(stderr, "Note: %zu trailing bytes after parsed weights\n", model->blob_size - offset);
    }

    return 0;
}


static void free_model(Model *model)
{
    free(model->layers);
    free(model->blob);
    memset(model, 0, sizeof(*model));
}


static size_t runtime_bytes(const Runner *runner)
{
    const Model *model = runner->model;
    size_t kv_bytes = (size_t)model->n_layers * model->ctx * model->n_embed * sizeof(float) * 2u;
    size_t core = (size_t)model->n_embed * sizeof(float) * 7u;
    size_t ffn = (size_t)model->n_embed * 4u * sizeof(float);
    size_t logits = (size_t)model->vocab * sizeof(float);
    size_t scores = (size_t)model->ctx * sizeof(float);
    size_t tokens = (size_t)model->ctx * sizeof(unsigned int);
    size_t heads = (size_t)model->n_heads * sizeof(float);
    size_t conductance = (size_t)model->n_layers * model->n_heads * sizeof(float);
    size_t meta = sizeof(*runner) + sizeof(*model) + (size_t)model->n_layers * sizeof(LayerView);
    return kv_bytes + core + ffn + logits + scores + tokens + heads + conductance + meta;
}


static float f16_to_float_le(const unsigned char *src)
{
    uint16_t f16 = read_u16_le(src);
    uint32_t sign = (uint32_t)(f16 >> 15) & 1u;
    uint32_t exp = (uint32_t)(f16 >> 10) & 0x1Fu;
    uint32_t frac = (uint32_t)f16 & 0x03FFu;
    float value;

    if (exp == 0) {
        value = ((float)frac / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        value = 65504.0f;
    } else {
        float mantissa = 1.0f + ((float)frac / 1024.0f);
        int exponent = (int)exp - 15;
        value = ldexpf(mantissa, exponent);
    }

    return sign ? -value : value;
}


static void matmul_q8(
    const int8_t *weights,
    const unsigned char *scales,
    const float *input,
    float *output,
    int in_dim,
    int out_dim
)
{
    int out_idx;
    for (out_idx = 0; out_idx < out_dim; ++out_idx) {
        float acc = 0.0f;
        const int8_t *row_w = weights + (size_t)out_idx * (size_t)in_dim;
        const unsigned char *row_s = scales + ((size_t)out_idx * (size_t)in_dim / Q_BLOCK) * 2u;
        int block;

        for (block = 0; block < in_dim; block += Q_BLOCK) {
            float scale = f16_to_float_le(row_s + (size_t)(block / Q_BLOCK) * 2u);
            int limit = (block + Q_BLOCK < in_dim) ? (block + Q_BLOCK) : in_dim;
            int j;
            for (j = block; j < limit; ++j) {
                acc += (float)row_w[j] * scale * input[j];
            }
        }
        output[out_idx] = acc;
    }
}


static void rms_norm(float *vec, int len)
{
    union {
        float f;
        uint32_t i;
    } u;
    float sum_sq = 0.0f;
    float mean_sq;
    float inv_rms;
    int i;

    for (i = 0; i < len; ++i) {
        sum_sq += vec[i] * vec[i];
    }

    mean_sq = sum_sq / (float)len + 1e-6f;
    u.f = mean_sq;
    u.i = 0x5f3759dfu - (u.i >> 1);
    inv_rms = u.f;
    inv_rms = inv_rms * (1.5f - 0.5f * mean_sq * inv_rms * inv_rms);
    inv_rms = inv_rms * (1.5f - 0.5f * mean_sq * inv_rms * inv_rms);

    for (i = 0; i < len; ++i) {
        vec[i] *= inv_rms;
    }
}


static void softmax_inplace(float *vec, int len)
{
    float max_val;
    float sum = 0.0f;
    int i;

    if (len <= 0) {
        return;
    }

    max_val = vec[0];
    for (i = 1; i < len; ++i) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }

    for (i = 0; i < len; ++i) {
        float x = vec[i] - max_val;
        float z;
        float e;
        if (x < -20.0f) {
            vec[i] = 0.0f;
            continue;
        }

        z = x * (1.0f / 128.0f);
        e = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666667f + z * 0.04166667f)));
        e = e * e;
        e = e * e;
        e = e * e;
        e = e * e;
        e = e * e;
        e = e * e;
        e = e * e;
        vec[i] = e;
        sum += e;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (i = 0; i < len; ++i) {
            vec[i] *= inv_sum;
        }
    }
}


static void embed_lookup(const Model *model, unsigned char token, float *out)
{
    float scale = model->em_scale / 127.0f;
    size_t base = (size_t)token * model->n_embed;
    unsigned int i;

    for (i = 0; i < model->n_embed; ++i) {
        out[i] = (float)model->emb_table[base + i] * scale;
    }
}


static void project_to_logits(const Model *model, const float *x, float *logits)
{
    float scale = model->em_scale / 127.0f;
    unsigned int v;
    for (v = 0; v < model->vocab; ++v) {
        float acc = 0.0f;
        size_t base = (size_t)v * model->n_embed;
        unsigned int i;
        for (i = 0; i < model->n_embed; ++i) {
            acc += (float)model->emb_table[base + i] * scale * x[i];
        }
        logits[v] = acc;
    }
}


static void pse_init(Runner *runner)
{
    size_t total = (size_t)runner->model->n_layers * runner->model->n_heads;
    size_t i;
    for (i = 0; i < total; ++i) {
        runner->conductance[i] = 1.0f;
    }
    runner->entropy_ema = 0.0f;
    runner->token_counter = 0;
}


static uint32_t pse_entropy(Runner *runner)
{
    uint32_t c = (uint32_t)clock() ^ (runner->token_counter * 1664525u + 1013904223u);
    c ^= c << 13;
    c ^= c >> 17;
    c ^= c << 5;
    return c;
}


static void pse_burst_inject(Runner *runner, float *x)
{
    unsigned int n_embed = runner->model->n_embed;
    union {
        float f;
        uint32_t i;
    } u;
    float sum_sq = 0.0f;
    float mean_sq;
    float inv_rms;
    float rms;
    float strength;
    uint32_t ent;
    int dims;
    unsigned int i;

    runner->token_counter++;
    if ((runner->token_counter % PSE_BURST_INTERVAL) != 0u) {
        return;
    }

    ent = pse_entropy(runner);
    for (i = 0; i < n_embed; ++i) {
        sum_sq += x[i] * x[i];
    }

    mean_sq = sum_sq / (float)n_embed + 1e-8f;
    u.f = mean_sq;
    u.i = 0x5f3759dfu - (u.i >> 1);
    inv_rms = u.f;
    inv_rms = inv_rms * (1.5f - 0.5f * mean_sq * inv_rms * inv_rms);
    rms = 1.0f / (inv_rms + 1e-8f);
    strength = rms * PSE_BURST_STRENGTH;
    dims = (n_embed < PSE_BURST_DIMS) ? (int)n_embed : PSE_BURST_DIMS;

    for (i = 0; i < (unsigned int)dims; ++i) {
        int dim = (int)((ent >> (i & 15u)) & 0x7Fu);
        float noise;
        if (dim >= (int)n_embed) {
            dim %= (int)n_embed;
        }
        noise = ((ent >> (i + 16u)) & 1u) ? strength : -strength;
        x[dim] += noise;
        ent = ent * 1664525u + 1013904223u;
    }
}


static void pse_physarum_update(Runner *runner, int layer_idx, const float *sharpness)
{
    float max_sharp = 0.0f;
    unsigned int h;

    for (h = 0; h < runner->model->n_heads; ++h) {
        if (sharpness[h] > max_sharp) {
            max_sharp = sharpness[h];
        }
    }
    if (max_sharp < 1e-6f) {
        return;
    }

    for (h = 0; h < runner->model->n_heads; ++h) {
        float norm = sharpness[h] / max_sharp;
        float *cond = &runner->conductance[conductance_offset(runner, layer_idx, (int)h)];
        if (norm > 0.5f) {
            *cond += PSE_PHYSARUM_REINFORCE * (norm - 0.5f);
        } else {
            *cond -= PSE_PHYSARUM_DECAY * (0.5f - norm);
        }
        if (*cond < PSE_PHYSARUM_MIN) {
            *cond = PSE_PHYSARUM_MIN;
        }
        if (*cond > PSE_PHYSARUM_MAX) {
            *cond = PSE_PHYSARUM_MAX;
        }
    }
}


static void pse_physarum_check_reset(Runner *runner, const float *logits)
{
    float max_val = logits[PRINTABLE_START];
    float count = 0.0f;
    float ent;
    float delta;
    unsigned int i;

    for (i = PRINTABLE_START + 1; i <= PRINTABLE_END; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    for (i = PRINTABLE_START; i <= PRINTABLE_END; ++i) {
        if (logits[i] > max_val - 10.0f) {
            count += 1.0f;
        }
    }

    ent = count / 95.0f;
    delta = ent - runner->entropy_ema;
    runner->entropy_ema = runner->entropy_ema * 0.9f + ent * 0.1f;

    if (delta > 0.5f) {
        pse_init(runner);
    }
}


static void attention_layer(Runner *runner, const LayerView *layer, int layer_idx, int pos, float *x)
{
    const Model *model = runner->model;
    int n_ctx = (pos + 1 < (int)model->ctx) ? pos + 1 : (int)model->ctx;
    float inv_sqrt_hd = 0.17678f;
    unsigned int i;
    unsigned int h;

    memcpy(runner->residual, x, model->n_embed * sizeof(float));
    rms_norm(x, (int)model->n_embed);

    matmul_q8(layer->wq, layer->sq, x, runner->q, (int)model->n_embed, (int)model->n_embed);
    matmul_q8(layer->wk, layer->sk, x, runner->k_cur, (int)model->n_embed, (int)model->n_embed);
    matmul_q8(layer->wv, layer->sv, x, runner->v_cur, (int)model->n_embed, (int)model->n_embed);

    if (pos < (int)model->ctx) {
        memcpy(kv_k_slot(runner, layer_idx, pos), runner->k_cur, model->n_embed * sizeof(float));
        memcpy(kv_v_slot(runner, layer_idx, pos), runner->v_cur, model->n_embed * sizeof(float));
    }

    memset(runner->attn_out, 0, model->n_embed * sizeof(float));
    for (h = 0; h < model->n_heads; ++h) {
        float cond = runner->conductance[conductance_offset(runner, layer_idx, (int)h)];
        const float *q_head;
        float max_score = -1e9f;
        float sum_score = 0.0f;
        unsigned int d;
        int t;

        if (cond < PSE_PHYSARUM_PRUNE) {
            runner->head_sharpness[h] = 0.0f;
            continue;
        }

        q_head = runner->q + (size_t)h * model->head_dim;
        for (t = 0; t < n_ctx; ++t) {
            const float *k_head = kv_k_slot(runner, layer_idx, t) + (size_t)h * model->head_dim;
            float score = 0.0f;
            for (d = 0; d < model->head_dim; ++d) {
                score += q_head[d] * k_head[d];
            }
            runner->attn_scores[t] = score * inv_sqrt_hd;
            if (runner->attn_scores[t] > max_score) {
                max_score = runner->attn_scores[t];
            }
            sum_score += runner->attn_scores[t];
        }

        runner->head_sharpness[h] = max_score - (sum_score / (float)n_ctx);
        softmax_inplace(runner->attn_scores, n_ctx);

        for (d = 0; d < model->head_dim; ++d) {
            float acc = 0.0f;
            for (t = 0; t < n_ctx; ++t) {
                const float *v_head = kv_v_slot(runner, layer_idx, t) + (size_t)h * model->head_dim;
                acc += runner->attn_scores[t] * v_head[d];
            }
            runner->attn_out[(size_t)h * model->head_dim + d] = acc * cond;
        }
    }

    pse_physarum_update(runner, layer_idx, runner->head_sharpness);
    matmul_q8(layer->wo, layer->so, runner->attn_out, runner->proj_out, (int)model->n_embed, (int)model->n_embed);

    for (i = 0; i < model->n_embed; ++i) {
        x[i] = runner->residual[i] + runner->proj_out[i];
    }

    memcpy(runner->residual, x, model->n_embed * sizeof(float));
    rms_norm(x, (int)model->n_embed);
    matmul_q8(layer->wff1, layer->sff1, x, runner->ff_buf, (int)model->n_embed, (int)model->n_embed * 4);
    for (i = 0; i < model->n_embed * 4u; ++i) {
        if (runner->ff_buf[i] < 0.0f) {
            runner->ff_buf[i] = 0.0f;
        }
    }
    matmul_q8(layer->wff2, layer->sff2, runner->ff_buf, runner->ff_out, (int)model->n_embed * 4, (int)model->n_embed);
    for (i = 0; i < model->n_embed; ++i) {
        x[i] = runner->residual[i] + runner->ff_out[i];
    }
}


static unsigned char greedy_sample(const float *logits)
{
    int best = PRINTABLE_START;
    int i;
    for (i = PRINTABLE_START + 1; i <= PRINTABLE_END; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return (unsigned char)best;
}


static void reset_runner(Runner *runner)
{
    const Model *model = runner->model;
    size_t kv_count = (size_t)model->n_layers * model->ctx * model->n_embed;
    memset(runner->kv.k, 0, kv_count * sizeof(float));
    memset(runner->kv.v, 0, kv_count * sizeof(float));
    memset(runner->x, 0, model->n_embed * sizeof(float));
    memset(runner->logits, 0, model->vocab * sizeof(float));
    memset(runner->tokens, 0, model->ctx * sizeof(unsigned int));
    runner->kv.pos = 0;
    runner->seq_len = 0;
    pse_init(runner);
}


static int init_runner(Runner *runner, Model *model)
{
    size_t kv_count = (size_t)model->n_layers * model->ctx * model->n_embed;
    memset(runner, 0, sizeof(*runner));
    runner->model = model;

    runner->kv.k = (float *)calloc(kv_count, sizeof(float));
    runner->kv.v = (float *)calloc(kv_count, sizeof(float));
    runner->x = (float *)calloc(model->n_embed, sizeof(float));
    runner->logits = (float *)calloc(model->vocab, sizeof(float));
    runner->tokens = (unsigned int *)calloc(model->ctx, sizeof(unsigned int));
    runner->q = (float *)calloc(model->n_embed, sizeof(float));
    runner->k_cur = (float *)calloc(model->n_embed, sizeof(float));
    runner->v_cur = (float *)calloc(model->n_embed, sizeof(float));
    runner->attn_out = (float *)calloc(model->n_embed, sizeof(float));
    runner->ff_buf = (float *)calloc(model->n_embed * 4u, sizeof(float));
    runner->attn_scores = (float *)calloc(model->ctx, sizeof(float));
    runner->residual = (float *)calloc(model->n_embed, sizeof(float));
    runner->proj_out = (float *)calloc(model->n_embed, sizeof(float));
    runner->ff_out = (float *)calloc(model->n_embed, sizeof(float));
    runner->head_sharpness = (float *)calloc(model->n_heads, sizeof(float));
    runner->conductance = (float *)calloc((size_t)model->n_layers * model->n_heads, sizeof(float));

    if (!runner->kv.k || !runner->kv.v || !runner->x || !runner->logits || !runner->tokens ||
        !runner->q || !runner->k_cur || !runner->v_cur || !runner->attn_out || !runner->ff_buf ||
        !runner->attn_scores || !runner->residual || !runner->proj_out || !runner->ff_out ||
        !runner->head_sharpness || !runner->conductance) {
        fprintf(stderr, "Out of memory allocating runner state\n");
        return -1;
    }

    reset_runner(runner);
    return 0;
}


static void free_runner(Runner *runner)
{
    free(runner->kv.k);
    free(runner->kv.v);
    free(runner->x);
    free(runner->logits);
    free(runner->tokens);
    free(runner->q);
    free(runner->k_cur);
    free(runner->v_cur);
    free(runner->attn_out);
    free(runner->ff_buf);
    free(runner->attn_scores);
    free(runner->residual);
    free(runner->proj_out);
    free(runner->ff_out);
    free(runner->head_sharpness);
    free(runner->conductance);
    memset(runner, 0, sizeof(*runner));
}


static unsigned char next_token(Runner *runner, unsigned char input_token)
{
    const Model *model = runner->model;
    int pos = runner->kv.pos;
    unsigned int layer_idx;
    unsigned char token;

    embed_lookup(model, input_token, runner->x);

    for (layer_idx = 0; layer_idx < model->n_layers; ++layer_idx) {
        attention_layer(runner, &model->layers[layer_idx], (int)layer_idx, pos, runner->x);
    }

    rms_norm(runner->x, (int)model->n_embed);
    pse_burst_inject(runner, runner->x);
    project_to_logits(model, runner->x, runner->logits);
    pse_physarum_check_reset(runner, runner->logits);

    token = greedy_sample(runner->logits);

    if (runner->kv.pos < (int)model->ctx - 1) {
        runner->kv.pos++;
    } else {
        unsigned int l;
        size_t row_bytes = (size_t)(model->ctx - 1) * model->n_embed * sizeof(float);
        for (l = 0; l < model->n_layers; ++l) {
            float *kbase = runner->kv.k + (size_t)l * model->ctx * model->n_embed;
            float *vbase = runner->kv.v + (size_t)l * model->ctx * model->n_embed;
            memmove(kbase, kbase + model->n_embed, row_bytes);
            memmove(vbase, vbase + model->n_embed, row_bytes);
        }
    }

    if (runner->seq_len < (int)model->ctx) {
        runner->tokens[runner->seq_len++] = input_token;
    }

    return token;
}


static char *join_args(int argc, char **argv, int start)
{
    size_t total = 1;
    int i;
    char *out;
    char *cursor;

    for (i = start; i < argc; ++i) {
        total += strlen(argv[i]) + 1;
    }

    out = (char *)malloc(total);
    if (!out) {
        return NULL;
    }

    cursor = out;
    for (i = start; i < argc; ++i) {
        size_t len = strlen(argv[i]);
        memcpy(cursor, argv[i], len);
        cursor += len;
        if (i + 1 < argc) {
            *cursor++ = ' ';
        }
    }
    *cursor = '\0';
    return out;
}


static void print_usage(const char *argv0)
{
    fprintf(stderr, "Usage: %s [-w weights.bin] [-n max_tokens] prompt...\n", argv0);
    fprintf(stderr, "Default weights path: %s\n", DEFAULT_WEIGHTS);
}


static int generate_stream(Runner *runner, const char *prompt, int max_tokens)
{
    size_t prompt_len = strlen(prompt);
    unsigned char token = 0;
    int emitted = 0;
    clock_t begin;
    clock_t end;
    double seconds;
    size_t total_runtime;
    size_t total_memory;

    reset_runner(runner);
    for (size_t i = 0; i < prompt_len; ++i) {
        token = next_token(runner, (unsigned char)prompt[i]);
    }

    printf("Output: ");
    fflush(stdout);
    begin = clock();
    while (emitted < max_tokens) {
        token = next_token(runner, token);
        if (token == 0) {
            break;
        }
        putchar((int)token);
        fflush(stdout);
        emitted++;
    }
    end = clock();
    putchar('\n');

    seconds = (double)(end - begin) / (double)CLOCKS_PER_SEC;
    total_runtime = runtime_bytes(runner);
    total_memory = total_runtime + runner->model->blob_size;
    printf(
        "Stats: prompt_tok=%zu generated=%d tok/s=%.2f runtime=%.1f KB weights=%.1f KB total=%.1f KB\n",
        prompt_len,
        emitted,
        emitted / (seconds > 1e-9 ? seconds : 1e-9),
        total_runtime / 1024.0,
        runner->model->blob_size / 1024.0,
        total_memory / 1024.0
    );
    return 0;
}


int main(int argc, char **argv)
{
    const char *weights_path = DEFAULT_WEIGHTS;
    int max_tokens = DEFAULT_MAX_TOKENS;
    int argi = 1;
    char *prompt;
    Model model;
    Runner runner;

    while (argi < argc && argv[argi][0] == '-') {
        if (!strcmp(argv[argi], "-h") || !strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        if (!strcmp(argv[argi], "-w")) {
            if (argi + 1 >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            weights_path = argv[argi + 1];
            argi += 2;
            continue;
        }
        if (!strcmp(argv[argi], "-n")) {
            if (argi + 1 >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            max_tokens = atoi(argv[argi + 1]);
            if (max_tokens <= 0) {
                fprintf(stderr, "max_tokens must be positive\n");
                return 1;
            }
            argi += 2;
            continue;
        }
        fprintf(stderr, "Unknown option: %s\n", argv[argi]);
        print_usage(argv[0]);
        return 1;
    }

    if (argi >= argc) {
        print_usage(argv[0]);
        return 1;
    }

    prompt = join_args(argc, argv, argi);
    if (!prompt) {
        fprintf(stderr, "Out of memory joining prompt arguments\n");
        return 1;
    }

    if (load_model(&model, weights_path) != 0) {
        free(prompt);
        free_model(&model);
        return 1;
    }

    if (init_runner(&runner, &model) != 0) {
        free(prompt);
        free_runner(&runner);
        free_model(&model);
        return 1;
    }

    printf(
        "Loaded %s\nModel: layers=%u embed=%u heads=%u ctx=%u vocab=%u em_scale=%.3f\nPrompt: %s\n",
        weights_path,
        model.n_layers,
        model.n_embed,
        model.n_heads,
        model.ctx,
        model.vocab,
        model.em_scale,
        prompt
    );

    generate_stream(&runner, prompt, max_tokens);

    free(prompt);
    free_runner(&runner);
    free_model(&model);
    return 0;
}
