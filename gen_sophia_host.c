// SPDX-License-Identifier: MIT
/*
 * gen_sophia_host.c — host-side Sophia dialog generator
 *
 * Compiles nano_gpt.c natively (x86-64 Linux) on sophia5070node.
 * The SEAI header was written on a LE x86 host, so all fields are
 * native. The 4-byte magic "SEAI" is read as LE 0x49414553 on x86
 * (fixed in nano_gpt_host.c with the extra || check).
 *
 * Build:
 *   gcc -O2 -o gen_sophia_host gen_sophia_host.c nano_gpt_host.c \
 *       -I. -Ihostinc -lm
 *
 * Usage:
 *   ./gen_sophia_host [weights_path] [prompt] [max_tokens]
 */

#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "nano_gpt.h"

/* ---- Static storage ---- */
static SGAIState   g_sgai;
static SGAIKVCache g_kv;

int main(int argc, char **argv)
{
    const char *weights_path = (argc > 1) ? argv[1]
        : "/home/sophia5070node/n64dev/legend_of_elya_rom/filesystem/sophia_weights.bin";
    const char *prompt = (argc > 2) ? argv[2]
        : "Who are you? Tell me your name.";
    int max_tokens = (argc > 3) ? atoi(argv[3]) : 80;

    /* Load weights */
    FILE *f = fopen(weights_path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open: %s\n", weights_path); return 1; }
    fseek(f, 0, SEEK_END);
    long wsz = ftell(f);
    rewind(f);
    static uint8_t wbuf[300 * 1024];
    if (wsz > (long)sizeof(wbuf)) {
        fprintf(stderr, "ERROR: weights too large (%ld)\n", wsz);
        fclose(f); return 1;
    }
    size_t nr = fread(wbuf, 1, wsz, f);
    fclose(f);
    if ((long)nr != wsz) { fprintf(stderr, "ERROR: short read\n"); return 1; }
    fprintf(stderr, "[host-gen] loaded %ld bytes\n", wsz);

    /* Init model */
    memset(&g_sgai, 0, sizeof(g_sgai));
    memset(&g_kv,   0, sizeof(g_kv));
    g_sgai.kv = &g_kv;
    sgai_init(&g_sgai, wbuf);

    /* sgai_init overwrites kv via memalign — reclaim our static buffer */
    if (g_sgai.kv && g_sgai.kv != &g_kv) {
        free(g_sgai.kv);
        g_sgai.kv = &g_kv;
        memset(&g_kv, 0, sizeof(g_kv));
    }

    if (!g_sgai.is_loaded) {
        fprintf(stderr, "ERROR: sgai_init failed (is_loaded=0)\n");
        return 1;
    }
    fprintf(stderr, "[host-gen] model ready\n");
    fprintf(stderr, "[host-gen] prompt: \"%s\"  max_tokens=%d\n", prompt, max_tokens);

    /* Generate */
    static uint8_t outbuf[128];
    memset(outbuf, 0, sizeof(outbuf));
    sgai_generate(&g_sgai,
                  (const uint8_t *)prompt, (int)strlen(prompt),
                  outbuf, max_tokens, 128);

    /* Debug: raw hex */
    fprintf(stderr, "[host-gen] raw out: ");
    for (int i = 0; i < 32; i++) fprintf(stderr, "%02X ", outbuf[i]);
    fprintf(stderr, "\n");

    /* Print printable ASCII */
    int out_len = 0;
    for (int i = 0; i < max_tokens && outbuf[i]; i++) {
        uint8_t c = outbuf[i];
        if (c >= 0x20 && c <= 0x7E) { putchar(c); out_len++; }
        else if (c == '\n' || c == '\r' || c == '\t') { putchar(' '); out_len++; }
    }
    putchar('\n');
    fprintf(stderr, "[host-gen] %d printable chars\n", out_len);
    return 0;
}
