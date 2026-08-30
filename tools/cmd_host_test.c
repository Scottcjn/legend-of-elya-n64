// SPDX-License-Identifier: MIT
/* cmd_host_test.c — does the model actually EMIT commands, and is the trie
 * the thing that shapes them?
 *
 * Runs the ROM's own nano_gpt.c natively with SGAI_CMD_BAND on and the two
 * hooks wired to the generated trie, exactly as the game will. This answers
 * the question the design named as its likeliest failure: whether a
 * 3.2M-parameter byte-level model learns to emit 0x01 at the right moment.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "nano_gpt.h"
#include "../cmd_trie.h"
#include "../nano_gpt.c"

static int   g_npc = 0;
static int   g_node = 0;
static int   g_in_cmd = 0;
static char  g_cmd[64];
static int   g_cmdlen = 0;
static int   g_emitted = 0, g_parsed = 0;

static int  cmd_allowed(void) { return 1; }
static int  cmd_mask(uint8_t *mask) { return cmd_legal(g_node, g_npc, mask); }
static void cmd_byte(uint8_t b)
{
    if (b == CMD_START) { g_in_cmd = 1; g_node = 0; g_cmdlen = 0; g_emitted++; return; }
    if (!g_in_cmd) return;
    if (b == CMD_END) {
        g_cmd[g_cmdlen] = 0; g_in_cmd = 0; sgai_cmd_active = 0;
        g_parsed++;
        printf("      -> COMMAND: %s\n", g_cmd);
        return;
    }
    int nx = cmd_advance(g_node, b);
    if (nx < 0) { printf("      -> trie rejected byte %02x\n", b); g_in_cmd = 0; sgai_cmd_active = 0; return; }
    g_node = nx;
    if (g_cmdlen < (int)sizeof g_cmd - 1) g_cmd[g_cmdlen++] = (char)b;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s blob npc_index [prompts...]\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("blob"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(n); if (fread(buf, 1, n, f) != (size_t)n) return 1; fclose(f);
    g_npc = atoi(argv[2]);

    static SGAIState st; static SGAIScratch sc;
    sgai_init_ex(&st, buf, SGAI_ENGINE_CPU, &sc, 0);
    if (!st.is_loaded) { fprintf(stderr, "blob did not load\n"); return 1; }

    sgai_cmd_allowed = cmd_allowed;
    sgai_cmd_mask    = cmd_mask;
    sgai_cmd_byte    = cmd_byte;

    for (int a = 3; a < argc; a++) {
        const char *p = argv[a];
        sgai_reset(&st);
        sgai_cmd_active = 0; g_in_cmd = 0;
        uint8_t tok = 0;
        for (const char *c = p; *c; c++) tok = sgai_next_token(&st, (uint8_t)*c, 0);
        printf("  %-28s ", p);
        for (int i = 0; i < 64; i++) {
            if (tok == '\n') break;
            if (tok >= 32 && tok < 127 && !g_in_cmd) putchar(tok);
            tok = sgai_next_token(&st, tok, 0);
        }
        putchar('\n');
    }
    printf("\ncommands emitted=%d parsed=%d\n", g_emitted, g_parsed);
    return 0;
}
