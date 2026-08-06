/* samp_probe.c — replays update_generating_step() EXACTLY (legend_of_elya.c
 * 1500-1575) and, with -DSAMP_TRACE, dumps the inside of every sample_logits
 * call on the temperature path: top-5 pre-penalty distribution, which chars the
 * repetition penalty zeroed and how much mass that removed, the surviving
 * total, the uniform draw, whether the total<=0 fallback fired, and the rank of
 * the char actually chosen.
 *
 * The prompt string is built the way start_dialog_from_prompt() builds it:
 *   persona_prefix + dialog option, e.g. "sage says: Who are you?: "
 *
 * usage: samp_probe BLOB "PROMPT" NRUNS [TEMP_Q8] [TRACEFILE]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"

static void *slurp(const char *p, long *out_len) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(l);
    if (fread(b, 1, l, f) != (size_t)l) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f); if (out_len) *out_len = l; return b;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s BLOB PROMPT NRUNS [TEMP_Q8] [TRACEFILE]\n", argv[0]); return 1; }
    long blen = 0;
    void *blob = slurp(argv[1], &blen);
    const char *pr = argv[2];
    int R = atoi(argv[3]);
    uint32_t temp = (argc > 4) ? (uint32_t)atoi(argv[4]) : 64u;

    static SGAIState st;
    sgai_init(&st, blob);
    if (!st.is_loaded) { fprintf(stderr, "blob NOT loaded (%ld B)\n", blen); return 2; }

    long tot = 0, dbl = 0, ended = 0, n = 0, bangs = 0, chars = 0;
    for (int r = 0; r < R; r++) {
#ifdef SAMP_TRACE
        if (argc > 5 && r == 0) {
            samp_trace_open(argv[5]);
            fprintf(samp_trace_fp, "# blob=%s prompt=\"%s\" temp_q8=%u run=%d\n",
                    argv[1], pr, temp, r);
        } else { if (samp_trace_fp) { fclose(samp_trace_fp); samp_trace_fp = NULL; } }
#endif
        sgai_reset(&st);
        host_rng_reseed(0x12345678u + (uint32_t)r * 2654435761u);

        /* ---- phase 0: feed the prompt at temperature 0 ---- */
        int plen = (int)strlen(pr);
        uint8_t tok = 0;
        for (int i = 0; i < plen; i++) tok = sgai_next_token(&st, (uint8_t)pr[i], 0);
        /* legend_of_elya.c:1518-1524 keeps this prediction in G.gen_last_tok and
         * FEEDS it back on the next call, but never appends it to dialog_buf.
         * It is the first character of the model's answer and the player never
         * sees it.  Printed here as [dropped=X]. */
        uint8_t dropped = tok;

        /* ---- phase 1: generate at temperature_q8 ---- */
        char buf[128]; int L = 0, out = 0, fin = 0;
        for (;;) {
            uint8_t t = sgai_next_token(&st, tok, temp);
            tok = t; out++;
            if (t == '\n') { fin = 1; break; }
            if (t == '.' && out >= 8) { fin = 1; break; }
            if (t >= 32 && t <= 126 && L < 80) buf[L++] = (char)t;
            if (L >= 80) break;
            if (out > 200) break;
        }
        buf[L] = 0;
        for (int i = 1; i < L; i++) if (buf[i] == buf[i-1]) dbl++;
        for (int i = 0; i < L; i++) { chars++; if (buf[i] == '!') bangs++; }
        tot += L; ended += fin; n++;
        if (r < 64) printf("  run%-2d | [dropped=%c] %s%s\n", r, (dropped>=32&&dropped<=126)?dropped:'?', buf, fin ? " [.]" : " [TRUNC]");
    }
    printf("  == temp_q8=%u  runs=%ld  mean len %.1f  clean-terminations %ld/%ld  "
           "doubled-letter pairs %ld  '!' %ld/%ld (%.2f%%)\n",
           temp, n, (double)tot / n, ended, n, dbl, bangs, chars,
           100.0 * (double)bangs / (chars ? chars : 1));
    return 0;
}
