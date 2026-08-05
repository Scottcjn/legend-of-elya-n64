// SPDX-License-Identifier: MIT
/*
 * host_eval.c — run the REAL nano_gpt.c natively as a quality reference.
 *
 * This is deliberately NOT a numpy re-implementation.  It compiles
 * nano_gpt.c itself (the exact translation unit the ROM builds) against a
 * freestanding libdragon shim, so the fast-inverse-sqrt rms_norm, the Taylor
 * exp() in softmax_f, the greedy argmax restricted to ASCII 32..126, and the
 * PSE Physarum router are all the ROM's code, not an approximation of it.
 *
 * Two differences from the ROM, both compile-time and both documented:
 *   HOST_BUILD     - f16_to_float does not byte-swap (the blob is LE and so is
 *                    the host), and sample_logits' CP0 RNG is an LCG.  Neither
 *                    is reached in greedy mode except the f16 swap.
 *   BENCH_DET_PSE  - pse_entropy() is a counter LCG instead of CP0 Count, so
 *                    two runs of the same weights give the same tokens.
 *
 * Usage:
 *   host_eval REF.bin [CAND.bin ...] --prompt "..." [--prompt "..."] [-n 24]
 *
 * The first blob is the reference.  For every other blob it reports the
 * generated text side by side, per-token top-1 agreement with the reference,
 * and mean absolute logit error over ASCII 32..126.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nano_gpt.h"

/* nano_gpt.c's statics are file-local; include it so we can reach the engine
 * without exporting anything. */
#include "../nano_gpt.c"

#define MAX_BLOBS 12
#define MAX_PROMPTS 8
#define MAX_TOK 256
#define FORCE_MAX 512

static void *slurp(const char *p, size_t *len)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return b;
}

/* Greedy decode.  Records the logit vector at every emitted position. */
static void run(const void *blob, const char *prompt, int n_new,
                uint8_t *out, float *logit_log)
{
    static SGAIState st;
    sgai_init(&st, blob);
    if (!st.is_loaded) { fprintf(stderr, "blob rejected by sgai_init\n"); exit(1); }
    sgai_reset(&st);

    int plen = (int)strlen(prompt);
    uint8_t tok = 0;
    for (int i = 0; i < plen; i++)
        tok = sgai_next_token(&st, (uint8_t)prompt[i], 0);

    for (int i = 0; i < n_new; i++) {
        if (i) tok = sgai_next_token(&st, tok, 0);
        out[i] = tok;
        if (logit_log)
            memcpy(logit_log + (size_t)i * SGAI_VOCAB, st.logits,
                   SGAI_VOCAB * sizeof(float));
    }
    free(st.kv);
    st.kv = NULL;
}

/* Teacher-forced pass: drive the model with a FIXED token sequence and record
 * the argmax + logits at every position.  This is the honest per-token
 * agreement measure.  Free-running top-1 collapses to noise after the first
 * divergence — one different token changes every token after it — so it
 * measures "when did they first disagree", not "how often do they disagree". */
static int forced(const void *blob, const uint8_t *seq, int n,
                  uint8_t *argmax_out, float *logit_log)
{
    static SGAIState st;
    sgai_init(&st, blob);
    if (!st.is_loaded) { fprintf(stderr, "blob rejected by sgai_init\n"); exit(1); }
    sgai_reset(&st);
    int m = n;
    for (int i = 0; i < m; i++) {
        argmax_out[i] = sgai_next_token(&st, seq[i], 0);
        if (logit_log)
            memcpy(logit_log + (size_t)i * SGAI_VOCAB, st.logits,
                   SGAI_VOCAB * sizeof(float));
    }
    free(st.kv);
    st.kv = NULL;
    return m;
}

static void show(const uint8_t *b, int n)
{
    putchar('"');
    for (int i = 0; i < n; i++) {
        if (b[i] >= 32 && b[i] < 127) putchar(b[i]);
        else printf("\\x%02x", b[i]);
    }
    putchar('"');
}

int main(int argc, char **argv)
{
    const char *blobs[MAX_BLOBS]; int nblob = 0;
    const char *prompts[MAX_PROMPTS]; int nprompt = 0;
    const char *forced_file = NULL;
    const char *dump_file = NULL;
    int n_new = 24;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompts[nprompt++] = argv[++i];
        else if (!strcmp(argv[i], "--forced") && i + 1 < argc) forced_file = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_file = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = atoi(argv[++i]);
        else blobs[nblob++] = argv[i];
    }
    if (nblob < 1) { fprintf(stderr, "need at least a reference blob\n"); return 1; }
    if (nprompt == 0) prompts[nprompt++] = "Who are you?";

    /* ---- teacher-forced arm ---------------------------------------- */
    static uint8_t fseq[FORCE_MAX];
    static uint8_t fref[FORCE_MAX], fcnd[FORCE_MAX];
    static float   fref_log[FORCE_MAX * SGAI_VOCAB], fcnd_log[FORCE_MAX * SGAI_VOCAB];
    int fn = 0;
    if (forced_file) {
        size_t fl; unsigned char *t = slurp(forced_file, &fl);
        fn = (int)(fl < FORCE_MAX ? fl : FORCE_MAX);
        memcpy(fseq, t, (size_t)fn);
        free(t);
    }

    static uint8_t ref_out[MAX_PROMPTS][MAX_TOK];
    static float ref_log[MAX_PROMPTS][MAX_TOK * SGAI_VOCAB];
    static uint8_t cnd_out[MAX_TOK];
    static float cnd_log[MAX_TOK * SGAI_VOCAB];

    size_t len;
    void *ref = slurp(blobs[0], &len);
    printf("REF  %s  (%zu B)\n", blobs[0], len);
    for (int p = 0; p < nprompt; p++) {
        run(ref, prompts[p], n_new, ref_out[p], ref_log[p]);
        printf("  %-16s -> ", prompts[p]); show(ref_out[p], n_new); putchar('\n');
    }

    if (fn) { forced(ref, fseq, fn, fref, fref_log); printf("  [teacher-forced over %d tokens]\n", fn); }

    /* --dump writes the reference blob's own results so two DIFFERENTLY BUILT
     * host_eval binaries (e.g. float32 KV vs int8 KV) can be compared. */
    if (dump_file) {
        FILE *df = fopen(dump_file, "wb");
        if (!df) { perror("dump"); return 1; }
        for (int p = 0; p < nprompt; p++) fwrite(ref_out[p], 1, (size_t)n_new, df);
        if (fn) fwrite(fref, 1, (size_t)fn, df);
        fclose(df);
        fprintf(stderr, "dumped %d free-run + %d forced tokens to %s\n",
                nprompt * n_new, fn, dump_file);
    }

    for (int b = 1; b < nblob; b++) {
        size_t l2;
        void *cnd = slurp(blobs[b], &l2);
        printf("\nCAND %s  (%zu B, %.4fx smaller)\n", blobs[b], l2, (double)len / (double)l2);
        long agree_tot = 0, agree_n = 0;
        double err_sum = 0.0; long err_n = 0;
        for (int p = 0; p < nprompt; p++) {
            run(cnd, prompts[p], n_new, cnd_out, cnd_log);
            int ag = 0;
            for (int i = 0; i < n_new; i++) if (cnd_out[i] == ref_out[p][i]) ag++;
            agree_tot += ag; agree_n += n_new;
            for (int i = 0; i < n_new; i++)
                for (int v = 32; v <= 126; v++) {
                    err_sum += fabs((double)cnd_log[(size_t)i * SGAI_VOCAB + v]
                                    - (double)ref_log[p][(size_t)i * SGAI_VOCAB + v]);
                    err_n++;
                }
            printf("  %-16s -> ", prompts[p]); show(cnd_out, n_new);
            printf("   top1 %3d/%d = %5.1f%%\n", ag, n_new, 100.0 * ag / n_new);
        }
        printf("  free-run  top1 %ld/%ld = %.1f%%   mean|dlogit| = %.4f\n",
               agree_tot, agree_n, 100.0 * (double)agree_tot / (double)agree_n,
               err_sum / (double)err_n);
        if (fn) {
            forced(cnd, fseq, fn, fcnd, fcnd_log);
            int ag = 0; double e = 0.0; long en = 0;
            for (int i = 0; i < fn; i++) {
                if (fcnd[i] == fref[i]) ag++;
                for (int v = 32; v <= 126; v++) {
                    e += fabs((double)fcnd_log[(size_t)i*SGAI_VOCAB+v]
                            - (double)fref_log[(size_t)i*SGAI_VOCAB+v]);
                    en++;
                }
            }
            printf("  forced    top1 %d/%d = %.1f%%   mean|dlogit| = %.4f\n",
                   ag, fn, 100.0*ag/fn, e/(double)en);
        }
        free(cnd);
    }
    return 0;
}
