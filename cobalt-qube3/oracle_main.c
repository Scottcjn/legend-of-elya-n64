#include "nano_gpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float g_em_override = 0.0f;

int main(int argc, char **argv) {
    const char *wpath = "sophia.bin";
    const char *prompt = "The ";
    int maxtok = 200;
    unsigned int temp_q8 = 200;
    int i, plen;
    FILE *f;
    long sz;
    void *wbuf;
    static SGAIState st;
    static unsigned char out[512];

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i+1<argc) wpath = argv[++i];
        else if (!strcmp(argv[i], "-p") && i+1<argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1<argc) maxtok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1<argc) temp_q8 = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-e") && i+1<argc) g_em_override = atof(argv[++i]);
    }

    f = fopen(wpath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", wpath); return 1; }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    wbuf = malloc(sz);
    if (!wbuf || (long)fread(wbuf, 1, sz, f) != sz) { fprintf(stderr, "read fail\n"); return 1; }
    fclose(f);
    fprintf(stderr, "[oracle] %ld bytes weights, prompt=\"%s\" n=%d\n", sz, prompt, maxtok);

    sgai_init(&st, wbuf);
    if (!st.is_loaded) { fprintf(stderr, "[oracle] bad weights\n"); return 1; }
    if (g_em_override > 0.0f) st.em_scale = g_em_override;

    plen = strlen(prompt);
    if (maxtok > 500) maxtok = 500;
    sgai_generate(&st, (const unsigned char*)prompt, plen, out, maxtok, temp_q8);
    printf("%s%s\n", prompt, out);
    return 0;
}
