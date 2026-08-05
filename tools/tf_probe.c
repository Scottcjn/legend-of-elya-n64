/* tf_probe.c — teacher-forced argmax dump.  Drive the REAL engine with a FIXED
 * byte sequence and write one argmax byte per position.  Cross-binary
 * comparison of these dumps isolates per-position prediction damage from
 * error compounding. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"
static void *slurp(const char*p,size_t*n){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);
 long l=ftell(f);fseek(f,0,SEEK_SET);void*b=malloc(l);fread(b,1,l,f);fclose(f);*n=l;return b;}
int main(int argc,char**argv){ /* blob seqfile outfile */
  size_t bl,sl; void*blob=slurp(argv[1],&bl); unsigned char*seq=slurp(argv[2],&sl);
  static SGAIState st; sgai_init(&st,blob); if(!st.is_loaded){fprintf(stderr,"reject\n");return 1;}
  sgai_reset(&st);
  FILE*o=fopen(argv[3],"wb");
  for(size_t i=0;i<sl;i++){ uint8_t a=sgai_next_token(&st,seq[i],0); fputc(a,o); }
  fclose(o); return 0;
}
