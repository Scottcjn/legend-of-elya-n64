/* ce_probe.c — teacher-forced cross-entropy (bits/char) of the REAL engine on a
 * fixed byte stream.  Corpus-free quality comparison between runtime arms:
 * same weights, same text, only the runtime differs, so lower BPC == the
 * runtime is preserving the model better. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"
static void *slurp(const char*p,size_t*n){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);
 long l=ftell(f);fseek(f,0,SEEK_SET);void*b=malloc(l);fread(b,1,l,f);fclose(f);*n=l;return b;}
int main(int argc,char**argv){ /* blob seqfile [csv] */
  size_t bl,sl; void*blob=slurp(argv[1],&bl); unsigned char*seq=slurp(argv[2],&sl);
  static SGAIState st; sgai_init(&st,blob); if(!st.is_loaded){fprintf(stderr,"reject\n");return 1;}
  sgai_reset(&st);
  FILE*csv = (argc>3)?fopen(argv[3],"w"):NULL;
  double tot=0; long cnt=0;
  for(size_t i=0;i+1<sl;i++){
    sgai_next_token(&st,seq[i],0);
    /* log-sum-exp over the ASCII range the sampler actually uses (32..126) */
    double mx=-1e30; for(int v=32;v<=126;v++) if(st.logits[v]>mx) mx=st.logits[v];
    double s=0; for(int v=32;v<=126;v++) s+=exp((double)st.logits[v]-mx);
    int t=seq[i+1];
    double lp;
    if(t<32||t>126) continue;                       /* skip out-of-range targets */
    lp = ((double)st.logits[t]-mx) - log(s);
    tot += -lp/log(2.0); cnt++;
    if(csv) fprintf(csv,"%zu,%.6f\n",i,-lp/log(2.0));
  }
  if(csv) fclose(csv);
  printf("%-28s  BPC = %8.4f   over %ld positions\n", argv[1], tot/cnt, cnt);
  return 0;
}
