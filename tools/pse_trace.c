/* pse_trace.c — free-run the REAL nano_gpt.c and print the PSE conductance
 * state after every token, alongside the token emitted. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"

static void *slurp(const char *p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);
 fseek(f,0,SEEK_SET);void*b=malloc(n);fread(b,1,n,f);fclose(f);return b;}

int main(int argc,char**argv){
  const char*blob=argv[1]; const char*prompt=argv[2]; int n=atoi(argv[3]);
  static SGAIState st; sgai_init(&st,slurp(blob)); sgai_reset(&st);
  int plen=strlen(prompt); uint8_t tok=0;
  for(int i=0;i<plen;i++){ tok=sgai_next_token(&st,(uint8_t)prompt[i],0); }
  printf("pos char  cmin  cmean cmax  nAtMax nAtMin  ent_ema\n");
  for(int i=0;i<n;i++){
    if(i) tok=sgai_next_token(&st,tok,0);
    float mn=9e9,mx=-9e9,sum=0; int hi=0,lo=0;
    for(int l=0;l<SGAI_N_LAYERS;l++)for(int h=0;h<SGAI_N_HEADS;h++){
      float c=pse_state.conductance[l][h];
      if(c<mn)mn=c; if(c>mx)mx=c; sum+=c;
      if(c>1.4999f)hi++; if(c<0.5001f)lo++;
    }
    printf("%3d %c    %.3f %.3f %.3f  %2d/%d  %2d/%d   %.4f\n",
      i,(tok>=32&&tok<127)?tok:'?',mn,sum/(SGAI_N_LAYERS*SGAI_N_HEADS),mx,
      hi,SGAI_N_LAYERS*SGAI_N_HEADS,lo,SGAI_N_LAYERS*SGAI_N_HEADS,pse_state.entropy_ema);
  }
  return 0;
}
