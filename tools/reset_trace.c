#include <stdio.h>
#include <stdlib.h>
#include <string.h>
long pse_reset_count = 0; float pse_last_delta = 0;
#include "nano_gpt.h"
#include "../nano_gpt.c"
static void *slurp(const char*p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long l=ftell(f);
 fseek(f,0,SEEK_SET);void*b=malloc(l);fread(b,1,l,f);fclose(f);return b;}
int main(int argc,char**argv){
  static SGAIState st; sgai_init(&st,slurp(argv[1])); sgai_reset(&st);
  const char*pr=argv[2]; int n=atoi(argv[3]); int temp=(argc>4)?atoi(argv[4]):0;
  int plen=strlen(pr); uint8_t tok=0; float dmax=-1e9;
  for(int i=0;i<plen;i++) tok=sgai_next_token(&st,(uint8_t)pr[i],0);
  long before=pse_reset_count;
  for(int i=0;i<n;i++){ if(i) tok=sgai_next_token(&st,tok,(uint32_t)temp);
    if(pse_last_delta>dmax) dmax=pse_last_delta; }
  printf("prompt=%-24s n=%d temp=%d  resets_during_gen=%ld  max_delta=%.4f (threshold 0.5)\n",
         pr,n,temp,pse_reset_count-before,dmax);
  return 0;
}
