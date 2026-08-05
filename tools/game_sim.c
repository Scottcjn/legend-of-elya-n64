/* game_sim.c — replays update_generating()'s EXACT loop on the host:
 * prompt at temperature 0, output at temperature_q8 = 64, stop on '\n',
 * stop on '.' once >= 8 output chars, stop at 80 chars. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"
static void *slurp(const char*p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long l=ftell(f);
 fseek(f,0,SEEK_SET);void*b=malloc(l);fread(b,1,l,f);fclose(f);return b;}
int main(int argc,char**argv){ /* blob prompt nruns */
  void*blob=slurp(argv[1]); const char*pr=argv[2]; int R=atoi(argv[3]);
  static SGAIState st; sgai_init(&st,blob);
  long tot=0, dbl=0, ended=0, n=0;
  for(int r=0;r<R;r++){
    sgai_reset(&st);
    host_rng_reseed(0x12345678u + r*2654435761u);
    int plen=strlen(pr); uint8_t tok=0;
    for(int i=0;i<plen;i++) tok=sgai_next_token(&st,(uint8_t)pr[i],0);
    char buf[96]; int L=0, out=0, fin=0;
    for(;;){
      uint8_t t = sgai_next_token(&st,tok,64); tok=t; out++;
      if(t=='\n'){ fin=1; break; }
      if(t=='.' && out>=8){ fin=1; break; }
      if(t>=32&&t<=126&&L<80) buf[L++]=(char)t;
      if(L>=80) break;
    }
    buf[L]=0;
    for(int i=1;i<L;i++) if(buf[i]==buf[i-1]) dbl++;
    tot+=L; ended+=fin; n++;
    if(r<3) printf("   run%d %s%s\n", r, buf, fin?" [.]":" [TRUNC80]");
  }
  printf("  -> mean len %.1f   ended-with-period %ld/%ld   doubled-letter pairs %ld\n",
         (double)tot/n, ended, n, dbl);
  return 0;
}
