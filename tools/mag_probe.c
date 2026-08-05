/* mag_probe.c — per-token tensor statistics + per-token wall time, free-running. */
#define MAG_TRACE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nano_gpt.h"
#include "../nano_gpt.c"
static void *slurp(const char*p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long l=ftell(f);
 fseek(f,0,SEEK_SET);void*b=malloc(l);fread(b,1,l,f);fclose(f);return b;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){ /* blob prompt n [temp] */
  static SGAIState st; sgai_init(&st,slurp(argv[1])); sgai_reset(&st);
  const char*pr=argv[2]; int n=atoi(argv[3]); int temp=(argc>4)?atoi(argv[4]):0;
  int plen=strlen(pr); uint8_t tok=0;
  printf("phase pos char ms      ffzero%%  |x|max   |x|mean  |attn|max  logitmax  logitmin  denorm nonfin  cmean\n");
  for(int i=0;i<plen+n;i++){
    memset(&mag,0,sizeof(mag));
    double t0=now();
    uint8_t in = (i<plen)?(uint8_t)pr[i]:tok;
    tok = sgai_next_token(&st,in,(i<plen)?0:(uint32_t)temp);
    double dt=(now()-t0)*1000.0;
    double lmax=-1e30,lmin=1e30;
    for(int v=32;v<=126;v++){ if(st.logits[v]>lmax)lmax=st.logits[v]; if(st.logits[v]<lmin)lmin=st.logits[v]; }
    double cs=0; for(int l=0;l<SGAI_N_LAYERS;l++)for(int h=0;h<SGAI_N_HEADS;h++) cs+=pse_state.conductance[l][h];
    printf("%-5s %3d %c   %6.1f %7.2f %9.3f %9.4f %9.3f %9.2f %9.2f %5ld %5ld  %.3f\n",
      (i<plen)?"pmt":"gen", i, (tok>=32&&tok<127)?tok:'?', dt,
      100.0*mag.ff_zero/mag.ff_n, mag.x_absmax, mag.x_absmean/mag.x_n, mag.attn_absmax,
      lmax, lmin, mag.denorm, mag.nonfinite, cs/(SGAI_N_LAYERS*SGAI_N_HEADS));
  }
  return 0;
}
