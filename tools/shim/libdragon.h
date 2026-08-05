/* Minimal freestanding stand-in for <libdragon.h>, used ONLY by the bare-metal
 * bench harness.  nano_gpt.c needs almost nothing from libdragon: types, and
 * debugf().  Keeping this separate means nano_gpt.c is compiled UNMODIFIED
 * (except the documented BENCH_DET_PSE guard) for the bench. */
#ifndef BENCH_LIBDRAGON_SHIM_H
#define BENCH_LIBDRAGON_SHIM_H
#include <stdint.h>
#include <stddef.h>
void bench_log(const char *s);
#define debugf(...) ((void)0)
#endif
