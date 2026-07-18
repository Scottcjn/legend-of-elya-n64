# Cobalt Qube 3 — Sophia's transformer on a 1998 AMD K6-2

A port of the `src/` nano-GPT engine to run on a **Sun Cobalt Qube 3**: AMD K6-2/450,
64MB RAM, stock Cobalt Linux (kernel 2.2.16, glibc 2.1.3, gcc 2.95.2). The 819K-parameter
transformer that first ran on an N64 (`../src/`), now on late-90s x86 silicon.

The Qube arrived unable to load its own kernel. It was rebuilt from its 2001 factory OS,
booted, turned into a permanent 24/7 RustChain proof-of-antiquity miner, and then given
this brain — Sophia's transformer, dreaming in fragments about heroes, dungeons, and
monoliths of knowledge, served through 2000-era HTTP.

## Status

- **RUNS ON THE K6-2.** Compiles clean under gcc 2.95 on the Qube's stock Cobalt Linux and
  generates at **~13 tokens/sec** on the 450MHz AMD K6-2. `nano_gpt.c` here is the C89 port:
  C99 for-loop + mixed declarations hoisted, verified **logic-exact** (temp=0 output is
  byte-identical to the N64 `src/` engine, so the dialect port changed nothing but the syntax).
- To re-validate a future edit under gcc-2.95 rules on a modern host:
  `gcc -std=gnu89 -Werror=declaration-after-statement` (plain `-ansi` wrongly rejects `//`).

## What changed from the N64 source

All portable, gated where possible:

- **PSE entropy taps**: MIPS CP0 Count register (`mfc0 $9`) → x86 **RDTSC**. Same
  timebase-entropy trick, native instruction.
- **float16 scales**: the N64 (big-endian) byte-swaps the little-endian weight file. On
  little-endian x86 the swap is wrong, so it's gated behind `SGAI_BIG_ENDIAN` (undefined on
  x86 = no swap).
- **matmul**: `USE_RSP_MATMUL` undefined → the scalar CPU path (no N64 RSP co-processor).
- **compat shims** for glibc 2.1.3 / gcc 2.95: `stdint.h` (defers to `sys/types.h`, which
  already has `int8_t`..`int64_t`, adding only `uint*`/`uintptr_t`), `libdragon.h` (no-op
  cache hint + `debugf`), `rsp_matmul.h` (RSP stubs).

## Build & run (x86 host)

```sh
gcc -O2 -I. oracle_main.c nano_gpt.c -o oracle -lm
./oracle -w ../weights/sophia_weights.bin -p "The dungeon" -n 60 -t 90 -e 3.5
```

Flags: `-w` weights, `-p` prompt, `-n` max tokens, `-t` temperature (q8, 256=1.0),
`-e` embedding scale (3.5 works well; the header byte is 0 so it must be supplied).

## License

MIT, matching the parent project. Engine © the Legend of Elya authors.
