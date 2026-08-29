# The frame and memory budget — Legend of Elya, MoE lane

**Summary.** At 5.87 tok/s a generated token costs **7.95 M CP0 cycles (169.6 ms,
10.17 frames)**, so the design unit is the token, not the frame: the game gets
ten 60 Hz frames per character it speaks, and the *entire* rest of the game —
RDP, logic, audio — measures **≤1.5 % of that** (F-RT005/006, in-game vs
headless). Inference owns ~98.5 % of the CPU. The costs no frame budget contains
are per-turn: a **7.53 M-cycle (161 ms, 9.6 frame) expert switch** and a **1 MB
expert DMA measured cold at 401 ms**. RDRAM is not the constraint — ~5.34 MB of
8 MB is free. **I ran nothing new for this document**; every figure re-derives
committed logs (`probe/moe_probe_2026-08-28.log`, `probe/ec_probe_2026-08-28.log`,
`probe/rate_epi_overlap_*`, `docs/rate_logs/`), and one re-derivation contradicts
two published findings — see the last section.

## Clock

CP0 counts at **46,875,000 Hz** (F-RT010: 1 s = 46,875,001 counts). One 59.94 Hz
frame = **782,032 counts**. Cross-check: `moe_probe` prompt 0 spends 244 vblanks
on 24 tokens = 190.8 M cycles, which is 190.8 M of its printed 295.4 M `gen`
window (the rest is the 14-byte prompt). Self-consistent.

## Per frame (782,032 cycles)

| item | cycles/frame | source |
|---|---:|---|
| RSP matvec (chunked, 16 chunks) + CPU epilogue, overlapped | ~770,000 | **measured**, F-R028 rate ÷ frame |
| RDP drawing + game logic + audio, all of it | **≤11,700** | **measured envelope**: in-game costs 1.1–1.5 % more than headless (F-RT005/006) |
| vblank ISR, vbl counter | not separated | — |

A token spans 10.17 frames, so there is no per-frame split of the inference
work. The split *inside a token* is measured only for the **dense 8L** model
(F-R026, overlap ON): stage 3.5 %, dispatch 2.5 %, epilogue 70.6 %, blocked wait
23.4 %. **Not measured for the 4L experts.**

## Per turn (one NPC line)

| item | cost | frames | measured? |
|---|---:|---:|---|
| route (`moe_route`, substring) | <10 k cycles (est.) | <0.02 | **estimate**, never instrumented |
| expert acquire, steady state | **~100 cycles** | ~0 | measured, F-R028 |
| expert acquire, cold (1 MB PI DMA) | **18,806,947 = 401 ms** | 24.0 | measured, F-R028 prompt 0 |
| expert **switch** (`sgai_init_ex` RSP lane permute) | **7,533,473 = 161 ms** | **9.6** | measured, F-R028, all 6 prompts within 0.3 % |
| prompt ingestion, per byte | **7,469,764 = 159 ms** | 9.6 | derived from the same log (295.4 M − 190.8 M) ÷ 14 |
| generation, per token | **7,950,659 = 170 ms** | 10.17 | measured, F-R028 |
| a median answer (27 chars, C028 newline-EOS) | **4.6 s** | 275 | derived |
| a 14-byte prompt + median answer, cold expert | **7.4 s** | 442 | derived |

## Per boot

`dfs_init`/`display_init`/`rdpq_init`/`ec_init`: **not measured**. The only
bounded component is the first acquire+swap, 561 ms.

## RDRAM (8,388,608 B)

| region | bytes | source |
|---|---:|---|
| 2 expert slots @ 1,048,592 | 2,097,184 | measured (`moe_probe.c`, bss) |
| framebuffer, 320×240×16bpp ×2 | 307,200 | measured from the `display_init` call |
| KV cache, 4L/256d int8, ctx 128 | 294,920 | formula in `nano_gpt.h`, validated against ares' printed 589,832 for 8L |
| game text+data (base link, T9) | 182,788 | measured |
| other .bss (base) | 19,844 | measured |
| `SGAIScratch` | 12,288 | measured (struct) |
| heap+stack reserve | 131,072 | **judgement**, from `tools/rdram_budget.py` |
| audio buffers (22.05 kHz ×4) | ~36,000 | **estimate**, not measured |
| rdpq command buffers | ~64,000 | **estimate**, not measured |
| **free** | **~5,343,000** | |

Cartridge: the bank is 5,243,072 B of the 64 MB EverDrive X7 image; the SD path
(libcart/FatFs, `sd:/`) is untested here.

## What breaks first

1. **Prefetch lead vs 1 MB DMA.** A 1 MB expert takes 401 ms cold — **2.4
   tokens** of lead. `moe_probe` hid it only because its prefetch started a whole
   4 s turn early. **Watch: `EC.misses` > 0 in-game, or any `acquire` > 100 k
   cycles.** Under two tokens of lead, the player eats a visible hitch.
2. **The 161 ms switch, once per topic change.** 9.6 dropped frames, unhidable in
   a vsync loop. **Watch: switches per minute × 161 ms.** At >1 switch per line
   the game stutters on every line; the fix is chunking `sgai_init_ex` across
   ≥10 frames, which nothing has been written for.
3. **CPU headroom, which is ~1.5 %.** The whole non-inference game measures
   ≤11,700 cycles/frame today with a near-empty scene. **Watch: the in-game vs
   headless delta.** Past ~10 % the rate falls under 5.3 tok/s and a median reply
   crosses 5 s.

## What I did NOT verify

- Nothing was built, booted or run for this document; all figures re-derive
  committed logs, and no number anywhere came from silicon (ares 147 only).
- **Contradiction found, unresolved:** F-R027 calls 1,480,530 cycles "~15.8 ms"
  and F-R028 calls 7,533,473 "~80 ms". Both imply a 93.75 MHz CP0. F-RT010 and
  the vblank cross-check above both give 46.875 MHz, i.e. **161 ms and 31.6 ms —
  2× worse than published**. One of these is wrong and I did not settle it on
  hardware. Consequently the PI rate is 5.19 MB/s (160 KB), not the "~10 MB/s"
  in F-R027 — and the cold 1 MB acquire implies only 2.61 MB/s, a 2× gap from
  the 160 KB figure that nothing here explains.
- Audio buffer bytes, rdpq buffer bytes, `moe_route` cost, boot time, and the
  per-phase split for 4-layer experts: all estimates or unmeasured.
- The in-game (`-DRATE_INGAME`) figure has never been re-measured with the
  overlap on, nor at all for the MoE build; the ≤1.5 % RDP envelope is carried
  over from the dense 8L runs.
