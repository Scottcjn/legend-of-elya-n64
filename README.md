# Legend of Elya — World's First LLM on Nintendo 64

[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz48L3N2Zz4=)](https://github.com/Scottcjn/Rustchain/blob/main/BCOS.md)


An original N64 homebrew ROM featuring **Sophia Elya** — an AI NPC
powered by a nano-GPT transformer running **live inference on the MIPS R4300i CPU**.
No precomputed responses. No lookup tables. Real matrix multiply, real softmax, real
attention — on a 93.75 MHz CPU from 1996.

> **Video demos**:
> - [Full Demo (58s)](https://bottube.ai/watch/7GL90ftLqvh) — Complete walkthrough with multiple prompts
> - [First Coherent Output (69s)](https://bottube.ai/watch/shFVLBT0kHY) — ~~61.8 tok/s~~ generating coherent English
>   ([**the 61.8 tok/s in this video is withdrawn** — the counter on screen was
>   broken and the real figure is 1.2 tok/s; see *The tok/s counter was wrong*
>   below](#the-toks-counter-was-wrong-2026-08-13))
>
> **Download ROM**: [`legend_of_elya.z64`](legend_of_elya.z64) — ready to run in ares emulator or EverDrive 64

![N64 LLM Screenshot](screenshots/n64_llm_ibm_power8.png)

---

## What It Does

- Press **A** near Sophia Elya to trigger AI dialog
- The N64 CPU runs a full 8-layer transformer: embedding → attention → FFN → logits → sampling
- Output tokens appear character-by-character with a live **tok/s counter**
- Each response is different — seeded by CPU oscillator jitter (hardware entropy)
- 32 prompts covering identity, Elya lore, RustChain, hardware trivia
- Runs in the [ares](https://ares-emu.net) emulator and on **real N64 hardware** via EverDrive 64

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | **6,356,992** (6.36M) |
| Layers | 8 |
| Embedding dim | 256 |
| Attention heads | 8 (32-dim each) |
| Vocabulary | 256 (byte-level ASCII) |
| Context window | 128 tokens |
| Quantization | **Ternary** (SEQ2, 2-bit weights + float16 block scales, 32-weight blocks) |
| Weight file | **1,984 KB** on cartridge ROM (`filesystem/sophia_weights.bin`, 2,031,628 B) |
| Inference math | **Float32** on MIPS R4300i FPU |
| Speed | **1.23 tok/s** scalar, **2.19 tok/s** RSP overlay — measured under ares, never on silicon ([how](#the-toks-counter-was-wrong-2026-08-13)) |
| KV cache | 256 KB in RDRAM |
| Total RDRAM | ~263 KB (KV cache + 7KB scratch) |

### Key Implementation Details

- **Float32 inference** — all activations, attention scores, and accumulations are IEEE 754 float32
- **On-the-fly dequantization** — weights stay compressed in ROM (2 bits each in the shipped
  ternary blob, 8 in an int8 blob); dequantized per matmul
- **Custom Taylor exp()** — range-reduction `exp(x) = exp(x/128)^128` with degree-4 Taylor series and 7 squarings. Uses **zero float-to-int casts** to avoid the R4300i's missing `trunc.w.s` instruction
- **Quake III fast inverse sqrt** — `0x5f3759df` bit trick with 2 Newton-Raphson iterations for RMS normalization
- **Big-endian aware** — weight file is little-endian (Python export), N64 is big-endian. `swap16`/`swap32` helpers handle byte-order conversion for header fields and float16 scales
- **Hardware entropy** — MIPS CP0 Count register XOR'd with frame counter for RNG seeding
- **Greedy sampling** — pure argmax over printable ASCII (32-126), matching proven x86 reference quality
- **Embedding scale restoration** — Q8 export normalizes to [-1,1]; the original scale factor (em=3.5) is stored in header byte and restored at init

---

## The tok/s counter was wrong (2026-08-13)

**This project published ~60 tok/s. The real figure is 1.23 tok/s. The counter
that produced the 60 was not measuring anything.**

The number reached four places — this README, the video description, the
elyanlabs.ai homepage, and the launch article — before anyone audited it. It has
been pulled from the homepage and the article. This section is the correction,
and it stays visible.

### What the old counter computed

From `src/legend_of_elya.c` at `bf97959` (2026-02-22), the build whose on-screen
readout was filmed:

```c
int elapsed = G.frame - G.gen_start_frame;
if (elapsed > 0)
    G.gen_toks_sec = (float)G.gen_out_count * 60.0f / (float)elapsed;
```

`G.frame` counts **game-loop iterations**, and this loop generates **exactly one
token per iteration** (`update_generating_step()` is called once per pass through
`main()`'s `while(1)`, and makes exactly one `sgai_next_token()` call). So
`gen_out_count` and `elapsed` both advance by exactly 1 every time that line runs.
Their ratio is pinned at ~1 by construction, and the readout is pinned at the
literal `60.0f` in the source.

Nothing on the right-hand side depends on how long anything took. **It reports ~60
on any machine at any speed** — it is a constant wearing the costume of a
measurement. The `60.0f` encodes an assumption that one loop iteration takes 1/60 s.
It takes 0.81 s.

The published **61.8** is the same expression's off-by-one: the first-token fix
sets `gen_out_count = 1` in the same iteration that zeroes `gen_start_frame`, so
the readout is `60·(k+1)/k`, converging on 60 from above. At `k = 33`,
`60 × 34 / 33 = 61.818` → displayed `61.8`. That is the number in the video
description, to the digit.

This is not a rounding error or a clock problem. **The old counter never measured
elapsed time at all.**

### Reproduced, not argued

The old expression is evaluated alongside the new one, on the same tokens, in the
same run, under the same emulator (`make base EXTRA="-DRATE_PROBE -DRATE_INGAME"`):

```
RATE HUD n=1  vbl=47  toks_vbl=1.27  toks_cp0=1.25  toks_legacyframe=60.00
RATE HUD n=9  vbl=434 toks_vbl=1.24  toks_cp0=1.24  toks_legacyframe=60.00
RATE HUD n=19 vbl=930 toks_vbl=1.22  toks_cp0=1.22  toks_legacyframe=60.00
```

`toks_legacyframe` is `60.00` on every token of every run. It is a constant.

### Was the emulator lying about the clock? No.

The suspicion going in was that CP0 COUNT — the CPU cycle counter the counter's
`CYCLES_TO_US()` reads — does not advance against real time under emulation, which
would make the emulator the culprit. **It does not hold up. ares advances CP0 COUNT
correctly**, and that matters more than the counter bug, because it means every
CP0-derived measurement in `docs/` stands.

The experiment: burn a known number of CP0 counts in a spin loop, count the VI
vblanks that elapse. Vblank is 59.94 Hz by video standard, not by opinion about CPU
speed.

| target | CP0 counts | → seconds | vblanks | → seconds | disagreement |
|--------|-----------:|----------:|--------:|----------:|-------------:|
| 1 s    |  46,875,000 |  1.0000  |      60 |   1.0010  | 0.10 % |
| 5 s    | 234,375,000 |  5.0000  |     299 |   4.9883  | 0.23 % |
| 5 s (RSP build) | 234,375,001 | 5.0000 | 299 | 4.9883 | 0.23 % |

The two clocks inside the emulated console agree to better than a quarter of a
percent. The residual is the 60 Hz-vs-59.94 Hz rounding in the spin target, not
drift.

A third clock settles what the first two cannot, since both come from inside the
emulated machine: the **host** clock, from `scripts/ares_rate_run.py`. 5.000
emulated seconds took **9.334 host seconds** — headless ares runs this workload at
about **0.54x real time**. That is the emulator being slow, which is honest and
harmless: emulated time is faithful, so a rate derived from it is a rate the
console would show. It is *host* wall clock that must not be quoted.

### Honest figures

Shipped configuration — SEQ2 ternary blob (2,031,628 B), `ctx=128`, `SGAI_KV_INT8`,
`PSE_FIX=1`, `REP_FIX=1`, output at `temperature_q8 = 64`, 16 generated tokens,
ares 147 headless. Both clocks quoted; both arms emit byte-identical text.

| build | CP0 counts | vblanks | tok/s (CP0) | **tok/s (vblank)** | s/token |
|-------|-----------:|--------:|------------:|-------------------:|--------:|
| scalar CPU (`make base`) | 609,575,279 | 778 | 1.230 | **1.23** | 0.81 |
| RSP overlay (`make base-rsp-ovl`) | 342,795,454 | 438 | 2.188 | **2.19** | 0.46 |

Those are pure inference, headless. Driving the **real vsync-locked game loop**
(`-DRATE_INGAME`, so the RDP renders a frame between tokens as it does for a
player) costs a little more, and this is what is actually on screen:

| build | in-game tok/s | vs headless |
|-------|--------------:|------------:|
| scalar CPU | **1.22** | −0.8 % |
| RSP overlay | **2.16** | −1.4 % |

Both builds emit the same 19 characters, `Call me Sophia Elya`, in every arm above —
the instrumentation changed no token.

**RSP overlay speedup over scalar: 1.78x** (1.776x on vblanks, 1.778x on CP0), with
`rsp=1968 cpu=0` — every matmul really did go to the vector unit.

Two things that number is not:

- It is **not the 4.769x** quoted elsewhere in this repo. That figure is a ratio of
  *CP0 cycle counts*, not tok/s, measured on the **int8** blob, which is not what
  ships. On the shipped ternary blob the RSP wins 1.78x, because ternary already
  cut the scalar path's work by more than half.
- It is **not a real-hardware number**. See *What has not been checked* below.

Prompt ingestion is quoted separately and is not folded into tok/s: the game's
25-token persona-prefixed prompt costs **~20 s** (1,222 vblanks) on the scalar build
before the first output token exists.

### What the new counter does

It counts **VI vblanks**. Vblank is 59.94 Hz on NTSC hardware because that is the
rate the console shoves fields at a television — a property of the video standard,
not of anyone's belief about CPU speed. The Sega Genesis port of this engine already
counts vblanks for exactly this reason.

A consequence worth stating, because it is the tell that the number came off a
clock: each token takes 48–50 vblanks, so the reported rate is **quantised to
59.94/k** — 1.25, 1.22, 1.20 and nothing in between. Arithmetic that is not
measuring time does not quantise like that.

CP0 COUNT is still computed, into `perf_toks_cp0`, and never displayed. It is kept
so the two clocks can be compared on any target — including silicon — without a
rebuild. On the two builds above they agree to 0.19 % and 0.08 %.

The denominator was also fixed to cover the output phase only. It previously ran
from the start of prompt ingestion while the numerator counted output tokens only,
which made the readout a cumulative average that crept upward and never arrived:
**0.54 tok/s displayed against 1.23 actual**. Under-reporting by 2.3x is still
misreporting.

### What has not been checked

- **Nothing here has been run on real N64 silicon.** Every number in this section is
  ares 147. ares's cycle model is the reason to trust it — independently calibrated
  in `docs/EMULATOR_ACCURACY.md` at 1.00 cycle per instruction, 40.25 cycles per
  D-cache miss and 268.62 cycles per uncached cartridge read, all of which are the
  documented real R4300i figures — but a cycle model is not a console. An EverDrive
  64 run is the outstanding check, and the new counter is built to survive it: on
  silicon the vblank and CP0 numbers must agree, and if they do not, the ROM prints
  both.
- The RSP overlay has never been run on silicon either, and ares emits a
  cache-coherency warning at the first overlay switch (`FINDINGS.md` F-O004) that
  could not be confirmed benign off-hardware.
- **Reproduce it**: `make base EXTRA="-DRATE_PROBE"` then
  `python3 scripts/ares_rate_run.py <rom.z64> <log>`. Add `-DRATE_INGAME` to drive
  the real vsync-locked game loop instead of the headless probe.

---

## Acceleration: RSP Build and the RPI Engine

The scalar 1.23 tok/s figure above is the **scalar CPU baseline** — the plain
`legend_of_elya.z64` build runs everything on the VR4300. Two faster paths ship in this repo:

### RSP-accelerated build (`make base-rsp`)

`rsp_matmul.S` is hand-written RSP microcode that runs the matmul on the N64's
8-lane vector unit: the CPU driver (`matmul_rsp_drv.c`) converts float32 activations
to int16 fixed-point, DMAs tiles into the RSP's 4KB DMEM, and dispatches 8 output
rows per call. `nano_gpt.c` uses it when built with `USE_RSP_MATMUL`
(the `legend_of_elya_rsp.z64` target). **Measured 1.78× over the scalar baseline**
on the shipped ternary blob under ares (2.19 vs 1.23 tok/s). The "projected 4-8×"
that stood here was never a tok/s measurement: the 4.769× it came from is a ratio of
CP0 cycle counts on the **int8** blob, which is not what ships. Still pending on real
hardware — benchmark reports welcome.

### RPI engine (`rpi/`) — zero-multiply inference

RPI (Resonant Permutation Inference) is a second, separate engine: instead of
matrix multiply, Sophia speaks through **permutation tables** (bigram/trigram
counts distilled from a larger teacher model, xorshift32 PRNG, zero multiplies,
zero FPU). The 868KB `sophia_game_n64.rpi` model fits in ROM alongside the game.
Roughly ~100× faster than the transformer on the R4300i (**estimated, not measured** —
no RPI build has been through the vblank counter) —
a different quality/speed trade, not a transformer.

---

## Files

| File | Purpose |
|------|---------|
| `nano_gpt.c` | Float32 GPT inference engine (MIPS R4300i) |
| `nano_gpt.h` | Model struct definitions, KV cache, API |
| `rsp_matmul.S` / `matmul_rsp_drv.c` | RSP vector-unit matmul microcode + CPU driver (`make base-rsp`) |
| `rpi/` | RPI zero-multiply permutation inference engine + 868KB game model |
| `multi_npc.c` | Expansion Pak multi-NPC mode (3 AI characters) |
| `legend_of_elya.c` | Game: dungeon scene, sprites, dialog, music, HUD, D-pad keyboard |
| `train_sophia_v5.py` | PyTorch training + Q8 weight export |
| `train_sophia_v8.py` | v8 training pipeline — 4 NPC personality packs |
| `reference_cli.c` | Host-side reference CLI (`make reference`) for x86 parity testing |
| `weights/sophia_weights.bin` | Pre-trained v5 weights (458KB, ready to use) |
| `Makefile` | libdragon build system |
| `src/` | Latest source snapshots |
| `screenshots/` | Working N64 LLM screenshots |
| `mining/` | **Optional** RustChain mining attestation module |

---

## Quick Start

### Option 1: Use Pre-built ROM

Download `legend_of_elya.z64` from [Releases](../../releases) and load in ares emulator or copy to EverDrive SD card.

### Option 2: Build from Source

Requires [libdragon](https://github.com/DragonMinded/libdragon) toolchain:

```bash
# Set toolchain path
export N64_INST=/path/to/mips64-toolchain

# Place weights in filesystem/
cp weights/sophia_weights.bin filesystem/

# Build (base ROM)
make clean && make base

# Or build the RSP-accelerated variant
make base-rsp   # -> legend_of_elya_rsp.z64

# Run in ares
ares legend_of_elya.z64
```

### Option 3: Train Your Own Model

```bash
# Requires PyTorch + CUDA GPU
python3 train_sophia_v5.py
# ~20 min on RTX 5070, exports filesystem/sophia_weights.bin
```

---

## Pre-trained Weights

The `weights/sophia_weights.bin` file contains a pre-trained v5 model (819K params, Q8 format, 458KB).
**This is not the model the ROM ships.** The shipped blob is
`filesystem/sophia_weights.bin` — a SEQ2 ternary 6.36M-parameter model, 2,031,628 B,
retrained 2026-08-06. The v5 file is kept for reference and for the small-config arm
of the scaling table below.

Training corpus covers: Sophia Elya identity, RustChain blockchain, Elya lore, N64 hardware, PowerPC architecture, dungeon/RPG dialog.

**Weight file format:**

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `0x53454149` ("SEAI"), little-endian |
| 4 | 1 | n_layers (4) |
| 5 | 2 | n_embed (128) |
| 7 | 1 | n_heads (4) |
| 8 | 2 | vocab_size (256) |
| 10 | 1 | ctx_len (64) |
| 11 | 1 | em_scale_x16 (56 = 3.5 × 16) |
| 12 | 32768 | Embedding table (256 × 128, int8) |
| 32780 | ... | Layer weights (int8) + scales (float16) × 4 layers |

---

## Honest Limitations

- **6.36M parameters, ternary weights.** Responses are short and sometimes imprecise. Expected at this scale and quantization. The achievement is real-time transformer inference on 1996 hardware.
- **1.23 tok/s scalar, 2.19 tok/s on the RSP overlay** — a sentence takes the better part of a minute, and the 25-token prompt costs ~20 s before the first character appears. Measured under ares; see *The tok/s counter was wrong* above.
- **Context window is 128 tokens.** Prompt + response must fit in 128 bytes.
- **No memory between dialogs.** KV cache resets each conversation.
- **Byte-level vocabulary.** One ASCII character per token — no subword tokenization.
- **Training corpus is small.** More data and epochs will improve coherence.

---

## Roadmap: N64 LLM SDK

The goal is to shrink, optimize, and package this into a **reusable SDK** that any N64 homebrew developer can drop into their game to give NPCs real language understanding.

### Phase 1: Core Engine (DONE)
- [x] Float32 transformer inference on MIPS R4300i
- [x] Q8 quantized weights with on-the-fly dequantization
- [x] Custom math (Taylor exp, fast inverse sqrt) avoiding missing R4300i instructions
- [x] Big-endian weight loading from ROM filesystem
- [x] Hardware entropy from CPU oscillator
- [x] Working demo ROM with dialog system

### Phase 2: Model Quality (IN PROGRESS)
- [ ] Extended training corpus (500+ QA pairs across game domains)
- [ ] Longer training runs (200K+ steps) for better convergence
- [ ] Context-aware prompting (NPC name, location, game state as prefix tokens)
- [x] Multiple personality weights — Personality Pack Trainer, 4 distinct NPC personas (`train_sophia_v8.py`)
- [ ] Fine-tune for specific game genres (RPG, adventure, puzzle)

### Phase 3: Performance Optimization
- [x] **RSP microcode acceleration** — implemented (`rsp_matmul.S` + `matmul_rsp_drv.c`, `make base-rsp`): 8-lane int16 matmul on the RSP, 8 output rows per dispatch. Measured 1.78× over scalar VR4300 on the shipped ternary blob under ares (2.19 vs 1.23 tok/s); on-hardware tok/s still pending
- [ ] **Q4 quantization** — halve weight size to ~230KB, fit more model or more NPCs
- [ ] **Tiled matmul** — process weights in cache-friendly blocks to reduce RDRAM stalls
- [ ] **Speculative generation** — pre-generate during idle frames (exploration, cutscenes)
- [ ] **KV cache sharing** — multiple NPCs sharing embedding + early layers, diverging at output

### Phase 4: SDK Release
- [ ] **`n64_llm.h` / `n64_llm.c`** — single-file drop-in library
- [ ] **Simple API**:
  ```c
  // Init with weight data from ROM
  N64LLM_State *npc = n64llm_init(rom_weights, weight_size);

  // Set NPC personality context
  n64llm_set_context(npc, "You are a blacksmith in the Crystal Caverns.");

  // Generate response to player input
  char response[128];
  n64llm_generate(npc, "Do you sell shields?", response, sizeof(response));

  // Per-frame generation (non-blocking, 1 token per frame)
  int done = n64llm_step(npc);
  ```
- [x] **Multiple NPC support** — Expansion Pak multi-NPC mode, 3 AI characters (`multi_npc.c`)
- [x] **Weight format tools** — Python training pipeline for custom NPC personalities (`train_sophia_v8.py`)
- [x] **Expansion Pak support** — 8MB mode working: 6.36M-param Large model with two-phase training, multi-NPC mode
- [ ] **Example ROMs** — tavern scene with 3 NPCs, shop with merchant, quest giver

### Phase 5: Advanced Features
- [x] **Player text input** — on-screen D-pad keyboard
- [ ] **Game state injection** — feed inventory, health, location as context tokens
- [ ] **Emotional state** — NPC mood affects response style (scared, friendly, hostile)
- [ ] **Memory** — persist key facts across conversations using save file
- [ ] **Multi-language** — vocabulary supports full 256-byte range for accented characters
- [ ] **RSP-only inference** — entire forward pass on RSP, freeing VR4300 for game logic

### Size Targets

| Config | Layers | Embed | Params | Weight Size | RAM (KV+scratch) | Use Case |
|--------|--------|-------|--------|-------------|-------------------|----------|
| Tiny | 2 | 64 | ~100K | ~60KB | ~70KB | Simple responses, many NPCs |
| Small | 4 | 128 | 819K | 458KB | 263KB | Legacy v5 (`weights/sophia_weights.bin`) |
| Medium | 6 | 192 | ~2.8M | ~1.5MB | 600KB | Rich dialog, Expansion Pak |
| Large | 8 | 256 | 6.36M | 1,984KB ternary | 576KB int8 KV | **Current** — Expansion Pak, two-phase training |

---

## Why This Matters

Every "AI NPC" in modern games is a cloud API call. This runs **entirely on the cartridge** — no internet, no server, no loading screen. The VR4300 does the matrix math. The ROM holds the weights. The RDRAM holds the KV cache.

It's the same transformer architecture as GPT — just 6.36M parameters instead of 175 billion. And it runs on hardware that predates Google.

If we can make a transformer talk on 8MB of RAM and a 93MHz MIPS CPU, the excuses for cloud-dependent "AI" in games evaporate.

---

## Screenshots

| IBM POWER8 Response | Elya Crystal Response |
|---------------------|----------------------|
| ![](screenshots/n64_llm_ibm_power8.png) | ![](screenshots/n64_llm_zelda_triforce.png) |

---

## Optional: RustChain Mining Module

The `mining/` directory contains an optional **proof-of-antiquity** mining module that lets a real N64 earn RTC (RustChain Token) rewards by submitting hardware attestations to the [RustChain](https://rustchain.org) blockchain.

**How it works:**
- N64 runs 5 hardware fingerprint checks (CPU PRId, COUNT timing, VI scan, memory ratio, anti-emulation)
- Results are written to controller pak via joybus → Raspberry Pi Pico relays over USB → Python host bridge submits to RustChain node
- Real chain data (epoch, slot, balance, miner count) flows back: RustChain API → Python → USB → Pico → pak READ → N64 display
- N64 gets a **3.0x antiquity multiplier** as vintage hardware (1996 silicon)
- Wallet is hardware-derived from RDRAM config registers + CP0 PRId — unique per console

**Requirements:** N64 + EverDrive 64 + Raspberry Pi Pico + USB cable

See [`mining/README.md`](mining/README.md) for full setup instructions.

---

## Credits

Built by [Elyan Labs](https://rustchain.org).

- **Engine**: nano-GPT float32 inference on MIPS R4300i
- **Game**: libdragon SDK, pixel art, dungeon adventure
- **Training**: PyTorch on RTX 5070
- **Platform**: [BoTTube](https://bottube.ai) for video hosting

Source is open — build it, train it, improve it, port it.
