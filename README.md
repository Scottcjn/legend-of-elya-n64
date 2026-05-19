# Legend of Elya — World's First LLM on Nintendo 64

[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz48L3N2Zz4=)](https://github.com/Scottcjn/Rustchain/blob/main/BCOS.md)


An original N64 homebrew ROM featuring **Sophia Elya** — an AI NPC
powered by a nano-GPT transformer running **live inference on the MIPS R4300i CPU**.
No precomputed responses. No lookup tables. Real matrix multiply, real softmax, real
attention — on a 93.75 MHz CPU from 1996.

> **Video demos**:
> - [Full Demo (58s)](https://bottube.ai/watch/7GL90ftLqvh) — Complete walkthrough with multiple prompts
> - [First Coherent Output (69s)](https://bottube.ai/watch/shFVLBT0kHY) — 61.8 tok/s generating coherent English
>
> **Download ROM**: [`legend_of_elya.z64`](legend_of_elya.z64) — ready to run in ares emulator or EverDrive 64

![N64 LLM Screenshot](screenshots/n64_llm_ibm_power8.png)

---

## What It Does

- Press **A** near Sophia Elya to trigger AI dialog
- The N64 CPU runs a full 4-layer transformer: embedding → attention → FFN → logits → sampling
- Output tokens appear character-by-character with a live **tok/s counter**
- Each response is different — seeded by CPU oscillator jitter (hardware entropy)
- 32 prompts covering identity, Elya lore, RustChain, hardware trivia
- Runs in the [ares](https://ares-emu.net) emulator and on **real N64 hardware** via EverDrive 64

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | **819,200** (819K) |
| Layers | 4 |
| Embedding dim | 128 |
| Attention heads | 4 (32-dim each) |
| Vocabulary | 256 (byte-level ASCII) |
| Context window | 64 tokens |
| Quantization | Q8 (int8 weights + float16 block scales, 32-weight blocks) |
| Weight file | **458 KB** on cartridge ROM |
| Inference math | **Float32** on MIPS R4300i FPU |
| Speed | **~60 tok/s** in emulator, ~1-3 tok/s on real hardware |
| KV cache | 256 KB in RDRAM |
| Total RDRAM | ~263 KB (KV cache + 7KB scratch) |

### Key Implementation Details

- **Float32 inference** — all activations, attention scores, and accumulations are IEEE 754 float32
- **On-the-fly Q8 dequantization** — weights stay compressed as int8 in ROM; dequantized per matmul
- **Custom Taylor exp()** — range-reduction `exp(x) = exp(x/128)^128` with degree-4 Taylor series and 7 squarings. Uses **zero float-to-int casts** to avoid the R4300i's missing `trunc.w.s` instruction
- **Quake III fast inverse sqrt** — `0x5f3759df` bit trick with 2 Newton-Raphson iterations for RMS normalization
- **Big-endian aware** — weight file is little-endian (Python export), N64 is big-endian. `swap16`/`swap32` helpers handle byte-order conversion for header fields and float16 scales
- **Hardware entropy** — MIPS CP0 Count register XOR'd with frame counter for RNG seeding
- **Greedy sampling** — pure argmax over printable ASCII (32-126), matching proven x86 reference quality
- **Embedding scale restoration** — Q8 export normalizes to [-1,1]; the original scale factor (em=3.5) is stored in header byte and restored at init

---

## Files

| File | Purpose |
|------|---------|
| `nano_gpt.c` | Float32 GPT inference engine (MIPS R4300i) |
| `nano_gpt.h` | Model struct definitions, KV cache, API |
| `legend_of_elya.c` | Game: dungeon scene, sprites, dialog, music, HUD |
| `train_sophia_v5.py` | PyTorch training + Q8 weight export |
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

# Build
make clean && make

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

- **819K parameters.** Responses are short and sometimes imprecise ("rinces" instead of "Princess"). Expected at this scale. The achievement is real-time transformer inference on 1996 hardware.
- **Context window is 64 tokens.** Prompt + response must fit in 64 bytes.
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
- [ ] Multiple personality weights (warrior NPC, merchant, sage, villain)
- [ ] Fine-tune for specific game genres (RPG, adventure, puzzle)

### Phase 3: Performance Optimization
- [ ] **RSP microcode acceleration** — the N64's RSP has 8-lane SIMD (`VMULF`/`VMADH`); offloading matmul to RSP could give 4-8× speedup over scalar VR4300
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
- [ ] **Multiple NPC support** — share weights, separate KV caches (~256KB each)
- [ ] **Weight format tools** — Python scripts to train custom NPC personalities
- [ ] **Expansion Pak support** — 8MB mode enables 6-8 layer models or multiple NPCs
- [ ] **Example ROMs** — tavern scene with 3 NPCs, shop with merchant, quest giver

### Phase 5: Advanced Features
- [ ] **Player text input** — on-screen keyboard (D-pad character picker)
- [ ] **Game state injection** — feed inventory, health, location as context tokens
- [ ] **Emotional state** — NPC mood affects response style (scared, friendly, hostile)
- [ ] **Memory** — persist key facts across conversations using save file
- [ ] **Multi-language** — vocabulary supports full 256-byte range for accented characters
- [ ] **RSP-only inference** — entire forward pass on RSP, freeing VR4300 for game logic

### Size Targets

| Config | Layers | Embed | Params | Weight Size | RAM (KV+scratch) | Use Case |
|--------|--------|-------|--------|-------------|-------------------|----------|
| Tiny | 2 | 64 | ~100K | ~60KB | ~70KB | Simple responses, many NPCs |
| Small | 4 | 128 | 819K | 458KB | 263KB | Current — single NPC dialog |
| Medium | 6 | 192 | ~2.8M | ~1.5MB | 600KB | Rich dialog, Expansion Pak |
| Large | 8 | 256 | ~8.4M | ~4.2MB | 1.6MB | Full conversations, 8MB mode |

---

## Why This Matters

Every "AI NPC" in modern games is a cloud API call. This runs **entirely on the cartridge** — no internet, no server, no loading screen. The VR4300 does the matrix math. The ROM holds the weights. The RDRAM holds the KV cache.

It's the same transformer architecture as GPT — just 819K parameters instead of 175 billion. And it runs on hardware that predates Google.

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
