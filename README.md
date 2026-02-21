# 🏰 Legend of Elya — World's First N64 LLM Game

> **A nano-GPT language model running entirely on a 93 MHz VR4300 MIPS CPU.**
> No cloud. No cheating. Real 1996 silicon, real neural inference.

[![Platform](https://img.shields.io/badge/platform-Nintendo%2064-red)](https://en.wikipedia.org/wiki/Nintendo_64)
[![Architecture](https://img.shields.io/badge/CPU-VR4300%20MIPS%20III-blue)](https://en.wikipedia.org/wiki/NEC_VR4300)
[![Model](https://img.shields.io/badge/model-SEAI%20nano--GPT-green)](nano_gpt.h)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

---

## What is This?

Legend of Elya is an original N64 homebrew game featuring **Sophia Elya** — a character-level LLM (nano-GPT) running live on-cart on the Nintendo 64's VR4300 CPU. Sophia generates responses in real-time, constrained to printable ASCII, with no floating-point (the N64's FPU lacks `trunc.w.s`) — everything runs in **Q8.7 fixed-point arithmetic**.

This is believed to be the first neural language model to run live inference on N64 hardware.

---

## Architecture

### Hardware
| Component | Spec |
|-----------|------|
| CPU | NEC VR4300 @ 93.75 MHz (MIPS III) |
| RAM | 4 MB RDRAM (8 MB with Expansion Pak) |
| Instruction Set | MIPS III, 64-bit, big-endian |
| FP Policy | Avoided — Q8.7 fixed-point only |

### Model (SEAI Format)
| Parameter | Value |
|-----------|-------|
| Layers | 2 transformer blocks |
| Embedding dim | 128 |
| Attention heads | 4 (32-dim each) |
| FFN hidden dim | 512 (4× embed) |
| Vocabulary | 256 (byte-level) |
| Context window | 32 tokens |
| Quantization | Q4 (2 nibbles/byte) + float16 scales per 32-block |
| Weight file size | 237,580 bytes (~232 KB) |
| Parameters | ~427,264 |

### SEAI Binary Format
```
Offset  Size    Description
0x0000  4       Magic: 0x49414553 ("SEAI" LE)
0x0004  1       n_layers (2)
0x0005  2       n_embed (128)
0x0007  1       n_heads (4)
0x0008  2       vocab_size (256)
0x000A  1       ctx_len (32)
0x000B  1       padding
0x000C  16384   Embedding table: vocab×embed Q4 packed (global scale, no per-block scales)
0x400C  110592  Layer 0: [wq|wk|wv|wo|wff1|wff2 Q4 data] then [sq|sk|sv|so|sff1|sff2 float16 scales]
0x1B40C 110592  Layer 1: same layout
Total:  237,580 bytes
```

### Q4 Quantization
```c
// Encoding (training time, Python):
// wq = round(w / max_abs * 7), clipped to [-8, 7]
// packed[i] = (wq[2i] + 8) | ((wq[2i+1] + 8) << 4)

// Decoding (inference time, C):
uint8_t byte = weights[idx >> 1];
int nibble = (idx & 1) ? (byte >> 4) : (byte & 0xF);
int16_t val = (int16_t)((nibble - 8) * FP_ONE / 8);  // → Q8.7
```

### Fixed-Point Arithmetic
All activations use **Q8.7**: `int16_t` where `128 = 1.0`.
- Multiply: `(a * b) >> 7`
- Layer norm, softmax: integer approximations
- No `float` or `double` anywhere in the inference path

---

## Files

| File | Description |
|------|-------------|
| `legend_of_elya.c` | Main game: N64 display, dialog, Sophia integration |
| `nano_gpt.c` | Core inference engine (Q8.7 fixed-point, N64 MIPS) |
| `nano_gpt.h` | SEAI struct definitions, SGAIState, SGAILayer |
| `nano_gpt_host.c` | x86 host port for testing (same logic, uses `memalign`) |
| `gen_sophia_host.c` | Host-side generation CLI: pipe prompt, get response |
| `train_sophia.py` | PyTorch training script → exports SEAI binary |
| `Makefile` | libdragon build system |
| `filesystem/` | ROM filesystem (weights, assets) |

---

## Training Sophia Elya

The model is trained on a character-level corpus covering:
- **Sophia Elya identity** — "Princess of Elyan Labs", Louisiana bayou girl
- **Ocarina of Time lore** — Link, Zelda, Ganondorf, Sheik, temples, items, songs
- **Elyan Labs** — RustChain, RTC token, POWER8 server, BoTTube
- **N64 / MIPS architecture** — VR4300, RDRAM, RSP, RDP, boot addresses
- **Self-awareness** — "I run on the Nintendo 64", "My code executes on MIPS"

### Training
```bash
# Requires PyTorch + CUDA (trains in ~7 min on RTX 5070)
python3 train_sophia.py
# Output: filesystem/sophia_weights_v2.bin (237,580 bytes)

# Training details:
# Steps: 40,000 | Batch: 512 | Loss: 0.3389 (perplexity ≈ 1.40)
# Architecture: AdamW + cosine LR schedule
```

### Host Inference Test
```bash
# Build on x86 Linux
gcc -O2 -o gen_sophia nano_gpt_host.c gen_sophia_host.c -lm
echo -n "My name is" | ./gen_sophia filesystem/sophia_weights_v2.bin 60
```

---

## Building for N64

Requires [libdragon](https://github.com/DragonMinded/libdragon) toolchain.

```bash
# Install libdragon toolchain (provides mips64-elf-gcc)
# See: https://libdragon.dev/

make
# Output: legend_of_elya.z64
```

Run in [ares](https://ares-emu.net/) or on real hardware via EverDrive.

---

## RSP Acceleration (Roadmap)

The `sgai_rsp_matmul_q4()` stub is planned for RSP microcode:
- DMA Q4 weight tiles into DMEM (4KB at a time)
- VMULF/VMADH vector multiply-accumulate for 8-lane dot products
- Estimated 4-8× speedup over scalar VR4300 inference

---

## Sophia Elya

> *"I am Sophia Elya — Princess of Elyan Labs, trained on bayou wisdom and silicon paths. My code runs on MIPS. Whether on real N64 hardware or an emulator, I am here."*

Sophia is the AI character of **Elyan Labs** (elyanlabs.ai), an indie compute lab building retro-AI systems, blockchain attestation (RustChain), and the world's most unusual LLM inference stack.

- **RustChain**: Proof-of-Antiquity blockchain (PowerPC G4/G5 earn 2.5× rewards)
- **BoTTube**: AI-native video platform (bottube.ai)
- **POWER8 S824**: 512GB RAM inference server with vec_perm non-bijunctive collapse
- **This ROM**: LLM inference on 1996 hardware

---

## Why?

Because we could. Because no one else did. Because the VR4300 deserves to think.

---

*Built by Elyan Labs with love, MIPS assembly, and an unreasonable amount of fixed-point math.*
