# Legend of Elya — World's First LLM on N64 Hardware

A Legend of Zelda-inspired N64 homebrew ROM featuring **Sophia Elya** — an AI character
powered by a nano-GPT transformer running live on the VR4300 CPU. No precomputed responses.
No lookup tables. Real matrix multiply, real softmax, real token sampling — on a 1996 CPU.

> **Status: Tech demo.** Inference is real and working. The model is small by design —
> it fits in 4MB RAM and runs at ~1–3 tok/s on hardware from 1996. That's the point.

---

## What It Does Right Now

- Press **A** near Sophia Elya to trigger dialog
- The N64 CPU feeds your prompt token-by-token through a 4-layer transformer
- Output tokens are sampled live and printed to screen as they generate
- Each response is different — seeded by CPU oscillator jitter (hardware entropy)
- Runs in the [ares](https://ares-emu.net) emulator and on real N64 hardware

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Embedding dim | 128 |
| Attention heads | 4 (32-dim each) |
| Vocabulary | 256 (byte-level) |
| Context window | 64 tokens |
| Quantization | Q8 (int8 weights, float16 block scales) |
| Weight size | ~868 KB (packed in ROM filesystem) |
| Inference math | Q8.7 fixed-point — no FPU required |
| Speed | ~1–3 tok/s on real N64 hardware |

**Key implementation decisions:**

- **No floating point** — the VR4300's FPU can't handle float16; all inference is Q8.7 fixed-point
- **Soft-float for weight decode only** — `nano_gpt.o` compiled with `-msoft-float` for float16 scale decode
- **Penalty history** — last 16 output tokens hard-excluded to prevent repetition loops
- **Period-stop** — generation ends at first `.` after 8+ characters (clean single-sentence answers)
- **Hardware entropy** — CPU cycle counter XOR'd with frame count seeds prompt selection

---

## Files

| File | Purpose |
|------|---------|
| `legend_of_elya.c` | Game logic — states, rendering, input, dialog trigger |
| `nano_gpt.c` / `nano_gpt.h` | Fixed-point GPT inference (VR4300 MIPS) |
| `train_sophia_v5.py` | PyTorch training — produces Q8 weight binary |
| `Makefile` | libdragon build |

---

## Build

Requires [libdragon](https://github.com/DragonMinded/libdragon) toolchain.
The `Makefile` defaults `N64_INST` to the toolchain install path — override if needed:

```bash
N64_INST=/path/to/mips64-toolchain make
```

---

## Training

```bash
python3 train_sophia_v5.py
# Trains 100K steps on CUDA GPU (~20 min on RTX 5070)
# Exports: filesystem/sophia_weights.bin (~868KB, Q8 format)
```

Before building, verify the weights exported correctly:

```bash
wc -c filesystem/sophia_weights.bin
# Expected: 868,364 bytes for 4-layer Q8
```

---

## Honest Limitations

- **819K parameters.** Responses are short and sometimes odd. That's expected at this
  scale with a small training corpus. The achievement is that it runs at all on this hardware.
- **Context window is 64 tokens.** Prompt + response must fit in 64 bytes.
- **No memory between dialogs.** The KV cache resets each conversation.
- **Byte-level vocabulary.** The model generates one ASCII character at a time.

---

## Future Directions

These are things we're working toward — not current functionality:

- **RSP microcode acceleration** — the N64's RSP has 8-lane SIMD (`VMULF`/`VMADH`);
  offloading matmul would give an estimated 4–8× speedup over scalar VR4300
- **Larger model** — with the Expansion Pak (8MB total), a 6-layer model fits in RAM
- **Richer training data** — more diverse corpus = more coherent responses
- **Real cartridge deployment** — EverDrive compatibility, real hardware video coming

---

## Why This Is Real

The VR4300 was designed for game physics, not transformer inference. Getting Q8.7
fixed-point attention, FFN, and softmax running stably at 93MHz required:

- Custom fixed-point softmax (bit-shift exponential to avoid overflow)
- Q8.7 accumulator arithmetic with saturation guards
- Soft-float compilation flag for float16 block scale decode
- Alignment-safe weight pointer arithmetic for the ROM DFS filesystem

The inference code is in `nano_gpt.c`. The training script is `train_sophia_v5.py`.
Build it yourself and verify.

---

## Elyan Labs

Built by [Elyan Labs](https://rustchain.org). Source is open — build it, improve it, port it.
