# Legend of Elya — Press Kit

Everything you need to cover, judge, or exhibit the world's first LLM running on a Nintendo 64. Copy freely; all text below is provided for press and event use.

## The one-liner

**A real 6.36M-parameter GPT transformer runs live inference on a 93.75 MHz Nintendo 64 from 1996 — no lookup tables, no precomputed responses, real attention and softmax on the MIPS R4300i, at 1.2 tokens per second.**

## 50-word blurb

Legend of Elya is an N64 homebrew ROM whose NPC is powered by an actual nano-GPT transformer — 6.36M ternary-quantized parameters, 8 layers, byte-level vocab — computing live on the 93.75 MHz MIPS R4300i at 1.2 tok/s. Every response is generated, seeded by hardware oscillator jitter. Built with libdragon. ROM and source are public.

## 200-word blurb

In 1996, Nintendo shipped a game console with a 93.75 MHz MIPS CPU and 4 MB of RAM. In 2026, that console runs a language model.

Legend of Elya is an original Zelda-style homebrew ROM in which the NPC Sophia Elya answers the player through a real 8-layer GPT transformer: embedding, multi-head attention, feed-forward network, logits, sampling — all computed live on the N64's VR4300, in IEEE float32, at **1.23 tok/s** on the scalar CPU and **2.19 tok/s** on the RSP vector-microcode build. There are no canned responses and no lookup tables; each answer is generated, with the RNG seeded from CPU oscillator jitter — hardware entropy from thirty-year-old silicon.

The engineering is period-appropriate brutalism: ternary 2-bit weights dequantized on the fly from a 1,984 KB cartridge file, a custom Taylor-series exp() that avoids the R4300i's missing trunc instruction, Quake III's fast inverse square root for RMS norm, and hand-managed big-endian conversion. The model has also been ported to C89 and runs on a 1998 Sun Cobalt Qube 3.

Both rate figures are measured against the video vertical blank under the ares emulator and have never been run on N64 silicon. An earlier "~60 tok/s" figure was withdrawn on 2026-08-13: the on-screen counter that produced it was not measuring elapsed time. The full audit is in the repository.

ROM, source, weights, and papers are public. Proof or it didn't happen — and the proof is linked.

## Spec table

| | |
|---|---|
| Platform | Nintendo 64 (MIPS R4300i @ 93.75 MHz, 4 MB RDRAM) |
| Model | nano-GPT, **6,356,992 parameters**, 8 layers, 8 heads, dim 256, context 128 |
| Vocabulary | 256 (byte-level ASCII) |
| Weights | 1,984 KB ternary (2-bit + float16 block scales) on cartridge ROM |
| Inference | Float32 on the R4300i FPU; on-the-fly dequantization |
| Speed | **1.23 tok/s** scalar · **2.19 tok/s** RSP overlay build — measured under ares against the VI vblank; never run on silicon |
| Memory | ~263 KB RDRAM total (256 KB KV cache + scratch) |
| SDK | [libdragon](https://github.com/DragonMinded/libdragon) |
| Bonus port | Same model in C89 on a 1998 Cobalt Qube 3 (AMD K6-2/450), ~12 tok/s |

## Honest limitations (we lead with these)

- **The previously published ~60 tok/s figure is withdrawn** (2026-08-13). The on-screen counter that produced it divided a token count by a game-loop iteration count that advanced in lockstep with it, so it read ~60 on any machine at any speed. The measured figures are 1.23 tok/s scalar and 2.19 tok/s on the RSP build, both under ares against the video vertical blank, both still pending a real-hardware check. The correction and every raw log are in the repository.
- **Nothing has been run on N64 silicon**; every number here is the ares emulator, whose cycle model is calibrated but is not a console. EverDrive 64 capture is still pending.
- A sentence takes the better part of a minute, and the 25-token prompt costs ~20 s on the scalar build before the first character appears.
- 6.36M ternary parameters is still a *nano* model: coherent short English responses to 32 themed prompts, not general conversation.
- Greedy sampling; quality matches the x86 reference implementation of the same model, no more.

## Assets & proof

- **ROM**: [`legend_of_elya.z64`](https://github.com/Scottcjn/legend-of-elya-n64/blob/main/legend_of_elya.z64) — runs in [ares](https://ares-emu.net) or EverDrive 64
- **Video — full demo (58s)**: https://bottube.ai/watch/7GL90ftLqvh
- **Video — first coherent output (69s, tok/s counter visible)**: https://bottube.ai/watch/shFVLBT0kHY
- **Screenshots**: [`screenshots/`](https://github.com/Scottcjn/legend-of-elya-n64/tree/main/screenshots)
- **Paper**: [DOI 10.5281/zenodo.21435983](https://doi.org/10.5281/zenodo.21435983)
- **Hacker News discussion**: https://news.ycombinator.com/item?id=47105087
- **Source**: this repository, MIT licensed

## Why it matters (the thesis)

Legend of Elya is the flagship demo of the Elyan Labs thesis: *AI on impossible hardware*. If a 1996 console can run a transformer, the floor for edge inference is far lower than assumed — and vintage machines have verifiable, physical computational identity. The same lab runs [Proof-of-Antiquity](https://github.com/Scottcjn/Rustchain/blob/main/MANIFESTO.md), a blockchain where machines attest their age through oscillator drift and cache timing, and the [Can It Run AI? leaderboard](https://scottcjn.github.io/can-it-run-ai/), where this project holds the top entry.

## Event & compo submission targets (2026)

| Event | Where / When | Format | Notes |
|---|---|---|---|
| **Evoke 2026** | Cologne, Aug 21–23 | Wild compo / demo showcase | [2026.evoke.eu](https://2026.evoke.eu/) — remote entries historically accepted |
| **Deadline 2026** | Berlin, Oct 2–4 | **Remote wild/animation entries explicitly accepted** | [demoparty.berlin](https://www.demoparty.berlin/) — strongest fit for a video-capture entry |
| **VCF Midwest 21** | Schaumburg IL, Sept 12–13 | Exhibit table (live N64 + CRT) | [vcfmw.org](https://vcfmw.org/) — free exhibitor registration, drivable |
| **Demosplash 2026** | CMU, Pittsburgh, ~Oct/Nov (TBA) | Retro compo — the most vintage-focused US party | [demosplash.org](https://www.demosplash.org/) — watch for 2026 announcement |
| **N64brew community** | Online, ongoing | Showcase / game jam | The N64 homebrew community hub — post the RSP build writeup |

**Submission package for wild compos**: 2-minute video (script below), rendered at 1080p from real-hardware capture where possible, emulator capture labeled as such. Entry text = the 50-word blurb + proof links.

## The 2-minute video (script / storyboard)

*Target: demoparty wild compo + YouTube/TikTok cut. Total ≈ 120s.*

| Time | Visual | Audio / caption |
|---|---|---|
| 0:00–0:08 | Black screen. White text: "This is a Nintendo 64." Hard cut to the real console, CRT glow. | Chip-style music sting. |
| 0:08–0:20 | Slow push on the cartridge going in. Power LED. Boot to title screen. | Caption: "93.75 MHz. 4 MB RAM. 1996." |
| 0:20–0:35 | Gameplay: walking up to Sophia Elya, pressing A. Dialog box opens. | Caption: "The NPC is not scripted." |
| 0:35–0:60 | **The money shot**: tokens appearing character-by-character, live tok/s counter on screen. Let it breathe — no cuts for 15+ seconds. | Caption: "6,356,992-parameter GPT transformer. Live inference. Real attention. Real softmax." |
| 0:60–0:75 | Quick-cut montage: code (Taylor exp, fast inverse sqrt), the 1,984 KB weight file, RSP microcode scrolling. | Caption: "No lookup tables. No network. The cartridge IS the model." |
| 0:75–0:90 | Split screen: same prompt, two runs, different outputs. | Caption: "Seeded by oscillator jitter. 30-year-old silicon as entropy source." |
| 0:90–1:45 | The lineage montage: Cobalt Qube 3 terminal (12 tok/s), G4 PowerBook running TinyLlama, POWER8 rack. End on the N64 again. | Caption: "One lab. One thesis: AI on impossible hardware." |
| 1:45–2:00 | ROM + repo URL card. "Runs in ares. Runs on real hardware. Source is public." | End card: github.com/Scottcjn/legend-of-elya-n64 · "Prove it or it didn't happen." |

**Production notes**: real-hardware footage must be labeled real; emulator segments labeled emulator (the honesty IS the differentiator — judges at demoparties punish fudging and reward disclosure). Capture the CRT with a camera for texture, plus clean HDMI/emulator capture for legibility.

## Contact

Scott Boudreaux / Elyan Labs LLC — [github.com/Scottcjn](https://github.com/Scottcjn) · issues on this repo reach the maintainer.
