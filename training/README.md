# Training and quantization tooling

The scripts that produced the shipped model (`filesystem/sophia_weights.bin`,
SEQ2 ternary, 2,031,628 bytes, answering 12/12 of the game's dialogue prompts).

**Checkpoints are deliberately not committed.** The twelve trained models are
~1 GB and every one is reproducible from `train_sophia_v9_qat.py` with a fixed
seed. Only the shipped blob lives in the repository.

## What each script is for

| script | role |
|---|---|
| `train_sophia_v9_qat.py` | quantization-aware trainer. Schemes: `none / int3 / int4 / int5 / int6 / int8 / ternary` (TWN, `--tau`) `/ ternary_bn` (BitNet absmean). Config-driven shape. |
| `train_mix.py` | corpus replication-mix sweep, paraphrase holdout, blob export |
| `eval_qat.py` | numpy reference implementation of `nano_gpt.c`. **The oracle.** |
| `eval12.py` | scores the 12 game dialogue prompts — the metric that actually matters |
| `ptq_from_ckpt.py` | post-training-quantization control: loads a float checkpoint into a quantized config |
| `qat_npz_to_seq.py` | **the exporter to use.** Re-emits from the same `.npz` the oracle reads. |
| `score_rom.py`, `compare_rom_host.py` | compare ROM output against the oracle |
| `run_eval_all.py`, `report.py`, `report_sweep.py` | evaluation harnesses |

## Four things that will bite you

**1. Quantization-aware training is not optional at low bit widths.** Post-training
ternarization of this model produced a fit loss of 6.63. The corpus is
byte-level, so `ln(256) = 5.545` — that is worse than uniform random over bytes.
The same 2,031,628-byte artifact trained with the quantizer in the loop reaches
0.1012 against an fp32 control's 0.1007. No post-hoc threshold sweep recovers it,
because QAT does not work by finding quantization-friendly weights: mean |w|
0.0748 vs 0.0826, kurtosis 3.85 vs 3.86, and both ternarize to the same 42.5%
zeros. The network absorbs the error during training instead.

**2. Do not select checkpoints on validation loss.** It runs backwards here.
Validation rises monotonically 3.77 → 5.33 while the game score goes 11 → 12, and
`best_val@1500` emits `"Shufles vectors in one cycle."` Select on the 12/12
behavioural metric (`eval12.py`) and take the final checkpoint.

**3. The v7 corpus has no held-out split.** `build_v7_corpus` makes "val" a
reshuffle of the same lines, so val ≡ fit by construction and the number is
meaningless as a generalization signal. `train_mix.py` builds a paraphrase
holdout instead: for any key with three or more answers the last moves to
validation, so every key keeps at least two and no game prompt loses its answer.
That is 44 of 400 lines (11%) genuinely unseen, fit 0.19 against val 5.5.

**4. Use `qat_npz_to_seq.py`, not the trainer's own `.bin` writer.** That writer
is broken three ways and fails silently: it stamps `SEAI` magic for every scheme
(so ternary decodes as int8), packs LSB-first while the runtime reads MSB-first,
and stores offset-binary while the runtime decodes two's complement. Always prove
`max|dW| = 0` against the `.npz` the oracle reads before trusting a blob.

## Reproducing the shipped model

```sh
python3 training/train_mix.py --mix 800/800/400 --quant ternary_bn --steps 9000
python3 training/qat_npz_to_seq.py <run>.npz filesystem/sophia_weights.bin
python3 training/eval12.py filesystem/sophia_weights.bin      # expect 12/12
```

Then build and verify on the target:

```sh
N64_INST=$HOME/n64-toolchain/mips64-toolchain make base EXTRA="-DGAME12_PROBE"
flatpak run dev.ares.ares --system "Nintendo 64" legend_of_elya.z64
```

`GAME12_PROBE` runs all twelve prompts through the real runtime at temperature 0
and at the game's `temperature_q8 = 64`, printing over IS-Viewer. It runs inside
`game_init()` because headless emulation has no GPU, so the ROM shows a black
screen while it works — that is expected, and the output is on stdout.

Findings journals: `docs/N64_RETRAIN_FINDINGS.md`, `docs/N64_COHERENCE_FINDINGS.md`,
`docs/N64_RSP_FINDINGS.md`, `docs/N64_MEASUREMENT_FINDINGS.md`,
`docs/EMULATOR_ACCURACY.md`.
