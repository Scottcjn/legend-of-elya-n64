# Bounty Fix Analysis

## Issue
- **URL:** https://github.com/Scottcjn/legend-of-elya-n64/issues/4
- **Title:** [BOUNTY: 250 RTC] RSP Matmul Microkernel — Accelerate Inference on N64 Vector Coprocessor
- **Bounty:** $250
- **Labels:** bounty

## Tech Stack


## Issue Body
## Bounty: 250 RTC

Implement an RSP-accelerated int8 GEMM microkernel for the Transformer's linear layers.

### Background
The N64's Reality Signal Processor (RSP) is a 128-bit SIMD vector unit designed for audio/graphics. We want to repurpose it as a neural accelerator for the matmul inner loop.

Current `rsp_matmul.S` exists as a prototype. This bounty is for making it production-quality.

### Acceptance Criteria
- [ ] RSP code integrated in the build system (works with `make`)
- [ ] Benchmark harness: before/after tok/s for CPU-only vs RSP-accelerated
- [ ] Minimum **2x speedup** over pure CPU path on the existing model
- [ ] Numerical parity with CPU path (or documented tolerance)
- [ ] Works in ares emulator; bonus for real hardware testing
- [ ] Short markdown writeup explaining the RSP approach

### Resources
- `rsp_matmul.S` — existing RSP prototype
- `matmul_rsp_drv.c` — CPU-side RSP driver
- `nano_gpt.c` — inference engine (see matmul loops)
- [libdragon RSP docs](https://li

## Analysis
[AI Agent should analyze the issue and implement the fix here]

## Testing Notes
- [ ] Understand the issue requirements
- [ ] Implement the fix
- [ ] Run existing tests
- [ ] Add tests if needed
- [ ] Verify fix addresses the issue
