# Lane 6 — Dialogue UX at 5.87 tok/s

**Summary.** At 5.87 tok/s with a byte tokenizer, one token is one character, so generation *is* the typewriter: ~170 ms/char, ~4.6 s for a 27-char answer (C028/C029 median). The current loop (`update_generating_step()`, `legend_of_elya.c:1503`) calls `sgai_next_token()` synchronously once per game-loop iteration and sets `G.dialog_char = G.dialog_len` immediately, so during generation the render loop runs at the token rate — ~5.9 fps, not 60 — and every character pops the instant it exists. This design does three things: slices inference to a frame deadline so the loop keeps 60 fps; buffers characters and reveals them **by word**, so speech reads as deliberate rather than as a stuttering drip; and moves both the 80 ms `sgai_init_ex()` swap and the 1 MB expert DMA off the dialogue's critical path onto the player's walk. I ran nothing for this document; it is read from the repo.

## Substates (inside `STATE_GENERATING`, top-level states unchanged)

```c
typedef enum { DLG_ROUTE, DLG_SWAP, DLG_PROMPT, DLG_STREAM,
               DLG_DRAIN, DLG_DONE, DLG_ABORT } DlgPhase;
```

`G.dlg` holds `phase`, `expert`, `reveal_len`, `reveal_vbl`, `word_end`, `abort`, `over_budget_frames`.

| phase | per frame | exit |
|---|---|---|
| DLG_ROUTE | `moe_route(prompt)`; `ec_request()` if not resident | resident → SWAP; else stay (spinner) |
| DLG_SWAP | `sgai_init_step(&ST, budget)` — resumable permute | done → PROMPT |
| DLG_PROMPT | `sgai_pump()` on prompt bytes, output discarded | `gen_ppos>=gen_plen` → STREAM |
| DLG_STREAM | `sgai_pump()`; on token: append, maybe advance word | EOS/80 chars → DRAIN |
| DLG_DRAIN | reveal only; no inference | `reveal_len==dialog_len` → DONE |
| DLG_DONE | — | A / B → `STATE_DIALOG` / `STATE_DUNGEON` |
| DLG_ABORT | finish current pump slice, discard | → `STATE_DIALOG` with text so far |

## Never blocking the frame

Two new resumable entry points in `nano_gpt.c`:

```c
int sgai_pump(SgaiState *s, uint8_t in_tok, int temp_q8,
              uint32_t deadline_ticks, uint8_t *out_tok);  /* SGAI_MORE|SGAI_TOKEN */
int sgai_init_step(SgaiState *s, const uint8_t *blob, uint32_t deadline_ticks);
```

The yield point already exists: F-R026 issues the matvec as `RSP_MM_EPI_CHUNKS` (16) row slices, each with its own syncpoint and parameter block. `sgai_pump()` returns after the first chunk whose completion crosses `deadline_ticks`; state lives in `SgaiState`, not on the stack. Caller: `deadline = TICKS_READ() + FRAME_CYCLES - RESERVE`, `RESERVE` ≈ 4 ms for RDP + `draw_text`. Same for the swap: permute a bounded byte count of the 1,048,588 B blob per frame, so ~80 ms becomes ~5 frames of animation.

**This changes no arithmetic.** The xchk ROM (`make xchk`, F-R024) must still report byte-identical CPU-vs-RSP output, and the ares run must still emit the six F-R028 answers unchanged, before `sgai_pump` is believed.

## Reveal, and what the player does meanwhile

- Characters land in `dialog_buf`; `reveal_len` is separate. A word is revealed only when its terminating space, `.` or `\n` has been generated — ~1 word/s, whole words, no half-words on screen.
- Cursor `_` blinks on `g_vbl`, not on token arrival, so it is alive during a 170 ms token.
- **Walking is allowed while generating.** `STATE_GENERATING` still runs `scene_dungeon()`; the D-pad moves Elya, camera follows, sparks update. Leaving the NPC's radius sets `abort`.
- **B = skip:** sets `abort`; the pump finishes its slice, then DRAIN reveals the rest at 60 chars/s. B again closes. **A = hurry:** reveal jumps to `dialog_len`. Neither can skip generation, and neither is allowed to say "done" while `phase != DLG_DONE`.

## Prefetch on the walk

The 1 MB acquire hides (F-R028: 18.8 M cycles cold, ~100 warm). Warm it before the dialogue exists, using the *generated* router only — never a second keyword table:

1. Player within ~3 tiles of NPC *n*: `ec_prefetch(&EC, moe_route(NPC_DIALOG_OPTIONS[n][0]), cur)`.
2. In `STATE_DIALOG_SELECT`, every cursor move: route the highlighted option, `ec_prefetch(..., cur)`.
3. On A: if `ec_resident()` → skip ROUTE's wait entirely; DLG_SWAP starts the same frame.

`ec_prefetch()` is a no-op when wrong. `ec_poll()` must be reached every frame (call `ec_resident()` in the dungeon tick) or `inflight_slot` sticks and prefetch silently switches itself off — the bug already documented in `expert_cache.c`.

## Designed so it cannot lie

- HUD (`perf_show`) shows `hit/miss/pf` from `ExpertCache` and `over_budget_frames` = frames where the loop exceeded one vblank. A UX claiming 60 fps with a nonzero, unshown counter is exactly this repo's recurring failure.
- Assert `reveal_len <= dialog_len <= 80` each frame; violation forces DLG_ABORT and prints to ISViewer.
- `DLG_STREAM` requires `ec_resident(expert)`; if false, hard-fail visibly rather than generating from whatever is in the slot.
- Prefetch and generation must call the same `moe_route()`; a UX-local copy is a bug by the repo's own rule.

## What I did NOT verify

- **Everything.** I read the repo; I built nothing, ran no ROM, no ares, no host suite.
- That `sgai_next_token()` state is slice-able at a chunk boundary, or that `SgaiState` holds enough to resume — I did not read `nano_gpt.c`'s inner loop.
- Per-layer / per-chunk cost. 170 ms/token ÷ 16 chunks ≈ 10 ms is arithmetic on the published rate, not a measurement; if a chunk exceeds one frame, this design fails and needs finer slicing.
- Whether `sgai_init_ex()`'s permutation is resumable at all, or the 4 ms `RESERVE` is enough for the RDP pass plus `draw_text`.
- That pump-slicing preserves bit-exactness (F-R024/F-R018's 176/176) — asserted as a gate, not observed.
- Console behaviour: all cited numbers are ares 147. Also unverified: EverDrive SD paths, RDRAM headroom with two 1 MB slots plus KV plus framebuffer under this state machine, and whether walking during generation perturbs the vblank-based rate readout.
