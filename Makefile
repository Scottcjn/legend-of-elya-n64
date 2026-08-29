# Legend of Elya - N64 Homebrew ROM
# World's First LLM-powered Nintendo 64 Game
#
# ROMs:
#   make base        -> legend_of_elya.z64             (CPU LLM, standalone)
#   make base-rsp    -> legend_of_elya_rsp.z64         (CPU+RSP LLM, standalone ucode)
#                                                     ** DOES NOT RENDER: rsp_load()
#                                                     evicts rdpq's rspq microcode.
#                                                     Headless measurement only. **
#   make base-rsp-ovl-> legend_of_elya_rsp_ovl.z64     (CPU+RSP LLM as an rspq OVERLAY)
#                                                     ** the playable RSP build **
#   make mining      -> legend_of_elya_mining.z64      (CPU LLM + Pico + RTC mining)
#   make rpc-mining  -> legend_of_elya_rpc_mining.z64  (RPC LLM + Pico + RTC mining)
#   make 3d          -> legend_of_elya_3d.z64          (3D + combat)
#   make all         -> all five
#
# Host Tools:
#   make reference   -> reference_cli                  (Host-side reference runner)

N64_INST ?= /home/sophia5070node/n64dev/mips64-toolchain
BUILD_DIR = build
HOST_CC ?= gcc
HOST_CFLAGS ?= -O3 -std=c99 -Wall -Wextra -pedantic

ifeq ($(MAKECMDGOALS),reference)
else
include $(N64_INST)/n64.mk
endif

# --- Model residency knobs (FINDINGS T8/T9) --------------------------------
# The float32 KV cache is 2,097,156 B. Weights 6,750,220 + that = 8,847,376,
# which is 458,768 B over the 8,388,608 B an Expansion Pak console has, so the
# shipped ROM's weight load is guarded off and the transformer never runs.
# An int8 KV cache is 589,828 B and brings the total to 7,340,048 B, which
# fits, with the weights left exactly as they are.
#   make base SGAI_KV=float32   -> old behaviour (does not fit)
SGAI_KV ?= int8
ifeq ($(SGAI_KV),int8)
CFLAGS += -DSGAI_KV_INT8
endif

# matmul_q8's block scale is constant across each 32-weight block, so it does
# not belong in the inner loop: 32 mul.s become 1. GCC will not do it itself —
# n64.mk passes -fno-associative-math, and it IS a reassociation.
# Measured on the REAL ROM under ares, 16 tokens, output byte-identical:
#   base  1,674,524,492 CP0 counts     hoist 1,438,601,627     -14.09 %
# (mupen64plus reported only -7.47 % for the same change because it prices
#  memory access at zero; ares models it. Static instruction counting of the
#  function said it got BIGGER, 110 -> 114 instructions, and would have
#  rejected it.)
#   make base SGAI_HOIST=0   -> the un-hoisted inner loop
SGAI_HOIST ?= 1
ifeq ($(SGAI_HOIST),1)
CFLAGS += -DOPT_HOIST_SCALE
endif

# --- Weight format (FINDINGS F-T009/F-T011) --------------------------------
# SGAI_BITS is the WIDEST weight blob this build will accept.  It sizes the
# static weight buffer, so a ternary build reserves 2,031,632 B instead of
# 6,750,224 B.  MEASURED on the real ROM under ares, 20 forward passes,
# ctx=128, output verified token-exact against the numpy oracle in both arms:
#   bits=8  blob 6,750,220 B   1,709,531,205 CP0   bss 7,358,860
#   bits=2  blob 2,031,628 B     792,761,864 CP0   bss 2,640,268   -53.6 %
# The blob in filesystem/ must be no wider than this: a "SEQ5" blob in a
# SGAI_BITS=2 build is rejected loudly by the size guard, not silently mangled.
#   make base SGAI_BITS=2   -> ternary build (needs a SEQ2 blob)
# Default is now 2: the shipped blob is a SEQ2 ternary model (2,031,628 B),
# retrained 2026-08-06. It answers 12/12 of the game's dialogue prompts
# where the previous SEAI int8 blob answered 0/12, and it is 3.32x smaller.
# Build with SGAI_BITS=8 only if you also swap in an int8 SEAI blob; the
# size guard rejects a mismatch loudly rather than mangling it silently.
SGAI_BITS ?= 2
CFLAGS += -DSGAI_WEIGHT_BITS=$(SGAI_BITS)

# --- Verification builds ----------------------------------------------------
# SGAI_PSE=0 pins the PSE Physarum conductance at 1.0 and drops the burst
# entropy injection, which is what n64qat/eval_qat.py models.  Only then are
# the ROM and the numpy oracle computing the same function, so a token
# mismatch can be attributed to the weight kernel.  OFF by default: the
# shipping ROM keeps PSE.
SGAI_PSE ?= 1
ifeq ($(SGAI_PSE),0)
CFLAGS += -DSGAI_PSE_OFF
endif

# --- PSE Physarum conductance fix (docs/N64_COHERENCE_FINDINGS.md) ----------
# `attn_out[d] = acc * cond` is an uncalibrated per-head gain. It starts at 1.0
# (the trained model) and RATCHETS: the sharpest head always has norm == 1.0 so
# it is always reinforced, and REINFORCE (0.1) is 5x DECAY (0.02), every token
# of every layer. Traced on the shipped blob: 3 of 64 heads welded to the +1.5
# clamp by token 7, 8 by token 15 -- exactly where output diverges from the
# reference -- and 25 by token 31. `pse_physarum_check_reset` never fires
# (0/720 tokens), so nothing undoes it inside a conversation. That is why the
# failure is one-way: coherent early, never fully coherent again.
#
# Renormalising to mean 1.0 keeps the RELATIVE routing the router is for, and
# removes the net gain the model has no learned scale to absorb. Narrowing the
# clamp bounds the residual. Measured, bits per char, lower is better:
#   as shipped (band +-0.50)         18.9873   diverges at token 15
#   SGAI_PSE=0 (router off entirely) 15.4510   exact vs reference
#   this fix  (normalize, +-0.03)    15.3730   diverges at token 56
# So the fix beats disabling the router outright. Game-loop replay over 80 runs:
# shipped 40/80 clean sentence terminations, this 80/80.
# Set PSE_FIX=0 to reproduce the old shipped behaviour.
PSE_FIX ?= 1
ifeq ($(PSE_FIX),1)
CFLAGS += -DPSE_COND_NORMALIZE -DPSE_PHYSARUM_MIN=0.97f -DPSE_PHYSARUM_MAX=1.03f
endif

# --- Repetition penalty fix (docs/N64_SAMPLING_FINDINGS.md) -----------------
# THIS IS THE ONE THAT WAS MISSING.  f32aeff added the SGAI_NO_REP_PENALTY guard
# to nano_gpt.c and 1bc17e5's commit message claims the penalty is fixed, but no
# commit ever added -DSGAI_NO_REP_PENALTY to a build, so every ROM ever shipped
# still runs the hard zero.  PSE_FIX got wired up; this did not.
#
# `sample_logits` at temperature_q8 != 0 does `probs[t] = 0.0f` for the last 3
# emitted characters.  This is a CHARACTER model, so that forbids every doubled
# letter and, far worse, forbids the space -- which English needs every ~5
# characters against a 3-character ban window.  At temperature_q8 = 64 the
# softmax is saturated (raw logits reach +-77, inv_temp = 4), so probs is a
# one-hot: zeroing the top entry zeroes ALL the mass, `total` underflows to
# exactly 0.0f, and the `total <= 0` fallback fires.  Instrumented on the real
# sampling path, prompt "sage says: Who are you?: ", 80 characters:
#   top-1 pre-penalty probability >= 0.999999   68/80   the DISTRIBUTION IS FINE
#   chosen char was the argmax                  68/80
#   chosen char was a real tail draw (rank>=2)   1/80
#   steps where the penalty removed >= 0.999 mass  10/80
#   characters emitted as '!'                      8    ALL 8 from the fallback
# 20 seeded runs of the exact game loop, same prompt, temp_q8 = 64:
#   as shipped   mean 75.4 chars   4/20 clean terminations   0 doubled   8.16% '!'
#   this fix     mean 33.2 chars  20/20 clean terminations  14 doubled   0.00% '!'
# A SOFT penalty is a measured no-op (REP_FACTOR 0.5 / 0.25 / 0.01 all produce
# byte-identical transcripts to no penalty at all), so removal is the fix.
# Set REP_FIX=0 to reproduce the shipped behaviour.
REP_FIX ?= 1
ifeq ($(REP_FIX),1)
CFLAGS += -DSGAI_NO_REP_PENALTY
else
CFLAGS += -DSGAI_FALLBACK_ASCII_SCAN
endif

# --- First-character fix (docs/N64_SAMPLING_FINDINGS.md) --------------------
# update_generating_step() keeps the model's first predicted character in
# G.gen_last_tok, feeds it back correctly, and never appends it to dialog_buf.
# Every NPC line loses its first character: "I am Sophia" reaches the player as
# " am Sophia", "The N64 runs a transformer" as "he N64 runs a transformer".
# Set FIRSTCHAR_FIX=0 to reproduce the shipped behaviour.
FIRSTCHAR_FIX ?= 1
ifeq ($(FIRSTCHAR_FIX),1)
CFLAGS += -DSGAI_EMIT_FIRST_TOKEN
endif

# Extra flags for one-off verification builds, e.g. EXTRA=-DBOOT_PROBE.
# APPENDED, never assigned: `make CFLAGS=...` on the command line silently
# discards every knob above it (that mistake produced an 8,866,956-byte bss
# before it was caught).
CFLAGS += $(EXTRA)

all: legend_of_elya.z64 legend_of_elya_rsp.z64 legend_of_elya_mining.z64 legend_of_elya_rpc_mining.z64 legend_of_elya_3d.z64


# --- Base ROM (CPU LLM, standalone, no Pico) ---
base: legend_of_elya.z64

$(BUILD_DIR)/legend_of_elya.dfs: filesystem/sophia_weights.bin filesystem/dungeon.wav64 filesystem/library.wav64 filesystem/forge.wav64

$(BUILD_DIR)/legend_of_elya.elf: $(BUILD_DIR)/legend_of_elya.o $(BUILD_DIR)/nano_gpt.o

legend_of_elya.z64: N64_ROM_TITLE="Legend of Elya"
legend_of_elya.z64: $(BUILD_DIR)/legend_of_elya.dfs

# --- RSP-accelerated ROM (CPU+RSP LLM, standalone, no Pico) ---
base-rsp: legend_of_elya_rsp.z64

$(BUILD_DIR)/legend_of_elya_rsp.dfs: filesystem/sophia_weights.bin filesystem/dungeon.wav64 filesystem/library.wav64 filesystem/forge.wav64

$(BUILD_DIR)/nano_gpt_rsp.o: nano_gpt.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/matmul_rsp2.o: matmul_rsp2.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/legend_of_elya_rsp.o: legend_of_elya.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/legend_of_elya_rsp.elf: $(BUILD_DIR)/legend_of_elya_rsp.o $(BUILD_DIR)/nano_gpt_rsp.o $(BUILD_DIR)/matmul_rsp2.o $(BUILD_DIR)/rsp_mm2.o

legend_of_elya_rsp.z64: N64_ROM_TITLE="Elya RSP"
legend_of_elya_rsp.z64: $(BUILD_DIR)/legend_of_elya_rsp.dfs

# --- RSP-as-rspq-OVERLAY ROM (the playable one) -------------------------------
# Same LLM as base-rsp, but the matmul is an rspq overlay instead of a
# standalone microcode, so it coexists with the rdpq ucode rdpq_init()
# installs.  base-rsp renders nothing after sgai_init(); this one does.
base-rsp-ovl: legend_of_elya_rsp_ovl.z64

$(BUILD_DIR)/legend_of_elya_rsp_ovl.dfs: filesystem/sophia_weights.bin filesystem/dungeon.wav64 filesystem/library.wav64 filesystem/forge.wav64

$(BUILD_DIR)/nano_gpt_rsp_ovl.o: nano_gpt.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/matmul_rsp2_ovl.o: matmul_rsp2.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DRSP_MM_OVERLAY -o $@ $<

$(BUILD_DIR)/legend_of_elya_rsp_ovl.o: legend_of_elya.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/legend_of_elya_rsp_ovl.elf: $(BUILD_DIR)/legend_of_elya_rsp_ovl.o $(BUILD_DIR)/nano_gpt_rsp_ovl.o $(BUILD_DIR)/matmul_rsp2_ovl.o $(BUILD_DIR)/rsp_mm2_ovl.o

legend_of_elya_rsp_ovl.z64: N64_ROM_TITLE="Elya RSP OVL"
legend_of_elya_rsp_ovl.z64: $(BUILD_DIR)/legend_of_elya_rsp_ovl.dfs

# --- Dual-processor consensus probe ------------------------------------------
# TWO models resident: a ternary one on the MIPS core and an int8 one on the
# RSP, voting per token.  Its own main() (dual_probe.c) rather than another
# #ifdef in legend_of_elya.c, because the game's globals are built around one
# model and this needs two of everything -- two weight buffers, two KV caches,
# two activation scratch blocks.
#
# Its filesystem is filesystem_dual/, NOT filesystem/: it carries a second
# 3.4 MB blob and every other ROM here would grow by that much if it shared.
#
#   make dual                       ternary CPU + int8 RSP, 16 tokens, shift 5
#   make dual EXTRA=-DDUAL_SHIFT=0  the naive sum, i.e. the failure mode
dual: legend_of_elya_dual.z64

# DUALFS selects which pair of blobs is baked in, so the mixed-format ROM and
# the same-format control are the same build with a different filesystem.
DUALFS ?= filesystem_dual
$(BUILD_DIR)/legend_of_elya_dual.dfs: $(DUALFS)/sophia_weights.bin $(DUALFS)/sophia_int8.bin
	@mkdir -p $(BUILD_DIR)
	$(N64_MKDFS) $@ $(DUALFS)/

$(BUILD_DIR)/dual_probe.o: dual_probe.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/legend_of_elya_dual.elf: $(BUILD_DIR)/dual_probe.o $(BUILD_DIR)/nano_gpt_rsp_ovl.o $(BUILD_DIR)/matmul_rsp2_ovl.o $(BUILD_DIR)/rsp_mm2_ovl.o

legend_of_elya_dual.z64: N64_ROM_TITLE="Elya Dual"
legend_of_elya_dual.z64: $(BUILD_DIR)/legend_of_elya_dual.dfs

# --- AUDIO-ONLY ROM: isolate the music from the model (audio_test.c) ---
audiotest: legend_of_elya_audiotest.z64

$(BUILD_DIR)/legend_of_elya_audiotest.dfs: filesystem/dungeon.wav64 filesystem/library.wav64 filesystem/forge.wav64
	@rm -rf $(BUILD_DIR)/aufs && mkdir -p $(BUILD_DIR)/aufs
	cp filesystem/dungeon.wav64 filesystem/library.wav64 filesystem/forge.wav64 $(BUILD_DIR)/aufs/
	$(N64_MKDFS) $@ $(BUILD_DIR)/aufs/

$(BUILD_DIR)/audio_test.o: audio_test.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/legend_of_elya_audiotest.elf: $(BUILD_DIR)/audio_test.o

legend_of_elya_audiotest.z64: N64_ROM_TITLE="Elya Audio"
legend_of_elya_audiotest.z64: $(BUILD_DIR)/legend_of_elya_audiotest.dfs

# --- PREFETCH ROM: does proximity prefetch hide the expert load? (prefetch_probe.c) ---
#
#   make prefetchprobe    four scripted walks, control vs prefetch, cart only
prefetchprobe: legend_of_elya_pfprobe.z64

$(BUILD_DIR)/legend_of_elya_pfprobe.dfs: $(MOE_BANK)
	@rm -rf $(BUILD_DIR)/pffs && mkdir -p $(BUILD_DIR)/pffs
	cp $(MOE_BANK) $(BUILD_DIR)/pffs/sophia_moe.bin
	$(N64_MKDFS) $@ $(BUILD_DIR)/pffs/

$(BUILD_DIR)/prefetch_probe.o: prefetch_probe.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -DRSP_MM_EPI_OVERLAP -o $@ $<

$(BUILD_DIR)/legend_of_elya_pfprobe.elf: $(BUILD_DIR)/prefetch_probe.o $(BUILD_DIR)/nano_gpt_ec.o $(BUILD_DIR)/matmul_rsp2_ec.o $(BUILD_DIR)/rsp_mm2_ovl.o $(BUILD_DIR)/expert_cache.o

legend_of_elya_pfprobe.z64: N64_ROM_TITLE="Elya Prefetch"
legend_of_elya_pfprobe.z64: $(BUILD_DIR)/legend_of_elya_pfprobe.dfs

# --- SD ROM: can an expert stream off the EverDrive SD card? (sd_probe.c) ---
#
#   make sdprobe                     prints ABSENT without a flashcart
#   make sdprobe EXTRA="-DSD_LBA=..." raw-sector arms need the file's LBA
sdprobe: legend_of_elya_sdprobe.z64

$(BUILD_DIR)/legend_of_elya_sdprobe.dfs: $(MOE_BANK) filesystem/sophia_weights.bin
	@rm -rf $(BUILD_DIR)/sdfs && mkdir -p $(BUILD_DIR)/sdfs
	cp filesystem/sophia_weights.bin $(BUILD_DIR)/sdfs/
	cp $(MOE_BANK) $(BUILD_DIR)/sdfs/sophia_moe.bin
	$(N64_MKDFS) $@ $(BUILD_DIR)/sdfs/

$(BUILD_DIR)/sd_probe.o: sd_probe.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -DRSP_MM_EPI_OVERLAP -o $@ $<

$(BUILD_DIR)/legend_of_elya_sdprobe.elf: $(BUILD_DIR)/sd_probe.o $(BUILD_DIR)/nano_gpt_ec.o $(BUILD_DIR)/matmul_rsp2_ec.o $(BUILD_DIR)/rsp_mm2_ovl.o $(BUILD_DIR)/expert_cache.o

legend_of_elya_sdprobe.z64: N64_ROM_TITLE="Elya SD"
legend_of_elya_sdprobe.z64: $(BUILD_DIR)/legend_of_elya_sdprobe.dfs

# --- MoE ROM: route -> stream -> swap -> generate (moe_probe.c) ---
#
#   make moeprobe    needs filesystem_moe/sophia_moe.bin (training/make_moe_bank.py)
moeprobe: legend_of_elya_moeprobe.z64

# MOE_BANK selects which bank is baked in.  The DFS is built from a STAGING
# directory holding exactly one bank, never from a directory that happens to
# contain several: mkdfs packs whatever it finds, so keeping both banks in
# filesystem_moe/ silently doubled the ROM to 10 MB and shipped a second,
# unused 5 MB copy of the weights.
MOE_BANK ?= banks/sophia_moe_sep.bin
$(BUILD_DIR)/legend_of_elya_moeprobe.dfs: $(MOE_BANK)
	@rm -rf $(BUILD_DIR)/moefs && mkdir -p $(BUILD_DIR)/moefs
	cp $(MOE_BANK) $(BUILD_DIR)/moefs/sophia_moe.bin
	$(N64_MKDFS) $@ $(BUILD_DIR)/moefs/

$(BUILD_DIR)/moe_probe.o: moe_probe.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -DRSP_MM_EPI_OVERLAP -o $@ $<

$(BUILD_DIR)/legend_of_elya_moeprobe.elf: $(BUILD_DIR)/moe_probe.o $(BUILD_DIR)/nano_gpt_ec.o $(BUILD_DIR)/matmul_rsp2_ec.o $(BUILD_DIR)/rsp_mm2_ovl.o $(BUILD_DIR)/expert_cache.o

legend_of_elya_moeprobe.z64: N64_ROM_TITLE="Elya MoE"
legend_of_elya_moeprobe.z64: $(BUILD_DIR)/legend_of_elya_moeprobe.dfs

# --- EC ROM: does an expert-sized PI DMA hide behind a token? (ec_probe.c) ---
#
#   make ecprobe     BASE / BLOCK / HIDE arms over the same 16 real tokens
ecprobe: legend_of_elya_ecprobe.z64

$(BUILD_DIR)/legend_of_elya_ecprobe.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/ec_probe.o: ec_probe.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -DRSP_MM_EPI_OVERLAP -o $@ $<

$(BUILD_DIR)/expert_cache.o: src/expert_cache.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/nano_gpt_ec.o: nano_gpt.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/matmul_rsp2_ec.o: matmul_rsp2.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DRSP_MM_OVERLAY -DRSP_MM_EPI_OVERLAP -DRSP_PHASE_TIMING -o $@ $<

$(BUILD_DIR)/legend_of_elya_ecprobe.elf: $(BUILD_DIR)/ec_probe.o $(BUILD_DIR)/nano_gpt_ec.o $(BUILD_DIR)/matmul_rsp2_ec.o $(BUILD_DIR)/rsp_mm2_ovl.o $(BUILD_DIR)/expert_cache.o

legend_of_elya_ecprobe.z64: N64_ROM_TITLE="Elya EC"
legend_of_elya_ecprobe.z64: $(BUILD_DIR)/legend_of_elya_ecprobe.dfs

# --- XCHK ROM: CPU-vs-RSP cross-check on screen (xchk_probe.c) ---
#
#   make xchk        same ternary blob on both engines, verdict drawn on screen
xchk: legend_of_elya_xchk.z64

$(BUILD_DIR)/legend_of_elya_xchk.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/xchk_probe.o: xchk_probe.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/legend_of_elya_xchk.elf: $(BUILD_DIR)/xchk_probe.o $(BUILD_DIR)/nano_gpt_rsp_ovl.o $(BUILD_DIR)/matmul_rsp2_ovl.o $(BUILD_DIR)/rsp_mm2_ovl.o

legend_of_elya_xchk.z64: N64_ROM_TITLE="Elya XCHK"
legend_of_elya_xchk.z64: $(BUILD_DIR)/legend_of_elya_xchk.dfs

# --- Mining ROM (CPU LLM + Pico bridge + RTC mining) ---
mining: legend_of_elya_mining.z64

$(BUILD_DIR)/legend_of_elya_mining.dfs: filesystem/sophia_weights.bin
	@mkdir -p $(BUILD_DIR)
	$(N64_MKDFS) $@ filesystem/

$(BUILD_DIR)/n64_attest.o: mining/n64/n64_attest.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -Imining/n64 -o $@ $<

$(BUILD_DIR)/legend_of_elya_mining.o: legend_of_elya_mining.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -Imining/n64 -o $@ $<

$(BUILD_DIR)/legend_of_elya_mining.elf: $(BUILD_DIR)/legend_of_elya_mining.o $(BUILD_DIR)/nano_gpt.o $(BUILD_DIR)/n64_attest.o

legend_of_elya_mining.z64: N64_ROM_TITLE="Elya Mining"
legend_of_elya_mining.z64: $(BUILD_DIR)/legend_of_elya_mining.dfs

# --- RPC Mining ROM (RPC LLM + Pico bridge + RTC mining) ---
rpc-mining: legend_of_elya_rpc_mining.z64

$(BUILD_DIR)/legend_of_elya_rpc_mining.dfs: filesystem/sophia_weights.bin
	@mkdir -p $(BUILD_DIR)
	$(N64_MKDFS) $@ filesystem/

$(BUILD_DIR)/n64_llm_rpc_mining.o: bridge/n64/n64_llm_rpc.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_MINING -DUSE_RPC_LLM -Ibridge/n64 -Imining/n64 -o $@ $<

$(BUILD_DIR)/legend_of_elya_rpc_mining.o: legend_of_elya_mining.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RPC_LLM -Imining/n64 -Ibridge/n64 -o $@ $<

$(BUILD_DIR)/legend_of_elya_rpc_mining.elf: $(BUILD_DIR)/legend_of_elya_rpc_mining.o $(BUILD_DIR)/nano_gpt.o $(BUILD_DIR)/n64_attest.o $(BUILD_DIR)/n64_llm_rpc_mining.o

legend_of_elya_rpc_mining.z64: N64_ROM_TITLE="Elya RPC Mine"
legend_of_elya_rpc_mining.z64: $(BUILD_DIR)/legend_of_elya_rpc_mining.dfs

# --- 3D ROM (Lopie model + combat + RTC rewards) ---
3d: legend_of_elya_3d.z64

$(BUILD_DIR)/legend_of_elya_3d.dfs: filesystem/sophia_weights.bin
	@mkdir -p $(BUILD_DIR)
	$(N64_MKDFS) $@ filesystem/

$(BUILD_DIR)/legend_of_elya_3d.o: legend_of_elya_3d.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/legend_of_elya_3d.elf: $(BUILD_DIR)/legend_of_elya_3d.o $(BUILD_DIR)/nano_gpt.o

legend_of_elya_3d.z64: N64_ROM_TITLE="Elya 3D"
legend_of_elya_3d.z64: $(BUILD_DIR)/legend_of_elya_3d.dfs

clean:
	# libdragon's n64.mk does not make objects depend on CFLAGS, and this
	# target used to be EMPTY. `make clean && make base EXTRA=<flags>` then
	# rebuilt nothing and produced a byte-identical ROM, so any A/B that
	# varied only CFLAGS silently compared a ROM against itself.
	# Only $(BUILD_DIR): the .z64 files are TRACKED artifacts and some of
	# them (legend_of_elya_rpc.z64) are not rebuilt by any target here, so
	# `rm *.z64` would delete them permanently. Removing the objects and
	# .elf is enough to force a genuine recompile.
	rm -rf $(BUILD_DIR)

# --- Reference CLI (x86/ARM host) ---
reference: reference_cli

reference_cli: reference_cli.c
	$(CC) -o reference_cli reference_cli.c -lm -std=c99

.PHONY: reference
	rm -rf $(BUILD_DIR) legend_of_elya.z64 legend_of_elya_rsp.z64 legend_of_elya_rsp_ovl.z64 legend_of_elya_mining.z64 legend_of_elya_rpc_mining.z64 legend_of_elya_3d.z64 reference_cli

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all base base-rsp base-rsp-ovl dual mining rpc-mining 3d reference clean

reference: reference_cli
reference_cli: reference_cli.c
	gcc -O3 -std=c99 -Wall -o $@ $< -lm

