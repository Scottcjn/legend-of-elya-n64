# Legend of Elya - N64 Homebrew ROM
# World's First LLM-powered Nintendo 64 Game
#
# ROMs:
#   make base        -> legend_of_elya.z64             (CPU LLM, standalone)
#   make base-rsp    -> legend_of_elya_rsp.z64         (CPU+RSP LLM, standalone)
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
SGAI_BITS ?= 8
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

# Extra flags for one-off verification builds, e.g. EXTRA=-DBOOT_PROBE.
# APPENDED, never assigned: `make CFLAGS=...` on the command line silently
# discards every knob above it (that mistake produced an 8,866,956-byte bss
# before it was caught).
CFLAGS += $(EXTRA)

all: legend_of_elya.z64 legend_of_elya_rsp.z64 legend_of_elya_mining.z64 legend_of_elya_rpc_mining.z64 legend_of_elya_3d.z64


# --- Base ROM (CPU LLM, standalone, no Pico) ---
base: legend_of_elya.z64

$(BUILD_DIR)/legend_of_elya.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/legend_of_elya.elf: $(BUILD_DIR)/legend_of_elya.o $(BUILD_DIR)/nano_gpt.o

legend_of_elya.z64: N64_ROM_TITLE="Legend of Elya"
legend_of_elya.z64: $(BUILD_DIR)/legend_of_elya.dfs

# --- RSP-accelerated ROM (CPU+RSP LLM, standalone, no Pico) ---
base-rsp: legend_of_elya_rsp.z64

$(BUILD_DIR)/legend_of_elya_rsp.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/nano_gpt_rsp.o: nano_gpt.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/matmul_rsp_drv.o: matmul_rsp_drv.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/legend_of_elya_rsp.o: legend_of_elya.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RSP_MATMUL -o $@ $<

$(BUILD_DIR)/legend_of_elya_rsp.elf: $(BUILD_DIR)/legend_of_elya_rsp.o $(BUILD_DIR)/nano_gpt_rsp.o $(BUILD_DIR)/matmul_rsp_drv.o $(BUILD_DIR)/rsp_matmul.o

legend_of_elya_rsp.z64: N64_ROM_TITLE="Elya RSP"
legend_of_elya_rsp.z64: $(BUILD_DIR)/legend_of_elya_rsp.dfs

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

# --- Reference CLI (x86/ARM host) ---
reference: reference_cli

reference_cli: reference_cli.c
	$(CC) -o reference_cli reference_cli.c -lm -std=c99

.PHONY: reference
	rm -rf $(BUILD_DIR) legend_of_elya.z64 legend_of_elya_rsp.z64 legend_of_elya_mining.z64 legend_of_elya_rpc_mining.z64 legend_of_elya_3d.z64 reference_cli

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all base base-rsp mining rpc-mining 3d reference clean

reference: reference_cli
reference_cli: reference_cli.c
	gcc -O3 -std=c99 -Wall -o $@ $< -lm

