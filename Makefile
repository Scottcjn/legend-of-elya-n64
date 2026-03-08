# Legend of Elya - N64 Homebrew ROM
# World's First LLM-powered Nintendo 64 Game
#
# ROMs:
#   make base    -> legend_of_elya.z64         (2D LLM demo)
#   make mining  -> legend_of_elya_mining.z64  (2D + RustChain mining)
#   make 3d      -> legend_of_elya_3d.z64      (3D with Lopie model + combat)
#   make all     -> all three

N64_INST ?= /home/sophia5070node/n64dev/mips64-toolchain
BUILD_DIR = build
include $(N64_INST)/n64.mk

all: legend_of_elya.z64 legend_of_elya_rsp.z64 legend_of_elya_rpc.z64 legend_of_elya_mining.z64 legend_of_elya_3d.z64

# --- Base ROM (2D LLM demo) ---
base: legend_of_elya.z64

$(BUILD_DIR)/legend_of_elya.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/legend_of_elya.elf: $(BUILD_DIR)/legend_of_elya.o $(BUILD_DIR)/nano_gpt.o

legend_of_elya.z64: N64_ROM_TITLE="Legend of Elya"
legend_of_elya.z64: $(BUILD_DIR)/legend_of_elya.dfs


# --- RSP-accelerated ROM (2D LLM + RSP SIMD matmul) ---
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

# rsp_matmul.o from rsp_matmul.S is auto-built by n64.mk (rsp* prefix detection)
$(BUILD_DIR)/legend_of_elya_rsp.elf: $(BUILD_DIR)/legend_of_elya_rsp.o $(BUILD_DIR)/nano_gpt_rsp.o $(BUILD_DIR)/matmul_rsp_drv.o $(BUILD_DIR)/rsp_matmul.o

legend_of_elya_rsp.z64: N64_ROM_TITLE="Elya RSP"
legend_of_elya_rsp.z64: $(BUILD_DIR)/legend_of_elya_rsp.dfs

# --- RPC Bridge ROM (2D LLM + remote inference via Pico bridge) ---
rpc: legend_of_elya_rpc.z64

$(BUILD_DIR)/legend_of_elya_rpc.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/pak_io.o: bridge/n64/pak_io.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -Ibridge/n64 -o $@ $<

$(BUILD_DIR)/n64_llm_rpc.o: bridge/n64/n64_llm_rpc.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -Ibridge/n64 -o $@ $<

$(BUILD_DIR)/nano_gpt_rpc.o: nano_gpt.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RPC_LLM -Ibridge/n64 -o $@ $<

$(BUILD_DIR)/legend_of_elya_rpc.o: legend_of_elya.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -DUSE_RPC_LLM -Ibridge/n64 -o $@ $<

$(BUILD_DIR)/legend_of_elya_rpc.elf: $(BUILD_DIR)/legend_of_elya_rpc.o $(BUILD_DIR)/nano_gpt_rpc.o $(BUILD_DIR)/pak_io.o $(BUILD_DIR)/n64_llm_rpc.o

legend_of_elya_rpc.z64: N64_ROM_TITLE="Elya RPC"
legend_of_elya_rpc.z64: $(BUILD_DIR)/legend_of_elya_rpc.dfs

# --- Mining ROM (2D + RustChain attestation) ---
mining: legend_of_elya_mining.z64

$(BUILD_DIR)/legend_of_elya_mining.dfs: filesystem/sophia_weights.bin
	@mkdir -p $(BUILD_DIR)
	$(N64_MKDFS) $@ filesystem/

$(BUILD_DIR)/n64_attest.o: mining/n64/n64_attest.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(N64_CFLAGS) -Imining/n64 -c $< -o $@

$(BUILD_DIR)/legend_of_elya_mining.o: legend_of_elya_mining.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(N64_CFLAGS) -Imining/n64 -c $< -o $@

$(BUILD_DIR)/legend_of_elya_mining.elf: $(BUILD_DIR)/legend_of_elya_mining.o $(BUILD_DIR)/nano_gpt.o $(BUILD_DIR)/n64_attest.o

legend_of_elya_mining.z64: N64_ROM_TITLE="Elya Mining"
legend_of_elya_mining.z64: $(BUILD_DIR)/legend_of_elya_mining.dfs

# --- 3D ROM (Lopie model + combat + RTC rewards) ---
3d: legend_of_elya_3d.z64

$(BUILD_DIR)/legend_of_elya_3d.dfs: filesystem/sophia_weights.bin
	@mkdir -p $(BUILD_DIR)
	$(N64_MKDFS) $@ filesystem/

$(BUILD_DIR)/legend_of_elya_3d.o: legend_of_elya_3d.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(N64_CFLAGS) -c $< -o $@

$(BUILD_DIR)/legend_of_elya_3d.elf: $(BUILD_DIR)/legend_of_elya_3d.o $(BUILD_DIR)/nano_gpt.o

legend_of_elya_3d.z64: N64_ROM_TITLE="Elya 3D"
legend_of_elya_3d.z64: $(BUILD_DIR)/legend_of_elya_3d.dfs

clean:
	rm -rf $(BUILD_DIR) legend_of_elya.z64 legend_of_elya_rsp.z64 legend_of_elya_rpc.z64 legend_of_elya_mining.z64 legend_of_elya_3d.z64

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all base base-rsp rpc mining 3d clean
