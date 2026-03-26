/**
 * Multi-NPC Mode — Expansion Pak 8MB Support
 * 
 * Enables 3 simultaneous NPC sessions with shared weights
 * but separate KV caches and personas.
 * 
 * Memory Map (8MB Expansion Pak):
 *   0x80000000 - 0x80100000 : Code + stack (1 MB)
 *   0x80100000 - 0x80380000 : Shared model weights (2.5 MB, ROM-mapped)
 *   0x80380000 - 0x804C8000 : NPC 0 KV cache (Sophia, ~1.3 MB)
 *   0x804C8000 - 0x80610000 : NPC 1 KV cache (Forge Master, ~1.3 MB)
 *   0x80610000 - 0x80758000 : NPC 2 KV cache (Librarian, ~1.3 MB)
 *   0x80758000 - 0x80800000 : Scratch/display buffers (~672 KB)
 *
 * Memory Map (4MB, single NPC fallback):
 *   0x80000000 - 0x80080000 : Code + stack (512 KB)
 *   0x80080000 - 0x80300000 : Shared model weights (2.5 MB)
 *   0x80300000 - 0x80400000 : NPC 0 KV cache only (1 MB, reduced ctx=64)
 * 
 * Bounty: legend-of-elya-n64 #5 (200 RTC)
 */

#ifndef MULTI_NPC_H
#define MULTI_NPC_H

#include "nano_gpt.h"

/* ── Constants ───────────────────────────────────────────────── */

#define MAX_NPCS_8MB      3
#define MAX_NPCS_4MB      1
#define NPC_NAME_LEN      32
#define NPC_PREFIX_LEN    128
#define NPC_COLOR_DEFAULT 0xFFFFFFFF

/* ── NPC Persona ─────────────────────────────────────────────── */

typedef struct {
    char      name[NPC_NAME_LEN];
    char      prefix[NPC_PREFIX_LEN];   /* Persona prompt prepended to context */
    uint32_t  text_color;               /* RGBA for display */
    uint32_t  name_color;               /* RGBA for name label */
    uint8_t   temperature_q8;           /* Personality temperature (0=cold, 255=wild) */
} NPCPersona;

/* ── NPC Session ─────────────────────────────────────────────── */

typedef struct {
    SGAIState   state;                  /* Inference state (shares weights ptr) */
    NPCPersona  persona;               /* Character definition */
    uint8_t     active;                 /* Is this NPC slot active? */
    uint8_t     last_output[256];       /* Last generated response */
    int         last_output_len;
} NPCSession;

/* ── Multi-NPC Manager ───────────────────────────────────────── */

typedef struct {
    NPCSession  npcs[MAX_NPCS_8MB];
    int         num_npcs;               /* 1 (4MB) or 3 (8MB) */
    int         active_npc;             /* Currently selected NPC (0-2) */
    int         has_expansion_pak;      /* RAM detection result */
    uint32_t    total_ram_bytes;        /* Detected RAM size */
    const void *shared_weights;         /* ROM-mapped weights (shared) */
} MultiNPCManager;

/* ── Default Personas ────────────────────────────────────────── */

/* Defined in multi_npc.c */
extern const NPCPersona PERSONA_SOPHIA;
extern const NPCPersona PERSONA_FORGE_MASTER;
extern const NPCPersona PERSONA_LIBRARIAN;

/* ── API ─────────────────────────────────────────────────────── */

/**
 * Detect Expansion Pak and initialize NPCs.
 * Returns number of active NPCs (1 or 3).
 */
int multi_npc_init(MultiNPCManager *mgr, const void *rom_weights);

/**
 * Reset all NPC sessions (clear KV caches).
 */
void multi_npc_reset(MultiNPCManager *mgr);

/**
 * Select active NPC by index (0-2).
 * Returns 0 on success, -1 if index out of range.
 */
int multi_npc_select(MultiNPCManager *mgr, int index);

/**
 * Generate response from the active NPC.
 * Prompt is prepended with the NPC's persona prefix.
 * Returns number of tokens generated.
 */
int multi_npc_generate(MultiNPCManager *mgr, 
                       const uint8_t *prompt, int prompt_len,
                       uint8_t *output, int max_tokens);

/**
 * Generate responses from ALL NPCs for the same prompt.
 * Used for the demo mode showing different personalities.
 * Fills output buffers for each NPC.
 */
int multi_npc_generate_all(MultiNPCManager *mgr,
                           const uint8_t *prompt, int prompt_len,
                           uint8_t outputs[][256], int *output_lens,
                           int max_tokens);

/**
 * Get persona info for display.
 */
const NPCPersona *multi_npc_get_persona(const MultiNPCManager *mgr, int index);

/**
 * Get the number of active NPCs.
 */
int multi_npc_count(const MultiNPCManager *mgr);

/* ── RAM Detection ───────────────────────────────────────────── */

/**
 * Detect N64 RAM size at boot.
 * Returns total bytes (4194304 for 4MB, 8388608 for 8MB).
 */
uint32_t detect_ram_size(void);

/**
 * Check if Expansion Pak is present.
 */
int has_expansion_pak(void);

#endif /* MULTI_NPC_H */
