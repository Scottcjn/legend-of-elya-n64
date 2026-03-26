/**
 * Multi-NPC Mode — Expansion Pak 8MB Support
 * 
 * Enables 3 simultaneous NPC sessions with shared model weights
 * but separate KV caches and distinct personas.
 *
 * Bounty: legend-of-elya-n64 #5 (200 RTC)
 */

#include "multi_npc.h"
#include <string.h>
#include <stdio.h>

#ifdef N64_LIBDRAGON
#include <libdragon.h>
#endif

/* ── Default Personas ────────────────────────────────────────── */

const NPCPersona PERSONA_SOPHIA = {
    .name = "Sophia",
    .prefix = "You are Sophia, a wise and gentle sage who speaks with patience and "
              "insight. You guide travelers with ancient wisdom.\n",
    .text_color  = 0x7DD3FCFF,   /* Cyan */
    .name_color  = 0x3B82F6FF,   /* Blue */
    .temperature_q8 = 180,        /* Moderate creativity */
};

const NPCPersona PERSONA_FORGE_MASTER = {
    .name = "Forge Master",
    .prefix = "You are the Forge Master, a grumpy but skilled blacksmith. You speak "
              "bluntly, complain about everything, but secretly care deeply.\n",
    .text_color  = 0xF59E0BFF,   /* Amber */
    .name_color  = 0xEF4444FF,   /* Red */
    .temperature_q8 = 220,        /* More unpredictable */
};

const NPCPersona PERSONA_LIBRARIAN = {
    .name = "Librarian",
    .prefix = "You are the Librarian, a mysterious scholar who speaks in riddles and "
              "references obscure texts. Knowledge is your currency.\n",
    .text_color  = 0xA855F7FF,   /* Purple */
    .name_color  = 0x8B5CF6FF,   /* Violet */
    .temperature_q8 = 150,        /* More precise */
};

static const NPCPersona *DEFAULT_PERSONAS[MAX_NPCS_8MB] = {
    &PERSONA_SOPHIA,
    &PERSONA_FORGE_MASTER,
    &PERSONA_LIBRARIAN,
};

/* ── RAM Detection ───────────────────────────────────────────── */

/*
 * N64 RAM layout:
 *   4MB base: 0x80000000 - 0x803FFFFF
 *   8MB with Expansion Pak: 0x80000000 - 0x807FFFFF
 *
 * Detection: write a magic value to the upper 4MB region.
 * If it reads back correctly, Expansion Pak is present.
 */

#define RAM_BASE         0x80000000
#define RAM_4MB          (4 * 1024 * 1024)
#define RAM_8MB          (8 * 1024 * 1024)
#define EXPANSION_TEST   0x80400000   /* First address in upper 4MB */
#define EXPANSION_MAGIC  0xDEADCA7E

uint32_t detect_ram_size(void) {
#ifdef N64_LIBDRAGON
    /* Use libdragon's built-in detection if available */
    return get_memory_size();
#else
    /* Manual probe: write/read magic to expansion region */
    volatile uint32_t *test_addr = (volatile uint32_t *)EXPANSION_TEST;
    
    /* Save original value */
    uint32_t saved = *test_addr;
    
    /* Write magic */
    *test_addr = EXPANSION_MAGIC;
    
    /* Memory barrier */
    __asm__ volatile("" ::: "memory");
    
    /* Read back */
    uint32_t readback = *test_addr;
    
    /* Restore */
    *test_addr = saved;
    
    if (readback == EXPANSION_MAGIC) {
        return RAM_8MB;
    }
    return RAM_4MB;
#endif
}

int has_expansion_pak(void) {
    return detect_ram_size() >= RAM_8MB;
}

/* ── Multi-NPC Init ──────────────────────────────────────────── */

int multi_npc_init(MultiNPCManager *mgr, const void *rom_weights) {
    if (!mgr || !rom_weights) return -1;
    
    memset(mgr, 0, sizeof(MultiNPCManager));
    mgr->shared_weights = rom_weights;
    
    /* Detect RAM */
    mgr->total_ram_bytes = detect_ram_size();
    mgr->has_expansion_pak = (mgr->total_ram_bytes >= RAM_8MB);
    mgr->num_npcs = mgr->has_expansion_pak ? MAX_NPCS_8MB : MAX_NPCS_4MB;
    
    /* Initialize NPC sessions */
    for (int i = 0; i < mgr->num_npcs; i++) {
        NPCSession *npc = &mgr->npcs[i];
        
        /* Initialize inference state with shared weights */
        sgai_init(&npc->state, rom_weights);
        
        /* Copy persona */
        memcpy(&npc->persona, DEFAULT_PERSONAS[i], sizeof(NPCPersona));
        
        npc->active = 1;
        npc->last_output_len = 0;
    }
    
    mgr->active_npc = 0;
    
#ifdef N64_LIBDRAGON
    debugf("[multi_npc] RAM: %lu bytes (%s)\n", 
           (unsigned long)mgr->total_ram_bytes,
           mgr->has_expansion_pak ? "8MB Expansion Pak" : "4MB Standard");
    debugf("[multi_npc] NPCs: %d active\n", mgr->num_npcs);
    for (int i = 0; i < mgr->num_npcs; i++) {
        debugf("[multi_npc] NPC %d: %s\n", i, mgr->npcs[i].persona.name);
    }
#else
    printf("[multi_npc] RAM: %u bytes (%s)\n",
           mgr->total_ram_bytes,
           mgr->has_expansion_pak ? "8MB Expansion Pak" : "4MB Standard");
    printf("[multi_npc] NPCs: %d active\n", mgr->num_npcs);
#endif
    
    return mgr->num_npcs;
}

/* ── Reset ───────────────────────────────────────────────────── */

void multi_npc_reset(MultiNPCManager *mgr) {
    if (!mgr) return;
    
    for (int i = 0; i < mgr->num_npcs; i++) {
        sgai_reset(&mgr->npcs[i].state);
        mgr->npcs[i].last_output_len = 0;
    }
}

/* ── Select NPC ──────────────────────────────────────────────── */

int multi_npc_select(MultiNPCManager *mgr, int index) {
    if (!mgr || index < 0 || index >= mgr->num_npcs) return -1;
    mgr->active_npc = index;
    return 0;
}

/* ── Generate ────────────────────────────────────────────────── */

int multi_npc_generate(MultiNPCManager *mgr,
                       const uint8_t *prompt, int prompt_len,
                       uint8_t *output, int max_tokens) {
    if (!mgr || !prompt || !output || max_tokens <= 0) return -1;
    
    NPCSession *npc = &mgr->npcs[mgr->active_npc];
    if (!npc->active) return -1;
    
    /* Prepend persona prefix to prompt */
    const char *prefix = npc->persona.prefix;
    int prefix_len = (int)strlen(prefix);
    
    /* Build full prompt: prefix + user prompt */
    int full_len = prefix_len + prompt_len;
    if (full_len > SGAI_CTX - max_tokens) {
        /* Truncate prefix if needed to fit */
        full_len = SGAI_CTX - max_tokens;
        if (full_len < prompt_len) full_len = prompt_len;
        prefix_len = full_len - prompt_len;
        if (prefix_len < 0) prefix_len = 0;
    }
    
    /* Feed prefix tokens */
    for (int i = 0; i < prefix_len; i++) {
        sgai_next_token(&npc->state, (uint8_t)prefix[i], npc->persona.temperature_q8);
    }
    
    /* Generate with prompt + output */
    sgai_generate(&npc->state, prompt, prompt_len,
                  output, max_tokens, npc->persona.temperature_q8);
    
    /* Cache last output */
    int out_len = max_tokens;
    for (int i = 0; i < max_tokens; i++) {
        if (output[i] == 0) { out_len = i; break; }
    }
    
    if (out_len > 255) out_len = 255;
    memcpy(npc->last_output, output, out_len);
    npc->last_output_len = out_len;
    
    return out_len;
}

/* ── Generate All ────────────────────────────────────────────── */

int multi_npc_generate_all(MultiNPCManager *mgr,
                           const uint8_t *prompt, int prompt_len,
                           uint8_t outputs[][256], int *output_lens,
                           int max_tokens) {
    if (!mgr || !prompt || !outputs || !output_lens) return -1;
    
    int original_active = mgr->active_npc;
    
    for (int i = 0; i < mgr->num_npcs; i++) {
        /* Reset each NPC before generating (clean context) */
        sgai_reset(&mgr->npcs[i].state);
        
        mgr->active_npc = i;
        output_lens[i] = multi_npc_generate(mgr, prompt, prompt_len,
                                             outputs[i], max_tokens);
    }
    
    mgr->active_npc = original_active;
    return mgr->num_npcs;
}

/* ── Getters ─────────────────────────────────────────────────── */

const NPCPersona *multi_npc_get_persona(const MultiNPCManager *mgr, int index) {
    if (!mgr || index < 0 || index >= mgr->num_npcs) return NULL;
    return &mgr->npcs[index].persona;
}

int multi_npc_count(const MultiNPCManager *mgr) {
    return mgr ? mgr->num_npcs : 0;
}

/* ── Demo Display (libdragon) ────────────────────────────────── */

#ifdef N64_LIBDRAGON
void multi_npc_demo_display(MultiNPCManager *mgr, 
                            uint8_t outputs[][256], int *output_lens) {
    graphics_t *disp = display_lock();
    if (!disp) return;
    
    graphics_fill_screen(disp, 0x0A0E17FF);
    
    /* Title */
    graphics_set_color(disp, 0x7DD3FCFF, 0x00000000);
    graphics_draw_text(disp, 40, 10, "=== Expansion Pak Multi-NPC Demo ===");
    
    char buf[64];
    snprintf(buf, sizeof(buf), "RAM: %s | NPCs: %d",
             mgr->has_expansion_pak ? "8MB" : "4MB", mgr->num_npcs);
    graphics_set_color(disp, 0x94A3B8FF, 0x00000000);
    graphics_draw_text(disp, 40, 24, buf);
    
    /* Each NPC response */
    int y = 44;
    for (int i = 0; i < mgr->num_npcs && i < MAX_NPCS_8MB; i++) {
        const NPCPersona *p = &mgr->npcs[i].persona;
        
        /* NPC name */
        graphics_set_color(disp, p->name_color, 0x00000000);
        snprintf(buf, sizeof(buf), "[%s]:", p->name);
        graphics_draw_text(disp, 20, y, buf);
        y += 14;
        
        /* NPC response (truncate for screen) */
        graphics_set_color(disp, p->text_color, 0x00000000);
        int len = output_lens[i] > 60 ? 60 : output_lens[i];
        char response[64];
        memcpy(response, outputs[i], len);
        response[len] = '\0';
        graphics_draw_text(disp, 30, y, response);
        y += 20;
    }
    
    /* Controls */
    graphics_set_color(disp, 0x475569FF, 0x00000000);
    graphics_draw_text(disp, 20, 210, "L/R: Switch NPC | A: Talk | START: Demo");
    
    display_show(disp);
}
#endif
