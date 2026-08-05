// SPDX-License-Identifier: MIT
/**
 * Legend of Elya - Nintendo 64 Homebrew
 * World's First LLM-powered N64 Game
 *
 * FIXED: Single-buffer rendering - rdpq_detach_wait() + graphics_draw_text()
 * eliminates the console_render() double-buffer flicker.
 *
 * v2: Legend of Elya splash screen with balloons + per-token tok/s indicator
 * v3: LOZ Dungeon Theme square-wave music via libdragon audio
 */

#include <libdragon.h>
#include <graphics.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "nano_gpt.h"

#ifdef USE_RPC_LLM
#include "n64_llm_rpc.h"
#endif

// ─── Room & NPC System ───────────────────────────────────────────────────────

typedef enum {
    ROOM_DUNGEON = 0,    // Original dungeon — Sophia Elya
    ROOM_LIBRARY,        // Eastern library — The Librarian
    ROOM_FORGE,          // Western forge — Grumpy Blacksmith
    ROOM_COUNT
} RoomID;

// Room exit edges — which direction leads where
typedef struct {
    int8_t north;   // room ID or -1
    int8_t south;
    int8_t east;
    int8_t west;
} RoomExits;

static const RoomExits ROOM_MAP[ROOM_COUNT] = {
    [ROOM_DUNGEON] = { -1,         -1,  ROOM_LIBRARY,  ROOM_FORGE },
    [ROOM_LIBRARY] = { -1,         -1, -1,             ROOM_DUNGEON },
    [ROOM_FORGE]   = { -1,         -1,  ROOM_DUNGEON, -1 },
};

static const char *ROOM_NAMES[ROOM_COUNT] = {
    "Crystal Dungeon",
    "Arcane Library",
    "Ember Forge",
};

// NPC profile — persona injected before player prompt to steer model output
typedef struct {
    const char *name;           // display name
    const char *persona_prefix; // prepended to prompt (max ~16 chars to fit CTX)
    int         sprite_x;       // x position in room
    int         sprite_y_base;  // base y (before bob)
    uint8_t     color_r, color_g, color_b;  // primary palette
    uint8_t     dress_r, dress_g, dress_b;  // dress/outfit color
    RoomID      room;           // which room this NPC lives in
} NPCProfile;

#define NPC_COUNT 3
static const NPCProfile NPC_PROFILES[NPC_COUNT] = {
    { // Sophia Elya — guide of the realm (original NPC)
        "Sophia Elya", "sage says: ",
        204, 72,
        80, 50, 120,   // purple torso
        60, 30, 100,   // purple dress
        ROOM_DUNGEON
    },
    { // Aldric — mysterious librarian
        "Aldric the Keeper", "scholar says: ",
        210, 76,
        40, 60, 100,   // dark blue torso
        30, 40,  80,   // dark blue robe
        ROOM_LIBRARY
    },
    { // Brunhild — grumpy blacksmith
        "Brunhild", "smith says: ",
        190, 74,
        120, 80, 40,   // brown torso
        100, 60, 20,   // brown apron
        ROOM_FORGE
    },
};

// Contextual dialog options per NPC (D-pad selectable)
#define DIALOG_OPTIONS 4
static const char *NPC_DIALOG_OPTIONS[NPC_COUNT][DIALOG_OPTIONS] = {
    { // Sophia
        "Who are you?: ",
        "What lurks here?: ",
        "Tell me a secret.: ",
        "What is RustChain?: ",
    },
    { // Aldric
        "What is your name?: ",
        "What is proof of antiquity?: ",
        "What is MIPS?: ",
        "How big is your model?: ",
    },
    { // Brunhild
        "What do I need here?: ",
        "What is the G4?: ",
        "What is AltiVec?: ",
        "What is vec_perm?: ",
    },
};

// Canned fallback responses per NPC (when weights not loaded)
static const char *NPC_CANNED[NPC_COUNT][DIALOG_OPTIONS] = {
    { // Sophia
        "I am Sophia Elya, guide of the realm.",
        "Skeletons and bats haunt these halls.",
        "Seek the silver key behind the great statue.",
        "RustChain proves old silicon still matters.",
    },
    { // Aldric
        "I am Aldric, keeper of forbidden tomes.",
        "Antiquity rewards those with vintage iron.",
        "MIPS runs the VR4300 inside this cartridge.",
        "An 819K transformer, fit in 8 megabytes.",
    },
    { // Brunhild
        "A stronger blade. Now leave me be.",
        "The G4 with AltiVec, two point five times.",
        "AltiVec: vector power in old PowerPC.",
        "Vec_perm shuffles 16 bytes in one cycle.",
    },
};

// ─── Game State ───────────────────────────────────────────────────────────────

typedef enum {
    STATE_ANNIVERSARY,   // Legend of Elya splash screen
    STATE_TITLE,
    STATE_DUNGEON,
    STATE_DIALOG_SELECT, // D-pad to pick dialog option
    STATE_DIALOG,
    STATE_GENERATING,
    STATE_KEYBOARD,      // Virtual keyboard for player text input
    STATE_ROOM_TRANSITION, // brief fade between rooms
} GameState;

// Forge spark particle (static pool, no alloc)
#define MAX_SPARKS 12
typedef struct {
    int16_t x, y;
    int8_t  dx, dy;
    uint8_t life;    // frames remaining
} Spark;

typedef struct {
    GameState state;
    int dialog_char;
    int dialog_done;
    uint8_t dialog_buf[128];
    int dialog_len;
    int frame;
    uint32_t anniversary_cp0;  // CP0 Count at boot for real-time splash duration
    // Room system
    RoomID   current_room;
    int      transition_timer;  // frames remaining in room transition fade
    RoomID   transition_target; // room we're transitioning to
    int      dialog_select_idx; // currently highlighted dialog option (0-3)
    int      current_npc;       // NPC index in current room (-1 if none)
    // Player position (for room edge detection)
    int      player_x;
    int      player_y;
    // Forge sparks
    Spark    sparks[MAX_SPARKS];
    // AI
    SGAIState ai;
    SGAIKVCache kv;
    int ai_ready;
    /* Set when the weight blob was found but did not fit wbuf, so the reason
     * the game fell back to canned dialogue is inspectable. 0 when unused. */
    int ai_load_failed_size;
    int ai_load_capacity;
    int prompt_idx;
    // Per-frame generation state (enables tok/s display)
    uint8_t gen_pbuf[64];   // copy of current prompt bytes
    int gen_plen;           // prompt byte count
    int gen_ppos;           // bytes fed so far (prompt phase)
    uint8_t gen_last_tok;   // last token for chaining
    int gen_out_count;      // output tokens generated
    int gen_start_frame;    // frame when output phase began
    float gen_toks_sec;     // computed tokens/second
    // Music sequencer
    int music_note_idx;     // current note in sequence
    int music_sample_pos;   // samples elapsed in current note
    int music_phase;        // square wave phase accumulator
    // Combat & HUD
    int attack_timer;       // frames remaining in attack animation (0 = idle)
    int attack_target;      // 0 = stalfos, 1 = keese
    int hearts;             // half-heart count: 8 = 4 full hearts, 0 = dead
    int magic;              // magic bar 0-128 (128 = full)
    // Performance monitoring
    uint32_t perf_frame_start;   // CP0 COUNT at frame start
    uint32_t perf_gen_cycles;    // cycles spent in sgai_next_token this frame
    uint32_t perf_gen_total_us;  // total generation time in microseconds
    uint32_t perf_gen_start_us;  // timestamp when generation started
    float    perf_cpu_pct;       // CPU% used by inference (0-100)
    float    perf_toks_precise;  // precise tok/s using cycle counter
    int      perf_show;          // 1 = show performance overlay
    // Virtual keyboard
    int      kb_row;             // cursor row (0-3)
    int      kb_col;             // cursor column (0-9)
    char     kb_input[64];       // player input buffer
    int      kb_len;             // current input length
    int      kb_debounce;        // frame counter for input debounce
#ifdef USE_RPC_LLM
    int      rpc_active;         // 1 = bridge detected, using RPC
    int      rpc_pending;        // 1 = waiting for RPC response
    uint32_t rpc_send_us;        // timestamp when RPC request sent
#endif
} GameCtx;

static GameCtx G;

/* Forward declarations for cross-referenced functions */
static void start_dialog_from_prompt(int npc_idx, const char *prompt, int use_persona);

/* N64 hardware entropy — XOR CPU cycle counter low bits with frame,
 * last token, AND prompt_idx (sequential counter).
 * prompt_idx is the critical fallback: emulators run TICKS_READ()
 * deterministically, so we rely on prompt_idx advancing each A-press
 * to guarantee a different topic every conversation.
 * On real hardware, TICKS jitter adds extra unpredictability. */
/* N64 R4300i runs at 93.75 MHz. CP0 COUNT increments every other cycle = 46.875 MHz.
 * So 1 microsecond = ~46.875 counts.  We use TICKS_READ() which reads CP0 COUNT. */
#define CYCLES_TO_US(c)   ((uint32_t)(c) / 47)
#define US_TO_CYCLES(us)  ((uint32_t)(us) * 47)

/* One NTSC frame = 1/60s = 16667us = ~781,250 CP0 counts */
#define FRAME_CYCLES  781250

#define N64_ENTROPY() ((uint32_t)(TICKS_READ())                    \
                       ^ ((uint32_t)G.frame << 3)                  \
                       ^ ((uint32_t)G.gen_last_tok * 2654435761u)  \
                       ^ ((uint32_t)G.prompt_idx  * 40503u))

/* Post-generation output filter — remove training data artifacts.
 * "helpmeet" was in the QA training corpus but is wrong for a game.
 * In-place replacement keeps same buffer length (no shift needed).
 * "guardian" is exactly 8 chars = same as "helpmeet". */
static void filter_dialog_buf(void) {
    char *buf = (char *)G.dialog_buf;
    /* replace "helpmeet" → "guardian" (8 chars = 8 chars, in-place safe) */
    char *p = buf;
    while ((p = strstr(p, "helpmeet")) != NULL) {
        memcpy(p, "guardian", 8);
        p += 8;
    }
    /* replace "Flameholder" → "Elyan Labs " (11 chars = 11 chars) */
    p = buf;
    while ((p = strstr(p, "Flameholder")) != NULL) {
        memcpy(p, "Elyan Labs ", 11);
        p += 11;
    }
}

// Fallback responses when weights not available (no helpmeet/title language)
static const char *CANNED[] = {
    "I am Sophia Elya, guide of the realm.",
    "Vintage hardware earns real RTC rewards.",
    "The G4 and G5 are my favorite miners.",
    "RustChain proves old silicon still matters.",
    "The VR4300 inside this cartridge is real.",
    "Seek the silver key behind the great statue.",
    "Many adventurers have braved these halls.",
    "Elyan Labs built me to run on 8 megabytes.",
    "The RSP and RDP team up to draw these halls.",
    "PowerPC G4 earns two point five times RTC.",
    "Three attestation nodes guard the network.",
    "Ancient silicon dreams in proof of antiquity.",
    "The legend of Elya endures in silicon.",
    "This dungeon holds secrets only brave find.",
    "I was trained on 50 thousand steps of lore.",
    "Press A near me anytime, weary traveler.",
};
#define N_CANNED 16

/* Prompt pool — exact QA_PAIRS keys from training data.
 * v5 CTX=64 gives room for prompts up to ~20 chars,
 * leaving 44+ tokens for Sophia's response.
 * Entropy from N64 CPU oscillator selects the prompt each conversation. */
static const char *PROMPTS[] = {
    /* identity */
    "Who are you?: ",
    "What is your name?: ",
    "Where are you from?: ",
    "What is your purpose?: ",
    /* dungeon / game */
    "What lurks here?: ",
    "How do I proceed?: ",
    "What do I need here?: ",
    "Tell me a secret.: ",
    /* RustChain */
    "What is RustChain?: ",
    "What is RTC?: ",
    "How do I earn RTC?: ",
    "What is a node?: ",
    "What is proof of antiquity?: ",
    "What is epoch?: ",
    /* hardware */
    "What is the G4?: ",
    "What is the G5?: ",
    "What is POWER8?: ",
    "What is AltiVec?: ",
    "What is vec_perm?: ",
    "What runs this ROM?: ",
    "What is the VR4300?: ",
    /* N64 lore */
    "What console is this?: ",
    "What is MIPS?: ",
    "How big is your model?: ",
    "What language runs you?: ",
    /* Elya lore */
    "What is Elyan Labs?: ",
    "Who is the Helpmeet?: ",
    "What is the Study?: ",
    "Who guards the realm?: ",
    "What is the Triforce?: ",
    "Who is the Flameholder?: ",
    "What is proof of work?: ",
};
#define N_PROMPTS 32

// ─── Music: Legend of Elya Theme (Original) ──────────────────────────────────
// Original composition for Legend of Elya.
// Key: A minor / C major modal. BPM ~110. Mysterious dungeon atmosphere
// with a wistful, exploratory feel. Rising minor arpeggio opening,
// chromatic tension, then resolution through the natural minor scale.
// Notes: A4=440, C5=523, D5=587, E5=659, F5=698, G5=784, A5=880,
//        B4=494, Bb4=466, G4=392, F4=349, E4=330
// 0 = rest/silence

#define MUSIC_FREQ         22050        // 22kHz, plenty for square wave
#define MUSIC_BPM          110
#define MUSIC_EIGHTH       (MUSIC_FREQ * 60 / (MUSIC_BPM * 2))   // ~6013 samples
#define MUSIC_ATTACK       350          // samples of fade-in per note
#define MUSIC_DECAY_START  (MUSIC_EIGHTH - 450) // start fade-out near end

static const uint16_t DUNGEON_FREQ[] = {
    // Phrase 1: mysterious ascending A minor arpeggio
    330,  440,  523,  659,   // E4 A4 C5 E5  (Am arpeggio, wistful)
    523,  659,  523,    0,   // C5 E5 C5 rest
    // Phrase 2: tension — chromatic climb F5→G5, then descend
    698,  784,  880,    0,   // F5 G5 A5 rest  (climax, high A)
    784,  698,    0,  587,   // G5 F5 rest D5   (falling back)
    // Phrase 3: melancholy descent through natural minor
    659,  587,  523,    0,   // E5 D5 C5 rest  (stepwise descent)
    494,  523,  587,  659,   // B4 C5 D5 E5   (re-ascend, hope)
    // Phrase 4: resolution — settle into tonic with gentle fade
    880,    0,  659,    0,   // A5 rest E5 rest  (octave call)
    523,  440,    0,    0,   // C5 A4 rest rest  (home)
};
#define DUNGEON_LEN 32

// Duration: 1=eighth note, 2=quarter note (held)
static const uint8_t DUNGEON_DUR[] = {
    // Phrase 1
    1, 1, 1, 2,   // E4 A4 C5 E5(held)  (lingering on the 5th)
    1, 2, 1, 1,   // C5 E5(held) C5 rest
    // Phrase 2
    1, 1, 2, 1,   // F5 G5 A5(held) rest  (sustained climax)
    1, 2, 1, 1,   // G5 F5(held) rest D5
    // Phrase 3
    1, 1, 2, 1,   // E5 D5 C5(held) rest
    1, 1, 1, 1,   // B4 C5 D5 E5  (quick run back up)
    // Phrase 4
    2, 1, 2, 1,   // A5(held) rest E5(held) rest
    2, 2, 1, 1,   // C5(held) A4(held) rest rest
};

static void music_update(void) {
    if (!audio_can_write()) return;

    short *buf = audio_write_begin();
    int nsamples = audio_get_buffer_length();

    for (int i = 0; i < nsamples; i++) {
        int note_samples = (int)DUNGEON_DUR[G.music_note_idx] * MUSIC_EIGHTH;
        uint16_t freq    = DUNGEON_FREQ[G.music_note_idx];

        int16_t sample = 0;
        if (freq > 0) {
            int period = MUSIC_FREQ / (int)freq;
            if (period > 0) {
                // Square wave
                int16_t amp = 5000;
                // Simple attack/decay envelope to avoid clicks
                if (G.music_sample_pos < MUSIC_ATTACK)
                    amp = (int16_t)((int32_t)amp * G.music_sample_pos / MUSIC_ATTACK);
                else if (G.music_sample_pos > MUSIC_DECAY_START)
                    amp = (int16_t)((int32_t)amp * (note_samples - G.music_sample_pos)
                                    / (note_samples - MUSIC_DECAY_START));
                sample = (G.music_phase < period / 2) ? amp : -amp;
                G.music_phase = (G.music_phase + 1) % period;
            }
        } else {
            G.music_phase = 0;
        }

        buf[i * 2]     = sample;   // left
        buf[i * 2 + 1] = sample;   // right

        // Advance note timer
        if (++G.music_sample_pos >= note_samples) {
            G.music_sample_pos = 0;
            G.music_phase      = 0;
            G.music_note_idx   = (G.music_note_idx + 1) % DUNGEON_LEN;
        }
    }

    audio_write_end();
}

// ─── rdpq fill helper ────────────────────────────────────────────────────────

static void fillrect(int x, int y, int w, int h, color_t c) {
    rdpq_set_mode_fill(c);
    rdpq_fill_rectangle(x, y, x + w, y + h);
}

// ─── Balloon drawing (in RDP pass) ───────────────────────────────────────────

// Festive balloon colors
static const color_t BALLOON_COLORS[6] = {
    {255, 60,  60,  255},   // red
    {255, 160, 30,  255},   // orange
    {240, 220, 0,   255},   // yellow
    {60,  210, 80,  255},   // green
    {60,  140, 255, 255},   // blue
    {220, 60,  255, 255},   // purple
};

// Curved string offsets (precomputed, avoids sinf per pixel)
static const int STRING_DX[14] = { 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 0 };

static void draw_balloon(int cx, int cy, color_t c) {
    // Oval body using layered horizontal rects
    fillrect(cx-4,  cy-10,  9,  2, c);
    fillrect(cx-7,  cy-8,  15,  2, c);
    fillrect(cx-9,  cy-6,  19,  3, c);
    fillrect(cx-10, cy-3,  21,  3, c);   // widest
    fillrect(cx-10, cy,    21,  3, c);   // widest
    fillrect(cx-9,  cy+3,  19,  3, c);
    fillrect(cx-7,  cy+6,  15,  2, c);
    fillrect(cx-4,  cy+8,   9,  2, c);
    // Knot
    fillrect(cx-2,  cy+10,  5,  3, c);
    // Curvy string (precomputed offsets, no sinf)
    for (int i = 0; i < 14; i++)
        fillrect(cx + STRING_DX[i], cy+13+i, 1, 1, RGBA32(190, 190, 190, 255));
}

// ─── Anniversary Scene (RDP pass) ─────────────────────────────────────────────

// Balloon x positions and frame-phase offsets for variety
static const int BALLOON_X[6]     = { 28, 72, 118, 165, 210, 262 };
static const int BALLOON_PHASE[6] = {  0, 40,  80,  20,  60,  10 };

static void scene_anniversary(void) {
    int f = G.frame;

    // Deep blue-black starry background
    fillrect(0, 0, 320, 240, RGBA32(4, 4, 28, 255));

    // Twinkling stars (deterministic positions, brightness flickers)
    for (int i = 0; i < 32; i++) {
        int sx = (i * 97 + 13) % 316 + 2;
        int sy = (i * 53 + 7)  % 195 + 2;
        int bright = 80 + (((f + i * 17) >> 3) & 1) * 120;
        fillrect(sx, sy, 1, 1, RGBA32(bright, bright, bright, 255));
    }

    // Elya crystal gem (cyan/teal, centered around x=152)
    {
        int cx = 152, cy = 28;
        color_t gem_bright = RGBA32(80, 220, 255, 255);
        color_t gem_mid    = RGBA32(40, 160, 200, 255);
        color_t gem_dark   = RGBA32(20, 100, 160, 255);
        // Top facet — narrow peak widening to center
        for (int row = 0; row < 12; row++) {
            int w = row * 2 + 2;
            color_t c = (row < 4) ? gem_bright : gem_mid;
            fillrect(cx - row, cy + row*2, w, 2, c);
        }
        // Bottom facet — widest at center, narrowing to point
        for (int row = 0; row < 14; row++) {
            int w = 24 - row * 2;
            if (w < 2) w = 2;
            color_t c = (row > 10) ? gem_bright : gem_dark;
            fillrect(cx - w/2, cy + 24 + row*2, w, 2, c);
        }
    }

    // Floating balloons - each drifts upward at slightly different speed
    for (int i = 0; i < 6; i++) {
        int period = 200 + i * 20;   // frames to cross full height
        int raw_y  = 270 - (((f + BALLOON_PHASE[i]) % period) * 270 / period);
        // Gentle horizontal sway using frame counter (no sinf)
        int sway = ((f + BALLOON_PHASE[i]) >> 3) & 1 ? 2 : -2;
        if (raw_y > -30 && raw_y < 245) {
            draw_balloon(BALLOON_X[i] + sway, raw_y, BALLOON_COLORS[i]);
        }
    }

    // Gold border
    fillrect(0,   0,   320, 3, RGBA32(215, 175, 0, 255));
    fillrect(0,   237, 320, 3, RGBA32(215, 175, 0, 255));
    fillrect(0,   0,   3, 240, RGBA32(215, 175, 0, 255));
    fillrect(317, 0,   3, 240, RGBA32(215, 175, 0, 255));
}

// ─── Room Palette (per-room wall/floor/bg colors) ────────────────────────────

typedef struct {
    uint8_t bg_r, bg_g, bg_b;         // upper background
    uint8_t wall_base;                  // base shade for wall bricks
    int8_t  wall_r_off, wall_g_off, wall_b_off; // color tint offsets
    uint8_t floor_base;                 // base shade for floor tiles
    int8_t  floor_r_off, floor_g_off, floor_b_off;
} RoomPalette;

static const RoomPalette ROOM_PALETTES[ROOM_COUNT] = {
    [ROOM_DUNGEON] = { 8,4,16,   28, 0,-6,4,   18, 0,0,4 },     // original purple-gray
    [ROOM_LIBRARY] = { 12,8,24,  24, -4,2,12,   22, -2,0,8 },    // deep blue stone
    [ROOM_FORGE]   = { 16,6,4,   30, 8,-2,-4,   20, 6,-2,-4 },   // warm reddish stone
};

// ─── Forge spark update ─────────────────────────────────────────────────────

static void update_sparks(void) {
    for (int i = 0; i < MAX_SPARKS; i++) {
        Spark *s = &G.sparks[i];
        if (s->life == 0) {
            // Spawn new spark from anvil area (every ~8 frames per slot)
            if ((G.frame + i * 7) % 10 == 0) {
                s->x = 90 + (G.frame * 13 + i * 37) % 20 - 10;
                s->y = 90;
                s->dx = ((G.frame + i) % 5) - 2;
                s->dy = -(2 + ((G.frame + i * 3) % 3));
                s->life = 12 + (i % 8);
            }
            continue;
        }
        s->x += s->dx;
        s->y += s->dy;
        s->dy += 1;  // gravity
        s->life--;
    }
}

static void draw_sparks(void) {
    for (int i = 0; i < MAX_SPARKS; i++) {
        Spark *s = &G.sparks[i];
        if (s->life == 0) continue;
        uint8_t bright = (s->life > 8) ? 255 : s->life * 30;
        fillrect(s->x, s->y, 2, 2, RGBA32(255, bright, bright/3, 255));
    }
}

// ─── Draw NPC sprite (generic, palette from NPCProfile) ─────────────────────

static void draw_npc_sprite(const NPCProfile *npc, int f) {
    int bob = (int)(sinf(f * 0.08f) * 2.0f);
    int sx = npc->sprite_x, sy = npc->sprite_y_base + bob;

    // Dress/robe
    fillrect(sx-6, sy+14, 18, 28,
             RGBA32(npc->dress_r, npc->dress_g, npc->dress_b, 255));
    // Torso
    fillrect(sx-4, sy+8, 14, 14,
             RGBA32(npc->color_r, npc->color_g, npc->color_b, 255));
    // Head (skin)
    fillrect(sx-3, sy, 12, 12, RGBA32(220,180,140, 255));
    // Hair — varies by NPC index for visual differentiation
    if (npc->room == ROOM_DUNGEON) {
        // Sophia: auburn hair
        fillrect(sx-4, sy-2, 14, 5, RGBA32(80, 30, 10, 255));
    } else if (npc->room == ROOM_LIBRARY) {
        // Aldric: gray/silver hair
        fillrect(sx-4, sy-2, 14, 5, RGBA32(140,140,150, 255));
        // Beard
        fillrect(sx-1, sy+9, 8, 5, RGBA32(140,140,150, 255));
    } else {
        // Brunhild: dark hair + headband
        fillrect(sx-4, sy-2, 14, 5, RGBA32(40, 30, 20, 255));
        fillrect(sx-5, sy+1, 16, 2, RGBA32(180, 50, 20, 255)); // red headband
    }
    // Eyes
    fillrect(sx,   sy+3, 2, 2, RGBA32(20, 20, 80, 255));
    fillrect(sx+5, sy+3, 2, 2, RGBA32(20, 20, 80, 255));
}

// ─── Room-specific scene elements ────────────────────────────────────────────

static void draw_room_walls_floor(RoomID room) {
    const RoomPalette *p = &ROOM_PALETTES[room];

    // Upper background
    fillrect(0, 0, 320, 148, RGBA32(p->bg_r, p->bg_g, p->bg_b, 255));

    // Stone wall rows
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 12; col++) {
            int offset = (row & 1) * 16;
            int bx = col * 32 + offset - 16;
            int by = row * 18;
            if (bx + 30 < 0 || bx > 320) continue;
            int shade = p->wall_base + ((col + row) % 3) * 7;
            int r = shade + p->wall_r_off; if (r < 0) r = 0; if (r > 255) r = 255;
            int g = shade + p->wall_g_off; if (g < 0) g = 0; if (g > 255) g = 255;
            int b = shade + p->wall_b_off; if (b < 0) b = 0; if (b > 255) b = 255;
            fillrect(bx+1, by+1, 30, 16, RGBA32(r, g, b, 255));
        }
    }

    // Floor
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 11; col++) {
            int shade = p->floor_base + ((col + row) & 1) * 7;
            int r = shade + p->floor_r_off; if (r < 0) r = 0;
            int g = shade + p->floor_g_off; if (g < 0) g = 0;
            int b = shade + p->floor_b_off; if (b < 0) b = 0;
            fillrect(col*32, 100 + row*10, 31, 9, RGBA32(r, g, b, 255));
        }
    }

    // Floor line
    fillrect(0, 148, 320, 2, RGBA32(40,30,60,255));
}

// Library-specific props: bookshelves, reading desk, candles
static void draw_library_props(int f) {
    // Bookshelves on back wall (3 tall shelves)
    for (int shelf = 0; shelf < 3; shelf++) {
        int bx = 20 + shelf * 100;
        // Shelf frame (dark wood)
        fillrect(bx, 30, 60, 60, RGBA32(50, 30, 15, 255));
        fillrect(bx+2, 32, 56, 2, RGBA32(70, 45, 20, 255));  // top
        fillrect(bx+2, 52, 56, 2, RGBA32(70, 45, 20, 255));  // mid shelf
        fillrect(bx+2, 72, 56, 2, RGBA32(70, 45, 20, 255));  // mid shelf 2
        fillrect(bx+2, 88, 56, 2, RGBA32(70, 45, 20, 255));  // bottom
        // Books (colored spines on each shelf)
        for (int b = 0; b < 7; b++) {
            int br = 60 + ((shelf*7+b) * 37) % 140;
            int bg = 20 + ((shelf*7+b) * 53) % 80;
            int bb = 30 + ((shelf*7+b) * 71) % 120;
            fillrect(bx + 4 + b*8, 34, 6, 18, RGBA32(br, bg, bb, 255));
            fillrect(bx + 4 + b*8, 54, 6, 18, RGBA32(bg, bb, br, 255));
            fillrect(bx + 4 + b*8, 74, 6, 14, RGBA32(bb, br, bg, 255));
        }
    }
    // Reading desk
    fillrect(100, 108, 50, 8, RGBA32(80, 50, 20, 255));
    fillrect(105, 116, 6, 20, RGBA32(60, 35, 15, 255));
    fillrect(139, 116, 6, 20, RGBA32(60, 35, 15, 255));
    // Open book on desk
    fillrect(108, 104, 18, 6, RGBA32(220, 210, 180, 255));
    fillrect(128, 104, 18, 6, RGBA32(230, 220, 190, 255));
    // Candle on desk (flickering)
    int flick = (f / 5) & 1;
    fillrect(147, 96 - flick, 3 + flick, 6, RGBA32(255, 200 + flick*40, 80, 255));
    fillrect(148, 102, 2, 6, RGBA32(200, 190, 170, 255));
    // Ambient glow around candle
    if ((f / 8) & 1)
        fillrect(144, 92, 10, 8, RGBA32(255, 220, 100, 30));
}

// Forge-specific props: anvil, furnace, bellows
static void draw_forge_props(int f) {
    int flick = (f / 3) & 1;

    // Large furnace (back wall, left)
    fillrect(20, 30, 80, 60, RGBA32(50, 35, 30, 255));   // stone body
    fillrect(24, 34, 72, 52, RGBA32(30, 20, 15, 255));    // inner
    // Fire inside furnace
    fillrect(30, 50, 60, 30, RGBA32(200+flick*40, 80+flick*30, 0, 255));
    fillrect(35, 45, 50, 20, RGBA32(255, 160+flick*40, 20, 255));
    fillrect(42, 40, 36, 12, RGBA32(255, 220, 80+flick*40, 255));
    // Furnace opening (arch shape)
    fillrect(38, 76, 44, 14, RGBA32(20, 12, 8, 255));
    fillrect(40, 74, 40, 4, RGBA32(20, 12, 8, 255));

    // Anvil (center floor)
    fillrect(82, 104, 36, 6, RGBA32(100, 100, 110, 255));  // top
    fillrect(86, 110, 28, 12, RGBA32(80, 80, 90, 255));     // body
    fillrect(90, 122, 20, 14, RGBA32(70, 70, 80, 255));     // base
    // Anvil horn (left point)
    fillrect(74, 104, 10, 4, RGBA32(90, 90, 100, 255));
    fillrect(72, 105, 4, 2, RGBA32(80, 80, 90, 255));
    // Anvil highlight
    fillrect(84, 104, 32, 1, RGBA32(150, 150, 160, 255));

    // Weapon rack (right wall)
    fillrect(240, 40, 4, 55, RGBA32(60, 40, 20, 255));
    fillrect(270, 40, 4, 55, RGBA32(60, 40, 20, 255));
    fillrect(238, 40, 38, 3, RGBA32(60, 40, 20, 255));
    fillrect(238, 70, 38, 3, RGBA32(60, 40, 20, 255));
    // Swords on rack
    fillrect(250, 42, 2, 26, RGBA32(180, 195, 210, 255));
    fillrect(260, 42, 2, 26, RGBA32(190, 200, 215, 255));
    fillrect(250, 72, 2, 22, RGBA32(170, 185, 200, 255));

    // Bellows (next to furnace)
    fillrect(105, 68, 20, 16, RGBA32(90, 55, 25, 255));
    int bellow_open = (f / 20) & 1;
    fillrect(105, 64 - bellow_open * 3, 20, 6, RGBA32(70, 40, 15, 255));
    fillrect(112, 84, 6, 8, RGBA32(60, 40, 20, 255));

    // Ambient furnace glow on floor
    if (flick)
        fillrect(20, 90, 100, 10, RGBA32(80, 30, 0, 50));

    // Draw sparks (particle effect from anvil)
    draw_sparks();
}

// ─── Dungeon Scene (now room-aware) ──────────────────────────────────────────

static void scene_dungeon(void) {
    int f = G.frame;
    RoomID room = G.current_room;

    // Draw walls and floor with room palette
    draw_room_walls_floor(room);

    // Room-specific props
    if (room == ROOM_LIBRARY) {
        draw_library_props(f);
    } else if (room == ROOM_FORGE) {
        draw_forge_props(f);
    }

    // ── Keese position (only in dungeon room) ────────────────────────────────
    int kx = 0, ky = 0;
    if (room == ROOM_DUNGEON) {
        kx = 150 + (int)(sinf(f * 0.045f) * 55.0f) + (int)(sinf(f * 0.13f) * 18.0f);
        ky =  38 + (int)(sinf(f * 0.031f) * 22.0f);
    }

    // ── Attack auto-trigger (only in dungeon) ────────────────────────────────
    if (room == ROOM_DUNGEON && G.state == STATE_DUNGEON && G.attack_timer <= 0
            && f > 120 && (f % 180) == 0) {
        G.attack_timer  = 42;
        G.attack_target = (f / 180) & 1;
    }
    int atk = G.attack_timer;
    if (atk > 0) G.attack_timer--;

    // Room-specific torches
    if (room == ROOM_DUNGEON) {
        int flick = (f / 4) & 1;
        fillrect(18, 56, 8+flick*2, 14, RGBA32(255, 140+flick*30, 0, 255));
        fillrect(16, 66, 12+flick*2, 4, RGBA32(200, 60, 0, 255));
        fillrect(22, 70, 2, 18, RGBA32(80, 60, 40, 255));
    }

    // NPC sprite — find NPC for current room and draw
    {
        int npc_idx = -1;
        for (int i = 0; i < NPC_COUNT; i++) {
            if (NPC_PROFILES[i].room == room) { npc_idx = i; break; }
        }
        G.current_npc = npc_idx;
        if (npc_idx >= 0) {
            draw_npc_sprite(&NPC_PROFILES[npc_idx], f);
        }
    }

    // Sophia-specific extras: shield + sword (only in dungeon)
    if (room == ROOM_DUNGEON) {
        int bob = (int)(sinf(f * 0.08f) * 2.0f);
        int sx = 204, sy = 72 + bob;

        // ── Shield (left arm, kite-style) ────────────────────────────────
        {
            int shx = sx - 20, shy = sy + 6;
            fillrect(shx+1, shy,      10,  2, RGBA32(20,  50, 130, 255));
            fillrect(shx,   shy+2,    12, 10, RGBA32(20,  50, 130, 255));
            fillrect(shx+1, shy+12,   10,  4, RGBA32(20,  50, 130, 255));
            fillrect(shx+2, shy+16,    8,  3, RGBA32(20,  50, 130, 255));
            fillrect(shx+4, shy+19,    4,  4, RGBA32(20,  50, 130, 255));
            fillrect(shx,   shy,      12,  2, RGBA32(215,175,  0, 255));
            fillrect(shx,   shy+2,     2, 20, RGBA32(215,175,  0, 255));
            fillrect(shx+10,shy+2,     2, 20, RGBA32(215,175,  0, 255));
            fillrect(shx+5, shy+4,     2,  2, RGBA32(80,220,255, 255));
            fillrect(shx+3, shy+6,     6,  3, RGBA32(40,160,200, 255));
            fillrect(shx+5, shy+9,     2,  2, RGBA32(80,220,255, 255));
        }

        // ── Sword (right arm, blade raised upward) ───────────────────────
        {
            int swx = sx + 13;
            fillrect(swx,   sy-16,  2, 28, RGBA32(195,215,235, 255));
            fillrect(swx+1, sy-16,  1, 14, RGBA32(240,250,255, 255));
            fillrect(swx,   sy-18,  2,  2, RGBA32(220,235,250, 255));
            fillrect(swx-5, sy+12,  12,  3, RGBA32(215,175,  0, 255));
            fillrect(swx-5, sy+12,  12,  1, RGBA32(255,220, 60, 255));
            fillrect(swx,   sy+15,   2,  7, RGBA32(110, 60, 15, 255));
            fillrect(swx-1, sy+22,   4,  3, RGBA32(215,175,  0, 255));
            int gleam = (f / 7) % 22;
            fillrect(swx, sy + 10 - gleam, 1, 4, RGBA32(255,255,255, 200));
        }
    }

    // Dungeon-only enemies and combat
    if (room == ROOM_DUNGEON) {
        int sx = 204, sy = 72 + (int)(sinf(f * 0.08f) * 2.0f);

        // Stalfos skeleton (left side)
        int ex = 80, ey = 78;
        fillrect(ex-4, ey,     12, 10, RGBA32(220,220,200,255));
        fillrect(ex-2, ey+2,    3,  3, RGBA32(8,4,16,255));
        fillrect(ex+4, ey+2,    3,  3, RGBA32(8,4,16,255));
        fillrect(ex-3, ey+10,  10,  4, RGBA32(200,200,180,255));
        fillrect(ex-5, ey+14,  14, 16, RGBA32(180,180,160,255));
        for (int r = 0; r < 3; r++)
            fillrect(ex-5, ey+15+r*5, 14, 2, RGBA32(70,70,55,255));
        fillrect(ex-4, ey+30,   4, 14, RGBA32(180,180,160,255));
        fillrect(ex+4, ey+30,   4, 14, RGBA32(180,180,160,255));
        if (atk > 0 && atk <= 22 && G.attack_target == 0)
            fillrect(ex-5, ey, 14, 44, RGBA32(255, 255, 255, 200));

        // Keese (bat enemy)
        int wing = (f / 5) & 1;
        fillrect(kx-3, ky-2, 6, 5, RGBA32(25, 15, 35, 255));
        if (wing == 0) {
            fillrect(kx-12, ky-5,  9, 6, RGBA32(50, 35, 70, 255));
            fillrect(kx+3,  ky-5,  9, 6, RGBA32(50, 35, 70, 255));
            fillrect(kx-12, ky-1,  4, 3, RGBA32(35, 22, 50, 255));
            fillrect(kx+8,  ky-1,  4, 3, RGBA32(35, 22, 50, 255));
        } else {
            fillrect(kx-12, ky+1,  9, 5, RGBA32(50, 35, 70, 255));
            fillrect(kx+3,  ky+1,  9, 5, RGBA32(50, 35, 70, 255));
            fillrect(kx-10, ky-2,  4, 3, RGBA32(35, 22, 50, 255));
            fillrect(kx+6,  ky-2,  4, 3, RGBA32(35, 22, 50, 255));
        }
        fillrect(kx-1, ky-1, 2, 2, RGBA32(255, 60,  20, 255));
        fillrect(kx+2, ky-1, 2, 2, RGBA32(255, 60,  20, 255));
        if (atk > 0 && atk <= 22 && G.attack_target == 1)
            fillrect(kx-12, ky-5, 25, 13, RGBA32(255, 255, 255, 200));

        // Treasure Chest
        {
            int cx = 146, cy = 112;
            fillrect(cx,    cy+10, 28, 20, RGBA32(100, 62, 18, 255));
            fillrect(cx,    cy,    28, 12, RGBA32(130, 82, 28, 255));
            fillrect(cx+2,  cy-1,  24,  2, RGBA32(155, 100, 40, 255));
            fillrect(cx,    cy+10, 28,  2, RGBA32(215, 175,  0, 255));
            fillrect(cx,    cy+28, 28,  2, RGBA32(215, 175,  0, 255));
            fillrect(cx,    cy,    28,  2, RGBA32(215, 175,  0, 255));
            fillrect(cx,    cy,     2, 30, RGBA32(215, 175,  0, 255));
            fillrect(cx+26, cy,     2, 30, RGBA32(215, 175,  0, 255));
            fillrect(cx+11, cy+8,   6,  6, RGBA32(215, 175,  0, 255));
            fillrect(cx+13, cy+9,   2,  2, RGBA32(30,  20,   5, 255));
            fillrect(cx+13, cy+11,  2,  3, RGBA32(30,  20,   5, 255));
            int glow = ((f / 30) & 1) ? 60 : 20;
            fillrect(cx+3,  cy+3,   8,  2, RGBA32(255, 230, 100, glow));
            fillrect(cx+14, cy+3,   8,  2, RGBA32(255, 230, 100, glow));
        }

        // Attack slash trail
        if (atk > 22) {
            int prog = 42 - atk;
            int swx_tip = sx + 14, swy_tip = sy - 14;
            int tx = (G.attack_target == 0) ? ex + 1 : kx;
            int ty = (G.attack_target == 0) ? ey + 4 : ky;
            int lx = swx_tip + ((tx - swx_tip) * prog) / 19;
            int ly = swy_tip + ((ty - swy_tip) * prog) / 19;
            fillrect(lx-2, ly-2, 6, 6, RGBA32(255, 255, 120, 255));
            fillrect(lx-1, ly-1, 4, 4, RGBA32(255, 240, 60,  255));
            if (prog > 2) {
                int lx2 = swx_tip + ((tx - swx_tip) * (prog - 3)) / 19;
                int ly2 = swy_tip + ((ty - swy_tip) * (prog - 3)) / 19;
                fillrect(lx2-1, ly2-1, 4, 4, RGBA32(220, 200, 40, 180));
            }
            if (prog > 5) {
                int lx3 = swx_tip + ((tx - swx_tip) * (prog - 6)) / 19;
                int ly3 = swy_tip + ((ty - swy_tip) * (prog - 6)) / 19;
                fillrect(lx3,   ly3,   2, 2, RGBA32(180, 140, 20, 120));
            }
        } else if (atk > 0) {
            int hx2 = (G.attack_target == 0) ? ex + 1 : kx;
            int hy2 = (G.attack_target == 0) ? ey + 4 : ky;
            int sp  = 22 - atk;
            fillrect(hx2 + sp,     hy2 - sp/2, 3, 3, RGBA32(255, 230,   0, 255));
            fillrect(hx2 - sp,     hy2 + sp/2, 3, 3, RGBA32(255, 200,  50, 255));
            fillrect(hx2 + sp/2,   hy2 + sp,   3, 3, RGBA32(255, 160,   0, 255));
            fillrect(hx2 - sp/2,   hy2 - sp,   3, 3, RGBA32(255, 100,   0, 255));
            if (sp < 9)
                fillrect(hx2 - 3, hy2 - 3, 7, 7, RGBA32(255, 255, 200, 255));
        }
    } // end ROOM_DUNGEON enemies

    // ── Room exit indicators (arrows at edges) ─────────────────────────────
    {
        const RoomExits *ex = &ROOM_MAP[room];
        color_t arrow_col = RGBA32(200, 180, 100, 180);
        if (ex->east >= 0) {    // right arrow
            fillrect(312, 72, 4, 8, arrow_col);
            fillrect(308, 76, 4, 4, arrow_col);
        }
        if (ex->west >= 0) {    // left arrow
            fillrect(4, 72, 4, 8, arrow_col);
            fillrect(8, 76, 4, 4, arrow_col);
        }
        if (ex->north >= 0) {   // up arrow
            fillrect(156, 16, 8, 4, arrow_col);
            fillrect(160, 20, 4, 4, arrow_col);
        }
        if (ex->south >= 0) {   // down arrow
            fillrect(156, 142, 8, 4, arrow_col);
            fillrect(160, 138, 4, 4, arrow_col);
        }
    }

    // ── HUD: 4 Hearts + Magic Bar + Room Name (drawn last = on top) ──────
    fillrect(0, 0, 320, 14, RGBA32(0, 0, 0, 255));   // dark HUD band

    // 4 heart containers — each 8×6 px, gap of 2px → 10px per heart
    for (int h = 0; h < 4; h++) {
        int hx = 4 + h * 12;
        int hy = 3;
        // Full heart = 2 half-hearts. G.hearts tracks half-hearts.
        color_t hcol = (G.hearts >= (h * 2 + 2)) ? RGBA32(220,  30,  30, 255) :
                       (G.hearts == (h * 2 + 1)) ? RGBA32(220,  30,  30, 255) :
                                                    RGBA32( 60,  12,  12, 255);
        // Heart shape: two bumps + wide body + narrowing point
        fillrect(hx+1, hy,   2, 2, hcol);   // left bump
        fillrect(hx+5, hy,   2, 2, hcol);   // right bump
        fillrect(hx,   hy+1, 8, 3, hcol);   // middle body
        fillrect(hx+1, hy+4, 6, 1, hcol);   // taper 1
        fillrect(hx+2, hy+5, 4, 1, hcol);   // taper 2
        fillrect(hx+3, hy+6, 2, 1, hcol);   // tip
        // Half-heart: overlay right half dark when half-full
        if (G.hearts == (h * 2 + 1)) {
            color_t hdim = RGBA32(60, 12, 12, 255);
            fillrect(hx+4, hy,   4, 2, hdim);
            fillrect(hx+4, hy+1, 4, 3, hdim);
            fillrect(hx+4, hy+4, 3, 1, hdim);
            fillrect(hx+5, hy+5, 2, 1, hdim);
        }
    }

    // Magic bar — green gradient, right side of HUD
    {
        int bx = 200, by = 4;
        fillrect(bx,    by,    68,  6, RGBA32( 10,  10,  30, 255)); // background
        int fill = (G.magic * 64) / 128;                             // 0-64 px
        if (fill > 0) {
            fillrect(bx+2, by+1, fill, 2, RGBA32( 80, 255, 130, 255)); // top highlight
            fillrect(bx+2, by+3, fill, 2, RGBA32( 40, 200,  90, 255)); // lower fill
        }
        fillrect(bx,    by,    68,  1, RGBA32( 80, 180,  80, 255)); // top border
        fillrect(bx,    by+5,  68,  1, RGBA32( 80, 180,  80, 255)); // bottom border
        fillrect(bx,    by,     1,  6, RGBA32( 80, 180,  80, 255)); // left border
        fillrect(bx+67, by,     1,  6, RGBA32( 80, 180,  80, 255)); // right border
    }

    // Performance bars (RDP-drawn during generation)
    if (G.perf_show && (G.state == STATE_GENERATING || G.state == STATE_DIALOG)) {
        int bar_y = 140;

        // CPU bar background
        fillrect(28, bar_y, 84, 6, RGBA32(20, 20, 20, 255));
        // CPU bar fill (red→orange gradient based on load)
        {
            int cpu_fill = (int)(G.perf_cpu_pct * 0.80f);
            if (cpu_fill > 80) cpu_fill = 80;
            if (cpu_fill < 1) cpu_fill = 1;
            if (G.perf_cpu_pct > 80.0f) {
                fillrect(30, bar_y+1, cpu_fill, 4, RGBA32(255, 60, 30, 255));  // red = heavy
            } else if (G.perf_cpu_pct > 40.0f) {
                fillrect(30, bar_y+1, cpu_fill, 4, RGBA32(255, 160, 30, 255)); // orange
            } else {
                fillrect(30, bar_y+1, cpu_fill, 4, RGBA32(80, 220, 80, 255));  // green = light
            }
        }
        // CPU bar border
        fillrect(28, bar_y,   84, 1, RGBA32(100, 100, 100, 255));
        fillrect(28, bar_y+5, 84, 1, RGBA32(100, 100, 100, 255));
        fillrect(28, bar_y,    1, 6, RGBA32(100, 100, 100, 255));
        fillrect(111, bar_y,   1, 6, RGBA32(100, 100, 100, 255));

#ifdef USE_RSP_MATMUL
        // RSP bar (always shows "active" during generation since RSP does matmul)
        fillrect(130, bar_y, 84, 6, RGBA32(20, 20, 20, 255));
        if (G.state == STATE_GENERATING) {
            fillrect(132, bar_y+1, 60, 4, RGBA32(30, 200, 255, 255));  // cyan = RSP active
        }
        fillrect(130, bar_y,   84, 1, RGBA32(100, 100, 100, 255));
        fillrect(130, bar_y+5, 84, 1, RGBA32(100, 100, 100, 255));
        fillrect(130, bar_y,    1, 6, RGBA32(100, 100, 100, 255));
        fillrect(213, bar_y,   1, 6, RGBA32(100, 100, 100, 255));
#endif
#ifdef USE_RPC_LLM
        if (G.rpc_active) {
            // RPC bar (purple = remote inference)
            fillrect(130, bar_y, 84, 6, RGBA32(20, 20, 20, 255));
            if (G.state == STATE_GENERATING && G.rpc_pending) {
                // Pulsing bar while waiting for RPC response
                int pulse = (G.frame / 4) % 60;
                fillrect(132, bar_y+1, pulse + 20, 4, RGBA32(180, 80, 255, 255));
            } else if (G.perf_gen_total_us > 0) {
                // Solid bar showing latency quality after response
                // 200ms = full (80px), 2000ms = minimum (8px)
                int latency_ms = (int)(G.perf_gen_total_us / 1000);
                int rpc_fill = 80 - (latency_ms - 200) / 25;
                if (rpc_fill > 80) rpc_fill = 80;
                if (rpc_fill < 8)  rpc_fill = 8;
                fillrect(132, bar_y+1, rpc_fill, 4, RGBA32(140, 60, 220, 255));
            }
            // Purple border
            fillrect(130, bar_y,   84, 1, RGBA32(140, 80, 180, 255));
            fillrect(130, bar_y+5, 84, 1, RGBA32(140, 80, 180, 255));
            fillrect(130, bar_y,    1, 6, RGBA32(140, 80, 180, 255));
            fillrect(213, bar_y,   1, 6, RGBA32(140, 80, 180, 255));
        }
#endif

        // Tok/s numeric right-aligned
        {
            int whole = (int)G.perf_toks_precise;
            int frac  = (int)((G.perf_toks_precise - (float)whole) * 10.0f);
            if (whole > 99) { whole = 99; frac = 9; }
            char tsbuf[12];
            int i = 0;
            if (whole >= 10) tsbuf[i++] = '0' + whole / 10;
            tsbuf[i++] = '0' + whole % 10;
            tsbuf[i++] = '.';
            tsbuf[i++] = '0' + frac;
            tsbuf[i++] = ' ';
            tsbuf[i++] = 't';
            tsbuf[i++] = 'o';
            tsbuf[i++] = 'k';
            tsbuf[i++] = '/';
            tsbuf[i++] = 's';
            tsbuf[i] = '\0';
            // Will be drawn with graphics_draw_text in SW pass
        }
    }
}

static void scene_dialog_box(void) {
    fillrect(8, 150, 304, 80, RGBA32(0, 0, 60, 255));
    fillrect(8, 150, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 228, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 150,   2, 80, RGBA32(215,175,0,255));
    fillrect(310,150,  2, 80, RGBA32(215,175,0,255));
    fillrect(11, 153, 298, 1, RGBA32(100,80,20,255));
}

// ─── Dialog Selection Scene (D-pad option picker) ────────────────────────────

static void scene_dialog_select(void) {
    // Semi-transparent overlay box with 4 dialog options
    fillrect(8, 150, 304, 80, RGBA32(0, 0, 50, 255));
    fillrect(8, 150, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 228, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 150,   2, 80, RGBA32(215,175,0,255));
    fillrect(310,150,  2, 80, RGBA32(215,175,0,255));
    // Highlight selected option
    int sel_y = 158 + G.dialog_select_idx * 16;
    fillrect(12, sel_y, 296, 14, RGBA32(80, 60, 0, 255));
    // Gold selector arrow
    fillrect(14, sel_y + 3, 6, 2, RGBA32(215, 175, 0, 255));
    fillrect(18, sel_y + 1, 2, 6, RGBA32(215, 175, 0, 255));
}

// ─── Room transition fade overlay ────────────────────────────────────────────

static void scene_room_transition(void) {
    // Simple fade to black and back: timer goes 20→0
    int alpha = 0;
    if (G.transition_timer > 10) {
        // Fading out (20→11): alpha 0→255
        alpha = (20 - G.transition_timer) * 255 / 10;
    } else {
        // Fading in (10→0): alpha 255→0
        alpha = G.transition_timer * 255 / 10;
    }
    if (alpha > 255) alpha = 255;
    if (alpha < 0)   alpha = 0;
    // Full screen dark overlay
    fillrect(0, 0, 320, 240, RGBA32(0, 0, 0, (uint8_t)alpha));
}

// ─── Virtual Keyboard ────────────────────────────────────────────────────────
// 4 rows x 10 cols D-pad character picker for player text input to Sophia

static const char KB_GRID[4][11] = {
    "ABCDEFGHIJ",
    "KLMNOPQRST",
    "UVWXYZ .,?",
    "0123456789",
};

static void scene_keyboard(void) {
    /* Dark background */
    fillrect(0, 0, 320, 240, RGBA32(10, 5, 20, 255));

    /* Input display area (top) */
    fillrect(8, 8, 304, 24, RGBA32(0, 0, 40, 255));
    fillrect(8, 8, 304, 1, RGBA32(215,175,0,255));
    fillrect(8, 31, 304, 1, RGBA32(215,175,0,255));
    fillrect(8, 8, 1, 24, RGBA32(215,175,0,255));
    fillrect(311, 8, 1, 24, RGBA32(215,175,0,255));

    /* Keyboard grid background */
    fillrect(20, 80, 280, 120, RGBA32(0, 0, 50, 255));

    /* Highlight selected cell */
    int cx = 28 + G.kb_col * 28;
    int cy = 84 + G.kb_row * 28;
    fillrect(cx, cy, 24, 24, RGBA32(180, 140, 0, 255));

    /* Grid cell borders */
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 10; c++) {
            int x = 28 + c * 28;
            int y = 84 + r * 28;
            fillrect(x, y, 24, 1, RGBA32(80, 60, 120, 255));
            fillrect(x, y+23, 24, 1, RGBA32(80, 60, 120, 255));
            fillrect(x, y, 1, 24, RGBA32(80, 60, 120, 255));
            fillrect(x+23, y, 1, 24, RGBA32(80, 60, 120, 255));
        }
    }

    /* Bottom bar */
    fillrect(20, 210, 280, 1, RGBA32(100, 80, 20, 255));
}

static void draw_keyboard_text(surface_t *disp) {
    graphics_draw_text(disp, 16, 2, "Ask Sophia:");

    /* Show current input with cursor */
    char ibuf[66];
    int len = G.kb_len;
    if (len > 60) len = 60;
    memcpy(ibuf, G.kb_input, len);
    ibuf[len] = '_';
    ibuf[len + 1] = '\0';
    graphics_draw_text(disp, 14, 14, ibuf);

    /* Draw keyboard grid characters */
    char cbuf[2] = {0, 0};
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 10; c++) {
            cbuf[0] = KB_GRID[r][c];
            int x = 36 + c * 28;
            int y = 90 + r * 28;
            graphics_draw_text(disp, x, y, cbuf);
        }
    }

    /* Instructions */
    graphics_draw_text(disp, 20, 216, "[A]Select [B]Delete [START]Send");
    /* Show char count */
    char cntbuf[8];
    cntbuf[0] = '0' + (G.kb_len / 10) % 10;
    cntbuf[1] = '0' + G.kb_len % 10;
    cntbuf[2] = '/';
    cntbuf[3] = '6';
    cntbuf[4] = '0';
    cntbuf[5] = '\0';
    graphics_draw_text(disp, 270, 2, cntbuf);
}

static void handle_keyboard_input(struct controller_data *k) {
    if (G.kb_debounce > 0) { G.kb_debounce--; return; }

    /* D-pad navigation */
    if (k->c[0].up)    { G.kb_row = (G.kb_row + 3) % 4; G.kb_debounce = 5; }
    if (k->c[0].down)  { G.kb_row = (G.kb_row + 1) % 4; G.kb_debounce = 5; }
    if (k->c[0].left)  { G.kb_col = (G.kb_col + 9) % 10; G.kb_debounce = 5; }
    if (k->c[0].right) { G.kb_col = (G.kb_col + 1) % 10; G.kb_debounce = 5; }

    /* A = select character */
    if (k->c[0].A && G.kb_len < 60) {
        char ch = KB_GRID[G.kb_row][G.kb_col];
        /* Convert to lowercase for model input */
        if (ch >= 'A' && ch <= 'Z') ch = ch + 32;
        G.kb_input[G.kb_len++] = ch;
        G.kb_debounce = 8;
    }

    /* B = backspace */
    if (k->c[0].B && G.kb_len > 0) {
        G.kb_len--;
        G.kb_input[G.kb_len] = '\0';
        G.kb_debounce = 8;
    }

    /* START = send to current NPC */
    if (k->c[0].start && G.kb_len > 0) {
        /* Build prompt string from keyboard input, add ": " suffix for model */
        char prompt_str[66];
        int plen = G.kb_len;
        if (plen > 60) plen = 60;
        memcpy(prompt_str, G.kb_input, plen);
        prompt_str[plen++] = ':';
        prompt_str[plen++] = ' ';
        prompt_str[plen] = '\0';

        int npc = G.current_npc;
        if (npc < 0) npc = 0;
        start_dialog_from_prompt(npc, prompt_str, 1);

        /* Clear keyboard buffer for next time */
        G.kb_len = 0;
        memset(G.kb_input, 0, sizeof(G.kb_input));
    }

    /* Z = cancel, back to dungeon */
    if (k->c[0].Z) {
        G.state = STATE_DUNGEON;
        G.kb_debounce = 10;
    }
}

// ─── CPU text overlay ────────────────────────────────────────────────────────

static void draw_text(surface_t *disp) {
    switch (G.state) {

    case STATE_ANNIVERSARY:
        graphics_draw_text(disp,  64, 72,  "World's First N64 LLM");
        graphics_draw_text(disp,  80, 86,  "Legend of Elya");
        graphics_draw_text(disp,  80, 100, "Elyan Labs  2026");
        graphics_draw_text(disp,  96, 168, "from Elyan Labs");
        graphics_draw_text(disp,  68, 182, "World's First N64 LLM");
        if ((G.frame / 20) & 1)
            graphics_draw_text(disp, 80, 220, "Press START to continue");
        break;

    case STATE_TITLE:
        graphics_draw_text(disp, 104, 50, "LEGEND OF ELYA");
        graphics_draw_text(disp,  80, 68, "Nintendo 64 Homebrew");
        graphics_draw_text(disp, 120, 84, "Elyan Labs");
        graphics_draw_text(disp,  76,103, "World's First N64 LLM");
        if (G.ai_ready && G.ai.is_loaded)
            graphics_draw_text(disp,  84, 118, "[Sophia AI: LOADED]");
        else if (G.ai_ready)
            graphics_draw_text(disp,  64, 118, "[AI: file ok, magic?]");
        else if (G.ai_load_failed_size > 0) {
            /* Make the failure LOUD.  This branch was silent for months: the
             * weight blob grew past a hardcoded buffer and the game just fell
             * back to canned dialogue with no way to tell from the screen.
             * Print the two numbers that explain it. */
            char why[40];
            graphics_draw_text(disp,  52, 118, "[Sophia AI: WEIGHTS TOO BIG]");
            snprintf(why, sizeof(why), "blob %d B > buffer %d B",
                     G.ai_load_failed_size, G.ai_load_capacity);
            graphics_draw_text(disp,  44, 131, why);
        } else
            graphics_draw_text(disp,  72, 118, "[Sophia AI: Demo Mode]");
        graphics_draw_text(disp,  80, 155, "Press START to enter");
        graphics_draw_text(disp, 104, 170, "the dungeon...");
        break;

    case STATE_DUNGEON:
        // Room name in HUD (right of magic bar)
        graphics_draw_text(disp, 60, 3, ROOM_NAMES[G.current_room]);
        graphics_draw_text(disp, 186, 3,  "MP");   // magic bar label
        // Context-sensitive help text
        if (G.current_room == ROOM_DUNGEON)
            graphics_draw_text(disp,  10, 220, "[A]Talk [B]Keyboard  (auto-attack)");
        else
            graphics_draw_text(disp, 10, 220, "[A]Talk [B]Keyboard [L/R]Move");
        break;

    case STATE_DIALOG_SELECT: {
        // Room name in HUD
        graphics_draw_text(disp, 60, 3, ROOM_NAMES[G.current_room]);
        graphics_draw_text(disp, 186, 3,  "MP");
        // NPC name
        if (G.current_npc >= 0)
            graphics_draw_text(disp, 16, 154, NPC_PROFILES[G.current_npc].name);
        else
            graphics_draw_text(disp, 16, 154, "???");
        // Dialog options
        {
            int npc = G.current_npc;
            if (npc < 0) npc = 0;
            for (int i = 0; i < DIALOG_OPTIONS; i++) {
                const char *opt = NPC_DIALOG_OPTIONS[npc][i];
                // Truncate display (remove trailing ": ")
                char obuf[32];
                int olen = 0;
                for (int j = 0; opt[j] && opt[j] != ':' && olen < 30; j++)
                    obuf[olen++] = opt[j];
                obuf[olen] = '\0';
                int oy = 162 + i * 16;
                graphics_draw_text(disp, 26, oy, obuf);
            }
        }
        graphics_draw_text(disp, 10, 230, "[A]Select [B]Cancel [D-pad]Choose");
        break;
    }

    case STATE_ROOM_TRANSITION:
        // Brief flash, no text needed
        break;

    case STATE_KEYBOARD:
        draw_keyboard_text(disp);
        break;

    case STATE_DIALOG:
    case STATE_GENERATING: {
        // Room name in HUD
        graphics_draw_text(disp, 60, 3, ROOM_NAMES[G.current_room]);
        // NPC name instead of hardcoded "Sophia Elya:"
        if (G.current_npc >= 0)
            graphics_draw_text(disp, 16, 158, NPC_PROFILES[G.current_npc].name);
        else
            graphics_draw_text(disp, 16, 158, "Sophia Elya:");

        // Performance overlay during generation
        if (G.state == STATE_GENERATING && G.gen_out_count > 0) {
            // Tok/s (precise)
            char spdbuf[16];
            int whole = (int)G.perf_toks_precise;
            int frac  = (int)((G.perf_toks_precise - (float)whole) * 10.0f);
            if (whole > 99) whole = 99;
            spdbuf[0] = (whole >= 10) ? ('0' + whole / 10) : ' ';
            spdbuf[1] = '0' + (whole % 10);
            spdbuf[2] = '.';
            spdbuf[3] = '0' + frac;
            spdbuf[4] = ' ';
            spdbuf[5] = 't';
            spdbuf[6] = '/';
            spdbuf[7] = 's';
            spdbuf[8] = '\0';
            graphics_draw_text(disp, 200, 158, spdbuf);

            // Total gen time in ms
            {
                int ms = (int)(G.perf_gen_total_us / 1000);
                char tbuf[12];
                int ti = 0;
                if (ms >= 1000) { tbuf[ti++] = '0' + (ms / 1000) % 10; ms %= 1000; }
                if (ms >= 100 || ti > 0) tbuf[ti++] = '0' + (ms / 100) % 10;
                tbuf[ti++] = '0' + (ms / 10) % 10;
                tbuf[ti++] = '0' + ms % 10;
                tbuf[ti++] = 'm'; tbuf[ti++] = 's'; tbuf[ti] = '\0';
                graphics_draw_text(disp, 260, 158, tbuf);
            }
        }

        // Performance bars (show during and briefly after generation)
        if (G.perf_show && (G.state == STATE_GENERATING || G.perf_cpu_pct > 0.0f)) {
            int bar_y = 145;  // just above dialog box

            // CPU inference bar (red/orange)
            {
                int cpu_fill = (int)(G.perf_cpu_pct * 0.80f);  // 80px max
                if (cpu_fill > 80) cpu_fill = 80;
                if (cpu_fill < 0) cpu_fill = 0;
                graphics_draw_text(disp, 4, bar_y - 1, "CPU");
            }
#ifdef USE_RSP_MATMUL
            // RSP indicator (cyan)
            graphics_draw_text(disp, 44, bar_y - 1, "RSP");
#endif
#ifdef USE_RPC_LLM
            if (G.rpc_active) {
                // RPC label and latency display
                graphics_draw_text(disp, 44, bar_y - 1, "RPC");
                // Show RPC latency in ms when available
                if (G.perf_gen_total_us > 0 && G.state == STATE_DIALOG) {
                    char rpcbuf[12];
                    int ms = (int)(G.perf_gen_total_us / 1000);
                    int ri = 0;
                    if (ms >= 1000) { rpcbuf[ri++] = '0' + (ms/1000)%10; }
                    if (ms >= 100 || ri > 0) rpcbuf[ri++] = '0' + (ms/100)%10;
                    rpcbuf[ri++] = '0' + (ms/10)%10;
                    rpcbuf[ri++] = '0' + ms%10;
                    rpcbuf[ri++] = 'm'; rpcbuf[ri++] = 's'; rpcbuf[ri] = '\0';
                    graphics_draw_text(disp, 82, bar_y - 1, rpcbuf);
                } else if (G.rpc_pending) {
                    // Pulsing dots while waiting
                    int dots = (G.frame / 10) % 4;
                    char dotbuf[5] = "    ";
                    for (int d = 0; d < dots; d++) dotbuf[d] = '.';
                    dotbuf[dots] = '\0';
                    graphics_draw_text(disp, 82, bar_y - 1, dotbuf);
                }
            }
#endif
        }

        // Character reveal with word-wrap
        int show = (G.dialog_char < 90) ? G.dialog_char : 90;
        char linebuf[37];
        int lb = 0, col = 0, line_y = 174;

        for (int i = 0; i < show; i++) {
            unsigned char c = G.dialog_buf[i];
            if (c < 32 || c > 126) continue;  /* skip any residual non-printable */
            if (col >= 34 && c == ' ') {
                linebuf[lb] = '\0';
                if (lb > 0) graphics_draw_text(disp, 16, line_y, linebuf);
                line_y += 12;
                lb = 0; col = 0;
            } else if (lb < 35) {
                linebuf[lb++] = c;
                col++;
            }
        }
        if (G.state == STATE_GENERATING && lb < 35)
            linebuf[lb++] = '_';
        linebuf[lb] = '\0';
        if (lb > 0) graphics_draw_text(disp, 16, line_y, linebuf);

        if (G.dialog_done && ((G.frame / 20) & 1))
            graphics_draw_text(disp, 20, 220, "[A] Next  [B] Close");
        break;
    }
    }
}

// ─── Per-frame generation (one token per frame) ───────────────────────────────

static void update_generating_step(void) {
#ifdef USE_RPC_LLM
    /* RPC path: send prompt to bridge, poll for response */
    if (G.rpc_active) {
        if (!G.rpc_pending && G.gen_plen > 0) {
            /* Build prompt string from gen_pbuf */
            char prompt_str[65];
            int plen = G.gen_plen;
            if (plen > 64) plen = 64;
            for (int i = 0; i < plen; i++)
                prompt_str[i] = (char)G.gen_pbuf[i];
            prompt_str[plen] = '\0';

            /* Send RPC request: max 80 tokens, temp=0.25 (Q8=64) */
            if (llm_rpc_request(prompt_str, 80, 64)) {
                G.rpc_pending = 1;
                G.rpc_send_us = CYCLES_TO_US(TICKS_READ());
                debugf("RPC sent: '%s'\n", prompt_str);
            }
        }

        if (G.rpc_pending) {
            int status = llm_rpc_poll();
            if (status == LLM_STATUS_READY) {
                /* Got response! Copy to dialog buffer */
                int rlen = g_llm_rpc.response_len;
                if (rlen > 80) rlen = 80;
                for (int i = 0; i < rlen; i++) {
                    uint8_t c = g_llm_rpc.response_buf[i];
                    if (c >= 32 && c <= 126)
                        G.dialog_buf[G.dialog_len++] = c;
                    if (c == '\n' || c == '\0') break;
                }
                G.dialog_char = G.dialog_len;

                /* Compute RPC latency */
                uint32_t now_us = CYCLES_TO_US(TICKS_READ());
                G.perf_gen_total_us = now_us - G.rpc_send_us;
                if (G.perf_gen_total_us > 1000) {
                    G.perf_toks_precise = (float)G.dialog_len * 1000000.0f / (float)G.perf_gen_total_us;
                    G.gen_toks_sec = G.perf_toks_precise;
                }
                G.perf_cpu_pct = 5.0f;  /* RPC = minimal CPU usage */

                filter_dialog_buf();
                G.dialog_done = 1;
                G.state = STATE_DIALOG;
                G.rpc_pending = 0;
                debugf("RPC response (%d us): '%s'\n",
                       (int)G.perf_gen_total_us, G.dialog_buf);
                return;
            } else if (status == LLM_STATUS_ERROR) {
                /* RPC failed, fall through to local inference */
                G.rpc_pending = 0;
                G.rpc_active = 0;  /* Disable RPC, use local from now on */
                debugf("RPC error — falling back to local LLM\n");
            }
            /* Still pending — show waiting animation */
            if (G.rpc_pending) {
                G.gen_out_count++;
                G.perf_cpu_pct = 2.0f;
                return;
            }
        }
    }
#endif

    if (!G.ai_ready) {
        // Canned mode: reveal one character every other frame
        if ((G.frame & 1) == 0 && G.dialog_char < G.dialog_len)
            G.dialog_char++;
        if (G.dialog_char >= G.dialog_len) {
            G.dialog_done = 1;
            G.state = STATE_DIALOG;
        }
        return;
    }

    if (G.gen_ppos < G.gen_plen) {
        // Phase 0: feed prompt tokens (discard output; temperature=0)
        {
            uint32_t _t0 = TICKS_READ();
            G.gen_last_tok = sgai_next_token(&G.ai,
                                              G.gen_pbuf[G.gen_ppos], 0);
            G.perf_gen_cycles += TICKS_READ() - _t0;
        }
        G.gen_ppos++;
        if (G.gen_ppos >= G.gen_plen) {
            // Prompt fully fed — gen_last_tok now holds the model's prediction
            // from the last prompt token (greedy argmax, printable ASCII).
            // DO NOT overwrite it — that prediction seeds the first output token.
            G.gen_start_frame = G.frame;
            G.gen_out_count   = 0;
        }
    } else {
        // Phase 1: generate one output token
        // temp_q8=64 → T=0.25 (mild randomness — varied but coherent outputs for demo)
        uint32_t _t0 = TICKS_READ();
        uint8_t tok = sgai_next_token(&G.ai, G.gen_last_tok, 64);
        G.perf_gen_cycles += TICKS_READ() - _t0;
        G.gen_last_tok = tok;
        G.gen_out_count++;

        // Newline = end of Q&A response (training separator); treat like EOS
        if (tok == '\n') tok = 0;
        // Period = end of first sentence — stop here for a clean response.
        // Every training answer ends with "." so the model reliably emits one.
        // Require 8+ output chars first to skip any period inside abbreviations.
        if (tok == '.' && G.gen_out_count >= 8) tok = 0;

        // Append token — sample_logits already restricts to printable ASCII 32-126,
        // but double-check here as defensive measure (unsigned char cast matters)
        if (tok != 0 && (unsigned char)tok >= 32 && (unsigned char)tok <= 126
            && G.dialog_len < (int)sizeof(G.dialog_buf) - 1) {
            G.dialog_buf[G.dialog_len++] = tok;
            G.dialog_char = G.dialog_len;   // show immediately
        }

        // Update tok/s — precise using CP0 cycle counter
        {
            uint32_t now_us = CYCLES_TO_US(TICKS_READ());
            uint32_t elapsed_us = now_us - G.perf_gen_start_us;
            if (elapsed_us > 1000) {  // at least 1ms elapsed
                G.perf_toks_precise = (float)G.gen_out_count * 1000000.0f / (float)elapsed_us;
                G.gen_toks_sec = G.perf_toks_precise;
                G.perf_gen_total_us = elapsed_us;
            }
            // CPU% = inference cycles / frame cycles
            // Average over frames since gen started
            int frames_elapsed = G.frame - G.gen_start_frame;
            if (frames_elapsed > 0) {
                uint32_t total_frame_cycles = (uint32_t)frames_elapsed * FRAME_CYCLES;
                // perf_gen_cycles accumulates across all frames
                G.perf_cpu_pct = (float)G.perf_gen_cycles * 100.0f / (float)total_frame_cycles;
                if (G.perf_cpu_pct > 100.0f) G.perf_cpu_pct = 100.0f;
            }
        }

        // Stop when null/newline token, max output, or buffer full
        if (tok == 0 || G.dialog_len >= 80) {
            filter_dialog_buf();   /* strip training artifacts (helpmeet etc.) */
            G.dialog_done = 1;
            G.state = STATE_DIALOG;
        }
    }
}

// ─── Dialog logic ─────────────────────────────────────────────────────────────

/* Open the dialog option selector (D-pad menu) */
static void open_dialog_select(void) {
    G.state = STATE_DIALOG_SELECT;
    G.dialog_select_idx = 0;
}

/* Start AI generation from a specific prompt string.
 * npc_idx: which NPC is speaking (for persona prefix + canned fallback).
 * prompt: the chosen prompt string (from NPC_DIALOG_OPTIONS or keyboard). */
static void start_dialog_from_prompt(int npc_idx, const char *prompt, int use_persona) {
    G.state       = STATE_GENERATING;
    G.dialog_char = 0;
    G.dialog_done = 0;
    G.dialog_len  = 0;
    G.gen_out_count   = 0;
    G.gen_start_frame = G.frame;
    G.gen_toks_sec    = 0.0f;
    G.perf_gen_cycles   = 0;
    G.perf_gen_total_us = 0;
    G.perf_gen_start_us = CYCLES_TO_US(TICKS_READ());
    G.perf_cpu_pct      = 0.0f;
    G.perf_toks_precise = 0.0f;
    G.perf_show         = 1;
#ifdef USE_RPC_LLM
    G.rpc_pending       = 0;
#endif
    memset(G.dialog_buf, 0, sizeof(G.dialog_buf));
    G.prompt_idx++;

    if (G.ai_ready) {
        sgai_reset(&G.ai);

        /* Build prompt: optional persona prefix + actual prompt.
         * Persona prefix steers the model's voice without burning too many tokens.
         * E.g. "smith says: What is the G4?: " */
        int plen = 0;
        if (use_persona && npc_idx >= 0 && npc_idx < NPC_COUNT) {
            const char *prefix = NPC_PROFILES[npc_idx].persona_prefix;
            int pfx_len = (int)strlen(prefix);
            if (pfx_len > 20) pfx_len = 20;  // cap prefix to preserve response room
            memcpy(G.gen_pbuf, prefix, pfx_len);
            plen = pfx_len;
        }
        int prlen = (int)strlen(prompt);
        if (plen + prlen > (int)sizeof(G.gen_pbuf) - 1)
            prlen = (int)sizeof(G.gen_pbuf) - 1 - plen;
        memcpy(G.gen_pbuf + plen, prompt, prlen);
        plen += prlen;

        G.gen_plen     = plen;
        G.gen_ppos     = 0;
        G.gen_last_tok = G.gen_pbuf[0];
    } else {
        /* Canned fallback: use NPC-specific canned responses */
        int opt_idx = G.dialog_select_idx;
        if (opt_idx < 0 || opt_idx >= DIALOG_OPTIONS) opt_idx = 0;
        int ni = npc_idx;
        if (ni < 0 || ni >= NPC_COUNT) ni = 0;
        const char *resp = NPC_CANNED[ni][opt_idx];
        strncpy((char *)G.dialog_buf, resp, sizeof(G.dialog_buf) - 1);
        G.dialog_len = (int)strlen(resp);
        G.gen_plen   = 0;
        G.gen_ppos   = 0;
    }
}

/* Legacy start_dialog: opens the D-pad option selector */
static void start_dialog(void) {
    open_dialog_select();
}

// ─── Input ────────────────────────────────────────────────────────────────────

/* Begin room transition to target room */
static void begin_room_transition(RoomID target) {
    G.state = STATE_ROOM_TRANSITION;
    G.transition_target = target;
    G.transition_timer = 20;  // 20 frames total fade
}

static void handle_input(void) {
    controller_scan();
    struct controller_data k = get_keys_down();
    switch (G.state) {
    case STATE_ANNIVERSARY:
        if (k.c[0].start || k.c[0].A) G.state = STATE_TITLE;
        break;
    case STATE_TITLE:
        if (k.c[0].start || k.c[0].A) G.state = STATE_DUNGEON;
        break;
    case STATE_DUNGEON:
        if (k.c[0].A) start_dialog();
        if (k.c[0].B) {
            G.state = STATE_KEYBOARD;
            G.kb_row = 0; G.kb_col = 0;
            G.kb_len = 0; G.kb_debounce = 10;
            memset(G.kb_input, 0, sizeof(G.kb_input));
        }
        /* Room transitions via L/R shoulder buttons */
        {
            const RoomExits *ex = &ROOM_MAP[G.current_room];
            if (k.c[0].L && ex->west >= 0)
                begin_room_transition((RoomID)ex->west);
            if (k.c[0].R && ex->east >= 0)
                begin_room_transition((RoomID)ex->east);
            /* D-pad left/right also transition rooms (natural movement) */
            if (k.c[0].left && ex->west >= 0)
                begin_room_transition((RoomID)ex->west);
            if (k.c[0].right && ex->east >= 0)
                begin_room_transition((RoomID)ex->east);
        }
        break;
    case STATE_DIALOG_SELECT:
        /* D-pad up/down to navigate options */
        if (k.c[0].up)
            G.dialog_select_idx = (G.dialog_select_idx + DIALOG_OPTIONS - 1) % DIALOG_OPTIONS;
        if (k.c[0].down)
            G.dialog_select_idx = (G.dialog_select_idx + 1) % DIALOG_OPTIONS;
        /* A = confirm selection, start generation */
        if (k.c[0].A) {
            int npc = G.current_npc;
            if (npc < 0) npc = 0;
            const char *prompt = NPC_DIALOG_OPTIONS[npc][G.dialog_select_idx];
            start_dialog_from_prompt(npc, prompt, 1);
        }
        /* B = cancel, return to exploration */
        if (k.c[0].B)
            G.state = STATE_DUNGEON;
        break;
    case STATE_ROOM_TRANSITION:
        /* Non-interactive during transition */
        break;
    case STATE_KEYBOARD:
        handle_keyboard_input(&k);
        break;
    case STATE_DIALOG:
        if (k.c[0].A) start_dialog();
        if (k.c[0].B) G.state = STATE_DUNGEON;
        break;
    case STATE_GENERATING:
        break;
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

static void game_init(void) {
    memset(&G, 0, sizeof(G));
    G.state = STATE_ANNIVERSARY;
    /* Record boot time via CP0 Count — increments at 46.875 MHz (half CPU clock).
     * Used for real-time splash duration, immune to emulator frame-rate variation. */
    uint32_t _cp0;
    asm volatile("mfc0 %0, $9" : "=r"(_cp0));
    G.anniversary_cp0 = _cp0;
    G.ai.kv  = &G.kv;
#ifdef USE_RPC_LLM
    /* Try to detect Pico bridge for RPC inference */
    G.rpc_active = llm_rpc_detect();
    if (G.rpc_active) {
        debugf("RPC bridge detected! Using remote inference.\n");
    } else {
        debugf("No RPC bridge — using on-cartridge LLM.\n");
    }
#endif

    G.hearts = 8;    // 4 full hearts
    G.magic  = 128;  // full magic bar
    G.current_room = ROOM_DUNGEON;
    G.current_npc  = 0;  // Sophia
    G.dialog_select_idx = 0;
    G.player_x = 160;
    G.player_y = 120;

    int fd = dfs_open("/sophia_weights.bin");
    if (fd >= 0) {
        /* Sized for the model that is actually in filesystem/: 8 layers,
         * 256 embed, int8 weights = 6,750,220 B, rounded up to an 8-byte
         * multiple.  It used to be a hardcoded 3 MB describing a 6-layer
         * 192-embed model that no longer exists, which is exactly how the
         * silent fallback below came to be permanently taken. */
        static uint8_t wbuf[SGAI_WEIGHT_BUF_BYTES] __attribute__((aligned(8)));
        int sz = dfs_size(fd);
        if (sz > 0 && sz <= (int)sizeof(wbuf)) {
            dfs_read(wbuf, 1, sz, fd);
            dfs_close(fd);
            sgai_init(&G.ai, wbuf);
            /* sgai_init heap-allocates its own KV cache; we use the static
             * G.kv instead, so hand the allocation back rather than leaking
             * it. On an 8 MB machine that leak was 2 MB with the float32
             * cache and is 589,828 B with the int8 one. */
            if (G.ai.kv != NULL && G.ai.kv != &G.kv) {
                free(G.ai.kv);
            }
            G.ai.kv   = &G.kv;
            G.ai_ready = 1;
        } else {
            /* The model did not fit and the game silently falls back to canned
             * dialogue. This has bitten us once already and gave no signal at
             * all: up to b8788d5 the blob was 868,364 bytes and loaded fine,
             * then f11042a ("Large model, 8.4M params, Expansion Pak") raised
             * it to 6,750,220 without growing wbuf, so this branch has been
             * taken ever since and the transformer never ran.
             *
             * Growing wbuf is not the fix. Even with the Expansion Pak fitted,
             * weights 6,750,220 + KV 2,097,156 = 8,847,376 exceeds the 8,388,608
             * byte RDRAM ceiling by 458,768 bytes before code, stack or
             * framebuffer. The large model has to be STREAMED, which is what
             * src/expert_cache.c exists for and it is not yet wired to the
             * weight path.
             *
             * Record why, so the failure is inspectable instead of invisible. */
            G.ai_load_failed_size = sz;
            G.ai_load_capacity    = (int)sizeof(wbuf);
            dfs_close(fd);
        }
    }

#if defined(COH_PROBE) || defined(BOOT_PROBE)
/* Two headless verification probes share this IS-Viewer preamble.
 * Each body is separately gated below and only one builds at a time. */
    /* COH_PROBE — coherence/throughput probe.  Runs inside game_init() because
     * headless ares has no GPU and dies at the first rendered frame (F39).
     * Prints ONE LINE PER TOKEN with the CP0 cycle delta for that token, so
     * per-token cost vs position is measured on the real target rather than
     * modelled.  Uses the REAL game sampling path: prompt at temperature 0,
     * output at temperature_q8 = 64, exactly as update_generating() does. */
    /* Headless verification only (FINDINGS F37/F-T012).  Runs inside
     * game_init(), i.e. before main() ever renders a frame, because the
     * headless ares harness has no GPU and dies at the first frame (F39).
     * Prints over the IS-Viewer window, which accepts 32-BIT WRITES ONLY —
     * byte stores silently drop three of every four characters. */
    {
        static char pl[256];
        #define ISV(s) do {                                                    \
            const char *_s = (s);                                              \
            uint32_t _n = (uint32_t)strlen(_s), _w = (_n + 3) & ~3u;           \
            for (uint32_t _i = 0; _i < _w; _i += 4) {                          \
                uint32_t _v = 0;                                               \
                for (int _b = 0; _b < 4; _b++) {                               \
                    uint32_t _c = (_i + _b < _n) ? (uint8_t)_s[_i + _b] : 0;   \
                    _v |= _c << (24 - 8 * _b);                                 \
                }                                                              \
                *(volatile uint32_t *)(uintptr_t)(0xB3FF0020ul + _i) = _v;     \
            }                                                                  \
            *(volatile uint32_t *)(uintptr_t)(0xB3FF0014ul) = _n;              \
        } while (0)

#ifdef COH_PROBE
        sprintf(pl, "COH rdy=%d bits=%d ctx=%d pse=%d\n", G.ai_ready, G.ai.w_bits,
                (int)SGAI_CTX,
#ifdef SGAI_PSE_OFF
                0
#else
                1
#endif
                );
        ISV(pl);
        if (G.ai_ready) {
            const char *p0 = COH_PROMPT;
            int P = (int)strlen(p0);
            static char out[COH_NGEN + 1];
            uint32_t t0, t1, cum = 0;
            sgai_reset(&G.ai);
            uint8_t tok = 0;
            for (int i = 0; i < P; i++) {
                asm volatile("mfc0 %0, $9" : "=r"(t0));
                tok = sgai_next_token(&G.ai, (uint8_t)p0[i], 0);
                asm volatile("mfc0 %0, $9" : "=r"(t1));
                cum += (t1 - t0);
                sprintf(pl, "T pmt %3d cp0=%u cum=%u ch=%d\n", i,
                        (unsigned)(t1 - t0), (unsigned)cum, (int)tok);
                ISV(pl);
            }
            for (int i = 0; i < COH_NGEN; i++) {
                asm volatile("mfc0 %0, $9" : "=r"(t0));
                if (i) tok = sgai_next_token(&G.ai, tok, COH_TEMP);
                asm volatile("mfc0 %0, $9" : "=r"(t1));
                cum += (t1 - t0);
                out[i] = (char)tok;
                sprintf(pl, "T gen %3d cp0=%u cum=%u ch=%d\n", i,
                        (unsigned)(t1 - t0), (unsigned)cum, (int)tok);
                ISV(pl);
            }
            out[COH_NGEN] = 0;
            sprintf(pl, "COH TEXT %s\n", out); ISV(pl);
        }
        ISV("COH_DONE\n");
        while (1) { }
    }
#endif /* COH_PROBE */

#ifdef BOOT_PROBE
        sprintf(pl, "BOOT rdy=%d bits=%d bufbytes=%d ctx=%d failsz=%d cap=%d\n",
                G.ai_ready, G.ai.w_bits, (int)SGAI_WEIGHT_BUF_BYTES,
                (int)SGAI_CTX, G.ai_load_failed_size, G.ai_load_capacity);
        ISV(pl);
        if (G.ai_ready) {
            sgai_reset(&G.ai);
#ifdef PROBE_LONG
            const char *p = "Who are you?";
            const int PLEN = 12, NGEN = 48;
#else
            const char *p = "Elya";
            const int PLEN = 4, NGEN = 16;
#endif
            uint8_t tok = 0;
            for (int i = 0; i < PLEN; i++)
                tok = sgai_next_token(&G.ai, (uint8_t)p[i], 0);
            static char out[64]; static char tl[420]; int tn = 0;
            uint32_t t0, t1, total = 0;
            for (int i = 0; i < NGEN; i++) {
                asm volatile("mfc0 %0, $9" : "=r"(t0));
                if (i) tok = sgai_next_token(&G.ai, tok, 0);
                asm volatile("mfc0 %0, $9" : "=r"(t1));
                total += (t1 - t0);
                out[i] = (char)tok;
                tn += sprintf(tl + tn, "%d ", (int)tok);
            }
            out[NGEN] = 0;
            { int off = 0; while (off < tn) { char sav = tl[off+120]; int cut = (off+120<tn)?120:(tn-off);
                sprintf(pl, "GAME TOKS %.*s\n", cut, tl+off); ISV(pl); (void)sav; off += cut; } }
            sprintf(pl, "GAME TEXT %s\n", out);  ISV(pl);
            sprintf(pl, "GAME CP0 gen16=%u\n", (unsigned)total); ISV(pl);
        }
#ifdef USE_RSP_MATMUL
        { extern uint32_t rsp_mm_calls_rsp, rsp_mm_calls_cpu;
          extern uint32_t rsp_t_stage, rsp_t_disp, rsp_t_epi;
          extern uint32_t rsp_n_wdma, rsp_b_w, rsp_b_out;
          sprintf(pl, "RSPPATH rsp=%u cpu=%u\n", (unsigned)rsp_mm_calls_rsp, (unsigned)rsp_mm_calls_cpu); ISV(pl);
          sprintf(pl, "RSPPHASE stage=%u disp=%u epi=%u\n",
                  (unsigned)rsp_t_stage, (unsigned)rsp_t_disp, (unsigned)rsp_t_epi); ISV(pl);
          sprintf(pl, "RSPDMA wdma=%u wKiB=%u outKiB=%u\n",
                  (unsigned)rsp_n_wdma, (unsigned)rsp_b_w, (unsigned)rsp_b_out); ISV(pl); }
#endif
        ISV("BOOT_DONE\n");
        while (1) { }
    }
#endif /* BOOT_PROBE */
#endif
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    controller_init();
    timer_init();
    dfs_init(DFS_DEFAULT_LOCATION);
    rdpq_init();
    audio_init(MUSIC_FREQ, 4);   // 22kHz, 4 buffers for smooth square-wave

    game_init();

    while (1) {
        G.frame++;

        // Auto-advance anniversary screen after ~5 seconds using CP0 real-time clock.
        // 46,875,000 ticks = 1 second on N64. Immune to emulator frame-rate variation.
        if (G.state == STATE_ANNIVERSARY) {
            uint32_t _now;
            asm volatile("mfc0 %0, $9" : "=r"(_now));
            if ((_now - G.anniversary_cp0) >= 46875000u * 5u)
                G.state = STATE_TITLE;
        }

        // Room transition logic
        if (G.state == STATE_ROOM_TRANSITION) {
            G.transition_timer--;
            if (G.transition_timer == 10) {
                // Midpoint: switch room
                G.current_room = G.transition_target;
                G.attack_timer = 0;  // reset combat on room change
            }
            if (G.transition_timer <= 0) {
                G.state = STATE_DUNGEON;
            }
        }

        // Update forge sparks
        if (G.current_room == ROOM_FORGE)
            update_sparks();

        // Per-frame AI generation step
        if (G.state == STATE_GENERATING)
            update_generating_step();

        // Get ONE surface for this frame
        surface_t *disp = display_get();

        // ── RDP graphics pass ──────────────────────────────────────────────
        G.perf_frame_start = TICKS_READ();
        rdpq_attach(disp, NULL);

        if (G.state == STATE_ANNIVERSARY) {
            scene_anniversary();
        } else if (G.state == STATE_TITLE) {
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 20, 255));
            fillrect(30,  30, 260, 6, RGBA32(180, 140, 0, 255));
            fillrect(30, 130, 260, 6, RGBA32(180, 140, 0, 255));
        } else if (G.state == STATE_KEYBOARD) {
            scene_keyboard();
        } else {
            scene_dungeon();
            if (G.state == STATE_DIALOG_SELECT)
                scene_dialog_select();
            if (G.state == STATE_DIALOG || G.state == STATE_GENERATING)
                scene_dialog_box();
            if (G.state == STATE_ROOM_TRANSITION)
                scene_room_transition();
        }

        // Wait for RDP to finish before CPU text pass
        rdpq_detach_wait();

        // ── CPU text pass (same surface, no buffer switch → no flicker) ───
        draw_text(disp);

        display_show(disp);

        // ── Music ──────────────────────────────────────────────────────────
        music_update();

        // ── Input ──────────────────────────────────────────────────────────
        handle_input();
    }

    return 0;
}
