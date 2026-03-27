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

// ─── Game State ───────────────────────────────────────────────────────────────

typedef enum {
    STATE_ANNIVERSARY,   // Legend of Elya splash screen
    STATE_TITLE,
    STATE_DUNGEON,
    STATE_DIALOG,
    STATE_GENERATING,
    STATE_KEYBOARD,      // On-screen D-pad keyboard for custom NPC prompts
} GameState;

typedef struct {
    GameState state;
    int dialog_char;
    int dialog_done;
    uint8_t dialog_buf[128];
    int dialog_len;
    int frame;
    uint32_t anniversary_cp0;  // CP0 Count at boot for real-time splash duration
    // AI
    SGAIState ai;
    SGAIKVCache kv;
    int ai_ready;
    int prompt_idx;
    // Per-frame generation state (enables tok/s display)
    uint8_t gen_pbuf[64];   // copy of current prompt bytes
    int gen_plen;           // prompt byte count
    int gen_ppos;           // bytes fed so far (prompt phase)
    uint8_t gen_last_tok;   // last token for chaining
    int gen_out_count;      // output tokens generated
    int gen_start_frame;    // frame when output phase began
    float gen_toks_sec;     // computed tokens/second
    // On-screen keyboard state
    char kb_buf[33];        // 32-char input buffer + null terminator
    int  kb_len;            // number of characters typed
    int  kb_col;            // cursor column (0..KB_COLS-1)
    int  kb_row;            // cursor row (0..KB_ROWS-1)
    int  kb_blink;          // cursor blink frame counter
    // Music sequencer
    int music_note_idx;     // current note in sequence
    int music_sample_pos;   // samples elapsed in current note
    int music_phase;        // square wave phase accumulator
    // Combat & HUD
    int attack_timer;       // frames remaining in attack animation (0 = idle)
    int attack_target;      // 0 = stalfos, 1 = keese
    int hearts;             // half-heart count: 8 = 4 full hearts, 0 = dead
    int magic;              // magic bar 0-128 (128 = full)
} GameCtx;

static GameCtx G;

// Forward declaration (defined later after music/audio setup)
static void fillrect(int x, int y, int w, int h, color_t c);

// ─── On-Screen D-Pad Keyboard ─────────────────────────────────────────────────
// Grid: 7 columns × 6 rows = 42 keys
// Layout: A-Z (26) + space + punctuation + BACK(←) + DONE(✓)
// Fits comfortably on 320×240 at N64 resolution

#define KB_COLS      7
#define KB_ROWS      6
#define KB_MAX_LEN   32

// Character map: row-major, 7 cols × 6 rows = 42 entries
// \x01 = BACK(delete), \x02 = DONE(submit), \x00 = unused
static const char KB_KEYS[KB_ROWS][KB_COLS] = {
    { 'A', 'B', 'C', 'D', 'E', 'F', 'G' },
    { 'H', 'I', 'J', 'K', 'L', 'M', 'N' },
    { 'O', 'P', 'Q', 'R', 'S', 'T', 'U' },
    { 'V', 'W', 'X', 'Y', 'Z', ' ', '!' },
    { '?', '.', ',', '\'',':', '-', '0' },
    { '1', '2', '3', '4', '5', '\x01', '\x02' },
};

// Key cell dimensions and top-left origin on screen
#define KB_CELL_W    40    // px per key cell (320 / 8 ≈ 40)
#define KB_CELL_H    16    // px per key cell
#define KB_ORIG_X    12    // left margin
#define KB_ORIG_Y    82    // top of grid (leaves room for typed text above)

// Label strings for special keys
#define KB_LABEL_BACK  "<"
#define KB_LABEL_DONE  "OK"

// ─── Keyboard draw (RDP fill pass) ────────────────────────────────────────────

static void scene_keyboard_bg(void) {
    // Dark background panel
    fillrect(0, 0, 320, 240, RGBA32(4, 4, 28, 255));

    // Title bar
    fillrect(0, 0, 320, 14, RGBA32(0, 0, 60, 255));
    fillrect(0, 13, 320, 1, RGBA32(215, 175, 0, 255));

    // Input text box (shows typed characters)
    fillrect(8, 20, 304, 18, RGBA32(0, 0, 40, 255));
    fillrect(8, 20, 304, 1,  RGBA32(100, 100, 200, 255));
    fillrect(8, 37, 304, 1,  RGBA32(100, 100, 200, 255));
    fillrect(8, 20,   1, 18, RGBA32(100, 100, 200, 255));
    fillrect(311,20,  1, 18, RGBA32(100, 100, 200, 255));

    // Hint bar at very bottom
    fillrect(0, 228, 320, 12, RGBA32(0, 0, 40, 255));
    fillrect(0, 228, 320, 1,  RGBA32(215, 175, 0, 255));

    // Draw key grid cells
    for (int r = 0; r < KB_ROWS; r++) {
        for (int c = 0; c < KB_COLS; c++) {
            char key = KB_KEYS[r][c];
            if (key == '\x00') continue;

            int x = KB_ORIG_X + c * KB_CELL_W;
            int y = KB_ORIG_Y + r * KB_CELL_H;
            int is_cursor = (r == G.kb_row && c == G.kb_col);

            color_t bg, border;
            if (is_cursor) {
                // Highlighted cursor cell
                bg     = RGBA32(200, 160, 0, 255);
                border = RGBA32(255, 220, 60, 255);
            } else if (key == '\x01') {
                // BACK key — reddish
                bg     = RGBA32(100, 30, 30, 255);
                border = RGBA32(180, 60, 60, 255);
            } else if (key == '\x02') {
                // DONE key — greenish
                bg     = RGBA32(20, 80, 20, 255);
                border = RGBA32(60, 200, 60, 255);
            } else {
                bg     = RGBA32(20, 20, 60, 255);
                border = RGBA32(60, 60, 120, 255);
            }

            // Fill cell background (inset 1px)
            fillrect(x + 1, y + 1, KB_CELL_W - 2, KB_CELL_H - 2, bg);
            // Border (1px outline)
            fillrect(x,     y,     KB_CELL_W,     1,              border);
            fillrect(x,     y + KB_CELL_H - 1, KB_CELL_W, 1,     border);
            fillrect(x,     y,     1,             KB_CELL_H,      border);
            fillrect(x + KB_CELL_W - 1, y, 1,    KB_CELL_H,      border);
        }
    }
}

// ─── Keyboard text pass (CPU overlay) ─────────────────────────────────────────

static void draw_keyboard_text(surface_t *disp) {
    // Title
    graphics_draw_text(disp, 70, 3, "Type your question for Sophia:");

    // Typed text buffer
    char display_buf[35];
    int dlen = G.kb_len;
    if (dlen > 32) dlen = 32;
    for (int i = 0; i < dlen; i++)
        display_buf[i] = G.kb_buf[i];
    // Blinking cursor
    if ((G.kb_blink / 15) & 1)
        display_buf[dlen++] = '_';
    display_buf[dlen] = '\0';
    graphics_draw_text(disp, 12, 25, display_buf);

    // Draw key labels
    for (int r = 0; r < KB_ROWS; r++) {
        for (int c = 0; c < KB_COLS; c++) {
            char key = KB_KEYS[r][c];
            if (key == '\x00') continue;

            int x = KB_ORIG_X + c * KB_CELL_W;
            int y = KB_ORIG_Y + r * KB_CELL_H;
            // Center label: each char ~6px wide, cell 40px → offset ~(40-6)/2 = 17
            // For single char center: x + (40 - 6) / 2 = x + 17
            char label[3];
            if (key == '\x01') {
                label[0] = '<'; label[1] = '\0';
            } else if (key == '\x02') {
                label[0] = 'O'; label[1] = 'K'; label[2] = '\0';
            } else if (key == ' ') {
                label[0] = '_'; label[1] = '\0';  // visible space indicator
            } else {
                label[0] = key; label[1] = '\0';
            }

            // Center single char in cell
            int lw = (key == '\x02') ? 12 : 6;  // OK = 2 chars * 6
            int tx = x + (KB_CELL_W - lw) / 2;
            int ty = y + (KB_CELL_H - 7) / 2 + 1;
            graphics_draw_text(disp, tx, ty, label);
        }
    }

    // Hint bar
    graphics_draw_text(disp, 8, 230, "D:move  A:type  B:del  START:send");
}

// ─── Keyboard input handler ───────────────────────────────────────────────────

static void keyboard_init(void) {
    memset(G.kb_buf, 0, sizeof(G.kb_buf));
    G.kb_len  = 0;
    G.kb_col  = 0;
    G.kb_row  = 0;
    G.kb_blink = 0;
}

/* Submit the typed keyboard buffer as an LLM prompt.
 * Appends ": " suffix so it matches training format, then enters GENERATING. */
static void keyboard_submit(void) {
    char prompt[40];
    int plen = G.kb_len;
    if (plen <= 0) return;
    if (plen > 30) plen = 30;
    memcpy(prompt, G.kb_buf, plen);
    prompt[plen++] = ':';
    prompt[plen++] = ' ';
    prompt[plen]   = '\0';

    G.state           = STATE_GENERATING;
    G.dialog_char     = 0;
    G.dialog_done     = 0;
    G.dialog_len      = 0;
    G.gen_out_count   = 0;
    G.gen_start_frame = G.frame;
    G.gen_toks_sec    = 0.0f;
    memset(G.dialog_buf, 0, sizeof(G.dialog_buf));

    if (G.ai_ready) {
        sgai_reset(&G.ai);
        if (plen > (int)sizeof(G.gen_pbuf) - 1)
            plen = (int)sizeof(G.gen_pbuf) - 1;
        memcpy(G.gen_pbuf, prompt, plen);
        G.gen_plen     = plen;
        G.gen_ppos     = 0;
        G.gen_last_tok = G.gen_pbuf[0];
    } else {
        // Canned fallback — use frame counter for variety
        uint32_t entropy = (uint32_t)(TICKS_READ()) ^ ((uint32_t)G.frame << 3);
        const char *resp = CANNED[entropy % N_CANNED];
        strncpy((char *)G.dialog_buf, resp, sizeof(G.dialog_buf) - 1);
        G.dialog_len = (int)strlen(resp);
        G.gen_plen   = 0;
        G.gen_ppos   = 0;
    }
}

static void handle_keyboard_input(struct controller_data *k) {
    G.kb_blink++;

    // D-pad navigation — wraps around grid edges
    if (k->c[0].up)    { G.kb_row = (G.kb_row - 1 + KB_ROWS) % KB_ROWS; G.kb_blink = 0; }
    if (k->c[0].down)  { G.kb_row = (G.kb_row + 1) % KB_ROWS;           G.kb_blink = 0; }
    if (k->c[0].left)  { G.kb_col = (G.kb_col - 1 + KB_COLS) % KB_COLS; G.kb_blink = 0; }
    if (k->c[0].right) { G.kb_col = (G.kb_col + 1) % KB_COLS;           G.kb_blink = 0; }

    // A button: select character or trigger special action
    if (k->c[0].A) {
        char key = KB_KEYS[G.kb_row][G.kb_col];
        if (key == '\x01') {
            // BACK key — delete last character
            if (G.kb_len > 0) G.kb_len--;
            G.kb_buf[G.kb_len] = '\0';
        } else if (key == '\x02') {
            // OK/DONE key — submit prompt to LLM
            keyboard_submit();
        } else if (G.kb_len < KB_MAX_LEN) {
            // Regular character — append to buffer
            G.kb_buf[G.kb_len++] = key;
            G.kb_buf[G.kb_len]   = '\0';
        }
    }

    // B button: backspace shortcut (always accessible regardless of cursor)
    if (k->c[0].B) {
        if (G.kb_len > 0) G.kb_len--;
        G.kb_buf[G.kb_len] = '\0';
    }

    // Start button: submit prompt directly (quick submit shortcut)
    if (k->c[0].start && G.kb_len > 0) {
        keyboard_submit();
    }
}

/* N64 hardware entropy — XOR CPU cycle counter low bits with frame,
 * last token, AND prompt_idx (sequential counter).
 * prompt_idx is the critical fallback: emulators run TICKS_READ()
 * deterministically, so we rely on prompt_idx advancing each A-press
 * to guarantee a different topic every conversation.
 * On real hardware, TICKS jitter adds extra unpredictability. */
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

// ─── Dungeon Scene ────────────────────────────────────────────────────────────

static void scene_dungeon(void) {
    int f = G.frame;

    // ── Keese position (hoisted — needed by attack overlay too) ───────────────
    int kx = 150 + (int)(sinf(f * 0.045f) * 55.0f) + (int)(sinf(f * 0.13f) * 18.0f);
    int ky =  38 + (int)(sinf(f * 0.031f) * 22.0f);

    // ── Attack auto-trigger (every ~3 seconds while in dungeon) ──────────────
    if (G.state == STATE_DUNGEON && G.attack_timer <= 0
            && f > 120 && (f % 180) == 0) {
        G.attack_timer  = 42;
        G.attack_target = (f / 180) & 1;  // alternate stalfos / keese
    }
    int atk = G.attack_timer;
    if (atk > 0) G.attack_timer--;

    // Sky/background
    fillrect(0, 0, 320, 148, RGBA32(8, 4, 16, 255));

    // Stone wall rows
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 12; col++) {
            int offset = (row & 1) * 16;
            int bx = col * 32 + offset - 16;
            int by = row * 18;
            if (bx + 30 < 0 || bx > 320) continue;
            int shade = 28 + ((col + row) % 3) * 7;
            fillrect(bx+1, by+1, 30, 16, RGBA32(shade, shade-6, shade+4, 255));
        }
    }

    // Floor
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 11; col++) {
            int shade = 18 + ((col + row) & 1) * 7;
            fillrect(col*32, 100 + row*10, 31, 9, RGBA32(shade,shade,shade+4,255));
        }
    }

    // Torch (flickering)
    int flick = (f / 4) & 1;
    fillrect(18, 56, 8+flick*2, 14, RGBA32(255, 140+flick*30, 0, 255));
    fillrect(16, 66, 12+flick*2, 4, RGBA32(200, 60, 0, 255));
    fillrect(22, 70, 2, 18, RGBA32(80, 60, 40, 255));

    // Sophia Elya pixel sprite (right side)
    int bob = (int)(sinf(f * 0.08f) * 2.0f);
    int sx = 204, sy = 72 + bob;

    // ── Shield (left arm, kite-style) ────────────────────────────────────
    {
        int shx = sx - 20, shy = sy + 6;
        // Kite-shaped body (wider in middle, narrows to point)
        fillrect(shx+1, shy,      10,  2, RGBA32(20,  50, 130, 255));
        fillrect(shx,   shy+2,    12, 10, RGBA32(20,  50, 130, 255));
        fillrect(shx+1, shy+12,   10,  4, RGBA32(20,  50, 130, 255));
        fillrect(shx+2, shy+16,    8,  3, RGBA32(20,  50, 130, 255));
        fillrect(shx+4, shy+19,    4,  4, RGBA32(20,  50, 130, 255));
        // Gold trim border
        fillrect(shx,   shy,      12,  2, RGBA32(215,175,  0, 255));
        fillrect(shx,   shy+2,     2, 20, RGBA32(215,175,  0, 255));
        fillrect(shx+10,shy+2,     2, 20, RGBA32(215,175,  0, 255));
        // Elya gem emblem (mini crystal)
        fillrect(shx+5, shy+4,     2,  2, RGBA32(80,220,255, 255));  // top
        fillrect(shx+3, shy+6,     6,  3, RGBA32(40,160,200, 255));  // middle
        fillrect(shx+5, shy+9,     2,  2, RGBA32(80,220,255, 255));  // bottom
    }

    // ── Sophia body ───────────────────────────────────────────────────────
    fillrect(sx-6, sy+14, 18, 28, RGBA32(60, 30, 100, 255));  // dress
    fillrect(sx-4, sy+8,  14, 14, RGBA32(80, 50, 120, 255));  // torso
    fillrect(sx-3, sy,    12, 12, RGBA32(220,180,140, 255));   // head
    fillrect(sx-4, sy-2,  14,  5, RGBA32(80, 30, 10, 255));   // hair
    fillrect(sx,   sy+3,   2,  2, RGBA32(20, 20, 80, 255));   // left eye
    fillrect(sx+5, sy+3,   2,  2, RGBA32(20, 20, 80, 255));   // right eye

    // ── Sword (right arm, blade raised upward) ────────────────────────────
    {
        int swx = sx + 13;
        // Blade (silver, drawn first so guard overlaps cleanly)
        fillrect(swx,   sy-16,  2, 28, RGBA32(195,215,235, 255)); // main blade
        fillrect(swx+1, sy-16,  1, 14, RGBA32(240,250,255, 255)); // edge highlight
        // Blade tip
        fillrect(swx,   sy-18,  2,  2, RGBA32(220,235,250, 255));
        // Crossguard (gold)
        fillrect(swx-5, sy+12,  12,  3, RGBA32(215,175,  0, 255));
        fillrect(swx-5, sy+12,  12,  1, RGBA32(255,220, 60, 255)); // guard highlight
        // Handle
        fillrect(swx,   sy+15,   2,  7, RGBA32(110, 60, 15, 255));
        // Pommel (gold ball)
        fillrect(swx-1, sy+22,   4,  3, RGBA32(215,175,  0, 255));
        // Animated gleam — travels up the blade every ~2 seconds
        int gleam = (f / 7) % 22;
        fillrect(swx, sy + 10 - gleam, 1, 4, RGBA32(255,255,255, 200));
    }

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
    // Hit flash: Stalfos (phase 2, atk 20→1, target=stalfos)
    if (atk > 0 && atk <= 22 && G.attack_target == 0)
        fillrect(ex-5, ey, 14, 44, RGBA32(255, 255, 255, 200));

    // ── Keese (bat enemy) — animated wings, erratic flight path ────────────
    // kx/ky hoisted to top of function for attack overlay use
    int wing = (f / 5) & 1;   // wing flap: every 5 frames
    // Dark body
    fillrect(kx-3, ky-2, 6, 5, RGBA32(25, 15, 35, 255));
    // Wings: alternate up-spread / down-spread
    if (wing == 0) {
        fillrect(kx-12, ky-5,  9, 6, RGBA32(50, 35, 70, 255));  // left wing up
        fillrect(kx+3,  ky-5,  9, 6, RGBA32(50, 35, 70, 255));  // right wing up
        fillrect(kx-12, ky-1,  4, 3, RGBA32(35, 22, 50, 255));  // left fold
        fillrect(kx+8,  ky-1,  4, 3, RGBA32(35, 22, 50, 255));  // right fold
    } else {
        fillrect(kx-12, ky+1,  9, 5, RGBA32(50, 35, 70, 255));  // left wing down
        fillrect(kx+3,  ky+1,  9, 5, RGBA32(50, 35, 70, 255));  // right wing down
        fillrect(kx-10, ky-2,  4, 3, RGBA32(35, 22, 50, 255));  // left fold
        fillrect(kx+6,  ky-2,  4, 3, RGBA32(35, 22, 50, 255));  // right fold
    }
    // Glowing red eyes
    fillrect(kx-1, ky-1, 2, 2, RGBA32(255, 60,  20, 255));
    fillrect(kx+2, ky-1, 2, 2, RGBA32(255, 60,  20, 255));
    // Hit flash: Keese (phase 2 of attack, atk 20→1, target=keese)
    if (atk > 0 && atk <= 22 && G.attack_target == 1)
        fillrect(kx-12, ky-5, 25, 13, RGBA32(255, 255, 255, 200));

    // ── Treasure Chest ─────────────────────────────────────────────────────
    // Sits on the floor, center-right of dungeon
    {
        int cx = 146, cy = 112;
        // Chest body (lower)
        fillrect(cx,    cy+10, 28, 20, RGBA32(100, 62, 18, 255));
        // Chest lid (upper, slightly lighter)
        fillrect(cx,    cy,    28, 12, RGBA32(130, 82, 28, 255));
        // Curved lid top highlight
        fillrect(cx+2,  cy-1,  24,  2, RGBA32(155, 100, 40, 255));
        // Gold trim — horizontal bands
        fillrect(cx,    cy+10, 28,  2, RGBA32(215, 175,  0, 255));  // lid seam
        fillrect(cx,    cy+28, 28,  2, RGBA32(215, 175,  0, 255));  // bottom
        fillrect(cx,    cy,    28,  2, RGBA32(215, 175,  0, 255));  // top
        // Gold trim — vertical sides
        fillrect(cx,    cy,     2, 30, RGBA32(215, 175,  0, 255));  // left
        fillrect(cx+26, cy,     2, 30, RGBA32(215, 175,  0, 255));  // right
        // Center lock plate
        fillrect(cx+11, cy+8,   6,  6, RGBA32(215, 175,  0, 255));
        // Keyhole
        fillrect(cx+13, cy+9,   2,  2, RGBA32(30,  20,   5, 255));
        fillrect(cx+13, cy+11,  2,  3, RGBA32(30,  20,   5, 255));
        // Shimmer glow on lid — pulses every ~30 frames
        int glow = ((f / 30) & 1) ? 60 : 20;
        fillrect(cx+3,  cy+3,   8,  2, RGBA32(255, 230, 100, glow));
        fillrect(cx+14, cy+3,   8,  2, RGBA32(255, 230, 100, glow));
    }

    // ── Attack slash trail ──────────────────────────────────────────────────
    if (atk > 22) {
        // Swing phase (atk 42→23): slash arc from sword tip toward enemy
        int prog = 42 - atk;         // 0 (start) → 19 (end of swing)
        int swx_tip = sx + 14;       // sword tip x
        int swy_tip = sy - 14;       // sword tip y
        int tx = (G.attack_target == 0) ? ex + 1 : kx;
        int ty = (G.attack_target == 0) ? ey + 4 : ky;
        // Interpolate: tip → enemy, progress 0-19 out of 19 steps
        int lx = swx_tip + ((tx - swx_tip) * prog) / 19;
        int ly = swy_tip + ((ty - swy_tip) * prog) / 19;
        // Bright spark at leading edge
        fillrect(lx-2, ly-2, 6, 6, RGBA32(255, 255, 120, 255));
        fillrect(lx-1, ly-1, 4, 4, RGBA32(255, 240, 60,  255));
        // Short trail 3 steps behind
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
        // Impact spark phase (atk 22→1): radiating sparks at hit point
        int hx2 = (G.attack_target == 0) ? ex + 1 : kx;
        int hy2 = (G.attack_target == 0) ? ey + 4 : ky;
        int sp  = 22 - atk;          // 0-21, grows outward
        fillrect(hx2 + sp,     hy2 - sp/2, 3, 3, RGBA32(255, 230,   0, 255));
        fillrect(hx2 - sp,     hy2 + sp/2, 3, 3, RGBA32(255, 200,  50, 255));
        fillrect(hx2 + sp/2,   hy2 + sp,   3, 3, RGBA32(255, 160,   0, 255));
        fillrect(hx2 - sp/2,   hy2 - sp,   3, 3, RGBA32(255, 100,   0, 255));
        // Central bright star (fades after sp>8)
        if (sp < 9)
            fillrect(hx2 - 3, hy2 - 3, 7, 7, RGBA32(255, 255, 200, 255));
    }

    // ── HUD: 4 Hearts + Magic Bar (drawn last = on top) ─────────────────────
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

    // Floor line
    fillrect(0, 148, 320, 2, RGBA32(40,30,60,255));
}

static void scene_dialog_box(void) {
    fillrect(8, 150, 304, 80, RGBA32(0, 0, 60, 255));
    fillrect(8, 150, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 228, 304, 2,  RGBA32(215,175,0,255));
    fillrect(8, 150,   2, 80, RGBA32(215,175,0,255));
    fillrect(310,150,  2, 80, RGBA32(215,175,0,255));
    fillrect(11, 153, 298, 1, RGBA32(100,80,20,255));
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
        else
            graphics_draw_text(disp,  72, 118, "[Sophia AI: Demo Mode]");
        graphics_draw_text(disp,  80, 155, "Press START to enter");
        graphics_draw_text(disp, 104, 170, "the dungeon...");
        break;

    case STATE_DUNGEON:
        graphics_draw_text(disp, 186, 3,  "MP");   // magic bar label
        graphics_draw_text(disp,  10, 220, "[A] Talk  [B] Type  (auto-attack)");
        break;

    case STATE_KEYBOARD:
        draw_keyboard_text(disp);
        break;

    case STATE_DIALOG:
    case STATE_GENERATING: {
        graphics_draw_text(disp, 16, 158, "Sophia Elya:");

        // Tok/s indicator during live generation
        if (G.state == STATE_GENERATING && G.gen_out_count > 0) {
            char spdbuf[10];
            int whole = (int)G.gen_toks_sec;
            int frac  = (int)((G.gen_toks_sec - (float)whole) * 10.0f);
            if (whole > 99) whole = 99;
            spdbuf[0] = (whole >= 10) ? ('0' + whole / 10) : ' ';
            spdbuf[1] = '0' + (whole % 10);
            spdbuf[2] = '.';
            spdbuf[3] = '0' + frac;
            spdbuf[4] = 't';
            spdbuf[5] = '/';
            spdbuf[6] = 's';
            spdbuf[7] = '\0';
            graphics_draw_text(disp, 212, 158, spdbuf);
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
        G.gen_last_tok = sgai_next_token(&G.ai,
                                          G.gen_pbuf[G.gen_ppos], 0);
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
        uint8_t tok = sgai_next_token(&G.ai, G.gen_last_tok, 64);
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

        // Update tok/s
        int elapsed = G.frame - G.gen_start_frame;
        if (elapsed > 0)
            G.gen_toks_sec = (float)G.gen_out_count * 60.0f / (float)elapsed;

        // Stop when null/newline token, max output, or buffer full
        if (tok == 0 || G.dialog_len >= 80) {
            filter_dialog_buf();   /* strip training artifacts (helpmeet etc.) */
            G.dialog_done = 1;
            G.state = STATE_DIALOG;
        }
    }
}

// ─── Dialog logic ─────────────────────────────────────────────────────────────

static void start_dialog(void) {
    G.state       = STATE_GENERATING;
    G.dialog_char = 0;
    G.dialog_done = 0;
    G.dialog_len  = 0;
    G.gen_out_count   = 0;
    G.gen_start_frame = G.frame;
    G.gen_toks_sec    = 0.0f;
    memset(G.dialog_buf, 0, sizeof(G.dialog_buf));

    /* Hardware entropy seed selection — RIP-PoA oscillator trick:
     * XOR CPU cycle counter low bits with frame + last token hash.
     * Low bits of TICKS_READ() vary ~12 bits per A-press due to
     * player reaction time jitter and music sample phase offset.
     * This guarantees a different prompt is chosen nearly every time. */
    uint32_t entropy = N64_ENTROPY();
    int idx = (int)(entropy % N_PROMPTS);
    G.prompt_idx++;   /* also increment for sequential tracking */

    if (G.ai_ready) {
        sgai_reset(&G.ai);
        const char *p = PROMPTS[idx];
        int plen = (int)strlen(p);
        /* Feed bare prompt — no seed prefix.
         * Context=32 tokens; seeds were burning 13-14 tokens leaving <5 for
         * response. Bare prompt (13-20 chars) leaves 12-19 response tokens. */
        if (plen > (int)sizeof(G.gen_pbuf) - 1)
            plen = (int)sizeof(G.gen_pbuf) - 1;
        memcpy(G.gen_pbuf, p, plen);
        G.gen_plen     = plen;
        G.gen_ppos     = 0;
        G.gen_last_tok = G.gen_pbuf[0];
    } else {
        // Canned fallback: entropy-selected from N_CANNED pool
        const char *resp = CANNED[idx % N_CANNED];
        strncpy((char *)G.dialog_buf, resp, sizeof(G.dialog_buf) - 1);
        G.dialog_len = (int)strlen(resp);
        G.gen_plen   = 0;   // signals canned path in update_generating_step
        G.gen_ppos   = 0;
    }
}

// ─── Input ────────────────────────────────────────────────────────────────────

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
            // B opens on-screen keyboard for custom player input
            keyboard_init();
            G.state = STATE_KEYBOARD;
        }
        break;
    case STATE_DIALOG:
        if (k.c[0].A) start_dialog();
        if (k.c[0].B) G.state = STATE_DUNGEON;
        break;
    case STATE_GENERATING:
        break;
    case STATE_KEYBOARD:
        handle_keyboard_input(&k);
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
    G.hearts = 8;    // 4 full hearts
    G.magic  = 128;  // full magic bar

    int fd = dfs_open("/sophia_weights.bin");
    if (fd >= 0) {
        static uint8_t wbuf[1024 * 1024] __attribute__((aligned(8)));  /* 1MB for v5 Q8 4-layer */
        int sz = dfs_size(fd);
        if (sz > 0 && sz <= (int)sizeof(wbuf)) {
            dfs_read(wbuf, 1, sz, fd);
            dfs_close(fd);
            sgai_init(&G.ai, wbuf);
            G.ai.kv   = &G.kv;
            G.ai_ready = 1;
        } else {
            dfs_close(fd);
        }
    }
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

        // Per-frame AI generation step
        if (G.state == STATE_GENERATING)
            update_generating_step();

        // Get ONE surface for this frame
        surface_t *disp = display_get();

        // ── RDP graphics pass ──────────────────────────────────────────────
        rdpq_attach(disp, NULL);

        if (G.state == STATE_ANNIVERSARY) {
            scene_anniversary();
        } else if (G.state == STATE_TITLE) {
            fillrect(0, 0, 320, 240, RGBA32(0, 0, 20, 255));
            fillrect(30,  30, 260, 6, RGBA32(180, 140, 0, 255));
            fillrect(30, 130, 260, 6, RGBA32(180, 140, 0, 255));
        } else if (G.state == STATE_KEYBOARD) {
            scene_keyboard_bg();
        } else {
            scene_dungeon();
            if (G.state == STATE_DIALOG || G.state == STATE_GENERATING)
                scene_dialog_box();
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
