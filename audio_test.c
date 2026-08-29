// SPDX-License-Identifier: MIT
/*
 * audio_test.c — is the music itself broken, or is the LLM breaking it?
 *
 * The game plays a VADPCM track through libdragon's mixer while also driving
 * the RSP for the transformer matmul, ~98.5% of the CPU is inference, and a
 * token is ~330 ms against ~66 ms of audio buffer. That is at least three
 * suspects tangled together, so this ROM removes two of them: the same three
 * assets, the same mixer calls, the same 22.05 kHz / 4 buffers — and NO model,
 * NO RSP matmul, nothing else running.
 *
 *   clean here  -> the assets and the mixer path are fine, and the game's
 *                  problem is contention with inference (or my per-layer
 *                  sgai_tick hook, which calls mixer_poll -- itself an rspq
 *                  user -- in the middle of a matmul dispatch).
 *   broken here -> the problem is the asset or the mixer call sequence, and
 *                  none of the LLM work is implicated at all.
 *
 * A/B/C switch tracks with the D-pad so all three can be heard.
 * Build: make audiotest
 */
#include <libdragon.h>
#include <stdio.h>
#include <string.h>

static char pl[192];
#define ISV(s) do {                                                        \
    const char *_s = (s);                                                  \
    uint32_t _n = (uint32_t)strlen(_s), _w = (_n + 3) & ~3u;               \
    for (uint32_t _i = 0; _i < _w; _i += 4) {                              \
        uint32_t _v = 0;                                                   \
        for (int _b = 0; _b < 4; _b++) {                                   \
            uint32_t _c = (_i + _b < _n) ? (uint8_t)_s[_i + _b] : 0;       \
            _v |= _c << (24 - 8 * _b);                                     \
        }                                                                  \
        *(volatile uint32_t *)(uintptr_t)(0xB3FF0020ul + _i) = _v;         \
    }                                                                      \
    *(volatile uint32_t *)(uintptr_t)(0xB3FF0014ul) = _n;                  \
} while (0)

static volatile uint32_t g_vbl = 0;
static void vbl(void) { g_vbl++; }

int main(void)
{
    display_init(RESOLUTION_320x240, DEPTH_16_BPP, 2, GAMMA_NONE, ANTIALIAS_RESAMPLE);
    timer_init();
    register_VI_handler(vbl);
    dfs_init(DFS_DEFAULT_LOCATION);
    controller_init();

    /* ONE VARIABLE. The game calls rdpq_init() and its audio is garbled; this
     * ROM did not and its audio was clean. rdpq and the mixer are both rspq
     * users, so this build adds nothing except that call. -DNO_RDPQ omits it
     * again, so the same binary tests both sides. */
#ifndef NO_RDPQ
    rdpq_init();
#endif
    audio_init(22050, 4);
    mixer_init(2);

    static const char *const TRACKS[] = {
        "rom:/dungeon.wav64", "rom:/library.wav64", "rom:/forge.wav64"
    };
    static const char *const NAMES[] = { "DUNGEON", "LIBRARY", "FORGE" };

    static wav64_t song;
    int cur = 0, debounce = 0;
    wav64_open(&song, TRACKS[cur]);
    wav64_set_loop(&song, true);
    wav64_play(&song, 0);
    mixer_ch_set_vol(0, 0.55f, 0.55f);

    uint32_t polls = 0, writes = 0;
    sprintf(pl, "AU INIT rdpq=%d freq=%d buflen=%d vi=%d\n",
#ifdef NO_RDPQ
            0,
#else
            1,
#endif
            audio_get_frequency(), audio_get_buffer_length(),
            (int)(get_tv_type() == TV_PAL ? 50 : 60));
    ISV(pl);
    uint32_t t_start = g_vbl;

    while (1) {
        controller_scan();
        struct controller_data k = get_keys_down();
        if (debounce > 0) debounce--;
        if (!debounce && (k.c[0].left || k.c[0].right)) {
            cur = (cur + (k.c[0].right ? 1 : 2)) % 3;
            /* Stop the channel AND close the old wav64 before reopening the
             * same struct.  Reopening an already-open wav64_t leaks its state
             * and leaves the mixer pointing at a half-torn-down sample --
             * which is the crash on track change. */
            mixer_ch_stop(0);
            wav64_close(&song);
            wav64_open(&song, TRACKS[cur]);
            wav64_set_loop(&song, true);
            wav64_play(&song, 0);
            mixer_ch_set_vol(0, 0.55f, 0.55f);
            debounce = 20;
        }

        /* Exactly the game's call sequence, and nothing else competing. */
        polls++;
        if (audio_can_write()) {
            short *buf = audio_write_begin();
            mixer_poll(buf, audio_get_buffer_length());
            audio_write_end();
            writes++;
        }

        surface_t *disp = display_get();
        graphics_fill_screen(disp, 0);
        #ifdef NO_RDPQ
        graphics_draw_text(disp, 40, 40, "AUDIO ONLY -- rdpq_init NOT called");
#else
        graphics_draw_text(disp, 40, 40, "AUDIO ONLY -- rdpq_init() CALLED");
#endif
        graphics_draw_text(disp, 40, 60, "D-pad L/R changes track");
        char l[64];
        sprintf(l, "track: %s", NAMES[cur]);
        graphics_draw_text(disp, 40, 90, l);
        sprintf(l, "buffer %d samples @ %d Hz", audio_get_buffer_length(),
                audio_get_frequency());
        graphics_draw_text(disp, 40, 110, l);
        sprintf(l, "frames %u   mixer_poll %u   (%u%% of frames)",
                (unsigned)polls, (unsigned)writes,
                (unsigned)(polls ? writes * 100 / polls : 0));
        graphics_draw_text(disp, 40, 130, l);
        graphics_draw_text(disp, 40, 160,
            "If THIS is clean, the assets are fine.");
        display_show(disp);

        /* Report the RATE, which is what "slow music" is about: how many
         * samples the mixer has actually produced against how much wall time
         * has passed. play_ratio should sit at ~1.00; below that the track
         * literally plays slow. */
        if ((polls % 300) == 0 && polls) {
            uint32_t el_vbl = g_vbl - t_start;
            float secs = el_vbl / 59.94f;
            uint64_t samples = (uint64_t)writes * audio_get_buffer_length();
            uint32_t ratio_x100 = secs > 0.1f
                ? (uint32_t)((samples / secs) * 100.0f / audio_get_frequency()) : 0;
            sprintf(pl, "AU RATE frames=%u writes=%u (%u%%) vbl=%u secs=%d "
                        "samples=%u play_ratio=%u.%02u\n",
                    (unsigned)polls, (unsigned)writes,
                    (unsigned)(writes * 100 / polls), (unsigned)el_vbl,
                    (int)secs, (unsigned)samples,
                    (unsigned)(ratio_x100 / 100), (unsigned)(ratio_x100 % 100));
            ISV(pl);
            if (polls >= 1800) { ISV("AU_DONE\n"); while (1) {} }
        }
    }
}
