#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""compose.py — original music for Legend of Elya, written here, owned by us.

WHY NOT A MODEL. The obvious move is a small music model, but the licensing is
the problem, not the compute: MusicGen's weights are CC-BY-NC-4.0 (the code is
MIT, the weights are not), which is a poor fit for a game anyone might ship.
This lab already did that homework once — feverdream's audio lane went looking
for a public-domain Rossini and documented every search that came up empty
(audio/LICENSES.md there). Composing it ourselves ends the question: these are
our notes, in a text file, diffable and editable.

WHY NOT A SQUARE WAVE. legend_of_elya.c's music_update() writes one square-wave
channel straight into the audio buffer, bypassing libdragon's mixer entirely.
That is why it sounds like a beeper: it IS one. The pipeline here is the one
feverdream's audio lane proved on console —
    MIDI -> fluidsynth (GM soundfont) -> 22.05 kHz mono s16 -> audioconv64 VADPCM
— played through the real RSP mixer at a measured ~30k CPU cycles per vblank.

Three tracks, one per room, each a seamless loop:
  dungeon  D minor, slow, low strings and a distant harp   — dread, patient
  library  D dorian, music box and warm pad                — curious, safe
  forge    D minor, timpani and brass stabs over an anvil  — heat, work

Usage: python3 audio/compose.py            (writes audio/*.mid)
"""
import struct, sys, os

TPQ = 480  # ticks per quarter note

def vlq(n):
    out = bytearray([n & 0x7F]); n >>= 7
    while n: out.insert(0, (n & 0x7F) | 0x80); n >>= 7
    return bytes(out)

class Track:
    """Collects events with absolute timestamps and delta-encodes them ONLY at
    the end, after a stable sort by time.

    This is not a style preference. MIDI delta times are relative to the
    previous event, so the event stream must be in time order. Emitting
    note-on/note-off as each part is written interleaves the parts out of
    order; a naive writer then clamps the negative deltas to zero, every
    later event slides forward, and the piece stretches and smears. That bug
    turned a 53-second dungeon loop into 127 seconds of overlapping mush --
    audible as 'dirty' audio long before it looks wrong in a hex dump."""
    def __init__(self): self.ev = []          # (abs_tick, order, bytes)
    def _add(self, at, data, order=1):
        self.ev.append((int(at), order, bytes(data)))
    # note-offs sort BEFORE note-ons at the same tick (order 0), so a repeated
    # pitch retriggers cleanly instead of the off killing the new note.
    def prog(self, at, ch, p):    self._add(at, [0xC0|ch, p], 0)
    def ctrl(self, at, ch, c, v): self._add(at, [0xB0|ch, c, v], 0)
    def on(self, at, ch, n, v):   self._add(at, [0x90|ch, n, v], 2)
    def off(self, at, ch, n):     self._add(at, [0x80|ch, n, 0], 0)
    def note(self, at, dur, ch, n, v):
        self.on(at, ch, n, v); self.off(at + dur, ch, n)
    def tempo(self, at, bpm):
        us = int(60_000_000 / bpm)
        self._add(at, b'\xFF\x51\x03' + us.to_bytes(3, 'big'), 0)
    def end(self, at):
        self._add(at, b'\xFF\x2F\x00', 3)
        out = bytearray(); prev = 0
        for tick, _order, data in sorted(self.ev, key=lambda e: (e[0], e[1])):
            out += vlq(tick - prev) + data
            prev = tick
        return bytes(out)

def write_midi(path, chunks):
    hdr = b'MThd' + struct.pack('>IHHH', 6, 1, len(chunks), TPQ)
    body = b''.join(b'MTrk' + struct.pack('>I', len(c)) + c for c in chunks)
    open(path, 'wb').write(hdr + body)
    print("wrote %s (%d bytes, %d tracks)" % (path, len(hdr) + len(body), len(chunks)))

Q, H, W, E = TPQ, TPQ*2, TPQ*4, TPQ//2

def dungeon(bars=16):
    """D minor. A low pedal that never resolves, and a harp figure that answers
    it late — the room is waiting for you, not welcoming you."""
    t = Track(); t.tempo(0, 72)
    t.prog(0, 0, 48)   # string ensemble
    t.prog(0, 1, 46)   # harp
    t.prog(0, 2, 42)   # cello
    root = [50, 50, 48, 46]                    # D  D  C  Bb  (natural minor descent)
    harp = [69, 72, 74, 77, 74, 72]            # A  C  D  F  D  C
    at = 0
    for b in range(bars):
        r = root[b % 4]
        t.note(at, W - 40, 2, r - 12, 62)      # cello pedal, an octave down
        t.note(at, H - 30, 0, r, 54)           # strings
        t.note(at + H, H - 30, 0, r + 7, 48)   # ... and its fifth
        if b % 2 == 1:                          # harp answers on odd bars only
            for i, n in enumerate(harp):
                t.note(at + H + i * E, E - 20, 1, n, 40 + (i % 3) * 6)
        at += W
    return [t.end(at)]

def library(bars=16):
    """D dorian — the minor with a raised sixth, which is why it sounds curious
    rather than sad. Music box over a warm pad."""
    t = Track(); t.tempo(0, 88)
    t.prog(0, 0, 10)   # music box
    t.prog(0, 1, 89)   # warm pad
    mel = [74, 76, 77, 79, 77, 76, 74, 72, 74, 77, 81, 79, 77, 76, 74, 72]
    pad = [50, 53, 55, 53]
    at = 0
    for b in range(bars):
        t.note(at, W - 60, 1, pad[b % 4] - 12, 40)
        for i in range(4):
            n = mel[(b * 4 + i) % len(mel)]
            t.note(at + i * Q, Q - 40, 0, n, 52 if i == 0 else 44)
        at += W
    return [t.end(at)]

def forge(bars=16):
    """D minor, but driven. Timpani on the beat, brass stabs off it, and an
    anvil (GM 'triangle' on the percussion channel) every other bar."""
    t = Track(); t.tempo(0, 104)
    t.prog(0, 0, 61)   # brass section
    t.prog(0, 1, 47)   # timpani
    stabs = [50, 50, 53, 55]
    at = 0
    for b in range(bars):
        r = stabs[b % 4]
        for i in range(4):
            t.note(at + i * Q, E - 10, 1, 38, 70 if i % 2 == 0 else 48)  # timpani
        t.note(at + E,      Q - 20, 0, r,      66)
        t.note(at + E + Q,  Q - 20, 0, r + 7,  58)
        if b % 2 == 0:
            t.note(at + H + Q, E, 9, 81, 80)   # ch 10 percussion: anvil-ish
        at += W
    return [t.end(at)]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    write_midi(os.path.join(here, "dungeon.mid"), dungeon())
    write_midi(os.path.join(here, "library.mid"), library())
    write_midi(os.path.join(here, "forge.mid"),   forge())
