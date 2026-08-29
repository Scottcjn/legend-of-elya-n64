#!/usr/bin/env bash
# Render the composed MIDI to looping VADPCM wav64, the pipeline feverdream's
# audio lane proved on console (MIDI -> fluidsynth -> 22050 Hz mono s16 ->
# audioconv64 vadpcm, played through the real RSP mixer).
#
# Two things here are load-bearing and were learned the hard way:
#  * VADPCM (--wav-compress 1), never Opus (3): Opus asserts at wav64_open
#    unless wav64_init_compression(3) is called at startup.
#  * TRIM TO THE EXACT BAR LENGTH. fluidsynth keeps rendering the reverb tail
#    past the last note, so an untrimmed loop plays the music and then several
#    seconds of near-silence before wrapping -- which sounds broken.
set -euo pipefail
cd "$(dirname "$0")/.."
AC64=$HOME/n64-toolchain/mips64-toolchain/bin/audioconv64
SF2=/usr/share/sounds/sf2/FluidR3_GM.sf2
mkdir -p build/audio filesystem
python3 audio/compose.py

# name  bars  bpm   (must match audio/compose.py)
render() {
    local name=$1 bars=$2 bpm=$3
    local secs
    secs=$(python3 -c "print(f'{$bars*4*60/$bpm:.6f}')")
    fluidsynth -ni -g 0.6 -O s16 -r 22050 -T wav \
        -F build/audio/${name}_st.wav "$SF2" audio/${name}.mid >/dev/null 2>&1
    # mono, trimmed to exactly one loop, peak-normalised to -1 dBFS, tiny
    # fades so the wrap has no click.
    sox build/audio/${name}_st.wav -c 1 build/audio/${name}.wav \
        trim 0 "$secs" gain -n -1 fade t 0.02 "$secs" 0.05
    "$AC64" --wav-compress 1 --wav-loop true --wav-loop-offset 0 \
        -o filesystem build/audio/${name}.wav >/dev/null
    printf "  %-9s %6.1fs loop  %8s bytes\n" "$name" "$secs" \
        "$(stat -c%s filesystem/${name}.wav64)"
}

render dungeon 16 72
render library 16 88
render forge   16 104
echo "music built into filesystem/"
