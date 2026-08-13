#!/usr/bin/env python3
"""Run a ROM under ares and timestamp every line it prints with the HOST clock.

Why a pty: ares writes the IS-Viewer text to stdout with the C library's
default buffering.  Redirected to a file or a pipe that is BLOCK buffered, so
nothing appears until the process exits and every line lands with the same
apparent timestamp -- useless for measuring how long the emulated machine
actually took.  Handing ares a pty makes it line buffered, so each line
surfaces when it is written and the host timestamp on it means something.

That host timestamp is the third clock in the experiment:
    CP0 COUNT   emulated CPU cycles   (printed by the ROM)
    VI vblank   emulated field rate   (printed by the ROM)
    host clock  real seconds          (added here)

ares operating notes, already paid for, do not rediscover:
  * flatpak CANNOT see /tmp -- the ROM must be staged under $HOME.
  * ares loses stdout on SIGTERM.  Send SIGINT.
  * ares validates the IPL2 checksum, so the ROM must be a stock libdragon
    build (the Makefile's chksum64 step handles this).
  * mupen64plus is not an acceptable substitute: it cannot boot libdragon
    ROMs and prices an uncached cart read at ~0 cycles against ares's 268.6.

Usage:  ares_rate_run.py <rom.z64> <logfile> [done_marker] [timeout_s]
"""
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time

ARES = ["flatpak", "run", "dev.ares.ares", "--system", "Nintendo 64",
        "--no-file-prompt",
        "--setting", "Video/Driver=None",
        "--setting", "Audio/Driver=None"]


def main() -> int:
    rom = os.path.abspath(sys.argv[1])
    logpath = os.path.abspath(sys.argv[2])
    marker = sys.argv[3] if len(sys.argv) > 3 else "RATE_DONE"
    timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0

    if not rom.startswith(os.path.expanduser("~")):
        print(f"REFUSING: ROM must live under $HOME, flatpak cannot see {rom}")
        return 2

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        ["xvfb-run", "-a"] + ARES + [rom],
        stdout=slave, stderr=slave, stdin=subprocess.DEVNULL,
        close_fds=True, start_new_session=True)
    os.close(slave)

    t0 = time.time()
    buf = b""
    lines = []
    saw_marker = False
    try:
        while True:
            if time.time() - t0 > timeout:
                lines.append((time.time() - t0, "HOST TIMEOUT"))
                break
            r, _, _ = select.select([master], [], [], 0.05)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    text = raw.decode("utf-8", "replace").rstrip("\r")
                    lines.append((time.time() - t0, text))
                    if marker in text:
                        saw_marker = True
                if saw_marker:
                    break
            elif proc.poll() is not None:
                break
    finally:
        # SIGINT, never SIGTERM: ares loses stdout on SIGTERM.
        for sig in (signal.SIGINT, signal.SIGKILL):
            if proc.poll() is not None:
                break
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except ProcessLookupError:
                break
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
        subprocess.run(["pkill", "-INT", "-x", "ares"], check=False)
        time.sleep(1.0)
        subprocess.run(["pkill", "-9", "-x", "ares"], check=False)
        try:
            os.close(master)
        except OSError:
            pass

    with open(logpath, "w") as fh:
        for dt, text in lines:
            fh.write(f"{dt:10.3f}  {text}\n")

    # Host wall clock between named markers, so the emulated elapsed time the
    # ROM reports can be compared against real seconds.
    stamps = {}
    for dt, text in lines:
        m = re.search(r"RATE (\w+)_(START|END)", text)
        if m:
            stamps[f"{m.group(1)}_{m.group(2)}"] = dt
    with open(logpath, "a") as fh:
        for phase in ("A", "B"):
            s, e = stamps.get(f"{phase}_START"), stamps.get(f"{phase}_END")
            if s is not None and e is not None:
                fh.write(f"HOSTCLOCK phase {phase}: {e - s:.3f} s wall\n")
    with open(logpath) as fh:
        sys.stdout.write(fh.read())
    return 0 if saw_marker else 1


if __name__ == "__main__":
    sys.exit(main())
