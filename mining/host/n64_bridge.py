#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
N64 Pico Bridge — Host-side serial bridge for RTC attestation.

Connects to Pico via USB serial. Relays controller pak writes from N64.
Can send button presses to Pico for remote control.

Usage:
    python3 n64_bridge.py                    # Monitor pak writes
    python3 n64_bridge.py --press A          # Press A button
    python3 n64_bridge.py --attest           # Wait for attestation data
    python3 n64_bridge.py --submit           # Attest + submit to RustChain
"""

import serial
import sys
import time
import struct
import hashlib
import json
import argparse
import threading
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200

# Attestation protocol: N64 writes fingerprint data starting at pak address 0x0000
# Page 0x0000: header (magic, version, num_checks, total_pages)
# Page 0x0020+: check results (32 bytes each)
ATTEST_MAGIC = b"N64A"  # N64 Attestation magic
ATTEST_VERSION = 1

# RustChain endpoint
RUSTCHAIN_NODE = "https://50.28.86.131"

# Chain state page format (32 bytes at pak address 0x8000)
# See plan doc for full field layout.
CHAIN_STATE_SIZE = 32
CHAIN_POLL_INTERVAL = 10  # seconds between RustChain API polls


class N64Bridge:
    def __init__(self, port=SERIAL_PORT):
        self.port_path = port
        self.ser = None
        self.pak_pages = {}  # addr -> 32 bytes
        self.running = True

    def connect(self):
        """Connect to Pico serial port."""
        try:
            self.ser = serial.Serial(self.port_path, SERIAL_BAUD, timeout=0.1)
            time.sleep(0.5)  # Wait for Pico to settle after USB enum
            # Flush any startup messages
            while self.ser.in_waiting:
                line = self.ser.readline().decode("ascii", errors="replace").strip()
                if line:
                    print(f"[pico] {line}")
            return True
        except serial.SerialException as e:
            print(f"ERROR: Cannot open {self.port_path}: {e}")
            print("Is the Pico plugged in? Check with: ls /dev/ttyACM*")
            return False

    def close(self):
        if self.ser:
            self.ser.close()

    def press_button(self, button, duration=0.1):
        """Send button press to Pico. button = 'A','B','S','Z','U','D','L','R'"""
        upper = button.upper()
        lower = button.lower()
        self.ser.write(upper.encode())
        time.sleep(duration)
        self.ser.write(lower.encode())
        print(f"[btn] Pressed {upper} for {duration}s")

    def release_all(self):
        """Release all buttons."""
        self.ser.write(b"0")

    def read_line(self, timeout=1.0):
        """Read a line from Pico serial."""
        self.ser.timeout = timeout
        try:
            raw = self.ser.readline()
            if raw:
                return raw.decode("ascii", errors="replace").strip()
        except Exception:
            pass
        return None

    def monitor(self, callback=None):
        """Monitor serial output. Calls callback(event_type, data) for each event."""
        print("[monitor] Listening for N64 pak events...")
        while self.running:
            line = self.read_line(timeout=0.5)
            if not line:
                continue

            if line.startswith("PAK_W:"):
                # Format: PAK_W:ADDR:HEXDATA
                parts = line.split(":", 2)
                if len(parts) == 3:
                    addr = int(parts[1], 16)
                    hex_data = parts[2]
                    data = bytes.fromhex(hex_data)
                    self.pak_pages[addr] = data
                    print(f"[pak_w] addr=0x{addr:04X} data={hex_data[:32]}...")
                    if callback:
                        callback("pak_write", addr, data)

            elif line.startswith("PAK_R:"):
                addr = int(line.split(":")[1], 16)
                print(f"[pak_r] addr=0x{addr:04X}")

            elif line.startswith("N64_BRIDGE:"):
                print(f"[pico] {line}")

            else:
                print(f"[serial] {line}")

    def wait_for_attestation(self, timeout=30.0):
        """Wait for complete attestation data from N64.

        Protocol:
          Page 0x0000: [N64A] [ver:1] [num_checks:1] [total_pages:1] [reserved:27]
          Page 0x0020+: check data (32 bytes each)

        Returns dict with fingerprint data or None on timeout.
        """
        print(f"[attest] Waiting for N64 attestation data (timeout={timeout}s)...")
        start = time.time()
        header_received = False
        num_pages = 0
        pages_received = set()

        while time.time() - start < timeout:
            line = self.read_line(timeout=0.5)
            if not line or not line.startswith("PAK_W:"):
                continue

            parts = line.split(":", 2)
            if len(parts) != 3:
                continue

            addr = int(parts[1], 16)
            data = bytes.fromhex(parts[2])
            self.pak_pages[addr] = data

            # Check for header at page 0x0000
            if addr == 0x0000 and data[:4] == ATTEST_MAGIC:
                version = data[4]
                num_checks = data[5]
                num_pages = data[6]
                header_received = True
                print(f"[attest] Header: v{version}, {num_checks} checks, {num_pages} pages")

            elif header_received and addr >= 0x0020:
                page_idx = (addr - 0x0020) // 32
                pages_received.add(page_idx)
                print(f"[attest] Page {page_idx + 1}/{num_pages} received")

                if len(pages_received) >= num_pages:
                    print("[attest] All pages received!")
                    return self._parse_attestation(num_checks, num_pages)

        print("[attest] Timeout waiting for attestation data")
        return None

    def _parse_attestation(self, num_checks, num_pages):
        """Parse attestation data from collected pak pages."""
        # Extract wallet ID from header (bytes 9-28 of page 0x0000)
        header = self.pak_pages.get(0x0000, b"\x00" * 32)
        # Header format: [N64A:4][ver:1][num_checks:1][num_pages:1][all_passed:1][total_passed:1][wallet:20]
        wallet_id = header[9:9+20].decode("ascii", errors="replace").rstrip("\x00")

        fp = {
            "device_arch": "n64_r4300i",
            "device_family": "MIPS",
            "wallet_id": wallet_id,
            "checks": {},
            "raw_pages": {},
        }

        for i in range(num_pages):
            addr = 0x0020 + i * 32
            data = self.pak_pages.get(addr, b"\x00" * 32)
            fp["raw_pages"][f"0x{addr:04X}"] = data.hex()

            # Each check page: [check_id:1] [passed:1] [data:30]
            check_id = data[0]
            passed = data[1]
            check_data = data[2:]

            check_names = {
                1: "cpu_prid",
                2: "count_timing",
                3: "vi_scan",
                4: "memory_ratio",
                5: "pi_timing",
                6: "rdram_config",
                7: "anti_emulation",
            }

            name = check_names.get(check_id, f"check_{check_id}")
            fp["checks"][name] = {
                "passed": bool(passed),
                "data": check_data.hex(),
            }

            # Extract specific values based on check type
            if check_id == 1:  # CPU PRId
                prid = struct.unpack(">I", check_data[:4])[0]
                fp["checks"][name]["prid"] = f"0x{prid:08X}"
                # R4300i should be 0x00000B22 or similar
                fp["checks"][name]["is_r4300i"] = (prid & 0xFFFF00) == 0x000B00

            elif check_id == 2:  # COUNT timing
                cycles = struct.unpack(">I", check_data[:4])[0]
                fp["checks"][name]["cycles"] = cycles

            elif check_id == 3:  # VI scan
                vi_delta = struct.unpack(">I", check_data[:4])[0]
                fp["checks"][name]["vi_delta"] = vi_delta

            elif check_id == 4:  # Memory ratio
                ratio = struct.unpack(">I", check_data[:4])[0]
                fp["checks"][name]["cached_uncached_ratio"] = ratio

        # Compute fingerprint hash
        raw = b""
        for addr in sorted(self.pak_pages.keys()):
            if 0x0000 <= addr < 0x0000 + (num_pages + 1) * 32:
                raw += self.pak_pages[addr]
        fp["fingerprint_hash"] = hashlib.sha256(raw).hexdigest()[:32]

        # Determine overall pass/fail
        total = len(fp["checks"])
        passed = sum(1 for c in fp["checks"].values() if c.get("passed", False))
        fp["total_checks"] = total
        fp["passed_checks"] = passed
        fp["all_passed"] = (passed == total)
        fp["mining_eligible"] = (passed >= 4)  # 4/5 threshold matches N64 ROM

        return fp

    # ─── Chain State (Host → Pico → N64) ─────────────────────────────────
    # Build 32-byte chain state page from RustChain API responses.

    def build_chain_state(self, epoch_data, balance_data, eligibility_data,
                          accepted, rejected):
        """Build 32-byte chain state page from RustChain API data.

        Returns bytes(32) matching the format the N64 ROM expects at pak 0x8000.
        """
        buf = bytearray(CHAIN_STATE_SIZE)

        # Bytes 0-1: magic "RC"
        buf[0] = 0x52  # 'R'
        buf[1] = 0x43  # 'C'
        # Byte 2: version
        buf[2] = 0x01
        # Byte 3: flags
        flags = 0x04  # Bit2 = data_valid
        if eligibility_data.get("eligible", False):
            flags |= 0x02  # Bit1 = eligible
        buf[3] = flags

        # Bytes 4-5: epoch (BE uint16)
        epoch = int(epoch_data.get("epoch", 0)) & 0xFFFF
        struct.pack_into(">H", buf, 4, epoch)

        # Bytes 6-9: slot (BE uint32)
        slot = int(epoch_data.get("slot", epoch_data.get("current_slot", 0)))
        struct.pack_into(">I", buf, 6, slot)

        # Bytes 10-13: balance in milli-RTC (BE uint32)
        balance_rtc = float(balance_data.get("amount_rtc",
                            balance_data.get("balance", 0)))
        balance_milli = int(balance_rtc * 1000) & 0xFFFFFFFF
        struct.pack_into(">I", buf, 10, balance_milli)

        # Bytes 14-17: accepted count (BE uint32)
        struct.pack_into(">I", buf, 14, int(accepted) & 0xFFFFFFFF)

        # Bytes 18-21: rejected count (BE uint32)
        struct.pack_into(">I", buf, 18, int(rejected) & 0xFFFFFFFF)

        # Bytes 22-23: multiplier * 100 (BE uint16), default 300 = 3.0x for N64
        mult = int(eligibility_data.get("multiplier_x100", 300)) & 0xFFFF
        struct.pack_into(">H", buf, 22, mult)

        # Byte 24: miners enrolled
        buf[24] = min(int(epoch_data.get("enrolled_miners",
                         epoch_data.get("miners_enrolled", 0))), 255)

        # Byte 25: rotation size
        buf[25] = min(int(eligibility_data.get("rotation_size", 0)), 255)

        # Bytes 26-27: turn offset (BE uint16)
        turn_offset = int(eligibility_data.get("turn_offset",
                         eligibility_data.get("your_turn_in", 0))) & 0xFFFF
        struct.pack_into(">H", buf, 26, turn_offset)

        # Bytes 28-31: XOR checksum (XOR of bytes 0-27, repeated 4x)
        xor_byte = 0
        for i in range(28):
            xor_byte ^= buf[i]
        buf[28] = xor_byte
        buf[29] = xor_byte
        buf[30] = xor_byte
        buf[31] = xor_byte

        return bytes(buf)

    def send_chain_state(self, state_bytes):
        """Send chain state to Pico via 'CHAIN:HEXDATA\\n' serial command."""
        hex_str = state_bytes.hex().upper()
        cmd = f"CHAIN:{hex_str}\n"
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()

    def poll_rustchain(self, wallet_id):
        """Poll RustChain APIs for current chain state.

        Returns (epoch_data, balance_data, eligibility_data) dicts.
        """
        try:
            import requests
        except ImportError:
            return {}, {}, {}

        epoch_data = {}
        balance_data = {}
        eligibility_data = {}

        try:
            r = requests.get(f"{RUSTCHAIN_NODE}/epoch", verify=False, timeout=5)
            if r.status_code == 200:
                epoch_data = r.json()
        except Exception as e:
            print(f"[chain] epoch poll error: {e}")

        try:
            r = requests.get(f"{RUSTCHAIN_NODE}/wallet/balance",
                             params={"miner_id": wallet_id},
                             verify=False, timeout=5)
            if r.status_code == 200:
                balance_data = r.json()
        except Exception as e:
            print(f"[chain] balance poll error: {e}")

        try:
            r = requests.get(f"{RUSTCHAIN_NODE}/lottery/eligibility",
                             params={"miner_id": wallet_id},
                             verify=False, timeout=5)
            if r.status_code == 200:
                eligibility_data = r.json()
        except Exception as e:
            print(f"[chain] eligibility poll error: {e}")

        return epoch_data, balance_data, eligibility_data

    def continuous_mode(self, wallet_id):
        """Continuous bridge mode: monitor attestations + push chain state.

        - Reads serial for PAK_W events (attestation data from N64)
        - On complete attestation: auto-submit to RustChain
        - Every CHAIN_POLL_INTERVAL seconds: poll RustChain + push state to Pico
        """
        print("=" * 60)
        print(f"N64 CONTINUOUS BRIDGE — wallet: {wallet_id}")
        print("=" * 60)
        print(f"Node: {RUSTCHAIN_NODE}")
        print(f"Chain poll interval: {CHAIN_POLL_INTERVAL}s")
        print("Listening for attestations + pushing chain state...")
        print()

        last_chain_poll = 0
        accepted = 0
        rejected = 0
        header_received = False
        num_pages = 0
        pages_received = set()

        while self.running:
            # --- Read serial (non-blocking, 100ms timeout) ---
            line = self.read_line(timeout=0.1)
            if line:
                if line.startswith("PAK_W:"):
                    parts = line.split(":", 2)
                    if len(parts) == 3:
                        try:
                            addr = int(parts[1], 16)
                            data = bytes.fromhex(parts[2])
                        except ValueError:
                            continue  # Truncated or malformed hex
                        self.pak_pages[addr] = data

                        # Check for attestation header
                        if addr == 0x0000 and data[:4] == ATTEST_MAGIC:
                            num_pages = data[6]
                            header_received = True
                            pages_received = set()
                            print(f"[attest] Header received, expecting {num_pages} pages")

                        elif header_received and addr >= 0x0020:
                            page_idx = (addr - 0x0020) // 32
                            pages_received.add(page_idx)
                            print(f"[attest] Page {page_idx + 1}/{num_pages}")

                            if len(pages_received) >= num_pages:
                                # Complete attestation — auto-submit
                                print("[attest] Complete! Submitting...")
                                header = self.pak_pages.get(0x0000, b"\x00" * 32)
                                fp = self._parse_attestation(header[5], num_pages)
                                if fp and fp.get("mining_eligible", False):
                                    ok = self.submit_to_rustchain(fp)
                                    if ok:
                                        accepted += 1
                                    else:
                                        rejected += 1
                                else:
                                    rejected += 1
                                header_received = False
                                pages_received = set()

                elif line.startswith("CHAIN_OK"):
                    pass  # Acknowledgment from Pico
                elif line.startswith("CHAIN_ERR"):
                    print(f"[chain] Pico error: {line}")
                elif line.startswith("POLL #"):
                    pass  # Suppress poll spam in continuous mode
                else:
                    print(f"[serial] {line}")

            # --- Poll RustChain + push chain state ---
            now = time.time()
            if now - last_chain_poll >= CHAIN_POLL_INTERVAL:
                last_chain_poll = now
                epoch_data, balance_data, elig_data = self.poll_rustchain(wallet_id)

                if epoch_data:
                    state = self.build_chain_state(
                        epoch_data, balance_data, elig_data,
                        accepted, rejected
                    )
                    self.send_chain_state(state)

                    ep = epoch_data.get("epoch", "?")
                    sl = epoch_data.get("slot", epoch_data.get("current_slot", "?"))
                    bal = balance_data.get("amount_rtc",
                          balance_data.get("balance", "?"))
                    print(f"[chain] Pushed: epoch={ep} slot={sl} "
                          f"balance={bal} RTC  ok={accepted} fail={rejected}")

    def submit_to_rustchain(self, fingerprint):
        """Submit attestation to RustChain node."""
        try:
            import requests
        except ImportError:
            print("[submit] ERROR: requests library not installed")
            print("         pip install requests")
            return False

        wallet_id = fingerprint.get("wallet_id", "n64-unknown")
        # Use fingerprint hash as hardware_id to ensure uniqueness
        hw_hash = fingerprint.get("fingerprint_hash", "")
        payload = {
            "miner": wallet_id,
            "miner_id": wallet_id,
            "nonce": int(time.time()),
            "device": {
                "device_model": "Nintendo 64",
                "device_arch": "n64_r4300i",
                "device_family": "MIPS",
                "hardware_id": wallet_id,
                "cpu_serial": fingerprint.get("checks", {}).get("cpu_prid", {}).get("prid", ""),
                "device_id": hw_hash,
            },
            "signals": {"macs": [wallet_id]},  # Unique ID for hw binding
            "fingerprint": fingerprint,
            "report": {
                "type": "n64_hardware_attestation",
                "timestamp": int(time.time()),
                "checks_passed": fingerprint.get("passed_checks", 0),
                "checks_total": fingerprint.get("total_checks", 0),
            },
        }

        print(f"[submit] Submitting to {RUSTCHAIN_NODE}/attest/submit")
        print(f"[submit] Fingerprint hash: {fingerprint['fingerprint_hash']}")
        print(f"[submit] Checks passed: {fingerprint['all_passed']}")

        try:
            resp = requests.post(
                f"{RUSTCHAIN_NODE}/attest/submit",
                json=payload,
                verify=False,
                timeout=10,
            )
            print(f"[submit] Response: {resp.status_code} {resp.text[:200]}")
            return resp.status_code == 200
        except Exception as e:
            print(f"[submit] ERROR: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="N64 Pico Bridge — RTC Attestation")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial port")
    parser.add_argument("--press", help="Press a button (A,B,S,Z,U,D,L,R)")
    parser.add_argument("--attest", action="store_true", help="Wait for attestation")
    parser.add_argument("--submit", action="store_true", help="Attest + submit to RustChain")
    parser.add_argument("--monitor", action="store_true", help="Monitor all pak events")
    parser.add_argument("--continuous", metavar="WALLET_ID",
                        help="Continuous bridge: auto-submit attestations + push chain state")
    args = parser.parse_args()

    bridge = N64Bridge(args.port)
    if not bridge.connect():
        sys.exit(1)

    try:
        if args.continuous:
            bridge.continuous_mode(args.continuous)

        elif args.press:
            bridge.press_button(args.press)
            time.sleep(0.2)

        elif args.attest or args.submit:
            print("=" * 50)
            print("N64 RTC ATTESTATION")
            print("=" * 50)
            print("Waiting for N64 to send attestation data...")
            print("(Press A on N64 or run: python3 n64_bridge.py --press A)")
            print()

            fp = bridge.wait_for_attestation(timeout=60)
            if fp:
                print()
                print("=" * 50)
                print("ATTESTATION RESULTS")
                print("=" * 50)
                print(json.dumps(fp, indent=2))

                if args.submit and fp.get("mining_eligible", False):
                    print()
                    bridge.submit_to_rustchain(fp)
                elif args.submit and not fp.get("mining_eligible", False):
                    print(f"[submit] SKIPPED — only {fp.get('passed_checks',0)}/{fp.get('total_checks',0)} checks passed (need 4)")
            else:
                print("No attestation data received.")

        else:
            # Default: monitor mode
            bridge.monitor()

    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
