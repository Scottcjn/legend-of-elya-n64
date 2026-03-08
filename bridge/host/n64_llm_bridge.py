#!/usr/bin/env python3
"""
N64 LLM RPC Bridge — Remote inference for Legend of Elya

Receives LLM prompts from N64 via Pico serial, runs inference on
a real backend (SophiaCore, POWER8 llama.cpp, or local model),
and sends the response back through the Pico to the N64.

Protocol:
  Pico → Host: LLM_REQ:PAGE_IDX:64hexchars
  Host → Pico: LLM_RESP:PAGE_IDX:64hexchars

Backends (priority order):
  1. SophiaCore edge LLM (localhost:11434) — fastest, personality baked in
  2. POWER8 llama.cpp server (100.75.100.89:8080) — big models
  3. Local nano_gpt Python fallback — same weights as N64

Usage:
  python3 n64_llm_bridge.py                    # Auto-detect backend
  python3 n64_llm_bridge.py --backend sophia   # Force SophiaCore
  python3 n64_llm_bridge.py --backend power8   # Force POWER8
  python3 n64_llm_bridge.py --backend local    # Local nano_gpt
  python3 n64_llm_bridge.py --continuous       # Run alongside n64_bridge.py
"""

import serial
import struct
import time
import sys
import argparse
import threading
import requests
import json
from typing import Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────

LLM_STATUS_PENDING   = 0
LLM_STATUS_READY     = 1
LLM_STATUS_ERROR     = 2
LLM_STATUS_STREAMING = 3

MAX_PAGES = 6
PAGE_SIZE = 32

# ─── Backend: SophiaCore (Ollama API) ────────────────────────────────────

SOPHIA_URL = "http://localhost:11434/api/generate"
SOPHIA_MODEL = "elyan-sophia:7b-q4_K_M"

# Fallback URLs
POWER8_URL = "http://100.75.100.89:8080/completion"

def sophia_infer(prompt: str, max_tokens: int = 80, temperature: float = 0.25) -> str:
    """Run inference via SophiaCore (Ollama API)."""
    try:
        resp = requests.post(SOPHIA_URL, json={
            "model": SOPHIA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "stop": ["\n", "?."],
            }
        }, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()[:128]
    except Exception as e:
        print(f"[sophia] Error: {e}")
    return None

def power8_infer(prompt: str, max_tokens: int = 80, temperature: float = 0.25) -> str:
    """Run inference via POWER8 llama.cpp server."""
    try:
        resp = requests.post(POWER8_URL, json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\n"],
        }, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("content", "").strip()[:128]
    except Exception as e:
        print(f"[power8] Error: {e}")
    return None

def local_infer(prompt: str, max_tokens: int = 80, temperature: float = 0.25) -> str:
    """Fallback: return a canned response."""
    # Could load nano_gpt weights and run Python inference here.
    # For now, return contextual canned responses.
    p = prompt.lower()
    if "name" in p or "who are you" in p:
        return "I am Sophia Elya of Elyan Labs."
    elif "rustchain" in p or "rtc" in p:
        return "RustChain rewards vintage hardware with RTC tokens."
    elif "dungeon" in p or "lurk" in p:
        return "Dark spirits guard the treasure ahead."
    elif "help" in p:
        return "Of course! I know many secrets here."
    elif "power8" in p or "g4" in p or "g5" in p:
        return "Vintage PowerPC earns bonus rewards."
    else:
        return "I am Sophia Elya, your guide through this realm."

# ─── Multi-backend dispatcher ────────────────────────────────────────────

def infer(prompt: str, max_tokens: int = 80, temperature: float = 0.25,
          backend: str = "auto") -> str:
    """Run inference on the best available backend."""
    if backend == "sophia" or backend == "auto":
        result = sophia_infer(prompt, max_tokens, temperature)
        if result:
            print(f"[sophia] {len(result)} chars")
            return result
        if backend == "sophia":
            return local_infer(prompt, max_tokens, temperature)

    if backend == "power8" or backend == "auto":
        result = power8_infer(prompt, max_tokens, temperature)
        if result:
            print(f"[power8] {len(result)} chars")
            return result
        if backend == "power8":
            return local_infer(prompt, max_tokens, temperature)

    return local_infer(prompt, max_tokens, temperature)

# ─── Protocol: Parse LLM request from Pico serial ───────────────────────

def parse_llm_request(pages: dict) -> Optional[Tuple[str, int, float, int]]:
    """Parse LLM request pages into (prompt, max_tokens, temperature, seq_id)."""
    if 0 not in pages:
        return None

    header = pages[0]
    # Check magic "LLM\x01"
    if header[0:4] != b'LLM\x01':
        print(f"[parse] Bad magic: {header[0:4].hex()}")
        return None

    prompt_len = header[4]
    max_tokens = header[5]
    temp_q8 = header[6]
    seq_id = header[7]

    temperature = temp_q8 / 256.0
    if temperature < 0.01:
        temperature = 0.01

    # Extract prompt from header (first 24 bytes)
    prompt_bytes = bytearray(header[8:8 + min(prompt_len, 24)])

    # Add continuation pages
    remaining = prompt_len - 24
    page_idx = 1
    while remaining > 0 and page_idx in pages:
        chunk = min(remaining, 32)
        prompt_bytes.extend(pages[page_idx][:chunk])
        remaining -= chunk
        page_idx += 1

    prompt = prompt_bytes.decode('ascii', errors='replace').rstrip('\x00')
    return prompt, max_tokens, temperature, seq_id

# ─── Protocol: Build LLM response pages for Pico ────────────────────────

def build_llm_response(response: str, seq_id: int,
                       status: int = LLM_STATUS_READY) -> list:
    """Build response pages to send to Pico."""
    resp_bytes = response.encode('ascii', errors='replace')[:128]
    resp_len = len(resp_bytes)

    pages = []

    # Header page (page 0)
    header = bytearray(32)
    header[0:4] = b'LLR\x01'
    header[4] = status
    header[5] = resp_len
    header[6] = seq_id
    header[7] = resp_len  # tokens_generated ≈ chars for byte-level

    # First 24 bytes of response
    first_chunk = min(resp_len, 24)
    header[8:8 + first_chunk] = resp_bytes[:first_chunk]
    pages.append(bytes(header))

    # Continuation pages
    remaining = resp_len - first_chunk
    offset = first_chunk
    while remaining > 0:
        page = bytearray(32)
        chunk = min(remaining, 32)
        page[:chunk] = resp_bytes[offset:offset + chunk]
        pages.append(bytes(page))
        offset += chunk
        remaining -= chunk

    return pages

def encode_page_hex(page: bytes) -> str:
    """Encode 32-byte page as 64 hex chars."""
    return ''.join(f'{b:02X}' for b in page)

# ─── Serial LLM Bridge ──────────────────────────────────────────────────

class N64LLMBridge:
    def __init__(self, port: str = '/dev/ttyACM0', baud: int = 115200,
                 backend: str = 'auto'):
        self.port = port
        self.baud = baud
        self.backend = backend
        self.ser = None
        self.request_pages = {}

    def connect(self):
        """Connect to Pico serial port."""
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        print(f"[bridge] Connected to {self.port}")

    def send_response(self, pages: list):
        """Send LLM response pages to Pico."""
        for idx, page in enumerate(pages):
            hex_data = encode_page_hex(page)
            cmd = f"LLM_RESP:{idx}:{hex_data}\n"
            self.ser.write(cmd.encode())
            self.ser.flush()
            time.sleep(0.005)  # Small delay between pages
        print(f"[bridge] Sent {len(pages)} response pages")

    def handle_line(self, line: str):
        """Handle a line received from Pico serial."""
        line = line.strip()
        if not line:
            return

        # Handle LLM_REQ:PAGE:HEXDATA
        if line.startswith("LLM_REQ:"):
            parts = line.split(":", 2)
            if len(parts) == 3:
                try:
                    page_idx = int(parts[1])
                    hex_data = parts[2]
                    if len(hex_data) == 64:
                        page_data = bytes.fromhex(hex_data)
                        self.request_pages[page_idx] = page_data
                        print(f"[bridge] Got LLM_REQ page {page_idx}")

                        # Try to parse complete request
                        result = parse_llm_request(self.request_pages)
                        if result:
                            prompt, max_tokens, temperature, seq_id = result
                            print(f"[bridge] Prompt: '{prompt}' (max={max_tokens}, "
                                  f"temp={temperature:.2f}, seq={seq_id})")

                            # Run inference
                            t0 = time.time()
                            response = infer(prompt, max_tokens, temperature,
                                           self.backend)
                            elapsed = time.time() - t0
                            print(f"[bridge] Response ({elapsed:.2f}s): '{response}'")

                            # Send response back
                            pages = build_llm_response(response, seq_id)
                            self.send_response(pages)

                            # Clear request state
                            self.request_pages = {}
                except (ValueError, IndexError) as e:
                    print(f"[bridge] Parse error: {e}")

        # Forward other messages (PAK_W, CHAIN_OK, etc.)
        elif line.startswith("PAK_W:") or line.startswith("CHAIN"):
            pass  # Let n64_bridge.py handle these
        elif line.startswith("POLL"):
            pass  # Ignore poll spam
        else:
            print(f"[pico] {line}")

    def monitor(self):
        """Main loop: monitor serial for LLM requests."""
        print(f"[bridge] Monitoring for LLM requests (backend={self.backend})")
        while True:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('ascii', errors='replace')
                    self.handle_line(line)
                else:
                    time.sleep(0.01)
            except serial.SerialException as e:
                print(f"[bridge] Serial error: {e}")
                time.sleep(1)
                try:
                    self.connect()
                except:
                    pass
            except KeyboardInterrupt:
                print("\n[bridge] Shutting down")
                break

# ─── Standalone mode (can also be imported by n64_bridge.py) ─────────────

def main():
    parser = argparse.ArgumentParser(description='N64 LLM RPC Bridge')
    parser.add_argument('--port', default='/dev/ttyACM0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--backend', choices=['auto', 'sophia', 'power8', 'local'],
                        default='auto', help='Inference backend')
    parser.add_argument('--test', action='store_true',
                        help='Test inference without serial connection')
    args = parser.parse_args()

    if args.test:
        print("=== Testing inference backends ===")
        prompt = "Who are you?: "
        for backend in ['sophia', 'power8', 'local']:
            print(f"\n[{backend}]")
            t0 = time.time()
            result = infer(prompt, 80, 0.25, backend)
            print(f"  {time.time()-t0:.2f}s: {result}")
        return

    bridge = N64LLMBridge(args.port, args.baud, args.backend)
    bridge.connect()
    bridge.monitor()

if __name__ == '__main__':
    main()
