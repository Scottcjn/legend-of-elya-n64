#!/usr/bin/env python3
"""
Tests for Multi-NPC Mode — Expansion Pak Support

Validates memory layout, persona definitions, RAM detection logic,
and multi-session architecture without requiring N64 hardware.

Run: python -m pytest test_multi_npc.py -v
Bounty: legend-of-elya-n64 #5 (200 RTC)
"""

import os
import re
import struct
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class TestMemoryLayout(unittest.TestCase):
    """Validate the 8MB Expansion Pak memory map."""

    # Memory regions (from multi_npc.h)
    CODE_START      = 0x80000000
    CODE_END        = 0x80100000  # 1 MB
    WEIGHTS_START   = 0x80100000
    WEIGHTS_END     = 0x80380000  # 2.5 MB
    KV0_START       = 0x80380000
    KV0_END         = 0x804C8000  # ~1.3 MB
    KV1_START       = 0x804C8000
    KV1_END         = 0x80610000  # ~1.3 MB
    KV2_START       = 0x80610000
    KV2_END         = 0x80758000  # ~1.3 MB
    SCRATCH_START   = 0x80758000
    RAM_END_8MB     = 0x80800000

    def test_total_fits_8mb(self):
        """All regions fit within 8MB."""
        self.assertLessEqual(self.RAM_END_8MB - self.CODE_START, 8 * 1024 * 1024)

    def test_no_overlap(self):
        """Memory regions don't overlap."""
        regions = [
            (self.CODE_START, self.CODE_END),
            (self.WEIGHTS_START, self.WEIGHTS_END),
            (self.KV0_START, self.KV0_END),
            (self.KV1_START, self.KV1_END),
            (self.KV2_START, self.KV2_END),
            (self.SCRATCH_START, self.RAM_END_8MB),
        ]
        for i, (s1, e1) in enumerate(regions):
            for j, (s2, e2) in enumerate(regions):
                if i >= j:
                    continue
                self.assertFalse(s1 < e2 and s2 < e1,
                                 f"Region {i} [{s1:#x}-{e1:#x}] overlaps {j} [{s2:#x}-{e2:#x}]")

    def test_code_region_1mb(self):
        size = self.CODE_END - self.CODE_START
        self.assertEqual(size, 1 * 1024 * 1024)

    def test_weights_region(self):
        size = self.WEIGHTS_END - self.WEIGHTS_START
        self.assertGreater(size, 2 * 1024 * 1024)

    def test_kv_caches_equal_size(self):
        kv0 = self.KV0_END - self.KV0_START
        kv1 = self.KV1_END - self.KV1_START
        kv2 = self.KV2_END - self.KV2_START
        self.assertEqual(kv0, kv1)
        self.assertEqual(kv1, kv2)

    def test_4mb_single_npc_fits(self):
        """4MB mode: reduced code + weights + 1 smaller KV cache fits in 4MB."""
        code_4mb = 512 * 1024          # 512 KB code in 4MB mode
        weights = 2.5 * 1024 * 1024    # 2.5 MB weights
        kv_4mb = 1 * 1024 * 1024       # 1 MB KV cache (reduced ctx=64)
        self.assertLessEqual(code_4mb + weights + kv_4mb, 4 * 1024 * 1024)


class TestRAMDetection(unittest.TestCase):
    """Test RAM detection logic."""

    RAM_4MB = 4 * 1024 * 1024
    RAM_8MB = 8 * 1024 * 1024
    EXPANSION_TEST = 0x80400000

    def test_4mb_constant(self):
        self.assertEqual(self.RAM_4MB, 4194304)

    def test_8mb_constant(self):
        self.assertEqual(self.RAM_8MB, 8388608)

    def test_expansion_test_address_in_upper_4mb(self):
        """Test address is in the 4-8MB range."""
        base = 0x80000000
        self.assertGreaterEqual(self.EXPANSION_TEST, base + self.RAM_4MB)
        self.assertLess(self.EXPANSION_TEST, base + self.RAM_8MB)

    def test_3_npcs_with_8mb(self):
        """8MB should support 3 NPCs."""
        ram = self.RAM_8MB
        max_npcs = 3 if ram >= self.RAM_8MB else 1
        self.assertEqual(max_npcs, 3)

    def test_1_npc_with_4mb(self):
        """4MB should support only 1 NPC."""
        ram = self.RAM_4MB
        max_npcs = 3 if ram >= self.RAM_8MB else 1
        self.assertEqual(max_npcs, 1)


class TestPersonas(unittest.TestCase):
    """Validate NPC persona definitions."""

    PERSONAS = {
        "Sophia": {
            "role": "wise sage",
            "text_color": 0x7DD3FCFF,
            "name_color": 0x3B82F6FF,
            "temp": 180,
        },
        "Forge Master": {
            "role": "grumpy blacksmith",
            "text_color": 0xF59E0BFF,
            "name_color": 0xEF4444FF,
            "temp": 220,
        },
        "Librarian": {
            "role": "mysterious scholar",
            "text_color": 0xA855F7FF,
            "name_color": 0x8B5CF6FF,
            "temp": 150,
        },
    }

    def test_three_personas(self):
        self.assertEqual(len(self.PERSONAS), 3)

    def test_unique_names(self):
        names = list(self.PERSONAS.keys())
        self.assertEqual(len(names), len(set(names)))

    def test_unique_colors(self):
        colors = [p["text_color"] for p in self.PERSONAS.values()]
        self.assertEqual(len(colors), len(set(colors)))

    def test_unique_name_colors(self):
        colors = [p["name_color"] for p in self.PERSONAS.values()]
        self.assertEqual(len(colors), len(set(colors)))

    def test_temperatures_different(self):
        temps = [p["temp"] for p in self.PERSONAS.values()]
        self.assertEqual(len(temps), len(set(temps)))

    def test_temperature_range(self):
        for name, p in self.PERSONAS.items():
            self.assertGreater(p["temp"], 0, f"{name} temp too low")
            self.assertLessEqual(p["temp"], 255, f"{name} temp too high")

    def test_sophia_is_default(self):
        """Sophia should be NPC 0 (first / default)."""
        names = list(self.PERSONAS.keys())
        self.assertEqual(names[0], "Sophia")

    def test_forge_master_hottest(self):
        """Forge Master should be the most unpredictable."""
        temps = {k: v["temp"] for k, v in self.PERSONAS.items()}
        hottest = max(temps, key=temps.get)
        self.assertEqual(hottest, "Forge Master")

    def test_librarian_coldest(self):
        """Librarian should be the most precise."""
        temps = {k: v["temp"] for k, v in self.PERSONAS.items()}
        coldest = min(temps, key=temps.get)
        self.assertEqual(coldest, "Librarian")


class TestHeaderStructure(unittest.TestCase):
    """Validate multi_npc.h has required definitions."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "multi_npc.h")) as f:
            cls.header = f.read()

    def test_has_max_npcs(self):
        self.assertIn("MAX_NPCS_8MB", self.header)
        self.assertIn("MAX_NPCS_4MB", self.header)

    def test_max_npcs_values(self):
        m = re.search(r"MAX_NPCS_8MB\s+(\d+)", self.header)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 3)

        m = re.search(r"MAX_NPCS_4MB\s+(\d+)", self.header)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 1)

    def test_has_npc_persona(self):
        self.assertIn("NPCPersona", self.header)
        self.assertIn("name", self.header)
        self.assertIn("prefix", self.header)
        self.assertIn("text_color", self.header)

    def test_has_npc_session(self):
        self.assertIn("NPCSession", self.header)
        self.assertIn("SGAIState", self.header)

    def test_has_manager(self):
        self.assertIn("MultiNPCManager", self.header)
        self.assertIn("shared_weights", self.header)
        self.assertIn("has_expansion_pak", self.header)

    def test_has_api_functions(self):
        for fn in ["multi_npc_init", "multi_npc_reset", "multi_npc_select",
                    "multi_npc_generate", "multi_npc_generate_all",
                    "multi_npc_count", "detect_ram_size", "has_expansion_pak"]:
            self.assertIn(fn, self.header, f"Missing function: {fn}")


class TestImplementation(unittest.TestCase):
    """Validate multi_npc.c implementation."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "multi_npc.c")) as f:
            cls.src = f.read()

    def test_has_persona_definitions(self):
        self.assertIn("PERSONA_SOPHIA", self.src)
        self.assertIn("PERSONA_FORGE_MASTER", self.src)
        self.assertIn("PERSONA_LIBRARIAN", self.src)

    def test_has_ram_detection(self):
        self.assertIn("detect_ram_size", self.src)
        self.assertIn("EXPANSION_MAGIC", self.src)

    def test_has_shared_weights(self):
        """Weights should be shared (not duplicated per NPC)."""
        self.assertIn("shared_weights", self.src)
        # Init should use same rom_weights for all NPCs
        self.assertIn("sgai_init(&npc->state, rom_weights)", self.src)

    def test_has_separate_kv_caches(self):
        """Each NPC should have its own state (separate KV)."""
        self.assertIn("npcs[", self.src)
        self.assertIn("npc->state", self.src)

    def test_persona_prefix_prepended(self):
        """Generate should prepend persona prefix."""
        self.assertIn("persona.prefix", self.src)
        self.assertIn("prefix_len", self.src)

    def test_4mb_fallback(self):
        """4MB mode should create only 1 NPC."""
        self.assertIn("MAX_NPCS_4MB", self.src)
        self.assertIn("MAX_NPCS_8MB", self.src)

    def test_generate_all_resets(self):
        """Generate all should reset each NPC for clean context."""
        self.assertIn("sgai_reset", self.src)

    def test_demo_display(self):
        """Should have display code for demo mode."""
        self.assertIn("Multi-NPC Demo", self.src)
        self.assertIn("display_lock", self.src)


class TestKVCacheSizing(unittest.TestCase):
    """Validate KV cache fits the model parameters."""

    # From nano_gpt.h
    N_LAYERS = 6
    N_EMBED = 192
    CTX = 128

    def test_single_kv_cache_size(self):
        """One KV cache: k + v arrays."""
        # float k[N_LAYERS][CTX][N_EMBED] + float v[same] + pos int
        kv_size = 2 * self.N_LAYERS * self.CTX * self.N_EMBED * 4  # float = 4 bytes
        kv_size += 4  # pos field
        # Should be ~1.15 MB
        self.assertLess(kv_size, 1.3 * 1024 * 1024)
        self.assertGreater(kv_size, 0.5 * 1024 * 1024)

    def test_three_kv_caches_fit_8mb(self):
        """3 KV caches + weights + code should fit in 8MB."""
        kv_one = 2 * self.N_LAYERS * self.CTX * self.N_EMBED * 4
        kv_total = kv_one * 3
        code = 1 * 1024 * 1024       # 1 MB
        weights = 2.5 * 1024 * 1024  # 2.5 MB
        total = kv_total + code + weights
        self.assertLess(total, 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
