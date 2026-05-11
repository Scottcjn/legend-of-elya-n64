import importlib.util
import io
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_weights_simple",
        ROOT / "scripts" / "generate_weights_simple.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_persona_weights_is_deterministic_and_shaped():
    generator = load_generator()

    first = generator.generate_persona_weights("sophia", generator.PERSONA_SEEDS["sophia"])
    second = generator.generate_persona_weights("sophia", generator.PERSONA_SEEDS["sophia"])
    blacksmith = generator.generate_persona_weights(
        "blacksmith",
        generator.PERSONA_SEEDS["blacksmith"],
    )

    assert first["embed"].shape == (
        generator.MODEL_CONFIG["vocab_size"],
        generator.MODEL_CONFIG["n_embed"],
    )
    assert len(first["layers"]) == generator.MODEL_CONFIG["n_layer"]
    assert first["layers"][0]["attention_in"].shape == (
        generator.MODEL_CONFIG["n_embed"],
        3 * generator.MODEL_CONFIG["n_embed"],
    )
    np.testing.assert_array_equal(first["embed"], second["embed"])
    assert not np.array_equal(first["embed"], blacksmith["embed"])


def test_export_q8_tensor_writes_scale_then_int8_payload():
    generator = load_generator()
    output = io.BytesIO()

    generator.export_q8_tensor(output, np.array([-2.0, 0.0, 2.0], dtype=np.float32))

    payload = output.getvalue()
    scale = struct.unpack("<f", payload[:4])[0]
    quantized = np.frombuffer(payload[4:], dtype=np.int8)

    assert scale == np.float32(2.0 / 127.0)
    np.testing.assert_array_equal(quantized, np.array([-127, 0, 127], dtype=np.int8))


def test_create_weight_file_writes_expected_header_and_stable_content(tmp_path):
    generator = load_generator()
    output_path = tmp_path / "guard.bin"
    repeat_path = tmp_path / "guard-repeat.bin"

    size = generator.create_weight_file("guard", str(output_path))
    repeat_size = generator.create_weight_file("guard", str(repeat_path))

    data = output_path.read_bytes()
    magic, n_layer, n_embed, n_head, vocab_size, ctx_len, em_scale = struct.unpack(
        "<IBHBHBB",
        data[:12],
    )

    assert size == output_path.stat().st_size
    assert repeat_size == repeat_path.stat().st_size
    assert data == repeat_path.read_bytes()
    assert magic == 0x53454149
    assert n_layer == generator.MODEL_CONFIG["n_layer"]
    assert n_embed == generator.MODEL_CONFIG["n_embed"]
    assert n_head == generator.MODEL_CONFIG["n_head"]
    assert vocab_size == generator.MODEL_CONFIG["vocab_size"]
    assert ctx_len == generator.MODEL_CONFIG["ctx_len"]
    assert em_scale == 56
    assert size > 12
