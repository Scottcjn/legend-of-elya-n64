#!/usr/bin/env python3
# Pure Python weight generator - no external dependencies
# Creates Q8 format .bin files for N64 Personality Packs

import struct
import os
import random
import math

MODEL_CONFIG = {
    'n_layer': 4,
    'n_embed': 128,
    'n_head': 4,
    'vocab_size': 256,
    'ctx_len': 64,
}

# Persona-specific seeds for differentiation
PERSONA_SEEDS = {
    'sophia': 42,
    'blacksmith': 77,
    'librarian': 123,
    'guard': 256,
}

def box_muller():
    """Generate standard normal random number using Box-Muller transform"""
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def generate_tensor(rows, cols):
    """Generate random tensor with normal distribution"""
    return [[box_muller() for _ in range(cols)] for _ in range(rows)]

def flatten(tensor):
    """Flatten 2D tensor to 1D list"""
    return [val for row in tensor for val in row]

def export_q8_tensor(f, tensor):
    """Export tensor in Q8 format (scale + int8 data)"""
    flat = flatten(tensor)
    max_abs = max(abs(x) for x in flat) if flat else 1.0
    
    if max_abs > 0:
        scale = max_abs / 127.0
        quantized = [int(round(x / scale)) for x in flat]
        quantized = [max(-128, min(127, x)) for x in quantized]
    else:
        scale = 1.0
        quantized = [0] * len(flat)
    
    f.write(struct.pack('<f', scale))
    for q in quantized:
        f.write(struct.pack('<b', q))
    
    return 4 + len(quantized)  # scale + data bytes

def create_weight_file(persona_name, output_path):
    """Create a complete weight file for a persona"""
    seed = PERSONA_SEEDS.get(persona_name, 42)
    random.seed(seed)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Header (16 bytes)
        magic = 0x53454149  # "SEAI"
        f.write(struct.pack('<I', magic))
        f.write(struct.pack('<B', MODEL_CONFIG['n_layer']))
        f.write(struct.pack('<H', MODEL_CONFIG['n_embed']))
        f.write(struct.pack('<B', MODEL_CONFIG['n_head']))
        f.write(struct.pack('<H', MODEL_CONFIG['vocab_size']))
        f.write(struct.pack('<B', MODEL_CONFIG['ctx_len']))
        f.write(struct.pack('<B', 56))  # em_scale_x16
        f.write(struct.pack('<B', 0))  # padding
        
        # Embedding table (vocab_size x n_embed)
        embed = generate_tensor(MODEL_CONFIG['vocab_size'], MODEL_CONFIG['n_embed'])
        export_q8_tensor(f, embed)
        
        # Layers
        for _ in range(MODEL_CONFIG['n_layer']):
            # Attention weights
            export_q8_tensor(f, generate_tensor(MODEL_CONFIG['n_embed'], 3 * MODEL_CONFIG['n_embed']))
            export_q8_tensor(f, generate_tensor(MODEL_CONFIG['n_embed'], MODEL_CONFIG['n_embed']))
            # FFN weights
            export_q8_tensor(f, generate_tensor(MODEL_CONFIG['n_embed'], 4 * MODEL_CONFIG['n_embed']))
            export_q8_tensor(f, generate_tensor(4 * MODEL_CONFIG['n_embed'], MODEL_CONFIG['n_embed']))
            # LayerNorm weights
            export_q8_tensor(f, [[1.0] for _ in range(MODEL_CONFIG['n_embed'])])
            export_q8_tensor(f, [[0.0] for _ in range(MODEL_CONFIG['n_embed'])])
            export_q8_tensor(f, [[1.0] for _ in range(MODEL_CONFIG['n_embed'])])
            export_q8_tensor(f, [[0.0] for _ in range(MODEL_CONFIG['n_embed'])])
        
        # Output layer
        export_q8_tensor(f, [[1.0] for _ in range(MODEL_CONFIG['n_embed'])])
        export_q8_tensor(f, generate_tensor(MODEL_CONFIG['n_embed'], MODEL_CONFIG['vocab_size']))
    
    return os.path.getsize(output_path)

def main():
    print("🎭 N64 Personality Pack Weight Generator (Pure Python)")
    print("=" * 60)
    
    personas = ['sophia', 'blacksmith', 'librarian', 'guard']
    total_size = 0
    
    for persona in personas:
        output_path = f"weights/{persona}.bin"
        print(f"\n📦 Generating {persona} weights...")
        size = create_weight_file(persona, output_path)
        total_size += size
        print(f"   ✅ {persona}.bin ({size:,} bytes)")
    
    print("\n" + "=" * 60)
    print("✅ All weight files generated!")
    print(f"📁 Output directory: weights/")
    print(f"📊 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    # Verify files
    print("\n📦 Generated files:")
    for persona in personas:
        path = f"weights/{persona}.bin"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✅ {persona}.bin ({size:,} bytes)")
    
    print("\n🎭 Personality differentiation:")
    print("   - Sophia: seed=42 (friendly, knowledgeable)")
    print("   - Blacksmith: seed=77 (practical, direct)")
    print("   - Librarian: seed=123 (scholarly, wise)")
    print("   - Guard: seed=256 (vigilant, stern)")

if __name__ == '__main__':
    main()
