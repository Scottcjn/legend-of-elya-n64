#!/usr/bin/env python3
# Simplified weight generator for N64 Personality Packs
# Creates Q8 format .bin files without PyTorch dependency

import struct
import numpy as np
import os

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

def generate_persona_weights(persona_name, seed):
    """Generate deterministic weights for a persona"""
    np.random.seed(seed)
    
    # Generate embedding table (vocab_size x n_embed)
    embed = np.random.randn(MODEL_CONFIG['vocab_size'], MODEL_CONFIG['n_embed']).astype(np.float32)
    
    # Generate layer weights
    layers = []
    for i in range(MODEL_CONFIG['n_layer']):
        layer = {
            'attention_in': np.random.randn(MODEL_CONFIG['n_embed'], 3 * MODEL_CONFIG['n_embed']).astype(np.float32),
            'attention_out': np.random.randn(MODEL_CONFIG['n_embed'], MODEL_CONFIG['n_embed']).astype(np.float32),
            'ffn_up': np.random.randn(MODEL_CONFIG['n_embed'], 4 * MODEL_CONFIG['n_embed']).astype(np.float32),
            'ffn_down': np.random.randn(4 * MODEL_CONFIG['n_embed'], MODEL_CONFIG['n_embed']).astype(np.float32),
            'ln1_weight': np.ones(MODEL_CONFIG['n_embed'], dtype=np.float32),
            'ln1_bias': np.zeros(MODEL_CONFIG['n_embed'], dtype=np.float32),
            'ln2_weight': np.ones(MODEL_CONFIG['n_embed'], dtype=np.float32),
            'ln2_bias': np.zeros(MODEL_CONFIG['n_embed'], dtype=np.float32),
        }
        layers.append(layer)
    
    # Output layer
    ln_f = np.ones(MODEL_CONFIG['n_embed'], dtype=np.float32)
    lm_head = np.random.randn(MODEL_CONFIG['n_embed'], MODEL_CONFIG['vocab_size']).astype(np.float32)
    
    return {
        'embed': embed,
        'layers': layers,
        'ln_f': ln_f,
        'lm_head': lm_head,
    }

def export_q8_tensor(f, tensor):
    """Export tensor in Q8 format"""
    flat = tensor.flatten()
    max_abs = np.max(np.abs(flat))
    
    if max_abs > 0:
        scale = max_abs / 127.0
        quantized = np.round(flat / scale).astype(np.int8)
    else:
        scale = 1.0
        quantized = np.zeros_like(flat, dtype=np.int8)
    
    f.write(struct.pack('<f', scale))
    f.write(quantized.tobytes())
    return len(quantized) * 2  # scale + data

def create_weight_file(persona_name, output_path):
    """Create a complete weight file for a persona"""
    seed = PERSONA_SEEDS.get(persona_name, 42)
    weights = generate_persona_weights(persona_name, seed)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Header
        magic = 0x53454149  # "SEAI"
        f.write(struct.pack('<I', magic))
        f.write(struct.pack('<B', MODEL_CONFIG['n_layer']))
        f.write(struct.pack('<H', MODEL_CONFIG['n_embed']))
        f.write(struct.pack('<B', MODEL_CONFIG['n_head']))
        f.write(struct.pack('<H', MODEL_CONFIG['vocab_size']))
        f.write(struct.pack('<B', MODEL_CONFIG['ctx_len']))
        f.write(struct.pack('<B', 56))  # em_scale_x16
        
        # Embedding
        export_q8_tensor(f, weights['embed'])
        
        # Layers
        for layer in weights['layers']:
            export_q8_tensor(f, layer['attention_in'])
            export_q8_tensor(f, layer['attention_out'])
            export_q8_tensor(f, layer['ffn_up'])
            export_q8_tensor(f, layer['ffn_down'])
            export_q8_tensor(f, layer['ln1_weight'])
            export_q8_tensor(f, layer['ln1_bias'])
            export_q8_tensor(f, layer['ln2_weight'])
            export_q8_tensor(f, layer['ln2_bias'])
        
        # Output
        export_q8_tensor(f, weights['ln_f'])
        export_q8_tensor(f, weights['lm_head'])
    
    file_size = os.path.getsize(output_path)
    return file_size

def main():
    print("🎭 N64 Personality Pack Weight Generator")
    print("=" * 50)
    
    personas = ['sophia', 'blacksmith', 'librarian', 'guard']
    
    for persona in personas:
        output_path = f"weights/{persona}.bin"
        print(f"\n📦 Generating {persona} weights...")
        size = create_weight_file(persona, output_path)
        print(f"   ✅ {persona}.bin ({size} bytes)")
    
    print("\n✅ All weight files generated!")
    print(f"📁 Output: weights/")
    
    # List files
    print("\n📦 Generated files:")
    for persona in personas:
        path = f"weights/{persona}.bin"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   - {persona}.bin ({size:,} bytes)")

if __name__ == '__main__':
    main()
