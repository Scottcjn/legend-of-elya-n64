#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight Export Tool for N64 LLM
Exports trained PyTorch model weights to N64-compatible .bin format

Usage:
    python export_weights.py --input model.pt --output sophia.bin
"""

import torch
import struct
import numpy as np
import argparse
import os

MODEL_CONFIG = {
    'n_layer': 4,
    'n_embed': 128,
    'n_head': 4,
    'vocab_size': 256,
    'ctx_len': 64,
}


def export_q8_tensor(f, tensor, name=""):
    """Export a tensor in Q8 format (int8 + scale)"""
    flat = tensor.detach().numpy().flatten()
    max_abs = np.max(np.abs(flat))
    
    if max_abs > 0:
        scale = max_abs / 127.0
        quantized = np.round(flat / scale).astype(np.int8)
    else:
        scale = 1.0
        quantized = np.zeros_like(flat, dtype=np.int8)
    
    # Write scale (float32) + data (int8)
    f.write(struct.pack('<f', scale))
    f.write(quantized.tobytes())
    
    return len(quantized)


def export_weights(input_path, output_path):
    """Export PyTorch model weights to N64 .bin format"""
    
    # Load state dict
    if input_path.endswith('.pt') or input_path.endswith('.pth'):
        state_dict = torch.load(input_path, map_location='cpu')
    else:
        raise ValueError("Input must be .pt or .pth file")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Write header
        magic = 0x53454149  # "SEAI"
        f.write(struct.pack('<I', magic))
        f.write(struct.pack('<B', MODEL_CONFIG['n_layer']))
        f.write(struct.pack('<H', MODEL_CONFIG['n_embed']))
        f.write(struct.pack('<B', MODEL_CONFIG['n_head']))
        f.write(struct.pack('<H', MODEL_CONFIG['vocab_size']))
        f.write(struct.pack('<B', MODEL_CONFIG['ctx_len']))
        f.write(struct.pack('<B', 56))  # em_scale_x16
        
        bytes_written = 16  # Header size
        
        # Export embedding table
        if 'token_embedding_table.weight' in state_dict:
            n = export_q8_tensor(f, state_dict['token_embedding_table.weight'], 'embedding')
            bytes_written += n * 2  # scale + data
        
        # Export transformer layers
        for i in range(MODEL_CONFIG['n_layer']):
            layer_prefix = f'layers.{i}.'
            
            # Attention
            for name, param in state_dict.items():
                if name.startswith(f'{layer_prefix}sa.'):
                    n = export_q8_tensor(f, param, name)
                    bytes_written += n * 2
            
            # FFN
            for name, param in state_dict.items():
                if name.startswith(f'{layer_prefix}ffn.'):
                    n = export_q8_tensor(f, param, name)
                    bytes_written += n * 2
            
            # LayerNorm
            for name, param in state_dict.items():
                if name.startswith(f'{layer_prefix}ln'):
                    n = export_q8_tensor(f, param, name)
                    bytes_written += n * 2
        
        # Export output layer
        if 'ln_f.weight' in state_dict:
            n = export_q8_tensor(f, state_dict['ln_f.weight'], 'ln_f')
            bytes_written += n * 2
        
        if 'lm_head.weight' in state_dict:
            n = export_q8_tensor(f, state_dict['lm_head.weight'], 'lm_head')
            bytes_written += n * 2
    
    file_size = os.path.getsize(output_path)
    print(f"✅ Exported {output_path} ({file_size} bytes)")
    return file_size


def main():
    parser = argparse.ArgumentParser(description='Export model weights to N64 format')
    parser.add_argument('--input', '-i', required=True, help='Input PyTorch model file (.pt)')
    parser.add_argument('--output', '-o', required=True, help='Output .bin file')
    
    args = parser.parse_args()
    
    print(f"📤 Exporting weights...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    
    export_weights(args.input, args.output)
    
    print("\n✅ Export complete!")


if __name__ == '__main__':
    main()
