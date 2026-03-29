#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Personality Pack Trainer for Legend of Elya N64
Trains 4 distinct NPC personality weight files from base model

Usage:
    python train_personas.py --persona all  # Train all 4 personalities
    python train_personas.py --persona sophia  # Train specific persona
"""

import torch
import torch.nn as nn
import json
import yaml
import os
import struct
import numpy as np
import argparse
from datetime import datetime

# Model configuration (matching nano_gpt.h)
MODEL_CONFIG = {
    'n_layer': 4,
    'n_embed': 128,
    'n_head': 4,
    'vocab_size': 256,
    'ctx_len': 64,
    'quantization': 'Q8'
}


class NanoGPT(nn.Module):
    """Simplified NanoGPT model matching the N64 implementation"""
    def __init__(self, config):
        super().__init__()
        self.n_layer = config['n_layer']
        self.n_embed = config['n_embed']
        self.n_head = config['n_head']
        self.vocab_size = config['vocab_size']
        
        # Embedding
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embed'])
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layer'])
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config['n_embed'])
        self.lm_head = nn.Linear(config['n_embed'], config['vocab_size'])
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MODEL_CONFIG['ctx_len']:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config['n_embed'] // config['n_head']
        self.sa = nn.MultiheadAttention(
            config['n_embed'], 
            config['n_head'], 
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config['n_embed'], 4 * config['n_embed']),
            nn.GELU(),
            nn.Linear(4 * config['n_embed'], config['n_embed'])
        )
        self.ln1 = nn.LayerNorm(config['n_embed'])
        self.ln2 = nn.LayerNorm(config['n_embed'])
    
    def forward(self, x):
        attn_out, _ = self.sa(x, x, x, is_causal=True)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


def load_training_data(data_path):
    """Load and preprocess training data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by persona
    persona_data = {}
    for persona_name, persona_info in data['personas'].items():
        persona_data[persona_name] = persona_info['samples']
    
    return persona_data


def load_personas_config(config_path):
    """Load personas configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('personas', {})


def encode_text(text):
    """Simple byte-level tokenization (ASCII)"""
    return [ord(c) for c in text if 0 <= ord(c) < 256]


def create_training_pairs(samples):
    """Create training pairs from samples"""
    pairs = []
    for sample in samples:
        prompt = sample['prompt']
        response = sample['response']
        # Combine prompt and response for training
        full_text = prompt + " " + response
        tokens = encode_text(full_text)
        
        if len(tokens) > 2:
            pairs.append((tokens[:-1], tokens[1:]))
    
    return pairs


def train_persona(model, persona_name, samples, config, epochs=100, lr=1e-3, batch_size=4):
    """Train model for a specific persona"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create training pairs
    training_pairs = create_training_pairs(samples)
    
    if len(training_pairs) == 0:
        print(f"   ⚠️ No training pairs for {persona_name}")
        return model
    
    print(f"\n🎭 Training {persona_name}...")
    print(f"   Samples: {len(samples)}")
    print(f"   Training pairs: {len(training_pairs)}")
    print(f"   Epochs: {epochs}")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(training_pairs), batch_size):
            batch_pairs = training_pairs[i:i+batch_size]
            
            # Pad sequences to same length
            max_len = max(len(x) for x, _ in batch_pairs)
            
            x_batch = []
            y_batch = []
            for x, y in batch_pairs:
                # Pad with zeros
                x_padded = x + [0] * (max_len - len(x))
                y_padded = y + [0] * (max_len - len(y))
                x_batch.append(x_padded)
                y_batch.append(y_padded)
            
            x_tensor = torch.tensor(x_batch, dtype=torch.long)
            y_tensor = torch.tensor(y_batch, dtype=torch.long)
            
            optimizer.zero_grad()
            _, loss = model(x_tensor, y_tensor)
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def export_weights_q8(model, output_path):
    """Export weights in N64-compatible Q8 format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    state_dict = model.state_dict()
    
    with open(output_path, 'wb') as f:
        # Write header (magic + config)
        magic = 0x53454149  # "SEAI" in little-endian
        f.write(struct.pack('<I', magic))
        f.write(struct.pack('<B', MODEL_CONFIG['n_layer']))  # n_layers
        f.write(struct.pack('<H', MODEL_CONFIG['n_embed']))  # n_embed
        f.write(struct.pack('<B', MODEL_CONFIG['n_head']))  # n_heads
        f.write(struct.pack('<H', MODEL_CONFIG['vocab_size']))  # vocab_size
        f.write(struct.pack('<B', MODEL_CONFIG['ctx_len']))  # ctx_len
        f.write(struct.pack('<B', 56))  # em_scale_x16 (3.5 * 16)
        
        # Export embedding table (Q8)
        embed_weights = state_dict['token_embedding_table.weight'].detach().numpy()
        export_q8_tensor(f, embed_weights)
        
        # Export layer weights
        for i, layer in enumerate(model.layers):
            # Attention weights
            if hasattr(layer.sa, 'in_proj_weight'):
                in_proj = layer.sa.in_proj_weight.detach().numpy()
                export_q8_tensor(f, in_proj)
            
            # FFN weights
            for name, param in layer.ffn.named_parameters():
                export_q8_tensor(f, param.detach().numpy())
            
            # LayerNorm weights
            for name, param in layer.ln1.named_parameters():
                export_q8_tensor(f, param.detach().numpy())
            for name, param in layer.ln2.named_parameters():
                export_q8_tensor(f, param.detach().numpy())
        
        # Output layer
        ln_f_weight = state_dict['ln_f.weight'].detach().numpy()
        export_q8_tensor(f, ln_f_weight)
        
        lm_head_weight = state_dict['lm_head.weight'].detach().numpy()
        export_q8_tensor(f, lm_head_weight)
    
    file_size = os.path.getsize(output_path)
    print(f"✅ Weights exported to {output_path} ({file_size} bytes)")


def export_q8_tensor(f, tensor):
    """Export a tensor in Q8 format (int8 + scale)"""
    flat = tensor.flatten()
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


def main():
    parser = argparse.ArgumentParser(description='Train NPC personality packs')
    parser.add_argument('--persona', type=str, default='all', 
                       choices=['all', 'sophia', 'blacksmith', 'librarian', 'guard'],
                       help='Which persona to train')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='weights', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print("📂 Loading training data...")
    persona_data = load_training_data('data/training_data.json')
    personas_config = load_personas_config('data/personas.yaml')
    
    # Determine which personas to train
    if args.persona == 'all':
        personas_to_train = ['sophia', 'blacksmith', 'librarian', 'guard']
    else:
        personas_to_train = [args.persona]
    
    # Train each persona
    for persona_name in personas_to_train:
        if persona_name not in persona_data:
            print(f"⚠️ No data for persona: {persona_name}")
            continue
        
        # Initialize fresh model for each persona
        model = NanoGPT(MODEL_CONFIG)
        
        # Train
        model = train_persona(
            model, 
            persona_name, 
            persona_data[persona_name],
            MODEL_CONFIG,
            epochs=args.epochs,
            lr=args.lr
        )
        
        # Export weights
        output_path = f"{args.output_dir}/{persona_name}.bin"
        export_weights_q8(model, output_path)
    
    print("\n✅ Training complete!")
    print(f"📁 Output directory: {args.output_dir}/")
    
    # List generated files
    print("\n📦 Generated weight files:")
    for persona in personas_to_train:
        path = f"{args.output_dir}/{persona}.bin"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   - {persona}.bin ({size} bytes)")


if __name__ == '__main__':
    main()
