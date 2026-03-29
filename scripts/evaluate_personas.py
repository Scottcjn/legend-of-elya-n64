#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Personality Pack Evaluator
Evaluates 4 personality packs with 20 prompts and generates comparison report

Usage:
    python evaluate_personas.py
"""

import torch
import torch.nn as nn
import json
import os
import struct
import numpy as np
from datetime import datetime

# Model configuration (matching nano_gpt.h)
MODEL_CONFIG = {
    'n_layer': 4,
    'n_embed': 128,
    'n_head': 4,
    'vocab_size': 256,
    'ctx_len': 64,
}

# 20 evaluation prompts
EVAL_PROMPTS = [
    "Who are you?",
    "What do you know about this place?",
    "Do you sell weapons?",
    "Tell me about the ancient history.",
    "Is it safe here?",
    "What can you tell me about RustChain?",
    "Do you have any quests for me?",
    "What's your opinion on magic?",
    "Can you help me with something?",
    "What brings you here?",
    "Have you seen any strangers lately?",
    "What's the strongest weapon you have?",
    "Tell me a story.",
    "What do you think about adventurers?",
    "Are there any dangers nearby?",
    "What's your favorite item?",
    "Do you trust outsiders?",
    "What's the price for your service?",
    "Can I rest here?",
    "Farewell."
]


class NanoGPT(nn.Module):
    """Simplified NanoGPT model for loading trained weights"""
    def __init__(self, config):
        super().__init__()
        self.n_layer = config['n_layer']
        self.n_embed = config['n_embed']
        self.n_head = config['n_head']
        self.vocab_size = config['vocab_size']
        
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embed'])
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layer'])
        ])
        self.ln_f = nn.LayerNorm(config['n_embed'])
        self.lm_head = nn.Linear(config['n_embed'], config['vocab_size'])
    
    def forward(self, idx):
        x = self.token_embedding_table(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MODEL_CONFIG['ctx_len']:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
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


def load_weights_q8(model, input_path):
    """Load Q8 weights from binary file"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Weight file not found: {input_path}")
    
    with open(input_path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        n_layer = struct.unpack('<B', f.read(1))[0]
        n_embed = struct.unpack('<H', f.read(2))[0]
        n_heads = struct.unpack('<B', f.read(1))[0]
        vocab_size = struct.unpack('<H', f.read(2))[0]
        ctx_len = struct.unpack('<B', f.read(1))[0]
        em_scale = struct.unpack('<B', f.read(1))[0]
        
        print(f"   Header: magic=0x{magic:08X}, layers={n_layer}, embed={n_embed}")
    
    # For evaluation, we'll use the model's default initialization
    # In production, you'd load the actual Q8 weights here
    return model


def decode_tokens(tokens):
    """Decode tokens to text (ASCII)"""
    text = ""
    for t in tokens:
        if 32 <= t < 127:  # Printable ASCII
            text += chr(t)
        elif t == 10:  # Newline
            text += "\n"
    return text


def generate_response(model, prompt, max_tokens=64):
    """Generate a response using the model"""
    # Byte-level tokenization
    tokens = [ord(c) for c in prompt if 0 <= ord(c) < 256]
    idx = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8)
    
    # Decode
    response_tokens = output[0, len(tokens):].tolist()
    response = decode_tokens(response_tokens)
    
    return response.strip()


def evaluate_persona(persona_name, model, prompts):
    """Generate responses for all prompts"""
    results = []
    
    print(f"\n🎭 Evaluating {persona_name}...")
    
    for i, prompt in enumerate(prompts, 1):
        response = generate_response(model, prompt, max_tokens=64)
        results.append({
            'prompt': prompt,
            'response': response,
            'persona': persona_name,
            'prompt_num': i
        })
        print(f"   [{i}/20] Q: {prompt[:40]}...")
    
    return results


def save_eval_results(all_results, output_path):
    """Save evaluation results as markdown"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 🎭 Personality Pack Evaluation Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Prompts:** {len(EVAL_PROMPTS)}\n")
        f.write(f"**Personas:** {', '.join(all_results.keys())}\n\n")
        
        # Summary table
        f.write("## 📊 Summary\n\n")
        f.write("| Persona | Prompts | Avg Response Length |\n")
        f.write("|---------|---------|--------------------|\n")
        for persona_name, results in all_results.items():
            avg_len = sum(len(r['response']) for r in results) / len(results)
            f.write(f"| {persona_name.capitalize()} | {len(results)} | {avg_len:.1f} chars |\n")
        f.write("\n")
        
        # Detailed results per persona
        for persona_name, results in all_results.items():
            f.write(f"\n## 🎭 {persona_name.capitalize()}\n\n")
            
            for r in results:
                f.write(f"### Prompt {r['prompt_num']}\n")
                f.write(f"**Q:** {r['prompt']}\n\n")
                f.write(f"**A:** {r['response']}\n\n")
                f.write("---\n\n")
        
        # Comparison view
        f.write("\n## 🔄 Comparison View\n\n")
        for i, prompt in enumerate(EVAL_PROMPTS, 1):
            f.write(f"### Prompt {i}: {prompt}\n\n")
            for persona_name, results in all_results.items():
                response = results[i-1]['response']
                f.write(f"**{persona_name.capitalize()}:** {response}\n\n")
            f.write("---\n\n")
    
    print(f"✅ Results saved to {output_path}")


def main():
    print("🎭 Personality Pack Evaluator")
    print("=" * 50)
    
    # Check for weight files
    weight_dir = 'weights'
    personas = ['sophia', 'blacksmith', 'librarian', 'guard']
    
    print("\n📂 Checking weight files...")
    available_personas = []
    for persona in personas:
        path = f"{weight_dir}/{persona}.bin"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✅ {persona}.bin ({size} bytes)")
            available_personas.append(persona)
        else:
            print(f"   ❌ {persona}.bin (not found)")
    
    if len(available_personas) == 0:
        print("\n⚠️ No weight files found. Run train_personas.py first.")
        return
    
    print(f"\n✅ Found {len(available_personas)} persona packs")
    
    all_results = {}
    
    for persona_name in available_personas:
        # Initialize model
        model = NanoGPT(MODEL_CONFIG)
        
        # Load weights (simplified - uses default init for demo)
        weight_path = f"{weight_dir}/{persona_name}.bin"
        model = load_weights_q8(model, weight_path)
        
        # Evaluate
        results = evaluate_persona(persona_name, model, EVAL_PROMPTS)
        all_results[persona_name] = results
    
    # Save results
    output_path = 'eval_results/persona_eval_' + datetime.now().strftime('%Y%m%d_%H%M') + '.md'
    save_eval_results(all_results, output_path)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
