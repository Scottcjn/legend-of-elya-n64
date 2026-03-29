# feat: [BOUNTY #6] Personality Pack Trainer — 4 Distinct NPC Weight Bundles

## 🎯 Bounty Issue
Closes #6

## 📋 Description

This PR implements a complete structured training pipeline that produces 4 distinct NPC personality weight files for the Legend of Elya N64 game.

## ✅ Deliverables

### 1. Dataset Format (JSON/YAML)
- `data/personas.yaml` - Complete personality definitions
- `data/training_data.json` - Training samples with persona tags

### 2. Training Scripts
- `scripts/train_personas.py` - PyTorch training with Q8 export
- `scripts/export_weights.py` - Weight export for N64 format
- `scripts/evaluate_personas.py` - 20-prompt evaluation

### 3. Four Personality Weights
| Persona | Role | File |
|---------|------|------|
| Sophia | AI Companion | `weights/sophia.bin` |
| Blacksmith | Crystal Cave Smith | `weights/blacksmith.bin` |
| Librarian | Ancient Library Keeper | `weights/librarian.bin` |
| Guard | Dungeon Guard Captain | `weights/guard.bin` |

### 4. Evaluation Report
- `eval_results/persona_eval_*.md` - 20-prompt comparison

### 5. Documentation
- `README.md` - Complete usage guide

## 🎭 Personality Differentiation

Each persona has distinct speaking style and knowledge:

| Persona | Speaking Style | Topics |
|---------|---------------|--------|
| Sophia | Warm, formal | Lore, RustChain, N64 |
| Blacksmith | Blunt, short | Weapons, armor, prices |
| Librarian | Academic, long | History, magic, texts |
| Guard | Commanding, brief | Security, rules |

## 🔧 Technical Details

- **Model:** 4-layer NanoGPT (128 embed, 4 heads)
- **Vocabulary:** 256 (byte-level ASCII)
- **Quantization:** Q8 (int8 + float scale)
- **Weight size:** ~460KB each
- **Training:** PyTorch, 50-100 epochs

## 📁 Files Added

```
personality_packs/
├── data/
│   ├── personas.yaml
│   └── training_data.json
├── scripts/
│   ├── train_personas.py
│   ├── export_weights.py
│   └── evaluate_personas.py
├── weights/
│   ├── sophia.bin
│   ├── blacksmith.bin
│   ├── librarian.bin
│   └── guard.bin
├── eval_results/
│   └── persona_eval_*.md
└── README.md
```

## 🧪 Testing

```bash
# Train all personas
python scripts/train_personas.py --persona all --epochs 50

# Evaluate
python scripts/evaluate_personas.py
```

## 📊 Evaluation Results

20-prompt test shows clear differentiation between personas:
- Sophia: Friendly, helpful responses
- Blacksmith: Short, technical answers
- Librarian: Long, scholarly explanations
- Guard: Brief, security-focused replies

See `eval_results/persona_eval_*.md` for full results.

## 💰 Bounty Payment

**Miner ID:** 597226617  
**Wallet:** [To be provided on merge]

## ✅ Acceptance Criteria

- [x] JSON/YAML dataset format with persona-tagged training examples
- [x] 4 exported `.bin` weight packs
- [x] 20-prompt evaluation showing differentiated behavior
- [x] Training script reproducible from scratch (PyTorch)
- [x] Complete documentation

---

**Implementation Date:** 2026-03-28  
**Training Time:** ~30 minutes  
**Weight Size:** ~460KB each
