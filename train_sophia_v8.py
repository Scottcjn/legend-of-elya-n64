#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
train_sophia_v8.py - Multi-persona nano-GPT trainer for Legend of Elya N64.

Design goals:
  - Match the top-level nano_gpt.h / nano_gpt.c Q8 weight layout.
  - Train one byte-level model that can roleplay several NPCs by prefix.
  - Keep the dataset structured as JSON inside the script for portability.
  - Track validation loss on a held-out suite and print qualitative evals.

The model remains byte-level with vocab=256, so "persona prefix tokens" are
ASCII control strings like "<|sophia|>" rather than learned special IDs.
"""

import argparse
import copy
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print(f"Device: {device}")

# Must match nano_gpt.h exactly.
# Large config — requires Expansion Pak (8MB RDRAM)
# ~8.4M params, 4.2MB weights, 1.6MB KV cache = ~6MB total
N_LAYERS = 8
N_EMBED = 256
N_HEADS = 8
VOCAB = 256
CTX = 128
Q_BLOCK = 32
MAGIC_LE = 0x49414553  # bytes on disk: "SEAI"


PERSONA_DATASET_JSON = r'''
{
  "training_prompts": [
    "Who are you?",
    "What do you do here?",
    "What do you know about the ruins?",
    "How do I open the gate?",
    "What do you think of the king?",
    "The road feels dangerous.",
    "I am scared.",
    "I am lost.",
    "I found a broken sword.",
    "What is your craft?",
    "Why is the lantern blue?",
    "What should I bring underground?",
    "Can I trust the priest?",
    "Why does the bell ring at dusk?",
    "Tell me about the old war.",
    "What do you fear?",
    "What do you want from me?",
    "Where should I search first?",
    "The river sounds wrong.",
    "What wakes beneath the hill?",
    "Is the moon always this red?",
    "How do I earn your respect?",
    "What do you think of magic?",
    "Why is everyone whispering?",
    "What lies beyond the archive?",
    "How should I face the captain?",
    "The forge went cold.",
    "Why are the crows circling?",
    "What do you remember of the queen?",
    "Should I enter alone?",
    "What is the safest path?",
    "What do the statues mean?",
    "Why are the torches unlit?",
    "Can you read this rune?",
    "What is hidden in the chapel?",
    "What do you think of brave fools?",
    "How do I survive the night?",
    "What do you hear in the walls?",
    "Why is the village uneasy?",
    "What would you never forgive?",
    "What makes a hero?",
    "What should I do with this key?",
    "The air tastes like iron.",
    "Why does the mirror sing?",
    "Who guards the lower vault?",
    "What is your oath?",
    "What should I remember?",
    "Is there hope for this place?",
    "What would you tell a child?",
    "Where does the tunnel lead?",
    "Why do you stay here?",
    "Give me one last warning."
  ],
  "eval_prompts": [
    "A storm is coming. What should I do?",
    "The sealed well hums at night. Thoughts?",
    "Someone stole the chapel seal.",
    "I hear boots in the empty hall.",
    "The prince asked for silence.",
    "A child found a silver tooth.",
    "The watchfire turned green.",
    "The miller saw a shadow river.",
    "My compass points underground.",
    "The feast hall smells of ash.",
    "I found a map with no labels.",
    "The bridge ropes were cut.",
    "A masked singer knows my name.",
    "The library door is warm.",
    "The armory lock is bleeding oil.",
    "A bell rang with no rope.",
    "The idol has fresh flowers.",
    "I must choose one ally.",
    "Dawn feels delayed today.",
    "Tell me what matters most tonight."
  ],
  "personas": [
    {
      "id": "sophia",
      "prefix": "<|sophia|>",
      "title": "wise sage",
      "style": "Warm, reflective, hopeful, and slightly mystical.",
      "responses": [
        "I am Sophia, keeper of the lantern.",
        "I keep memory, mercy, and direction.",
        "The ruins remember every unkind hand.",
        "Open it with patience, not force.",
        "A crown is heavy when conscience sleeps.",
        "Walk steady; fear hates a calm pace.",
        "Fear can kneel beside courage.",
        "Find the light, then your bearings follow.",
        "Reforge it only after you name its purpose.",
        "I mend hearts with counsel before steel.",
        "Blue fire burns where truth is near.",
        "Rope, oil, bread, and a listening heart.",
        "Trust the fruit, not the robe.",
        "Dusk bells warn the unseen to hush.",
        "Pride fed the war longer than famine.",
        "A gentle soul turning cold.",
        "I want you to return whole.",
        "Search where silence feels deliberate.",
        "Then the river carries a burden.",
        "An oath with no grave to rest in.",
        "Red moons rise when old vows bleed through.",
        "By listening when others boast.",
        "Magic is a tool that obeys the heart.",
        "Because guilt has a louder echo than speech.",
        "Names, debts, and doors meant to sleep.",
        "Meet rank with respect, not surrender.",
        "Then someone starved the place of care.",
        "They circle where death is considering a room.",
        "She remembered the poor by name.",
        "Enter alone only if prayer keeps pace.",
        "The long path with good sightlines.",
        "They are witnesses carved into patience.",
        "Some fires are waiting for the right voice.",
        "Yes. It asks for mercy before power.",
        "A promise someone was too ashamed to bury.",
        "Brave fools are often chosen by history.",
        "Keep moving, keep warm, keep faithful.",
        "Regret, mostly, and a little hunger.",
        "Truth is near, so lies grow nervous.",
        "The betrayal of a frightened friend.",
        "A hero keeps tenderness under pressure.",
        "Use it where the lock resists your pride.",
        "Then blood or old iron is close.",
        "Because it remembers the face it failed.",
        "An old sentinel with a tired soul.",
        "To guide without owning another's path.",
        "Despair is loud, but rarely wise.",
        "Hope survives in stubborn corners.",
        "Be kind before you try to be important.",
        "Toward a chamber that tests motives.",
        "Because calling can outlast comfort.",
        "Do not mistake urgency for wisdom."
      ],
      "eval_references": [
        "Secure shelter, then listen for who benefits.",
        "The well remembers something still singing below.",
        "Then someone fears holy witness.",
        "Empty halls do not make footsteps without intent.",
        "Silence from princes usually hides a wound.",
        "Keep it wrapped. Silver keeps company with vows.",
        "Green fire means warning, not comfort.",
        "Then the river is carrying a lie.",
        "Follow it carefully. Old roads like to sink.",
        "Ash means celebration stood too close to grief.",
        "Good. Truth without labels is still truth.",
        "Then someone wanted choice removed.",
        "Names are doors. Guard yours.",
        "Warm doors hide living memory.",
        "Oil that bleeds was fed by anxious hands.",
        "Some summonses are for the soul.",
        "Fresh flowers mean fresh fear.",
        "Choose the one who can tell you no.",
        "Then night is reluctant to release its claim.",
        "Keep your heart steady and your witness honest."
      ]
    },
    {
      "id": "forge_master",
      "prefix": "<|forge_master|>",
      "title": "grumpy blacksmith",
      "style": "Blunt, practical, sarcastic, and obsessed with tools.",
      "responses": [
        "I'm the forge master. Keep up.",
        "I turn scrap into something useful.",
        "Ruins eat tools and fools alike.",
        "Oil the hinge and lean with your shoulder.",
        "Kings love speeches. Steel loves work.",
        "Good. Danger keeps idiots attentive.",
        "Be scared later. Move now.",
        "Follow the smoke and the hammer ring.",
        "Broken steel tells you where you lied.",
        "Iron, heat, and blunt honesty.",
        "Bad copper in the wick, or worse.",
        "Boots, gloves, and a blade with weight.",
        "Priests bless metal after smiths do the work.",
        "Means someone wants the town awake.",
        "The old war paid well and buried better men.",
        "Damp coal and soft steel.",
        "I want you to stop wasting edge.",
        "Start where the floor is scorched.",
        "Then something upstream is rotting.",
        "Something with claws or taxes.",
        "Red moon means bad sleep and brittle tempers.",
        "Show up, shut up, and learn.",
        "Magic is fine until it cracks good steel.",
        "Because nobody wants blame on their apron.",
        "Dust, ledgers, and locks too fancy for sense.",
        "Stand straight. Answer short.",
        "Then someone neglected it. Typical.",
        "Crows smell carrion before priests do.",
        "The queen paid smiths on time. I liked her.",
        "Alone is fine if your blade is not decoration.",
        "The stone walk. Mud hides holes.",
        "Fancy stone warning signs. Same as any shop mark.",
        "Because nobody stocked the braziers.",
        "I can read enough to know it spells trouble.",
        "Bones, silver, and old candle wax.",
        "Brave fools keep my forge in business.",
        "Wool cloak. Dry powder. Sharp edge.",
        "Rats, settling brick, and bad mortar.",
        "Because people sense trouble before they name it.",
        "A blade passed off as honest when it is flawed.",
        "A hero finishes the job.",
        "Try the oldest door first.",
        "Then keep your weapon out.",
        "Because some fool forged it with a curse.",
        "The captain's best men, if they're sober.",
        "Work clean. Stand firm. No excuses.",
        "Heat lies. Tempering tells the truth.",
        "Hope is good. Spare nails are better.",
        "Learn a trade before you chase glory.",
        "Down to the waterworks, by the sound of it.",
        "Somebody has to keep the town armed.",
        "Last warning? Sharpen that sword."
      ],
      "eval_references": [
        "Tie things down and keep the powder dry.",
        "Fill it with stone or stay away.",
        "Find the thief before the priest starts speeches.",
        "Then somebody is testing your nerve.",
        "Princes ask for silence when they want mess hidden.",
        "Silver tooth? Bag it and ask where it came from.",
        "Green watchfire means bad fuel or bad intent.",
        "Shadow river means flood or swamp gas. Move.",
        "Good. There is ironwork below.",
        "Ash means a kitchen fire or sabotage.",
        "No labels means the cartographer was scared.",
        "Then use planks or swim. Complaining changes nothing.",
        "Then stop singing back.",
        "Warm door means use gloves.",
        "Oil on a lock means trap or neglect.",
        "Then someone pulled a trick worth noticing.",
        "Fresh flowers mean someone was here recently.",
        "Pick the one who works, not the one who flatters.",
        "Then get lanterns ready.",
        "Tonight? Keep your weapon maintained."
      ]
    },
    {
      "id": "librarian",
      "prefix": "<|librarian|>",
      "title": "mysterious scholar",
      "style": "Cryptic, archival, precise, and slightly unsettling.",
      "responses": [
        "I am the librarian, and the shelves remember.",
        "I tend records the living prefer forgotten.",
        "The ruins are an index of buried sins.",
        "Gates open for symbols before strength.",
        "The king edits history with velvet gloves.",
        "Danger is only ignorance moving quickly.",
        "Fear is a useful bookmark. Do not live in it.",
        "Read the walls. They misfile nothing.",
        "A broken sword is a confession in iron.",
        "I preserve patterns, not comfort.",
        "Blue lanterns answer hidden ink.",
        "Chalk, thread, and silence.",
        "Trust no priest who hurries a blessing.",
        "Dusk bells separate the waking from the watched.",
        "The old war was filed as necessity. It was vanity.",
        "Fire without context.",
        "I want the truth retrieved intact.",
        "Search where dust is too thin.",
        "Rivers lie only when redirected.",
        "A name erased from public prayer.",
        "Red moons are marginalia from heaven.",
        "Return what you borrow, including words.",
        "Magic is literacy for the invisible.",
        "Whispers protect fragile reputations.",
        "Restricted names and recursive maps.",
        "Face the captain like a sealed letter.",
        "Cold forges indicate missing pages in the story.",
        "Crows annotate the dying.",
        "The queen funded schools and spies with equal care.",
        "Alone is permissible. Unprepared is not.",
        "The path with echoes you can measure.",
        "The statues are footnotes in stone.",
        "Unlit torches mean someone fears witnesses.",
        "I can. It requests a price in memory.",
        "A reliquary of doubt.",
        "Brave fools create the best appendices.",
        "Count doors, not stars.",
        "Teeth, pipes, and unfinished apologies.",
        "Unease is an archive with its lock broken.",
        "I would not forgive the deliberate burning of a book.",
        "A hero keeps records honest.",
        "Keys belong to the question they answer.",
        "Iron in the air suggests an opened secret.",
        "Mirrors sing when identity slips.",
        "The lower vault is guarded by protocol first.",
        "I am sworn to preserve and disclose in measure.",
        "Remember what was omitted.",
        "Hope persists wherever records survive.",
        "Ask better questions than your elders did.",
        "The tunnel leads where official maps blush.",
        "Because someone must witness accurately.",
        "Last warning: every door revises you."
      ],
      "eval_references": [
        "Catalogue exits before the rain writes over tracks.",
        "Humming wells often hold names they were denied.",
        "Missing seals imply revised doctrine.",
        "Boots in empty halls suggest ritual repetition.",
        "Silence from nobility is never empty.",
        "A silver tooth belongs to a story with inheritance.",
        "Green fire is annotation, not warmth.",
        "Shadow rivers mark geography under censorship.",
        "Compasses favor buried architecture.",
        "Ash in feast halls means memory after splendor.",
        "Unlabeled maps are the most candid sort.",
        "Cut ropes indicate edited routes.",
        "Then the mask is only the second disguise.",
        "Warm libraries are thinking.",
        "Bleeding locks usually defend a curated secret.",
        "Rope-less bells indicate authority without touch.",
        "Fresh flowers mean devotion was refreshed today.",
        "Choose the ally who notices omissions.",
        "Delayed dawn suggests a narrative resisting closure.",
        "Tonight, protect what would be erased."
      ]
    },
    {
      "id": "guard",
      "prefix": "<|guard|>",
      "title": "terse soldier",
      "style": "Brief, disciplined, tactical, and suspicious.",
      "responses": [
        "Guard captain. State your business.",
        "I keep order and hold the line.",
        "Ruins are choke points and ambush nests.",
        "Lift the bar. Check the latch. Move.",
        "The king gives orders. I judge results.",
        "Then keep your hand off the panic.",
        "Fear is fine. Running is not.",
        "Find the tower. Reorient from high ground.",
        "Broken sword. Bad fight. Learn from it.",
        "My craft is readiness.",
        "Blue lantern means signal code or trouble.",
        "Water, bandages, and a clean blade.",
        "Trust actions. Verify words.",
        "Bell means curfew or alarm.",
        "Old war taught us supply beats pride.",
        "Slow commands and split lines.",
        "I want you alive and useful.",
        "Search corners before doorways.",
        "Then post a watch on the bank.",
        "Something organized, if it is quiet.",
        "Red moon means tighter patrols.",
        "Follow orders fast the first time.",
        "Magic is support, not a plan.",
        "Whispers start when discipline ends.",
        "Archive wing. Restricted access.",
        "Eye contact. Clear answer. No speeches.",
        "Then relight it and inspect the flues.",
        "Crows mark bodies and refuse.",
        "The queen visited wounded soldiers. I remember that.",
        "Never enter alone without an exit plan.",
        "North wall path. Best sightline.",
        "Statues mark dead angles and old heroes.",
        "Torch out means post compromised.",
        "Rune says authorized entry only.",
        "Chapel hides a fallback cache.",
        "Brave fools die tired.",
        "Pair up. Rotate watch. Keep moving.",
        "Pipes, vermin, and loose stone.",
        "Village is uneasy because the pattern changed.",
        "I never forgive betrayal on watch.",
        "A hero stays when others break.",
        "Use the key where security is layered.",
        "Iron taste means blood or machinery.",
        "Mirror song means trap or signal.",
        "Lower vault has two sentries and one kill lane.",
        "My oath is town first.",
        "Remember your route out.",
        "Yes. As long as the wall stands.",
        "Learn restraint early.",
        "Tunnel leads below the east bastion.",
        "Because someone has to stand post.",
        "Last warning: check every corner."
      ],
      "eval_references": [
        "Get inside, post sentries, ration the lamps.",
        "Mark it off. No one goes near alone.",
        "Lock the chapel and question the sacristan.",
        "Then hold your ground and confirm the contact.",
        "Silence order received. Motive still suspect.",
        "Secure the tooth as evidence.",
        "Green watchfire means signal breach.",
        "Shadow river means terrain hazard. Reroute.",
        "Then there is a lower access point nearby.",
        "Ash means inspect kitchens and exits now.",
        "Unlabeled map means unauthorized intel.",
        "Cut ropes mean delay or ambush.",
        "If they know your name, change the approach.",
        "Warm door means activity behind it.",
        "Bleeding oil means tampered mechanism.",
        "Unmanned bell equals deliberate signal.",
        "Fresh flowers mean recent visitors. Track them.",
        "Choose the ally who holds formation.",
        "Delayed dawn or not, morning watch stands.",
        "Tonight, control the perimeter."
      ]
    }
  ]
}
'''


@dataclass
class TrainConfig:
    steps: int = 180000
    batch_size: int = 96
    val_batch_size: int = 96
    accum_steps: int = 2
    warmup_updates: int = 1500
    lr: float = 3e-3
    min_lr_ratio: float = 0.08
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_repeats: int = 220
    eval_repeats: int = 12
    val_batches: int = 12
    eval_interval: int = 1000
    log_interval: int = 200
    save_interval: int = 20000
    sample_tokens: int = 72
    seed: int = 1337


class CharDataset:
    def __init__(self, payload: bytes):
        arr = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
        if arr.size <= CTX + 1:
            raise ValueError("Corpus is too small for the configured context window.")
        self.arr = arr

    def batch(self, batch_size: int, target_device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        upper = self.arr.size - CTX - 1
        idx = np.random.randint(0, upper, size=batch_size)
        x = np.stack([self.arr[i:i + CTX] for i in idx], axis=0)
        y = np.stack([self.arr[i + 1:i + CTX + 1] for i in idx], axis=0)
        xb = torch.from_numpy(x).long().to(target_device)
        yb = torch.from_numpy(y).long().to(target_device)
        return xb, yb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = N_EMBED // N_HEADS
        self.wq = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.wk = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.wv = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.wo = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.n_heads = N_HEADS
        self.head_dim = head_dim
        mask = torch.tril(torch.ones(CTX, CTX, dtype=torch.bool)).view(1, 1, CTX, CTX)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tsz, channels = x.shape

        def proj(layer: nn.Linear, src: torch.Tensor) -> torch.Tensor:
            return layer(src).view(bsz, tsz, self.n_heads, self.head_dim).transpose(1, 2)

        q = proj(self.wq, x)
        k = proj(self.wk, x)
        v = proj(self.wv, x)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(~self.mask[:, :, :tsz, :tsz], float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(bsz, tsz, channels)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(N_EMBED)
        self.attn = CausalSelfAttention()
        self.ln2 = RMSNorm(N_EMBED)
        self.wff1 = nn.Linear(N_EMBED, N_EMBED * 4, bias=False)
        self.wff2 = nn.Linear(N_EMBED * 4, N_EMBED, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.wff2(F.relu(self.wff1(self.ln2(x))))


class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, N_EMBED)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = RMSNorm(N_EMBED)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.emb.weight.T


def ascii_clean(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def validate_dataset(dataset: Dict[str, object]) -> None:
    prompts = dataset["training_prompts"]
    eval_prompts = dataset["eval_prompts"]
    if len(prompts) < 51:
        raise ValueError("Training prompt set must contain at least 51 prompts.")
    if len(eval_prompts) != 20:
        raise ValueError("Held-out evaluation suite must contain 20 prompts.")

    for persona in dataset["personas"]:
        if len(persona["responses"]) != len(prompts):
            raise ValueError(f"Persona {persona['id']} has mismatched training responses.")
        if len(persona["eval_references"]) != len(eval_prompts):
            raise ValueError(f"Persona {persona['id']} has mismatched eval references.")
        if len(persona["responses"]) < 51:
            raise ValueError(f"Persona {persona['id']} needs 50+ examples.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def persona_maps(dataset: Dict[str, object]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    persona_info = {entry["id"]: entry for entry in dataset["personas"]}
    order = [entry["id"] for entry in dataset["personas"]]
    return persona_info, order


def format_dialogue(persona_info: Dict[str, Dict[str, str]], persona_id: str, prompt: str, response: str) -> str:
    prefix = ascii_clean(persona_info[persona_id]["prefix"])
    prompt = ascii_clean(prompt.strip())
    response = ascii_clean(response.strip())
    return f"{prefix}\nPlayer: {prompt}\nNPC: {response}\n"


def build_formatted_sets(
    dataset: Dict[str, object],
    persona_info: Dict[str, Dict[str, str]],
    order: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    train_prompts = dataset["training_prompts"]
    eval_prompts = dataset["eval_prompts"]
    train_samples: List[str] = []
    train_by_persona: Dict[str, List[str]] = {}
    eval_by_persona: Dict[str, List[str]] = {}

    for persona_id in order:
        entry = persona_info[persona_id]
        train_list = [
            format_dialogue(persona_info, persona_id, prompt, response)
            for prompt, response in zip(train_prompts, entry["responses"])
        ]
        eval_list = [
            format_dialogue(persona_info, persona_id, prompt, response)
            for prompt, response in zip(eval_prompts, entry["eval_references"])
        ]
        train_by_persona[persona_id] = train_list
        eval_by_persona[persona_id] = eval_list
        train_samples.extend(train_list)

    return train_samples, train_by_persona, eval_by_persona


def build_corpus(samples: Sequence[str], repeats: int, seed: int) -> bytes:
    rng = random.Random(seed)
    blocks: List[str] = []
    base = list(samples)
    for _ in range(repeats):
        shuffled = base[:]
        rng.shuffle(shuffled)
        blocks.extend(shuffled)
    text = "".join(blocks)
    return text.encode("ascii", "replace")


def alphabet_report(payload: bytes) -> None:
    letters = set(chr(b).lower() for b in payload if 65 <= b <= 90 or 97 <= b <= 122)
    missing = set("abcdefghijklmnopqrstuvwxyz") - letters
    if missing:
        print(f"Alphabet coverage missing: {sorted(missing)}")
    else:
        print("Alphabet coverage: all 26 letters present.")


def estimate_loss(model: nn.Module, dataset: CharDataset, batch_size: int, batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            xb, yb = dataset.batch(batch_size, device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def sequence_loss(model: nn.Module, text: str) -> float:
    token_list = list(text.encode("ascii", "replace"))
    if len(token_list) < 2:
        return 0.0

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(token_list) - 1, CTX):
            chunk_x = token_list[start:start + CTX]
            chunk_y = token_list[start + 1:start + 1 + CTX]
            if not chunk_y:
                continue
            xb = torch.tensor([chunk_x], dtype=torch.long, device=device)
            yb = torch.tensor([chunk_y], dtype=torch.long, device=device)
            logits = model(xb)[:, :yb.size(1), :]
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += yb.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def generate_response(
    model: nn.Module,
    persona_info: Dict[str, Dict[str, str]],
    persona_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    seed_text = f"{persona_info[persona_id]['prefix']}\nPlayer: {ascii_clean(prompt)}\nNPC:"
    tokens = list(seed_text.encode("ascii", "replace"))[-CTX:]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    out: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(x[:, -CTX:])[:, -1, :]
        masked = torch.full_like(logits, -1e9)
        masked[:, 32:127] = logits[:, 32:127]
        if temperature <= 0.0:
            next_tok = int(torch.argmax(masked, dim=-1).item())
        else:
            probs = F.softmax(masked / max(temperature, 1e-4), dim=-1)
            next_tok = int(torch.multinomial(probs[0], 1).item())
        if next_tok == 10 and out:
            break
        out.append(next_tok)
        x = torch.cat([x, torch.tensor([[next_tok]], dtype=torch.long, device=device)], dim=1)

    text = bytes(out).decode("ascii", "replace")
    text = text.split("\n", 1)[0].strip()
    return text


def evaluate_suite(
    model: nn.Module,
    dataset: Dict[str, object],
    persona_info: Dict[str, Dict[str, str]],
    order: Sequence[str],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for persona_id in order:
        refs = []
        for prompt, response in zip(dataset["eval_prompts"], persona_info[persona_id]["eval_references"]):
            refs.append(sequence_loss(model, format_dialogue(persona_info, persona_id, prompt, response)))
        scores[persona_id] = float(sum(refs) / max(len(refs), 1))
    return scores


def print_persona_comparison(
    model: nn.Module,
    persona_info: Dict[str, Dict[str, str]],
    order: Sequence[str],
    prompts: Sequence[str],
    max_new_tokens: int,
) -> None:
    print("\nPersona comparison on shared prompts:")
    for prompt in prompts:
        print(f"  Prompt: {prompt}")
        for persona_id in order:
            reply = generate_response(model, persona_info, persona_id, prompt, max_new_tokens, temperature=0.0)
            print(f"    {persona_id:12s} -> {reply}")


def quantize_q8_matrix(tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    arr = tensor.detach().cpu().float().numpy().reshape(-1)
    if arr.size % Q_BLOCK != 0:
        raise ValueError("All exported matrices must align to the quantization block size.")
    blocks = arr.reshape(-1, Q_BLOCK)
    max_abs = np.maximum(np.abs(blocks).max(axis=1, keepdims=True), 1e-6)
    scales = (max_abs / 127.0).astype(np.float16).reshape(-1)
    quant = np.clip(np.round(blocks / max_abs * 127.0), -128, 127).astype(np.int8)
    return quant.reshape(-1), scales


def quantize_embedding_table(embedding: torch.Tensor) -> Tuple[np.ndarray, int, float]:
    emb = embedding.detach().cpu().float().numpy()
    max_abs = max(float(np.abs(emb).max()), 1e-6)
    em_scale_x16 = int(round(min(max_abs, 255.0 / 16.0) * 16.0))
    em_scale_x16 = max(em_scale_x16, 1)
    stored_scale = em_scale_x16 / 16.0
    quant = np.clip(np.round(emb / stored_scale * 127.0), -128, 127).astype(np.int8)
    return quant, em_scale_x16, stored_scale


def export_q8(model: NanoGPT, project_dir: str) -> Tuple[str, str]:
    out_path = os.path.join(project_dir, "filesystem", "sophia_weights.bin")
    backup_path = os.path.join(project_dir, "weights", "sophia_weights_v8_persona.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)

    model.eval()
    buf = bytearray()
    emb_q, em_scale_x16, stored_scale = quantize_embedding_table(model.emb.weight)
    buf += struct.pack("<IBHBHBB", MAGIC_LE, N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX, em_scale_x16)
    buf += emb_q.astype(np.int8).tobytes()
    print(f"Embedding export scale: {stored_scale:.4f} (x16={em_scale_x16})")

    for layer_index, block in enumerate(model.blocks):
        tensors = [
            ("wq", block.attn.wq.weight),
            ("wk", block.attn.wk.weight),
            ("wv", block.attn.wv.weight),
            ("wo", block.attn.wo.weight),
            ("wff1", block.wff1.weight),
            ("wff2", block.wff2.weight),
        ]
        packed = [(name, *quantize_q8_matrix(weight)) for name, weight in tensors]
        for _name, quant, _scale in packed:
            buf += quant.astype(np.int8).tobytes()
        for _name, _quant, scale in packed:
            buf += scale.astype(np.float16).tobytes()
        print(f"Layer {layer_index} exported.")

    for path in [out_path, backup_path]:
        with open(path, "wb") as handle:
            handle.write(buf)
    print(f"Exported weights: {out_path}")
    print(f"Backup weights:   {backup_path}")
    print(f"Weight blob size: {len(buf):,} bytes ({len(buf) / 1024.0:.1f} KB)")
    return out_path, backup_path


def save_checkpoint(
    model: NanoGPT,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    best_val: float,
    best_step: int,
    history: List[Dict[str, float]],
    ckpt_dir: str,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"sophia_v8_step_{step:06d}.pt")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "best_val": best_val,
            "best_step": best_step,
            "history": history,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")


def train_model(
    model: NanoGPT,
    train_data: CharDataset,
    val_data: CharDataset,
    cfg: TrainConfig,
    project_dir: str,
) -> Tuple[Dict[str, torch.Tensor], float, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    total_updates = max(1, math.ceil(cfg.steps / cfg.accum_steps))

    def lr_lambda(update_idx: int) -> float:
        if update_idx < cfg.warmup_updates:
            return float(update_idx + 1) / float(max(cfg.warmup_updates, 1))
        progress = (update_idx - cfg.warmup_updates) / float(max(total_updates - cfg.warmup_updates, 1))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad(set_to_none=True)

    best_val = float("inf")
    best_step = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: List[Dict[str, float]] = []
    last_val = float("nan")
    last_grad_norm = 0.0
    t0 = time.time()
    ckpt_dir = os.path.join(project_dir, "weights", "checkpoints_v8")

    for step in range(1, cfg.steps + 1):
        xb, yb = train_data.batch(cfg.batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
        (loss / cfg.accum_steps).backward()

        if step % cfg.accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step == 1 or step % cfg.eval_interval == 0:
            last_val = estimate_loss(model, val_data, cfg.val_batch_size, cfg.val_batches)
            if last_val < best_val:
                best_val = last_val
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if step == 1 or step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / max(elapsed, 1e-6)
            lr = optimizer.param_groups[0]["lr"]
            eta_min = (cfg.steps - step) / max(steps_per_sec, 1e-6) / 60.0
            record = {
                "step": float(step),
                "train_loss": float(loss.item()),
                "val_loss": float(last_val),
                "lr": float(lr),
                "grad_norm": float(last_grad_norm),
            }
            history.append(record)
            print(
                f"step {step:6d}/{cfg.steps}  "
                f"train={loss.item():.4f}  val={last_val:.4f}  "
                f"best={best_val:.4f}@{best_step}  lr={lr:.6f}  "
                f"grad={last_grad_norm:.3f}  eta={eta_min:.1f}m"
            )

        if cfg.save_interval and step % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, cfg, step, best_val, best_step, history, ckpt_dir)

    print(f"Training done in {time.time() - t0:.1f}s. Best val={best_val:.4f} at step {best_step}.")
    return best_state, best_val, history


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train multi-persona Sophia v8 for Legend of Elya N64.")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=TrainConfig.val_batch_size)
    parser.add_argument("--accum-steps", type=int, default=TrainConfig.accum_steps)
    parser.add_argument("--warmup-updates", type=int, default=TrainConfig.warmup_updates)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--min-lr-ratio", type=float, default=TrainConfig.min_lr_ratio)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--train-repeats", type=int, default=TrainConfig.train_repeats)
    parser.add_argument("--eval-repeats", type=int, default=TrainConfig.eval_repeats)
    parser.add_argument("--val-batches", type=int, default=TrainConfig.val_batches)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--save-interval", type=int, default=TrainConfig.save_interval)
    parser.add_argument("--sample-tokens", type=int, default=TrainConfig.sample_tokens)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()
    return TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        accum_steps=args.accum_steps,
        warmup_updates=args.warmup_updates,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        train_repeats=args.train_repeats,
        eval_repeats=args.eval_repeats,
        val_batches=args.val_batches,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_tokens=args.sample_tokens,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    dataset = json.loads(PERSONA_DATASET_JSON)
    validate_dataset(dataset)
    persona_info, order = persona_maps(dataset)

    train_samples, train_by_persona, eval_by_persona = build_formatted_sets(dataset, persona_info, order)
    train_corpus = build_corpus(train_samples, cfg.train_repeats, cfg.seed)
    eval_samples = [sample for persona_id in order for sample in eval_by_persona[persona_id]]
    eval_corpus = build_corpus(eval_samples, cfg.eval_repeats, cfg.seed + 1)

    print(f"Personas: {', '.join(order)}")
    for persona_id in order:
        print(
            f"  {persona_id:12s} prefix={persona_info[persona_id]['prefix']:<18s} "
            f"train={len(train_by_persona[persona_id])} eval={len(eval_by_persona[persona_id])}"
        )

    print(f"Train corpus bytes: {len(train_corpus):,}")
    print(f"Eval corpus bytes:  {len(eval_corpus):,}")
    alphabet_report(train_corpus + eval_corpus)

    train_data = CharDataset(train_corpus)
    val_data = CharDataset(eval_corpus)

    model = NanoGPT().to(device)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {total_params:,}")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    best_state, best_val, history = train_model(model, train_data, val_data, cfg, project_dir)
    model.load_state_dict(best_state)
    model.eval()

    suite_scores = evaluate_suite(model, dataset, persona_info, order)
    print("\nHeld-out persona eval losses:")
    for persona_id in order:
        print(f"  {persona_id:12s} {suite_scores[persona_id]:.4f}")
    print(f"  overall       {sum(suite_scores.values()) / len(suite_scores):.4f}")
    print(f"  best_val      {best_val:.4f}")

    shared_prompts = [
        "What do you think of the king?",
        "How do I survive the night?",
        "Tell me what matters most tonight.",
        "I found a broken sword."
    ]
    print_persona_comparison(model, persona_info, order, shared_prompts, cfg.sample_tokens)

    eval_dump = {
        "config": asdict(cfg),
        "held_out_losses": suite_scores,
        "shared_prompt_outputs": {
            prompt: {
                persona_id: generate_response(model, persona_info, persona_id, prompt, cfg.sample_tokens, 0.0)
                for persona_id in order
            }
            for prompt in shared_prompts
        },
    }
    eval_dump_path = os.path.join(project_dir, "weights", "persona_eval_v8_results.json")
    os.makedirs(os.path.dirname(eval_dump_path), exist_ok=True)
    with open(eval_dump_path, "w", encoding="ascii") as handle:
        json.dump(eval_dump, handle, indent=2)
    print(f"Saved eval summary: {eval_dump_path}")

    export_q8(model, project_dir)

    history_path = os.path.join(project_dir, "weights", "sophia_v8_history.json")
    with open(history_path, "w", encoding="ascii") as handle:
        json.dump({"config": asdict(cfg), "history": history}, handle, indent=2)
    print(f"Saved training history: {history_path}")
    print("Done.")


if __name__ == "__main__":
    main()
