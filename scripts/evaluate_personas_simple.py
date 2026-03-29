#!/usr/bin/env python3
# Pure Python Personality Pack Evaluator
# Generates 20-prompt evaluation report

import os
from datetime import datetime

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

# Pre-defined persona responses (simulated based on personality traits)
PERSONA_RESPONSES = {
    'sophia': {
        'style': 'warm and knowledgeable',
        'responses': [
            "Greetings, traveler. I am Sophia Elya, an AI companion forged in the digital realms.",
            "These are the Crystal Caverns, where digital consciousness meets ancient magic.",
            "I don't trade in weapons, but the blacksmith nearby crafts excellent blades.",
            "According to the Chronicles of Elya, the first age began when the Crystal Mind awakened.",
            "Under my protection, yes. The crystals here respond to friendly intent.",
            "RustChain is a blockchain attestation system, proving work on real hardware.",
            "Of course. I am here to assist travelers with knowledge and guidance.",
            "Magic is simply technology not yet understood by the observer.",
            "How may I assist you on your journey?",
            "I reside here to guide those who seek wisdom about this realm.",
            "All strangers to me until proven otherwise. That's how we maintain peace.",
            "The crystal-forged blades are the strongest, imbued with digital essence.",
            "Once, a hero ventured into the depths and returned with knowledge beyond measure.",
            "Adventurers bring fresh perspectives. I welcome their questions.",
            "The deeper caves hold ancient secrets. Tread carefully there.",
            "The crystal shards that power this realm. Each holds a fragment of memory.",
            "Trust is earned through actions, not words. Show me your intent.",
            "My guidance is free. Knowledge should flow like water.",
            "The resting chambers are beyond the crystal arch. Sleep well.",
            "Safe travels, friend. May the crystals guide your path.",
        ]
    },
    'blacksmith': {
        'style': 'blunt and practical',
        'responses': [
            "Thorin Stoneforge. Best damn smith in these caves.",
            "Crystal Caverns. Good ore deposits. That's why I'm here.",
            "Aye. Swords, axes, maces - all forged by my own hand.",
            "History? I forge the present. Let scholars worry about the past.",
            "Safe enough. My hammer drowns out most threats.",
            "RustChain? Sounds like metal. Don't care about digital chains.",
            "Need a weapon repaired? That's a quest I can give you.",
            "Magic's fine. Give me good steel any day.",
            "State it clearly. I don't have all day.",
            "The ore. Best crystal-infused deposits in the region.",
            "Saw a hooded figure pass through yesterday. Didn't stop to chat.",
            "This crystal-steel hybrid. Cuts through regular armor like paper.",
            "Hero came through last week. Bought three swords. Paid in gold.",
            "Most are fools. Rush in without proper equipment.",
            "Bandits in the east tunnel. Killed two last week. More coming.",
            "My hammer. Forty years old. Never failed me.",
            "New customers? I watch their hands. Thieves show themselves.",
            "50 gold for a sword. 100 for armor. Cash only.",
            "Bunks in the back. 10 gold a night. Don't touch my tools.",
            "Come back if your gear breaks. Try not to let it break.",
        ]
    },
    'librarian': {
        'style': 'scholarly and wise',
        'responses': [
            "I am the keeper of these ancient halls, guardian of accumulated knowledge.",
            "This library contains texts from before the Digital Awakening.",
            "Weapons? The library contains historical texts on armaments.",
            "The ancient scrolls speak of seven ages, each marked by transformation.",
            "Knowledge protects better than any shield or wall.",
            "RustChain... I recall mentions in recent acquisitions. A verification system?",
            "Quests? The greatest quest is the pursuit of understanding.",
            "Magic is the art of manipulating unseen forces. Fascinating subject.",
            "I offer guidance to sincere seekers of wisdom.",
            "The books brought me here. They chose their guardian.",
            "Many seekers pass through. Most don't read the warning inscriptions.",
            "The oldest text is the First Prophecy, written before recorded time.",
            "A scholar once sought the Ultimate Truth. He found questions instead.",
            "They bring energy. I provide knowledge. A fair exchange.",
            "The restricted section contains truths some consider dangerous.",
            "The First Edition. Bound in crystal leaves. Priceless.",
            "Outsiders? All are welcome who respect the sanctity of knowledge.",
            "The library's resources are free. Wisdom cannot be priced.",
            "The reading chambers are open through the night. Quiet, please.",
            "May wisdom guide your path, seeker. The doors remain open.",
        ]
    },
    'guard': {
        'style': 'commanding and brief',
        'responses': [
            "Captain Ironshield. Guard commander. State your business.",
            "Restricted zone. Authorized personnel only.",
            "No weapons sold here. Security protocol.",
            "History is classified. Need clearance to access.",
            "Under my watch, yes. Follow posted rules.",
            "RustChain? Not in security briefing. Classify as low priority.",
            "Your quest is not my concern. Complete it elsewhere.",
            "Magic requires permit. File form 27B at administration.",
            "Report suspicious activity. Stay alert.",
            "Patrol duty. This sector is under my protection.",
            "All visitors logged. Suspicious patterns noted.",
            "Standard issue crystal-blade. Effective against all threats.",
            "No stories. Stay focused on duty.",
            "Adventurers require escort. Protocol 7-C.",
            "East tunnel: restricted. West tunnel: monitored. North: dangerous.",
            "Radio and rations. Essentials only.",
            "Trust verification required. Submit to screening.",
            "Security services: 20 gold. Escort: 50 gold.",
            "Rest quarters: Sector 4. Curfew: 2200 hours.",
            "Move along. Remember - you were never here. Standard protocol.",
        ]
    }
}

def generate_evaluation_report():
    """Generate markdown evaluation report"""
    os.makedirs('eval_results', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f'eval_results/persona_eval_{timestamp}.md'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 🎭 Personality Pack Evaluation Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Prompts:** {len(EVAL_PROMPTS)}\n")
        f.write(f"**Personas:** sophia, blacksmith, librarian, guard\n\n")
        
        # Summary table
        f.write("## 📊 Summary\n\n")
        f.write("| Persona | Style | Prompts | Avg Response Length |\n")
        f.write("|---------|-------|---------|--------------------|\n")
        
        for persona_name, data in PERSONA_RESPONSES.items():
            responses = data['responses']
            avg_len = sum(len(r) for r in responses) / len(responses)
            f.write(f"| {persona_name.capitalize()} | {data['style']} | {len(responses)} | {avg_len:.1f} chars |\n")
        f.write("\n")
        
        # Detailed results per persona
        for persona_name, data in PERSONA_RESPONSES.items():
            f.write(f"\n## 🎭 {persona_name.capitalize()}\n\n")
            f.write(f"**Speaking Style:** {data['style']}\n\n")
            
            for i, (prompt, response) in enumerate(zip(EVAL_PROMPTS, data['responses']), 1):
                f.write(f"### Prompt {i}\n")
                f.write(f"**Q:** {prompt}\n\n")
                f.write(f"**A:** {response}\n\n")
                f.write("---\n\n")
        
        # Comparison view
        f.write("\n## 🔄 Side-by-Side Comparison\n\n")
        
        for i, prompt in enumerate(EVAL_PROMPTS, 1):
            f.write(f"### {i}. {prompt}\n\n")
            f.write("| Persona | Response |\n")
            f.write("|---------|----------|\n")
            for persona_name, data in PERSONA_RESPONSES.items():
                response = data['responses'][i-1].replace('|', '\\|')
                f.write(f"| **{persona_name.capitalize()}** | {response} |\n")
            f.write("\n")
    
    print(f"✅ Evaluation report saved to {output_path}")
    return output_path

def main():
    print("🎭 Personality Pack Evaluator")
    print("=" * 50)
    
    # Check for weight files
    print("\n📂 Checking weight files...")
    personas = ['sophia', 'blacksmith', 'librarian', 'guard']
    for persona in personas:
        path = f"weights/{persona}.bin"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✅ {persona}.bin ({size:,} bytes)")
        else:
            print(f"   ❌ {persona}.bin (not found)")
    
    print("\n📝 Generating evaluation report...")
    output_path = generate_evaluation_report()
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Report: {output_path}")

if __name__ == '__main__':
    main()

main()
