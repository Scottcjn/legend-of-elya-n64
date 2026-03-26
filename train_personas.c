#include "nano_gpt.h"
#include <libdragon.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define MAX_PERSONA_NAME 32

typedef struct {
    const char *name;
    const char *identity_pairs[10];
    const char *qa_pairs[10];
} Persona;

void train_personas(Persona personas[], size_t persona_count) {
    for (size_t i = 0; i < persona_count; i++) {
        const Persona *p = &personas[i];
        printf("Training persona: %s\n", p->name);

        for (int j = 0; j < 10 && p->identity_pairs[j]; j++) {
            printf("Identity Pair %d: %s\n", j+1, p->identity_pairs[j]);
        }

        for (int k = 0; k < 10 && p->qa_pairs[k]; k++) {
            printf("QA Pair %d: %s\n", k+1, p->qa_pairs[k]);
        }

        // Placeholder: Save logic for compiled SGAIWeights.
        printf("[INFO] Save weights for persona: %s\n", p->name);
    }
}

int main() {
    Persona personas[] = {
        {
            "Sophia", 
            {"Who are you?: I am Sophia.", "What is your name?: Sophia."},
            {"Tell me about this dungeon.: Dark halls hide secrets.", "What is your wisdom?: Protect knowledge."}
        },
        {
            "Guard", 
            {"Who are you?: I am the Guard.", "What is your duty?: Protect."},
            {"Who built gates?: Elyan crafted them."}
        }
    };
    
    train_personas(personas, sizeof(personas)/sizeof(personas[0]));
    return 0;
}