@qai.logic(name="complete_code", requires=["model", "tokenizer"], max_seconds=10.0,
            description="Complete a code prompt using the transformer")
def complete_code(prompt: str) -> str:
    """Complete a code prompt."""
    return model.generate(prompt, max_new_tokens=20, tokenizer=tokenizer,
                           temperature=0.0, use_cache=True)

# closure deps (pass these to load_logic_source(namespace=...)): ['model', 'tokenizer']
# required namespace names: ['model', 'tokenizer']
