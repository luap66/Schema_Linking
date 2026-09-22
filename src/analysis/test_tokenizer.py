from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
tokens = t.convert_ids_to_tokens(t.encode("« test column»"))
print("Tokens:", tokens)
print("« positions:", [i for i, tok in enumerate(tokens) if "«" in tok])
print("» positions:", [i for i, tok in enumerate(tokens) if "»" in tok])
