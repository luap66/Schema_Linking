"""
ExSL Training Script
====================
Trains ExSL (Extractive Schema Linking) on Spider Train (7 000 examples).
Uses DeepSeek Coder 6.7B with QLoRA (4-bit NF4 + LoRA on q_proj/v_proj) and a
binary relevance head on top of the « / » marker hidden states.

Saves:
    exsl_lora/      — LoRA adapter weights (PEFT format)
    exsl_head.pt    — state_dict of the w_relevance linear head
"""

import re

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from model import ExSLModel
from spider_data import get_spider_train
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
HIDDEN_SIZE = 4096 # Anzahl der Vektordimensionen die einen Token repräsentieren in dem LLM
NUM_EPOCHS = 2
LR = 5e-6
GRAD_ACCUM_STEPS = 16   # effective batch size 16 - VRAM zu klein für einen Chunk deswegen wird nur ein Chunk zur Zeit verarbeitet, der Gradient jedoch für 16 Chunks berechnet und daraus weight updates gemacht
MAX_TOKENS = 1024

# Bei Lora werden statt der echten Gewichte des LLMs eigene Matrizen an die Gewichte der Target_Modules gehangen. Diese sind kleiner und somit mit weniger Rechenaufwand zu updaten.
LORA_RANK = 16 # Größe der Matrizen, die an die Gewichte gehangen werden
LORA_ALPHA = 32 # Gewichtung LoRa-Beitrag
LORA_DROPOUT = 0.05
# Nur die Query- und Value-Projektionen, weil für Finetuning am effektivsten
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
SAVE_LORA_PATH = "exsl_lora"
SAVE_HEAD_PATH = "exsl_head.pt"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_candidate_labels(prompt: str, gold_schema: dict) -> list[float]:
    """Extract « table col» candidates from a prompt and assign binary labels.

    A candidate (table, col) receives label 1.0 if col appears in
    gold_schema[table], otherwise 0.0.
    """
    candidates = re.findall(r'«\s+(\S+)\s+(\S+)\s*»', prompt)
    labels = []
    for table, col in candidates:
        gold_cols = [c.lower() for c in gold_schema.get(table.lower(), [])]
        labels.append(1.0 if col.lower() in gold_cols else 0.0)
    return labels


def build_training_samples(raw_data: list[dict]) -> list[dict]:
    """Expand dataset items into individual training samples.

    Each item from get_spider_train() may contain several prompt chunks
    (when the schema was split to fit the context window).  Every chunk
    becomes its own sample with the appropriate binary labels.

    Returns:
        List of {"prompt": str, "labels": list[float]}
    """
    samples = []
    for item in raw_data:
        gold = item["gold_schema"]
        for prompt in item["input"]:
            labels = _get_candidate_labels(prompt, gold)
            if labels:
                samples.append({"prompt": prompt, "labels": labels})
    return samples


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # Speichern der Gewichte in 4-bit 
        bnb_4bit_compute_dtype=torch.float16, # Berechnungen werden mit 16 bit gemacht für mehr Genauigkeit
        bnb_4bit_use_double_quant=True, # Quantisiert die Quantisierungskonstanten (Quantisierungskonstanten können aus 4 bit Gewicht wieder sinnvoll in 16 bit Gewicht umwandeln)
        bnb_4bit_quant_type="nf4", # Normalverteilte quantisierte Gewichte
    )
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        output_hidden_states=True # Gibt Hidden States aller Layer zurück – wir brauchen den letzten für den Klassifikationskopf
    )
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", # Bias-Parameter werden nicht trainiert
    )
    base_model = get_peft_model(base_model, lora_config) # Erweitert die relevanten Schichten des Basemodels um die trainierbaren LoRa-Matrizen
    base_model.print_trainable_parameters()

    model = ExSLModel(base_model, hidden_size=HIDDEN_SIZE) # Verbindet das quantisierte LLM mit dem Klassifikationskopf zu einem gemeinsamen Modell
    # Lineare Layer auf die gleiche Hardware legen, wie das LLM
    device = next(base_model.parameters()).device
    model.w_relevance = model.w_relevance.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Marker position helper
# ---------------------------------------------------------------------------

def find_marker_positions(tokens: list[str]) -> tuple[list[int], list[int]]:
    """Return token indices of « and » markers.

    DeepSeek Coder encodes the UTF-8 bytes of « and » as individual tokens.
    Both the single-token form ('Â«') and the multi-byte split form are handled
    by checking for substring membership.
    """
    open_pos = [i for i, t in enumerate(tokens) if "«" in t]
    close_pos = [i for i, t in enumerate(tokens) if "»" in t]
    return open_pos, close_pos


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    print("=== ExSL Training ===")
    print(f"Model:   {MODEL_NAME}")
    print(f"Epochs:  {NUM_EPOCHS}  |  LR: {LR}  |  Grad-Accum: {GRAD_ACCUM_STEPS}")

    model, tokenizer = setup_model_and_tokenizer()

    print("\nLoading Spider training data …")
    raw_data = get_spider_train(tokenizer, MAX_TOKENS)
    samples = build_training_samples(raw_data)
    print(f"Training samples (prompt chunks): {len(samples)}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        model.train() # Aktiviert Trainingsmodus
        optimizer.zero_grad() # Setzt alle Gradienten auf null

        total_loss = 0.0
        logged_steps = 0

        pbar = tqdm(samples, desc=f"Epoch {epoch + 1}", dynamic_ncols=True) # Initialisierung des visuellen Forschrittsbalkens
        for step, sample in enumerate(pbar):
            prompt = sample["prompt"]
            labels = torch.tensor(sample["labels"], dtype=torch.float32).to("cuda") # weil BCEWithLogitsLoss float32 erwartet

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_TOKENS,
                truncation=True,
            ).to("cuda")

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            open_pos, close_pos = find_marker_positions(tokens)

            # Candidates that were truncated away have no markers — skip them
            n = min(len(open_pos), len(close_pos), len(labels))
            if n == 0:
                continue

            logits = model(
                enc["input_ids"], # Token Ids des Propmts
                enc["attention_mask"], # Da die Prompts unterschiedlich viele Tokens haben, aber die Länge für alle Prompts gleich lang sein müssen, werden Padding-Tokens als Lückenfüller eingefügt. 
                                        # Die Attention-Mask zeigt an was tatsächliche Tokens und was Padding-Tokens sind.
                open_pos[:n],
                close_pos[:n],
            )
            # Da nur ein Sample zur Zeit verabeitet werden kann und ein Sample nicht zu stark ins Gewicht fallen soll, wird durch die "simulierte Batchgröße" geteilt.
            loss = loss_fn(logits, labels[:n]) / GRAD_ACCUM_STEPS
            # Berechnung der Gradienten für jedes trainierbare Gewicht. Die Gradienten werden innerhalb des "simulierten Batches" akkumuliert.
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step() # Anpassung der Gewichte mithilfe der Gradienten
                optimizer.zero_grad() # Zurücksetzen der Gradienten
                logged_steps += 1
                avg = total_loss / (logged_steps * GRAD_ACCUM_STEPS)
                pbar.set_postfix(loss=f"{avg:.4f}")

        # Flush remaining gradients at epoch end
        if (len(samples) % GRAD_ACCUM_STEPS) != 0:
            optimizer.step()
            optimizer.zero_grad()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving LoRA adapter → {SAVE_LORA_PATH}/")
    model.base.save_pretrained(SAVE_LORA_PATH)
    print(f"Saving relevance head → {SAVE_HEAD_PATH}")
    torch.save(model.w_relevance.state_dict(), SAVE_HEAD_PATH)
    print("Done.")


if __name__ == "__main__":
    train()
