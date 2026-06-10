"""
ExSL Training Script — Full Finetuning
=======================================
Trains ExSL (Extractive Schema Linking) on Spider Train (7 000 examples).
Uses DeepSeek Coder 6.7B with full finetuning (no quantization, no LoRA)
and a binary relevance head on top of the « / » marker hidden states.

Memory requirements on H100 (80 GB VRAM):
    Model weights  (bf16) ~13 GB
    Gradients      (bf16) ~13 GB
    AdamW optimizer (fp32) ~54 GB
    → Total ~80 GB — gradient checkpointing keeps it within budget.

Saves:
    exsl_full/        — full model weights (HuggingFace format)
    exsl_head_full.pt — state_dict of the w_relevance linear head
"""

import argparse
import os
import re

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from model import ExSLModel
from spider_data import get_spider_train

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ExSL Full Finetuning")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing")
    return parser.parse_args()

args = parse_args()

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME = args.model
HIDDEN_SIZE = 4096
NUM_EPOCHS = args.epochs
LR = args.lr
GRAD_ACCUM_STEPS = args.grad_accum
MAX_TOKENS = args.max_tokens
# Output directory — override with OUTPUT_DIR env var (e.g. /app/output in Docker)
_OUT = os.environ.get("OUTPUT_DIR", ".")
SAVE_MODEL_PATH = os.path.join(_OUT, "exsl_full")
SAVE_HEAD_PATH = os.path.join(_OUT, "exsl_head_full.pt")

USE_GRADIENT_CHECKPOINTING = not args.no_grad_ckpt

# H100 natively supports bf16; this falls back to fp16 on older GPUs.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ---------------------------------------------------------------------------
# Data helpers  (identical to train.py)
# ---------------------------------------------------------------------------

def _get_candidate_labels(prompt: str, gold_schema: dict) -> list[float]:
    candidates = re.findall(r'«\s+(\S+)\s+(\S+)\s*»', prompt)
    labels = []
    for table, col in candidates:
        gold_cols = [c.lower() for c in gold_schema.get(table.lower(), [])]
        labels.append(1.0 if col.lower() in gold_cols else 0.0)
    return labels


def build_training_samples(raw_data: list[dict]) -> list[dict]:
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

    print(f"Loading {MODEL_NAME} in {DTYPE} (full weights, no quantization) …")
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        output_hidden_states=True,
    )

    if USE_GRADIENT_CHECKPOINTING:
        base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    model = ExSLModel(base_model, hidden_size=HIDDEN_SIZE)
    device = next(base_model.parameters()).device
    model.w_relevance = model.w_relevance.to(device).to(torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Marker position helper
# ---------------------------------------------------------------------------

def find_marker_positions(tokens: list[str]) -> tuple[list[int], list[int]]:
    open_pos = [i for i, t in enumerate(tokens) if "«" in t]
    close_pos = [i for i, t in enumerate(tokens) if "»" in t]
    return open_pos, close_pos


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    print("=== ExSL Training — Full Finetuning ===")
    print(f"Model:   {MODEL_NAME}")
    print(f"Epochs:  {NUM_EPOCHS}  |  LR: {LR}  |  Grad-Accum: {GRAD_ACCUM_STEPS}")
    print(f"dtype:   {DTYPE}")

    model, tokenizer = setup_model_and_tokenizer()

    print("\nLoading Spider training data …")
    raw_data = get_spider_train(tokenizer, MAX_TOKENS)
    samples = build_training_samples(raw_data)
    print(f"Training samples (prompt chunks): {len(samples)}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    device = next(model.base.parameters()).device

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        logged_steps = 0

        pbar = tqdm(samples, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        for step, sample in enumerate(pbar):
            prompt = sample["prompt"]
            labels = torch.tensor(sample["labels"], dtype=torch.float32).to(device)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_TOKENS,
                truncation=True,
            ).to(device)

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            open_pos, close_pos = find_marker_positions(tokens)

            n = min(len(open_pos), len(close_pos), len(labels))
            if n == 0:
                continue

            logits = model(
                enc["input_ids"],
                enc["attention_mask"],
                open_pos[:n],
                close_pos[:n],
            )
            loss = loss_fn(logits, labels[:n]) / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # Gradient clipping helps stabilise full finetuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                logged_steps += 1
                avg = total_loss / (logged_steps * GRAD_ACCUM_STEPS)
                pbar.set_postfix(loss=f"{avg:.4f}")

        # Flush remaining gradients at epoch end
        if (len(samples) % GRAD_ACCUM_STEPS) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving full model → {SAVE_MODEL_PATH}/")
    # Save the base model (without the ExSLModel wrapper) in HF format
    model.base.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)
    print(f"Saving relevance head → {SAVE_HEAD_PATH}")
    torch.save(model.w_relevance.state_dict(), SAVE_HEAD_PATH)
    print("Done.")


if __name__ == "__main__":
    train()
