"""
ExSL Evaluation Script — Full Finetuning
=========================================
Evaluates a fully finetuned ExSL model (no LoRA, no quantization).

Usage:
    python evaluate_full.py

Expects trained artefacts:
    exsl_full/          — full model weights (HuggingFace format)
    exsl_head_full.pt   — w_relevance linear head state_dict
"""

import re

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from model import ExSLModel
from spider_data import get_spider_val
from spider_ent_data import get_ent_gold_schema_neu, get_spider_ent_data, spider_ent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "exsl_full"
HEAD_PATH = "exsl_head_full.pt"
HIDDEN_SIZE = 4096
MAX_TOKENS = 1024
LOGIT_THRESHOLD = -3.0

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(tokenizer):
    base_model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map={"": "cuda:0"},
        output_hidden_states=True,
    )
    model = ExSLModel(base_model, hidden_size=HIDDEN_SIZE)
    model.w_relevance.load_state_dict(
        torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
    )
    model.w_relevance = model.w_relevance.to("cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers  (identical to evaluate.py)
# ---------------------------------------------------------------------------

def _find_marker_positions(tokens: list[str]) -> tuple[list[int], list[int]]:
    open_pos = [i for i, t in enumerate(tokens) if "«" in t]
    close_pos = [i for i, t in enumerate(tokens) if "»" in t]
    return open_pos, close_pos


def predict_schema(model, tokenizer, item: dict) -> dict[str, list[str]]:
    pred_schema: dict[str, list[str]] = {}

    for prompt in item["input"]:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
        ).to("cuda")

        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        open_pos, close_pos = _find_marker_positions(tokens)

        candidates = re.findall(r"«\s+(\S+)\s+(\S+)\s*»", prompt)
        n = min(len(open_pos), len(close_pos), len(candidates))
        if n == 0:
            continue

        with torch.no_grad():
            logits = model(
                enc["input_ids"],
                enc["attention_mask"],
                open_pos[:n],
                close_pos[:n],
            )

        for (table, col), logit in zip(candidates[:n], logits.cpu().tolist()):
            if logit >= LOGIT_THRESHOLD:
                pred_schema.setdefault(table.lower(), []).append(col.lower())

    return pred_schema


# ---------------------------------------------------------------------------
# Metrics  (identical to evaluate.py)
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list[dict], gold_schemas: list[dict]) -> dict:
    total_tp = total_pred = total_gold = 0
    for pred, gold in zip(predictions, gold_schemas):
        pred_cols = {(t.lower(), c.lower()) for t, cols in pred.items() for c in cols if t and c}
        gold_cols = {(t.lower(), c.lower()) for t, cols in gold.items() for c in (cols or []) if t and c}
        total_tp += len(pred_cols & gold_cols)
        total_pred += len(pred_cols)
        total_gold += len(gold_cols)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gold if total_gold > 0 else 0.0
    beta_sq = 36
    denom = beta_sq * precision + recall
    f6 = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "f6": f6,
            "tp": total_tp, "pred": total_pred, "gold": total_gold}


def _print_metrics(name: str, m: dict) -> None:
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F6        : {m['f6']:.4f}")
    print(f"  (TP={m['tp']}, Pred={m['pred']}, Gold={m['gold']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_model(tokenizer)

    print("\nLoading Spider Dev …")
    spider_dev_data = get_spider_val(tokenizer, MAX_TOKENS)
    preds_dev, golds_dev = [], []
    for item in tqdm(spider_dev_data, desc="Spider Dev", dynamic_ncols=True):
        preds_dev.append(predict_schema(model, tokenizer, item))
        golds_dev.append(item["gold_schema"])
    _print_metrics("Spider Dev", compute_metrics(preds_dev, golds_dev))

    print("\nLoading Spider-Ent …")
    spider_ent_data = get_spider_ent_data(tokenizer, MAX_TOKENS)
    preds_ent, golds_ent = [], []
    for item, question in tqdm(zip(spider_ent_data, spider_ent), total=len(spider_ent), desc="Spider-Ent", dynamic_ncols=True):
        preds_ent.append(predict_schema(model, tokenizer, item))
        golds_ent.append(get_ent_gold_schema_neu(question))
    _print_metrics("Spider-Ent", compute_metrics(preds_ent, golds_ent))


if __name__ == "__main__":
    evaluate()
