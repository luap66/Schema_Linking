"""
ExSL Evaluation Script
======================
Evaluates a trained ExSL model on:
    • Spider Dev  (1 034 examples) — reproduces paper results
    • Spider-Ent  (  602 examples) — zero-shot transfer to enterprise schemas

Metrics reported: micro-averaged Precision, Recall, and F6 (β=6, recall-heavy).

Usage:
    python evaluate.py

Expects trained artefacts:
    exsl_lora/      — LoRA adapter (PEFT format)
    exsl_head.pt    — w_relevance linear head state_dict
"""

import re

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from src.model import ExSLModel
from src.data_loaders.spider_data import get_spider_schema_ddl_and_candidates, get_spider_val, spider_val
from src.data_loaders.spider_ent_data import (
    get_ent_gold_schema,
    get_spider_ent_data,
    schema_with_parsed_candidates as ent_schema_with_parsed_candidates,
    spider_ent,
)
from src.utils import add_missing_bridge_tables

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
LORA_PATH = "exsl_lora"
HEAD_PATH = "exsl_head.pt"
HIDDEN_SIZE = 4096
MAX_TOKENS = 1024
LOGIT_THRESHOLD = -3.0   # sigmoid threshold for positive prediction


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(tokenizer):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        output_hidden_states=True,
        device_map={"": "cuda:0"},
    )
    base_model = PeftModel.from_pretrained(base_model, LORA_PATH)

    model = ExSLModel(base_model, hidden_size=HIDDEN_SIZE)
    model.w_relevance.load_state_dict(
        torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
    )
    model.w_relevance = model.w_relevance.to("cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _find_marker_positions(tokens: list[str]) -> tuple[list[int], list[int]]:
    open_pos = [i for i, t in enumerate(tokens) if "«" in t]
    close_pos = [i for i, t in enumerate(tokens) if "»" in t]
    return open_pos, close_pos


def predict_schema(model, tokenizer, item: dict) -> dict[str, list[str]]:
    """Run inference on all prompt chunks of one data item.

    Merges predictions across chunks back into a single {table: [columns]} dict.
    A candidate column is predicted as relevant when sigmoid(logit) >= THRESHOLD.
    """
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

        # Extract candidates from the original (untruncated) prompt text
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
            
        logits_list = logits.cpu().tolist()
        
        for (table, col), logit in zip(candidates[:n], logits_list):
            if logit >= LOGIT_THRESHOLD:
                t_low = table.lower()
                pred_schema.setdefault(t_low, []).append(col.lower())

    return pred_schema


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[dict], gold_schemas: list[dict]
) -> dict:
    """Micro-averaged Precision, Recall, F6 over (table, column) pairs.

    F6 = (1 + 6²) * P * R / (6² * P + R)
    Heavily weights recall — missing a required column is penalised more than
    including a spurious one.
    """
    total_tp = 0
    total_pred = 0
    total_gold = 0

    for pred, gold in zip(predictions, gold_schemas):
        pred_cols = {
            (t.lower(), c.lower())
            for t, cols in pred.items()
            for c in cols
            if t and c
        }
        gold_cols = {
            (t.lower(), c.lower())
            for t, cols in gold.items()
            for c in (cols or [])
            if t and c
        }
        total_tp += len(pred_cols & gold_cols)
        total_pred += len(pred_cols)
        total_gold += len(gold_cols)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gold if total_gold > 0 else 0.0
    beta_sq = 36  # β = 6
    denom = beta_sq * precision + recall
    f6 = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f6": f6,
        "tp": total_tp,
        "pred": total_pred,
        "gold": total_gold,
    }


def _print_metrics(name: str, m: dict) -> None:
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F6        : {m['f6']:.4f}")
    print(f"  (TP={m['tp']}, Pred={m['pred']}, Gold={m['gold']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(tokenizer)

    # ------------------------------------------------------------------
    # Spider Dev
    # ------------------------------------------------------------------
    print("\nLoading Spider Dev …")
    spider_dev_data = get_spider_val(tokenizer, MAX_TOKENS)
    spider_schema_ddls_and_candidates = get_spider_schema_ddl_and_candidates()

    preds_dev_no_bridge, preds_dev_bridged, golds_dev = [], [], []
    for item, question in tqdm(
        zip(spider_dev_data, spider_val), total=len(spider_val), desc="Spider Dev", dynamic_ncols=True
    ):
        pred_schema = predict_schema(model, tokenizer, item)
        db_tables = spider_schema_ddls_and_candidates[question["db_id"]]
        preds_dev_no_bridge.append(pred_schema)
        preds_dev_bridged.append(add_missing_bridge_tables(pred_schema, db_tables))
        golds_dev.append(item["gold_schema"])

    # ------------------------------------------------------------------
    # Spider-Ent
    # ------------------------------------------------------------------
    print("\nLoading Spider-Ent …")
    spider_ent_data = get_spider_ent_data(tokenizer, MAX_TOKENS)

    preds_ent_no_bridge, preds_ent_bridged, golds_ent = [], [], []
    for item, question in tqdm(
        zip(spider_ent_data, spider_ent),
        total=len(spider_ent),
        desc="Spider-Ent",
        dynamic_ncols=True,
    ):
        pred_schema = predict_schema(model, tokenizer, item)
        db_tables = ent_schema_with_parsed_candidates[question["data_asset"]]
        preds_ent_no_bridge.append(pred_schema)
        preds_ent_bridged.append(add_missing_bridge_tables(pred_schema, db_tables))
        # Gold schema must be translated to enterprise column names
        golds_ent.append(get_ent_gold_schema(question))

    # ------------------------------------------------------------------
    # Results, reported separately without vs. with bridge-table completion
    # ------------------------------------------------------------------
    print(f"\n{'#'*40}\n#  Ohne Bridge-Tables\n{'#'*40}")
    _print_metrics("Spider Dev", compute_metrics(preds_dev_no_bridge, golds_dev))
    _print_metrics("Spider-Ent", compute_metrics(preds_ent_no_bridge, golds_ent))

    print(f"\n{'#'*40}\n#  Mit Bridge-Tables\n{'#'*40}")
    _print_metrics("Spider Dev", compute_metrics(preds_dev_bridged, golds_dev))
    _print_metrics("Spider-Ent", compute_metrics(preds_ent_bridged, golds_ent))


if __name__ == "__main__":
    evaluate()
