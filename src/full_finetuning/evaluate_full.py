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

import argparse
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ExSL Full Finetuning — Evaluation")
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=-3.0, help="Logit threshold for positive prediction")
    parser.add_argument("--dataset", choices=["spider", "spider_ent", "both"], default="both")
    # Defaults to the repo root, resolved relative to this file so it does not
    # depend on the working directory the script is launched from.
    repo_root = Path(__file__).resolve().parents[2]
    _out = os.environ.get("OUTPUT_DIR", str(repo_root))
    parser.add_argument("--model_path", default=os.path.join(_out, "exsl_full"))
    parser.add_argument("--head_path", default=os.path.join(_out, "exsl_head_full.pt"))
    return parser.parse_args()

args = parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = args.model_path
HEAD_PATH = args.head_path
HIDDEN_SIZE = 4096
MAX_TOKENS = args.max_tokens
LOGIT_THRESHOLD = args.threshold

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(tokenizer):
    base_model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
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
    print(f"Max Tokens : {MAX_TOKENS}")
    print(f"Args: {args}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_model(tokenizer)

    if args.dataset in ("spider", "both"):
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

    if args.dataset in ("spider_ent", "both"):
        print("\nLoading Spider-Ent …")
        spider_ent_data = get_spider_ent_data(tokenizer, MAX_TOKENS)
        preds_ent_no_bridge, preds_ent_bridged, golds_ent = [], [], []
        for item, question in tqdm(zip(spider_ent_data, spider_ent), total=len(spider_ent), desc="Spider-Ent", dynamic_ncols=True):
            pred_schema = predict_schema(model, tokenizer, item)
            db_tables = ent_schema_with_parsed_candidates[question["data_asset"]]
            preds_ent_no_bridge.append(pred_schema)
            preds_ent_bridged.append(add_missing_bridge_tables(pred_schema, db_tables))
            golds_ent.append(get_ent_gold_schema(question))

    print(f"\n{'#'*40}\n#  Ohne Bridge-Tables\n{'#'*40}")
    if args.dataset in ("spider", "both"):
        _print_metrics("Spider Dev", compute_metrics(preds_dev_no_bridge, golds_dev))
    if args.dataset in ("spider_ent", "both"):
        _print_metrics("Spider-Ent", compute_metrics(preds_ent_no_bridge, golds_ent))

    print(f"\n{'#'*40}\n#  Mit Bridge-Tables\n{'#'*40}")
    if args.dataset in ("spider", "both"):
        _print_metrics("Spider Dev", compute_metrics(preds_dev_bridged, golds_dev))
    if args.dataset in ("spider_ent", "both"):
        _print_metrics("Spider-Ent", compute_metrics(preds_ent_bridged, golds_ent))


if __name__ == "__main__":
    evaluate()
