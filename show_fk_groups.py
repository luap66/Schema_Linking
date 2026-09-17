"""Zeigt fuer eine Spider-DB (Standard: baseball_1)
1) die FK-connected-components (group_tables_by_fk_component)
2) die tatsaechlichen Chunks, die create_schema_linker_input mit einem echten
   Tokenizer + Context-Window erzeugt, inkl. welche Tabellen pro Chunk enthalten
   sind und welche referenzierten FK-Tabellen in einem ANDEREN Chunk landen.
"""

import re

from transformers import AutoTokenizer

from spider_data import get_spider_schema_ddl_and_candidates, spider_train
from utils import group_tables_by_fk_component, create_schema_linker_input, _extract_fk_targets

DB_ID = "baseball_1"
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
MAX_TOKENS = 1024  # wie in train.py

_TABLE_NAME_IN_CHUNK = re.compile(r'CREATE TABLE\s+`?(\S+?)`?\s*\(')


def tables_in_chunk(chunk: str) -> list[str]:
    return [name.lower() for name in _TABLE_NAME_IN_CHUNK.findall(chunk)]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    db_tables = get_spider_schema_ddl_and_candidates()[DB_ID]
    name_to_table = {t["candidates"]["table"].lower(): t for t in db_tables}

    # --- 1) FK-Gruppen ---
    groups = group_tables_by_fk_component(db_tables)
    print(f"{DB_ID}: {len(db_tables)} Tabellen, {len(groups)} FK-Gruppe(n)\n")
    for i, group in enumerate(groups, start=1):
        names = [t["candidates"]["table"] for t in group]
        print(f"Gruppe {i} ({len(names)} Tabellen): {names}")

    # --- 2) Tatsaechliches Chunking mit Tokenizer ---
    question_text = next(q["question"] for q in spider_train if q["db_id"] == DB_ID)
    print(f"\nBeispiel-Frage: {question_text!r}")

    chunks = create_schema_linker_input(db_tables, question_text, MAX_TOKENS, DB_ID, tokenizer)
    print(f"\n=> {len(chunks)} Chunk(s) bei max_tokens={MAX_TOKENS}\n")

    chunk_tables = [tables_in_chunk(c) for c in chunks]

    for i, names in enumerate(chunk_tables, start=1):
        print(f"Chunk {i} ({len(names)} Tabellen): {names}")

    # --- 3) FK-Referenzen, die in einem ANDEREN Chunk landen ---
    print("\nFehlende FK-Referenzen pro Chunk:")
    found_any = False
    for i, names in enumerate(chunk_tables, start=1):
        for name in names:
            table = name_to_table.get(name)
            if not table:
                continue
            targets = _extract_fk_targets(table.get("ddl") or "")
            missing = [t for t in targets if t in name_to_table and t not in names]
            if missing:
                found_any = True
                print(f"  Chunk {i}: '{name}' referenziert {missing}, die NICHT in Chunk {i} sind")
    if not found_any:
        print("  (keine - alle FK-Referenzen sind innerhalb desselben Chunks)")


if __name__ == "__main__":
    main()
