"""Findet Hub-Tabellen (Tabellen, die von vielen anderen Tabellen per FK referenziert werden)
in allen Spider- und Spider-Ent-Schemas, die NICHT komplett in MAX_TOKENS passen.

Hub-Tabellen sind fuer das Schema-Chunking (siehe utils.create_schema_linker_input) besonders
kritisch: sobald ein Schema zu gross fuer einen Chunk ist und die FK-Komponente des Hubs selbst
nicht mehr in einen Chunk passt, faellt das Chunking auf reines sequenzielles Packen zurueck -
Satelliten-Tabellen landen dann in einem anderen Chunk als der Hub, den sie referenzieren.
"""

import sys
from collections import defaultdict

from transformers import AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # z.B. wenn stdout umgeleitet ist und kein TextIOWrapper ist

from src.data_loaders.spider_data import get_spider_schema_ddl_and_candidates
from src.data_loaders.spider_ent_data import schema_with_parsed_candidates as ent_schemas
from src.utils import _extract_fk_targets, create_schema_linker_input

MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
MAX_TOKENS = 3000
MIN_REFERENCED_BY = 3  # ab wie vielen referenzierenden Tabellen ein Table als "Hub" gilt


def schema_fits(db_tables: list, tokenizer, max_tokens: int, db_id) -> bool:
    """Prueft, ob das GESAMTE Schema (ohne Frage) in einen einzigen Chunk passt."""
    chunks = create_schema_linker_input(db_tables, "", max_tokens, db_id, tokenizer)
    return len(chunks) <= 1


def find_hub_tables(db_tables: list, min_referenced_by: int = MIN_REFERENCED_BY) -> list:
    """Gibt [(table_name, referenced_by_count, [referencing_tables])] sortiert nach Count absteigend."""
    name_to_table = {
        t["candidates"]["table"].lower(): t
        for t in db_tables if t["candidates"].get("table")
    }

    referenced_by = defaultdict(set)
    for name, table in name_to_table.items():
        for target in _extract_fk_targets(table.get("ddl") or ""):
            if target in name_to_table:
                referenced_by[target].add(name)

    hubs = [
        (name, len(referencers), sorted(referencers))
        for name, referencers in referenced_by.items()
        if len(referencers) >= min_referenced_by
    ]
    hubs.sort(key=lambda x: -x[1])
    return hubs


def report(schemas: dict, tokenizer, label: str):
    print(f"\n=== {label} ===")
    any_hit = False
    for schema_id, db_tables in schemas.items():
        if schema_fits(db_tables, tokenizer, MAX_TOKENS, schema_id):
            continue
        hubs = find_hub_tables(db_tables)
        if not hubs:
            continue
        any_hit = True
        print(f"\n{schema_id}: {len(db_tables)} Tabellen, passt NICHT in {MAX_TOKENS} Tokens")
        for name, count, referencing_tables in hubs:
            print(f"  HUB '{name}': wird von {count} Tabellen referenziert -> {referencing_tables}")
    if not any_hit:
        print("  (keine Hub-Tabellen in ueberlaufenden Schemas gefunden)")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    spider_schemas = get_spider_schema_ddl_and_candidates()
    report(spider_schemas, tokenizer, "Spider (train/val)")
    report(ent_schemas, tokenizer, "Spider-Ent (data assets)")


if __name__ == "__main__":
    main()
