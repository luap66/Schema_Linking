"""Findet FK-connected-components (siehe utils.group_tables_by_fk_component), die NICHT
komplett in MAX_TOKENS passen und dabei eine Hub-Tabelle enthalten.

Das ist genau der Fall, in dem create_schema_linker_input die FK-Garantie aufgibt und auf
reines sequenzielles Packen zurueckfaellt (siehe utils._pack_components_into_chunks,
Zeilen 334-340) - Satelliten-Tabellen koennen dann in einem anderen Chunk landen als der
Hub, den sie referenzieren.
"""

import sys

from transformers import AutoTokenizer

from spider_data import get_spider_schema_ddl_and_candidates
from spider_ent_data import schema_with_parsed_candidates as ent_schemas
from utils import group_tables_by_fk_component, _pack_tables_into_chunks
from find_hub_tables import find_hub_tables

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # z.B. wenn stdout umgeleitet ist und kein TextIOWrapper ist

MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
MAX_TOKENS = 3000


def group_fits(group: list, tokenizer, max_tokens: int, schema_id) -> bool:
    """Prueft, ob eine einzelne FK-Gruppe (ohne Frage) in einen einzigen Chunk passt."""
    chunks = _pack_tables_into_chunks(group, "", max_tokens, schema_id, tokenizer)
    return len(chunks) <= 1


def report(schemas: dict, tokenizer, label: str):
    print(f"\n=== {label} ===")
    any_hit = False
    for schema_id, db_tables in schemas.items():
        for group in group_tables_by_fk_component(db_tables):
            if len(group) <= 1:
                continue
            if group_fits(group, tokenizer, MAX_TOKENS, schema_id):
                continue
            hubs = find_hub_tables(group)
            if not hubs:
                continue
            any_hit = True
            names = [t["candidates"]["table"] for t in group]
            print(f"\n{schema_id}: FK-Gruppe mit {len(names)} Tabellen passt NICHT in "
                  f"{MAX_TOKENS} Tokens: {names}")
            for name, count, referencing_tables in hubs:
                print(f"  HUB '{name}': wird von {count} Tabellen in dieser Gruppe "
                      f"referenziert -> {referencing_tables}")
    if not any_hit:
        print("  (keine ueberlaufenden Gruppen mit Hub-Tabelle gefunden)")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    spider_schemas = get_spider_schema_ddl_and_candidates()
    report(spider_schemas, tokenizer, "Spider (train/val)")
    report(ent_schemas, tokenizer, "Spider-Ent (data assets)")


if __name__ == "__main__":
    main()
