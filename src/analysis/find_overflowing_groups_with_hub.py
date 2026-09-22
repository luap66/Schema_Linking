"""Findet FK-connected-components (siehe utils.group_tables_by_fk_component), die NICHT
komplett in MAX_TOKENS passen und dabei eine Hub-Tabelle enthalten.

Das ist genau der Fall, in dem create_schema_linker_input die FK-Garantie aufgibt und auf
reines sequenzielles Packen zurueckfaellt (siehe utils._pack_components_into_chunks,
Zeilen 334-340) - Satelliten-Tabellen koennen dann in einem anderen Chunk landen als der
Hub, den sie referenzieren.

Standardmaessig werden nur aggregierte Zahlen pro Datensatz ausgegeben (Anzahl ueberlaufender
FK-Gruppen, davon mit Hub-Tabelle, betroffene Hub- und Satelliten-Tabellen). Mit --verbose
wird zusaetzlich die bisherige Detail-Auflistung pro Gruppe ausgegeben.
"""

import argparse
import sys

from transformers import AutoTokenizer

from src.data_loaders.spider_data import get_spider_schema_ddl_and_candidates
from src.data_loaders.spider_ent_data import schema_with_parsed_candidates as ent_schemas
from src.utils import group_tables_by_fk_component, _pack_tables_into_chunks
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


def collect_overflowing_groups(schemas: dict, tokenizer) -> list:
    """Returns one entry per FK-connected component (across all schemas) that has more than
    one table and does not fit into a single chunk: {schema_id, tables, hubs}, where hubs is
    the find_hub_tables() result restricted to that group (empty list if no hub is present)."""
    overflowing_groups = []
    for schema_id, db_tables in schemas.items():
        for group in group_tables_by_fk_component(db_tables):
            if len(group) <= 1:
                continue
            if group_fits(group, tokenizer, MAX_TOKENS, schema_id):
                continue
            overflowing_groups.append({
                "schema_id": schema_id,
                "tables": [t["candidates"]["table"] for t in group],
                "hubs": find_hub_tables(group),
            })
    return overflowing_groups


def aggregate(overflowing_groups: list) -> dict:
    """Aggregates the per-group results into split-level counts."""
    groups_with_hub = [g for g in overflowing_groups if g["hubs"]]

    hub_tables = {(g["schema_id"], name) for g in groups_with_hub for name, _, _ in g["hubs"]}
    satellite_tables = {
        (g["schema_id"], referencer)
        for g in groups_with_hub
        for _, _, referencing_tables in g["hubs"]
        for referencer in referencing_tables
    }

    return {
        "overflowing_groups": len(overflowing_groups),
        "overflowing_groups_with_hub": len(groups_with_hub),
        "affected_hub_tables": len(hub_tables),
        "affected_satellite_tables": len(satellite_tables),
    }


def report(schemas: dict, tokenizer, label: str, verbose: bool):
    print(f"\n=== {label} ===")
    overflowing_groups = collect_overflowing_groups(schemas, tokenizer)
    stats = aggregate(overflowing_groups)

    print(f"  Ueberlaufende FK-Gruppen (>{MAX_TOKENS} Tokens): {stats['overflowing_groups']}")
    share = (stats["overflowing_groups_with_hub"] / stats["overflowing_groups"] * 100
             if stats["overflowing_groups"] else 0.0)
    print(f"  Davon mit mindestens einer Hub-Tabelle: "
          f"{stats['overflowing_groups_with_hub']} ({share:.1f}%)")
    print(f"  Betroffene Hub-Tabellen (schema-eindeutig): {stats['affected_hub_tables']}")
    print(f"  Betroffene Satelliten-Tabellen (schema-eindeutig): {stats['affected_satellite_tables']}")

    if verbose:
        groups_with_hub = [g for g in overflowing_groups if g["hubs"]]
        if not groups_with_hub:
            print("  (keine ueberlaufenden Gruppen mit Hub-Tabelle gefunden)")
        for g in groups_with_hub:
            print(f"\n{g['schema_id']}: FK-Gruppe mit {len(g['tables'])} Tabellen passt NICHT "
                  f"in {MAX_TOKENS} Tokens: {g['tables']}")
            for name, count, referencing_tables in g["hubs"]:
                print(f"  HUB '{name}': wird von {count} Tabellen in dieser Gruppe "
                      f"referenziert -> {referencing_tables}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Overflowing FK-groups with hub tables")
    parser.add_argument("--verbose", action="store_true",
                         help="Zusaetzlich die Detail-Auflistung pro Gruppe ausgeben")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    spider_schemas = get_spider_schema_ddl_and_candidates()
    spider_stats = report(spider_schemas, tokenizer, "Spider (train/val)", args.verbose)
    ent_stats = report(ent_schemas, tokenizer, "Spider-Ent (data assets)", args.verbose)

    print("\n=== Gesamt ===")
    total = {
        key: spider_stats[key] + ent_stats[key]
        for key in spider_stats
    }
    share = (total["overflowing_groups_with_hub"] / total["overflowing_groups"] * 100
             if total["overflowing_groups"] else 0.0)
    print(f"  Ueberlaufende FK-Gruppen: {total['overflowing_groups']}")
    print(f"  Davon mit mindestens einer Hub-Tabelle: "
          f"{total['overflowing_groups_with_hub']} ({share:.1f}%)")
    print(f"  Betroffene Hub-Tabellen (schema-eindeutig): {total['affected_hub_tables']}")
    print(f"  Betroffene Satelliten-Tabellen (schema-eindeutig): {total['affected_satellite_tables']}")


if __name__ == "__main__":
    main()
