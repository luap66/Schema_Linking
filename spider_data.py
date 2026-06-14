import json
import re

from datasets import load_dataset, Dataset
from utils import create_schema_linker_input, parse_orig_sql, parse_ddl

from dotenv import load_dotenv

load_dotenv()

with open('data/spider/tables.json', 'r', encoding='utf-8') as f:
    spider_tables = json.load(f)

spider_train = load_dataset("xlangai/spider", split="train")

spider_val = load_dataset("xlangai/spider", split="validation")


def get_spider_train(tokenizer, max_tokens) -> list[dict]:
    return get_spider_x_y_set(spider_train, tokenizer, max_tokens)

def get_spider_val(tokenizer, max_tokens) -> list[dict]:
    return get_spider_x_y_set(spider_val, tokenizer, max_tokens)

def get_spider_x_y_set(data_set: Dataset, tokenizer, max_tokens) -> list[dict]:
    schema_linker_inputs = []
    spider_schema_ddls = generate_spider_ddl(spider_tables)
    spider_schema_ddls_and_candidates = {}
    for db, ddls in spider_schema_ddls.items():
        db_table_info = []
        for ddl in ddls:
            parsed_ddl = parse_ddl(ddl)
            db_table_info.append({"ddl": ddl, "candidates": parsed_ddl})
        spider_schema_ddls_and_candidates[db] = db_table_info

    for q in data_set:
        db_tables = spider_schema_ddls_and_candidates[q['db_id']]
        schema_linker_input = create_schema_linker_input(db_tables, q['question'], max_tokens, tokenizer)
        gold_schema = parse_orig_sql(q['query'])

        # Schema-Lookup: {table_name_lower: set(col_name_lower)} aus den echten DB-Spalten
        real_schema = {}
        for schema_table in db_tables:
            t = schema_table['candidates']['table'].lower()
            real_schema[t] = {c.lower() for c in schema_table['candidates']['columns']}

        # Gold-Schema gegen echtes Schema filtern: nur Tabellen/Spalten behalten die wirklich existieren
        filtered_gold = {}
        for table, columns in gold_schema.items():
            if table.lower() not in real_schema:
                continue
            valid_cols = [c for c in columns if c.lower() in real_schema[table.lower()]]
            filtered_gold[table] = valid_cols
        gold_schema = filtered_gold

        for table, columns in gold_schema.items():
            # SELECT * erzeugt leere Column-Liste — erste Spalte der Tabelle eintragen
            if len(columns) == 0:
                for schema_table in db_tables:
                    schema_table_name = schema_table['candidates']['table']
                    if schema_table_name.lower() == table.lower():
                        columns.append(schema_table['candidates']['columns'][0])
        schema_linker_inputs.append({"input": schema_linker_input, "gold_schema": gold_schema, "sql": q['query']})
    return schema_linker_inputs

def generate_spider_ddl(tables_json: list) -> dict[str, list]:
    """Generates DDL strings from Spider tables.json.
    Returns dict: {db_id: {table_name: ddl_string}}"""
    schema = {}
    for db in tables_json:
        db_id = db['db_id']
        schema[db_id] = []

        tables = db['table_names_original']
        primary_keys = set(db['primary_keys'])

        # Group columns by table index
        columns_by_table = {i: [] for i in range(len(tables))}
        for col_idx, (table_idx, col_name) in enumerate(db['column_names_original']):
            if table_idx == -1:  # skip * wildcard column
                continue
            col_type = db['column_types'][col_idx]
            sql_type = 'TEXT' if col_type == 'text' else 'NUMBER'
            columns_by_table[table_idx].append((col_idx, col_name, sql_type))

        for table_idx, table_name in enumerate(tables):
            cols = columns_by_table[table_idx]
            pk_col_set = {col_name.replace(' ', '_') for col_idx, col_name, _ in cols if col_idx in primary_keys}

            col_defs = []
            for _, col_name, sql_type in cols:
                # Leerzeichen in Spaltennamen durch Unterstriche ersetzen,
                # damit « table column » Marker als zwei Tokens funktionieren
                col_name = col_name.replace(' ', '_')
                pk_suffix = ' PRIMARY KEY' if col_name in pk_col_set else ''
                col_defs.append(f'    {col_name} {sql_type}{pk_suffix}')

            fk_defs = []
            for fk_from, fk_to in db['foreign_keys']:
                if db['column_names_original'][fk_from][0] == table_idx:
                    fk_from_col = db['column_names_original'][fk_from][1].replace(' ', '_')
                    fk_to_table = db['table_names_original'][db['column_names_original'][fk_to][0]]
                    fk_to_col = db['column_names_original'][fk_to][1].replace(' ', '_')
                    fk_defs.append(f'    FOREIGN KEY({fk_from_col})\n      REFERENCES {fk_to_table}({fk_to_col})')

            all_defs = col_defs + fk_defs
            ddl = f'CREATE TABLE {table_name} (\n' + ',\n'.join(all_defs) + ' );'
            schema[db_id].append(ddl)

    return schema