from datasets import load_dataset, Dataset
from utils import create_schema_linker_input, parse_orig_sql, parse_ddl
import json

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
        schema_linker_inputs.append({"input": schema_linker_input, "gold_schema": gold_schema})
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
            col_defs = [f'`{col_name}` {sql_type}' for _, col_name, sql_type in cols]

            pk_cols = [col_name for col_idx, col_name, _ in cols if col_idx in primary_keys]
            if pk_cols:
                col_defs.append(f'PRIMARY KEY ({", ".join([f"`{c}`" for c in pk_cols])})')

            for fk_from, fk_to in db['foreign_keys']:
                if db['column_names_original'][fk_from][0] == table_idx:
                    fk_from_col = db['column_names_original'][fk_from][1]
                    fk_to_table = db['table_names_original'][db['column_names_original'][fk_to][0]]
                    fk_to_col = db['column_names_original'][fk_to][1]
                    col_defs.append(f'FOREIGN KEY (`{fk_from_col}`) REFERENCES `{fk_to_table}` (`{fk_to_col}`)')

            ddl = f'CREATE TABLE `{table_name}` (\n' + ',\n'.join([f'  {d}' for d in col_defs]) + '\n);'
            schema[db_id].append(ddl)

    return schema