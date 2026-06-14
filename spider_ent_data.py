import json

from utils import parse_orig_sql, create_schema_linker_input, parse_ddl

with open('data/spider_ent/Spider-Ent.json', 'r', encoding='utf-8') as f:
    spider_ent = json.load(f)

with open('data/spider_ent/data_assets.json', 'r', encoding='utf-8') as f:
    schema = json.load(f)

with open('data/spider_ent/column_name_mappings.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)
    column_mappings = {
        db: {
            t.lower(): {k.lower(): v for k, v in cols.items()}
            for t, cols in tables.items()
        }
        for db, tables in raw.items()
    }

with open('data/spider_ent/table_name_mappings.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)
    table_mappings = {
        db: {k.lower(): v for k, v in tables.items()}
        for db, tables in raw.items()
    }


# Schema-Candidates auf Modulebene vorberechnen, damit get_ent_gold_schema_neu darauf zugreifen kann
schema_with_parsed_candidates = {}
for _data_asset, _ddls in schema.items():
    _data_asset_tables = []
    for _id, _ddl in _ddls.items():
        parsed_ddl = parse_ddl(_ddl)
        _data_asset_tables.append({"ddl": _ddl, "candidates": parsed_ddl})
    schema_with_parsed_candidates[_data_asset] = _data_asset_tables


def get_gold_tables_ddls(item: dict) -> list:
    item_db = schema[item['data_asset']]
    gold_ids = item['gold_table_ids']
    tables = [item_db[str(i)] for i in gold_ids if str(i) in item_db]
    return tables


def get_spider_ent_data(tokenizer, max_tokens):
    schema_linker_inputs = []

    for q in spider_ent:
        db_ddls_and_candidates = schema_with_parsed_candidates.get(q['data_asset'])
        schema_linker_input = create_schema_linker_input(db_ddls_and_candidates, q['question'], max_tokens, tokenizer)
        gold_schema = parse_orig_sql(q['original_SQL'])

        # Gold-Schema gegen echtes Schema filtern
        real_schema = {}
        for schema_table in db_ddls_and_candidates:
            t = schema_table['candidates']['table'].lower()
            real_schema[t] = {c.lower() for c in schema_table['candidates']['columns']}
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
                for schema_table in db_ddls_and_candidates:
                    schema_table_name = schema_table['candidates']['table']
                    if schema_table_name.lower() == table.lower():
                        columns.append(schema_table['candidates']['columns'][0])
        schema_linker_inputs.append({"input": schema_linker_input, "gold_schema": gold_schema})
    return schema_linker_inputs

# Noch nötig?
def translate_column_names(ent_table_name: str, orig_column_names: list) -> list:
    """Maps the column names of a table from the original Spider Benchmark to the column name of Spider-Ent"""
    table_info = {}
    for category, tables in table_mappings.items():
        for orig_table, ent_table in tables.items():
            columns = column_mappings.get(category).get(orig_table)
            table_info[ent_table] = {"category": category, 'original_name': orig_table, 'columns': columns}
    column_translation = table_info.get(ent_table_name).get('columns')
    lower_map = {k.lower(): v for k, v in column_translation.items()}
    return [lower_map.get(col.lower()) for col in orig_column_names]


def get_ent_gold_schema_neu(question: dict) -> dict[str, list]:
    """Returns all tables and columns used in the SQL of a question translated to the ent schema"""
    gold_schema = {}
    eval_db = question['eval_db']
    orig_gold_schema = parse_orig_sql(question['original_SQL'])
    # SELECT * Fix: leere Column-Listen mit erster Spalte der Tabelle füllen
    data_asset_tables = schema_with_parsed_candidates.get(question['data_asset'], [])
    for orig_table, orig_columns in orig_gold_schema.items():
        if len(orig_columns) == 0:
            for schema_table in data_asset_tables:
                schema_table_name = schema_table['candidates']['table']
                if schema_table_name.lower() == orig_table.lower():
                    orig_columns.append(schema_table['candidates']['columns'][0])
        ent_table_name = table_mappings.get(eval_db).get(orig_table.lower())
        ent_columns = []
        for orig_column in orig_columns:
            try:
                ent_column = column_mappings.get(eval_db).get(orig_table.lower()).get(orig_column.lower())
            except AttributeError:
                continue
            ent_columns.append(ent_column)
        gold_schema[ent_table_name] = ent_columns
    return gold_schema