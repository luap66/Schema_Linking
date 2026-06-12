import re
from collections import defaultdict

import sqlglot
import sqlglot.expressions as exp
import torch
import torch.nn as nn


def parse_ddl(ddl: str) -> dict:
    """Creates a list of strings in the form of '<table> <column>' for a given table schema string"""
    cleaned = re.sub(r'/\*.*?\*/', '', ddl)
    try:
        fixed = re.sub(r'(`\w+`\s+\w+(?:\(\d+\))?)\)', r'\1', cleaned)
        tree = sqlglot.parse_one(fixed, dialect="mysql")
        table_name = tree.find(exp.Table).name if tree.find(exp.Table) else None
        columns = [col.name for col in tree.find_all(exp.ColumnDef)]
    except sqlglot.errors.ParseError:
        # Fallback: extract table name and column definitions line by line
        table_match = re.search(r'CREATE TABLE `(\w+)`', cleaned)
        table_name = table_match.group(1) if table_match else None
        # Match lines starting with a backtick column name followed by a type word — excludes PRIMARY KEY, FOREIGN KEY etc.
        columns = re.findall(r'^\s*`(\w+)`\s+\w', cleaned, re.MULTILINE)
    return {"table": table_name, "columns": columns}


def parse_orig_sql(sql) -> dict:
    tree = sqlglot.parse_one(sql)

    result = defaultdict(set)

    def local_tables_of_select(select_node):
        """Tabellen direkt im FROM/JOIN dieses SELECTs, ohne in Subqueries abzusteigen."""
        def collect(node):
            if isinstance(node, exp.Subquery):
                return []
            tables = []
            if isinstance(node, exp.Table):
                tables.append(node)
            for child in node.args.values():
                if child is None:
                    continue
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, exp.Expression):
                            tables.extend(collect(item))
                elif isinstance(child, exp.Expression):
                    tables.extend(collect(child))
            return tables

        local = []
        from_clause = select_node.args.get('from_') or select_node.args.get('from')
        if from_clause:
            local.extend(collect(from_clause))
        for join in (select_node.args.get('joins') or []):
            local.extend(collect(join))
        return local

    # SELECT * behandeln: Tabellen erfassen, auch wenn keine expliziten Spalten genannt werden
    for star in tree.find_all(exp.Star):
        parent_select = star.find_ancestor(exp.Select)
        if parent_select is None:
            continue
        local_tables = local_tables_of_select(parent_select)
        for t in local_tables:
            table_name = t.name.lower()
            # Tabelle ins Result aufnehmen (leere Menge = alle Spalten via *)
            result[table_name]  # defaultdict erzeugt leeres set

    # Alle genutzten Columns sammeln
    for col in tree.find_all(exp.Column):
        table_alias = col.table.lower()
        col_name = col.name.lower()
        if not col_name:
            continue

        parent_select = col.find_ancestor(exp.Select)

        if table_alias:
            # Alias-Map nur aus dem lokalen Scope dieses SELECTs bauen. Denn in einem SQL können mehrere SELECTs enthalten sein, bspw. bei Intersect-Statements.
            if parent_select is not None:
                local_tables = local_tables_of_select(parent_select)
                local_alias_map = {t.alias.lower(): t.name for t in local_tables if t.alias}
            else:
                local_alias_map = {}
            table_name = local_alias_map.get(table_alias, table_alias).lower()
            result[table_name].add(col_name)
        else:
            # Lokalen Scope: naechster umgebender SELECT
            if parent_select is None:
                continue
            local_tables = local_tables_of_select(parent_select)
            if len(local_tables) == 1:
                result[local_tables[0].name.lower()].add(col_name)

    return {k: list(v) for k, v in result.items()}


def has_join_and_alias(sql: str) -> dict:
    tree = sqlglot.parse_one(sql)

    joins = list(tree.find_all(exp.Join))
    tables_with_alias = [t for t in tree.find_all(exp.Table) if t.alias]

    return {
        "has_join": len(joins) > 0,
        "has_alias": len(tables_with_alias) > 0,
        "aliases": {t.alias: t.name for t in tables_with_alias},
        "result": len(joins) > 0 and len(tables_with_alias) > 0
    }


def create_schema_linker_input(tables_ddl_canditates: list, question_text: str, context_window: int, tokenizer) -> list:
    """Builds schema linker inputs for a single question. Splits the database schema into multiple
    chunks if the full schema exceeds the context window size.
    Returns a list of prompt strings, each containing a subset of tables and their candidate columns."""
    schema_linker_inputs = []
    collected_ddl_input = ""
    collected_columns_input = ""
    for table in tables_ddl_canditates:

        table_ddl = table.get("ddl")
        table_name = table.get('candidates').get('table')
        column_list = table.get('candidates').get('columns')

        current_column_input = ""

        for column in column_list:
            current_column_input = current_column_input + "\n« " + table_name + " " + column + "»"

        new_collected = collected_ddl_input + table_ddl + question_text + collected_columns_input + current_column_input
        token_count = len(tokenizer.encode(new_collected))
        if token_count >= context_window:
            # Append old strings to the schema linker inputs, because adding another table would surpass the context window size.
            schema_input_in_context_window_size = collected_ddl_input + "\nTo answer: " + question_text + "\nWe need columns:" + collected_columns_input
            schema_linker_inputs.append(schema_input_in_context_window_size)
            # Reset the schema_input variables to the values of this table, so the table can be included in the next schema linker inputs.
            collected_ddl_input = table_ddl
            collected_columns_input = current_column_input
        else:
            # The schema input is still small enough, so the schema of the current table can be concatenated to the schema_input item
            collected_ddl_input = collected_ddl_input + "\n" + table_ddl
            collected_columns_input = collected_columns_input + current_column_input

    # Append the last remaining block
    if collected_ddl_input:
        schema_linker_inputs.append(
            collected_ddl_input + "\nTo answer: " + question_text + "\nWe need columns:" + collected_columns_input
        )

    return schema_linker_inputs
