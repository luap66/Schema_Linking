import re
from collections import defaultdict

from mo_sql_parsing import parse as mo_parse


def parse_ddl(ddl: str) -> dict:
    """Creates a list of strings in the form of '<table> <column>' for a given table schema string"""
    cleaned = re.sub(r'/\*.*?\*/', '', ddl)
    # Regex-basiertes Parsing: robuster als sqlglot bei Sonderzeichen in Spaltennamen
    table_match = re.search(r'CREATE TABLE\s+`?(\S+?)`?\s*\(', cleaned)
    table_name = table_match.group(1) if table_match else None
    # Spaltenname = alles vor dem Typ-Keyword, erlaubt Sonderzeichen wie %, ()
    # Excludes FOREIGN KEY, PRIMARY KEY constraint lines
    columns = re.findall(r'^\s+(?!FOREIGN\s+KEY|PRIMARY\s+KEY)(\S+)\s+(?:NUMBER|TEXT|INTEGER|REAL|BLOB)\b',
                         cleaned, re.MULTILINE | re.IGNORECASE)
    return {"table": table_name, "columns": columns}


# ---------------------------------------------------------------------------
# SQL-Parsing mit mo-sql-parsing (wie im ExSL-Paper)
# ---------------------------------------------------------------------------

# JOIN-Typen die mo-sql-parsing als Keys verwendet
_JOIN_KEYS = ('join', 'left join', 'right join', 'inner join',
              'cross join', 'left outer join', 'full join', 'full outer join')


def parse_orig_sql(sql) -> dict:
    try:
        tree = mo_parse(sql)
    except Exception:
        # Fallback für syntaktisch ungültige SQLs (z.B. ORDER BY ... INTERSECT)
        # Teile am Set-Operator und parse jeden Teil einzeln
        result = defaultdict(set)
        for part in re.split(r'\b(UNION ALL|UNION|INTERSECT|EXCEPT)\b', sql, flags=re.IGNORECASE):
            part = part.strip()
            if not part or part.upper() in ('UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT'):
                continue
            try:
                sub_tree = mo_parse(part)
                _walk_query(sub_tree, result)
            except Exception:
                continue
        return {k: list(v) for k, v in result.items()}
    result = defaultdict(set)
    _walk_query(tree, result)
    return {k: list(v) for k, v in result.items()}


def _walk_query(node, result):
    """Verarbeitet Set-Operationen (UNION/INTERSECT/EXCEPT) oder einzelne SELECTs."""
    if not isinstance(node, dict):
        return
    for op in ('union', 'union_all', 'intersect', 'except'):
        if op in node:
            items = node[op] if isinstance(node[op], list) else [node[op]]
            for item in items:
                _walk_query(item, result)
            return
    if 'select' in node or 'select_distinct' in node:
        _handle_select(node, result)


def _handle_select(node, result):
    """Extrahiert Tabellen und Spalten aus einem einzelnen SELECT."""
    alias_map = {}
    tables = []
    _extract_tables(node.get('from'), alias_map, tables, result)

    # SELECT * → Tabelle ohne Spalten erfassen
    sel = node.get('select') or node.get('select_distinct')
    if _has_star(sel):
        for t in tables:
            result[t.lower()]

    # Spaltenreferenzen aus allen Klauseln sammeln
    refs = []
    for key in ('select', 'select_distinct', 'where', 'groupby', 'orderby', 'having'):
        _collect_refs(node.get(key), refs, result)
    # ON-Klauseln stecken in der FROM-Struktur
    _collect_on_refs(node.get('from'), refs, result)

    # Alias-Auflösung
    for ref in refs:
        if '.' in ref:
            alias, col = ref.split('.', 1)
            tbl = alias_map.get(alias, alias).lower()
            result[tbl].add(col.lower())
        elif len(tables) == 1:
            result[tables[0].lower()].add(ref.lower())


def _has_star(sel):
    if isinstance(sel, dict) and 'all_columns' in sel:
        return True
    if isinstance(sel, list):
        return any(isinstance(s, dict) and 'all_columns' in s for s in sel)
    return False


def _extract_tables(clause, alias_map, tables, result):
    """Extrahiert Tabellennamen und Aliase aus FROM/JOIN-Klauseln."""
    if clause is None:
        return
    if isinstance(clause, str):
        tables.append(clause)
    elif isinstance(clause, dict):
        # Direkte Tabellenreferenz: {"value": "table_name", "name": "alias"}
        if 'value' in clause:
            val = clause['value']
            if isinstance(val, str):
                tables.append(val)
                if 'name' in clause:
                    alias_map[clause['name']] = val
            elif isinstance(val, dict):
                _walk_query(val, result)  # Subquery in FROM
        # JOIN-Klauseln
        for jk in _JOIN_KEYS:
            if jk in clause:
                j = clause[jk]
                if isinstance(j, str):
                    tables.append(j)
                elif isinstance(j, dict):
                    if 'value' in j and isinstance(j['value'], str):
                        tables.append(j['value'])
                        if 'name' in j:
                            alias_map[j['name']] = j['value']
                    elif 'value' in j and isinstance(j['value'], dict):
                        _walk_query(j['value'], result)
    elif isinstance(clause, list):
        for item in clause:
            _extract_tables(item, alias_map, tables, result)


def _collect_on_refs(clause, refs, result):
    """Sammelt Spaltenreferenzen aus ON-Klauseln innerhalb von FROM."""
    if clause is None:
        return
    if isinstance(clause, dict):
        if 'on' in clause:
            _collect_refs(clause['on'], refs, result)
        for val in clause.values():
            if isinstance(val, (dict, list)):
                _collect_on_refs(val, refs, result)
    elif isinstance(clause, list):
        for item in clause:
            _collect_on_refs(item, refs, result)


def _collect_refs(node, refs, result):
    """Sammelt rekursiv alle Spaltenreferenz-Strings aus dem Parse-Baum."""
    if node is None:
        return
    if isinstance(node, str):
        if node != '*':
            refs.append(node)
    elif isinstance(node, (int, float, bool)):
        return
    elif isinstance(node, dict):
        if 'literal' in node:
            return  # Single-Quoted String-Literal
        if 'select' in node or 'select_distinct' in node:
            _walk_query(node, result)  # Subquery
            return
        if 'all_columns' in node:
            return
        for key, val in node.items():
            if key in ('name', 'sort'):  # Alias (AS ...) und Sortierrichtung überspringen
                continue
            _collect_refs(val, refs, result)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, refs, result)


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
