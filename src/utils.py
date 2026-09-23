import math
import re
from collections import defaultdict, deque

from mo_sql_parsing import parse as mo_parse


# Zeilen ohne Spaltendefinition: Tabellenende, Constraints, FK-Fortsetzung, Leerzeilen
_NON_COLUMN_LINE = re.compile(r'^\s*(?:\)|PRIMARY\s+KEY|FOREIGN\s+KEY|REFERENCES|UNIQUE|CONSTRAINT|$)', re.IGNORECASE)
# Spaltenname = Backtick-Name (darf Leerzeichen enthalten) oder erstes Wort der Zeile (darf %, () enthalten)
_COLUMN_NAME = re.compile(r'^\s*(?:`([^`]+)`|([^\s,]+))')


def parse_ddl(ddl: str) -> dict:
    """Creates a list of strings in the form of '<table> <column>' for a given table schema string"""
    cleaned = re.sub(r'/\*.*?\*/', '', ddl, flags=re.DOTALL)
    table_match = re.search(r'CREATE TABLE\s+`?(\S+?)`?\s*\(', cleaned)
    table_name = table_match.group(1) if table_match else None
    # Eine Spalte pro Zeile, unabhängig von Einrückung (Spider: 4 Spaces, Spider-Ent: keine) und Typ
    # (Spider-Ent nutzt u.a. VARCHAR, DATETIME, BOOLEAN und hat Spalten ohne Typ)
    columns = []
    for line in cleaned.split('\n')[1:]:
        if _NON_COLUMN_LINE.match(line):
            continue
        match = _COLUMN_NAME.match(line)
        if match:
            columns.append(match.group(1) or match.group(2))
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
            tbl = alias_map.get(alias.lower(), alias).lower()
            result[tbl].add(col.lower())
        elif len(tables) == 1:
            result[tables[0].lower()].add(ref.lower())

    # Tabellen die im FROM stehen aber keine Spalten zugeordnet bekamen
    # (z.B. SELECT count(*) FROM t, oder SELECT * FROM t)
    # → mit leerer Spaltenliste erfassen, damit spider_data.py die erste Spalte eintragen kann
    for t in tables:
        if t.lower() not in result:
            result[t.lower()]


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
        # Subquery oder Set-Operation direkt in FROM (ohne {"value": ...} Wrapper)
        if any(k in clause for k in ('select', 'select_distinct', 'union', 'union_all', 'intersect', 'except')):
            _walk_query(clause, result)
            return
        # Direkte Tabellenreferenz: {"value": "table_name", "name": "alias"}
        if 'value' in clause:
            val = clause['value']
            if isinstance(val, str):
                tables.append(val)
                if 'name' in clause:
                    alias_map[clause['name'].lower()] = val
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
                            alias_map[j['name'].lower()] = j['value']
                    elif 'value' in j and isinstance(j['value'], dict):
                        _walk_query(j['value'], result)
    elif isinstance(clause, list):
        for item in clause:
            _extract_tables(item, alias_map, tables, result)


def _collect_on_refs(clause, refs, result):
    """Sammelt Spaltenreferenzen aus ON-Klauseln innerhalb von FROM.
    Stoppt an Subquery-/Set-Op-Grenzen, da diese ihren eigenen Scope haben."""
    if clause is None:
        return
    if isinstance(clause, dict):
        # Nicht in Subqueries oder Set-Operationen absteigen
        if any(k in clause for k in ('select', 'select_distinct', 'union', 'union_all', 'intersect', 'except')):
            return
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


# Gemessen mit dem deepseek-coder Tokenizer: Spider ~2.9 Zeichen/Token (Median), Spider-Ent ~2.2 (Minimum 2.16).
# Bewusst niedriger gewählt, damit ein geschätzter Chunk nie mehr echte Tokens hat als das Context Window.
CHARS_PER_TOKEN_ESTIMATE = 2.0


def count_tokens(text: str, tokenizer=None) -> int:
    """Counts the tokens of a text with the tokenizer, or estimates them from the character count if no tokenizer is given."""
    if tokenizer is None:
        return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)
    return len(tokenizer.encode(text))


def build_schema_linker_prompt(ddl_input: str, question_text: str, columns_input: str) -> str:
    """Builds the prompt string exactly as it is passed to the model."""
    return ddl_input + "\nTo answer: " + question_text + "\nWe need columns:" + columns_input


# Matches "FOREIGN KEY (from_col, ...) REFERENCES `table` (to_col, ...)" as embedded in the generated
# DDL strings of both Spider (spider_data.generate_spider_ddl) and Spider-Ent (data_assets.json).
# Composite keys only contribute their first column - Spider FKs are single-column in practice.
_FK_EDGE_REGEX = re.compile(
    r'FOREIGN KEY\s*\(\s*`?([^`,)\s]+)`?[^)]*\)\s*REFERENCES\s*`?([^`\s(]+)`?\s*\(\s*`?([^`,)\s]+)`?',
    re.IGNORECASE,
)


def _extract_fk_edges(ddl: str) -> list:
    """Returns (from_column_lower, to_table_lower, to_column_lower) for every FOREIGN KEY ...
    REFERENCES ... constraint embedded in a table's DDL."""
    return [(from_col.lower(), to_table.lower(), to_col.lower())
            for from_col, to_table, to_col in _FK_EDGE_REGEX.findall(ddl)]


def _extract_fk_targets(ddl: str) -> set:
    """Returns the lowercased names of tables a table's DDL references via FOREIGN KEY ... REFERENCES."""
    return {to_table for _, to_table, _ in _extract_fk_edges(ddl)}


def _build_fk_adjacency(name_to_ddl: dict) -> dict:
    """Undirected FK adjacency (by lowercased table name), built from generated DDL strings via
    _extract_fk_targets, restricted to the tables in name_to_ddl."""
    adjacency = defaultdict(set)
    for name, ddl in name_to_ddl.items():
        for ref in _extract_fk_targets(ddl):
            if ref in name_to_ddl:
                adjacency[name].add(ref)
                adjacency[ref].add(name)
    return adjacency


def _build_fk_column_adjacency(name_to_ddl: dict) -> dict:
    """Undirected FK adjacency that also records which column of each table participates in the
    relationship: adjacency[a][b] is the column of `a` used to join with `b`. If several foreign
    keys connect the same two tables, the first one found wins - composite/multi-FK relationships
    between the same table pair are rare in Spider and not modelled further."""
    adjacency = defaultdict(dict)
    for name, ddl in name_to_ddl.items():
        for from_col, to_table, to_col in _extract_fk_edges(ddl):
            if to_table not in name_to_ddl:
                continue
            adjacency[name].setdefault(to_table, from_col)
            adjacency[to_table].setdefault(name, to_col)
    return adjacency


def group_tables_by_fk_component(tables_ddl_canditates: list) -> list:
    """Partitions tables into foreign-key-connected components (undirected, transitive): tables
    joined via a foreign key - directly or through intermediate tables - end up in the same group.
    A table with no foreign-key relation to any other candidate forms its own single-table group.
    The relative order of tables from tables_ddl_canditates is preserved within and across groups,
    so behaviour degenerates to the original table order when no foreign keys are present.
    Uses a plain adjacency-dict + DFS (connected components), not a graph library, since the graphs
    here are tiny (a few dozen tables/edges per database)."""
    name_to_table = {}
    for table in tables_ddl_canditates:
        name = table.get('candidates').get('table')
        if name:
            name_to_table[name.lower()] = table

    adjacency = _build_fk_adjacency({name: t.get('ddl') or '' for name, t in name_to_table.items()})

    visited = set()
    groups = []
    for table in tables_ddl_canditates:
        name = table.get('candidates').get('table')
        if not name or name.lower() in visited:
            continue
        component = set()
        stack = [name.lower()]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        visited |= component
        groups.append([t for t in tables_ddl_canditates
                        if t.get('candidates').get('table') and t.get('candidates').get('table').lower() in component])

    return groups


def _pack_tables_into_chunks(tables_ddl_canditates: list, question_text: str, context_window: int, db_id, tokenizer=None,
                              overflow_stats: dict = None) -> list:
    """Greedily packs tables (in the given order) into as few prompt chunks as fit the context window.
    Without a tokenizer the token count is estimated from the character count (see CHARS_PER_TOKEN_ESTIMATE).
    Without a context window the whole schema ends up in a single chunk.
    If overflow_stats is given, every split appends the number of tokens the next table would have exceeded
    the context window by to overflow_stats[db_id]."""
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

        new_ddl_input = collected_ddl_input + "\n" + table_ddl
        new_columns_input = collected_columns_input + current_column_input
        exceeded_by = 0
        if context_window is not None:
            # Count the complete prompt incl. question template, otherwise chunks end up a few tokens too long and get truncated
            new_prompt = build_schema_linker_prompt(new_ddl_input, question_text, new_columns_input)
            exceeded_by = count_tokens(new_prompt, tokenizer) - context_window

        # A table that does not even fit into an empty chunk gets its own chunk instead of producing an empty one
        if exceeded_by > 0 and collected_ddl_input:
            # Append old strings to the schema linker inputs, because adding another table would surpass the context window size.
            schema_linker_inputs.append(build_schema_linker_prompt(collected_ddl_input, question_text, collected_columns_input))
            # Reset the schema_input variables to the values of this table, so the table can be included in the next schema linker inputs.
            collected_ddl_input = table_ddl
            collected_columns_input = current_column_input
            if overflow_stats is not None:
                overflow_stats.setdefault(db_id, []).append(exceeded_by)
        else:
            # The schema input is still small enough, so the schema of the current table can be concatenated to the schema_input item
            collected_ddl_input = new_ddl_input
            collected_columns_input = new_columns_input

    # Append the last remaining block
    if collected_ddl_input:
        schema_linker_inputs.append(build_schema_linker_prompt(collected_ddl_input, question_text, collected_columns_input))

    return schema_linker_inputs


def _pack_components_into_chunks(components: list, question_text: str, context_window: int, db_id, tokenizer=None,
                                  overflow_stats: dict = None) -> list:
    """Bin-packs whole foreign-key-connected components (see group_tables_by_fk_component) into chunks:
    as many complete components as fit are combined into one chunk, so unrelated small components still
    get packed densely together. A chunk boundary only ever falls inside a component when that single
    component alone does not fit into an (otherwise empty) chunk - in that case it is split table-by-table
    via _pack_tables_into_chunks, same as before."""
    schema_linker_inputs = []
    pending = []  # tables of whole components accumulated so far that still fit together in one chunk
    for component in components:
        candidate = pending + component
        if len(_pack_tables_into_chunks(candidate, question_text, context_window, db_id, tokenizer)) <= 1:
            pending = candidate
            continue
        if pending:
            schema_linker_inputs.extend(
                _pack_tables_into_chunks(pending, question_text, context_window, db_id, tokenizer, overflow_stats)
            )
        if len(_pack_tables_into_chunks(component, question_text, context_window, db_id, tokenizer)) <= 1:
            pending = component
        else:
            # This single component alone already overflows a chunk - split it internally.
            schema_linker_inputs.extend(
                _pack_tables_into_chunks(component, question_text, context_window, db_id, tokenizer, overflow_stats)
            )
            pending = []
    if pending:
        schema_linker_inputs.extend(
            _pack_tables_into_chunks(pending, question_text, context_window, db_id, tokenizer, overflow_stats)
        )
    return schema_linker_inputs


def create_schema_linker_input(tables_ddl_canditates: list, question_text: str, context_window: int, db_id, tokenizer=None,
                               overflow_stats: dict = None) -> list:
    """Builds schema linker inputs for a single question. Splits the database schema into multiple
    chunks if the full schema exceeds the context window size.

    The whole schema is first packed in its original table order (see _pack_tables_into_chunks).
    If that already fits into a single chunk, it is returned unchanged - this is the case for
    almost all Spider-Train and all Spider-Val questions.
    Only when a split is actually unavoidable (the schema overflows the context window) are the
    tables first grouped into foreign-key-connected components (see group_tables_by_fk_component),
    which are then bin-packed into chunks (see _pack_components_into_chunks) - so a split never
    separates tables that are joined via a foreign key unless a single component is itself too
    large for one chunk, while unrelated small components still get packed densely together.

    Without a tokenizer the token count is estimated from the character count (see CHARS_PER_TOKEN_ESTIMATE).
    Without a context window the whole schema ends up in a single chunk.
    If overflow_stats is given, every split appends the number of tokens the next table would have exceeded
    the context window by to overflow_stats[db_id].
    Returns a list of prompt strings, each containing a subset of tables and their candidate columns."""
    if tokenizer is not None and context_window is None:
        raise ValueError("context_window is required when a tokenizer is given")

    trial = _pack_tables_into_chunks(tables_ddl_canditates, question_text, context_window, db_id, tokenizer)
    if len(trial) <= 1:
        return trial

    components = group_tables_by_fk_component(tables_ddl_canditates)
    return _pack_components_into_chunks(components, question_text, context_window, db_id, tokenizer, overflow_stats)

def get_gold_schema(query:str, db_tables):
    gold_schema = parse_orig_sql(query)

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
    return gold_schema


def _shortest_path_between_sets(adjacency: dict, sources: set, targets: set) -> list:
    """Multi-source BFS shortest path from any node in sources to any node in targets.
    Returns the full path (a source .. a target) or None if no target is reachable."""
    visited = set(sources)
    queue = deque([[s] for s in sources])
    while queue:
        path = queue.popleft()
        for neighbor in adjacency.get(path[-1], ()):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor in targets:
                return new_path
            visited.add(neighbor)
            queue.append(new_path)
    return None


def add_missing_bridge_tables(pred_schema: dict, tables_ddl_canditates: list) -> dict:
    """Completes a predicted schema (as produced by schema-linking inference, without access to
    gold SQL) with bridge tables: tables that only contribute a JOIN condition between other
    relevant tables and never a SELECT/WHERE/GROUP BY/... column (see Bridge-Tables.ipynb).

    Tables in pred_schema that are not reachable from each other via other predicted tables are
    connected via the shortest path through the *full* FK graph of the database (built the same
    way as group_tables_by_fk_component). Every table on that path not already in pred_schema is
    added with the FK column(s) that actually tie it into the path - the column(s) of that table
    used in its FOREIGN KEY relationship to its neighbour(s) on the path - since the column-level
    gold schema (get_gold_schema/parse_orig_sql) credits exactly those join columns for a bridge
    table, not an arbitrary one. Predicted tables with no FK path between them at all (e.g.
    disconnected schema components) are left as-is, since there is no FK-justified way to add a
    JOIN between them.
    """
    name_to_ddl = {
        table['candidates']['table'].lower(): table.get('ddl') or ''
        for table in tables_ddl_canditates if table.get('candidates', {}).get('table')
    }
    column_adjacency = _build_fk_column_adjacency(name_to_ddl)
    full_adjacency = {name: set(neighbors) for name, neighbors in column_adjacency.items()}

    result = {t.lower(): list(cols) for t, cols in pred_schema.items()}
    predicted = set(result.keys())
    if len(predicted) < 2:
        return result

    # Connected components of the predicted tables, using only direct predicted-predicted FK edges
    components = []
    unvisited = set(predicted)
    while unvisited:
        component = {next(iter(unvisited))}
        stack = list(component)
        while stack:
            node = stack.pop()
            for neighbor in (full_adjacency.get(node, set()) & predicted) - component:
                component.add(neighbor)
                stack.append(neighbor)
        components.append(component)
        unvisited -= component

    # Greedily connect the nearest pair of components until one component remains (or no more
    # FK path exists between the rest) - the common case is exactly two components one hop apart.
    while len(components) > 1:
        best = None  # (path, i, j)
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                path = _shortest_path_between_sets(full_adjacency, components[i], components[j])
                if path and (best is None or len(path) < len(best[0])):
                    best = (path, i, j)
        if best is None:
            break
        path, i, j = best
        for pos in range(1, len(path) - 1):
            bridge_table = path[pos]
            if bridge_table in result:
                continue
            join_cols = [column_adjacency[bridge_table][neighbor]
                         for neighbor in (path[pos - 1], path[pos + 1])]
            # dedupe while preserving order, in case both neighbours join on the same column
            result[bridge_table] = list(dict.fromkeys(join_cols))
        merged = components[i] | components[j] | set(path)
        components = [c for k, c in enumerate(components) if k not in (i, j)] + [merged]

    return result