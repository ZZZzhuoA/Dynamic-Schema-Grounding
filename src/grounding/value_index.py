import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "to", "was",
    "were", "what", "when", "where", "which", "who", "with",
}


def normalize_value(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[_\-/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem_token(token):
    token = token.casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ly"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def value_tokens(value, remove_stopwords=False):
    tokens = [stem_token(token) for token in TOKEN_RE.findall(normalize_value(value))]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in QUERY_STOPWORDS]
    return [token for token in tokens if len(token) >= 2 or token.isdigit()]


def quote_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def find_sqlite_files(db_root):
    files = {}
    for path in Path(db_root).rglob("*.sqlite"):
        files.setdefault(path.stem, path)
        files.setdefault(path.parent.name, path)
    return files


def read_schema_catalog(graph_file):
    catalog = {}
    with Path(graph_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            inputs = example.get("inference_inputs", {})
            db_id = inputs.get("db_id")
            if db_id and db_id not in catalog:
                catalog[db_id] = inputs.get("schema_nodes", [])
    return catalog


def initialize_index(connection):
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS value_entries (
            id INTEGER PRIMARY KEY,
            db_id TEXT NOT NULL,
            schema_item_id INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            raw_value TEXT NOT NULL,
            frequency INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            UNIQUE(db_id, schema_item_id, normalized_value)
        );
        CREATE TABLE IF NOT EXISTS value_tokens (
            db_id TEXT NOT NULL,
            token TEXT NOT NULL,
            value_id INTEGER NOT NULL,
            UNIQUE(db_id, token, value_id)
        );
        CREATE INDEX IF NOT EXISTS idx_value_tokens_lookup
            ON value_tokens(db_id, token);
        CREATE INDEX IF NOT EXISTS idx_value_entries_schema
            ON value_entries(db_id, schema_item_id);
        CREATE TABLE IF NOT EXISTS indexed_databases (
            db_id TEXT PRIMARY KEY,
            sqlite_path TEXT NOT NULL,
            indexed_column_count INTEGER NOT NULL,
            indexed_value_count INTEGER NOT NULL
        );
        """
    )


def searchable_column(node, include_numeric=False):
    if node.get("type") != "column":
        return False
    dtype = normalize_value(node.get("data_type"))
    if include_numeric:
        return True
    numeric_types = {"integer", "real", "number", "float", "double", "decimal", "numeric"}
    return dtype not in numeric_types


def fetch_distinct_values(connection, table, column, max_values, max_value_chars):
    sql = (
        f"SELECT CAST({quote_identifier(column)} AS TEXT), COUNT(*) "
        f"FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(column)} IS NOT NULL "
        f"GROUP BY {quote_identifier(column)} LIMIT ?"
    )
    try:
        rows = connection.execute(sql, (int(max_values),)).fetchall()
    except sqlite3.Error:
        return []
    result = []
    for raw_value, frequency in rows:
        raw_value = str(raw_value).strip()
        if not raw_value or len(raw_value) > max_value_chars:
            continue
        normalized = normalize_value(raw_value)
        tokens = sorted(set(value_tokens(normalized)))
        if not normalized or not tokens:
            continue
        result.append((normalized, raw_value, int(frequency or 1), tokens))
    return result


def fetch_table_values(connection, table, columns, max_values, max_value_chars):
    """Scan a table once and collect bounded distinct values for all requested columns."""
    if not columns:
        return {}
    select_list = ", ".join(quote_identifier(column) for column in columns)
    try:
        cursor = connection.execute(
            f"SELECT {select_list} FROM {quote_identifier(table)}"
        )
    except sqlite3.Error:
        return {}
    values_by_column = {column: {} for column in columns}
    try:
        for row in cursor:
            for column, raw_value in zip(columns, row):
                if raw_value is None:
                    continue
                raw_value = str(raw_value).strip()
                if not raw_value or len(raw_value) > max_value_chars:
                    continue
                normalized = normalize_value(raw_value)
                if not normalized:
                    continue
                existing = values_by_column[column].get(normalized)
                if existing is not None:
                    existing[1] += 1
                    continue
                if len(values_by_column[column]) >= max_values:
                    continue
                tokens = sorted(set(value_tokens(normalized)))
                if tokens:
                    values_by_column[column][normalized] = [raw_value, 1, tokens]
    except sqlite3.Error:
        return {}
    return {
        column: [
            (normalized, raw_value, frequency, tokens)
            for normalized, (raw_value, frequency, tokens) in values.items()
        ]
        for column, values in values_by_column.items()
    }


def index_database(index_connection, db_id, sqlite_path, schema_nodes, args):
    indexed_columns = 0
    pending_entries = []
    pending_tokens = {}
    source = sqlite3.connect(str(sqlite_path))
    try:
        nodes_by_table = defaultdict(list)
        for node in schema_nodes:
            if not searchable_column(node, args.include_numeric_values):
                continue
            table = node.get("table")
            column = node.get("column")
            if not table or not column:
                continue
            nodes_by_table[table].append(node)
        for table, table_nodes in nodes_by_table.items():
            table_values = fetch_table_values(
                source,
                table,
                [node["column"] for node in table_nodes],
                args.max_values_per_column,
                args.max_value_chars,
            )
            for node in table_nodes:
                column = node["column"]
                values = table_values.get(column, [])
                if not values:
                    continue
                indexed_columns += 1
                for normalized, raw_value, frequency, tokens in values:
                    item_id = int(node["id"])
                    pending_entries.append(
                        (
                            db_id,
                            item_id,
                            table,
                            column,
                            normalized,
                            raw_value,
                            frequency,
                            len(tokens),
                        )
                    )
                    pending_tokens[(item_id, normalized)] = tokens
    finally:
        source.close()
    index_connection.execute("DELETE FROM value_tokens WHERE db_id=?", (db_id,))
    index_connection.execute("DELETE FROM value_entries WHERE db_id=?", (db_id,))
    index_connection.executemany(
        """
        INSERT INTO value_entries(
            db_id, schema_item_id, table_name, column_name,
            normalized_value, raw_value, frequency, token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pending_entries,
    )
    value_ids = {
        (int(item_id), normalized): int(value_id)
        for value_id, item_id, normalized in index_connection.execute(
            "SELECT id, schema_item_id, normalized_value FROM value_entries WHERE db_id=?",
            (db_id,),
        )
    }
    token_rows = []
    for key, tokens in pending_tokens.items():
        value_id = value_ids[key]
        token_rows.extend((db_id, token, value_id) for token in tokens)
    index_connection.executemany(
        "INSERT INTO value_tokens(db_id, token, value_id) VALUES (?, ?, ?)",
        token_rows,
    )
    indexed_values = len(pending_entries)
    index_connection.execute(
        """
        INSERT OR REPLACE INTO indexed_databases(
            db_id, sqlite_path, indexed_column_count, indexed_value_count
        ) VALUES (?, ?, ?, ?)
        """,
        (db_id, str(sqlite_path), indexed_columns, indexed_values),
    )
    index_connection.commit()
    return {"indexed_column_count": indexed_columns, "indexed_value_count": indexed_values}


class ValueIndex:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Value index not found: {self.path}")
        self.connection = sqlite3.connect(str(self.path))

    def close(self):
        self.connection.close()

    def query(
        self,
        db_id,
        question,
        evidence="",
        max_candidates=500,
        max_matches_per_column=3,
        min_score=0.6,
    ):
        query_text = normalize_value(f"{question or ''} {evidence or ''}")
        query_tokens = sorted(set(value_tokens(query_text, remove_stopwords=True)))
        if not query_tokens:
            return []
        placeholders = ",".join("?" for _ in query_tokens)
        sql = f"""
            SELECT e.id, e.schema_item_id, e.table_name, e.column_name,
                   e.normalized_value, e.raw_value, e.frequency, e.token_count,
                   COUNT(DISTINCT t.token) AS matched_tokens
            FROM value_tokens AS t
            JOIN value_entries AS e ON e.id = t.value_id
            WHERE t.db_id = ? AND t.token IN ({placeholders})
            GROUP BY e.id
            ORDER BY matched_tokens DESC, e.frequency DESC
            LIMIT ?
        """
        rows = self.connection.execute(
            sql,
            [db_id, *query_tokens, int(max_candidates)],
        ).fetchall()
        by_column = defaultdict(list)
        padded_query = f" {query_text} "
        for row in rows:
            (
                _, item_id, table, column, normalized, raw_value,
                frequency, token_count, matched_tokens,
            ) = row
            phrase_match = f" {normalized} " in padded_query
            coverage = matched_tokens / max(token_count, 1)
            length_factor = min(1.0, math.log2(token_count + 1))
            score = 1.0 if phrase_match else coverage * (0.85 + 0.15 * length_factor)
            if score < min_score:
                continue
            by_column[int(item_id)].append(
                {
                    "schema_item_id": int(item_id),
                    "table": table,
                    "column": column,
                    "value": raw_value,
                    "normalized_value": normalized,
                    "score": float(score),
                    "matched_tokens": int(matched_tokens),
                    "token_count": int(token_count),
                    "frequency": int(frequency),
                    "phrase_match": bool(phrase_match),
                }
            )
        matches = []
        for item_id, item_matches in by_column.items():
            item_matches.sort(key=lambda item: (item["score"], item["token_count"]), reverse=True)
            matches.append(
                {
                    "schema_item_id": item_id,
                    "score": item_matches[0]["score"],
                    "matches": item_matches[:max_matches_per_column],
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches
