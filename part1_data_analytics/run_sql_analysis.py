"""
Capstone Part 1 - Task 1 : SQL-based traffic analysis.

Rebuilds `traffic.db` from the raw CSV, executes every statement in the
three SQL files under `sql/`, and writes the results to
`outputs/sql_results.md` so that a grader can read the answers without
installing SQLite tooling.

Run from the repository root:

    python part1_data_analytics/run_sql_analysis.py
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PART1_DIR = Path(__file__).resolve().parent
REPO_ROOT = PART1_DIR.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
DB_PATH = PART1_DIR / "outputs" / "traffic.db"
SQL_DIR = PART1_DIR / "sql"
RESULTS_MD = PART1_DIR / "outputs" / "sql_results.md"

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]


def configure_logging() -> None:
    """Console logging for the entry-point script."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def load_csv_into_sqlite(csv_path: Path, db_path: Path) -> sqlite3.Connection:
    """Create the database and load the comma-delimited CSV into `traffic`.

    The holiday column is read as raw text so that the literal string
    'None' is preserved. (Reading this file with pandas' defaults would
    silently convert 'None' to a missing value, because 'None' is one of
    pandas' default NA tokens.)
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    ddl = (SQL_DIR / "01_load_and_verify.sql").read_text(encoding="utf-8")
    # Execute only the DDL portion (everything before the verification header).
    ddl_only = ddl.split("-- VERIFICATION QUERIES")[0]
    conn.executescript(ddl_only)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected CSV header.\n  expected: {EXPECTED_COLUMNS}\n  found:    {header}"
            )
        rows = [
            (
                r[0],                    # holiday  (text, keeps 'None')
                float(r[1]),             # temp
                float(r[2]),             # rain_1h
                float(r[3]),             # snow_1h
                int(r[4]),               # clouds_all
                r[5], r[6], r[7],        # weather_main, description, date_time
                int(r[8]),               # traffic_volume
            )
            for r in reader
        ]

    conn.executemany(
        "INSERT INTO traffic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    logger.info("Loaded %s rows x %s columns into %s", len(rows), len(EXPECTED_COLUMNS), db_path.name)
    return conn


def split_statements(sql_text: str) -> list[tuple[str, str]]:
    """Split a SQL file into (label, statement) pairs.

    The label is taken from the most recent `-- X9. description` comment
    seen before the statement, which is how the SQL files in this project
    name their queries.
    """
    statements: list[tuple[str, str]] = []
    buffer: list[str] = []
    label = "statement"

    for raw_line in sql_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("--"):
            body = stripped.lstrip("-").strip()
            # A label looks like "A1. Total and average traffic volume per year"
            head = body.split(".")[0]
            if 1 < len(head) <= 4 and head[0].isalpha() and head[1:].isdigit():
                label = body
            continue
        buffer.append(raw_line)
        if stripped.endswith(";"):
            stmt = "\n".join(buffer).strip()
            if stmt:
                statements.append((label, stmt))
            buffer = []
            label = "statement"

    tail = "\n".join(buffer).strip()
    if tail:
        statements.append((label, tail))
    return statements


def render_table(columns: list[str], rows: list[tuple]) -> str:
    """Render a result set as a GitHub-flavoured markdown table."""
    if not rows:
        return "_(no rows returned)_\n"
    widths = [len(c) for c in columns]
    text_rows = []
    for row in rows:
        cells = ["" if v is None else str(v) for v in row]
        widths = [max(w, len(c)) for w, c in zip(widths, cells)]
        text_rows.append(cells)

    head = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join(
        "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
        for cells in text_rows
    )
    return f"{head}\n{rule}\n{body}\n"


def run_sql_file(conn: sqlite3.Connection, path: Path, out) -> None:
    """Execute every statement in a SQL file and write its results."""
    out.write(f"\n\n## {path.name}\n")
    sql_text = path.read_text(encoding="utf-8")

    for label, stmt in split_statements(sql_text):
        head = stmt.lstrip().upper()
        if head.startswith(("DROP", "CREATE", "INSERT")):
            continue  # DDL already applied by the loader
        try:
            cur = conn.execute(stmt)
        except sqlite3.Error:
            logger.error("Query failed: %s", label, exc_info=True)
            continue
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        out.write(f"\n### {label}\n\n")
        out.write("```sql\n" + stmt.strip() + "\n```\n\n")
        out.write(render_table(columns, rows))
        logger.info("%-55s -> %s row(s)", label[:55], len(rows))


def main() -> int:
    configure_logging()
    try:
        conn = load_csv_into_sqlite(RAW_CSV, DB_PATH)
    except (OSError, ValueError, sqlite3.Error):
        logger.error("Could not build the SQLite database", exc_info=True)
        return 1

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    try:
        with RESULTS_MD.open("w", encoding="utf-8") as out:
            out.write("# Part 1 - SQL Analysis Results\n\n")
            out.write(
                "Generated by `part1_data_analytics/run_sql_analysis.py` from "
                "`data/raw/Metro_Interstate_Traffic_Volume.csv`.\n"
            )
            for name in (
                "01_load_and_verify.sql",
                "02_annual_traffic_trends.sql",
                "03_holiday_temperature.sql",
            ):
                run_sql_file(conn, SQL_DIR / name, out)
    except OSError:
        logger.error("Could not write the results file", exc_info=True)
        return 1
    finally:
        conn.close()

    logger.info("SQL analysis written to %s", RESULTS_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
