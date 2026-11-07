"""Database operations and PostgreSQL connectivity for Arzi."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_USER = "postgres"
DEFAULT_MAINTENANCE_DB = "postgres"
TARGET_DB = "arzi"


def get_connection_config(dbname: str | None = None) -> dict[str, Any]:
    """Retrieve database connection settings from standard environment variables."""
    host = os.environ.get("PGHOST", DEFAULT_HOST)
    port_str = os.environ.get("PGPORT", str(DEFAULT_PORT))
    try:
        port = int(port_str)
    except ValueError:
        port = DEFAULT_PORT

    user = os.environ.get("PGUSER", DEFAULT_USER)
    password = os.environ.get("PGPASSWORD")

    # If dbname is explicitly provided, use it; otherwise fall back to PGDATABASE or TARGET_DB
    target_database = dbname if dbname is not None else os.environ.get("PGDATABASE", TARGET_DB)

    config: dict[str, Any] = {
        "host": host,
        "port": port,
        "user": user,
        "dbname": target_database,
    }
    if password is not None:
        config["password"] = password
    return config


@contextmanager
def get_connection(
    dbname: str | None = TARGET_DB, autocommit: bool = False
) -> Iterator[psycopg.Connection]:
    """Provide a managed PostgreSQL connection, defaulting to the application database."""
    config = get_connection_config(dbname=dbname)
    conn = psycopg.connect(**config, autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def get_server_version(conn: psycopg.Connection) -> str:
    """Query and return the PostgreSQL server version string."""
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        row = cur.fetchone()
        return str(row[0]) if row else "Unknown"


def ensure_database_exists(db_name: str = TARGET_DB) -> bool:
    """Ensure the target database exists, creating it via maintenance DB if necessary."""
    with get_connection(dbname=DEFAULT_MAINTENANCE_DB, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,)
            )
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(
                    sql.SQL("CREATE DATABASE {};").format(
                        sql.Identifier(db_name)
                    )
                )
                return False  # Created new
            return True  # Already existed


def apply_schema(conn: psycopg.Connection, schema_path: Path | str) -> None:
    """Read and execute SQL schema definition."""
    path = Path(schema_path)
    sql_text = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_text)


def load_raw_records_copy(
    conn: psycopg.Connection,
    columns: list[str],
    row_iterator: Iterator[tuple[Any, ...]],
) -> int:
    """Load raw complaint records into arzi.complaints_raw using PostgreSQL COPY."""
    col_identifiers = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    query = sql.SQL("COPY arzi.complaints_raw ({}) FROM STDIN").format(
        col_identifiers
    )

    count = 0
    with conn.cursor() as cur:
        with cur.copy(query) as copy:
            for row in row_iterator:
                copy.write_row(row)
                count += 1
    return count


def load_prepared_records_copy(
    conn: psycopg.Connection,
    columns: list[str],
    row_iterator: Iterator[tuple[Any, ...]],
) -> int:
    """Load prepared complaint records into arzi.complaints using PostgreSQL COPY."""
    col_identifiers = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    query = sql.SQL("COPY arzi.complaints ({}) FROM STDIN").format(
        col_identifiers
    )

    count = 0
    with conn.cursor() as cur:
        with cur.copy(query) as copy:
            for row in row_iterator:
                copy.write_row(row)
                count += 1
    return count


def truncate_target_tables(conn: psycopg.Connection) -> None:
    """Safely truncate project tables for repeatable reloading within a transaction."""
    with conn.cursor() as cur:
        cur.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'arzi' AND table_name = 'complaint_topics'
                ) THEN
                    TRUNCATE TABLE arzi.complaint_topics, arzi.complaints, arzi.complaints_raw;
                ELSE
                    TRUNCATE TABLE arzi.complaints, arzi.complaints_raw;
                END IF;
            END $$;
            """
        )


def run_validation_checks(conn: psycopg.Connection) -> dict[str, Any]:
    """Execute validation queries against loaded tables and return structured metrics."""
    metrics: dict[str, Any] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM arzi.complaints_raw;")
        metrics["raw_count"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM arzi.complaints;")
        metrics["prepared_count"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT complaint_id) FROM arzi.complaints;")
        metrics["distinct_ids"] = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM arzi.complaints WHERE complaint_id IS NULL;"
        )
        metrics["missing_ids"] = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT complaint_id FROM arzi.complaints
                GROUP BY complaint_id HAVING COUNT(*) > 1
            ) dup;
            """
        )
        metrics["duplicate_ids"] = cur.fetchone()[0]

        cur.execute(
            "SELECT MIN(date_received), MAX(date_received) FROM arzi.complaints;"
        )
        min_date, max_date = cur.fetchone()
        metrics["min_date"] = str(min_date) if min_date else None
        metrics["max_date"] = str(max_date) if max_date else None

        cur.execute(
            "SELECT COUNT(*) FROM arzi.complaints WHERE has_narrative = TRUE;"
        )
        metrics["narrative_count"] = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COALESCE(SUM(dup_count), 0) FROM (
                SELECT COUNT(*) - 1 AS dup_count
                FROM arzi.complaints
                WHERE has_narrative = TRUE
                GROUP BY consumer_complaint_narrative
                HAVING COUNT(*) > 1
            ) sub;
            """
        )
        metrics["repeated_narratives"] = cur.fetchone()[0]

        cur.execute(
            """
            SELECT
                COUNT(DISTINCT product),
                COUNT(DISTINCT issue),
                COUNT(DISTINCT company)
            FROM arzi.complaints;
            """
        )
        p_count, i_count, c_count = cur.fetchone()
        metrics["distinct_products"] = p_count
        metrics["distinct_issues"] = i_count
        metrics["distinct_companies"] = c_count

        cur.execute(
            """
            SELECT timely_response, COUNT(*)
            FROM arzi.complaints
            GROUP BY timely_response
            ORDER BY count DESC;
            """
        )
        metrics["timely_response_dist"] = cur.fetchall()

    return metrics
