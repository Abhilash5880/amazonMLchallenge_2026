"""
Candidate generation and disk-backed blocking for the
Amazon ML Challenge 2026 entity-resolution pipeline.

The production implementation uses a SQLite-backed index
for Source 2 and Source 3 so that the full datasets do not
need to remain in RAM.
"""

from collections import defaultdict
from pathlib import Path
import sqlite3
import time

import pandas as pd

from .preprocessing import (
    prepare_small,
    add_block_keys,
)


# ============================================================
# BLOCKING CONFIGURATION
# ============================================================

BLOCK_COLUMNS = {
    "name": "country_name_key",
    "addr": "country_addr_key",
    "name_addr": "country_name_addr_key",
}

INDEX_READ_CHUNK = 100_000
SQL_KEY_BATCH_SIZE = 1_000


# ============================================================
# SQLITE INDEX
# ============================================================

def create_index_database(index_path: Path) -> sqlite3.Connection:
    """
    Create the disk-backed SQLite database used for
    Source 2 / Source 3 candidate lookup.
    """

    index_path = Path(index_path)

    if index_path.exists():
        index_path.unlink()

    conn = sqlite3.connect(index_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=FILE")
    cur.execute("PRAGMA cache_size=-262144")

    cur.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source INTEGER NOT NULL,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,
            country TEXT NOT NULL,
            country_name_key TEXT NOT NULL,
            country_addr_key TEXT NOT NULL,
            country_name_addr_key TEXT NOT NULL
        )
    """)

    conn.commit()

    return conn


def build_test_index(
    conn: sqlite3.Connection,
    path: Path,
    source_number: int,
    chunk_size: int = INDEX_READ_CHUNK,
) -> None:
    """
    Stream a Source 2 or Source 3 TSV into the SQLite
    blocking index.
    """

    insert_sql = """
        INSERT INTO records (
            entity_id,
            source,
            business_name,
            business_address,
            country,
            country_name_key,
            country_addr_key,
            country_name_addr_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )

    total = 0
    t0 = time.time()

    for chunk in reader:

        chunk = prepare_small(chunk)
        chunk = add_block_keys(chunk)

        rows = zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["business_address"],
            chunk["country"],
            chunk["country_name_key"],
            chunk["country_addr_key"],
            chunk["country_name_addr_key"],
        )

        conn.executemany(
            insert_sql,
            [
                (
                    entity_id,
                    source_number,
                    business_name,
                    business_address,
                    country,
                    country_name_key,
                    country_addr_key,
                    country_name_addr_key,
                )
                for (
                    entity_id,
                    business_name,
                    business_address,
                    country,
                    country_name_key,
                    country_addr_key,
                    country_name_addr_key,
                ) in rows
            ],
        )

        total += len(chunk)

        conn.commit()

        del chunk

    print(
        f"Indexed S{source_number}: "
        f"{total:,} rows | "
        f"{(time.time() - t0) / 60:.2f} min"
    )


def create_blocking_indexes(
    conn: sqlite3.Connection,
) -> None:
    """
    Create SQLite lookup indexes for the three blocking
    strategies for both Source 2 and Source 3.
    """

    conn.execute(
        "CREATE INDEX idx_s2_name "
        "ON records(source, country_name_key)"
    )

    conn.execute(
        "CREATE INDEX idx_s2_addr "
        "ON records(source, country_addr_key)"
    )

    conn.execute(
        "CREATE INDEX idx_s2_name_addr "
        "ON records(source, country_name_addr_key)"
    )

    conn.execute(
        "CREATE INDEX idx_s3_name "
        "ON records(source, country_name_key)"
    )

    conn.execute(
        "CREATE INDEX idx_s3_addr "
        "ON records(source, country_addr_key)"
    )

    conn.execute(
        "CREATE INDEX idx_s3_name_addr "
        "ON records(source, country_name_addr_key)"
    )

    conn.commit()


# ============================================================
# CANDIDATE LOOKUP
# ============================================================

def query_candidates_for_keys(
    conn: sqlite3.Connection,
    source: int,
    column: str,
    keys,
):
    """
    Retrieve Source 2 or Source 3 records matching a batch
    of blocking keys.
    """

    keys = [key for key in keys if key]

    if not keys:
        return []

    placeholders = ",".join(
        "?" for _ in keys
    )

    sql = f"""
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            {column}
        FROM records
        WHERE source = ?
          AND {column} IN ({placeholders})
    """

    return conn.execute(
        sql,
        [source, *keys],
    ).fetchall()