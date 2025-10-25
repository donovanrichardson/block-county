#!/usr/bin/env python3
"""
Aggregate aland/awater from blocks_2020 by LEFT(geoid,12) and add these sums
to matching rows in county_centroids2. Processing happens in batches with a
progress bar so you can observe progress.

Behavior:
- Creates a temporary aggregation table in the DB (session-scoped) with
  geoid12, aland_sum, awater_sum.
- Detects whether the county table uses `county_geoid` or `geoid` and uses that
  column as the join key.
- Finds the set of matching geoid12 values present in both the temp table and
  the county table, and updates county rows in batches.
- Each update adds the aggregated aland/awater to the existing values
  (i.e., aland = COALESCE(aland,0) + aland_sum).
- Has --dry-run which prints counts and a small sample but does not modify data.

Usage:
    python normalize/update_county_from_blocks.py [--dry-run] [--batch-size N]

Options:
    --dry-run      Do not perform updates; only print counts and a sample.
    --batch-size   Number of keys to update per batch (default 500).

Notes:
- Update DB_CRED at the top to use your DB credentials.
- Requires psycopg2 and tqdm: `pip install psycopg2-binary tqdm`
"""

import argparse
import math
import sys
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

# ----------------------------
# CONFIG — edit as needed
# ----------------------------
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}

COUNTY_TABLE = "county_centroids2"
BLOCKS_TABLE = "blocks_2020"


def psql_conn():
    return psycopg2.connect(**DB_CRED)


def detect_county_key(conn) -> str:
    """Detect whether county table has 'county_geoid' or 'geoid' and return the column name."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s AND column_name IN ('county_geoid','geoid')
        """, (COUNTY_TABLE,))
        rows = [r[0] for r in cur.fetchall()]
    if 'county_geoid' in rows:
        return 'county_geoid'
    if 'geoid' in rows:
        return 'geoid'
    raise RuntimeError(f"Neither 'county_geoid' nor 'geoid' found on table public.{COUNTY_TABLE}")


def create_temp_aggregation(conn):
    """Removed: no temp table creation. Placeholder kept for API compatibility."""
    # Intentionally left blank: aggregation is performed on-the-fly per batch.
    return


def get_matching_keys(conn, county_key: str) -> List[str]:
    """Return a list of distinct geoid12 values derived from blocks_2020 that match the county table.

    This queries blocks_2020 and joins to the county table on LEFT(geoid,12) = county_key.
    No temp table is created; keys are returned as a list of strings.
    """
    sql = f"""
        SELECT DISTINCT LEFT(b.geoid20, 12) AS geoid12
        FROM public.{BLOCKS_TABLE} b
        JOIN public.{COUNTY_TABLE} c
          ON c.{county_key} = LEFT(b.geoid20, 12)
        ORDER BY geoid12
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def sample_aggregation(conn, limit=10):
    """Return a small sample of aggregated rows computed on-the-fly from blocks_2020.

    This returns a list of dicts with keys: geoid12, aland_sum, awater_sum.
    """
    sql = f"""
        SELECT LEFT(geoid20,12) AS geoid12,
               SUM(COALESCE(aland20,0))::bigint AS aland_sum,
               SUM(COALESCE(awater20,0))::bigint AS awater_sum
        FROM public.{BLOCKS_TABLE}
        GROUP BY LEFT(geoid20,12)
        ORDER BY geoid12
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return cur.fetchall()


def update_batches(conn, county_key: str, keys: List[str], batch_size: int = 500):
    """Update county rows in batches by aggregating blocks_2020 on-the-fly per batch.

    For each batch of geoid12 keys we compute sums from blocks_2020 with a WHERE
    clause filtering to that batch, then join the subquery in the UPDATE.
    """
    total_updated = 0
    total = len(keys)
    if total == 0:
        return 0

    batches = math.ceil(total / batch_size)
    pbar = tqdm(total=total, desc="Updating county rows", unit="rows")
    try:
        for i in tqdm(range(batches), desc="Batches", unit="batch", leave=False):
            start = i * batch_size
            end = start + batch_size
            batch_keys = keys[start:end]

            # Aggregate blocks for this batch on-the-fly and update county rows by join
            update_sql = f"""
                UPDATE public.{COUNTY_TABLE} AS c
                SET aland = COALESCE(c.aland, 0) + agg.aland_sum,
                    awater = COALESCE(c.awater, 0) + agg.awater_sum
                FROM (
                    SELECT LEFT(geoid20,12) AS geoid12,
                           SUM(COALESCE(aland20,0))::bigint AS aland_sum,
                           SUM(COALESCE(awater20,0))::bigint AS awater_sum
                    FROM public.{BLOCKS_TABLE}
                    WHERE LEFT(geoid20,12) = ANY(%s)
                    GROUP BY LEFT(geoid20,12)
                ) AS agg
                WHERE c.{county_key} = agg.geoid12;
            """
            with conn.cursor() as cur:
                cur.execute(update_sql, (batch_keys,))
                affected = int(cur.rowcount or 0)
            conn.commit()
            total_updated += affected
            pbar.update(len(batch_keys))
    finally:
        pbar.close()
    return total_updated


def main(argv):
    parser = argparse.ArgumentParser(description="Aggregate blocks by LEFT(geoid20,12) and add to county_centroids2 in batches")
    parser.add_argument("--dry-run", action="store_true", help="Do not perform updates; only print counts and sample")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of keys to process per batch")
    args = parser.parse_args(argv)

    print("Connecting to database...")
    conn = psql_conn()

    try:
        county_key = detect_county_key(conn)
        print(f"Using county key column: {county_key}")

        total_agg_rows = 0
        # Count distinct aggregated keys directly from the blocks table (no temp table)
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(DISTINCT LEFT(geoid20,12)) FROM public.{BLOCKS_TABLE}")
            total_agg_rows = cur.fetchone()[0]
        print(f"Total distinct geoid12 values present in {BLOCKS_TABLE}: {total_agg_rows}")

        # sample aggregated rows (computed on-the-fly)
        print("Sample aggregated rows (from blocks_2020):")
        for r in sample_aggregation(conn, limit=5):
            print(r)

        matching_keys = get_matching_keys(conn, county_key)
        print(f"Total keys that match county table (to update): {len(matching_keys)}")

        if args.dry_run:
            print("DRY RUN: no updates will be performed.")
            return

        if not matching_keys:
            print("No matching keys to update; exiting.")
            return

        print(f"Updating in batches of {args.batch_size}...")
        updated = update_batches(conn, county_key, matching_keys, batch_size=args.batch_size)
        print(f"Done. Total rows affected (updated): {updated}")

    finally:
        conn.close()


if __name__ == '__main__':
    main(sys.argv[1:])
