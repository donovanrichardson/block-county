#!/usr/bin/env python3
"""
Create aland and awater columns on county_centroids2 and populate them
from tracts_2020 by summing tract aland/awater per county.

Behavior:
- Adds two BIGINT columns on `county_centroids2` if they don't already exist:
    - aland
    - awater
- Aggregates values from `tracts_2020` grouped by LEFT(geoid, 5) (county geoid)
  and updates `county_centroids2` where `county_centroids2.geoid = LEFT(tracts_2020.geoid, 5)`.

Usage:
    python normalize/create_county_aland_awater.py [--dry-run]

Options:
    --dry-run   Print the SQL that would be executed and a sample of aggregated
                values, but do not modify the database.

Notes:
- Update DB_CRED to match your database credentials.
- Requires psycopg2 (or psycopg2-binary) to be installed.
"""

import argparse
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

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
TRACT_TABLE = "tracts_2020"


def psql_conn():
    return psycopg2.connect(**DB_CRED)


def add_columns_if_missing(conn):
    """Add aland and awater columns to county_centroids2 if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s AND column_name IN ('aland','awater');
        """, (COUNTY_TABLE,))
        existing = {row[0] for row in cur.fetchall()}
        stmts = []
        if 'aland' not in existing:
            stmts.append(f"ALTER TABLE public.{COUNTY_TABLE} ADD COLUMN aland bigint;")
        if 'awater' not in existing:
            stmts.append(f"ALTER TABLE public.{COUNTY_TABLE} ADD COLUMN awater bigint;")
        for s in stmts:
            print("Executing:", s)
            cur.execute(s)
    conn.commit()


def aggregate_tract_values(conn, sample_limit=10):
    """Return aggregated aland/awater per county (LEFT(geoid,5)).

    Returns a list of dict rows with keys: county_geoid, aland_sum, awater_sum.
    Also returns a small sample for printing.
    """
    sql = f"""
        SELECT LEFT(geoid,5) AS county_geoid,
            SUM(COALESCE(aland,0))::bigint AS aland_sum,
            SUM(COALESCE(awater,0))::bigint AS awater_sum
        FROM public.{TRACT_TABLE}
        GROUP BY LEFT(geoid,5)
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    sample = rows[:sample_limit]
    return rows, sample


def update_county_table(conn):
    """Update county_centroids2 aland/awater using aggregated tracts.

    This performs a single SQL UPDATE with a FROM subquery for efficiency.
    """
    with conn.cursor() as cur:
        update_sql = f"""
            UPDATE public.{COUNTY_TABLE} AS c
            SET aland = agg.aland_sum,
                awater = agg.awater_sum
            FROM (
                SELECT LEFT(geoid,5) AS county_geoid,
                    SUM(COALESCE(aland,0))::bigint AS aland_sum,
                    SUM(COALESCE(awater,0))::bigint AS awater_sum
                FROM public.{TRACT_TABLE}
                GROUP BY LEFT(geoid,5)
            ) AS agg
            WHERE c.county_geoid = agg.county_geoid;
        """
        print("Running UPDATE to populate aland and awater on", COUNTY_TABLE)
        cur.execute(update_sql)
    conn.commit()


def main(argv):
    parser = argparse.ArgumentParser(description="Add and populate aland/awater on county_centroids2")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify the DB; just print what would run")
    args = parser.parse_args(argv)

    print("Connecting to database...")
    conn = psql_conn()

    try:
        if args.dry_run:
            print("DRY RUN: Will not modify the database. Showing sample aggregated results and SQL.")
            rows, sample = aggregate_tract_values(conn)
            print(f"Total counties with tract data: {len(rows)}")
            print("Sample rows:")
            for r in sample:
                print(r)
            print("\nSample UPDATE SQL:\n")
            print("""
            UPDATE public.{COUNTY_TABLE} AS c
            SET aland = agg.aland_sum,
                awater = agg.awater_sum
            FROM (
                SELECT LEFT(geoid,5) AS county_geoid,
                    SUM(COALESCE(aland,0))::bigint AS aland_sum,
                    SUM(COALESCE(awater,0))::bigint AS awater_sum
                FROM public.{TRACT_TABLE}
                GROUP BY LEFT(geoid,5)
            ) AS agg
            WHERE c.county_geoid = agg.county_geoid;
            """)
            return

        # Non-dry run: perform changes
        print("Adding columns if missing...")
        add_columns_if_missing(conn)

        print("Updating county table with aggregated values from tracts...")
        update_county_table(conn)

        print("Done. aland and awater updated on", COUNTY_TABLE)

    finally:
        conn.close()


if __name__ == '__main__':
    main(sys.argv[1:])

