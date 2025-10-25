#!/usr/bin/env python3
"""
Update county_centroids2.aland and awater by copying values from tracts_2020
where county_centroids2.county_geoid matches tracts_2020.geoid (exact match).

Behavior:
- Does NOT create columns (assumes aland/awater already exist on county_centroids2).
- Performs a single efficient UPDATE ... FROM ... WHERE join to update only rows
  that have a matching tract geoid. Rows without a match are left unchanged.
- Has a --dry-run option that prints counts and sample rows but does not modify the DB.

Usage:
    python normalize/update_county_from_tracts.py [--dry-run]

Notes:
- Update DB_CRED to match your environment.
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


def sample_matches(conn, limit=10):
    """Return count of matches and a small sample of matching rows.

    Matches are rows where c.county_geoid = t.geoid. Returns (count, sample_rows).
    sample_rows are dicts with keys: county_geoid, county_geoid_exists, aland, awater
    where aland/awater are from tracts_2020.
    """
    count_sql = f"SELECT COUNT(*) FROM public.{COUNTY_TABLE} AS c JOIN public.{TRACT_TABLE} AS t ON c.county_geoid = t.geoid"
    sample_sql = f"SELECT c.county_geoid, t.geoid AS tract_geoid, t.aland, t.awater FROM public.{COUNTY_TABLE} AS c JOIN public.{TRACT_TABLE} AS t ON c.county_geoid = t.geoid LIMIT %s"
    with conn.cursor() as cur:
        cur.execute(count_sql)
        count = cur.fetchone()[0]
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sample_sql, (limit,))
        sample = cur.fetchall()
    return count, sample


def update_from_tracts(conn):
    """Perform the UPDATE that copies aland and awater from tracts_2020 to county_centroids2
    where county_centroids2.county_geoid = tracts_2020.geoid.

    Returns the number of rows affected.
    """
    update_sql = f"""
        UPDATE public.{COUNTY_TABLE} AS c
        SET aland = t.aland,
            awater = t.awater
        FROM public.{TRACT_TABLE} AS t
        WHERE c.county_geoid = t.geoid
        AND (c.aland IS DISTINCT FROM t.aland OR c.awater IS DISTINCT FROM t.awater);
    """
    with conn.cursor() as cur:
        cur.execute(update_sql)
        affected = cur.rowcount
    conn.commit()
    return affected


def main(argv):
    parser = argparse.ArgumentParser(description="Copy aland/awater from tracts_2020 to county_centroids2 by matching geoid")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify the DB; just print counts and samples")
    args = parser.parse_args(argv)

    print("Connecting to database...")
    conn = psql_conn()

    try:
        count, sample = sample_matches(conn, limit=10)
        print(f"Total matching county rows that have a tract match: {count}")
        if sample:
            print("Sample matching rows (county_geoid, tract_geoid, aland, awater):")
            for r in sample:
                print(r)
        else:
            print("No sample rows found (no matches)")

        if args.dry_run:
            print("DRY RUN: no updates performed.")
            return

        if count == 0:
            print("No matching rows found; nothing to update.")
            return

        print("Performing update: copying aland/awater from matching tracts into county table...")
        affected = update_from_tracts(conn)
        print(f"Update complete. Rows affected (updated): {affected}")

    finally:
        conn.close()


if __name__ == '__main__':
    main(sys.argv[1:])

