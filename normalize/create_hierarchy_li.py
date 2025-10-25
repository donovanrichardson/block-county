#!/usr/bin/env python3
"""
Create the Long Island (LI) record in the `hll` table by merging specific counties
and summing their populations.

Behavior:
- Computes merged geometry from `public.counties_2020.geom` for counties in COUNTY_GEOIDS
  via a robust ST_Union (makes geometries valid, extracts polygons, forces MULTIPOLYGON, sets SRID 4269).
- Computes total population by joining `public.county_centroids2` to `public.counties_2020`
  on geoid, filtering to counties in COUNTY_GEOIDS, and aggregating SUM(pop) per county
  (to avoid double-counting from smaller centroid features).
- Inserts a single row into `public.hll` with:
    parent = NULL
    hierarchy = 'LI'
    level = '0'
    label = 0 (numeric)
    geom = merged geometry
    centroid_geom = NULL
    pop = total population
  The `id` is generated with `gen_random_uuid()::text` (requires pgcrypto).
- Uses ON CONFLICT on the unique (hierarchy, level, label) constraint to upsert.

Usage:
    python normalize/create_hierarchy_li.py [--dry-run]

Options:
    --dry-run   Print the SQL/steps without modifying the database.

Notes:
- Update DB_CRED at the top to match your environment, or run with env-configured DB.
- Requires psycopg2 and PostGIS; script ensures PostGIS and pgcrypto extensions.
- Requires `county_centroids2` and `counties_2020` tables to exist with matching geoid columns.
"""

import argparse
import sys
import psycopg2
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
SCHEMA = "public"
COUNTIES_TABLE = "counties_2020"
COUNTY_CENTROIDS = "county_centroids2"
HLL_TABLE = "hll"

# Long Island counties (FIPS codes: Nassau, Suffolk, Queens, Brooklyn)
COUNTY_GEOIDS = ["36103", "36059", "36081", "36047"]


def psql_conn():
    return psycopg2.connect(**DB_CRED)


def ensure_extensions(cur):
    # Ensure PostGIS and pgcrypto (for gen_random_uuid) are available
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")


def compute_total_population(cur):
    """Compute total population by joining county_centroids2 to counties_2020 on geoid
    and aggregating SUM(pop) per county to avoid double-counting.

    Streams per-county aggregates with a server-side cursor and shows progress with tqdm.
    Filters to only counties in COUNTY_GEOIDS.
    If the join or tables are missing, raises an error (no fallback).
    
    Note: county_centroids2 uses 'county_geoid' as the primary key (not 'geoid').
    """
    # Count how many distinct counties will be produced by the join, for progress
    cur.execute(
        f"SELECT COUNT(DISTINCT cty.geoid) FROM {SCHEMA}.{COUNTY_CENTROIDS} cc JOIN {SCHEMA}.{COUNTIES_TABLE} cty ON cc.county_geoid = cty.geoid WHERE cty.geoid = ANY(%s);",
        (COUNTY_GEOIDS,)
    )
    total_rows = cur.fetchone()[0] or 0
    
    total = 0
    # stream the per-county aggregated sums to avoid loading everything into memory
    stream_cur = cur.connection.cursor()
    stream_cur.itersize = 1000
    stream_cur.execute(
        f"SELECT cty.geoid, SUM(cc.pop)::numeric AS county_pop FROM {SCHEMA}.{COUNTY_CENTROIDS} cc JOIN {SCHEMA}.{COUNTIES_TABLE} cty ON cc.county_geoid = cty.geoid WHERE cty.geoid = ANY(%s) GROUP BY cty.geoid ORDER BY cty.geoid;",
        (COUNTY_GEOIDS,)
    )
    with tqdm(total=total_rows, desc="Summing population (Long Island counties)", unit="count") as pbar:
        for row in stream_cur:
            county_pop = row[1] if row and row[1] is not None else 0
            total += int(county_pop)
            pbar.update(1)
    stream_cur.close()
    return total


def compute_merged_geom(cur, batch_size=500):
    """Compute merged geometry by performing ST_Union in batches and showing progress.

    Strategy:
    - Fetch county geoids from counties_2020 that are in COUNTY_GEOIDS.
    - Create a temporary table to hold per-batch union geometries.
    - For each batch of geoids, compute a server-side ST_Union(geom) and INSERT the result into the temp table.
    - After processing all batches, ST_Union the temp table geometries into the final geometry.
    """
    # Count total counties matching geoid filter
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{COUNTIES_TABLE} WHERE geoid = ANY(%s);", (COUNTY_GEOIDS,))
    total = cur.fetchone()[0] or 0
    if total == 0:
        raise RuntimeError(f"No counties found in {SCHEMA}.{COUNTIES_TABLE} matching COUNTY_GEOIDS")

    # Fetch all geoids for Long Island
    cur.execute(f"SELECT geoid FROM {SCHEMA}.{COUNTIES_TABLE} WHERE geoid = ANY(%s) ORDER BY geoid;", (COUNTY_GEOIDS,))
    geoids = [r[0] for r in cur.fetchall()]

    # Create a temp table to store batch unions
    cur.execute("CREATE TEMP TABLE temp_geom_union (geom geometry);")

    # Process batches
    with tqdm(total=total, desc="Batch-unioning Long Island counties", unit="count") as pbar:
        for i in range(0, len(geoids), batch_size):
            batch = geoids[i:i+batch_size]
            # compute server-side union for this batch and insert into temp table
            cur.execute(
                f"INSERT INTO temp_geom_union (geom) SELECT ST_MakeValid(ST_Union(geom)) FROM {SCHEMA}.{COUNTIES_TABLE} WHERE geoid = ANY(%s);",
                (batch,)
            )
            pbar.update(len(batch))

    # Final union of batch geometries into a clean MULTIPOLYGON with SRID 4269
    cur.execute(f"SELECT ST_AsBinary(ST_SetSRID(ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_Union(geom)),3)),4269)) FROM temp_geom_union;")
    row = cur.fetchone()
    if not row or row[0] is None:
        raise RuntimeError("Failed to compute merged geometry from Long Island counties")
    return row[0]


def upsert_hll(cur, merged_geom_wkb, total_pop, dry_run=False):
    # Insert or update a single Long Island record. We rely on the unique constraint
    # (hierarchy, level, label) to perform upsert. The constraint name used when
    # creating the table was `{HLL_TABLE}_hierarchy_level_label_unique`.
    insert_sql = f"""
    INSERT INTO {SCHEMA}.{HLL_TABLE} (id, parent, hierarchy, level, label, geom, centroid_geom, pop)
    VALUES (gen_random_uuid()::text, NULL, 'LI', '0', 0, %s, NULL, %s)
    ON CONFLICT ON CONSTRAINT {HLL_TABLE}_hierarchy_level_label_unique
    DO UPDATE SET geom = EXCLUDED.geom, pop = EXCLUDED.pop, parent = EXCLUDED.parent, centroid_geom = EXCLUDED.centroid_geom
    RETURNING id;
    """
    if dry_run:
        print("DRY RUN: the following INSERT would be executed:")
        print(insert_sql)
        print("Parameters: geom=WKB_GEOM, pop=", total_pop)
        return None

    cur.execute(insert_sql, (merged_geom_wkb, total_pop))
    row = cur.fetchone()
    if row:
        return row[0]
    return None


def main(argv):
    parser = argparse.ArgumentParser(description="Create Long Island record in hll table by merging selected counties")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying the DB")
    args = parser.parse_args(argv)

    with psql_conn() as conn, conn.cursor() as cur:
        print("Ensuring required extensions (postgis, pgcrypto) and checking input tables...")
        ensure_extensions(cur)

        # Verify counties table exists
        cur.execute("SELECT to_regclass(%s);", (f"{SCHEMA}.{COUNTIES_TABLE}",))
        if cur.fetchone()[0] is None:
            raise RuntimeError(f"Source counties table {SCHEMA}.{COUNTIES_TABLE} does not exist")

        # Verify hll table exists
        cur.execute("SELECT to_regclass(%s);", (f"{SCHEMA}.{HLL_TABLE}",))
        if cur.fetchone()[0] is None:
            raise RuntimeError(f"Target table {SCHEMA}.{HLL_TABLE} does not exist. Run normalize/create_hll_table.py first.")

        print("Computing total population for Long Island counties...")
        total_pop = compute_total_population(cur)
        print(f"  total_pop = {total_pop}")

        print("Computing merged geometry for Long Island counties...")
        merged_wkb = compute_merged_geom(cur, batch_size=500)

        print("Inserting/updating record in hll table...")
        new_id = upsert_hll(cur, merged_wkb, total_pop, dry_run=args.dry_run)
        if not args.dry_run:
            conn.commit()
            print(f"Inserted/updated hll record (Long Island) with id: {new_id}")
        else:
            print("Dry run complete. No changes were made.")


if __name__ == '__main__':
    main(sys.argv[1:])

