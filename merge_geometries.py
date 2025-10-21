#!/usr/bin/env python3
"""
Merge block-level geometries from public.blocks_2020 into county, tract, and block group geometries.
Creates new tables: county_geom, tract_geom, blockgroup_geom.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from shapely import wkb
from shapely.ops import unary_union
from tqdm import tqdm

DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
BLOCK_TABLE = "blocks_2020"

LEVELS = [
    {"name": "county_geom", "fips_len": 5, "pk": "county_geoid"},
    {"name": "tract_geom", "fips_len": 11, "pk": "tract_geoid"},
    {"name": "blockgroup_geom", "fips_len": 12, "pk": "blockgroup_geoid"},
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.{table} (
    {pk} text PRIMARY KEY,
    geom geometry(MultiPolygon, 4269) NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO {schema}.{table} ({pk}, geom)
SELECT LEFT(geoid20, {fips_len}) AS {pk}, ST_Union(geom)::geometry(MultiPolygon, 4269)
FROM {schema}.{block_table}
GROUP BY {pk}
ON CONFLICT ({pk}) DO UPDATE SET geom = EXCLUDED.geom;
"""

def fetch_blocks_for_level(cur, fips_len):
    cur.execute(f"""
        SELECT LEFT(geoid20, %s) AS fips, geom
        FROM {SCHEMA}.{BLOCK_TABLE}
    """, (fips_len,))
    rows = cur.fetchall()
    # Group by fips
    groups = {}
    for fips, geom in rows:
        if fips not in groups:
            groups[fips] = []
        if geom is not None:
            groups[fips].append(wkb.loads(geom, hex=True))
    return groups

INSERT_PY_SQL = """
INSERT INTO {schema}.{table} ({pk}, geom)
VALUES (%s, ST_GeomFromWKB(%s, 4269))
ON CONFLICT ({pk}) DO UPDATE SET geom = EXCLUDED.geom;
"""

def merge_geometries():
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        for level in LEVELS:
            table = level["name"]
            fips_len = level["fips_len"]
            pk = level["pk"]
            print(f"Creating table {table} if not exists...")
            cur.execute(CREATE_SQL.format(schema=SCHEMA, table=table, pk=pk))
            conn.commit()
            print(f"Fetching blocks for {table}...")
            groups = fetch_blocks_for_level(cur, fips_len)
            print(f"Merging {len(groups)} geometries for {table} (Python union)...")
            for fips, geoms in tqdm(groups.items(), desc=f"Union {table}", unit="group"):
                if not geoms:
                    continue
                unioned = unary_union(geoms)
                # Convert to WKB for insertion
                wkb_bytes = unioned.wkb
                cur.execute(INSERT_PY_SQL.format(schema=SCHEMA, table=table, pk=pk), (fips, wkb_bytes))
            conn.commit()
            print(f"Done: {table}")

def main():
    print("Starting geometry merge...")
    merge_geometries()
    print("All geometry merges complete.")

if __name__ == "__main__":
    main()
