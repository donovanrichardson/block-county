#!/usr/bin/env python3
"""
Compute population-weighted centroid for each Delaware county using block-level population and geometry.

Connects to the same database as llm.py.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

# Use the same DB_CRED as llm.py
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
BLOCK_TABLE = "de_blocks_2020"
POP_TABLE = "de_block_pop_2020"

SQL = f"""
WITH weighted AS (
  SELECT
    b.countyfp20,
    MIN(LEFT(b.geoid20, 5)) AS county_geoid,
    SUM(ST_X(ST_Centroid(b.geom)) * p.pop)::float AS weighted_x,
    SUM(ST_Y(ST_Centroid(b.geom)) * p.pop)::float AS weighted_y,
    SUM(p.pop) AS pop
  FROM {SCHEMA}.{BLOCK_TABLE} b
  JOIN {SCHEMA}.{POP_TABLE} p ON b.geoid20 = p.geoid20
  GROUP BY b.countyfp20
)
SELECT
  countyfp20,
  county_geoid,
  ST_X(ST_SetSRID(ST_MakePoint(weighted_x / pop, weighted_y / pop), 4269)) AS centroid_x,
  ST_Y(ST_SetSRID(ST_MakePoint(weighted_x / pop, weighted_y / pop), 4269)) AS centroid_y,
  pop
FROM weighted
ORDER BY countyfp20;
"""

def main():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CRED)
    table_name = "county_centroids2"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        print(f"Creating {table_name} table if not exists...")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{table_name} (
                county_geoid text PRIMARY KEY,
                centroid_geom geometry(Point, 4269) NOT NULL,
                pop integer NOT NULL
            );
        """)
        conn.commit()
        print("Calculating population-weighted centroids and populations for each county...")
        cur.execute(SQL)
        results = cur.fetchall()
        print(f"Inserting/updating {len(results)} county centroids...")
        for row in tqdm(results, desc="Saving centroids", unit="county"):
            county_geoid = row['county_geoid']
            centroid_x = row['centroid_x']
            centroid_y = row['centroid_y']
            pop = row['pop']
            cur.execute(f"""
                INSERT INTO {SCHEMA}.{table_name} (county_geoid, centroid_geom, pop)
                VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4269), %s)
                ON CONFLICT (county_geoid) DO UPDATE
                SET centroid_geom = EXCLUDED.centroid_geom, pop = EXCLUDED.pop;
            """, (county_geoid, centroid_x, centroid_y, pop))
        conn.commit()
        print(f"\nCounty population-weighted centroids saved to {table_name} table:")
        for row in results:
            print(f"County {row['county_geoid']}: centroid_x={row['centroid_x']:.6f}, centroid_y={row['centroid_y']:.6f}, pop={row['pop']}")
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
