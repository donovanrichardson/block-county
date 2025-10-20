#!/usr/bin/env python3
"""
Compute population-weighted centroid for each county using block-level population and geometry.

Connects to the same database as block_import.py.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

# Use the same DB_CRED as block_import.py
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
BLOCK_TABLE = "blocks_2020"
POP_TABLE = "block_pop_2020"

# --- Only process counties missing a centroid ---
# This query will be built dynamically to only include counties not yet in the centroid table.
SQL_TEMPLATE = """
WITH weighted AS (
  SELECT
    LEFT(b.geoid20, 5) AS county_geoid,
    SUM(ST_X(ST_Centroid(b.geom)) * b.pop20)::float AS weighted_x,
    SUM(ST_Y(ST_Centroid(b.geom)) * b.pop20)::float AS weighted_y,
    SUM(b.pop20) AS pop
  FROM {schema}.{block} b
  WHERE LEFT(b.geoid20, 5) IN ({placeholders})
  GROUP BY county_geoid
)
SELECT
  county_geoid,
  ST_X(ST_SetSRID(ST_MakePoint(weighted_x / pop, weighted_y / pop), 4269)) AS centroid_x,
  ST_Y(ST_SetSRID(ST_MakePoint(weighted_x / pop, weighted_y / pop), 4269)) AS centroid_y,
  pop
FROM weighted
ORDER BY county_geoid;
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

        # --- Find counties missing a centroid ---
        # Use LEFT(geoid20, 5) as the unique county identifier
        print("Checking for counties with missing centroids...")
        cur.execute(f'''
            SELECT county_geoid FROM (
                SELECT DISTINCT LEFT(geoid20, 5) AS county_geoid
                FROM {SCHEMA}.{BLOCK_TABLE}
            ) sub
            WHERE county_geoid NOT IN (SELECT county_geoid FROM {SCHEMA}.{table_name})
        ''')
        missing_counties = [row['county_geoid'] for row in cur.fetchall()]
        # Comment: Only counties not present in county_centroids2 will be processed
        if not missing_counties:
            print("All counties already have centroids in the table. Nothing to do.")
            conn.close()
            return

        print(f"Found {len(missing_counties)} counties missing centroids: {missing_counties}")

        # --- Build parameterized query for missing counties ---
        placeholders = ",".join(["%s"] * len(missing_counties))
        sql = SQL_TEMPLATE.format(
            schema=SCHEMA,
            block=BLOCK_TABLE,
            pop=POP_TABLE,
            placeholders=placeholders,
        )
        print("Calculating population-weighted centroids and populations for missing counties...")
        cur.execute(sql, missing_counties)
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
# --- End of script ---
# Comments added to explain changes: Only counties missing a centroid are processed, using LEFT(geoid20, 5) as the unique county identifier for safety and correctness.
