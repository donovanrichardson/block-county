#!/usr/bin/env python3
"""
Compute population-weighted centroid for each Delaware county using block-level population and geometry.

Connects to the same database as llm.py.
"""
import psycopg2
from psycopg2.extras import RealDictCursor

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
SELECT
  b.countyfp20,
  ST_X(ST_SetSRID(ST_MakePoint(
    SUM(ST_X(ST_Centroid(b.geom)) * p.pop)::float / SUM(p.pop),
    SUM(ST_Y(ST_Centroid(b.geom)) * p.pop)::float / SUM(p.pop)
  ), 4269)) AS centroid_x,
  ST_Y(ST_SetSRID(ST_MakePoint(
    SUM(ST_X(ST_Centroid(b.geom)) * p.pop)::float / SUM(p.pop),
    SUM(ST_Y(ST_Centroid(b.geom)) * p.pop)::float / SUM(p.pop)
  ), 4269)) AS centroid_y
FROM {SCHEMA}.{BLOCK_TABLE} b
JOIN {SCHEMA}.{POP_TABLE} p ON b.geoid20 = p.geoid20
GROUP BY b.countyfp20
ORDER BY b.countyfp20;
"""

def main():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CRED)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        print("Calculating population-weighted centroids for each county...")
        cur.execute(SQL)
        results = cur.fetchall()
        print("\nCounty population-weighted centroids:")
        for row in results:
            print(f"County {row['countyfp20']}: centroid_x={row['centroid_x']:.6f}, centroid_y={row['centroid_y']:.6f}")
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()

