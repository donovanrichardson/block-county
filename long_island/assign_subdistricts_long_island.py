#!/usr/bin/env python3
"""
Assign tracts to sub-districts within each county-level district using K-Medoids clustering on tract centroids.
Results are inserted into the 'district' table with type=11, geoid=tract geoid, and parent=r<region-parent>r<sub-region-label>.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from normalize.clustering import cluster_and_insert

DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
CENTROID_TABLE = "county_centroids2"
DISTRICT_TABLE = "district_li3"
BLOCK_TABLE = "blocks_2020"

N_CLUSTERS = 4  # Default, can be changed per region

def create_district_table():
    """Create the district table if it does not exist."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{DISTRICT_TABLE} (
                geoid text PRIMARY KEY,
                type text NOT NULL,
                parent text NOT NULL,
                medioid boolean NOT NULL
            );
        """)
        conn.commit()

def get_tracts_on_long_island():
    """Fetch all unique tract geoids (first 11 digits) in the given counties from blocks_2020."""
    county_geoids = ["36103", "36059", "36081", "36047"]  # Suffolk, Nassau, Queens, Kings
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        placeholders = ','.join(['%s'] * len(county_geoids))
        cur.execute(f"""
            SELECT DISTINCT LEFT(geoid20, 11) AS tract_geoid
            FROM {SCHEMA}.{BLOCK_TABLE}
            WHERE LEFT(geoid20, 5) IN ({placeholders})
        """, county_geoids)
        return [row[0] for row in cur.fetchall()]

def fetch_tract_centroids(tract_geoids):
    """Fetch tract centroids and populations from county_centroids2 where type=11."""
    if not tract_geoids:
        return []
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        placeholders = ','.join(['%s'] * len(tract_geoids))
        cur.execute(f"""
            SELECT county_geoid, ST_Y(centroid_geom) AS lat, ST_X(centroid_geom) AS lon, pop, aland, awater
            FROM {SCHEMA}.{CENTROID_TABLE}
            WHERE type = '11' AND county_geoid IN ({placeholders})
        """, tract_geoids)
        return cur.fetchall()


# --- Main process ---
def main():
    create_district_table()
    region_label = "LI"
    total_start = time.time()
    print("Fetching Tracts on Long Island...")
    tracts = get_tracts_on_long_island()
    print(f"  Tracts on Long Island: {len(tracts)}")
    tract_centroids = fetch_tract_centroids(tracts)
    print(f"  Tract centroids fetched: {len(tract_centroids)}")
    # extract below
    cluster_and_insert(tract_centroids, N_CLUSTERS, region_label, DB_CRED, SCHEMA, DISTRICT_TABLE)
    print(f"  Sub-districts assigned and inserted")
    #extract above
    total_elapsed = time.time() - total_start
    print(f"All regions processed in {total_elapsed:.2f} seconds.")
    print("Done. Tract-level sub-district assignments saved.")

if __name__ == "__main__":
    main()
