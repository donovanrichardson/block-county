#!/usr/bin/env python3
"""
Assign counties to districts using K-Medoids clustering on population-weighted centroids.
Creates/updates the 'district' table with county assignments and medoid status.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn_extra.cluster import KMedoids

# Database connection config (edit as needed)
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
CENTROID_TABLE = "county_centroids_2"
DISTRICT_TABLE = "district"

N_CLUSTERS = 19  # Change as needed


def fetch_county_centroids():
    """Fetch county geoid, centroid_geom, and pop for type = 'county'."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT county_geoid, centroid_geom, pop
            FROM {SCHEMA}.{CENTROID_TABLE}
            WHERE type = 'county'
        """)
        rows = cur.fetchall()
    return rows


def create_district_table():
    """Create the district table if it does not exist."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{DISTRICT_TABLE} (
                geoid text PRIMARY KEY,
                type text NOT NULL,
                parent integer NOT NULL,
                medioid boolean NOT NULL
            );
        """)
        conn.commit()


def assign_districts(counties):
    """Run K-Medoids clustering and return assignments and medoid status."""
    # Extract coordinates from centroid_geom (WKT to x/y)
    coords = []
    geoids = []
    for row in counties:
        geoids.append(row['county_geoid'])
        # Use ST_X and ST_Y from PostGIS, but here we fetch as WKT
        # So we need to get x/y from centroid_geom
        # Use psycopg2's geometry adaptation or ask for ST_X/ST_Y in SQL
        # For simplicity, fetch x/y in SQL
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT county_geoid, ST_X(centroid_geom) AS x, ST_Y(centroid_geom) AS y
            FROM {SCHEMA}.{CENTROID_TABLE}
            WHERE type = 'county'
        """)
        rows = cur.fetchall()
        coords = np.array([[row['x'], row['y']] for row in rows])
        geoids = [row['county_geoid'] for row in rows]
    # Run K-Medoids
    kmedoids = KMedoids(n_clusters=N_CLUSTERS, random_state=1776)
    labels = kmedoids.fit_predict(coords)
    medoid_indices = kmedoids.medoid_indices_
    medoid_geoids = set([geoids[i] for i in medoid_indices])
    # Prepare results
    results = []
    for i, geoid in enumerate(geoids):
        results.append({
            'geoid': geoid,
            'type': 'county',
            'parent': int(labels[i]),
            'medioid': geoid in medoid_geoids
        })
    return results


def insert_districts(assignments):
    """Insert assignments into the district table."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        for row in assignments:
            cur.execute(f"""
                INSERT INTO {SCHEMA}.{DISTRICT_TABLE} (geoid, type, parent, medioid)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (geoid) DO UPDATE
                SET parent = EXCLUDED.parent, medioid = EXCLUDED.medioid;
            """, (row['geoid'], row['type'], row['parent'], row['medioid']))
        conn.commit()


def main():
    print("Fetching county centroids...")
    counties = fetch_county_centroids()
    print(f"Fetched {len(counties)} counties.")
    print("Creating district table if needed...")
    create_district_table()
    print("Assigning districts using K-Medoids...")
    assignments = assign_districts(counties)
    print(f"Inserting {len(assignments)} assignments into district table...")
    insert_districts(assignments)
    print("Done. District assignments saved.")

if __name__ == "__main__":
    main()

