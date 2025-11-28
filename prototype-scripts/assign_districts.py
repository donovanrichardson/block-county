#!/usr/bin/env python3
"""
Assign counties to districts using K-Medoids clustering on population-weighted centroids.
Creates/updates the 'district' table with county assignments and medoid status.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn_extra.cluster import KMedoids
import networkx as nx
from scipy.spatial import Delaunay
from pyproj import Transformer
from tqdm import tqdm
import time

# Database connection config (edit as needed)
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
CENTROID_TABLE = "county_centroids2"
DISTRICT_TABLE = "district2"

N_CLUSTERS = 19  # Change as needed


def fetch_county_centroids():
    start = time.time()
    print("[Step 1] Fetching county centroids from database...")
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT county_geoid, ST_Y(centroid_geom) AS lat, ST_X(centroid_geom) AS lon, pop
            FROM {SCHEMA}.{CENTROID_TABLE}
            WHERE type = 'county'
        """)
        rows = cur.fetchall()
    elapsed = time.time() - start
    print(f"Fetched {len(rows)} counties in {elapsed:.2f} seconds.")
    return rows

# Haversine function for great-circle distance in kilometers
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


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


def assign_districts(counties):
    print("[Step 2] Projecting centroids to Web Mercator for Delaunay triangulation...")
    start = time.time()
    geoids = [row['county_geoid'] for row in counties]
    lats = np.array([row['lat'] for row in counties])
    lons = np.array([row['lon'] for row in counties])
    pops = np.array([row['pop'] for row in counties])
    n = len(counties)
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    coords_merc = np.column_stack([xs, ys])
    elapsed = time.time() - start
    print(f"Projection completed in {elapsed:.2f} seconds.")

    print("[Step 3] Performing Delaunay triangulation...")
    start = time.time()
    tri = Delaunay(coords_merc)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            edges.add(tuple(sorted((a, b))))
    elapsed = time.time() - start
    print(f"Delaunay triangulation completed in {elapsed:.2f} seconds. {len(edges)} edges created.")

    print("[Step 4] Building graph with population-weighted haversine edge weights...")
    start = time.time()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for a, b in tqdm(list(edges), desc="Adding edges to graph", unit="edge"):
        dist = haversine(lats[a], lons[a], lats[b], lons[b])
        pop_a = pops[a] if pops[a] > 0 else 1e-9
        pop_b = pops[b] if pops[b] > 0 else 1e-9
        weight = ((dist / (2 * np.sqrt(pop_a))) + (dist / (2 * np.sqrt(pop_b))))**2
        G.add_edge(a, b, weight=weight)
    elapsed = time.time() - start
    print(f"Graph built in {elapsed:.2f} seconds.")

    print("[Step 5] Computing shortest-path distance matrix (Dijkstra)...")
    start = time.time()
    sp_length = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    elapsed = time.time() - start
    print(f"Shortest-path computation completed in {elapsed:.2f} seconds.")

    print("[Step 6] Building distance matrix...")
    start = time.time()
    dist_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="Building distance matrix", unit="county"):
        for j in range(n):
            dist_matrix[i, j] = sp_length[i][j] if j in sp_length[i] else np.inf
    elapsed = time.time() - start
    print(f"Distance matrix built in {elapsed:.2f} seconds.")

    print("[Step 7] Running K-Medoids clustering...")
    start = time.time()
    kmedoids = KMedoids(n_clusters=N_CLUSTERS, metric="precomputed", random_state=1776)
    labels = kmedoids.fit_predict(dist_matrix)
    medoid_indices = kmedoids.medoid_indices_
    medoid_geoids = set([geoids[i] for i in medoid_indices])
    elapsed = time.time() - start
    print(f"K-Medoids clustering completed in {elapsed:.2f} seconds.")

    print("[Step 8] Preparing results...")
    results = []
    for i, geoid in enumerate(geoids):
        results.append({
            'geoid': geoid,
            'type': 'county',
            'parent': int(labels[i]),
            'medioid': geoid in medoid_geoids
        })
    print(f"Prepared {len(results)} county assignments.")
    return results


def insert_districts(assignments):
    print("[Step 9] Inserting assignments into district table...")
    start = time.time()
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        for row in tqdm(assignments, desc="Inserting", unit="county"):
            cur.execute(f"""
                INSERT INTO {SCHEMA}.{DISTRICT_TABLE} (geoid, type, parent, medioid)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (geoid) DO UPDATE
                SET parent = EXCLUDED.parent, medioid = EXCLUDED.medioid;
            """, (row['geoid'], row['type'], row['parent'], row['medioid']))
        conn.commit()
    elapsed = time.time() - start
    print(f"Inserted {len(assignments)} assignments in {elapsed:.2f} seconds.")


def main():
    print("Starting county-to-district assignment process...")
    total_start = time.time()
    counties = fetch_county_centroids()
    create_district_table()
    assignments = assign_districts(counties)
    insert_districts(assignments)
    total_elapsed = time.time() - total_start
    print(f"All steps completed in {total_elapsed:.2f} seconds.")
    print("Done. District assignments saved.")

if __name__ == "__main__":
    main()
