#!/usr/bin/env python3
"""
Assign tracts to sub-districts within each county-level district using K-Medoids clustering on tract centroids.
Results are inserted into the 'district' table with type=11, geoid=tract geoid, and parent=r<region-parent>r<sub-region-label>.
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

DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
CENTROID_TABLE = "county_centroids2"
DISTRICT_TABLE = "district"
BLOCK_TABLE = "blocks_2020"

N_CLUSTERS = 19  # Default, can be changed per region

# --- Utility functions ---
def get_county_regions():
    """Fetch all unique county-level region labels (parent) from the district table."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT DISTINCT parent FROM {SCHEMA}.{DISTRICT_TABLE}
            WHERE type = 'county'
        """)
        return [row['parent'] for row in cur.fetchall()]

def get_counties_in_region(region_label):
    """Fetch all county geoids assigned to a given region label."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT geoid FROM {SCHEMA}.{DISTRICT_TABLE}
            WHERE type = 'county' AND parent = %s
        """, (region_label,))
        return [row['geoid'] for row in cur.fetchall()]

def get_tracts_in_counties(county_geoids):
    """Fetch all unique tract geoids (first 11 digits) in the given counties from blocks_2020."""
    if not county_geoids:
        return []
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
            SELECT county_geoid, ST_Y(centroid_geom) AS lat, ST_X(centroid_geom) AS lon, pop
            FROM {SCHEMA}.{CENTROID_TABLE}
            WHERE type = '11' AND county_geoid IN ({placeholders})
        """, tract_geoids)
        return cur.fetchall()

# --- Clustering logic ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def cluster_tracts(tracts, n_clusters, parent_label):
    if len(tracts) < n_clusters:
        n_clusters = max(1, len(tracts))
    geoids = [row['county_geoid'] for row in tracts]
    lats = np.array([row['lat'] for row in tracts])
    lons = np.array([row['lon'] for row in tracts])
    pops = np.array([row['pop'] for row in tracts])
    n = len(tracts)
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    coords_merc = np.column_stack([xs, ys])
    tri = Delaunay(coords_merc)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            edges.add(tuple(sorted((a, b))))
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for a, b in edges:
        dist = haversine(lats[a], lons[a], lats[b], lons[b])
        pop_a = max(pops[a], 0.5)
        pop_b = max(pops[b], 0.5)
        weight = (dist / (2 * np.sqrt(pop_a))) + (dist / (2 * np.sqrt(pop_b)))
        G.add_edge(a, b, weight=weight)
    sp_length = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = sp_length[i][j] if j in sp_length[i] else np.inf
    # Diagnostic: print largest value in dist_matrix
    print("Max value in dist_matrix:", np.nanmax(dist_matrix))
    # Sample 1000 random entries and print the largest 5
    if n > 1:
        sample_size = min(1000, n * n)
        flat = dist_matrix.flatten()
        sample = np.random.choice(flat, size=sample_size, replace=False)
        largest_5 = np.sort(sample)[-5:]
        print("Largest 5 values in a sample of 1000 from dist_matrix:", largest_5)
    # Find and print tract pairs with infinite distance todo remove this it makes the thing take longer
    inf_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and np.isinf(dist_matrix[i, j]):
                inf_pairs.append((geoids[i], geoids[j]))
    if inf_pairs:
        print(f"Found {len(inf_pairs)} tract pairs with infinite distance:")
        for pair in inf_pairs[:20]:  # Print up to 20 pairs for brevity
            print(f"  {pair[0]} <-> {pair[1]}")
        if len(inf_pairs) > 20:
            print(f"  ...and {len(inf_pairs)-20} more.")
    else:
        print("No tract pairs with infinite distance found.")
    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=1776)
    labels = kmedoids.fit_predict(dist_matrix)
    medoid_indices = kmedoids.medoid_indices_
    medoid_geoids = set([geoids[i] for i in medoid_indices])
    results = []
    for i, geoid in enumerate(geoids):
        results.append({
            'geoid': geoid,
            'type': '11',
            'parent': f"r{parent_label}r{labels[i]}",
            'medioid': geoid in medoid_geoids
        })
    return results

def insert_subdistricts(assignments):
    print("[Step] Inserting sub-district assignments into district table...")
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        for row in tqdm(assignments, desc="Inserting", unit="tract"):
            cur.execute(f"""
                INSERT INTO {SCHEMA}.{DISTRICT_TABLE} (geoid, type, parent, medioid)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (geoid) DO UPDATE
                SET parent = EXCLUDED.parent, medioid = EXCLUDED.medioid;
            """, (row['geoid'], row['type'], row['parent'], row['medioid']))
        conn.commit()

# --- Main process ---
def main():
    print("Starting tract-level sub-district assignment process...")
    total_start = time.time()
    county_regions = get_county_regions()
    print(f"Found {len(county_regions)} county-level regions.")
    for region_label in tqdm(county_regions, desc="Processing regions", unit="region"):
        print(f"\nProcessing region: {region_label}")
        counties = get_counties_in_region(region_label)
        print(f"  Counties in region: {len(counties)}")
        tracts = get_tracts_in_counties(counties)
        print(f"  Tracts in counties: {len(tracts)}")
        tract_centroids = fetch_tract_centroids(tracts)
        print(f"  Tract centroids fetched: {len(tract_centroids)}")
        if not tract_centroids:
            print(f"  No tract centroids found for region {region_label}. Skipping.")
            continue
        assignments = cluster_tracts(tract_centroids, N_CLUSTERS, region_label)
        print(f"  Sub-districts assigned: {len(assignments)}")
        insert_subdistricts(assignments)
    total_elapsed = time.time() - total_start
    print(f"All regions processed in {total_elapsed:.2f} seconds.")
    print("Done. Tract-level sub-district assignments saved.")

if __name__ == "__main__":
    main()
