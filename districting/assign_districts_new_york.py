#!/usr/bin/env python3
"""
Create hypothetical congressional districts for New York State using recursive binary K-Medoids clustering.
Fetches tract centroids (type=11) from county_centroids2 for all tracts beginning with state FIPS 36 (New York).
Recursively splits regions into 2 clusters, rebalancing populations, until each cluster represents ~1 district.
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
DISTRICT_TABLE = "district_ny"

# Number of congressional districts to create for New York State
N_DISTRICTS = 26  # Can be changed to any integer >= 2


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


def fetch_ny_tract_centroids():
    """Fetch all tract centroids (type=11) for New York State (county_geoid begins with 36)."""
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT county_geoid, ST_Y(centroid_geom) AS lat, ST_X(centroid_geom) AS lon, pop
            FROM {SCHEMA}.{CENTROID_TABLE}
            WHERE type = '11' AND county_geoid LIKE '36%'
        """)
        return cur.fetchall()


def haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two lat/lon points in kilometers."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_distance_matrix(tracts):
    """
    Build a weighted distance matrix using Delaunay triangulation and Dijkstra shortest paths.
    Returns the distance matrix and tract metadata.
    """
    geoids = [row['county_geoid'] for row in tracts]
    lats = np.array([row['lat'] for row in tracts])
    lons = np.array([row['lon'] for row in tracts])
    pops = np.array([row['pop'] for row in tracts])
    n = len(tracts)
    
    print(f"  Building distance matrix for {n} tracts...")
    
    # Transform to Mercator for Delaunay
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    coords_merc = np.column_stack([xs, ys])
    
    # Build Delaunay triangulation
    tri = Delaunay(coords_merc)
    edges = set()
    for simplex in tqdm(tri.simplices, desc="  Building Delaunay edges", unit="triangle", leave=False):
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            edges.add(tuple(sorted((a, b))))
    
    # Build graph with weighted edges
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    
    for a, b in tqdm(edges, desc="  Adding weighted edges", unit="edge", leave=False):
        dist = haversine(lats[a], lons[a], lats[b], lons[b])
        pop_a = max(pops[a], 1e-9)
        pop_b = max(pops[b], 1e-9)
        weight = ((dist / (2 * np.sqrt(pop_a))) + (dist / (2 * np.sqrt(pop_b))))**2
        G.add_edge(a, b, weight=weight)
    
    # Compute all-pairs shortest paths
    sp_length = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    dist_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="  Building distance matrix", unit="row", leave=False):
        for j in range(n):
            dist_matrix[i, j] = sp_length[i][j] if j in sp_length[i] else np.inf
    
    # Diagnostic output
    max_dist = np.nanmax(dist_matrix[~np.isinf(dist_matrix)]) if np.any(~np.isinf(dist_matrix)) else 0
    print(f"  Max finite distance in matrix: {max_dist:.2e}")
    
    return dist_matrix, geoids, lats, lons, pops


def find_best_medoid(cluster_indices, dist_matrix):
    """
    Find the medoid of a cluster: the point with minimum sum of distances to all other points.
    Returns the index (in the original distance matrix) of the medoid.
    """
    min_sum_dist = np.inf
    best_idx = cluster_indices[0]
    for idx in cluster_indices:
        sum_dist = sum(dist_matrix[idx, j] for j in cluster_indices)
        if sum_dist < min_sum_dist:
            min_sum_dist = sum_dist
            best_idx = idx
    return best_idx


def recursive_split(tract_indices, dist_matrix, geoids, pops, n_districts, total_pop, cluster_name, all_results):
    """
    Recursively split tracts into districts using binary K-Medoids.
    
    Parameters:
    - tract_indices: list of indices (into dist_matrix) for tracts in this cluster
    - dist_matrix: full distance matrix for all tracts (computed once at the top level)
    - geoids: list of all tract geoids
    - pops: array of all tract populations
    - n_districts: number of districts this cluster should be split into (>= 2)
    - total_pop: total population of all input tracts (for calculating target district size)
    - cluster_name: string identifier for this cluster (e.g., "", "0", "01", etc.)
    - all_results: list to accumulate results (modified in place)
    """
    n_tracts = len(tract_indices)
    cluster_pop = sum(pops[i] for i in tract_indices)
    
    print(f"[Split] Cluster '{cluster_name}' has {n_tracts} tracts, pop={cluster_pop:.0f}, target={n_districts} districts")
    
    if n_tracts < 2:
        # Base case: only one tract, assign it
        idx = tract_indices[0]
        all_results.append({
            'geoid': geoids[idx],
            'type': '11',
            'parent': cluster_name if cluster_name else '0',
            'medioid': True  # Single tract is its own medoid
        })
        return
    
    # Extract submatrix for this cluster
    sub_matrix = dist_matrix[np.ix_(tract_indices, tract_indices)]
    
    # Run K-Medoids with k=2 to split into two clusters
    kmedoids = KMedoids(n_clusters=2, metric="precomputed", init="k-medoids++")
    sub_labels = kmedoids.fit_predict(sub_matrix)
    
    # Map back to original indices
    cluster_0_indices = [tract_indices[i] for i in range(n_tracts) if sub_labels[i] == 0]
    cluster_1_indices = [tract_indices[i] for i in range(n_tracts) if sub_labels[i] == 1]
    
    pop_0 = sum(pops[i] for i in cluster_0_indices)
    pop_1 = sum(pops[i] for i in cluster_1_indices)
    
    # Label the smaller cluster as t_0, larger as t_1
    if pop_0 <= pop_1:
        t_0_indices = cluster_0_indices
        t_1_indices = cluster_1_indices
        p_0 = pop_0
        p_1 = pop_1
    else:
        t_0_indices = cluster_1_indices
        t_1_indices = cluster_0_indices
        p_0 = pop_1
        p_1 = pop_0
    
    cn_0 = cluster_name + "0"
    cn_1 = cluster_name + "1"
    
    print(f"  Initial split: cluster {cn_0} pop={p_0:.0f}, cluster {cn_1} pop={p_1:.0f}")

    # Rebalancing step
    # Target district population
    target_district_pop = total_pop / n_districts
    
    # Check if p_0 needs rebalancing
    # "until p_0 reaches at or beyond the next mod (p_t/n)" means:
    # We want p_0 to be a multiple of target_district_pop.
    # If p_0 is not exactly a multiple, we transfer tracts from t_1 to t_0
    # until p_0 >= next_multiple, where next_multiple = ceil(p_0 / target_district_pop) * target_district_pop

    # llm says:
    # It's valid Python but fragile: using `%` with a float and comparing to zero can fail due to floating-point error. Use a tolerance (or `np.isclose`) instead.
    #
    # ```python
    # # use a small tolerance to avoid floating-point equality issues
    # tol = 1e-6
    # if p_0 > 0 and not np.isclose(p_0 % target_district_pop, 0, atol=tol):
    #     # rebalancing logic...
    #     ...
    # ```

    if p_0 % target_district_pop != 0 or p_0 == 0:
        next_multiple = (np.floor(p_0 / target_district_pop)+1) * target_district_pop
        
        # Find medoids for both clusters
        m_0_idx = find_best_medoid(t_0_indices, dist_matrix)
        m_1_idx = find_best_medoid(t_1_indices, dist_matrix)

        # Calculate d_0 and d_1 for all tracts in t_1
        ratios = []
        for idx in t_1_indices:
            d_0 = dist_matrix[idx, m_0_idx]
            d_1 = dist_matrix[idx, m_1_idx]
            if d_0 > 0:
                r_1 = d_1 / d_0
            else:
                r_1 = 0 if d_1 == 0 else np.inf
            ratios.append((r_1, idx))

        # Sort by r_1 descending (highest r_1 = closest to m_0 relative to m_1)
        ratios.sort(reverse=True)

        # Transfer tracts from t_1 to t_0 until p_0 reaches the next multiple
        transferred = []
        for r_1, idx in ratios:
            if p_0 >= next_multiple:
                break
            transferred.append(idx)
            p_0 += pops[idx]
            p_1 -= pops[idx]

        # Update clusters
        if transferred:
            t_0_indices.extend(transferred)
            t_1_indices = [idx for idx in t_1_indices if idx not in transferred]
            print(f"  Rebalanced: transferred {len(transferred)} tracts from {cn_1} to {cn_0}")
            print(f"  After rebalancing: {cn_0} pop={p_0:.0f}, {cn_1} pop={p_1:.0f}")
    
    # Determine how many districts each cluster should contain
    # Round to nearest integer based on population proportion
    n_0 = round(p_0 / target_district_pop)
    n_1 = round(p_1 / target_district_pop)
    
    # Ensure at least 1 district per cluster if it has tracts todo even though all should have tracts and population
    n_0 = max(1, n_0) if t_0_indices else 0
    n_1 = max(1, n_1) if t_1_indices else 0

    print(f"  Cluster {cn_0} will have {n_0} districts, cluster {cn_1} will have {n_1} districts")
    
    # Process each cluster
    for cluster_indices, cluster_pop, cluster_n, cluster_name_new in [
        (t_0_indices, p_0, n_0, cn_0),
        (t_1_indices, p_1, n_1, cn_1)
    ]:
        if not cluster_indices:
            continue

        if cluster_n == 1:
            # This cluster represents exactly 1 district, finalize it
            medoid_idx = find_best_medoid(cluster_indices, dist_matrix)
            for idx in tqdm(cluster_indices, desc=f"  Finalizing district {cluster_name_new}", unit="tract", leave=False):
                all_results.append({
                    'geoid': geoids[idx],
                    'type': '11',
                    'parent': cluster_name_new,
                    'medioid': (idx == medoid_idx)
                })
        else:
            # This cluster needs further splitting
            recursive_split(cluster_indices, dist_matrix, geoids, pops, cluster_n, total_pop, cluster_name_new, all_results)


def insert_districts(assignments):
    """Insert district assignments into the database."""
    print(f"\n[Insert] Inserting {len(assignments)} tract assignments into {DISTRICT_TABLE}...")
    with psycopg2.connect(**DB_CRED) as conn, conn.cursor() as cur:
        for row in tqdm(assignments, desc="Inserting", unit="tract"):
            cur.execute(f"""
                INSERT INTO {SCHEMA}.{DISTRICT_TABLE} (geoid, type, parent, medioid)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (geoid) DO UPDATE
                SET parent = EXCLUDED.parent, medioid = EXCLUDED.medioid, type = EXCLUDED.type;
            """, (row['geoid'], row['type'], row['parent'], row['medioid']))
        conn.commit()
    print("Insert complete.")


def main():
    total_start = time.time()
    
    print(f"=== New York State Congressional District Assignment ===")
    print(f"Target: {N_DISTRICTS} districts\n")
    
    # Create table
    create_district_table()
    
    # Fetch all NY tract centroids
    print("[Step 1] Fetching tract centroids for New York State (FIPS 36)...")
    tracts = fetch_ny_tract_centroids()
    print(f"  Found {len(tracts)} tracts")
    
    if len(tracts) == 0:
        print("No tracts found. Exiting.")
        return
    
    total_pop = sum(row['pop'] for row in tracts)
    print(f"  Total population: {total_pop:.0f}")
    print(f"  Target district population: {total_pop / N_DISTRICTS:.0f}\n")
    
    # Build distance matrix (computed once, reused throughout recursion)
    print("[Step 2] Building distance matrix...")
    dist_matrix, geoids, lats, lons, pops = build_distance_matrix(tracts)
    print()
    
    # Run recursive splitting
    print("[Step 3] Running recursive K-Medoids clustering...")
    all_tract_indices = list(range(len(tracts)))
    results = []
    recursive_split(all_tract_indices, dist_matrix, geoids, pops, N_DISTRICTS, total_pop, "", results)
    print(f"\n  Total tracts assigned: {len(results)}")
    
    # Insert into database
    print("\n[Step 4] Inserting results into database...")
    insert_districts(results)
    
    total_elapsed = time.time() - total_start
    print(f"\n=== Complete in {total_elapsed:.2f} seconds ===")


if __name__ == "__main__":
    main()

