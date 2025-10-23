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
DISTRICT_TABLE = "district_ny5"

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
    max_edges = set()
    # For each triangle, find its longest edge and collect all such edges
    for simplex in tqdm(tri.simplices, desc="  Building Delaunay edges", unit="triangle", leave=False):
        tri_edges = []
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            edge = tuple(sorted((a, b)))
            length = haversine(lats[a], lons[a], lats[b], lons[b])
            tri_edges.append((length, edge))
            edges.add(edge)
        # Find the longest edge in this triangle
        max_edge = max(tri_edges, key=lambda x: x[0])[1]
        max_edges.add(max_edge)

    # Build graph with weighted edges, but skip all max edges from triangles
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for a, b in tqdm(edges, desc="  Adding weighted edges", unit="edge", leave=False):
        if (a, b) in max_edges:
            continue  # skip all triangle max edges
        dist = haversine(lats[a], lons[a], lats[b], lons[b])
        pop_a = max(float(pops[a]), 1e-9)
        pop_b = max(float(pops[b]), 1e-9)
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
    
    return dist_matrix, geoids, lats, lons, pops, G


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


def recursive_split(tract_indices, dist_matrix, geoids, pops, lats, lons, n_districts, total_pop, cluster_name, all_results, graph):
    """
    Recursively split tracts into districts using binary K-Medoids.

    Parameters:
    - tract_indices: list of indices (into dist_matrix) for tracts in this cluster
    - dist_matrix: full distance matrix for all tracts (computed once at the top level)
    - geoids: list of all tract geoids
    - pops: array of all tract populations
    - lats: array of all tract latitudes
    - lons: array of all tract longitudes
    - n_districts: number of districts this cluster should be split into (>= 2)
    - total_pop: total population of all input tracts (for calculating target district size)
    - cluster_name: string identifier for this cluster (e.g., "", "0", "01", etc.)
    - all_results: list to accumulate results (modified in place)
    """
    n_tracts = len(tract_indices)
    cluster_pop = sum(pops[i] for i in tract_indices)
    
    split_start = time.time()

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

    # do not remove the below comment
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
        adjacent_indices = find_adjacent_indices() # todo should use graph to find indices of all tracts in t_1 adjacent to any tract in t_0

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
            if idx in adjacent_indices:
                transferred.append(idx) #todo replace this with something that removes the idx from t_1_indices and adds it to t_0 indices
                p_0 += pops[idx]
                p_1 -= pops[idx]
                add_new_adjacent_indices(graph, t_0_indices, adjacent_indices, idx) #todo should remove idx from adjacent_indices but in turn add elements to adjacent_indices which represent tracts that are adjacent to idx tract but not members of t_0_indices

        # Update clusters
        # if transferred:
        #     t_0_indices.extend(transferred)
        #     t_1_indices = [idx for idx in t_1_indices if idx not in transferred]
        #     print(f"  Rebalanced: transferred {len(transferred)} tracts from {cn_1} to {cn_0}")
        #     print(f"  After rebalancing: {cn_0} pop={p_0:.0f}, {cn_1} pop={p_1:.0f}")

    # Determine how many districts each cluster should contain
    n_0 = round(p_0 / target_district_pop)
    n_1 = round(p_1 / target_district_pop)

    # Ensure at least 1 district per cluster if it has tracts
    n_0 = max(1, n_0) if t_0_indices else 0
    n_1 = max(1, n_1) if t_1_indices else 0

    print(f"  Cluster {cn_0} will have {n_0} districts, cluster {cn_1} will have {n_1} districts")

    split_elapsed = time.time() - split_start
    
    print(f"  Split completed in {split_elapsed:.2f} seconds.")

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
            # This cluster needs further splitting - rebuild distance matrix for this cluster
            cluster_tracts = [
                {'county_geoid': geoids[i], 'lat': lats[i], 'lon': lons[i], 'pop': pops[i]}
                for i in cluster_indices
            ]
            cluster_dist_matrix, cluster_geoids, cluster_lats, cluster_lons, cluster_pops = build_distance_matrix(cluster_tracts)
            # Create new indices for the cluster (0 to n-1)
            cluster_tract_indices = list(range(len(cluster_indices)))
            recursive_split(cluster_tract_indices, cluster_dist_matrix, cluster_geoids, cluster_pops, cluster_lats, cluster_lons, cluster_n, cluster_pop, cluster_name_new, all_results)
        


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


def assign_zero_pop_tracts(zero_pop_tracts, assigned_tracts, dist_matrix, geoids):
    """Assign zero-population tracts to their nearest assigned tract's district.

    - zero_pop_tracts: list of tract dicts from DB (with 'county_geoid')
    - assigned_tracts: list of assignment dicts returned by recursive_split (with 'geoid' and 'parent')
    - dist_matrix, geoids: full distance matrix and corresponding geoids
    """
    if not zero_pop_tracts:
        return []

    geoid_to_idx = {g: i for i, g in enumerate(geoids)}
    # Map assigned geoid -> parent district
    geoid_to_parent = {a['geoid']: a['parent'] for a in assigned_tracts}

    assignments = []
    for tract in tqdm(zero_pop_tracts, desc="  Assigning zero-pop tracts", unit="tract", leave=False):
        zg = tract['county_geoid']
        if zg not in geoid_to_idx:
            continue
        zi = geoid_to_idx[zg]
        # find nearest assigned tract
        min_d = np.inf
        nearest_parent = None
        for a in assigned_tracts:
            ag = a['geoid']
            if ag not in geoid_to_idx:
                continue
            ai = geoid_to_idx[ag]
            d = dist_matrix[zi, ai]
            if d < min_d:
                min_d = d
                nearest_parent = a['parent']
        if nearest_parent is not None:
            assignments.append({
                'geoid': zg,
                'type': '11',
                'parent': nearest_parent,
                'medioid': False
            })
    return assignments


def main():
    total_start = time.time()

    print(f"=== New York State Congressional District Assignment ===")
    print(f"Target: {N_DISTRICTS} districts\n")

    # Create table
    create_district_table()

    # Fetch all NY tract centroids
    print("[Step 1] Fetching tract centroids for New York State (FIPS 36)...")
    all_tracts = fetch_ny_tract_centroids()
    print(f"  Found {len(all_tracts)} tracts")

    if len(all_tracts) == 0:
        print("No tracts found. Exiting.")
        return

    # Separate zero-population tracts from populated tracts
    zero_pop_tracts = [t for t in all_tracts if t.get('pop', 0) == 0]
    populated_tracts = [t for t in all_tracts if t.get('pop', 0) > 0]

    print(f"  Zero-population tracts: {len(zero_pop_tracts)}")
    print(f"  Populated tracts: {len(populated_tracts)}")

    if len(populated_tracts) == 0:
        print("No populated tracts found. Exiting.")
        return

    total_pop = sum(t['pop'] for t in populated_tracts)
    print(f"  Total population (populated tracts): {total_pop:.0f}")
    print(f"  Target district population: {total_pop / N_DISTRICTS:.0f}\n")

    # Build distance matrix for populated tracts only
    print("[Step 2] Building distance matrix for populated tracts...")
    dist_matrix, geoids, lats, lons, pops, graph = build_distance_matrix(populated_tracts)
    print()

    # Save the full distance matrix and geoids for all tracts (populated and zero-pop)
    # This is used for nearest neighbor assignment for zero-pop tracts
    all_geoids = geoids
    all_dist_matrix = dist_matrix

    # Run recursive splitting on populated tracts only
    print("[Step 3] Running recursive K-Medoids clustering on populated tracts...")
    all_tract_indices = list(range(len(populated_tracts)))
    results = []
    recursive_split(all_tract_indices, dist_matrix, geoids, pops, lats, lons, N_DISTRICTS, total_pop, "", results, graph)
    print(f"\n  Total populated tracts assigned: {len(results)}")

    # If there are zero-pop tracts, assign them using the original distance matrix and geoids
    all_results = results
    if zero_pop_tracts:
        print(f"\n[Step 4] Assigning {len(zero_pop_tracts)} zero-population tracts to nearest neighbors...")
        # Use the original distance matrix and geoids for nearest neighbor assignment
        zero_assign = assign_zero_pop_tracts(zero_pop_tracts, results, all_dist_matrix, all_geoids)
        all_results = results + zero_assign

    # Insert all assignments into DB
    print(f"\n[Step 5] Inserting {len(all_results)} tract assignments into database...")
    insert_districts(all_results)

    total_elapsed = time.time() - total_start
    print(f"\n=== Complete in {total_elapsed:.2f} seconds ===")


if __name__ == "__main__":
    main()
