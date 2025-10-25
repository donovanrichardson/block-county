#!/usr/bin/env python3
"""
Clustering utilities for tract sub-district assignment.
"""
import psycopg2
import numpy as np
from sklearn_extra.cluster import KMedoids
import networkx as nx
from scipy.spatial import Delaunay
from pyproj import Transformer
from tqdm import tqdm

# Database configuration (imported from parent module)
DB_CRED = None
SCHEMA = None
DISTRICT_TABLE = None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def cluster_and_insert(tract_centroids, n_clusters, region_label, db_cred, schema, district_table):
    """
    Cluster tracts into sub-districts and insert assignments into the database.
    
    Args:
        tract_centroids: List of tract centroid dictionaries with 'county_geoid', 'lat', 'lon', 'pop'
        n_clusters: Number of clusters to create
        region_label: Label for the region (e.g., "LI")
        db_cred: Database credentials dictionary
        schema: Database schema name
        district_table: District table name
    """
    tract_centroids = [row for row in tract_centroids if row.get('aland', 0) > 0]
    
    # Cluster the tracts
    assignments = _cluster_tracts(tract_centroids, n_clusters, region_label)
    
    # Insert into database
    _insert_subdistricts(assignments, db_cred, schema, district_table)


def _cluster_tracts(tracts, n_clusters, parent_label):
    """Cluster tracts into sub-districts using K-Medoids on Delaunay graph."""
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
    for simplex in tqdm(tri.simplices, desc="Building Delaunay edges", unit="triangle"):
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            edges.add(tuple(sorted((a, b))))
    
    G = nx.Graph()
    for i in tqdm(range(n), desc="Adding nodes to graph", unit="node"):
        G.add_node(i)
    
    for a, b in tqdm(edges, desc="Adding edges to graph", unit="edge"):
        dist = haversine(lats[a], lons[a], lats[b], lons[b])
        # Ensure pop values are Python floats to avoid type warnings with max()
        pop_a = max(float(pops[a]), 1e-9)
        pop_b = max(float(pops[b]), 1e-9)
        weight = ((dist / (2 * np.sqrt(pop_a))) + (dist / (2 * np.sqrt(pop_b))))**2
        # alt_weight = (((dist / (2 * np.sqrt(pop_a))) + (dist / (2 * np.sqrt(pop_b))))**2)/(pop_a+pop_b) #todo do not remove alt_weight comment
        G.add_edge(a, b, weight=weight)

    # For nodes with zero population, keep only their single shortest physical edge (by haversine distance),
    # set that edge's weight to 0 so it doesn't contribute to path lengths, and remove any other incident edges.
    # This ensures zero-pop nodes are attached to the graph by their closest neighbor only.
    for i in tqdm(range(n), desc="Pruning zero-pop nodes", unit="node"):
        if pops[i] <= 0:
            incident = list(G.edges(i, data=True))  # list to avoid mutation issues while removing
            if not incident:
                continue
            # Find the incident edge with the shortest haversine distance
            min_dist = float('inf')
            min_neighbor = None
            for u, v, data in incident:
                neighbor = v if u == i else u
                d = data['weight']
                if d < min_dist:
                    min_dist = d
                    min_neighbor = neighbor
            # Set the chosen edge's weight to 0 (if it exists) and remove all other edges incident to i
            if min_neighbor is not None and G.has_edge(i, min_neighbor):
                # Use networkx helper to set edge attribute to avoid static type warnings
                nx.set_edge_attributes(G, {(i, min_neighbor): {'weight': 0}})
            for u, v, data in incident:
                neighbor = v if u == i else u
                if neighbor == min_neighbor:
                    continue
                if G.has_edge(i, neighbor):
                    G.remove_edge(i, neighbor)

    sp_length = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    dist_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="Building distance matrix (rows)", unit="row"):
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
    # inf_pairs = []
    # for i in tqdm(range(n), desc="Checking for infinite pairs", unit="row"):
    #     for j in range(n):
    #         if i != j and np.isinf(dist_matrix[i, j]):
    #             inf_pairs.append((geoids[i], geoids[j]))
    # if inf_pairs:
    #     print(f"Found {len(inf_pairs)} tract pairs with infinite distance:")
    #     for pair in inf_pairs[:20]:  # Print up to 20 pairs for brevity
    #         print(f"  {pair[0]} <-> {pair[1]}")
    #     if len(inf_pairs) > 20:
    #         print(f"  ...and {len(inf_pairs)-20} more.")
    # else:
    #     print("No tract pairs with infinite distance found.")
    
    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed") # removed random state due to heuristic (non random) initialization DO NOT REMVOE COMMENT
    labels = kmedoids.fit_predict(dist_matrix)
    medoid_indices = kmedoids.medoid_indices_
    medoid_geoids = set([geoids[i] for i in medoid_indices])
    results = []
    for i, geoid in tqdm(list(enumerate(geoids)), desc="Building results", unit="tract"):
        results.append({
            'geoid': geoid,
            'type': '11',
            'parent': f"r{parent_label}r{labels[i]}",
            'medioid': geoid in medoid_geoids
        })
    return results


def _insert_subdistricts(assignments, db_cred, schema, district_table):
    """Insert sub-district assignments into the database."""
    print("[Step] Inserting sub-district assignments into district table...")
    with psycopg2.connect(**db_cred) as conn, conn.cursor() as cur:
        for row in tqdm(assignments, desc="Inserting", unit="tract"):
            cur.execute(f"""
                INSERT INTO {schema}.{district_table} (geoid, type, parent, medioid)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (geoid) DO UPDATE
                SET parent = EXCLUDED.parent, medioid = EXCLUDED.medioid;
            """, (row['geoid'], row['type'], row['parent'], row['medioid']))
        conn.commit()
