```py
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import Delaunay
import networkx as nx
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", default="./data/counties_2023_with_pop.gpkg")
    parser.add_argument("--layer", default="counties_2023")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--out", default="./data/counties_2023_with_regions.gpkg")
    args = parser.parse_args()

    print("Reading data...")
    # Read data
    gdf = gpd.read_file(args.gpkg, layer=args.layer)
    # Exclude Alaska (02) and Hawaii (15)
    gdf = gdf[~gdf["STATEFP"].isin(["02", "15"])].copy()

    print("Calculating centroids...")
    # Get centroids in WGS84
    gdf["centroid"] = gdf.geometry.centroid

    print("Projecting to Web Mercator...")
    # Project to Web Mercator
    gdf = gdf.set_geometry("centroid")
    gdf = gdf.to_crs(3857)
    coords = np.array(list(gdf.geometry.apply(lambda p: (p.x, p.y))))

    print("Performing Delaunay triangulation...")
    # Delaunay triangulation
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            edges.add(tuple(sorted((a, b))))

    print("Building graph with great-circle distances as weights...")
    # Build graph with great-circle distances as weights
    gdf = gdf.to_crs(4326)
    coords_latlon = np.array(list(gdf.geometry.apply(lambda p: (p.y, p.x))))
    def haversine(p1, p2):
        # Returns distance in km
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    G = nx.Graph()
    for i, (lat, lon) in enumerate(coords_latlon):
        G.add_node(i, pos=(lon, lat))
    # Get population array (assume column is 'POPULATION')
    populations = gdf["total_population"].values
    for a, b in edges:
        dist = haversine(coords_latlon[a], coords_latlon[b])
        pop_a = populations[a] if populations[a] > 0 else 0.5
        pop_b = populations[b] if populations[b] > 0 else 0.5
        # Transform the weight as specified
        transformed_weight = (dist / (2 * np.sqrt(pop_a))) + (dist / (2 * np.sqrt(pop_b)))
        G.add_edge(a, b, weight=transformed_weight)

    import time
    print("Computing shortest-path distance matrix...")
    start_time = time.time()
    sp_length = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    elapsed = time.time() - start_time
    print(f"all_pairs_dijkstra_path_length took {elapsed:.2f} seconds")
    n = len(gdf)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i % (n // 20) == 0 and j == 0:
                print(f"Processing node {i**2+j}/{n**2}...")
            dist_matrix[i, j] = sp_length[i][j] if j in sp_length[i] else np.inf

    print("Running K-Medoids clustering...")
    kmedoids = KMedoids(n_clusters=19, metric="precomputed", random_state=1776)
    labels = kmedoids.fit_predict(dist_matrix)
    gdf["region"] = labels

    print("Plotting results...")
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, column="region", categorical=True, legend=True, cmap="tab10", linewidth=0.1, edgecolor="k")
    ax.set_title("County Regions (K-Medoids, k=9)")
    ax.axis("off")
    plt.show()

    # Export if requested
    if args.export:
        print("Exporting results...")
        gdf = gdf.set_geometry("geometry")
        gdf.drop(columns=["centroid"], inplace=True)
        gdf.to_file(args.out, layer=args.layer, driver="GPKG")
        print(f"Exported with region labels to {args.out}")

if __name__ == "__main__":
    try:
        print("Starting k-medoids region assignment process...")
        main()
        print("Process completed successfully.")
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)

```