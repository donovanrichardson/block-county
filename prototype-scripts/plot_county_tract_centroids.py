#!/usr/bin/env python3
"""
Plot Delaware block geographies merged by county and census tract, with population-weighted county centroids.
- County boundaries: thick lines
- Tract boundaries: thin lines
- Centroids: points
Logs progress for each major step.
"""
import geopandas as gpd
import pandas as pd
import psycopg2
from shapely.geometry import Point
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# SQL to get block geometries and population
BLOCKS_SQL = f"""
SELECT b.geoid20, b.countyfp20, b.tractce20, b.geom, p.pop
FROM {SCHEMA}.{BLOCK_TABLE} b
JOIN {SCHEMA}.{POP_TABLE} p ON b.geoid20 = p.geoid20;
"""

# SQL to get population-weighted centroids
CENTROID_SQL = f"""
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

def log(msg):
    print(f"[LOG] {msg}")

def main():
    log("Connecting to database and loading block data...")
    conn = psycopg2.connect(**DB_CRED)
    # Load blocks as GeoDataFrame
    blocks = gpd.read_postgis(BLOCKS_SQL, conn, geom_col='geom', crs='EPSG:4269')
    log(f"Loaded {len(blocks)} blocks.")

    # Merge blocks by county
    log("Merging blocks by county...")
    counties = blocks.dissolve(by='countyfp20')
    log(f"Merged into {len(counties)} counties.")

    # Merge blocks by tract
    log("Merging blocks by tract...")
    tracts = blocks.dissolve(by=['countyfp20', 'tractce20'])
    log(f"Merged into {len(tracts)} tracts.")

    # Get population-weighted centroids
    log("Fetching population-weighted centroids from database...")
    centroids_df = pd.read_sql(CENTROID_SQL, conn)
    centroids_df['geometry'] = centroids_df.apply(lambda r: Point(r['centroid_x'], r['centroid_y']), axis=1)
    centroids = gpd.GeoDataFrame(centroids_df, geometry='geometry', crs='EPSG:4269')
    log(f"Fetched {len(centroids)} centroids.")
    conn.close()

    # Plot
    log("Plotting counties, tracts, and centroids...")
    fig, ax = plt.subplots(figsize=(10, 10))
    counties.boundary.plot(ax=ax, linewidth=2, edgecolor='black', label='County')
    tracts.boundary.plot(ax=ax, linewidth=0.7, edgecolor='blue', label='Tract')
    centroids.plot(ax=ax, color='red', markersize=50, label='Centroid')
    plt.legend()
    plt.title("Delaware Counties, Tracts, and Population-Weighted Centroids")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    log("Showing plot...")
    plt.show()
    log("Done.")

if __name__ == "__main__":
    main()

