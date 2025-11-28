# Comparison: assign_districts.py vs kmedioids.py.md

This document describes the similarities and differences between the Python script `assign_districts.py` and the algorithm outlined in `kmedioids.py.md`.

## Similarities

1. **Purpose**
   - Both scripts assign counties to districts (or regions) using clustering algorithms, specifically K-Medoids.
   - Both use county centroids and population data as the basis for clustering.

2. **Clustering Algorithm**
   - Both use the K-Medoids algorithm from `sklearn_extra.cluster` to partition counties into a specified number of clusters (districts/regions).
   - Both use a precomputed distance matrix as input to K-Medoids.

3. **Delaunay Triangulation & Graph Construction**
   - Both scripts perform Delaunay triangulation on projected county centroids to define adjacency (edges) between counties.
   - Both build a graph where nodes are counties and edges are Delaunay connections.

4. **Population-Weighted Edge Costs**
   - Both scripts assign edge weights using a population-weighted transformation of the great-circle (haversine) distance between centroids:
     - `weight = (dist / (2 * sqrt(pop_a))) + (dist / (2 * sqrt(pop_b)))`

5. **Shortest-Path Distance Matrix**
   - Both compute the shortest-path distance matrix between all counties using Dijkstra's algorithm on the constructed graph.
   - This matrix is used as the input for K-Medoids clustering.

6. **Cluster Assignment Output**
   - Both assign each county to a district/region and identify which counties are medoids (cluster centers).

## Differences

1. **Data Sources and Formats**
   - `assign_districts.py` fetches data directly from a PostGIS database table (`county_centroids_2`).
   - `kmedioids.py.md` reads data from a GeoPackage file using GeoPandas, and works with a DataFrame.

2. **Geometry Handling**
   - `assign_districts.py` uses SQL queries and `pyproj` for coordinate projection.
   - `kmedioids.py.md` uses GeoPandas for geometry operations and CRS transformations.

3. **Population Column Name**
   - `assign_districts.py` expects a `pop` column in the database.
   - `kmedioids.py.md` expects a `total_population` column in the GeoDataFrame.

4. **Exclusion of Alaska and Hawaii**
   - `kmedioids.py.md` explicitly excludes Alaska and Hawaii by filtering out FIPS codes `02` and `15`.
   - `assign_districts.py` does not perform this exclusion unless the source table is pre-filtered.

5. **Visualization and Export**
   - `kmedioids.py.md` includes code for plotting the results and exporting the labeled regions to a GeoPackage file.
   - `assign_districts.py` does not include visualization or export; it writes results to a database table (`district`).

6. **Command-Line Interface**
   - `kmedioids.py.md` uses `argparse` for command-line arguments and options.
   - `assign_districts.py` is designed to be run as a script with hardcoded parameters.

7. **Error Handling and Progress Reporting**
   - `kmedioids.py.md` provides more robust error handling and progress reporting throughout its workflow:
     - It wraps the main process in a try/except block, printing clear error messages and exiting with a non-zero status on failure.
     - Progress is reported for each major step (reading data, calculating centroids, triangulation, graph building, shortest-path computation, clustering, plotting, and exporting).
     - For computationally intensive steps (e.g., shortest-path matrix calculation), it prints elapsed time and periodic progress updates (such as node processing status during matrix construction).
     - The use of `argparse` allows for graceful handling of invalid or missing command-line arguments.
   - `assign_districts.py` has minimal error handling and progress reporting:
     - It prints basic status messages for major steps (fetching data, creating tables, assigning districts, inserting results).
     - There is no try/except block around the main process, so any uncaught exception will terminate the script with a stack trace.
     - No progress bars or timing information are provided for long-running steps (e.g., Delaunay triangulation, shortest-path computation, clustering, or database inserts).
     - The script does not validate input parameters or handle database connection errors gracefully.
   - **Summary:**
     - `kmedioids.py.md` is better suited for interactive or production use, where feedback and resilience to errors are important.
     - `assign_districts.py` is more basic, suitable for batch or internal use where minimal feedback is acceptable and errors are handled externally.

## Summary Table

| Feature                        | assign_districts.py | kmedioids.py.md  |
|--------------------------------|---------------------|------------------|
| Data source                    | PostGIS table       | GeoPackage       |
| Geometry library               | pyproj, SQL         | GeoPandas        |
| Population column              | pop                 | total_population |
| Delaunay triangulation         | Yes                 | Yes              |
| Population-weighted edge costs | Yes                 | Yes              |
| Shortest-path matrix           | Yes                 | Yes              |
| K-Medoids clustering           | Yes                 | Yes              |
| Medoid identification          | Yes                 | Yes              |
| Output format                  | Database table      | GeoPackage       |
| Visualization                  | No                  | Yes              |
| CLI/arguments                  | No                  | Yes              |
| Exclude AK/HI                  | No                  | Yes              |

## Conclusion

Both scripts implement the same core algorithm for clustering counties into districts using population-weighted, graph-based shortest-path distances and K-Medoids. The main differences are in data handling, output format, and user interface. The database-centric `assign_districts.py` is suited for integration with PostGIS workflows, while `kmedioids.py.md` is more flexible for geospatial analysis and visualization in Python.
