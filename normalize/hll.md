create a new script that will create a table called hll. it needs the following columns:

id text primary key
parent text (correspoonds to id of parent geom)
hierarchy text
level text
label text
geom (for polygons), geometry(MultiPolygon,4269)
centroid_geom (for centroids) geometry(Point,4269)
pop numeric(12,0)

all columns need the appropriate indices.

hierarchy, level, label combination must be unique.


## Why and how `county_centroids` is used in `create_hierarchy_start.py`

The `create_hierarchy_start.py` script computes the total US population for the top-level `hll` record (hierarchy='US', level='0') by **joining `county_centroids2` to `counties_2020` on their geoid columns and aggregating SUM(pop) per county**. This approach avoids double-counting when `county_centroids2` contains multiple smaller features per county.

Why this join is necessary:

- `county_centroids2` may contain multiple rows per county (e.g., one row per tract centroid, one row per block group centroid, etc.). Simply summing `pop` from `county_centroids2` would count the same county's population multiple times.
- By joining to `counties_2020` on geoid and aggregating, we ensure one population sum per county (avoiding duplicates).
- This provides a canonical, accurate total population for the US-level hierarchy record.

How the script implements this (implementation details):

1. **Count distinct counties from the join**: the script runs:
   ```sql
   SELECT COUNT(DISTINCT cty.geoid) 
   FROM county_centroids2 cc 
   JOIN counties_2020 cty ON cc.county_geoid = cty.geoid
   ```
   This count is used to seed a tqdm progress bar.

2. **Stream per-county aggregates**: the script opens an unnamed server-side cursor and streams:
   ```sql
   SELECT cty.geoid, SUM(cc.pop)::numeric AS county_pop 
   FROM county_centroids2 cc 
   JOIN counties_2020 cty ON cc.county_geoid = cty.geoid
   GROUP BY cty.geoid 
   ORDER BY cty.geoid
   ```
   The cursor uses `itersize = 1000` to fetch rows in manageable chunks, avoiding loading the entire result into Python memory.

3. **Accumulate per-county sums**: for each streamed row, the script extracts the `county_pop` value, converts to int, and adds it to a running total. The tqdm progress bar advances once per county.

4. **No fallback**: if the join fails (e.g., tables missing, geoid mismatch, or no rows matched), the script raises an error immediately. There is no fallback to other tables or approaches. Both `county_centroids2` and `counties_2020` must exist and have compatible geoid columns.

Why this design (rationale):

- **Correctness**: joining ensures we count each county exactly once, avoiding silent double-counting that could corrupt the hierarchy.
- **Explicitness**: requiring both tables and a successful join makes failures loud and obvious, rather than silently falling back to an incorrect data source.
- **Resource efficiency**: server-side cursors with streaming avoid loading large result sets into Python memory; tqdm provides user-friendly progress feedback.

Notes and prerequisites

- `county_centroids2.county_geoid` (primary key) must match `counties_2020.geoid` for the join to succeed.
- `county_centroids2` must have a `pop` column (integer) containing the population values to aggregate.
- `counties_2020` must have a `geoid` column and a `geom` column for geometry merging.
- If the join returns zero rows, the script will fail with an error (not return a zero total).
