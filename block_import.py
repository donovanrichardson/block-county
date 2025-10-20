#!/usr/bin/env python3
"""
Download 2020 census blocks, load to PostGIS, then fetch 2020 PL block population and load to a joinable table.

Tables created:
  - public.blocks_2020 (geometry + TIGER attributes; PK = geoid20)
  - public.block_pop_2020 (geoid20, pop)

Join key:
  - geoid20 (state+county+tract+block; unique per block)

Indexes created with rationale in SQL comments.
"""

import os
import io
import csv
import sys
import json
import time
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

import requests
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

import county_pop_weighted_centroid  # Import the centroid calculation script

# ----------------------------
# CONFIG — edit as needed
# ----------------------------
PG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "gis",
    "user": "postgres",
    "password": "postgres",
}

# Database credentials used by psycopg2 and ogr2ogr. Keep separate so both
# the Python DB connection and the ogr2ogr PG: connection string use the
# same values and can be changed in one place.
DB_CRED = {
    "host": PG["host"],
    "port": PG["port"],
    "dbname": "block-county",
    "user": "block-county",  # must be superuser or have needed privileges
    "password": "your_password_here",
}
SCHEMA = "public"
BLOCK_TABLE = "blocks_2020"
POP_TABLE = "block_pop_2020"
# Keep native shapefile CRS (NAD83 / EPSG:4269). Change to "EPSG:5070" if you want Albers.
TARGET_EPSG = "EPSG:4269"



# 2020 PL (DEC/PL) API: total population is P1_001N
# We request NAME for debugging/spot checks; keys returned: NAME, P1_001N, state, county, tract, block
PL_API_URL = "https://api.census.gov/data/2020/dec/pl"

# ----------------------------
# Helpers
# ----------------------------
def psql_conn():
    # Use DB_CRED for connection parameters
    return psycopg2.connect(
        host=DB_CRED["host"],
        port=DB_CRED["port"],
        dbname=DB_CRED["dbname"],
        user=DB_CRED["user"],
        password=DB_CRED["password"],
    )

def run_ogr2ogr(shp_path, table_fullname, epsg_target):
    """
    Load shapefile into PostGIS using ogr2ogr.
    -nlt MULTIPOLYGON because blocks are polygons; TIGER block geometries are polygons/multipolygons.
    -lco GEOMETRY_NAME=geom ensures column is 'geom' (conventional).
    -nln sets output table name.
    -progress for feedback.
    """
    # Build the ogr2ogr PostgreSQL connection string from DB_CRED
    pg_conn = (
        f"PG:host={DB_CRED['host']} port={DB_CRED['port']} dbname={DB_CRED['dbname']} "
        f"user={DB_CRED['user']} password={DB_CRED['password']}"
    )
    cmd = [
        "ogr2ogr",
        "-append",
        "-f", "PostgreSQL",
        pg_conn,
        shp_path,
        "-nln", table_fullname,
        "-lco", "GEOMETRY_NAME=geom",
        "-nlt", "MULTIPOLYGON",
        "-lco", "FID=gid",
        "-progress"
    ] #todo Warning 1: Layer creation options ignored since an existing layer is being appended to.
    if epsg_target:
        cmd += ["-t_srs", epsg_target]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_schema_and_extensions():
    with psql_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.commit()

def create_block_indexes():
    """
    Add constraints & indexes on the geometry table with rationale.

    Index plan:
    - PRIMARY KEY on geoid20:
        * GEOID20 is unique per block feature and is the natural join key to block-level attributes.
        * Using it as PK guarantees uniqueness and accelerates equality joins.
    - GiST on geom:
        * A spatial index enabling fast bounding-box filters and spatial joins (ST_Intersects, ST_DWithin, etc.).
    - BTREE on (countyfp20, tractce20):
        * Common filter/group-by dimensions when working within a county or tract; improves planner choices for those queries.
    """
    with psql_conn() as conn, conn.cursor() as cur:
        tbl = f"{SCHEMA}.{BLOCK_TABLE}"
        # Ensure geoid20 is text
        cur.execute(f"""
            ALTER TABLE {tbl}
            ALTER COLUMN geoid20 TYPE text;
        """)
        # Add PK on geoid20 if not exists
        cur.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema = '{SCHEMA}' AND table_name = '{BLOCK_TABLE}' AND constraint_type = 'PRIMARY KEY'
                ) THEN
                    ALTER TABLE {tbl}
                    ADD CONSTRAINT {BLOCK_TABLE}_pk PRIMARY KEY (geoid20);
                END IF;
            END$$;
        """)
        # Spatial index on geom
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {BLOCK_TABLE}_geom_gix
            ON {tbl}
            USING GIST (geom);
            COMMENT ON INDEX {BLOCK_TABLE}_geom_gix
            IS 'GiST spatial index on geometry: accelerates spatial filters (e.g., ST_Intersects, ST_DWithin).';
        """)
        # Composite BTREE index for common admin filters
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {BLOCK_TABLE}_county_tract_idx
            ON {tbl} (countyfp20, tractce20);
            COMMENT ON INDEX {BLOCK_TABLE}_county_tract_idx
            IS 'BTREE on (countyfp20, tractce20): speeds filtering/grouping within county/tract boundaries.';
        """)
        # Index on blockce20
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {BLOCK_TABLE}_blockce20_idx
            ON {tbl} (blockce20);
            COMMENT ON INDEX {BLOCK_TABLE}_blockce20_idx
            IS 'BTREE on blockce20: enables fast filtering/grouping by block code within tracts/counties.';
        """)
        # Index on aland20
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {BLOCK_TABLE}_aland20_idx
            ON {tbl} (aland20);
            COMMENT ON INDEX {BLOCK_TABLE}_aland20_idx
            IS 'BTREE on aland20: enables fast filtering/sorting by land area.';
        """)
        conn.commit()

def get_county_codes():
    """
    Query the database for all distinct county codes in the block table. Returns a list of countyfp20 strings.
    If the population table does not exist, falls back to a simple distinct query.
    """
    with psql_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(f"SELECT DISTINCT countyfp20 FROM {SCHEMA}.{BLOCK_TABLE} bt WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.{POP_TABLE} pt WHERE bt.geoid20 = pt.geoid20) order by countyfp20 desc;")
            return [row[0] for row in cur.fetchall()]
        except psycopg2.errors.UndefinedTable:
            conn.rollback()  # Rollback the failed transaction
            cur.execute(f"SELECT DISTINCT countyfp20 FROM {SCHEMA}.{BLOCK_TABLE};")
            return [row[0] for row in cur.fetchall()]

def fetch_block_population(county_code, fips):
    """
    Pull block-level population from the 2020 DEC/PL API for a specific county in a specific state.  
    We construct geoid20 = state + county + tract + block and store P1_001N as pop.
    """
    print(f"Fetching block-level population for county {county_code} from 2020 DEC/PL API…")
    params = {
        "get": "NAME,P1_001N",
        "for": "block:*",
        "in": f"state:{fips}+county:{county_code}",
    }
    r = requests.get(PL_API_URL, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()  # first row is header
    header = rows[0]
    idx_name = header.index("NAME")
    idx_pop = header.index("P1_001N")
    idx_state = header.index("state")
    idx_county = header.index("county")
    idx_tract = header.index("tract")
    idx_block = header.index("block")

    data = []
    for row in tqdm(rows[1:], desc=f"Processing population records for county {county_code}", unit="block"):
        state = row[idx_state]
        county = row[idx_county]
        tract = row[idx_tract]
        block = row[idx_block]
        geoid20 = f"{state}{county}{tract}{block}"
        pop = int(row[idx_pop])
        data.append((geoid20, pop))
    return data

def load_population_table(pop_rows):
    """
    Create the population table if it does not exist and load values.
    Also add PK + BTREE index with rationale.
    Returns the number of records inserted and time elapsed in seconds.
    """
    import time
    start_time = time.time()
    with psql_conn() as conn, conn.cursor() as cur:
        tbl = f"{SCHEMA}.{POP_TABLE}"
        # Only create the table if it does not exist
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {tbl}(
                geoid20 text PRIMARY KEY,
                pop integer NOT NULL
            );
            COMMENT ON TABLE {tbl} IS '2020 PL (DEC/PL) total population (P1_001N) at block level for each state; keyed by geoid20 for joins to de_blocks_2020.';
            COMMENT ON COLUMN {tbl}.geoid20 IS 'Census Block GEOID20 (state+county+tract+block)';
            COMMENT ON COLUMN {tbl}.pop IS 'Total population (P1_001N) from 2020 DEC/PL.';
        """)
        # Bulk insert with tqdm progress
        chunk_size = 10000
        total = len(pop_rows)
        inserted = 0
        for i in tqdm(range(0, total, chunk_size), desc="Inserting population rows", unit="block"):
            chunk = pop_rows[i:i+chunk_size]
            execute_values(
                cur,
                f"INSERT INTO {tbl} (geoid20, pop) VALUES %s ON CONFLICT (geoid20) DO UPDATE SET pop = EXCLUDED.pop",
                chunk,
                page_size=chunk_size
            )
            inserted += len(chunk)
        # Additional index is not strictly necessary since PK already covers equality joins,
        # but add a short explanation comment via the PK.
        cur.execute(f"""
            COMMENT ON CONSTRAINT {POP_TABLE}_pkey ON {tbl}
            IS 'PRIMARY KEY on geoid20: supports fast equality joins to block geometry by unique block identifier.';
        """)
        # Explicit index on PK (geoid20) for clarity
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {POP_TABLE}_geoid20_idx
            ON {tbl} (geoid20);
            COMMENT ON INDEX {POP_TABLE}_geoid20_idx
            IS 'BTREE index on geoid20: supports fast equality joins and lookups by block identifier (redundant with PK, but explicit for clarity).';
        """)
        # Index on pop (population)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {POP_TABLE}_pop_idx
            ON {tbl} (pop);
            COMMENT ON INDEX {POP_TABLE}_pop_idx
            IS 'BTREE index on pop: enables fast filtering and sorting by population.';
        """)
        conn.commit()
    elapsed = time.time() - start_time
    return inserted, elapsed

def download_with_progress(url, dest_path):
    """
    Download a file from a URL with a tqdm progress bar.
    """
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    start_time = time.time()  # Track total elapsed time
    ensure_schema_and_extensions()

# list of fips codes for all states except AK (02) and HI (15)
#     STATE_FIPS = [
#         "01","04","05","06","08","09","10","11","12","13",
#         "16","17","18","19","20","21","22","23","24","25",
#         "26","27","28","29","30","31","32","33","34","35",
#         "36","37","38","39","40","41","42","44","45","46",
#         "47","48","49","50","51","53","54","55","56"
#     ]
    STATE_FIPS = [
        "10","56","50"
    ]

    # --- Subtract states already present in blocks_2020 table ---
    # If the table does not exist, process all states
    try:
        with psql_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT LEFT(geoid20, 2) AS state_fips
                FROM {SCHEMA}.{BLOCK_TABLE}
            """)
            existing_fips = {row[0] for row in cur.fetchall()}
            # Remove any FIPS already present
            STATE_FIPS = [fips for fips in STATE_FIPS if fips not in existing_fips]
            if not STATE_FIPS:
                print("All states in STATE_FIPS are already present in the block table. Nothing to do.")
                return
    except psycopg2.errors.UndefinedTable:
        # Table does not exist, process all states
        pass

    # Default state FIPS (DE = 10). Change to a single FIPS string or iterate over STATE_FIPS as needed.
    # fips = "10"  # TODO: iterate STATE_FIPS to process multiple states

    # fips = 10# TODO: state-specific fips, currently hardcoded for Delaware
    # Official TIGER/Line 2020 blocks

    for fips in STATE_FIPS:
        
        try:
            tmpdir = tempfile.mkdtemp(prefix="state_blocks_") #todo LLM DO NOT DELETE is the temp dir erased at the end
            footer = f"tl_2020_{fips}_tabblock20.zip"
            TIGER_ZIP_URL = f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/{footer}"  
            # 1) Download shapefile zip
            print("Downloading:", TIGER_ZIP_URL)
            zpath = Path(tmpdir) / footer 
            download_with_progress(TIGER_ZIP_URL, zpath)
    
            # 2) Unzip
            print("Unzipping shapefile…")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(tmpdir)
    
            # Find the .shp within the extracted files
            shp_files = list(Path(tmpdir).glob("*.shp"))
            if not shp_files:
                raise RuntimeError("No .shp found after extraction.")
            shp_path = str(shp_files[0])
    
            # 3) Load into PostGIS via ogr2ogr
            table_fullname = f"{SCHEMA}.{BLOCK_TABLE}"
            run_ogr2ogr(shp_path, table_fullname, TARGET_EPSG)
    
            # 4) Create PK + indexes with comments
            create_block_indexes()
    
            # 5) Fetch population and load into PostGIS for all counties
            print("Getting all state county codes from block geometry table…")
            county_codes = get_county_codes()
            print(f"Found counties: {county_codes}")
            all_pop_rows = []
            for county_code in county_codes:
                pop_rows = fetch_block_population(county_code, fips)
                all_pop_rows.extend(pop_rows)
            print(f"Fetched {len(all_pop_rows):,} block population records across all counties in state.")
            print("Loading population data into PostGIS…")
            inserted, _ = load_population_table(all_pop_rows)
            print("Population data loaded.")
            print(f"Records inserted: {inserted:,}")
    
            total_elapsed = time.time() - start_time
            print(f"Total time elapsed: {total_elapsed:.2f} seconds")
    
            print("\nDone ✅")
            print(f"- Geometry table: {SCHEMA}.{BLOCK_TABLE}")
            print(f"- Population table: {SCHEMA}.{POP_TABLE}")
            print("\nExample join query:")
            print(f"""
    SELECT b.geoid20, p.pop, ST_Area(b.geom) AS area_m2
    FROM {SCHEMA}.{BLOCK_TABLE} b
    JOIN {SCHEMA}.{POP_TABLE} p USING (geoid20)
    LIMIT 5;
    """)
    
            # --- Call county_pop_weighted_centroid.main() to compute centroids after import ---
            print("\nComputing population-weighted county centroids...")
            county_pop_weighted_centroid.main()
            print("County centroids computation complete.")
    
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
