#!/usr/bin/env python3
"""
Download Delaware 2020 census blocks, load to PostGIS, then fetch 2020 PL block population and load to a joinable table.

Tables created:
  - public.de_blocks_2020 (geometry + TIGER attributes; PK = geoid20)
  - public.de_block_pop_2020 (geoid20, pop)

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
SCHEMA = "public"
BLOCK_TABLE = "de_blocks_2020"
POP_TABLE = "de_block_pop_2020"
# Keep native shapefile CRS (NAD83 / EPSG:4269). Change to "EPSG:5070" if you want Albers.
TARGET_EPSG = "EPSG:4269"

# Official TIGER/Line 2020 blocks for Delaware (state FIPS 10)
TIGER_ZIP_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_10_tabblock20.zip"

# 2020 PL (DEC/PL) API: total population is P1_001N
# We request NAME for debugging/spot checks; keys returned: NAME, P1_001N, state, county, tract, block
PL_API_URL = "https://api.census.gov/data/2020/dec/pl"
PL_API_PARAMS = {
    "get": "NAME,P1_001N",
    "for": "block:*",
    "in": "state:10",  # Delaware
}

# ----------------------------
# Helpers
# ----------------------------
def psql_conn():
    return psycopg2.connect(
        host=PG["host"],
        port=PG["port"],
        dbname=PG["dbname"],
        user=PG["user"],
        password=PG["password"],
    )

def run_ogr2ogr(shp_path, table_fullname, epsg_target):
    """
    Load shapefile into PostGIS using ogr2ogr.
    -nlt MULTIPOLYGON because blocks are polygons; TIGER block geometries are polygons/multipolygons.
    -lco GEOMETRY_NAME=geom ensures column is 'geom' (conventional).
    -nln sets output table name.
    -progress for feedback.
    """
    cmd = [
        "ogr2ogr",
        "-overwrite",
        "-f", "PostgreSQL",
        f"PG:host={PG['host']} port={PG['port']} dbname={PG['dbname']} user={PG['user']} password={PG['password']}",
        shp_path,
        "-nln", table_fullname,
        "-lco", "GEOMETRY_NAME=geom",
        "-nlt", "MULTIPOLYGON",
        "-lco", "FID=gid",
        "-progress"
    ]
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
        # Some TIGER fields are fixed-width text; ensure the key column exists in the imported schema.
        # GEOID20 exists in TABBLOCK20; make it PK.
        cur.execute(f"""
            ALTER TABLE {tbl}
            ALTER COLUMN geoid20 TYPE text;
        """)
        # Drop any existing PK if re-running
        cur.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = '{tbl}'::regclass
                      AND contype = 'p'
                ) THEN
                    ALTER TABLE {tbl} DROP CONSTRAINT (SELECT conname FROM pg_constraint WHERE conrelid = '{tbl}'::regclass AND contype='p' LIMIT 1);
                END IF;
            END$$;
        """)
        # Add PK on geoid20
        cur.execute(f"""
            ALTER TABLE {tbl}
            ADD CONSTRAINT {BLOCK_TABLE}_pk PRIMARY KEY (geoid20);
            COMMENT ON CONSTRAINT {BLOCK_TABLE}_pk ON {tbl}
            IS 'PRIMARY KEY on geoid20: unique per census block; ideal for equality joins with attribute tables.';
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
        conn.commit()

def fetch_block_population():
    """
    Pull block-level population from the 2020 DEC/PL API for Delaware.
    We construct geoid20 = state + county + tract + block and store P1_001N as pop.
    """
    print("Fetching block-level population from 2020 DEC/PL API for Delaware…")
    r = requests.get(PL_API_URL, params=PL_API_PARAMS, timeout=120)
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
    for row in rows[1:]:
        state = row[idx_state]
        county = row[idx_county]
        tract = row[idx_tract]
        block = row[idx_block]
        # Construct 2020 block GEOID20 = state(2) + county(3) + tract(6) + block(4)
        geoid20 = f"{state}{county}{tract}{block}"
        pop = int(row[idx_pop])
        data.append((geoid20, pop))
    return data

def load_population_table(pop_rows):
    """
    Create/replace the population table and load values.
    Also add PK + BTREE index with rationale.
    """
    with psql_conn() as conn, conn.cursor() as cur:
        tbl = f"{SCHEMA}.{POP_TABLE}"
        cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(f"""
            CREATE TABLE {tbl}(
                geoid20 text PRIMARY KEY,
                pop integer NOT NULL
            );
            COMMENT ON TABLE {tbl} IS '2020 PL (DEC/PL) total population (P1_001N) at block level for Delaware; keyed by geoid20 for joins to de_blocks_2020.';
            COMMENT ON COLUMN {tbl}.geoid20 IS 'Census Block GEOID20 (state+county+tract+block)';
            COMMENT ON COLUMN {tbl}.pop IS 'Total population (P1_001N) from 2020 DEC/PL.';
        """)
        # Bulk insert
        execute_values(
            cur,
            f"INSERT INTO {tbl} (geoid20, pop) VALUES %s ON CONFLICT (geoid20) DO UPDATE SET pop = EXCLUDED.pop",
            pop_rows,
            page_size=10000
        )
        # Additional index is not strictly necessary since PK already covers equality joins,
        # but add a short explanation comment via the PK.
        cur.execute(f"""
            COMMENT ON CONSTRAINT {POP_TABLE}_pkey ON {tbl}
            IS 'PRIMARY KEY on geoid20: supports fast equality joins to block geometry by unique block identifier.';
        """)
        conn.commit()

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
    ensure_schema_and_extensions()

    tmpdir = tempfile.mkdtemp(prefix="de_blocks_")
    try:
        # 1) Download shapefile zip
        print("Downloading:", TIGER_ZIP_URL)
        zpath = Path(tmpdir) / "tl_2020_10_tabblock20.zip"
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

        # 5) Fetch population and load into PostGIS
        print("Fetching block-level population from 2020 DEC/PL API for Delaware…")
        pop_rows = fetch_block_population()
        print(f"Fetched {len(pop_rows):,} block population records.")
        print("Loading population data into PostGIS…")
        load_population_table(pop_rows)
        print("Population data loaded.")

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

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
