#!/usr/bin/env python3
"""
Download 2020 census tracts, load to PostGIS, and create indexes.
Table created:
  - public.tracts_2020 (geometry + TIGER attributes; PK = geoid)
"""
import os
import sys
import time
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
import requests
import psycopg2
from tqdm import tqdm

# ----------------------------
# CONFIG — edit as needed
# ----------------------------
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
TRACT_TABLE = "tracts_2020"
TARGET_EPSG = "EPSG:4269"

STATE_FIPS = [
    "56", "50", "11", "38",
    "46", "10", "30", "44",
    "23", "33", "16", "54",
    "31", "35", "28", "05",
    "19", "20", "32", "49",
    "09", "41", "40", "21",
    "22", "01", "45", "08",
    # "27", 
    "55", "24", "29",
    "18", "47", "25", "04",
    "53", "51", "34", "26",
    "37", "13", "39", "17",
    "42", "36", "12", "48",
    "06"
]

# ----------------------------
# Helpers
# ----------------------------
def psql_conn():
    return psycopg2.connect(**DB_CRED)

def run_ogr2ogr(shp_path, table_fullname, epsg_target):
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
    ]
    if epsg_target:
        cmd += ["-t_srs", epsg_target]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_schema_and_extensions():
    with psql_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.commit()

def create_tract_indexes():
    with psql_conn() as conn, conn.cursor() as cur:
        tbl = f"{SCHEMA}.{TRACT_TABLE}"
        # Ensure geoid is text
        cur.execute(f"""
            ALTER TABLE {tbl}
            ALTER COLUMN geoid TYPE text;
        """)
        # Add tract_11 column if not exists
        cur.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = '{SCHEMA}' AND table_name = '{TRACT_TABLE}' AND column_name = 'tract_11'
                ) THEN
                    ALTER TABLE {tbl} ADD COLUMN tract_11 text;
                END IF;
            END$$;
        """)
        # Populate tract_11 with first 11 chars of geoid
        cur.execute(f"""
            UPDATE {tbl}
            SET tract_11 = LEFT(geoid, 11)
            WHERE tract_11 IS NULL OR tract_11 <> LEFT(geoid, 11);
        """)
        # Add PK on geoid if not exists
        cur.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema = '{SCHEMA}' AND table_name = '{TRACT_TABLE}' AND constraint_type = 'PRIMARY KEY'
                ) THEN
                    ALTER TABLE {tbl}
                    ADD CONSTRAINT {TRACT_TABLE}_pk PRIMARY KEY (geoid);
                END IF;
            END$$;
        """)
        # Spatial index on geom
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {TRACT_TABLE}_geom_gix
            ON {tbl}
            USING GIST (geom);
        """)
        # Index on tract_11 for fast lookup
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {TRACT_TABLE}_tract_11_idx
            ON {tbl} (tract_11);
        """)
        conn.commit()

def download_with_progress(url, dest_path):
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
    start_time = time.time()
    ensure_schema_and_extensions()
    for fips in STATE_FIPS:
        state_start = time.time()
        try:
            tmpdir = tempfile.mkdtemp(prefix="state_tracts_")
            footer = f"tl_2020_{fips}_tract.zip"
            TIGER_ZIP_URL = f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/{footer}"
            print("Downloading:", TIGER_ZIP_URL)
            zpath = Path(tmpdir) / footer
            download_with_progress(TIGER_ZIP_URL, zpath)
            print("Unzipping shapefile…")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(tmpdir)
            shp_files = list(Path(tmpdir).glob("*.shp"))
            if not shp_files:
                raise RuntimeError("No .shp found after extraction.")
            shp_path = str(shp_files[0])
            table_fullname = f"{SCHEMA}.{TRACT_TABLE}"
            run_ogr2ogr(shp_path, table_fullname, TARGET_EPSG)
            create_tract_indexes()
            print(f"Done ✅ - Imported tracts for state {fips}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            total_elapsed = time.time() - state_start
            print(f"Total time elapsed (State): {total_elapsed:.2f} seconds")
    total_elapsed = time.time() - start_time
    print(f"Total time elapsed: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
