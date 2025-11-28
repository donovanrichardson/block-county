#!/usr/bin/env python3
"""
Download 2020 US counties (national TL_2020_US_COUNTY), load to PostGIS, and create indexes.
Table created:
  - public.counties_2020 (geometry + TIGER attributes; PK = geoid)

This mirrors the style of `tract_import.py` but downloads the single national ZIP
`tl_2020_us_county.zip` so there is no per-state loop. Per user request, no
`county_5` column is created because the county GEOID is already 5 characters.
"""
import os
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
COUNTY_TABLE = "counties_2020"
TARGET_EPSG = "EPSG:4269"

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


def create_county_indexes():
    with psql_conn() as conn, conn.cursor() as cur:
        tbl = f"{SCHEMA}.{COUNTY_TABLE}"
        # Ensure geoid is text
        cur.execute(f"""
            ALTER TABLE {tbl}
            ALTER COLUMN geoid TYPE text;
        """)
        # Add PK on geoid if not exists
        cur.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema = '{SCHEMA}' AND table_name = '{COUNTY_TABLE}' AND constraint_type = 'PRIMARY KEY'
                ) THEN
                    ALTER TABLE {tbl}
                    ADD CONSTRAINT {COUNTY_TABLE}_pk PRIMARY KEY (geoid);
                END IF;
            END$$;
        """)
        # Spatial index on geom
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {COUNTY_TABLE}_geom_gix
            ON {tbl}
            USING GIST (geom);
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
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="us_counties_")
        # footer = "tl_2020_us_county.zip"
        TIGER_ZIP_URL = f"https://pub-a835f667d17f4b6691fafec7e9ede33d.r2.dev/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
        print("Downloading:", TIGER_ZIP_URL)
        zpath = Path(tmpdir) / "countys"
        download_with_progress(TIGER_ZIP_URL, zpath)
        print("Unzipping shapefile…")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).glob("*.shp"))
        if not shp_files:
            raise RuntimeError("No .shp found after extraction.")
        shp_path = str(shp_files[0])
        table_fullname = f"{SCHEMA}.{COUNTY_TABLE}"
        run_ogr2ogr(shp_path, table_fullname, TARGET_EPSG)
        create_county_indexes()
        print(f"Done ✅ - Imported US counties into {table_fullname}")
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        total_elapsed = time.time() - start_time
        print(f"Total time elapsed: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
