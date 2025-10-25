#!/usr/bin/env python3
"""
Create the `hll` table and appropriate indices in PostGIS.

Table schema (public.hll):
  - id text PRIMARY KEY
  - parent text
  - hierarchy text
  - level text
  - label text
  - geom geometry(MultiPolygon,4269)
  - centroid_geom geometry(Point,4269)
  - pop numeric(12,0)

Requirements implemented:
- UNIQUE constraint on (hierarchy, level, label)
- Indices on all columns (B-tree for text/numeric, GIST for geometry columns)
- Ensures PostGIS extension exists

Usage:
    python normalize/create_hll_table.py [--dry-run]

Options:
    --dry-run    Print the SQL that would be executed without modifying the DB.

Update DB_CRED in this file to match your environment.
"""

import argparse
import sys
import psycopg2

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
TABLE = "hll"


def psql_conn():
    return psycopg2.connect(**DB_CRED)


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} (
  id text PRIMARY KEY,
  parent text,
  hierarchy text,
  level numeric(3,0),
  label text,
  geom geometry(MultiPolygon,4269),
  centroid_geom geometry(Point,4269),
  pop numeric(12,0)
);
"""


def create_indices_and_constraints(cur):
    # Ensure PostGIS extension exists is handled outside this function
    # Unique constraint on (hierarchy, level, label)
    cur.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_schema = '{SCHEMA}' AND tc.table_name = '{TABLE}' AND tc.constraint_type = 'UNIQUE'
                AND EXISTS (
                    SELECT 1 FROM information_schema.key_column_usage kcu
                    WHERE kcu.table_schema = '{SCHEMA}' AND kcu.table_name = '{TABLE}'
                    AND kcu.column_name IN ('hierarchy','level','label')
                )
            ) THEN
                ALTER TABLE {SCHEMA}.{TABLE}
                ADD CONSTRAINT {TABLE}_hierarchy_level_label_unique UNIQUE (hierarchy, level, label);
            END IF;
        END$$;
    """)

    # Indices for text and numeric columns (B-tree)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_parent_idx ON {SCHEMA}.{TABLE} (parent);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_hierarchy_idx ON {SCHEMA}.{TABLE} (hierarchy);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_level_idx ON {SCHEMA}.{TABLE} (level);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_label_idx ON {SCHEMA}.{TABLE} (label);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_pop_idx ON {SCHEMA}.{TABLE} (pop);")

    # Spatial indices (GIST)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_geom_gix ON {SCHEMA}.{TABLE} USING GIST (geom);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_centroid_geom_gix ON {SCHEMA}.{TABLE} USING GIST (centroid_geom);")


def ensure_postgis(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")


def main(argv):
    parser = argparse.ArgumentParser(description="Create hll table and indices in the database")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args(argv)

    sqls = [CREATE_TABLE_SQL]

    # Additional SQL for indices and constraints are executed via psycopg2 calls in create_indices_and_constraints

    if args.dry_run:
        print("DRY RUN: SQL statements that would be executed:\n")
        for s in sqls:
            print(s)
        print("(Indices and constraints will be created via psycopg2 API calls — see script.)")
        return

    with psql_conn() as conn, conn.cursor() as cur:
        print("Ensuring PostGIS extension and creating table...")
        ensure_postgis(cur)
        cur.execute(CREATE_TABLE_SQL)
        print("Creating indices and constraints...")
        create_indices_and_constraints(cur)
        conn.commit()
        print(f"Table {SCHEMA}.{TABLE} ensured with indices and constraints.")


if __name__ == '__main__':
    main(sys.argv[1:])
