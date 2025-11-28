#!/usr/bin/env python3
"""
Create block group (BG) 2020 geometries and centroids from block-level data.

Creates a table block_group_2020 with:
  - block_group_geoid (first 12 digits of block GEOID20): PK
  - geom: ST_Union of all block geometries in the BG
  - pop: sum of block populations
  - centroid_geom: population-weighted centroid

Indexes:
  - GiST on geom for spatial queries
  - GiST on centroid_geom for spatial queries
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

# Database credentials (same as block_import.py)
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
SCHEMA = "public"
BLOCK_TABLE = "blocks_2020"
BG_TABLE = "block_group_2020"


def psql_conn():
    """Create a database connection using DB_CRED."""
    return psycopg2.connect(
        host=DB_CRED["host"],
        port=DB_CRED["port"],
        dbname=DB_CRED["dbname"],
        user=DB_CRED["user"],
        password=DB_CRED["password"],
    )


def create_bg_table():
    """Create the block_group_2020 table with proper constraints and indexes."""
    with psql_conn() as conn, conn.cursor() as cur:
        tbl = f"{SCHEMA}.{BG_TABLE}"
        
        # Drop table if exists (for idempotency during development)
        cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        
        # Create the table
        cur.execute(f"""
            CREATE TABLE {tbl} (
                block_group_geoid text PRIMARY KEY,
                geom geometry(MultiPolygon, 4269) NOT NULL,
                pop integer NOT NULL,
                centroid_geom geometry(Point, 4269) NOT NULL
            );
            COMMENT ON TABLE {tbl}
            IS 'Block Group 2020 geometries and centroids. Aggregated from block-level data.';
            COMMENT ON COLUMN {tbl}.block_group_geoid
            IS 'Block Group GEOID20 (first 12 digits of block GEOID: state+county+tract+block_group).';
            COMMENT ON COLUMN {tbl}.geom
            IS 'ST_Union of all block geometries in the block group (MultiPolygon, EPSG:4269).';
            COMMENT ON COLUMN {tbl}.pop
            IS 'Total population: sum of pop from all blocks in the block group.';
            COMMENT ON COLUMN {tbl}.centroid_geom
            IS 'Population-weighted centroid (Point, EPSG:4269).';
        """)
        
        # Add spatial indexes
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {BG_TABLE}_geom_gix
            ON {tbl}
            USING GIST (geom);
            COMMENT ON INDEX {BG_TABLE}_geom_gix
            IS 'GiST spatial index on geom: accelerates spatial filters and joins.';
        """)
        
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {BG_TABLE}_centroid_geom_gix
            ON {tbl}
            USING GIST (centroid_geom);
            COMMENT ON INDEX {BG_TABLE}_centroid_geom_gix
            IS 'GiST spatial index on centroid_geom: accelerates spatial filters and joins on centroid.';
        """)
        
        conn.commit()
        print(f"✓ Created table {tbl}")


def populate_bg_table():
    """
    Populate the block_group_2020 table by aggregating blocks.
    
    Groups blocks by their first 12 digits (block group GEOID), unions geometries,
    sums populations, and calculates population-weighted centroids.
    """
    with psql_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        tbl = f"{SCHEMA}.{BG_TABLE}"
        block_tbl = f"{SCHEMA}.{BLOCK_TABLE}"
        
        # Fetch distinct block groups and their block data
        print("Fetching block group data...")
        cur.execute(f"""
            SELECT
                LEFT(b.geoid20, 12) AS block_group_geoid,
                ARRAY_AGG(b.gid) AS block_gids,
                COUNT(*) AS block_count
            FROM {block_tbl} b
            GROUP BY block_group_geoid
            ORDER BY block_group_geoid;
        """)
        
        bgs = cur.fetchall()
        print(f"Found {len(bgs)} block groups")
        
        # Process each block group
        bg_data = []
        for bg_row in tqdm(bgs, desc="Processing block groups", unit="BG"):
            bg_geoid = bg_row['block_group_geoid']
            
            # Fetch blocks for this BG with their geometries and populations
            cur.execute(f"""
                SELECT
                    b.gid,
                    b.geom,
                    COALESCE(b.pop20, 0) AS pop
                FROM {block_tbl} b
                WHERE LEFT(b.geoid20, 12) = %s;
            """, (bg_geoid,))
            
            blocks = cur.fetchall()
            
            if not blocks:
                continue
            
            # Sum population
            total_pop = sum(b['pop'] for b in blocks)
            
            # Calculate population-weighted centroid using SQL
            # This is done via a subquery that we'll use in the INSERT
            bg_data.append({
                'block_group_geoid': bg_geoid,
                'block_ids': tuple(b['gid'] for b in blocks),
            })
        
        # Now insert BGs using SQL that calculates unions and centroids
        print("\nInserting block groups into database...")
        insert_count = 0
        for bg_info in tqdm(bg_data, desc="Inserting block groups", unit="BG"):
            bg_geoid = bg_info['block_group_geoid']
            block_ids = bg_info['block_ids']
            
            # Convert block_ids to SQL array format
            block_ids_sql = ', '.join(str(gid) for gid in block_ids)
            
            cur.execute(f"""
                INSERT INTO {tbl} (block_group_geoid, geom, pop, centroid_geom)
                WITH bg_blocks AS (
                    SELECT
                        b.geom,
                        COALESCE(b.pop20, 0) AS pop,
                        ST_Centroid(b.geom) AS block_centroid
                    FROM {block_tbl} b
                    WHERE b.gid IN ({block_ids_sql})
                )
                SELECT
                    %s AS block_group_geoid,
                    ST_Union(geom)::geometry(MultiPolygon, 4269) AS geom,
                    SUM(pop)::integer AS pop,
                    ST_SetSRID(
                        ST_MakePoint(
                            SUM(ST_X(block_centroid) * pop) / GREATEST(SUM(pop), 1),
                            SUM(ST_Y(block_centroid) * pop) / GREATEST(SUM(pop), 1)
                        ),
                        4269
                    )::geometry(Point, 4269) AS centroid_geom
                FROM bg_blocks;
            """, (bg_geoid,))
            
            insert_count += 1
        
        conn.commit()
        print(f"✓ Inserted {insert_count} block groups")


def verify_results():
    """Print summary statistics about the created block groups."""
    with psql_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        tbl = f"{SCHEMA}.{BG_TABLE}"
        
        cur.execute(f"""
            SELECT
                COUNT(*) AS bg_count,
                SUM(pop) AS total_pop,
                AVG(pop)::integer AS avg_pop,
                MIN(pop) AS min_pop,
                MAX(pop) AS max_pop
            FROM {tbl};
        """)
        
        result = cur.fetchone()
        print("\n=== Block Group 2020 Summary ===")
        print(f"Total block groups: {result['bg_count']}")
        print(f"Total population: {result['total_pop']}")
        print(f"Average BG population: {result['avg_pop']}")
        print(f"Min BG population: {result['min_pop']}")
        print(f"Max BG population: {result['max_pop']}")


def main():
    """Main entry point."""
    print("Creating block group 2020 table...")
    create_bg_table()
    
    print("\nPopulating block group data...")
    populate_bg_table()
    
    print("\nVerifying results...")
    verify_results()
    
    print("\n✓ Block group 2020 table creation complete!")


if __name__ == "__main__":
    main()


