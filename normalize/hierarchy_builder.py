#!/usr/bin/env python3
"""
Interactive CLI tool for building spatial hierarchies incrementally.

Uses Inquirer to guide the user through selecting a parent record, choosing
centroid level, running the clustering algorithm, and inserting child records
into the hll table.

Usage:
    python normalize/hierarchy_builder.py
"""

import sys
import psycopg2
from InquirerPy.base import Choice
from psycopg2.extras import RealDictCursor
from collections import defaultdict
from InquirerPy import inquirer
from tqdm import tqdm

# Import the clustering algorithm (public API only)
try:
    from clustering import cluster_and_insert
except ImportError:
    print("Error: clustering.py not found. Make sure it's in the same directory or in PYTHONPATH.")
    sys.exit(1)

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
HLL_TABLE = "hll"
COUNTY_CENTROIDS = "county_centroids2"
COUNTIES_TABLE = "counties_2020"
TRACTS_TABLE = "tracts_2020"
BLOCKS_TABLE = "blocks_2020"

# Default number of clusters
DEFAULT_K = 19

# Centroid level mappings
CENTROID_LEVEL_MAP = {
    'county': {
        'table': COUNTIES_TABLE,
        'type': 'county',
        'geoid_transform': None  # geoid as-is
    },
    '11': {
        'table': TRACTS_TABLE,
        'type': '11',
        'geoid_transform': None  # geoid as-is
    },
    '12': {
        'table': BLOCKS_TABLE,
        'type': '12',
        'geoid_transform': 'LEFT(geoid, 12)'  # first 12 digits
    }
}


def psql_conn():
    return psycopg2.connect(**DB_CRED)


def get_hierarchies(cur):
    """Fetch distinct hierarchies from hll table."""
    cur.execute(f"SELECT DISTINCT hierarchy FROM {SCHEMA}.{HLL_TABLE} ORDER BY hierarchy;")
    return [row['hierarchy'] for row in cur.fetchall()]


def get_level_zero_records(cur, hierarchy):
    """Fetch level-0 records for a given hierarchy."""
    cur.execute(
        f"SELECT id, hierarchy, label, pop FROM {SCHEMA}.{HLL_TABLE} WHERE hierarchy = %s AND level = 0 ORDER BY label;",
        (hierarchy,)
    )
    return cur.fetchall()


def get_children(cur, parent_id):
    """Fetch children of a given parent record."""
    cur.execute(
        f"SELECT id, level, label, pop FROM {SCHEMA}.{HLL_TABLE} WHERE parent = %s ORDER BY label;",
        (parent_id,)
    )
    return cur.fetchall()


def get_record(cur, record_id):
    """Fetch a single record by id."""
    cur.execute(
        f"SELECT id, parent, hierarchy, level, label, pop, ST_AsText(geom) as geom_wkt FROM {SCHEMA}.{HLL_TABLE} WHERE id = %s;",
        (record_id,)
    )
    return cur.fetchone()


def drill_down_to_leaf(cur, hierarchy):
    """Interactively drill down from level 0 to a leaf record."""
    # Start at level 0
    level_zero = get_level_zero_records(cur, hierarchy)
    if not level_zero:
        print(f"No level-0 records found for hierarchy '{hierarchy}'")
        return None
    
    # Display level 0 options - create tuples of (display, value)
    choices = [Choice(r['id'], name=f"ID: {r['id']}, Label: {r['label']}, Pop: {r['pop']}") for r in level_zero]
    
    selected_id = inquirer.select(
        message=f"Select level-0 record for hierarchy '{hierarchy}':",
        choices=choices
    ).execute()
    
    if selected_id is None:
        return None
    
    current_id = selected_id
    
    # Keep drilling down until we reach a leaf
    while True:
        children = get_children(cur, current_id)
        if not children:
            # This is a leaf node
            return current_id
        
        # Present children for selection - use Choice objects with value and name
        choices = [Choice(r['id'], name=f"ID: {r['id']}, Level: {r['level']}, Label: {r['label']}, Pop: {r['pop']}") for r in children]
        choices.append(Choice("__stop__", name="** Select this record (stop drilling) **"))
        choices.append(Choice("__back__", name="** Go back **"))
        
        selection = inquirer.select(
            message=f"Children of {current_id} (or select this record as parent):",
            choices=choices
        ).execute()
        
        if selection is None:
            return None
        
        if selection == "__stop__":
            return current_id
        elif selection == "__back__":
            # For simplicity, we'll just return None and let user restart
            # (A full implementation could maintain a stack of visited nodes)
            print("Going back not implemented yet. Please restart selection.")
            return None
        else:
            # Selection is the child ID
            current_id = selection


def select_parent_record(cur):
    """Main flow: select hierarchy and drill down to a leaf record."""
    # Get hierarchies
    hierarchies = get_hierarchies(cur)
    if not hierarchies:
        print("No hierarchies found in the database.")
        return None
    
    # Ask user to select hierarchy
    hierarchy = inquirer.select(
        message="Select a hierarchy:",
        choices=hierarchies
    ).execute()
    
    if hierarchy is None:
        return None
    
    # Ask user: manual entry or drill down?
    mode = inquirer.select(
        message="How do you want to select the parent record?",
        choices=['Drill down interactively', 'Enter record ID manually']
    ).execute()
    
    if mode is None:
        return None
    
    if mode == 'Enter record ID manually':
        record_id = inquirer.text(
            message="Enter the record ID:"
        ).execute()
        
        if not record_id:
            return None
            
        # Validate that the record exists
        record = get_record(cur, record_id)
        if not record:
            print(f"Record ID '{record_id}' not found.")
            return None
        return record_id
    else:
        # Drill down interactively
        return drill_down_to_leaf(cur, hierarchy)


def get_centroids_within_geometry(cur, parent_geom_wkb, centroid_level):
    """Fetch centroids from county_centroids2 that fall within the parent geometry.
    
    Returns a list of dicts with keys: county_geoid, lat, lon, pop, aland.
    """
    # Query centroids that fall within the parent geometry
    cur.execute(f"""
        SELECT county_geoid, 
               ST_Y(centroid_geom) AS lat, 
               ST_X(centroid_geom) AS lon, 
               pop,
               COALESCE(aland, 0) AS aland
        FROM {SCHEMA}.{COUNTY_CENTROIDS}
        WHERE type = %s 
          AND ST_Within(centroid_geom, ST_GeomFromWKB(%s, 4269))
        ORDER BY county_geoid;
    """, (centroid_level, parent_geom_wkb))
    
    return [dict(row) for row in cur.fetchall()]


def main():
    print("=== Spatial Hierarchy Builder ===\n")
    
    with psql_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Step 1: Select parent record
            print("Step 1: Select the parent record")
            parent_id = select_parent_record(cur)
            if not parent_id:
                print("No parent record selected. Exiting.")
                return
            
            # Fetch parent record details
            parent_record = get_record(cur, parent_id)
            if not parent_record:
                print(f"Could not fetch details for record {parent_id}. Exiting.")
                return
            
            print(f"\nSelected parent record:")
            print(f"  ID: {parent_record['id']}")
            print(f"  Hierarchy: {parent_record['hierarchy']}")
            print(f"  Level: {parent_record['level']}")
            print(f"  Label: {parent_record['label']}")
            print(f"  Population: {parent_record['pop']}")
            
            parent_level = int(parent_record['level'])
            new_level = parent_level + 1
            hierarchy = parent_record['hierarchy']
            
            print(f"\nNew child level will be: {new_level}")
            
            # Step 2: Select centroid level
            print("\nStep 2: Select centroid level")
            centroid_level = inquirer.select(
                message="Choose centroid level:",
                choices=['county', '11', '12']
            ).execute()
            
            if centroid_level is None:
                print("No centroid level selected. Exiting.")
                return
            
            print(f"Selected centroid level: {centroid_level}")
            
            # Step 3: Ask for number of clusters (K)
            print("\nStep 3: Set number of clusters")
            k_input = inquirer.number(
                message=f"Enter number of clusters:",
                default=DEFAULT_K,
                min_allowed=1,
                max_allowed=1000
            ).execute()
            
            if k_input is None:
                print("No K value provided. Exiting.")
                return
            
            k = int(k_input)
            print(f"Number of clusters: {k}")
            
            # Step 4: Fetch parent geometry and centroids
            print("\nStep 4: Fetching centroids within parent geometry...")
            cur.execute(f"SELECT ST_AsBinary(geom) FROM {SCHEMA}.{HLL_TABLE} WHERE id = %s;", (parent_id,))
            parent_geom_row = cur.fetchone()
            print(parent_geom_row)
            if not parent_geom_row or not parent_geom_row['st_asbinary']:
                print("Could not fetch parent geometry. Exiting.")
                return
            
            parent_geom_wkb = parent_geom_row['st_asbinary']
            centroids = get_centroids_within_geometry(cur, parent_geom_wkb, centroid_level)
            
            if not centroids:
                print(f"No centroids of type '{centroid_level}' found within parent geometry. Exiting.")
                return
            
            print(f"Found {len(centroids)} centroids within parent geometry.")
            
            # Step 5: Run clustering algorithm
            print("\nStep 5: Running clustering algorithm...")
            region_label = str(parent_record['label'])
            
            # Call the public API which returns assignments
            assignments = cluster_and_insert(centroids, k, region_label)
            
            if not assignments:
                print("Clustering returned no assignments. Exiting.")
                return
            
            print(f"Clustering complete. Generated {len(assignments)} assignments.")
            
            # Step 6: Insert assignments into database
            print("\nStep 6: Inserting child records into database...")
            
            # Group assignments by label
            label_groups = defaultdict(list)
            for assignment in assignments:
                label_groups[assignment['label']].append(assignment)
            
            # Get the geometry table info for this centroid level
            level_info = CENTROID_LEVEL_MAP[centroid_level]
            geom_table = level_info['table']
            
            with tqdm(total=len(label_groups), desc="Inserting records", unit="record") as pbar:
                for label, group in label_groups.items():
                    geoids = [rec['geoid'] for rec in group]
                    medioid_geoid = next((rec['geoid'] for rec in group if rec['medioid']), None)
                    total_pop = sum(rec.get('pop', 0) for rec in group)
                    
                    # Merge geometries based on centroid level
                    if centroid_level == '12':
                        # For block groups (12), filter by LEFT(geoid, 12)
                        cur.execute(f"""
                            SELECT ST_AsBinary(ST_SetSRID(ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_Union(geom)), 3)), 4269)) AS merged_geom
                            FROM {SCHEMA}.{geom_table}
                            WHERE LEFT(geoid, 12) = ANY(%s);
                        """, (geoids,))
                    else:
                        # For county or tract level
                        cur.execute(f"""
                            SELECT ST_AsBinary(ST_SetSRID(ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_Union(geom)), 3)), 4269)) AS merged_geom
                            FROM {SCHEMA}.{geom_table}
                            WHERE geoid = ANY(%s);
                        """, (geoids,))
                    
                    merged_geom_row = cur.fetchone()
                    print(merged_geom_row)
                    if not merged_geom_row or not merged_geom_row['merged_geom']:
                        print(f"Warning: Could not merge geometries for label {label}, skipping.")
                        pbar.update(1)
                        continue
                    
                    merged_geom_wkb = merged_geom_row['merged_geom']
                    
                    # Get centroid_geom from the medioid
                    centroid_geom_wkb = None
                    if medioid_geoid:
                        cur.execute(f"""
                            SELECT ST_AsBinary(centroid_geom)
                            FROM {SCHEMA}.{COUNTY_CENTROIDS}
                            WHERE county_geoid = %s AND type = %s;
                        """, (medioid_geoid, centroid_level))
                        centroid_row = cur.fetchone()
                        print(centroid_row)
                        if centroid_row:
                            centroid_geom_wkb = centroid_row['st_asbinary']

                    for item in (parent_id, hierarchy, new_level, label, merged_geom_wkb, centroid_geom_wkb, total_pop):
                        print(item, type(item))
                            
                    # Insert into hll
                    cur.execute(f"""
                        INSERT INTO {SCHEMA}.{HLL_TABLE} (id, parent, hierarchy, level, label, geom, centroid_geom, pop)
                        VALUES (gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (parent_id, hierarchy, new_level, int(label), merged_geom_wkb, centroid_geom_wkb, int(total_pop)))

                    fetchone = cur.fetchone()
                    print(fetchone)
                    new_id = fetchone['id']
                    pbar.update(1)
            
            print(f"Successfully inserted {len(label_groups)} child records.")
            
            conn.commit()
            print("\n=== Hierarchy building complete! ===")


if __name__ == '__main__':
    main()

