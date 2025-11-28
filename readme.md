# prereqs

Must install pipenv on your machine
```sh
pip install --user pipenv
```

Run `pipenv install` in the root folder. Python version 3.9 is expected.

Then run `pipenv shell` to activate the virtual environment.

Ensure a PostgreSQL server is running and a database with the PostGIS extension and a user with the credentials below exists:

```py
DB_CRED = {
    "host": "localhost",
    "port": 5432,
    "dbname": "block-county",
    "user": "block-county",
    "password": "your_password_here",
}
```

Ther should also be the following Databases existing. You will need to download and import Tiger and population data for blocks, block groups, tracts, and counties within the USA.

**Info:** the imported TIGER data will require about 12GB of free space.

### DDL

#### Block Table

used for imported TIGER data at the block level

```sql
CREATE TABLE blocks_2020 (
    gid SERIAL PRIMARY KEY,
    geom geometry(MultiPolygon,4269),
    statefp20 character varying(2),
    countyfp20 character varying(3),
    tractce20 character varying(6),
    blockce20 character varying(4),
    geoid20 text,
    name20 character varying(10),
    mtfcc20 character varying(5),
    ur20 character varying(5),
    uace20 character varying(5),
    uatype20 character varying(5),
    funcstat20 character varying(1),
    aland20 numeric(14,0),
    awater20 numeric(14,0),
    intptlat20 character varying(11),
    intptlon20 character varying(12),
    housing20 numeric(12,0),
    pop20 numeric(12,0)
);

-- Indices -------------------------------------------------------

CREATE UNIQUE INDEX blocks_2020_pkey ON blocks_2020(gid int4_ops);
CREATE INDEX blocks_2020_geom_geom_idx ON blocks_2020 USING GIST (geom gist_geometry_ops_2d);
CREATE INDEX blocks_2020_geom_gix ON blocks_2020 USING GIST (geom gist_geometry_ops_2d);
CREATE INDEX blocks_2020_county_tract_idx ON blocks_2020(countyfp20 text_ops,tractce20 text_ops);
CREATE INDEX blocks_2020_blockce20_idx ON blocks_2020(blockce20 text_ops);
CREATE INDEX blocks_2020_aland20_idx ON blocks_2020(aland20 numeric_ops);

```

#### Tract Table

Used for imported TIGER data at the tract level

```sql
CREATE TABLE tracts_2020 (
    gid SERIAL PRIMARY KEY,
    geom geometry(MultiPolygon,4269),
    statefp character varying(2),
    countyfp character varying(3),
    tractce character varying(6),
    geoid text,
    name character varying(7),
    namelsad character varying(20),
    mtfcc character varying(5),
    funcstat character varying(1),
    aland numeric(14,0),
    awater numeric(14,0),
    intptlat character varying(11),
    intptlon character varying(12),
    tract_11 text
);

-- Indices -------------------------------------------------------

CREATE UNIQUE INDEX tracts_2020_pkey ON tracts_2020(gid int4_ops);
CREATE INDEX tracts_2020_geom_geom_idx ON tracts_2020 USING GIST (geom gist_geometry_ops_2d);
CREATE INDEX tracts_2020_geom_gix ON tracts_2020 USING GIST (geom gist_geometry_ops_2d);
CREATE INDEX tracts_2020_tract_11_idx ON tracts_2020(tract_11 text_ops);
```

### County Table
Used for imported TIGER data at the county level

```sql
CREATE TABLE counties_2020 (
    gid SERIAL PRIMARY KEY,
    geom geometry(MultiPolygon,4269),
    statefp character varying(2),
    countyfp character varying(3),
    countyns character varying(8),
    geoid text,
    name character varying(100),
    namelsad character varying(100),
    lsad character varying(2),
    classfp character varying(2),
    mtfcc character varying(5),
    csafp character varying(3),
    cbsafp character varying(5),
    metdivfp character varying(5),
    funcstat character varying(1),
    aland numeric(14,0),
    awater numeric(14,0),
    intptlat character varying(11),
    intptlon character varying(12)
);

-- Indices -------------------------------------------------------

CREATE UNIQUE INDEX counties_2020_pkey ON counties_2020(gid int4_ops);
CREATE INDEX counties_2020_geom_geom_idx ON counties_2020 USING GIST (geom gist_geometry_ops_2d);
CREATE INDEX counties_2020_geom_gix ON counties_2020 USING GIST (geom gist_geometry_ops_2d);
```

#### Geographic Centroids Table

This table contains centroids that I constructed from a centrer-of-population calculation at the block level, for the Counties, Tracts, and Block Groups of the US. Despite its name, this table's scope is not limited to counties.

```sql
CREATE TABLE county_centroids2 (
    county_geoid text PRIMARY KEY,
    centroid_geom geometry(Point,4269) NOT NULL,
    pop integer NOT NULL,
    type text,
    parent text,
    aland bigint,
    awater bigint
);

-- Indices -------------------------------------------------------

CREATE UNIQUE INDEX county_centroids2_pkey ON county_centroids2(county_geoid text_ops);
```

#### Hierarchy of K-Medoid Clusters Table

This table contains the output of the K-Medoids clustering. It also allows for hierarchical clustering, such that an output cluster can be used as the parent of another clustering operation. For the correct operation of the K-Medoids algorithm, each record in the table must have a population value greater than zero. 

HLL stands for *Hierarchy*, *Level*, *Label*, three columns used in this table.

```sql
CREATE TABLE hll (
    id text PRIMARY KEY,
    parent text,
    hierarchy text,
    level numeric(3,0),
    label numeric(3,0),
    geom geometry(MultiPolygon,4269),
    centroid_geom geometry(Point,4269),
    pop numeric(12,0),
    geocoded text,
    CONSTRAINT hll_hierarchy_level_label_unique UNIQUE (hierarchy, level, label)
);

-- Indices -------------------------------------------------------

CREATE UNIQUE INDEX hll_pkey ON hll(id text_ops);
CREATE INDEX hll_parent_idx ON hll(parent text_ops);
CREATE INDEX hll_hierarchy_idx ON hll(hierarchy text_ops);
CREATE INDEX hll_level_idx ON hll(level numeric_ops);
CREATE INDEX hll_pop_idx ON hll(pop numeric_ops);
CREATE INDEX hll_geom_gix ON hll USING GIST (geom gist_geometry_ops_2d);
CREATE INDEX hll_centroid_geom_gix ON hll USING GIST (centroid_geom gist_geometry_ops_2d);
CREATE UNIQUE INDEX hll_hierarchy_level_label_unique ON hll(hierarchy text_ops,level numeric_ops,label numeric_ops);
CREATE INDEX hll_label_idx ON hll(label numeric_ops);

```

After importing the requisite data, to run the interactive shell, you must run the following command:

```sh
python hierarchy_builder.py
```

**Congrats!**  
You can now use the K-Medoids algorithm to cluster your desired regions of the US to your heart's desire!