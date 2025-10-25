# create a script that will union the block geometries from the block table to create block groups. the resulting table should have approximarely this ddl. should also have spatial index on the geometry columns.:

# CREATE TABLE block_group_2020 (
#     county_geoid text PRIMARY KEY, -- made from the first 12 digits of the block geoid
#     geom geometry(MultiPolygon,4269), -- unioned geometry of all blocks in the block group
#     pop integer NOT NULL, -- sum of the populations of all blocks in the block group
#     centroid_geom geometry(Point,4269) NOT NULL, -- population-weighted centroid of the block group, obtained by the same general method used in county_pop_weighted_centroid.py
#     );