create a script similar to [create_hierarchy_start.py](create_hierarchy_start.py) except that it only includes the counties whose geoid is in county_geoids = ["36103", "36059", "36081", "36047"]

- id: whatever the default value is
- parent: null
- hierarchy: 'LI'
- level: 0
- label: 0 (i altered the schema to make label numeric)
- geom: the merged geometry of LI counties
- centroid_geom: null
- pop: the total population of LI counties combined