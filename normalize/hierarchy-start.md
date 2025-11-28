create a script which will merge all the counties in the US toghether, and insert the resulting single record into the hll table with teh following values:
- id: whatever the default value is
- parent: null
- hierarchy: 'US'
- level: 0
- label: 0 (i altered the schema to make label numeric)
- geom: the merged geometry of all counties
- centroid_geom: null
- pop: the total population of all counties combined