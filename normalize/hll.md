create a new script that will create a table called hll. it needs the following columns:

id text primary key
parent text (correspoonds to id of parent geom)
hierarchy text
level text
label text
geom (for polygons), geometry(MultiPolygon,4269)
centroid_geom (for centroids) geometry(Point,4269)
pop numeric(12,0)

all columns need the appropriate indices.

hierarchy, level, label combination must be unique.