# Dump (compressed custom format; good default)
pg_dump -h localhost -p 5432 -U donovan \
-Fc -Z9 \
-f block-county_$(date +%F).dump \
block-county
