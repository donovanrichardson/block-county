# Prompt

Create a script in this directory basically exactly like [assign_subdistricts_long_island.py](../long_island/assign_subdistricts_long_island.py) except that it will be used to create hypothetical congressional districts for New York State. This means it should grab all tracts centroids beginning with 36 in county_centroids2 (county_geoid begins with 36 AND type = "11"). Then the script should accept an integer of 2 or higher for the number of districts to split the region into. 

## Procedure

- initial inputs are county_centroids2 tract centroids t (geoid beginning with 36) and number of districts n >= 2, and a cluster name cn that begins as empty in the first iteration. the population of the tracts is referred to as p_t
- just like before, a delaunay is used to create a graph of the t g_t. and a distance matrix is created m_t.
  - m_t should be computed once and used throughout the several calls to this function.
- as before, n and m_t are used as inputs to kmedioids
- the kmedioids outputs clusters of t, called t_0, and t_1, and medioids m_0 and m_1.
  - at this point, summarize the populations of the clusters; the one with the lower population will be labled t_0 and get the cluster name (cn_0) of concat(cn,0), and the other t_1 will get the cluster name (cn_1) of concat(cn,1)
- the population of t_0 us p_0. if p_0 mod (p_t/n) is exactly 0, then skip this step (future impl should use a tolerance but i am not sure what that should be)
  - all tracts have a distance from m_0 called d_0, and a distance from m_1 called d_1.
  - create a list of tracts in t_1 sorted by d_1/d_0, this ratio will be called r_1. These r_1 values should all be less than 1.
  - tracts in t_1 with the highest r_1 will then be successively reassigned to t_0 until p_0 reaches at or beyond the next `mod (p_t/n)`. Because t_0 initially had the lower population, there should be no case in which t_0 ends up with all of the input tracts t. in your impl please add a comment showing that you understand what  "until p_0 reaches at or beyond the next `mod (p_t/n)`" means.
- for both clusters c (t_0, t_1 are the members of c)
  - population of c is p_c
  - if round(p_c/(p_t/n)) = 1, then do the normal procedure of inserting the members of c into the DISTRICT_TABLE with `parent` equal to its new cluster name (the value of cn_0 or cn_1), and it should get medioid = true if it is the member of the cluster with the lowest sum distance to all other members of c according to m_t
  - else if the population of c is hiigher (rounds to 2 or more), then recursively call this entire procedure with input tracts being the members of c, input number of districts round(p_c/(p_t/n)), and cluster name equal to the cluster name of c (cn_0 or cn_1)
- IMPORTANT: please add progress indicators where appropriate.