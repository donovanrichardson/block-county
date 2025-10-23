# Analysis of Recursive Splitting Problems

## Problem 1: Zero-Population Districts Being Created Repeatedly

### What's Happening
The logs show that K-Medoids is repeatedly splitting the 5,411 tracts into:
- Cluster 0: 1 tract with population 0
- Cluster 1: 5,410 tracts with population 20,201,249

This happens 53 times in a row, peeling off one zero-population tract at a time.

### Root Cause
New York State has many zero-population tracts (likely parks, water bodies, industrial zones, etc.). The K-Medoids algorithm, when using the weighted distance matrix, tends to isolate these zero-population tracts because:

1. **Distance weighting formula**: `weight = ((dist / (2 * sqrt(pop_a))) + (dist / (2 * sqrt(pop_b))))**2`
2. For zero-population tracts, `pop` is set to `1e-9` (very small), making `1/sqrt(pop)` extremely large
3. This causes edges connected to zero-population tracts to have extremely high weights
4. K-Medoids sees these high-weight connections and isolates the zero-population tracts into their own cluster

### Why Rebalancing Doesn't Fix It
The rebalancing logic checks `if p_0 % target_district_pop != 0 or p_0 == 0`, but:
- When `p_0 = 0`, `next_multiple = ceil(0 / 776971) * 776971 = 0`
- So the condition `p_0 >= next_multiple` is immediately true (0 >= 0)
- No tracts are transferred, and the zero-population cluster is assigned as "1 district"

## Problem 2: Districts with Far More Than Target Population

### What's Happening
At the end of the splitting process, we see districts like:
- `111111111111111111111111111111111111111111111111111100`: pop=2,889,860 (target: 776,971)
- `111111111111111111111111111111111111111111111111111101`: pop=2,550,710 (target: 776,971)

These are 3-4x larger than the target district population.

### Root Cause
The recursive stopping condition is:
```python
if cluster_n == 1:
    # This cluster represents exactly 1 district, finalize it
```

This means a cluster stops splitting when `round(cluster_pop / target_pop) == 1`. 

The problem:
- A cluster with population 2,889,860 has `round(2889860 / 776971) = round(3.72) = 4`
- But wait, the logs show this was assigned `n_0 = 1` district!
- This happens because after rebalancing transferred 599 tracts, the **smaller** cluster got most of the population
- The code assigned it 1 district based on rounding: `round(2889860 / 776971) = 4` (should be 4, not 1!)

Actually, looking more carefully:
- Initial split: cluster 100 pop=900,771, cluster 101 pop=4,539,799
- After rebalance: cluster 100 pop=2,889,860, cluster 101 pop=2,550,710
- Assigned: cluster 100 will have **1** districts, cluster 101 will have **1** districts

The issue is that `round(2889860 / 776971) = 4`, but the parent cluster only had 7 districts to allocate total. The algorithm is trying to balance to match the parent's district count, not to match the target population.

### The Real Problem
The algorithm passes `n_districts` (the number of districts the parent cluster should split into) but this gets out of sync with the actual populations after multiple rebalancing steps. The split of 7 districts into child clusters based on their populations after rebalancing doesn't properly account for the fact that we're trying to create districts of ~776,971 population each.

## Solutions Needed

### Solution 1: Filter Out Zero-Population Tracts
Before running K-Medoids, filter out tracts with zero (or very low) population. Handle them separately by assigning them to their nearest neighbor district after the main clustering is complete.

### Solution 2: Fix District Count Calculation
Instead of using `round(p_i / target_district_pop)`, use a more careful allocation that:
1. Ensures the sum of child district counts equals the parent district count
2. Respects population proportions
3. Never allows a cluster to be finalized with population >> target_district_pop

### Solution 3: Add Stopping Criteria Based on Population Deviation
A cluster should only be finalized (stop splitting) if:
- `cluster_n == 1` AND
- `abs(cluster_pop - target_district_pop) / target_district_pop < threshold` (e.g., 50% tolerance)

Or continue splitting even if `cluster_n == 1` if the population is too far from target.

## Recommended Fix

1. **Pre-filter zero-population tracts**: Remove them before clustering, assign them to nearest districts afterward
2. **Fix district allocation logic**: Use a more careful algorithm that ensures child district counts sum to parent count
3. **Add recursion stopping criteria**: Only stop when population is reasonably close to target OR we can't split further

