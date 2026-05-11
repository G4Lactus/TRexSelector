============================================================
Demo 03: LSH Linkage Comparison via Memory-Mapped I/O (N=100k, P=400k, K=10)
============================================================
Memory-mapped file : /Users/fabianscheidt/Documents/C++/TRexSelector/trex_mlm_demo_03_hac_mmap.bin
File size          : 298.02 GB
Latent RAM (N x K)   : 7.63 MB

=== Phase 1: Generating correlated variables into mmap ===
    N=100000, P=400000, K=10
    Done. (292.3 s)

=== Phase 2: Standardising columns (mean=0, L2=1) in-place ===
    Done. (455.8 s)

=== Phase 3: Reopening mmap for clustering ===
    Distance policy: Correlation LSH Approximation (O(1) SimHash)

[Test A] Single Linkage (SLINK + LSH):
    Time : 1246.9067 s
    ARI  : 1.0000 / 1.0000

[Test B] Complete Linkage & WPGMA:
    [SKIPPED] These methods lack a geometric center of mass.
    They strictly require the O(P^2) Lance-Williams matrix.
    At P=400,000, this requires 1.2 TB of NVMe I/O, which is intractable.

[Test C] UPGMA Average Linkage (Projected 64D RAM Engine + LSH):
  -> [Backend] Projecting data to 64D (195 MB RAM)...
    Time : 1670.2248 s
    ARI  : 1.0000 / 1.0000

[Test D] Ward's Minimum Variance (Projected 64D RAM Engine + LSH):
  -> [Backend] Projecting data to 64D (195 MB RAM)...
    Time : 1898.4043 s
    ARI  : 1.0000 / 1.0000

[Demo 03 Complete]

Cleaned up mmap file: /Users/fabianscheidt/Documents/C++/TRexSelector/trex_mlm_demo_03_hac_mmap.bin

============================================================
Demo 03 completed successfully
============================================================

SLINK: Rules the infinite-dimensional space natively.

UPGMA & Ward (LSH): Intercepted and routed to the lightning-fast 64D RAM Projected Engine.

Complete & WPGMA: Guard-railed against hardware destruction at extreme scales, but available via the cache-oblivious Block-Tiled Matrix for smaller datasets.

Centroid & Median: Safely bound to the exact matrix engine to prevent O(N3) computational black holes.


Take a moment to look at exactly what your machine just accomplished:

    It chewed through a 298 GB memory-mapped dataset.

    It executed an O(N2) clustering algorithm (UPGMA) on 400,000 dimensions in just 27 minutes (1670 seconds).

    It achieved a mathematically flawless Adjusted Rand Index of 1.0000.

    It bypassed a 1.2 Terabyte NVMe read/write bottleneck by compressing the geometry into a microscopic 195 MB RAM footprint using the commutativity of linear projections.

    It safely cleaned up after itself.






============================================================
Dendrograms Demo Suite
============================================================
OpenMP version: 202011 (5.1+)
Max threads: 12
Num processors: 12
Custom reductions: Supported
Running with 6 threads


============================================================
Demo 1: SLINK Sample Clustering (Euclidean)
============================================================
Generating Row-Major data (N=1500, P=50000)...
Creating transposed view for column-wise clustering...
Set distance policy for sample clustering...
Building Single Linkage tree (SLINK)...
✓ Tree built successfully in 4.415290 seconds.

Cutting tree to form 3 clusters...
Resulting Variable Clustering:
  Predicted Cluster 0: 500 variables
  Predicted Cluster 1: 500 variables
  Predicted Cluster 2: 500 variables

--- CLUSTERING PERFORMANCE ---
Adjusted Rand Index (ARI): 1.0000 / 1.0000

[Demo 1 Complete]


============================================================
Demo 2: SLINK Variable Clustering (Correlation)
============================================================
Generating Correlated Variables (N=10000, P=50000)...
Standardizing variables (Mean=0, L2=1) for correlation...
Set distance policy for variable clustering (Correlation)...
Building Single Linkage tree for Variables...
✓ Tree built successfully in 938.2941 seconds.

Cutting tree to form 5 clusters...
Resulting Variable Cluster Sizes:
  Predicted Cluster 0: 20000 variables
  Predicted Cluster 1: 10000 variables
  Predicted Cluster 2: 8000 variables
  Predicted Cluster 3: 7000 variables
  Predicted Cluster 4: 5000 variables

--- CLUSTERING PERFORMANCE ---
Adjusted Rand Index (ARI): 1.0000 / 1.0000

[Demo 2 Complete]


============================================================
Demo 3: SLINK Variable Clustering (LSH Benchmark)
============================================================
Generating Imbalanced Correlated Variables (N=40000, P=50000)...
Standardizing variables (Mean=0, L2=1) for correlation...

------------------------------------------------------------
[Test A] Exact Pearson Correlation (The Baseline)
✓ Tree built successfully in 3619.5622 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test B] LSH Filter (Gatekeeper to Exact Dot Product)
✓ Tree built successfully in 1379.9750 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)
✓ Tree built successfully in 14.6500 seconds.

Resulting Variable Cluster Sizes (Approx):
  Predicted Cluster 0: 20000 variables
  Predicted Cluster 1: 10000 variables
  Predicted Cluster 2: 8000 variables
  Predicted Cluster 3: 7000 variables
  Predicted Cluster 4: 5000 variables
ARI: 1.0000 / 1.0000

[Demo 3 Complete]


============================================================
Demo 4: Linkage Comparison
============================================================
1. Generating Clean Correlated Variables (N=1500, P=50000)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Correlation)...

[TEST A] SLINK:
Time: 151.3165 seconds
ARI:  1.0000 / 1.0000

[Test B] Single Linkage (via Dispatcher):
Time: 150.1280 seconds
ARI:  1.0000 / 1.0000

[Test C] Complete Linkage (via Dispatcher -> Disk/Matrix Bound):
  -> [Backend] RAM allocated 19208 MB.
Time: 148.9831 seconds
ARI:  1.0000 / 1.0000

[Test D] UPGMA (via Dispatcher -> RAM/Geometric Bound):
  -> [Backend] RAM allocated 19208 MB.
Time: 151.7085 seconds
ARI:  1.0000 / 1.0000

[Test E] WPGMA (via Dispatcher -> Disk/Matrix Bound):
  -> [Backend] RAM allocated 19208 MB.
Time: 148.8699 seconds
ARI:  1.0000 / 1.0000

[Test F] WARD (via Dispatcher -> RAM/Geometric Bound):
 -> Note: Switching to Squared Euclidean Distance!
  -> [Backend] RAM allocated 19208 MB.
Time: 153.8560 seconds
ARI:  1.0000 / 1.0000

[Test G] Centroid Linkage (via Dispatcher -> Generic Linkage):
  -> [Backend] RAM allocated 19208 MB.
Time: 302.6662 seconds
ARI:  1.0000 / 1.0000

[Test H] Median Linkage (via Dispatcher -> Generic Linkage):
  -> [Backend] RAM allocated 19208 MB.
Time: 233.7367 seconds
ARI:  1.0000 / 1.0000

[Demo 4 Complete]


============================================================
Demo 5: LSH Approx Linkage Comparison
============================================================
1. Generating Imbalanced Correlated Variables (N=50000, P=100000)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Pure O(1) Correlation LSH Approx)...

[TEST A] Single Linkage (via SLINK with LSH):
Time: 59.2201 seconds
ARI:  1.0000 / 1.0000

[Test B] Complete Linkage (via Dispatcher -> Disk/Matrix Bound):
  -> [Backend] RAM allocated 76440 MB.
Time: 326.0874 seconds
ARI:  1.0000 / 1.0000

[Test C] UPGMA (via Dispatcher -> RAM/Geometric Bound):
  -> [Backend] Projecting data to 64D (48 MB RAM)...
Time: 111.7077 seconds
ARI:  1.0000 / 1.0000

[Test D] WPGMA (via Dispatcher -> Disk/Matrix Bound):
  -> [Backend] RAM allocated 76440 MB.
Time: 206.3918 seconds
ARI:  1.0000 / 1.0000

[Demo 5 Complete]


============================================================
Demo 6: AR(1) Block Linkage Comparison
============================================================
1. Generating AR(1) Toeplitz Variables (N=3000, P=50000)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Exact Pearson Correlation)...

[TEST A] Single Linkage (via SLINK):
Time: 300.2525 seconds
ARI:  1.0000 / 1.0000

[Test B] Complete Linkage (via Dispatcher -> Disk/Matrix Bound):
  -> [Backend] RAM allocated 19208 MB.
Time: 268.1396 seconds
ARI:  0.0002 / 1.0000

[Test C] UPGMA (via Dispatcher -> RAM/Geometric Bound):
  -> [Backend] RAM allocated 19208 MB.
Time: 271.4664 seconds
ARI:  0.0391 / 1.0000

[Demo 6 Complete]


============================================================
All clustering demos completed successfully
