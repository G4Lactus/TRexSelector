# Summary: Modern Hierarchical, Agglomerative Clustering Algorithms

**Author:** Daniel Müllner (2011) | **Scope:** Sections 1 & 2 (Foundations)

## 1. The Core Objective

* Despite algorithms being solved theoretically by Sibson, Rohlf, and Murtagh, mainstream software
  (R, SciPy) largely relied on suboptimal implementations.
* The paper's goal is to bridge this gap by providing strictly optimal algorithms for the seven
  standard SAHN (sequential, agglomerative, hierarchic, nonoverlapping) methods: single, complete,
  average, weighted, Ward, centroid, and median.

## 2. Input Limits and the $\Omega(N^2)$ Boundary

* The standard "stored matrix" input is a dissimilarity index representing $\Theta(N^2)$ pairwise
   distances.
* Because hierarchical clustering is sensitive to every single input value, every algorithm is
    mathematically bounded by a minimum time and space complexity of $\Omega(N^2)$.

## 3. The "Stepwise Dendrogram" Output Standard

* Modern software expects a "stepwise dendrogram" as output: an $(N-1) \times 3$ list recording the
   two merged nodes and the merging distance at each step.

* **Inversions:** Unlike abstract mathematical dendrograms, the stepwise data structure natively accommodates inversions (where merging distances decrease).
  
* **Ties:** If multiple clusters merge at the exact same distance, the execution order matters.
  Algorithms that ignore tie-breaking (like standard SLINK) fail to produce a valid stepwise
  dendrogram for tied data.

## 4. Memory-Efficient Label Generation

* To achieve peak efficiency, algorithms should perform in-place distance updates by reusing the
 array indices of the merged clusters.
* The raw output can then be converted into standard software node labels
  (e.g., SciPy or R conventions) in $\Theta(N)$ time using a Union-Find data structure.

## 5. The "Primitive" Baseline and Lance-Williams

* The procedural, brute-force search algorithm requires $\Theta(N^3)$ time because it searches the
    entire active distance matrix at every step.

* Distance updates between merged clusters and remaining clusters are handled in $O(1)$ time via
     the Lance-Williams formula.

* Among the seven standard methods, only **Centroid** and **Median** fail the condition preventing
    inversions, making them mathematically distinct from the others.

**Scope:** Section 3 (Algorithms)

## 1. Overview of the Three Algorithms

* The section presents three highly efficient algorithms for the "stored matrix" approach: `GENERIC_LINKAGE`, `NN-CHAIN`, and `MST-LINKAGE`.
* Any advanced algorithm is mathematically correct if and only if it produces an output that
  perfectly matches one of the valid stepwise dendrograms produced by the primitive baseline algorithm.

## 2. Section 3.1: The Generic Clustering Algorithm (`GENERIC_LINKAGE`)

* **Purpose:** This is a universal algorithm capable of using *any* distance update formula, making it the only algorithm in the paper that can safely handle dendrogram inversions (which are caused by the Centroid and Median methods).
* **Mechanism:** It builds upon Anderberg's nearest-neighbor approach but uses a priority queue (binary heap) to track lower bounds of distances (`mindist`) and candidates for nearest neighbors (`n_nghbr`).
* **Lazy Evaluation:** Instead of aggressively recalculating a cluster's nearest neighbor every time a merge happens, the algorithm lazily defers the search until that cluster's lower-bound distance reaches the top of the priority queue.
* **Complexity:** While the theoretical worst-case remains $O(N^3)$ due to potential repeated nearest-neighbor searches, the lazy evaluation prevents this behavior in practice, resulting in significantly faster real-world execution.

## 3. Section 3.2: The Nearest-Neighbor-Chain Algorithm (`NN-CHAIN`)

* **Purpose:** Originally described by Murtagh, this algorithm is mathematically proven by Müllner to work for the Single, Complete, Average, Weighted, and Ward methods.
* **The Reducibility Wall:** Müllner provides a strict proof correcting previous literature: `NN-CHAIN` works *only* if the distance update formula satisfies the "reducibility property" and if the distance between disjoint clusters is independent of the order in which they were merged.
* **Mechanism:** It searches for "reciprocal nearest neighbors" (where cluster A is closest to B, and B is closest to A) and merges them immediately, regardless of whether smaller distances exist elsewhere in the matrix.
* **Post-Processing:** `NN-CHAIN` outputs an "unsorted dendrogram". To become a valid stepwise dendrogram, it must be stably sorted by distance, followed by a $\Theta(N)$ Union-Find pass to generate correct node labels.

## 4. Section 3.3: The Single Linkage Algorithm (`MST-LINKAGE`)

* **Purpose:** Based on Gower and Ross's observation that Single Linkage clustering can be derived from a Minimum Spanning Tree (MST), Müllner presents Rohlf's algorithm (a simplified Prim's algorithm).
* **Handling Ties:** Müllner explicitly proves that `MST-LINKAGE` naturally and correctly resolves exact distance ties, whereas Sibson's standard `SLINK` algorithm does not.
* **Post-Processing:** Just like `NN-CHAIN`, the `MST-LINKAGE` core algorithm produces an unsorted dendrogram that seamlessly flows through the exact same stable sort and Union-Find pipeline.

**Author:** Daniel Müllner (2011) | **Scope:** Sections 4, 5, & 6

## 1. Section 4: Performance & Recommendations

* **Theoretical vs. Practical:** Müllner benchmarks both asymptotic worst-case bounds and empirical use-case CPU times.
* **Centroid/Median:** Anderberg's $O(N^3)$ primitive algorithm consistently hits its cubic worst-case in random data, whereas `GENERIC_LINKAGE` stays comfortably in a quadratic curve. 
* **The Final Recommendations:**
  * *Single Linkage:* `MST-LINKAGE` is recommended for its speed and low memory ($O(N)$) footprint. *(Note: Our modern C++ compiler actually favors Sibson's SIMD-friendly layout).*
  * *Complete, Average, WPGMA, Ward:* `NN-CHAIN` is the definitive choice, guaranteeing $O(N^2)$ execution with no performance drawbacks.
  * *Centroid, Median:* `GENERIC_LINKAGE` is strictly required to avoid the inversion traps and $O(N^3)$ bottlenecks.

## 2. Section 5: Alternative Algorithms

* **Sibson's SLINK:** Müllner acknowledges SLINK shares the optimal $O(N^2)$ time and $O(N)$ space properties, and uniquely reads inputs in a fixed sequence, which is great for disk-streaming. However, it requires a secondary lexicographical order check to safely handle ties.
* **Defays' Complete Linkage:** Müllner explicitly warns against using Defays' (1977) algorithm, stating it "definitely is not an algorithm for complete linkage clustering, as the complete linkage method is commonly defined".
* **Day-Edelsbrunner:** While mathematically achieving $O(N^2 \log N)$ using $N$ priority queues, the massive bookkeeping overhead makes it practically inefficient. 

## 3. Section 6: Extension to Vector Data (The Hyperscale Solution)

* **Breaking the $\Omega(N^2)$ Memory Wall:** If the input is raw $D$-dimensional vectors rather than a precomputed dissimilarity matrix, the lower limit on memory no longer applies.
* **On-The-Fly Computation:** Algorithms like `MST-LINKAGE` and `NN-CHAIN` (specifically for Ward's method) can compute distances strictly on-the-fly using geometric cluster centroids.
* **Parallelization:** For very high-dimensional data, the actual distance computations (dot products/Euclidean norms) become the bottleneck, making parallelized nearest-neighbor searches highly effective.

---

## Simulation results

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
✓ Tree built successfully in 4.401208 seconds.

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
✓ Tree built successfully in 959.4239 seconds.

Cutting tree to form 5 clusters...
Resulting Sample Cluster Sizes:
  Predicted Cluster 0: 20000 samples
  Predicted Cluster 1: 10000 samples
  Predicted Cluster 2: 8000 samples
  Predicted Cluster 3: 7000 samples
  Predicted Cluster 4: 5000 samples

--- CLUSTERING PERFORMANCE ---
Adjusted Rand Index (ARI): 1.0000 / 1.0000

[Demo 2 Complete]


============================================================
Demo 3: SLINK Variable Clustering (Correlation + LSH)
============================================================
Generating Correlated Variables (N=1500, P=50000)...
Standardizing variables (Mean=0, L2=1) for correlation...
Set distance policy for variable clustering (Correlation + LSH)...
Building Single Linkage tree (LSH Engine Active)...
  -> Initializing LSH Hyperplanes for 50000 variables...
✓ Tree built successfully in 53.4219 seconds.

Cutting tree to form 5 clusters...
Resulting Sample Cluster Sizes:
  Predicted Cluster 0: 20000 samples
  Predicted Cluster 1: 10000 samples
  Predicted Cluster 2: 8000 samples
  Predicted Cluster 3: 7000 samples
  Predicted Cluster 4: 5000 samples

--- CLUSTERING PERFORMANCE ---
Adjusted Rand Index (ARI): 1.0000 / 1.0000

[Demo 3 Complete]


============================================================
Demo 4: Linkage Comparison
============================================================
1. Generating Clean Correlated Variables (N=1500, P=50000)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Correlation)...

[TEST A] SLINK:
Time: 154.6600 seconds
ARI:  1.0000 / 1.0000

[Test B] CLINK (Complete Linkage via condensed NN-Chain):
Time: 190.7134 seconds
ARI:  1.0000 / 1.0000

[Test C] UPGMA (Average Linkage via condensed NN-Chain):
Time: 189.3886 seconds
ARI:  1.0000 / 1.0000

[Test D] WPGMA (Weighted Average Linkage via condensed NN-Chain):
Time: 188.7456 seconds
ARI:  1.0000 / 1.0000

[Test E] WARD (Ward's Minimum Variance via condensed NN-Chain):
 -> Note: Switching to Squared Euclidean Distance!
Time: 195.6451 seconds
ARI:  1.0000 / 1.0000

[Test F] Single Linkage via condensed NN-CHAIN (Sanity Check with SLINK):
Time: 188.2486 seconds
ARI:  1.0000 / 1.0000

[Test G] Rohlf's MST-based SLINK:
Time: 328.6945 seconds
ARI:  1.0000 / 1.0000

[Test H] Centroid Linkage via Müllner's Generic_Linkage:
 -> Note: Switching to Squared Euclidean Distance & 2D Matrix!
Time: 599.6465 seconds
ARI:  1.0000 / 1.0000

[Test I] Median Linkage via Müllner's Generic_Linkage:
 -> Note: Switching to Squared Euclidean Distance & 2D Matrix!
Time: 447.5223 seconds
ARI:  1.0000 / 1.0000

[Demo 4 Complete]



============================================================
Demo 3: SLINK Variable Clustering (LSH Benchmark)
============================================================
Generating Imbalanced Correlated Variables (N=10000, P=50000)...
Standardizing variables (Mean=0, L2=1) for correlation...

------------------------------------------------------------
[Test A] Exact Pearson Correlation (The Baseline)
✓ Tree built successfully in 921.032577 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test B] LSH Filter (Gatekeeper to Exact Dot Product)
  -> Initializing LSH Hyperplanes for 50000 variables...
✓ Tree built successfully in 360.5243 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)
  -> Initializing LSH Hyperplanes for 50000 variables...
✓ Tree built successfully in 19.5841 seconds.

Resulting Variable Cluster Sizes (Approx):
  Predicted Cluster 0: 20000 variables
  Predicted Cluster 1: 10000 variables
  Predicted Cluster 2: 8000 variables
  Predicted Cluster 3: 7000 variables
  Predicted Cluster 4: 5000 variables
ARI: 1.0000 / 1.0000

[Demo 3 Complete]

Generating Imbalanced Correlated Variables (N=40000, P=50000)...
Standardizing variables (Mean=0, L2=1) for correlation...

------------------------------------------------------------
[Test A] Exact Pearson Correlation (The Baseline)
✓ Tree built successfully in 3649.579695 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test B] LSH Filter (Gatekeeper to Exact Dot Product)
  -> Initializing LSH Hyperplanes for 50000 variables...
✓ Tree built successfully in 1387.9006 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)
  -> Initializing LSH Hyperplanes for 50000 variables...
✓ Tree built successfully in 14.3546 seconds.

Resulting Variable Cluster Sizes (Approx):
  Predicted Cluster 0: 20000 variables
  Predicted Cluster 1: 10000 variables
  Predicted Cluster 2: 8000 variables
  Predicted Cluster 3: 7000 variables
  Predicted Cluster 4: 5000 variables
ARI: 1.0000 / 1.0000

[Demo 3 Complete]


============================================================
Demo 5: LSH Approx Linkage Comparison
============================================================
1. Generating Imbalanced Correlated Variables (N=25000, P=50000)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Pure O(1) Correlation LSH Approx)...

[TEST A] Single Linkage (via SLINK with LSH):
  -> Initializing LSH Hyperplanes for 50000 variables...
Time: 16.049214 seconds
ARI:  1.0000 / 1.0000

[Test B] CLINK (Complete Linkage via NN-Chain with LSH):
  -> Initializing LSH Hyperplanes for 50000 variables...
Time: 26.4025 seconds
ARI:  1.0000 / 1.0000

[Test C] UPGMA (Average Linkage via NN-Chain with LSH):
  -> Initializing LSH Hyperplanes for 50000 variables...
Time: 26.3337 seconds
ARI:  1.0000 / 1.0000

[Test D] WPGMA (Weighted Average via NN-Chain with LSH):
  -> Initializing LSH Hyperplanes for 50000 variables...
Time: 25.2710 seconds
ARI:  1.0000 / 1.0000

[Demo 5 Complete]
