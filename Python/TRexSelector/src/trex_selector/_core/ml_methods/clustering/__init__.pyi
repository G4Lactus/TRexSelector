import numpy as np

# ---------------------------------------------------------------------------
# LinkageMethod
# ---------------------------------------------------------------------------

class LinkageMethod:
    Ward: LinkageMethod
    Average: LinkageMethod
    Complete: LinkageMethod
    Single: LinkageMethod
    WPGMA: LinkageMethod
    Median: LinkageMethod
    Centroid: LinkageMethod

# ---------------------------------------------------------------------------
# DistanceMetric
# ---------------------------------------------------------------------------

class DistanceMetric:
    Euclidean: DistanceMetric
    Correlation: DistanceMetric
    Manhattan: DistanceMetric
    Correlation_LSH_Filter: DistanceMetric
    Correlation_LSH_Approx: DistanceMetric

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def agglomerative_cluster(
    data: np.ndarray,
    method: LinkageMethod,
    metric: DistanceMetric = ...,
    use_mmap: bool = ...,
) -> np.ndarray: ...

def cut_tree(
    linkage: np.ndarray,
    num_orig_objs: int,
    num_clusters: int,
) -> list[int]: ...
