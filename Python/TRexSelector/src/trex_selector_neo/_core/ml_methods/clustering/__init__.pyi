"""
Hierarchical clustering algorithms
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Average', 'Centroid', 'Complete', 'Correlation', 'Correlation_LSH_Approx', 'Correlation_LSH_Filter', 'DistanceMetric', 'Euclidean', 'LinkageMethod', 'Manhattan', 'Median', 'Single', 'WPGMA', 'Ward', 'agglomerative_cluster', 'cut_tree']
class DistanceMetric:
    """
    Members:
    
      Euclidean
    
      Correlation
    
      Manhattan
    
      Correlation_LSH_Filter
    
      Correlation_LSH_Approx
    """
    Correlation: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.Correlation: 1>
    Correlation_LSH_Approx: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.Correlation_LSH_Approx: 4>
    Correlation_LSH_Filter: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.Correlation_LSH_Filter: 3>
    Euclidean: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.Euclidean: 0>
    Manhattan: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.Manhattan: 2>
    __members__: typing.ClassVar[dict[str, DistanceMetric]]  # value = {'Euclidean': <DistanceMetric.Euclidean: 0>, 'Correlation': <DistanceMetric.Correlation: 1>, 'Manhattan': <DistanceMetric.Manhattan: 2>, 'Correlation_LSH_Filter': <DistanceMetric.Correlation_LSH_Filter: 3>, 'Correlation_LSH_Approx': <DistanceMetric.Correlation_LSH_Approx: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LinkageMethod:
    """
    Members:
    
      Ward
    
      Average
    
      Complete
    
      Single
    
      WPGMA
    
      Median
    
      Centroid
    """
    Average: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.Average: 1>
    Centroid: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.Centroid: 6>
    Complete: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.Complete: 2>
    Median: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.Median: 5>
    Single: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.Single: 3>
    WPGMA: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.WPGMA: 4>
    Ward: typing.ClassVar[LinkageMethod]  # value = <LinkageMethod.Ward: 0>
    __members__: typing.ClassVar[dict[str, LinkageMethod]]  # value = {'Ward': <LinkageMethod.Ward: 0>, 'Average': <LinkageMethod.Average: 1>, 'Complete': <LinkageMethod.Complete: 2>, 'Single': <LinkageMethod.Single: 3>, 'WPGMA': <LinkageMethod.WPGMA: 4>, 'Median': <LinkageMethod.Median: 5>, 'Centroid': <LinkageMethod.Centroid: 6>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def agglomerative_cluster(data: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], method: LinkageMethod, metric: DistanceMetric = ..., use_mmap: bool = False) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Perform hierarchical agglomerative clustering, returning a SciPy-style [N-1, 4] linkage matrix.
    """
def cut_tree(linkage: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], num_orig_objs: typing.SupportsInt | typing.SupportsIndex, num_clusters: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
    """
    Cut a hierarchical linkage matrix to form exactly `num_clusters` flat clusters.
    """
Average: LinkageMethod  # value = <LinkageMethod.Average: 1>
Centroid: LinkageMethod  # value = <LinkageMethod.Centroid: 6>
Complete: LinkageMethod  # value = <LinkageMethod.Complete: 2>
Correlation: DistanceMetric  # value = <DistanceMetric.Correlation: 1>
Correlation_LSH_Approx: DistanceMetric  # value = <DistanceMetric.Correlation_LSH_Approx: 4>
Correlation_LSH_Filter: DistanceMetric  # value = <DistanceMetric.Correlation_LSH_Filter: 3>
Euclidean: DistanceMetric  # value = <DistanceMetric.Euclidean: 0>
Manhattan: DistanceMetric  # value = <DistanceMetric.Manhattan: 2>
Median: LinkageMethod  # value = <LinkageMethod.Median: 5>
Single: LinkageMethod  # value = <LinkageMethod.Single: 3>
WPGMA: LinkageMethod  # value = <LinkageMethod.WPGMA: 4>
Ward: LinkageMethod  # value = <LinkageMethod.Ward: 0>
