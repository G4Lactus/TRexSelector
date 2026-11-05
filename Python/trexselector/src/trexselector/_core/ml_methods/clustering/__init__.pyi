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
    Pairwise distance between data columns for the agglomerative clustering: exact metrics and
    two locality-sensitive hashing (SimHash) approximations of the correlation distance.
    
    References
    ----------
    M. S. Charikar, "Similarity Estimation Techniques from Rounding Algorithms," in Proc.
      ACM Symp. Theory Comput. (STOC), 2002.
    P. Indyk and R. Motwani, "Approximate Nearest Neighbors: Towards Removing the Curse of
      Dimensionality," in Proc. 30th ACM Symp. Theory Comput. (STOC), 1998, pp. 604-613.
    P. Li, T. J. Hastie, and K. W. Church, "Very Sparse Random Projections," in Proc. ACM
      SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD), 2006.
    
    
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
    @typing.overload
    def __eq__(self, other: DistanceMetric) -> bool:
        ...
    @typing.overload
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
    @typing.overload
    def __ne__(self, other: DistanceMetric) -> bool:
        ...
    @typing.overload
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
    Linkage criterion of the agglomerative clustering, applied through the Lance-Williams
    update when two clusters merge.
    
    References
    ----------
    G. N. Lance and W. T. Williams, "A General Theory of Classificatory Sorting Strategies:
      I. Hierarchical Systems," Comput. J., vol. 9, pp. 373-380, 1967.
    D. Müllner, "Modern Hierarchical, Agglomerative Clustering Algorithms," arXiv:1109.2378,
      2011.
    
    
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
    @typing.overload
    def __eq__(self, other: LinkageMethod) -> bool:
        ...
    @typing.overload
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
    @typing.overload
    def __ne__(self, other: LinkageMethod) -> bool:
        ...
    @typing.overload
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
def agglomerative_cluster(data: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], method: LinkageMethod, metric: DistanceMetric = ..., use_mmap: bool = False, verbose: bool = False, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Perform hierarchical agglomerative clustering, returning a SciPy-style [N-1, 4] linkage matrix. verbose=True prints the backend's allocation line (the size of the distance matrix in RAM or in the memory-mapped file). n_threads is the thread budget of the distance loops (None: the library's default budget, utils.get_num_threads(), initially 1).
    
    References
    ----------
    R. Sibson, "SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method,"
      Comput. J., vol. 16, no. 1, pp. 30-34, 1973.
    J.-P. Benzécri, "Construction d'une classification ascendante hiérarchique par la
      recherche en chaîne des voisins réciproques," Les Cahiers de l'Analyse des Données,
      vol. 7, no. 2, pp. 209-218, 1982.
    J. Juan, "Programme de classification hiérarchique par l'algorithme de la recherche en
      chaîne des voisins réciproques," Les Cahiers de l'Analyse des Données, vol. 7, no. 2,
      pp. 219-225, 1982.
    G. N. Lance and W. T. Williams, "A General Theory of Classificatory Sorting Strategies:
      I. Hierarchical Systems," Comput. J., vol. 9, pp. 373-380, 1967.
    D. Müllner, "Modern Hierarchical, Agglomerative Clustering Algorithms," arXiv:1109.2378,
      2011.
    F. Murtagh and P. Contreras, "Algorithms for Hierarchical Clustering: An Overview,"
      Wiley Interdiscip. Rev. Data Min. Knowl. Discov., vol. 2, no. 1, pp. 86-97, 2012,
      doi: 10.1002/widm.53.
    F. Murtagh and P. Contreras, "Algorithms for Hierarchical Clustering: An Overview, II,"
      Wiley Interdiscip. Rev. Data Min. Knowl. Discov., vol. 7, no. 6, Art. no. e1219, 2017,
      doi: 10.1002/widm.1219.
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
