import numpy as np
from typing import overload
from trex_selector._core.ml_methods.clustering import LinkageMethod

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLoopStrategy:
    SKIPL: LLoopStrategy
    STANDARD: LLoopStrategy
    HCONCAT: LLoopStrategy
    PERMUTATION: LLoopStrategy
    PERMUTATION_DIRECT: LLoopStrategy
    DIRECT: LLoopStrategy

class SolverTypeForTRex:
    TLARS: SolverTypeForTRex
    TLASSO: SolverTypeForTRex
    TENET: SolverTypeForTRex
    TSTEPWISE: SolverTypeForTRex
    TSTAGEWISE: SolverTypeForTRex
    TOMP: SolverTypeForTRex
    TGP: SolverTypeForTRex
    TACGP: SolverTypeForTRex
    TMP: SolverTypeForTRex
    TNCGMP: SolverTypeForTRex
    TOOLS: SolverTypeForTRex
    TAFS: SolverTypeForTRex

# ---------------------------------------------------------------------------
# SolverHyperparameters
# ---------------------------------------------------------------------------

class SolverHyperparameters:
    lambda2: float
    rho_afs: float
    ncgmp_variant: int
    tol: float
    def __init__(self) -> None: ...

# ---------------------------------------------------------------------------
# DummyDistribution
# ---------------------------------------------------------------------------

class DummyDistribution:
    class Type:
        Normal: DummyDistribution.Type
        Uniform: DummyDistribution.Type
        Rademacher: DummyDistribution.Type
        StudentT: DummyDistribution.Type
        Laplace: DummyDistribution.Type
        Gumbel: DummyDistribution.Type
        Holtsmark: DummyDistribution.Type
        Triangle: DummyDistribution.Type
        UniformSphere: DummyDistribution.Type
        Mammen: DummyDistribution.Type
        ConstrainedSparseRademacher: DummyDistribution.Type
        Logistic: DummyDistribution.Type
    uniform_a: float
    student_t_df: float
    laplace_location: float
    laplace_scale: float
    gumbel_location: float
    gumbel_scale: float
    holtsmark_location: float
    holtsmark_scale: float
    triangle_a: float
    triangle_b: float
    triangle_c: float
    sphere_dim: int
    mammen_param_p1: float
    mammen_param_p2: float
    constrained_sparse_rademacher_s: float
    logistic_location: float
    logistic_scale: float
    def __init__(self) -> None: ...
    def get_type(self) -> DummyDistribution.Type: ...
    @staticmethod
    def normal() -> DummyDistribution: ...
    @staticmethod
    def rademacher() -> DummyDistribution: ...
    @staticmethod
    def mammen(p1: float = ..., p2: float = ...) -> DummyDistribution: ...
    @staticmethod
    def uniform(a: float = ...) -> DummyDistribution: ...
    @staticmethod
    def student_t(df: float = ...) -> DummyDistribution: ...
    @staticmethod
    def laplace(location: float = ..., scale: float = ...) -> DummyDistribution: ...
    @staticmethod
    def gumbel(location: float = ..., scale: float = ...) -> DummyDistribution: ...
    @staticmethod
    def holtsmark(location: float = ..., scale: float = ...) -> DummyDistribution: ...
    @staticmethod
    def triangle(a: float = ..., b: float = ..., c: float = ...) -> DummyDistribution: ...
    @staticmethod
    def uniform_sphere(dim: int = ...) -> DummyDistribution: ...
    @staticmethod
    def constrained_sparse_rademacher(s: float) -> DummyDistribution: ...
    @staticmethod
    def logistic(location: float = ..., scale: float = ...) -> DummyDistribution: ...

# ---------------------------------------------------------------------------
# TRexControlParameter
# ---------------------------------------------------------------------------

class TRexControlParameter:
    K: int
    max_dummy_multiplier: int
    use_max_T_stop: bool
    opt_threshold: float
    lloop_strategy: LLoopStrategy
    tloop_stagnation_stop: bool
    tloop_max_stagnant_steps: int
    parallel_rnd_experiments: bool
    use_memory_mapping: bool
    max_outer_threads: int
    max_inner_threads: int
    solver_type: SolverTypeForTRex
    solver_params: SolverHyperparameters
    dummy_distribution: DummyDistribution
    def __init__(self) -> None: ...

# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------

class SelectionResult:
    selected_var: np.ndarray
    v_thresh: float
    r_mat: np.ndarray
    T_stop: int
    num_dummies: int
    dummy_factor_L: int
    fdp_hat_mat: np.ndarray
    phi_mat: np.ndarray
    phi_prime: np.ndarray
    voting_grid: np.ndarray

# ---------------------------------------------------------------------------
# TRexSelector (C++ binding)
# ---------------------------------------------------------------------------

class TRexSelector:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tFDR: float = ...,
        trex_control: TRexControlParameter = ...,
        seed: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    def select(self) -> SelectionResult: ...
    def getTFDR(self) -> float: ...
    def getK(self) -> int: ...
    def getMaxNumDummies(self) -> int: ...
    def getMaxTStop(self) -> bool: ...
    def getStagnationCheck(self) -> bool: ...
    def getXMeans(self) -> np.ndarray: ...
    def getXL2Norms(self) -> np.ndarray: ...
    def getYMean(self) -> float: ...
    def getSelectedVar(self) -> np.ndarray: ...
    def getSelectedIndices(self) -> np.ndarray: ...
    def getNumSelected(self) -> int: ...
    def getTStop(self) -> int: ...
    def getVotingGrid(self) -> np.ndarray: ...
    def getVotingThreshold(self) -> float: ...
    def getFDPHatMat(self) -> np.ndarray: ...
    def getPhiMat(self) -> np.ndarray: ...
    def getRMat(self) -> np.ndarray: ...
    def getPhiPrime(self) -> np.ndarray: ...
    def getDummyMultiplierL(self) -> int: ...

# ---------------------------------------------------------------------------
# DA-TRex
# ---------------------------------------------------------------------------

class DAMethod:
    AR1: DAMethod
    EQUI: DAMethod
    BT: DAMethod
    NN: DAMethod
    PRIOR_GROUPS: DAMethod

class TRexDAControlParameter:
    method: DAMethod
    cor_coef: float
    rho_thr_DA: float
    hc_linkage: LinkageMethod
    hc_grid_length: int
    prior_groups: list[list[int]] # nested list of groups
    rho_grid_labels: list[float]
    def __init__(self) -> None: ...

class DASelectionResult(SelectionResult):
    rho_thresh: float
    rho_grid: np.ndarray
    method: DAMethod
    cor_coef: float
    fdp_hat_array_bt: list[np.ndarray]
    phi_array_bt: list[np.ndarray]
    r_array_bt: list[np.ndarray]

class TRexDASelector(TRexSelector):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tFDR: float = ...,
        da_control: TRexDAControlParameter = ...,
        trex_control: TRexControlParameter = ...,
        seed: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    def select(self) -> DASelectionResult: ...
    def getDAResult(self) -> DASelectionResult: ...

# ---------------------------------------------------------------------------
# GVS-TRex
# ---------------------------------------------------------------------------

class GVSType:
    EN: GVSType
    IEN: GVSType

class TRexGVSControlParameter:
    gvs_type: GVSType
    corr_max: float
    hc_linkage: LinkageMethod
    lambda2_lars: float
    prior_groups: list[int] # flat list
    def __init__(self) -> None: ...

class GVSSelectionResult(SelectionResult):
    lambda2_used: float
    gvs_type: GVSType
    max_clusters: int
    hc_method_used: str
    groups_vec: np.ndarray

class TRexGVSSelector(TRexSelector):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tFDR: float = ...,
        gvs_control: TRexGVSControlParameter = ...,
        trex_control: TRexControlParameter = ...,
        seed: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    def select(self) -> GVSSelectionResult: ...
    def getGVSResult(self) -> GVSSelectionResult: ...

# ---------------------------------------------------------------------------
# Screen-TRex
# ---------------------------------------------------------------------------

class ScreenTRexMethod:
    TREX: ScreenTRexMethod
    TREX_DA_AR1: ScreenTRexMethod
    TREX_DA_EQUI: ScreenTRexMethod
    TREX_DA_BLOCK_EQUI: ScreenTRexMethod

class ScreenTRexControlParameter:
    use_bootstrap_CI: bool
    R_boot: int
    ci_grid_step: float
    trex_method: ScreenTRexMethod
    rho_thr_DA: float
    cor_coef: float
    n_blocks: int
    def __init__(self) -> None: ...

class ScreenTRexSelectionResult(SelectionResult):
    estimated_FDR: float
    used_bootstrap: bool
    confidence_level: float
    beta_mat: np.ndarray
    dummy_betas: np.ndarray
    estimated_correlation: float

class TRexScreeningSelector(TRexSelector):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        screen_control: ScreenTRexControlParameter = ...,
        trex_control: TRexControlParameter = ...,
        seed: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    def select(self) -> ScreenTRexSelectionResult: ...
    def getScreenResult(self) -> ScreenTRexSelectionResult: ...

# ---------------------------------------------------------------------------
# Biobank Screen-TRex
# ---------------------------------------------------------------------------

class BiobankScreenTRexControl:
    target_FDR_trex: float
    lower_bound_FDR: float
    upper_bound_FDR: float
    def __init__(self) -> None: ...

class BiobankScreenTRexResult:
    phenotype_index: int
    selected_indices: np.ndarray
    estimated_FDR: float
    method_used: str
    estimated_FDR_screen_ordinary: float
    estimated_FDR_screen_bootstrap: float
    selected_indices_screen_ordinary: np.ndarray
    selected_indices_screen_bootstrap: np.ndarray
    used_fallback_trex: bool

class TRexBiobankScreeningSelector:
    @overload
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        biosctrex_ctrl: BiobankScreenTRexControl = ...,
        seed: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        biosctrex_ctrl: BiobankScreenTRexControl = ...,
        seed: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    def screenPhenotype(self) -> BiobankScreenTRexResult: ...
    def screenPhenotypes(self) -> list[BiobankScreenTRexResult]: ...

# ---------------------------------------------------------------------------
# T-Rex Sparse PCA
# ---------------------------------------------------------------------------

class SPCAMode:
    ActiveSet: SPCAMode
    """Ridge regression on the T-Rex active set to compute loadings."""
    Thresholded: SPCAMode
    """Keep and normalize the top |A| ordinary PCA loadings."""

class TRexSPCAControlParameter:
    mode: SPCAMode
    """Sparse loading strategy (default: ActiveSet)."""
    lambda2: float
    """Ridge penalty for ActiveSet loading assembly (default: 1e-6)."""
    seed: int
    """Random seed forwarded to each per-PC T-Rex run (-1 = hardware entropy)."""
    trex_ctrl: TRexControlParameter
    """TRexControlParameter forwarded to each per-PC T-Rex run."""
    def __init__(self) -> None: ...

class TRexSPCAResult:
    Z: np.ndarray
    """Sparse PC score matrix (n × M)."""
    V: np.ndarray
    """Sparse loading matrix (p × M)."""
    active_sets: list[list[int]]
    """Support indices per PC (list of M index vectors, 0-based)."""
    adjusted_ev: np.ndarray
    """Marginal adjusted explained variance per component (M-vector)."""
    cumulative_ev: np.ndarray
    """Cumulative percentage of explained variance (M-vector)."""
    def __init__(self) -> None: ...

class TRexSPCA:
    """T-Rex Sparse PCA orchestrator — all methods are static."""
    def __init__(self) -> None: ...
    @staticmethod
    def select(
        X: np.ndarray,
        M: int,
        tFDR: float,
        spca_ctrl: TRexSPCAControlParameter = ...,
    ) -> TRexSPCAResult: ...
