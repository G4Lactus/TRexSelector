"""
T-Rex Selector variants and core algorithms
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AR1', 'ActiveSet', 'BT', 'BTSelectionMode', 'BiobankScreenTRexControl', 'BiobankScreenTRexResult', 'CV_1SE_CCD', 'CV_1SE_SVD', 'CV_MIN_CCD', 'CV_MIN_SVD', 'DAMethod', 'DASelectionResult', 'DummyDistribution', 'EN', 'ENSolverType', 'EQUI', 'FeasibleOnly', 'GVSSelectionResult', 'GVSType', 'HCONCAT', 'IEN', 'L2', 'LLoopStrategy', 'LambdaSelectionMethod', 'NN', 'ONDEMAND', 'PERMUTATION', 'PERMUTATION_ONDEMAND', 'PRIOR_GROUPS', 'RFaithful', 'SKIPL', 'SPCAMode', 'STANDARD', 'ScalingMode', 'ScreenTRexControlParameter', 'ScreenTRexMethod', 'ScreenTRexSelectionResult', 'SelectionResult', 'SolverHyperparameters', 'SolverTypeForTRex', 'TACGP', 'TAFS', 'TENET', 'TENET_AUG', 'TGP', 'TIENET_AUG', 'TLARS', 'TLASSO', 'TMP', 'TNCGMP', 'TOMP', 'TOOLS', 'TREX', 'TREX_DA_AR1', 'TREX_DA_BLOCK_EQUI', 'TREX_DA_EQUI', 'TRexBiobankScreeningSelector', 'TRexControlParameter', 'TRexDAControlParameter', 'TRexDASelector', 'TRexGVSControlParameter', 'TRexGVSSelector', 'TRexSPCAControlParameter', 'TRexSPCAResult', 'TRexSPCASelector', 'TRexScreeningSelector', 'TRexSelector', 'TSTAGEWISE', 'TSTEPWISE', 'Thresholded', 'ZSCORE']
class BTSelectionMode:
    """
    Cell-selection policy for the BT (binary-tree) DA method.
    
    Members:
    
      FeasibleOnly : Only FDP-feasible cells may be selected (default; controls realized FDR).
    
      RFaithful : Reproduces the R reference; may inflate FDR at high within-group correlation.
    """
    FeasibleOnly: typing.ClassVar[BTSelectionMode]  # value = <BTSelectionMode.FeasibleOnly: 0>
    RFaithful: typing.ClassVar[BTSelectionMode]  # value = <BTSelectionMode.RFaithful: 1>
    __members__: typing.ClassVar[dict[str, BTSelectionMode]]  # value = {'FeasibleOnly': <BTSelectionMode.FeasibleOnly: 0>, 'RFaithful': <BTSelectionMode.RFaithful: 1>}
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
class BiobankScreenTRexControl:
    """
    Control parameters for TRexBiobankScreeningSelector.
    """
    def __init__(self) -> None:
        ...
    @property
    def lower_bound_FDR(self) -> float:
        """
        Lower bound on the FDR search range.
        """
    @lower_bound_FDR.setter
    def lower_bound_FDR(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def target_FDR_trex(self) -> float:
        """
        Target FDR for the core T-Rex fallback (default 0.1).
        """
    @target_FDR_trex.setter
    def target_FDR_trex(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trex_screen_ctrl(self) -> ScreenTRexControlParameter:
        """
        Nested Screen-TRex control shared by both the screening and fallback-T-Rex paths. Set the screening method via trex_screen_ctrl.trex_method and the base algorithm (K, solver_type, ...) via trex_screen_ctrl.trex_ctrl. Its lloop_strategy is forced to STANDARD by the default ctor.
        """
    @trex_screen_ctrl.setter
    def trex_screen_ctrl(self, arg0: ScreenTRexControlParameter) -> None:
        ...
    @property
    def upper_bound_FDR(self) -> float:
        """
        Upper bound on the FDR search range.
        """
    @upper_bound_FDR.setter
    def upper_bound_FDR(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class BiobankScreenTRexResult:
    """
    Result for a single phenotype returned by screenPhenotype() or screenPhenotypes().
    """
    @property
    def estimated_FDR(self) -> float:
        """
        Estimated FDR of the selection.
        """
    @property
    def estimated_FDR_screen_bootstrap(self) -> float:
        """
        Estimated FDR from the bootstrap screening method.
        """
    @property
    def estimated_FDR_screen_ordinary(self) -> float:
        """
        Estimated FDR from the ordinary (non-bootstrap) screening method.
        """
    @property
    def method_used(self) -> str:
        """
        Method used: 'screening_ordinary', 'screening_bootstrap', or 'trex'.
        """
    @property
    def phenotype_index(self) -> int:
        """
        0-based column index of the screened phenotype.
        """
    @property
    def selected_indices(self) -> list[int]:
        """
        0-based indices of selected predictors.
        """
    @property
    def selected_indices_screen_bootstrap(self) -> list[int]:
        """
        Selected indices from the bootstrap screening method (0-based).
        """
    @property
    def selected_indices_screen_ordinary(self) -> list[int]:
        """
        Selected indices from the ordinary screening method (0-based).
        """
    @property
    def used_fallback_trex(self) -> bool:
        """
        True if the T-Rex fallback was used instead of screening.
        """
class DAMethod:
    """
    Data Augmentation methods.
    
    Members:
    
      AR1 : Autoregressive order 1.
    
      EQUI : Equicorrelated structure.
    
      BT : Block-Toeplitz structure.
    
      NN : Nearest-Neighbor approach.
    
      PRIOR_GROUPS : Prior group correlation structures.
    """
    AR1: typing.ClassVar[DAMethod]  # value = <DAMethod.AR1: 0>
    BT: typing.ClassVar[DAMethod]  # value = <DAMethod.BT: 2>
    EQUI: typing.ClassVar[DAMethod]  # value = <DAMethod.EQUI: 1>
    NN: typing.ClassVar[DAMethod]  # value = <DAMethod.NN: 3>
    PRIOR_GROUPS: typing.ClassVar[DAMethod]  # value = <DAMethod.PRIOR_GROUPS: 4>
    __members__: typing.ClassVar[dict[str, DAMethod]]  # value = {'AR1': <DAMethod.AR1: 0>, 'EQUI': <DAMethod.EQUI: 1>, 'BT': <DAMethod.BT: 2>, 'NN': <DAMethod.NN: 3>, 'PRIOR_GROUPS': <DAMethod.PRIOR_GROUPS: 4>}
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
class DASelectionResult(SelectionResult):
    """
    Extended SelectionResult holding specific DA outputs.
    """
    @property
    def cor_coef(self) -> float:
        """
        Auto-estimated or supplied correlation coefficient.
        """
    @property
    def fdp_hat_array_bt(self) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Per-rho-level FDP estimate matrices (BT method only).
        """
    @property
    def method(self) -> DAMethod:
        """
        The resulting DA method applied.
        """
    @property
    def phi_array_bt(self) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Per-rho-level deflated Phi matrices (BT method only).
        """
    @property
    def r_array_bt(self) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Per-rho-level R matrices (BT method only).
        """
    @property
    def rho_grid(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Grid values used for rho.
        """
    @property
    def rho_thresh(self) -> float:
        """
        The threshold applied for rho.
        """
class DummyDistribution:
    """
    Specification of the dummy variable distribution injected by T-Rex.
    """
    class Type:
        """
        Dummy distribution type enumeration.
        
        Members:
        
          Normal
        
          Uniform
        
          Rademacher
        
          StudentT
        
          Laplace
        
          Gumbel
        
          Triangle
        
          UniformSphere
        
          Mammen
        
          ConstrainedSparseRademacher
        
          Logistic
        """
        ConstrainedSparseRademacher: typing.ClassVar[DummyDistribution.Type]  # value = <Type.ConstrainedSparseRademacher: 9>
        Gumbel: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Gumbel: 5>
        Laplace: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Laplace: 4>
        Logistic: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Logistic: 10>
        Mammen: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Mammen: 8>
        Normal: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Normal: 0>
        Rademacher: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Rademacher: 2>
        StudentT: typing.ClassVar[DummyDistribution.Type]  # value = <Type.StudentT: 3>
        Triangle: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Triangle: 6>
        Uniform: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Uniform: 1>
        UniformSphere: typing.ClassVar[DummyDistribution.Type]  # value = <Type.UniformSphere: 7>
        __members__: typing.ClassVar[dict[str, DummyDistribution.Type]]  # value = {'Normal': <Type.Normal: 0>, 'Uniform': <Type.Uniform: 1>, 'Rademacher': <Type.Rademacher: 2>, 'StudentT': <Type.StudentT: 3>, 'Laplace': <Type.Laplace: 4>, 'Gumbel': <Type.Gumbel: 5>, 'Triangle': <Type.Triangle: 6>, 'UniformSphere': <Type.UniformSphere: 7>, 'Mammen': <Type.Mammen: 8>, 'ConstrainedSparseRademacher': <Type.ConstrainedSparseRademacher: 9>, 'Logistic': <Type.Logistic: 10>}
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
    ConstrainedSparseRademacher: typing.ClassVar[DummyDistribution.Type]  # value = <Type.ConstrainedSparseRademacher: 9>
    Gumbel: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Gumbel: 5>
    Laplace: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Laplace: 4>
    Logistic: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Logistic: 10>
    Mammen: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Mammen: 8>
    Normal: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Normal: 0>
    Rademacher: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Rademacher: 2>
    StudentT: typing.ClassVar[DummyDistribution.Type]  # value = <Type.StudentT: 3>
    Triangle: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Triangle: 6>
    Uniform: typing.ClassVar[DummyDistribution.Type]  # value = <Type.Uniform: 1>
    UniformSphere: typing.ClassVar[DummyDistribution.Type]  # value = <Type.UniformSphere: 7>
    @staticmethod
    def constrained_sparse_rademacher(s: typing.SupportsFloat | typing.SupportsIndex) -> DummyDistribution:
        """
        Create Constrained Sparse Rademacher distribution.
        """
    @staticmethod
    def gumbel(location: typing.SupportsFloat | typing.SupportsIndex = 0.0, scale: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> DummyDistribution:
        """
        Create Gumbel distribution.
        """
    @staticmethod
    def laplace(location: typing.SupportsFloat | typing.SupportsIndex = 0.0, scale: typing.SupportsFloat | typing.SupportsIndex = 0.7071067811865475) -> DummyDistribution:
        """
        Create Laplace distribution.
        """
    @staticmethod
    def logistic(location: typing.SupportsFloat | typing.SupportsIndex = 0.0, scale: typing.SupportsFloat | typing.SupportsIndex = 0.5513288954217921) -> DummyDistribution:
        """
        Create Logistic distribution.
        """
    @staticmethod
    def mammen(p1: typing.SupportsFloat | typing.SupportsIndex = 0.7236067977499789, p2: typing.SupportsFloat | typing.SupportsIndex = 0.27639320225002106) -> DummyDistribution:
        """
        Create Mammen two-point distribution.
        """
    @staticmethod
    def normal() -> DummyDistribution:
        """
        Create Normal distribution.
        """
    @staticmethod
    def rademacher() -> DummyDistribution:
        """
        Create Rademacher distribution.
        """
    @staticmethod
    def student_t(df: typing.SupportsFloat | typing.SupportsIndex = 5.0) -> DummyDistribution:
        """
        Create Student's t-distribution.
        """
    @staticmethod
    def triangle(a: typing.SupportsFloat | typing.SupportsIndex = -2.449489742783178, b: typing.SupportsFloat | typing.SupportsIndex = 0.0, c: typing.SupportsFloat | typing.SupportsIndex = 2.449489742783178) -> DummyDistribution:
        """
        Create Triangle distribution.
        """
    @staticmethod
    def uniform(a: typing.SupportsFloat | typing.SupportsIndex = 1.7320508075688772) -> DummyDistribution:
        """
        Create Uniform U(-a, a) distribution.
        """
    @staticmethod
    def uniform_sphere(dim: typing.SupportsInt | typing.SupportsIndex = 3) -> DummyDistribution:
        """
        Create Uniform distribution on the unit d-sphere.
        """
    def __init__(self) -> None:
        """
        Default constructor: Normal distribution.
        """
    def get_type(self) -> DummyDistribution.Type:
        """
        Get the distribution type.
        """
    @property
    def constrained_sparse_rademacher_s(self) -> float:
        ...
    @constrained_sparse_rademacher_s.setter
    def constrained_sparse_rademacher_s(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gumbel_location(self) -> float:
        ...
    @gumbel_location.setter
    def gumbel_location(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gumbel_scale(self) -> float:
        ...
    @gumbel_scale.setter
    def gumbel_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def laplace_location(self) -> float:
        ...
    @laplace_location.setter
    def laplace_location(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def laplace_scale(self) -> float:
        ...
    @laplace_scale.setter
    def laplace_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def logistic_location(self) -> float:
        ...
    @logistic_location.setter
    def logistic_location(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def logistic_scale(self) -> float:
        ...
    @logistic_scale.setter
    def logistic_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mammen_param_p1(self) -> float:
        ...
    @mammen_param_p1.setter
    def mammen_param_p1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mammen_param_p2(self) -> float:
        ...
    @mammen_param_p2.setter
    def mammen_param_p2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sphere_dim(self) -> int:
        ...
    @sphere_dim.setter
    def sphere_dim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def student_t_df(self) -> float:
        ...
    @student_t_df.setter
    def student_t_df(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def triangle_a(self) -> float:
        ...
    @triangle_a.setter
    def triangle_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def triangle_b(self) -> float:
        ...
    @triangle_b.setter
    def triangle_b(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def triangle_c(self) -> float:
        ...
    @triangle_c.setter
    def triangle_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def uniform_a(self) -> float:
        ...
    @uniform_a.setter
    def uniform_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ENSolverType:
    """
    Elastic-Net solver variant used when gvs_type == EN.
    
    Members:
    
      TENET : Gram-based Elastic Net (ridge absorbed via Cholesky update).
    
      TENET_AUG : Augmented-LASSO Elastic Net (row-augmented system).
    """
    TENET: typing.ClassVar[ENSolverType]  # value = <ENSolverType.TENET: 0>
    TENET_AUG: typing.ClassVar[ENSolverType]  # value = <ENSolverType.TENET_AUG: 1>
    __members__: typing.ClassVar[dict[str, ENSolverType]]  # value = {'TENET': <ENSolverType.TENET: 0>, 'TENET_AUG': <ENSolverType.TENET_AUG: 1>}
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
class GVSSelectionResult(SelectionResult):
    """
    SelectionResult extended with GVS-specific fields.
    """
    @property
    def group_labels(self) -> list[str]:
        """
        Unique group label integers.
        """
    @property
    def groups_vec(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Group assignment vector (0-based group index per predictor).
        """
    @property
    def gvs_type(self) -> GVSType:
        """
        GVS method used (EN or IEN).
        """
    @property
    def hc_method_used(self) -> str:
        """
        Hierarchical clustering linkage method used.
        """
    @property
    def lambda2_used(self) -> float:
        """
        L2 penalty lambda2 used (auto-selected via GCV if not specified).
        """
    @property
    def max_clusters(self) -> int:
        """
        Number of variable groups identified by hierarchical clustering.
        """
class GVSType:
    """
    Group Variable Selection method variants.
    
    Members:
    
      EN : Elastic-Net group variable selection (uses TENET solver).
    
      IEN : Informed Elastic-Net using group structure from clustering (uses TIENETAug solver).
    """
    EN: typing.ClassVar[GVSType]  # value = <GVSType.EN: 0>
    IEN: typing.ClassVar[GVSType]  # value = <GVSType.IEN: 1>
    __members__: typing.ClassVar[dict[str, GVSType]]  # value = {'EN': <GVSType.EN: 0>, 'IEN': <GVSType.IEN: 1>}
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
class LLoopStrategy:
    """
    L-loop strategy controlling how dummy variables are constructed in each experiment.
    
    Members:
    
      SKIPL : Stored, one-shot: fixed dummies = max_dummy_multiplier * p.

      STANDARD : Stored; fresh dummies at each L-loop iteration (conservative).

      HCONCAT : Stored; horizontally expand dummy matrices across L (faster, same result).

      PERMUTATION : Stored base dummy matrix + deterministic row permutations per experiment.

      PERMUTATION_ONDEMAND : Seed-derived base + row permutations per experiment; nothing stored.

      ONDEMAND : Seed-derived independent dummies per experiment; nothing stored.
    """
    HCONCAT: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.HCONCAT: 2>
    ONDEMAND: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.ONDEMAND: 5>
    PERMUTATION: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.PERMUTATION: 3>
    PERMUTATION_ONDEMAND: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.PERMUTATION_ONDEMAND: 4>
    SKIPL: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.SKIPL: 0>
    STANDARD: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.STANDARD: 1>
    __members__: typing.ClassVar[dict[str, LLoopStrategy]]  # value = {'SKIPL': <LLoopStrategy.SKIPL: 0>, 'STANDARD': <LLoopStrategy.STANDARD: 1>, 'HCONCAT': <LLoopStrategy.HCONCAT: 2>, 'PERMUTATION': <LLoopStrategy.PERMUTATION: 3>, 'PERMUTATION_ONDEMAND': <LLoopStrategy.PERMUTATION_ONDEMAND: 4>, 'ONDEMAND': <LLoopStrategy.ONDEMAND: 5>}
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
class LambdaSelectionMethod:
    """
    Rule for auto-selecting the ridge parameter lambda_2 when lambda_2 < 0.
    
    Members:
    
      CV_1SE_SVD : JacobiSVD ridge CV, 1-SE rule.
    
      CV_MIN_SVD : JacobiSVD ridge CV, minimum-CV-error rule.
    
      CV_1SE_CCD : Coordinate-descent ridge CV, glmnet-faithful 1-SE rule (default).
    
      CV_MIN_CCD : Coordinate-descent ridge CV, glmnet-faithful minimum rule.
    """
    CV_1SE_CCD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_1SE_CCD: 2>
    CV_1SE_SVD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_1SE_SVD: 0>
    CV_MIN_CCD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_MIN_CCD: 3>
    CV_MIN_SVD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_MIN_SVD: 1>
    __members__: typing.ClassVar[dict[str, LambdaSelectionMethod]]  # value = {'CV_1SE_SVD': <LambdaSelectionMethod.CV_1SE_SVD: 0>, 'CV_MIN_SVD': <LambdaSelectionMethod.CV_MIN_SVD: 1>, 'CV_1SE_CCD': <LambdaSelectionMethod.CV_1SE_CCD: 2>, 'CV_MIN_CCD': <LambdaSelectionMethod.CV_MIN_CCD: 3>}
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
class SPCAMode:
    """
    Members:
    
      ActiveSet : Ridge regression on the T-Rex active set to compute loadings.
    
      Thresholded : Keep and normalize the top |A| ordinary PCA loadings.
    """
    ActiveSet: typing.ClassVar[SPCAMode]  # value = <SPCAMode.ActiveSet: 0>
    Thresholded: typing.ClassVar[SPCAMode]  # value = <SPCAMode.Thresholded: 1>
    __members__: typing.ClassVar[dict[str, SPCAMode]]  # value = {'ActiveSet': <SPCAMode.ActiveSet: 0>, 'Thresholded': <SPCAMode.Thresholded: 1>}
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
class ScalingMode:
    """
    Column scaling convention applied to X (and dummies) after centering.
    
    Members:
    
      L2 : Divide each centered column by its L2 norm (unit L2 norm).
    
      ZSCORE : Divide each centered column by its sample SD (n-1); glmnet standardize=TRUE convention.
    """
    L2: typing.ClassVar[ScalingMode]  # value = <ScalingMode.L2: 0>
    ZSCORE: typing.ClassVar[ScalingMode]  # value = <ScalingMode.ZSCORE: 1>
    __members__: typing.ClassVar[dict[str, ScalingMode]]  # value = {'L2': <ScalingMode.L2: 0>, 'ZSCORE': <ScalingMode.ZSCORE: 1>}
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
class ScreenTRexControlParameter:
    """
    Control parameters for TRexScreeningSelector.
    """
    def __init__(self) -> None:
        ...
    @property
    def R_boot(self) -> int:
        """
        Number of bootstrap replicates (default 1000).
        """
    @R_boot.setter
    def R_boot(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ci_grid_step(self) -> float:
        """
        Step size for the confidence-interval grid.
        """
    @ci_grid_step.setter
    def ci_grid_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cor_coef(self) -> float:
        """
        Fixed correlation coefficient for AR1 or equicorrelated structure. Estimated automatically if <= -1. Only applies for TREX_DA_AR1 and TREX_DA_EQUI.
        """
    @cor_coef.setter
    def cor_coef(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_blocks(self) -> int:
        """
        Number of equicorrelated blocks. Only applies for TREX_DA_BLOCK_EQUI.
        """
    @n_blocks.setter
    def n_blocks(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rho_thr_DA(self) -> float:
        """
        Correlation threshold passed to the inner DA step. Only applies for TREX_DA_AR1 and TREX_DA_EQUI.
        """
    @rho_thr_DA.setter
    def rho_thr_DA(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trex_ctrl(self) -> TRexControlParameter:
        """
        Nested base T-Rex algorithmic control parameters (overridden by the separate trex_control argument to TRexScreeningSelector).
        """
    @trex_ctrl.setter
    def trex_ctrl(self, arg0: TRexControlParameter) -> None:
        ...
    @property
    def trex_method(self) -> ScreenTRexMethod:
        """
        Screening method variant to use (see ScreenTRexMethod).
        """
    @trex_method.setter
    def trex_method(self, arg0: ScreenTRexMethod) -> None:
        ...
    @property
    def use_bootstrap_CI(self) -> bool:
        """
        Use bootstrap confidence intervals for FDR estimation (default false).
        """
    @use_bootstrap_CI.setter
    def use_bootstrap_CI(self, arg0: bool) -> None:
        ...
class ScreenTRexMethod:
    """
    Screening TRex baseline variants.
    
    Members:
    
      TREX : Standard T-Rex screening without data augmentation.
    
      TREX_DA_AR1 : T-Rex screening with AR(1) data augmentation.
    
      TREX_DA_EQUI : T-Rex screening with equicorrelated data augmentation.
    
      TREX_DA_BLOCK_EQUI : T-Rex screening with block-equicorrelated data augmentation.
    """
    TREX: typing.ClassVar[ScreenTRexMethod]  # value = <ScreenTRexMethod.TREX: 0>
    TREX_DA_AR1: typing.ClassVar[ScreenTRexMethod]  # value = <ScreenTRexMethod.TREX_DA_AR1: 1>
    TREX_DA_BLOCK_EQUI: typing.ClassVar[ScreenTRexMethod]  # value = <ScreenTRexMethod.TREX_DA_BLOCK_EQUI: 3>
    TREX_DA_EQUI: typing.ClassVar[ScreenTRexMethod]  # value = <ScreenTRexMethod.TREX_DA_EQUI: 2>
    __members__: typing.ClassVar[dict[str, ScreenTRexMethod]]  # value = {'TREX': <ScreenTRexMethod.TREX: 0>, 'TREX_DA_AR1': <ScreenTRexMethod.TREX_DA_AR1: 1>, 'TREX_DA_EQUI': <ScreenTRexMethod.TREX_DA_EQUI: 2>, 'TREX_DA_BLOCK_EQUI': <ScreenTRexMethod.TREX_DA_BLOCK_EQUI: 3>}
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
class ScreenTRexSelectionResult(SelectionResult):
    """
    SelectionResult extended with screening-specific fields.
    """
    @property
    def beta_mat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Coefficient matrix averaged over K experiments (p x T_stop).
        """
    @property
    def confidence_level(self) -> float:
        """
        Bootstrap confidence level achieved.
        """
    @property
    def dummy_betas(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Non-zero dummy coefficients collected during screening.
        """
    @property
    def estimated_FDR(self) -> float:
        """
        Estimated false discovery rate.
        """
    @property
    def estimated_correlation(self) -> float:
        """
        Estimated correlation used by DA-based screening variants.
        """
    @property
    def used_bootstrap(self) -> bool:
        """
        Whether bootstrap CI was used for FDR estimation.
        """
class SelectionResult:
    """
    Return value of TRexSelector.select(): selected variables and diagnostic matrices.
    """
    @property
    def T_stop(self) -> int:
        """
        T_stop: maximum number of LARS steps run.
        """
    @property
    def dummy_factor_L(self) -> int:
        """
        Dummy multiplier L used.
        """
    @property
    def fdp_hat_mat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        FDP hat matrix (p x |voting_grid|).
        """
    @property
    def num_dummies(self) -> int:
        """
        Total number of dummy variables used.
        """
    @property
    def phi_mat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Phi matrix: relative occurrence counts per predictor.
        """
    @property
    def phi_prime(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Smoothed selection probability vector phi'.
        """
    @property
    def r_mat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        R matrix of empirical tail probabilities (p x T_stop).
        """
    @property
    def selected_var(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Selection indicator vector (length p; 1 = selected, 0 = not selected).
        """
    @property
    def v_thresh(self) -> float:
        """
        Voting threshold v applied to control the FDR.
        """
    @property
    def voting_grid(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Voting grid: per-candidate vote totals across K experiments.
        """
class SolverHyperparameters:
    """
    Hyperparameters for the selected T-Rex solver.
    """
    def __init__(self) -> None:
        ...
    @property
    def lambda2(self) -> float:
        """
        L2 penalty for ENET-type solvers.
        """
    @lambda2.setter
    def lambda2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ncgmp_variant(self) -> int:
        """
        NCGMP variant: 0 = LineSearch, 1 = FullyCorrective.
        """
    @ncgmp_variant.setter
    def ncgmp_variant(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rho_afs(self) -> float:
        """
        Shrinkage step size for AFS solver in (0, 1].
        """
    @rho_afs.setter
    def rho_afs(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tol(self) -> float:
        """
        Numerical tolerance for solver steps.
        """
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SolverTypeForTRex:
    """
    Solver algorithm variants available for T-Rex.
    
    Members:
    
      TLARS : Terminating LARS.
    
      TLASSO : Terminating LASSO.
    
      TENET : Terminating Elastic Net.
    
      TSTEPWISE : Terminating Stepwise.
    
      TSTAGEWISE : Terminating Stagewise.
    
      TOMP : Terminating OMP.
    
      TGP : Terminating Gradient Pursuit.
    
      TACGP : Terminating Approximate Conjugate Gradient Pursuit.
    
      TMP : Terminating Matching Pursuit.
    
      TNCGMP : Terminating Norm-Corrected Generalized Matching Pursuit.
    
      TOOLS : Terminating Orthogonal Least Squares.
    
      TAFS : Terminating AFS.
    
      TENET_AUG : Terminating Elastic Net via augmented LASSO (GVS).
    
      TIENET_AUG : Terminating Informed Elastic Net via augmented LASSO (GVS, gvs_type=IEN).
    """
    TACGP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TACGP: 9>
    TAFS: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TAFS: 13>
    TENET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TENET: 2>
    TENET_AUG: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TENET_AUG: 3>
    TGP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TGP: 8>
    TIENET_AUG: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TIENET_AUG: 4>
    TLARS: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TLARS: 0>
    TLASSO: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TLASSO: 1>
    TMP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TMP: 10>
    TNCGMP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TNCGMP: 11>
    TOMP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TOMP: 7>
    TOOLS: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TOOLS: 12>
    TSTAGEWISE: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TSTAGEWISE: 6>
    TSTEPWISE: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TSTEPWISE: 5>
    __members__: typing.ClassVar[dict[str, SolverTypeForTRex]]  # value = {'TLARS': <SolverTypeForTRex.TLARS: 0>, 'TLASSO': <SolverTypeForTRex.TLASSO: 1>, 'TENET': <SolverTypeForTRex.TENET: 2>, 'TSTEPWISE': <SolverTypeForTRex.TSTEPWISE: 5>, 'TSTAGEWISE': <SolverTypeForTRex.TSTAGEWISE: 6>, 'TOMP': <SolverTypeForTRex.TOMP: 7>, 'TGP': <SolverTypeForTRex.TGP: 8>, 'TACGP': <SolverTypeForTRex.TACGP: 9>, 'TMP': <SolverTypeForTRex.TMP: 10>, 'TNCGMP': <SolverTypeForTRex.TNCGMP: 11>, 'TOOLS': <SolverTypeForTRex.TOOLS: 12>, 'TAFS': <SolverTypeForTRex.TAFS: 13>, 'TENET_AUG': <SolverTypeForTRex.TENET_AUG: 3>, 'TIENET_AUG': <SolverTypeForTRex.TIENET_AUG: 4>}
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
class TRexBiobankScreeningSelector:
    """
    Biobank-scale T-Rex selector for FDR-controlled variable selection across one or many phenotypes.
    """
    @typing.overload
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], bio_ctrl: BiobankScreenTRexControl = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = False) -> None:
        """
        Construct for a single phenotype (1-D y). X and y are accessed zero-copy.
        """
    @typing.overload
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], Y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], bio_ctrl: BiobankScreenTRexControl = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = False) -> None:
        """
        Construct for multiple phenotypes (2-D Y). X and Y are accessed zero-copy.
        """
    def screenPhenotype(self) -> BiobankScreenTRexResult:
        """
        Run screening for the single loaded phenotype.
        """
    def screenPhenotypes(self) -> list[BiobankScreenTRexResult]:
        """
        Run screening for all loaded phenotypes.
        """
class TRexControlParameter:
    """
    Control parameters for the core T-Rex selection algorithm.
    """
    def __init__(self) -> None:
        ...
    @property
    def K(self) -> int:
        """
        Number of random simulation experiments to run over variable inclusions.
        """
    @K.setter
    def K(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def dummy_distribution(self) -> DummyDistribution:
        """
        Distribution used to generate dummy variables.
        """
    @dummy_distribution.setter
    def dummy_distribution(self, arg0: DummyDistribution) -> None:
        ...
    @property
    def lloop_strategy(self) -> LLoopStrategy:
        """
        L-loop strategy for constructing dummy variables.
        """
    @lloop_strategy.setter
    def lloop_strategy(self, arg0: LLoopStrategy) -> None:
        ...
    @property
    def max_dummy_multiplier(self) -> int:
        """
        Maximum dummy multiplier L (default 10).
        """
    @max_dummy_multiplier.setter
    def max_dummy_multiplier(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_inner_threads(self) -> int:
        """
        Thread count for inner solver operations.
        """
    @max_inner_threads.setter
    def max_inner_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_outer_threads(self) -> int:
        """
        Thread count for K repetitions (outer parallelism).
        """
    @max_outer_threads.setter
    def max_outer_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def opt_threshold(self) -> float:
        """
        Voting threshold parameter (default 0.75).
        """
    @opt_threshold.setter
    def opt_threshold(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def parallel_rnd_experiments(self) -> bool:
        """
        Run K random experiments in parallel using OpenMP.
        """
    @parallel_rnd_experiments.setter
    def parallel_rnd_experiments(self, arg0: bool) -> None:
        ...
    @property
    def scaling_mode(self) -> ScalingMode:
        """
        Column scaling convention applied to X after centering (L2 or ZSCORE).
        """
    @scaling_mode.setter
    def scaling_mode(self, arg0: ScalingMode) -> None:
        ...
    @property
    def solver_params(self) -> SolverHyperparameters:
        """
        Hyperparameters for the selected solver.
        """
    @solver_params.setter
    def solver_params(self, arg0: SolverHyperparameters) -> None:
        ...
    @property
    def solver_type(self) -> SolverTypeForTRex:
        """
        Solver algorithm to use for T-Rex.
        """
    @solver_type.setter
    def solver_type(self, arg0: SolverTypeForTRex) -> None:
        ...
    @property
    def tloop_max_stagnant_steps(self) -> int:
        """
        Maximum consecutive stagnant T-loop steps before early stopping.
        """
    @tloop_max_stagnant_steps.setter
    def tloop_max_stagnant_steps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tloop_stagnation_stop(self) -> bool | None:
        """
        T-loop early stopping on support-set stagnation. None (default) resolves by solver family: disabled for equiangular LARS-path solvers (TLARS/TLASSO/TENET/TENET_AUG, matching the R reference), enabled for greedy solvers (guards their noise trap). Set True/False to override.
        """
    @tloop_stagnation_stop.setter
    def tloop_stagnation_stop(self, arg0: bool | None) -> None:
        ...
    @property
    def use_max_T_stop(self) -> bool:
        """
        If true, always run solvers to T_stop regardless of early stopping.
        """
    @use_max_T_stop.setter
    def use_max_T_stop(self, arg0: bool) -> None:
        ...
    @property
    def use_memory_mapping(self) -> bool:
        """
        Use out-of-core memory-mapped dummy matrix.
        """
    @use_memory_mapping.setter
    def use_memory_mapping(self, arg0: bool) -> None:
        ...
class TRexDAControlParameter:
    """
    Control parameters for TRexDASelector.
    """
    def __init__(self) -> None:
        ...
    @property
    def bt_selection_mode(self) -> BTSelectionMode:
        """
        Cell-selection policy for the BT method (FeasibleOnly or RFaithful).
        """
    @bt_selection_mode.setter
    def bt_selection_mode(self, arg0: BTSelectionMode) -> None:
        ...
    @property
    def cor_coef(self) -> float:
        """
        Correlation coefficient parameter.
        """
    @cor_coef.setter
    def cor_coef(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hc_grid_length(self) -> int:
        """
        Grid length for hierarchical clustering.
        """
    @hc_grid_length.setter
    def hc_grid_length(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def hc_linkage(self) -> ...:
        """
        Hierarchical clustering linkage method.
        """
    @hc_linkage.setter
    def hc_linkage(self, arg0: ...) -> None:
        ...
    @property
    def method(self) -> DAMethod:
        """
        The data augmentation method to use.
        """
    @method.setter
    def method(self, arg0: DAMethod) -> None:
        ...
    @property
    def prior_groups(self) -> list[list[int]]:
        """
        Prior groupings provided to the selector.
        """
    @prior_groups.setter
    def prior_groups(self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]]) -> None:
        ...
    @property
    def rho_grid_labels(self) -> list[float]:
        """
        Labels for the correlation grid.
        """
    @rho_grid_labels.setter
    def rho_grid_labels(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def rho_thr_DA(self) -> float:
        """
        Correlation threshold for DA.
        """
    @rho_thr_DA.setter
    def rho_thr_DA(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def trex_ctrl(self) -> TRexControlParameter:
        """
        Nested base T-Rex algorithmic control parameters (overridden by the separate trex_control argument to TRexDASelector).
        """
    @trex_ctrl.setter
    def trex_ctrl(self, arg0: TRexControlParameter) -> None:
        ...
class TRexDASelector(TRexSelector):
    """
    TRex Selector with Data Augmentation support.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tFDR: typing.SupportsFloat | typing.SupportsIndex = 0.1, da_control: TRexDAControlParameter = ..., trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = True) -> None:
        """
        Construct the DA selector. X and y are accessed zero-copy via Eigen::Map.
        """
    def getDAResult(self) -> DASelectionResult:
        """
        Fetch the detailed DA selection result.
        """
class TRexGVSControlParameter:
    """
    Control parameters for TRexGVSSelector.
    """
    def __init__(self) -> None:
        ...
    @property
    def corr_max(self) -> float:
        """
        Maximum pairwise correlation for automatic cluster formation.
        """
    @corr_max.setter
    def corr_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_folds(self) -> int:
        """
        Number of folds for cross-validated lambda_2 selection.
        """
    @cv_n_folds.setter
    def cv_n_folds(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_lambda(self) -> int:
        """
        Number of points on the lambda grid searched during CV.
        """
    @cv_n_lambda.setter
    def cv_n_lambda(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_seed(self) -> int:
        """
        Seed for the CV fold-permutation RNG (-1 = derive from the T-Rex seed).
        """
    @cv_seed.setter
    def cv_seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def en_solver(self) -> ENSolverType:
        """
        EN solver variant when gvs_type == EN (TENET or TENET_AUG).
        """
    @en_solver.setter
    def en_solver(self, arg0: ENSolverType) -> None:
        ...
    @property
    def group_labels(self) -> list[str]:
        """
        Optional human-readable cluster names (size must equal number of clusters).
        """
    @group_labels.setter
    def group_labels(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def gvs_type(self) -> GVSType:
        """
        GVS method variant (EN = Elastic Net, IEN = Informed Elastic Net).
        """
    @gvs_type.setter
    def gvs_type(self, arg0: GVSType) -> None:
        ...
    @property
    def hc_linkage(self) -> ...:
        """
        Hierarchical clustering linkage method.
        """
    @hc_linkage.setter
    def hc_linkage(self, arg0: ...) -> None:
        ...
    @property
    def lambda2_method(self) -> LambdaSelectionMethod:
        """
        Rule for auto-selecting lambda_2 when lambda_2 < 0.
        """
    @lambda2_method.setter
    def lambda2_method(self, arg0: LambdaSelectionMethod) -> None:
        ...
    @property
    def lambda_2(self) -> float:
        """
        Ridge L2 penalty in LARS units (< 0 = auto-select, 0 = pure TLASSO, > 0 = fixed).
        """
    @lambda_2.setter
    def lambda_2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def prior_groups(self) -> list[int]:
        """
        Manually specified group assignments (0-based; empty = auto-cluster).
        """
    @prior_groups.setter
    def prior_groups(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def tenet_aug_use_lars(self) -> bool:
        """
        For TENET_AUG: use a pure-LARS inner solver (R-parity diagnostic).
        """
    @tenet_aug_use_lars.setter
    def tenet_aug_use_lars(self, arg0: bool) -> None:
        ...
    @property
    def trex_ctrl(self) -> TRexControlParameter:
        """
        Nested base T-Rex algorithmic control parameters (used directly by TRexSPCASelector; overridden by the separate trex_control argument in TRexGVSSelector).
        """
    @trex_ctrl.setter
    def trex_ctrl(self, arg0: TRexControlParameter) -> None:
        ...
class TRexGVSSelector(TRexSelector):
    """
    TRex Selector with Group Variable Selection support.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tFDR: typing.SupportsFloat | typing.SupportsIndex = 0.1, gvs_control: TRexGVSControlParameter = ..., trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = True) -> None:
        """
        Construct the GVS selector. X and y are accessed zero-copy via Eigen::Map.
        """
    def getGVSResult(self) -> GVSSelectionResult:
        """
        Return the GVSSelectionResult after calling select().
        """
class TRexSPCAControlParameter:
    """
    Algorithmic control parameters for the T-Rex Sparse PCA selector.
    """
    def __init__(self) -> None:
        ...
    @property
    def en_solver(self) -> ENSolverType:
        """
        EN solver variant for the per-PC GVS(EN) sub-selector (TENET or TENET_AUG; default TENET_AUG).
        """
    @en_solver.setter
    def en_solver(self, arg0: ENSolverType) -> None:
        ...
    @property
    def gvs_ctrl(self) -> TRexGVSControlParameter:
        """
        TRexGVSControlParameter forwarded to each per-PC GVS run. Set gvs_ctrl.gvs_type = GVSType.EN or IEN. Set gvs_ctrl.lambda_2 > 0 to bypass auto-determination. The base T-Rex parameters are configured via gvs_ctrl.trex_ctrl.
        """
    @gvs_ctrl.setter
    def gvs_ctrl(self, arg0: TRexGVSControlParameter) -> None:
        ...
    @property
    def lambda2_ridge_loadings(self) -> float:
        """
        Ridge penalty for ActiveSet loading assembly (default: 1e-6).
        """
    @lambda2_ridge_loadings.setter
    def lambda2_ridge_loadings(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mode(self) -> SPCAMode:
        """
        Sparse loading strategy (default: ActiveSet).
        """
    @mode.setter
    def mode(self, arg0: SPCAMode) -> None:
        ...
class TRexSPCAResult:
    """
    Container for the T-Rex Sparse PCA output.
    """
    def __init__(self) -> None:
        ...
    @property
    def V(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Sparse loading matrix (p × M).
        """
    @V.setter
    def V(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def Z(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Sparse PC score matrix (n × M).
        """
    @Z.setter
    def Z(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def active_sets(self) -> list[list[int]]:
        """
        Support indices per PC (list of M index vectors, 0-based).
        """
    @active_sets.setter
    def active_sets(self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]]) -> None:
        ...
    @property
    def adjusted_ev(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Marginal adjusted explained variance per component (M-vector).
        """
    @adjusted_ev.setter
    def adjusted_ev(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    @property
    def cumulative_ev(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Cumulative percentage of explained variance (M-vector).
        """
    @cumulative_ev.setter
    def cumulative_ev(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
class TRexSPCASelector:
    """
    T-Rex Sparse PCA selector — instance wrapper (zero-copy Eigen::Map).
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], M: typing.SupportsInt | typing.SupportsIndex, tFDR: typing.SupportsFloat | typing.SupportsIndex, ctrl: TRexSPCAControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = False) -> None:
        """
        Construct the SPCA selector. X is accessed zero-copy via Eigen::Map.
        """
    def getResult(self) -> TRexSPCAResult:
        """
        Return the TRexSPCAResult after calling select().
        """
    def select(self) -> None:
        """
        Run T-Rex Sparse PCA selection.
        """
class TRexScreeningSelector(TRexSelector):
    """
    T-Rex Selector with fast variable screening via dummy comparison.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], screen_control: ScreenTRexControlParameter = ..., trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = True) -> None:
        """
        Construct the screening selector. X and y are accessed zero-copy via Eigen::Map.
        """
    def getScreenResult(self) -> ScreenTRexSelectionResult:
        """
        Return the ScreenTRexSelectionResult after calling select().
        """
class TRexSelector:
    """
    T-Rex Selector: FDR-controlled variable selection via dummy comparison.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tFDR: typing.SupportsFloat | typing.SupportsIndex = 0.1, trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = True) -> None:
        """
        Construct the selector. X and y are accessed zero-copy via Eigen::Map.
        """
    def getDummyMultiplierL(self) -> int:
        """
        Return L: the dummy multiplier used.
        """
    def getFDPHatMat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return the FDP hat matrix.
        """
    def getK(self) -> int:
        """
        Return K: the number of random experiments.
        """
    def getMaxNumDummies(self) -> int:
        """
        Return the maximum number of dummy variables per L level.
        """
    def getMaxTStop(self) -> bool:
        """
        Return whether use_max_T_stop is active.
        """
    def getNumSelected(self) -> int:
        """
        Return the number of selected predictors.
        """
    def getPhiMat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return the Phi occurrence-count matrix.
        """
    def getPhiPrime(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Return the smoothed selection probability vector phi'.
        """
    def getRMat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return the R matrix of empirical tail probabilities.
        """
    def getSelectedIndices(self) -> list[int]:
        """
        Return 0-based indices of selected predictors.
        """
    def getSelectedVar(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Return the binary selection indicator vector (1 = selected).
        """
    def getStagnationCheck(self) -> bool:
        """
        Return whether T-loop stagnation stopping is active.
        """
    def getTFDR(self) -> float:
        """
        Return the target FDR.
        """
    def getTStop(self) -> int:
        """
        Return T_stop: the LARS step count used.
        """
    def getVotingGrid(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Return the voting grid.
        """
    def getVotingThreshold(self) -> float:
        """
        Return the voting threshold v.
        """
    def getXL2Norms(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Return the column-wise L2 norms of X after centering.
        """
    def getXMeans(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Return the column means of X.
        """
    def getYMean(self) -> float:
        """
        Return the mean of y.
        """
    def select(self) -> SelectionResult:
        """
        Run the T-Rex algorithm and return a SelectionResult.
        """
AR1: DAMethod  # value = <DAMethod.AR1: 0>
ActiveSet: SPCAMode  # value = <SPCAMode.ActiveSet: 0>
BT: DAMethod  # value = <DAMethod.BT: 2>
CV_1SE_CCD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_1SE_CCD: 2>
CV_1SE_SVD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_1SE_SVD: 0>
CV_MIN_CCD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_MIN_CCD: 3>
CV_MIN_SVD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_MIN_SVD: 1>
EN: GVSType  # value = <GVSType.EN: 0>
EQUI: DAMethod  # value = <DAMethod.EQUI: 1>
FeasibleOnly: BTSelectionMode  # value = <BTSelectionMode.FeasibleOnly: 0>
HCONCAT: LLoopStrategy  # value = <LLoopStrategy.HCONCAT: 2>
IEN: GVSType  # value = <GVSType.IEN: 1>
L2: ScalingMode  # value = <ScalingMode.L2: 0>
NN: DAMethod  # value = <DAMethod.NN: 3>
ONDEMAND: LLoopStrategy  # value = <LLoopStrategy.ONDEMAND: 5>
PERMUTATION: LLoopStrategy  # value = <LLoopStrategy.PERMUTATION: 3>
PERMUTATION_ONDEMAND: LLoopStrategy  # value = <LLoopStrategy.PERMUTATION_ONDEMAND: 4>
PRIOR_GROUPS: DAMethod  # value = <DAMethod.PRIOR_GROUPS: 4>
RFaithful: BTSelectionMode  # value = <BTSelectionMode.RFaithful: 1>
SKIPL: LLoopStrategy  # value = <LLoopStrategy.SKIPL: 0>
STANDARD: LLoopStrategy  # value = <LLoopStrategy.STANDARD: 1>
TACGP: SolverTypeForTRex  # value = <SolverTypeForTRex.TACGP: 9>
TAFS: SolverTypeForTRex  # value = <SolverTypeForTRex.TAFS: 13>
TENET: ENSolverType  # value = <ENSolverType.TENET: 0>
TENET_AUG: ENSolverType  # value = <ENSolverType.TENET_AUG: 1>
TGP: SolverTypeForTRex  # value = <SolverTypeForTRex.TGP: 8>
TIENET_AUG: SolverTypeForTRex  # value = <SolverTypeForTRex.TIENET_AUG: 4>
TLARS: SolverTypeForTRex  # value = <SolverTypeForTRex.TLARS: 0>
TLASSO: SolverTypeForTRex  # value = <SolverTypeForTRex.TLASSO: 1>
TMP: SolverTypeForTRex  # value = <SolverTypeForTRex.TMP: 10>
TNCGMP: SolverTypeForTRex  # value = <SolverTypeForTRex.TNCGMP: 11>
TOMP: SolverTypeForTRex  # value = <SolverTypeForTRex.TOMP: 7>
TOOLS: SolverTypeForTRex  # value = <SolverTypeForTRex.TOOLS: 12>
TREX: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX: 0>
TREX_DA_AR1: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX_DA_AR1: 1>
TREX_DA_BLOCK_EQUI: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX_DA_BLOCK_EQUI: 3>
TREX_DA_EQUI: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX_DA_EQUI: 2>
TSTAGEWISE: SolverTypeForTRex  # value = <SolverTypeForTRex.TSTAGEWISE: 6>
TSTEPWISE: SolverTypeForTRex  # value = <SolverTypeForTRex.TSTEPWISE: 5>
Thresholded: SPCAMode  # value = <SPCAMode.Thresholded: 1>
ZSCORE: ScalingMode  # value = <ScalingMode.ZSCORE: 1>
