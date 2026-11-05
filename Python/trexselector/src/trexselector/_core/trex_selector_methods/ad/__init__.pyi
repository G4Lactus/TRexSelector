"""
Augmented-dummy (AD) T-Rex selectors: classical, DA, GVS, screening, SPCA, Tikhonov
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AR1', 'ActiveSet', 'BT', 'BTSelectionMode', 'BiobankScreenTRexControl', 'BiobankScreenTRexResult', 'CALIBRATED', 'CV_1SE_CCD', 'CV_1SE_IEN_CCD', 'CV_1SE_SVD', 'CV_1SE_TIK_SVD', 'CV_MIN_CCD', 'CV_MIN_IEN_CCD', 'CV_MIN_SVD', 'CV_MIN_TIK_SVD', 'DAMethod', 'DASelectionResult', 'DummyDistribution', 'EN', 'ENSolverType', 'EQUI', 'FeasibleOnly', 'GROUP_CENTERING', 'GROUP_MAJORITY', 'GVSSelectionResult', 'GVSType', 'HCONCAT', 'IEN', 'L2', 'LLoopStrategy', 'Lambda2CVDiagnostics', 'Lambda2PickInfo', 'Lambda2PilotResult', 'Lambda2PowerRuleResult', 'Lambda2Probe', 'LambdaSelectionMethod', 'MAJORITY', 'NN', 'NONE', 'PERMUTATION', 'PERMUTATION_SEEDED', 'PILOT_KNEE', 'PRIOR_GROUPS', 'PilotStatistic', 'RFaithful', 'SEEDED', 'SKIPL', 'SPARSE_GENERAL', 'SPCAMode', 'STANDARD', 'ScalingMode', 'ScreenTRexControlParameter', 'ScreenTRexMethod', 'ScreenTRexSelectionResult', 'SelectionResult', 'SolverHyperparameters', 'SolverTypeForTRex', 'TACGP', 'TAFS', 'TCDENET', 'TCDIENET', 'TCDLASSO', 'TCDTIKNET', 'TENET', 'TENET_AUG', 'TGP', 'TIENET', 'TIENET_AUG', 'TIKNET', 'TLARS', 'TLASSO', 'TMP', 'TNCGMP', 'TOMP', 'TOOLS', 'TREX', 'TREX_DA_AR1', 'TREX_DA_BLOCK_EQUI', 'TREX_DA_EQUI', 'TRexBiobankScreeningSelector', 'TRexControlParameter', 'TRexDAControlParameter', 'TRexDASelector', 'TRexGVSControlParameter', 'TRexGVSSelector', 'TRexSPCAControlParameter', 'TRexSPCAResult', 'TRexSPCASelector', 'TRexScreeningSelector', 'TRexSelector', 'TRexTikhonovControlParameter', 'TRexTikhonovSelector', 'TSTAGEWISE', 'TSTEPWISE', 'Thresholded', 'TikLambda2Method', 'TikPenaltyStructure', 'ZSCORE', 'build_group_centering_k', 'gamma_to_k', 'normalize_design', 'tikhonov_lambda2_power_rule']
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
    @typing.overload
    def __eq__(self, other: BTSelectionMode) -> bool:
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
    def __ne__(self, other: BTSelectionMode) -> bool:
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
        Nested Screen-TRex control shared by both the screening and fallback-T-Rex paths. Set the screening method via trex_screen_ctrl.trex_method and the base algorithm (K, solver_type, ...) via trex_screen_ctrl.trex_ctrl. Its lloop_strategy defaults to STANDARD; accepted values are STANDARD, SEEDED, PERMUTATION, and PERMUTATION_SEEDED (validated at construction).
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
    @typing.overload
    def __eq__(self, other: DAMethod) -> bool:
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
    def __ne__(self, other: DAMethod) -> bool:
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
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    E. Mammen, "Bootstrap and Wild Bootstrap for High Dimensional Linear Models," Ann.
      Statist., vol. 21, no. 1, pp. 255-285, 1993, doi: 10.1214/aos/1176349025.
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
        @typing.overload
        def __eq__(self, other: DummyDistribution.Type) -> bool:
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
        def __ne__(self, other: DummyDistribution.Type) -> bool:
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
    
      TCDENET : Coordinate-descent Elastic Net (exact fixed-lambda1 minimizers per dummy crossing).
    """
    TCDENET: typing.ClassVar[ENSolverType]  # value = <ENSolverType.TCDENET: 2>
    TENET: typing.ClassVar[ENSolverType]  # value = <ENSolverType.TENET: 0>
    TENET_AUG: typing.ClassVar[ENSolverType]  # value = <ENSolverType.TENET_AUG: 1>
    __members__: typing.ClassVar[dict[str, ENSolverType]]  # value = {'TENET': <ENSolverType.TENET: 0>, 'TENET_AUG': <ENSolverType.TENET_AUG: 1>, 'TCDENET': <ENSolverType.TCDENET: 2>}
    @typing.overload
    def __eq__(self, other: ENSolverType) -> bool:
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
    def __ne__(self, other: ENSolverType) -> bool:
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
    def lambda2_cv(self) -> Lambda2CVDiagnostics:
        """
        Health record of the lambda_2 tuner (Lambda2CVDiagnostics).
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
    
      IEN : Informed Elastic-Net using group structure from clustering (solver via trex_control.solver_type: TIENET default, TIENET_AUG, TCDIENET).
    """
    EN: typing.ClassVar[GVSType]  # value = <GVSType.EN: 0>
    IEN: typing.ClassVar[GVSType]  # value = <GVSType.IEN: 1>
    __members__: typing.ClassVar[dict[str, GVSType]]  # value = {'EN': <GVSType.EN: 0>, 'IEN': <GVSType.IEN: 1>}
    @typing.overload
    def __eq__(self, other: GVSType) -> bool:
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
    def __ne__(self, other: GVSType) -> bool:
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
class LLoopStrategy:
    """
    L-loop strategy controlling how dummy variables are constructed in each experiment.
    
    Members:
    
      SKIPL : Stored, one-shot: fixed dummies = max_dummy_multiplier * p.
    
      STANDARD : Stored; fresh dummies at each L-loop iteration (conservative).
    
      HCONCAT : Stored; horizontally expand dummy matrices across L (faster, same result).
    
      PERMUTATION : Stored base dummy matrix + deterministic row permutations per experiment.
    
      PERMUTATION_SEEDED : Seed-derived base + row permutations per experiment; nothing stored.
    
      SEEDED : The storage-free twin of STANDARD: fresh dummies at every L step, re-derived from the seed per experiment and bit-identical to STANDARD for the same seed; nothing stored.
    """
    HCONCAT: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.HCONCAT: 2>
    PERMUTATION: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.PERMUTATION: 3>
    PERMUTATION_SEEDED: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.PERMUTATION_SEEDED: 4>
    SEEDED: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.SEEDED: 5>
    SKIPL: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.SKIPL: 0>
    STANDARD: typing.ClassVar[LLoopStrategy]  # value = <LLoopStrategy.STANDARD: 1>
    __members__: typing.ClassVar[dict[str, LLoopStrategy]]  # value = {'SKIPL': <LLoopStrategy.SKIPL: 0>, 'STANDARD': <LLoopStrategy.STANDARD: 1>, 'HCONCAT': <LLoopStrategy.HCONCAT: 2>, 'PERMUTATION': <LLoopStrategy.PERMUTATION: 3>, 'PERMUTATION_SEEDED': <LLoopStrategy.PERMUTATION_SEEDED: 4>, 'SEEDED': <LLoopStrategy.SEEDED: 5>}
    @typing.overload
    def __eq__(self, other: LLoopStrategy) -> bool:
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
    def __ne__(self, other: LLoopStrategy) -> bool:
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
class Lambda2CVDiagnostics:
    """
    Health record of a lambda_2 tuner run: whether it ran, its cost, the truncation-edge count, and the boundary-pick guard fields for the returned pick (grid_index / grid_size, at_grid_edge, profile_at_edge, boundary_pick).
    """
    def __init__(self) -> None:
        ...
    @property
    def at_grid_edge(self) -> bool:
        """
        The returned lambda_2 is the first or last grid point: the grid boundary, not a CV optimum.
        """
    @property
    def boundary_pick(self) -> bool:
        """
        at_grid_edge or profile_at_edge: the selector warned once and returned the pick unchanged.
        """
    @property
    def dfmax_ratio(self) -> float:
        """
        IEN tuner: the support-cap ratio it was called with (cv_dfmax_ratio, as passed; <= 0 disabled the cap). 0 when no IEN-geometry tuner ran.
        """
    @property
    def grid_index(self) -> int:
        """
        0-based index of the returned lambda_2 on the tuner's grid (-1: no tuner ran).
        """
    @property
    def grid_size(self) -> int:
        """
        Number of lambda_2 grid points searched (0: no tuner ran).
        """
    @property
    def low_dimensional(self) -> bool:
        """
        p <= cap: the cap was inert.
        """
    @property
    def n_profile_at_edge(self) -> int:
        """
        lambda_2 values whose profiled lambda_1 optimum landed on a truncation edge (0 = healthy).
        """
    @property
    def profile_at_edge(self) -> bool:
        """
        IEN tuner: the profiled lambda_1 optimum of the chosen lambda_2 sat on the truncation edge of the lambda_1 path.
        """
    @property
    def profile_at_support_cap(self) -> bool:
        """
        IEN tuner: profile_at_edge, and that edge is the support cap (raise cv_dfmax_ratio to move it).
        """
    @property
    def ran(self) -> bool:
        """
        A tuner ran (False: lambda_2 came from the caller).
        """
    @property
    def seconds(self) -> float:
        """
        Wall-clock seconds spent resolving lambda_2.
        """
    @property
    def support_cap(self) -> int:
        """
        Support cap (glmnet dfmax) applied, or -1 when disabled.
        """
class Lambda2PickInfo:
    """
    How the knee was read off a selection-count profile (the pilot rule and the power rule).
    """
    def __init__(self) -> None:
        ...
    @property
    def index(self) -> int:
        """
        0-based index of the pick on the grid.
        """
    @property
    def max_selected(self) -> int:
        """
        Largest selection count on the grid.
        """
    @property
    def n_ties(self) -> int:
        """
        Grid values within the tolerance of the maximum (at tolerance 0: the values attaining it).
        """
class Lambda2PilotResult:
    """
    Record of a lambda_2 pilot run (PILOT_KNEE): the grid, the per-value counts and curves, the pick, and the pilot's size.
    """
    def __init__(self) -> None:
        ...
    @property
    def K_pilot(self) -> int:
        """
        Pilot experiments per grid value.
        """
    @property
    def L_0(self) -> int:
        """
        Pilot dummy multiplier.
        """
    @property
    def T_pilot(self) -> int:
        """
        Pilot T-steps.
        """
    @property
    def at_lower_end(self) -> bool:
        """
        The pick is the first grid value.
        """
    @property
    def at_upper_end(self) -> bool:
        """
        The pick is the last grid value.
        """
    @property
    def calibrated(self) -> list[int]:
        """
        Calibrated count per grid value.
        """
    @property
    def counts(self) -> list[int]:
        """
        Selection count per grid value under pilot_statistic.
        """
    @property
    def curve_majority(self) -> list[list[int]]:
        """
        Per grid value: majority count per pilot T-step.
        """
    @property
    def curve_soft(self) -> list[list[float]]:
        """
        Per grid value: calibrated (soft) count per pilot T-step.
        """
    @property
    def dummy_tag(self) -> int:
        """
        Dummy stream tag the pilot drew under.
        """
    @property
    def grid(self) -> list[float]:
        """
        The lambda_2 grid probed.
        """
    @property
    def group_majority(self) -> list[int]:
        """
        Group-majority count per grid value.
        """
    @property
    def groups_qualifying(self) -> list[int]:
        """
        Groups qualifying per grid value.
        """
    @property
    def index(self) -> int:
        """
        0-based index of the picked grid value.
        """
    @property
    def lambda2(self) -> float:
        """
        The picked lambda_2.
        """
    @property
    def pick(self) -> Lambda2PickInfo:
        """
        Lambda2PickInfo of the pick.
        """
    @property
    def probe_seconds(self) -> list[float]:
        """
        Wall-clock seconds per grid value.
        """
    @property
    def ran(self) -> bool:
        """
        A pilot ran (False: lambda_2 came from the caller or a CV tuner).
        """
    @property
    def seconds(self) -> float:
        """
        Wall-clock seconds of the whole pilot.
        """
class Lambda2PowerRuleResult:
    """
    The full lambda_2 profile and its knee: every grid value's selection, the pick and the boundary flags.
    """
    def __init__(self) -> None:
        ...
    def counts(self) -> list[int]:
        """
        The selection counts, ascending in lambda_2.
        """
    def selection(self) -> list[int]:
        """
        The selection at the chosen lambda_2 (0-based, ascending).
        """
    @property
    def at_lower_end(self) -> bool:
        """
        The pick is the first grid value (widen the grid before trusting it).
        """
    @property
    def at_upper_end(self) -> bool:
        """
        The pick is the last grid value.
        """
    @property
    def index(self) -> int:
        """
        0-based index of the pick into probes.
        """
    @property
    def lambda2(self) -> float:
        """
        The chosen lambda_2.
        """
    @property
    def pick(self) -> Lambda2PickInfo:
        """
        Lambda2PickInfo of the pick.
        """
    @property
    def probes(self) -> list[Lambda2Probe]:
        """
        Every grid value's selection, ascending in lambda_2.
        """
    @property
    def seconds(self) -> float:
        """
        Wall-clock seconds of the whole profile.
        """
class Lambda2Probe:
    """
    One grid value's selection in the full lambda_2 profile of tikhonov_lambda2_power_rule().
    """
    def __init__(self) -> None:
        ...
    @property
    def T_stop(self) -> int:
        """
        The calibrated stopping time of the run.
        """
    @property
    def lambda2(self) -> float:
        """
        The lambda_2 this probe ran at.
        """
    @property
    def seconds(self) -> float:
        """
        Wall-clock seconds of the run.
        """
    @property
    def selected(self) -> list[int]:
        """
        The selected variables (0-based, ascending).
        """
class LambdaSelectionMethod:
    """
    Rule for auto-selecting the ridge parameter lambda_2 when lambda_2 < 0.
    
    Members:
    
      CV_1SE_SVD : JacobiSVD ridge CV, 1-SE rule.
    
      CV_MIN_SVD : JacobiSVD ridge CV, minimum-CV-error rule.
    
      CV_1SE_CCD : Coordinate-descent ridge CV, glmnet-faithful 1-SE rule (default).
    
      CV_MIN_CCD : Coordinate-descent ridge CV, glmnet-faithful minimum rule.
    
      CV_1SE_IEN_CCD : IEN-geometry coordinate-descent CV (lambda1 profiled, group-consuming), 1-SE rule. IEN default.
    
      CV_MIN_IEN_CCD : IEN-geometry coordinate-descent CV (lambda1 profiled, group-consuming), minimum rule.
    
      CV_1SE_TIK_SVD : Generalized-Tikhonov SVD ridge CV (IEN geometry), 1-SE rule.
    
      CV_MIN_TIK_SVD : Generalized-Tikhonov SVD ridge CV (IEN geometry), minimum rule.
    """
    CV_1SE_CCD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_1SE_CCD: 2>
    CV_1SE_IEN_CCD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_1SE_IEN_CCD: 4>
    CV_1SE_SVD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_1SE_SVD: 0>
    CV_1SE_TIK_SVD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_1SE_TIK_SVD: 6>
    CV_MIN_CCD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_MIN_CCD: 3>
    CV_MIN_IEN_CCD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_MIN_IEN_CCD: 5>
    CV_MIN_SVD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_MIN_SVD: 1>
    CV_MIN_TIK_SVD: typing.ClassVar[LambdaSelectionMethod]  # value = <LambdaSelectionMethod.CV_MIN_TIK_SVD: 7>
    __members__: typing.ClassVar[dict[str, LambdaSelectionMethod]]  # value = {'CV_1SE_SVD': <LambdaSelectionMethod.CV_1SE_SVD: 0>, 'CV_MIN_SVD': <LambdaSelectionMethod.CV_MIN_SVD: 1>, 'CV_1SE_CCD': <LambdaSelectionMethod.CV_1SE_CCD: 2>, 'CV_MIN_CCD': <LambdaSelectionMethod.CV_MIN_CCD: 3>, 'CV_1SE_IEN_CCD': <LambdaSelectionMethod.CV_1SE_IEN_CCD: 4>, 'CV_MIN_IEN_CCD': <LambdaSelectionMethod.CV_MIN_IEN_CCD: 5>, 'CV_1SE_TIK_SVD': <LambdaSelectionMethod.CV_1SE_TIK_SVD: 6>, 'CV_MIN_TIK_SVD': <LambdaSelectionMethod.CV_MIN_TIK_SVD: 7>}
    @typing.overload
    def __eq__(self, other: LambdaSelectionMethod) -> bool:
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
    def __ne__(self, other: LambdaSelectionMethod) -> bool:
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
class PilotStatistic:
    """
    Selection count the pilot rule reads per grid value.
    
    Members:
    
      MAJORITY : Variables selected in more than half of the pilot experiments.
    
      CALIBRATED : Calibrated count (soft curve).
    
      GROUP_MAJORITY : Groups whose selected fraction exceeds pilot_group_fraction.
    """
    CALIBRATED: typing.ClassVar[PilotStatistic]  # value = <PilotStatistic.CALIBRATED: 1>
    GROUP_MAJORITY: typing.ClassVar[PilotStatistic]  # value = <PilotStatistic.GROUP_MAJORITY: 2>
    MAJORITY: typing.ClassVar[PilotStatistic]  # value = <PilotStatistic.MAJORITY: 0>
    __members__: typing.ClassVar[dict[str, PilotStatistic]]  # value = {'MAJORITY': <PilotStatistic.MAJORITY: 0>, 'CALIBRATED': <PilotStatistic.CALIBRATED: 1>, 'GROUP_MAJORITY': <PilotStatistic.GROUP_MAJORITY: 2>}
    @typing.overload
    def __eq__(self, other: PilotStatistic) -> bool:
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
    def __ne__(self, other: PilotStatistic) -> bool:
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
class SPCAMode:
    """
    Members:
    
      ActiveSet : Ridge regression on the T-Rex active set to compute loadings.
    
      Thresholded : Keep and normalize the top |A| ordinary PCA loadings.
    """
    ActiveSet: typing.ClassVar[SPCAMode]  # value = <SPCAMode.ActiveSet: 0>
    Thresholded: typing.ClassVar[SPCAMode]  # value = <SPCAMode.Thresholded: 1>
    __members__: typing.ClassVar[dict[str, SPCAMode]]  # value = {'ActiveSet': <SPCAMode.ActiveSet: 0>, 'Thresholded': <SPCAMode.Thresholded: 1>}
    @typing.overload
    def __eq__(self, other: SPCAMode) -> bool:
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
    def __ne__(self, other: SPCAMode) -> bool:
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
class ScalingMode:
    """
    Working-design convention of the T-Rex selectors. L2 and ZSCORE center X and y and scale the columns of X (dummies follow); NONE leaves X and y exactly as supplied (user-managed centering and scaling, e.g. a pre-normalized design or an intercept-free dictionary), with the dummies kept on the unit-L2 standard.
    
    Members:
    
      L2 : Divide each centered column by its L2 norm (unit L2 norm).
    
      ZSCORE : Divide each centered column by its sample SD (n-1); glmnet standardize=TRUE convention.
    
      NONE : Leave X and y untouched (no centering, no scaling); the design should be unit-L2-commensurate since the dummies keep the unit-L2 standard. A design pre-normalized with normalize_design() selects under NONE exactly what the raw design selects under L2.
    """
    L2: typing.ClassVar[ScalingMode]  # value = <ScalingMode.L2: 0>
    NONE: typing.ClassVar[ScalingMode]  # value = <ScalingMode.NONE: 2>
    ZSCORE: typing.ClassVar[ScalingMode]  # value = <ScalingMode.ZSCORE: 1>
    __members__: typing.ClassVar[dict[str, ScalingMode]]  # value = {'L2': <ScalingMode.L2: 0>, 'ZSCORE': <ScalingMode.ZSCORE: 1>, 'NONE': <ScalingMode.NONE: 2>}
    @typing.overload
    def __eq__(self, other: ScalingMode) -> bool:
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
    def __ne__(self, other: ScalingMode) -> bool:
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
    @typing.overload
    def __eq__(self, other: ScreenTRexMethod) -> bool:
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
    def __ne__(self, other: ScreenTRexMethod) -> bool:
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
    def cd_certified_localization(self) -> bool:
        """
        CCD family: True (default) = certified crossing localization (duality-gap and coordinate-status certificates, closed-form continuation, records at the knot); False = LEGACY probe bisection with the support-nesting stop.
        """
    @cd_certified_localization.setter
    def cd_certified_localization(self, arg0: bool) -> None:
        ...
    @property
    def cd_gram_cap(self) -> int:
        """
        CCD family: column cap for the cached Gram matrix (0 keeps the solver default).
        """
    @cd_gram_cap.setter
    def cd_gram_cap(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_lambda_rel_tol(self) -> float:
        """
        CCD family: relative lambda step tolerance of the crossing search (<= 0 keeps the solver default).
        """
    @cd_lambda_rel_tol.setter
    def cd_lambda_rel_tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_max_sweeps(self) -> int:
        """
        CCD family: maximum coordinate sweeps per subproblem (0 keeps the solver default).
        """
    @cd_max_sweeps.setter
    def cd_max_sweeps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_tol_certify(self) -> float:
        """
        CCD family: KKT certification tolerance; applied only when set together with cd_tol_probe (both > 0).
        """
    @cd_tol_certify.setter
    def cd_tol_certify(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_tol_probe(self) -> float:
        """
        CCD family: probing tolerance; applied only when set together with cd_tol_certify (both > 0).
        """
    @cd_tol_probe.setter
    def cd_tol_probe(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def exch_tie_alpha(self) -> float:
        """
        Exchangeable-tie band width for greedy solvers (TOMP/TAFS) in pairwise ranking-noise sd units; 0 = off. Recommended under trex+DA: 0.25.
        """
    @exch_tie_alpha.setter
    def exch_tie_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def exch_tie_floor(self) -> float:
        """
        Minimum |correlation| for exchangeable-tie candidates in (0, 1); ignored unless exch_tie_alpha > 0.
        """
    @exch_tie_floor.setter
    def exch_tie_floor(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
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
    def tenet_aug_use_lars(self) -> bool:
        """
        TENET_AUG: False (default) = standard LASSO inner path; True = optional pure-LARS inner path that never drops variables.
        """
    @tenet_aug_use_lars.setter
    def tenet_aug_use_lars(self, arg0: bool) -> None:
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
    
      TENET_AUG : Terminating Elastic Net via augmented LASSO; accepted wherever TENET runs (LASSO inner path standard, LARS via tenet_aug_use_lars).
    
      TCDLASSO : Terminating coordinate-descent LASSO (exact penalized-lasso minimizers per crossing; certified crossing localization by default).
    
      TCDENET : Terminating coordinate-descent Elastic Net.
    
      TIENET : Terminating Informed Elastic Net, native pathwise (GVS-only, gvs_type=IEN; IEN default).
    
      TIENET_AUG : Terminating Informed Elastic Net via augmented LASSO (GVS-only, gvs_type=IEN).
    
      TCDIENET : Terminating coordinate-descent Informed Elastic Net (GVS-only, gvs_type=IEN).
    
      TCDTIKNET : Terminating coordinate-descent Tikhonov Net (sparse-K; needs a penalty matrix, TRexTikhonovSelector only — not yet bound in Python).
    
      TIKNET : Terminating Tikhonov Net, native pathwise (sparse-K, FOLDED; TRexTikhonovSelector only).
    """
    TACGP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TACGP: 15>
    TAFS: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TAFS: 19>
    TCDENET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TCDENET: 10>
    TCDIENET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TCDIENET: 11>
    TCDLASSO: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TCDLASSO: 9>
    TCDTIKNET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TCDTIKNET: 12>
    TENET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TENET: 2>
    TENET_AUG: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TENET_AUG: 3>
    TGP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TGP: 14>
    TIENET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TIENET: 4>
    TIENET_AUG: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TIENET_AUG: 5>
    TIKNET: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TIKNET: 6>
    TLARS: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TLARS: 0>
    TLASSO: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TLASSO: 1>
    TMP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TMP: 16>
    TNCGMP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TNCGMP: 17>
    TOMP: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TOMP: 13>
    TOOLS: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TOOLS: 18>
    TSTAGEWISE: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TSTAGEWISE: 8>
    TSTEPWISE: typing.ClassVar[SolverTypeForTRex]  # value = <SolverTypeForTRex.TSTEPWISE: 7>
    __members__: typing.ClassVar[dict[str, SolverTypeForTRex]]  # value = {'TLARS': <SolverTypeForTRex.TLARS: 0>, 'TLASSO': <SolverTypeForTRex.TLASSO: 1>, 'TENET': <SolverTypeForTRex.TENET: 2>, 'TSTEPWISE': <SolverTypeForTRex.TSTEPWISE: 7>, 'TSTAGEWISE': <SolverTypeForTRex.TSTAGEWISE: 8>, 'TOMP': <SolverTypeForTRex.TOMP: 13>, 'TGP': <SolverTypeForTRex.TGP: 14>, 'TACGP': <SolverTypeForTRex.TACGP: 15>, 'TMP': <SolverTypeForTRex.TMP: 16>, 'TNCGMP': <SolverTypeForTRex.TNCGMP: 17>, 'TOOLS': <SolverTypeForTRex.TOOLS: 18>, 'TAFS': <SolverTypeForTRex.TAFS: 19>, 'TENET_AUG': <SolverTypeForTRex.TENET_AUG: 3>, 'TCDLASSO': <SolverTypeForTRex.TCDLASSO: 9>, 'TCDENET': <SolverTypeForTRex.TCDENET: 10>, 'TIENET': <SolverTypeForTRex.TIENET: 4>, 'TIENET_AUG': <SolverTypeForTRex.TIENET_AUG: 5>, 'TCDIENET': <SolverTypeForTRex.TCDIENET: 11>, 'TCDTIKNET': <SolverTypeForTRex.TCDTIKNET: 12>, 'TIKNET': <SolverTypeForTRex.TIKNET: 6>}
    @typing.overload
    def __eq__(self, other: SolverTypeForTRex) -> bool:
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
    def __ne__(self, other: SolverTypeForTRex) -> bool:
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
class TRexBiobankScreeningSelector:
    """
    Biobank-scale T-Rex selector for FDR-controlled variable selection across one or many phenotypes.
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "False Discovery Rate Control for Fast
      Screening of Large-Scale Genomics Biobanks," in Proc. 22nd IEEE Statist. Signal
      Process. Workshop (SSP), 2023, pp. 666-670.
    F. Scheidt, J. Machkour, and M. Muma, "Solving FDR-Controlled Sparse Regression Problems
      with Five Million Variables on a Laptop," in Proc. IEEE 9th Int. Workshop Comput. Adv.
      Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 116-120.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
        Thread budget of the solvers' own passes over the design (default 1, serial; a whole number of at least 1). Live when the experiments run in sequence; inside a parallel run the solvers stay serial. A pass splits only above about a million entries.
        """
    @max_inner_threads.setter
    def max_inner_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_outer_threads(self) -> int:
        """
        Threads of the parallel K-loop (default 1; a whole number of at least 1), capped at K. Ignored while parallel_rnd_experiments is False.
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
        Run the K random experiments in parallel (default False: in sequence). Inside a parallel run every solver runs serially; the lean L-loop strategies (SEEDED, PERMUTATION, PERMUTATION_SEEDED) always run their experiments in sequence.
        """
    @parallel_rnd_experiments.setter
    def parallel_rnd_experiments(self, arg0: bool) -> None:
        ...
    @property
    def scaling_mode(self) -> ScalingMode:
        """
        Working-design convention: L2 (default) or ZSCORE center X and y and scale X; NONE leaves X and y as supplied (user-managed; dummies keep the unit-L2 standard). Non-finite entries and dead columns are rejected in every mode (ValueError).
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
        Prior grouping constraints (label vectors of length p); sub-clustering runs within their finest common refinement.
        """
    @prior_groups.setter
    def prior_groups(self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]]) -> None:
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
    Dependency-aware T-Rex selector (T-Rex+DA): FDR-controlled variable selection for dependent
    variables, with a correlation-structure model (AR1, EQUI, BT, NN or PRIOR_GROUPS) that
    accounts for the dependencies among the variables.
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "High-Dimensional False Discovery Rate Control
      for Dependent Variables," Signal Process., vol. 234, Art. no. 109990, 2025,
      doi: 10.1016/j.sigpro.2025.109990.
    J. Machkour, D. P. Palomar, and M. Muma, "FDR-Controlled Portfolio Optimization for
      Sparse Financial Index Tracking," in Proc. IEEE Int. Conf. Acoust., Speech, Signal
      Process. (ICASSP), 2025, doi: 10.1109/ICASSP49660.2025.10890629.
    J. Machkour, M. Muma, and D. P. Palomar, "FDR-Controlled Sparse Index Tracking with
      Autoregressive Stock Dependency Models," in Proc. Eur. Signal Process. Conf.
      (EUSIPCO), 2024, pp. 2662-2666.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
    def cv_dfmax_ratio(self) -> float:
        """
        IEN-geometry CD tuners (CV_*_IEN_CCD): support cap as a fraction of the smallest fold's training size (default 0.8; <= 0 disables the cap; larger values allow larger supports at a higher tuner cost; must be finite).
        """
    @cv_dfmax_ratio.setter
    def cv_dfmax_ratio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
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
        Number of points on the lambda grid searched during CV (EN-shaped rules).
        """
    @cv_n_lambda.setter
    def cv_n_lambda(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_lambda1(self) -> int:
        """
        IEN-geometry CD tuners (CV_*_IEN_CCD): points on the profiled lambda_1 path per lambda_2 value (default 40).
        """
    @cv_n_lambda1.setter
    def cv_n_lambda1(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_lambda2(self) -> int:
        """
        IEN-geometry CD tuners: points on the lambda_2 grid (default 100).
        """
    @cv_n_lambda2.setter
    def cv_n_lambda2(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
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
        EN solver variant when gvs_type == EN: TENET (default), TENET_AUG, or TCDENET.
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
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "False Discovery Rate Control for Grouped
      Variable Selection in High-Dimensional Linear Models Using the T-Knock Filter," in
      Proc. 30th Eur. Signal Process. Conf. (EUSIPCO), 2022, pp. 892-896.
    J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped
      Variable Selection and FDR Control in Genomics Research," in Proc. IEEE 9th Int.
      Workshop Comput. Adv. Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 466-470.
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tFDR: typing.SupportsFloat | typing.SupportsIndex = 0.1, gvs_control: TRexGVSControlParameter = ..., trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = True) -> None:
        """
        Construct the GVS selector. X and y are accessed zero-copy via Eigen::Map.
        """
    def getGVSResult(self) -> GVSSelectionResult:
        """
        Return the GVSSelectionResult after calling select().
        """
    def getLambda2CVDiagnostics(self) -> Lambda2CVDiagnostics:
        """
        Return the lambda_2 tuner health record of the last select().
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
        TRexGVSControlParameter forwarded to each per-PC GVS run. gvs_ctrl.gvs_type must stay GVSType.EN (T-Rex SPCA is defined on the elastic net, IEN raises ValueError). Set gvs_ctrl.lambda_2 > 0 to bypass auto-determination. The base T-Rex parameters are configured via gvs_ctrl.trex_ctrl.
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
    
    References
    ----------
    J. Machkour, A. Breloy, M. Muma, D. P. Palomar, and F. Pascal, "Sparse PCA with False
      Discovery Rate Controlled Variable Selection," in Proc. IEEE 49th Int. Conf. Acoust.,
      Speech, Signal Process. (ICASSP), 2024, pp. 9716-9720.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "False Discovery Rate Control for Fast
      Screening of Large-Scale Genomics Biobanks," in Proc. 22nd IEEE Statist. Signal
      Process. Workshop (SSP), 2023, pp. 666-670.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
    def getNumDummies(self) -> int:
        """
        Number of dummies used by the last select() (the L-loop's calibration).
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
    def restore(self) -> None:
        """
        Give X back before select(): undo the in-place normalization applied at construction and release the buffer claim (idempotent; a no-op after select(), which restores X itself). A restored selector cannot run: select() then raises, construct a new one. Without it, an unselected selector gives X back only when it is garbage-collected.
        """
    def select(self) -> SelectionResult:
        """
        Run the T-Rex algorithm and return a SelectionResult.
        """
class TRexTikhonovControlParameter:
    """
    Control parameters for TRexTikhonovSelector: the penalty (a dense p x p PSD matrix tikhonov_K, or GROUP_CENTERING built from groups), the lambda_2 rule, and the cluster-aware dummies.
    """
    def __init__(self) -> None:
        ...
    @property
    def cluster_dummies(self) -> bool:
        """
        Cluster-aware MVN dummies (default True); False = i.i.d. dummies (the K = I anchor).
        """
    @cluster_dummies.setter
    def cluster_dummies(self, arg0: bool) -> None:
        ...
    @property
    def corr_max(self) -> float:
        """
        HAC cut: maximum within-cluster correlation when prior_groups is empty.
        """
    @corr_max.setter
    def corr_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_dfmax_ratio(self) -> float:
        """
        IEN-geometry CD tuners: support cap as a fraction of the smallest fold's training size (default 0.8; <= 0 disables the cap; larger values allow larger supports at a higher tuner cost; must be finite).
        """
    @cv_dfmax_ratio.setter
    def cv_dfmax_ratio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_folds(self) -> int:
        """
        Folds of the CV tuners.
        """
    @cv_n_folds.setter
    def cv_n_folds(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_lambda(self) -> int:
        """
        SVD tuners: points on the lambda_2 grid.
        """
    @cv_n_lambda.setter
    def cv_n_lambda(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_lambda1(self) -> int:
        """
        IEN-geometry CD tuners: points on the profiled lambda_1 path per lambda_2 value.
        """
    @cv_n_lambda1.setter
    def cv_n_lambda1(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_n_lambda2(self) -> int:
        """
        IEN-geometry CD tuners: points on the lambda_2 grid.
        """
    @cv_n_lambda2.setter
    def cv_n_lambda2(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cv_seed(self) -> int:
        """
        Seed for the CV fold permutation (-1 = derive from the T-Rex seed).
        """
    @cv_seed.setter
    def cv_seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def fold_dummy_coupling(self) -> bool:
        """
        FOLDED dummy coupling: dummies take the K block of their real counterpart (default True).
        """
    @fold_dummy_coupling.setter
    def fold_dummy_coupling(self, arg0: bool) -> None:
        ...
    @property
    def hc_linkage(self) -> ...:
        """
        HAC linkage (ml_methods.clustering.LinkageMethod).
        """
    @hc_linkage.setter
    def hc_linkage(self, arg0: ...) -> None:
        ...
    @property
    def lambda2_method(self) -> TikLambda2Method:
        """
        Rule for resolving lambda_2 when lambda_2 < 0 (default PILOT_KNEE, the pilot knee; the CV tuners are the alternatives).
        """
    @lambda2_method.setter
    def lambda2_method(self, arg0: TikLambda2Method) -> None:
        ...
    @property
    def lambda_2(self) -> float:
        """
        Tikhonov penalty (solver scale): < 0 = resolve by lambda2_method, 0 = no ridge, > 0 = fixed.
        """
    @lambda_2.setter
    def lambda_2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def penalty_structure(self) -> TikPenaltyStructure:
        """
        SPARSE_GENERAL (tikhonov_K given) or GROUP_CENTERING (K from the groups; TCDTIKNET then runs through the group-centering hooks of TCDIENET, so its warnings carry the tag [TCDIENET]).
        """
    @penalty_structure.setter
    def penalty_structure(self, arg0: TikPenaltyStructure) -> None:
        ...
    @property
    def pilot_K(self) -> int:
        """
        Pilot experiments per grid value.
        """
    @pilot_K.setter
    def pilot_K(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def pilot_L_multiplier(self) -> int:
        """
        Pilot dummy multiplier (in [1, trex_control.max_dummy_multiplier]).
        """
    @pilot_L_multiplier.setter
    def pilot_L_multiplier(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def pilot_T(self) -> int:
        """
        Pilot T-steps.
        """
    @pilot_T.setter
    def pilot_T(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def pilot_grid(self) -> list[float]:
        """
        lambda_2 grid of the pilot rule (keep 0 on it so that no penalty is an admissible answer).
        """
    @pilot_grid.setter
    def pilot_grid(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def pilot_group_fraction(self) -> float:
        """
        GROUP_MAJORITY: fraction of a group that must be selected.
        """
    @pilot_group_fraction.setter
    def pilot_group_fraction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pilot_knee_tolerance(self) -> float:
        """
        Pilot rule: selections below the largest count that still count as on the plateau (0 reads the largest count).
        """
    @pilot_knee_tolerance.setter
    def pilot_knee_tolerance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pilot_shared_dummies(self) -> bool:
        """
        Pilot draws its dummies under the main run's first block tag.
        """
    @pilot_shared_dummies.setter
    def pilot_shared_dummies(self, arg0: bool) -> None:
        ...
    @property
    def pilot_statistic(self) -> PilotStatistic:
        """
        Selection count the pilot reads (MAJORITY, CALIBRATED, GROUP_MAJORITY).
        """
    @pilot_statistic.setter
    def pilot_statistic(self, arg0: PilotStatistic) -> None:
        ...
    @property
    def prior_groups(self) -> list[int]:
        """
        Group assignment (0-based, length p; empty = HAC clusters from corr_max / hc_linkage).
        """
    @prior_groups.setter
    def prior_groups(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def tikhonov_K(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        The p x p PSD penalty matrix K as a dense array (converted to sparse internally; read back dense). Required under SPARSE_GENERAL.
        """
    @tikhonov_K.setter
    def tikhonov_K(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def trex_ctrl(self) -> TRexControlParameter:
        """
        Nested base T-Rex control (overridden by the separate trex_control argument of TRexTikhonovSelector).
        """
    @trex_ctrl.setter
    def trex_ctrl(self, arg0: TRexControlParameter) -> None:
        ...
class TRexTikhonovSelector(TRexSelector):
    """
    T-Rex selector with a general Tikhonov penalty (TCDTIKNET / TIKNET) and cluster-aware dummies.
    
    References
    ----------
    A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization
      Method," Soviet Math. Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl.
      Akad. Nauk SSSR 151 (1963), no. 3, 501-504).
    Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov
      Regularization Problems in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25,
      2018, doi: 10.1016/j.cam.2018.04.049.
    A. J. Smola and R. Kondor, "Kernels and Regularization on Graphs," in Learning Theory
      and Kernel Machines (COLT/Kernel 2003), Lecture Notes in Artificial Intelligence
      vol. 2777, 2003, pp. 144-158.
    F. R. K. Chung, Spectral Graph Theory. CBMS Regional Conference Series in Mathematics,
      no. 92. American Mathematical Society, 1997.
    A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal
      Problems," Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article
      in the Special 40th Anniversary Issue).
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tFDR: typing.SupportsFloat | typing.SupportsIndex = 0.1, tik_control: TRexTikhonovControlParameter = ..., trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = -1, verbose: bool = True) -> None:
        """
        Construct the Tikhonov selector. X and y are accessed zero-copy via Eigen::Map.
        """
    def getLambda2CVDiagnostics(self) -> Lambda2CVDiagnostics:
        """
        Return the lambda_2 tuner health record of the last select().
        """
    def getLambda2PilotResult(self) -> Lambda2PilotResult:
        """
        Return the lambda_2 pilot record of the last select().
        """
    def getLambda2Used(self) -> float:
        """
        lambda_2 used by the last select() (solver scale); -1 before select().
        """
class TikLambda2Method:
    """
    Rule for resolving lambda_2 when tik_control.lambda_2 < 0: the pilot knee (default) or a k-fold CV tuner. The knee of the full selection profile is tikhonov_lambda2_power_rule().
    
    Members:
    
      CV_1SE_IEN_CCD : IEN-geometry coordinate-descent CV (lambda_1 profiled), 1-SE rule.
    
      CV_MIN_IEN_CCD : IEN-geometry coordinate-descent CV, minimum rule.
    
      CV_1SE_TIK_SVD : Generalized-Tikhonov SVD ridge CV, 1-SE rule.
    
      CV_MIN_TIK_SVD : Generalized-Tikhonov SVD ridge CV, minimum rule.
    
      PILOT_KNEE : The pilot knee (default): a few random experiments per grid value, the smallest grid value whose selection count is within pilot_knee_tolerance of the maximum.
    """
    CV_1SE_IEN_CCD: typing.ClassVar[TikLambda2Method]  # value = <TikLambda2Method.CV_1SE_IEN_CCD: 0>
    CV_1SE_TIK_SVD: typing.ClassVar[TikLambda2Method]  # value = <TikLambda2Method.CV_1SE_TIK_SVD: 2>
    CV_MIN_IEN_CCD: typing.ClassVar[TikLambda2Method]  # value = <TikLambda2Method.CV_MIN_IEN_CCD: 1>
    CV_MIN_TIK_SVD: typing.ClassVar[TikLambda2Method]  # value = <TikLambda2Method.CV_MIN_TIK_SVD: 3>
    PILOT_KNEE: typing.ClassVar[TikLambda2Method]  # value = <TikLambda2Method.PILOT_KNEE: 4>
    __members__: typing.ClassVar[dict[str, TikLambda2Method]]  # value = {'CV_1SE_IEN_CCD': <TikLambda2Method.CV_1SE_IEN_CCD: 0>, 'CV_MIN_IEN_CCD': <TikLambda2Method.CV_MIN_IEN_CCD: 1>, 'CV_1SE_TIK_SVD': <TikLambda2Method.CV_1SE_TIK_SVD: 2>, 'CV_MIN_TIK_SVD': <TikLambda2Method.CV_MIN_TIK_SVD: 3>, 'PILOT_KNEE': <TikLambda2Method.PILOT_KNEE: 4>}
    @typing.overload
    def __eq__(self, other: TikLambda2Method) -> bool:
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
    def __ne__(self, other: TikLambda2Method) -> bool:
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
class TikPenaltyStructure:
    """
    Where the penalty matrix comes from.
    
    Members:
    
      SPARSE_GENERAL : tik_control.tikhonov_K is the p x p PSD penalty matrix (dense array, converted to sparse).
    
      GROUP_CENTERING : K = I - J/p_m per group, built from prior_groups (or the HAC clusters).
    """
    GROUP_CENTERING: typing.ClassVar[TikPenaltyStructure]  # value = <TikPenaltyStructure.GROUP_CENTERING: 1>
    SPARSE_GENERAL: typing.ClassVar[TikPenaltyStructure]  # value = <TikPenaltyStructure.SPARSE_GENERAL: 0>
    __members__: typing.ClassVar[dict[str, TikPenaltyStructure]]  # value = {'SPARSE_GENERAL': <TikPenaltyStructure.SPARSE_GENERAL: 0>, 'GROUP_CENTERING': <TikPenaltyStructure.GROUP_CENTERING: 1>}
    @typing.overload
    def __eq__(self, other: TikPenaltyStructure) -> bool:
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
    def __ne__(self, other: TikPenaltyStructure) -> bool:
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
def build_group_centering_k(prior_groups: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Build the GROUP_CENTERING penalty K = blockdiag(I - J/p_m) from a 0-based group assignment (dense out).
    """
def gamma_to_k(gamma: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Build the penalty matrix K = Gamma^T Gamma from a (q x p) Tikhonov operator Gamma (dense in, dense out).
    """
def normalize_design(X: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], scaling_mode: ScalingMode = ..., eps: typing.SupportsFloat | typing.SupportsIndex = 1e-06, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> dict:
    """
    Apply the selectors' data normalization to copies of X and y and return a dict with keys 'X', 'y' (the normalized copies), 'X_means', 'X_scales' (the per-column divisors: L2 norms for L2, sample SDs for ZSCORE, ones for NONE) and 'y_mean'. Under ScalingMode.NONE the copies equal the inputs and the parameters are the identity. eps is the dead-column threshold on the working L2 norm (the selectors use the solver tolerance, default 1e-6). n_threads is the thread budget of the column passes (None: the library's default budget, utils.get_num_threads(), initially 1). Raises ValueError on a non-finite entry or a dead column (constant under L2 / ZSCORE, all-zero under NONE), naming the column (0-based).
    """
def tikhonov_lambda2_power_rule(X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tFDR: typing.SupportsFloat | typing.SupportsIndex = 0.1, tik_control: TRexTikhonovControlParameter = ..., trex_control: TRexControlParameter = ..., seed: typing.SupportsInt | typing.SupportsIndex = 0, grid: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0], knee_tolerance: typing.SupportsFloat | typing.SupportsIndex = 1.0, verbose: bool = False) -> Lambda2PowerRuleResult:
    """
    The knee of the full lambda_2 profile of the Tikhonov T-Rex: one FDR-controlled selection per grid value with the same seed (common dummies), then the smallest lambda_2 whose selection is within knee_tolerance selections of the largest (0 reads the largest selection, ties toward the smallest lambda_2). This is the reference rule behind TikLambda2Method.PILOT_KNEE at the cost of one selection per grid value; it returns every probe's selection. X (F-contiguous float64) and y are accessed zero-copy: every probe normalizes X in place and hands it back, so X is unchanged on return. The pick depends on y and lies outside the fixed-lambda_2 FDR guarantee that each probe carries.
    """
AR1: DAMethod  # value = <DAMethod.AR1: 0>
ActiveSet: SPCAMode  # value = <SPCAMode.ActiveSet: 0>
BT: DAMethod  # value = <DAMethod.BT: 2>
CALIBRATED: PilotStatistic  # value = <PilotStatistic.CALIBRATED: 1>
CV_1SE_CCD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_1SE_CCD: 2>
CV_1SE_IEN_CCD: TikLambda2Method  # value = <TikLambda2Method.CV_1SE_IEN_CCD: 0>
CV_1SE_SVD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_1SE_SVD: 0>
CV_1SE_TIK_SVD: TikLambda2Method  # value = <TikLambda2Method.CV_1SE_TIK_SVD: 2>
CV_MIN_CCD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_MIN_CCD: 3>
CV_MIN_IEN_CCD: TikLambda2Method  # value = <TikLambda2Method.CV_MIN_IEN_CCD: 1>
CV_MIN_SVD: LambdaSelectionMethod  # value = <LambdaSelectionMethod.CV_MIN_SVD: 1>
CV_MIN_TIK_SVD: TikLambda2Method  # value = <TikLambda2Method.CV_MIN_TIK_SVD: 3>
EN: GVSType  # value = <GVSType.EN: 0>
EQUI: DAMethod  # value = <DAMethod.EQUI: 1>
FeasibleOnly: BTSelectionMode  # value = <BTSelectionMode.FeasibleOnly: 0>
GROUP_CENTERING: TikPenaltyStructure  # value = <TikPenaltyStructure.GROUP_CENTERING: 1>
GROUP_MAJORITY: PilotStatistic  # value = <PilotStatistic.GROUP_MAJORITY: 2>
HCONCAT: LLoopStrategy  # value = <LLoopStrategy.HCONCAT: 2>
IEN: GVSType  # value = <GVSType.IEN: 1>
L2: ScalingMode  # value = <ScalingMode.L2: 0>
MAJORITY: PilotStatistic  # value = <PilotStatistic.MAJORITY: 0>
NN: DAMethod  # value = <DAMethod.NN: 3>
NONE: ScalingMode  # value = <ScalingMode.NONE: 2>
PERMUTATION: LLoopStrategy  # value = <LLoopStrategy.PERMUTATION: 3>
PERMUTATION_SEEDED: LLoopStrategy  # value = <LLoopStrategy.PERMUTATION_SEEDED: 4>
PILOT_KNEE: TikLambda2Method  # value = <TikLambda2Method.PILOT_KNEE: 4>
PRIOR_GROUPS: DAMethod  # value = <DAMethod.PRIOR_GROUPS: 4>
RFaithful: BTSelectionMode  # value = <BTSelectionMode.RFaithful: 1>
SEEDED: LLoopStrategy  # value = <LLoopStrategy.SEEDED: 5>
SKIPL: LLoopStrategy  # value = <LLoopStrategy.SKIPL: 0>
SPARSE_GENERAL: TikPenaltyStructure  # value = <TikPenaltyStructure.SPARSE_GENERAL: 0>
STANDARD: LLoopStrategy  # value = <LLoopStrategy.STANDARD: 1>
TACGP: SolverTypeForTRex  # value = <SolverTypeForTRex.TACGP: 15>
TAFS: SolverTypeForTRex  # value = <SolverTypeForTRex.TAFS: 19>
TCDENET: ENSolverType  # value = <ENSolverType.TCDENET: 2>
TCDIENET: SolverTypeForTRex  # value = <SolverTypeForTRex.TCDIENET: 11>
TCDLASSO: SolverTypeForTRex  # value = <SolverTypeForTRex.TCDLASSO: 9>
TCDTIKNET: SolverTypeForTRex  # value = <SolverTypeForTRex.TCDTIKNET: 12>
TENET: ENSolverType  # value = <ENSolverType.TENET: 0>
TENET_AUG: ENSolverType  # value = <ENSolverType.TENET_AUG: 1>
TGP: SolverTypeForTRex  # value = <SolverTypeForTRex.TGP: 14>
TIENET: SolverTypeForTRex  # value = <SolverTypeForTRex.TIENET: 4>
TIENET_AUG: SolverTypeForTRex  # value = <SolverTypeForTRex.TIENET_AUG: 5>
TIKNET: SolverTypeForTRex  # value = <SolverTypeForTRex.TIKNET: 6>
TLARS: SolverTypeForTRex  # value = <SolverTypeForTRex.TLARS: 0>
TLASSO: SolverTypeForTRex  # value = <SolverTypeForTRex.TLASSO: 1>
TMP: SolverTypeForTRex  # value = <SolverTypeForTRex.TMP: 16>
TNCGMP: SolverTypeForTRex  # value = <SolverTypeForTRex.TNCGMP: 17>
TOMP: SolverTypeForTRex  # value = <SolverTypeForTRex.TOMP: 13>
TOOLS: SolverTypeForTRex  # value = <SolverTypeForTRex.TOOLS: 18>
TREX: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX: 0>
TREX_DA_AR1: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX_DA_AR1: 1>
TREX_DA_BLOCK_EQUI: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX_DA_BLOCK_EQUI: 3>
TREX_DA_EQUI: ScreenTRexMethod  # value = <ScreenTRexMethod.TREX_DA_EQUI: 2>
TSTAGEWISE: SolverTypeForTRex  # value = <SolverTypeForTRex.TSTAGEWISE: 8>
TSTEPWISE: SolverTypeForTRex  # value = <SolverTypeForTRex.TSTEPWISE: 7>
Thresholded: SPCAMode  # value = <SPCAMode.Thresholded: 1>
ZSCORE: ScalingMode  # value = <ScalingMode.ZSCORE: 1>
