"""
Augmented-dummy (AD) T-solvers: the classical family running on [X, D]
"""
from __future__ import annotations
import typing
from . import afs_based
from . import cd_based
from . import lars_based
from . import omp_based
__all__: list[str] = ['CENTERING', 'GroupPenaltyVariant', 'L2', 'MEAN', 'ScalingMode', 'SolverOptions', 'ZSCORE', 'afs_based', 'cd_based', 'lars_based', 'omp_based']
class GroupPenaltyVariant:
    """
    Structured penalty of the group CCD solver (TCDIENET).
    
    Members:
    
      MEAN : K = blockdiag(1 1^T / p_m): the canonical IEN group-sum penalty.
    
      CENTERING : K = blockdiag(I - 1 1^T / p_m): within-group centering.
    """
    CENTERING: typing.ClassVar[GroupPenaltyVariant]  # value = <GroupPenaltyVariant.CENTERING: 1>
    MEAN: typing.ClassVar[GroupPenaltyVariant]  # value = <GroupPenaltyVariant.MEAN: 0>
    __members__: typing.ClassVar[dict[str, GroupPenaltyVariant]]  # value = {'MEAN': <GroupPenaltyVariant.MEAN: 0>, 'CENTERING': <GroupPenaltyVariant.CENTERING: 1>}
    @typing.overload
    def __eq__(self, other: GroupPenaltyVariant) -> bool:
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
    def __ne__(self, other: GroupPenaltyVariant) -> bool:
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
    Column scaling convention applied by the solver after centering.
    
    Members:
    
      L2 : Divide each centered column by its L2 norm (unit L2 norm).
    
      ZSCORE : Divide each centered column by its sample SD (n-1).
    """
    L2: typing.ClassVar[ScalingMode]  # value = <ScalingMode.L2: 0>
    ZSCORE: typing.ClassVar[ScalingMode]  # value = <ScalingMode.ZSCORE: 1>
    __members__: typing.ClassVar[dict[str, ScalingMode]]  # value = {'L2': <ScalingMode.L2: 0>, 'ZSCORE': <ScalingMode.ZSCORE: 1>}
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
class SolverOptions:
    """
    Every T-solver input except the data X, D and y. Each solver reads the fields it needs and ignores the rest; the defaults are the solver-level defaults. Construct with keywords (`SolverOptions(lambda2=0.5, normalize=False)`) or set attributes.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        All defaults.
        """
    @typing.overload
    def __init__(self, **kwargs) -> None:
        """
        Keyword constructor: every keyword must name a SolverOptions field.
        """
    def __repr__(self) -> str:
        ...
    @property
    def cd_certified_localization(self) -> bool:
        """
        CD family: True (default) CERTIFIED crossing localization, False LEGACY.
        """
    @cd_certified_localization.setter
    def cd_certified_localization(self, arg0: bool) -> None:
        ...
    @property
    def cd_gram_cap(self) -> int:
        """
        CD family: ever-active Gram slots cached before falling back to naive sweeps. Default 400.
        """
    @cd_gram_cap.setter
    def cd_gram_cap(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_lambda_rel_tol(self) -> float:
        """
        CD family: relative lambda tolerance of the crossing search, in (0, 1). Default 1e-3.
        """
    @cd_lambda_rel_tol.setter
    def cd_lambda_rel_tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_max_sweeps(self) -> int:
        """
        CD family: maximum coordinate sweeps per fixed-lambda solve. Default 2000.
        """
    @cd_max_sweeps.setter
    def cd_max_sweeps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_tol_certify(self) -> float:
        """
        CD family: certification-sweep tolerance. Default 1e-10.
        """
    @cd_tol_certify.setter
    def cd_tol_certify(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cd_tol_probe(self) -> float:
        """
        CD family: probe-sweep tolerance. Default 1e-8.
        """
    @cd_tol_probe.setter
    def cd_tol_probe(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def exch_tie_alpha(self) -> float:
        """
        Exchangeable-tie band width for the greedy solvers (TOMP/TAFS families); 0 (default) = off.
        """
    @exch_tie_alpha.setter
    def exch_tie_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def exch_tie_floor(self) -> float:
        """
        Minimum |correlation| for exchangeable-tie candidates in [0, 1). Default 0.5.
        """
    @exch_tie_floor.setter
    def exch_tie_floor(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fold_dummies(self) -> bool:
        """
        TCDTIKNET dummy coupling: False (default) INDEPENDENT_RIDGE, True FOLDED.
        """
    @fold_dummies.setter
    def fold_dummies(self, arg0: bool) -> None:
        ...
    @property
    def group_penalty(self) -> GroupPenaltyVariant:
        """
        TCDIENET structured penalty (GroupPenaltyVariant). Default MEAN.
        """
    @group_penalty.setter
    def group_penalty(self, arg0: GroupPenaltyVariant) -> None:
        ...
    @property
    def intercept(self) -> bool:
        """
        Center X and D in place (a writer, as for normalize) and the solver's internal copy of y (an intercept is fitted). False when the caller pre-centered. Default True.
        """
    @intercept.setter
    def intercept(self, arg0: bool) -> None:
        ...
    @property
    def kappa_dummy(self) -> float:
        """
        Ridge weight of the dummy block (TCDENET; TCDTIKNET under INDEPENDENT_RIDGE). Default 1.0.
        """
    @kappa_dummy.setter
    def kappa_dummy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lambda2(self) -> float:
        """
        L2 (ridge / group-ridge / Tikhonov) penalty weight, >= 0; read by the elastic-net, IEN and Tikhonov solvers. Default 0 (no quadratic penalty).
        """
    @lambda2.setter
    def lambda2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_threads(self) -> int:
        """
        Thread budget of the solver's own passes over the design (correlation refreshes, step-length search, standardization). Default 1 (serial); a whole number of at least 1. A pass splits only above about a million entries and never inside a parallel region of the caller. The T-Rex selectors set it from max_inner_threads.
        """
    @n_threads.setter
    def n_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ncgmp_variant(self) -> omp_based.NCGMPVariant:
        """
        TNCGMP variant (NCGMPVariant). Default LineSearch.
        """
    @ncgmp_variant.setter
    def ncgmp_variant(self, arg0: omp_based.NCGMPVariant) -> None:
        ...
    @property
    def normalize(self) -> bool:
        """
        Scale each column of X (and D) to unit L2 norm (or unit SD, see scaling_mode), IN PLACE through the views (the solver becomes a writer and claims the buffers until restore() or destruction). False when the caller pre-scaled. Default True.
        """
    @normalize.setter
    def normalize(self, arg0: bool) -> None:
        ...
    @property
    def precentered(self) -> bool:
        """
        The caller centered X, D and y (intercept=False on centered data, as in the T-Rex selectors). The centered columns span at most n - 1 dimensions, so the solver counts the intercept as with intercept=True: the active set is capped at n - 1 and the degrees of freedom and Cp include the intercept. The data are not touched. Redundant when intercept is True. Default False.
        """
    @precentered.setter
    def precentered(self, arg0: bool) -> None:
        ...
    @property
    def rho_afs(self) -> float:
        """
        TAFS shrinkage step in (0, 1]; 1 (default) is forward stepwise.
        """
    @rho_afs.setter
    def rho_afs(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def scaling_mode(self) -> ScalingMode:
        """
        Column scaling applied when normalize is True: ScalingMode.L2 (default) or ZSCORE.
        """
    @scaling_mode.setter
    def scaling_mode(self, arg0: ScalingMode) -> None:
        ...
    @property
    def tenet_aug_use_lars(self) -> bool:
        """
        TENETAug / TIENETAug: pure-LARS inner path that never drops variables (True) instead of the LASSO inner path (False, default).
        """
    @tenet_aug_use_lars.setter
    def tenet_aug_use_lars(self, arg0: bool) -> None:
        ...
    @property
    def tie_seed(self) -> int:
        """
        Seed of the tie-break engine. Default 0: a fixed seed, so a solver run is reproducible also where exact ties occur; < 0 seeds from std::random_device (nondeterministic).
        """
    @tie_seed.setter
    def tie_seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tol(self) -> float:
        """
        Numerical tolerance of the solver's internal comparisons. Default: machine epsilon.
        """
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def verbose(self) -> bool:
        """
        Print status and diagnostics. Default False.
        """
    @verbose.setter
    def verbose(self, arg0: bool) -> None:
        ...
CENTERING: GroupPenaltyVariant  # value = <GroupPenaltyVariant.CENTERING: 1>
L2: ScalingMode  # value = <ScalingMode.L2: 0>
MEAN: GroupPenaltyVariant  # value = <GroupPenaltyVariant.MEAN: 0>
ZSCORE: ScalingMode  # value = <ScalingMode.ZSCORE: 1>
