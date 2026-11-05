"""
Tests that all public symbols declared in __all__ are importable and non-None.
"""
import importlib
import pytest


MODULES_TO_CHECK = [
    "trexselector",
    "trexselector.ml_methods",
    "trexselector.ml_methods.clustering",
    "trexselector.tsolvers",
    "trexselector.tsolvers.ad",
    "trexselector.tsolvers.ad.lars_based",
    "trexselector.tsolvers.ad.omp_based",
    "trexselector.tsolvers.ad.afs_based",
    "trexselector.tsolvers.ad.cd_based",
    "trexselector.trex_selector_methods",
    "trexselector.trex_selector_methods.ad",
    "trexselector.utils",
]


@pytest.mark.parametrize("module_name", MODULES_TO_CHECK)
def test_all_symbols_importable(module_name):
    mod = importlib.import_module(module_name)
    all_list = getattr(mod, "__all__", [])
    assert len(all_list) > 0, f"{module_name}.__all__ is empty"
    for name in all_list:
        obj = getattr(mod, name, None)
        assert obj is not None, f"{module_name}.{name} is None or missing"


def test_top_level_submodules_accessible():
    import trexselector
    assert trexselector.ml_methods is not None
    assert trexselector.tsolvers is not None
    assert trexselector.utils is not None


def test_top_level_classes_importable():
    from trexselector import (
        LLoopStrategy,
        SolverTypeForTRex,
        SolverHyperparameters,
        DummyDistribution,
        TRexControlParameter,
        SelectionResult,
        TRexSelector,
        TRexDASelector,
        TRexDAControlParameter,
        DAMethod,
        DASelectionResult,
        TRexGVSSelector,
        TRexGVSControlParameter,
        GVSType,
        GVSSelectionResult,
        TRexScreeningSelector,
        ScreenTRexControlParameter,
        ScreenTRexMethod,
        ScreenTRexSelectionResult,
        TRexBiobankScreeningSelector,
        BiobankScreenTRexControl,
        BiobankScreenTRexResult,
    )

    # Spot-check
    assert callable(TRexSelector)
    assert callable(TRexDASelector)
    assert callable(TRexGVSSelector)
    assert callable(TRexScreeningSelector)
    assert callable(TRexBiobankScreeningSelector)
    assert callable(DummyDistribution)
    assert callable(LLoopStrategy)
    assert callable(SolverTypeForTRex)
    assert callable(SolverHyperparameters)
    assert callable(TRexControlParameter)
    assert callable(SelectionResult)
    assert callable(TRexDAControlParameter)
    assert callable(DAMethod)
    assert callable(DASelectionResult)
    assert callable(TRexGVSControlParameter)
    assert callable(GVSType)
    assert callable(GVSSelectionResult)
    assert callable(ScreenTRexControlParameter)
    assert callable(ScreenTRexMethod)
    assert callable(ScreenTRexSelectionResult)
    assert callable(BiobankScreenTRexControl)
    assert callable(BiobankScreenTRexResult)


def test_utils_symbols_importable():
    from trexselector.utils import (
        AccessMode,
        MemoryMappedMatrix,
        numpy_to_memmap,
        convert_to_memory_mapped,
        compute_fdp,
        compute_tpp,
        compute_precision,
        compute_recall,
        compute_fdp_dense,
        compute_tpp_dense,
        true_positives_count,
        false_positives_count,
        false_negatives_count,
        f1_score,
        get_max_threads,
        get_num_threads,
        set_num_threads,
    )
    assert callable(compute_fdp)
    assert callable(compute_tpp)
    assert callable(compute_precision)
    assert callable(compute_recall)
    assert callable(compute_fdp_dense)
    assert callable(compute_tpp_dense)
    assert callable(true_positives_count)
    assert callable(false_positives_count)
    assert callable(false_negatives_count)
    assert callable(f1_score)
    assert callable(get_max_threads)
    assert callable(get_num_threads)
    assert callable(set_num_threads)
    assert callable(numpy_to_memmap)
    assert callable(convert_to_memory_mapped)
    assert callable(MemoryMappedMatrix)
    assert AccessMode.ReadOnly is not None
    assert AccessMode.ReadWrite is not None


def test_tsolver_symbols_importable():
    from trexselector.tsolvers.ad import ScalingMode
    from trexselector.tsolvers.ad.lars_based import (
        TLARS_Solver,
        TLASSO_Solver,
        TSTEPWISE_Solver,
        TSTAGEWISE_Solver,
        TENET_Solver,
        TENETAug_Solver,
        TIENETAug_Solver,
    )
    from trexselector.tsolvers.ad.omp_based import (
        TOMP_Solver,
        TGP_Solver,
        TACGP_Solver,
        TMP_Solver,
        TOOLS_Solver,
        TNCGMP_Solver,
        NCGMPVariant,
    )
    from trexselector.tsolvers.ad.afs_based import TAFS_Solver

    for cls in (TLARS_Solver, TLASSO_Solver, TSTEPWISE_Solver, TSTAGEWISE_Solver,
                TENET_Solver, TENETAug_Solver, TIENETAug_Solver, TOMP_Solver,
                TGP_Solver, TACGP_Solver, TMP_Solver, TOOLS_Solver,
                TNCGMP_Solver, TAFS_Solver):
        assert callable(cls)
    assert ScalingMode.L2 is not None
    assert ScalingMode.ZSCORE is not None
    from trexselector.tsolvers.ad import SolverOptions, GroupPenaltyVariant
    from trexselector.tsolvers.ad.cd_based import (
        TCDLASSO_Solver, TCDENET_Solver, TCDIENET_Solver, TCDTIKNET_Solver)
    from trexselector.tsolvers.ad.lars_based import TIENET_Solver, TIKNET_Solver
    for cls in (TCDLASSO_Solver, TCDENET_Solver, TCDIENET_Solver, TCDTIKNET_Solver,
                TIENET_Solver, TIKNET_Solver):
        assert callable(cls)
    assert SolverOptions().lambda2 == 0.0
    assert GroupPenaltyVariant.MEAN is not None
    assert NCGMPVariant.LineSearch is not None


def test_ml_methods_symbols_importable():
    from trexselector.ml_methods import (
        ZScoreScaler,
        LpNormScaler,
        NormType,
    )
    assert callable(ZScoreScaler)
    assert callable(LpNormScaler)
    assert NormType.L1 is not None
    assert NormType.L2 is not None


def test_clustering_symbols_importable():
    from trexselector.ml_methods.clustering import (
        LinkageMethod,
        DistanceMetric,
        agglomerative_cluster,
        cut_tree,
    )
    assert callable(agglomerative_cluster)
    assert callable(cut_tree)
    assert LinkageMethod.Ward is not None
    assert DistanceMetric.Euclidean is not None


def test_dummy_mechanism_levels():
    """One subpackage per dummy mechanism, mirroring cpp/src/{tsolvers,trex_selector_methods}/ad."""
    import trexselector
    from trexselector.tsolvers import ad as tsolvers_ad
    from trexselector.trex_selector_methods import ad as selectors_ad
    assert trexselector.tsolvers.__all__ == ["ad"]
    assert trexselector.trex_selector_methods.__all__ == ["ad"]
    assert tsolvers_ad.lars_based.TLARS_Solver is not None
    assert tsolvers_ad.SolverOptions is not None
    # the root re-exports the AD selectors as the everyday spelling
    assert trexselector.TRexControlParameter is selectors_ad.TRexControlParameter
    assert trexselector.SolverTypeForTRex is selectors_ad.SolverTypeForTRex
    # the compiled extension carries the same level
    from trexselector._core.tsolvers.ad.cd_based import TCDTIKNET_Solver
    from trexselector._core.trex_selector_methods.ad import TRexTikhonovSelector
    assert callable(TCDTIKNET_Solver) and callable(TRexTikhonovSelector)


def test_tikhonov_symbols_importable():
    from trexselector import (
        TRexTikhonovSelector,
        TRexTikhonovControlParameter,
        TikLambda2Method,
        PilotStatistic,
        TikPenaltyStructure,
        Lambda2PickInfo,
        Lambda2PilotResult,
        Lambda2CVDiagnostics,
        gamma_to_k,
        build_group_centering_k,
    )
    for obj in (TRexTikhonovSelector, TRexTikhonovControlParameter, TikLambda2Method,
                PilotStatistic, TikPenaltyStructure, Lambda2PickInfo, Lambda2PilotResult,
                Lambda2CVDiagnostics, gamma_to_k, build_group_centering_k):
        assert callable(obj)
    assert TikLambda2Method.CV_1SE_IEN_CCD is not None
    assert TikPenaltyStructure.GROUP_CENTERING is not None
