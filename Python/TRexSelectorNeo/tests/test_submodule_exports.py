"""
Tests that all public symbols declared in __all__ are importable and non-None.
"""
import importlib
import pytest


MODULES_TO_CHECK = [
    "trex_selector_neo",
    "trex_selector_neo.ml_methods",
    "trex_selector_neo.ml_methods.clustering",
    "trex_selector_neo.tsolvers",
    "trex_selector_neo.tsolvers.lars_based",
    "trex_selector_neo.tsolvers.omp_based",
    "trex_selector_neo.tsolvers.afs_based",
    "trex_selector_neo.utils",
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
    import trex_selector_neo
    assert trex_selector_neo.ml_methods is not None
    assert trex_selector_neo.tsolvers is not None
    assert trex_selector_neo.utils is not None


def test_top_level_classes_importable():
    from trex_selector_neo import (
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
    from trex_selector_neo.utils import (
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
    assert callable(set_num_threads)
    assert callable(numpy_to_memmap)
    assert callable(convert_to_memory_mapped)
    assert callable(MemoryMappedMatrix)
    assert AccessMode.ReadOnly is not None
    assert AccessMode.ReadWrite is not None


def test_tsolver_symbols_importable():
    from trex_selector_neo.tsolvers import ScalingMode
    from trex_selector_neo.tsolvers.lars_based import (
        TLARS_Solver,
        TLASSO_Solver,
        TSTEPWISE_Solver,
        TSTAGEWISE_Solver,
        TENET_Solver,
        TENETAug_Solver,
        TIENETAug_Solver,
    )
    from trex_selector_neo.tsolvers.omp_based import (
        TOMP_Solver,
        TGP_Solver,
        TACGP_Solver,
        TMP_Solver,
        TOOLS_Solver,
        TNCGMP_Solver,
        NCGMPVariant,
    )
    from trex_selector_neo.tsolvers.afs_based import TAFS_Solver

    for cls in (TLARS_Solver, TLASSO_Solver, TSTEPWISE_Solver, TSTAGEWISE_Solver,
                TENET_Solver, TENETAug_Solver, TIENETAug_Solver, TOMP_Solver,
                TGP_Solver, TACGP_Solver, TMP_Solver, TOOLS_Solver,
                TNCGMP_Solver, TAFS_Solver):
        assert callable(cls)
    assert ScalingMode.L2 is not None
    assert ScalingMode.ZSCORE is not None
    assert NCGMPVariant.LineSearch is not None


def test_ml_methods_symbols_importable():
    from trex_selector_neo.ml_methods import (
        ZScoreScaler,
        LpNormScaler,
        NormType,
    )
    assert callable(ZScoreScaler)
    assert callable(LpNormScaler)
    assert NormType.L1 is not None
    assert NormType.L2 is not None


def test_clustering_symbols_importable():
    from trex_selector_neo.ml_methods.clustering import (
        LinkageMethod,
        DistanceMetric,
        agglomerative_cluster,
        cut_tree,
    )
    assert callable(agglomerative_cluster)
    assert callable(cut_tree)
    assert LinkageMethod.Ward is not None
    assert DistanceMetric.Euclidean is not None
