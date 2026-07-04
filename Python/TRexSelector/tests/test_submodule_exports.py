"""
Tests that all public symbols declared in __all__ are importable and non-None.
"""
import importlib
import pytest


MODULES_TO_CHECK = [
    "trex_selector",
    "trex_selector.ml_methods",
    "trex_selector.ml_methods.clustering",
    "trex_selector.tsolvers",
    "trex_selector.tsolvers.lars_based",
    "trex_selector.tsolvers.omp_based",
    "trex_selector.tsolvers.afs_based",
    "trex_selector.utils",
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
    import trex_selector
    assert trex_selector.ml_methods is not None
    assert trex_selector.tsolvers is not None
    assert trex_selector.utils is not None


def test_top_level_classes_importable():
    from trex_selector import (
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
    from trex_selector.utils import (
        AccessMode,
        MemoryMappedMatrix,
        numpy_to_memmap,
        compute_fdp,
        compute_tpp,
        compute_precision,
        compute_recall,
        compute_fdp_dense,
        compute_tpp_dense,
        get_max_threads,
        set_num_threads,
    )
    assert callable(compute_fdp)
    assert callable(compute_tpp)
    assert callable(compute_precision)
    assert callable(compute_recall)
    assert callable(compute_fdp_dense)
    assert callable(compute_tpp_dense)
    assert callable(get_max_threads)
    assert callable(set_num_threads)
    assert callable(numpy_to_memmap)
    assert callable(MemoryMappedMatrix)
    assert AccessMode.ReadOnly is not None
    assert AccessMode.ReadWrite is not None


def test_ml_methods_symbols_importable():
    from trex_selector.ml_methods import (
        ZScoreScaler,
        LpNormScaler,
        NormType,
    )
    assert callable(ZScoreScaler)
    assert callable(LpNormScaler)
    assert NormType.L1 is not None
    assert NormType.L2 is not None


def test_clustering_symbols_importable():
    from trex_selector.ml_methods.clustering import (
        LinkageMethod,
        DistanceMetric,
        agglomerative_cluster,
        cut_tree,
    )
    assert callable(agglomerative_cluster)
    assert callable(cut_tree)
    assert LinkageMethod.Ward is not None
    assert DistanceMetric.Euclidean is not None
