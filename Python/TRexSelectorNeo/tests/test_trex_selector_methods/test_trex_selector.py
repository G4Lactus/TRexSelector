"""
Tests for TRexSelector: properties, type checks, and consistency.
"""
import numpy as np
import pytest
import trex_selector_neo
from trex_selector_neo import DummyDistribution, SolverTypeForTRex, SolverHyperparameters


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    assert sel is not None


def test_dimension_mismatch_raises(signal_data, fast_control):
    X, y, n, p = signal_data
    y_bad = np.random.randn(n + 10)
    with pytest.raises(Exception):
        trex_selector_neo.TRexSelector(X, y_bad, trex_control=fast_control, verbose=False)


# ---------------------------------------------------------------------------
# Post-selection property types and shapes
# ---------------------------------------------------------------------------

def test_properties_after_selection(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()

    # Scalar properties
    assert isinstance(sel.tFDR, float)
    assert isinstance(sel.v_thresh, float)
    assert isinstance(sel.y_mean, float)
    assert isinstance(sel.K, int)
    assert isinstance(sel.T_stop, int)
    assert isinstance(sel.L, int)
    assert isinstance(sel.num_selected, int)

    # Vector properties
    assert isinstance(sel.selected_var, np.ndarray)
    assert sel.selected_var.shape == (p,)
    assert isinstance(sel.selected_indices, list)

    assert isinstance(sel.x_means, np.ndarray)
    assert sel.x_means.shape == (p,)
    assert isinstance(sel.x_l2_norms, np.ndarray)
    assert sel.x_l2_norms.shape == (p,)

    assert isinstance(sel.voting_grid, np.ndarray)
    assert isinstance(sel.phi_prime, np.ndarray)
    assert sel.phi_prime.shape == (p,)

    # Matrix properties
    assert isinstance(sel.fdp_hat_mat, np.ndarray)
    assert sel.fdp_hat_mat.ndim == 2
    assert isinstance(sel.phi_mat, np.ndarray)
    assert sel.phi_mat.ndim == 2
    assert isinstance(sel.r_mat, np.ndarray)
    assert sel.r_mat.ndim == 2


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def test_selected_var_indices_consistent(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()

    sv = sel.selected_var
    idx = sel.selected_indices

    assert len(sv) == p
    assert all(v in (0, 1) for v in sv)
    assert sel.num_selected == int(sv.sum())

    # Indices are 0-based and within [0, p)
    if idx:
        assert all(0 <= i < p for i in idx)

    # Consistency: selected_var[i] == 1 iff i in selected_indices
    for i in idx:
        assert sv[i] == 1


# ---------------------------------------------------------------------------
# TRexControlParameter new fields
# ---------------------------------------------------------------------------

def test_trex_control_new_fields():
    ctrl = trex_selector_neo.TRexControlParameter()

    # Threading
    ctrl.max_outer_threads = 2
    ctrl.max_inner_threads = 2
    assert ctrl.max_outer_threads == 2
    assert ctrl.max_inner_threads == 2

    # Memory mapping
    ctrl.use_memory_mapping = False
    assert ctrl.use_memory_mapping is False

    # Solver type
    ctrl.solver_type = trex_selector_neo.SolverTypeForTRex.TLASSO
    assert ctrl.solver_type == trex_selector_neo.SolverTypeForTRex.TLASSO

    # Solver hyperparameters
    hp = trex_selector_neo.SolverHyperparameters()
    hp.lambda2 = 0.05
    hp.tol = 1e-8
    ctrl.solver_params = hp
    assert ctrl.solver_params.lambda2 == pytest.approx(0.05)

    # Dummy distribution
    ctrl.dummy_distribution = trex_selector_neo.DummyDistribution.student_t(df=5.0)
    assert ctrl.dummy_distribution.get_type() == trex_selector_neo.DummyDistribution.Type.StudentT

    # Scaling mode
    ctrl.scaling_mode = trex_selector_neo.ScalingMode.ZSCORE
    assert ctrl.scaling_mode == trex_selector_neo.ScalingMode.ZSCORE


def test_trex_control_with_tlasso_solver(signal_data):
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = trex_selector_neo.SolverTypeForTRex.TLASSO
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# SelectionResult fields
# ---------------------------------------------------------------------------

def test_selection_result_fields(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    res = sel.select()

    assert isinstance(res.selected_var, np.ndarray)
    assert isinstance(res.v_thresh, float)
    assert isinstance(res.T_stop, int)
    assert isinstance(res.num_dummies, int)
    assert isinstance(res.dummy_factor_L, int)
    assert isinstance(res.fdp_hat_mat, np.ndarray)
    assert isinstance(res.phi_mat, np.ndarray)
    assert isinstance(res.phi_prime, np.ndarray)
    assert isinstance(res.r_mat, np.ndarray)
    assert isinstance(res.voting_grid, np.ndarray)


# ---------------------------------------------------------------------------
# LLoopStrategy variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    trex_selector_neo.LLoopStrategy.SKIPL,
    trex_selector_neo.LLoopStrategy.STANDARD,
    trex_selector_neo.LLoopStrategy.HCONCAT,
    trex_selector_neo.LLoopStrategy.PERMUTATION,
    trex_selector_neo.LLoopStrategy.PERMUTATION_ONDEMAND,
    trex_selector_neo.LLoopStrategy.ONDEMAND,
])
def test_lloop_strategy_variants(signal_data, strategy):
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    ctrl.lloop_strategy = strategy
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# DummyDistribution integration
# ---------------------------------------------------------------------------

def test_dummy_rademacher_runs(signal_data):
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    ctrl.dummy_distribution = trex_selector_neo.DummyDistribution.rademacher()
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


def test_dummy_uniform_runs(signal_data):
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    ctrl.dummy_distribution = trex_selector_neo.DummyDistribution.uniform(a=1.5)
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# SolverTypeForTRex variants (TOMP)
# ---------------------------------------------------------------------------

def test_solver_type_tomp(signal_data):
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = trex_selector_neo.SolverTypeForTRex.TOMP
    sel = trex_selector_neo.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# DummyDistribution — factory methods and parameter assertions
# ---------------------------------------------------------------------------

class TestDummyDistributionFactories:
    def test_normal(self):
        d = DummyDistribution.normal()
        assert d.get_type() == DummyDistribution.Type.Normal

    def test_rademacher(self):
        d = DummyDistribution.rademacher()
        assert d.get_type() == DummyDistribution.Type.Rademacher

    def test_mammen_default(self):
        d = DummyDistribution.mammen()
        assert d.get_type() == DummyDistribution.Type.Mammen

    def test_mammen_custom(self):
        d = DummyDistribution.mammen(p1=0.7, p2=0.3)
        assert d.get_type() == DummyDistribution.Type.Mammen
        assert d.mammen_param_p1 == pytest.approx(0.7)
        assert d.mammen_param_p2 == pytest.approx(0.3)

    def test_uniform(self):
        d = DummyDistribution.uniform(a=2.0)
        assert d.get_type() == DummyDistribution.Type.Uniform
        assert d.uniform_a == pytest.approx(2.0)

    def test_student_t(self):
        d = DummyDistribution.student_t(df=10.0)
        assert d.get_type() == DummyDistribution.Type.StudentT
        assert d.student_t_df == pytest.approx(10.0)

    def test_laplace(self):
        d = DummyDistribution.laplace(location=0.0, scale=0.5)
        assert d.get_type() == DummyDistribution.Type.Laplace
        assert d.laplace_location == pytest.approx(0.0)
        assert d.laplace_scale == pytest.approx(0.5)

    def test_gumbel(self):
        d = DummyDistribution.gumbel(location=1.0, scale=2.0)
        assert d.get_type() == DummyDistribution.Type.Gumbel
        assert d.gumbel_location == pytest.approx(1.0)
        assert d.gumbel_scale == pytest.approx(2.0)

    def test_triangle(self):
        d = DummyDistribution.triangle(a=-1.0, b=0.0, c=1.0)
        assert d.get_type() == DummyDistribution.Type.Triangle
        assert d.triangle_a == pytest.approx(-1.0)
        assert d.triangle_b == pytest.approx(0.0)
        assert d.triangle_c == pytest.approx(1.0)

    def test_uniform_sphere(self):
        d = DummyDistribution.uniform_sphere(dim=4)
        assert d.get_type() == DummyDistribution.Type.UniformSphere
        assert d.sphere_dim == 4

    def test_constrained_sparse_rademacher(self):
        d = DummyDistribution.constrained_sparse_rademacher(s=0.5)
        assert d.get_type() == DummyDistribution.Type.ConstrainedSparseRademacher
        assert d.constrained_sparse_rademacher_s == pytest.approx(0.5)

    def test_logistic(self):
        d = DummyDistribution.logistic(location=0.0, scale=0.55)
        assert d.get_type() == DummyDistribution.Type.Logistic
        assert d.logistic_location == pytest.approx(0.0)
        assert d.logistic_scale == pytest.approx(0.55)

    def test_default_constructor_is_normal(self):
        d = DummyDistribution()
        assert d.get_type() == DummyDistribution.Type.Normal


class TestDummyDistributionWritableFields:
    def test_student_t_df_writable(self):
        d = DummyDistribution.student_t(df=5.0)
        d.student_t_df = 10.0
        assert d.student_t_df == pytest.approx(10.0)

    def test_laplace_scale_writable(self):
        d = DummyDistribution.laplace()
        d.laplace_scale = 2.0
        assert d.laplace_scale == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SolverTypeForTRex — all enum values accessible
# ---------------------------------------------------------------------------

class TestSolverTypeForTRex:
    _all_values = [
        "TLARS", "TLASSO", "TENET", "TSTEPWISE", "TSTAGEWISE",
        "TOMP", "TGP", "TACGP", "TMP", "TNCGMP", "TOOLS", "TAFS",
    ]

    @pytest.mark.parametrize("name", _all_values)
    def test_enum_accessible(self, name):
        val = getattr(SolverTypeForTRex, name)
        assert val is not None

    def test_enum_inequality(self):
        assert SolverTypeForTRex.TLARS != SolverTypeForTRex.TLASSO


# ---------------------------------------------------------------------------
# SolverHyperparameters
# ---------------------------------------------------------------------------

class TestSolverHyperparameters:
    def test_default_constructor(self):
        hp = SolverHyperparameters()
        assert hp is not None

    def test_field_assignment(self):
        hp = SolverHyperparameters()
        hp.lambda2 = 0.01
        hp.rho_afs = 0.5
        hp.ncgmp_variant = 1
        hp.tol = 1e-6
        assert hp.lambda2 == pytest.approx(0.01)
        assert hp.rho_afs == pytest.approx(0.5)
        assert hp.ncgmp_variant == 1
        assert hp.tol == pytest.approx(1e-6)

    def test_exchangeable_tie_fields(self):
        hp = SolverHyperparameters()
        # Defaults: off, floor 0.5.
        assert hp.exch_tie_alpha == pytest.approx(0.0)
        assert hp.exch_tie_floor == pytest.approx(0.5)
        hp.exch_tie_alpha = 0.25
        hp.exch_tie_floor = 0.7
        assert hp.exch_tie_alpha == pytest.approx(0.25)
        assert hp.exch_tie_floor == pytest.approx(0.7)
