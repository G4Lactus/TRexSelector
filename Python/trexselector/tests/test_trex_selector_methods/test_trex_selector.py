"""
Tests for TRexSelector: properties, type checks, and consistency.
"""
import numpy as np
import pytest
import trexselector
from trexselector import DummyDistribution, SolverTypeForTRex, SolverHyperparameters


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    assert sel is not None


def test_dimension_mismatch_raises(signal_data, fast_control):
    X, y, n, p = signal_data
    y_bad = np.random.randn(n + 10)
    with pytest.raises(Exception):
        trexselector.TRexSelector(X, y_bad, trex_control=fast_control, verbose=False)


# ---------------------------------------------------------------------------
# Post-selection property types and shapes
# ---------------------------------------------------------------------------

def test_properties_after_selection(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
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
    sel = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
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
    ctrl = trexselector.TRexControlParameter()

    # Threading
    ctrl.max_outer_threads = 2
    ctrl.max_inner_threads = 2
    assert ctrl.max_outer_threads == 2
    assert ctrl.max_inner_threads == 2

    # Memory mapping
    ctrl.use_memory_mapping = False
    assert ctrl.use_memory_mapping is False

    # Solver type
    ctrl.solver_type = trexselector.SolverTypeForTRex.TLASSO
    assert ctrl.solver_type == trexselector.SolverTypeForTRex.TLASSO

    # Solver hyperparameters
    hp = trexselector.SolverHyperparameters()
    hp.lambda2 = 0.05
    hp.tol = 1e-8
    ctrl.solver_params = hp
    assert ctrl.solver_params.lambda2 == pytest.approx(0.05)

    # Dummy distribution
    ctrl.dummy_distribution = trexselector.DummyDistribution.student_t(df=5.0)
    assert ctrl.dummy_distribution.get_type() == trexselector.DummyDistribution.Type.StudentT

    # Scaling mode
    ctrl.scaling_mode = trexselector.ScalingMode.ZSCORE
    assert ctrl.scaling_mode == trexselector.ScalingMode.ZSCORE


def test_trex_control_with_tlasso_solver(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = trexselector.SolverTypeForTRex.TLASSO
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# SelectionResult fields
# ---------------------------------------------------------------------------

def test_selection_result_fields(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
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
    trexselector.LLoopStrategy.SKIPL,
    trexselector.LLoopStrategy.STANDARD,
    trexselector.LLoopStrategy.HCONCAT,
    trexselector.LLoopStrategy.PERMUTATION,
    trexselector.LLoopStrategy.PERMUTATION_SEEDED,
    trexselector.LLoopStrategy.SEEDED,
])
def test_lloop_strategy_variants(signal_data, strategy):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.lloop_strategy = strategy
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


@pytest.mark.parametrize("strategy,multiplier,stop,expected", [
    (trexselector.LLoopStrategy.SKIPL, 10, None, True),
    (trexselector.LLoopStrategy.SKIPL, 5, None, False),
    (trexselector.LLoopStrategy.SKIPL, 20, True, False),
    (trexselector.LLoopStrategy.STANDARD, 20, None, False),
])
def test_skipl_large_budget_without_stop_warns(signal_data, capfd, strategy, multiplier, stop, expected):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = SolverTypeForTRex.TLARS
    ctrl.lloop_strategy = strategy
    ctrl.max_dummy_multiplier = multiplier
    ctrl.tloop_stagnation_stop = stop
    trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    assert ("lloop_strategy SKIPL" in capfd.readouterr().err) == expected


# ---------------------------------------------------------------------------
# DummyDistribution integration
# ---------------------------------------------------------------------------

def test_dummy_rademacher_runs(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.dummy_distribution = trexselector.DummyDistribution.rademacher()
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


def test_dummy_uniform_runs(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.dummy_distribution = trexselector.DummyDistribution.uniform(a=1.5)
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# SolverTypeForTRex variants (TOMP)
# ---------------------------------------------------------------------------

def test_solver_type_tomp(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = trexselector.SolverTypeForTRex.TOMP
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
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


# ---------------------------------------------------------------------------
# CCD family + construction gates (bindings-parity additions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["TCDLASSO", "TCDENET", "TENET_AUG"])
def test_ccd_and_aug_solvers_run(signal_data, solver):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = getattr(trexselector.SolverTypeForTRex, solver)
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


def test_cd_knobs_and_tenet_aug_use_lars_bound():
    hp = trexselector.SolverHyperparameters()
    assert hp.cd_lambda_rel_tol > 0
    assert hp.cd_tol_certify < 0 and hp.cd_tol_probe < 0
    assert hp.cd_gram_cap == 0 and hp.cd_max_sweeps == 0
    assert hp.tenet_aug_use_lars is False
    hp.tenet_aug_use_lars = True
    assert hp.tenet_aug_use_lars is True
    assert hp.cd_certified_localization is True
    hp.cd_certified_localization = False
    assert hp.cd_certified_localization is False


@pytest.mark.parametrize("solver", ["TIENET", "TIENET_AUG", "TCDIENET"])
def test_ien_family_gated_at_construction(signal_data, solver):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = getattr(trexselector.SolverTypeForTRex, solver)
    with pytest.raises(ValueError, match="GVS|group assignment"):
        trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)


@pytest.mark.parametrize("solver", ["TCDTIKNET", "TIKNET"])
def test_tikhonov_family_gated_at_construction(signal_data, solver):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = getattr(trexselector.SolverTypeForTRex, solver)
    with pytest.raises(ValueError, match="Tikhonov"):
        trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)


@pytest.mark.parametrize("certified", [True, False])
def test_cd_localization_modes_run(signal_data, certified):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = trexselector.SolverTypeForTRex.TCDLASSO
    ctrl.solver_params.cd_certified_localization = certified
    sel = trexselector.TRexSelector(X, y, trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    assert res.selected_var.shape[0] == p


# ---------------------------------------------------------------------------
# ScalingMode.NONE (user-managed design) and input rejection
# ---------------------------------------------------------------------------

def _unit_l2_columns(X):
    return X / np.linalg.norm(X, axis=0, keepdims=True)


def _planted_design(seed=5, n=300, p=60):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 5.0 * X[:, 1] - 3.0 * X[:, 2] + 2.0 * X[:, 3] + rng.standard_normal(n)
    return X, y


def _control(mode, K=20):
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = K
    ctrl.scaling_mode = mode
    return ctrl


def test_scaling_mode_none_is_exposed():
    assert trexselector.ScalingMode.NONE is not None
    ctrl = trexselector.TRexControlParameter()
    assert ctrl.scaling_mode == trexselector.ScalingMode.L2  # default unchanged
    ctrl.scaling_mode = trexselector.ScalingMode.NONE
    assert ctrl.scaling_mode == trexselector.ScalingMode.NONE


def test_normalize_design_reproduces_the_selector_preprocessing():
    X, y = _planted_design(n=40, p=6)
    X0, y0 = X.copy(), y.copy()

    d = trexselector.normalize_design(X, y)  # L2 default
    np.testing.assert_array_equal(X, X0)  # inputs untouched
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_allclose(d["X"].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(d["X"], axis=0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(d["X_means"], X.mean(axis=0))
    np.testing.assert_allclose(d["X_scales"], np.linalg.norm(X - X.mean(axis=0), axis=0))
    assert d["y_mean"] == pytest.approx(y.mean())
    np.testing.assert_allclose(d["y"], y - y.mean())

    z = trexselector.normalize_design(X, y, trexselector.ScalingMode.ZSCORE)
    np.testing.assert_allclose(np.linalg.norm(z["X"], axis=0), np.sqrt(39.0), rtol=1e-12)
    np.testing.assert_allclose(z["X_scales"], X.std(axis=0, ddof=1))

    none = trexselector.normalize_design(X, y, trexselector.ScalingMode.NONE)
    np.testing.assert_array_equal(none["X"], X0)
    np.testing.assert_array_equal(none["y"], y0)
    assert np.all(none["X_means"] == 0.0)
    assert np.all(none["X_scales"] == 1.0)
    assert none["y_mean"] == 0.0


def test_none_on_prenormalized_design_equals_l2_on_raw_design():
    X, y = _planted_design()

    sel_l2 = trexselector.TRexSelector(
        X.copy(), y.copy(), tFDR=0.2,
        trex_control=_control(trexselector.ScalingMode.L2), seed=11, verbose=False)
    r_l2 = sel_l2.select()

    d = trexselector.normalize_design(X, y)
    sel_none = trexselector.TRexSelector(
        d["X"], d["y"], tFDR=0.2,
        trex_control=_control(trexselector.ScalingMode.NONE), seed=11, verbose=False)
    r_none = sel_none.select()

    assert r_l2.T_stop == r_none.T_stop
    np.testing.assert_array_equal(r_l2.selected_var, r_none.selected_var)
    np.testing.assert_array_equal(r_l2.phi_prime, r_none.phi_prime)  # bitwise
    selected = np.flatnonzero(r_l2.selected_var)
    assert selected.size > 0
    assert set(selected).issubset({1, 2, 3})


def test_none_leaves_x_untouched_and_reports_identity_parameters():
    X, y = _planted_design(n=80, p=20)
    X = np.asfortranarray(_unit_l2_columns(X))
    X0, y0 = X.copy(), y.copy()

    sel = trexselector.TRexSelector(
        X, y, tFDR=0.2, trex_control=_control(trexselector.ScalingMode.NONE, K=5),
        seed=3, verbose=False)
    np.testing.assert_array_equal(sel.X, X0)  # the wrapper's buffer is the C++ buffer
    sel.select()
    np.testing.assert_array_equal(sel.X, X0)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    assert np.all(sel._selector.getXMeans() == 0.0)
    assert np.all(sel._selector.getXL2Norms() == 1.0)
    assert sel._selector.getYMean() == 0.0


@pytest.mark.parametrize("mode", ["L2", "ZSCORE", "NONE"])
def test_non_finite_input_is_rejected_in_every_mode(mode):
    X, y = _planted_design(n=40, p=8)
    X = _unit_l2_columns(X)
    scaling_mode = getattr(trexselector.ScalingMode, mode)

    Xn = X.copy()
    Xn[2, 3] = np.nan
    with pytest.raises(ValueError, match=r"X\(2, 3\)"):
        trexselector.TRexSelector(Xn, y, trex_control=_control(scaling_mode, K=3),
                                       verbose=False)
    with pytest.raises(ValueError, match=r"X\(2, 3\)"):
        trexselector.normalize_design(Xn, y, scaling_mode)

    yi = y.copy()
    yi[1] = np.inf
    # The selector's parameter validation catches y first (its own message);
    # the normalizer names the entry when called directly.
    with pytest.raises(ValueError, match="NaN or Inf"):
        trexselector.TRexSelector(X, yi, trex_control=_control(scaling_mode, K=3),
                                       verbose=False)
    with pytest.raises(ValueError, match=r"y\(1\)"):
        trexselector.normalize_design(X, yi, scaling_mode)


def test_constant_column_is_rejected_under_l2_and_a_constant_atom_passes_under_none():
    X, y = _planted_design(n=40, p=8)
    X[:, 4] = 3.5
    with pytest.raises(ValueError, match="column 4.*constant"):
        trexselector.TRexSelector(X, y, trex_control=_control(trexselector.ScalingMode.L2, K=3),
                                       verbose=False)
    with pytest.raises(ValueError, match="column 4"):
        trexselector.normalize_design(X, y, trexselector.ScalingMode.ZSCORE)

    # Intercept-free dictionary: a unit-L2 constant (DC) atom is legitimate
    # under NONE, and an all-zero column is not.
    Xd = _unit_l2_columns(_planted_design(n=40, p=8)[0])
    Xd[:, 4] = 1.0 / np.sqrt(40.0)
    sel = trexselector.TRexSelector(Xd, y, trex_control=_control(trexselector.ScalingMode.NONE, K=3),
                                         verbose=False)
    assert sel.select() is not None
    Xz = Xd.copy()
    Xz[:, 4] = 0.0
    with pytest.raises(ValueError, match="column 4"):
        trexselector.TRexSelector(Xz, y, trex_control=_control(trexselector.ScalingMode.NONE, K=3),
                                       verbose=False)


# ---------------------------------------------------------------------------
# restore(): give X back before select()
# ---------------------------------------------------------------------------

def test_restore_before_select_gives_x_back(fast_control):
    rng = np.random.default_rng(77)
    n, p = 50, 30
    X = np.asfortranarray(rng.standard_normal((n, p)))   # shared, not copied
    y = X[:, 0] * 2.0 + rng.standard_normal(n)
    X0 = X.copy(order="F")
    sel = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    assert not np.allclose(X, X0)                        # normalized in place at construction
    sel.restore()
    assert np.allclose(X, X0, atol=1e-12)
    sel.restore()                                        # idempotent
    assert np.allclose(X, X0, atol=1e-12)
    with pytest.raises(RuntimeError, match="restore"):
        sel.select()
    sel2 = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    sel2.select()                                        # a fresh selector runs and restores X itself
    assert np.allclose(X, X0, atol=1e-12)
    sel2.restore()                                       # no-op after select()
    assert np.allclose(X, X0, atol=1e-12)


def test_num_dummies_is_exposed(fast_control):
    rng = np.random.default_rng(78)
    n, p = 40, 20
    X = np.asfortranarray(rng.standard_normal((n, p)))
    y = X[:, 0] * 2.0 + rng.standard_normal(n)
    sel = trexselector.TRexSelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()
    assert sel.num_dummies >= p
