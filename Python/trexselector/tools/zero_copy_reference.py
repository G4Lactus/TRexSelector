#!/usr/bin/env python
"""Bit-identity harness of the Python ML surface.

    python tools/zero_copy_reference.py record  <dir>   # on the current tree
    python tools/zero_copy_reference.py compare <dir>   # after a change

Every ML class is run on fixed synthetic data of three shapes and every
output is saved as a .npz; compare requires bit equality (np.array_equal on
float64 views), so a change at the binding boundary or a copy fusion must
reproduce the recording exactly. The reference is platform-specific; keep it
out of git.
"""

import sys
from pathlib import Path

import numpy as np

from trexselector.ml_methods import (ElasticNet, ElasticNetCV, LpNormScaler,
                                          NormType, PCA, RidgeCV, RidgeSolver,
                                          SVDSolver, ZScoreScaler)
from trexselector.ml_methods.clustering import (DistanceMetric, LinkageMethod,
                                                     agglomerative_cluster, cut_tree)

SHAPES = {"tall": (400, 30), "wide": (60, 900), "square": (120, 120)}


def make_data(n, p):
    rng = np.random.default_rng(1000 + p)
    X = np.asfortranarray(rng.standard_normal((n, p)))
    beta = np.zeros(p)
    beta[0] = 2.0
    beta[3] = -1.5
    beta[min(7, p - 1)] = 1.0
    rng2 = np.random.default_rng(77 + n)
    y = X @ beta + 0.5 * rng2.standard_normal(n)
    return X, y


def run_case(n, p):
    X, y = make_data(n, p)
    X0 = X.copy(order="F")
    out = {}

    cv = RidgeCV()
    cv.fit(X, y, 5, 60, 1000.0, 11)
    out.update(rcv_lambdas=cv.lambdas(), rcv_mse=cv.cv_mse(), rcv_sem=cv.cv_sem(),
               rcv_min=np.array([cv.cv_min(), cv.cv_1se(), cv.index_min(), cv.index_1se()]))

    en = ElasticNet()
    en.fit(X, y, 0.7, 40, -1.0, True, True, False, 100000, 1e-7)
    out.update(en_lambdas=en.lambdas(), en_coef=en.coef(), en_a0=en.intercepts(),
               en_dev=en.dev_ratio(), en_conv=np.array([en.converged(), en.n_nonconverged()]),
               en_pred=en.predict(np.asfortranarray(X[:7])))
    grid = en.lambdas()[5:15]
    en2 = ElasticNet()
    en2.fit_grid(X, y, grid, 0.7, False, True, True, 100000, 1e-7)
    out.update(eng_lambdas=en2.lambdas(), eng_coef=en2.coef(), eng_a0=en2.intercepts())

    ecv = ElasticNetCV()
    ecv.fit(X, y, 0.5, 5, 40, -1.0, 3, True, True, 100000, 1e-7)
    out.update(ecv_lambdas=ecv.lambdas(), ecv_mse=ecv.cv_mse(), ecv_sem=ecv.cv_sem(),
               ecv_min=np.array([ecv.cv_min(), ecv.cv_1se(), ecv.index_min(), ecv.index_1se()]))

    r = SVDSolver().compute(X, 5)
    out.update(svd_U=r.U, svd_S=r.S, svd_V=r.V)

    pca = PCA()
    pca.fit(X, 4)
    out["pca_pre_X"] = X.copy(order="F")
    out.update(pca_V=pca.get_loadings(), pca_ev=pca.get_explained_variance(),
               pca_means=pca.get_means(), pca_norms=pca.get_norms(),
               pca_tr=pca.transform(np.asfortranarray(X0[:6])))
    pca.restore()
    out["pca_restored_X"] = X.copy(order="F")

    out["ridge"] = np.column_stack([RidgeSolver.solve(X, y, lam) for lam in (1e-2, 1.0, 1e2)])

    Xs = X.copy(order="F")
    Xs[:, 1] = 3.0
    z = ZScoreScaler(True, True)
    z.fit_transform_inplace(Xs)
    out.update(z_X=Xs.copy(order="F"), z_centers=z.get_centers(), z_scales=z.get_scales(),
               z_dropped=np.asarray(z.get_dropped_indices(), dtype=np.int64))
    z.inverse_transform_inplace(Xs)
    out["z_back"] = Xs.copy(order="F")
    l1 = LpNormScaler(NormType.L1, False, True)
    l1.fit_transform_inplace(Xs)
    out.update(lp_X=Xs.copy(order="F"), lp_scales=l1.get_scales())
    l1.inverse_transform_inplace(Xs)
    out["lp_back"] = Xs.copy(order="F")

    Xc = np.asfortranarray(X[: min(n, 60)])
    out["hac_average"] = np.asarray(agglomerative_cluster(Xc, LinkageMethod.Average,
                                                          DistanceMetric.Euclidean))
    hw = np.asarray(agglomerative_cluster(Xc, LinkageMethod.Ward, DistanceMetric.Euclidean))
    out["hac_ward"] = hw
    out["hac_cut"] = np.asarray(cut_tree(hw, Xc.shape[0], 3))

    out["X_final"] = X.copy(order="F")
    out["X_untouched"] = np.array([np.array_equal(X, X0)])
    return out


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("record", "compare"):
        sys.exit("usage: zero_copy_reference.py record|compare <dir>")
    mode, d = sys.argv[1], Path(sys.argv[2])
    d.mkdir(parents=True, exist_ok=True)
    results = {f"{nm}__{k}": np.asarray(v) for nm, (n, p) in SHAPES.items()
               for k, v in run_case(n, p).items()}
    file = d / "python_ml_reference.npz"
    if mode == "record":
        np.savez(file, **results)
        print("recorded", file)
        return
    ref = np.load(file)
    bad = [k for k in results if k not in ref.files or not np.array_equal(results[k], ref[k])]
    missing = [k for k in ref.files if k not in results]
    for k in bad + missing:
        print("DIFFERS:", k)
    if bad or missing:
        sys.exit("Python ML reference: DIFFERENCES")
    print("Python ML reference: identical")


if __name__ == "__main__":
    main()
