"""
Smoke test: TRexSelector runs correctly with OMP_NUM_THREADS=1 (single OpenMP thread).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import trex_selector_neo


def test_single_openmp_thread():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 20))
    y = X[:, 0] * 5 + X[:, 1] * -3 + rng.standard_normal(50)
    ts = trex_selector_neo.TRexSelector(X, y, verbose=False)
    res = ts.select()
    assert res is not None
    assert isinstance(ts.selected_indices, list)
