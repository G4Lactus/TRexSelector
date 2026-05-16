import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import trex_selector
print("Imported")
np.random.seed(42)
X = np.random.randn(50, 20)
y = X[:, 0] * 5 + X[:, 1] * -3 + np.random.randn(50)
print("Testing Selection")
ts = trex_selector.TRexSelector(X, y)
res = ts.select()
print("Success. Selected:", ts.selected_indices)
