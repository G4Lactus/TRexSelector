import numpy as np
import trex_selector
print("Imported")
np.random.seed(42)
X = np.random.randn(50, 20)
y = X[:, 0] * 5 + X[:, 1] * -3 + np.random.randn(50)
print("Testing Selection")
try:
    ts = trex_selector.TRexSelector(X, y)
    res = ts.select()
    print("Success. Selected:", ts.selected_indices)
except Exception as e:
    print("Failed")
