import numpy as np
import plotly.express as px

# 1. Setup and Initialization
np.random.seed(42)
n_snps = 150
mat = np.zeros((n_snps, n_snps))

# 2. Define Main LD Blocks (start, end, base_correlation)
blocks = [
    (0, 45, 0.4),
    (45, 80, 0.8),
    (80, 90, 0.95),
    (90, 110, 0.0),
    (110, 140, 0.7),
    (140, 150, 0.5)
]

# Populate blocks with distance-decay and alternating signs
for start, end, base_corr in blocks:
    if base_corr == 0:
        continue
    for i in range(start, end):
        for j in range(start, end):
            if i == j:
                mat[i, j] = 1.0
            else:
                sign = 1 if np.random.rand() > 0.4 else -1
                dist = abs(i - j)
                decay = np.exp(-dist * 0.1) # exponential decay by distance
                corr = base_corr * decay * sign + np.random.normal(0, 0.1)
                mat[i, j] = np.clip(corr, -1, 1)
                mat[j, i] = mat[i, j]

# 3. Inject Long-Range LD (off-diagonal blocks)
long_range = [(10, 20, 120, 130, 0.6), (50, 60, 90, 100, -0.5)]
for r_start, r_end, c_start, c_end, strength in long_range:
    for i in range(r_start, r_end):
        for j in range(c_start, c_end):
            sign = 1 if np.random.rand() > 0.5 else -1
            val = np.clip(strength * sign + np.random.normal(0, 0.1), -1, 1)
            mat[i, j] = val
            mat[j, i] = val

# 4. Add Background Noise and Ensure Symmetry
noise = np.random.normal(0, 0.05, (n_snps, n_snps))
noise = (noise + noise.T) / 2
mat = mat + noise

np.fill_diagonal(mat, 1.0)
mat = np.clip(mat, -1, 1)
mat = (mat + mat.T) / 2 # enforce strict symmetry

# 5. Plotly Interactive Rendering
fig = px.imshow(
    mat,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1
)

fig.update_layout(
    title="Simulated LD Block Correlation Matrix",
    xaxis_title="SNP Index",
    yaxis_title="SNP Index",
    xaxis=dict(showticklabels=True, tickmode='linear', tick0=0, dtick=30),
    yaxis=dict(showticklabels=True, tickmode='linear', tick0=0, dtick=30),
    width=700,
    height=700
)

# Render interactively
fig.show()
