# Script for multi-panel plot of subunits for one cell, calculated using different sparsity parameter values
# Import dependencies
from stnmf import STNMF
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # needed for compressed h5 files
import tqdm
import einops
from pathlib import Path
import math

# %% Set parameters
data_root = Path(
    r"F:\Laura\zebrafish_15_01_2026\Phase_00\4px_20Hz_40mins_shuffle_idx_12"
)

cell_idx = 21
save_root = data_root / f"cell_{cell_idx}"

sparsity_vals = np.arange(0.1, 2.1, 0.1)
r_subunits = np.arange(140, 150, 5)
pixel_size = 4.0

# %% Processing
# Load snippets
snippets_path = data_root / f"cell_{cell_idx}/snippets.h5"

with h5py.File(snippets_path, "r") as f:
    snippets = f["snippets"][:]
    size_gb = snippets.size * snippets.dtype.itemsize / (1024**3)
    print(f"Snippets shape: {snippets.shape}")
    print(f"Size of snippets: {size_gb:.2f} GB")

# Crop to remove borders
snippets = snippets[100:]

# Spike-triggered average
sta = np.mean(snippets, axis=0)  # (T, H, W)

# Background image for plotting
mse_snippets = np.max((sta - 0.5) ** 2, axis=0)

T, H, W = sta.shape
N = snippets.shape[0]

# Signed projection
W_signed = (sta - 0.5).astype(np.float32)  # (T, H, W)
projected_snippets = np.zeros((N, H, W), dtype=np.float32)

for t in tqdm.tqdm(range(T), desc="Projecting snippets"):
    projected_snippets += (snippets[:, t].astype(np.float32) - 0.5) * W_signed[t]

# STNMF expects (H, W, N)
projected_snippets = einops.rearrange(projected_snippets, "n h w -> h w n")

# %% Plotting
n_sparsity = len(sparsity_vals)
n_r = len(r_subunits)
# n_cols = math.ceil(math.sqrt(n_sparsity))
# n_rows = math.ceil(n_sparsity / n_cols)
n_cols = math.ceil(math.sqrt(n_r))
n_rows = math.ceil(n_r / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
axes = np.array(axes).reshape(-1)

# for idx, sp in enumerate(sparsity_vals):
for idx, i in enumerate(r_subunits):
    # print(f"Running STNMF (sparsity = {sp:.1f})")

    stnmf = STNMF(
        projected_snippets, sparsity=0.5, r=i
    )  # run stnmf for each sparsity value
    stnmf.pixel_size = pixel_size

    ax = axes[idx]
    ax.imshow(mse_snippets, cmap="gray")
    # ax.set_title(f"sparsity = {sp:.1f}", fontsize=10)
    ax.set_title(f"r = {i:.1f}", fontsize=10)

    # Flip y-axis to match image coordinates
    ax.set_ylim(ax.get_ylim()[::-1])

    # Plot subunit contours
    for contour in stnmf.outlines:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="white")

    ax.axis("off")

# turn off unused axes
for j in range(idx + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# %%
