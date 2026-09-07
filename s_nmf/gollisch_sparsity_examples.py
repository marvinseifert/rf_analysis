from stnmf import STNMF
from stnmf.callbacks import consensus
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import tqdm
import hdbscan
import numpy as np
import einops
from scipy.spatial.distance import cdist
from pathlib import Path

from laura_scripts.subunit_sparsity_plots import pixel_size
from s_nmf.factorization import semi_nmf_hals
from rf_torch.parameters import Cell_Params
from pickle import dump, load


# %% Load data

data_root = Path(
    r"F:\Laura\zebrafish_15_01_2026\Phase_00\4px_20Hz_40mins_shuffle_idx_12"
)
cell_idx = 21
save_root = data_root / f"cell_{cell_idx}"

# %%
with h5py.File(data_root / f"cell_{cell_idx}/snippets.h5", "r") as f:
    print(f["snippets"].shape)
    # calculate size in GB
    size_gb = f["snippets"].size * f["snippets"].dtype.itemsize / (1024**3)
    print(f"Size of snippets: {size_gb:.2f} GB")
    snippets = f["snippets"][:]
# Exclude first 100 snippets as they may contain artifacts
snippets = snippets[100:]

# %% RMS
sta = np.mean(snippets, axis=0)  # Calculate the mean of all snippets
mse_snippets = np.max(
    (np.mean(snippets, axis=0) - 0.5) ** 2, axis=0
)  # Calculate MSE for each snippet
# %% Sanity plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap="gray")
# flip the y-axis
ax.set_ylim(ax.get_ylim()[::-1])
fig.show()
# %% projection
T, H, W = sta.shape
N = snippets.shape[0]

# ---- Signed projection (keeps ON/OFF polarity) ----
W_signed = (sta - 0.5).astype(np.float32)  # (T,H,W)
projected_snippets = np.zeros((N, H, W), np.float32)

for t in tqdm.tqdm(range(T)):
    projected_snippets += (snippets[:, t, :, :].astype(np.float32) - 0.5) * W_signed[t]
projected_snippets = einops.rearrange(projected_snippets, "n h w -> h w n")  # (N, H*W)

# %% Test sparsity levels
# !!!!! Warning, this will take some time (30 mins to hours), depending on the number of snippets
nr_repetitions = 1
arguments = dict(sparsities=[0, 0.5, 1, 1.5, 2], num_rep=nr_repetitions)
results = dict()
nr_components = 150
#
# # %%
# STNMF(
#     projected_snippets,
#     callback=consensus,
#     callback_data=results,
#     callback_kwargs=arguments,
#     r=nr_components,
# )
# %% Results
# print("The stability of the decomposition at different sparsity levels:")
# for sparsity, stability in zip(arguments["sparsities"], results["cpcc"]):
#     print(f"Sparsity: {sparsity}, Stability: {stability:.4f}")

# %% This cell will run STNMF with the best sparsity level found above and with 10 repetitions for consensus analysis
# best_sparsity = np.nanargmax(results["cpcc"])
# run STNMF with the best sparsity
# stnmf = STNMF(
#     projected_snippets,
#     callback=consensus,
#     callback_kwargs={
#         "sparsities": arguments["sparsities"][best_sparsity],
#         "num_rep": nr_repetitions,
#     },
#     r=nr_components,
#     sparsity=arguments["sparsities"][best_sparsity],
# )

best_sparsity = 1.5
stnmf = STNMF(
    projected_snippets,
    callback=consensus,
    callback_kwargs={
        "sparsities": [best_sparsity],
        "num_rep": nr_repetitions,
    },
    r=nr_components,
    sparsity=best_sparsity,
)

# %% Get the polarities of all subunits
subunits = stnmf.subunits  # shape: (num_subunits, x, y)

# Find the peak intensity value for each subunit and get its sign
# This mimics what the library is trying to do internally
manual_polarities = []
for i in range(subunits.shape[0]):
    s = subunits[i]
    # Find value with largest absolute magnitude
    peak_val = s.flat[np.abs(s).argmax()]
    # Assign +1 for ON, -1 for OFF
    manual_polarities.append(1 if peak_val > 0 else -1)

print(manual_polarities)
# %% Plot results
polarity_colours = {1: "red", -1: "blue"}
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap="gray")
# flip the y-axis
ax.set_ylim(ax.get_ylim()[::-1])

for c_idx, contour in enumerate(stnmf.outlines):
    ax.plot(
        contour[:, 1],
        contour[:, 0],
        linewidth=2,
        color=polarity_colours[manual_polarities[c_idx]],
    )
fig.show()
# %%

fig = stnmf.plot(colors="#2980b9")

fig.show()

# %% Subunit triggered averages
k = 0  # subunit index

weights = stnmf.h[k]
weights = weights / np.sum(weights)

sub_sta_full = np.tensordot(weights, snippets, axes=(0, 0))  # -> (T, X, Y)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.var(sub_sta_full, axis=0), cmap="gray")
contour = stnmf.outlines[k]
ax.plot(
    contour[:, 1],
    contour[:, 0],
    linewidth=2,
    color=polarity_colours[manual_polarities[c_idx]],
)
# flip the y-axis
ax.set_ylim(ax.get_ylim()[::-1])
fig.show()

# %%
s = subunits[k]
s = s / np.linalg.norm(s)

temporal_kernel = np.tensordot(sub_sta_full, s, axes=([1, 2], [0, 1]))  # -> (T,)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(temporal_kernel)
fig.show()
# %%
np.save(save_root / "s_nmf_contours.npy", stnmf.outlines)
np.save(save_root / "s_nmf_subunits.npy", stnmf.subunits)
np.save(save_root / "snippet_mse.npy", mse_snippets)

# %% Subunit kernel plots

# Normalize weights (K, N)
weights = stnmf.h / stnmf.h.sum(axis=1, keepdims=True)

# Normalize subunits (K, X, Y)
subunits_norm = subunits / np.linalg.norm(subunits, axis=(1, 2), keepdims=True)

# Compute all subunit STAs at once: (K, T, X, Y)
sub_sta_full = np.tensordot(weights, snippets, axes=(1, 0))

# Project each STA onto its corresponding subunit → (K, T)
nr_components = sub_sta_full.shape[0]
T = sub_sta_full.shape[1]

temporal_kernels = np.empty((nr_components, T))

for k in range(nr_components):
    temporal_kernels[k] = np.tensordot(
        sub_sta_full[k], subunits_norm[k], axes=([1, 2], [0, 1])  # (T, X, Y)  # (X, Y)
    )

# Plot
fig, ax = plt.subplots(10, 15, figsize=(20, 20))
ax = ax.flatten()

for k in range(nr_components):
    ax[k].plot(temporal_kernels[k])
    ax[k].set_title(f"subunit={k}")

plt.tight_layout()
fig.show()
