from stnmf import STNMF

import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import tqdm
import hdbscan
import numpy as np
import einops
from scipy.spatial.distance import cdist
from pathlib import Path
from s_nmf.factorization import semi_nmf_hals
from rf_torch.parameters import Cell_Params
from pickle import dump, load

# %%
############# Semi NMF parameters #############
n_components = 20
n_runs = 30
sparsity = 1e-2
topk_per_feature = 1
min_cluster_size = 10
n_cluster_samples = 5
# %%
data_root = Path(
    r"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/kilosort4/4px_20Hz_shuffle_535nm_idx_12")
cell_idx = 506
save_root = data_root / f"cell_{cell_idx}"
cell_params = Cell_Params.load_from_root_json(save_root)
cell_params["s_nmf_analysis"]["n_components"] = n_components
cell_params["s_nmf_analysis"]["n_runs"] = n_runs
cell_params["s_nmf_analysis"]["sparsity"] = sparsity
cell_params["s_nmf_analysis"]["top_k_per_feature"] = 1
cell_params["s_nmf_analysis"]["min_cluster_size"] = min_cluster_size
cell_params["s_nmf_analysis"]["n_cluster_samples"] = n_cluster_samples

# %%
with h5py.File(data_root / f"cell_{cell_idx}/snippets.h5",
               "r") as f:
    print(f["snippets"].shape)
    # calculate size in GB
    size_gb = f["snippets"].size * f["snippets"].dtype.itemsize / (1024 ** 3)
    print(f"Size of snippets: {size_gb:.2f} GB")
    snippets = f["snippets"][:]

# %%
snippets = snippets[100:]  # Crop the snippets to remove borders
# random_indices = np.random.choice(snippets.shape[0], size=min(10000, snippets.shape[0]), replace=False)
# snippets = snippets[random_indices]  # Subsample snippets if there are too many
# %%
sta = np.mean(snippets, axis=0)  # Calculate the mean of all snippets
mse_snippets = np.max((np.mean(snippets, axis=0) - 0.5) ** 2, axis=0)  # Calculate MSE for each snippet
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap='gray')
# flip the y-axis
ax.set_ylim(ax.get_ylim()[::-1])
fig.show()
# %%


T, H, W = sta.shape
N = snippets.shape[0]

# ---- Signed projection (keeps ON/OFF polarity) ----
W_signed = (sta - 0.5).astype(np.float32)  # (T,H,W)
projected_snippets = np.zeros((N, H, W), np.float32)

for t in tqdm.tqdm(range(T)):
    projected_snippets += (snippets[:, t, :, :].astype(np.float32) - 0.5) * W_signed[t]
projected_snippets = einops.rearrange(projected_snippets, 'n h w -> h w n')  # (N, H*W)
# %%
stnmf = STNMF(projected_snippets, r=30)
stnmf.pixel_size = 4.0

# %%
fig = stnmf.plot(colors="#2980b9")

fig.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap='gray')
# flip the y-axis
ax.set_ylim(ax.get_ylim()[::-1])

for contour in stnmf.outlines:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='white')
fig.show()
# %%
np.save(save_root / "s_nmf_contours.npy", stnmf.outlines)
np.save(save_root / "s_nmf_subunits.npy", stnmf.subunits)
np.save(save_root / "snippet_mse.npy", mse_snippets)
# %% save cell params
