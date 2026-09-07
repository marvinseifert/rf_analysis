# Use Gollisch semi-negative matrix factorisation to identify subunits in receptive fields
# The orientation of the final plot is aligned with the gollisch pipeline scripts, but not my other receptive field functions...

# %% Import dependencies
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import tqdm
import hdbscan
import numpy as np
import einops
from bokeh.core.property.visual import FontSize
from scipy.spatial.distance import cdist
from stnmf import STNMF
from pathlib import Path
from s_nmf.factorization import semi_nmf_hals
from rf_torch.parameters import Cell_Params
from pickle import dump, load
from matplotlib import colormaps

# %% 1. RUN SNMF for each cell and save to file

# Set semi-negative matrix factorisation parameters
n_components = 20
n_runs = 30
sparsity = 1e-2
topk_per_feature = 1
min_cluster_size = 10
n_cluster_samples = 5

# Load data
data_root = Path(
    r"F:\Laura\zebrafish_02_12_2025\Phase_01\4px_20Hz_40mins_shuffle_idx_4"
)

# Select cells to process
cells = [200]

for cell_idx in cells:
    print(f"\nProcessing Cell {cell_idx} ...")
    save_root = data_root / f"cell_{cell_idx}"
    cell_params = Cell_Params.load_from_root_json(save_root)
    cell_params["s_nmf_analysis"]["n_components"] = n_components
    cell_params["s_nmf_analysis"]["n_runs"] = n_runs
    cell_params["s_nmf_analysis"]["sparsity"] = sparsity
    cell_params["s_nmf_analysis"]["top_k_per_feature"] = 1
    cell_params["s_nmf_analysis"]["min_cluster_size"] = min_cluster_size
    cell_params["s_nmf_analysis"]["n_cluster_samples"] = n_cluster_samples

    with h5py.File(data_root / f"cell_{cell_idx}/snippets.h5", "r") as f:
        print(f["snippets"].shape)
        # calculate size in GB
        size_gb = f["snippets"].size * f["snippets"].dtype.itemsize / (1024**3)
        print(f"Size of snippets: {size_gb:.2f} GB")
        snippets = f["snippets"][:]

    snippets = snippets[100:]  # Crop the snippets to remove borders

    sta = np.mean(snippets, axis=0)  # Calculate the mean of all snippets
    mse_snippets = np.max(
        (np.mean(snippets, axis=0) - 0.5) ** 2, axis=0
    )  # Calculate MSE for each snippet

    # %%
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mse_snippets, cmap="gray")
    # flip the y-axis
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.show()
    # %%
    T, H, W = sta.shape
    N = snippets.shape[0]

    # Signed projection (keeps ON/OFF polarity)
    W_signed = (sta - 0.5).astype(np.float32)  # (T,H,W)
    projected_snippets = np.zeros((N, H, W), np.float32)

    for t in tqdm.tqdm(range(T)):
        projected_snippets += (
            snippets[:, t, :, :].astype(np.float32) - 0.5
        ) * W_signed[t]
    projected_snippets = einops.rearrange(
        projected_snippets, "n h w -> h w n"
    )  # (N, H*W)

    stnmf = STNMF(projected_snippets, r=30)
    stnmf.pixel_size = 4.0
    # %%
    # fig = stnmf.plot(colors="#2980b9")

    fig = stnmf.plot()
    fig.show()

    # %%
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mse_snippets, cmap="gray")
    # flip the y-axis
    ax.set_ylim(ax.get_ylim()[::-1])

    for contour in stnmf.outlines:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="white")
    fig.show()
    # %%
    # save outputs
    np.save(save_root / "s_nmf_contours.npy", stnmf.outlines)
    np.save(save_root / "s_nmf_subunits.npy", stnmf.subunits)
    np.save(save_root / "snippet_mse.npy", mse_snippets)

    print(f"Finished processing Cell {cell_idx}")

# %%
# # %% 3. PLOTTING THE ABOVE< BUT BY COLOUR LABEL (short, long, white) - TEST.
# # Alignment still seems slightly off between receptive fields and subunits
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Set parameters
# root = Path(r"F:\Laura\zebrafish_02_12_2025\Phase_01\4px_20Hz_40mins_shuffle_idx_4")
# quality = np.load(root / "quality.npy")
# cell_labels = np.load(root / "cell_labels.npy", allow_pickle=True)  # cell IDs + labels
#
# cells = [6, 8, 9, 32, 33, 50, 51, 52, 53, 54, 55, 56, 64, 73, 74, 75, 76 ,79,
#          90, 104, 106, 113, 138, 163, 178, 180, 181, 182, 199, 200, 203,
#          204, 205, 206, 207, 231, 233, 250, 262, 264, 269, 271, 275, 288, 292, 316]
#
# # Map label to color
# label_to_color = {"white": "black", "long": "red", "short": "darkblue"}
#
# # Set figure
# fig, axes = plt.subplots(figsize=(10, 10), dpi=300)
#
# for idx, cell in enumerate(cells):
#
#     # Plot background mse receptive fields (optional)
#     image = np.load(root / f"cell_{int(cell)}/snippet_mse.npy")
#     x_centre = quality[cell, 2]
#     y_centre = quality[cell, 3]
#     h, w = image.shape
#     extent = [x_centre - w / 2, x_centre + w / 2, y_centre - h / 2, y_centre + h / 2]
#     # axes.imshow(image, cmap="Greys", origin="lower", extent=extent, alpha=0.4)
#
#     # Find cell's label safely
#     matching_rows = cell_labels[cell_labels[:, 0] == cell]
#     if len(matching_rows) == 0:
#         colour = "black"  # fallback if cell not in cell_labels
#     else:
#         colour_label = matching_rows[0, 1]  # column 1 contains label
#         colour = label_to_color.get(colour_label, "black")  # map label to color
#
#     # Plot subunits on top
#     subunit_root = root / f"cell_{cell}"
#     contours = np.load(subunit_root / "s_nmf_contours.npy", allow_pickle=True)
#
#     for contour in contours:
#         contour = np.asarray(contour)
#         axes.plot(contour[:, 1] + x_centre - w / 2,
#                   contour[:, 0] + y_centre - h / 2,
#                   color=colour, linewidth=0.4)
#
# # Plot ticks and labels
# axes.set_ylim([0, 600])
# axes.set_xlim([0, 600])
# axes.set_yticks(np.arange(0, 601, 100))
# axes.set_xticks(np.arange(0, 601, 100))
# axes.set_xticklabels(np.arange(0, 2401, 400))
# axes.set_yticklabels(np.arange(0, 2401, 400))
# axes.set_xlabel("µm")
# axes.set_ylabel("µm")
#
# # Add legend for label colors
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], color='darkblue', lw=2, label='short'),
#     Line2D([0], [0], color='black', lw=2, label='white'),
#     Line2D([0], [0], color='red', lw=2, label='long')
# ]
# axes.legend(handles=legend_elements, frameon=False, fontsize=8)
#
# # Add scale bar, 8 µm = 2 pixels
# bar_length_px = 2
# x_start_bar = 570
# x_end_bar = x_start_bar + bar_length_px
# y_pos = 570
# axes.plot([x_start_bar, x_end_bar], [y_pos, y_pos], color='red', linewidth=3)
# axes.text(x_start_bar - 13, y_pos + 11, "8 µm", color='red', fontsize=15)
#
# fig.suptitle("Subunit contours (SNMF) for cells with QI > 20 (white 4px_20Hz_shuffle noise)")
# fig.tight_layout()
# fig.show()
#
