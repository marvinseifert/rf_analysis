# %%
def cutout_nans(data, inset=0):
    import numpy as np

    # Compute mask of valid (non-NaN) pixels
    x_slice = slice(
        np.where(np.any(data > 0, axis=0))[0][0] + inset,
        np.where(np.any(data > 0, axis=0))[0][-1] + 1 - inset,
        1,
    )
    y_slice = slice(
        np.where(np.any(data > 0, axis=1))[0][0] + inset,
        np.where(np.any(data > 0, axis=1))[0][-1] + 1 - inset,
        1,
    )
    return data.isel(x=x_slice, y=y_slice)


# %% High-elongation RF panel (pooled across datasets)

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------
path_phase0 = Path(
    r"F:\Laura\zebrafish_05_11_2025\Phase_01\noise_analysis\noise_data.nc"
)

path_phase1 = Path(
    r"F:\Laura\zebrafish_15_01_2026\Phase_00\noise_analysis\noise_data.nc"
)

path_phase2 = Path(
    r"F:\Laura\zebrafish_26_02_2026\Phase_00\noise_analysis\noise_data.nc"
)
dataset_phase0 = xr.load_dataset(path_phase0)
dataset_phase1 = xr.load_dataset(path_phase1)
dataset_phase2 = xr.load_dataset(path_phase2)
# %%
path_phase3 = Path(
    r"F:\Laura\zebrafish_02_12_2025\Phase_01\noise_analysis\noise_data.nc"
)
dataset_phase3 = xr.load_dataset(path_phase3)
datasets = [dataset_phase0, dataset_phase1, dataset_phase2, dataset_phase3]
# %%
# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
channel = "12px_20Hz_shuffle"
threshold = 1.41
crop_size = 600
half_crop = crop_size / 2


# ------------------------------------------------------------
# HELPER: COLLECT HIGH-ELONGATION CELLS
# ------------------------------------------------------------
def collect_candidates(dataset):
    tilt = dataset["tilt"].sel(channel=channel)
    elongation = 1 / tilt

    cell_ids = elongation.coords["cell_index"].values
    values = elongation.values

    valid = np.isfinite(values) & (values >= threshold)

    return list(zip([dataset] * np.sum(valid), cell_ids[valid], values[valid]))


# ------------------------------------------------------------
# FILTER + SORT (same as before)
# ------------------------------------------------------------
candidates = []
for ds in datasets:
    candidates += collect_candidates(ds)

candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

print(f"Total cells above threshold: {len(candidates)}")

# ------------------------------------------------------------
# DYNAMIC GRID SIZE
# ------------------------------------------------------------
n_cells = len(candidates)

n_cols = 4
n_rows = int(np.ceil(n_cells / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(5 * n_cols, 5 * n_rows),
    constrained_layout=False,
)
fig.subplots_adjust(hspace=0.05)

# fig.patch.set_alpha(0)

# flatten axes for easy indexing
axes = np.array(axes).reshape(-1)

# ------------------------------------------------------------
# PLOT ALL CELLS
# ------------------------------------------------------------
for idx, (dataset, cell_id, elong) in enumerate(candidates):
    ax = axes[idx]
    ax.patch.set_alpha(0)
    ax.grid(False)

    # --------------------------------------------------------
    # LOAD RF
    # --------------------------------------------------------
    rf_image = dataset["cm_most_important"].sel(
        channel=channel,
        cell_index=cell_id,
    )

    if rf_image.size == 0 or np.isnan(rf_image).all():
        print(f"Skipping empty RF: cell {cell_id}")
        continue

    rf_image_crop = cutout_nans(rf_image)

    # --------------------------------------------------------
    # CENTER CROP
    # --------------------------------------------------------
    x_coords = rf_image_crop.coords[rf_image_crop.dims[-1]].values
    y_coords = rf_image_crop.coords[rf_image_crop.dims[-2]].values

    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2

    x_min = x_center - half_crop
    x_max = x_center + half_crop
    y_min = y_center - half_crop
    y_max = y_center + half_crop

    rf_image_crop = rf_image_crop.sel(
        {
            rf_image_crop.dims[-1]: slice(x_min, x_max),
            rf_image_crop.dims[-2]: slice(y_min, y_max),
        }
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    im = rf_image_crop.plot.imshow(
        ax=ax,
        cmap="coolwarm",
        add_colorbar=False,
    )

    ax.set_aspect(1)
    ax.tick_params(axis="both", labelsize=12)

    # --------------------------------------------------------
    # ELONGATION LABEL
    # --------------------------------------------------------
    ax.text(
        0.5,
        1.05,
        f"elongation: {elong:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    # --------------------------------------------------------
    # CLEAN AXES LABELS
    # --------------------------------------------------------
    ax.set_xlabel("")
    ax.set_ylabel("")

# ------------------------------------------------------------
# TURN OFF UNUSED AXES
# ------------------------------------------------------------
for j in range(len(candidates), len(axes)):
    axes[j].axis("off")

plt.show()

# %% PLOT ACTUAL ------------------------------------------------------------
# USER-DEFINED CELL IDS
# ------------------------------------------------------------
target_cells = [200, 251, 257, 219, 51, 223, 189, 186]

# ------------------------------------------------------------
# FILTER + SORT
# ------------------------------------------------------------
selected = [c for c in candidates if c[1] in target_cells]
selected = sorted(selected, key=lambda x: x[2], reverse=True)

# optional manual ordering
idxs = [0, 1, 3, 4, 5, 6, 7, 8]
selected = [selected[i] for i in idxs if i < len(selected)]

print("Plotting cells in order of elongation:")
for _, cell_id, elong in selected:
    print(f"cell {cell_id} → elongation {elong:.3f}")

# ------------------------------------------------------------
# FIGURE SETUP
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

fig = plt.figure(figsize=(20, 10), constrained_layout=True)
fig.patch.set_alpha(0)
gs = gridspec.GridSpec(2, 4, figure=fig)

axes = []
for i in range(2):
    for j in range(4):
        axes.append(fig.add_subplot(gs[i, j]))

# ------------------------------------------------------------
# PLOT LOOP
# ------------------------------------------------------------
for idx, (dataset, cell_id, elong) in enumerate(selected):
    ax = axes[idx]
    ax.patch.set_alpha(0)
    ax.grid(False)

    # --------------------------------------------------------
    # LOAD RF
    # --------------------------------------------------------
    rf_image = dataset["cm_most_important"].sel(
        channel=channel,
        cell_index=cell_id,
    )

    if rf_image.size == 0 or np.isnan(rf_image).all():
        print(f"Skipping empty RF: cell {cell_id}")
        ax.axis("off")
        continue

    rf_image_crop = cutout_nans(rf_image)

    # --------------------------------------------------------
    # CENTER CROP
    # --------------------------------------------------------
    x_coords = rf_image_crop.coords[rf_image_crop.dims[-1]].values
    y_coords = rf_image_crop.coords[rf_image_crop.dims[-2]].values

    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2

    x_min = x_center - half_crop
    x_max = x_center + half_crop
    y_min = y_center - half_crop
    y_max = y_center + half_crop

    rf_image_crop = rf_image_crop.sel(
        {
            rf_image_crop.dims[-1]: slice(x_min, x_max),
            rf_image_crop.dims[-2]: slice(y_min, y_max),
        }
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    im = rf_image_crop.plot.imshow(
        ax=ax,
        cmap="coolwarm",
        add_colorbar=False,
        add_labels=False,
    )

    ax.set_aspect(1)
    ax.tick_params(axis="both", labelsize=15)

    ax.set_title("")
    ax.title.set_visible(False)

    # --------------------------------------------------------
    # INDIVIDUAL COLORBAR (NO LABEL EXCEPT ONE)
    # --------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    cbar.formatter = formatter
    cbar.update_ticks()

    cbar.ax.tick_params(labelsize=10)

    # Only label bottom-right subplot (index 7 in 2x4 grid)
    if idx == len(selected) - 1:
        cbar.set_label("Covariance", fontsize=22, labelpad=15)

    # --------------------------------------------------------
    # ELONGATION LABEL
    # --------------------------------------------------------
    ax.text(
        0.5,
        1.05,
        f"elongation: {elong:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
    )

    # --------------------------------------------------------
    # AXES LABELS
    # --------------------------------------------------------
    row = idx // 4
    col = idx % 4

    if col == 0:
        ax.set_ylabel("y position (µm)", fontsize=20)
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")
    if row == 1:
        ax.set_xlabel("x position (µm)", fontsize=20, labelpad=10)
    else:
        ax.set_xlabel("")

# ------------------------------------------------------------
# TURN OFF UNUSED PANELS
# ------------------------------------------------------------
for j in range(len(selected), 8):
    axes[j].axis("off")

plt.show()
