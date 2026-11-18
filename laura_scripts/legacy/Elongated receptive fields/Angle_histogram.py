# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %% Dataset paths
dataset_paths = {
    # "dataset_1": Path(
    #     r"F:\Laura\zebrafish_05_11_2025\Phase_00\noise_analysis\noise_data.nc"
    # ),
    "dataset_2": Path(
        r"F:\Laura\zebrafish_05_11_2025\Phase_01\noise_analysis\noise_data.nc"
    ),
    "dataset_3": Path(
        r"F:\Laura\zebrafish_02_12_2025\Phase_01\noise_analysis\noise_data.nc"
    ),
    # "dataset_4": Path(
    #     r"F:\Laura\zebrafish_18_12_2025\Phase_00\noise_analysis\noise_data.nc"
    # ),
    "dataset_5": Path(
        r"F:\Laura\zebrafish_15_01_2026\Phase_00\noise_analysis\noise_data.nc"
    ),
    "dataset_6": Path(
        r"F:\Laura\zebrafish_26_02_2026\Phase_00\noise_analysis\noise_data.nc"
    ),
    # "dataset_7": Path(
    #     r"F:\Laura\zebrafish_13_05_2026\Phase_00\noise_analysis\noise_data.nc"
    # ),
    # "dataset_8": Path(
    #     r"F:\Laura\zebrafish_14_05_2026\Phase_00\noise_analysis\noise_data.nc"
    # ),
}

CHANNELS_TO_LOAD = {
    # "dataset_1": ["white_12px"],
    "dataset_2": ["12px_20Hz_shuffle"],
    "dataset_3": ["12px_20Hz_shuffle", "4px_20Hz_40mins_shuffle"],
    # "dataset_4": ["white_4px"],
    "dataset_5": ["12px_20Hz_shuffle", "4px_20Hz_40mins_shuffle"],
    "dataset_6": ["12px_20Hz_shuffle", "4px_20Hz_40mins_shuffle"],
    # "dataset_7": ["8px_20Hz_20mins_shuffle_x4"],
    # "dataset_8": ["12px_20Hz_25mins_shuffle_x12"],
}
# %% Load datasets

datasets = {}
for name, path in dataset_paths.items():
    ds = xr.load_dataset(path)

    print(f"\n{name}")
    print(ds["channel"].values)

    for ch in CHANNELS_TO_LOAD[name]:
        key = f"{name}_{ch}"
        datasets[key] = ds.sel(channel=ch)

# %% Functions


# Calculate orientation differences between cell pairs within each dataset
def compute_orientation_differences(
    dataset,
    qi_limit=20,
    tilt_limit=None,
):
    """
    Compute pairwise orientation differences for one dataset.
    """

    # Quality values
    quality = dataset["quality"]

    # Cells above quality threshold
    top_cells = quality.cell_index.values[quality.values > qi_limit]

    # Tilt values
    tilts = dataset["tilt"].sel(cell_index=top_cells)
    # tilts = dataset["tilt"].sel(channel=channel, cell_index=top_cells)

    # Valid cells
    valid_cells = top_cells[((tilts < tilt_limit) & np.isfinite(tilts)).values]

    # Orientation angles
    # angles = dataset["angle"].sel(channel=channel, cell_index=valid_cells).values
    angles = dataset["angle"].sel(cell_index=valid_cells).values

    # Remove NaNs
    angles = angles[np.isfinite(angles)]

    # Pairwise orientation differences
    diff_matrix = np.abs(angles[:, None] - angles[None, :])

    # Orientation symmetry (0–180°)
    diff_matrix = np.minimum(diff_matrix, 180 - diff_matrix)

    # Unique pairs only
    diffs = diff_matrix[np.triu_indices(len(angles), k=1)]

    return diffs, valid_cells


# Plot overlaid histogram
def plot_indv_histograms(diff_dict, dataset_names=None, bins=None):
    if dataset_names is None:
        dataset_names = list(diff_dict.keys())

    # exclude small recordings
    dataset_names = [name for name in dataset_names if len(diff_dict[name]) >= 10]

    # common bin edges across all datasets
    all_data = np.concatenate([diff_dict[name] for name in dataset_names])
    # bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
    bin_edges = np.linspace(0, 90, bins + 1)
    fig, ax = plt.subplots()

    bottom = np.zeros(len(bin_edges) - 1)

    # colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    # total_pairs = 0
    # for name, color in zip(dataset_names, colors):
    #     counts, _ = np.histogram(diff_dict[name], bins=bin_edges)
    #     counts = counts / counts.sum() * 100
    #     data = diff_dict[name]
    #     total_pairs += len(data)
    #     ax.bar(
    #         bin_edges[:-1],
    #         counts,
    #         width=np.diff(bin_edges),
    #         bottom=bottom,
    #         label=f"{name} (n cells ={(1 + np.sqrt(1 + 8 * len(data))) / 2})",
    #         color=color,
    #         alpha=0.7,
    #         align="edge",
    #     )

    total_pairs = 0
    for name in dataset_names:
        counts, _ = np.histogram(diff_dict[name], bins=bin_edges)
        counts = counts / counts.sum() * 100
        data = diff_dict[name]
        total_pairs += len(data)
        ax.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            bottom=bottom,
            label=f"{name} (n cells ={(1 + np.sqrt(1 + 8 * len(data))) / 2})",
            color="steelblue",
            alpha=0.7,
            align="edge",
        )
        bottom += counts
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.set_xlabel("Orientation difference (degrees)")
    ax.set_ylabel("Count %")
    ax.set_title(
        f"Between-cell orientation difference \n(tilt limit: {TILT_LIMIT}), (total between-cell comparisons: {total_pairs})"
    )
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(diff_dict, dataset_names=None, bins=20):
    if dataset_names is None:
        dataset_names = list(diff_dict.keys())

    dataset_names = [name for name in dataset_names if len(diff_dict[name]) >= 10]

    bin_edges = np.linspace(0, 90, bins + 1)

    # limit to first 3 datasets for right panel (as you requested)
    dataset_names = dataset_names[:3]

    n = len(dataset_names)

    fig = plt.figure(figsize=(10, 6))

    gs = fig.add_gridspec(
        nrows=n,
        ncols=2,
        width_ratios=[2.2, 1],  # 👈 left is bigger
        wspace=0.35,
        hspace=0.6,
    )

    # =========================
    # LEFT: COMBINED (SPAN ALL ROWS)
    # =========================
    ax_combined = fig.add_subplot(gs[:, 0])

    colors = plt.get_cmap("Dark2")(np.linspace(0, 1, n))
    color_map = dict(zip(dataset_names, colors))

    bottom = np.zeros(len(bin_edges) - 1)

    for name in dataset_names:
        data = diff_dict[name]
        counts, _ = np.histogram(data, bins=bin_edges)
        counts = counts / counts.sum() * 100

        ax_combined.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            bottom=bottom,
            color=color_map[name],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            align="edge",
            label=name,
        )

        bottom += counts

    ax_combined.set_title("Datasets combined", fontsize=12)
    ax_combined.set_ylabel("Count (%)")
    ax_combined.set_xlabel("Orientation difference (°)")
    ax_combined.set_xticks([0, 20, 40, 60, 80, 90])

    # ax_combined.legend(frameon=False, fontsize=8)

    # =========================
    # RIGHT: INDIVIDUAL DATASETS
    # =========================
    for i, name in enumerate(dataset_names):
        ax = fig.add_subplot(gs[i, 1])
        data = diff_dict[name]

        counts, _ = np.histogram(data, bins=bin_edges)
        counts = counts / counts.sum() * 100

        ax.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            color=color_map[name],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            align="edge",
        )
        if i == len(dataset_names) - 1:
            ax.set_xlabel("Orientation difference (°)")
        else:
            ax.set_xlabel("")
        ax.set_title(f"Dataset {i+1}", fontsize=10)
        ax.set_ylabel("Count (%)")
        ax.set_xticks([0, 20, 40, 60, 80, 90])

        ax.tick_params(axis="both", labelsize=8, length=3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # optional: transparent background for publication
    fig.patch.set_alpha(0)

    plt.tight_layout()
    plt.show()


def plot_multipanel_histograms(
    datasets,
    tilt_limits,
    bin_settings=(18, 36),
    qi_limit=None,
    exclusion_threshold=None,
):
    n_rows = len(tilt_limits)
    n_cols = len(bin_settings)

    fig, axes = plt.subplots(
        n + 1, 1, figsize=(5.5, 2.2 * (n + 1)), sharex=True  # 👈 narrower width (was 7)
    )
    # handle case of single row
    axes = np.atleast_2d(axes)

    colors = plt.cm.tab10(range(len(datasets)))

    for i, tilt in enumerate(tilt_limits):
        # compute diffs once per tilt
        all_diffs = {
            name: compute_orientation_differences(
                dataset,
                qi_limit=qi_limit,
                tilt_limit=tilt,
            )
            for name, dataset in datasets.items()
        }

        dataset_names = list(all_diffs.keys())

        # exclude small recordings
        dataset_names = [
            name
            for name in dataset_names
            if len(all_diffs[name]) >= exclusion_threshold
        ]

        # common bins for this tilt
        all_data = np.concatenate([all_diffs[name] for name in dataset_names])

        for j, bins in enumerate(bin_settings):
            ax = axes[i, j]

            bin_edges = np.linspace(
                all_data.min(),
                all_data.max(),
                bins + 1,
            )

            # bottom = np.zeros(len(bin_edges) - 1)
            total_pairs = 0
            for k, (name, color) in enumerate(zip(dataset_names, colors)):
                data = all_diffs[name]

                counts, _ = np.histogram(data, bins=bin_edges)
                counts = counts / counts.sum() * 100

                # for name, color in zip(dataset_names, colors):
                #     data = all_diffs[name]
                #
                #     counts, _ = np.histogram(data, bins=bin_edges)
                #     counts = counts / counts.sum() * 100
                #
                #     total_pairs += len(data)

                # ax.bar(
                #     bin_edges[:-1],
                #     counts,
                #     width=np.diff(bin_edges),
                #     color=color,
                #     alpha=0.4,
                #     edgecolor="black",
                #     linewidth=0.5,
                #     align="edge",
                #     label=name,
                # )
                n_datasets = len(dataset_names)
                # bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
                bin_edges = np.arange(0, 91, 10)
                bin_width = np.diff(bin_edges)[0]
                sub_width = bin_width / n_datasets
                ax.bar(
                    bin_edges[:-1] + k * sub_width,
                    counts,
                    width=sub_width,
                    color=color,
                    alpha=0.7,
                    align="edge",
                    label=name,
                )
                # bottom += counts

            # formatting
            ax.set_xlim(0, 90)

            if j == 0:
                ax.set_ylabel("Count %")

            if i == n_rows - 1:
                ax.set_xlabel("Orientation difference (°)")

            if i == 0:
                bin_width = 90 / bins
                ax.set_title(f"Bins = {bin_width:.1f}°")

            if j == 0:
                ax.text(
                    0.02,
                    0.95,
                    f"tilt ≤ {tilt:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                )

            ax.text(
                0.98,
                0.95,
                f"N = {total_pairs}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

    handles = [
        plt.Line2D([0], [0], color=color, lw=6, alpha=0.4, label=name)
        for name, color in zip(dataset_names, colors)
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(dataset_names), 5),
        frameon=False,
    )

    plt.tight_layout()
    plt.show()


# %% Run functions
# Set parameters
QI_LIMIT = 20
TILT_LIMIT = 0.71

# Calculate orientation differences
# all_diffs = {
#     name: compute_orientation_differences(
#         dataset,
#         # channel=CHANNEL,
#         qi_limit=QI_LIMIT,
#         tilt_limit=TILT_LIMIT,
#     )
#     for name, dataset in datasets.items()
# }
results = {
    name: compute_orientation_differences(
        dataset,
        qi_limit=QI_LIMIT,
        tilt_limit=TILT_LIMIT,
    )
    for name, dataset in datasets.items()
}

all_diffs = {name: res[0] for name, res in results.items()}
valid_cells = {name: res[1] for name, res in results.items()}

# Plot histograms - for all data, or subsets
# plot_histograms(all_diffs, dataset_names=None, bins=9)
plot_indv_histograms(
    all_diffs,
    dataset_names=[
        "dataset_2_12px_20Hz_shuffle",
        "dataset_6_12px_20Hz_shuffle",
        # "dataset_5_4px_20Hz_40mins_shuffle",
        "dataset_5_12px_20Hz_shuffle",
        "dataset_3_12px_20Hz_shuffle",
    ],
    bins=9,
)

# %%
plot_multipanel_histograms(
    datasets=datasets,
    tilt_limits=np.arange(0.67, 0.85, 0.01),
    bin_settings=(9, 18),
    qi_limit=QI_LIMIT,
    exclusion_threshold=10,
)


# # %%
# def plot_centre_surrounds(dataset, channel, qi_limit, tilt_limit):
#     import matplotlib.pyplot as plt
#     import math
#     import numpy as np
#
#     quality = dataset["quality"]
#     tilt = dataset["tilt"]
#
#     q_vals = quality.sel(channel=channel)
#     t_vals = tilt.sel(channel=channel)
#
#     if tilt_limit is not None:
#         mask = (
#             (q_vals.values > qi_limit)
#             & (t_vals.values < tilt_limit)
#             & np.isfinite(t_vals.values)
#         )
#     else:
#         mask = (q_vals.values > qi_limit) & np.isfinite(t_vals.values)
#
#     top_cells = q_vals.cell_index.values[mask]
#
#     if len(top_cells) == 0:
#         print("No cells pass filters")
#         return
#
#     # --- plotting layout ---
#     n_cols = math.ceil(math.sqrt(len(top_cells)))
#     n_rows = math.ceil(len(top_cells) / n_cols)
#
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
#     axes = np.array(axes).flatten()
#
#     for idx, cell_id in enumerate(top_cells):
#         rf_image = dataset["cm_most_important"].sel(channel=channel, cell_index=cell_id)
#
#         rf_image_crop = cutout_nans(rf_image)
#
#         rf_image_crop.plot.imshow(ax=axes[idx], cmap="coolwarm", add_colorbar=False)
#
#         axes[idx].set_title(
#             f"Cell {int(cell_id)}, tilt: {t_vals.sel(cell_index=cell_id).values:.2f}"
#         )
#         axes[idx].set_xlabel("")
#         axes[idx].set_ylabel("")
#         axes[idx].set_aspect("equal")
#
#     # turn off unused axes
#     for ax in axes[len(top_cells) :]:
#         ax.axis("off")
#
#     fig.suptitle(
#         f"Receptive fields | qi > {qi_limit} | {channel}",
#         fontsize=16,
#     )
#
#     plt.tight_layout()
#     plt.show()
#
#
# # %%
# # Load dataset(s)
# path_to_data = Path(
#     r"F:\Laura\zebrafish_05_11_2025\Phase_01\noise_analysis\noise_data.nc"
# )
# ds = xr.load_dataset(path_to_data)
# # %%
# ds = ds.sel(cell_index=[219, 223, 78, 61, 115.0])
# plot_centre_surrounds(ds, channel="12px_20Hz_shuffle", qi_limit=20, tilt_limit=None)


# %%
def compute_orientation_differences_weighted(
    dataset,
    qi_limit=20,
    weight_mode="inverse",
    eps=1e-3,
    k=3.0,
):
    """
    Compute pairwise orientation differences weighted by average tilt.
    NO tilt filtering — all cells included.
    """

    # --- quality filter only ---
    quality = dataset["quality"]
    valid_cells = quality.cell_index.values[quality.values > qi_limit]

    # --- get angles and tilts ---
    angles = dataset["angle"].sel(cell_index=valid_cells).values
    tilts = dataset["tilt"].sel(cell_index=valid_cells).values

    # --- remove NaNs (but NOT based on tilt threshold) ---
    mask = np.isfinite(angles) & np.isfinite(tilts)
    angles = angles[mask]
    tilts = tilts[mask]

    # --- pairwise angle differences ---
    diff_matrix = np.abs(angles[:, None] - angles[None, :])
    diff_matrix = np.minimum(diff_matrix, 180 - diff_matrix)

    # --- pairwise average tilt ---
    tilt_matrix = (tilts[:, None] + tilts[None, :]) / 2

    # --- unique pairs ---
    iu = np.triu_indices(len(angles), k=1)

    diffs = diff_matrix[iu]
    avg_tilts = tilt_matrix[iu]

    # --- weights ---
    if weight_mode == "inverse":
        weights = 1.0 / (avg_tilts + eps)
    elif weight_mode == "exp":
        weights = np.exp(-k * avg_tilts)
    else:
        raise ValueError("weight_mode must be 'inverse' or 'exp'")

    return diffs, weights, valid_cells


# %%
def plot_weighted_histograms(diff_dict, weight_dict, bins=9, dataset_names=None):
    if dataset_names is None:
        dataset_names = list(diff_dict.keys())

    dataset_names = [name for name in dataset_names if len(diff_dict[name]) >= 10]

    bin_edges = np.linspace(0, 90, bins + 1)

    fig, ax = plt.subplots()

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))

    for name, color in zip(dataset_names, colors):
        diffs = diff_dict[name]
        weights = weight_dict[name]

        counts, _ = np.histogram(diffs, bins=bin_edges, weights=weights)

        # normalize to %
        counts = counts / counts.sum() * 100

        ax.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            align="edge",
            alpha=0.7,
            color=color,
            label=name,
        )

    ax.set_xlabel("Orientation difference (°)")
    ax.set_ylabel("Weighted %")
    ax.set_xlim(0, 90)
    ax.legend()
    plt.tight_layout()
    plt.show()


# %%
results = {
    name: compute_orientation_differences_weighted(
        dataset,
        qi_limit=QI_LIMIT,
        # tilt_limit=TILT_LIMIT,
        weight_mode="inverse",
    )
    for name, dataset in datasets.items()
}

all_diffs = {name: r[0] for name, r in results.items()}
all_weights = {name: r[1] for name, r in results.items()}
valid_cells = {name: r[2] for name, r in results.items()}

# %%
plot_weighted_histograms(
    all_diffs,
    all_weights,
    bins=9,
    dataset_names=[
        "dataset_2_12px_20Hz_shuffle",
        "dataset_6_12px_20Hz_shuffle",
        "dataset_5_12px_20Hz_shuffle",
        "dataset_3_12px_20Hz_shuffle",
    ],
)
