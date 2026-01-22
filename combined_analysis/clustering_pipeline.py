from clustering import rf_clustering
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from pickle import load as load_pickle
from importlib import reload
import numpy as np
import xarray as xr
from organize.configs import Recording_Config
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
import plotly.express as px
from scaling.cuts import get_non_nan_slices, cutout_nans
from tqdm import tqdm
import matplotlib
from aquarel import load_theme

coolwarm_heatmap = matplotlib.colormaps["coolwarm"]
heatmap_max = coolwarm_heatmap(1.0)
heatmap_min = coolwarm_heatmap(0.0)
theme = load_theme("scientific")
theme.apply()

channel_colours = ["red", "green", "black"]


# %% Functions
def zoom_extent(xarray, zoom_factor):
    """Return extent [xmin, xmax, ymin, ymax] zoomed around the center."""
    x_coords = xarray.coords["x"]
    y_coords = xarray.coords["y"]
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2
    x_half_range = (x_coords.max() - x_coords.min()) / 2 / zoom_factor
    y_half_range = (y_coords.max() - y_coords.min()) / 2 / zoom_factor
    return [
        x_center - x_half_range,
        x_center + x_half_range,
        y_center - y_half_range,
        y_center + y_half_range,
    ]


def all_arrays_identical(arrays):
    first = arrays[0]
    return all(np.array_equal(a, first) for a in arrays[1:])


# %%
for_clustering = [
    "center_outline_um",
    "in_out_outline_um",
    "surround_outline_um",
    "sta_single_pixel",
]
# %%
# We can load datasets from several different recordings.
path_to_data = [
    Path(
        "/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_13_11_2025/Phase_00/noise_analysis/noise_data.nc"
    ),
    Path(
        r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_18_11_2025/Phase_00/noise_analysis/noise_data.nc"
    ),
]

# Load the config files:
configs = [Recording_Config.load_from_root_json(path.parent) for path in path_to_data]


# %% check which channels can be used for clustering matching channel names
channel_names = []
for config in configs:
    channel_names = channel_names + config.channel_names
unique_channel_names, counts = np.unique(np.asarray(channel_names), return_counts=True)

valid_channels = unique_channel_names[counts == len(configs)]
print(
    f"Found the following consistent channels across recordings: {valid_channels}. These will be used for clustering."
)
# %%
# 1. Load and align recordings
datasets = []
for path, config in zip(path_to_data, configs):
    # Lazy load only the variables you need
    ds = xr.open_dataset(path)[
        for_clustering
    ]  # Note: This will not interpolate cm_most_important or rms (This can be done later for subsamples that shall be plotted)
    # Filter for valid channels and add the recording coordinate
    ds = ds.sel(channel=valid_channels).expand_dims(recording=[config.overview.name])
    datasets.append(ds)

# 2. Global Interpolation (Align grids across recordings once)
# Create a master grid for all spatial/temporal dimensions
ref_ds = datasets[0]
target_coords = {
    dim: max([d.coords[dim].values for d in datasets], key=len)
    for dim in ["time_max", "degrees", "radius"]
    if dim in ref_ds.dims
}

# Interpolate all datasets to match the master grid
aligned_datasets = [ds.interp(target_coords, method="linear") for ds in datasets]

# 3. Combine and Stack
# Concat along 'recording' first.
# Resulting dims: (recording, channel, cell_index, x, y, etc.)
combined_ds = xr.concat(aligned_datasets, dim="recording")
# fill dimensions that are only 0 with NaNs

# Create your sample index (Observations)
# Size will be: nr_recordings * nr_cells
obs_stacked = combined_ds.stack(index=("recording", "cell_index"))

# Collapse all variables and all other dimensions (channel, x, y...) into the feature vector
# Size will be: (index, all_remaining_features)
final_array = obs_stacked.to_stacked_array("all_features", sample_dims=["index"])
# %% Drop any cells with NaNs
final_array = final_array.fillna(0)  # TODO Need to check for all ==0
final_array[np.all(final_array.values == 0, axis=1), :] = np.nan
# cells that have all NaNs across all features are dropped
final_array = final_array.dropna("index", how="all")

print(
    f"Final data array shape for clustering: {final_array.shape[0]} cells and {final_array.shape[1]} features."
)
# %%
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest()
outlier_labels = iso_forest.fit_predict(final_array)

# %%
final_array_clean = final_array.sel(
    index=final_array.coords["index"].values[outlier_labels == 1]
)
inliner_indices = final_array_clean.coords["index"].values
inliner_indices_pd = inlier_indices = pd.MultiIndex.from_tuples(
    final_array_clean.coords["index"].values, names=["recording", "cell_index"]
)  # This extracts the index as an actual pandas MultiIndex for easier handling later on
# %% Scale data
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(final_array_clean)

# %% PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_data_scaled)  # only use inliers
# plot PCA result
inlier_indices = final_array_clean.coords["index"].values
recording_ids = [idx[0] for idx in inlier_indices]
cell_indices = [idx[1] for idx in inlier_indices]
fig = px.scatter(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hover_data={"recording": recording_ids, "cell_index": cell_indices},
    labels={"x": "PC1", "y": "PC2"},
)
fig.update_traces(marker=dict(size=3))
fig.show()

# %% Calculate distance matrix
from sklearn.metrics import pairwise_distances

distance_matrix = pairwise_distances(all_data_scaled, metric="cosine")
fig, ax = plt.subplots()
cax = ax.imshow(distance_matrix, cmap="viridis")
fig.colorbar(cax, ax=ax, label="Cosine Distance")
ax.set_title("Distance Matrix")
fig.show()

# %% Clustering
from sklearn.cluster import AffinityPropagation

clustering_model = AffinityPropagation(damping=0.5, random_state=42)
labels = clustering_model.fit_predict(all_data_scaled)

# %% Plot clusters in pca space
fig = px.scatter(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    color=labels.astype(str),
    hover_data={"recording": recording_ids, "cell_index": cell_indices},
    labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
)
fig.update_traces(marker=dict(size=3))
fig.show()
# %% Now I need to cluster all the data
# Here we plot one figure per cluster. The first figure shows heatmaps of cm_most_important and rms.
# This plots quite a bit of data and the plotting can take a couple of minutes.
# Parameters:
top_cell_nr = 5  # How many cells to plot per cluster (maximally)
zoom = (
    2  # This can zoom the heatmaps by a factor, set to 1 if you want to original size
)
datasets_heatmaps = {}
heatmap_names = [
    "cm_most_important",
    "rms",
]

data_to_load = heatmap_names + [
    "quality"
]  # heatmap names and quality for choosing the best cells to plot
save_path = Path(r"/media/mawa/New Volume/cluster_all/test")
for path, config in zip(path_to_data, configs):
    # Lazy load only the variables you need
    ds = xr.open_dataset(path, chunks={"channel": 100})[data_to_load]
    # Filter for valid channels and add the recording coordinate
    ds = ds.sel(channel=valid_channels).expand_dims(recording=[config.overview.name])
    datasets_heatmaps[config.overview.name] = ds

unique_labels = np.unique(labels)
for label_idx, label in tqdm(
    enumerate(unique_labels), desc="Plotting clusters", total=len(unique_labels)
):
    max_cell_nr_in_cluster = min(np.sum(labels == label), top_cell_nr)
    required_height = max_cell_nr_in_cluster * 2 * 2  # 2 rows per cell, 5 inches each
    required_width = required_height // (max_cell_nr_in_cluster // 2 + 1)
    fig, ax = plt.subplots(
        nrows=max_cell_nr_in_cluster * len(heatmap_names),  # times two for cm and rms
        ncols=valid_channels.shape[0] + 1,  # +1 for overlay
        figsize=(required_width, required_height),
        sharey="row",
        sharex="row",
        gridspec_kw={
            "hspace": 0.05,  # Small spacing within cell pairs
            "wspace": 0.05,
        },
    )
    # Add extra spacing between cell groups
    for cell_idx in range(1, max_cell_nr_in_cluster):
        row_idx = cell_idx * 2
        for col_idx in range(ax.shape[1]):
            pos = ax[row_idx, col_idx].get_position()
            ax[row_idx, col_idx].set_position(
                [pos.x0, pos.y0 - 0.02 * cell_idx, pos.width, pos.height]
            )
            ax[row_idx + 1, col_idx].set_position(
                [
                    ax[row_idx + 1, col_idx].get_position().x0,
                    ax[row_idx + 1, col_idx].get_position().y0 - 0.02 * cell_idx,
                    ax[row_idx + 1, col_idx].get_position().width,
                    ax[row_idx + 1, col_idx].get_position().height,
                ]
            )
    cluster_indices = labels == label
    cluster_cell_indices = inliner_indices_pd[cluster_indices]
    unique_recordings = cluster_cell_indices.unique("recording")
    cm_list, rms_list, quality_list = [], [], []
    for rec in unique_recordings:
        ds_heatmap = datasets_heatmaps[rec]
        cell_indices_rec = cluster_cell_indices[
            cluster_cell_indices.get_level_values("recording") == rec
        ].get_level_values("cell_index")
        cm_data = ds_heatmap["cm_most_important"].sel(
            cell_index=cell_indices_rec
        )  # this is still lazy
        rms_data = ds_heatmap["rms"].sel(cell_index=cell_indices_rec)
        quality = ds_heatmap["quality"].sel(cell_index=cell_indices_rec)
        quality_list.append(quality.compute())
        cm_list.append(cm_data)
        rms_list.append(rms_data)
    # Concatenate all recordings for this cluster
    cm_cluster = xr.concat(cm_list, dim="recording").compute()  # Materialize the data
    rms_cluster = xr.concat(rms_list, dim="recording").compute()
    quality_cluster = xr.concat(quality_list, dim="recording")
    # Sort all datasets by average quality across channels
    avg_quality = quality_cluster.mean(dim="channel")
    avg_quality_flat = avg_quality.stack(flat_index=("recording", "cell_index"))
    sorted_indices = avg_quality_flat.argsort()[::-1][
        :max_cell_nr_in_cluster
    ]  # Top top_cell_nr highest quality
    selected_indices = avg_quality_flat.coords["flat_index"].values[
        sorted_indices.values
    ]
    cm_cluster_flat = cm_cluster.stack(flat_index=("recording", "cell_index"))
    rms_cluster_flat = rms_cluster.stack(flat_index=("recording", "cell_index"))

    # Finally, we can plot
    plotting_position = 0
    for cell_to_plot in range(max_cell_nr_in_cluster):
        for channel_idx, channel in enumerate(valid_channels):
            cm_cutout = None
            rms_cutout = None
            try:
                cm_cutout = cutout_nans(
                    cm_cluster_flat.sel(
                        {
                            "channel": channel,
                            "flat_index": selected_indices[cell_to_plot],
                        }
                    )
                )
                cutout_min_max = np.nanmax(np.abs(cm_cutout.values))
            except IndexError:
                ax[plotting_position, channel_idx].axis("off")

            try:
                rms_cutout = cutout_nans(
                    rms_cluster_flat.sel(
                        {
                            "channel": channel,
                            "flat_index": selected_indices[cell_to_plot],
                        }
                    )
                )
            except IndexError:
                ax[plotting_position + 1, channel_idx].axis("off")

            if (
                cm_cutout is None and rms_cutout is None
            ):  # No data, so dont need to plot
                # Switch subplot off
                continue
            # ax[plotting_position, channel_idx].set_title(
            #     f"Channel {channel}, Recording {selected_indices[cell_to_plot][0]}, Cell {selected_indices[cell_to_plot][1]}"
            # )
            ax[plotting_position, channel_idx].imshow(
                cm_cutout.values,
                extent=zoom_extent(cm_cutout, zoom),
                cmap="coolwarm",
                vmin=-cutout_min_max,
                vmax=cutout_min_max,
            )
            # remove ticks, labels, frame, etc.
            ax[plotting_position, channel_idx].set_xticks([])
            ax[plotting_position, channel_idx].set_yticks([])
            ax[plotting_position, channel_idx].spines["top"].set_visible(False)
            ax[plotting_position, channel_idx].spines["right"].set_visible(False)
            ax[plotting_position, channel_idx].spines["left"].set_visible(False)
            ax[plotting_position, channel_idx].spines["bottom"].set_visible(False)
            ax[plotting_position, -1].set_xticks([])
            ax[plotting_position, -1].set_yticks([])
            ax[plotting_position, -1].spines["top"].set_visible(False)
            ax[plotting_position, -1].spines["right"].set_visible(False)
            ax[plotting_position, -1].spines["left"].set_visible(False)
            ax[plotting_position, -1].spines["bottom"].set_visible(False)

            ax[plotting_position + 1, channel_idx].imshow(
                rms_cutout.values,
                extent=zoom_extent(rms_cutout, zoom),
                cmap="Greys",
            )
            ax[plotting_position + 1, channel_idx].set_xticks([])
            ax[plotting_position + 1, channel_idx].set_yticks([])
            ax[plotting_position + 1, channel_idx].spines["top"].set_visible(False)
            ax[plotting_position + 1, channel_idx].spines["right"].set_visible(False)
            ax[plotting_position + 1, channel_idx].spines["left"].set_visible(False)
            ax[plotting_position + 1, channel_idx].spines["bottom"].set_visible(False)
        if cm_cutout is None and rms_cutout is None:  # No data, so dont need to plot
            # Switch subplot off
            continue
        # Now plot the overlay in the last column
        rgb_image = rms_cluster_flat.sel(
            {
                "flat_index": selected_indices[cell_to_plot],
            }
        ).transpose("x", "y", "channel")
        rgb_image_mean = rgb_image.mean(dim="channel", skipna=True)
        rgb_slices = get_non_nan_slices(rgb_image_mean.values)
        rgb_image_cut = rgb_image.isel(x=rgb_slices[0], y=rgb_slices[1])
        rgb_image_cut = rgb_image_cut.fillna(0)
        # change to uint8
        normalized = (
            (
                rgb_image_cut.values
                - rgb_image_cut.values.min(axis=(0, 1), keepdims=True)
            )
            / (
                rgb_image_cut.values.max(axis=(0, 1), keepdims=True)
                - rgb_image_cut.values.min(axis=(0, 1), keepdims=True)
            )
            * 255
        )
        rgb_image_cut = rgb_image_cut.copy(data=normalized.astype(np.uint8))
        ax[plotting_position + 1, -1].imshow(
            rgb_image_cut.values,
            extent=zoom_extent(rgb_image_cut, zoom),
        )
        ax[plotting_position + 1, -1].set_xticks([])
        ax[plotting_position + 1, -1].set_yticks([])
        ax[plotting_position + 1, -1].spines["top"].set_visible(False)
        ax[plotting_position + 1, -1].spines["right"].set_visible(False)
        ax[plotting_position + 1, -1].spines["left"].set_visible(False)
        ax[plotting_position + 1, -1].spines["bottom"].set_visible(False)

        plotting_position += 2
    fig.savefig(save_path / f"cluster_{label}_heatmaps.png", dpi=300)
    plt.close(fig)

# close heatmap datasets
for ds in datasets_heatmaps.values():
    ds.close()
# %%
# Plot additional statistics per cluster
datasets_stats = []
for path, config in zip(path_to_data, configs):
    # Lazy load only the variables you need
    ds = xr.open_dataset(path)[for_clustering + ["quality"]]
    # Filter for valid channels and add the recording coordinate
    ds = ds.sel(channel=valid_channels).expand_dims(recording=[config.overview.name])
    datasets_stats.append(ds)

combined_ds_stats = xr.concat(datasets_stats, dim="recording")
combined_ds_stats = combined_ds_stats.stack(
    index=("recording", "cell_index")
)  # Create multiindex for easier indexing
column_ratios = [1, 1, 1, 1, 1]
n_cols = 5
n_rows = len(unique_labels)
global_line_width = 0.1
fig, axs = plt.subplots(
    ncols=n_cols,
    nrows=n_rows,
    figsize=(10, 25),
    sharex="col",
    sharey="col",
    # 1. Define custom column widths using gridspec_kw
    gridspec_kw={"width_ratios": column_ratios},
)
# Transform axes to polar
polar_cols = [0, 1]
for row in range(n_rows):
    for col in polar_cols:
        axs[row, col].remove()
        subplot_index = row * n_cols + col + 1
        axs[row, col] = fig.add_subplot(
            n_rows, n_cols, subplot_index, projection="polar"
        )
# Remove all ticks
for ax in axs.flatten():
    ax.tick_params(
        axis="both",  # Apply to both x and y axes
        which="both",  # Apply to both major and minor ticks
        bottom=False,  # Remove bottom ticks
        top=False,  # Remove top ticks
        left=False,  # Remove left ticks
        right=False,  # Remove right ticks
        labelbottom=False,  # Remove x-axis labels
        labelleft=False,  # Remove y-axis labels
    )

for label_idx, label in enumerate(unique_labels):
    cluster_indices = labels == label
    cluster_cell_indices = inliner_indices_pd[cluster_indices]
    selected_data = combined_ds_stats.sel(index=list(cluster_cell_indices))
    # first plot: Center outline
    axs[label_idx, 0].set_title(f"Cluster {label} Center Outlines")
    for channel_idx, channel in enumerate(valid_channels):
        # Plot center outlines
        channel_data = selected_data.sel(channel=channel)
        axs[label_idx, 0].plot(
            np.deg2rad(channel_data.coords["degree"].values),
            channel_data["center_outline_um"].values,
            c=channel_colours[channel_idx],
            linewidth=global_line_width,
        )
        # plot mean
        axs[label_idx, 0].plot(
            np.deg2rad(channel_data.coords["degree"].values),
            channel_data["center_outline_um"].mean(dim="index").values,
            c="black",
            linewidth=global_line_width * 5,
        )
        axs[label_idx, 0].set_theta_zero_location("E")
        axs[label_idx, 0].set_rgrids(
            [
                int(
                    np.round(
                        channel_data["center_outline_um"].max().item() * i / 3, -1
                    ),
                )
                for i in range(1, 4)
            ]
        )
        # Plot surround outlines
        axs[label_idx, 1].plot(
            np.deg2rad(channel_data.coords["degree"].values),
            channel_data["surround_outline_um"].values,
            c=channel_colours[channel_idx],
            linewidth=global_line_width,
        )
        # plot mean
        axs[label_idx, 1].plot(
            np.deg2rad(channel_data.coords["degree"].values),
            channel_data["surround_outline_um"].mean(dim="index").values,
            c="black",
            linewidth=global_line_width * 5,
        )
        axs[label_idx, 1].set_theta_zero_location("E")
        axs[label_idx, 1].set_rgrids(
            [
                int(
                    np.round(
                        channel_data["surround_outline_um"].max().item() * i / 3, -1
                    ),
                )
                for i in range(1, 4)
            ]
        )
        # PLot in out outline
        axs[label_idx, 2].plot(
            channel_data.coords["radius"].values,
            channel_data["in_out_outline_um"].values,
            c=channel_colours[channel_idx],
            linewidth=global_line_width,
        )
        # plot mean
        axs[label_idx, 2].plot(
            channel_data.coords["radius"].values,
            channel_data["in_out_outline_um"].mean(dim="index").values,
            c="black",
            linewidth=global_line_width * 5,
        )
        # Plot sta
        axs[label_idx, 3].plot(
            channel_data.coords["time_max"].values,
            channel_data["sta_single_pixel"].values,
            c=channel_colours[channel_idx],
            linewidth=global_line_width,
        )
        # plot mean
        axs[label_idx, 3].plot(
            channel_data.coords["time_max"].values,
            channel_data["sta_single_pixel"].mean(dim="index").values,
            c="black",
            linewidth=global_line_width * 5,
        )
fig.savefig(save_path / f"cluster_{label}_stats.png", dpi=300)
