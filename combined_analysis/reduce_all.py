import polars as pl
import numpy as np
from pathlib import Path
import pickle
from organize.dataframe import Noise_store
from dim_reduction.circular_reduction import polar_transform
from location.channel_handling import masking_square
from polarspike import Overview
from tqdm import tqdm
import warnings
from rf_torch.parameters import Noise_Params
import matplotlib.pyplot as plt
from smoothing.gaussian import smooth_ker
from covariance.filtering import cov_filtering_sum


def check_border_constrains(cX, cY, alys_settings):
    # Check if the cell is too close to the border
    if cY > alys_settings["noise"][channel]["extended_size"][1] - int(
            alys_settings["data_dict"]["max_half_cut_px"][1]):
        cY = alys_settings["noise"][channel]["extended_size"][1] - int(
            alys_settings["data_dict"]["max_half_cut_px"][1])

    if cX > alys_settings["noise"][channel]["extended_size"][0] - int(
            alys_settings["data_dict"]["max_half_cut_px"][0]):
        cX = alys_settings["noise"][channel]["extended_size"][0] - int(
            alys_settings["data_dict"]["max_half_cut_px"][0])
    if cY < alys_settings["data_dict"]["max_half_cut_px"][1]:
        cY = alys_settings["data_dict"]["max_half_cut_px"][1]
    if cX < alys_settings["data_dict"]["max_half_cut_px"][0]:
        cX = alys_settings["data_dict"]["max_half_cut_px"][0]

    return cX, cY


# %% Load json file with basic information

project_root = Path(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_05_11_2025/Phase_01")

# %% Set parameters
thresholding_dict: dict[str, float] = {
    "threshold": 20,  # The quality threshold for the cells.
}
circular_reduction_dict: dict[str, int] = {
    "degree_step": 10,  # The step size in degrees for the circular reduction.
}
# % automatically
circular_reduction_dict["degree_bins"] = 360 // circular_reduction_dict["degree_step"]

data_dict: dict[
    str, Path | int | list[str] | list[
        str] | str | str | np.ndarray | np.ndarray | np.ndarray | tuple | tuple | int] = {
    "project_root": project_root,
    # The root folder of the project.
    "nr_channels": 2,
    "channels": [
        "white",
        "560_nm",

    ],  # The channels (wavelengths) that were recorded. bw="bw"
    "channel_paths": [
        "12px_20Hz_shuffle_idx_6",
        "4px_20Hz_40mins_shuffle_560_idx_8",
        # The paths to the channels, must match the order of the channels.
    ],  # The paths to the channels, must match the order of the channels.
    "output_folder": "noise_analysis",  # The folder to store the results in.
    "sta_file": "kernel.npy",
    "channel_sizes": np.zeros(3),  # This is just a placeholder
    "channel_scaling_factor": np.zeros(3),  # This is just a placeholder
    "cut_size_um": np.array([800, 800]),
    "max_cut_size_px": (0, 0),
    "max_half_cut_px": (0, 0),
    "max_noise_bins": 0
}

noise_dicts = {}

# %% load noise sizes and match sizes to common ground
channel_sizes = np.zeros((len(data_dict["channel_paths"]), 3))
for ch_idx, channel in enumerate(data_dict["channel_paths"]):
    channel_sizes[ch_idx] = np.load(data_dict["project_root"] / channel / "cell_0" / data_dict["sta_file"]).shape

# find largest of all dimensions
max_sizes = np.max(channel_sizes, axis=0)
scaling_factors = max_sizes / channel_sizes

data_dict["channel_sizes"] = channel_sizes
data_dict["channel_scaling_factor"] = scaling_factors

# %%

# Example: Define parameters for each noise stimulus
# You can iterate over your noise stimuli and populate this
for noise_idx, noise_location in enumerate(data_dict["channel_paths"]):  # Add more as needed
    noise_params = Noise_Params.load_from_root_json(project_root / noise_location)
    noise_params.project_root = project_root

    noise_dicts[data_dict["channels"][noise_idx]] = {
        "cut_size": data_dict["cut_size_um"] / noise_params["noise_pixel_size"],
        # Adjust based on your Noise_Params attributes
        "total_sta_len": noise_params.total_sta_len,
        "post_spike_len": noise_params.post_spike_len
        # Add other relevant parameters
    }

    noise_dicts[data_dict["channels"][noise_idx]]["half_cut"] = noise_dicts[data_dict["channels"][noise_idx]][
                                                                    "cut_size"] // 2
    noise_dicts[data_dict["channels"][noise_idx]]["border_cut"] = noise_dicts[data_dict["channels"][noise_idx]][
                                                                      "cut_size"] // 4
    noise_dicts[data_dict["channels"][noise_idx]]["buffer"] = noise_dicts[data_dict["channels"][noise_idx]][
        "border_cut"]
    noise_dicts[data_dict["channels"][noise_idx]]["extended_size"] = (
        data_dict["channel_sizes"][noise_idx, 1] + noise_dicts[data_dict["channels"][noise_idx]]["cut_size"][0],
        data_dict["channel_sizes"][noise_idx, 2] + noise_dicts[data_dict["channels"][noise_idx]]["cut_size"][1],
    )

alys_settings = {
    "noise": noise_dicts,
    "thresholding": thresholding_dict,
    "circular_reduction": circular_reduction_dict,
    "data_dict": data_dict,
}

# Make a new folder for the results
output_folder = (
        alys_settings["data_dict"]["project_root"] / alys_settings["data_dict"]["output_folder"]
)
Path.mkdir(output_folder, exist_ok=True)

# Save the settings
with open(output_folder / "settings.pkl", "wb") as f:
    pickle.dump(alys_settings, f)
# %% Step one, load the quality data
positions_x = []
positions_y = []
quality_list = []
good_cells = []
nr_cells = np.zeros(len(alys_settings["data_dict"]["channel_paths"]), dtype=int)
paths = [
    alys_settings["data_dict"]["project_root"] / path
    for path in alys_settings["data_dict"]["channel_paths"]
]
for path_idx, path in enumerate(paths):
    try:
        quality = np.load(path / "quality.npy")
        nr_cells[path_idx] = quality.shape[0]
    except FileNotFoundError:
        exit(f"Quality file not found in {path}")
    good_cells.append(
        quality[quality[:, 0] > alys_settings["thresholding"]["threshold"], 1].astype(
            int
        )
    )
    positions_x.append(quality[:, 2])
    positions_y.append(quality[:, 3])
    quality_list.append(quality[:, 0])

assert np.all(
    nr_cells == nr_cells[0]
), "The number of cells is not equal across channels, all channels must have the same number of cells."

good_cells = np.unique(np.concatenate(good_cells))

positions_x = np.array(positions_x, dtype=int)
positions_y = np.array(positions_y, dtype=int)
quality_all = np.array(quality_list)

quality_below_threshold = quality_all < alys_settings["thresholding"]["threshold"]

# For each cell, if any channel is below threshold, replace with position from best quality channel
for cell_idx in range(positions_x.shape[1]):
    below_mask = quality_below_threshold[:, cell_idx]
    if np.any(below_mask) and not np.all(below_mask):
        # Find channel with highest quality for this cell
        best_channel = np.argmax(quality_all[:, cell_idx])
        # Replace positions for channels below threshold
        positions_x[below_mask, cell_idx] = np.round(positions_x[best_channel, cell_idx] * (
                1 / alys_settings["data_dict"]["channel_scaling_factor"][below_mask, 1])).astype(int)
        positions_y[below_mask, cell_idx] = np.round(positions_y[best_channel, cell_idx] * (
                1 / alys_settings["data_dict"]["channel_scaling_factor"][below_mask, 2])).astype(int)

# %% Sanity plot
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="Mean of empty slice",
    )
    best_cell = np.nanargmax(np.nanmean(quality_all, axis=0))
best_channel = np.argmax(quality_all[:, best_cell])

sta = np.load(alys_settings["data_dict"]["project_root"] / data_dict["channel_paths"][
    best_channel] / f"cell_{best_cell}" / "kernel.npy")

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(np.var(sta, axis=0), cmap="Greys")
ax.scatter(positions_x[0, best_cell], positions_y[0, best_cell], color="red")
ax.set_title("Sanity plot, should show best cell RF with matching red marker on RF")
fig.show()
# %% Create polars dataframe to store all the results
noise_store = Noise_store(alys_settings, nr_cells=nr_cells[0])
# df.write_parquet(output_folder / "noise_df")
noise_store.save()

# %% Add positions to the dataframe
positions = np.vstack((positions_x.flatten(), positions_y.flatten())).T
positions_series = pl.Series("positions", positions, dtype=pl.Array(pl.UInt16, 2))
quality_series = pl.Series("quality", quality_all.flatten(), dtype=pl.Float64)
# Add the positions to the dataframe
noise_store.df = noise_store.df.with_columns(positions_series)
noise_store.df = noise_store.df.with_columns(quality_series)

cut_size = np.array([0, 0])
noise_bins = 0
for channel in noise_dicts:
    if noise_dicts[channel]["cut_size"][0] > cut_size[0]:
        cut_size[0] = noise_dicts[channel]["cut_size"][0]
    if noise_dicts[channel]["cut_size"][1] > cut_size[1]:
        cut_size[1] = noise_dicts[channel]["cut_size"][1]
    if noise_dicts[channel]["total_sta_len"] > noise_bins:
        noise_bins = noise_dicts[channel]["total_sta_len"]
alys_settings["data_dict"]["max_cut_size_px"] = tuple(cut_size)
alys_settings["data_dict"]["max_half_cut_px"] = tuple((cut_size / 2).astype(int))
alys_settings["data_dict"]["max_noise_bins"] = noise_bins
cm_most_important_store = np.zeros(
    (
        len(noise_store.df),
        alys_settings["data_dict"]["max_cut_size_px"][0],
        alys_settings["data_dict"]["max_cut_size_px"][1]
    )
)
# cm_signed_store = np.zeros_like(cm_most_important_store)
sta_single_store = np.zeros((len(noise_store.df), alys_settings["data_dict"]["max_noise_bins"]))

recording = Overview.Recording.load(
    alys_settings["data_dict"]["project_root"] / "overview"
)

circular_reduction_dict["max_radius"] = int(
    np.sqrt(np.max(alys_settings["data_dict"]["max_cut_size_px"]) ** 2 + np.max(
        alys_settings["data_dict"]["max_cut_size_px"]) ** 2) // 2
)
# %% Step two sta extension and covariance matrix
# This is a for loop, because the data is too large for running in parallel.
for row in tqdm(noise_store.df.iter_rows(), total=len(noise_store.df), desc="Calculating covariance matrices"):
    cell_idx = row[noise_store.df.get_column_index("cell_index")]

    channel = row[noise_store.df.get_column_index("channel")]

    try:
        sta_data = np.load(
            noise_store.df.filter((pl.col("cell_index") == cell_idx) & (pl.col("channel") == channel))[
                "sta_path"].item())
    except FileNotFoundError:
        print(f"File not found for item {cell_idx}")
        continue
    original_sta_shape = sta_data.shape
    sta_extended = (
        np.full(
            (
                original_sta_shape[0],
                original_sta_shape[1] + alys_settings["data_dict"]["max_cut_size_px"][0],
                original_sta_shape[2] + alys_settings["data_dict"]["max_cut_size_px"][1],
            ), np.median(sta_data), dtype=float
        )
    )
    sta_extended[
        :,
        alys_settings["data_dict"]["max_half_cut_px"][0]: -alys_settings["data_dict"]["max_half_cut_px"][0],
        alys_settings["data_dict"]["max_half_cut_px"][1]: -alys_settings["data_dict"]["max_half_cut_px"][1],
    ] = sta_data
    sta_data = sta_extended
    del sta_extended
    cX, cY = (
            noise_store.df.filter((pl.col("cell_index") == cell_idx) & (pl.col("channel") == channel))[
                "positions"].to_numpy()[0] + np.array(alys_settings["data_dict"]["max_half_cut_px"])
    )

    # Check if the cell is too close to the border
    cX, cY = check_border_constrains(cX, cY, alys_settings)
    #
    # Create mask and extract subset
    mask = masking_square(
        int(alys_settings["noise"][channel]["extended_size"][1]),
        int(alys_settings["noise"][channel]["extended_size"][0]),
        (int(cX), int(cY)),
        alys_settings["data_dict"]["max_cut_size_px"][1],
        alys_settings["data_dict"]["max_cut_size_px"][0],
    )
    subset_flat = sta_data[:, mask]
    subset = np.reshape(
        subset_flat,
        (
            original_sta_shape[0],
            alys_settings["data_dict"]["max_cut_size_px"][1],
            alys_settings["data_dict"]["max_cut_size_px"][0],
        ),
    )

    if cell_idx == 0:
        fig, ax = plt.subplots(figsize=(10, 10))
        variance_map = np.var(smooth_ker(subset), axis=0)
        ax.imshow(np.var(subset, axis=0), cmap="gray")
        ax.set_title("Sanity plot subset")
        max_pos = np.unravel_index(np.argmax(variance_map), variance_map.shape)
        ax.scatter(max_pos[1], max_pos[0], color='red', s=100, marker='x')
        fig.show()

    # Nail the location of the center
    variance_map = np.var(smooth_ker(subset), axis=0)
    cY_subset, cX_subset = np.unravel_index(np.argmax(variance_map), variance_map.shape)
    cX = cX + cX_subset - alys_settings["data_dict"]["max_half_cut_px"][1]
    cY = cY + cY_subset - alys_settings["data_dict"]["max_half_cut_px"][0]
    cX, cY = check_border_constrains(cX, cY, alys_settings)

    # update the position in the dataframe
    noise_store.df = noise_store.df.with_columns(
        pl.when((pl.col("cell_index") == cell_idx) & (pl.col("channel") == channel))
        .then(pl.lit([cX, cY]))
        .otherwise(pl.col("positions"))
        .alias("positions")
    )
    mask = masking_square(
        int(alys_settings["noise"][channel]["extended_size"][1]),
        int(alys_settings["noise"][channel]["extended_size"][0]),
        (int(cX), int(cY)),
        alys_settings["data_dict"]["max_cut_size_px"][1],
        alys_settings["data_dict"]["max_cut_size_px"][0],
    )
    subset_flat = sta_data[:, mask]
    subset = np.reshape(
        subset_flat,
        (
            original_sta_shape[0],
            alys_settings["data_dict"]["max_cut_size_px"][1],
            alys_settings["data_dict"]["max_cut_size_px"][0],
        ),
    )

    stimulus_id = int(noise_store.df[cell_idx, "sta_path"].split("_")[-2][:1])
    nr_of_spikes = \
        recording.spikes_df.query(
            f"stimulus_index=={stimulus_id}&cell_index=={noise_store.df[cell_idx, 'cell_index']}")[
            "nr_of_spikes"].values[0]
    sta_per_spike = (subset / nr_of_spikes - 0.5)

    # Calculate the covariance matrix
    # cm = np.max(sta_per_spike ** 2, axis=0)
    # most_important_pixel = np.unravel_index(np.argmax(cm), cm.shape)
    # max_time = np.argmax(np.abs(sta_per_spike[:, most_important_pixel[0], most_important_pixel[1]]))
    # cm_signed = sta_per_spike[max_time, :, :]
    #
    # sta_single = subset[:, most_important_pixel[0], most_important_pixel[1]]
    # sta_single_store[cell_idx, :] = sta_single
    #
    # # Fill in self covariance (variance) of the most important pixel
    # cm_most_important_store[cell_idx, :, :] = cm
    # cm_signed_store[cell_idx, :, :] = cm_signed
    # Calculate covariance across pixels (time x pixels)

    time_bins, h, w = sta_per_spike.shape
    flat = sta_per_spike.reshape(time_bins, h * w).astype(np.float32, copy=False)
    flat -= flat.mean(axis=0, keepdims=True)

    # First principal component loadings over pixels
    try:
        _, s, Vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        print(f"SVD did not converge for cell {cell_idx}, skipping.")
        continue
    loadings = Vt[0]  # shape: (h*w,)
    most_idx = int(np.argmax(np.abs(loadings)))

    # Covariance with the most-important pixel without forming full cov matrix
    cov_with_most = (flat.T @ flat[:, most_idx]) / (time_bins - 1)
    cm_most_important_store[cell_idx, :, :] = cov_with_most.reshape(h, w)

    # Store STA time course of the selected pixel
    pix_y, pix_x = divmod(most_idx, w)
    sta_single_store[cell_idx, :time_bins] = sta_per_spike[:, pix_y, pix_x]

    #

# %% Add the cm_most_important to the dataframe
cm_most_important_store_flat = np.array(
    [
        cm_most_important_store[idx].flatten()
        for idx in range(cm_most_important_store.shape[0])
    ]
)
# cm_signed_store_flat = np.array(
#     [
#         cm_signed_store[idx].flatten() for idx in range(cm_signed_store.shape[0])
#     ])
# %%
cm_most_important_series = pl.Series(
    "cm_most_important",
    cm_most_important_store_flat,
    dtype=pl.Array(pl.Float64,
                   (alys_settings["data_dict"]["max_cut_size_px"][0] * alys_settings["data_dict"]["max_cut_size_px"][
                       1],)),
)

# cm_signed_series = pl.Series(
#     "cm_signed",
#     cm_signed_store_flat,
#     dtype=pl.Array(pl.Float64, alys_settings["noise"]["cut_size"] ** 2),
# )

sta_single_series = pl.Series(
    "sta_single",
    sta_single_store,
    dtype=pl.Array(pl.Float64, sta_single_store.shape[1]),
)

# %%
noise_store.df = noise_store.df.with_columns(cm_most_important_series)
noise_store.df = noise_store.df.with_columns(sta_single_series)
# noise_store.df = noise_store.df.with_columns(cm_signed_series)
# %%
noise_store.save()

# %%
center_outline = np.zeros(
    (len(noise_store.df), alys_settings["circular_reduction"]["degree_bins"])
)
surround_outline = np.zeros_like(center_outline)
in_out_outline = np.zeros(
    (len(noise_store.df), alys_settings["circular_reduction"]["max_radius"])
)
center_std = np.zeros((len(noise_store.df)))
surround_std = np.zeros_like(center_std)
for idx in range(len(noise_store.df)):
    polar_cm = polar_transform(
        cm_most_important_store[idx, :, :],
        (alys_settings["data_dict"]["max_half_cut_px"][1], alys_settings["data_dict"]["max_half_cut_px"][0]),
        alys_settings["circular_reduction"]["max_radius"],
    )
    entries = []
    for deg_10 in range(0, 360, alys_settings["circular_reduction"]["degree_step"]):
        entries.append(
            np.mean(
                polar_cm[
                    deg_10: deg_10
                            + alys_settings["circular_reduction"]["degree_step"],
                    :,
                ],
                axis=0,
            )
        )
    entries = np.asarray(entries)
    entries_pos = entries.copy()
    entries_pos[entries_pos < 0] = 0
    entries_mean = (
            np.mean(entries_pos / np.max(entries_pos), axis=1) * entries_pos.shape[1]
    )

    entries_neg = entries.copy()
    entries_neg[entries_neg > 0] = 0
    entries_mean_neg = (
            np.mean(entries_neg / np.min(entries_neg), axis=1) * entries_neg.shape[1]
    )
    entries_mean[np.isnan(entries_mean)] = 0
    entries_mean_neg[np.isnan(entries_mean_neg)] = 0
    caused_var_pos = []
    caused_var_neg = []
    for i in range(36):
        caused_var_pos.append(
            np.std(entries[i, :]) / np.std(entries[i, int(np.ceil(entries_mean[i])):])
        )
        caused_var_neg.append(
            np.std(entries[i, :])
            / np.std(entries[i, int(np.ceil(entries_mean_neg[i])):])
        )
    center_std[idx] = np.mean(caused_var_pos)
    surround_std[idx] = np.mean(caused_var_neg)

    center_outline[idx, :] = entries_mean
    surround_outline[idx, :] = entries_mean_neg
    in_out_outline[idx, :] = np.mean(entries, axis=0)

# %% Add results to the dataframe
center_outline = pl.Series(
    "center_outline",
    center_outline,
    dtype=pl.Array(pl.Float64, alys_settings["circular_reduction"]["degree_bins"]),
)
surround_outline = pl.Series(
    "surround_outline",
    surround_outline,
    dtype=pl.Array(pl.Float64, alys_settings["circular_reduction"]["degree_bins"]),
)
in_out_outline = pl.Series(
    "in_out_outline",
    in_out_outline,
    dtype=pl.Array(pl.Float64, alys_settings["circular_reduction"]["max_radius"]),
)
rf_std = pl.Series(
    "rf_std", np.max(np.vstack([center_std, surround_std]), axis=0), dtype=pl.Float64
)
nr_columns = len(noise_store.df.columns)

for series_idx, series in enumerate(
        [center_outline, surround_outline, in_out_outline, rf_std]
):
    noise_store.df.insert_column(nr_columns + series_idx, series)

# %% Update good cells

noise_store.df = noise_store.df.with_columns(
    (pl.col("cm_most_important").arr.max() > 0).alias("good_cells")
)
noise_store.save()

# %% Calculate some stats
degrees = np.arange(0, 360, alys_settings["circular_reduction"]["degree_step"])


def calc_size(outline):
    outline = np.array(outline)
    for deg_idx in range(len(degrees[:-2])):
        size = outline[deg_idx] * outline[deg_idx + 1] * np.sin(np.deg2rad(10))
        size = np.stack(
            [
                size,
                outline[-1,] * outline[0] * np.sin(np.deg2rad(10)),
            ]
        )
    cell_size = np.sum(size, axis=0)
    return cell_size


# %%
noise_store.df = noise_store.df.with_columns(
    (pl.col("center_outline").map_elements(calc_size, pl.Float64)).alias("center_size")
)
# %%
noise_store.df = noise_store.df.with_columns(
    pl.col("surround_outline")
    .map_elements(calc_size, pl.Float64)
    .alias("surround_size")
)
# %%
noise_store.save()
