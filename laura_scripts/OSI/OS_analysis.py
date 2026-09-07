# Imports

from pathlib import Path
from polarspike import Overview
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr

# %%
# FOR MOVING_BAR and MOVING_BAR_TINY (existing stimuli) they are only 600x600 or 800x800 pixels respectively.
# These do not cover the full array, especially the 600x600 pixels.
# Therefore, want to generate a list of cells that are actually exposed to the stimuli.
# This is generated in the alignment pipeline re-written script, and is specifically for 15_05_2026 currently.
# The one ending in 400px is for tiny moving bar, and the one ending in 300px is for moving bar.
central_cells_path = Path(
    r"F:\Laura\zebrafish_15_05_2026\Phase_00"
    r"\noise_analysis"
    r"\zebrafish_15_05_2026_phase_00_central_RF_cells_300px.npy"
)

central_cell_ids = np.load(central_cells_path)

print(central_cell_ids)
print(type(central_cell_ids))

# %% Dataset settings

dataset_configs = [
    {
        "recording_id": "15_05_2026_p0",
        "overview_path": r"F:\Laura\zebrafish_15_05_2026\Phase_00\overview",
        "noise_data_path": r"F:\Laura\zebrafish_15_05_2026\Phase_00\noise_analysis\noise_data.nc",
        "stimulus_index": 4,  # moving bar stim
        "rf_channel": "12px_20Hz_25mins_shuffle_x12_white",
    },
    # # Add more datasets here.
    # {
    #     "recording_id": "05_11_2025_p1",
    #     "overview_path": r"F:\Laura\zebrafish_05_11_2025\Phase_01\overview",
    #     "noise_data_path": r"F:\Laura\zebrafish_05_11_2025\Phase_01\noise_analysis\noise_data.nc",
    #     "stimulus_index": 5,
    #     "rf_channel": "12px_20Hz_shuffle",
    # },
    # {
    #     "recording_id": "02_12_2025_p1",
    #     "overview_path": r"F:\Laura\zebrafish_02_12_2025\Phase_01\overview",
    #     "noise_data_path": r"F:\Laura\zebrafish_02_12_2025\Phase_01\noise_analysis\noise_data.nc",
    #     "stimulus_index": 3,
    #     "rf_channel": "12px_20Hz_shuffle",
    # },
    #     {
    #         "recording_id": "26_02_2026_p0",
    #         "overview_path": r"F:\Laura\zebrafish_26_02_2026\Phase_00\overview",
    #         "noise_data_path": r"F:\Laura\zebrafish_26_02_2026\Phase_00\noise_analysis\noise_data.nc",
    #         "stimulus_index": 3,
    #         "rf_channel": "12px_20Hz_shuffle",
    #     },
]

# Choose which datasets to run.
# Options:
#   "all"
#   "15_01_2026_p0"
#   ["15_01_2026_p0"]
#   ["15_01_2026_p0", "example_dataset_2"]

# datasets_to_run = "all"
datasets_to_run = ["15_05_2026_p0"]


# %% ============================================================================
# Analysis settings
# ============================================================================

# Moving-bar timing.
# Original stimulus: nt = 600, then stimulus = stimulus[::2].
# Therefore each direction is 300 displayed frames at 60 Hz.
frames_per_direction = 300
seconds_per_frame = 1 / 60
expected_repeats = 10

# Triggered spike table columns.
# trigger         = frame index within one repeat, e.g. 0...2399
# repeat          = repeat index, e.g. 0...9
# times_triggered = spike time relative to repeat start, in samples
trigger_col = "trigger"
repeat_col = "repeat"
time_col = "times_triggered"

# Spike sample rate.
sample_rate_hz = 20_000

# Peak-response OSI settings.
# For each direction, this builds a PSTH across repeats and uses the maximum
# PSTH bin as the response.
peak_response_bin_size_s = 0.1

# Which response metric to use for polar plots, filtering, and scatter.
#
# Options:
#   "peak_response" = peak PSTH response per direction
#   "total_spikes"  = total spike rate across the full direction block
# polar_response_metric = "peak_response"
polar_response_metric = "total_spikes"

# Main filters.
quality_threshold = 20
min_spikes = 100

# Tilt is redefined everywhere as:
#   tilt = 1 / raw_tilt
#
# Therefore larger transformed tilt means smaller original/raw tilt.
tilt_threshold = None
tilt_filter_direction = "greater_equal"  # "greater_equal" or "less_equal"

# Direction-selective filter.
plot_only_strong_ds = True
dsi_threshold = 0.0

# Testing option.
max_cells_to_analyse = None  # set to e.g. 30 while testing

# Plot options.
plot_combined_scatter = True
plot_multipanel_polar = False
plot_single_cell_example = False

# Scatter plot options.
scatter_point_size = 35
scatter_alpha = 0.7
scatter_show_legend = True
scatter_colour_by_dataset = True

# Single-cell example options.
single_cell_recording_id = "05_11_2025_p1"
single_cell_id = 12


# %% ============================================================================
# Moving-bar direction definitions
# ============================================================================

# This order matches your DIRECTIONS dictionary:
# right, left, up, down, up-right, up-left, down-right, down-left

# direction_names = np.array(
#     [
#         "right",
#         "left",
#         "up",
#         "down",
#         "up-right",
#         "up-left",
#         "down-right",
#         "down-left",
#     ]
# )
#
# # Polar convention:
# # 0°   = up
# # 90°  = right
# # 180° = down
# # 270° = left
#
# directions_deg = np.array(
#     [
#         90,  # right
#         270,  # left
#         0,  # up
#         180,  # down
#         45,  # up-right
#         315,  # up-left
#         135,  # down-right
#         225,  # down-left
#     ]
# )
# corrected for open GL change in coordinates and projector flip
DIRECTION_NAMES = np.array(
    [
        "up",
        "down",
        "right",
        "left",
        "up-right",
        "down-right",
        "up-left",
        "down-left",
    ]
)

DIRECTIONS_DEG = np.array(
    [
        0,
        180,
        90,
        270,
        45,
        135,
        315,
        225,
    ],
    dtype=float,
)


# %% ============================================================================
# General helpers
# ============================================================================


def safe_nanmax(values):
    values = np.asarray(values, dtype=float)

    if np.all(~np.isfinite(values)):
        return np.nan

    return np.nanmax(values)


def transform_tilt(raw_tilt):
    """
    Redefine tilt as:
        tilt = 1 / raw_tilt
    """

    raw_tilt = float(raw_tilt)

    if not np.isfinite(raw_tilt) or raw_tilt == 0:
        return np.nan

    return 1 / raw_tilt


def select_dataset_configs(dataset_configs, datasets_to_run):
    """
    Select which dataset configs to analyse.

    datasets_to_run can be:
        "all"
        a single recording_id string
        a list of recording_id strings
    """

    if datasets_to_run == "all":
        return dataset_configs

    if isinstance(datasets_to_run, str):
        datasets_to_run = [datasets_to_run]

    selected_configs = [
        config
        for config in dataset_configs
        if config["recording_id"] in datasets_to_run
    ]

    found_ids = {config["recording_id"] for config in selected_configs}

    missing_ids = [
        recording_id
        for recording_id in datasets_to_run
        if recording_id not in found_ids
    ]

    if len(missing_ids) > 0:
        available_ids = [config["recording_id"] for config in dataset_configs]

        raise ValueError(
            f"These requested recording_id values were not found: {missing_ids}\n"
            f"Available recording_id values are:\n{available_ids}"
        )

    return selected_configs


# %% ============================================================================
# Selectivity calculations
# ============================================================================


def circular_vector_direction_stats(response_by_direction, directions_deg):
    """
    Calculate vector-average preferred direction, vector DSI,
    preferred orientation, and OSI.

    Direction vector:
        uses normal angles, so 0° and 180° oppose each other.

    Orientation vector:
        uses doubled angles, so 0° and 180° are treated as the same axis.

    OSI:
        |sum(R(theta) * exp(2i theta))| / sum(R(theta))
    """

    response_by_direction = np.asarray(response_by_direction, dtype=float)
    directions_deg = np.asarray(directions_deg, dtype=float)

    if response_by_direction.shape != directions_deg.shape:
        raise ValueError(
            "response_by_direction and directions_deg must have the same shape."
        )

    responses = response_by_direction.copy()
    responses[~np.isfinite(responses)] = 0

    total_response = np.sum(responses)

    if total_response == 0:
        return {
            "vector_preferred_direction_deg": np.nan,
            "vector_DSI": np.nan,
            "preferred_orientation_deg": np.nan,
            "OSI": np.nan,
            "direction_vector_length": np.nan,
            "orientation_vector_length": np.nan,
        }

    theta = np.deg2rad(directions_deg)

    direction_vector = np.sum(responses * np.exp(1j * theta))
    direction_vector_length = np.abs(direction_vector)

    vector_dsi = direction_vector_length / total_response
    vector_preferred_direction_deg = np.rad2deg(np.angle(direction_vector)) % 360

    orientation_vector = np.sum(responses * np.exp(2j * theta))
    orientation_vector_length = np.abs(orientation_vector)

    osi = orientation_vector_length / total_response
    preferred_orientation_deg = (np.rad2deg(np.angle(orientation_vector)) / 2) % 180

    return {
        "vector_preferred_direction_deg": vector_preferred_direction_deg,
        "vector_DSI": vector_dsi,
        "preferred_orientation_deg": preferred_orientation_deg,
        "OSI": osi,
        "direction_vector_length": direction_vector_length,
        "orientation_vector_length": orientation_vector_length,
    }


def compute_peak_opposite_dsi(response_by_direction, directions_deg):
    """
    Peak-opposite DSI:
        DSI = (R_pref - R_opp) / (R_pref + R_opp)
    """

    response_by_direction = np.asarray(response_by_direction, dtype=float)
    directions_deg = np.asarray(directions_deg, dtype=float)

    if response_by_direction.shape != directions_deg.shape:
        raise ValueError(
            "response_by_direction and directions_deg must have the same shape."
        )

    if len(response_by_direction) == 0 or np.all(~np.isfinite(response_by_direction)):
        return {
            "peak_opposite_DSI": np.nan,
            "peak_preferred_direction_deg": np.nan,
            "opposite_direction_deg": np.nan,
            "preferred_response": np.nan,
            "opposite_response": np.nan,
        }

    responses_for_argmax = response_by_direction.copy()
    responses_for_argmax[~np.isfinite(responses_for_argmax)] = -np.inf

    preferred_index = np.argmax(responses_for_argmax)
    preferred_response = response_by_direction[preferred_index]
    peak_preferred_direction_deg = directions_deg[preferred_index]

    opposite_target_deg = (peak_preferred_direction_deg + 180) % 360

    direction_diffs = np.abs(directions_deg - opposite_target_deg)
    direction_diffs = np.minimum(direction_diffs, 360 - direction_diffs)

    opposite_index = np.argmin(direction_diffs)
    opposite_direction_deg = directions_deg[opposite_index]
    opposite_response = response_by_direction[opposite_index]

    if not np.isfinite(preferred_response) or not np.isfinite(opposite_response):
        peak_opposite_dsi = np.nan
    else:
        denominator = preferred_response + opposite_response

        if denominator == 0:
            peak_opposite_dsi = 0
        else:
            peak_opposite_dsi = (preferred_response - opposite_response) / denominator

    return {
        "peak_opposite_DSI": peak_opposite_dsi,
        "peak_preferred_direction_deg": peak_preferred_direction_deg,
        "opposite_direction_deg": opposite_direction_deg,
        "preferred_response": preferred_response,
        "opposite_response": opposite_response,
    }


# %% ============================================================================
# Dataset helpers
# ============================================================================


def check_rf_channel_exists(dataset, rf_channel, recording_id=None):
    available_channels = dataset["channel"].values

    if rf_channel not in available_channels:
        prefix = f"Dataset {recording_id}: " if recording_id is not None else ""

        raise ValueError(
            f"{prefix}requested rf_channel {rf_channel!r} was not found.\n"
            f"Available channels are:\n{available_channels}"
        )

    return rf_channel


def get_cell_quality_and_tilt(dataset, cell_id, rf_channel):
    """
    Get quality, raw tilt, and transformed tilt for one cell.
    """

    if "quality" not in dataset:
        raise KeyError("Dataset does not contain variable 'quality'.")

    if "tilt" not in dataset:
        raise KeyError("Dataset does not contain variable 'tilt'.")

    cell_id = int(cell_id)

    if cell_id not in dataset["cell_index"].values:
        raise KeyError(f"Cell {cell_id} is not present in dataset['cell_index'].")

    if rf_channel not in dataset["channel"].values:
        raise KeyError(f"Channel {rf_channel!r} is not present in dataset['channel'].")

    quality = float(
        dataset["quality"].sel(cell_index=cell_id, channel=rf_channel).values
    )

    raw_tilt = float(dataset["tilt"].sel(cell_index=cell_id, channel=rf_channel).values)

    tilt = transform_tilt(raw_tilt)

    return {
        "quality": quality,
        "raw_tilt": raw_tilt,
        "tilt": tilt,
    }


# %% ============================================================================
# Triggered-spike timing helpers
# ============================================================================


def build_repeat_mapping(spikes_stimulus, repeat_col, expected_repeats):
    """
    Build mapping from repeat labels to 0-based repeat index.
    """

    if repeat_col not in spikes_stimulus.columns:
        raise KeyError(
            f"repeat_col={repeat_col!r} was not found.\n"
            f"Available columns are:\n{spikes_stimulus.columns}"
        )

    repeat_values = spikes_stimulus[repeat_col].to_numpy()
    unique_values = np.sort(np.unique(repeat_values))

    if len(unique_values) != expected_repeats:
        print(
            f"\nWarning: found {len(unique_values)} unique repeat values, "
            f"but expected {expected_repeats}."
        )

    # 0-based: 0...9
    if unique_values.min() == 0 and unique_values.max() == expected_repeats - 1:
        return {int(value): int(value) for value in unique_values}

    # 1-based: 1...10
    if unique_values.min() == 1 and unique_values.max() == expected_repeats:
        return {int(value): int(value) - 1 for value in unique_values}

    # Fallback: arbitrary ordered labels.
    return {int(value): idx for idx, value in enumerate(unique_values)}


def split_spikes_by_direction_and_repeat(
    spikes_single,
    direction_names,
    frames_per_direction,
    seconds_per_frame,
    expected_repeats,
    time_col,
    trigger_col,
    repeat_col,
    repeat_mapping,
    sample_rate_hz,
):
    """
    Split one cell's spikes by direction and repeat.

    Direction is determined from:
        direction_index = trigger // frames_per_direction

    Raster x-position is:
        time_in_direction = time_in_repeat - direction_start_time
    """

    for col in [time_col, trigger_col, repeat_col]:
        if col not in spikes_single.columns:
            raise KeyError(
                f"Column {col!r} was not found in spikes_single.\n"
                f"Available columns are:\n{spikes_single.columns}"
            )

    direction_names = np.asarray(direction_names)
    n_directions = len(direction_names)

    direction_duration_s = frames_per_direction * seconds_per_frame
    frames_per_repeat = frames_per_direction * n_directions

    trigger_frames = spikes_single[trigger_col].to_numpy().astype(int)
    repeat_values = spikes_single[repeat_col].to_numpy()
    times_triggered_samples = spikes_single[time_col].to_numpy().astype(float)

    repeat_index_per_spike = np.array(
        [repeat_mapping.get(int(value), -1) for value in repeat_values],
        dtype=int,
    )

    valid = (
        np.isfinite(times_triggered_samples)
        & (trigger_frames >= 0)
        & (trigger_frames < frames_per_repeat)
        & (repeat_index_per_spike >= 0)
        & (repeat_index_per_spike < expected_repeats)
    )

    trigger_frames = trigger_frames[valid]
    repeat_index_per_spike = repeat_index_per_spike[valid]
    times_triggered_samples = times_triggered_samples[valid]

    empty_rasters = {
        str(direction_name): [np.array([]) for _ in range(expected_repeats)]
        for direction_name in direction_names
    }

    if len(trigger_frames) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(n_directions, dtype=int),
            "n_spikes": 0,
        }

    direction_index_per_spike = trigger_frames // frames_per_direction

    time_in_repeat_s = times_triggered_samples / sample_rate_hz
    direction_start_s = direction_index_per_spike * direction_duration_s
    time_in_direction_s = time_in_repeat_s - direction_start_s

    valid_direction = (
        (direction_index_per_spike >= 0)
        & (direction_index_per_spike < n_directions)
        & (time_in_direction_s >= 0)
        & (time_in_direction_s < direction_duration_s)
    )

    direction_index_per_spike = direction_index_per_spike[valid_direction]
    repeat_index_per_spike = repeat_index_per_spike[valid_direction]
    time_in_direction_s = time_in_direction_s[valid_direction]

    if len(direction_index_per_spike) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(n_directions, dtype=int),
            "n_spikes": 0,
        }

    spike_counts = np.bincount(
        direction_index_per_spike,
        minlength=n_directions,
    )

    raster_by_direction = {}

    for direction_index, direction_name in enumerate(direction_names):
        repeat_spike_times = []

        for repeat_index in range(expected_repeats):
            this_repeat_times = time_in_direction_s[
                (direction_index_per_spike == direction_index)
                & (repeat_index_per_spike == repeat_index)
            ]

            repeat_spike_times.append(this_repeat_times)

        raster_by_direction[str(direction_name)] = repeat_spike_times

    return {
        "raster_by_direction": raster_by_direction,
        "spike_counts": spike_counts,
        "n_spikes": int(len(time_in_direction_s)),
    }


def compute_peak_response_by_direction_from_rasters(
    raster_by_direction,
    direction_names,
    direction_duration_s,
    expected_repeats,
    bin_size_s=0.25,
):
    """
    For each direction:
        1. Pool spikes across repeats.
        2. Build a PSTH across the direction block.
        3. Take the maximum PSTH bin as the response.

    Returns peak response in spikes/s/repeat.
    """

    direction_names = np.asarray(direction_names)

    bins = np.arange(
        0,
        direction_duration_s + bin_size_s,
        bin_size_s,
    )

    if bins[-1] < direction_duration_s:
        bins = np.append(bins, direction_duration_s)

    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2

    peak_response_by_direction = np.zeros(len(direction_names), dtype=float)
    peak_time_by_direction = np.full(len(direction_names), np.nan, dtype=float)

    psth_by_direction = {}

    for direction_index, direction_name in enumerate(direction_names):
        direction_name = str(direction_name)

        repeat_spike_times = raster_by_direction[direction_name]

        non_empty_repeats = [
            np.asarray(spike_times, dtype=float)
            for spike_times in repeat_spike_times
            if len(spike_times) > 0
        ]

        if len(non_empty_repeats) == 0:
            all_spike_times = np.array([])
        else:
            all_spike_times = np.concatenate(non_empty_repeats)

        if len(all_spike_times) == 0:
            psth = np.zeros(len(bin_centers), dtype=float)
            peak_response = 0.0
            peak_time = np.nan

        else:
            counts, _ = np.histogram(
                all_spike_times,
                bins=bins,
            )

            psth = counts / (bin_widths * expected_repeats)

            peak_bin = int(np.argmax(psth))
            peak_response = float(psth[peak_bin])
            peak_time = float(bin_centers[peak_bin])

        peak_response_by_direction[direction_index] = peak_response
        peak_time_by_direction[direction_index] = peak_time

        psth_by_direction[direction_name] = {
            "bin_centers": bin_centers,
            "psth": psth,
        }

    return {
        "peak_response_by_direction": peak_response_by_direction,
        "peak_time_by_direction": peak_time_by_direction,
        "psth_by_direction": psth_by_direction,
    }


# %% ============================================================================
# Direction tuning calculation
# ============================================================================


def calculate_tuning_for_cell(
    spikes_stimulus,
    dataset,
    cell_id,
    rf_channel,
    direction_names,
    directions_deg,
    frames_per_direction,
    seconds_per_frame,
    expected_repeats,
    time_col,
    trigger_col,
    repeat_col,
    repeat_mapping,
    sample_rate_hz,
    peak_response_bin_size_s,
):
    """
    Calculate direction/orientation tuning for one cell.

    Two response metrics are calculated:
        1. total_spikes
        2. peak_response
    """

    cell_id = int(cell_id)

    cell_info = get_cell_quality_and_tilt(
        dataset=dataset,
        cell_id=cell_id,
        rf_channel=rf_channel,
    )

    spikes_single = spikes_stimulus.filter(pl.col("cell_index") == cell_id)

    if len(spikes_single) == 0:
        return None

    split_data = split_spikes_by_direction_and_repeat(
        spikes_single=spikes_single,
        direction_names=direction_names,
        frames_per_direction=frames_per_direction,
        seconds_per_frame=seconds_per_frame,
        expected_repeats=expected_repeats,
        time_col=time_col,
        trigger_col=trigger_col,
        repeat_col=repeat_col,
        repeat_mapping=repeat_mapping,
        sample_rate_hz=sample_rate_hz,
    )

    spike_counts = split_data["spike_counts"]
    n_spikes = split_data["n_spikes"]

    if n_spikes == 0:
        return None

    direction_duration_s = frames_per_direction * seconds_per_frame
    total_time_per_direction_s = expected_repeats * direction_duration_s

    # -------------------------------------------------------------------------
    # 1. Total-spike response metric
    # -------------------------------------------------------------------------

    total_spike_response_by_direction = spike_counts / total_time_per_direction_s

    total_peak_stats = compute_peak_opposite_dsi(
        total_spike_response_by_direction,
        directions_deg,
    )

    total_vector_stats = circular_vector_direction_stats(
        total_spike_response_by_direction,
        directions_deg,
    )

    # -------------------------------------------------------------------------
    # 2. Peak-response metric
    # -------------------------------------------------------------------------

    peak_response_data = compute_peak_response_by_direction_from_rasters(
        raster_by_direction=split_data["raster_by_direction"],
        direction_names=direction_names,
        direction_duration_s=direction_duration_s,
        expected_repeats=expected_repeats,
        bin_size_s=peak_response_bin_size_s,
    )

    peak_response_by_direction = peak_response_data["peak_response_by_direction"]

    peak_response_peak_stats = compute_peak_opposite_dsi(
        peak_response_by_direction,
        directions_deg,
    )

    peak_response_vector_stats = circular_vector_direction_stats(
        peak_response_by_direction,
        directions_deg,
    )

    return {
        "cell_id": cell_id,
        "spikes_single": spikes_single,
        "n_spikes": n_spikes,
        "n_spikes_raw": len(spikes_single),
        "rf_channel": rf_channel,
        "quality": cell_info["quality"],
        "raw_tilt": cell_info["raw_tilt"],
        "tilt": cell_info["tilt"],
        "direction_names": direction_names,
        "directions_deg": directions_deg,
        "spike_counts": spike_counts,
        "raster_by_direction": split_data["raster_by_direction"],
        "total_spike_response_by_direction": total_spike_response_by_direction,
        "peak_opposite_DSI_total_spikes": total_peak_stats["peak_opposite_DSI"],
        "peak_preferred_direction_deg_total_spikes": total_peak_stats[
            "peak_preferred_direction_deg"
        ],
        "opposite_direction_deg_total_spikes": total_peak_stats[
            "opposite_direction_deg"
        ],
        "vector_DSI_total_spikes": total_vector_stats["vector_DSI"],
        "vector_preferred_direction_deg_total_spikes": total_vector_stats[
            "vector_preferred_direction_deg"
        ],
        "OSI_total_spikes": total_vector_stats["OSI"],
        "preferred_orientation_deg_total_spikes": total_vector_stats[
            "preferred_orientation_deg"
        ],
        "peak_response_by_direction": peak_response_by_direction,
        "peak_time_by_direction": peak_response_data["peak_time_by_direction"],
        "psth_by_direction": peak_response_data["psth_by_direction"],
        "peak_response_bin_size_s": peak_response_bin_size_s,
        "peak_opposite_DSI_peak_response": peak_response_peak_stats[
            "peak_opposite_DSI"
        ],
        "peak_preferred_direction_deg_peak_response": peak_response_peak_stats[
            "peak_preferred_direction_deg"
        ],
        "opposite_direction_deg_peak_response": peak_response_peak_stats[
            "opposite_direction_deg"
        ],
        "vector_DSI_peak_response": peak_response_vector_stats["vector_DSI"],
        "vector_preferred_direction_deg_peak_response": peak_response_vector_stats[
            "vector_preferred_direction_deg"
        ],
        "OSI_peak_response": peak_response_vector_stats["OSI"],
        "preferred_orientation_deg_peak_response": peak_response_vector_stats[
            "preferred_orientation_deg"
        ],
    }


# %% ============================================================================
# Response-metric switch
# ============================================================================


def get_response_metric_for_plot(result, response_metric="peak_response"):
    """
    Select which response metric to use.

    Options:
        "peak_response"
        "total_spikes"
    """

    if response_metric == "peak_response":
        return {
            "response_by_direction": result["peak_response_by_direction"],
            "OSI": result["OSI_peak_response"],
            "DSI": result["peak_opposite_DSI_peak_response"],
            "preferred_direction_deg": result[
                "peak_preferred_direction_deg_peak_response"
            ],
            "preferred_orientation_deg": result[
                "preferred_orientation_deg_peak_response"
            ],
            "label": "Peak response",
            "short_label": "peak",
        }

    if response_metric == "total_spikes":
        return {
            "response_by_direction": result["total_spike_response_by_direction"],
            "OSI": result["OSI_total_spikes"],
            "DSI": result["peak_opposite_DSI_total_spikes"],
            "preferred_direction_deg": result[
                "peak_preferred_direction_deg_total_spikes"
            ],
            "preferred_orientation_deg": result[
                "preferred_orientation_deg_total_spikes"
            ],
            "label": "Total spike rate",
            "short_label": "total",
        }

    raise ValueError("response_metric must be 'peak_response' or 'total_spikes'.")


# %% ============================================================================
# Filtering and dataframe helpers
# ============================================================================


def apply_main_filters(
    results,
    response_metric="peak_response",
    quality_threshold=None,
    min_spikes=0,
    plot_only_strong_ds=False,
    dsi_threshold=0.4,
    tilt_threshold=None,
    tilt_filter_direction="greater_equal",
):
    """
    Apply quality, spike count, DSI, and optional transformed-tilt filters.
    """

    filtered = []

    for result in results:
        metric = get_response_metric_for_plot(
            result=result,
            response_metric=response_metric,
        )

        if (
            np.isfinite(result["tilt"])
            and np.isfinite(metric["OSI"])
            and np.isfinite(metric["DSI"])
        ):
            filtered.append(result)

    if quality_threshold is not None:
        filtered = [
            result
            for result in filtered
            if np.isfinite(result["quality"]) and result["quality"] >= quality_threshold
        ]

    filtered = [result for result in filtered if result["n_spikes"] >= min_spikes]

    if tilt_threshold is not None:
        if tilt_filter_direction == "greater_equal":
            filtered = [
                result
                for result in filtered
                if np.isfinite(result["tilt"]) and result["tilt"] >= tilt_threshold
            ]

        elif tilt_filter_direction == "less_equal":
            filtered = [
                result
                for result in filtered
                if np.isfinite(result["tilt"]) and result["tilt"] <= tilt_threshold
            ]

        else:
            raise ValueError(
                "tilt_filter_direction must be 'greater_equal' or 'less_equal'."
            )

    if plot_only_strong_ds:
        filtered = [
            result
            for result in filtered
            if get_response_metric_for_plot(
                result=result,
                response_metric=response_metric,
            )["DSI"]
            >= dsi_threshold
        ]

    return filtered


def make_results_dataframe(results):
    if len(results) == 0:
        return None

    rows = []

    for result in results:
        rows.append(
            {
                "recording_id": result.get("recording_id", ""),
                "cell_index": int(result["cell_id"]),
                "global_cell_id": result.get(
                    "global_cell_id",
                    f"{result.get('recording_id', '')}__cell_{result['cell_id']}",
                ),
                "stimulus_index": result.get("stimulus_index", np.nan),
                "rf_channel": str(result["rf_channel"]),
                "quality": result["quality"],
                "raw_tilt": result["raw_tilt"],
                "tilt": result["tilt"],
                "n_spikes": result["n_spikes"],
                "OSI_peak_response": result["OSI_peak_response"],
                "DSI_peak_response": result["peak_opposite_DSI_peak_response"],
                "preferred_orientation_deg_peak_response": result[
                    "preferred_orientation_deg_peak_response"
                ],
                "peak_preferred_direction_deg_peak_response": result[
                    "peak_preferred_direction_deg_peak_response"
                ],
                "OSI_total_spikes": result["OSI_total_spikes"],
                "DSI_total_spikes": result["peak_opposite_DSI_total_spikes"],
                "preferred_orientation_deg_total_spikes": result[
                    "preferred_orientation_deg_total_spikes"
                ],
                "peak_preferred_direction_deg_total_spikes": result[
                    "peak_preferred_direction_deg_total_spikes"
                ],
            }
        )

    return pl.DataFrame(rows).to_pandas()


def build_scatter_dataframe(results, response_metric):
    rows = []

    for result in results:
        metric = get_response_metric_for_plot(
            result=result,
            response_metric=response_metric,
        )

        rows.append(
            {
                "recording_id": result["recording_id"],
                "cell_index": int(result["cell_id"]),
                "global_cell_id": result["global_cell_id"],
                "tilt": result["tilt"],
                "raw_tilt": result["raw_tilt"],
                "OSI": metric["OSI"],
                "DSI": metric["DSI"],
                "quality": result["quality"],
                "n_spikes": result["n_spikes"],
                "stimulus_index": result["stimulus_index"],
                "rf_channel": result["rf_channel"],
            }
        )

    if len(rows) == 0:
        return None

    scatter_df = pl.DataFrame(rows).to_pandas()

    scatter_df = scatter_df[
        np.isfinite(scatter_df["tilt"]) & np.isfinite(scatter_df["OSI"])
    ].copy()

    return scatter_df


# %% ============================================================================
# Dataset-level runner
# ============================================================================


def analyse_one_dataset(
    config,
    direction_names,
    directions_deg,
    frames_per_direction,
    seconds_per_frame,
    expected_repeats,
    trigger_col,
    repeat_col,
    time_col,
    sample_rate_hz,
    peak_response_bin_size_s,
    max_cells_to_analyse=None,
):
    """
    Load one recording + RF/noise dataset and calculate OSI/DSI for all cells.

    Each config can have a different:
        - overview_path
        - noise_data_path
        - stimulus_index
        - rf_channel
    """

    recording_id = config["recording_id"]
    overview_path = config["overview_path"]
    noise_data_path = Path(config["noise_data_path"])
    stimulus_index = int(config["stimulus_index"])
    rf_channel = config["rf_channel"]

    print("\n" + "=" * 80)
    print(f"Analysing dataset: {recording_id}")
    print("=" * 80)
    print(f"Overview path: {overview_path}")
    print(f"Noise dataset: {noise_data_path}")
    print(f"Moving-bar stimulus index: {stimulus_index}")
    print(f"RF channel: {rf_channel}")

    recording = Overview.Recording.load(overview_path)

    # Use open_dataset, not load_dataset, to avoid loading huge RF maps for
    # every dataset into memory.
    dataset = xr.open_dataset(noise_data_path)

    try:
        check_rf_channel_exists(
            dataset=dataset,
            rf_channel=rf_channel,
            recording_id=recording_id,
        )

        spikes_stimulus = recording.get_spikes_triggered(
            [{"stimulus_index": stimulus_index}],
            [{"cell_index": "all"}],
            pandas=False,
        )

        for required_col in ["cell_index", time_col, trigger_col, repeat_col]:
            if required_col not in spikes_stimulus.columns:
                raise KeyError(
                    f"Required column {required_col!r} not found in triggered "
                    f"spikes for dataset {recording_id}.\n"
                    f"Available columns are:\n{spikes_stimulus.columns}"
                )

        available_cells = (
            spikes_stimulus.select("cell_index")
            .unique()
            .sort("cell_index")["cell_index"]
            .to_numpy()
        )

        print(f"Number of available cells: {len(available_cells)}")

        n_directions = len(directions_deg)
        direction_duration_s = frames_per_direction * seconds_per_frame
        frames_per_repeat = frames_per_direction * n_directions
        repeat_duration_s = direction_duration_s * n_directions

        print("\nStimulus timing assumptions:")
        print(f"Frames per direction: {frames_per_direction}")
        print(f"Direction duration: {direction_duration_s:.3f} s")
        print(f"Frames per repeat: {frames_per_repeat}")
        print(f"Repeat duration: {repeat_duration_s:.3f} s")
        print(f"Expected repeats: {expected_repeats}")

        print("\nTriggered table check:")
        print(
            spikes_stimulus.select(
                pl.col(time_col).min().alias("min_times_triggered_samples"),
                pl.col(time_col).max().alias("max_times_triggered_samples"),
                (pl.col(time_col).max() / sample_rate_hz).alias(
                    "max_times_triggered_s"
                ),
                pl.col(trigger_col).min().alias("min_trigger_frame"),
                pl.col(trigger_col).max().alias("max_trigger_frame"),
                pl.col(trigger_col).n_unique().alias("n_unique_trigger_frames"),
                pl.col(repeat_col).n_unique().alias("n_unique_repeats"),
            )
        )

        repeat_mapping = build_repeat_mapping(
            spikes_stimulus=spikes_stimulus,
            repeat_col=repeat_col,
            expected_repeats=expected_repeats,
        )

        all_cell_ids = available_cells.copy()

        if max_cells_to_analyse is not None:
            all_cell_ids = all_cell_ids[:max_cells_to_analyse]

        dataset_results = []

        for this_cell_id in all_cell_ids:
            try:
                result = calculate_tuning_for_cell(
                    spikes_stimulus=spikes_stimulus,
                    dataset=dataset,
                    cell_id=this_cell_id,
                    rf_channel=rf_channel,
                    direction_names=direction_names,
                    directions_deg=directions_deg,
                    frames_per_direction=frames_per_direction,
                    seconds_per_frame=seconds_per_frame,
                    expected_repeats=expected_repeats,
                    time_col=time_col,
                    trigger_col=trigger_col,
                    repeat_col=repeat_col,
                    repeat_mapping=repeat_mapping,
                    sample_rate_hz=sample_rate_hz,
                    peak_response_bin_size_s=peak_response_bin_size_s,
                )

            except (KeyError, ValueError) as exc:
                print(f"Skipping cell {this_cell_id} in {recording_id}: {exc}")
                continue

            if result is not None:
                result["recording_id"] = recording_id
                result["global_cell_id"] = f"{recording_id}__cell_{int(this_cell_id)}"
                result["overview_path"] = overview_path
                result["noise_data_path"] = str(noise_data_path)
                result["stimulus_index"] = stimulus_index
                result["rf_channel"] = rf_channel

                dataset_results.append(result)

        print(
            f"\nCalculated moving-bar tuning for "
            f"{len(dataset_results)} cells in {recording_id}."
        )

    finally:
        dataset.close()

    return dataset_results


# %% ============================================================================
# Polar plotting helpers
# ============================================================================


def close_polar_curve(directions_deg, responses):
    directions_deg = np.asarray(directions_deg)
    responses = np.asarray(responses, dtype=float)

    sort_idx = np.argsort(directions_deg % 360)

    theta_sorted = np.deg2rad(directions_deg[sort_idx])
    responses_sorted = responses[sort_idx]

    theta_closed = np.concatenate([theta_sorted, theta_sorted[:1]])
    responses_closed = np.concatenate([responses_sorted, responses_sorted[:1]])

    return theta_closed, responses_closed


def add_motion_direction_arrows_to_polar_axis(
    ax,
    tick_degrees=None,
    radius=0.55,
    arrow_length=0.07,
    linewidth=1.2,
    mutation_scale=9,
):
    if tick_degrees is None:
        tick_degrees = np.arange(0, 360, 45)

    for motion_deg in tick_degrees:
        theta = np.deg2rad(motion_deg)

        x = 0.5 + radius * np.sin(theta)
        y = 0.5 + radius * np.cos(theta)

        dx = arrow_length * np.sin(theta)
        dy = arrow_length * np.cos(theta)

        ax.annotate(
            "",
            xy=(x + dx / 2, y + dy / 2),
            xytext=(x - dx / 2, y - dy / 2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=linewidth,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=mutation_scale,
            ),
            annotation_clip=False,
        )


def plot_pooled_polar_tuning(
    ax,
    result,
    response_metric="peak_response",
    show_title=True,
    show_legend=False,
    show_peak_direction_line=True,
    show_orientation_axis=True,
    show_motion_direction_arrows=True,
):
    metric = get_response_metric_for_plot(
        result=result,
        response_metric=response_metric,
    )

    response_by_direction = metric["response_by_direction"]

    theta_closed, rates_closed = close_polar_curve(
        result["directions_deg"],
        response_by_direction,
    )

    (rate_line,) = ax.plot(
        theta_closed,
        rates_closed,
        linewidth=2,
        marker="o",
        markersize=4,
        linestyle="-",
        label=metric["label"],
    )

    ax.fill(
        theta_closed,
        rates_closed,
        alpha=0.10,
    )

    max_rate = safe_nanmax(response_by_direction)

    if np.isfinite(max_rate) and max_rate > 0:
        if show_peak_direction_line:
            peak_deg = metric["preferred_direction_deg"]

            if np.isfinite(peak_deg):
                peak_rad = np.deg2rad(peak_deg)

                ax.plot(
                    [peak_rad, peak_rad],
                    [0, max_rate],
                    linewidth=1.5,
                    linestyle=":",
                    color=rate_line.get_color(),
                    label="Peak direction",
                )

        if show_orientation_axis:
            orientation_deg = metric["preferred_orientation_deg"]

            if np.isfinite(orientation_deg):
                orientation_rad = np.deg2rad(orientation_deg)
                opposite_orientation_rad = np.deg2rad((orientation_deg + 180) % 360)

                ax.plot(
                    [orientation_rad, orientation_rad],
                    [0, max_rate],
                    linewidth=1.2,
                    linestyle="--",
                    color=rate_line.get_color(),
                    alpha=0.7,
                    label="Preferred orientation",
                )

                ax.plot(
                    [opposite_orientation_rad, opposite_orientation_rad],
                    [0, max_rate],
                    linewidth=1.2,
                    linestyle="--",
                    color=rate_line.get_color(),
                    alpha=0.7,
                )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(-22.5)

    tick_degrees = np.arange(0, 360, 45)

    ax.set_xticks(np.deg2rad(tick_degrees))
    ax.set_xticklabels([""] * len(tick_degrees))

    if show_motion_direction_arrows:
        add_motion_direction_arrows_to_polar_axis(
            ax=ax,
            tick_degrees=tick_degrees,
            radius=0.55,
            arrow_length=0.07,
            linewidth=1.2,
            mutation_scale=9,
        )

    ax.tick_params(labelsize=7)

    if show_title:
        ax.set_title(
            f"{result['recording_id']}\n"
            f"Cell {result['cell_id']}\n"
            f"OSI={metric['OSI']:.2f}, "
            f"DSI={metric['DSI']:.2f}\n"
            f"Q={result['quality']:.1f}, "
            f"tilt={result['tilt']:.2f}",
            f"Bin size={peak_response_bin_size_s}s",
            fontsize=8,
        )

    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            fontsize=8,
        )


def plot_multipanel_polar_tuning(
    plot_results,
    n_cols=5,
    response_metric="peak_response",
):
    n_cells = len(plot_results)

    if n_cells == 0:
        print("No cells to plot.")
        return

    n_rows = int(np.ceil(n_cells / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.4 * n_rows),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )

    axs = axs.ravel()

    for ax_idx, result in enumerate(plot_results):
        plot_pooled_polar_tuning(
            ax=axs[ax_idx],
            result=result,
            response_metric=response_metric,
            show_title=True,
            show_legend=False,
            show_peak_direction_line=True,
            show_orientation_axis=True,
            show_motion_direction_arrows=True,
        )

    for ax_idx in range(len(plot_results), len(axs)):
        axs[ax_idx].axis("off")

    fig.suptitle(
        f"Moving-bar tuning | response metric = {response_metric}",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()


# %% ============================================================================
# Single-cell plot helpers
# ============================================================================


def plot_small_raster_box(
    ax,
    repeat_spike_times,
    direction_duration_s,
    expected_repeats=10,
    raster_facecolor="0.90",
    show_xlabel=False,
):
    """
    Plot one raster containing one row per stimulus repeat.
    """

    ax.set_facecolor(raster_facecolor)

    ax.eventplot(
        repeat_spike_times,
        lineoffsets=np.arange(expected_repeats),
        linelengths=0.65,
        linewidths=0.6,
        colors="black",
    )

    ax.set_xlim(0, direction_duration_s)
    ax.set_ylim(-0.5, expected_repeats - 0.5)
    ax.invert_yaxis()

    # Label time from 0 to 5 seconds in 1-second intervals
    x_ticks = np.arange(
        0,
        direction_duration_s + 0.01,
        1,
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [f"{tick:.0f}" for tick in x_ticks],
        fontsize=5,
    )

    ax.tick_params(
        axis="x",
        length=2,
        width=0.6,
        pad=1,
    )

    # Add "s" only when requested
    if show_xlabel:
        ax.set_xlabel(
            "s",
            fontsize=5,
            labelpad=1,
        )

    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.25")


def plot_small_psth_box(
    ax,
    psth_data,
    direction_duration_s,
    common_y_max,
    psth_facecolor="white",
    show_y_scale=False,
):
    """
    Plot the mean firing-rate PSTH above one raster.

    The supplied PSTH has already been pooled across repeats and divided by:
        bin width x expected number of repeats

    Therefore its units are spikes/s, averaged across repeats.
    """

    bin_centers = np.asarray(
        psth_data["bin_centers"],
        dtype=float,
    )
    mean_firing_rate = np.asarray(
        psth_data["psth"],
        dtype=float,
    )

    ax.set_facecolor(psth_facecolor)

    ax.plot(
        bin_centers,
        mean_firing_rate,
        linewidth=1.0,
        color="black",
    )

    ax.fill_between(
        bin_centers,
        0,
        mean_firing_rate,
        alpha=0.15,
        color="black",
    )

    ax.set_xlim(0, direction_duration_s)
    ax.set_ylim(0, common_y_max)
    ax.set_xticks([])

    if show_y_scale:
        ax.set_yticks([0, common_y_max])
        ax.set_yticklabels(
            ["0", f"{common_y_max:.0f}"],
            fontsize=5,
        )
        ax.set_ylabel(
            "Hz",
            fontsize=5,
            labelpad=0,
        )
        ax.tick_params(
            axis="y",
            length=2,
            width=0.6,
            pad=1,
        )
    else:
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def get_common_psth_y_max(
    psth_by_direction,
    direction_names,
    padding_fraction=0.08,
):
    """
    Return one shared PSTH y-axis maximum for all eight directions.

    Using a common scale makes the small PSTHs directly comparable.
    """

    all_values = []

    for direction_name in direction_names:
        direction_name = str(direction_name)

        if direction_name not in psth_by_direction:
            continue

        values = np.asarray(
            psth_by_direction[direction_name]["psth"],
            dtype=float,
        )

        finite_values = values[np.isfinite(values)]

        if finite_values.size > 0:
            all_values.append(finite_values)

    if len(all_values) == 0:
        return 1.0

    maximum = float(np.max(np.concatenate(all_values)))

    if maximum <= 0:
        return 1.0

    return maximum * (1 + padding_fraction)


def get_polar_arrow_center_in_figure_coords(
    fig,
    ax_polar,
    motion_deg,
    arrow_radius=0.60,
):
    theta = np.deg2rad(motion_deg)

    x_axes = 0.5 + arrow_radius * np.sin(theta)
    y_axes = 0.5 + arrow_radius * np.cos(theta)

    xy_display = ax_polar.transAxes.transform((x_axes, y_axes))
    xy_fig = fig.transFigure.inverted().transform(xy_display)

    return xy_fig[0], xy_fig[1]


def compute_raster_positions_from_polar_arrows(
    fig,
    ax_polar,
    raster_box_size=0.13,
    arrow_radius=0.60,
    raster_gap_from_arrow=0.035,
    show_psth=True,
    psth_box_height=0.045,
    psth_gap=0.006,
):
    """
    Calculate the lower-left position of each raster.

    When show_psth=True, space is reserved above every raster for a PSTH.
    The entire raster-plus-PSTH tile is positioned outside the polar arrows.
    """

    raster_width = raster_box_size
    raster_height = raster_box_size
    gap_from_arrow = raster_gap_from_arrow

    if show_psth:
        total_tile_height = raster_height + psth_gap + psth_box_height
    else:
        total_tile_height = raster_height

    direction_to_angle = {
        "up": 0,
        "up-right": 45,
        "right": 90,
        "down-right": 135,
        "down": 180,
        "down-left": 225,
        "left": 270,
        "up-left": 315,
    }

    arrow_xy = {
        direction_name: get_polar_arrow_center_in_figure_coords(
            fig=fig,
            ax_polar=ax_polar,
            motion_deg=motion_deg,
            arrow_radius=arrow_radius,
        )
        for direction_name, motion_deg in direction_to_angle.items()
    }

    raster_positions = {}

    # Above the polar plot.
    x, y = arrow_xy["up"]
    raster_positions["up"] = (
        x - raster_width / 2,
        y + gap_from_arrow,
    )

    # Below the polar plot.
    x, y = arrow_xy["down"]
    raster_positions["down"] = (
        x - raster_width / 2,
        y - gap_from_arrow - total_tile_height,
    )

    # Right and left of the polar plot.
    x, y = arrow_xy["right"]
    raster_positions["right"] = (
        x + gap_from_arrow,
        y - total_tile_height / 2,
    )

    x, y = arrow_xy["left"]
    raster_positions["left"] = (
        x - gap_from_arrow - raster_width,
        y - total_tile_height / 2,
    )

    diagonal_gap = gap_from_arrow / np.sqrt(2)

    x, y = arrow_xy["up-right"]
    raster_positions["up-right"] = (
        x + diagonal_gap,
        y + diagonal_gap,
    )

    x, y = arrow_xy["up-left"]
    raster_positions["up-left"] = (
        x - diagonal_gap - raster_width,
        y + diagonal_gap,
    )

    x, y = arrow_xy["down-right"]
    raster_positions["down-right"] = (
        x + diagonal_gap,
        y - diagonal_gap - total_tile_height,
    )

    x, y = arrow_xy["down-left"]
    raster_positions["down-left"] = (
        x - diagonal_gap - raster_width,
        y - diagonal_gap - total_tile_height,
    )

    return raster_positions


def plot_single_cell_summary(
    result,
    frames_per_direction,
    seconds_per_frame,
    expected_repeats=10,
    response_metric="peak_response",
    show_peak_direction_line=True,
    show_orientation_axis=True,
    show_motion_direction_arrows=True,
    raster_box_size=0.13,
    raster_box_facecolor="0.90",
    single_cell_arrow_radius=0.60,
    single_cell_arrow_length=0.075,
    single_cell_arrow_linewidth=1.8,
    single_cell_arrow_mutation_scale=18,
    raster_gap_from_arrow=0.035,
    show_psth=True,
    psth_box_height=0.045,
    psth_gap=0.006,
    psth_facecolor="white",
    show_psth_scale=True,
):
    """
    Plot the polar tuning curve surrounded by one raster per direction.

    When show_psth=True, a compact mean firing-rate PSTH is placed directly
    above every raster. The PSTH is averaged across expected_repeats.
    """

    direction_duration_s = frames_per_direction * seconds_per_frame
    raster_by_direction = result["raster_by_direction"]
    psth_by_direction = result["psth_by_direction"]

    metric = get_response_metric_for_plot(
        result=result,
        response_metric=response_metric,
    )

    common_psth_y_max = get_common_psth_y_max(
        psth_by_direction=psth_by_direction,
        direction_names=result["direction_names"],
    )

    fig = plt.figure(figsize=(7.2, 7.2))

    ax_polar = fig.add_axes(
        [0.30, 0.27, 0.40, 0.40],
        projection="polar",
    )

    plot_pooled_polar_tuning(
        ax=ax_polar,
        result=result,
        response_metric=response_metric,
        show_title=False,
        show_legend=False,
        show_peak_direction_line=show_peak_direction_line,
        show_orientation_axis=show_orientation_axis,
        show_motion_direction_arrows=False,
    )

    if show_motion_direction_arrows:
        add_motion_direction_arrows_to_polar_axis(
            ax=ax_polar,
            tick_degrees=np.arange(0, 360, 45),
            radius=single_cell_arrow_radius,
            arrow_length=single_cell_arrow_length,
            linewidth=single_cell_arrow_linewidth,
            mutation_scale=single_cell_arrow_mutation_scale,
        )

    raster_positions = compute_raster_positions_from_polar_arrows(
        fig=fig,
        ax_polar=ax_polar,
        raster_box_size=raster_box_size,
        arrow_radius=single_cell_arrow_radius,
        raster_gap_from_arrow=raster_gap_from_arrow,
        show_psth=show_psth,
        psth_box_height=psth_box_height,
        psth_gap=psth_gap,
    )

    for direction_name, (left, bottom) in raster_positions.items():
        ax_raster = fig.add_axes(
            [
                left,
                bottom,
                raster_box_size,
                raster_box_size,
            ]
        )

        plot_small_raster_box(
            ax=ax_raster,
            repeat_spike_times=raster_by_direction[direction_name],
            direction_duration_s=direction_duration_s,
            expected_repeats=expected_repeats,
            raster_facecolor=raster_box_facecolor,
            show_xlabel=(direction_name == "left"),
        )
        if show_psth:
            ax_psth = fig.add_axes(
                [
                    left,
                    bottom + raster_box_size + psth_gap,
                    raster_box_size,
                    psth_box_height,
                ]
            )

            plot_small_psth_box(
                ax=ax_psth,
                psth_data=psth_by_direction[direction_name],
                direction_duration_s=direction_duration_s,
                common_y_max=common_psth_y_max,
                psth_facecolor=psth_facecolor,
                # Only the left-most tile needs labels because all PSTHs
                # use the same y-axis range.
                show_y_scale=(show_psth_scale and direction_name == "left"),
            )

    fig.suptitle(
        (
            f"{result['recording_id']} | Cell {result['cell_id']}\n"
            f"{metric['label']} | "
            f"Q={result['quality']:.1f}, "
            f"tilt={result['tilt']:.2f}, "
            f"DSI={metric['DSI']:.2f}, "
            f"OSI={metric['OSI']:.2f},"
            f"Bin size={peak_response_bin_size_s}s"
        ),
        fontsize=13,
        y=0.98,
    )

    plt.show()


def print_single_cell_summary(result):
    print("\n==============================")
    print(f"{result['recording_id']} | Cell {result['cell_id']}")
    print("==============================")

    print(f"RF channel: {result['rf_channel']}")
    print(f"Stimulus index: {result['stimulus_index']}")
    print(f"Quality: {result['quality']:.3f}")
    print(f"Raw tilt: {result['raw_tilt']:.3f}")
    print(f"Transformed tilt = 1/raw_tilt: {result['tilt']:.3f}")
    print(f"Spikes used: {result['n_spikes']}")
    print(f"Peak-response bin size: {result['peak_response_bin_size_s']:.3f} s")

    print("\nDirection responses:")
    print("Direction      deg   total Hz   peak Hz   peak time s   spikes")

    for i, direction_name in enumerate(result["direction_names"]):
        print(
            f"{str(direction_name):>10s} | "
            f"{result['directions_deg'][i]:>5.1f} | "
            f"{result['total_spike_response_by_direction'][i]:>8.3f} | "
            f"{result['peak_response_by_direction'][i]:>7.3f} | "
            f"{result['peak_time_by_direction'][i]:>11.3f} | "
            f"{result['spike_counts'][i]:>6}"
        )

    print("\nPeak-response selectivity:")
    print(f"OSI_peak_response: {result['OSI_peak_response']:.3f}")
    print(f"DSI_peak_response: {result['peak_opposite_DSI_peak_response']:.3f}")

    print("\nTotal-spike selectivity:")
    print(f"OSI_total_spikes: {result['OSI_total_spikes']:.3f}")
    print(f"DSI_total_spikes: {result['peak_opposite_DSI_total_spikes']:.3f}")


# %% ============================================================================
# Run selected datasets
# ============================================================================

selected_dataset_configs = select_dataset_configs(
    dataset_configs=dataset_configs,
    datasets_to_run=datasets_to_run,
)

print("\nDatasets selected for analysis:")
for config in selected_dataset_configs:
    print(f"  {config['recording_id']}")

all_pooled_results = []

for config in selected_dataset_configs:
    dataset_results = analyse_one_dataset(
        config=config,
        direction_names=direction_names,
        directions_deg=directions_deg,
        frames_per_direction=frames_per_direction,
        seconds_per_frame=seconds_per_frame,
        expected_repeats=expected_repeats,
        trigger_col=trigger_col,
        repeat_col=repeat_col,
        time_col=time_col,
        sample_rate_hz=sample_rate_hz,
        peak_response_bin_size_s=peak_response_bin_size_s,
        max_cells_to_analyse=max_cells_to_analyse,
    )

    all_pooled_results.extend(dataset_results)

print("\n" + "=" * 80)
print(f"Total cells analysed: {len(all_pooled_results)}")
print("=" * 80)

if len(all_pooled_results) == 0:
    raise ValueError("No valid tuning results were calculated.")


# %% ============================================================================
# Summary dataframe
# ============================================================================

pooled_summary_df = make_results_dataframe(all_pooled_results)

print("\nSummary sorted by OSI_peak_response:")
print(pl.from_pandas(pooled_summary_df).sort("OSI_peak_response", descending=True))

print("\nCells per dataset:")
print(
    pl.from_pandas(pooled_summary_df)
    .group_by("recording_id")
    .agg(pl.len().alias("n_cells"))
    .sort("recording_id")
)


# %% ============================================================================
# Apply filters once
# ============================================================================

filtered_results = apply_main_filters(
    all_pooled_results,
    response_metric=polar_response_metric,
    quality_threshold=quality_threshold,
    min_spikes=min_spikes,
    plot_only_strong_ds=plot_only_strong_ds,
    dsi_threshold=dsi_threshold,
    tilt_threshold=tilt_threshold,
    tilt_filter_direction=tilt_filter_direction,
)

filtered_df = make_results_dataframe(filtered_results)

print(f"\nNumber of cells passing filters: {len(filtered_results)}")


# %% ============================================================================
# Build fast lookup for single-cell plotting
# ============================================================================

result_by_global_cell_id = {
    result["global_cell_id"]: result for result in all_pooled_results
}

results_by_recording_and_cell = {
    (result["recording_id"], int(result["cell_id"])): result
    for result in all_pooled_results
}


def list_available_plot_cells(sort_by="OSI_peak_response", descending=True, n=20):
    if pooled_summary_df is None:
        print("No pooled results available.")
        return None

    if sort_by not in pooled_summary_df.columns:
        raise ValueError(
            f"sort_by={sort_by!r} is not a valid column.\n"
            f"Available columns are:\n{list(pooled_summary_df.columns)}"
        )

    sorted_df = (
        pl.from_pandas(pooled_summary_df).sort(sort_by, descending=descending).head(n)
    )

    print(f"\nTop {n} cells sorted by {sort_by}:")
    print(sorted_df)

    return sorted_df


def get_cell_result(cell_id_to_plot, recording_id=None):
    cell_id_to_plot = int(cell_id_to_plot)

    if recording_id is not None:
        key = (recording_id, cell_id_to_plot)

        if key not in results_by_recording_and_cell:
            raise ValueError(
                f"Cell {cell_id_to_plot} was not found for recording_id "
                f"{recording_id!r}."
            )

        return results_by_recording_and_cell[key]

    matches = [
        result
        for result in all_pooled_results
        if int(result["cell_id"]) == cell_id_to_plot
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise ValueError(f"Cell {cell_id_to_plot} was not found.")

    matching_recordings = [result["recording_id"] for result in matches]

    raise ValueError(
        f"Cell ID {cell_id_to_plot} exists in multiple datasets: "
        f"{matching_recordings}\n"
        f"Call get_cell_result/plot_cell with recording_id=..."
    )


def plot_cell(
    cell_id_to_plot,
    recording_id=None,
    response_metric=None,
    raster_box_size=0.13,
    single_cell_arrow_radius=0.60,
    single_cell_arrow_length=0.075,
    single_cell_arrow_linewidth=1.8,
    single_cell_arrow_mutation_scale=18,
    raster_gap_from_arrow=0.035,
    show_motion_direction_arrows=True,
    show_psth=True,
    psth_box_height=0.045,
    psth_gap=0.006,
    psth_facecolor="white",
    show_psth_scale=True,
    print_summary=True,
):
    """
    Plot one cell's polar tuning, direction rasters, and optional PSTHs.

    The PSTHs show mean firing rate across expected_repeats and use the
    peak_response_bin_size_s bin width defined near the top of the script.
    """

    if response_metric is None:
        response_metric = polar_response_metric

    result = get_cell_result(
        cell_id_to_plot=cell_id_to_plot,
        recording_id=recording_id,
    )

    if print_summary:
        print_single_cell_summary(result)

    plot_single_cell_summary(
        result=result,
        frames_per_direction=frames_per_direction,
        seconds_per_frame=seconds_per_frame,
        expected_repeats=expected_repeats,
        response_metric=response_metric,
        raster_box_size=raster_box_size,
        single_cell_arrow_radius=single_cell_arrow_radius,
        single_cell_arrow_length=single_cell_arrow_length,
        single_cell_arrow_linewidth=single_cell_arrow_linewidth,
        single_cell_arrow_mutation_scale=single_cell_arrow_mutation_scale,
        raster_gap_from_arrow=raster_gap_from_arrow,
        show_motion_direction_arrows=show_motion_direction_arrows,
        show_psth=show_psth,
        psth_box_height=psth_box_height,
        psth_gap=psth_gap,
        psth_facecolor=psth_facecolor,
        show_psth_scale=show_psth_scale,
    )

    return result


# %% ============================================================================
# Combined scatter plot: OSI versus transformed tilt
# ============================================================================


def plot_osi_vs_tilt_scatter(
    filtered_results,
    response_metric=polar_response_metric,
    colour_by_dataset=True,
    point_size=35,
    alpha=0.7,
    show_legend=True,
):
    scatter_df = build_scatter_dataframe(
        results=filtered_results,
        response_metric=response_metric,
    )

    if scatter_df is None or len(scatter_df) == 0:
        print("No finite cells available for scatter plot.")
        return None

    n_cells_scatter = len(scatter_df)

    pearson_r = np.nan
    pearson_p = np.nan

    if n_cells_scatter > 1:
        try:
            from scipy.stats import pearsonr

            pearson_result = pearsonr(
                scatter_df["tilt"],
                scatter_df["OSI"],
            )

            pearson_r = pearson_result.statistic
            pearson_p = pearson_result.pvalue

        except ImportError:
            pearson_r = np.corrcoef(
                scatter_df["tilt"],
                scatter_df["OSI"],
            )[0, 1]
            pearson_p = np.nan

    fig, ax = plt.subplots(figsize=(6.4, 5.5))

    if colour_by_dataset:
        for recording_id, group_df in scatter_df.groupby("recording_id"):
            ax.scatter(
                group_df["tilt"],
                group_df["OSI"],
                alpha=alpha,
                s=point_size,
                label=f"{recording_id} (n={len(group_df)})",
            )
    else:
        ax.scatter(
            scatter_df["tilt"],
            scatter_df["OSI"],
            alpha=alpha,
            s=point_size,
            color="0.25",
        )

    ax.set_xlabel("RF tilt")
    ax.set_ylabel("OSI")
    ax.set_title(
        f"RF tilt versus OSI\n"
        f"response metric = {response_metric}, n = {n_cells_scatter}"
    )

    if np.isfinite(pearson_r):
        if np.isfinite(pearson_p):
            stats_text = f"Pearson r = {pearson_r:.3f}\n" f"p = {pearson_p:.3g}"
        else:
            stats_text = f"Pearson r = {pearson_r:.3f}"

        ax.text(
            0.97,
            0.97,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.8,
            ),
        )

    if colour_by_dataset and show_legend:
        ax.legend(
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    plt.tight_layout()
    plt.show()

    return scatter_df


if plot_combined_scatter:
    scatter_df = plot_osi_vs_tilt_scatter(
        filtered_results=filtered_results,
        response_metric="peak_response",
        colour_by_dataset=scatter_colour_by_dataset,
        point_size=scatter_point_size,
        alpha=scatter_alpha,
        show_legend=scatter_show_legend,
    )


# %% ============================================================================
# Optional multipanel polar plot
# ============================================================================

if plot_multipanel_polar:
    plot_results = sorted(
        filtered_results,
        key=lambda result: get_response_metric_for_plot(
            result,
            response_metric=polar_response_metric,
        )["OSI"],
        reverse=True,
    )

    plot_multipanel_polar_tuning(
        plot_results=plot_results,
        n_cols=5,
        response_metric=polar_response_metric,
    )


# %% ============================================================================
# Optional single-cell plot
# ============================================================================

if plot_single_cell_example:
    single_result = plot_cell(
        cell_id_to_plot=single_cell_id,
        recording_id=single_cell_recording_id,
        response_metric=polar_response_metric,
        single_cell_arrow_radius=0.60,
        raster_box_size=0.13,
        raster_gap_from_arrow=0.035,
        print_summary=True,
    )


# %% ============================================================================
# Convenience examples
# ============================================================================

# IMPORTANT:
# datasets_to_run needs to be set near the top of the script, BEFORE the
# "Run selected datasets" section.
#
# Run one dataset:
# datasets_to_run = ["15_01_2026_p0"]
#
# Run all datasets:
# datasets_to_run = "all"


# %% ============================================================================
# Plot one cell after the script has run
# ============================================================================

single_result = plot_cell(
    cell_id_to_plot=13,
    recording_id="15_05_2026_p0",
    response_metric="peak_response",
)

single_result = plot_cell(
    cell_id_to_plot=13,
    recording_id="15_05_2026_p0",
    response_metric="total_spikes",
)


# %% ============================================================================
# List top cells
# ============================================================================

list_available_plot_cells(
    sort_by="OSI_peak_response",
    descending=True,
    n=20,
)

list_available_plot_cells(
    sort_by="tilt",
    descending=True,
    n=20,
)


# %% ============================================================================
# Get and plot top 10 most tilted filtered cells
# ============================================================================

top_10_tilted_filtered_df = (
    pl.from_pandas(filtered_df)
    .filter(pl.col("tilt").is_not_nan())
    .sort("tilt", descending=True)
    .head(10)
    .to_pandas()
)

print("\nTop 10 most tilted filtered cells:")
print(
    top_10_tilted_filtered_df[
        [
            "recording_id",
            "cell_index",
            "tilt",
            "raw_tilt",
            "quality",
            "n_spikes",
            "OSI_peak_response",
            "OSI_total_spikes",
        ]
    ]
)

for _, row in top_10_tilted_filtered_df.iterrows():
    plot_cell(
        cell_id_to_plot=int(row["cell_index"]),
        recording_id=row["recording_id"],
        response_metric="peak_response",
        print_summary=False,
    )


# %% ============================================================================
# Get and plot bottom 10 least tilted filtered cells
# ============================================================================

bottom_10_tilted_filtered_df = (
    pl.from_pandas(filtered_df)
    .filter(pl.col("tilt").is_not_nan())
    .sort("tilt", descending=False)
    .head(10)
    .to_pandas()
)

print("\nBottom 10 least tilted filtered cells:")
print(
    bottom_10_tilted_filtered_df[
        [
            "recording_id",
            "cell_index",
            "tilt",
            "raw_tilt",
            "quality",
            "n_spikes",
            "OSI_peak_response",
            "OSI_total_spikes",
        ]
    ]
)

for _, row in bottom_10_tilted_filtered_df.iterrows():
    plot_cell(
        cell_id_to_plot=int(row["cell_index"]),
        recording_id=row["recording_id"],
        response_metric="total_spikes",
        print_summary=False,
    )
# %%
# %% ============================================================================
# Scatter plot: top 10 most tilted + bottom 10 least tilted cells
# OSI versus transformed tilt
# ============================================================================
import pandas as pd

n_extreme_cells = 10

# Get extreme groups from the already-filtered dataframe
top_10_tilted_df = (
    pl.from_pandas(filtered_df)
    .filter(pl.col("tilt").is_not_nan())
    .sort("tilt", descending=True)
    .head(n_extreme_cells)
    .with_columns(pl.lit("Most tilted").alias("tilt_group"))
    .to_pandas()
)

bottom_10_tilted_df = (
    pl.from_pandas(filtered_df)
    .filter(pl.col("tilt").is_not_nan())
    .sort("tilt", descending=False)
    .head(n_extreme_cells)
    .with_columns(pl.lit("Least tilted").alias("tilt_group"))
    .to_pandas()
)

# Combine
extreme_tilt_scatter_df = pd.concat(
    [top_10_tilted_df, bottom_10_tilted_df],
    ignore_index=True,
)

# Optional: remove duplicates in case top and bottom groups overlap
extreme_tilt_scatter_df = extreme_tilt_scatter_df.drop_duplicates(
    subset=["recording_id", "cell_index"]
).copy()

print("\nCells included in extreme-tilt scatter:")
print(
    extreme_tilt_scatter_df[
        [
            "recording_id",
            "cell_index",
            "tilt",
            "raw_tilt",
            "OSI_peak_response",
            "OSI_total_spikes",
            "quality",
            "n_spikes",
            "tilt_group",
        ]
    ]
)

# Choose which OSI to plot
if polar_response_metric == "total_spikes":
    osi_col = "OSI_peak_response"
else:
    osi_col = "OSI_total_spikes"

# Plot
fig, ax = plt.subplots(figsize=(5.5, 5))

most_df = extreme_tilt_scatter_df[
    extreme_tilt_scatter_df["tilt_group"] == "Most tilted"
]
least_df = extreme_tilt_scatter_df[
    extreme_tilt_scatter_df["tilt_group"] == "Least tilted"
]

ax.scatter(
    most_df["tilt"],
    most_df[osi_col],
    color="red",
    alpha=0.8,
    s=45,
    label=f"Most tilted (n = {len(most_df)})",
)

ax.scatter(
    least_df["tilt"],
    least_df[osi_col],
    color="blue",
    alpha=0.8,
    s=45,
    label=f"Least tilted (n = {len(least_df)})",
)

ax.set_xlabel("RF tilt")
ax.set_ylabel("OSI")
ax.set_title(
    f"OSI versus RF tilt\n" f"Top {n_extreme_cells} most and least tilted cells"
)

ax.legend()
plt.tight_layout()
plt.show()
