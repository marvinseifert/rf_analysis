from pathlib import Path
import numpy as np
import xarray as xr

from loading.load_sta import load_and_realign_center
from dim_reduction.rms_error import calculate_rms

from scipy import ndimage
from scipy.ndimage import gaussian_filter


def calculate_weighted_sta_for_cells(
    noise_data_path,
    recording_config,
    collapse_2d_config,
    cell_ids,
    channels,
    qi_limit=None,
    weight_method="threshold",
    threshold_fraction_by_channel=None,
    percentile_by_channel=None,
    max_diameter_um_by_channel=None,
    sigma_by_channel=None,
    connected_component=False,
    pixel_size_um_by_channel=None,
    add_offset=True,
    verbose=True,
    return_masks=False,
):
    """
    Calculate weighted STA time courses for selected cells/channels
    without rerunning the full pipeline.

    Weighting methods
    -----------------
    weight_method="all_positive"
        Use all positive covariance pixels.

    weight_method="threshold"
        Use positive covariance pixels above a channel-specific fraction
        of the maximum positive covariance.

    weight_method="percentile"
        Use positive covariance pixels above a channel-specific percentile.
        Example: 95 keeps the top 5% of positive covariance pixels.

    weight_method="diameter"
        Use positive covariance pixels within a physical diameter around
        the most important pixel.

    Parameters
    ----------
    noise_data_path : str or Path
        Path to existing noise_data.nc.

    recording_config : Recording_Config
        Same Recording_Config used in the original pipeline.

    collapse_2d_config : Collapse_2d_Config
        Same Collapse_2d_Config used in the original pipeline.

    cell_ids : list[int]
        Cell indices to process.

    channels : list[str]
        Channel names to process.

    qi_limit : float or None
        Optional quality threshold.

    weight_method : str
        One of:
        "all_positive", "threshold", "percentile", "diameter"

    threshold_fraction_by_channel : dict or None
        Used when weight_method="threshold".
        Values are fractions of maximum positive covariance.

    percentile_by_channel : dict or None
        Used when weight_method="percentile".
        Values should be percentiles, e.g. 90, 95, 97.

    max_diameter_um_by_channel : dict or None
        Used when weight_method="diameter".
        Diameter in micrometres.

    sigma_by_channel : dict or None
        Optional Gaussian smoothing sigma, in pixels, applied to covariance map
        before selecting weights.

    connected_component : bool
        If True, keep only the connected covariance region containing the
        most important pixel.

    pixel_size_um_by_channel : dict or None
        Optional override for physical pixel size in um.
        If None, uses recording_config.channel_configs[channel].pixel_size.

    add_offset : bool
        If True, adds 0.5 to match your existing STA convention.

    verbose : bool
        If True, prints number of pixels and physical area used.

    Returns
    -------
    xr.DataArray
        Weighted STA with dimensions:
        cell_index, channel, time_max
    """

    valid_methods = {
        "all_positive",
        "threshold",
        "percentile",
        "diameter",
    }

    if weight_method not in valid_methods:
        raise ValueError(
            f"weight_method must be one of {valid_methods}. " f"Got: {weight_method}"
        )

    ds = xr.load_dataset(noise_data_path)

    # ------------------------------------------------------------
    # Default settings
    # ------------------------------------------------------------
    if threshold_fraction_by_channel is None:
        threshold_fraction_by_channel = {
            "2px_20Hz_20mins_shuffle": 0.6,
            "4px_20Hz_40mins_shuffle": 0.5,
            "12px_20Hz_shuffle": 0.4,
        }

    if percentile_by_channel is None:
        percentile_by_channel = {
            "2px_20Hz_20mins_shuffle": 95,
            "4px_20Hz_40mins_shuffle": 95,
            "12px_20Hz_shuffle": 90,
        }

    if max_diameter_um_by_channel is None:
        max_diameter_um_by_channel = {
            "2px_20Hz_20mins_shuffle": 200,
            "4px_20Hz_40mins_shuffle": 160,
            "12px_20Hz_shuffle": 160,
        }

    if sigma_by_channel is None:
        sigma_by_channel = {
            "2px_20Hz_20mins_shuffle": 2.0,
            "4px_20Hz_40mins_shuffle": 1.0,
            "12px_20Hz_shuffle": 0.0,
        }

    if pixel_size_um_by_channel is None:
        pixel_size_um_by_channel = {}

    target_time = ds["time_max"].values

    weighted_sta_results = []
    mask_results = []
    weight_map_results = []
    cov_map_results = []

    for cell_id in cell_ids:
        channel_results = []
        channel_mask_results = []
        channel_weight_map_results = []
        channel_cov_map_results = []

        for channel in channels:
            # ------------------------------------------------------------
            # Validate cell/channel
            # ------------------------------------------------------------
            if cell_id not in ds.cell_index.values:
                print(f"Cell {cell_id} not found in dataset.")
                channel_results.append(None)
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                continue

            if channel not in ds.channel.values:
                print(f"Channel {channel} not found in dataset.")
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                channel_results.append(None)
                continue

            cell_data = ds.sel(cell_index=cell_id, channel=channel)

            if np.isnan(cell_data["quality"].item()):
                print(f"Skipping cell {cell_id}, {channel}: quality is NaN.")
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                channel_results.append(None)
                continue

            if qi_limit is not None:
                if cell_data["quality"].item() < qi_limit:
                    print(
                        f"Skipping cell {cell_id}, {channel}: "
                        f"quality {cell_data['quality'].item():.2f} < {qi_limit}."
                    )
                    channel_mask_results.append(None)
                    channel_weight_map_results.append(None)
                    channel_cov_map_results.append(None)
                    channel_results.append(None)
                    continue

            # ------------------------------------------------------------
            # Reload and realign STA cutout
            # ------------------------------------------------------------
            position = ds["positions"].sel(
                cell_index=cell_id,
                channel=channel,
            )

            result = load_and_realign_center(
                recording_config,
                collapse_2d_config,
                cell_id,
                channel,
                position,
            )

            if result is None:
                print(f"Could not load/realign cell {cell_id}, {channel}.")
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                channel_results.append(None)
                continue

            subset, c_x, c_y, var_coordinates = result

            # ------------------------------------------------------------
            # Recalculate STA-per-spike
            # ------------------------------------------------------------
            stimulus_id = int(
                Path(cell_data["sta_path"].item()).parts[-3].split("_")[-1]
            )

            nr_of_spikes = recording_config.overview.spikes_df.query(
                f"stimulus_index == {stimulus_id} and cell_index == {cell_id}"
            )["nr_of_spikes"].values[0]

            sta_per_spike_raw, rms = calculate_rms(subset, nr_of_spikes)

            time_bins, h, w = sta_per_spike_raw.shape

            raw_flat = sta_per_spike_raw.values.reshape(time_bins, h * w)

            flat_centered = raw_flat - raw_flat.mean(axis=0, keepdims=True)

            # ------------------------------------------------------------
            # SVD: find most important pixel
            # ------------------------------------------------------------
            try:
                _, _, Vt = np.linalg.svd(flat_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                print(f"SVD failed for cell {cell_id}, {channel}.")
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                channel_results.append(None)
                continue

            loadings = Vt[0]
            most_idx = int(np.argmax(np.abs(loadings)))

            pix_y, pix_x = divmod(most_idx, w)

            # ------------------------------------------------------------
            # Covariance of every pixel with most important pixel
            # ------------------------------------------------------------
            cov_with_most = (flat_centered.T @ flat_centered[:, most_idx]) / (
                time_bins - 1
            )

            cov_map = cov_with_most.reshape(h, w)

            # ------------------------------------------------------------
            # Optional smoothing of covariance map
            # ------------------------------------------------------------
            sigma = sigma_by_channel.get(channel, 0.0)

            if sigma > 0:
                cov_map = gaussian_filter(cov_map, sigma=sigma)

            # ------------------------------------------------------------
            # Start with positive covariance only
            # ------------------------------------------------------------
            weights_map = np.clip(cov_map, 0, None)

            if weights_map.sum() <= 0:
                print(f"No positive covariance pixels for cell {cell_id}, {channel}.")
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                channel_results.append(None)
                continue

            # ------------------------------------------------------------
            # Select pixels according to weighting method
            # ------------------------------------------------------------
            if weight_method == "all_positive":
                mask = weights_map > 0

            elif weight_method == "threshold":
                threshold_fraction = threshold_fraction_by_channel.get(channel, 0.3)
                threshold = threshold_fraction * weights_map.max()
                mask = weights_map >= threshold

            elif weight_method == "percentile":
                positive_values = weights_map[weights_map > 0]

                if positive_values.size == 0:
                    print(f"No positive pixels for cell {cell_id}, {channel}.")
                    channel_mask_results.append(None)
                    channel_weight_map_results.append(None)
                    channel_cov_map_results.append(None)
                    channel_results.append(None)
                    continue

                percentile = percentile_by_channel.get(channel, 95)
                threshold = np.percentile(positive_values, percentile)
                mask = weights_map >= threshold

            elif weight_method == "diameter":
                pixel_size_um = pixel_size_um_by_channel.get(
                    channel,
                    float(recording_config.channel_configs[channel].pixel_size),
                )

                max_diameter_um = max_diameter_um_by_channel.get(channel, 160)
                max_radius_um = max_diameter_um / 2

                yy, xx = np.indices((h, w))

                distance_um = np.sqrt(
                    ((yy - pix_y) * pixel_size_um) ** 2
                    + ((xx - pix_x) * pixel_size_um) ** 2
                )

                mask = (weights_map > 0) & (distance_um <= max_radius_um)

            # ------------------------------------------------------------
            # Optional connected component restriction
            # ------------------------------------------------------------
            if connected_component:
                labels, n_labels = ndimage.label(mask)

                main_label = labels[pix_y, pix_x]

                if main_label == 0:
                    print(
                        f"Most important pixel did not survive mask "
                        f"for cell {cell_id}, {channel}."
                    )
                    channel_mask_results.append(None)
                    channel_weight_map_results.append(None)
                    channel_cov_map_results.append(None)
                    channel_results.append(None)
                    continue

                mask = labels == main_label

            # ------------------------------------------------------------
            # Apply final mask
            # ------------------------------------------------------------
            weights_map = weights_map * mask

            if weights_map.sum() <= 0:
                print(
                    f"No pixels survived weighting method for "
                    f"cell {cell_id}, {channel}."
                )
                channel_mask_results.append(None)
                channel_weight_map_results.append(None)
                channel_cov_map_results.append(None)
                channel_results.append(None)
                continue

            final_mask = mask.copy()
            final_weights_map = weights_map.copy()
            final_cov_map = cov_map.copy()
            weights = weights_map.ravel()
            weights = weights / weights.sum()

            # ------------------------------------------------------------
            # Weighted STA across selected pixels
            # ------------------------------------------------------------
            sta_weighted = raw_flat @ weights

            if add_offset:
                sta_weighted = sta_weighted + 0.5

            da = xr.DataArray(
                sta_weighted.astype(np.float32),
                dims=["time"],
                coords={
                    "time": sta_per_spike_raw.time.values,
                },
                name="STA_weighted_pixels",
            )

            # Interpolate onto the existing dataset time axis
            da = da.interp(time=target_time, kwargs={"fill_value": 0})

            channel_results.append(da)
            channel_mask_results.append(final_mask)
            channel_weight_map_results.append(final_weights_map)
            channel_cov_map_results.append(final_cov_map)

            # ------------------------------------------------------------
            # Diagnostics
            # ------------------------------------------------------------
            if verbose:
                n_pixels_used = int(np.sum(weights > 0))

                pixel_size_um = pixel_size_um_by_channel.get(
                    channel,
                    float(recording_config.channel_configs[channel].pixel_size),
                )

                area_um2 = n_pixels_used * pixel_size_um**2

                sta_amp = float(np.nanmax(sta_weighted) - np.nanmin(sta_weighted))

                print(
                    f"Cell {cell_id}, {channel}: "
                    f"method={weight_method}, "
                    f"pixels={n_pixels_used}, "
                    f"area={area_um2:.1f} µm², "
                    f"STA amplitude={sta_amp:.4f}"
                )

        weighted_sta_results.append(channel_results)
        mask_results.append(channel_mask_results)
        weight_map_results.append(channel_weight_map_results)
        cov_map_results.append(channel_cov_map_results)

    # ------------------------------------------------------------
    # Convert nested results into one xarray DataArray
    # ------------------------------------------------------------
    output = np.full(
        (
            len(cell_ids),
            len(channels),
            len(target_time),
        ),
        np.nan,
        dtype=np.float32,
    )

    for i, row in enumerate(weighted_sta_results):
        for j, item in enumerate(row):
            if item is not None:
                output[i, j, :] = item.values

    weighted_sta_da = xr.DataArray(
        output,
        dims=["cell_index", "channel", "time_max"],
        coords={
            "cell_index": cell_ids,
            "channel": channels,
            "time_max": target_time,
        },
        name=f"STA_weighted_pixels_{weight_method}",
    )

    if return_masks:
        return weighted_sta_da, mask_results, weight_map_results, cov_map_results

    return weighted_sta_da


# %%
from pathlib import Path
from organize.configs import Recording_Config, Collapse_2d_Config
from location.x_array import x_y_and_scale


rec_object = Recording_Config(
    root_path=Path(r"F:\Laura\zebrafish_26_02_2026\Phase_00"),
)

rec_object.add_channel(
    stimulus_id=8,
    name="2px_20Hz_20mins_shuffle",
    colour="blue",
)
rec_object.add_channel(
    stimulus_id=4,
    name="4px_20Hz_40mins_shuffle",
    colour="green",
)
rec_object.add_channel(
    stimulus_id=12,
    name="12px_20Hz_shuffle",
    colour="red",
)

collapse_2d_config = Collapse_2d_Config(
    recording_config=rec_object,
    cut_size_um=x_y_and_scale(800, 800),
)

noise_data_path = (
    rec_object.root_path / rec_object.output_folder / "noise_analysis" / "noise_data.nc"
)

dataset = xr.load_dataset(noise_data_path)
rec_config = Recording_Config.load_from_root_json(noise_data_path.parent)
# %%


# sta_weighted = calculate_weighted_sta_for_cells(
#     noise_data_path=noise_data_path,
#     recording_config=rec_object,
#     collapse_2d_config=collapse_2d_config,
#     cell_ids=[235, 135],
#     channels=[
#         "2px_20Hz_20mins_shuffle",
#         "4px_20Hz_40mins_shuffle",
#         "12px_20Hz_shuffle",
#     ],
#     qi_limit=20,
#     weight_method="all_positive",
#     connected_component=True,
# )
#
# sta_weighted = calculate_weighted_sta_for_cells(
#     noise_data_path=noise_data_path,
#     recording_config=rec_object,
#     collapse_2d_config=collapse_2d_config,
#     cell_ids=[235, 135],
#     channels=[
#         "2px_20Hz_20mins_shuffle",
#         "4px_20Hz_40mins_shuffle",
#         "12px_20Hz_shuffle",
#     ],
#     qi_limit=20,
#     weight_method="threshold",
#     threshold_fraction_by_channel={
#         "2px_20Hz_20mins_shuffle": 0.6,
#         "4px_20Hz_40mins_shuffle": 0.5,
#         "12px_20Hz_shuffle": 0.4,
#     },
#     connected_component=False,
# )

(
    sta_weighted,
    mask_results,
    weight_map_results,
    cov_map_results,
) = calculate_weighted_sta_for_cells(
    noise_data_path=noise_data_path,
    recording_config=rec_object,
    collapse_2d_config=collapse_2d_config,
    cell_ids=[235, 135],
    channels=[
        "2px_20Hz_20mins_shuffle",
        "4px_20Hz_40mins_shuffle",
        "12px_20Hz_shuffle",
    ],
    qi_limit=20,
    weight_method="percentile",
    percentile_by_channel={
        "2px_20Hz_20mins_shuffle": 97,
        "4px_20Hz_40mins_shuffle": 97,
        "12px_20Hz_shuffle": 97,
    },
    connected_component=False,
    return_masks=True,
)

# sta_weighted = calculate_weighted_sta_for_cells(
#     noise_data_path=noise_data_path,
#     recording_config=rec_object,
#     collapse_2d_config=collapse_2d_config,
#     cell_ids=[235, 135],
#     channels=[
#         "2px_20Hz_20mins_shuffle",
#         "4px_20Hz_40mins_shuffle",
#         "12px_20Hz_shuffle",
#     ],
#     qi_limit=20,
#     weight_method="diameter",
#     max_diameter_um_by_channel={
#         "2px_20Hz_20mins_shuffle": 200,
#         "4px_20Hz_40mins_shuffle": 160,
#         "12px_20Hz_shuffle": 160,
#     },
#     pixel_size_um_by_channel={
#         "2px_20Hz_20mins_shuffle": 5,
#         "4px_20Hz_40mins_shuffle": 5,
#         "12px_20Hz_shuffle": 5,
#     },
#     connected_component=True,
# )
# %% Plots


def plot_weighted_and_single_sta_by_cell_and_channel(
    sta_weighted,
    ds,
    figsize_per_panel=(4, 3),
    sharey=False,
    reverse_channels=False,
):
    """
    Plot weighted STA and single-pixel STA for each selected cell and channel.

    Top block:
        STA_weighted_pixels

    Bottom block:
        sta_single_pixel from the original dataset
    """

    cell_ids = sta_weighted.cell_index.values
    channels = sta_weighted.channel.values

    if reverse_channels:
        channels = channels[::-1]

    time_weighted = sta_weighted.time_max.values
    time_single = ds.time_max.values

    n_cells = len(cell_ids)
    n_channels = len(channels)

    n_rows = n_cells * 2
    n_cols = n_channels

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            figsize_per_panel[0] * n_cols,
            figsize_per_panel[1] * n_rows,
        ),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    # ============================================================
    # TOP BLOCK: WEIGHTED STA
    # ============================================================
    for row, cell_id in enumerate(cell_ids):
        for col, channel in enumerate(channels):
            ax = axes[row, col]

            sta = sta_weighted.sel(
                cell_index=cell_id,
                channel=channel,
            )

            ax.plot(
                time_weighted,
                sta.values,
                linewidth=2,
            )

            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.axvline(0, color="gray", linestyle=":", linewidth=1)

            ax.set_title(
                f"Cell {int(cell_id)}\n{channel}",
                fontsize=11,
            )

            if col == 0:
                ax.set_ylabel("Weighted STA", fontsize=12)

            ax.tick_params(axis="both", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # ============================================================
    # BOTTOM BLOCK: SINGLE-PIXEL STA
    # ============================================================
    for row, cell_id in enumerate(cell_ids):
        plot_row = row + n_cells

        for col, channel in enumerate(channels):
            ax = axes[plot_row, col]

            sta_single = ds["sta_single_pixel"].sel(
                cell_index=cell_id,
                channel=channel,
            )

            ax.plot(
                time_single,
                sta_single.values,
                linewidth=2,
            )

            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.axvline(0, color="gray", linestyle=":", linewidth=1)

            ax.set_title(
                f"Cell {int(cell_id)}\n{channel}",
                fontsize=11,
            )

            if plot_row == n_rows - 1:
                ax.set_xlabel("Time before spike (ms)", fontsize=12)

            if col == 0:
                ax.set_ylabel("Single-pixel STA", fontsize=12)

            ax.tick_params(axis="both", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


plot_weighted_and_single_sta_by_cell_and_channel(
    sta_weighted,
    ds,
    reverse_channels=True,
)
# %%


def plot_weighted_pixel_masks(
    mask_results,
    weight_map_results,
    cov_map_results,
    cell_ids,
    channels,
    reverse_channels=False,
    show_weights=True,
    figsize_per_panel=(4, 4),
):
    """
    Plot which pixels from the STA cutout were used for weighted STA.

    Parameters
    ----------
    mask_results : nested list
        Output mask list from calculate_weighted_sta_for_cells(..., return_masks=True).

    weight_map_results : nested list
        Final covariance weight maps after masking.

    cov_map_results : nested list
        Original covariance maps before masking.

    cell_ids : list
        Cell IDs in the same order used for calculation.

    channels : list
        Channel names in the same order used for calculation.

    reverse_channels : bool
        Reverse channel order in the plot.

    show_weights : bool
        If True, plot weighted covariance map.
        If False, plot binary selected-pixel mask.
    """

    plot_channels = list(channels)

    if reverse_channels:
        plot_channels = plot_channels[::-1]

    n_rows = len(cell_ids)
    n_cols = len(plot_channels)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            figsize_per_panel[0] * n_cols,
            figsize_per_panel[1] * n_rows,
        ),
        squeeze=False,
    )

    for row, cell_id in enumerate(cell_ids):
        for col, channel in enumerate(plot_channels):
            ax = axes[row, col]

            original_channel_index = list(channels).index(channel)

            mask = mask_results[row][original_channel_index]
            weights_map = weight_map_results[row][original_channel_index]
            cov_map = cov_map_results[row][original_channel_index]

            if mask is None:
                ax.set_title(f"Cell {cell_id}\n{channel}\nNo valid mask")
                ax.axis("off")
                continue

            if show_weights:
                image = weights_map
                title_label = "selected weighted pixels"
            else:
                image = mask.astype(float)
                title_label = "selected pixel mask"

            im = ax.imshow(image, origin="lower")
            ax.contour(mask, levels=[0.5], linewidths=1)

            pix_y, pix_x = np.unravel_index(
                np.nanargmax(np.abs(cov_map)),
                cov_map.shape,
            )

            ax.scatter(
                pix_x,
                pix_y,
                marker="x",
                s=80,
                linewidths=2,
            )

            ax.set_title(
                f"Cell {int(cell_id)}\n{channel}\n{title_label}",
                fontsize=10,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )

            if show_weights:
                cbar.set_label("Covariance weight", fontsize=9)
            else:
                cbar.set_label("Selected", fontsize=9)

    fig.tight_layout()
    plt.show()


plot_weighted_pixel_masks(
    mask_results=mask_results,
    weight_map_results=weight_map_results,
    cov_map_results=cov_map_results,
    cell_ids=[235, 135],
    channels=[
        "2px_20Hz_20mins_shuffle",
        "4px_20Hz_40mins_shuffle",
        "12px_20Hz_shuffle",
    ],
    reverse_channels=True,
    show_weights=True,
)

# %% Poster plot
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as mticker
from organize.configs import Recording_Config
from aquarel import load_theme


# =============================================================================
# Plot style
# =============================================================================

theme = load_theme("scientific")
theme.apply()

coolwarm_heatmap = matplotlib.colormaps["coolwarm"]


# =============================================================================
# Helper functions
# =============================================================================


def cutout_nans(data, inset=0):
    """
    Crop image to valid non-zero/non-NaN region.
    """

    valid_mask = np.nan_to_num(data.values) != 0

    x_valid = np.where(np.any(valid_mask, axis=0))[0]
    y_valid = np.where(np.any(valid_mask, axis=1))[0]

    if len(x_valid) == 0 or len(y_valid) == 0:
        return data

    x_slice = slice(
        x_valid[0] + inset,
        x_valid[-1] + 1 - inset,
        1,
    )

    y_slice = slice(
        y_valid[0] + inset,
        y_valid[-1] + 1 - inset,
        1,
    )

    return data.isel(x=x_slice, y=y_slice)


def add_scale_bar(ax, length_um=20, height_offset=0.05, pad=0.05, lw=2):
    """
    Add a horizontal scale bar in data coordinates.
    Assumes x-axis is in micrometers.
    """

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_start = x1 - (x1 - x0) * pad - length_um
    y_start = y1 - (y1 - y0) * pad

    ax.plot(
        [x_start, x_start + length_um],
        [y_start, y_start],
        color="black",
        linewidth=lw,
        solid_capstyle="butt",
    )

    ax.text(
        x_start + length_um / 2,
        y_start + 0.01 * (y1 - y0),
        f"{length_um} µm",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )


def plot_cm_and_weighted_sta(
    dataset,
    sta_weighted,
    rec_config,
    cells_to_plot,
    reverse_channels=True,
    figsize=(17, 18),
    scale_bar_um=20,
    share_sta_ylim=False,
):
    """
    Create a multipanel plot with covariance maps and weighted STA traces.

    Layout for two cells:
    Rows 0-1 : Cell 1
        Row 0 -> cm_most_important
        Row 1 -> weighted STA

    Rows 2-3 : Cell 2
        Row 2 -> cm_most_important
        Row 3 -> weighted STA

    Parameters
    ----------
    dataset : xr.Dataset
        Original processed noise_data.nc dataset.

    sta_weighted : xr.DataArray
        Weighted STA output from calculate_weighted_sta_for_cells.
        Expected dims: cell_index, channel, time_max
        or: cell_index, channel, time

    rec_config : Recording_Config
        Recording config, used for channel colours.

    cells_to_plot : list[int]
        Cells to plot.

    reverse_channels : bool
        If True, plots channels in reverse order.

    figsize : tuple
        Figure size.

    scale_bar_um : float
        Scale bar length in micrometres.

    share_sta_ylim : bool
        If True, all STA plots share one common y-axis scale.
        If False, each STA plot is scaled symmetrically around 0.5.
    """

    # ------------------------------------------------------------
    # Detect weighted STA time dimension
    # ------------------------------------------------------------
    if "time_max" in sta_weighted.dims:
        time_dim = "time_max"
    elif "time" in sta_weighted.dims:
        time_dim = "time"
    else:
        raise ValueError(
            f"Could not find time dimension in sta_weighted. "
            f"Found dims: {sta_weighted.dims}"
        )

    # ------------------------------------------------------------
    # Channel order
    # ------------------------------------------------------------
    channels = list(dataset.channel.values)

    if reverse_channels:
        channels_to_plot = channels[::-1]
    else:
        channels_to_plot = channels

    n_cells = len(cells_to_plot)
    n_channels = len(channels_to_plot)
    n_rows = n_cells * 2

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_channels,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    fig.patch.set_alpha(0)

    # ------------------------------------------------------------
    # Optional global STA y limit
    # ------------------------------------------------------------
    if share_sta_ylim:
        all_sta_values = []

        for cell in cells_to_plot:
            for channel in channels_to_plot:
                if (
                    cell in sta_weighted.cell_index.values
                    and channel in sta_weighted.channel.values
                ):
                    vals = sta_weighted.sel(
                        cell_index=cell,
                        channel=channel,
                    ).values

                    all_sta_values.append(vals)

        all_sta_values = np.concatenate(all_sta_values)
        sta_min = np.nanmin(all_sta_values)
        sta_max = np.nanmax(all_sta_values)

        global_max_deviation = max(
            abs(sta_min - 0.5),
            abs(sta_max - 0.5),
        )

        global_sta_ylim = (
            0.5 - global_max_deviation,
            0.5 + global_max_deviation,
        )
    else:
        global_sta_ylim = None

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    for cell_idx, cell in enumerate(cells_to_plot):
        cm_row = cell_idx * 2
        sta_row = cell_idx * 2 + 1

        cell_data = dataset.sel(cell_index=cell)

        for plot_col, channel in enumerate(channels_to_plot):
            original_ch_idx = channels.index(channel)

            # ============================================================
            # CM IMAGE
            # ============================================================
            ax_img = axs[cm_row, plot_col]

            cm_data = cell_data.sel(channel=channel)["cm_most_important"]
            cm_data_cut = cutout_nans(cm_data, inset=0)

            vmax = np.nanmax(np.abs(cm_data.values))

            im = cm_data_cut.plot.imshow(
                ax=ax_img,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                add_colorbar=True,
            )

            add_scale_bar(ax_img, length_um=scale_bar_um)

            ax_img.grid(False)
            ax_img.set_title("")
            ax_img.set_xlabel("x position (µm)", labelpad=4, fontsize=20)
            ax_img.set_ylabel("y position (µm)", labelpad=12, fontsize=20)
            ax_img.tick_params(axis="both", labelsize=18)

            # ------------------------------------------------------------
            # Colorbar formatting
            # ------------------------------------------------------------
            cbar = im.colorbar

            pos = cbar.ax.get_position()
            cbar.ax.set_position(
                [
                    pos.x0 - 0.1,
                    pos.y0,
                    pos.width * 0.8,
                    pos.height,
                ]
            )

            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))

            cbar.formatter = formatter
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=17)

            if plot_col == n_channels - 1:
                cbar.set_label("Covariance", labelpad=12, fontsize=20)
            else:
                cbar.set_label("")

            # ============================================================
            # WEIGHTED STA
            # ============================================================
            ax_sta = axs[sta_row, plot_col]

            sta = sta_weighted.sel(
                cell_index=cell,
                channel=channel,
            )

            time = sta[time_dim].values

            ax_sta.plot(
                time,
                sta.values,
                color=rec_config.channel_colours[original_ch_idx],
                linewidth=2,
            )

            ax_sta.axvline(0, color="black", linestyle="--", linewidth=1)

            if share_sta_ylim:
                ax_sta.set_ylim(global_sta_ylim)
            else:
                sta_min = np.nanmin(sta.values)
                sta_max = np.nanmax(sta.values)

                max_deviation = max(
                    abs(sta_min - 0.5),
                    abs(sta_max - 0.5),
                )

                ax_sta.set_ylim(
                    0.5 - max_deviation,
                    0.5 + max_deviation,
                )

            # Only bottom-right STA plot
            # ------------------------------------------------------------
            # Only tidy the bottom-right STA y-axis labels
            # Do NOT force y-limits
            # ------------------------------------------------------------
            if sta_row == n_rows - 1 and plot_col == n_channels - 1:
                ymin, ymax = ax_sta.get_ylim()

                ticks = [ymin, 0.5, ymax]

                ax_sta.set_yticks(ticks)
                ax_sta.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            ax_sta.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            ax_sta.grid(False)
            ax_sta.set_title("")

            ax_sta.set_xlabel("Time (ms)", labelpad=4, fontsize=20)
            ax_sta.set_ylabel("STA", labelpad=12, fontsize=20)

            ax_sta.set_xticks(
                [
                    -800,
                    -400,
                    0,
                ]
            )

            ax_sta.tick_params(axis="both", labelsize=18)

    plt.show()
    plt.close("all")


plot_cm_and_weighted_sta(
    dataset=dataset,
    sta_weighted=sta_weighted,
    rec_config=rec_config,
    cells_to_plot=[235, 135],
    reverse_channels=True,
    share_sta_ylim=False,
)
