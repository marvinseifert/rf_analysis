# %% ============================================================
# Combine two processed 12 px white-noise STA channels
# using raw centred STA sums
# zebrafish_15_05_2026 / Phase_00
# ============================================================

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.ndimage import shift as ndi_shift

from organize.configs import (
    Recording_Config,
    Collapse_2d_Config,
)

from location.x_array import x_y_and_scale
from loading.load_sta import load_and_realign_center


# %% ============================================================
# Recording config
# ============================================================

rec_object = Recording_Config(
    root_path=Path(r"F:\Laura\zebrafish_15_05_2026\Phase_00"),
)

rec_object.add_channel(
    stimulus_id=5,
    name="12px_20Hz_25mins_shuffle_x12_365",
    colour="blue",
)

rec_object.add_channel(
    stimulus_id=11,
    name="12px_20Hz_25mins_shuffle_x12_repeat_365",
    colour="green",
)

rec_object.add_channel(
    stimulus_id=15,
    name="12px_20Hz_25mins_shuffle_x12_white",
    colour="red",
)

rec_object.add_channel(
    stimulus_id=19,
    name="12px_20Hz_25mins_shuffle_x12_repeat_white",
    colour="red",
)

collapse_2d_config = Collapse_2d_Config(
    recording_config=rec_object,
    cut_size_um=x_y_and_scale(800, 800),
)

noise_data_path = (
    rec_object.root_path / rec_object.output_folder / "noise_analysis" / "noise_data.nc"
)

print(f"Using dataset: {noise_data_path}")

dataset = xr.load_dataset(noise_data_path)

print("\nChannels found in noise_data.nc:")
for ch in dataset.channel.values:
    print(f"  {ch}")


# %% ============================================================
# Helper functions
# ============================================================


def centred_sta_from_subset(subset, nr_of_spikes):
    """
    Convert raw spike-triggered stimulus sum into centred STA.

    This matches your original calculate_rms logic:

        sta_per_spike_raw = subset / nr_of_spikes - 0.5

    Returns
    -------
    xr.DataArray
        STA centred around zero.
    """

    return subset / nr_of_spikes - 0.5


def centred_sum_from_subset(subset, nr_of_spikes):
    """
    Convert raw spike-triggered stimulus sum into raw centred STA sum.

    This is the key object to add across recordings.

    Since:
        sta = subset / n_spikes - 0.5

    then:
        centred_sum = sta * n_spikes
                     = subset - 0.5 * n_spikes

    Adding centred_sum_1 + centred_sum_2 and dividing by total spikes is
    equivalent to treating the two recordings as one long recording.
    """

    return subset - (0.5 * nr_of_spikes)


def get_sta_peak_yx_from_centred_sta(sta_centred):
    """
    Find RF peak using the same RMS-style method as your original pipeline:

        rms = (sta_per_spike_raw ** 2).max(dim='time')

    The peak is the pixel with the largest STA deviation across time.
    """

    rms_map = (sta_centred**2).max(dim="time").values

    peak_y, peak_x = np.unravel_index(
        np.nanargmax(rms_map),
        rms_map.shape,
    )

    return int(peak_y), int(peak_x)


def shift_sta_like_array(
    array,
    shift_yx,
    shift_order=1,
    mode="constant",
    cval=0.0,
):
    """
    Shift a time × y × x array spatially.

    The time axis is not shifted.
    """

    shift_y, shift_x = shift_yx

    shifted_values = ndi_shift(
        array.values,
        shift=(0, shift_y, shift_x),
        order=shift_order,
        mode=mode,
        cval=cval,
    )

    shifted_array = xr.DataArray(
        shifted_values.astype(np.float32),
        dims=array.dims,
        coords=array.coords,
        name=array.name,
    )

    return shifted_array


def calculate_covariance_map_from_sta(combined_sta):
    """
    Calculate covariance map from a combined centred STA movie.

    This follows your original logic:
    1. flatten STA movie to time × pixels
    2. mean-centre each pixel over time
    3. use SVD to find the most informative pixel
    4. calculate covariance of all pixels with that pixel

    Returns
    -------
    combined_cm : np.ndarray
        Covariance map, shape y × x.

    most_idx : int
        Flattened index of SVD-defined most informative pixel.
    """

    time_bins, h, w = combined_sta.shape

    raw_flat = combined_sta.values.reshape(time_bins, h * w).astype(
        np.float32,
        copy=False,
    )

    flat_centered = raw_flat - raw_flat.mean(axis=0, keepdims=True)

    try:
        _, _, Vt = np.linalg.svd(flat_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, np.nan

    loadings = Vt[0]
    most_idx = int(np.argmax(np.abs(loadings)))

    cov_with_most = (flat_centered.T @ flat_centered[:, most_idx]) / (time_bins - 1)

    combined_cm = cov_with_most.reshape(h, w).astype(np.float32)

    return combined_cm, most_idx


# %% ============================================================
# Main function: combine raw centred STA sums
# ============================================================


def calculate_combined_sta_rfs(
    noise_data_path,
    recording_config,
    collapse_2d_config,
    cell_ids,
    channel_groups,
    qi_limit=None,
    align_before_combine=True,
    alignment_target="median_peak",
    shift_order=1,
    add_sta_offset=True,
    verbose=True,
):
    """
    Combine processed STA data across channels as if they were one longer
    recording.

    Key logic
    ---------
    For each contributing channel:

        raw centred STA sum = subset - 0.5 * n_spikes

    Then combine:

        combined centred STA =
            sum(raw centred STA sums) / sum(n_spikes)

    This is equivalent to:

        combined STA = (subset_1 + subset_2) / (n1 + n2) - 0.5

    but using the centred-sum form makes alignment safer because the
    baseline outside shifted regions is zero.

    Parameters
    ----------
    noise_data_path : str or Path
        Existing noise_data.nc path.

    recording_config : Recording_Config
        Recording config.

    collapse_2d_config : Collapse_2d_Config
        Collapse config.

    cell_ids : list[int]
        Cells to process.

    channel_groups : dict
        Example:
        {
            "12px_20Hz_50mins_white_combined": [
                "12px_20Hz_25mins_shuffle_x12_white",
                "12px_20Hz_25mins_shuffle_x12_repeat_white",
            ],
        }

    qi_limit : float or None
        Optional quality threshold.

    align_before_combine : bool
        If True, align each channel's STA to a common RF peak before summing.

    alignment_target : str
        "median_peak" or "first_peak".

    shift_order : int
        Interpolation order for spatial shift.
        0 = nearest neighbour, 1 = linear.

    add_sta_offset : bool
        If True, add 0.5 back to combined single-pixel STA trace for plotting.

    verbose : bool
        Print diagnostics.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - combined_rms
        - combined_cm_most_important
        - combined_sta_single_pixel
        - combined_most_idx
        - combined_total_spikes
        - alignment diagnostics as attrs
    """

    if alignment_target not in ["median_peak", "first_peak"]:
        raise ValueError("alignment_target must be 'median_peak' or 'first_peak'.")

    ds = xr.load_dataset(noise_data_path)

    group_names = list(channel_groups.keys())
    target_time = ds["time_max"].values

    rms_store = []
    cm_store = []
    sta_store = []
    most_idx_store = []
    spike_store = []

    peak_y_store = []
    peak_x_store = []
    shift_y_store = []
    shift_x_store = []

    for cell_id in cell_ids:
        cell_rms_group = []
        cell_cm_group = []
        cell_sta_group = []
        cell_most_idx_group = []
        cell_spike_group = []

        cell_peak_y_group = []
        cell_peak_x_group = []
        cell_shift_y_group = []
        cell_shift_x_group = []

        for group_name in group_names:
            channels_to_combine = channel_groups[group_name]

            centred_sums = []
            centred_stas_for_peak = []
            spike_counts = []
            included_channels = []

            if verbose:
                print(f"\nCell {cell_id}, group: {group_name}")

            # ------------------------------------------------------------
            # Load each raw subset and convert to raw centred STA sum
            # ------------------------------------------------------------
            for channel in channels_to_combine:
                if cell_id not in ds.cell_index.values:
                    print(f"Cell {cell_id} not found in dataset.")
                    continue

                if channel not in ds.channel.values:
                    print(f"Channel not found in dataset: {channel}")
                    print("Available channels are:")
                    for ch in ds.channel.values:
                        print(f"  {ch}")
                    continue

                cell_data = ds.sel(cell_index=cell_id, channel=channel)

                quality = cell_data["quality"].item()

                if np.isnan(quality):
                    print(f"Skipping cell {cell_id}, {channel}: quality is NaN.")
                    continue

                if qi_limit is not None and quality < qi_limit:
                    print(
                        f"Skipping cell {cell_id}, {channel}: "
                        f"quality {quality:.2f} < {qi_limit}."
                    )
                    continue

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
                    continue

                subset, c_x, c_y, var_coordinates = result

                stimulus_id = int(
                    Path(cell_data["sta_path"].item()).parts[-3].split("_")[-1]
                )

                spike_query = recording_config.overview.spikes_df.query(
                    f"stimulus_index == {stimulus_id} and cell_index == {cell_id}"
                )

                if len(spike_query) == 0:
                    print(
                        f"No spike count found for cell {cell_id}, "
                        f"stimulus {stimulus_id}."
                    )
                    continue

                nr_of_spikes = float(spike_query["nr_of_spikes"].values[0])

                if nr_of_spikes <= 0:
                    print(f"Skipping cell {cell_id}, {channel}: no spikes.")
                    continue

                centred_sta = centred_sta_from_subset(subset, nr_of_spikes)
                centred_sum = centred_sum_from_subset(subset, nr_of_spikes)

                centred_stas_for_peak.append(centred_sta)
                centred_sums.append(centred_sum)
                spike_counts.append(nr_of_spikes)
                included_channels.append(channel)

                if verbose:
                    peak_y, peak_x = get_sta_peak_yx_from_centred_sta(centred_sta)
                    print(
                        f"  included {channel}: "
                        f"quality={quality:.2f}, "
                        f"spikes={int(nr_of_spikes)}, "
                        f"peak_yx=({peak_y}, {peak_x})"
                    )

            # ------------------------------------------------------------
            # No usable channels
            # ------------------------------------------------------------
            if len(centred_sums) == 0:
                print(f"No valid channels for cell {cell_id}, group {group_name}.")

                cell_rms_group.append(None)
                cell_cm_group.append(None)
                cell_sta_group.append(None)
                cell_most_idx_group.append(np.nan)
                cell_spike_group.append(0)

                cell_peak_y_group.append([])
                cell_peak_x_group.append([])
                cell_shift_y_group.append([])
                cell_shift_x_group.append([])

                continue

            # ------------------------------------------------------------
            # Check shapes match
            # ------------------------------------------------------------
            shapes = [arr.shape for arr in centred_sums]

            if len(set(shapes)) != 1:
                raise ValueError(
                    f"STA shapes do not match for cell {cell_id}, "
                    f"group {group_name}: {shapes}. "
                    "You need interpolation before combining."
                )

            # ------------------------------------------------------------
            # Find peaks and optional alignment shifts
            # ------------------------------------------------------------
            original_peaks = [
                get_sta_peak_yx_from_centred_sta(sta) for sta in centred_stas_for_peak
            ]

            if alignment_target == "median_peak":
                target_peak_y = int(np.round(np.median([p[0] for p in original_peaks])))
                target_peak_x = int(np.round(np.median([p[1] for p in original_peaks])))

            elif alignment_target == "first_peak":
                target_peak_y, target_peak_x = original_peaks[0]

            target_peak_yx = (target_peak_y, target_peak_x)

            peak_y_list = []
            peak_x_list = []
            shift_y_list = []
            shift_x_list = []

            if align_before_combine:
                aligned_centred_sums = []

                for centred_sum, original_peak_yx, channel in zip(
                    centred_sums,
                    original_peaks,
                    included_channels,
                ):
                    peak_y, peak_x = original_peak_yx

                    shift_y = target_peak_y - peak_y
                    shift_x = target_peak_x - peak_x

                    aligned_sum = shift_sta_like_array(
                        centred_sum,
                        shift_yx=(shift_y, shift_x),
                        shift_order=shift_order,
                        mode="constant",
                        cval=0.0,
                    )

                    aligned_centred_sums.append(aligned_sum)

                    peak_y_list.append(peak_y)
                    peak_x_list.append(peak_x)
                    shift_y_list.append(shift_y)
                    shift_x_list.append(shift_x)

                    if verbose:
                        print(
                            f"  aligned {channel}: "
                            f"original_peak=({peak_y}, {peak_x}), "
                            f"target_peak={target_peak_yx}, "
                            f"shift=({shift_y}, {shift_x})"
                        )

                centred_sums = aligned_centred_sums

            else:
                for peak_y, peak_x in original_peaks:
                    peak_y_list.append(peak_y)
                    peak_x_list.append(peak_x)
                    shift_y_list.append(0)
                    shift_x_list.append(0)

                if verbose:
                    print("  alignment skipped")

            # ------------------------------------------------------------
            # Combine as one long recording
            # ------------------------------------------------------------
            total_spikes = float(np.sum(spike_counts))

            combined_centred_sum_values = np.sum(
                np.stack(
                    [arr.values for arr in centred_sums],
                    axis=0,
                ),
                axis=0,
            ).astype(np.float32)

            combined_sta_values = (combined_centred_sum_values / total_spikes).astype(
                np.float32
            )

            combined_sta = xr.DataArray(
                combined_sta_values,
                dims=centred_sums[0].dims,
                coords=centred_sums[0].coords,
                name="combined_sta_centred",
            )

            time_bins, h, w = combined_sta.shape

            # ------------------------------------------------------------
            # Correct RMS-style RF
            # Matches original:
            # rms = (sta_per_spike_raw ** 2).max(dim="time")
            # ------------------------------------------------------------
            combined_rms = (combined_sta**2).max(dim="time").values.astype(np.float32)

            # ------------------------------------------------------------
            # Covariance RF
            # ------------------------------------------------------------
            combined_cm, most_idx = calculate_covariance_map_from_sta(combined_sta)

            if combined_cm is None:
                print(f"SVD failed for cell {cell_id}, group {group_name}.")

                cell_rms_group.append(combined_rms)
                cell_cm_group.append(None)
                cell_sta_group.append(None)
                cell_most_idx_group.append(np.nan)
                cell_spike_group.append(total_spikes)

                cell_peak_y_group.append(peak_y_list)
                cell_peak_x_group.append(peak_x_list)
                cell_shift_y_group.append(shift_y_list)
                cell_shift_x_group.append(shift_x_list)

                continue

            # ------------------------------------------------------------
            # Single-pixel STA from SVD-defined most informative pixel
            # ------------------------------------------------------------
            pix_y, pix_x = divmod(int(most_idx), w)

            combined_sta_single = combined_sta[:, pix_y, pix_x]

            if add_sta_offset:
                combined_sta_single = combined_sta_single + 0.5

            combined_sta_single_interp = combined_sta_single.interp(
                time=target_time,
                kwargs={"fill_value": 0},
            )

            cell_rms_group.append(combined_rms)
            cell_cm_group.append(combined_cm)
            cell_sta_group.append(combined_sta_single_interp.values.astype(np.float32))
            cell_most_idx_group.append(most_idx)
            cell_spike_group.append(total_spikes)

            cell_peak_y_group.append(peak_y_list)
            cell_peak_x_group.append(peak_x_list)
            cell_shift_y_group.append(shift_y_list)
            cell_shift_x_group.append(shift_x_list)

            if verbose:
                print(
                    f"  combined as one long recording, "
                    f"total spikes={int(total_spikes)}, "
                    f"combined most_idx={int(most_idx)}"
                )

        rms_store.append(cell_rms_group)
        cm_store.append(cell_cm_group)
        sta_store.append(cell_sta_group)
        most_idx_store.append(cell_most_idx_group)
        spike_store.append(cell_spike_group)

        peak_y_store.append(cell_peak_y_group)
        peak_x_store.append(cell_peak_x_group)
        shift_y_store.append(cell_shift_y_group)
        shift_x_store.append(cell_shift_x_group)

    # ---------------------------------------------------------------------
    # Create output arrays
    # ---------------------------------------------------------------------
    example_rms = None

    for row in rms_store:
        for item in row:
            if item is not None:
                example_rms = item
                break
        if example_rms is not None:
            break

    if example_rms is None:
        raise ValueError("No valid combined RFs were calculated.")

    h, w = example_rms.shape

    rms_array = np.full(
        (len(cell_ids), len(group_names), h, w),
        np.nan,
        dtype=np.float32,
    )

    cm_array = np.full_like(rms_array, np.nan)

    sta_array = np.full(
        (len(cell_ids), len(group_names), len(target_time)),
        np.nan,
        dtype=np.float32,
    )

    most_idx_array = np.full(
        (len(cell_ids), len(group_names)),
        np.nan,
        dtype=np.float32,
    )

    spike_array = np.full(
        (len(cell_ids), len(group_names)),
        np.nan,
        dtype=np.float32,
    )

    for i in range(len(cell_ids)):
        for j in range(len(group_names)):
            if rms_store[i][j] is not None:
                rms_array[i, j, :, :] = rms_store[i][j]

            if cm_store[i][j] is not None:
                cm_array[i, j, :, :] = cm_store[i][j]

            if sta_store[i][j] is not None:
                sta_array[i, j, :] = sta_store[i][j]

            most_idx_array[i, j] = most_idx_store[i][j]
            spike_array[i, j] = spike_store[i][j]

    combined_ds = xr.Dataset(
        data_vars={
            "combined_rms": (
                ("cell_index", "combined_channel", "y", "x"),
                rms_array,
            ),
            "combined_cm_most_important": (
                ("cell_index", "combined_channel", "y", "x"),
                cm_array,
            ),
            "combined_sta_single_pixel": (
                ("cell_index", "combined_channel", "time_max"),
                sta_array,
            ),
            "combined_most_idx": (
                ("cell_index", "combined_channel"),
                most_idx_array,
            ),
            "combined_total_spikes": (
                ("cell_index", "combined_channel"),
                spike_array,
            ),
        },
        coords={
            "cell_index": cell_ids,
            "combined_channel": group_names,
            "y": np.arange(h),
            "x": np.arange(w),
            "time_max": target_time,
        },
    )

    combined_ds.attrs["alignment_peak_y"] = str(peak_y_store)
    combined_ds.attrs["alignment_peak_x"] = str(peak_x_store)
    combined_ds.attrs["alignment_shift_y"] = str(shift_y_store)
    combined_ds.attrs["alignment_shift_x"] = str(shift_x_store)
    combined_ds.attrs["align_before_combine"] = str(align_before_combine)
    combined_ds.attrs["alignment_target"] = alignment_target
    combined_ds.attrs[
        "combination_logic"
    ] = "combined_STA = sum(subset_i - 0.5*n_spikes_i) / sum(n_spikes_i)"

    return combined_ds


# %% ============================================================
# Run combination
# ============================================================

cell_ids = [235, 27, 9, 31, 53]

combined_ds = calculate_combined_sta_rfs(
    noise_data_path=noise_data_path,
    recording_config=rec_object,
    collapse_2d_config=collapse_2d_config,
    cell_ids=cell_ids,
    channel_groups={
        "12px_20Hz_50mins_white_combined": [
            "12px_20Hz_25mins_shuffle_x12_white",
            "12px_20Hz_25mins_shuffle_x12_repeat_white",
        ],
    },
    qi_limit=20,
    align_before_combine=True,  # change to False to skip alignment
    alignment_target="first_peak",  # "median_peak" or "first_peak"
    shift_order=1,
    verbose=True,
)

print(combined_ds)


# %% ============================================================
# Plotting helpers
# ============================================================


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


def plot_individual_vs_combined_rfs(
    dataset,
    combined_ds,
    cells_to_plot,
    individual_channels,
    combined_channel,
    rf_type="rms",
    figsize_per_panel=(5, 5),
    cmap="coolwarm",
    crop_individual=True,
):
    """
    Plot individually calculated RFs next to the combined RF.

    Columns:
        individual channel 1
        individual channel 2
        combined channel

    rf_type:
        "rms" or "covariance"
    """

    n_rows = len(cells_to_plot)
    n_cols = len(individual_channels) + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            figsize_per_panel[0] * n_cols,
            figsize_per_panel[1] * n_rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    for row, cell_id in enumerate(cells_to_plot):
        # ============================================================
        # Individual RFs
        # ============================================================
        for col, channel in enumerate(individual_channels):
            ax = axes[row, col]

            if rf_type == "covariance":
                rf = dataset["cm_most_important"].sel(
                    cell_index=cell_id,
                    channel=channel,
                )

                cbar_label = "Covariance"
                plot_cmap = cmap
                vmax = np.nanmax(np.abs(rf.values))
                vmin = -vmax

            elif rf_type == "rms":
                rf = dataset["rms"].sel(
                    cell_index=cell_id,
                    channel=channel,
                )

                cbar_label = "RMS"
                plot_cmap = "gray"
                vmin = np.nanmin(rf.values)
                vmax = np.nanmax(rf.values)

            else:
                raise ValueError("rf_type must be 'rms' or 'covariance'.")

            if crop_individual:
                rf_plot = cutout_nans(rf)
            else:
                rf_plot = rf

            im = rf_plot.plot.imshow(
                ax=ax,
                cmap=plot_cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
            )

            ax.set_title(
                f"Cell {int(cell_id)}\n{channel}",
                fontsize=11,
            )

            ax.set_xlabel("x position")
            ax.set_ylabel("y position")
            ax.grid(False)

            cbar = im.colorbar
            cbar.set_label(cbar_label)

            if rf_type == "covariance":
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                cbar.formatter = formatter
                cbar.update_ticks()

        # ============================================================
        # Combined RF
        # ============================================================
        ax = axes[row, -1]

        if rf_type == "covariance":
            rf = combined_ds["combined_cm_most_important"].sel(
                cell_index=cell_id,
                combined_channel=combined_channel,
            )

            cbar_label = "Combined covariance"
            plot_cmap = cmap
            vmax = np.nanmax(np.abs(rf.values))
            vmin = -vmax

        elif rf_type == "rms":
            rf = combined_ds["combined_rms"].sel(
                cell_index=cell_id,
                combined_channel=combined_channel,
            )

            cbar_label = "Combined RMS"
            plot_cmap = "gray"
            vmin = np.nanmin(rf.values)
            vmax = np.nanmax(rf.values)

        im = ax.imshow(
            rf.values,
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )

        ax.set_title(
            f"Cell {int(cell_id)}\n{combined_channel}",
            fontsize=11,
        )

        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.grid(False)

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        cbar.set_label(cbar_label)

        if rf_type == "covariance":
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            cbar.formatter = formatter
            cbar.update_ticks()

    plt.show()


# %%
# Plot individual vs combined RFs

plot_individual_vs_combined_rfs(
    dataset=dataset,
    combined_ds=combined_ds,
    cells_to_plot=cell_ids,
    individual_channels=[
        "12px_20Hz_25mins_shuffle_x12_white",
        "12px_20Hz_25mins_shuffle_x12_repeat_white",
    ],
    combined_channel="12px_20Hz_50mins_white_combined",
    rf_type="covariance  ",  # can change to "covariance"
)
