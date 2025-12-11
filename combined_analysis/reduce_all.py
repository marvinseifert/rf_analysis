from __future__ import annotations
import polars as pl
import numpy as np
from pathlib import Path
import pickle
from polars import from_numpy
from dim_reduction.circular_reduction import polar_transform
from location.channel_handling import masking_square
from polarspike import Overview
from tqdm import tqdm
import warnings
from rf_torch.parameters import Noise_Params
import matplotlib.pyplot as plt
from smoothing.gaussian import smooth_ker
from covariance.filtering import cov_filtering_sum
from location.border import check_border_constrains
from loading.load_sta import load_sta_subset
import xarray as xr
from location.x_array import x_y_and_scale
from location.locate import locate_across_channels
from importlib import reload
from organize.decorators import depends_on
from loading.load_sta import load_sta_as_xarray, sta_time_dimension
import einops
from location.locate import px_position_to_um, px_int_to_um


def fill_defaults(recording_config, collapse_2d_config):
    """
    I scope function for handling exceptions by appending default values to the stores.

    Returns
    -------
    tuple cm_most_important_store, sta_single_store
        Appended default values to the stores.

    """
    nr_cells = recording_config.overview.nr_cells
    nr_channels = recording_config.nr_channels
    cm_most_important_store = np.empty((nr_cells * nr_channels), dtype=object)
    sta_single_store = np.empty((nr_cells * nr_channels), dtype=object)
    iter_position = 0
    for cell_idx in range(nr_cells):
        for channel in range(nr_channels):
            cm_most_important_store[iter_position] = np.zeros(
                (
                    collapse_2d_config.cut_size_px[
                        recording_config.channel_names[channel]
                    ]
                    .loc["x"]
                    .item(),
                    collapse_2d_config.cut_size_px[
                        recording_config.channel_names[channel]
                    ]
                    .loc["y"]
                    .item(),
                )
            )
            sta_single_store[iter_position] = np.zeros(
                (
                    recording_config.channel_configs[
                        recording_config.channel_names[channel]
                    ].total_sta_len,
                )
            )
            iter_position += 1
    cell_sta_coordinates = np.empty((nr_cells, nr_channels), dtype=object)
    return cm_most_important_store, sta_single_store, cell_sta_coordinates


@depends_on("calculate_rf_quality")
def sta_2d_cov_collapse(
        recording_config: "Recording_Config",
        collapse_2d_config: "Collapse_2d_Config",
        analysis_folder: str,
):
    channel_roots = [
        channel["root_path"] for channel in recording_config.channel_configs.values()
    ]
    # Calculate common location across channels
    cell_qualities, positions = locate_across_channels(
        channel_roots,
        threshold=collapse_2d_config.threshold,
        channel_names=recording_config.channel_names,
    )

    # %% save into xarray Dataset
    # 1. Define dimensions
    nr_cells = recording_config.overview.nr_cells

    nr_channels = recording_config.nr_channels

    # 2. Reshape Quality Data (Must be 2D: N_cells x N_channels)
    # The flatten("F") is correct for column-major (channel-fastest) storage,
    # BUT it must be reshaped back to 2D *after* flattening.
    quality_data = (
        cell_qualities.sel(metrics="quality")["quality_metrics"]
        .to_numpy()  # This is likely 2D already (channel x cell_index) from xarray/polars
        .flatten("F")  # This flattens it 1D, Fortran-style (channel repeats first)
        .reshape(nr_cells, nr_channels, order="C")
        # ^^^^^ IMPORTANT: Reshape to (N_cells, N_channels), using C-order
        # because the flattened F-order data naturally arranges itself that way when reshaping to (N_cells, N_channels)
    )

    # 3. Reshape Positions Data (Must be 2D: N_cells x 2)
    # The positions data (original shape likely N_cells * N_channels x 2) must be condensed to one entry per cell.
    # Assuming positions["quality_metrics"].to_numpy() is (N_cells * N_channels, 2) and the positions are repeated per channel:
    # We take only the first channel's position slice, resulting in (N_cells, 2)
    positions_data = positions["quality_metrics"].to_numpy()

    # 4. Reshape sta_path Data (Must be 2D: N_cells x N_channels)
    sta_paths_1d = np.array(
        [
            f"{channel_roots[channel].parts[-1]}/cell_{cell_idx}/kernel.npy"
            for cell_idx in cell_qualities["cell_index"].values
            for channel in range(nr_channels)
        ]
    )
    # Reshape this 1D array back to 2D (N_cells, N_channels) - C-order is the default and correct here
    sta_paths_2d = sta_paths_1d.reshape(nr_cells, nr_channels)

    ds = xr.Dataset(
        data_vars={
            # ('cell_index', 'channel') shape is (N_cells, N_channels)
            "quality": (("cell_index", "channel"), quality_data),
            # ('cell_index', 'pos_dim') shape is (N_cells, 2)
            "positions": (("cell_index", "pos_dim"), positions_data),
            # ('cell_index', 'channel') shape is (N_cells, N_channels)
            "sta_path": (("cell_index", "channel"), sta_paths_2d),
        },
        coords={
            "cell_index": cell_qualities["cell_index"].values,
            "channel": recording_config.channel_names,  # ['610_nm', '535_nm']
            "pos_dim": ["x", "y"],
        },
    )
    # save as netcdf file
    ds.to_netcdf(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc"
    )

    # # %% Construct the results dataframe
    # schema = {
    #     "cell_index": pl.UInt32,
    #     "channel": pl.Categorical,
    #     "sta_path": pl.Utf8,
    #     "positions": pl.Array(pl.UInt16, 2),
    #     "quality": pl.Float64,
    # }
    # sta_paths = [
    #     f"{channel_root.parts[-1]}/cell_{cell_idx}/kernel.npy"
    #     for cell_idx in cell_qualities["cell_index"].values
    #     for channel_root in channel_roots
    # ]
    # data = {
    #     "cell_index": cell_qualities["cell_index"]
    #     .values.repeat(recording_config.nr_channels)
    #     .astype(np.uint32),
    #     "channel": recording_config.channel_names * cell_qualities.sizes["cell_index"],
    #     "sta_path": [path for path in sta_paths],
    #     "positions": positions["quality_metrics"]
    #     .to_numpy()
    #     .astype(np.uint16)
    #     .repeat(recording_config.nr_channels, axis=0),
    #     "quality": cell_qualities.sel(metrics="quality")["quality_metrics"]
    #     .to_numpy()
    #     .flatten("F"),
    # }
    # df = pl.DataFrame(data, schema=schema)
    # # save as parquet file
    # df.write_parquet(
    #     recording_config.root_path
    #     / recording_config.output_folder
    #     / analysis_folder
    #     / "noise_df.parquet"
    # )

    cm_most_important_store, sta_single_store, cell_sta_coordinates = fill_defaults(
        recording_config, collapse_2d_config
    )

    # %% Step two sta extension and covariance matrix
    # This is a for loop, because the data is too large for running in parallel.
    plot_trigger = 0
    store_position = 0
    for cell_idx in tqdm(
            range(recording_config.overview.nr_cells),
            desc="Calculating covariance matrices",
    ):
        # Select all data (quality, positions, sta_path) for the current cell_idx
        try:
            cell_data = ds.sel(cell_index=cell_idx)
        except KeyError:
            print(f"Cell index {cell_idx} not found in dataset, skipping.")
            store_position += 1
            continue
        # Extract relevant info from the row
        for channel in recording_config.channel_names:
            channel_data = cell_data.sel(channel=channel)
            if channel_data["quality"].item() < collapse_2d_config.threshold:
                store_position += 1
                continue

            try:
                # Load STA subset
                subset, c_x, c_y = load_sta_subset(
                    recording_config.root_path / channel_data["sta_path"].item(),
                    positions=tuple(channel_data["positions"].values.astype(int)),
                    subset_size=tuple(
                        [
                            collapse_2d_config.cut_size_px[channel].loc["x"].item(),
                            collapse_2d_config.cut_size_px[channel].loc["y"].item(),
                        ]
                    ),
                    dt_ms=recording_config.channel_configs[channel].dt_ms,
                    t_zero_index=recording_config.channel_configs[channel].total_sta_len
                                 - recording_config.channel_configs[channel].post_spike_bins,
                )
                subset = px_position_to_um(
                    subset,
                    recording_config.channel_configs[channel].pixel_size,
                    collapse_2d_config.extended_cut_size_px[channel],
                )
                cell_sta_coordinates[
                    cell_idx, recording_config.channel_names.index(channel)
                ] = xr.DataArray(
                    data=subset.var("time").values,
                    dims=["y", "x"],
                    coords={"x": subset.coords["x"], "y": subset.coords["y"]},
                )
            except FileNotFoundError:
                print(f"File not found for item {cell_idx}")
                store_position += 1
                continue

            if plot_trigger == 0:
                # Sanity plot for the first cell only
                plot_trigger = 1
                fig, ax = plt.subplots(figsize=(10, 10))
                ishow = subset.var("time").plot.imshow(
                    ax=ax,
                    cmap="gray",
                    x="x",
                    y="y",
                    cbar_kwargs={"shrink": 0.5},
                )
                ax.set_aspect("equal")
                ax.set_title("Sanity plot subset")
                c_x_um, c_y_um = px_int_to_um(
                    c_x,
                    c_y,
                    recording_config.channel_configs[channel].pixel_size,
                    collapse_2d_config.extended_cut_size_px[channel],
                )
                ax.scatter(
                    c_x_um,
                    c_y_um,
                    color="red",
                    marker="x",
                )
                fig.show()

            # Get number of spikes for normalization

            stimulus_id = int(channel_data["sta_path"].item().split("_")[-2][:1])
            nr_of_spikes = recording_config.overview.spikes_df.query(
                f"stimulus_index=={stimulus_id}&cell_index=={cell_idx}"
            )["nr_of_spikes"].values[0]

            sta_per_spike_raw = (
                    subset / nr_of_spikes - 0.5
            ).values  # shape: (time, h, w)

            time_bins, h, w = sta_per_spike_raw.shape

            flat = sta_per_spike_raw.reshape(time_bins, h * w).astype(
                np.float32, copy=False
            )

            flat_centered = flat - flat.mean(axis=0, keepdims=True)

            # SVD on the CENTERED data
            try:
                # Use flat_centered here, NOT flat (raw)
                _, s, Vt = np.linalg.svd(flat_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                print(f"SVD did not converge for cell {cell_idx}, skipping.")
                store_position += 1
                continue
            # Covariance with most important pixel
            loadings = Vt[0]  # shape: (h*w,)
            most_idx = int(np.argmax(np.abs(loadings)))
            cov_with_most = (flat_centered.T @ flat_centered[:, most_idx]) / (
                    time_bins - 1
            )
            cov_with_most = np.reshape(
                cov_with_most,
                (
                    h,
                    w,
                ),
            )
            cm_most_important_store[store_position] = cov_with_most
            # Store STA time course of the selected pixel
            pix_y, pix_x = divmod(most_idx, w)
            sta_single_store[store_position] = sta_per_spike_raw[:, pix_y, pix_x] + 0.5

        #

    # Data reshaping for optimap storing.
    # If the data are from noise with different pixel sizes, we need to interpolate them to the same coordinate system.
    # For example, if we have one noise at size 4 um per pixel and another at 8 um per pixel,
    # we need to interpolate the 8 um data to 4 um per pixel. We can do this using xarray's interpolation functions.
    # First, test if interpolation is needed, if not we can skip this step.

    pixel_sizes = [
        recording_config.channel_configs[channel].pixel_size
        for channel in recording_config.channel_names
    ]
    if not all(np.array(pixel_sizes) == pixel_sizes[0]):
        # Interpolation needed
        print("Interpolating data to common pixel size for storage.")
        common_pixel_size = min(pixel_sizes)
        # for idx, channel in enumerate(recording_config.channel_names):
        # TODO: Implement interpolation to common pixel size
    else:
        common_pixel_size = pixel_sizes[0]
        cm_most_important_store = np.stack(cm_most_important_store, axis=0)

    # Equally, we need to interpolate the time dimension if they differ across channels.
    # Here we need to consider the sampling rate but also the time of the spike t=0.
    channel_dt = [
        recording_config.channel_configs[channel].dt_ms
        for channel in recording_config.channel_names
    ]
    if not all(np.array(channel_dt) == channel_dt[0]):
        warnings.warn(
            "Channels have different time resolutions. Using time interpolation for storage."
        )
    else:
        common_dt = channel_dt[0]
        sta_single_store = np.stack(sta_single_store, axis=0)

    # Determine duration in ms of sta before the spike
    time_lengths = [
        (
                recording_config.channel_configs[channel].total_sta_len
                - recording_config.channel_configs[channel].post_spike_bins
        )
        * recording_config.channel_configs[channel].dt_ms
        for channel in recording_config.channel_names
    ]
    # Since we dont want to extrapolate, we use the minimum time length
    common_before_spike_ms = min(time_lengths)

    post_spike_time = [
        recording_config.channel_configs[channel].post_spike_bins
        * recording_config.channel_configs[channel].dt_ms
    ]
    common_post_spike_time = min(post_spike_time)

    common_total_sta_length = np.ceil(
        common_before_spike_ms / common_dt + common_post_spike_time / common_dt
    ).astype(int)

    # Caculate the size in um of the complete sta image. All cutouts from all channels will fit in this space.
    max_y_size_um = max(
        (
                recording_config.channel_configs[channel]["image_shape"].loc["y"].item()
                + collapse_2d_config.border_buffer_px[channel].loc["y"].item() * 2
        )  # both sides)
        * common_pixel_size
        for channel in recording_config.channel_names
    )
    max_x_size_um = max(
        (
                recording_config.channel_configs[channel]["image_shape"].loc["x"].item()
                + collapse_2d_config.border_buffer_px[channel].loc["x"].item() * 2
        )  # both sides)
        * common_pixel_size
        for channel in recording_config.channel_names
    )
    max_y_size_px = int(np.ceil(max_y_size_um / common_pixel_size))
    max_x_size_px = int(np.ceil(max_x_size_um / common_pixel_size))

    pixel_space = x_y_and_scale(max_x_size_px, max_y_size_px)

    # map common coordinates
    y_coords_px = np.arange(0, max_y_size_px)
    x_coords_px = np.arange(0, max_x_size_px)
    y_coords_um = (y_coords_px - max_y_size_px / 2) * common_pixel_size
    x_coords_um = (x_coords_px - max_x_size_px / 2) * common_pixel_size

    # Bring all toegther into an xarray Dataset for storage
    time_coordinates = np.arange(
        -int(common_before_spike_ms),
        int(common_post_spike_time),
        common_dt,
    )

    nr_cells = ds.sizes["cell_index"]
    nr_channels = ds.sizes["channel"]

    cm_final_array = np.full(
        (nr_cells, nr_channels, max_y_size_px, max_x_size_px), np.nan, dtype=np.float32
    )
    # Prepare the final (N_cells, N_channels, Max_Time) array
    sta_final_array = np.full(
        (nr_cells, nr_channels, common_total_sta_length), np.nan, dtype=np.float32
    )

    cm_most_important_store = einops.rearrange(
        cm_most_important_store,
        " (cell channel) y x -> cell channel y x ",
        cell=nr_cells,
        channel=nr_channels,
    )

    sta_single_store = einops.rearrange(
        sta_single_store,
        " (cell channel) time -> cell channel time ",
        cell=nr_cells,
        channel=nr_channels,
    )
    store_idx = 0
    for cell_idx in range(nr_cells):
        for channel_idx in range(nr_channels):
            coords = cell_sta_coordinates[cell_idx][channel_idx]
            if np.all(coords == None):
                store_idx += 1
                continue
            y_indices = np.searchsorted(y_coords_um, coords["y"].values)
            x_indices = np.searchsorted(x_coords_um, coords["x"].values)

            # create meshgrid for indexing
            y_grid, x_grid = np.ix_(y_indices, x_indices)
            # Place the cm_data at the correct coordinates
            cm_final_array[
                cell_idx,
                channel_idx,
                y_grid,
                x_grid,
            ] = cm_most_important_store[cell_idx, channel_idx, :, :]
            sta_final_array[cell_idx, channel_idx, :] = sta_single_store[
                cell_idx, channel_idx, :
            ]
            store_idx += 1

    # Add new dimensions and variables to the dataset
    ds = ds.assign_coords(
        y=np.arange(max_y_size_px),
        x=np.arange(max_x_size_px),
        time_max=np.arange(common_total_sta_length),
    )

    # Note: These coordinates are not fully correct, as they don't map to physical units
    # for the entire padded space. They are placeholders for NetCDF structure.
    # The actual physical coordinates should be stored as separate data variables if needed.

    # Add the padded arrays to the dataset
    ds["cm_most_important"] = (
        ("cell_index", "channel", "y" "", "x"),
        cm_final_array,
    )
    ds["sta_single_pixel"] = (
        ("cell_index", "channel", "time_max"),
        sta_final_array,
    )

    print("Finished, saving dataset.")

    # Save the updated dataset
    ds.to_netcdf(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc"
    )


@depends_on("calculate_rf_quality")
@depends_on("sta_2d_cov_collapse")
def circular_reduction(
        recording_config: "Recording_Config",
        collapse_2d_config: "Collapse_2d_Config",
        circular_reduction_config: "Circular_Reduction_Config",
        analysis_folder: str,
):
    df = pl.read_parquet(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_df.parquet"
    )

    # %%

    center_outline_store = xr.DataArray(
        np.zeros((len(df), int(360 / circular_reduction_config.degree_bins))),
        dims=["cell", "degree"],
        coords={
            "cell": df["cell_index"].to_numpy(),
            "degree": np.arange(0, 360, circular_reduction_config.degree_bins),
        },
    )
    surround_outline_store = xr.full_like(center_outline_store, 0)
    in_out_outline_store = (
        []
    )  # Need List, because radius can vary for different channels
    center_std = np.zeros((len(df)))
    surround_std = np.zeros_like(center_std)

    for row_idx, row in tqdm(
            enumerate(df.iter_rows()),
            total=len(df),
            desc="Performing circular reduction",
    ):
        channel = row[df.get_column_index("channel")]
        if row[df.get_column_index("quality")] < collapse_2d_config.threshold:
            in_out_outline_store.append(
                np.zeros(int(round(collapse_2d_config.max_radius_px[channel].item())))
            )
            continue
        center = (
            collapse_2d_config.half_cut_size_px[channel].loc["x"].item(),
            collapse_2d_config.half_cut_size_px[channel].loc["y"].item(),
        )
        polar_cm = polar_transform(
            np.asarray(row[df.get_column_index("cm_most_important")]).reshape(
                collapse_2d_config.cut_size_px[channel].loc["y"].item(),
                collapse_2d_config.cut_size_px[channel].loc["x"].item(),
            ),
            center,
            max_radius=int(round(collapse_2d_config.max_radius_px[channel].item())),
        )
        entries = []
        for deg_step in range(0, 360, circular_reduction_config.degree_bins):
            entries.append(
                np.mean(
                    polar_cm[
                        deg_step: deg_step + circular_reduction_config.degree_bins,
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

        center_outline_store[row_idx, :] = entries_mean
        surround_outline_store[row_idx, :] = entries_mean_neg
        in_out_outline_store.append(np.mean(entries, axis=0))

    # %% Add results to the dataframe
    center_outline = pl.Series(
        "center_outline",
        center_outline_store.values,
        dtype=pl.Array(pl.Float64, 360 // circular_reduction_config.degree_bins),
    )
    surround_outline = pl.Series(
        "surround_outline",
        surround_outline_store.values,
        dtype=pl.Array(pl.Float64, 360 // circular_reduction_config.degree_bins),
    )
    in_out_outline = pl.Series(
        "in_out_outline",
        in_out_outline_store,
        dtype=pl.List(pl.Float64),
    )

    nr_columns = len(df.columns)

    for series_idx, series in enumerate(
            [center_outline, surround_outline, in_out_outline]
    ):
        df.with_columns(series)

    df.write_parquet(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_df.parquet"
    )
    return
