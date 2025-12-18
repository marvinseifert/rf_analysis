from __future__ import annotations

import warnings
from itertools import product

import einops
import numpy as np
import xarray as xr
from tqdm import tqdm

from dim_reduction.circular_reduction import polar_transform
from loading.load_sta import load_and_realign_center
from location.locate import locate_across_channels
from location.x_array import x_y_and_scale
from organize.decorators import depends_on
import matplotlib.pyplot as plt
from location.locate import px_int_to_um
from dim_reduction.rms_error import calculate_rms
from dim_reduction.covariance import calculate_covariance


def plot_sanity_fig(
    subset: xr.DataArray,
    c_x: int,
    c_y: int,
    channel: str,
    recording_config: "Recording_Config",
    collapse_2d_config: "Collapse_2d_Config",
):
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


def fill_defaults(  # noqa: F821  # noqa: F821
    recording_config: "Recording_Config", collapse_2d_config: "Collapse_2d_Config"
) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function pre-fills the stores with default values (zeros) for all cells and channels based
    on the information from the recording and collapse configurations.

    Parameters
    ----------
    recording_config : Recording_Config
        The recording configuration containing details about the recording setup.
    collapse_2d_config : Collapse_2d_Config
        The collapse 2D configuration containing details about the cut sizes.
    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        A tuple containing:
        - cm_most_important_store: An array to store the most important covariance matrices.
        - sta_single_store: An array to store single pixel STA time courses.
        - rms_store: An array to store RMS values.
        - cell_sta_coordinates: An array to store STA coordinates for each cell and channel.

    """
    nr_cells = np.max(recording_config.overview.spikes_df["cell_index"]) + 1
    nr_channels = recording_config.nr_channels
    cm_most_important_store = np.empty((nr_cells * nr_channels), dtype=object)
    rms_store = np.empty_like(cm_most_important_store)
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
            rms_store[iter_position] = np.zeros(
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
    return cm_most_important_store, sta_single_store, rms_store, cell_sta_coordinates


@depends_on("calculate_rf_quality")
def sta_2d_cov_collapse(
    recording_config: "Recording_Config",
    collapse_2d_config: "Collapse_2d_Config",
    analysis_folder: str,
) -> None:
    """
    Perform STA 2D covariance collapse across multiple channels and save the results.
    Parameters
    ----------
    recording_config : Recording_Config
        The recording configuration containing details about the recording setup.
    collapse_2d_config : Collapse_2d_Config
        The collapse 2D configuration containing details about the cut sizes and thresholds.
    analysis_folder : str
        The folder where analysis results will be saved.
    Returns
    -------
    None (saves results to disk)
    """
    # 1. Extract some general parameters
    channel_roots = [
        channel["root_path"] for channel in recording_config.channel_configs.values()
    ]
    nr_cells = np.max(recording_config.overview.spikes_df["cell_index"]) + 1
    nr_channels = recording_config.nr_channels

    # 2. Locate cells across channels
    # This is done by taking the weighted average of the positions of the cells in each individual channel,
    # with the weights being the quality metrics of the cells in each channel.
    cell_qualities, positions = locate_across_channels(
        channel_roots,
        threshold=collapse_2d_config.threshold,
        channel_names=recording_config.channel_names,
        channel_configs=recording_config.channel_configs,
    )

    # 3. Reshape Quality Data (Must be 2D: N_cells,  N_channels)
    quality_data = (
        cell_qualities.sel(metrics="quality")
        .to_numpy()  # This is likely 2D already (channel x cell_index) from xarray/polars
        .flatten("F")  # This flattens it 1D, Fortran-style (channel repeats first)
        .reshape(nr_cells, nr_channels, order="C")
        # ^^^^^ IMPORTANT: Reshape to (N_cells, N_channels), using C-order
        # because the flattened F-order data naturally arranges itself that way when reshaping to (N_cells, N_channels)
    )

    # 4. Create STA paths array which is ( N_cells, N_channels)
    sta_paths_1d = np.array(
        [
            f"{channel_roots[channel].parts[-1]}/cell_{cell_idx}/kernel.npy"
            for cell_idx in cell_qualities["cell_index"].values
            for channel in range(nr_channels)
        ]
    )
    sta_paths_2d = sta_paths_1d.reshape(nr_cells, nr_channels)

    # 5. Create empty xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "quality": (("cell_index", "channel"), quality_data),
            "positions": (
                ("cell_index", "pos_dim", "channel"),
                einops.rearrange(
                    positions.to_numpy(),
                    "channel cells positions -> cells positions channel",
                ),
            ),
            "sta_path": (("cell_index", "channel"), sta_paths_2d),
        },
        coords={
            "cell_index": cell_qualities["cell_index"].values,
            "channel": recording_config.channel_names,
            "pos_dim": ["x", "y"],
        },
    )
    # save as netcdf file with compression
    ds.to_netcdf(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc",
        engine="netcdf4",
        encoding={var: {"zlib": True, "complevel": 9} for var in ds.data_vars},
    )
    # 6. Define default output values (zeros) for all variables to be calculated in the loop
    (
        cm_most_important_store,
        sta_single_store,
        rms_store,
        cell_sta_coordinates,
    ) = fill_defaults(recording_config, collapse_2d_config)

    plot_trigger = 0  # This is a switch to only plot the first cell for sanity check
    # 7. Loop over all cells and channels to calculate covariance matrices
    for cell_pos, cell_idx in tqdm(
        enumerate(ds["cell_index"].to_numpy()),
        total=ds.sizes["cell_index"],
        desc="Calculating covariance matrices",
    ):
        # Loop over channels
        for channel_index, channel in enumerate(recording_config.channel_names):
            channel_data = ds.sel(cell_index=cell_idx, channel=channel)
            flat_idx = cell_idx * nr_channels + channel_index
            # Skip low quality cells
            if np.isnan(channel_data["quality"].item()):
                continue
            if channel_data["quality"].item() < collapse_2d_config.threshold:
                continue

            result = load_and_realign_center(
                recording_config,
                collapse_2d_config,
                cell_idx,
                channel,
                positions.sel(cell_index=cell_idx, channel=channel),
            )
            # The function will return None if the cell could not be loaded or aligned
            if result is None:
                continue
            subset, c_x, c_y, var_coordinates = result
            cell_sta_coordinates[cell_idx, channel_index] = var_coordinates
            # update positions
            positions.loc[dict(cell_index=cell_idx, channel=channel)] = xr.DataArray(
                data=np.array([c_x, c_y]),
                dims=["metrics"],
                coords={"metrics": ["center_x", "center_y"]},
            )
            if plot_trigger == 0:
                # Sanity plot for the first cell only
                plot_trigger = 1
                plot_sanity_fig(
                    subset, c_x, c_y, channel, recording_config, collapse_2d_config
                )

            # Now that the data are loaded, we can calculate RMS and covariance map
            # 8. Calculate RMS
            stimulus_id = int(channel_data["sta_path"].item().split("_")[-2][:1])
            nr_of_spikes = recording_config.overview.spikes_df.query(
                f"stimulus_index=={stimulus_id}&cell_index=={cell_idx}"
            )["nr_of_spikes"].values[0]
            sta_per_spike_raw, rms = calculate_rms(subset, nr_of_spikes)
            rms_store[flat_idx] = rms.values.astype(np.float32, copy=False)
            time_bins, h, w = sta_per_spike_raw.shape

            # 9. Reshape data for SVD
            flat = sta_per_spike_raw.values.reshape(time_bins, h * w).astype(
                np.float32, copy=False
            )
            results = calculate_covariance(
                flat - flat.mean(axis=0, keepdims=True), time_bins
            )
            if results is None:
                print(
                    f"SVD failed, could not calculate covariance map for cell {cell_idx}."
                )
                continue
            cov_with_most, most_idx = results
            cov_with_most = np.reshape(
                cov_with_most,
                (
                    h,
                    w,
                ),
            )
            cm_most_important_store[flat_idx] = cov_with_most
            # Store STA time course of the selected pixel
            pix_y, pix_x = divmod(most_idx, w)
            sta_single_store[flat_idx] = (
                sta_per_spike_raw[:, pix_y, pix_x] + 0.5
            ).values

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
        sta_single_store[568] = sta_single_store[568][60:160]
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

    rms_final = np.full_like(cm_final_array, np.nan, dtype=np.float32)
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
    rms_store = einops.rearrange(
        np.stack(rms_store, axis=0),
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
            rms_final[
                cell_idx,
                channel_idx,
                y_grid,
                x_grid,
            ] = rms_store[cell_idx, channel_idx, :, :]
            # Place the sta time course at the correct time indices
            sta_final_array[cell_idx, channel_idx, :] = sta_single_store[
                cell_idx, channel_idx, :
            ]
            store_idx += 1

    # Add new dimensions and variables to the dataset
    ds = ds.assign_coords(
        y=y_coords_um,
        x=x_coords_um,
        time_max=np.arange(
            -int(common_before_spike_ms / common_dt),
            int(common_post_spike_time / common_dt),
            1,
        ),
    )

    # Note: These coordinates are not fully correct, as they don't map to physical units
    # for the entire padded space. They are placeholders for NetCDF structure.
    # The actual physical coordinates should be stored as separate data variables if needed.

    # Add the padded arrays to the dataset
    ds["cm_most_important"] = (
        ("cell_index", "channel", "y", "x"),
        cm_final_array,
    )
    ds["rms"] = (("cell_index", "channel", "y", "x"), rms_final)
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
        / "noise_data.nc",
        engine="netcdf4",
        encoding={var: {"zlib": True, "complevel": 9} for var in ds.data_vars},
    )


@depends_on("calculate_rf_quality")
@depends_on("sta_2d_cov_collapse")
def circular_reduction(
    recording_config: "Recording_Config",
    collapse_2d_config: "Collapse_2d_Config",
    circular_reduction_config: "Circular_Reduction_Config",
    analysis_folder: str,
):
    ds = xr.load_dataset(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc",
    )

    # %%

    center_outline_store = xr.DataArray(
        np.zeros(
            (
                ds.cell_index.shape[0],
                int(360 / circular_reduction_config.degree_bins),
                ds.channel.shape[0],
            )
        ),
        dims=["cell_index", "degree", "channel"],
        coords={
            "cell_index": ds.cell_index.values,
            "degree": np.arange(0, 360, circular_reduction_config.degree_bins),
            "channel": ds.channel.values,
        },
    )
    surround_outline_store = xr.full_like(center_outline_store, 0)
    max_radius = np.ceil(
        np.max([radius.values for radius in collapse_2d_config.max_radius_px.values()])
    ).astype(int)
    in_out_outline_store = xr.DataArray(
        np.zeros((ds.cell_index.shape[0], max_radius, ds.channel.shape[0])),
        dims=["cell_index", "radius", "channel"],
        coords={
            "cell_index": ds.cell_index.values,
            "radius": np.arange(0, max_radius),
            "channel": ds.channel.values,
        },
    )
    center_std = xr.DataArray(
        np.zeros((ds.cell_index.shape[0], ds.channel.shape[0])),
        dims=["cell_index", "channel"],
        coords={
            "cell_index": ds.cell_index.values,
            "channel": ds.channel.values,
        },
    )
    surround_std = xr.DataArray(
        np.zeros((ds.cell_index.shape[0], ds.channel.shape[0])),
        dims=["cell_index", "channel"],
        coords={
            "cell_index": ds.cell_index.values,
            "channel": ds.channel.values,
        },
    )
    channel_names = recording_config.channel_names
    with np.errstate(divide="ignore", invalid="ignore"):
        for (cell_position, cell_index), channel in tqdm(
            product(enumerate(ds.cell_index), channel_names),
            total=ds.cell_index.shape[0] * len(channel_names),
            desc="Performing circular reduction",
        ):
            if np.isnan(ds.sel(cell_index=cell_index, channel=channel).quality):
                continue
            if (  # Skip low quality cells
                ds.sel(cell_index=cell_index, channel=channel).quality
                < collapse_2d_config.threshold
            ):
                continue

            center = (
                collapse_2d_config.half_cut_size_px[channel].loc["x"].item(),
                collapse_2d_config.half_cut_size_px[channel].loc["y"].item(),
            )
            cm_most_important_unpadded = ds.sel(
                cell_index=cell_index, channel=channel
            ).cm_most_important.values[
                ~np.isnan(
                    ds.sel(
                        cell_index=cell_index, channel=channel
                    ).cm_most_important.values
                )
            ]
            polar_cm = polar_transform(
                cm_most_important_unpadded.reshape(
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
                            deg_step : deg_step + circular_reduction_config.degree_bins,
                            :,
                        ],
                        axis=0,
                    )
                )
            entries = np.asarray(entries)
            entries_pos = entries.copy()
            entries_pos[entries_pos < 0] = 0
            entries_mean = (
                np.mean(entries_pos / np.max(entries_pos), axis=1)
                * entries_pos.shape[1]
            )

            entries_neg = entries.copy()
            entries_neg[entries_neg > 0] = 0
            entries_mean_neg = (
                np.mean(entries_neg / np.min(entries_neg), axis=1)
                * entries_neg.shape[1]
            )
            entries_mean[np.isnan(entries_mean)] = 0
            entries_mean_neg[np.isnan(entries_mean_neg)] = 0

            center_outline_store.loc[
                dict(cell_index=cell_index, channel=channel)
            ] = entries_mean
            surround_outline_store.loc[
                dict(cell_index=cell_index, channel=channel)
            ] = entries_mean_neg
            in_out_outline_store.loc[
                dict(cell_index=cell_index, channel=channel)
            ] = np.mean(entries, axis=0)

    min_pixel_size = np.min(
        [
            recording_config.channel_configs[channel].pixel_size
            for channel in recording_config.channel_configs
        ]
    )
    # %% Add results to the datastore
    ds["center_outline_um"] = center_outline_store * min_pixel_size
    ds["surround_outline_um"] = surround_outline_store * min_pixel_size
    ds["in_out_outline_um"] = in_out_outline_store.assign_coords(
        radius=in_out_outline_store.radius.values * min_pixel_size
    )
    ds["center_std"] = center_std
    ds["surround_std"] = surround_std
    # Save the updated dataset
    print("Finished circular reduction, saving dataset.")
    ds.to_netcdf(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc",
        engine="netcdf4",
        encoding={var: {"zlib": True, "complevel": 9} for var in ds.data_vars},
    )

    return
