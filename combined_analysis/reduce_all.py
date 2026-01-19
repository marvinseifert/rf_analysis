from __future__ import annotations

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
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


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


def interpolate_sta_timecourse(
    array, min_time_old, max_time_old, min_time_new, max_time_new, new_time_bins
):
    """
    Interpolates a 1D array from old time coordinates to new time coordinates.
    """
    # Calculate size using round to avoid floating point truncation errors (e.g. 9.99->9)
    expected_size = int(np.round((max_time_new - min_time_new) / new_time_bins)) + 1

    # Check if interpolation is strictly necessary
    if (
        min_time_old == min_time_new
        and max_time_old == max_time_new
        and array.shape[0] == expected_size
    ):
        return array

    # Create time vectors
    old_time = np.linspace(min_time_old, max_time_old, array.shape[0])
    new_time = np.linspace(min_time_new, max_time_new, expected_size)

    # Use numpy's native 1D interpolation (faster and handles boundaries gracefully)
    # left=0, right=0 handles values outside the old time range (padding with zeros)
    new_array = np.interp(new_time, old_time, array, left=0, right=0)

    return new_array


def interpolate_2d_results(array, old_x_len, old_y_len, new_x_len, new_y_len):
    """
    Interpolates a 2D array from old coordinates to new coordinates using bilinear interpolation.

    Parameters
    ----------
    array : np.ndarray
        The 2D array to be interpolated.
    old_x_len : int
        The length of the original x-dimension.
    old_y_len : int
        The length of the original y-dimension.
    new_x_len : int
        The length of the new x-dimension.
    new_y_len : int
        The length of the new y-dimension.

    Returns
    -------
    np.ndarray
        The interpolated 2D array.
    """
    if old_x_len == new_x_len and old_y_len == new_y_len:
        return array  # No interpolation needed
    else:
        old_x = np.linspace(0, 1, old_x_len)
        old_y = np.linspace(0, 1, old_y_len)
        new_x = np.linspace(0, 1, new_x_len)
        new_y = np.linspace(0, 1, new_y_len)
        interp_obj = RegularGridInterpolator(
            (old_y, old_x), array, method="nearest", bounds_error=False, fill_value=0
        )
        new_y_grid, new_x_grid = np.meshgrid(new_y, new_x, indexing="ij")
        points = np.column_stack([new_y_grid.ravel(), new_x_grid.ravel()])
        new_array = interp_obj(points).reshape(new_y_len, new_x_len)
        return new_array


def interpolate_pixel_space(array, x_num, y_num):
    """
    Interpolates x/y coordinates and returns a list of (x, y) pairs
    corresponding to a flattened 2D image (Row-Major/C-style).
    """

    array = array.interp(
        x=np.linspace(array.coords["x"][0].item(), array.coords["x"][-1].item(), x_num)
    )

    array = array.interp(
        y=np.linspace(array.coords["y"][0].item(), array.coords["y"][-1].item(), y_num)
    )
    return array


def fill_defaults(  # noqa: F821  # noqa: F821
    recording_config: "Recording_Config", collapse_2d_config: "Collapse_2d_Config"
) -> tuple(
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, np.ndarray, np.ndarray
):
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
        - differing_channels_space: A boolean array indicating channels with differing pixel sizes.
        - differing_channels_time: A boolean array indicating channels with differing time resolutions.

    """
    nr_cells = np.max(recording_config.overview.spikes_df["cell_index"]) + 1
    nr_channels = recording_config.nr_channels
    pixel_sizes = [
        recording_config.channel_configs[channel].pixel_size
        for channel in recording_config.channel_names
    ]
    dts = [
        recording_config.channel_configs[channel].dt_ms
        for channel in recording_config.channel_names
    ]
    cell_indices = np.arange(nr_cells)
    if not all(np.array(pixel_sizes) == pixel_sizes[0]):
        # Interpolation needed
        # find which channels have different pixel sizes compared to the smallest one
        differing_channels_space = np.asarray(pixel_sizes) != min(pixel_sizes)
        print(
            f"Channels with differing pixel sizes: {np.asarray(recording_config.channel_names)[differing_channels_space]}"
        )
        common_pixel_size = min(pixel_sizes)
        print(f"Interpolating to common pixel size, pixel size: {common_pixel_size} um")
        # define common size by using the size from one of the channels with the smallest pixel size
        reference_channel = np.where(np.array(pixel_sizes) == common_pixel_size)[0][0]
        common_image_size = collapse_2d_config.cut_size_px[
            recording_config.channel_names[reference_channel]
        ]
    else:
        common_image_size = collapse_2d_config.cut_size_px[
            recording_config.channel_names[0]
        ]
        differing_channels_space = np.array([False] * nr_channels)
    # Pre-fill stores
    cm_most_important_store = xr.DataArray(
        np.zeros(
            (
                nr_cells,
                nr_channels,
                common_image_size.loc["y"].item(),
                common_image_size.loc["x"].item(),
            ),
            dtype=np.float32,
        ),
        dims=["cell_index", "channel", "y", "x"],
        coords={
            "cell_index": cell_indices,
            "channel": recording_config.channel_names,
            "x": np.arange(common_image_size.loc["x"].item()),
            "y": np.arange(common_image_size.loc["y"].item()),
        },
    )
    rms_store = xr.full_like(cm_most_important_store, 0)
    positions_placeholder = np.empty((nr_cells, nr_channels), dtype=object)

    if not all(np.array(dts) == dts[0]):
        # This can be done very simply by taking the min time, max time and calculating the number of bins
        common_dt = min(dts)
        print(
            f"Channels with differing time resolutions, interpolating to dt: {common_dt} ms"
        )
        differing_channels_time = np.asarray(dts) != common_dt
    else:
        common_dt = dts[0]
        differing_channels_time = np.array([False] * nr_channels)
    min_time = (
        -np.min(
            [
                total_sta_len - post_spike_bins
                for total_sta_len, post_spike_bins in zip(
                    [
                        recording_config.channel_configs[channel].total_sta_len
                        for channel in recording_config.channel_names
                    ],
                    [
                        recording_config.channel_configs[channel].post_spike_bins
                        for channel in recording_config.channel_names
                    ],
                )
            ]
        )
        * common_dt
    )
    max_time = (
        np.max(
            [
                post_spike_bins
                for post_spike_bins in [
                    recording_config.channel_configs[channel].post_spike_bins
                    for channel in recording_config.channel_names
                ]
            ]
        )
        * common_dt
    )
    common_time_frame = np.arange(
        min_time,
        max_time,
        common_dt,
    )
    sta_single_store = xr.DataArray(
        np.zeros((nr_cells, nr_channels, common_time_frame.shape[0]), dtype=np.float32),
        dims=["cell_index", "channel", "time"],
        coords={
            "cell_index": cell_indices,
            "channel": recording_config.channel_names,
            "time": common_time_frame,
        },
    )

    return (
        cm_most_important_store,
        sta_single_store,
        rms_store,
        positions_placeholder,
        differing_channels_space,
        differing_channels_time,
    )


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
            str(
                Path(channel_roots[channel].parts[-1])
                / f"cell_{cell_idx}"
                / "kernel.npy"
            )
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
        positions_placeholder,
        differing_channels_space,
        differing_channels_time,
    ) = fill_defaults(recording_config, collapse_2d_config)
    common_shape_x = cm_most_important_store.coords["x"].max().item() + 1
    common_shape_y = cm_most_important_store.coords["y"].max().item() + 1
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
            positions_placeholder[cell_pos, channel_index] = interpolate_pixel_space(
                var_coordinates,
                common_shape_x,
                common_shape_y,
            )
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
            stimulus_id = int(channel_data["sta_path"].item().split("_")[-1])
            nr_of_spikes = recording_config.overview.spikes_df.query(
                f"stimulus_index=={stimulus_id}&cell_index=={cell_idx}"
            )["nr_of_spikes"].values[0]
            sta_per_spike_raw, rms = calculate_rms(subset, nr_of_spikes)
            # Interpolate covariance map if needed, will return same data if no interpolation needed
            rms_store.loc[cell_idx, channel] = interpolate_2d_results(
                rms.values.astype(np.float32, copy=False),
                rms.sizes["y"],
                rms.sizes["x"],
                rms_store.loc[cell_idx, channel].sizes["y"],
                rms_store.loc[cell_idx, channel].sizes["x"],
            )
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
            # Interpolate covariance map if needed, will return same data if no interpolation needed
            cm_most_important_store.loc[cell_idx, channel] = interpolate_2d_results(
                cov_with_most,
                h,
                w,
                cm_most_important_store.loc[cell_idx, channel].sizes["y"],
                cm_most_important_store.loc[cell_idx, channel].sizes["x"],
            )
            # Store STA time course of the selected pixel
            pix_y, pix_x = divmod(most_idx, w)
            sta_single_store.loc[cell_idx, channel] = interpolate_sta_timecourse(
                (sta_per_spike_raw[:, pix_y, pix_x] + 0.5).values,
                sta_per_spike_raw[:, pix_y, pix_x].time.values[0],
                sta_per_spike_raw[:, pix_y, pix_x].time.values[-1],
                sta_single_store.loc[cell_idx, channel].time.values[0],
                sta_single_store.loc[cell_idx, channel].time.values[-1],
                abs(
                    sta_single_store.loc[cell_idx, channel].time.values[0]
                    - sta_single_store.loc[cell_idx, channel].time.values[1]
                ),
            )

    # Data reshaping for optimap storing.
    # If the data are from noise with different pixel sizes, we need to interpolate them to the same coordinate system.
    # For example, if we have one noise at size 4 um per pixel and another at 8 um per pixel,
    # we need to interpolate the 8 um data to 4 um per pixel. We can do this using xarray's interpolation functions.
    # First, test if interpolation is needed, if not we can skip this step.

    # Caculate the size in um of the complete sta image. All cutouts from all channels will fit in this space.
    reference_channel = np.asarray(recording_config.channel_names)[
        ~differing_channels_space
    ][0]
    common_pixel_size = recording_config.channel_configs[reference_channel].pixel_size
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

    nr_cells = ds.sizes["cell_index"]
    nr_channels = ds.sizes["channel"]

    cm_final_array = np.full(
        (nr_cells, nr_channels, max_y_size_px, max_x_size_px), np.nan, dtype=np.float32
    )

    rms_final = np.full_like(cm_final_array, np.nan, dtype=np.float32)
    # Prepare the final (N_cells, N_channels, Max_Time) array
    sta_final_array = np.full(
        (nr_cells, nr_channels, sta_single_store.shape[2]), np.nan, dtype=np.float32
    )

    # Store the cell coordinates for each cell and channel

    store_idx = 0
    for cell_idx in range(nr_cells):
        for channel_idx in range(nr_channels):
            coords = positions_placeholder[cell_idx][channel_idx]
            if np.all(coords == None):
                store_idx += 1
                continue
            y_indices = np.clip(
                np.searchsorted(y_coords_um, coords["y"].values), 0, max_y_size_px - 1
            )
            x_indices = np.clip(
                np.searchsorted(x_coords_um, coords["x"].values), 0, max_x_size_px - 1
            )

            # Ensure indices are unique and maintain the correct shape
            if len(np.unique(y_indices)) != len(y_indices) or len(
                np.unique(x_indices)
            ) != len(x_indices):
                # Use nearest neighbor assignment instead
                y_indices = np.arange(len(coords["y"].values)) + y_indices[0]
                x_indices = np.arange(len(coords["x"].values)) + x_indices[0]
                y_indices = np.clip(y_indices, 0, max_y_size_px - 1)
                x_indices = np.clip(x_indices, 0, max_x_size_px - 1)

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
        time_max=sta_single_store.coords["time"].values.astype(np.float32, copy=False),
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
    x_cut_size = int(
        np.max(
            [size.loc["x"].item() for size in collapse_2d_config.cut_size_px.values()]
        )
    )
    y_cut_size = int(
        np.max(
            [size.loc["y"].item() for size in collapse_2d_config.cut_size_px.values()]
        )
    )

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
        for cell_position, cell_index in enumerate(ds.cell_index.values):
            for channel in channel_names:
                if np.isnan(ds.sel(cell_index=cell_index, channel=channel).quality):
                    continue
                if (  # Skip low quality cells
                    ds.sel(cell_index=cell_index, channel=channel).quality
                    < collapse_2d_config.threshold
                ):
                    continue

                center = (
                    x_cut_size // 2,
                    y_cut_size // 2,
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
                try:
                    polar_cm = polar_transform(
                        cm_most_important_unpadded.reshape(
                            x_cut_size,
                            y_cut_size,
                        ),
                        center,
                        max_radius=max_radius,
                    )
                except ValueError:
                    print("Shape mismatch")
                entries = []
                for deg_step in range(0, 360, circular_reduction_config.degree_bins):
                    entries.append(
                        np.mean(
                            polar_cm[
                                deg_step : deg_step
                                + circular_reduction_config.degree_bins,
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
