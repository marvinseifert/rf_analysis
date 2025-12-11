from typing import Any

import numpy as np
import cv2
import networkx as nx
from xarray import Dataset, DataTree

from graph.build_graph import array_to_graph
import xarray as xr
from pathlib import Path


def identify_center(sta_std: np.ndarray) -> tuple[int, int]:
    """
    Identify the center of the STA by finding the largest contour in the image.

    :param sta_std: The STA_STD image.
    :return: The x and y coordinates of the center of the STA.
    """
    # Convert the image to 8 bit 3 channel image
    sta_std = sta_std / np.max(sta_std) * 255
    sta_std = sta_std.astype(np.uint8)
    # Extend the image to the 3 channel format
    sta_std = cv2.merge([sta_std, sta_std, sta_std])
    # Convert the image to grayscale
    sta_std = cv2.cvtColor(sta_std, cv2.COLOR_BGR2GRAY)
    # Apply the thresholding
    _, sta_std = cv2.threshold(sta_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find the contours
    contours, _ = cv2.findContours(sta_std, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    c = max(contours, key=cv2.contourArea)
    # Find the centroid of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def isolate_central_island(matrix: np.ndarray) -> np.ndarray:
    """
    Isolate the central island in a binary matrix by marking all non-central islands as False.
    :param matrix: A binary matrix.
    :return: The matrix with only the central island marked as True.
    """
    G = array_to_graph(matrix)
    largest_component = max(nx.connected_components(G), key=len)

    # Mark cells not in the largest component as False
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i, j) not in largest_component:
                matrix[i][j] = False
            else:
                matrix[i][
                    j
                ] = True  # Ensure all nodes in the largest component are True
    return matrix


def px_position_to_um(
        position_px: xr.DataArray, pixel_size: float, noise_size_2d: xr.DataArray
) -> xr.DataArray:
    """
    Convert pixel positions to micrometer positions. Position (0,0) is at the center of the noise stimulus, all
    positions are relative to that, so can be negative.

    Parameters
    ----------
    position_px : xr.DataArray
        Position in pixels.
    pixel_size : float
        Size of a pixel in micrometers.
    noise_size_2d : xr.DataArray
        Size of the noise stimulus in pixels (2D).
    Returns
    -------
    xr.DataArray
        An array of type float with positions in micrometers.

    """

    position_um = position_px.assign_coords(
        x=(position_px["x"] - noise_size_2d.loc["x"] / 2) * pixel_size,
        y=(position_px["y"] - noise_size_2d.loc["y"] / 2) * pixel_size,
    )
    return position_um


def px_int_to_um(
        c_x: int, c_y: int, pixel_size: float, noise_size_2d: xr.DataArray
) -> tuple[float, float]:
    """
    Convert integer pixel positions to micrometer positions. Position (0,0) is at the center of the noise stimulus, all
    positions are relative to that, so can be negative.

    Parameters
    ----------
    c_x : int
        X position in pixels.
    c_y : int
        Y position in pixels.
    pixel_size : float
        Size of a pixel in micrometers.
    noise_size_2d : xr.DataArray
        Size of the noise stimulus in pixels (2D).
    Returns
    -------
    tuple[float, float]
        A tuple with positions in micrometers (x_um, y_um).

    """

    x_um = (c_x - noise_size_2d.loc["x"] / 2) * pixel_size
    y_um = (c_y - noise_size_2d.loc["y"] / 2) * pixel_size
    return x_um.values.item(), y_um.values.item()


def um_position_to_px(
        position_um: xr.DataArray, pixel_size: float, noise_size_2d: xr.DataArray
) -> xr.DataArray:
    """
    Convert micrometer positions back to pixel positions relative to the noise center.
    Parameters
    ----------
    position_um : xr.DataArray
        Position in micrometers.
    pixel_size : float
        Size of a pixel in micrometers.
    noise_size_2d : xr.DataArray
        Size of the noise stimulus in pixels (2D).
    Returns
    -------
    xr.DataArray
        An array of type int with positions in pixels.
    """
    position_px = position_um.assign_coords(
        x=int((position_um["x"] / pixel_size) + (noise_size_2d.loc["x"] / 2).round()),
        y=int((position_um["y"] / pixel_size) + (noise_size_2d.loc["y"] / 2).round()),
    )
    return position_px


def locate_across_channels(
        channel_paths: list[Path], threshold: float = 20, channel_names: list[str] = None
) -> tuple[Dataset, DataTree | Any]:
    """
    Loads the quality.npy files from all channels and computes the mean positions across cells.
    Parameters
    ----------
    channel_paths : list[Path]
        List of paths to the channels.
    threshold : float
        Quality threshold below which weights are set to a small value.
    channel_names : list[str], optional
        List of channel names corresponding to the channel paths.
    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing:
        - final_quality_array: An xarray DataArray with quality metrics from all channels.
        - mean_positions: An xarray DataArray with the computed mean positions across channels.
    """
    # Assuming channel_paths is a list of Path objects

    # 1. INITIAL SETUP: Load the first channel and prepare it
    # Load the first channel's data
    first_channel_data = xr.load_dataset(channel_paths[0] / "quality.nc")
    first_channel_data = first_channel_data.rename_vars(
        {"__xarray_dataarray_variable__": "quality_metrics"}
    )
    # Ensure the 'channel' coordinate exists on the first data item
    if channel_names:
        first_channel_data = first_channel_data.assign_coords(
            channel=channel_names[
                0
            ]  # Assign the first channel's name as a scalar coord
        )

    # A list to hold all the individual xarray Datasets for concatenation
    all_channel_datasets = [first_channel_data]

    # 2. LOAD AND COLLECT ALL REMAINING CHANNEL DATA
    for i, channel_path in enumerate(channel_paths[1:], start=1):
        quality_array = xr.load_dataset(channel_path / "quality.nc")
        quality_array = quality_array.rename_vars(
            {"__xarray_dataarray_variable__": "quality_metrics"}
        )
        # Assign the correct channel name coordinate if available
        if channel_names:
            quality_array = quality_array.assign_coords(channel=channel_names[i])

        all_channel_datasets.append(quality_array)

    # 3. CONCATENATE ALL DATASETS
    # Concatenate the list of Datasets along the 'channel' dimension
    final_quality_array = xr.concat(
        all_channel_datasets,
        dim="channel",
        data_vars="all",  # Ensure all variables are concatenated
    )

    # 3. CORRECT THRESHOLDING AND WEIGHT PREPARATION

    # Extract the raw quality values (weights) for all channels
    weights_raw = final_quality_array.sel(
        metrics="quality"
    )  # Shape: (nr_cells, nr_channels)

    # Apply the threshold: set all weights below the threshold to 0.0
    # This is the correct way to exclude "bad" measurements.
    weights_masked = xr.where(weights_raw < threshold, 1e-6, weights_raw)

    # Expand weights to match the 2 (x,y) dimensions for np.average
    # Shape becomes: (nr_cells, 2, nr_channels)
    position_metrics = ["center_x", "center_y"]
    position_data = final_quality_array.sel(metrics=position_metrics)

    # 3a. Weighted Sum:
    # The weight array (cell, channel) automatically broadcasts across the metrics dimension (x, y)
    # Weighted_sum has dimensions (cell, metrics)
    weighted_sum = (position_data * weights_masked).sum(dim="channel")

    # 3b. Total Weight:
    # Total_weights has dimensions (cell)
    total_weights = weights_masked.sum(dim="channel")

    # 3c. Calculate Mean:
    # We divide the weighted sum by the total weights.
    # total_weights (cell) automatically broadcasts across the metrics dimension (x, y)
    # during the division.
    mean_positions = weighted_sum / total_weights
    # Optional: Handle the case where a cell has 0 total weight (returns NaN)
    # --- 4. HANDLE ZERO TOTAL WEIGHT ---

    # Identify cells where the total weight is zero
    no_weight_mask = total_weights == 0

    # Get the fallback position from the first channel (channel index 0)
    # Fallback_position has dimensions (cell, metrics)
    fallback_position = position_data.isel(channel=0).drop_vars("channel")

    # Use xr.where() to replace NaN (or arbitrary result) positions with the fallback position
    final_mean_positions = xr.where(no_weight_mask, fallback_position, mean_positions)

    return final_quality_array, final_mean_positions
