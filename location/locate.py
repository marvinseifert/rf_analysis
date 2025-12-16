from typing import Any

import numpy as np
import cv2
import networkx as nx
from xarray import Dataset, DataTree

from graph.build_graph import array_to_graph
import xarray as xr
from pathlib import Path
import math
from typing import List


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
        position_px: xr.DataArray,
        pixel_size: float,
        noise_size_2d: xr.DataArray,
        x_key="x",
        y_key="y",
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
        x=(position_px[x_key] - noise_size_2d.loc["x"] / 2) * pixel_size,
        y=(position_px[y_key] - noise_size_2d.loc["y"] / 2) * pixel_size,
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
        position_um: xr.DataArray,
        pixel_size: float,
        noise_size_2d: xr.DataArray,
        x_key="x",
        y_key="y",
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
    x_key : str
        Key for the x dimension.
    y_key : str
        Key for the y dimension.
    Returns
    -------
    xr.DataArray
        An array of type int with positions in pixels.
    """
    position_px = position_um.assign_coords(
        x=int(
            (position_um[x_key] / pixel_size) + (noise_size_2d.loc[x_key] / 2).round()
        ),
        y=int(
            (position_um[y_key] / pixel_size) + (noise_size_2d.loc[y_key] / 2).round()
        ),
    )
    return position_px


def locate_across_channels(
        channel_paths: list[Path],
        threshold: float = 20,
        channel_names: list[str] = None,
        channel_configs: dict = None,
) -> tuple[xr.DataArray, xr.DataArray]:
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
    channel_configs : dict, optional
        Dictionary of channel configurations.
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
    first_channel_data = xr.load_dataarray(channel_paths[0] / "quality.nc")
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
        quality_array = xr.load_dataarray(channel_path / "quality.nc")
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
    weights_masked = xr.where(weights_raw < threshold, 1e-6, weights_raw).values
    weights_masked = np.tile(weights_masked[:, :, np.newaxis], (1, 1, 2))

    # Expand weights to match the 2 (x,y) dimensions for np.average
    # Shape becomes: (nr_cells, 2, nr_channels)
    position_metrics = ["center_x", "center_y"]
    position_data = final_quality_array.sel(metrics=position_metrics)

    # Convert pixel positions to micrometer positions for each channel
    for channel in channel_names:
        position_data.loc[dict(metrics="center_x", channel=channel)] = (
                                                                               position_data.sel(metrics="center_x",
                                                                                                 channel=channel)
                                                                               -
                                                                               channel_configs[channel].image_shape.loc[
                                                                                   "x"].item() / 2
                                                                       ) * channel_configs[channel].pixel_size
        position_data.loc[dict(metrics="center_y", channel=channel)] = (
                                                                               position_data.sel(metrics="center_y",
                                                                                                 channel=channel)
                                                                               -
                                                                               channel_configs[channel].image_shape.loc[
                                                                                   "y"].item() / 2
                                                                       ) * channel_configs[channel].pixel_size

    #

    # Weighted average
    average_position = np.average(position_data, axis=0, weights=weights_masked)
    # We need to expand average_positions to all channels because they might have different image sizes and pixel sizes
    average_position = np.tile(
        average_position[np.newaxis, :, :], (len(channel_names), 1, 1)
    )
    # Prepare the final mean positions DataArray
    average_position = xr.DataArray(
        data=average_position,
        dims=["channel", "cell_index", "metrics"],
        coords={
            "channel": channel_names,
            "cell_index": final_quality_array.cell_index,
            "metrics": position_metrics,
        },
    )
    for channel in channel_names:
        average_position.loc[dict(metrics="center_x", channel=channel)] = (
            (
                    average_position.sel(metrics="center_x", channel=channel)
                    / channel_configs[channel].pixel_size
                    + channel_configs[channel].image_shape.loc["x"].item() / 2
            )
            .round()
            .fillna(0)
            .clip(0, channel_configs[channel].image_shape.loc["x"].item() - 1)
        )
        average_position.loc[dict(metrics="center_y", channel=channel)] = (
            (
                    average_position.sel(metrics="center_y", channel=channel)
                    / channel_configs[channel].pixel_size
                    + channel_configs[channel].image_shape.loc["y"].item() / 2
            )
            .round()
            .fillna(0)
            .clip(0, channel_configs[channel].image_shape.loc["y"].item() - 1)
        )
    average_position = average_position.astype(int)
    return final_quality_array, average_position


def project_radii_to_xy(radii: List[float]) -> np.ndarray:
    n = len(radii)
    delta_theta = (2 * math.pi) / n
    x_coords = []
    y_coords = []
    for i in range(n):
        theta = i * delta_theta
        r = radii[i]
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        x_coords.append(x)
        y_coords.append(y)
    return np.column_stack((x_coords, y_coords))
