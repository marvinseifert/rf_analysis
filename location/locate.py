import numpy as np
import cv2
import networkx as nx
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
        x=(position_px["x"] - noise_size_2d["x"] / 2) * pixel_size,
        y=(position_px["y"] - noise_size_2d["y"] / 2) * pixel_size,
    )
    return position_um


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
        x=int((position_um["x"] / pixel_size) + (noise_size_2d["x"] / 2).round()),
        y=int((position_um["y"] / pixel_size) + (noise_size_2d["y"] / 2).round()),
    )
    return position_px


def locate_across_channels(
        channel_paths: list[Path], threshold: float = 20, channel_names: list[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the quality.npy files from all channels and computes the mean positions across cells.
    ... [Documentation omitted for brevity]
    """
    first_quality_array = xr.load_dataset(channel_paths[0] / "quality.nc")
    # 1. INITIAL SETUP AND VALIDATION
    first_quality_array = first_quality_array.expand_dims(dim={"channel": 2})
    if channel_names:
        first_quality_array = first_quality_array.assign_coords(
            channel=("channel", channel_names)
        )

    # 2. LOAD ALL CHANNEL DATA (Correctly loading all channels before processing)
    for i, channel_path in enumerate(channel_paths[1:], start=1):
        quality_array = xr.load_dataset(channel_paths[0] / "quality.nc")
        first_quality_array[:, :, i] = quality_array

    # 3. CORRECT THRESHOLDING AND WEIGHT PREPARATION

    # Extract the raw quality values (weights) for all channels
    weights_raw = quality_store[:, 0, :]  # Shape: (nr_cells, nr_channels)

    # Apply the threshold: set all weights below the threshold to 0.0
    # This is the correct way to exclude "bad" measurements.
    weights_masked = np.where(weights_raw < threshold, 1e-6, weights_raw)

    # Expand weights to match the 2 (x,y) dimensions for np.average
    # Shape becomes: (nr_cells, 2, nr_channels)
    weights_expanded = np.tile(weights_masked[:, np.newaxis, :], (1, 2, 1))

    # 4. CALCULATE WEIGHTED MEAN POSITION
    mean_positions = np.average(
        quality_store[:, 2:4, :],  # Position data (x, y)
        axis=2,  # Average across the channel axis
        weights=weights_expanded,
    )

    # Optional: Handle the case where a cell has 0 total weight (returns NaN)
    total_weights = np.sum(weights_masked, axis=1)
    # Get the indices where total weight is zero
    no_weight_mask = total_weights == 0
    # Set the mean position for these cells to the position from the first channel (arbitrary default)
    mean_positions[no_weight_mask] = quality_store[no_weight_mask, 2:4, 0]

    return quality_store, mean_positions
