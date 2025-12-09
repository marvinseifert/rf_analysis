from typing import Tuple
from location.border import check_border_constrains
from location.channel_handling import masking_square
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional


def load_sta_as_xarray(
        sta_path: Path, dt_ms: float = 1.0, t_zero_index: Optional[int] = None
) -> xr.DataArray:
    """
    Load STA data from a .npy file and convert it to a labeled xarray.DataArray
    with dimensions (Time, X, Y).

    The function assumes the input NumPy array shape is (Time, Height, Width).

    Parameters
    ----------
    sta_path : Path
        Path to the `STA.npy` file.
    dt_ms : float, optional
        Time step (sampling interval) in milliseconds. Default is 1.0 ms.
    t_zero_index : Optional[int], optional
        The index corresponding to Time=0 (the spike trigger). If None,
        Time starts at 0.

    Returns
    -------
    xr.DataArray
        The 3D STA data with labeled dimensions (Time, X, Y).
    """
    try:
        sta_data_np = np.load(sta_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {sta_path}") from exc

    if sta_data_np.ndim != 3:
        raise ValueError(f"Expected 3 dimensions (Time, H, W), got {sta_data_np.ndim}.")

    # --- 1. Define Dimensions and Coordinates ---

    return _create_sta_dataarray(sta_data_np, dt_ms, t_zero_index)


def _create_sta_dataarray(
        sta_data_np: np.ndarray, dt_ms: float, t_zero_index: Optional[int]
) -> xr.DataArray:
    T, H, W = sta_data_np.shape
    t_zero = 0 if t_zero_index is None else t_zero_index

    time_indices = np.arange(T)
    time_coords = (time_indices - t_zero) * dt_ms

    x_coords = np.arange(W)
    y_coords = np.arange(H)

    sta_da = xr.DataArray(
        sta_data_np,
        coords={"time": time_coords, "y": y_coords, "x": x_coords},
        dims=["time", "y", "x"],
        name="spike_triggered_average",
    )

    if sta_da.dims != ("time", "x", "y"):
        sta_da = sta_da.transpose("time", "x", "y")

    return sta_da


def load_sta_subset(
        sta_path: Path,
        positions: Tuple[int, int],
        subset_size: Tuple[int, int],
        dt_ms: float = 10.0,
        t_zero_index: Optional[int] = 100,
) -> Tuple[xr.DataArray, float, float]:
    """
    Load a padded STA, extract a masked window, and return the subset along with updated center coordinates.

    Parameters
    ----------
    sta_path : Path
        Path to the `STA.npy` file for the requested channel.
    positions : Tuple[int, int]
        Initial cell position provided as (cX, cY).
    subset_size : Tuple[int, int]
        Border padding to apply in the (height, width) order.
    dt_ms : float, optional
        Time step (sampling interval) in milliseconds. Default is 10.0 ms.
    t_zero_index : Optional[int], optional
        The index corresponding to Time=0 (the spike trigger). Default is 100.

    Returns
    -------
    Tuple[np.ndarray, float, float]
        The extracted STA subset, updated cX, and updated cY.
    """
    try:
        sta_data = load_sta_as_xarray(sta_path, dt_ms, t_zero_index)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {sta_path}") from exc

    half_border_x = subset_size[1] // 2  # Width padding
    half_border_y = subset_size[0] // 2  # Height padding
    sta_data_padded = sta_data.pad(
        x=(half_border_x, half_border_x),
        y=(half_border_y, half_border_y),
        mode="median",
    ).astype(float)

    # 2. FIX: Create new, continuous coordinates and assign them to the padded array.
    # We assume 'x' and 'y' coordinates are simply 0-indexed positions.
    new_x_coords = np.arange(sta_data_padded.sizes["x"])
    new_y_coords = np.arange(sta_data_padded.sizes["y"])

    sta_data_padded = sta_data_padded.assign_coords(
        {"x": new_x_coords, "y": new_y_coords}
    )

    # Note: If your original 'x' and 'y' coordinates represented physical space (e.g., microns)
    # and were not just integer indices, you would need to calculate the new continuous
    # physical coordinates for the padded array here. Assuming for now they are indices.

    # 3. Calculate new center position (c_x is width, c_y is height)
    # positions[0] is cX (width/x), positions[1] is cY (height/y)
    # The new center is the old center shifted by the padding amount.
    c_x, c_y = positions[0] + half_border_x, positions[1] + half_border_y

    # 4. Check border constraints and update center
    c_x, c_y = check_border_constrains(
        c_x,
        c_y,
        (sta_data_padded.sizes["x"], sta_data_padded.sizes["y"]),
        (half_border_x, half_border_y),
    )

    # 5. Create the mask using the padded array's dimensions
    mask = masking_square(
        sta_data_padded.sizes["y"],
        sta_data_padded.sizes["x"],
        (int(c_x), int(c_y)),
        subset_size[1],  # mask width
        subset_size[0],  # mask height
    )

    # make mask an xarray with dimensions y, x
    mask_da = xr.DataArray(
        mask,
        coords={"y": sta_data_padded.coords["y"], "x": sta_data_padded.coords["x"]},
        dims=("y", "x"),
    )

    # 6. Apply mask and drop NaNs (this correctly subsets the data and coordinates)
    subset = sta_data_padded.where(mask_da, drop=True)

    return subset, c_x, c_y
