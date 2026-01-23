import xarray as xr


def zoom_extent(
    xarray: xr.DataArray, zoom_factor: float, x_coord: str = "x", y_coord: str = "y"
) -> list[float]:
    """Return extent [xmin, xmax, ymin, ymax] zoomed around the center.
    The output of this function can be used directly in matplotlib imshow extent parameter.
    """
    x_coords = xarray.coords[x_coord]
    y_coords = xarray.coords[y_coord]
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2
    x_half_range = (x_coords.max() - x_coords.min()) / 2 / zoom_factor
    y_half_range = (y_coords.max() - y_coords.min()) / 2 / zoom_factor
    return [
        x_center - x_half_range,
        x_center + x_half_range,
        y_center - y_half_range,
        y_center + y_half_range,
    ]
