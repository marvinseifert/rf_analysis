import xarray as xr
from typing import Union


def x_y_and_scale(
    x: Union[int, float], y: Union[int, float], name: str = "size_values"
) -> xr.DataArray:
    """
    Creates a 1-dimensional xarray DataArray containing 'x' and 'y' values
    as the data payload, labeled by a 'dimension' coordinate.

    This structure is optimized for easy array-wise arithmetic (e.g., division
    by a pixel size), replacing the original 0D DataArray structure which
    required cumbersome coordinate re-assignment.

    Parameters
    ----------
    x : int or float
        The X value (e.g., cut size in um).
    y : int or float
        The Y value (e.g., cut size in um).
    name : str, optional
        The name of the DataArray.

    Returns
    -------
    xr.DataArray
        A 1D DataArray with two elements, where coords['dimension']=['x', 'y'],
        and the data is [x, y].
    """
    # 1. Define the data values
    data_values = [x, y]

    # 2. Define the coordinates for the new dimension
    coords_dict = {"dimension": ["x", "y"]}
    dims_list = ["dimension"]

    # 3. Create and return the 1D DataArray
    da = xr.DataArray(data=data_values, coords=coords_dict, dims=dims_list, name=name)

    # Note: The ratio calculation (x/y) is removed as it was not relevant
    # for the dimension conversion calculation you were trying to perform.
    return da


def create_empty_dataarray() -> xr.DataArray:
    """Returns a minimal, empty xr.DataArray instance."""
    # Define it with appropriate dimensions/metadata your application expects.
    # Here, we create an array with no data but defined dimensions (x, y)
    return xr.DataArray(
        data=None,  # Or np.empty((0, 0)) if you need NumPy data
        coords={"x": [], "y": []},
        dims=("x", "y"),
    )
