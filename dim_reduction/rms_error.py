import xarray as xr


def calculate_rms(
    subset: xr.DataArray, nr_of_spikes: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate the RMS and mean of a given xarray DataArray subset.

    Parameters
    ----------
    subset : xr.DataArray
        The input DataArray subset for which to calculate RMS and mean.
    nr_of_spikes : int
        The number of spikes used for normalization.

    Returns
    -------
    tuple(xr.DataArray, xr.DataArray)
        A tuple containing the RMS and mean DataArrays.
    """
    sta_per_spike_raw: xr.DataArray = subset / nr_of_spikes - 0.5
    rms = (sta_per_spike_raw**2).max(dim="time")

    return sta_per_spike_raw, rms
