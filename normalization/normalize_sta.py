import numpy as np
import xarray as xr


def zscore(array: np.ndarray) -> np.ndarray:
    """
    Normalize the array to have a mean of 0 and a standard deviation of 1.
    :param array: A numpy array.
    :return: The z-score normalized array.
    """
    return (array - np.mean(array)) / np.std(array)


def zscore_sta(sta, trailing_samples: int = 30) -> np.ndarray:
    """
    Special z-score normalization over the time axis.

    Works with 1D arrays (n_samples,) and 2D arrays (n_samples, n_features).
    The mean is computed over the full time axis.
    The standard deviation is computed using only the last 30 samples of the time axis
    (per feature for 2D inputs).
    """
    arr = np.asarray(sta, dtype=float)

    if arr.ndim == 1:
        mean = arr.mean()
        tail = arr[-min(trailing_samples, arr.shape[0]) :]
        std = tail.std()
        if std == 0 or not np.isfinite(std):
            std = 1.0
        return (arr - mean) / std

    if arr.ndim == 2:
        mean = arr.mean(axis=1, keepdims=True)
        tail_len = min(trailing_samples, arr.shape[0])
        tail = arr[:, -tail_len:]
        std = tail.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (arr - mean) / std

    raise ValueError("zscore_sta expects a 1D or 2D array")


def zscore_xr_sta(
    sta: xr.DataArray, sta_name: str = "sta_single_pixel", time_name: str = "time_max"
) -> xr.DataArray:
    """
    Performs a noise renormalization (z-score) on the STA DataArray along the time axis.
    The std and mean are computed only considering the "Pure Noise" region (time > 0). This is done based on the
    time_name dimension of an xarray DataArray.
    Parameters
    ----------
    sta : xr.DataArray
        The STA DataArray to be normalized.
    sta_name : str
        The name of the STA variable within the DataArray.
    time_name : str
        The name of the time dimension within the DataArray.
    """
    # 1. Isolate the "Pure Noise" region
    # Assuming positive_time_mask is your pre-defined background window
    noise_floor = sta.sel({time_name: sta[time_name] > 0})[sta_name]

    # 2. Calculate Robust Baseline Statistics
    # Using 'ddof=1' for a more unbiased estimate of the population std
    bg_mean = noise_floor.mean(dim=time_name)
    bg_std = noise_floor.std(dim=time_name, ddof=1)

    # 3. Apply the transformation to the ENTIRE dataset
    # This tells us how many 'noise-sigmas' the entire signal is
    return (sta[sta_name] - bg_mean) / bg_std
