import numpy as np


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
        tail = arr[-min(trailing_samples, arr.shape[0]):]
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
