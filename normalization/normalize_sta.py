import numpy as np


def zscore(array: np.ndarray) -> np.ndarray:
    """
    Normalize the array to have a mean of 0 and a standard deviation of 1.
    :param array: A numpy array.
    :return: The z-score normalized array.
    """
    return (array - np.mean(array)) / np.std(array)


def zscore_sta(sta):
    """Special zscore over the time axis (axis=1). The standard deviation is calculated only considering the last
    30 entries of the time axis."""
    sta = np.array(sta)
    sta = sta - np.mean(sta)
    std = np.std(sta[-20:])
    sta = sta / std
    return sta
