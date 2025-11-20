import numpy as np
from scipy.signal import decimate


def decimate_ker(ker_sm: np.ndarray, factor: int = 5) -> np.ndarray:
    """
    Decimate the kernel by a factor in x and y direction.
    :param ker_sm: The smoothed kernel.
    :param factor: The decimation factor.
    :return: The decimated kernel.
    """
    ker_ds = decimate(
        decimate(ker_sm, factor, ftype="fir", axis=(1)), factor, ftype="fir", axis=2
    )
    ker_ds = ker_ds[:, 2 * factor: -2 * factor, 2 * factor: -2 * factor]
    return ker_ds
