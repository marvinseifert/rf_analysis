import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_ker(ker: np.ndarray, axes: tuple | None = None) -> np.ndarray:
    """
    Smooth the kernel using a Gaussian filter.
    :param ker: The sta_kernel to be smoothed.
    :return: The smoothed kernel.
    """

    return gaussian_filter(ker.astype(float), 5, mode="constant", cval=np.median(ker), axes=axes)
