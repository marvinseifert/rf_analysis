import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_ker(ker: np.ndarray, axes: tuple | None = None, sigma=5) -> np.ndarray:
    """
    Smooth the kernel using a Gaussian filter.
    Parameters
    ----------
    ker : np.ndarray
        The kernel to be smoothed.
    axes : tuple | None, optional
        The axes along which to apply the Gaussian filter. If None, all axes are used.
        Default is None.
    sigma : float, optional
        The standard deviation for Gaussian kernel. Default is 5.
    """

    return gaussian_filter(
        ker.astype(float), sigma, mode="constant", cval=np.median(ker), axes=axes
    )
