import numpy as np
from scipy.stats import zscore


def cov_filtering(
        ker: np.ndarray, shape: tuple = (100, 100)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the covariance matrix of the kernel and filter the pixels with the highest number of connections.
    :param ker: The sta_kernel to be filtered as flattened array.
    :param shape: The shape of the kernel in 2D.
    :return: The location of the important pixels and the covariance matrix.
    """
    ker_z = zscore(ker)

    ker_shape = ker_z.shape
    ker_z = ker_z.reshape(ker_shape[0], -1)
    cm = np.cov(ker_z.T)

    # Postprocessing
    diag = np.where(np.diag(np.diag(cm)))[0]
    cm[diag, diag] = 0
    # cm_max = np.max(cm, axis=0)
    cm_argmax = np.argmax(np.triu(cm), axis=0)
    pix, px_nr = np.unique(cm_argmax, return_counts=True)

    important_pixels = np.zeros(shape[0] * shape[1])
    important_pixels[pix] = px_nr
    important_pixels = important_pixels.reshape(shape)
    return important_pixels, cm


def cov_filtering_sum(
        ker: np.ndarray, shape: tuple = (100, 100)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the covariance matrix of the kernel and filter the pixels with the highest sum of connections.
    :param ker: The sta_kernel to be filtered as flattened array.
    :param shape: The shape of the kernel in 2D.
    :return: The location of the important pixels and the covariance matrix.
    """

    # ker_z = zscore(ker)
    ker_z = ker.copy()

    ker_shape = ker_z.shape
    ker_z = ker_z.reshape(ker_shape[0], -1)
    cm = np.cov(ker_z.T)

    # Postprocessing
    diag = np.where(np.diag(np.diag(cm)))[0]
    cm[diag, diag] = 0
    cm_sum = np.sum(cm, axis=0)
    # Get pixels with higest sum of connections
    important_sums = np.where(cm_sum > 0)[0]
    important_mins = np.where(cm_sum < 0)[0]

    important_pixels = np.zeros(shape[0] * shape[1])
    important_pixels[important_sums] = cm_sum[important_sums]
    important_pixels[important_mins] = cm_sum[important_mins]
    important_pixels = important_pixels.reshape(shape)
    return important_pixels, cm
