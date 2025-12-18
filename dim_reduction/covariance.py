import numpy as np


def calculate_covariance(sta_per_spike_flat: np.ndarray, time_bins) -> np.ndarray:
    """
    Calculate the covariance matrix of the flattened STA per spike data.

    Parameters
    ----------
    sta_per_spike_flat : np.ndarray
        Flattened STA per spike data with shape (time_bins, pixels).

    time_bins : int
        Number of time bins in the STA data.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape (pixels, pixels).
    """
    try:
        # Use flat_centered here, NOT flat (raw)
        _, s, Vt = np.linalg.svd(sta_per_spike_flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    loadings = Vt[0]  # shape: (h*w,)
    most_idx = int(np.argmax(np.abs(loadings)))
    cov_with_most = (sta_per_spike_flat.T @ sta_per_spike_flat[:, most_idx]) / (
        time_bins - 1
    )
    return cov_with_most, most_idx
