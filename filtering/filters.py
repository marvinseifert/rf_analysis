import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth band\-pass filter to the input data.

    Parameters
    ----------
    data : np.ndarray
        Input signal array. Filtering is applied along `axis\=0`.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order of the Butterworth filter (default: 5\).

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as `data`.
    """
    # Create bandpass filter coefficients in second\-order sections
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    # Apply the filter to the data
    filtered_data = sosfiltfilt(sos, data, axis=0)
    return filtered_data
