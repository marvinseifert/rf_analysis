import numpy as np
from scipy.signal import hilbert


def complex_pca(signals, n_components=None):
    """
    Complex PCA via SVD on the analytic signal.
    Args:
      signals       : array, shape (n_signals, n_samples), real-valued
      n_components  : int or None, number of components to keep
    Returns:
      U     : array, shape (n_signals, k), spatial maps (left singular vectors)
      S     : array, shape (k,), singular values
      Vh    : array, shape (k, n_samples), right singular vectors
      PCs   : array, shape (k, n_samples), S * Vh (complex principal time series)
      var_explained : array, shape (k,), fractional variance explained per component
    """
    # 1. Analytic signal
    A = hilbert(signals, axis=1)
    # 2. Center
    A = A - A.mean(axis=1, keepdims=True)
    # 3. SVD
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        Vh = Vh[:n_components, :]
    # 4. PCs and variance explained
    PCs = np.diag(S) @ Vh
    var_explained = (S ** 2) / np.sum(S ** 2)
    return U, S, Vh, PCs, var_explained
