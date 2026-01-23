import numpy as np


def cutout_nans(data, inset=0):
    # Compute mask of valid (non-NaN) pixels
    x_slice = slice(
        np.where(np.any(data > 0, axis=0))[0][0] + inset,
        np.where(np.any(data > 0, axis=0))[0][-1] + 1 - inset,
        1,
    )
    y_slice = slice(
        np.where(np.any(data > 0, axis=1))[0][0] + inset,
        np.where(np.any(data > 0, axis=1))[0][-1] + 1 - inset,
        1,
    )
    return data.isel(x=x_slice, y=y_slice)


def get_non_nan_slices(data, inset=0):
    x_slice = slice(
        np.where(np.any(data > 0, axis=0))[0][0] + inset,
        np.where(np.any(data > 0, axis=0))[0][-1] + 1 - inset,
        1,
    )
    y_slice = slice(
        np.where(np.any(data > 0, axis=1))[0][0] + inset,
        np.where(np.any(data > 0, axis=1))[0][-1] + 1 - inset,
        1,
    )
    return y_slice, x_slice
