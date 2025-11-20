import numpy as np


def check_border_entries_false(array: np.ndarray) -> bool:
    """
    Check if the first and last row and column of the array are all False.

    :param array: A 2D numpy array.
    :return: True if the first and last row and column are all False, False otherwise.
    """
    # Check if the first row and the last row are all False
    first_last_row_check = np.all(array[0, :] == False) and np.all(
        array[-1, :] == False
    )

    # Check if the first column and the last column are all False,
    # excluding the first and last element since they were checked above.
    first_last_column_check = np.all(array[1:-1, 0] == False) and np.all(
        array[1:-1, -1] == False
    )

    return first_last_row_check and first_last_column_check
