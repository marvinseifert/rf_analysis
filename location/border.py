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


def check_border_constrains(cX, cY, extended_size, max_half_cut_px):
    # Ensure inputs are treated as integers for boundary calculation
    # Casts are necessary if extended_size and max_half_cut_px elements are strings

    # Calculate boundaries
    max_x = int(extended_size[0]) - int(max_half_cut_px[0])
    max_y = int(extended_size[1]) - int(max_half_cut_px[1])
    min_x = int(max_half_cut_px[0])
    min_y = int(max_half_cut_px[1])

    # Apply clipping using min/max: max(min_value, coordinate) ensures it's not too small,
    # then min(max_value, result) ensures it's not too large.
    cX_out = min(max_x, max(min_x, int(cX)))
    cY_out = min(max_y, max(min_y, int(cY)))

    return cX_out, cY_out
