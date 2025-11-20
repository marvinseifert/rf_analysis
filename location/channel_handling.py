import numpy as np


def ident_surrounding_channels(
    channel: int, nr_neighbours: int = 1, **kwargs
) -> np.ndarray:
    """
    Identifies the surrounding channels of a channel on a non-square grid.

    Parameters
    ----------
    channel : int
        The channel index.
    nr_neighbours : int
        The number of surrounding channels that are identified. (Value indicates the number of "rings" around
        the central channel.)
    boxes : int
        The number of channels in one row/column of the grid, assuming a square grid.
        Legacy parameter, to be consistent with old version of this function.

    rows : int
        The number of channels in one row of the grid.
    columns : int
        The number of channels in one column of the grid.

    Returns
    -------
    channel_matrix : np.ndarray
        A matrix containing the indices of the channel and the surrounding channels on the grid as position indices.
        Channels outside the grid are np.NaN.
    """

    # Check if square grid shall be assumed.
    if kwargs["rows"] is None and kwargs["columns"] is None:
        rows = kwargs["boxes"]
        columns = kwargs["boxes"]
    else:
        rows = kwargs["rows"]
        columns = kwargs["columns"]

    # Create output matrix
    channel_x = 1 + 2 * nr_neighbours
    channel_matrix = np.empty((channel_x, channel_x))
    channel_matrix[:] = np.NaN

    # calculate the channel's row and column
    channel_row = channel // columns
    # channel_col = channel % columns

    for a, i in enumerate(range(-nr_neighbours, nr_neighbours + 1)):
        start_channel = channel + columns * i - nr_neighbours
        for j in range(channel_x):
            current_channel = start_channel + j
            if (
                0 <= current_channel < rows * columns
                and current_channel // columns == channel_row + i
            ):
                channel_matrix[j, a] = current_channel
    return channel_matrix


def masking(h: int, w: int, center: tuple = None, radius: int = None) -> np.ndarray:
    """
    Creates a circular mask around a given point with a given radius.

    Parameters
    ----------
    h : int
        The height of the input array.
    w : int
        The width of the input array.
    center : tuple
        The coordinates of the centre of the circle.
    radius : int
        The radius of the circle.
    Returns
    -------
    mask : np.ndarray
        A mask of the same size as the input array with all values outside the circular mask set to False.
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def masking_square(
    h: int, w: int, center: tuple = None, width: int = None, height: int = None
) -> np.ndarray:
    """
    Creates a rectangular mask around a given point with a given width and height.

    Parameters
    ----------
    h : int
        The height of the input array.
    w : int
        The width of the input array.
    center : tuple
        The coordinates of the center of the rectangle (x, y).
    width : int
        The width of the rectangle.
    height : int
        The height of the rectangle.
    Returns
    -------
    mask : np.ndarray
        A mask of the same size as the input array with all values outside the rectangular mask set to False.
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if width is None:  # use a default width, e.g., a quarter of the image width
        width = w // 4
    if height is None:  # use a default height, e.g., a quarter of the image height
        height = h // 4

    # Calculate the x and y boundaries of the rectangle
    left_edge = max(center[0] - width // 2, 0)
    right_edge = left_edge + width
    top_edge = max(center[1] - height // 2, 0)
    bottom_edge = top_edge + height

    # Create an array of the same size as the input array, initially all False
    mask = np.zeros((h, w), dtype=bool)

    # Set values within the rectangular bounds to True
    mask[top_edge:bottom_edge, left_edge:right_edge] = True

    return mask


def eliminate_singled(
    sta: np.ndarray, filter: bool = False, state: int = 1, **kwargs
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Eliminates single pixels in the STA by checking the surrounding pixels. The surrounding pixels are np.NaN.

    Parameters
    ----------
    sta : np.ndarray
        A map of pixels with value 1 and background with value = np.NaN.
    filter : bool
        If true, the surrounding pixels are filtered by a filter array.
    **kwargs : dict
        nr_filter_steps : int
            The number of subsequent filter steps that are applied.
        neighbours : int
            The number of surrounding pixels that are checked.
        filter_array : np.ndarray
            The boolean filter array that is applied to the surrounding pixels.
            [[False, True, False],[False, True, False], [False, True, False]]
            for example, would only check the pixels above and below the central pixel.
        filter_threshold : float
            The threshold that is applied to kick out pixels. In percent of surrounding pixels.
        max_channel_nr : int
            The maximum number of channels that are allowed to be accepted. If the number is exceeded, all pixels
            are considered to be noise.
    Returns
    -------
    position : np.ndarray
        The position of the pixels that survived the elimination.
    combined_filtered: np.ndarray
        The sta with the eliminated pixels set to np.NaN.
    """
    combined_filtered = sta.copy()
    channel_x = np.sqrt(sta.shape[0]).astype(int)

    for _ in range(kwargs["nr_filter_steps"]):
        # Iterate over the number of filter steps
        indices = np.where(~np.isnan(combined_filtered))[0]
        indices_penalty = np.empty_like(indices).astype(float)

        for index, i in zip(indices, range(indices.shape[0])):
            # Iterate over all pixels that are not np.NaN
            # Find the surrounding pixels
            s_channels = ident_surrounding_channels(index, **kwargs)

            if filter:
                # Filter the surrounding pixels
                s_channels = s_channels[kwargs["filter_array"]]
            # Remove surrounding pixels that are outside the image (=np.NaN)
            s_channels = s_channels[~np.isnan(s_channels)].astype(int)
            # Calculate the percentage of surrounding pixels that are np.NaN
            indices_penalty[i] = np.sum(np.isnan(combined_filtered[s_channels])).astype(
                float
            )
            # Normalize the percentage
            indices_penalty[i] = 1 - indices_penalty[i] / np.shape(s_channels)[0]
        # Check if the percentage is below the threshold
        indices_kick = indices[indices_penalty < kwargs["filter_threshold"]]
        combined_filtered[indices_kick] = np.NaN
    # Check if too many channels passed the filter
    if np.sum(~np.isnan(combined_filtered)) > kwargs["max_channel_nr"]:
        # print(np.sum(~np.isnan(combined_filtered)))
        print("All pixel rejected")
        combined_filtered[:] = np.NaN
        state = 0

    # Reshape the data to a 2D array
    reshaped_points = np.reshape(combined_filtered, (channel_x, channel_x))
    # Find the indices of the remaining pixels
    y_dots, x_dots = np.where(~np.isnan(reshaped_points))
    dots_combined = np.zeros((np.shape(y_dots)[0], 2))
    dots_combined[:, 1] = x_dots.astype(np.uint16)
    dots_combined[:, 0] = y_dots.astype(np.uint16)
    position = dots_combined.astype(int)

    return position, combined_filtered, state
