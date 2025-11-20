import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar


def plot_2d(
        sta: np.ndarray, pixel_size: int = None, cmap: str = "Greys", title: str = None, scale_size: int = 50,
        plot_scalebar=True
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D STA.

    :param sta: The STA.
    :param pixel_size: The size of a single pixel in micrometers
    :param cmap: The colormap to use.
    :param title: The title of the plot.
    :return: matplotlib.pyplot.Figure and matplotlib.pyplot.Axes
    """

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    im = ax.imshow(sta, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    if plot_scalebar:
        scalebar = ScaleBar(pixel_size, "um", fixed_value=scale_size)
    if title:
        ax.set_title(title)
    ax.add_artist(scalebar)
    return fig, ax


def add_scalebar(ax: plt.Axes, pixel_size: int = 1) -> None:
    """
    Add a scalebar to the plot based on the size of a single pixel.
    :param ax: The axis to add the scalebar to.
    :param pixel_size: The size of a single pixel in micrometers.
    :return: None
    """
    scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
    ax.add_artist(scalebar)
