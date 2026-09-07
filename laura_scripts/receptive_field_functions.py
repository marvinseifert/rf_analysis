# Receptive field functions
# For processing rf data from the new pipeline (from the noise_analysis file)


# %% Marvin's function for removing NaNs around receptive field
def cutout_nans(data, inset=0):
    import numpy as np

    if not np.any(data > 0):
        print("Empty RF detected")
        return data

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


# %% Function to plot a multipanel figure of each top QI (>20) cell's receptive field, calculated using mean squared error method
def plot_mse_rf(dataset, channel, qi_limit):
    # Import dependencies
    import matplotlib.pyplot as plt
    import math
    import numpy as np

    # Load quality file
    quality = dataset["quality"]

    # Select cell indices for each channel individually which have qi > 20
    top_cells_dict = {
        ch: quality.sel(channel=ch).cell_index.values[
            quality.sel(channel=ch).values > qi_limit
        ]
        for ch in quality.channel.values
    }
    # Can now index the top cells from a single channel:
    top_cells = top_cells_dict[channel]

    # Find appropriate number of columns and rows in plot
    n_cols = math.ceil(math.sqrt(len(top_cells)))
    n_rows = math.ceil(len(top_cells) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=300)
    axes = axes.flatten()

    for idx, cell_id in enumerate(top_cells):
        # Import root-mean-square receptive field image
        rf_image = dataset["rms"].sel(channel=channel, cell_index=cell_id)
        # Crop rf image using cutout_nans function
        rf_image_crop = cutout_nans(rf_image)
        # Skip empty RFs
        if rf_image_crop.size == 0 or not np.any(rf_image_crop.values > 0):
            print(f"Skipping empty RF for cell {cell_id}")
            continue
        # Use xarray imshow to plot with correct coordinates
        rf_image_crop.plot.imshow(
            ax=axes[idx],
            cmap="Greys",
            add_colorbar=False,
            x=None,  # or set to None
            y=None,  # or set to None
        )

        # Set subplot title and labels
        axes[idx].set_title(f"Cell {int(cell_id)}")

        bottom_left_index = (n_rows - 1) * n_cols
        if not idx == bottom_left_index:
            # axes[idx].set_xticks([])
            # axes[idx].set_yticks([])
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel("")
            axes[idx].set_aspect("equal")
        else:
            axes[idx].set_xlabel("x position (µm)")
            axes[idx].set_ylabel("y position (µm)")
            axes[idx].set_aspect("equal")

    fig.suptitle(
        f"Receptive fields for cells with qi >{qi_limit} ({channel})",
        fontsize=20,
        y=0.99,
    )
    plt.tight_layout()
    plt.show()


# %% Function to use covariance matrix to plot centre surround receptive field structure
def plot_centre_surrounds(dataset, channel, qi_limit):
    # Import dependencies
    import matplotlib.pyplot as plt
    import math

    # Load quality file
    quality = dataset["quality"]
    # Select cell indices for each channel individually which have qi > 20
    top_cells_dict = {
        ch: quality.sel(channel=ch).cell_index.values[
            quality.sel(channel=ch).values > qi_limit,
        ]
        for ch in quality.channel.values
    }
    # Can now index the top cells from a single channel:
    top_cells = top_cells_dict[channel]

    # Find appropriate number of columns and rows in plot
    n_cols = math.ceil(math.sqrt(len(top_cells)))
    n_rows = math.ceil(len(top_cells) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    for idx, cell_id in enumerate(top_cells):
        # Import co-variance receptive field image
        rf_image = dataset["cm_most_important"].sel(channel=channel, cell_index=cell_id)
        # Crop rf image using cutout_nans function
        rf_image_crop = cutout_nans(rf_image)
        # Use xarray imshow to plot with correct coordinates
        rf_image_crop.plot.imshow(
            ax=axes[idx],
            cmap="coolwarm",
            add_colorbar=False,  # important when plotting many subplots
            x=None,  # or set to None
            y=None,  # or set to None
        )

        # Set subplot title and labels
        axes[idx].set_title(f"Cell {int(cell_id)}")

        bottom_left_index = (n_rows - 1) * n_cols
        if not idx == bottom_left_index:
            # axes[idx].set_xticks([])
            # axes[idx].set_yticks([])
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel("")
            axes[idx].set_aspect("equal")
        else:
            axes[idx].set_xlabel("x position (µm)")
            axes[idx].set_ylabel("y position (µm)")
            axes[idx].set_aspect("equal")

    fig.suptitle(
        f"Receptive fields for cells with qi >{qi_limit} ({channel})",
        fontsize=20,
        y=0.99,
    )
    plt.tight_layout()
    plt.show()


# %% Plot the STA for each of the top QI cells [can help work out what type the cell is e.g. OFF].
# Locates the pixel of spatial noise presented that the cell has peak responsiveness to, and shows what it was doing on average, 80 frames pre and 20 frames post a spike


def plot_sta(dataset, channel, qi_limit):
    # Import dependencies
    import matplotlib.pyplot as plt
    import math

    # Load quality file
    quality = dataset["quality"]

    # Select cell indices for each channel individually which have qi > 20
    top_cells_dict = {
        ch: quality.sel(channel=ch).cell_index.values[
            quality.sel(channel=ch).values > qi_limit
        ]
        for ch in quality.channel.values
    }
    # Can now index the top cells from a single channel:
    top_cells = top_cells_dict[channel]

    # Find appropriate number of columns and rows in plot
    n_cols = math.ceil(math.sqrt(len(top_cells)))
    n_rows = math.ceil(len(top_cells) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    for idx, cell_id in enumerate(top_cells):
        sta = dataset["sta_single_pixel"].sel(channel=channel, cell_index=cell_id)
        axes[idx].plot(sta)
        axes[idx].set_title(f"Cell ID = {int(cell_id)}")
        axes[idx].axvline(x=80, linestyle="--", color="r")
        axes[idx].set_xlabel("frames")

    fig.suptitle(
        f"STA for cells with qi >{qi_limit} ({channel})",
        fontsize=20,
        y=0.99,
    )
    plt.tight_layout()
    plt.show()


# %% Function to extract and plot statistics from dataset


def plot_rf_statistics(dataset, channel, qi_limit, bin_size):
    import matplotlib.pyplot as plt
    import numpy as np

    quality = dataset["quality"]

    # Select cell indices for each channel individually which have qi > 20
    top_cells_dict = {
        ch: quality.sel(channel=ch).cell_index.values[
            quality.sel(channel=ch).values > qi_limit
        ]
        for ch in quality.channel.values
    }
    # Can now index the top cells from a single channel:
    top_cells = top_cells_dict[channel]

    # Plot histogram of top cell "TIR" (measure of roundness, the higher the more elongated)
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    dataset["tir"].sel(channel=channel, cell_index=top_cells).plot.hist(
        bins=bin_size, ax=ax
    )
    ax.set_xlabel("TIR")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Top cells, {channel}")
    plt.show()

    # Plot histogram of top cell "tilt"
    fig, ax = plt.subplots()
    dataset["tilt"].sel(channel=channel, cell_index=top_cells).plot.hist(
        bins=bin_size, ax=ax
    )
    ax.set_xlabel("Tilt")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Top cells, {channel}")
    plt.show()

    # Plot histogram of top cell "tilt" with fixed bin width
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract tilt values for the selected channel and top cells
    tilt_vals = dataset["tilt"].sel(channel=channel, cell_index=top_cells)
    n_cells = len(top_cells)

    # Define consistent bin edges (e.g., bin width = 0.05 from 0 to 1)
    bin_width = 0.02
    bins = np.arange(0, 1 + bin_width, bin_width)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(
        tilt_vals,
        bins=bins,
        color="#cfe8f3",  # pale blue fill
        edgecolor="#1f4e79",  # dark blue border
        linewidth=1.2,
        alpha=0.85,
    )

    # Axis labels and title
    ax.set_xlabel("Tilt")
    ax.set_ylabel("Number of cells")
    ax.set_xlim(0, 1)
    ax.set_title(f"{channel} (total n cells = {n_cells})")

    # Clean up spines and add light grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.show()

    # Plot histogram of top cell "centre size (mm2)"
    fig, ax = plt.subplots()
    dataset["center_size_mm2"].sel(channel=channel, cell_index=top_cells).plot.hist(
        bins=bin_size, ax=ax
    )
    ax.set_xlabel("Centre size (mm2)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Top cells, {channel}")
    plt.show()

    # Plot histogram of top cell "surround size (mm2)"
    fig, ax = plt.subplots()
    dataset["surround_size_mm2"].sel(channel=channel, cell_index=top_cells).plot.hist(
        bins=bin_size, ax=ax
    )
    ax.set_xlabel("Surround size (mm2)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Top cells, {channel}")
    plt.show()

    # Plot histogram of top cell "angles"
    fig, ax = plt.subplots()
    dataset["angle"].sel(channel=channel, cell_index=top_cells).plot.hist(
        bins=bin_size, ax=ax
    )
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Top cells, {channel}")
    plt.show()

    # Plot histogram of top cell "angles" which ALSO have a tilt value of < 0.9
    fig, ax = plt.subplots()
    # Extract tilts for top_cells
    tilts = dataset["tilt"].sel(channel=channel, cell_index=top_cells)
    # Create mask: tilt < 0.9 and not NaN
    valid_mask = (tilts < 0.9) & np.isfinite(tilts)
    # Apply mask to top_cells
    filtered_cells = top_cells[valid_mask.values]
    # Select angle of filtered cells only
    dataset["angle"].sel(channel=dataset.channel, cell_index=filtered_cells).plot.hist(
        bins=bin_size, ax=ax
    )
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Top cells, {channel}, with tilt < 0.9")
    plt.show()

    # Plot histogram of top cell "angles" which ALSO are weighted by tilt
    # --- Step 1: Extract tilts and angles for top_cells ---
    tilts = dataset["tilt"].sel(channel=channel, cell_index=top_cells)
    angles = dataset["angle"].sel(channel=channel, cell_index=top_cells)
    # Convert to numpy arrays
    tilts = np.array(tilts)
    angles = np.array(angles)
    # --- Step 2: Mask out NaNs ---
    valid_mask = np.isfinite(tilts) & np.isfinite(angles)
    tilts = tilts[valid_mask]
    angles = angles[valid_mask]
    weights = 1 - tilts
    # --- Step 3: Plot histogram weighted by tilt ---
    fig, ax = plt.subplots()
    ax.hist(
        angles, bins=bin_size, weights=weights, color="C0", edgecolor="k", alpha=0.7
    )
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Weighted count (by tilt)")
    ax.set_title(f"Top cells, {channel}, weighted by tilt")
    plt.show()

    # Plot the same as above, but in a semi-circular plot
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Step 1: Extract tilts and angles for top_cells ---
    tilts = dataset["tilt"].sel(channel=channel, cell_index=top_cells)
    angles = dataset["angle"].sel(channel=channel, cell_index=top_cells)

    # Convert to numpy arrays
    tilts = np.array(tilts)
    angles = np.array(angles)

    # --- Step 2: Mask out NaNs ---
    valid_mask = np.isfinite(tilts) & np.isfinite(angles)
    tilts = tilts[valid_mask]
    angles = angles[valid_mask]

    # --- Step 3: Compute weights ---
    weights = 1 - tilts

    # --- Step 4: Define bins for histogram (same as linear histogram) ---
    bin_edges = np.linspace(0, 180, bin_size + 1)  # degrees
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram with weights
    hist_values, _ = np.histogram(angles, bins=bin_edges, weights=weights)

    # Convert bin centers to radians for polar plot
    bin_centers_rad = np.deg2rad(bin_centers)

    # --- Step 5: Plot as half-polar histogram ---
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    # ax.set_theta_zero_location("E")  # 0° on +x axis (right)
    # ax.set_theta_direction(1)  # counterclockwise (90° is +y)
    ax.set_theta_zero_location("S")  # 0° at bottom
    ax.set_theta_direction(1)
    ax.set_thetamax(180)  # limit to 0–180°
    ax.set_thetamin(0)

    # Plot bars
    ax.bar(
        bin_centers_rad,  # angular position
        hist_values,  # height of bars
        width=np.deg2rad(bin_edges[1] - bin_edges[0]),  # bar width in radians
        bottom=0,
        edgecolor="midnightblue",
        color="green",
        alpha=0.6,
    )
    # ax.set_title(f"RF angle (degrees) weighted by tilt, {channel}", va="bottom")
    plt.show()


# %% Function to plot a multipanel figure where each of the top QI (>20) cell's receptive fields are plotted.
# Each cell is then overlaid by the receptive field of the most strongly (spatially) overlapping cell.
# Define the number of top overlapping cells to plot by "number of overlaps".


# def plot_top_cells_overlay(dataset, channel, qi_limit, number_of_overlaps):

#
# # %%
# import numpy as np
#
#
# def cutout_nans(data, inset=0):
#     # Compute mask of valid (non-NaN) pixels
#     x_slice = slice(
#         np.where(np.any(data > 0, axis=0))[0][0] + inset,
#         np.where(np.any(data > 0, axis=0))[0][-1] + 1 - inset,
#         1,
#     )
#     y_slice = slice(
#         np.where(np.any(data > 0, axis=1))[0][0] + inset,
#         np.where(np.any(data > 0, axis=1))[0][-1] + 1 - inset,
#         1,
#     )
#     return data.isel(x=x_slice, y=y_slice)
#
#
# def overlap_score(rf_main, rf_other):
#     """
#     Pixel-wise overlap score between two receptive fields.
#     """
#     main_mask = rf_main > 0
#     other_mask = rf_other > 0
#     return np.nansum(main_mask & other_mask)
#
#
# from pathlib import Path
# import xarray as xr
# import matplotlib.pyplot as plt
# import math
#
# path_to_data = Path(
#     r"F:\Laura\zebrafish_08_01_2026\Phase_00\noise_analysis\noise_data.nc"
# )
# dataset = xr.load_dataset(path_to_data)
# channel = "4px_20Hz_40mins_shuffle_460_560"
# qi_limit = 20
# number_of_overlaps = 1
#
#
# # Load quality file
# quality = dataset["quality"]
#
# # Select cell indices for each channel individually which have qi > 20
# top_cells_dict = {
#     ch: quality.sel(channel=ch).cell_index.values[
#         quality.sel(channel=ch).values > qi_limit
#     ]
#     for ch in quality.channel.values
# }
# # Can now index the top cells from a single channel:
# top_cells = top_cells_dict[channel]
#
# # Find appropriate number of columns and rows in plot
# n_cols = math.ceil(math.sqrt(len(top_cells)))
# n_rows = math.ceil(len(top_cells) / n_cols)
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
# axes = axes.flatten()
#
# for idx, cell_id in enumerate(top_cells):
#     # Import root-mean-square receptive field image of our "main cell"
#     rf_main = dataset["rms"].sel(channel=channel, cell_index=cell_id)
#     # Crop rf image using cutout_nans function
#     rf_main_crop = cutout_nans(rf_main)
#
#     # Find the coordinates of the centre of the main cell's receptive field
#     x_centre = rf_main_crop.x[rf_main_crop.sizes["x"] // 2].values
#     y_centre = rf_main_crop.y[rf_main_crop.sizes["y"] // 2].values
#
#     # Initiate overlaps list - to put overlapping cells into
#     overlaps = []  # resets overlaps for each new main cell
#
#     # Find overlaps with other top cells
#     for other_cell in top_cells:
#         if (
#             other_cell == cell_id
#         ):  # if the "other cell" is actually the "main cell" then ignore and continue
#             continue
#
#         # load receptive field of each other cell
#         rf_other = dataset["rms"].sel(channel=channel, cell_index=other_cell)
#         # crop receptive field of other cell
#         other_rf_crop = cutout_nans(rf_other)
#         score = overlap_score(rf_main, rf_other)
#         if score > 0:
#             overlaps.append((other_cell, score))
#
#         # Sort by strongest overlap
#         overlaps.sort(key=lambda x: x[1], reverse=True)
#         top_overlaps = overlaps[:number_of_overlaps]
#
#         # --- Overlay overlapping RFs ---
#         for other_cell, score in top_overlaps:
#             rf_other = dataset["rms"].sel(channel=channel, cell_index=other_cell)
#             rf_other = cutout_nans(rf_other)
#
#             # Mask to overlapping pixels only
#             rf_overlap = rf_other.where((rf_other > 0) & (rf_main > 0))
#
#             rf_overlap.plot.imshow(
#                 ax=axes,
#                 cmap="Reds",
#                 alpha=0.3,
#                 add_colorbar=False,
#             )
#
#         axes.set_title(f"Cell {cell_id}", fontsize=10)
#         axes.set_xlabel("")
#         axes.set_ylabel("")
#
#         # # Turn off unused axes
#         # for ax in axes[len(top_cells) :]:
#         #     ax.axis("off")
#
# plt.tight_layout()
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_weighted_rf_angles(
    combined_dataset, channel_substr="12px", qi_limit=20, bin_size=18, title=None
):
    """
    Plot a half-polar histogram of receptive field angles weighted by elongation (1 - tilt).

    Parameters:
    -----------
    combined_dataset : xarray.Dataset
        Dataset containing 'angle', 'tilt', and 'quality' variables.
    channel_substr : str
        Substring to select channels (default "12px").
    qi_limit : float
        Minimum quality threshold to select cells (default 20).
    bin_size : int
        Number of bins for histogram (default 18 → 10° bins).
    title : str or None
        Custom title for the plot. If None, a default title is used.
    """

    # --- Step 1: Select channels containing substring ---
    channels = [ch for ch in combined_dataset.channel.values if channel_substr in ch]

    angles_list = []
    tilts_list = []

    # --- Step 2: Extract angles and tilts for high-quality cells ---
    for ch in channels:
        quality = combined_dataset["quality"].sel(channel=ch)
        high_quality_mask = quality.values > qi_limit

        angles = combined_dataset["angle"].sel(channel=ch).values.flatten()
        tilts = combined_dataset["tilt"].sel(channel=ch).values.flatten()

        # Apply quality mask
        angles = angles[high_quality_mask.flatten()]
        tilts = tilts[high_quality_mask.flatten()]

        # Mask NaNs
        valid_mask = np.isfinite(angles) & np.isfinite(tilts)
        angles_list.append(angles[valid_mask])
        tilts_list.append(tilts[valid_mask])

    # --- Step 3: Concatenate all channels ---
    all_angles = np.concatenate(angles_list)
    all_tilts = np.concatenate(tilts_list)

    # --- Step 4: Compute weights ---
    weights = 1 - all_tilts  # elongated cells contribute more

    # --- Step 5: Define histogram bins ---
    bin_edges = np.linspace(0, 180, bin_size + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_values, _ = np.histogram(all_angles, bins=bin_edges, weights=weights)
    bin_centers_rad = np.deg2rad(bin_centers)
    bar_width = np.deg2rad(bin_edges[1] - bin_edges[0])

    # --- Step 6: Plot half-polar histogram ---
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.set_theta_zero_location("N")  # 0° at top
    ax.set_theta_direction(-1)  # clockwise
    ax.set_thetamax(180)
    ax.set_thetamin(0)

    ax.bar(
        bin_centers_rad,
        hist_values,
        width=bar_width,
        bottom=0,
        edgecolor="red",
        color="peachpuff",
        alpha=0.7,
    )

    # Circular gridlines
    ax.set_yticks(np.arange(0, np.max(hist_values) + 1, 1))

    # Optional x-axis labels every 10°
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 10)))
    ax.set_xticklabels([f"{i}°" for i in range(0, 181, 10)])

    # Title
    if title is None:
        title = f"Channels containing '{channel_substr}'\nRF angle (degrees) weighted by elongation"
    ax.set_title(title, va="bottom")

    plt.show()
