from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from organize.configs import Recording_Config
from aquarel import load_theme
import matplotlib


def cutout_nans(data, inset=0):
    import numpy as np

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


# %%
# Load the data
# path_to_data = Path(
#     r"F:\Laura\zebrafish_05_11_2025\Phase_01\noise_analysis\noise_data.nc"
# )
path_to_data = Path(
    r"F:\Laura\zebrafish_07_08_2026\Phase_00\noise_analysis\noise_data.nc"
)
dataset_single = xr.load_dataset(path_to_data)

# %%
channel = "32px_15Hz_20mins_shuffle_x4"
qi_limit = 20

# Load quality file
quality = dataset_single["quality"]

# Select cell indices for each channel individually which have qi > 20
top_cells_dict = {
    ch: quality.sel(channel=ch).cell_index.values[
        quality.sel(channel=ch).values > qi_limit
    ]
    for ch in quality.channel.values
}
# Can now index the top cells from a single channel:
top_cells = top_cells_dict[channel]

# %% Single cells rms/cm rfs, grid overlaid

cell_id = top_cells[36]
fig, axes = plt.subplots(figsize=(10, 10))
# fig.patch.set_alpha(0)
# axes.patch.set_alpha(0)
# Import root-mean-square receptive field image
rf_image = dataset_single["cm_most_important"].sel(channel=channel, cell_index=cell_id)
# Crop rf image using cutout_nans function
rf_image_crop = cutout_nans(rf_image)

# --- Crop RF image to 400 µm x 400 µm centered on current center ---

crop_size = 800  # microns. If want full, just say 800
half_crop = crop_size / 2

# Get coordinate arrays
x_coords = rf_image_crop.coords[rf_image_crop.dims[-1]].values
y_coords = rf_image_crop.coords[rf_image_crop.dims[-2]].values

# Compute center in µm
x_center = (x_coords.min() + x_coords.max()) / 2
y_center = (y_coords.min() + y_coords.max()) / 2

# Define crop bounds
x_min_crop = x_center - half_crop
x_max_crop = x_center + half_crop
y_min_crop = y_center - half_crop
y_max_crop = y_center + half_crop

# Crop
rf_image_crop = rf_image_crop.sel(
    {
        rf_image_crop.dims[-1]: slice(x_min_crop, x_max_crop),
        rf_image_crop.dims[-2]: slice(y_min_crop, y_max_crop),
    }
)

# Use xarray imshow to plot with correct coordinates
rf_image_crop.plot.imshow(
    ax=axes,
    cmap="coolwarm",
    add_colorbar=False,
)
axes.set_aspect(1)

# --- Add 10 µm scale bar ---
scale_um = 20  # scale bar in microns

# Use axis limits to place the bar inside the image
x_min, x_max = axes.get_xlim()
y_min, y_max = axes.get_ylim()

# Place bar a little inside the bottom-left corner
margin_x = (x_max - x_min) * 0.05
margin_y = (y_max - y_min) * 0.05

x_start = x_min + margin_x
y_start = y_min + margin_y

axes.hlines(
    y=y_start, xmin=x_start, xmax=x_start + scale_um, colors="black", linewidth=3
)
axes.text(
    x_start + scale_um / 2,
    y_start + 3,  # slightly above the bar
    f"{scale_um} µm",
    color="black",
    ha="center",
    va="bottom",
    fontsize=22,
)

# # Overlay 10µm x 10µm grid in central 100µm x 100µm region
# grid_total = 360  # total width/height of grid in µm
# grid_step = 10  # grid spacing in µm
#
# # Center coordinates of the image
# x_center = (x_min + x_max) / 2
# y_center = (y_min + y_max) / 2
#
# # Compute the extent of the grid
# x0 = x_center - grid_total / 2
# x1 = x_center + grid_total / 2
# y0 = y_center - grid_total / 2
# y1 = y_center + grid_total / 2

# Draw vertical lines
# for x in np.arange(x0, x1 + grid_step, grid_step):
# axes.vlines(ymin=y0, ymax=y1, x=x, colors="red", linewidth=1.5)

# Draw horizontal lines
# for y in np.arange(y0, y1 + grid_step, grid_step):
# axes.hlines(xmin=x0, xmax=x1, y=y, colors="red", linewidth=0.5)

# Set subplot title and labels
# axes.set_title(f"Cell {int(cell_id)}", fontsize=30)
axes.set_title("")
axes.set_xlabel("x position (µm)", fontsize=18)
axes.set_ylabel("y position (µm)", fontsize=18)

plt.tight_layout()
plt.show()

# %% Multi-panel RMS/CM RFs for 4 selected cells
# Independent normalization with matched-height colorbars
selected_cells = [229, 266, 268, 232]
# selected_cells = [246, 219, 77, 181]

crop_size = 600  # microns
half_crop = crop_size / 2
scale_um = 20  # scale bar length in µm

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(20, 5),
    constrained_layout=True,
)

fig.patch.set_alpha(0)

# ------------------------------------------------------------
# Plot each cell
# ------------------------------------------------------------
for ax, cell_id in zip(axes, selected_cells):
    ax.patch.set_alpha(0)

    # Load RF image
    rf_image = dataset_single["rms"].sel(
        channel=channel,
        cell_index=cell_id,
    )

    # Remove NaNs
    rf_image_crop = cutout_nans(rf_image)

    # Coordinates
    x_coords = rf_image_crop.coords[rf_image_crop.dims[-1]].values
    y_coords = rf_image_crop.coords[rf_image_crop.dims[-2]].values

    # Center
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2

    # Crop bounds
    x_min_crop = x_center - half_crop
    x_max_crop = x_center + half_crop
    y_min_crop = y_center - half_crop
    y_max_crop = y_center + half_crop

    # Crop
    rf_image_crop = rf_image_crop.sel(
        {
            rf_image_crop.dims[-1]: slice(x_min_crop, x_max_crop),
            rf_image_crop.dims[-2]: slice(y_min_crop, y_max_crop),
        }
    )

    # Plot
    im = rf_image_crop.plot.imshow(
        ax=ax,
        cmap="coolwarm",
        add_colorbar=False,
    )

    ax.set_aspect(1)

    # --------------------------------------------------------
    # Add colorbar matched to subplot height
    # --------------------------------------------------------
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,  # width of colorbar
        pad=0.04,  # spacing from subplot
    )
    if ax == axes[-1]:
        cbar.set_label("Root mean square", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    # --------------------------------------------------------
    # Add scale bar
    # --------------------------------------------------------
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    margin_x = (x_max - x_min) * 0.08
    margin_y = (y_max - y_min) * 0.08

    x_start = x_max - scale_um - margin_x
    y_start = y_max - margin_y

    ax.hlines(
        y=y_start,
        xmin=x_start,
        xmax=x_start + scale_um,
        colors="black",
        linewidth=2,
    )

    ax.text(
        x_start + scale_um / 2,
        y_start + 5,
        f"{scale_um} µm",
        color="black",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    # Titles
    ax.set_title(None)

    # Labels
    ax.set_xlabel("x position (µm)", fontsize=12)

    if ax == axes[0]:
        ax.set_ylabel("y position (µm)", fontsize=12)
    else:
        ax.set_ylabel("")

plt.show()
# %% Multi-panel RMS RFs for 4 selected cells
# Independent normalization with matched-height colorbars

selected_cells = [229, 266, 268, 232]

crop_size = 600  # microns
half_crop = crop_size / 2
scale_um = 20  # scale bar length in µm

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
    constrained_layout=True,
)

# Flatten axes for easy iteration
axes = axes.flatten()

fig.patch.set_alpha(0)

# ------------------------------------------------------------
# Plot each cell
# ------------------------------------------------------------
for i, (ax, cell_id) in enumerate(zip(axes, selected_cells)):
    # ax.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # --------------------------------------------------------
    # Load RF image
    # --------------------------------------------------------
    rf_image = dataset_single["rms"].sel(
        channel=channel,
        cell_index=cell_id,
    )

    # Skip invalid cells
    if np.isnan(rf_image.values).all():
        print(f"Cell {cell_id} is all NaNs")
        continue

    if not np.any(np.isfinite(rf_image.values)):
        print(f"Cell {cell_id} contains no finite values")
        continue

    # --------------------------------------------------------
    # Remove NaNs
    # --------------------------------------------------------
    rf_image_crop = cutout_nans(rf_image)

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------
    x_coords = rf_image_crop.coords[rf_image_crop.dims[-1]].values
    y_coords = rf_image_crop.coords[rf_image_crop.dims[-2]].values

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2

    # --------------------------------------------------------
    # Crop bounds
    # --------------------------------------------------------
    x_min_crop = x_center - half_crop
    x_max_crop = x_center + half_crop
    y_min_crop = y_center - half_crop
    y_max_crop = y_center + half_crop

    # --------------------------------------------------------
    # Crop image
    # --------------------------------------------------------
    rf_image_crop = rf_image_crop.sel(
        {
            rf_image_crop.dims[-1]: slice(x_min_crop, x_max_crop),
            rf_image_crop.dims[-2]: slice(y_min_crop, y_max_crop),
        }
    )

    # --------------------------------------------------------
    # Plot image
    # --------------------------------------------------------
    im = rf_image_crop.plot.imshow(
        ax=ax,
        cmap="viridis",
        add_colorbar=False,
    )

    ax.set_aspect("equal")

    # --------------------------------------------------------
    # Add colorbar
    # --------------------------------------------------------
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    import matplotlib.ticker as mticker

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))  # forces scientific notation

    cbar.formatter = formatter
    cbar.update_ticks()
    # Label only final colorbar
    if i == len(selected_cells) - 1:
        cbar.set_label("Root mean square error", fontsize=15, labelpad=15)

    cbar.ax.tick_params(labelsize=13)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    # --------------------------------------------------------
    # Add scale bar
    # --------------------------------------------------------
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    margin_x = (x_max - x_min) * 0.08
    margin_y = (y_max - y_min) * 0.08

    x_start = x_max - scale_um - margin_x
    y_start = y_max - margin_y

    ax.hlines(
        y=y_start,
        xmin=x_start,
        xmax=x_start + scale_um,
        colors="white",
        linewidth=2,
    )

    ax.text(
        x_start + scale_um / 2,
        y_start + 5,
        f"{scale_um} µm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=13,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    ax.set_xlabel("x position (µm)", fontsize=15)

    # Left column only
    if i in [0, 2]:
        ax.set_ylabel("y position (µm)", fontsize=15)
    else:
        ax.set_ylabel("")

    # Remove title
    ax.set_title("")

# ------------------------------------------------------------
# Show figure
# ------------------------------------------------------------
plt.show()
