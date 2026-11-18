from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from organize.configs import Recording_Config

# Load the data
path_to_data = Path(
    r"F:\Laura\zebrafish_19_02_2026\Phase_00\noise_analysis\noise_data.nc"
)

dataset = xr.load_dataset(path_to_data)
rec_config = Recording_Config.load_from_root_json(path_to_data.parent)

# %%
dataset = dataset.sel(channel=dataset.channel != "4px_20Hz_shuffle", drop=True)

# dataset = dataset.drop_sel(channel="4px_20Hz_40mins_shuffle_365")
# Find top cells
# Load quality index
quality = dataset["quality"]
# Set quality index threshold
qi_limit = 20
# Select cell indices for each channel individually which have qi > qi_limit
top_cells_mask = (quality > qi_limit).all(dim="channel")
# Extract the cell indices
top_cells = quality.cell_index.values[top_cells_mask.values]

# %%
import math
import matplotlib.patches as mpatches
from skimage import measure

top_cells = [57, 58, 60, 77, 78, 153, 178, 185, 206, 208, 209, 214, 224, 229, 262, 276]
# Find appropriate number of columns and rows in plot
n_cols = math.ceil(math.sqrt(len(top_cells)))
n_rows = math.ceil(len(top_cells) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10), dpi=300)
axes = axes.flatten()


for idx, cell_id in enumerate(top_cells):
    combined = dataset["rms"].sel(cell_index=cell_id)

    rgb_image = xr.DataArray(
        np.zeros((combined.y.size, combined.x.size, 3), dtype=np.uint8),
        coords={"y": combined.y, "x": combined.x, "channel": [0, 1, 2]},
        dims=["y", "x", "channel"],
    )

    for i, channel in enumerate(dataset.channel.values):
        channel_data = combined.sel(channel=channel).values
        channel_data = channel_data / np.nanmax(np.abs(channel_data))
        if rec_config.channel_colours[i] == "red":
            rgb_image.loc[:, :, 0] = (channel_data * 255).astype(
                np.uint8
            )  # rgb red channel
        elif rec_config.channel_colours[i] == "blue":
            rgb_image.loc[:, :, 1] = (channel_data * 255).astype(
                np.uint8
            )  # rgb green channel
        elif rec_config.channel_colours[i] == "white":
            rgb_image.loc[:, :, 2] = (channel_data * 255).astype(
                np.uint8
            )  # rgb blue channel

    # need to find the first and last non-zero pixel in x and y to crop the image
    x_slice = slice(
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=0))[0][0],
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=0))[0][-1] + 1,
        1,
    )
    y_slice = slice(
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=1))[0][0],
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=1))[0][-1] + 1,
        1,
    )

    rgb_image = rgb_image.isel(x=x_slice, y=y_slice)
    rgb_image.plot.imshow(ax=axes[idx])
    axes[idx].set_xlabel("x position (µm)")
    axes[idx].set_ylabel("y position (µm)")
    axes[idx].set_title(f"cell {cell_id}")
    # Make the plot square
    axes[idx].set_aspect("equal")  # <-- ensures square pixels / equal axes

    # --- Add 10 µm scale bar ---
    scale_um = 10  # scale bar in microns

    # Use axis limits to place the bar inside the image
    x_min, x_max = axes[idx].get_xlim()
    y_min, y_max = axes[idx].get_ylim()

    # Place bar a little inside the bottom-left corner
    margin_x = (x_max - x_min) * 0.05  # 2% margin
    margin_y = (y_max - y_min) * 0.05

    x_start = x_max - scale_um - margin_x  # start so bar fits inside axis
    y_start = y_max - margin_y

    axes[idx].hlines(
        y=y_start, xmin=x_start, xmax=x_start + scale_um, colors="white", linewidth=1.5
    )
    axes[idx].text(
        x_start + scale_um / 2,
        y_start + 10,  # slightly above the bar
        f"{scale_um} µm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # # --- Second scale bar (center of the plot) ---
    # x_center_start = (x_min + x_max) / 2 - scale_um / 2  # centered horizontally
    # y_center = (y_min + y_max) / 2  # centered vertically
    #
    # axes[idx].hlines(
    #     y=y_center,
    #     xmin=x_center_start,
    #     xmax=x_center_start + scale_um,
    #     colors="white",
    #     linewidth=1.5,
    # )

# Turn off any empty axes
for idx in range(len(top_cells), len(axes)):
    axes[idx].axis("off")

# Create custom legend handles
# handles = [
#     mpatches.Patch(color="red", label="4px_20Hz_40mins_shuffle_460_560"),
#     mpatches.Patch(color="green", label="4px_20Hz_40mins_shuffle"),
#     # mpatches.Patch(color="green", label="12px_20Hz_shuffle")
# ]

# Add a single legend to the figure
# fig.suptitle(
#     f"Overlapping RFs (interesting cells?)",
#     fontsize=25,
#     # y=0.99,
# )
# fig.legend(handles=handles, loc="lower right", fontsize=12)

fig.tight_layout()
fig.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Cells → RGB channels
# --------------------
cells = [205, 206]
cell_to_rgb = {
    cells[0]: 0,  # red
    cells[1]: 1,  # green
    # cells[2]: 2,  # blue
}

# --------------------
# Find index of recording channel to use
# --------------------
channel_idx = [i for i, c in enumerate(rec_config.channel_colours) if c == "red"][0]
channel = dataset.channel.values[channel_idx]

# --------------------
# Create empty RGB canvas
# --------------------
ref = dataset["rms"].isel(cell_index=0)

rgb_image = np.zeros((ref.y.size, ref.x.size, 3), dtype=float)
signal_sum = np.zeros((ref.y.size, ref.x.size), dtype=float)

# --------------------
# Fill RGB channels
# --------------------
for cell_id in cells:
    combined = dataset["rms"].sel(cell_index=cell_id)
    data = combined.sel(channel=channel).values

    max_val = np.nanmax(np.abs(data))
    if max_val == 0 or np.isnan(max_val):
        continue  # skip empty RFs safely

    data = data / max_val

    rgb_idx = cell_to_rgb[cell_id]
    rgb_image[..., rgb_idx] += data
    signal_sum += np.abs(data)

rgb_image = np.clip(rgb_image, 0, 1)

# --------------------
# Compute RF center SAFELY
# --------------------
mask = signal_sum > 0

if np.any(mask):
    yy, xx = np.where(mask)
    x_center = np.mean(ref.x.values[xx])
    y_center = np.mean(ref.y.values[yy])
else:
    # Fallback: image center
    x_center = ref.x.values.mean()
    y_center = ref.y.values.mean()

# --------------------
# Plot
# --------------------
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

ax.imshow(
    rgb_image,
    origin="lower",
    extent=[
        ref.x.min(),
        ref.x.max(),
        ref.y.min(),
        ref.y.max(),
    ],
)

# --------------------
# Crop to 200 × 200 µm
# --------------------
half_size = 300
ax.set_xlim(x_center - half_size, x_center + half_size)
ax.set_ylim(y_center - half_size, y_center + half_size)

# --------------------
# Formatting
# --------------------
ax.set_aspect("equal")
ax.set_xlabel("x position (µm)")
ax.set_ylabel("y position (µm)")
ax.set_title(f"Overlaid receptive fields {cells}")

# --- Add 10 µm scale bar ---
scale_um = 10  # scale bar in microns

# Use axis limits to place the bar inside the image
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Place bar a little inside the bottom-left corner
margin_x = (x_max - x_min) * 0.05  # 2% margin
margin_y = (y_max - y_min) * 0.05

x_start = x_max - scale_um - margin_x  # start so bar fits inside axis
y_start = y_max - margin_y

ax.hlines(
    y=y_start, xmin=x_start, xmax=x_start + scale_um, colors="white", linewidth=1.5
)
ax.text(
    x_start + scale_um / 2,
    y_start + 10,  # slightly above the bar
    f"{scale_um} µm",
    color="white",
    ha="center",
    va="bottom",
    fontsize=12,
)

plt.tight_layout()
plt.show()
