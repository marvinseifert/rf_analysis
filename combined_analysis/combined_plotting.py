"""
This script shows how one can plot data from a single recording after all analysis have been performed.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from organize.configs import Recording_Config
from aquarel import load_theme
import matplotlib

coolwarm_heatmap = matplotlib.colormaps["coolwarm"]
heatmap_max = coolwarm_heatmap(1.0)
heatmap_min = coolwarm_heatmap(0.0)
theme = load_theme("scientific")
theme.apply()


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


# %%
# First, we need to load the dataset
path_to_data = Path(
    "/home/mawa/nas_a/Marvin/chicken_13_05_2025/Phase_00/noise_analysis_test/noise_data.nc"
)
dataset = xr.load_dataset(path_to_data)

rec_config = Recording_Config.load_from_root_json(path_to_data.parent)

# %%
# look for cell with max quality and plot it
# This plot plots rfs of channels side by side for the best cell
dataset["quality"] = dataset["quality"].fillna(0)
best_cell = dataset["cell_index"][
    dataset["quality"].mean(dim="channel").argmax("cell_index")
]
# or sorted cells:
sorted_cells = (
    dataset.dropna(dim="cell_index", how="any", subset=["quality"])
    .mean(dim="channel", skipna=True)
    .sortby("quality", ascending=False)["cell_index"]
)
# %%
best_cell = sorted_cells[0].values

fig, axs = plt.subplots(
    ncols=dataset.channel.values.shape[0], nrows=1, figsize=(20, 10)
)
for ax, channel in zip(axs, dataset.channel.values):
    dataset.sel(cell_index=best_cell, channel=channel)["cm_most_important"].plot.imshow(
        ax=ax,
        cmap="coolwarm",
        vmin=-np.nanmax(
            np.abs(
                dataset.sel(cell_index=best_cell, channel=channel)[
                    "cm_most_important"
                ].values
            )
        ),
        vmax=np.nanmax(
            np.abs(
                dataset.sel(cell_index=best_cell, channel=channel)[
                    "cm_most_important"
                ].values
            )
        ),
        add_colorbar=False,
    )
fig.show()
# %% Alternatively, plot on top of each other using datashader

combined = dataset["rms"].sel(cell_index=best_cell)

rgb_image = xr.DataArray(
    np.zeros((combined.y.size, combined.x.size, 3), dtype=np.uint8),
    coords={"y": combined.y, "x": combined.x, "channel": [0, 1, 2]},
    dims=["y", "x", "channel"],
)

for i, channel in enumerate(dataset.channel.values):
    channel_data = combined.sel(channel=channel).values
    channel_data = channel_data / np.nanmax(np.abs(channel_data))
    if rec_config.channel_colours[i] == "red":
        rgb_image.loc[:, :, 0] = (channel_data * 255).astype(np.uint8)
    elif rec_config.channel_colours[i] == "green":
        rgb_image.loc[:, :, 1] = (channel_data * 255).astype(np.uint8)
    elif rec_config.channel_colours[i] == "blue":
        rgb_image.loc[:, :, 2] = (channel_data * 255).astype(np.uint8)

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
fig, ax = plt.subplots(figsize=(10, 10))
rgb_image.plot.imshow(ax=ax)
ax.set_xlabel("x position (µm)")
ax.set_ylabel("y position (µm)")
fig.show()

# %% Plot the first 10 best cells in a grid
n_cells = 20
n_cols = 5
n_rows = int(np.ceil(n_cells / n_cols))

fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 5, n_rows * 5))
for i in range(n_cells):
    ax = axs.flatten()[i]
    ax.set_aspect("equal")
    best_cell = sorted_cells[i].values
    combined = dataset["rms"].sel(cell_index=best_cell)

    rgb_image = xr.DataArray(
        np.zeros((combined.y.size, combined.x.size, 3), dtype=np.uint8),
        coords={"y": combined.y, "x": combined.x, "channel": [0, 1, 2]},
        dims=["y", "x", "channel"],
    )

    for j, channel in enumerate(dataset.channel.values):
        channel_data = combined.sel(channel=channel).values
        channel_data = channel_data / np.nanmax(np.abs(channel_data))
        if rec_config.channel_colours[j] == "white":
            rgb_image.loc[:, :, 0] = (channel_data * 255).astype(np.uint8)
        elif rec_config.channel_colours[j] == "green":
            rgb_image.loc[:, :, 1] = (channel_data * 255).astype(np.uint8)
        elif rec_config.channel_colours[j] == "blue":
            rgb_image.loc[:, :, 2] = (channel_data * 255).astype(np.uint8)

    if np.all(rgb_image == 0):
        continue

    # need to find the first and last non-zero pixel in x and y to crop the image

    x_slice = slice(
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=0))[0][0] + 50,
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=0))[0][-1]
        + 1
        - 50,
        1,
    )
    y_slice = slice(
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=1))[0][0] + 50,
        np.where(np.any(rgb_image.sum(dim="channel").values > 0, axis=1))[0][-1]
        + 1
        - 50,
        1,
    )

    rgb_image = rgb_image.isel(x=x_slice, y=y_slice)
    rgb_image.plot.imshow(ax=ax)

    ax.set_xlabel("x position (µm)")
    ax.set_ylabel("y position (µm)")
    # remove ticks and labels if not on the left or bottom
    if i % n_cols != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    if i < (n_rows - 1) * n_cols:
        ax.set_xticklabels([])
        ax.set_xlabel("")
#
fig.show()

# %% PLot overlay of all cells in one channel
channel_to_plot = "560_nm"
# create maximum intensity projection over all cells
combined = dataset["rms"].sel(channel=channel_to_plot)
combined = combined / combined.max(dim=["x", "y"], skipna=True)
combined = combined.fillna(0)
max_projection = combined.max(dim="cell_index")

x_slice = slice(
    np.where(np.any(max_projection > 0, axis=0))[0][0],
    np.where(np.any(max_projection > 0, axis=0))[0][-1] + 1,
    1,
)
y_slice = slice(
    np.where(np.any(max_projection > 0, axis=1))[0][0],
    np.where(np.any(max_projection > 0, axis=1))[0][-1] + 1,
    1,
)

fig, ax = plt.subplots(figsize=(10, 10))
max_projection.isel(x=x_slice, y=y_slice).plot.imshow(
    ax=ax, cmap="gray", vmin=0, vmax=1
)
ax.set_xlabel("x position (µm)")
ax.set_ylabel("y position (µm)")
fig.show()
# %% Go mental and plot all stats for best cell
best_cell = sorted_cells[0]  # dataset["quality"].argmax("cell_index").values[0]
best_cell_data = dataset.sel(cell_index=best_cell)
# 3rd column needs to be polar plot of locations
column_ratios = [1.5, 1.5, 1, 1, 1, 1, 1]

fig, axs = plt.subplots(
    ncols=7,
    nrows=2,
    figsize=(25, 10),
    sharex="col",
    sharey="col",
    # 1. Define custom column widths using gridspec_kw
    gridspec_kw={"width_ratios": column_ratios},
)
axs[0, 2].remove()
axs[0, 2] = fig.add_subplot(2, 7, 3, projection="polar")
axs[0, 3].remove()
axs[0, 3] = fig.add_subplot(2, 7, 4, projection="polar")
axs[1, 2].remove()
axs[1, 2] = fig.add_subplot(2, 7, 10, projection="polar")
axs[1, 3].remove()
axs[1, 3] = fig.add_subplot(2, 7, 11, projection="polar")
for c_idx, channel in enumerate(dataset.channel.values):
    cutout_nans(best_cell_data.sel(channel=channel)["rms"]).plot.imshow(
        ax=axs[c_idx, 0],
        cmap="gray",
        vmin=0,
        vmax=best_cell_data.sel(channel=channel)["rms"].max(),
        add_colorbar=False,
    )
    axs[c_idx, 0].set_aspect("equal")
    cutout_nans(best_cell_data.sel(channel=channel)["cm_most_important"]).plot.imshow(
        ax=axs[c_idx, 1],
        cmap="coolwarm",
        vmin=-np.nanmax(
            np.abs(best_cell_data.sel(channel=channel)["cm_most_important"].values)
        ),
        vmax=np.nanmax(
            np.abs(best_cell_data.sel(channel=channel)["cm_most_important"].values)
        ),
        add_colorbar=False,
    )

    axs[c_idx, 1].set_aspect("equal")
    radii = best_cell_data.sel(channel=channel)["center_outline_um"]
    axs[c_idx, 2].plot(
        np.deg2rad(radii.coords["degree"].values), radii.values, c=heatmap_max
    )
    # set direction to clockwise
    axs[c_idx, 2].set_theta_zero_location("E")
    axs[c_idx, 2].set_rgrids(
        [
            int(
                np.round(radii.max().item() * i / 3, -1),
            )
            for i in range(1, 4)
        ]
    )
    # axs[c_idx, 2].set_theta_direction(-1)
    tilt = best_cell_data.sel(channel=channel)["angle"].item()
    axs[c_idx, 2].annotate(
        "",
        xy=(
            np.deg2rad(tilt),
            (1 - best_cell_data.sel(channel=channel)["tilt"]) * radii.max().item() * 2,
        ),
        xytext=(
            np.deg2rad(180) + np.deg2rad(tilt),
            (1 - best_cell_data.sel(channel=channel)["tilt"]) * radii.max().item() * 2,
        ),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )

    # surround outline
    radii = best_cell_data.sel(channel=channel)["surround_outline_um"]
    axs[c_idx, 3].plot(
        np.deg2rad(radii.coords["degree"].values), radii.values, c=heatmap_min
    )

    # move 0 to left
    axs[c_idx, 3].set_theta_zero_location("E")
    axs[c_idx, 3].set_rgrids(
        [
            int(
                np.round(radii.max().item() * i / 3, -1),
            )
            for i in range(1, 4)
        ]
    )
    # set direction to clockwise
    # axs[c_idx, 3].set_theta_direction(-1)
    # in out outline
    best_cell_data.sel(channel=channel)["in_out_outline_um"].plot.line(
        ax=axs[c_idx, 4], c="black"
    )
    # only need 5 major ticks on y axis
    axs[c_idx, 4].set_yticks(
        np.linspace(
            best_cell_data.sel(channel=channel)["in_out_outline_um"].min().item(),
            best_cell_data.sel(channel=channel)["in_out_outline_um"].max().item(),
            5,
        ),
    )

    axs[c_idx, 4].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    # plot zero line for this plot
    axs[c_idx, 4].axhline(0, color="black", linestyle="--")
    best_cell_data.sel(channel=channel)["sta_single_pixel"].plot.line(
        ax=axs[c_idx, 5], c="black"
    )

    sta_min, sta_max = (
        best_cell_data["sta_single_pixel"].min().item(),
        best_cell_data["sta_single_pixel"].max().item(),
    )
    max_deviation = max(abs(sta_min - 0.5), abs(sta_max - 0.5))
    axs[c_idx, 5].set_yticks(
        np.round(
            np.linspace(0.5 - max_deviation, 0.5 + max_deviation, 5),
            2,
        )
    )
    axs[c_idx, 5].set_yticks(
        np.round(
            np.linspace(
                best_cell_data.sel(channel=channel)["sta_single_pixel"].min().item(),
                best_cell_data.sel(channel=channel)["sta_single_pixel"].max().item(),
                6,
            ),
            2,
        )
    )
    # plot dotted line at time 0
    axs[c_idx, 5].axvline(0, color="black", linestyle="--")

    stats_text = (
        f"Quality: {best_cell_data.sel(channel=channel)['quality'].item():.2f}\n\n"
        f"Center Size (mm²): {best_cell_data.sel(channel=channel)['center_size_mm2'].item():.2e}\n\n"
        f"Surround Size (mm²): {best_cell_data.sel(channel=channel)['surround_size_mm2'].item():.2e}\n\n"
        f"TIR (um): {best_cell_data.sel(channel=channel)['tir'].item():.2f}\n\n"
        f"Center Shift (um): {best_cell_data.sel(channel=channel)['center_shift'].item():.2f}\n\n"
        f"Tilt: {best_cell_data.sel(channel=channel)['tilt'].item():.2f}\n\n"
        f"Angle (deg): {best_cell_data.sel(channel=channel)['angle'].item():.2f}\n"
    )
    axs[c_idx, 6].text(0.1, 0.5, stats_text, fontsize=12, va="center")
    axs[c_idx, 6].axis("off")

    if c_idx == 0:
        axs[c_idx, 0].set_title("RMS", y=1.05)
        axs[c_idx, 1].set_title("covariance map", y=1.05)
        axs[c_idx, 2].set_title("center outline", y=1.5)
        axs[c_idx, 3].set_title("surround outline", y=1.5)
        axs[c_idx, 4].set_title("in-out outline", y=1.05)
        axs[c_idx, 5].set_title("STA single pixel", y=1.05)
        axs[c_idx, 6].set_title("Statistics", y=1.05)
    # remove titles for all but first row
    else:
        axs[c_idx, 0].set_title("")
        axs[c_idx, 1].set_title("")
        axs[c_idx, 2].set_title("")
        axs[c_idx, 3].set_title("")
        axs[c_idx, 4].set_title("")
        axs[c_idx, 5].set_title("")
        axs[c_idx, 6].set_title("")
# need to increase the margin between subplots
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()
plt.close("all")

# %% Go mental an plot all stats for best cell
best_cell = sorted_cells[20]  # dataset["quality"].argmax("cell_index").values[0]
best_cell_data = dataset.sel(cell_index=best_cell)
# 3rd column needs to be polar plot of locations
column_ratios = [1.5, 1.5, 1, 1, 1, 1, 1]

fig, axs = plt.subplots(
    ncols=7,
    nrows=3,
    figsize=(25, 10),
    sharex="col",
    sharey="col",
    # 1. Define custom column widths using gridspec_kw
    gridspec_kw={"width_ratios": column_ratios},
)
axs[0, 2].remove()
axs[0, 2] = fig.add_subplot(3, 7, 3, projection="polar")
axs[0, 3].remove()
axs[0, 3] = fig.add_subplot(3, 7, 4, projection="polar")
axs[1, 2].remove()
axs[1, 2] = fig.add_subplot(3, 7, 10, projection="polar")
axs[1, 3].remove()
axs[1, 3] = fig.add_subplot(3, 7, 11, projection="polar")
axs[2, 2].remove()
axs[2, 2] = fig.add_subplot(3, 7, 17, projection="polar")
axs[2, 3].remove()
axs[2, 3] = fig.add_subplot(3, 7, 18, projection="polar")

for c_idx, channel in enumerate(dataset.channel.values):
    cutout_nans(best_cell_data.sel(channel=channel)["rms"], 50).plot.imshow(
        ax=axs[c_idx, 0],
        cmap="gray",
        vmin=0,
        vmax=best_cell_data.sel(channel=channel)["rms"].max(),
        add_colorbar=False,
    )
    axs[c_idx, 0].set_aspect("equal")
    cutout_nans(
        best_cell_data.sel(channel=channel)["cm_most_important"], 50
    ).plot.imshow(
        ax=axs[c_idx, 1],
        cmap="coolwarm",
        vmin=-np.nanmax(
            np.abs(best_cell_data.sel(channel=channel)["cm_most_important"].values)
        ),
        vmax=np.nanmax(
            np.abs(best_cell_data.sel(channel=channel)["cm_most_important"].values)
        ),
        add_colorbar=False,
    )

    axs[c_idx, 1].set_aspect("equal")
    radii = best_cell_data.sel(channel=channel)["center_outline_um"]
    axs[c_idx, 2].plot(
        np.deg2rad(radii.coords["degree"].values), radii.values, c=heatmap_max
    )
    # set direction to clockwise
    axs[c_idx, 2].set_theta_zero_location("E")
    axs[c_idx, 2].set_rgrids(
        [
            int(
                np.round(radii.max().item() * i / 3, -1),
            )
            for i in range(1, 4)
        ]
    )
    # axs[c_idx, 2].set_theta_direction(-1)
    tilt = best_cell_data.sel(channel=channel)["angle"].item()
    axs[c_idx, 2].annotate(
        "",
        xy=(
            np.deg2rad(tilt),
            (1 - best_cell_data.sel(channel=channel)["tilt"]) * radii.max().item() * 2,
        ),
        xytext=(
            np.deg2rad(180) + np.deg2rad(tilt),
            (1 - best_cell_data.sel(channel=channel)["tilt"]) * radii.max().item() * 2,
        ),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )

    # surround outline
    radii = best_cell_data.sel(channel=channel)["surround_outline_um"]
    axs[c_idx, 3].plot(
        np.deg2rad(radii.coords["degree"].values), radii.values, c=heatmap_min
    )

    # move 0 to left
    axs[c_idx, 3].set_theta_zero_location("E")
    axs[c_idx, 3].set_rgrids(
        [
            int(
                np.round(radii.max().item() * i / 3, -1),
            )
            for i in range(1, 4)
        ]
    )
    # set direction to clockwise
    # axs[c_idx, 3].set_theta_direction(-1)
    # in out outline
    best_cell_data.sel(channel=channel)["in_out_outline_um"].plot.line(
        ax=axs[c_idx, 4], c=rec_config.channel_colours[c_idx]
    )
    # only need 5 major ticks on y axis
    axs[c_idx, 4].set_yticks(
        np.linspace(
            best_cell_data.sel(channel=channel)["in_out_outline_um"].min().item(),
            best_cell_data.sel(channel=channel)["in_out_outline_um"].max().item(),
            5,
        ),
    )

    axs[c_idx, 4].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    # plot zero line for this plot
    axs[c_idx, 4].axhline(0, color="black", linestyle="--")
    best_cell_data.sel(channel=channel)["sta_single_pixel"].plot.line(
        ax=axs[c_idx, 5], c=rec_config.channel_colours[c_idx]
    )

    sta_min, sta_max = (
        best_cell_data["sta_single_pixel"].min().item(),
        best_cell_data["sta_single_pixel"].max().item(),
    )
    max_deviation = max(abs(sta_min - 0.5), abs(sta_max - 0.5))
    axs[c_idx, 5].set_yticks(
        np.round(
            np.linspace(0.5 - max_deviation, 0.5 + max_deviation, 5),
            2,
        )
    )
    axs[c_idx, 5].set_yticks(
        np.round(
            np.linspace(
                best_cell_data.sel(channel=channel)["sta_single_pixel"].min().item(),
                best_cell_data.sel(channel=channel)["sta_single_pixel"].max().item(),
                6,
            ),
            2,
        )
    )
    # plot dotted line at time 0
    axs[c_idx, 5].axvline(0, color="black", linestyle="--")

    stats_text = (
        f"Quality: {best_cell_data.sel(channel=channel)['quality'].item():.2f}\n\n"
        f"Center Size (mm²): {best_cell_data.sel(channel=channel)['center_size_mm2'].item():.2e}\n\n"
        f"Surround Size (mm²): {best_cell_data.sel(channel=channel)['surround_size_mm2'].item():.2e}\n\n"
        f"TIR (um): {best_cell_data.sel(channel=channel)['tir'].item():.2f}\n\n"
        f"Center Shift (um): {best_cell_data.sel(channel=channel)['center_shift'].item():.2f}\n\n"
        f"Tilt: {best_cell_data.sel(channel=channel)['tilt'].item():.2f}\n\n"
        f"Angle (deg): {best_cell_data.sel(channel=channel)['angle'].item():.2f}\n"
    )
    axs[c_idx, 6].text(0.1, 0.5, stats_text, fontsize=12, va="center")
    axs[c_idx, 6].axis("off")

    if c_idx == 0:
        axs[c_idx, 0].set_title("RMS", y=1.05)
        axs[c_idx, 1].set_title("covariance map", y=1.05)
        axs[c_idx, 2].set_title("center outline", y=1.5)
        axs[c_idx, 3].set_title("surround outline", y=1.5)
        axs[c_idx, 4].set_title("in-out outline", y=1.05)
        axs[c_idx, 5].set_title("STA single pixel", y=1.05)
        axs[c_idx, 6].set_title("Statistics", y=1.2)
    # remove titles for all but first row
    else:
        axs[c_idx, 0].set_title("")
        axs[c_idx, 1].set_title("")
        axs[c_idx, 2].set_title("")
        axs[c_idx, 3].set_title("")
        axs[c_idx, 4].set_title("")
        axs[c_idx, 5].set_title("")
        axs[c_idx, 6].set_title("")

# add combination of channel in last row
combined = dataset["rms"].sel(cell_index=best_cell)
rgb_image = xr.DataArray(
    np.zeros((combined.y.size, combined.x.size, 3), dtype=np.uint8),
    coords={"y": combined.y, "x": combined.x, "channel": [0, 1, 2]},
    dims=["y", "x", "channel"],
)
for i, channel in enumerate(dataset.channel.values):
    channel_data = combined.sel(channel=channel).values
    channel_data = channel_data / np.nanmax(np.abs(channel_data))
    if rec_config.channel_colours[i] == "red":
        rgb_image.loc[:, :, 0] = (channel_data * 255).astype(np.uint8)
    elif rec_config.channel_colours[i] == "green":
        rgb_image.loc[:, :, 1] = (channel_data * 255).astype(np.uint8)
    elif rec_config.channel_colours[i] == "blue":
        rgb_image.loc[:, :, 2] = (channel_data * 255).astype(np.uint8)
# slice out nans
rgb_image = cutout_nans(rgb_image, inset=50)
rgb_image.plot.imshow(ax=axs[2, 0])
axs[2, 0].set_title("Combined RGB RMS", y=1.05)
axs[2, 0].set_aspect("equal")
# we cant combine covariance maps, so display blank
axs[2, 1].axis("off")
# polar plot of combined outlines
radii = best_cell_data["center_outline_um"]
for channel_idx, channel in enumerate(dataset.channel.values):
    axs[2, 2].plot(
        np.deg2rad(radii.coords["degree"].values),
        radii.sel(channel=channel).values,
        c=rec_config.channel_colours[channel_idx],
    )
# set direction to clockwise
axs[2, 2].set_theta_zero_location("E")
axs[2, 2].set_rgrids(
    [
        int(
            np.round(radii.max().item() * i / 3, -1),
        )
        for i in range(1, 4)
    ]
)
# add arrows for each channel
for channel_idx, channel in enumerate(dataset.channel.values):
    tilt = best_cell_data.sel(channel=channel)["angle"].item()
    axs[2, 2].annotate(
        "",
        xy=(
            np.deg2rad(tilt),
            (1 - best_cell_data.sel(channel=channel)["tilt"]) * radii.max().item() * 2,
        ),
        xytext=(
            np.deg2rad(180) + np.deg2rad(tilt),
            (1 - best_cell_data.sel(channel=channel)["tilt"]) * radii.max().item() * 2,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            color=rec_config.channel_colours[channel_idx],
            lw=2,
        ),
    )
# surround outline
radii = best_cell_data["surround_outline_um"]
for channel_idx, channel in enumerate(dataset.channel.values):
    axs[2, 3].plot(
        np.deg2rad(radii.coords["degree"].values),
        radii.sel(channel=channel).values,
        c=rec_config.channel_colours[channel_idx],
    )
# move 0 to left
axs[2, 3].set_theta_zero_location("E")
axs[2, 3].set_rgrids(
    [
        int(
            np.round(radii.max().item() * i / 3, -1),
        )
        for i in range(1, 4)
    ]
)
for channel_idx, channel in enumerate(dataset.channel.values):
    # plot in out outline
    best_cell_data.sel(channel=channel)["in_out_outline_um"].plot.line(
        ax=axs[2, 4], c=rec_config.channel_colours[channel_idx]
    )
# only need 5 major ticks on y axis
axs[2, 4].set_yticks(
    np.linspace(
        best_cell_data["in_out_outline_um"].min().item(),
        best_cell_data["in_out_outline_um"].max().item(),
        5,
    ),
)
# add STA single pixel for all channels
for channel_idx, channel in enumerate(dataset.channel.values):
    best_cell_data.sel(channel=channel)["sta_single_pixel"].plot.line(
        ax=axs[2, 5], c=rec_config.channel_colours[channel_idx]
    )
axs[2, 0].set_title("")
axs[2, 1].set_title("")
axs[2, 2].set_title("")
axs[2, 3].set_title("")
axs[2, 4].set_title("")
axs[2, 5].set_title("")
axs[2, 6].set_title("")
# need to increase the margin between subplots
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.show()
plt.close("all")
