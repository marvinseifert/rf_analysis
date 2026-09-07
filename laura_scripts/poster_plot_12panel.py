# %% Poster plot
# Multipanel plot

"""
Create a 4-row × 3-column multipanel figure for two cells.

Layout
------
Rows 0-1 : Cell 1
    Row 0 -> cm_most_important (with cutout_nans applied)
    Row 1 -> sta_single_pixel

Rows 2-3 : Cell 2
    Row 2 -> cm_most_important (with cutout_nans applied)
    Row 3 -> sta_single_pixel

Columns correspond to the three channels in the dataset.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from organize.configs import Recording_Config
from aquarel import load_theme
import matplotlib
import matplotlib.ticker as mticker

# =============================================================================
# Plot style
# =============================================================================

theme = load_theme("scientific")
theme.apply()

coolwarm_heatmap = matplotlib.colormaps["coolwarm"]

# =============================================================================
# Helper functions
# =============================================================================


def cutout_nans(data, inset=0):
    """
    Crop image to valid non-zero/non-NaN region.
    """

    valid_mask = np.nan_to_num(data.values) != 0

    x_valid = np.where(np.any(valid_mask, axis=0))[0]
    y_valid = np.where(np.any(valid_mask, axis=1))[0]

    if len(x_valid) == 0 or len(y_valid) == 0:
        return data

    x_slice = slice(
        x_valid[0] + inset,
        x_valid[-1] + 1 - inset,
        1,
    )

    y_slice = slice(
        y_valid[0] + inset,
        y_valid[-1] + 1 - inset,
        1,
    )

    return data.isel(x=x_slice, y=y_slice)


# %%
def add_scale_bar(ax, length_um=20, height_offset=0.05, pad=0.05, lw=2):
    """
    Add a horizontal scale bar in data coordinates.
    Assumes x-axis is in micrometers.
    """

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # place bar in top-right corner
    x_start = x1 - (x1 - x0) * pad - length_um
    y_start = y1 - (y1 - y0) * pad

    ax.plot(
        [x_start, x_start + length_um],
        [y_start, y_start],
        color="black",
        linewidth=lw,
        solid_capstyle="butt",
    )

    ax.text(
        x_start + length_um / 2,
        y_start + 0.01 * (y1 - y0),  # <-- move text upward
        f"{length_um} µm",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )


# %%
# =============================================================================
# Load dataset
# =============================================================================

path_to_data = Path(
    r"F:\Laura\zebrafish_26_02_2026\Phase_00\noise_analysis\noise_data.nc"
)

dataset = xr.load_dataset(path_to_data)

rec_config = Recording_Config.load_from_root_json(path_to_data.parent)

# =============================================================================
# Select cells
# =============================================================================
# %%
dataset["quality"] = dataset["quality"].fillna(0)

sorted_cells = (
    dataset.dropna(dim="cell_index", how="any", subset=["quality"])
    .mean(dim="channel", skipna=True)
    .sortby("quality", ascending=False)["cell_index"]
)

# Select two cells
# cell_1 = sorted_cells[0].item()
# cell_2 = sorted_cells[1].item()
cell_1 = 235
cell_2 = 135
cells_to_plot = [cell_1, cell_2]

print(f"Plotting cells: {cells_to_plot}")

# =============================================================================
# =============================================================================
# Create figure
# =============================================================================

fig, axs = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(17, 18),
    constrained_layout=True,
)

fig.patch.set_alpha(0)

# =============================================================================
# Plot
# =============================================================================

for cell_idx, cell in enumerate(cells_to_plot):
    cm_row = cell_idx * 2
    sta_row = cell_idx * 2 + 1

    cell_data = dataset.sel(cell_index=cell)

    for ch_idx, channel in enumerate(dataset.channel.values):
        swap_idx = [2, 1, 0][ch_idx]  # swaps column 0 <-> 2

        # ---------------------------------------------------------------------
        # CM IMAGE
        # ---------------------------------------------------------------------

        ax_img = axs[cm_row, swap_idx]

        cm_data = cell_data.sel(channel=channel)["cm_most_important"]
        cm_data_cut = cutout_nans(cm_data, inset=0)

        vmax = np.nanmax(np.abs(cm_data.values))

        im = cm_data_cut.plot.imshow(
            ax=ax_img,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            add_colorbar=True,
        )
        add_scale_bar(ax_img, length_um=20)
        ax_img.grid(False)
        ax_img.set_title("")
        ax_img.set_xlabel("x position (µm)", labelpad=4, fontsize=20)
        ax_img.set_ylabel("y position (µm)", labelpad=12, fontsize=20)
        ax_img.tick_params(axis="both", labelsize=18)
        # -------------------------------------------------------------
        # ONLY LABEL COLORBAR IN THIRD COLUMN
        # -------------------------------------------------------------
        cbar = im.colorbar

        # get current position
        pos = cbar.ax.get_position()

        # move it closer (reduce horizontal gap)
        cbar.ax.set_position(
            [
                pos.x0 - 0.1,  # move left (tune this)
                pos.y0,
                pos.width * 0.8,  # optional: make bar thinner
                pos.height,
            ]
        )
        import matplotlib.ticker as mticker

        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # force scientific notation

        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=17)
        # cbar.formatter.set_powerlimits((0, 0))
        # cbar.formatter.set_scientific(True)
        # cbar.update_ticks()
        #
        if swap_idx == 2:
            cbar.set_label("Covariance", labelpad=12, fontsize=20)
        else:
            cbar.set_label("")

        # ---------------------------------------------------------------------
        # STA SINGLE PIXEL
        # ---------------------------------------------------------------------

        ax_sta = axs[sta_row, swap_idx]

        sta = cell_data.sel(channel=channel)["sta_single_pixel"]

        sta.plot.line(
            ax=ax_sta,
            color=rec_config.channel_colours[ch_idx],
        )

        ax_sta.axvline(0, color="black", linestyle="--", linewidth=1)

        sta_min = sta.min().item()
        sta_max = sta.max().item()

        max_deviation = max(abs(sta_min - 0.5), abs(sta_max - 0.5))

        ax_sta.set_ylim(
            0.5 - max_deviation,
            0.5 + max_deviation,
        )

        ax_sta.grid(False)
        ax_sta.set_title("")

        ax_sta.set_xlabel("Time", labelpad=4, fontsize=20)
        ax_sta.set_ylabel("STA", labelpad=12, fontsize=20)
        ax_sta.set_xticks(
            [
                -800,
                -400,
                0,
            ]
        )
        ax_sta.tick_params(axis="both", labelsize=18)

        ax_sta.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
plt.show()
plt.close("all")

# %%
