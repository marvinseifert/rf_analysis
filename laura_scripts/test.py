from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as mticker

from scipy.ndimage import shift as ndi_shift

from organize.configs import (
    Recording_Config,
    Collapse_2d_Config,
)

from location.x_array import x_y_and_scale
from loading.load_sta import load_and_realign_center


# %% ============================================================
# Recording config
# ============================================================

rec_object = Recording_Config(
    root_path=Path(r"F:\Laura\zebrafish_15_05_2026\Phase_00"),
)

rec_object.add_channel(
    stimulus_id=15,
    name="12px_20Hz_25mins_shuffle_x12_white",
    colour="red",
)

rec_object.add_channel(
    stimulus_id=19,
    name="12px_20Hz_25mins_shuffle_x12_repeat_white",
    colour="red",
)

collapse_2d_config = Collapse_2d_Config(
    recording_config=rec_object,
    cut_size_um=x_y_and_scale(800, 800),
)

noise_data_path = (
    rec_object.root_path / rec_object.output_folder / "noise_analysis" / "noise_data.nc"
)

dataset = xr.load_dataset(noise_data_path)


# %%


def calculate_rms_from_kernel_array(kernel, nr_of_spikes):
    """
    Match original pipeline RMS calculation.

    Original:
        sta_per_spike_raw = subset / nr_of_spikes - 0.5
        rms = (sta_per_spike_raw ** 2).max(dim="time")

    Here:
        kernel is a numpy array with shape time × y × x
    """

    # Convert numpy kernel to xarray so we can use dim="time"
    subset = xr.DataArray(
        kernel,
        dims=["time", "y", "x"],
        coords={
            "time": np.arange(kernel.shape[0]),
            "y": np.arange(kernel.shape[1]),
            "x": np.arange(kernel.shape[2]),
        },
    )

    sta_per_spike_raw = subset / nr_of_spikes - 0.5
    rms = (sta_per_spike_raw**2).max(dim="time")

    return sta_per_spike_raw, rms


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


def plot_individual_vs_combined_rms_from_kernel(
    dataset,
    recording_config,
    combined_kernel_path,
    cells_to_plot,
    individual_channels,
    individual_stimulus_ids,
    combined_label="combined kernel",
    kernel_filename="kernel_combined.npy",
    figsize_per_panel=(5, 5),
    crop_individual=True,
    cmap="gray",
):
    """
    Plot individual RMS RFs from dataset next to combined RMS RFs calculated
    directly from saved kernel_combined.npy files.

    Individual RMS:
        loaded from dataset["rms"]

    Combined RMS:
        loaded from:
            combined_kernel_path / cell_X / kernel_combined.npy

    Combined spike count:
        summed from recording_config.overview.spikes_df using the two
        original stimulus IDs.

    RMS calculation matches original pipeline:
        sta_per_spike_raw = kernel / nr_of_spikes - 0.5
        rms = (sta_per_spike_raw ** 2).max(dim="time")
    """

    combined_kernel_path = Path(combined_kernel_path)

    n_rows = len(cells_to_plot)
    n_cols = len(individual_channels) + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            figsize_per_panel[0] * n_cols,
            figsize_per_panel[1] * n_rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    spikes_df = recording_config.overview.spikes_df

    for row, cell_id in enumerate(cells_to_plot):
        cell_id_int = int(cell_id)

        # ============================================================
        # Individual channel RMS RFs from dataset
        # ============================================================
        for col, channel in enumerate(individual_channels):
            ax = axes[row, col]

            rf = dataset["rms"].sel(
                cell_index=cell_id,
                channel=channel,
            )

            if crop_individual:
                rf_plot = cutout_nans(rf)
            else:
                rf_plot = rf

            vmin = np.nanmin(rf_plot.values)
            vmax = np.nanmax(rf_plot.values)

            im = rf_plot.plot.imshow(
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
            )

            ax.set_title(
                f"Cell {cell_id_int}\n{channel}",
                fontsize=11,
            )

            ax.set_xlabel("x position")
            ax.set_ylabel("y position")
            ax.grid(False)

            im.colorbar.set_label("RMS")

        # ============================================================
        # Combined RMS from kernel_combined.npy
        # ============================================================
        ax = axes[row, -1]

        combined_file = combined_kernel_path / f"cell_{cell_id_int}" / kernel_filename

        if not combined_file.exists():
            raise FileNotFoundError(f"Could not find: {combined_file}")

        combined_kernel = np.load(combined_file)

        if combined_kernel.ndim != 3:
            raise ValueError(
                f"Expected combined kernel shape time × y × x, "
                f"but got {combined_kernel.shape} for cell {cell_id_int}"
            )

        # ============================================================
        # Get total spike count from the original two stimulus IDs
        # ============================================================
        nr_of_spikes = 0

        for stimulus_id in individual_stimulus_ids:
            spike_row = spikes_df.query(
                f"stimulus_index == {stimulus_id} and cell_index == {cell_id_int}"
            )

            if len(spike_row) == 0:
                raise ValueError(
                    f"No spike count found for cell {cell_id_int}, "
                    f"stimulus_index {stimulus_id}"
                )

            nr_of_spikes += int(spike_row["nr_of_spikes"].values[0])

        # ============================================================
        # Calculate RMS using original pipeline logic
        # ============================================================
        sta_per_spike_raw, combined_rms = calculate_rms_from_kernel_array(
            combined_kernel,
            nr_of_spikes,
        )

        vmin = np.nanmin(combined_rms.values)
        vmax = np.nanmax(combined_rms.values)

        im = combined_rms.plot.imshow(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
        )

        ax.set_title(
            f"Cell {cell_id_int}\n{combined_label}\n" f"n spikes = {nr_of_spikes}",
            fontsize=11,
        )

        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.grid(False)

        im.colorbar.set_label("Combined RMS")

    plt.show()


# %%
combined_kernel_path = Path(
    r"F:\Laura\zebrafish_15_05_2026\Phase_00\combined_kernels_idx_20"
)

plot_individual_vs_combined_rms_from_kernel(
    dataset=dataset,
    recording_config=Recording_Config,
    combined_kernel_path=combined_kernel_path,
    cells_to_plot=[1, 10],
    individual_channels=[
        "12px_20Hz_25mins_shuffle_x12_white",
        "12px_20Hz_25mins_shuffle_x12_repeat_white",
    ],
    individual_stimulus_ids=[15, 19],
    combined_label="12px_20Hz_50mins_white_combined",
)
