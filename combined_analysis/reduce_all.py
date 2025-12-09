from __future__ import annotations
import polars as pl
import numpy as np
from pathlib import Path
import pickle
from polars import from_numpy
from dim_reduction.circular_reduction import polar_transform
from location.channel_handling import masking_square
from polarspike import Overview
from tqdm import tqdm
import warnings
from rf_torch.parameters import Noise_Params
import matplotlib.pyplot as plt
from smoothing.gaussian import smooth_ker
from covariance.filtering import cov_filtering_sum
from location.border import check_border_constrains
from loading.load_sta import load_sta_subset
import xarray as xr
from location.x_array import x_y_and_scale
from location.locate import locate_across_channels
from importlib import reload
from organize.decorators import depends_on
from loading.load_sta import load_sta_as_xarray


@depends_on("calculate_rf_quality")
def sta_2d_cov_collapse(
        recording_config: "Recording_Config",
        collapse_2d_config: "Collapse_2d_Config",
        analysis_folder: str,
):
    def default_exceptions():
        """
        I scope function for handling exceptions by appending default values to the stores.

        Returns
        -------
        tuple cm_most_important_store, sta_single_store
            Appended default values to the stores.

        """
        cm_most_important_store.append(
            np.zeros(
                (
                    collapse_2d_config.cut_size_px[channel].loc["x"].item()
                    * collapse_2d_config.cut_size_px[channel].loc["y"].item(),
                )
            )
        )
        sta_single_store.append(
            np.zeros((recording_config.channel_configs[channel].total_sta_len,))
        )
        return cm_most_important_store, sta_single_store

    channel_roots = [
        channel["root_path"] for channel in recording_config.channel_configs.values()
    ]
    # Calculate common location across channels
    cell_qualities, positions = locate_across_channels(
        channel_roots,
        threshold=collapse_2d_config.threshold,
        channel_names=recording_config.channel_names,
    )

    # %% Construct the results dataframe
    schema = {
        "cell_index": pl.UInt32,
        "channel": pl.Categorical,
        "sta_path": pl.Utf8,
        "positions": pl.Array(pl.UInt16, 2),
        "quality": pl.Float64,
    }
    sta_paths = [
        f"{channel_root.parts[-1]}/cell_{cell_idx}/kernel.npy"
        for cell_idx in cell_qualities["cell_index"].values
        for channel_root in channel_roots
    ]
    data = {
        "cell_index": cell_qualities["cell_index"]
        .values.repeat(recording_config.nr_channels)
        .astype(np.uint32),
        "channel": recording_config.channel_names * cell_qualities.sizes["cell_index"],
        "sta_path": [path for path in sta_paths],
        "positions": positions["quality_metrics"]
        .to_numpy()
        .astype(np.uint16)
        .repeat(recording_config.nr_channels, axis=0),
        "quality": cell_qualities.sel(metrics="quality")["quality_metrics"]
        .to_numpy()
        .flatten("F"),
    }
    df = pl.DataFrame(data, schema=schema)
    # save as parquet file
    df.write_parquet(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_df.parquet"
    )

    # %% Step two sta extension and covariance matrix
    # This is a for loop, because the data is too large for running in parallel.
    cm_most_important_store = []
    sta_single_store = []
    plot_trigger = 0
    for row_idx, row in tqdm(
            enumerate(df.iter_rows()),
            total=len(df),
            desc="Calculating covariance matrices",
    ):
        # Extract relevant info from the row
        cell_idx = row[df.get_column_index("cell_index")]
        channel = row[df.get_column_index("channel")]
        if row[df.get_column_index("quality")] < collapse_2d_config.threshold:
            default_exceptions()
            continue

        try:
            # Load STA subset
            subset, c_x, c_y = load_sta_subset(
                recording_config.root_path / row[df.get_column_index("sta_path")],
                positions=tuple(row[df.get_column_index("positions")]),
                subset_size=tuple(
                    [
                        collapse_2d_config.cut_size_px[channel].loc["x"].item(),
                        collapse_2d_config.cut_size_px[channel].loc["y"].item(),
                    ]
                ),
                dt_ms=recording_config.channel_configs[channel].dt_ms,
                t_zero_index=recording_config.channel_configs[channel].total_sta_len,
            )
        except FileNotFoundError:
            print(f"File not found for item {cell_idx}")
            default_exceptions()
            continue

        if plot_trigger == 0:
            # Sanity plot for the first cell only
            plot_trigger = 1
            fig, ax = plt.subplots(figsize=(10, 10))
            ishow = subset.var("time").plot.imshow(
                ax=ax,
                cmap="gray",
                x="x",
                y="y",
                cbar_kwargs={"shrink": 0.5},
            )
            ax.set_aspect("equal")
            ax.set_title("Sanity plot subset")
            ax.scatter(c_x, c_y, color="red", marker="x")
            fig.show()

        # Get number of spikes for normalization

        stimulus_id = int(df[cell_idx, "sta_path"].split("_")[-2][:1])
        nr_of_spikes = recording_config.overview.spikes_df.query(
            f"stimulus_index=={stimulus_id}&cell_index=={df[cell_idx, 'cell_index']}"
        )["nr_of_spikes"].values[0]

        sta_per_spike_raw = (subset / nr_of_spikes - 0.5).values  # shape: (time, h, w)

        time_bins, h, w = sta_per_spike_raw.shape

        flat = sta_per_spike_raw.reshape(time_bins, h * w).astype(
            np.float32, copy=False
        )

        flat_centered = flat - flat.mean(axis=0, keepdims=True)

        # SVD on the CENTERED data
        try:
            # Use flat_centered here, NOT flat (raw)
            _, s, Vt = np.linalg.svd(flat_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"SVD did not converge for cell {cell_idx}, skipping.")
            default_exceptions()
            continue
        # Covariance with most important pixel
        loadings = Vt[0]  # shape: (h*w,)
        most_idx = int(np.argmax(np.abs(loadings)))
        cov_with_most = (flat_centered.T @ flat_centered[:, most_idx]) / (time_bins - 1)
        cm_most_important_store.append(cov_with_most)
        # Store STA time course of the selected pixel
        pix_y, pix_x = divmod(most_idx, w)
        sta_single_store.append(sta_per_spike_raw[:, pix_y, pix_x] + 0.5)

        #

    # Add to the dataframe
    cm_most_important_series = pl.Series(
        "cm_most_important",
        cm_most_important_store,
        dtype=pl.List(
            pl.Float64,
        ),
    )

    sta_single_series = pl.Series(
        "sta_single",
        sta_single_store,
        dtype=pl.List(
            pl.Float64,
        ),
    )

    # %% Save the dataframe
    df = df.with_columns(cm_most_important_series)
    df = df.with_columns(sta_single_series)
    # noise_store.df = noise_store.df.with_columns(cm_signed_series)
    df.write_parquet(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_df.parquet"
    )
    return


@depends_on("calculate_rf_quality")
@depends_on("sta_2d_cov_collapse")
def circular_reduction(
        recording_config: "Recording_Config",
        collapse_2d_config: "Collapse_2d_Config",
        circular_reduction_config: Circular_Reduction_Config,
        analysis_folder: str,
):
    df = pl.read_parquet(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_df.parquet"
    )

    # %%

    center_outline_store = xr.DataArray(
        np.zeros((len(df), int(360 / circular_reduction_config.degree_bins))),
        dims=["cell", "degree"],
        coords={
            "cell": df["cell_index"].to_numpy(),
            "degree": np.arange(0, 360, circular_reduction_config.degree_bins),
        },
    )
    surround_outline_store = xr.full_like(center_outline_store, 0)
    in_out_outline_store = (
        []
    )  # Need List, because radius can vary for different channels
    center_std = np.zeros((len(df)))
    surround_std = np.zeros_like(center_std)

    for row_idx, row in tqdm(
            enumerate(df.iter_rows()),
            total=len(df),
            desc="Performing circular reduction",
    ):
        channel = row[df.get_column_index("channel")]
        if row[df.get_column_index("quality")] < collapse_2d_config.threshold:
            in_out_outline_store.append(
                np.zeros(int(round(collapse_2d_config.max_radius_px[channel].item())))
            )
            continue
        center = (
            collapse_2d_config.half_cut_size_px[channel].loc["x"].item(),
            collapse_2d_config.half_cut_size_px[channel].loc["y"].item(),
        )
        polar_cm = polar_transform(
            np.asarray(row[df.get_column_index("cm_most_important")]).reshape(
                collapse_2d_config.cut_size_px[channel].loc["y"].item(),
                collapse_2d_config.cut_size_px[channel].loc["x"].item(),
            ),
            center,
            max_radius=int(round(collapse_2d_config.max_radius_px[channel].item())),
        )
        entries = []
        for deg_step in range(0, 360, circular_reduction_config.degree_bins):
            entries.append(
                np.mean(
                    polar_cm[
                        deg_step: deg_step + circular_reduction_config.degree_bins,
                        :,
                    ],
                    axis=0,
                )
            )
        entries = np.asarray(entries)
        entries_pos = entries.copy()
        entries_pos[entries_pos < 0] = 0
        entries_mean = (
                np.mean(entries_pos / np.max(entries_pos), axis=1) * entries_pos.shape[1]
        )

        entries_neg = entries.copy()
        entries_neg[entries_neg > 0] = 0
        entries_mean_neg = (
                np.mean(entries_neg / np.min(entries_neg), axis=1) * entries_neg.shape[1]
        )
        entries_mean[np.isnan(entries_mean)] = 0
        entries_mean_neg[np.isnan(entries_mean_neg)] = 0

        center_outline_store[row_idx, :] = entries_mean
        surround_outline_store[row_idx, :] = entries_mean_neg
        in_out_outline_store.append(np.mean(entries, axis=0))

    # %% Add results to the dataframe
    center_outline = pl.Series(
        "center_outline",
        center_outline_store.values,
        dtype=pl.Array(pl.Float64, 360 // circular_reduction_config.degree_bins),
    )
    surround_outline = pl.Series(
        "surround_outline",
        surround_outline_store.values,
        dtype=pl.Array(pl.Float64, 360 // circular_reduction_config.degree_bins),
    )
    in_out_outline = pl.Series(
        "in_out_outline",
        in_out_outline_store,
        dtype=pl.List(pl.Float64),
    )

    nr_columns = len(df.columns)

    for series_idx, series in enumerate(
            [center_outline, surround_outline, in_out_outline]
    ):
        df.with_columns(series)

    df.write_parquet(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_df.parquet"
    )
    return
