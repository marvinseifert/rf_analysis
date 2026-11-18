import sys

sys.path.extend(["/home/mawa/PycharmProjects/polarspike"])
from IPython.display import display, HTML
from polarspike import (
    Overview,
    recording_overview,
    spiketrain_plots,
    colour_template,
    quality_tests,
)
import polars as pl
from bokeh.io import output_notebook
from bokeh.plotting import show
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from organize.configs import Recording_Config
from aquarel import load_theme
import matplotlib

# %% First, load the RF dataset

path_to_data = Path(
    r"F:\Laura\zebrafish_22_07_2026\Phase_00\noise_analysis\noise_data.nc"
)

dataset = xr.load_dataset(path_to_data)

# %% Make a per-cell RF metadata table

rf_cell_table = (
    dataset[["tilt", "quality"]]
    .to_dataframe()
    .reset_index()[["cell_index", "tilt", "quality"]]
    .drop_duplicates(subset=["cell_index"])
    .rename(
        columns={
            "tilt": "rf_tilt",
            "quality": "rf_qi",
        }
    )
)

rf_cell_table["cell_index"] = rf_cell_table["cell_index"].astype(int)

print("RF cell table:")
print(rf_cell_table.head())

# %% Import and process Polarspike data

spikes_on_disk = pl.scan_parquet(
    r"F:\Laura\zebrafish_15_01_2026\Phase_00\alldata.parquet"
)

recording = Overview.Recording.load(r"F:\Laura\zebrafish_15_01_2026\Phase_00\overview")

# Select just the first full-field / SCF stimulus
spikes = recording.get_spikes_triggered([{"stimulus_index": [0]}])

# Make a Polars version for Polarspike functions

if isinstance(spikes, pl.LazyFrame):
    spikes_pl = spikes.collect()

elif isinstance(spikes, pl.DataFrame):
    spikes_pl = spikes

elif isinstance(spikes, pd.DataFrame):
    spikes_pl = pl.from_pandas(spikes)

else:
    spikes_pl = pl.DataFrame(spikes)

# Important fix:
# Polarspike / Polars expects Int64, but some columns may come in as Int32.
# This casts all integer columns to Int64 before spiketrain_qi is called.
integer_columns = [
    col
    for col, dtype in spikes_pl.schema.items()
    if dtype
    in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]
]

spikes_pl = spikes_pl.with_columns(
    [pl.col(col).cast(pl.Int64) for col in integer_columns]
)

# Also explicitly cast the key columns, just to be safe
spikes_pl = spikes_pl.with_columns(
    [
        pl.col("cell_index").cast(pl.Int64),
        pl.col("stimulus_index").cast(pl.Int64),
    ]
)

print("Spikes Polars schema after casting:")
print(spikes_pl.schema)

print("Original spikes table:")
print(spikes_pl.head())

# Calculate full-field response QI

qi = quality_tests.spiketrain_qi(
    spikes_pl,
    max_window=2 * 16,
    max_repeat=5,
)

# Convert QI output to pandas and clean it

if isinstance(qi, pl.DataFrame):
    qi = qi.to_pandas()

elif isinstance(qi, pl.Series):
    qi = qi.to_pandas().rename("qi").reset_index()

elif isinstance(qi, pd.Series):
    qi = qi.rename("qi").reset_index()

else:
    qi = pd.DataFrame(qi).reset_index()

# If cell_index has come out as "index", rename it
if "cell_index" not in qi.columns and "index" in qi.columns:
    qi = qi.rename(columns={"index": "cell_index"})

if "cell_index" not in qi.columns and "level_0" in qi.columns:
    qi = qi.rename(columns={"level_0": "cell_index"})

# Add stimulus index, because this QI belongs to the selected full-field stimulus
stimulus_index = int(spikes_pl["stimulus_index"].unique()[0])
qi["stimulus_index"] = stimulus_index

# Rename the full-field quality column to qi
# RF quality is called rf_qi, so full-field response quality should be qi.
if "quality" in qi.columns:
    qi = qi.rename(columns={"quality": "qi"})

elif "spiketrain_qi" in qi.columns:
    qi = qi.rename(columns={"spiketrain_qi": "qi"})

elif "fullfield_qi" in qi.columns:
    qi = qi.rename(columns={"fullfield_qi": "qi"})

elif "qi" not in qi.columns:
    possible_qi_columns = [
        col
        for col in qi.columns
        if col not in ["cell_index", "stimulus_index", "index", "level_0"]
    ]

    if len(possible_qi_columns) == 1:
        qi = qi.rename(columns={possible_qi_columns[0]: "qi"})
    else:
        raise ValueError(
            "Could not automatically identify the full-field QI column. "
            f"The QI columns are: {list(qi.columns)}"
        )

qi["cell_index"] = qi["cell_index"].astype(int)
qi["stimulus_index"] = qi["stimulus_index"].astype(int)

# Keep only the useful QI columns
qi = qi[["cell_index", "stimulus_index", "qi"]]

print("Full-field QI table:")
print(qi.head())

# Convert spikes to pandas for plotting / merging

spikes_pd = spikes_pl.to_pandas()

spikes_pd["cell_index"] = spikes_pd["cell_index"].astype(int)
spikes_pd["stimulus_index"] = spikes_pd["stimulus_index"].astype(int)

# Join RF metadata and full-field QI onto the spike table

spikes_with_rf = spikes_pd.merge(
    rf_cell_table,
    on="cell_index",
    how="left",
    validate="many_to_one",
)

spikes_with_rf = spikes_with_rf.merge(
    qi,
    on=["cell_index", "stimulus_index"],
    how="left",
    validate="many_to_one",
)

print("New spikes table with RF metadata and full-field QI:")
print(spikes_with_rf.head())

print("New columns:")
print(spikes_with_rf.columns)

# # %% Optional: make a per-cell plotting table
#
# cell_table_for_plotting = (
#     spikes_with_rf[["cell_index", "stimulus_index", "rf_tilt", "rf_qi", "qi"]]
#     .drop_duplicates(subset=["cell_index", "stimulus_index"])
#     .sort_values(by="rf_qi", ascending=False)
# )
#
# print("Cell table for plotting/sorting:")
# print(cell_table_for_plotting.head(20))


# %%
def plot_raster_with_mean(
    spikes,
    t_start=0,
    t_end=32,
    bin_size=0.05,
    min_rf_qi=None,
    min_rf_tilt=None,
    max_rf_tilt=None,
    min_qi=None,
):
    """
    Plot full-field raster and mean PSTH.

    Designed to work with spikes_with_rf, which should contain:
        cell_index
        stimulus_index
        times_relative or times
        rf_tilt
        rf_qi
        qi

    Parameters
    ----------
    spikes : pandas.DataFrame
        Spike table. Use spikes_with_rf from the previous script.

    t_start : float
        Start time for plotting.

    t_end : float
        End time for plotting.

    bin_size : float
        PSTH bin size in seconds.

    min_rf_qi : float or None
        If given, only plot cells with rf_qi >= min_rf_qi.

    min_rf_tilt : float or None
        If given, only plot cells with rf_tilt >= min_rf_tilt.

    max_rf_tilt : float or None
        If given, only plot cells with rf_tilt <= max_rf_tilt.

    min_qi : float or None
        If given, only plot cells with full-field qi >= min_qi.

    Returns
    -------
    counts : pandas.DataFrame
        One row per cell, including ON/OFF metrics, polarity,
        rf_tilt, rf_qi, and qi.
    """

    # =================================================
    # COPY INPUT
    # =================================================

    df_all = spikes.copy()

    # =================================================
    # TIME COLUMN
    # =================================================

    if "times_relative" in df_all.columns:
        time_col = "times_relative"
    elif "times" in df_all.columns:
        time_col = "times"
    else:
        raise ValueError(
            "Could not find a spike time column. "
            "Expected either 'times_relative' or 'times'. "
            f"Available columns are: {list(df_all.columns)}"
        )

    # =================================================
    # CHECK REQUIRED COLUMNS
    # =================================================

    required_cols = ["cell_index", time_col]

    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(
                f"Missing required column: {col}. "
                f"Available columns are: {list(df_all.columns)}"
            )

    # =================================================
    # CELL METADATA TABLE
    # =================================================

    cell_metadata_cols = ["cell_index"]

    if "rf_tilt" in df_all.columns:
        cell_metadata_cols.append("rf_tilt")

    if "rf_qi" in df_all.columns:
        cell_metadata_cols.append("rf_qi")

    if "qi" in df_all.columns:
        cell_metadata_cols.append("qi")

    cell_metadata = (
        df_all[cell_metadata_cols]
        .drop_duplicates(subset=["cell_index"])
        .set_index("cell_index")
    )

    keep_cells = cell_metadata.index.to_numpy()

    # =================================================
    # OPTIONAL RF QI FILTER
    # =================================================

    if min_rf_qi is not None:
        if "rf_qi" not in cell_metadata.columns:
            raise ValueError(
                "min_rf_qi was given, but 'rf_qi' is not in the spikes table."
            )

        keep_cells = np.intersect1d(
            keep_cells,
            cell_metadata[cell_metadata["rf_qi"] >= min_rf_qi].index.to_numpy(),
        )

    # =================================================
    # OPTIONAL RF TILT FILTERS
    # =================================================

    if min_rf_tilt is not None:
        if "rf_tilt" not in cell_metadata.columns:
            raise ValueError(
                "min_rf_tilt was given, but 'rf_tilt' is not in the spikes table."
            )

        keep_cells = np.intersect1d(
            keep_cells,
            cell_metadata[cell_metadata["rf_tilt"] >= min_rf_tilt].index.to_numpy(),
        )

    if max_rf_tilt is not None:
        if "rf_tilt" not in cell_metadata.columns:
            raise ValueError(
                "max_rf_tilt was given, but 'rf_tilt' is not in the spikes table."
            )

        keep_cells = np.intersect1d(
            keep_cells,
            cell_metadata[cell_metadata["rf_tilt"] <= max_rf_tilt].index.to_numpy(),
        )

    # =================================================
    # OPTIONAL FULL-FIELD QI FILTER
    # =================================================

    if min_qi is not None:
        if "qi" not in cell_metadata.columns:
            raise ValueError("min_qi was given, but 'qi' is not in the spikes table.")

        keep_cells = np.intersect1d(
            keep_cells,
            cell_metadata[cell_metadata["qi"] >= min_qi].index.to_numpy(),
        )

    # =================================================
    # APPLY CELL FILTERS
    # =================================================

    df_all = df_all[df_all["cell_index"].isin(keep_cells)].copy()

    if df_all.empty:
        raise ValueError(
            "No cells left after filtering. "
            "Try relaxing min_rf_qi, min_rf_tilt, max_rf_tilt, or min_qi."
        )

    # =================================================
    # TIME FILTER
    # =================================================

    df = df_all[(df_all[time_col] >= t_start) & (df_all[time_col] <= t_end)].copy()

    if df.empty:
        raise ValueError(
            "No spikes left after time filtering. "
            "Try checking t_start/t_end or relaxing the cell filters."
        )

    # =================================================
    # ON / OFF BLOCK STRUCTURE
    # =================================================

    df["block"] = (df[time_col] // 2).astype(int)

    df["condition"] = np.where(df["block"] % 2 == 0, "ON", "OFF")

    # =================================================
    # POLARITY METRIC
    # =================================================

    counts = df.groupby(["cell_index", "condition"]).size().unstack(fill_value=0)

    if "ON" not in counts.columns:
        counts["ON"] = 0

    if "OFF" not in counts.columns:
        counts["OFF"] = 0

    counts["on_off_metric"] = (
        (counts["ON"] - counts["OFF"]) / (counts["ON"] + counts["OFF"] + 1e-9)
    ).fillna(0)

    # =================================================
    # CLASSIFY POLARITY
    # =================================================

    counts["polarity"] = "ON-OFF"

    counts.loc[counts["on_off_metric"] > 0.8, "polarity"] = "ON"
    counts.loc[counts["on_off_metric"] < -0.8, "polarity"] = "OFF"

    # =================================================
    # EARLY LIGHT-PHASE SPIKES
    # =================================================

    df["block_start"] = (df[time_col] // 2) * 2
    df["within_block"] = df[time_col] - df["block_start"]

    early_window = 1.5
    early_df = df[df["within_block"] < early_window]

    on_early = early_df[early_df["condition"] == "ON"].groupby("cell_index").size()

    off_early = early_df[early_df["condition"] == "OFF"].groupby("cell_index").size()

    counts["on_early"] = on_early.reindex(counts.index).fillna(0)
    counts["off_early"] = off_early.reindex(counts.index).fillna(0)

    # =================================================
    # EARLY RESPONSE METRIC FOR SORTING
    # =================================================

    counts["early_metric"] = 0.0

    counts.loc[counts["polarity"] == "ON", "early_metric"] = counts["on_early"]

    counts.loc[counts["polarity"] == "OFF", "early_metric"] = counts["off_early"]

    counts.loc[counts["polarity"] == "ON-OFF", "early_metric"] = (
        counts["on_early"] + counts["off_early"]
    )

    # =================================================
    # ADD RF / FULL-FIELD METADATA TO COUNTS
    # =================================================

    metadata_to_join = []

    if "rf_tilt" in df_all.columns:
        metadata_to_join.append("rf_tilt")

    if "rf_qi" in df_all.columns:
        metadata_to_join.append("rf_qi")

    if "qi" in df_all.columns:
        metadata_to_join.append("qi")

    if len(metadata_to_join) > 0:
        metadata = (
            df_all[["cell_index"] + metadata_to_join]
            .drop_duplicates(subset=["cell_index"])
            .set_index("cell_index")
        )

        counts = counts.join(metadata, how="left")

    # =================================================
    # POLARITY ORDER
    # =================================================

    category_order = {
        "ON-OFF": 0,
        "OFF": 1,
        "ON": 2,
    }

    counts["sort_order"] = counts["polarity"].map(category_order)

    # =================================================
    # FINAL SORT
    # =================================================
    # ON-OFF → OFF → ON
    # Within each group, strongest early response first

    counts = counts.sort_values(["sort_order", "early_metric"], ascending=[True, False])

    sorted_cells = counts.index.to_numpy()

    cell_to_y = {cell: len(sorted_cells) - 1 - i for i, cell in enumerate(sorted_cells)}

    # =================================================
    # PSTH
    # =================================================

    bins = np.arange(t_start, t_end + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts_hist, _ = np.histogram(df[time_col], bins=bins)

    n_cells = df["cell_index"].nunique()

    psth = counts_hist / (bin_size * n_cells)

    # =================================================
    # FIGURE
    # =================================================

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

    ax_psth = fig.add_subplot(gs[0])
    ax_raster = fig.add_subplot(gs[1], sharex=ax_psth)

    # =================================================
    # STIMULUS BARS
    # =================================================

    colors = [
        "#fd1717",
        "#000000",
        "#fe7c7c",
        "#000000",
        "#fafe7c",
        "#000000",
        "#dffc7e",
        "#000000",
        "#8afe7c",
        "#000000",
        "#7cfcfe",
        "#000000",
        "#7c86fe",
        "#000000",
        "#fe7cfe",
        "#000000",
    ]

    stim_labels = [
        "660",
        "",
        "610",
        "",
        "560",
        "",
        "535",
        "",
        "500",
        "",
        "460",
        "",
        "420",
        "",
        "365nm",
        "",
    ]

    for i, c in enumerate(colors):
        t0 = t_start + i * 2
        t1 = t0 + 2

        alpha = 0.15 if c != "#000000" else 0.08

        ax_raster.axvspan(t0, t1, color=c, alpha=alpha)

        ax_psth.axvspan(t0, t1, color=c, alpha=alpha)

    # =================================================
    # PSTH
    # =================================================

    ax_psth.plot(bin_centers, psth, color="black", linewidth=1.5)

    ax_psth.set_ylabel("Firing rate\n(spikes/s/cell)", fontsize=13)

    # ax_psth.set_yticks(np.arange(0, 151, 50))
    ax_psth.tick_params(axis="y", labelsize=13)

    ax_psth.set_xlim(t_start, t_end)
    ax_psth.set_xticks([])

    for i, label in enumerate(stim_labels):
        if label == "":
            continue

        t0 = t_start + i * 2
        t1 = t0 + 2

        ax_psth.text(
            (t0 + t1) / 2,
            1.08,
            label,
            transform=ax_psth.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=13,
        )

    # =================================================
    # RASTER
    # =================================================

    for _, row in df.iterrows():
        cell = row["cell_index"]

        if cell not in cell_to_y:
            continue

        y = cell_to_y[cell]

        ax_raster.vlines(
            row[time_col],
            y - 0.4,
            y + 0.4,
            color="black",
            linewidth=0.5,
        )

    ax_raster.set_xlim(t_start, t_end)
    ax_raster.set_ylim(-0.5, len(sorted_cells) - 0.5)

    ax_raster.set_xlabel("Time (s)", fontsize=13)
    ax_raster.set_xticks(np.arange(2, 34, 2))
    ax_raster.tick_params(axis="x", labelsize=13)

    ax_raster.set_ylabel("Cell", fontsize=13)
    ax_raster.set_yticks([])

    ax_psth.tick_params(axis="x", bottom=False, labelbottom=False)

    # =================================================
    # SCALE BAR
    # =================================================

    scale_cells = 10

    if len(sorted_cells) >= scale_cells:
        x = -0.01
        y0 = 0.02

        y1 = y0 + (scale_cells / len(sorted_cells))

        ax_raster.plot(
            [x, x],
            [y0, y1],
            transform=ax_raster.transAxes,
            color="black",
            linewidth=2,
            clip_on=False,
        )

        ax_raster.text(
            x - 0.005,
            (y0 + y1) / 2,
            "10",
            transform=ax_raster.transAxes,
            va="center",
            ha="right",
            fontsize=13,
            clip_on=False,
        )

    # =================================================
    # RIGHT-SIDE POLARITY LABELS
    # =================================================

    groups = [
        ("ON-OFF", "black"),
        ("OFF", "blue"),
        ("ON", "red"),
    ]

    x_line = 1.01
    y_top = 1.0

    for label, color in groups:
        n_group = np.sum(counts["polarity"] == label)

        if n_group == 0:
            continue

        frac = n_group / len(counts)
        y_bottom = y_top - frac

        ax_raster.plot(
            [x_line, x_line],
            [y_bottom, y_top],
            transform=ax_raster.transAxes,
            color=color,
            linewidth=3,
            clip_on=False,
        )

        ax_raster.text(
            x_line + 0.005,
            (y_bottom + y_top) / 2,
            label,
            fontsize=13,
            transform=ax_raster.transAxes,
            va="center",
        )

        y_top = y_bottom

    # =================================================
    # TITLE WITH FILTER INFORMATION
    # =================================================

    filter_text = []

    if min_rf_qi is not None:
        filter_text.append(f"RF QI ≥ {min_rf_qi}")

    if min_rf_tilt is not None:
        filter_text.append(f"RF tilt ≥ {min_rf_tilt}")

    if max_rf_tilt is not None:
        filter_text.append(f"RF tilt ≤ {max_rf_tilt}")

    if min_qi is not None:
        filter_text.append(f"FF QI ≥ {min_qi}")

    filter_text.append(f"n = {len(sorted_cells)} cells")

    ax_psth.set_title(", ".join(filter_text), fontsize=13, pad=40)

    # =================================================
    # FINAL
    # =================================================

    plt.tight_layout()
    plt.show()

    return counts


# %%
plot = plot_raster_with_mean(
    spikes_with_rf,
    min_rf_qi=0,
    min_qi=0.0,
    min_rf_tilt=0.801,
    max_rf_tilt=1,
)
# %%
