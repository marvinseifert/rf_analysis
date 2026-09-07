# %% Imports

import sys
from pathlib import Path

sys.path.extend(["/home/mawa/PycharmProjects/polarspike"])

from polarspike import Overview, quality_tests

import polars as pl
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# %% User settings

phase_path = Path(r"F:\Laura\zebrafish_22_07_2026\Phase_00")

rf_dataset_path = phase_path / "noise_analysis" / "noise_data.nc"
spikes_path = phase_path / "alldata.parquet"
overview_path = phase_path / "overview"

chirp_stimulus_index = 2

cell_thresholds_path = Path(
    r"C:\Users\Laura Steel\PycharmProjects\polarspike\cell_thresholds.npy"
)

# Chirp parameters
start_freq = 1
end_freq = 30
chirp_duration = 30
refresh_rate = 300
total_length_sec = 35
chirp_offset_sec = 3

# Plot/filter parameters
min_rf_qi = None
min_rf_tilt = None
max_rf_tilt = None
min_qi = None
min_rate = None

t_start = 0
t_end = 33
bin_size = 0.05


# %% Generate chirp stimulus trace


def generate_chirp_data(
    start_freq,
    end_freq,
    duration,
    refresh_rate,
    max_power=4095,
):
    """
    Generate exponential frequency chirp.

    Returns
    -------
    t : np.ndarray
        Time axis for the active chirp period.

    chirp_signal : np.ndarray
        PWM/brightness trace.

    freqs : np.ndarray
        Instantaneous frequency at each time point.
    """

    num_cycles = int(duration * refresh_rate)

    beta = np.log(end_freq / start_freq) / duration

    t = np.linspace(
        0,
        duration,
        num_cycles,
        endpoint=False,
    )

    exp_bt = np.exp(beta * t)

    freqs = start_freq * exp_bt

    phase = 2 * np.pi * (start_freq / beta) * (exp_bt - 1)

    raw_sine = np.sin(phase)

    chirp_signal = (raw_sine + 1.0) * max_power / 2.0
    chirp_signal = chirp_signal.astype(int)

    return t, chirp_signal, freqs


def make_padded_chirp_arrays(
    start_freq=1,
    end_freq=30,
    chirp_duration=30,
    refresh_rate=300,
    total_length_sec=35,
    chirp_offset_sec=3,
    max_power=4095,
):
    """
    Make full-length chirp arrays with padding before the chirp starts.
    """

    time, chirp_real, freq_actual = generate_chirp_data(
        start_freq=start_freq,
        end_freq=end_freq,
        duration=chirp_duration,
        refresh_rate=refresh_rate,
        max_power=max_power,
    )

    total_samples = int(total_length_sec * refresh_rate) - 1

    full_time_axis = np.arange(
        0,
        total_length_sec - 1 / refresh_rate,
        1 / refresh_rate,
    )

    chirp_complete = np.zeros(total_samples)
    c_freqs = np.zeros(total_samples)

    offset = int(refresh_rate * chirp_offset_sec)
    end_idx = offset + len(chirp_real)

    limit = min(len(chirp_complete), end_idx)
    insert_len = limit - offset

    if insert_len > 0:
        chirp_complete[offset:limit] = chirp_real[:insert_len]
        c_freqs[offset:limit] = freq_actual[:insert_len]

    return full_time_axis, chirp_complete, c_freqs


full_time_axis, chirp_complete, c_freqs = make_padded_chirp_arrays(
    start_freq=start_freq,
    end_freq=end_freq,
    chirp_duration=chirp_duration,
    refresh_rate=refresh_rate,
    total_length_sec=total_length_sec,
    chirp_offset_sec=chirp_offset_sec,
)


# %% Helper functions for loading and cleaning spike/QI data


def ensure_polars_int64(spikes):
    """
    Convert spike table to Polars and cast integer columns to Int64.

    This avoids the Polars/Python UDF error:
    Int32 is incompatible with expected Int64.
    """

    if isinstance(spikes, pl.LazyFrame):
        spikes_pl = spikes.collect()

    elif isinstance(spikes, pl.DataFrame):
        spikes_pl = spikes

    elif isinstance(spikes, pd.DataFrame):
        spikes_pl = pl.from_pandas(spikes)

    else:
        spikes_pl = pl.DataFrame(spikes)

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

    if len(integer_columns) > 0:
        spikes_pl = spikes_pl.with_columns(
            [pl.col(col).cast(pl.Int64) for col in integer_columns]
        )

    if "cell_index" in spikes_pl.columns:
        spikes_pl = spikes_pl.with_columns(pl.col("cell_index").cast(pl.Int64))

    if "stimulus_index" in spikes_pl.columns:
        spikes_pl = spikes_pl.with_columns(pl.col("stimulus_index").cast(pl.Int64))

    return spikes_pl


def clean_qi_output(qi_raw, stimulus_index):
    """
    Convert quality_tests.spiketrain_qi output into a dataframe with:
        cell_index
        stimulus_index
        qi
    """

    if isinstance(qi_raw, pl.DataFrame):
        qi = qi_raw.to_pandas()

    elif isinstance(qi_raw, pl.Series):
        qi = qi_raw.to_pandas().rename("qi").reset_index()

    elif isinstance(qi_raw, pd.Series):
        qi = qi_raw.rename("qi").reset_index()

    elif isinstance(qi_raw, pd.DataFrame):
        qi = qi_raw.copy()

    else:
        qi = pd.DataFrame(qi_raw)

    if "cell_index" not in qi.columns:
        qi = qi.reset_index()

    if "cell_index" not in qi.columns and "index" in qi.columns:
        qi = qi.rename(columns={"index": "cell_index"})

    if "cell_index" not in qi.columns and "level_0" in qi.columns:
        qi = qi.rename(columns={"level_0": "cell_index"})

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
                "Could not automatically identify the QI column. "
                f"QI columns are: {list(qi.columns)}"
            )

    qi["cell_index"] = qi["cell_index"].astype(int)
    qi["stimulus_index"] = int(stimulus_index)

    qi = qi[["cell_index", "stimulus_index", "qi"]]

    return qi


# %% Load RF metadata safely

dataset = xr.load_dataset(rf_dataset_path)

rf_cell_table_raw = (
    dataset[["tilt", "quality"]]
    .to_dataframe()
    .reset_index()[["cell_index", "channel", "tilt", "quality"]]
)


def first_non_nan(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return x.iloc[0]


rf_cell_table = rf_cell_table_raw.groupby("cell_index", as_index=False).agg(
    rf_tilt=("tilt", first_non_nan),
    rf_qi=("quality", "max"),
)

rf_cell_table["cell_index"] = rf_cell_table["cell_index"].astype(int)

print("RF cell table:")
print(rf_cell_table.head(30))

print("Cells with non-NaN RF tilt:")
print(rf_cell_table["rf_tilt"].notna().sum())

print("Cells with non-NaN RF QI:")
print(rf_cell_table["rf_qi"].notna().sum())

print("RF tilt summary:")
print(rf_cell_table["rf_tilt"].describe())

# %% Load chirp spikes

spikes_on_disk = pl.scan_parquet(spikes_path)

recording = Overview.Recording.load(overview_path)

spikes_chirp = recording.get_spikes_triggered(
    [{"stimulus_index": [chirp_stimulus_index]}]
)

spikes_chirp_pl = ensure_polars_int64(spikes_chirp)

print("Chirp spikes schema:")
print(spikes_chirp_pl.schema)

print("Chirp spikes:")
print(spikes_chirp_pl.head())


# %% Calculate chirp response QI

qi_raw = quality_tests.spiketrain_qi(
    spikes_chirp_pl,
    max_window=2 * 16,
    max_repeat=5,
)

qi = clean_qi_output(
    qi_raw=qi_raw,
    stimulus_index=chirp_stimulus_index,
)

print("Chirp QI table:")
print(qi.head())


# %% Convert chirp spikes to pandas and join RF metadata + chirp QI

spikes_chirp_pd = spikes_chirp_pl.to_pandas()

spikes_chirp_pd["cell_index"] = spikes_chirp_pd["cell_index"].astype(int)
spikes_chirp_pd["stimulus_index"] = spikes_chirp_pd["stimulus_index"].astype(int)

spikes_chirp_with_rf = spikes_chirp_pd.merge(
    rf_cell_table,
    on="cell_index",
    how="left",
    validate="many_to_one",
)

spikes_chirp_with_rf = spikes_chirp_with_rf.merge(
    qi,
    on=["cell_index", "stimulus_index"],
    how="left",
    validate="many_to_one",
)

print("Chirp spikes with RF metadata:")
print(spikes_chirp_with_rf.head())

print("Columns:")
print(spikes_chirp_with_rf.columns)


# %% Optional: add threshold-frequency data

if cell_thresholds_path.exists():
    cell_thresholds = np.load(cell_thresholds_path)

    cell_thresholds = pd.DataFrame(
        cell_thresholds,
        columns=["cell_index", "threshold_frequency"],
    )

    cell_thresholds["cell_index"] = cell_thresholds["cell_index"].astype(int)

    # Optional exclusion from your old script
    cell_thresholds = cell_thresholds[cell_thresholds["cell_index"] != 199]

    spikes_chirp_with_rf = spikes_chirp_with_rf.merge(
        cell_thresholds,
        on="cell_index",
        how="left",
        validate="many_to_one",
    )

    print("Added threshold_frequency.")
    print(cell_thresholds.head())

else:
    print("No cell_thresholds file found. Skipping threshold_frequency merge.")


# %% Plotting function
def plot_raster_with_mean_chirp_rf(
    spikes,
    full_time_axis,
    pwm_trace,
    frequency_trace=None,
    t_start=0,
    t_end=33,
    bin_size=0.05,
    min_rf_qi=None,
    min_rf_tilt=None,
    max_rf_tilt=None,
    min_qi=None,
    min_rate=None,
    sort_by="firing_rate",
    sort_ascending=True,
    chirp_offset_sec=3,
    single_cells=None,  # kept only so old function calls do not break
):
    """
    Plot chirp responses for cells selected using RF metadata.

    This version plots only:
        1. Chirp intensity trace
        2. Population PSTH
        3. Population raster

    It does NOT plot individual single-cell rasters/PSTHs below.

    Parameters
    ----------
    spikes : pandas.DataFrame
        Spike table with chirp spikes plus rf_tilt, rf_qi, and qi.

    full_time_axis : np.ndarray
        Time axis for the padded chirp stimulus.

    pwm_trace : np.ndarray
        Padded chirp intensity trace.

    frequency_trace : np.ndarray or None
        Optional padded instantaneous frequency trace.

    t_start, t_end : float
        Time window to plot.

    bin_size : float
        PSTH bin size in seconds.

    min_rf_qi : float or None
        Keep cells with rf_qi >= min_rf_qi.

    min_rf_tilt : float or None
        Keep cells with rf_tilt >= min_rf_tilt.

    max_rf_tilt : float or None
        Keep cells with rf_tilt <= max_rf_tilt.

    min_qi : float or None
        Keep cells with chirp response qi >= min_qi.

    min_rate : float or None
        Keep cells with firing rate >= min_rate.

    sort_by : str
        Column used to sort population raster.
        Useful options:
            "firing_rate"
            "threshold_frequency"
            "rf_tilt"
            "rf_qi"
            "qi"

    sort_ascending : bool
        Whether to sort the population raster in ascending order.

    chirp_offset_sec : float
        Time when the active chirp starts, used for x-axis labels.

    single_cells : ignored
        Kept only so old calls with single_cells=None do not error.

    Returns
    -------
    df : pandas.DataFrame
        Filtered spike dataframe used for plotting.

    cell_summary : pandas.DataFrame
        One row per plotted cell with metadata and firing rate.
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
            "Expected 'times_relative' or 'times'. "
            f"Available columns are: {list(df_all.columns)}"
        )

    # =================================================
    # REPEAT COLUMN
    # =================================================

    if "repeat" not in df_all.columns:
        print("Warning: repeat column missing. Creating a single repeat called 0.")
        df_all["repeat"] = 0

    df_all = df_all.dropna(subset=["repeat"]).copy()
    df_all["repeat"] = df_all["repeat"].astype(int)

    # =================================================
    # TIME FILTER
    # =================================================

    df_time = df_all[(df_all[time_col] >= t_start) & (df_all[time_col] <= t_end)].copy()

    if df_time.empty:
        raise ValueError("No spikes found in the selected time window.")

    # =================================================
    # CELL SUMMARY TABLE
    # =================================================

    metadata_cols = ["cell_index"]

    for col in ["rf_tilt", "rf_qi", "qi", "threshold_frequency"]:
        if col in df_time.columns:
            metadata_cols.append(col)

    cell_summary = (
        df_time[metadata_cols]
        .drop_duplicates(subset=["cell_index"])
        .set_index("cell_index")
    )

    duration = t_end - t_start

    spike_counts = df_time.groupby("cell_index").size()
    firing_rate = spike_counts / duration

    cell_summary["spike_count"] = spike_counts.reindex(cell_summary.index).fillna(0)
    cell_summary["firing_rate"] = firing_rate.reindex(cell_summary.index).fillna(0)

    keep_cells = cell_summary.index.to_numpy()

    # =================================================
    # APPLY RF QI FILTER
    # =================================================

    if min_rf_qi is not None:
        if "rf_qi" not in cell_summary.columns:
            raise ValueError("min_rf_qi was given, but rf_qi is missing.")

        keep_cells = np.intersect1d(
            keep_cells,
            cell_summary[cell_summary["rf_qi"] >= min_rf_qi].index.to_numpy(),
        )

    # =================================================
    # APPLY RF TILT FILTERS
    # =================================================

    if min_rf_tilt is not None:
        if "rf_tilt" not in cell_summary.columns:
            raise ValueError("min_rf_tilt was given, but rf_tilt is missing.")

        keep_cells = np.intersect1d(
            keep_cells,
            cell_summary[cell_summary["rf_tilt"] >= min_rf_tilt].index.to_numpy(),
        )

    if max_rf_tilt is not None:
        if "rf_tilt" not in cell_summary.columns:
            raise ValueError("max_rf_tilt was given, but rf_tilt is missing.")

        keep_cells = np.intersect1d(
            keep_cells,
            cell_summary[cell_summary["rf_tilt"] <= max_rf_tilt].index.to_numpy(),
        )

    # =================================================
    # APPLY CHIRP QI FILTER
    # =================================================

    if min_qi is not None:
        if "qi" not in cell_summary.columns:
            raise ValueError("min_qi was given, but qi is missing.")

        keep_cells = np.intersect1d(
            keep_cells,
            cell_summary[cell_summary["qi"] >= min_qi].index.to_numpy(),
        )

    # =================================================
    # APPLY FIRING RATE FILTER
    # =================================================

    if min_rate is not None:
        keep_cells = np.intersect1d(
            keep_cells,
            cell_summary[cell_summary["firing_rate"] >= min_rate].index.to_numpy(),
        )

    # =================================================
    # FINAL FILTERED DATA
    # =================================================

    df = df_time[df_time["cell_index"].isin(keep_cells)].copy()
    cell_summary = cell_summary.loc[keep_cells].copy()

    if df.empty:
        raise ValueError(
            "No cells/spikes left after filtering. "
            "Try relaxing min_rf_qi, min_rf_tilt, max_rf_tilt, min_qi, or min_rate."
        )

    # =================================================
    # SORT CELLS
    # =================================================

    if sort_by not in cell_summary.columns:
        print(f"sort_by='{sort_by}' not found. " "Using sort_by='firing_rate' instead.")
        sort_by = "firing_rate"

    cell_summary = cell_summary.sort_values(
        by=sort_by,
        ascending=sort_ascending,
    )

    sorted_cells = cell_summary.index.to_numpy()

    cell_to_y = {cell: len(sorted_cells) - 1 - i for i, cell in enumerate(sorted_cells)}

    # =================================================
    # POPULATION PSTH
    # =================================================

    bins = np.arange(t_start, t_end + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts_hist, _ = np.histogram(df[time_col], bins=bins)

    n_cells = df["cell_index"].nunique()

    psth = counts_hist / (bin_size * n_cells)

    # =================================================
    # FIGURE: CHIRP TRACE + PSTH + RASTER
    # =================================================

    fig = plt.figure(figsize=(16, 8))

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1, 1, 5],
        hspace=0.08,
    )

    ax_chirp = fig.add_subplot(gs[0])
    ax_psth = fig.add_subplot(gs[1], sharex=ax_chirp)
    ax_raster = fig.add_subplot(gs[2], sharex=ax_chirp)

    all_axes = [
        ax_chirp,
        ax_psth,
        ax_raster,
    ]

    for ax in all_axes:
        ax.set_facecolor("#e6e6e6")

    # =================================================
    # CHIRP INTENSITY TRACE
    # =================================================

    stim_mask = (full_time_axis >= t_start) & (full_time_axis <= t_end)

    ax_chirp.plot(
        full_time_axis[stim_mask],
        pwm_trace[stim_mask],
        color="darkgreen",
        linewidth=1.0,
    )

    ax_chirp.set_ylabel(
        "Chirp\nintensity",
        fontsize=13,
    )

    ax_chirp.set_ylim(0, np.max(pwm_trace))
    ax_chirp.set_yticks([0, np.max(pwm_trace)])
    ax_chirp.set_yticklabels([0, 1])
    ax_chirp.tick_params(axis="y", labelsize=13)
    ax_chirp.tick_params(axis="x", bottom=False, labelbottom=False)

    # Optional frequency trace on the right y-axis
    if frequency_trace is not None:
        ax_freq = ax_chirp.twinx()

        ax_freq.plot(
            full_time_axis[stim_mask],
            frequency_trace[stim_mask],
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

        ax_freq.set_ylabel(
            "Freq\n(Hz)",
            fontsize=11,
        )

        ax_freq.tick_params(axis="y", labelsize=11)

    # =================================================
    # POPULATION PSTH
    # =================================================

    ax_psth.plot(
        bin_centers,
        psth,
        color="black",
        linewidth=1.5,
    )

    ax_psth.set_ylabel(
        "Firing rate\n(spikes/s/cell)",
        fontsize=13,
    )

    ax_psth.set_xlim(t_start, t_end)
    ax_psth.tick_params(axis="y", labelsize=13)
    ax_psth.tick_params(axis="x", bottom=False, labelbottom=False)

    # =================================================
    # POPULATION RASTER
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

    ax_raster.set_ylabel("Cell", fontsize=13, labelpad=10)
    ax_raster.set_xlim(t_start, t_end)
    ax_raster.set_ylim(-0.5, len(sorted_cells) - 0.5)
    ax_raster.set_yticks([])

    # X-axis labels: show time from chirp start
    xticks = np.arange(
        chirp_offset_sec,
        chirp_offset_sec + 30 + 0.1,
        5,
    )

    xticklabels = [str(int(x - chirp_offset_sec)) for x in xticks]

    ax_raster.set_xticks(xticks)
    ax_raster.set_xticklabels(xticklabels)
    ax_raster.set_xlabel("Time from chirp start (s)", fontsize=13)
    ax_raster.tick_params(axis="x", labelsize=13)

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
        filter_text.append(f"chirp QI ≥ {min_qi}")

    if min_rate is not None:
        filter_text.append(f"FR ≥ {min_rate} spikes/s")

    filter_text.append(f"n = {len(sorted_cells)} cells")
    filter_text.append(f"sorted by {sort_by}")

    ax_chirp.set_title(
        ", ".join(filter_text),
        fontsize=13,
        pad=35,
    )

    # =================================================
    # BORDERS
    # =================================================

    for ax in all_axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.2)

    # =================================================
    # FINAL
    # =================================================

    plt.tight_layout()
    plt.show()

    return df, cell_summary


# %% Run plot
chirp_plot_df, chirp_cell_summary = plot_raster_with_mean_chirp_rf(
    spikes=spikes_chirp_with_rf,
    full_time_axis=full_time_axis,
    pwm_trace=chirp_complete,
    frequency_trace=c_freqs,
    t_start=t_start,
    t_end=t_end,
    bin_size=bin_size,
    min_rf_qi=0,
    min_rf_tilt=None,
    max_rf_tilt=0.8,
    min_qi=None,
    min_rate=None,
    sort_by="firing_rate",
    sort_ascending=True,
    chirp_offset_sec=chirp_offset_sec,
)
