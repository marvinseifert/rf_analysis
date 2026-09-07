from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl
from polarspike import Overview


def to_pandas(frame: Any) -> pd.DataFrame:
    """Convert Polarspike/polars/pandas output to an independent pandas DataFrame."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, pl.LazyFrame):
        return frame.collect().to_pandas()
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str:
    columns = set(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(
        f"None of these columns were found: {tuple(candidates)}. "
        f"Available columns: {sorted(columns)}"
    )


def normalise_repeat_labels(
    repeats: np.ndarray,
    expected_repeats: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Convert zero-based, one-based, or arbitrary repeat labels to 0..n-1."""
    unique_values = np.sort(np.unique(np.asarray(repeats, dtype=int)))

    if len(unique_values) == 0:
        mapping: dict[int, int] = {}
    elif unique_values.min() == 0 and unique_values.max() == expected_repeats - 1:
        mapping = {int(x): int(x) for x in unique_values}
    elif unique_values.min() == 1 and unique_values.max() == expected_repeats:
        mapping = {int(x): int(x) - 1 for x in unique_values}
    else:
        mapping = {int(x): i for i, x in enumerate(unique_values)}

    indices = np.array([mapping.get(int(x), -1) for x in repeats], dtype=int)
    return indices, mapping


def stimulus_cell_indices(recording: Any, stimulus_index: int) -> np.ndarray:
    """Return sorted cell IDs that have spikes assigned to one stimulus."""
    spikes_df = to_pandas(recording.spikes_df)

    if "stimulus_index" not in spikes_df.columns or "cell_index" not in spikes_df.columns:
        raise KeyError(
            "recording.spikes_df must contain 'stimulus_index' and 'cell_index'."
        )

    cells = (
        spikes_df.loc[
            spikes_df["stimulus_index"] == int(stimulus_index),
            "cell_index",
        ]
        .dropna()
        .astype(int)
        .unique()
    )
    return np.sort(cells)


def all_cell_indices(overview_path) -> np.ndarray:
    """Return every cell_index present in the recording overview."""
    recording = Overview.Recording.load(overview_path)
    spikes_df = to_pandas(recording.spikes_df)
    if "cell_index" not in spikes_df.columns:
        raise KeyError("recording.spikes_df has no 'cell_index' column.")
    return np.sort(spikes_df["cell_index"].dropna().astype(int).unique())


def get_cell_stimulus_spikes(
    recording: Any,
    cell_index: int,
    stimulus_index: int,
) -> pd.DataFrame:
    """Use the same Polarspike triggered-spike route as the supplied plotting scripts."""
    condition = {
        "stimulus_index": [int(stimulus_index)],
        "cell_index": [int(cell_index)],
    }
    return to_pandas(recording.get_spikes_triggered([condition]))


def repeated_stimulus_psth(
    spikes_df: pd.DataFrame,
    duration_s: float,
    expected_repeats: int,
    bin_size_s: float,
    time_col_candidates: Sequence[str] = (
        "times_relative",
        "times",
        "times_triggered",
        "time",
    ),
    repeat_col_candidates: Sequence[str] = ("repeat", "repeats", "trial"),
) -> dict[str, Any]:
    """
    Recreate the repeated-stimulus raster/PSTH logic used by the supplied
    SCF/csteps summary script, but return only analysis arrays.
    """
    bins = np.arange(0.0, duration_s + bin_size_s, bin_size_s, dtype=float)
    if bins.size < 2:
        bins = np.array([0.0, duration_s], dtype=float)
    elif bins[-1] < duration_s:
        bins = np.append(bins, duration_s)
    else:
        bins[-1] = duration_s

    bins = np.unique(bins)
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2.0

    empty_raster = [np.array([], dtype=float) for _ in range(expected_repeats)]

    if spikes_df.empty:
        return {
            "raster_by_repeat": empty_raster,
            "bins": bins,
            "bin_centers": bin_centers,
            "bin_widths": bin_widths,
            "psth": np.zeros(len(bin_centers), dtype=float),
            "n_spikes": 0,
        }

    time_col = first_existing(spikes_df.columns, time_col_candidates)
    repeat_col = next(
        (c for c in repeat_col_candidates if c in spikes_df.columns),
        None,
    )

    times = pd.to_numeric(spikes_df[time_col], errors="coerce").to_numpy(dtype=float)

    if repeat_col is not None:
        repeats_raw = pd.to_numeric(
            spikes_df[repeat_col], errors="coerce"
        ).to_numpy(dtype=float)
        finite_repeat = np.isfinite(repeats_raw)
        times = times[finite_repeat]
        repeats_raw = np.rint(repeats_raw[finite_repeat]).astype(int)
        repeat_idx, _ = normalise_repeat_labels(repeats_raw, expected_repeats)
        x_times = times.copy()
        finite_times = x_times[np.isfinite(x_times)]
        if finite_times.size and np.nanmax(finite_times) > duration_s + 1e-9:
            x_times = np.mod(x_times, duration_s)
    else:
        finite_times = times[np.isfinite(times)]
        if finite_times.size and np.nanmax(finite_times) > duration_s + 1e-9:
            repeat_idx = np.floor(times / duration_s).astype(int)
            x_times = np.mod(times, duration_s)
        else:
            repeat_idx = np.zeros(len(times), dtype=int)
            x_times = times.copy()

    valid = (
        np.isfinite(x_times)
        & (x_times >= 0.0)
        & (x_times <= duration_s)
        & (repeat_idx >= 0)
        & (repeat_idx < expected_repeats)
    )
    x_times = x_times[valid]
    repeat_idx = repeat_idx[valid]

    raster = [
        np.asarray(x_times[repeat_idx == rep], dtype=float)
        for rep in range(expected_repeats)
    ]

    counts, _ = np.histogram(x_times, bins=bins)
    psth = counts / (bin_widths * max(1, expected_repeats))

    return {
        "raster_by_repeat": raster,
        "bins": bins,
        "bin_centers": bin_centers,
        "bin_widths": bin_widths,
        "psth": psth,
        "n_spikes": int(len(x_times)),
    }


def box_smooth(values: np.ndarray, bin_size_s: float, window_s: float) -> np.ndarray:
    """Simple centred box smoothing used for transient/sustained peak estimates."""
    values = np.asarray(values, dtype=float)
    n = max(1, int(round(window_s / bin_size_s)))
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(values, kernel, mode="same")


def mean_in_window(
    times: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    mask = (times >= start_s) & (times < end_s)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(values[mask]))


def peak_in_window(
    times: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    mask = (times >= start_s) & (times < end_s)
    if not np.any(mask):
        return np.nan
    window = np.asarray(values[mask], dtype=float)
    if window.size == 0 or np.all(~np.isfinite(window)):
        return np.nan
    return float(np.nanmax(window))


def contrast_index(a: float, b: float) -> float:
    """Return (a-b)/(a+b), or NaN when the denominator is zero/invalid."""
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    denominator = a + b
    if denominator == 0:
        return np.nan
    return float((a - b) / denominator)
