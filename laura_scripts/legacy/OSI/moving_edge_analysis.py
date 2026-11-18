from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from polarspike import Overview


"""
Analyse 1-4 moving-bar stimuli in compact per-cell summary figures.

Main features
-------------
- Accept any subset of:
    ON  13 fps
    OFF 13 fps
    ON  60 fps
    OFF 60 fps
- Plot only the conditions you provide.
- Preserve the supplied order.
- Use exact direction timing metadata read from each H5 file.
- Use the projector/projection transformed direction mapping:

    left_to_right              -> up
    right_to_left              -> down
    bottom_to_top              -> right
    top_to_bottom              -> left
    bottom_left_to_top_right   -> up-right
    bottom_right_to_top_left   -> down-right
    top_left_to_bottom_right   -> up-left
    top_right_to_bottom_left   -> down-left

Figure layout
-------------
- 1 condition  -> 1 x 1
- 2 conditions -> 1 x 2
- 3 conditions -> 2 x 2, last slot unused
- 4 conditions -> 2 x 2

Each condition panel contains:
- a central polar tuning plot
- one repeat raster for each of the eight directions
- one PSTH above each raster
- panel title with DSI / OSI / RF quality / tilt

Important
---------
This script reads exact H5 timing metadata:
- Directions
- Direction_Start_Frame
- Direction_Frame_Count
- Frame_Rate

It therefore does not assume equal direction durations.
"""


# =============================================================================
# Direction definitions
# =============================================================================

# Plotting convention:
# 0 degrees = upward motion; angles increase clockwise.
DIRECTION_NAMES = np.array(
    [
        "right",
        "left",
        "up",
        "down",
        "up-right",
        "up-left",
        "down-right",
        "down-left",
    ]
)

DIRECTIONS_DEG = np.array(
    [90, 270, 0, 180, 45, 315, 135, 225],
    dtype=float,
)

# KEEPING YOUR PROJECTOR / PROJECTION TRANSFORMATION
H5_TO_PLOT_DIRECTION = {
    "left_to_right": "up",
    "right_to_left": "down",
    "bottom_to_top": "right",
    "top_to_bottom": "left",
    "bottom_left_to_top_right": "up-right",
    "bottom_right_to_top_left": "down-right",
    "top_left_to_bottom_right": "up-left",
    "top_right_to_bottom_left": "down-left",
}


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(frozen=True)
class MovingBarConditionConfig:
    """
    Recording and stimulus information for one moving-bar condition.
    """

    title: str
    stimulus_index: int
    h5_path: Path
    expected_repeats: int

    # Optional metadata checks
    expected_polarity: str | None = None
    expected_frame_rate_hz: float | None = None

    # Polarspike spike-table columns
    trigger_col: str = "trigger"
    repeat_col: str = "repeat"
    time_col: str = "times_triggered"

    # times_triggered can be:
    # - "seconds"
    # - "samples"
    # - "auto"
    time_unit: str = "seconds"
    sample_rate_hz: float = 20_000.0

    # Set to 1 only if trigger frames are one-based in the stored table
    trigger_frame_offset: int = 0

    # PSTH bin width in seconds
    bin_size_s: float = 0.25


@dataclass
class PlotStyleConfig:
    """
    Shared analysis and layout settings.
    """

    response_metric: str = "peak_response"  # "peak_response" or "mean_rate"

    direction_names: np.ndarray = field(default_factory=lambda: DIRECTION_NAMES.copy())
    directions_deg: np.ndarray = field(default_factory=lambda: DIRECTIONS_DEG.copy())

    raster_facecolor: str = "0.90"
    psth_facecolor: str = "white"

    show_peak_direction_line: bool = True
    show_orientation_axis: bool = True
    show_motion_direction_arrows: bool = True

    polar_arrow_radius: float = 0.60
    polar_arrow_length: float = 0.075
    polar_arrow_linewidth: float = 1.4
    polar_arrow_mutation_scale: float = 14

    raster_box_width: float = 0.16
    raster_box_height: float = 0.10
    raster_gap_from_arrow: float = 0.035
    psth_box_height: float = 0.045
    psth_gap: float = 0.006

    show_psth: bool = True
    show_psth_scale: bool = True

    # If True, use one shared radial scale across supplied conditions
    share_polar_scale_across_conditions: bool = False


@dataclass
class PlotConfig:
    recording_name: str
    overview_path: Path
    noise_data_path: Path
    output_dir: Path

    # Supply any 1-4 conditions in the order you want them plotted
    conditions: tuple[MovingBarConditionConfig, ...]

    style: PlotStyleConfig = field(default_factory=PlotStyleConfig)

    # RF variables are used only for filtering/ranking/title metadata
    rf_quality_variable: str = "quality"
    rf_tilt_variable: str = "tilt"
    rf_reference_channel: Any | None = None
    transform_tilt_reciprocal: bool = False

    selected_cells: Sequence[int] | None = None
    min_rf_quality: float | None = None
    min_tilt: float | None = None
    max_tilt: float | None = None
    max_cells: int | None = None

    save: bool = True
    show: bool = True
    figure_dpi: int = 160
    save_dpi: int = 300
    image_format: str = "png"
    figure_size: tuple[float, float] = (20.0, 20.0)


@dataclass(frozen=True)
class StimulusTiming:
    """
    Exact direction boundaries read from a generated H5 stimulus.
    """

    h5_direction_names: tuple[str, ...]
    plot_direction_names: tuple[str, ...]
    direction_start_frames: np.ndarray
    direction_frame_counts: np.ndarray
    frame_rate_hz: float
    total_frames: int
    polarity: str

    @property
    def direction_end_frames(self) -> np.ndarray:
        return self.direction_start_frames + self.direction_frame_counts

    @property
    def direction_start_s(self) -> np.ndarray:
        return self.direction_start_frames / self.frame_rate_hz

    @property
    def direction_duration_s(self) -> np.ndarray:
        return self.direction_frame_counts / self.frame_rate_hz

    @property
    def repeat_duration_s(self) -> float:
        return self.total_frames / self.frame_rate_hz


# =============================================================================
# General helpers
# =============================================================================


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, pl.LazyFrame):
        return frame.collect().to_pandas()
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _normalise_repeat_labels(
    repeats: np.ndarray,
    expected_repeats: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Convert zero-based, one-based, or arbitrary repeat labels to 0..n-1.
    """
    unique_values = np.sort(np.unique(repeats.astype(int)))

    if len(unique_values) == 0:
        mapping: dict[int, int] = {}
    elif unique_values.min() == 0 and unique_values.max() == expected_repeats - 1:
        mapping = {int(x): int(x) for x in unique_values}
    elif unique_values.min() == 1 and unique_values.max() == expected_repeats:
        mapping = {int(x): int(x) - 1 for x in unique_values}
    else:
        mapping = {int(x): index for index, x in enumerate(unique_values)}

    indices = np.array(
        [mapping.get(int(x), -1) for x in repeats],
        dtype=int,
    )
    return indices, mapping


def _decode_h5_strings(values: np.ndarray) -> tuple[str, ...]:
    decoded = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _read_scalar_text(dataset: h5py.Dataset, default: str = "unknown") -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.asarray(value).size == 1:
        return str(np.asarray(value).squeeze())
    return default


# =============================================================================
# H5 timing metadata
# =============================================================================


def load_stimulus_timing(h5_path: Path) -> StimulusTiming:
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"Moving-bar H5 file not found: {h5_path}")

    required = {
        "Frame_Rate",
        "Directions",
        "Direction_Start_Frame",
        "Direction_Frame_Count",
    }

    with h5py.File(h5_path, "r") as h5_file:
        missing = sorted(required.difference(h5_file.keys()))
        if missing:
            raise KeyError(
                f"{h5_path.name} is missing required metadata {missing}. "
                f"Available datasets: {sorted(h5_file.keys())}"
            )

        h5_direction_names = _decode_h5_strings(h5_file["Directions"][:])
        start_frames = np.asarray(h5_file["Direction_Start_Frame"][:], dtype=int)
        frame_counts = np.asarray(h5_file["Direction_Frame_Count"][:], dtype=int)
        frame_rate_hz = float(np.asarray(h5_file["Frame_Rate"][()]).squeeze())

        polarity = (
            _read_scalar_text(h5_file["Polarity"])
            if "Polarity" in h5_file
            else "unknown"
        )

        if "Noise" in h5_file:
            total_frames = int(h5_file["Noise"].shape[0])
        else:
            total_frames = int(np.max(start_frames + frame_counts))

    if frame_rate_hz <= 0:
        raise ValueError(f"Frame_Rate must be positive in {h5_path}")

    if not (len(h5_direction_names) == len(start_frames) == len(frame_counts)):
        raise ValueError(
            "Directions, Direction_Start_Frame and Direction_Frame_Count "
            f"have inconsistent lengths in {h5_path}"
        )

    if np.any(start_frames < 0) or np.any(frame_counts <= 0):
        raise ValueError(f"Invalid direction frame boundaries in {h5_path}")

    unknown = [name for name in h5_direction_names if name not in H5_TO_PLOT_DIRECTION]
    if unknown:
        raise ValueError(
            f"Unknown direction names in {h5_path.name}: {unknown}. "
            f"Known names: {sorted(H5_TO_PLOT_DIRECTION)}"
        )

    plot_direction_names = tuple(
        H5_TO_PLOT_DIRECTION[name] for name in h5_direction_names
    )

    expected_names = set(DIRECTION_NAMES.tolist())
    if set(plot_direction_names) != expected_names:
        raise ValueError(
            f"Expected exactly the eight transformed directions {sorted(expected_names)}, "
            f"but {h5_path.name} contains {list(plot_direction_names)}"
        )

    timing = StimulusTiming(
        h5_direction_names=h5_direction_names,
        plot_direction_names=plot_direction_names,
        direction_start_frames=start_frames,
        direction_frame_counts=frame_counts,
        frame_rate_hz=frame_rate_hz,
        total_frames=total_frames,
        polarity=polarity,
    )

    print(f"Loaded timing metadata: {h5_path.name}")
    print(
        f"  polarity={timing.polarity}, fps={timing.frame_rate_hz:g}, "
        f"repeat duration={timing.repeat_duration_s:.2f} s, "
        f"frames/repeat={timing.total_frames}"
    )
    for name, start, count, duration in zip(
        timing.plot_direction_names,
        timing.direction_start_frames,
        timing.direction_frame_counts,
        timing.direction_duration_s,
    ):
        print(
            f"  {name:>10s}: start frame {start:4d}, "
            f"{count:4d} frames, {duration:8.3f} s"
        )

    return timing


def validate_condition_against_timing(
    condition: MovingBarConditionConfig,
    timing: StimulusTiming,
) -> None:
    """
    Optional safety checks so you do not accidentally point a condition at the wrong H5.
    """
    if condition.expected_polarity is not None:
        observed = str(timing.polarity).strip().lower()
        expected = str(condition.expected_polarity).strip().lower()
        if observed != expected:
            raise ValueError(
                f"Condition '{condition.title}' expected polarity '{condition.expected_polarity}', "
                f"but H5 reports '{timing.polarity}'. File: {condition.h5_path}"
            )

    if condition.expected_frame_rate_hz is not None:
        if not np.isclose(
            float(condition.expected_frame_rate_hz), timing.frame_rate_hz
        ):
            raise ValueError(
                f"Condition '{condition.title}' expected fps "
                f"{condition.expected_frame_rate_hz}, but H5 reports "
                f"{timing.frame_rate_hz:g}. File: {condition.h5_path}"
            )


# =============================================================================
# RF metric helpers
# =============================================================================


def resolve_reference_channel(
    dataset: xr.Dataset,
    variable: str,
    requested_channel: Any | None,
) -> Any | None:
    if variable not in dataset:
        return None

    da = dataset[variable]
    if "channel" not in da.dims:
        return None

    if "channel" in da.coords:
        channels = [
            _python_scalar(value)
            for value in np.asarray(da["channel"].values).reshape(-1)
        ]
    else:
        channels = list(range(da.sizes["channel"]))

    if not channels:
        return None

    if requested_channel is None:
        return channels[0]

    if requested_channel not in channels:
        raise KeyError(
            f"Reference channel {requested_channel!r} was not found in "
            f"{variable!r}. Available channels: {channels}"
        )

    return requested_channel


def get_cell_metric(
    dataset: xr.Dataset,
    variable: str,
    cell_id: int,
    channel: Any | None,
) -> float:
    if variable not in dataset:
        return np.nan

    da = dataset[variable].sel(cell_index=int(cell_id))

    if channel is not None and "channel" in da.dims:
        da = da.sel(channel=channel)

    arr = np.asarray(da.squeeze(drop=True).values)
    return float(arr) if arr.size == 1 else np.nan


def get_quality_and_tilt(
    dataset: xr.Dataset,
    cell_id: int,
    cfg: PlotConfig,
    reference_channel: Any | None,
) -> tuple[float, float]:
    quality = get_cell_metric(
        dataset,
        cfg.rf_quality_variable,
        cell_id,
        reference_channel,
    )
    tilt = get_cell_metric(
        dataset,
        cfg.rf_tilt_variable,
        cell_id,
        reference_channel,
    )

    if cfg.transform_tilt_reciprocal and np.isfinite(tilt) and tilt != 0:
        tilt = 1.0 / tilt

    return quality, tilt


def select_cells(
    dataset: xr.Dataset,
    cfg: PlotConfig,
    reference_channel: Any | None,
) -> list[int]:
    if "cell_index" not in dataset.coords and "cell_index" not in dataset.dims:
        raise KeyError("The NetCDF dataset has no 'cell_index' coordinate.")

    cells = [int(value) for value in dataset["cell_index"].values]

    if cfg.selected_cells is not None:
        requested = {int(value) for value in cfg.selected_cells}
        cells = [cell for cell in cells if cell in requested]

    retained = []
    for cell in cells:
        quality, tilt = get_quality_and_tilt(
            dataset=dataset,
            cell_id=cell,
            cfg=cfg,
            reference_channel=reference_channel,
        )

        if cfg.min_rf_quality is not None and (
            not np.isfinite(quality) or quality < cfg.min_rf_quality
        ):
            continue

        if cfg.min_tilt is not None and (not np.isfinite(tilt) or tilt < cfg.min_tilt):
            continue

        if cfg.max_tilt is not None and (not np.isfinite(tilt) or tilt > cfg.max_tilt):
            continue

        retained.append(cell)

    if cfg.max_cells is not None:
        retained = retained[: cfg.max_cells]

    return retained


# =============================================================================
# Moving-bar spike splitting and analysis
# =============================================================================


def _times_to_seconds(
    times_triggered: np.ndarray,
    condition: MovingBarConditionConfig,
    timing: StimulusTiming,
) -> tuple[np.ndarray, str]:
    time_unit = condition.time_unit.lower()

    if time_unit == "seconds":
        return times_triggered, "seconds"

    if time_unit == "samples":
        return times_triggered / condition.sample_rate_hz, "samples"

    if time_unit != "auto":
        raise ValueError(
            "MovingBarConditionConfig.time_unit must be "
            "'seconds', 'samples', or 'auto'."
        )

    finite = times_triggered[np.isfinite(times_triggered)]
    max_time = float(np.nanmax(finite)) if finite.size else 0.0

    if max_time <= timing.repeat_duration_s * 2.0:
        return times_triggered, "seconds"

    return times_triggered / condition.sample_rate_hz, "samples"


def split_moving_bar_spikes(
    spikes_df: pd.DataFrame,
    condition: MovingBarConditionConfig,
    timing: StimulusTiming,
    plot_direction_names: np.ndarray,
) -> dict[str, Any]:
    """
    Split one cell's spikes using exact H5 direction frame boundaries.
    """
    for column in [condition.time_col, condition.trigger_col, condition.repeat_col]:
        if column not in spikes_df.columns:
            raise KeyError(
                f"Column {column!r} was not found for {condition.title}. "
                f"Available columns: {list(spikes_df.columns)}"
            )

    empty_rasters = {
        str(direction): [np.array([]) for _ in range(condition.expected_repeats)]
        for direction in plot_direction_names
    }

    if spikes_df.empty:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(len(plot_direction_names), dtype=int),
            "n_spikes": 0,
            "n_input_spikes": 0,
            "repeat_mapping": {},
            "detected_time_unit": condition.time_unit,
        }

    trigger_numeric = pd.to_numeric(
        spikes_df[condition.trigger_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    repeat_numeric = pd.to_numeric(
        spikes_df[condition.repeat_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    time_numeric = pd.to_numeric(
        spikes_df[condition.time_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_base = (
        np.isfinite(trigger_numeric)
        & np.isfinite(repeat_numeric)
        & np.isfinite(time_numeric)
    )

    trigger_frames = np.rint(trigger_numeric[finite_base]).astype(int)
    trigger_frames = trigger_frames - int(condition.trigger_frame_offset)
    repeat_values = np.rint(repeat_numeric[finite_base]).astype(int)
    times_triggered = time_numeric[finite_base]

    repeat_indices, repeat_mapping = _normalise_repeat_labels(
        repeat_values,
        condition.expected_repeats,
    )

    times_in_repeat_s, detected_time_unit = _times_to_seconds(
        times_triggered,
        condition,
        timing,
    )

    valid_repeat = (
        (repeat_indices >= 0)
        & (repeat_indices < condition.expected_repeats)
        & (trigger_frames >= 0)
        & (trigger_frames < timing.total_frames)
        & np.isfinite(times_in_repeat_s)
    )

    trigger_frames = trigger_frames[valid_repeat]
    repeat_indices = repeat_indices[valid_repeat]
    times_in_repeat_s = times_in_repeat_s[valid_repeat]

    # Assign each spike to the exact H5 direction interval
    h5_direction_indices = np.full(len(trigger_frames), -1, dtype=int)

    for h5_index, (start_frame, end_frame) in enumerate(
        zip(timing.direction_start_frames, timing.direction_end_frames)
    ):
        in_direction = (trigger_frames >= start_frame) & (trigger_frames < end_frame)
        h5_direction_indices[in_direction] = h5_index

    valid_direction = h5_direction_indices >= 0

    trigger_frames = trigger_frames[valid_direction]
    repeat_indices = repeat_indices[valid_direction]
    times_in_repeat_s = times_in_repeat_s[valid_direction]
    h5_direction_indices = h5_direction_indices[valid_direction]

    if len(trigger_frames) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(len(plot_direction_names), dtype=int),
            "n_spikes": 0,
            "n_input_spikes": int(len(spikes_df)),
            "repeat_mapping": repeat_mapping,
            "detected_time_unit": detected_time_unit,
        }

    direction_start_times = timing.direction_start_s[h5_direction_indices]
    direction_durations = timing.direction_duration_s[h5_direction_indices]
    time_in_direction_s = times_in_repeat_s - direction_start_times

    tolerance_s = 1.0 / timing.frame_rate_hz + 1e-9
    valid_time = (time_in_direction_s >= -tolerance_s) & (
        time_in_direction_s < direction_durations + tolerance_s
    )

    trigger_frames = trigger_frames[valid_time]
    repeat_indices = repeat_indices[valid_time]
    times_in_repeat_s = times_in_repeat_s[valid_time]
    h5_direction_indices = h5_direction_indices[valid_time]
    time_in_direction_s = time_in_direction_s[valid_time]

    if len(time_in_direction_s):
        current_durations = timing.direction_duration_s[h5_direction_indices]
        time_in_direction_s = np.clip(
            time_in_direction_s,
            0.0,
            np.maximum(current_durations - np.finfo(float).eps, 0.0),
        )

    plot_index_by_name = {
        str(name): index for index, name in enumerate(plot_direction_names)
    }

    direction_indices = np.array(
        [
            plot_index_by_name[timing.plot_direction_names[h5_index]]
            for h5_index in h5_direction_indices
        ],
        dtype=int,
    )

    spike_counts = np.bincount(
        direction_indices,
        minlength=len(plot_direction_names),
    )

    raster_by_direction: dict[str, list[np.ndarray]] = {}
    for direction_index, direction_name in enumerate(plot_direction_names):
        raster_by_direction[str(direction_name)] = [
            np.asarray(
                time_in_direction_s[
                    (direction_indices == direction_index)
                    & (repeat_indices == repeat_index)
                ],
                dtype=float,
            )
            for repeat_index in range(condition.expected_repeats)
        ]

    trigger_time_s = trigger_frames / timing.frame_rate_hz
    mismatch_s = np.abs(times_in_repeat_s - trigger_time_s)
    mismatch_median_s = float(np.nanmedian(mismatch_s)) if mismatch_s.size else np.nan
    mismatch_max_s = float(np.nanmax(mismatch_s)) if mismatch_s.size else np.nan

    return {
        "raster_by_direction": raster_by_direction,
        "spike_counts": spike_counts,
        "n_spikes": int(len(time_in_direction_s)),
        "n_input_spikes": int(len(spikes_df)),
        "repeat_mapping": repeat_mapping,
        "detected_time_unit": detected_time_unit,
        "trigger_time_mismatch_median_s": mismatch_median_s,
        "trigger_time_mismatch_max_s": mismatch_max_s,
    }


def circular_vector_direction_stats(
    responses: np.ndarray,
    directions_deg: np.ndarray,
) -> dict[str, float]:
    responses = np.asarray(responses, dtype=float).copy()
    directions_deg = np.asarray(directions_deg, dtype=float)

    responses[~np.isfinite(responses)] = 0.0
    total_response = float(np.sum(responses))

    if total_response == 0:
        return {
            "vector_preferred_direction_deg": np.nan,
            "vector_DSI": np.nan,
            "preferred_orientation_deg": np.nan,
            "OSI": np.nan,
        }

    theta = np.deg2rad(directions_deg)
    direction_vector = np.sum(responses * np.exp(1j * theta))
    orientation_vector = np.sum(responses * np.exp(2j * theta))

    return {
        "vector_preferred_direction_deg": np.rad2deg(np.angle(direction_vector)) % 360,
        "vector_DSI": np.abs(direction_vector) / total_response,
        "preferred_orientation_deg": np.rad2deg(np.angle(orientation_vector)) / 2 % 180,
        "OSI": np.abs(orientation_vector) / total_response,
    }


def compute_peak_opposite_dsi(
    responses: np.ndarray,
    directions_deg: np.ndarray,
) -> dict[str, float]:
    responses = np.asarray(responses, dtype=float)
    directions_deg = np.asarray(directions_deg, dtype=float)

    if len(responses) == 0 or np.all(~np.isfinite(responses)):
        return {
            "peak_opposite_DSI": np.nan,
            "peak_preferred_direction_deg": np.nan,
        }

    responses_for_argmax = responses.copy()
    responses_for_argmax[~np.isfinite(responses_for_argmax)] = -np.inf

    preferred_index = int(np.argmax(responses_for_argmax))
    preferred_response = responses[preferred_index]
    preferred_direction = directions_deg[preferred_index]

    opposite_target = (preferred_direction + 180) % 360
    differences = np.abs(directions_deg - opposite_target)
    differences = np.minimum(differences, 360 - differences)
    opposite_index = int(np.argmin(differences))
    opposite_response = responses[opposite_index]

    denominator = preferred_response + opposite_response
    dsi = (
        np.nan
        if denominator == 0
        else (preferred_response - opposite_response) / denominator
    )

    return {
        "peak_opposite_DSI": float(dsi),
        "peak_preferred_direction_deg": float(preferred_direction),
    }


def _direction_stats(
    responses: np.ndarray,
    directions_deg: np.ndarray,
) -> dict[str, Any]:
    return {
        "responses": np.asarray(responses, dtype=float),
        **compute_peak_opposite_dsi(responses, directions_deg),
        **circular_vector_direction_stats(responses, directions_deg),
    }


def compute_moving_bar_psths(
    raster_by_direction: dict[str, list[np.ndarray]],
    condition: MovingBarConditionConfig,
    direction_duration_by_name: dict[str, float],
    direction_names: np.ndarray,
) -> dict[str, Any]:
    peak_responses = np.zeros(len(direction_names), dtype=float)
    peak_times = np.full(len(direction_names), np.nan, dtype=float)
    psth_by_direction: dict[str, dict[str, np.ndarray]] = {}

    for direction_index, direction_name_value in enumerate(direction_names):
        direction_name = str(direction_name_value)
        duration_s = float(direction_duration_by_name[direction_name])

        bins = np.arange(
            0.0,
            duration_s + condition.bin_size_s,
            condition.bin_size_s,
        )

        if bins.size < 2:
            bins = np.array([0.0, duration_s], dtype=float)
        elif bins[-1] < duration_s:
            bins = np.append(bins, duration_s)
        else:
            bins[-1] = duration_s

        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([0.0, duration_s], dtype=float)

        bin_widths = np.diff(bins)
        bin_centers = bins[:-1] + bin_widths / 2

        repeat_arrays = raster_by_direction[direction_name]
        non_empty = [
            np.asarray(times, dtype=float) for times in repeat_arrays if len(times) > 0
        ]
        all_spike_times = np.concatenate(non_empty) if non_empty else np.array([])

        if len(all_spike_times) == 0:
            psth = np.zeros(len(bin_centers), dtype=float)
            peak_response = 0.0
            peak_time = np.nan
        else:
            counts, _ = np.histogram(all_spike_times, bins=bins)
            psth = counts / (bin_widths * condition.expected_repeats)
            peak_bin = int(np.argmax(psth))
            peak_response = float(psth[peak_bin])
            peak_time = float(bin_centers[peak_bin])

        peak_responses[direction_index] = peak_response
        peak_times[direction_index] = peak_time
        psth_by_direction[direction_name] = {
            "bins": bins,
            "bin_centers": bin_centers,
            "psth": psth,
            "duration_s": duration_s,
        }

    return {
        "peak_response_by_direction": peak_responses,
        "peak_time_by_direction": peak_times,
        "psth_by_direction": psth_by_direction,
    }


def analyse_moving_bar(
    spikes_df: pd.DataFrame,
    condition: MovingBarConditionConfig,
    timing: StimulusTiming,
    style: PlotStyleConfig,
) -> dict[str, Any]:
    split = split_moving_bar_spikes(
        spikes_df=spikes_df,
        condition=condition,
        timing=timing,
        plot_direction_names=style.direction_names,
    )

    duration_by_name = {
        timing.plot_direction_names[h5_index]: float(duration_s)
        for h5_index, duration_s in enumerate(timing.direction_duration_s)
    }

    duration_in_plot_order = np.array(
        [duration_by_name[str(name)] for name in style.direction_names],
        dtype=float,
    )

    total_time_per_direction_s = condition.expected_repeats * duration_in_plot_order
    mean_rate_response = np.divide(
        split["spike_counts"],
        total_time_per_direction_s,
        out=np.zeros_like(duration_in_plot_order, dtype=float),
        where=total_time_per_direction_s > 0,
    )

    peak_data = compute_moving_bar_psths(
        raster_by_direction=split["raster_by_direction"],
        condition=condition,
        direction_duration_by_name=duration_by_name,
        direction_names=style.direction_names,
    )

    return {
        **split,
        "condition": condition,
        "timing": timing,
        "direction_duration_by_name": duration_by_name,
        "psth_by_direction": peak_data["psth_by_direction"],
        "peak_time_by_direction": peak_data["peak_time_by_direction"],
        "mean_rate": _direction_stats(
            mean_rate_response,
            style.directions_deg,
        ),
        "peak_response": _direction_stats(
            peak_data["peak_response_by_direction"],
            style.directions_deg,
        ),
    }


def get_response_metric(
    condition_data: dict[str, Any],
    style: PlotStyleConfig,
) -> dict[str, Any]:
    if style.response_metric not in {"mean_rate", "peak_response"}:
        raise ValueError("response_metric must be 'peak_response' or 'mean_rate'.")

    metric = condition_data[style.response_metric]
    return {
        **metric,
        "label": (
            "Peak response"
            if style.response_metric == "peak_response"
            else "Mean firing rate"
        ),
    }


# =============================================================================
# Drawing helpers
# =============================================================================


def _close_polar_curve(
    directions_deg: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sort_index = np.argsort(np.asarray(directions_deg) % 360)
    theta_sorted = np.deg2rad(np.asarray(directions_deg)[sort_index])
    response_sorted = np.asarray(responses, dtype=float)[sort_index]

    return (
        np.concatenate([theta_sorted, theta_sorted[:1]]),
        np.concatenate([response_sorted, response_sorted[:1]]),
    )


def _add_motion_arrows(ax, style: PlotStyleConfig) -> None:
    for motion_deg in np.arange(0, 360, 45):
        theta = np.deg2rad(motion_deg)
        x = 0.5 + style.polar_arrow_radius * np.sin(theta)
        y = 0.5 + style.polar_arrow_radius * np.cos(theta)
        dx = style.polar_arrow_length * np.sin(theta)
        dy = style.polar_arrow_length * np.cos(theta)

        ax.annotate(
            "",
            xy=(x + dx / 2, y + dy / 2),
            xytext=(x - dx / 2, y - dy / 2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": style.polar_arrow_linewidth,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": style.polar_arrow_mutation_scale,
            },
            annotation_clip=False,
        )


def _draw_polar_tuning(
    ax,
    condition_data: dict[str, Any],
    style: PlotStyleConfig,
    radial_max: float | None = None,
) -> None:
    metric = get_response_metric(condition_data, style)
    theta, responses = _close_polar_curve(
        style.directions_deg,
        metric["responses"],
    )

    (line,) = ax.plot(
        theta,
        responses,
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        linestyle="-",
    )
    ax.fill(theta, responses, alpha=0.10)

    local_max = (
        float(np.nanmax(metric["responses"])) if len(metric["responses"]) else np.nan
    )

    if radial_max is not None and np.isfinite(radial_max) and radial_max > 0:
        ax.set_ylim(0, radial_max)
        line_extent = radial_max
    else:
        line_extent = local_max

    if np.isfinite(line_extent) and line_extent > 0:
        preferred_direction = metric["peak_preferred_direction_deg"]
        preferred_orientation = metric["preferred_orientation_deg"]

        if style.show_peak_direction_line and np.isfinite(preferred_direction):
            angle = np.deg2rad(preferred_direction)
            ax.plot(
                [angle, angle],
                [0, line_extent],
                linewidth=1.3,
                linestyle=":",
                color=line.get_color(),
            )

        if style.show_orientation_axis and np.isfinite(preferred_orientation):
            for angle_deg in [preferred_orientation, preferred_orientation + 180]:
                angle = np.deg2rad(angle_deg % 360)
                ax.plot(
                    [angle, angle],
                    [0, line_extent],
                    linewidth=1.0,
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.70,
                )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(-22.5)

    ticks = np.arange(0, 360, 45)
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels([""] * len(ticks))
    ax.tick_params(labelsize=5.5, pad=1)

    if style.show_motion_direction_arrows:
        _add_motion_arrows(ax, style)


def _compact_time_ticks(duration_s: float) -> np.ndarray:
    if duration_s <= 0:
        return np.array([0.0])

    middle = duration_s / 2
    return np.array([0.0, middle, duration_s])


def _format_time_tick(value: float, duration_s: float) -> str:
    if duration_s >= 10:
        return f"{value:.0f}"
    if duration_s >= 2:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_small_raster(
    ax,
    repeat_spike_times: list[np.ndarray],
    direction_duration_s: float,
    expected_repeats: int,
    style: PlotStyleConfig,
    show_xlabel: bool,
) -> None:
    ax.set_facecolor(style.raster_facecolor)
    ax.eventplot(
        repeat_spike_times,
        lineoffsets=np.arange(expected_repeats),
        linelengths=0.65,
        linewidths=0.55,
        colors="black",
    )

    ax.set_xlim(0, direction_duration_s)
    ax.set_ylim(-0.5, expected_repeats - 0.5)
    ax.invert_yaxis()

    ticks = _compact_time_ticks(direction_duration_s)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [_format_time_tick(tick, direction_duration_s) for tick in ticks],
        fontsize=4.5,
    )
    ax.tick_params(axis="x", length=1.8, width=0.5, pad=1)

    if show_xlabel:
        ax.set_xlabel("s", fontsize=5, labelpad=0.5)

    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def _draw_small_psth(
    ax,
    psth_data: dict[str, np.ndarray],
    direction_duration_s: float,
    common_y_max: float,
    style: PlotStyleConfig,
    show_y_scale: bool,
) -> None:
    bin_centers = np.asarray(psth_data["bin_centers"], dtype=float)
    firing_rate = np.asarray(psth_data["psth"], dtype=float)

    ax.set_facecolor(style.psth_facecolor)
    ax.plot(bin_centers, firing_rate, linewidth=0.9, color="black")
    ax.fill_between(bin_centers, 0, firing_rate, alpha=0.15, color="black")

    ax.set_xlim(0, direction_duration_s)
    ax.set_ylim(0, common_y_max)
    ax.set_xticks([])

    if show_y_scale:
        ax.set_yticks([0, common_y_max])
        ax.set_yticklabels(["0", f"{common_y_max:.0f}"], fontsize=4.5)
        ax.set_ylabel("Hz", fontsize=4.5, labelpad=0)
        ax.tick_params(axis="y", length=1.8, width=0.5, pad=1)
    else:
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")


def _common_psth_y_max(
    condition_data: dict[str, Any],
    style: PlotStyleConfig,
) -> float:
    values = []

    for direction_name in style.direction_names:
        psth = np.asarray(
            condition_data["psth_by_direction"][str(direction_name)]["psth"],
            dtype=float,
        )
        finite = psth[np.isfinite(psth)]
        if finite.size:
            values.append(finite)

    if not values:
        return 1.0

    maximum = float(np.max(np.concatenate(values)))
    return 1.0 if maximum <= 0 else maximum * 1.08


def _square_design_box(container) -> tuple[float, float, float, float]:
    root_figure = getattr(container, "figure", container)
    root_figure.canvas.draw()

    container_width_px = float(container.bbox.width)
    container_height_px = float(container.bbox.height)

    if container_width_px <= 0 or container_height_px <= 0:
        raise ValueError("The plotting container has zero width or height.")

    if container_width_px >= container_height_px:
        square_width = container_height_px / container_width_px
        square_height = 1.0
        square_left = (1.0 - square_width) / 2
        square_bottom = 0.0
    else:
        square_width = 1.0
        square_height = container_width_px / container_height_px
        square_left = 0.0
        square_bottom = (1.0 - square_height) / 2

    return square_left, square_bottom, square_width, square_height


def _map_box_from_square(
    square_box: tuple[float, float, float, float],
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> list[float]:
    square_left, square_bottom, square_width, square_height = square_box

    return [
        square_left + left * square_width,
        square_bottom + bottom * square_height,
        width * square_width,
        height * square_height,
    ]


def _moving_bar_layout(
    container,
    style: PlotStyleConfig,
) -> tuple[list[float], dict[str, tuple[float, float]], dict[str, float]]:
    square_box = _square_design_box(container)

    polar_local_box = (0.30, 0.27, 0.40, 0.40)
    polar_box = _map_box_from_square(square_box, *polar_local_box)

    raster_width = style.raster_box_width
    raster_height = style.raster_box_height
    gap = style.raster_gap_from_arrow

    total_tile_height = (
        raster_height + style.psth_gap + style.psth_box_height
        if style.show_psth
        else raster_height
    )

    direction_to_angle = {
        "up": 0,
        "up-right": 45,
        "right": 90,
        "down-right": 135,
        "down": 180,
        "down-left": 225,
        "left": 270,
        "up-left": 315,
    }

    polar_left, polar_bottom, polar_width, polar_height = polar_local_box

    arrow_xy = {}
    for direction_name, motion_deg in direction_to_angle.items():
        theta = np.deg2rad(motion_deg)
        x_axes = 0.5 + style.polar_arrow_radius * np.sin(theta)
        y_axes = 0.5 + style.polar_arrow_radius * np.cos(theta)
        arrow_xy[direction_name] = (
            polar_left + x_axes * polar_width,
            polar_bottom + y_axes * polar_height,
        )

    positions_local: dict[str, tuple[float, float]] = {}

    x, y = arrow_xy["up"]
    positions_local["up"] = (x - raster_width / 2, y + gap)

    x, y = arrow_xy["down"]
    positions_local["down"] = (
        x - raster_width / 2,
        y - gap - total_tile_height,
    )

    x, y = arrow_xy["right"]
    positions_local["right"] = (
        x + gap,
        y - total_tile_height / 2,
    )

    x, y = arrow_xy["left"]
    positions_local["left"] = (
        x - gap - raster_width,
        y - total_tile_height / 2,
    )

    diagonal_gap = gap / np.sqrt(2)

    x, y = arrow_xy["up-right"]
    positions_local["up-right"] = (
        x + diagonal_gap,
        y + diagonal_gap,
    )

    x, y = arrow_xy["up-left"]
    positions_local["up-left"] = (
        x - diagonal_gap - raster_width,
        y + diagonal_gap,
    )

    x, y = arrow_xy["down-right"]
    positions_local["down-right"] = (
        x + diagonal_gap,
        y - diagonal_gap - total_tile_height,
    )

    x, y = arrow_xy["down-left"]
    positions_local["down-left"] = (
        x - diagonal_gap - raster_width,
        y - diagonal_gap - total_tile_height,
    )

    positions = {}
    for direction_name, (left, bottom) in positions_local.items():
        mapped = _map_box_from_square(
            square_box,
            left,
            bottom,
            raster_width,
            raster_height,
        )
        positions[direction_name] = (mapped[0], mapped[1])

    dimensions = {
        "raster_width": raster_width * square_box[2],
        "raster_height": raster_height * square_box[3],
        "psth_height": style.psth_box_height * square_box[3],
        "psth_gap": style.psth_gap * square_box[3],
    }

    return polar_box, positions, dimensions


def draw_condition_panel(
    container,
    condition_data: dict[str, Any],
    cell_id: int,
    quality: float,
    tilt: float,
    style: PlotStyleConfig,
    radial_max: float | None = None,
) -> None:
    condition = condition_data["condition"]
    timing = condition_data["timing"]
    metric = get_response_metric(condition_data, style)
    common_y_max = _common_psth_y_max(condition_data, style)

    polar_box, positions, dimensions = _moving_bar_layout(container, style)

    ax_polar = container.add_axes(polar_box, projection="polar")
    _draw_polar_tuning(
        ax=ax_polar,
        condition_data=condition_data,
        style=style,
        radial_max=radial_max,
    )

    raster_width = dimensions["raster_width"]
    raster_height = dimensions["raster_height"]
    psth_height = dimensions["psth_height"]
    psth_gap = dimensions["psth_gap"]

    for direction_name, (left, bottom) in positions.items():
        direction_duration_s = condition_data["direction_duration_by_name"][
            direction_name
        ]

        ax_raster = container.add_axes([left, bottom, raster_width, raster_height])
        _draw_small_raster(
            ax=ax_raster,
            repeat_spike_times=condition_data["raster_by_direction"][direction_name],
            direction_duration_s=direction_duration_s,
            expected_repeats=condition.expected_repeats,
            style=style,
            show_xlabel=(direction_name == "left"),
        )

        if style.show_psth:
            ax_psth = container.add_axes(
                [
                    left,
                    bottom + raster_height + psth_gap,
                    raster_width,
                    psth_height,
                ]
            )
            _draw_small_psth(
                ax=ax_psth,
                psth_data=condition_data["psth_by_direction"][direction_name],
                direction_duration_s=direction_duration_s,
                common_y_max=common_y_max,
                style=style,
                show_y_scale=(style.show_psth_scale and direction_name == "left"),
            )

    quality_text = f"{quality:.1f}" if np.isfinite(quality) else "n/a"
    tilt_text = f"{tilt:.2f}" if np.isfinite(tilt) else "n/a"

    container.suptitle(
        f"{condition.title} | H5: {timing.polarity}, {timing.frame_rate_hz:g} fps\n"
        f"Cell {cell_id} | {metric['label']} | "
        f"Q={quality_text}, tilt={tilt_text}, "
        f"DSI={metric['peak_opposite_DSI']:.2f}, "
        f"OSI={metric['OSI']:.2f}",
        fontsize=10,
        y=0.985,
    )


def _shared_radial_max(
    condition_results: Iterable[dict[str, Any]],
    style: PlotStyleConfig,
) -> float | None:
    if not style.share_polar_scale_across_conditions:
        return None

    maxima = []
    for result in condition_results:
        responses = np.asarray(
            get_response_metric(result, style)["responses"],
            dtype=float,
        )
        finite = responses[np.isfinite(responses)]
        if finite.size:
            maxima.append(float(np.max(finite)))

    if not maxima:
        return 1.0

    maximum = max(maxima)
    return 1.0 if maximum <= 0 else maximum * 1.05


# =============================================================================
# Session class
# =============================================================================


class MovingBarSession:
    def __init__(
        self,
        cfg: PlotConfig,
        dataset: xr.Dataset | None = None,
    ):
        self.cfg = cfg
        self._owns_dataset = dataset is None

        if not 1 <= len(cfg.conditions) <= 4:
            raise ValueError(
                f"Provide between 1 and 4 moving-bar conditions; "
                f"received {len(cfg.conditions)}."
            )

        if dataset is None:
            print("No preloaded dataset supplied; loading NetCDF from disk...")
            self.dataset = xr.load_dataset(cfg.noise_data_path)
        else:
            print("Using the already-loaded NetCDF dataset.")
            self.dataset = dataset

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading Polarspike recording overview...")
        self.recording = Overview.Recording.load(str(cfg.overview_path))

        self.reference_channel = resolve_reference_channel(
            dataset=self.dataset,
            variable=cfg.rf_quality_variable,
            requested_channel=cfg.rf_reference_channel,
        )

        self.timings: dict[str, StimulusTiming] = {}
        for condition in cfg.conditions:
            timing = load_stimulus_timing(condition.h5_path)
            validate_condition_against_timing(condition, timing)
            self.timings[condition.title] = timing

        self.available_cells = select_cells(
            dataset=self.dataset,
            cfg=cfg,
            reference_channel=self.reference_channel,
        )

        self._spike_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._analysis_cache: dict[int, dict[str, Any]] = {}

        print(
            f"Session ready. {len(self.available_cells)} selected cells:\n"
            f"{self.available_cells}"
        )
        print(f"Reference RF channel: {self.reference_channel!r}")

    def _check_cell(self, cell_id: int) -> int:
        cell_id = int(cell_id)
        all_cells = {int(value) for value in self.dataset["cell_index"].values}

        if cell_id not in all_cells:
            raise ValueError(f"Cell {cell_id} is not present in the NetCDF dataset.")

        return cell_id

    def get_spikes(
        self,
        cell_id: int,
        stimulus_index: int,
    ) -> pd.DataFrame:
        key = (int(cell_id), int(stimulus_index))

        if key not in self._spike_cache:
            condition = {
                "stimulus_index": [int(stimulus_index)],
                "cell_index": [int(cell_id)],
            }
            self._spike_cache[key] = _to_pandas(
                self.recording.get_spikes_triggered([condition])
            )

        return self._spike_cache[key]

    def analyse_cell(
        self,
        cell_id: int,
        refresh: bool = False,
    ) -> dict[str, Any]:
        cell_id = self._check_cell(cell_id)

        if not refresh and cell_id in self._analysis_cache:
            return self._analysis_cache[cell_id]

        quality, tilt = get_quality_and_tilt(
            dataset=self.dataset,
            cell_id=cell_id,
            cfg=self.cfg,
            reference_channel=self.reference_channel,
        )

        condition_results = []

        for condition in self.cfg.conditions:
            spikes = self.get_spikes(cell_id, condition.stimulus_index)
            result = analyse_moving_bar(
                spikes_df=spikes,
                condition=condition,
                timing=self.timings[condition.title],
                style=self.cfg.style,
            )
            condition_results.append(result)

            counts = dict(
                zip(
                    self.cfg.style.direction_names,
                    result["spike_counts"].tolist(),
                )
            )

            print(
                f"Cell {cell_id} | {condition.title}: retained "
                f"{result['n_spikes']} / {result['n_input_spikes']} spikes; "
                f"time unit={result['detected_time_unit']}; counts={counts}"
            )

            mismatch = result.get("trigger_time_mismatch_max_s", np.nan)
            frame_duration = 1.0 / result["timing"].frame_rate_hz
            if np.isfinite(mismatch) and mismatch > 2.5 * frame_duration:
                print(
                    "  Warning: spike times and trigger-frame times differ by "
                    f"up to {mismatch:.3f} s. Check time_unit and "
                    "trigger_frame_offset for this condition."
                )

        analysed = {
            "cell_id": cell_id,
            "quality": quality,
            "tilt": tilt,
            "conditions": condition_results,
        }

        self._analysis_cache[cell_id] = analysed
        return analysed

    def plot_cell(
        self,
        cell_id: int,
        save: bool | None = None,
        show: bool | None = None,
    ):
        result = self.analyse_cell(cell_id)

        if save is None:
            save = self.cfg.save
        if show is None:
            show = self.cfg.show

        n_conditions = len(result["conditions"])

        if n_conditions == 1:
            figure_size = self.cfg.figure_size
            n_rows, n_cols = 1, 1
        elif n_conditions == 2:
            figure_size = self.cfg.figure_size
            n_rows, n_cols = 1, 2
        else:
            figure_size = self.cfg.figure_size
            n_rows, n_cols = 2, 2

        fig = plt.figure(
            figsize=figure_size,
            dpi=self.cfg.figure_dpi,
            constrained_layout=False,
        )

        grid = fig.add_gridspec(
            n_rows,
            n_cols,
            left=0.025,
            right=0.975,
            bottom=0.025,
            top=0.955,
            wspace=0.035,
            hspace=0.060,
        )

        subfigures = []
        for i in range(n_conditions):
            row = i // n_cols
            col = i % n_cols
            subfigures.append(fig.add_subfigure(grid[row, col]))

        radial_max = _shared_radial_max(
            result["conditions"],
            self.cfg.style,
        )

        for subfigure, condition_result in zip(subfigures, result["conditions"]):
            draw_condition_panel(
                container=subfigure,
                condition_data=condition_result,
                cell_id=cell_id,
                quality=result["quality"],
                tilt=result["tilt"],
                style=self.cfg.style,
                radial_max=radial_max,
            )

        output_path = (
            self.cfg.output_dir
            / f"{self.cfg.recording_name}_cell_{cell_id}_moving_bar_summary.{self.cfg.image_format}"
        )

        if save:
            fig.savefig(
                output_path,
                dpi=self.cfg.save_dpi,
                bbox_inches="tight",
            )
            print(f"Saved: {output_path}")

        if show:
            plt.show()

        return fig

    def plot_selected_cells(
        self,
        save: bool | None = None,
        show: bool | None = None,
    ) -> dict[int, Any]:
        figures = {}
        for cell_id in self.available_cells:
            figures[cell_id] = self.plot_cell(
                cell_id=cell_id,
                save=save,
                show=show,
            )
        return figures

    def quality_table(self) -> pd.DataFrame:
        if self.cfg.rf_quality_variable not in self.dataset:
            raise KeyError(
                f"{self.cfg.rf_quality_variable!r} was not found in the dataset."
            )

        rows = []
        for cell_id in self.dataset["cell_index"].values:
            quality, tilt = get_quality_and_tilt(
                dataset=self.dataset,
                cell_id=int(cell_id),
                cfg=self.cfg,
                reference_channel=self.reference_channel,
            )
            rows.append(
                {
                    "cell_index": int(cell_id),
                    "rf_quality": quality,
                    "tilt": tilt,
                }
            )

        table = pd.DataFrame(rows)
        table = (
            table.dropna(subset=["rf_quality"])
            .sort_values("rf_quality", ascending=False)
            .reset_index(drop=True)
        )
        table["rank"] = np.arange(1, len(table) + 1)
        return table

    def clear_cache(self, cell_id: int | None = None) -> None:
        if cell_id is None:
            self._spike_cache.clear()
            self._analysis_cache.clear()
            return

        cell_id = int(cell_id)
        self._analysis_cache.pop(cell_id, None)

        for key in list(self._spike_cache):
            if key[0] == cell_id:
                self._spike_cache.pop(key)

    def close(self, close_dataset: bool | None = None) -> None:
        if close_dataset is None:
            close_dataset = self._owns_dataset

        if close_dataset:
            self.dataset.close()

        plt.close("all")
        print("Moving-bar plotting session closed.")


# =============================================================================
# Example usage
# =============================================================================

# %% 1. Load the large NetCDF dataset once
NOISE_DATA_PATH = Path(
    r"F:\Laura\zebrafish_14_08_2026\Phase_00\noise_analysis\noise_data.nc"
)

try:
    dataset.close()
except (NameError, AttributeError):
    pass

dataset = xr.load_dataset(NOISE_DATA_PATH)

print("Loaded NetCDF dataset")
print("Dimensions:", dict(dataset.sizes))
print("Variables:", list(dataset.data_vars))
if "channel" in dataset.coords:
    print(
        "Channels:",
        [
            _python_scalar(value)
            for value in np.asarray(dataset["channel"].values).reshape(-1)
        ],
    )


# %% 2. Edit paths, stimulus indices and plotting parameters
STIMULUS_DIR = Path(r"F:\Laura\stimuli\stimuli_July_2026")

config = PlotConfig(
    recording_name="zebrafish_14_08_2026_phase_00",
    overview_path=Path(r"F:\Laura\zebrafish_14_08_2026\Phase_00\overview"),
    noise_data_path=NOISE_DATA_PATH,
    output_dir=Path(r"F:\Laura\zebrafish_14_08_2026\Phase_00\moving_bar_summary_plots"),
    # Supply ANY subset of 1-4 conditions in the order you want them plotted.
    # For the full set, recommended order is:
    #   1. ON  13 fps
    #   2. OFF 13 fps
    #   3. ON  60 fps
    #   4. OFF 60 fps
    conditions=(
        MovingBarConditionConfig(
            title="ON moving bar — 13 fps — 20 px",
            stimulus_index=15,  # CHANGE
            h5_path=STIMULUS_DIR
            / "F:\Laura\stimuli\stimuli_July_2026\moving_bar_weighted_13fps_ON_800px_wide_800px_high_20px_thick_30.0px_ramp_1.0px_per_frame.h5",
            expected_repeats=3,  # CHANGE if needed
            expected_polarity="ON",
            expected_frame_rate_hz=13,
            time_unit="seconds",
            bin_size_s=0.25,
        ),
        MovingBarConditionConfig(
            title="OFF moving bar — 13 fps — 20 px",
            stimulus_index=16,  # CHANGE
            h5_path=STIMULUS_DIR
            / "F:\Laura\stimuli\stimuli_July_2026\moving_edge_weighted_13fps_OFF_800px_wide_800px_high__20px_thick_30.0px_ramp_1.0px_per_frame.h5",
            expected_repeats=3,  # CHANGE if needed
            expected_polarity="OFF",
            expected_frame_rate_hz=13,
            time_unit="seconds",
            bin_size_s=0.25,
        ),
        # MovingBarConditionConfig(
        #     title="ON moving bar — 60 fps — 20 px",
        #     stimulus_index=7,  # CHANGE
        #     h5_path=STIMULUS_DIR / "YOUR_ON_60FPS_20PX_FILE.h5",
        #     expected_repeats=3,  # CHANGE if needed
        #     expected_polarity="ON",
        #     expected_frame_rate_hz=60,
        #     time_unit="seconds",
        #     bin_size_s=0.25,
        # ),
        # MovingBarConditionConfig(
        #     title="OFF moving bar — 60 fps — 20 px",
        #     stimulus_index=8,  # CHANGE
        #     h5_path=STIMULUS_DIR / "YOUR_OFF_60FPS_20PX_FILE.h5",
        #     expected_repeats=3,  # CHANGE if needed
        #     expected_polarity="OFF",
        #     expected_frame_rate_hz=60,
        #     time_unit="seconds",
        #     bin_size_s=0.25,
        # ),
    ),
    style=PlotStyleConfig(
        response_metric="peak_response",  # or "mean_rate"
        show_psth=True,
        show_psth_scale=True,
        share_polar_scale_across_conditions=False,
    ),
    rf_quality_variable="quality",
    rf_tilt_variable="tilt",
    rf_reference_channel="32px_15Hz_20mins_shuffle_x4",
    transform_tilt_reciprocal=True,
    # Example:
    # selected_cells=[235, 135]
    selected_cells=None,
    min_rf_quality=20,
    min_tilt=None,
    max_tilt=None,
    max_cells=None,
    save=True,
    show=True,
)


# %% 3. Create / refresh the session without reloading the NetCDF
try:
    moving_bar_session.close(close_dataset=False)
except (NameError, AttributeError):
    pass

moving_bar_session = MovingBarSession(
    cfg=config,
    dataset=dataset,
)


# %% 4. Inspect cells ranked by RF quality
quality_table = moving_bar_session.quality_table()
print(quality_table.head(20))


# %% 5. Plot one cell
CELL_ID = 94  # CHANGE
figure = moving_bar_session.plot_cell(
    cell_id=CELL_ID,
    save=False,
    show=True,
)


# %% 6. Optional: plot every retained cell
all_figures = moving_bar_session.plot_selected_cells(save=True, show=False)
