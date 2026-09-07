from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from polarspike import Overview
from tqdm import tqdm

from fff_common import (
    get_cell_stimulus_spikes,
    normalise_repeat_labels,
    stimulus_cell_indices,
)


# Exact projector/projection direction transformation from the supplied
# moving-edge script.
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
DIRECTIONS_DEG = np.array([90, 270, 0, 180, 45, 315, 135, 225], dtype=float)


@dataclass(frozen=True)
class MovingEdgeConditionConfig:
    """One moving-edge condition, e.g. ON 13 fps or OFF 60 fps."""

    output_name: str
    stimulus_index: int
    h5_path: Path
    expected_repeats: int
    expected_polarity: str | None = None
    expected_frame_rate_hz: float | None = None
    trigger_col: str = "trigger"
    repeat_col: str = "repeat"
    time_col: str = "times_triggered"
    time_unit: str = "seconds"
    sample_rate_hz: float = 20_000.0
    trigger_frame_offset: int = 0
    bin_size_s: float = 0.25
    response_metric: str = "peak_response"  # or "mean_rate"


@dataclass(frozen=True)
class StimulusTiming:
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


def _decode_h5_strings(values: np.ndarray) -> tuple[str, ...]:
    decoded = []
    for value in np.asarray(values).reshape(-1):
        decoded.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return tuple(decoded)


def _read_scalar_text(dataset: h5py.Dataset, default: str = "unknown") -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.asarray(value).size == 1:
        return str(np.asarray(value).squeeze())
    return default


def load_stimulus_timing(h5_path: Path) -> StimulusTiming:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Moving-edge H5 file not found: {h5_path}")

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
        total_frames = (
            int(h5_file["Noise"].shape[0])
            if "Noise" in h5_file
            else int(np.max(start_frames + frame_counts))
        )

    if frame_rate_hz <= 0:
        raise ValueError(f"Frame_Rate must be positive in {h5_path}")
    if not (len(h5_direction_names) == len(start_frames) == len(frame_counts)):
        raise ValueError("Direction metadata arrays have inconsistent lengths.")
    if np.any(start_frames < 0) or np.any(frame_counts <= 0):
        raise ValueError(f"Invalid direction frame boundaries in {h5_path}")

    unknown = [name for name in h5_direction_names if name not in H5_TO_PLOT_DIRECTION]
    if unknown:
        raise ValueError(f"Unknown H5 direction names: {unknown}")

    plot_direction_names = tuple(H5_TO_PLOT_DIRECTION[name] for name in h5_direction_names)
    expected_names = set(DIRECTION_NAMES.tolist())
    if set(plot_direction_names) != expected_names:
        raise ValueError(
            f"Expected the eight transformed directions {sorted(expected_names)}, "
            f"got {list(plot_direction_names)}"
        )

    return StimulusTiming(
        h5_direction_names=h5_direction_names,
        plot_direction_names=plot_direction_names,
        direction_start_frames=start_frames,
        direction_frame_counts=frame_counts,
        frame_rate_hz=frame_rate_hz,
        total_frames=total_frames,
        polarity=polarity,
    )


def validate_condition(condition: MovingEdgeConditionConfig, timing: StimulusTiming):
    if condition.expected_polarity is not None:
        observed = str(timing.polarity).strip().lower()
        expected = str(condition.expected_polarity).strip().lower()
        if observed != expected:
            raise ValueError(
                f"Condition {condition.output_name!r} expected polarity "
                f"{condition.expected_polarity!r}, but H5 reports {timing.polarity!r}."
            )

    if condition.expected_frame_rate_hz is not None and not np.isclose(
        float(condition.expected_frame_rate_hz), timing.frame_rate_hz
    ):
        raise ValueError(
            f"Condition {condition.output_name!r} expected "
            f"{condition.expected_frame_rate_hz} fps, but H5 reports "
            f"{timing.frame_rate_hz:g} fps."
        )


def _times_to_seconds(times_triggered, condition, timing):
    time_unit = condition.time_unit.lower()
    if time_unit == "seconds":
        return times_triggered
    if time_unit == "samples":
        return times_triggered / condition.sample_rate_hz
    if time_unit != "auto":
        raise ValueError("time_unit must be 'seconds', 'samples', or 'auto'.")

    finite = times_triggered[np.isfinite(times_triggered)]
    max_time = float(np.nanmax(finite)) if finite.size else 0.0
    if max_time <= timing.repeat_duration_s * 2.0:
        return times_triggered
    return times_triggered / condition.sample_rate_hz


def split_moving_edge_spikes(
    spikes_df: pd.DataFrame,
    condition: MovingEdgeConditionConfig,
    timing: StimulusTiming,
) -> dict[str, Any]:
    for column in [condition.time_col, condition.trigger_col, condition.repeat_col]:
        if column not in spikes_df.columns:
            raise KeyError(
                f"Column {column!r} not found for {condition.output_name}. "
                f"Available: {list(spikes_df.columns)}"
            )

    empty_rasters = {
        str(direction): [np.array([]) for _ in range(condition.expected_repeats)]
        for direction in DIRECTION_NAMES
    }
    if spikes_df.empty:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(len(DIRECTION_NAMES), dtype=int),
            "n_spikes": 0,
        }

    trigger_numeric = pd.to_numeric(
        spikes_df[condition.trigger_col], errors="coerce"
    ).to_numpy(dtype=float)
    repeat_numeric = pd.to_numeric(
        spikes_df[condition.repeat_col], errors="coerce"
    ).to_numpy(dtype=float)
    time_numeric = pd.to_numeric(
        spikes_df[condition.time_col], errors="coerce"
    ).to_numpy(dtype=float)

    finite = np.isfinite(trigger_numeric) & np.isfinite(repeat_numeric) & np.isfinite(time_numeric)
    trigger_frames = np.rint(trigger_numeric[finite]).astype(int)
    trigger_frames -= int(condition.trigger_frame_offset)
    repeat_values = np.rint(repeat_numeric[finite]).astype(int)
    times_triggered = time_numeric[finite]

    repeat_indices, _ = normalise_repeat_labels(repeat_values, condition.expected_repeats)
    times_in_repeat_s = _times_to_seconds(times_triggered, condition, timing)

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
            "spike_counts": np.zeros(len(DIRECTION_NAMES), dtype=int),
            "n_spikes": 0,
        }

    direction_start_times = timing.direction_start_s[h5_direction_indices]
    direction_durations = timing.direction_duration_s[h5_direction_indices]
    time_in_direction_s = times_in_repeat_s - direction_start_times

    tolerance_s = 1.0 / timing.frame_rate_hz + 1e-9
    valid_time = (time_in_direction_s >= -tolerance_s) & (
        time_in_direction_s < direction_durations + tolerance_s
    )
    repeat_indices = repeat_indices[valid_time]
    h5_direction_indices = h5_direction_indices[valid_time]
    time_in_direction_s = time_in_direction_s[valid_time]

    if len(time_in_direction_s):
        current_durations = timing.direction_duration_s[h5_direction_indices]
        time_in_direction_s = np.clip(
            time_in_direction_s,
            0.0,
            np.maximum(current_durations - np.finfo(float).eps, 0.0),
        )

    plot_index_by_name = {str(name): i for i, name in enumerate(DIRECTION_NAMES)}
    direction_indices = np.array(
        [
            plot_index_by_name[timing.plot_direction_names[h5_index]]
            for h5_index in h5_direction_indices
        ],
        dtype=int,
    )

    spike_counts = np.bincount(direction_indices, minlength=len(DIRECTION_NAMES))
    raster_by_direction = {}
    for direction_index, direction_name in enumerate(DIRECTION_NAMES):
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

    return {
        "raster_by_direction": raster_by_direction,
        "spike_counts": spike_counts,
        "n_spikes": int(len(time_in_direction_s)),
    }


def circular_vector_direction_stats(responses, directions_deg):
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


def compute_peak_opposite_dsi(responses, directions_deg):
    responses = np.asarray(responses, dtype=float)
    directions_deg = np.asarray(directions_deg, dtype=float)
    if len(responses) == 0 or np.all(~np.isfinite(responses)):
        return {"peak_opposite_DSI": np.nan, "peak_preferred_direction_deg": np.nan}
    values = responses.copy()
    values[~np.isfinite(values)] = -np.inf
    preferred_index = int(np.argmax(values))
    preferred_response = responses[preferred_index]
    preferred_direction = directions_deg[preferred_index]
    opposite_target = (preferred_direction + 180) % 360
    differences = np.abs(directions_deg - opposite_target)
    differences = np.minimum(differences, 360 - differences)
    opposite_index = int(np.argmin(differences))
    opposite_response = responses[opposite_index]
    denominator = preferred_response + opposite_response
    dsi = np.nan if denominator == 0 else (preferred_response - opposite_response) / denominator
    return {
        "peak_opposite_DSI": float(dsi),
        "peak_preferred_direction_deg": float(preferred_direction),
    }


def _direction_stats(responses):
    return {
        "responses": np.asarray(responses, dtype=float),
        **compute_peak_opposite_dsi(responses, DIRECTIONS_DEG),
        **circular_vector_direction_stats(responses, DIRECTIONS_DEG),
    }


def compute_peak_responses(raster_by_direction, condition, duration_by_name):
    peak_responses = np.zeros(len(DIRECTION_NAMES), dtype=float)
    for direction_index, direction_name_value in enumerate(DIRECTION_NAMES):
        direction_name = str(direction_name_value)
        duration_s = float(duration_by_name[direction_name])
        bins = np.arange(0.0, duration_s + condition.bin_size_s, condition.bin_size_s)
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

        non_empty = [
            np.asarray(times, dtype=float)
            for times in raster_by_direction[direction_name]
            if len(times) > 0
        ]
        all_spike_times = np.concatenate(non_empty) if non_empty else np.array([])
        if len(all_spike_times) == 0:
            peak_response = 0.0
        else:
            counts, _ = np.histogram(all_spike_times, bins=bins)
            psth = counts / (bin_widths * condition.expected_repeats)
            peak_response = float(psth[int(np.argmax(psth))])
        peak_responses[direction_index] = peak_response
    return peak_responses


def analyse_moving_edge(spikes_df, condition, timing):
    split = split_moving_edge_spikes(spikes_df, condition, timing)
    duration_by_name = {
        timing.plot_direction_names[h5_index]: float(duration_s)
        for h5_index, duration_s in enumerate(timing.direction_duration_s)
    }
    duration_in_plot_order = np.array(
        [duration_by_name[str(name)] for name in DIRECTION_NAMES], dtype=float
    )
    total_time_per_direction_s = condition.expected_repeats * duration_in_plot_order
    mean_rate = np.divide(
        split["spike_counts"],
        total_time_per_direction_s,
        out=np.zeros_like(duration_in_plot_order, dtype=float),
        where=total_time_per_direction_s > 0,
    )
    peak_response = compute_peak_responses(
        split["raster_by_direction"], condition, duration_by_name
    )
    return {
        "n_spikes": split["n_spikes"],
        "mean_rate": _direction_stats(mean_rate),
        "peak_response": _direction_stats(peak_response),
    }


def extract_moving_edge_features(
    spikes_df: pd.DataFrame,
    cell_index: int,
    condition: MovingEdgeConditionConfig,
    timing: StimulusTiming,
) -> dict[str, float | int] | None:
    analysis = analyse_moving_edge(spikes_df, condition, timing)
    if analysis["n_spikes"] == 0:
        return None
    if condition.response_metric not in {"mean_rate", "peak_response"}:
        raise ValueError("response_metric must be 'mean_rate' or 'peak_response'.")

    metric = analysis[condition.response_metric]
    prefix = f"moving_edge_{condition.output_name}"
    result: dict[str, float | int] = {
        "cell_index": int(cell_index),
        f"{prefix}_n_spikes": int(analysis["n_spikes"]),
        # DSI matches the value displayed in the supplied moving-edge figures.
        f"{prefix}_DSI": float(metric["peak_opposite_DSI"]),
        f"{prefix}_vector_DSI": float(metric["vector_DSI"]),
        f"{prefix}_OSI": float(metric["OSI"]),
        f"{prefix}_preferred_direction_deg": float(metric["peak_preferred_direction_deg"]),
        f"{prefix}_vector_preferred_direction_deg": float(metric["vector_preferred_direction_deg"]),
        f"{prefix}_preferred_orientation_deg": float(metric["preferred_orientation_deg"]),
    }
    for direction_name, response in zip(DIRECTION_NAMES, metric["responses"]):
        result[f"{prefix}_response_{str(direction_name).replace('-', '_')}"] = float(response)
    return result


def calculate_moving_edge_features(
    overview_path,
    conditions: Sequence[MovingEdgeConditionConfig],
) -> pd.DataFrame:
    """Calculate all supplied moving-edge conditions and merge them by cell_index."""
    if not conditions:
        return pd.DataFrame()

    recording = Overview.Recording.load(overview_path)
    condition_tables = []

    for condition in conditions:
        timing = load_stimulus_timing(condition.h5_path)
        validate_condition(condition, timing)
        cells = stimulus_cell_indices(recording, condition.stimulus_index)

        results = []
        desc = f"Moving edge: {condition.output_name}"
        for cell_index in tqdm(cells, desc=desc):
            spikes = get_cell_stimulus_spikes(
                recording, int(cell_index), condition.stimulus_index
            )
            result = extract_moving_edge_features(
                spikes, int(cell_index), condition, timing
            )
            if result is not None:
                results.append(result)

        condition_tables.append(pd.DataFrame(results))

    merged = None
    for table in condition_tables:
        if table.empty:
            continue
        merged = table if merged is None else merged.merge(
            table, on="cell_index", how="outer", validate="one_to_one"
        )

    return pd.DataFrame() if merged is None else merged.sort_values("cell_index").reset_index(drop=True)
