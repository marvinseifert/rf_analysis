from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from polarspike import Overview
from tqdm import tqdm

from fff_common import (
    get_cell_stimulus_spikes,
    normalise_repeat_labels,
    stimulus_cell_indices,
)


# Exact direction convention from the supplied multipanel summary script.
DIRECTION_NAMES = np.array(
    [
        "up",
        "down",
        "right",
        "left",
        "up-right",
        "down-right",
        "up-left",
        "down-left",
    ]
)
DIRECTIONS_DEG = np.array([0, 180, 90, 270, 45, 135, 315, 225], dtype=float)


@dataclass(frozen=True)
class MovingBarFeatureConfig:
    stimulus_index: int
    frames_per_direction: int = 300
    seconds_per_frame: float = 1 / 60
    expected_repeats: int = 10
    sample_rate_hz: float = 20_000.0
    time_unit: str = "seconds"
    trigger_col: str = "trigger"
    repeat_col: str = "repeat"
    time_col: str = "times_triggered"
    bin_size_s: float = 0.10
    response_metric: str = "peak_response"  # or "total_spikes"
    direction_names: np.ndarray = field(default_factory=lambda: DIRECTION_NAMES.copy())
    directions_deg: np.ndarray = field(default_factory=lambda: DIRECTIONS_DEG.copy())


def split_moving_bar_spikes(
    spikes_df: pd.DataFrame,
    cfg: MovingBarFeatureConfig,
) -> dict[str, Any]:
    """Exact moving-bar splitting logic from the supplied summary script."""
    for col in [cfg.time_col, cfg.trigger_col, cfg.repeat_col]:
        if col not in spikes_df.columns:
            raise KeyError(
                f"Column {col!r} not found in moving-bar table. "
                f"Available: {list(spikes_df.columns)}"
            )

    direction_names = np.asarray(cfg.direction_names)
    n_directions = len(direction_names)
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    frames_per_repeat = cfg.frames_per_direction * n_directions

    trigger_frames = pd.to_numeric(
        spikes_df[cfg.trigger_col], errors="coerce"
    ).to_numpy(dtype=float)
    repeat_values = pd.to_numeric(
        spikes_df[cfg.repeat_col], errors="coerce"
    ).to_numpy(dtype=float)
    times_triggered = pd.to_numeric(
        spikes_df[cfg.time_col], errors="coerce"
    ).to_numpy(dtype=float)

    finite = (
        np.isfinite(trigger_frames)
        & np.isfinite(repeat_values)
        & np.isfinite(times_triggered)
    )
    trigger_frames = np.rint(trigger_frames[finite]).astype(int)
    repeat_values = np.rint(repeat_values[finite]).astype(int)
    times_triggered = times_triggered[finite]

    repeat_indices, _ = normalise_repeat_labels(repeat_values, cfg.expected_repeats)
    valid = (
        (trigger_frames >= 0)
        & (trigger_frames < frames_per_repeat)
        & (repeat_indices >= 0)
        & (repeat_indices < cfg.expected_repeats)
    )
    trigger_frames = trigger_frames[valid]
    repeat_indices = repeat_indices[valid]
    times_triggered = times_triggered[valid]

    empty_rasters = {
        str(direction): [np.array([]) for _ in range(cfg.expected_repeats)]
        for direction in direction_names
    }
    if len(trigger_frames) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(n_directions, dtype=int),
            "n_spikes": 0,
        }

    direction_indices = trigger_frames // cfg.frames_per_direction

    if cfg.time_unit == "seconds":
        time_in_repeat_s = times_triggered
    elif cfg.time_unit == "samples":
        time_in_repeat_s = times_triggered / cfg.sample_rate_hz
    elif cfg.time_unit == "auto":
        repeat_duration_s = direction_duration_s * n_directions
        finite_times = times_triggered[np.isfinite(times_triggered)]
        max_time = float(np.nanmax(finite_times)) if finite_times.size else 0.0
        time_in_repeat_s = (
            times_triggered
            if max_time <= repeat_duration_s * 2
            else times_triggered / cfg.sample_rate_hz
        )
    else:
        raise ValueError("time_unit must be 'seconds', 'samples', or 'auto'.")

    direction_start_s = direction_indices * direction_duration_s
    time_in_direction_s = time_in_repeat_s - direction_start_s
    valid_direction = (
        (direction_indices >= 0)
        & (direction_indices < n_directions)
        & (time_in_direction_s >= 0)
        & (time_in_direction_s < direction_duration_s)
    )
    direction_indices = direction_indices[valid_direction]
    repeat_indices = repeat_indices[valid_direction]
    time_in_direction_s = time_in_direction_s[valid_direction]

    if len(direction_indices) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(n_directions, dtype=int),
            "n_spikes": 0,
        }

    spike_counts = np.bincount(direction_indices, minlength=n_directions)
    raster_by_direction = {}
    for direction_index, direction_name in enumerate(direction_names):
        raster_by_direction[str(direction_name)] = [
            np.asarray(
                time_in_direction_s[
                    (direction_indices == direction_index)
                    & (repeat_indices == repeat_index)
                ],
                dtype=float,
            )
            for repeat_index in range(cfg.expected_repeats)
        ]

    return {
        "raster_by_direction": raster_by_direction,
        "spike_counts": spike_counts,
        "n_spikes": int(len(time_in_direction_s)),
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
        return {"peak_opposite_DSI": np.nan, "peak_preferred_direction_deg": np.nan}

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
    dsi = np.nan if denominator == 0 else (preferred_response - opposite_response) / denominator
    return {
        "peak_opposite_DSI": float(dsi),
        "peak_preferred_direction_deg": float(preferred_direction),
    }


def _direction_stats(responses, directions_deg):
    return {
        "responses": np.asarray(responses, dtype=float),
        **compute_peak_opposite_dsi(responses, directions_deg),
        **circular_vector_direction_stats(responses, directions_deg),
    }


def compute_moving_bar_psths(raster_by_direction, cfg: MovingBarFeatureConfig):
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    bins = np.arange(0, direction_duration_s + cfg.bin_size_s, cfg.bin_size_s)
    if bins[-1] < direction_duration_s:
        bins = np.append(bins, direction_duration_s)
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2

    peak_responses = np.zeros(len(cfg.direction_names), dtype=float)
    for direction_index, direction_name in enumerate(cfg.direction_names):
        direction_name = str(direction_name)
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
            psth = counts / (bin_widths * cfg.expected_repeats)
            peak_response = float(psth[int(np.argmax(psth))])
        peak_responses[direction_index] = peak_response

    return peak_responses


def analyse_moving_bar(spikes_df, cfg: MovingBarFeatureConfig):
    split = split_moving_bar_spikes(spikes_df, cfg)
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    total_time_per_direction_s = cfg.expected_repeats * direction_duration_s
    total_spike_response = split["spike_counts"] / total_time_per_direction_s
    peak_response = compute_moving_bar_psths(split["raster_by_direction"], cfg)
    return {
        "n_spikes": split["n_spikes"],
        "total_spikes": _direction_stats(total_spike_response, cfg.directions_deg),
        "peak_response": _direction_stats(peak_response, cfg.directions_deg),
    }


def extract_moving_bar_features(
    spikes_df: pd.DataFrame,
    cell_index: int,
    cfg: MovingBarFeatureConfig,
) -> dict[str, float | int] | None:
    analysis = analyse_moving_bar(spikes_df, cfg)
    if analysis["n_spikes"] == 0:
        return None

    if cfg.response_metric not in {"peak_response", "total_spikes"}:
        raise ValueError("response_metric must be 'peak_response' or 'total_spikes'.")
    metric = analysis[cfg.response_metric]

    result: dict[str, float | int] = {
        "cell_index": int(cell_index),
        "moving_bar_n_spikes": int(analysis["n_spikes"]),
        # DSI below matches the value shown in the supplied summary figure.
        "moving_bar_DSI": float(metric["peak_opposite_DSI"]),
        "moving_bar_vector_DSI": float(metric["vector_DSI"]),
        "moving_bar_OSI": float(metric["OSI"]),
        "moving_bar_preferred_direction_deg": float(metric["peak_preferred_direction_deg"]),
        "moving_bar_vector_preferred_direction_deg": float(metric["vector_preferred_direction_deg"]),
        "moving_bar_preferred_orientation_deg": float(metric["preferred_orientation_deg"]),
    }

    for direction_name, response in zip(cfg.direction_names, metric["responses"]):
        result[f"moving_bar_response_{str(direction_name).replace('-', '_')}"] = float(response)

    return result


def calculate_moving_bar_features(
    overview_path,
    stimulus_id: int,
    **config_overrides,
) -> pd.DataFrame:
    recording = Overview.Recording.load(overview_path)
    cfg = MovingBarFeatureConfig(stimulus_index=int(stimulus_id), **config_overrides)
    cells = stimulus_cell_indices(recording, stimulus_id)

    results = []
    for cell_index in tqdm(cells, desc="Extracting moving-bar features"):
        spikes = get_cell_stimulus_spikes(recording, int(cell_index), stimulus_id)
        result = extract_moving_bar_features(spikes, int(cell_index), cfg)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)
