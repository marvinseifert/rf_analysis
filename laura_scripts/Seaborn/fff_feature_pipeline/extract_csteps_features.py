from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from polarspike import Overview
from tqdm import tqdm

from fff_common import (
    box_smooth,
    contrast_index,
    get_cell_stimulus_spikes,
    mean_in_window,
    peak_in_window,
    repeated_stimulus_psth,
    stimulus_cell_indices,
)


@dataclass(frozen=True)
class CStepsFeatureConfig:
    stimulus_index: int
    duration_s: float = 40.0
    expected_repeats: int = 3
    bin_size_s: float = 0.01
    step_span_s: float = 2.0
    contrasts_percent: tuple[float, ...] = (
        100, -100,
        90, -90,
        80, -80,
        70, -70,
        60, -60,
        50, -50,
        40, -40,
        30, -30,
        20, -20,
        10, -10,
    )

    # Polarity response amplitude: mean firing rate during the first second
    # after the +100% and -100% transitions.
    polarity_response_window_s: float = 1.0

    # Baden-lab-style transient/sustained windows.
    transient_window_s: tuple[float, float] = (0.080, 0.160)
    sustained_window_s: tuple[float, float] = (0.240, 2.000)
    smoothing_window_s: float = 0.040


def _window_relative_to_transition(
    transition_s: float,
    relative_window: tuple[float, float],
) -> tuple[float, float]:
    return (
        transition_s + relative_window[0],
        transition_s + relative_window[1],
    )


def extract_csteps_features(
    spikes_df: pd.DataFrame,
    cell_index: int,
    cfg: CStepsFeatureConfig,
) -> dict[str, float | int] | None:
    """
    Extract contrast-step polarity and kinetics for one cell.

    Polarity index:
        (A_ON - A_OFF) / (A_ON + A_OFF)

    Transience index:
        (A_transient - A_sustained) / (A_transient + A_sustained)

    Transient/sustained amplitudes are peak 40-ms-box-smoothed firing rates
    80-160 ms and 240-2000 ms after the transition, respectively.
    """
    analysis = repeated_stimulus_psth(
        spikes_df=spikes_df,
        duration_s=cfg.duration_s,
        expected_repeats=cfg.expected_repeats,
        bin_size_s=cfg.bin_size_s,
    )

    if analysis["n_spikes"] == 0:
        return None

    times = np.asarray(analysis["bin_centers"], dtype=float)
    psth = np.asarray(analysis["psth"], dtype=float)
    smoothed = box_smooth(psth, cfg.bin_size_s, cfg.smoothing_window_s)

    results: dict[str, float | int] = {
        "cell_index": int(cell_index),
        "csteps_n_spikes": int(analysis["n_spikes"]),
    }

    # Save the response curve for every contrast step. This is the mean firing
    # rate during the first polarity_response_window_s after each transition.
    for step_index, contrast in enumerate(cfg.contrasts_percent):
        transition_s = step_index * cfg.step_span_s
        response = mean_in_window(
            times,
            psth,
            transition_s,
            min(
                transition_s + cfg.polarity_response_window_s,
                transition_s + cfg.step_span_s,
            ),
        )
        sign = "pos" if contrast > 0 else "neg"
        magnitude = int(abs(contrast)) if float(abs(contrast)).is_integer() else abs(contrast)
        results[f"csteps_{sign}_{magnitude}_response_hz"] = response

    # The supplied stimulus begins +100%, then -100%.
    on_transition_s = 0.0
    off_transition_s = cfg.step_span_s

    on_response = mean_in_window(
        times,
        psth,
        on_transition_s,
        on_transition_s + cfg.polarity_response_window_s,
    )
    off_response = mean_in_window(
        times,
        psth,
        off_transition_s,
        off_transition_s + cfg.polarity_response_window_s,
    )

    on_tr_start, on_tr_end = _window_relative_to_transition(
        on_transition_s, cfg.transient_window_s
    )
    on_sus_start, on_sus_end = _window_relative_to_transition(
        on_transition_s, cfg.sustained_window_s
    )
    off_tr_start, off_tr_end = _window_relative_to_transition(
        off_transition_s, cfg.transient_window_s
    )
    off_sus_start, off_sus_end = _window_relative_to_transition(
        off_transition_s, cfg.sustained_window_s
    )

    on_transient = peak_in_window(times, smoothed, on_tr_start, on_tr_end)
    on_sustained = peak_in_window(times, smoothed, on_sus_start, on_sus_end)
    off_transient = peak_in_window(times, smoothed, off_tr_start, off_tr_end)
    off_sustained = peak_in_window(times, smoothed, off_sus_start, off_sus_end)

    polarity_index = contrast_index(on_response, off_response)
    on_ti = contrast_index(on_transient, on_sustained)
    off_ti = contrast_index(off_transient, off_sustained)

    # Compound TI follows whichever polarity dominates the 100% response.
    if np.isfinite(polarity_index):
        compound_ti = on_ti if polarity_index >= 0 else off_ti
    else:
        compound_ti = np.nan

    results.update(
        {
            "csteps_on_response_hz": on_response,
            "csteps_off_response_hz": off_response,
            "csteps_polarity_index": polarity_index,
            "csteps_on_transient_hz": on_transient,
            "csteps_on_sustained_hz": on_sustained,
            "csteps_off_transient_hz": off_transient,
            "csteps_off_sustained_hz": off_sustained,
            "csteps_on_transience_index": on_ti,
            "csteps_off_transience_index": off_ti,
            "csteps_compound_transience_index": compound_ti,
        }
    )

    return results


def calculate_csteps_features(
    overview_path,
    stimulus_id: int,
    **config_overrides,
) -> pd.DataFrame:
    recording = Overview.Recording.load(overview_path)
    cfg = CStepsFeatureConfig(stimulus_index=int(stimulus_id), **config_overrides)
    cells = stimulus_cell_indices(recording, stimulus_id)

    results = []
    for cell_index in tqdm(cells, desc="Extracting contrast-step features"):
        spikes = get_cell_stimulus_spikes(recording, int(cell_index), stimulus_id)
        result = extract_csteps_features(spikes, int(cell_index), cfg)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)
