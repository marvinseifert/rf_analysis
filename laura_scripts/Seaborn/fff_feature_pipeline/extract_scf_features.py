from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from polarspike import Overview
from tqdm import tqdm

from fff_common import (
    get_cell_stimulus_spikes,
    mean_in_window,
    repeated_stimulus_psth,
    stimulus_cell_indices,
)


@dataclass(frozen=True)
class SCFFeatureConfig:
    stimulus_index: int
    wavelengths_nm: tuple[int, ...] = (660, 610, 560, 535, 500, 460, 420, 365)
    expected_repeats: int = 5
    stimulus_span_s: float = 2.0
    bin_size_s: float = 0.01
    response_window_s: float = 1.0

    # A discrete spectral tuning is called opponent when both response signs are
    # present and the weaker polarity reaches at least this fraction of the
    # dominant polarity. 0.10 mirrors the 10% criterion used in Baden-lab
    # spectral-opponency analyses, but here it is applied to a spike-based SCF
    # ON-minus-OFF tuning proxy rather than a calcium spectral kernel.
    opponency_min_fraction: float = 0.10

    @property
    def duration_s(self) -> float:
        # Each wavelength is followed by one equally long dark period.
        return len(self.wavelengths_nm) * 2 * self.stimulus_span_s


def _opponency_metrics(
    tuning: np.ndarray,
    threshold_fraction: float,
) -> tuple[int, float, int]:
    tuning = np.asarray(tuning, dtype=float)
    finite = tuning[np.isfinite(tuning)]
    if finite.size == 0:
        return 0, np.nan, 0

    max_pos = float(np.max(finite))
    min_neg = float(np.min(finite))
    positive = max_pos > 0
    negative = min_neg < 0

    if not (positive and negative):
        return 0, 0.0, 0

    pos_amp = max_pos
    neg_amp = abs(min_neg)
    dominant = max(pos_amp, neg_amp)
    weaker = min(pos_amp, neg_amp)
    strength = weaker / dominant if dominant > 0 else np.nan
    opponent = int(np.isfinite(strength) and strength >= threshold_fraction)

    # Count sign changes across wavelength order after ignoring exact zeros/NaNs.
    signs = np.sign(tuning[np.isfinite(tuning) & (tuning != 0)])
    zero_crossings = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    return opponent, float(strength), zero_crossings


def extract_scf_features(
    spikes_df: pd.DataFrame,
    cell_index: int,
    cfg: SCFFeatureConfig,
) -> dict[str, float | int] | None:
    """
    Extract spike-based spectral features from single chromatic flashes.

    For each wavelength:
      ON response  = mean firing rate after colour onset
      OFF response = mean firing rate after transition back to dark
      signed tuning = ON - OFF
      amplitude = max(ON, OFF)

    The signed ON-minus-OFF tuning allows a simple colour-opponency descriptor
    while retaining the raw ON and OFF amplitudes separately.
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

    results: dict[str, float | int] = {
        "cell_index": int(cell_index),
        "scf_n_spikes": int(analysis["n_spikes"]),
    }

    signed_tuning = []
    amplitudes = []

    block_s = 2 * cfg.stimulus_span_s
    for wavelength_index, wavelength_nm in enumerate(cfg.wavelengths_nm):
        colour_onset_s = wavelength_index * block_s
        dark_onset_s = colour_onset_s + cfg.stimulus_span_s

        on_response = mean_in_window(
            times,
            psth,
            colour_onset_s,
            min(colour_onset_s + cfg.response_window_s, dark_onset_s),
        )
        off_response = mean_in_window(
            times,
            psth,
            dark_onset_s,
            min(
                dark_onset_s + cfg.response_window_s,
                colour_onset_s + block_s,
            ),
        )

        signed = (
            float(on_response - off_response)
            if np.isfinite(on_response) and np.isfinite(off_response)
            else np.nan
        )
        amplitude = (
            float(max(on_response, off_response))
            if np.isfinite(on_response) and np.isfinite(off_response)
            else np.nan
        )

        signed_tuning.append(signed)
        amplitudes.append(amplitude)

        prefix = f"scf_{int(wavelength_nm)}nm"
        results[f"{prefix}_on_response_hz"] = on_response
        results[f"{prefix}_off_response_hz"] = off_response
        results[f"{prefix}_signed_tuning_hz"] = signed
        results[f"{prefix}_amplitude_hz"] = amplitude

    signed_tuning_arr = np.asarray(signed_tuning, dtype=float)
    amplitudes_arr = np.asarray(amplitudes, dtype=float)

    if np.any(np.isfinite(amplitudes_arr)):
        preferred_idx = int(np.nanargmax(amplitudes_arr))
        preferred_wavelength = float(cfg.wavelengths_nm[preferred_idx])
        preferred_amplitude = float(amplitudes_arr[preferred_idx])
    else:
        preferred_wavelength = np.nan
        preferred_amplitude = np.nan

    opponent, opponency_strength, zero_crossings = _opponency_metrics(
        signed_tuning_arr,
        cfg.opponency_min_fraction,
    )

    results.update(
        {
            "scf_preferred_wavelength_nm": preferred_wavelength,
            "scf_preferred_amplitude_hz": preferred_amplitude,
            "scf_colour_opponent": int(opponent),
            "scf_opponency_strength": opponency_strength,
            "scf_zero_crossings": int(zero_crossings),
        }
    )

    return results


def calculate_scf_features(
    overview_path,
    stimulus_id: int,
    **config_overrides,
) -> pd.DataFrame:
    recording = Overview.Recording.load(overview_path)
    cfg = SCFFeatureConfig(stimulus_index=int(stimulus_id), **config_overrides)
    cells = stimulus_cell_indices(recording, stimulus_id)

    results = []
    for cell_index in tqdm(cells, desc="Extracting SCF features"):
        spikes = get_cell_stimulus_spikes(recording, int(cell_index), stimulus_id)
        result = extract_scf_features(spikes, int(cell_index), cfg)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)
