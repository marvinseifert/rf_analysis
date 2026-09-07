from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from polarspike import Overview, histograms
from tqdm import tqdm


# %% ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def moving_average(a, n=3):
    ret = np.cumsum(
        a,
        dtype=float,
    )

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1 :] / n


# %% ------------------------------------------------------------
# Generate chirp stimulus
# ------------------------------------------------------------


def generate_chirp_data(
    start_freq,
    end_freq,
    duration,
    refresh_rate,
    max_power=4095,
):
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

    chirp_signal = ((raw_sine + 1.0) * max_power / 2.0).astype(int)

    return (
        t,
        chirp_signal,
        freqs,
    )


# %% ------------------------------------------------------------
# Build full stimulus-frequency trace
# ------------------------------------------------------------


def build_chirp_frequency_trace(
    start_freq=1,
    end_freq=30,
    chirp_duration=30,
    total_duration=35,
    refresh_rate=300,
    chirp_offset=3,
):
    """
    Build the full chirp stimulus-frequency trace.

    The returned trace is sampled at the stimulus refresh rate.

    Default:
        300 Hz stimulus trace
        35 s total duration
        chirp starts at 3 s
        chirp goes from 1 Hz -> 30 Hz
    """

    _, _, freq_actual = generate_chirp_data(
        start_freq=start_freq,
        end_freq=end_freq,
        duration=chirp_duration,
        refresh_rate=refresh_rate,
    )

    total_samples = int(total_duration * refresh_rate) - 1

    c_freqs = np.zeros(
        total_samples,
        dtype=float,
    )

    offset = int(chirp_offset * refresh_rate)

    end_idx = offset + len(freq_actual)

    limit = min(
        len(c_freqs),
        end_idx,
    )

    insert_len = limit - offset

    if insert_len > 0:
        c_freqs[offset:limit] = freq_actual[:insert_len]

    return c_freqs


# %% ------------------------------------------------------------
# Resample chirp-frequency trace to PSTH timebase
# ------------------------------------------------------------


def resample_chirp_frequency_trace(
    c_freqs,
    signal_length,
    bin_size,
    c_freqs_refresh_rate=300,
):
    """
    Resample the original stimulus-frequency trace onto the PSTH/CWT
    timebase.

    This is essential when:

        stimulus trace = 300 Hz
        PSTH bin size   = 10 ms = 100 Hz

    Without this step, the first 3500 samples of the original
    10500-sample stimulus trace are incorrectly treated as the
    full 35 s recording.
    """

    c_freqs = np.asarray(
        c_freqs,
        dtype=float,
    )

    # Original stimulus timebase

    source_time = np.arange(
        len(c_freqs),
        dtype=float,
    ) / float(c_freqs_refresh_rate)

    # PSTH / CWT timebase

    target_time = np.arange(
        signal_length,
        dtype=float,
    ) * float(bin_size)

    # Interpolate stimulus frequency
    # onto PSTH time points

    c_freqs_resampled = np.interp(
        target_time,
        source_time,
        c_freqs,
        left=0.0,
        right=0.0,
    )

    return c_freqs_resampled


# %% ------------------------------------------------------------
# Extract chirp features for ONE CELL
# ------------------------------------------------------------


def extract_chirp_features(
    spikes,
    cell_index,
    stimulus_index,
    c_freqs,
    window_end=35,
    # --------------------------------------------------------
    # Faster PSTH
    # --------------------------------------------------------
    #
    # Was:
    #     0.01 / 3
    #
    # Now:
    #     0.01 = 10 ms
    bin_size=0.01,
    wavelet="cmor3.0-1.5",
    # --------------------------------------------------------
    # Faster wavelet resolution
    # --------------------------------------------------------
    #
    # Was:
    #     300
    #
    # Now:
    #     100
    n_freqs=100,
    widths_min=1,
    widths_max=1024,
    repeat_to_use=2,
    # --------------------------------------------------------
    # Search range
    # --------------------------------------------------------
    #
    # Was:
    #     150 with 300 frequency bins
    #
    # Now:
    #     50 with 100 frequency bins
    #
    # This keeps approximately the same relative search range.
    search_shift=50,
    # --------------------------------------------------------
    # Smoothing
    # --------------------------------------------------------
    #
    # Original:
    #
    #     400 bins
    #     × 3.33 ms
    #     ≈ 1.33 s
    #
    # New:
    #
    #     133 bins
    #     × 10 ms
    #     ≈ 1.33 s
    smooth_n=133,
    threshold_frac=0.10,
    # Original stimulus trace sampling rate
    c_freqs_refresh_rate=300,
    # FFT CWT is faster for these relatively long traces
    cwt_method="fft",
    # Shared cache used across cells
    analysis_cache=None,
):
    """
    Extract chirp features for one cell.

    Main outputs:

        chirp_power_max_frequency

            Stimulus frequency at which wavelet power is maximal.


        chirp_threshold_frequency

            Highest stimulus frequency at which smoothed wavelet
            power remains above 10% of maximum.


    Important changes from the previous version:

        1. The 300 Hz stimulus-frequency trace is resampled onto
           the PSTH timebase.

        2. PSTH uses 10 ms bins.

        3. CWT uses 100 scales instead of 300.

        4. Smoothing remains approximately 1.33 s.

        5. Frequency search range is scaled from 150 -> 50.

        6. CWT uses FFT convolution.

        7. Frequency-to-CWT-row mapping is cached across cells.

        8. Search around the expected chirp trajectory is
           vectorized rather than looped in Python.
    """

    # --------------------------------------------------------
    # Select cell
    # --------------------------------------------------------

    cell_spikes = spikes.query(
        "cell_index == @cell_index " "& stimulus_index == @stimulus_index"
    )

    if cell_spikes.empty:
        return None

    # --------------------------------------------------------
    # PSTH
    # --------------------------------------------------------

    psth, bins, repeat = histograms.psth_by_index(
        cell_spikes,
        index=["repeat"],
        bin_size=bin_size,
        window_end=window_end,
        return_idx=True,
    )

    if psth.shape[0] <= repeat_to_use:
        return None

    # --------------------------------------------------------
    # Spike signal
    # --------------------------------------------------------

    signal = psth[repeat_to_use] - psth[repeat_to_use].mean()

    # --------------------------------------------------------
    # Wavelet scales
    # --------------------------------------------------------

    widths = np.geomspace(
        widths_min,
        widths_max,
        num=n_freqs,
    )

    # --------------------------------------------------------
    # Continuous wavelet transform
    # --------------------------------------------------------

    cwtmatr, freqs = pywt.cwt(
        signal,
        widths,
        wavelet,
        sampling_period=bin_size,
        method=cwt_method,
    )

    # Preserve the previous trimming behaviour

    cwtmatr = np.abs(
        cwtmatr[
            :-1,
            :-1,
        ]
    )

    freqs_trim = freqs[:-1]

    if cwtmatr.shape[0] == 0 or cwtmatr.shape[1] == 0:
        return None

    # %% --------------------------------------------------------
    # Prepare frequency mapping
    # --------------------------------------------------------
    #
    # This mapping is identical for every cell in the recording,
    # so it only needs to be calculated once.

    if analysis_cache is None:
        analysis_cache = {}

    cache_key = (
        len(signal),
        float(bin_size),
        len(c_freqs),
        float(c_freqs_refresh_rate),
        str(wavelet),
        int(n_freqs),
        float(widths_min),
        float(widths_max),
        cwtmatr.shape[0],
        cwtmatr.shape[1],
    )

    if analysis_cache.get("cache_key") != cache_key:
        # ----------------------------------------------------
        # Correct stimulus-frequency timebase
        # ----------------------------------------------------

        c_freqs_resampled = resample_chirp_frequency_trace(
            c_freqs=c_freqs,
            signal_length=len(signal),
            bin_size=bin_size,
            c_freqs_refresh_rate=(c_freqs_refresh_rate),
        )

        # ----------------------------------------------------
        # Match CWT and stimulus lengths
        # ----------------------------------------------------

        T = min(
            cwtmatr.shape[1],
            len(c_freqs_resampled),
        )

        c_freqs_use = c_freqs_resampled[:T]

        # ----------------------------------------------------
        # Find nearest CWT frequency row
        #
        # OLD VERSION:
        #
        # for every time point:
        #     np.argmin(...)
        #
        # NEW VERSION:
        #
        # Calculate everything in one NumPy operation.
        # ----------------------------------------------------

        base_pos = np.argmin(
            np.abs(
                freqs_trim[
                    :,
                    None,
                ]
                - c_freqs_use[None, :]
            ),
            axis=0,
        ).astype(int)

        # ----------------------------------------------------
        # Store for all subsequent cells
        # ----------------------------------------------------

        analysis_cache.clear()

        analysis_cache.update(
            {
                "cache_key": cache_key,
                "T": T,
                "c_freqs_use": c_freqs_use,
                "base_pos": base_pos,
            }
        )

    # --------------------------------------------------------
    # Retrieve cached mapping
    # --------------------------------------------------------

    T = analysis_cache["T"]

    c_freqs_use = analysis_cache["c_freqs_use"]

    base_pos = analysis_cache["base_pos"]

    if T == 0:
        return None

    # %% --------------------------------------------------------
    # Search around expected chirp trajectory
    # --------------------------------------------------------
    #
    # Original version searched:
    #
    #     base_pos + shift
    #     base_pos - shift
    #
    # inside a Python loop.
    #
    # This performs exactly the same type of search but constructs
    # all candidate trajectories at once.

    if search_shift < 1:
        search_shift = 1

    positive_shifts = np.arange(
        search_shift,
        dtype=int,
    )

    negative_shifts = -np.arange(
        1,
        search_shift,
        dtype=int,
    )

    all_shifts = np.concatenate(
        [
            positive_shifts,
            negative_shifts,
        ]
    )

    # --------------------------------------------------------
    # Candidate CWT rows
    # --------------------------------------------------------

    candidate_positions = base_pos[None, :] + all_shifts[:, None]

    # Keep rows inside CWT bounds

    candidate_positions = np.clip(
        candidate_positions,
        0,
        cwtmatr.shape[0] - 1,
    )

    # --------------------------------------------------------
    # Time positions
    # --------------------------------------------------------

    time_positions = np.arange(
        T,
        dtype=int,
    )

    # --------------------------------------------------------
    # Extract every candidate trace
    # --------------------------------------------------------

    candidate_traces = cwtmatr[
        candidate_positions,
        time_positions[None, :],
    ]

    # --------------------------------------------------------
    # Find trajectory with greatest total power
    # --------------------------------------------------------

    candidate_totals = np.sum(
        candidate_traces,
        axis=1,
    )

    best_candidate = int(np.argmax(candidate_totals))

    max_power = candidate_traces[best_candidate].copy()

    # --------------------------------------------------------
    # Remove periods outside chirp
    # --------------------------------------------------------

    max_power[c_freqs_use == 0] = 0

    abs_power = np.abs(max_power)

    if len(abs_power) < smooth_n:
        return None

    # %% --------------------------------------------------------
    # Maximum-power frequency
    # --------------------------------------------------------

    power_max_position = int(np.argmax(abs_power))

    power_max = float(abs_power[power_max_position])

    power_max_frequency = float(c_freqs_use[power_max_position])

    # %% --------------------------------------------------------
    # Smooth power trace
    # --------------------------------------------------------

    filtered = moving_average(
        abs_power,
        smooth_n,
    )

    # --------------------------------------------------------
    # 10% power threshold
    # --------------------------------------------------------

    threshold = power_max * threshold_frac

    filtered_threshold_positions = np.where(filtered > threshold)[0]

    if len(filtered_threshold_positions) == 0:
        return None

    # --------------------------------------------------------
    # Highest point above threshold
    # --------------------------------------------------------

    filtered_threshold_position = int(filtered_threshold_positions[-1])

    # Account for moving-average shortening

    threshold_position = filtered_threshold_position + smooth_n - 1

    threshold_position = min(
        threshold_position,
        len(c_freqs_use) - 1,
    )

    threshold_frequency = float(c_freqs_use[threshold_position])

    # %% --------------------------------------------------------
    # Power after threshold
    # --------------------------------------------------------

    threshold_power = np.sum(abs_power[threshold_position:])

    # --------------------------------------------------------
    # Preserve previous power-ratio calculation
    # --------------------------------------------------------

    if threshold_power == 0:
        power_ratio = np.nan

    else:
        total_power = np.sum(abs_power)

        power_ratio = (
            np.sum(abs_power[:threshold_position]) / threshold_power
        ) / total_power

    # %% --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "cell_index": int(cell_index),
        "chirp_stimulus_index": int(stimulus_index),
        "chirp_power_max_frequency": power_max_frequency,
        "chirp_threshold_frequency": threshold_frequency,
        "chirp_power_ratio": float(power_ratio),
        "chirp_power_max": power_max,
        "chirp_power_max_position": power_max_position,
        "chirp_threshold_position": threshold_position,
        "chirp_n_spikes": int(len(cell_spikes)),
    }


# %% ------------------------------------------------------------
# Calculate chirp features for ALL CELLS
# ------------------------------------------------------------


def calculate_chirp_features(
    overview_path,
    stimulus_id,
    c_freqs=None,
    **extract_overrides,
):
    """
    Calculate chirp features for every cell in one recording.
    """

    # --------------------------------------------------------
    # Generate stimulus-frequency trace
    # --------------------------------------------------------

    if c_freqs is None:
        c_freqs = build_chirp_frequency_trace()

    # --------------------------------------------------------
    # Load recording
    # --------------------------------------------------------

    recording = Overview.Recording.load(overview_path)

    # --------------------------------------------------------
    # Select chirp stimulus
    # --------------------------------------------------------

    recording.dataframes["chirps"] = recording.spikes_df.loc[
        recording.spikes_df["stimulus_index"] == stimulus_id
    ].copy()

    spikes = recording.get_spikes_df(
        "chirps",
        carry=["stimulus_name"],
    )

    if spikes.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Cell IDs
    # --------------------------------------------------------

    cell_indices = np.sort(spikes["cell_index"].dropna().astype(int).unique())

    results = []

    # --------------------------------------------------------
    # Shared analysis cache
    # --------------------------------------------------------
    #
    # The frequency mapping only depends on:
    #
    #     PSTH bin size
    #     wavelet
    #     wavelet scales
    #     chirp frequency trace
    #
    # It does NOT depend on cell identity.
    #
    # Therefore it is calculated for the first cell and reused.

    analysis_cache = {}

    # --------------------------------------------------------
    # Extract all cells
    # --------------------------------------------------------

    for cell_index in tqdm(
        cell_indices,
        desc="Extracting chirp features",
        unit="cell",
    ):
        result = extract_chirp_features(
            spikes=spikes,
            cell_index=int(cell_index),
            stimulus_index=stimulus_id,
            c_freqs=c_freqs,
            analysis_cache=analysis_cache,
            **extract_overrides,
        )

        if result is not None:
            results.append(result)

    # --------------------------------------------------------
    # Return one row per cell
    # --------------------------------------------------------

    return pd.DataFrame(results)
