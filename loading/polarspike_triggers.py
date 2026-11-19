"""
loading/polarspike_triggers.py — work around incomplete last repeats in polarspike.

polarspike's get_spikes_triggered() assumes every repeat of a stimulus is complete:
it needs n_repeats * stimulus_repeat_logic + 1 triggers (one per frame plus the end).
If a few triggers are missing at the end (e.g. 15/10/2026 moving bar: 23,999 triggers
instead of 24,001), it stops with a polars ShapeError ("series 'start' has length ...
while series 'sub_t' has length ...").

Both functions change the loaded recording in memory only; the overview file on the
NAS is never changed.

pad_to_repeats(recording, stimulus_index, n_repeats)
    For stimuli that are always played a known number of times (the moving bar is
    always played 10 times): adds the missing triggers at the end, spaced by the
    stimulus's median trigger interval (one frame), so all repeats are kept.

drop_incomplete_repeats(recording, stimulus_index)
    For other stimuli: trims to the complete repeats.

    from loading.polarspike_triggers import pad_to_repeats
    recording = Overview.Recording.load(overview_path)
    pad_to_repeats(recording, stimulus_index=3, n_repeats=10)
"""
from __future__ import annotations

import numpy as np


def _stimulus_row(recording, stimulus_index: int):
    stimulus_df = recording.dataframes["stimulus_df"]
    rows = stimulus_df.index[stimulus_df["stimulus_index"] == stimulus_index]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one stimulus_df row for stimulus {stimulus_index}, found {len(rows)}"
        )
    return stimulus_df, rows[0]


def _set_triggers(stimulus_df, row, triggers: np.ndarray, n_repeats: int) -> None:
    """Store a new trigger list and the fields polarspike derives from it."""
    stimulus_df.at[row, "trigger_fr_relative"] = triggers
    if "trigger_int" in stimulus_df.columns:
        stimulus_df.at[row, "trigger_int"] = np.diff(triggers)
    # trigger_ends is left as it is: it is not always triggers[1:] (15/10 moving bar has
    # 21,600 entries), and get_spikes_triggered does not use it.
    stimulus_df.at[row, "end_fr"] = int(stimulus_df.at[row, "begin_fr"]) + int(triggers[-1])
    if "nr_repeats" in stimulus_df.columns:
        stimulus_df.at[row, "nr_repeats"] = n_repeats


def pad_to_repeats(recording, stimulus_index: int, n_repeats: int, verbose: bool = True) -> int:
    """Add missing end triggers so stimulus `stimulus_index` has `n_repeats` full repeats.

    The added triggers are spaced by the median interval of the recorded triggers.
    Returns n_repeats. Does nothing if the triggers already fill n_repeats repeats.
    Raises if more than one repeat's worth of triggers would have to be invented, or if
    there are already more triggers than n_repeats repeats need.
    """
    stimulus_df, row = _stimulus_row(recording, stimulus_index)
    triggers = np.asarray(stimulus_df.at[row, "trigger_fr_relative"])
    logic = int(stimulus_df.at[row, "stimulus_repeat_logic"])
    n_needed = n_repeats * logic + 1
    n_missing = n_needed - len(triggers)

    if n_missing == 0:
        return n_repeats
    if n_missing < 0:
        raise ValueError(
            f"Stimulus {stimulus_index}: {len(triggers)} triggers is more than {n_repeats} "
            f"repeats of {logic} need ({n_needed}); check n_repeats."
        )
    if n_missing > logic:
        raise ValueError(
            f"Stimulus {stimulus_index}: {n_missing} triggers missing (more than one repeat "
            f"of {logic}); not padding. Check that it really was played {n_repeats} times."
        )

    interval = int(round(np.median(np.diff(triggers))))
    added = triggers[-1] + interval * np.arange(1, n_missing + 1)
    padded = np.concatenate([triggers, added.astype(triggers.dtype)])

    # Warn if the padded end runs into the next stimulus.
    begin = int(stimulus_df.at[row, "begin_fr"])
    later = stimulus_df[stimulus_df["begin_fr"] > begin]["begin_fr"]
    if len(later) and begin + int(padded[-1]) > int(later.min()):
        print(
            f"Warning: stimulus {stimulus_index}: padded end overlaps the next stimulus by "
            f"{begin + int(padded[-1]) - int(later.min())} samples."
        )

    _set_triggers(stimulus_df, row, padded, n_repeats)
    if verbose:
        sample_rate = float(stimulus_df.at[row, "sampling_freq"]) if "sampling_freq" in stimulus_df.columns else np.nan
        print(
            f"Stimulus {stimulus_index}: {len(triggers)} triggers, {n_needed} needed for "
            f"{n_repeats} repeats of {logic}; added {n_missing} at the end, "
            f"{interval} samples apart ({interval / sample_rate * 1000:.1f} ms) "
            f"(in memory only)."
        )
    return n_repeats


def drop_incomplete_repeats(recording, stimulus_index: int, verbose: bool = True) -> int:
    """Trim stimulus `stimulus_index` to its complete repeats (in memory).

    Returns the number of complete repeats. Does nothing if the triggers already fit
    whole repeats (n_triggers == n_repeats * logic + 1).
    """
    stimulus_df, row = _stimulus_row(recording, stimulus_index)
    triggers = np.asarray(stimulus_df.at[row, "trigger_fr_relative"])
    logic = int(stimulus_df.at[row, "stimulus_repeat_logic"])
    n_complete = (len(triggers) - 1) // logic
    n_keep = n_complete * logic + 1
    if len(triggers) == n_keep:
        return n_complete
    if n_complete < 1:
        raise ValueError(
            f"Stimulus {stimulus_index}: only {len(triggers)} triggers, less than one "
            f"complete repeat of {logic}."
        )

    _set_triggers(stimulus_df, row, triggers[:n_keep], n_complete)
    if verbose:
        print(
            f"Stimulus {stimulus_index}: {len(triggers)} triggers do not fill whole repeats of "
            f"{logic}; using the {n_complete} complete repeats ({n_keep} triggers), "
            f"dropping {len(triggers) - n_keep} (in memory only)."
        )
    return n_complete
