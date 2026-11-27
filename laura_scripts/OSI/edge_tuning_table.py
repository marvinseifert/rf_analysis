"""
Moving-edge direction / orientation tuning for every cell, with a chance test (23/11/2026).

Recordings of the cone-alignment set that played the weighted moving edges (8 directions, 800 x 800 px):
  22/07/2026 P0: ON 13 fps (stim 4, x5), OFF 13 fps (stim 6, x5), ON 60 fps (stim 7, x3), OFF 60 fps (stim 8, x3)
                 (20 px edges assumed; notes say lots of dropped frames in all moving-bar stimuli that day)
  14/08/2026 P0: ON 13 fps (stim 15, x3), OFF 13 fps (stim 16, x3), 20 px
  07/08/2026 P0: ON 60 fps (stim 12 in the notes = stimulus_index 11, x4), 20 px; the OFF run crashed
ON = moving_bar_weighted_*_ON_*_20px file, OFF = moving_edge_weighted_*_OFF_*_20px file (as moving_edge_analysis.py).

Uses moving_edge_analysis.py unchanged (exact direction timing from the H5 file, date-dependent H5 -> projected
direction table, spike splitting per direction and repeat). Response per direction = peak of the repeat-averaged
PSTH (0.25 s bins) as in that script. Orientation vector (doubled angles) -> OSI and preferred motion AXIS
(polar frame: 0° = up, clockwise; the axis the edge moves along); direction vector -> DSI and preferred direction.

Chance test: direction labels permuted independently within each repeat (--n-shuffle times), PSTH peaks and
OSI / DSI recomputed; p = fraction of shuffles >= real.

Output: Box Figures/OSI/edge_tuning_table.csv (one row per cell x condition).
Run from the repo root (~10-15 min):
    poetry run python laura_scripts/OSI/edge_tuning_table.py
"""
import argparse
import ast
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n-shuffle", type=int, default=500)
parser.add_argument("--bin-s", type=float, default=0.25)
args = parser.parse_args()

REPO = Path(__file__).resolve().parents[2]
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/OSI")
STIM = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026")
ON = {13: "moving_bar_weighted_13fps_ON_800px_wide_800px_high_20px_thick_30.0px_ramp_1.0px_per_frame.h5",
      60: "moving_bar_weighted_60fps_ON_800px_wide_800px_high_20px_thick_30.0px_ramp_1.0px_per_frame.h5"}
OFF = {13: "moving_edge_weighted_13fps_OFF_800px_wide_800px_high__20px_thick_30.0px_ramp_1.0px_per_frame.h5",
       60: "moving_edge_weighted_60fps_OFF_800px_wide_800px_high__20px_thick_30.0px_ramp_1.0px_per_frame.h5"}
RECS = {  # recording: [(condition, stimulus_index, h5, repeats, polarity, fps)]
    "22_07_2026": [("ON_13fps", 4, ON[13], 5, "ON", 13), ("OFF_13fps", 6, OFF[13], 5, "OFF", 13),
                   ("ON_60fps", 7, ON[60], 3, "ON", 60), ("OFF_60fps", 8, OFF[60], 3, "OFF", 60)],
    "14_08_2026": [("ON_13fps", 15, ON[13], 3, "ON", 13), ("OFF_13fps", 16, OFF[13], 3, "OFF", 13)],
    "07_08_2026": [("ON_60fps", 11, ON[60], 4, "ON", 60)],
}

# ---- moving_edge_analysis.py: imports, classes, functions and constants above its example section ----
src = REPO / "laura_scripts" / "OSI" / "moving_edge_analysis.py"
text = src.read_text()
tree = ast.parse(text)
example_line = next(n.lineno for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "NOISE_DATA_PATH" for t in n.targets))
keep = [n for n in tree.body if n.lineno < example_line and isinstance(
    n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))]
_mod = types.ModuleType("moving_edge_analysis_functions")
sys.modules[_mod.__name__] = _mod  # dataclasses look their module up in sys.modules
exec(compile(ast.Module(body=keep, type_ignores=[]), str(src), "exec"), _mod.__dict__)
ns = _mod.__dict__
MovingBarConditionConfig = ns["MovingBarConditionConfig"]
load_stimulus_timing, split_moving_bar_spikes = ns["load_stimulus_timing"], ns["split_moving_bar_spikes"]
DIRECTION_NAMES, DIRECTIONS_DEG = ns["DIRECTION_NAMES"], ns["DIRECTIONS_DEG"]
pad_to_repeats, Overview, _to_pandas = ns["pad_to_repeats"], ns["Overview"], ns["_to_pandas"]

THETA = np.deg2rad(DIRECTIONS_DEG)


def vector_stats(resp):
    """resp (..., 8) -> OSI, preferred axis, DSI, preferred direction (polar frame)."""
    tot = resp.sum(-1)
    o = (resp * np.exp(2j * THETA)).sum(-1)
    d = (resp * np.exp(1j * THETA)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        osi, dsi = np.abs(o) / tot, np.abs(d) / tot
    return osi, (np.rad2deg(np.angle(o)) / 2) % 180, dsi, np.rad2deg(np.angle(d)) % 360


rows = []
for rec_name, conds in RECS.items():
    overview = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec_name}/Phase_00/overview")
    recording = Overview.Recording.load(str(overview))
    for cname, stim_index, h5, n_rep, pol, fps in conds:
        cond = MovingBarConditionConfig(title=cname, stimulus_index=stim_index, h5_path=STIM / h5,
                                        expected_repeats=n_rep, expected_polarity=pol, expected_frame_rate_hz=fps,
                                        time_unit="seconds", bin_size_s=args.bin_s)
        try:
            pad_to_repeats(recording, stim_index, n_rep)
        except ValueError as exc:  # e.g. 07/08: a few more triggers than n_rep full repeats (dropped frames?)
            print(f"WARNING {rec_name} {cname}: {exc} -> using the triggers as they are", flush=True)
        timing = load_stimulus_timing(cond.h5_path, str(overview))
        dur = {timing.plot_direction_names[i]: float(d) for i, d in enumerate(timing.direction_duration_s)}
        n_bins = int(np.ceil(max(dur.values()) / args.bin_s))
        try:
            spikes_all = _to_pandas(recording.get_spikes_triggered([{"stimulus_index": [stim_index]}]))
        except Exception as exc:  # 07/08: trigger count does not fit whole repeats -> polarspike cannot split it
            print(f"SKIPPED {rec_name} {cname}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        rng = np.random.default_rng(stim_index)
        n_done = 0
        for cell, sp in spikes_all.groupby("cell_index"):
            split = split_moving_bar_spikes(sp, cond, timing, DIRECTION_NAMES)
            counts = np.zeros((n_rep, len(DIRECTION_NAMES), n_bins))
            for j, name in enumerate(DIRECTION_NAMES):
                for r, times in enumerate(split["raster_by_direction"][str(name)][:n_rep]):
                    if len(times):
                        h, _ = np.histogram(times, bins=n_bins, range=(0, n_bins * args.bin_s))
                        counts[r, j] = h
            n_sp = int(counts.sum())
            if n_sp == 0:
                continue
            peak = counts.mean(0).max(-1) / args.bin_s  # peak of repeat-averaged PSTH per direction (Hz)
            osi, axis, dsi, pdir = vector_stats(peak)
            # chance: permute direction labels independently in every repeat
            perm = np.array([rng.permuted(np.tile(np.arange(8), (n_rep, 1)), axis=1) for _ in range(args.n_shuffle)])
            shuf = counts[np.arange(n_rep)[None, :, None], perm]  # (n_shuffle, rep, dir, bins)
            speak = shuf.mean(1).max(-1) / args.bin_s
            sosi, _, sdsi, _ = vector_stats(speak)
            rows.append(dict(recording=rec_name, condition=cname, polarity=pol, fps=fps, cell_index=int(cell),
                             n_spikes=n_sp, OSI=float(osi), p_OSI=float(np.mean(sosi >= osi)),
                             pref_motion_axis_polar=float(axis), DSI=float(dsi), p_DSI=float(np.mean(sdsi >= dsi)),
                             pref_direction_polar=float(pdir),
                             **{f"peak_{n}": float(p) for n, p in zip(DIRECTION_NAMES, peak)}))
            n_done += 1
        sub = pd.DataFrame([r for r in rows if r["recording"] == rec_name and r["condition"] == cname])
        print(f"{rec_name} {cname}: {n_done} cells; OSI p<0.05: {(sub.p_OSI < 0.05).sum()}, "
              f"DSI p<0.05: {(sub.p_DSI < 0.05).sum()}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "edge_tuning_table.csv", index=False)
print("saved", OUT / "edge_tuning_table.csv")
