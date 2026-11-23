"""
Claim 1C: does moving-bar orientation tuning follow the cone rows? (23/11/2026)

RF long axes run ACROSS the predicted radiating cone rows (summary_figure.py: main RF direction ~70° from the
rows, across-recording p = 0.007). If the row structure shapes what RGCs report, the same should show in their
responses to moving bars:
  * an RF elongated across the rows responds best to a bar whose edge lies along the RF long axis, i.e. a bar
    MOVING along the rows -> preferred motion axis ~parallel to the rows (difference ~0°), and
  * within a cell, preferred motion axis ~perpendicular to the RF long axis (difference ~90°).

Moving-bar tuning: OSI/OS_analysis.py functions, unchanged (8 directions, response = total spikes per direction
block, orientation vector with doubled angles; preferred_orientation_deg = the axis the bar MOVES along, polar
frame: 0° = up, clockwise; projected frame, date-dependent direction table).
RF long axis and cone-row axis per cell: rf_vs_cone_row_angle_all_recordings_cells.csv (rf_cone_row_angle.py),
converted from the projected frame (0° = +x right, towards +y down) to the polar frame: polar = (deg + 90) % 180.

Tests
  per cell     : |bar motion axis - row axis| and |bar motion axis - RF long axis| (0-90°), for tuned cells
                 (OSI >= --min-osi, >= --min-spikes spikes); cells near the stimulus edge left out.
  shuffle test : mean over cells of |bar motion axis - own row|; null = bar axes shuffled among cells of the SAME
                 recording (5000x). Tests whether bar preference follows each cell's local row as the rows fan out;
                 a fixed bias (projector, flattening) cannot produce that. A recording's mean row direction is not
                 used: rows fan out from a nearby ONH, so it is meaningless in most recordings.

Recordings with the 8-direction moving bar: 22/07, 14/08, 14/05, 15/10, 15/05 (07/05 has none; 07/08 only
weighted moving edges).

Run from the repo root (~20-30 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/bar_orientation_vs_cone_rows.py"
"""
import argparse
import ast
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laura_scripts.Alignment_pipeline.alignment_geometry import (
    moving_bar_direction_names,
    moving_bar_directions_deg,
    orientation_difference_deg,
)
from laura_scripts.Alignment_pipeline.alignment_recordings import noise_data_v2_file

parser = argparse.ArgumentParser()
parser.add_argument("--min-osi", type=float, default=0.1)
parser.add_argument("--min-spikes", type=int, default=100)
parser.add_argument("--metric", choices=["total_spikes", "peak_response"], default="peak_response",
                    help="total spikes per direction block (favours bars that stay longer in the RF) or peak PSTH")
args = parser.parse_args()

REPO = Path(__file__).resolve().parents[4]
FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")
CSV = FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv"
RECS = {  # recording -> (phase folder, moving-bar stimulus index, repeats)
    "22_07_2026": ("Phase_00", 3, 10),
    "14_08_2026": ("Phase_00", 3, 10),
    "14_05_2026": ("Phase_00", 3, 10),
    "15_10_2026": ("Phase_00", 3, 10),  # stimulus table says 9, triggers show 10
    "15_05_2026": ("Phase_00", 4, 10),
}

# ---- OS_analysis.py functions, unchanged (the file runs an analysis at import, so only imports + defs) ----
src = REPO / "laura_scripts" / "OSI" / "OS_analysis.py"
tree = ast.parse(src.read_text())
def _is_literal(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


# imports, function definitions, and simple settings (literal assignments: used as default arguments)
keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (isinstance(n, ast.Assign) and _is_literal(n.value))]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(src), "exec"), ns)
analyse_one_dataset = ns["analyse_one_dataset"]

cells = pd.read_csv(CSV)
cells["rf_axis_polar"] = (cells["rf_axis_projected_deg"] + 90) % 180
cells["row_axis_polar"] = (cells["cone_row_axis_projected_deg"] + 90) % 180

rows = []
for rec_name, (phase, stim_index, n_rep) in RECS.items():
    sub = cells[cells.recording == rec_name]
    channel = sub["channel"].iloc[0]
    root = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec_name}/{phase}")
    config = dict(recording_id=rec_name, overview_path=str(root / "overview"),
                  noise_data_path=str(noise_data_v2_file(rec_name)),  # same RF data as the RF/row table
                  stimulus_index=stim_index, rf_channel=channel)
    names = moving_bar_direction_names(rec_name)
    res = analyse_one_dataset(config=config, direction_names=names, directions_deg=moving_bar_directions_deg(names),
                              frames_per_direction=300, seconds_per_frame=1 / 60, expected_repeats=n_rep,
                              trigger_col="trigger", repeat_col="repeat", time_col="times_triggered",
                              sample_rate_hz=20_000, peak_response_bin_size_s=0.1)
    tun = pd.DataFrame([{"recording": rec_name, "cell_index": int(r["cell_id"]) if "cell_id" in r else
                         int(r["global_cell_id"].split("_cell_")[1]),
                         "osi": r[f"OSI_{args.metric}"], "bar_axis_polar": r[f"preferred_orientation_deg_{args.metric}"],
                         "dsi": r[f"vector_DSI_{args.metric}"],
                         "bar_direction_polar": r[f"vector_preferred_direction_deg_{args.metric}"],
                         "n_spikes": r["n_spikes"]} for r in res])
    rows.append(sub.merge(tun, on=["recording", "cell_index"], how="inner"))
    print(f"{rec_name}: {len(res)} cells with bar tuning, {len(rows[-1])} also in the RF/row table", flush=True)

df = pd.concat(rows, ignore_index=True)
df["bar_vs_row"] = orientation_difference_deg(df["bar_axis_polar"], df["row_axis_polar"])
df["bar_vs_rf"] = orientation_difference_deg(df["bar_axis_polar"], df["rf_axis_polar"])
df["rf_vs_row"] = orientation_difference_deg(df["rf_axis_polar"], df["row_axis_polar"])
tuned = df[(df.osi >= args.min_osi) & (df.n_spikes >= args.min_spikes) & (~df.near_edge.astype(bool))]
df.to_csv(FIG / f"bar_orientation_vs_cone_rows_cells_{args.metric}.csv", index=False)
print(f"\n{len(df)} cells with RF + bar tuning; {len(tuned)} tuned (OSI >= {args.min_osi}, >= {args.min_spikes} "
      f"spikes, not near the edge)")


def axial_mean(deg, w=None):
    w = np.ones(len(deg)) if w is None else np.asarray(w)
    z = np.sum(w * np.exp(2j * np.deg2rad(deg)))
    return (np.rad2deg(np.angle(z)) / 2) % 180


recs = [r for r in RECS if (tuned.recording == r).sum() >= 3]
# Per-cell test (each cell vs its OWN local row), null = bar axes shuffled among cells of the same recording.
# (A recording's mean row direction is NOT used: rows fan out from a nearby optic nerve head, so their mean is
# meaningless in most recordings. An earlier version compared mean axes and wrongly gave "across rows, p = 0.025".)
rng = np.random.default_rng(0)
real_mean = float(tuned.bar_vs_row.mean())
null = []
for _ in range(5000):
    parts = [orientation_difference_deg(rng.permutation(d.bar_axis_polar.values), d.row_axis_polar.values)
             for _, d in tuned.groupby("recording")]
    null.append(np.concatenate(parts).mean())
null = np.array(null)
p_across = float(np.mean(null >= real_mean))  # bar motion more ACROSS own rows than chance
p_along = float(np.mean(null <= real_mean))   # bar motion more ALONG own rows than chance
print(f"\nper cell (n = {len(tuned)}): mean |bar motion axis - own row| {real_mean:.1f}° "
      f"(within-recording shuffle {null.mean():.1f}°); p(across) = {p_across:.3f}, p(along) = {p_along:.3f}")
for r in recs:
    d = tuned[tuned.recording == r]
    print(f"  {r}: n tuned {len(d)}, bar vs own row median {d.bar_vs_row.median():.0f}° "
          f"({(d.bar_vs_row > 45).mean() * 100:.0f} % across), bar vs RF median {d.bar_vs_rf.median():.0f}°")
el = tuned[tuned.elongated.astype(bool)]
for name, d in [("all tuned", tuned), ("tuned + elongated RF", el)]:
    print(f"{name} (n = {len(d)}): bar vs rows median {d.bar_vs_row.median():.0f}° "
          f"({(d.bar_vs_row < 45).mean() * 100:.0f} % < 45°); bar vs RF median {d.bar_vs_rf.median():.0f}° "
          f"({(d.bar_vs_rf > 45).mean() * 100:.0f} % > 45°)")

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(2, len(recs) + 1, figsize=(4 * (len(recs) + 1), 8))
bins = np.linspace(0, 90, 7)
for j, r in enumerate(recs):
    d = tuned[tuned.recording == r]
    for i, (col, ttl, pred) in enumerate([("bar_vs_row", "bar motion axis vs own cone row", "0 = along rows, 90 = across"),
                                          ("bar_vs_rf", "bar motion axis vs RF long axis",
                                           "0 = along RF long axis, 90 = across")]):
        ax = axs[i, j]
        ax.hist(d[col], bins=bins, color="tab:blue", alpha=0.7, label="tuned cells")
        de = d[d.elongated.astype(bool)]
        ax.hist(de[col], bins=bins, histtype="step", color="k", lw=1.5, label="elongated RF")
        ax.axhline(len(d) / (len(bins) - 1), color="grey", ls="--", lw=0.8)
        ax.set_xticks([0, 45, 90]); ax.set_xlim(0, 90)
        ax.set_title(f"{r} (n = {len(d)})\n{ttl}", fontsize=9, loc="left")
        if j == 0:
            ax.set_ylabel(pred, fontsize=8)
            ax.legend(fontsize=7, frameon=False)
ax = axs[0, -1]
ax.hist(null, bins=40, color="0.7", label="bar axes shuffled within recording")
ax.axvline(real_mean, color="tab:blue", lw=2, label=f"real ({real_mean:.1f}°)")
ax.set_xlabel("mean |bar motion axis - own row| (°)\n(lower = along the rows, higher = across)")
ax.set_title(f"Per cell vs its own row, all tuned cells\np(more across than chance) = {p_across:.3f}, "
             f"p(more along) = {p_along:.3f}", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False)
ax = axs[1, -1]
ax.hist(tuned.bar_vs_row, bins=bins, color="tab:blue", alpha=0.7, label=f"bar vs rows (n = {len(tuned)})")
ax.hist(tuned.bar_vs_rf, bins=bins, histtype="step", color="tab:red", lw=2, label="bar vs RF long axis")
ax.axhline(len(tuned) / (len(bins) - 1), color="grey", ls="--", lw=0.8)
ax.set_xticks([0, 45, 90]); ax.set_xlim(0, 90)
ax.set_title("All recordings pooled", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False)
fig.suptitle(f"Do moving-bar orientation preferences follow the cone rows? Metric: {args.metric}. Tuned cells: OSI >= {args.min_osi}, "
             f">= {args.min_spikes} spikes, RF not near the stimulus edge. Bar motion axis = axis the preferred "
             "bar moves along (edge perpendicular to it). Dashed = evenly spread.", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"bar_orientation_vs_cone_rows_{args.metric}.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / f"bar_orientation_vs_cone_rows_{args.metric}.png")
