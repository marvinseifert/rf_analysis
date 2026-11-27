"""
Stricter tests of "RFs lie across the radiating cone rows" (23/11/2026). Companion to summary_figure.py,
which is left unchanged.

Why: summary_figure.py's main test compares each recording's mean RF direction with its MEAN row direction.
Where the optic nerve head (ONH) is on or near the array, the rows fan out in all directions and their mean is
meaningless (row-direction spread = |mean of exp(2i*angle)|, 1 = parallel rows, 0 = all directions).
Its pooled histograms compare each cell with its own row (fine) but against "evenly spread", which a recording-wide
bias (e.g. flattening stretching all RFs one way) could also produce.

Tests here (each cell vs its OWN row; cells: quality >= 10, inside the displayed stimulus, not near its edge,
as summary_figure.py; data rf_vs_cone_row_angle_all_recordings_cells.csv):
  1. within-recording shuffle: RF axes shuffled among cells of the same recording (5000x). Asks whether RFs
     follow the LOCAL row direction as the rows fan out; a recording-wide bias cannot do that.
  2. fake ONH positions: every cell's row direction recomputed as if the ONH were at a random position within
     1.5 mm of the recording's cell centroid (5000x). Asks whether the real ONH position is special.
Statistic: mean over cells of |RF long axis - own row| (0 = along the rows, 90 = across).

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_strict_tests.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
N_PERM, FAKE_ONH_RADIUS_UM = 5000, 1500
df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
ok = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
rng = np.random.default_rng(1)


def diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


spread = {r: float(abs(np.mean(np.exp(2j * np.deg2rad(d.cone_row_axis_projected_deg))))) for r, d in ok.groupby("recording")}


def run(sel):
    real = diff(sel.rf_axis_projected_deg, sel.cone_row_axis_projected_deg).mean()
    groups = [d for _, d in sel.groupby("recording")]
    null1 = np.array([np.concatenate([diff(rng.permutation(d.rf_axis_projected_deg.values),
                                           d.cone_row_axis_projected_deg.values) for d in groups]).mean()
                      for _ in range(N_PERM)])
    null2 = []
    for _ in range(N_PERM):
        parts = []
        for d in groups:
            P = np.c_[d.x_projected_um, d.y_projected_um]
            a, rad = rng.uniform(0, 2 * np.pi), FAKE_ONH_RADIUS_UM * np.sqrt(rng.uniform())
            o = P.mean(0) + rad * np.array([np.cos(a), np.sin(a)])
            rows = np.rad2deg(np.arctan2(P[:, 1] - o[1], P[:, 0] - o[0])) % 180
            parts.append(diff(d.rf_axis_projected_deg.values, rows))
        null2.append(np.concatenate(parts).mean())
    null2 = np.array(null2)
    return real, null1, null2, float(np.mean(null1 >= real)), float(np.mean(null2 >= real))


sets = {"all cells": ok, "elongated (> 1.3)": ok[ok.elongated.astype(bool)]}
res = {k: run(v) for k, v in sets.items()}
for k, (real, n1, n2, p1, p2) in res.items():
    print(f"{k} (n = {len(sets[k])}): real {real:.1f}°; test 1 null {n1.mean():.1f}°, p = {p1:.4f}; "
          f"test 2 null {n2.mean():.1f}°, p = {p2:.4f}")

fig = plt.figure(figsize=(20, 9))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.35, hspace=0.45)
for i, (k, (real, n1, n2, p1, p2)) in enumerate(res.items()):
    for j, (null, p, ttl) in enumerate([(n1, p1, "Test 1: RF axes shuffled within recording"),
                                        (n2, p2, f"Test 2: ONH at random positions (<= {FAKE_ONH_RADIUS_UM / 1000:.1f} mm)")]):
        ax = fig.add_subplot(gs[i, j])
        ax.hist(null, bins=50, color="0.75")
        ax.axvline(real, color="tab:blue", lw=2.5)
        ax.text(real, ax.get_ylim()[1] * 0.95, f" real {real:.1f}°", color="tab:blue", fontsize=9, va="top")
        ax.set_xlabel("mean |RF long axis - own cone row| (°)\nhigher = more across the rows")
        ax.set_title(f"{ttl}\n{k}, n = {len(sets[k])}: chance {null.mean():.1f}°, p = {p:.3f}", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

# per recording: % across + row spread
ax = fig.add_subplot(gs[0, 2])
recs = sorted(ok.recording.unique(), key=lambda r: spread[r])
for i, r in enumerate(recs):
    d = ok[ok.recording == r]
    de = d[d.elongated.astype(bool)]
    a = (diff(d.rf_axis_projected_deg, d.cone_row_axis_projected_deg) > 45).mean() * 100
    ax.barh(i + 0.2, a, 0.4, color="tab:blue", label="all cells" if i == 0 else None)
    if len(de):
        ae = (diff(de.rf_axis_projected_deg, de.cone_row_axis_projected_deg) > 45).mean() * 100
        ax.barh(i - 0.2, ae, 0.4, color="k", label="elongated" if i == 0 else None)
    ax.text(101, i, f"n = {len(d)} / {len(de)}", va="center", fontsize=7)
ax.axvline(50, color="grey", ls="--", lw=0.8)
ax.set_yticks(range(len(recs)), [f"{r}\nrow spread {spread[r]:.2f}" for r in recs], fontsize=7)
ax.set_xlim(0, 100)
ax.set_xlabel("% of cells whose RF lies ACROSS its own row (> 45°)")
ax.set_title("Per recording (ordered by row spread:\n0 = rows fan in all directions, 1 = parallel)", fontsize=9,
             loc="left")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[:, 3]); ax.axis("off")
r_all, r_el = res["all cells"], res["elongated (> 1.3)"]
txt = (
    "Summary\n\n"
    f"- RFs follow their OWN local row direction more than\n  chance as the rows fan out (test 1): all cells\n"
    f"  p = {r_all[3]:.3f}, elongated p = {r_el[3]:.3f}. A recording-wide\n  bias (flattening, projector) cannot "
    "produce this.\n\n"
    f"- The real ONH position explains RF directions better\n  than random ONH positions, but only borderline\n"
    f"  (test 2): all p = {r_all[4]:.3f}, elongated p = {r_el[4]:.3f}.\n\n"
    f"- Effect size is modest: mean {r_all[0]:.1f}° vs chance\n  {r_all[1].mean():.1f}° (all cells); 55-76 % of cells "
    "lie across\n  their row in every recording (50 % = no preference).\n\n"
    "- The across-recording test in summary_figure.py\n  (p = 0.007) compares MEAN row directions; with\n"
    "  row spread of 0.08-0.35 in 4 recordings these\n  means are not meaningful. Use test 1 instead.\n\n"
    "- Rows are still assumed to radiate from the ONH,\n  not measured; RF angles are noisy (9-23° between\n"
    "  noise stimuli), which dilutes the effect.")
ax.text(0, 1, txt, va="top", fontsize=10, family="sans-serif",
        bbox=dict(boxstyle="round", fc="#eef4fb", ec="tab:blue"))
fig.suptitle("RF long axis vs radiating cone rows: stricter tests (each cell compared with its own row). "
             "Same cells as the summary figure.", fontsize=12, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_vs_cone_rows_strict_tests.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_vs_cone_rows_strict_tests.png")
