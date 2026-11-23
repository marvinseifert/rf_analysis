"""
Where do RF long axes fall relative to their own cone row: across, along, or both? (23/11/2026)

Question (Laura): RFs lie across their own row more often than chance, but where do the other cells fall? If they
fall ALONG the rows, RFs would be aligned with the mosaic either way (two peaks, at 0° and 90°, dip at 45°).

Per cell: angle between RF long axis and the radiating row through its centre (0 = along, 90 = across), cells as
summary_figure.py (quality >= 10, inside the displayed stimulus, not near its edge); elongated = 1/tilt > 1.3.
  * fractions in three bands: along 0-15°, oblique 30-60°, across 75-90° (chance: 16.7 / 33.3 / 16.7 %)
  * "aligned either way" score: mean cos(4 x angle) (+1 = all along or across, -1 = all at 45°, 0 = random),
    tested with the within-recording shuffle and fake ONH positions (as rf_row_strict_tests.py)
  * "across vs along" score: mean -cos(2 x angle) (+1 = all across, -1 = all along), same tests
  * control: RF angles vs the stimulus pixel grid (mean cos(4 x RF angle in the stimulus frame)); square checkers
    could bias fitted RF axes towards the grid's x/y axes.

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_angle_distribution.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")
N_PERM, FAKE_ONH_RADIUS_UM, ELONGATION_MIN = 5000, 1500.0, 1.3
ACROSS, ALONG, OBLIQUE = "#1e6fd9", "#e0a800", "0.6"
df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
ok = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
ok["elong"] = 1 / ok.tilt > ELONGATION_MIN


def diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


ok["to_row"] = diff(ok.rf_axis_projected_deg, ok.cone_row_axis_projected_deg)
rng = np.random.default_rng(3)


def scores(angles):
    t = np.deg2rad(angles)
    return float(np.cos(4 * t).mean()), float(-np.cos(2 * t).mean())


def tests(sel):
    groups = [t for _, t in sel.groupby("recording")]
    real4, real2 = scores(sel.to_row)
    n1, n2 = [], []
    for _ in range(N_PERM):
        a1 = np.concatenate([diff(rng.permutation(t.rf_axis_projected_deg.values), t.cone_row_axis_projected_deg.values)
                             for t in groups])
        parts = []
        for t in groups:
            P = np.c_[t.x_projected_um, t.y_projected_um]
            a, rad = rng.uniform(0, 2 * np.pi), FAKE_ONH_RADIUS_UM * np.sqrt(rng.uniform())
            o = P.mean(0) + rad * np.array([np.cos(a), np.sin(a)])
            rows = np.degrees(np.arctan2(P[:, 1] - o[1], P[:, 0] - o[0])) % 180
            parts.append(diff(t.rf_axis_projected_deg.values, rows))
        n1.append(scores(a1)); n2.append(scores(np.concatenate(parts)))
    n1, n2 = np.array(n1), np.array(n2)
    return dict(n=len(sel), c4=real4, c2=real2,
                p4_shuffle=float(np.mean(n1[:, 0] >= real4)), p4_onh=float(np.mean(n2[:, 0] >= real4)),
                p2_shuffle=float(np.mean(n1[:, 1] >= real2)), p2_onh=float(np.mean(n2[:, 1] >= real2)),
                null4=n1[:, 0], null2=n1[:, 1])


def bands(a):
    a = np.asarray(a)
    return (a < 15).mean() * 100, ((a >= 30) & (a < 60)).mean() * 100, (a >= 75).mean() * 100


sets = {"All cells": ok, f"Elongated (> {ELONGATION_MIN})": ok[ok.elong]}
res = {k: tests(v) for k, v in sets.items()}
for k, r in res.items():
    print(f"{k} (n = {r['n']}): along-or-across score {r['c4']:+.3f} (p shuffle {r['p4_shuffle']:.3f}, p ONH "
          f"{r['p4_onh']:.3f}); across-vs-along score {r['c2']:+.3f} (p shuffle {r['p2_shuffle']:.3f}, "
          f"p ONH {r['p2_onh']:.3f}); bands {bands(sets[k].to_row)}")
grid4_all = float(np.cos(4 * np.deg2rad(ok.rf_angle_stimulus_deg)).mean())
grid4_el = float(np.cos(4 * np.deg2rad(ok[ok.elong].rf_angle_stimulus_deg)).mean())

# ------------------------------------------------------------------ figure
recs = sorted(ok.recording.unique())
n_rec_cols = int(np.ceil(len(recs) / 2))
fig = plt.figure(figsize=(21, 10))
gs = fig.add_gridspec(2, 4 + n_rec_cols, width_ratios=[1.4, 1.4, 1.1, 1.1] + [0.9] * n_rec_cols,
                      hspace=0.6, wspace=0.55)
bins = np.arange(0, 91, 7.5)
for i, (k, sel) in enumerate(sets.items()):
    r = res[k]
    ax = fig.add_subplot(gs[i, 0])
    counts, _, patches = ax.hist(sel.to_row, bins=bins, edgecolor="white")
    for p, left in zip(patches, bins[:-1]):
        p.set_facecolor(ALONG if left < 15 else ACROSS if left >= 75 else (OBLIQUE if 30 <= left < 60 else "0.8"))
    ax.axhline(len(sel) * 7.5 / 90, color="k", ls="--", lw=1, label="chance (evenly spread)")
    ax.set_xlim(0, 90); ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("angle between RF long axis and its own cone row (°)\n0 = along, 90 = across")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"{k}, pooled (n = {r['n']})", loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[i, 1])
    b = bands(sel.to_row)
    x = np.arange(3)
    ax.bar(x, b, color=[ALONG, OBLIQUE, ACROSS], edgecolor="k")
    ax.scatter(x, [100 / 6, 100 / 3, 100 / 6], marker="_", s=900, color="k", lw=2, zorder=3, label="chance")
    for xi, v in zip(x, b):
        ax.text(xi, v + 1, f"{v:.0f} %", ha="center", fontsize=10)
    ax.set_xticks(x, ["along\n0-15°", "oblique\n30-60°", "across\n75-90°"])
    ax.set_ylabel("% of cells")
    ax.set_ylim(0, 45)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Where do the cells fall?", loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for j, (null, real, p1, p2, ttl, lab) in enumerate([
            (r["null4"], r["c4"], r["p4_shuffle"], r["p4_onh"], "Along OR across?",
             "mean cos(4 x angle)\n+1 = all along or across, -1 = all at 45°"),
            (r["null2"], r["c2"], r["p2_shuffle"], r["p2_onh"], "Across > along?",
             "mean -cos(2 x angle)\n+1 = all across, -1 = all along")]):
        ax = fig.add_subplot(gs[i, 2 + j])
        ax.hist(null, bins=50, color="0.75")
        ax.axvline(real, color=ACROSS, lw=2.5)
        ax.set_xlabel(lab, fontsize=8.5)
        ax.set_yticks([])
        ax.set_title(f"{ttl} real {real:+.3f}\nshuffle p = {p1:.3f}\nfake ONH p = {p2:.3f}", loc="left", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)

for m, rec in enumerate(recs):
    t = ok[ok.recording == rec]
    ax = fig.add_subplot(gs[m % 2, 4 + m // 2])
    b = bands(t.to_row)
    ax.bar(range(3), b, color=[ALONG, OBLIQUE, ACROSS], edgecolor="k")
    ax.scatter(range(3), [100 / 6, 100 / 3, 100 / 6], marker="_", s=400, color="k", lw=2, zorder=3)
    ax.set_xticks(range(3), ["along", "obl.", "across"], fontsize=8)
    ax.set_ylim(0, 50)
    ax.set_title(f"{rec.replace('_', '/')} (n = {len(t)})", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

fig.text(0.01, -0.02,
         "Answer: the cells that are not across their row are NOT along it. Along-row RFs are rarer than chance "
         "(about 10 % vs 16.7 %), oblique ones are at chance, across-row ones slightly above chance: one tilt away "
         "from the rows, not two peaks.\n"
         f"Control: RF angles lean slightly towards the stimulus pixel grid's x/y axes (mean cos(4 x angle) "
         f"{grid4_all:+.2f} all, {grid4_el:+.2f} elongated), a possible effect of square checkers; the rows are not "
         "aligned with the grid, so this cannot create row alignment. Cells as summary_figure.py; rows assumed to "
         "radiate from the optic nerve head.", fontsize=9.5, va="top")
fig.suptitle("Do RFs line up with the cone mosaic EITHER along or across the rows? Angle between each RF long axis "
             "and its own radiating cone row (7 recordings)", fontsize=12, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_row_angle_distribution.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_row_angle_distribution.png")
