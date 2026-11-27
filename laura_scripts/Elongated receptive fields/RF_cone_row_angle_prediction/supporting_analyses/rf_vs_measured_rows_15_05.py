"""
15/05/2026: RF long axes vs the MEASURED cone rows (traced by Laura), instead of rows assumed to radiate from the
optic nerve head (ONH). (23/11/2026)

Traced rows: Box Imaging/20260515_dragonfly_microscope/15_05_2026_cones_rows.csv (row_id, point_id, x/y pixels;
red-pen tracing extracted onto 15_05_2026_cones.jpg, which matches 15_05_2026_p0_cones_flipped.jpg, the image used
by the alignment: green-channel correlation 0.78 unflipped vs <= 0.10 flipped/rotated).
Image px -> projected µm: alignment_geometry A_image1_to_projected_um (v2 cache), as rf_cone_row_angle.py.

Local measured row direction at each RF centre: every traced row with points within --radius-um of the centre
contributes the principal axis (PCA) of its points inside that circle, weighted by the number of points; the row
direction = axial mean of those axes (also its consistency, 0-1). Cells with no traced row nearby are left out.
Cells: from rf_vs_cone_row_angle_all_recordings_cells.csv (quality >= 10, inside the displayed stimulus, not near
its edge); RF long axis rf_axis_projected_deg; ONH-radial row cone_row_axis_projected_deg (for comparison).

Tests (one recording, so the shuffle permutes RF axes among its cells, 5000x):
  across-vs-along score mean -cos(2 x angle) (+1 all across, -1 all along), and
  along-or-across score  mean  cos(4 x angle) (+1 all along or across, -1 all at 45°),
  each with RF axes shuffled among the cells; bands along 0-15 / oblique 30-60 / across 75-90° (chance 16.7/33.3/16.7).
Same statistics with the ONH-radial rows on the SAME cells, for comparison.

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_vs_measured_rows_15_05.py"
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from PIL import Image

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file

parser = argparse.ArgumentParser()
parser.add_argument("--radius-um", type=float, default=50.0)
parser.add_argument("--n-perm", type=int, default=5000)
args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = None
REC = "15_05_2026"
IMG_DIR = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Experiments/Imaging/20260515_dragonfly_microscope")
FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
HERE = Path(__file__).resolve().parents[1]
ACROSS, ALONG = "#1e6fd9", "#e0a800"
ROW_CMAP = LinearSegmentedColormap.from_list("row_angle", ["#ffd23f", "#3bb273", ACROSS])
ELONGATION_MIN = 1.3

with contextlib.redirect_stdout(io.StringIO()):
    cache = alignment_cache_v2_file(REC)
    geo = g.load_alignment_geometry(cache)
Ai = geo.A_image1_to_projected_um
cache_np = np.load(cache)
half_w = int(cache_np["displayed_stim_width_px"]) * geo.corrected_stim_um_per_px / 2
half_h = int(cache_np["displayed_stim_height_px"]) * geo.corrected_stim_um_per_px / 2

rows = pd.read_csv(IMG_DIR / "15_05_2026_cones_rows.csv")
P = np.c_[rows.x_pixels, rows.y_pixels, np.ones(len(rows))] @ Ai.T
rows["px"], rows["py"] = P[:, 0], P[:, 1]
onh = json.loads((HERE / "optic_nerve_head_positions.json").read_text())[REC]
onh_xy = (Ai @ np.r_[onh["x_px"], onh["y_px"], 1])[:2]

cells = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
cells = cells[(cells.recording == REC) & (cells.quality >= 10) & cells.inside_displayed_stimulus.astype(bool)
              & ~cells.near_edge.astype(bool)].copy()


def diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


meas, consist, nrows = [], [], []
for _, c in cells.iterrows():
    near = rows[np.hypot(rows.px - c.x_projected_um, rows.py - c.y_projected_um) <= args.radius_um]
    zs, ws = [], []
    for _, seg in near.groupby("row_id"):
        if len(seg) < 5:
            continue
        X = np.c_[seg.px, seg.py]
        X = X - X.mean(0)
        evals, evecs = np.linalg.eigh(X.T @ X)
        v = evecs[:, -1]
        ang = np.degrees(np.arctan2(v[1], v[0])) % 180
        zs.append(np.exp(2j * np.deg2rad(ang)))
        ws.append(len(seg))
    if not zs:
        meas.append(np.nan); consist.append(np.nan); nrows.append(0); continue
    z = np.sum(np.array(ws) * np.array(zs)) / np.sum(ws)
    meas.append(np.degrees(np.angle(z)) / 2 % 180); consist.append(abs(z)); nrows.append(len(zs))
cells["measured_row_deg"], cells["row_consistency"], cells["n_rows_near"] = meas, consist, nrows
cov = cells.measured_row_deg.notna()
print(f"{REC}: {len(cells)} cells, {cov.sum()} with a traced row within {args.radius_um:.0f} µm "
      f"(median {np.median(cells.n_rows_near[cov]):.0f} rows, consistency {np.median(cells.row_consistency[cov]):.2f})")
c = cells[cov].copy()
c["elong"] = 1 / c.tilt > ELONGATION_MIN
c["to_measured"] = diff(c.rf_axis_projected_deg, c.measured_row_deg)
c["to_radial"] = diff(c.rf_axis_projected_deg, c.cone_row_axis_projected_deg)
c["measured_vs_radial"] = diff(c.measured_row_deg, c.cone_row_axis_projected_deg)
print(f"measured vs ONH-radial row direction at the cells: median difference {c.measured_vs_radial.median():.0f}° "
      f"(IQR {c.measured_vs_radial.quantile(0.25):.0f}-{c.measured_vs_radial.quantile(0.75):.0f}°)")
rng = np.random.default_rng(5)


def stats(sel, rowcol):
    a = diff(sel.rf_axis_projected_deg, sel[rowcol])
    t = np.deg2rad(a)
    s2, s4 = float(-np.cos(2 * t).mean()), float(np.cos(4 * t).mean())
    null = []
    for _ in range(args.n_perm):
        tt = np.deg2rad(diff(rng.permutation(sel.rf_axis_projected_deg.values), sel[rowcol].values))
        null.append((-np.cos(2 * tt).mean(), np.cos(4 * tt).mean()))
    null = np.array(null)
    return dict(n=len(sel), s2=s2, p2=float(np.mean(null[:, 0] >= s2)), s4=s4, p4=float(np.mean(null[:, 1] >= s4)),
                along=float((a < 15).mean() * 100), oblique=float(((a >= 30) & (a < 60)).mean() * 100),
                across=float((a >= 75).mean() * 100), angles=a)


res = {}
for subset, sel in [("all", c), ("elongated", c[c.elong])]:
    for rowname, col in [("measured rows", "measured_row_deg"), ("ONH-radial rows", "cone_row_axis_projected_deg")]:
        r = stats(sel, col)
        res[(subset, rowname)] = r
        print(f"{subset:9s} vs {rowname:15s} (n = {r['n']}): along {r['along']:.0f} %, oblique {r['oblique']:.0f} %, "
              f"across {r['across']:.0f} %; across-vs-along {r['s2']:+.3f} (p = {r['p2']:.3f}); "
              f"along-or-across {r['s4']:+.3f} (p = {r['p4']:.3f})")
c.to_csv(FIG / "rf_vs_measured_rows_15_05_2026_cells.csv", index=False)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(22, 11))
gs = fig.add_gridspec(2, 4, width_ratios=[1.7, 1, 1, 1.05], hspace=0.45, wspace=0.35)
ax = fig.add_subplot(gs[:, 0])
img = Image.open(cone_image_file(REC))
iw, ih = img.size
small = np.asarray(img.convert("L").reduce(4))
img.close()
ax.set_facecolor("black")
ax.imshow(small, cmap="gray", extent=[0, iw, ih, 0], origin="upper", alpha=0.7,
          transform=Affine2D(Ai) + ax.transData)
for _, seg in rows.groupby("row_id"):
    ax.plot(seg.px.values, seg.py.values, ",", color="#ff6a2b", alpha=0.6, ms=0.5, zorder=4)
ax.plot([-half_w, half_w, half_w, -half_w, -half_w], [-half_h, -half_h, half_h, half_h, -half_h], "w--", lw=1, zorder=5)
L = 40
for _, r in c.iterrows():
    a = np.deg2rad(r.rf_axis_projected_deg)
    ax.plot([r.x_projected_um - L * np.cos(a), r.x_projected_um + L * np.cos(a)],
            [r.y_projected_um - L * np.sin(a), r.y_projected_um + L * np.sin(a)], color=ROW_CMAP(r.to_measured / 90),
            lw=2.2 if r.elong else 1.0, solid_capstyle="round", zorder=6)
ax.plot(*onh_xy, "o", color="red", ms=10, mec="white", mew=1.5, zorder=8)
xs, ys = np.r_[-half_w, half_w, onh_xy[0]], np.r_[-half_h, half_h, onh_xy[1]]
ax.set_xlim(xs.min() - 80, xs.max() + 80); ax.set_ylim(ys.max() + 80, ys.min() - 80)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"15/05/2026: traced cone rows (orange); RF long axes\ncoloured by angle to the MEASURED local row\n"
             f"(yellow = along, blue = across; thick = elongated)\nred = optic nerve head; n = {len(c)}",
             loc="left", fontsize=10)

ax = fig.add_subplot(gs[0, 1])
ax.hist(c.measured_vs_radial, bins=np.arange(0, 91, 7.5), color="0.5", edgecolor="white")
ax.set_xlabel("measured row direction vs ONH-radial row (°)")
ax.set_ylabel("cells")
ax.set_xlim(0, 90)
ax.set_title(f"Do the traced rows radiate from the ONH?\nmedian difference {c.measured_vs_radial.median():.0f}° at the "
             "cells", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

bins = np.arange(0, 91, 10)
for j, subset in enumerate(["all", "elongated"]):
    ax = fig.add_subplot(gs[1, 1 + j])
    rm, rr = res[(subset, "measured rows")], res[(subset, "ONH-radial rows")]
    ax.hist(rm["angles"], bins=bins, color=ACROSS, alpha=0.75, edgecolor="white", label="vs measured row")
    ax.hist(rr["angles"], bins=bins, histtype="step", color="k", lw=1.8, label="vs ONH-radial row")
    ax.axhline(rm["n"] * 10 / 90, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 90); ax.set_xticks([0, 45, 90], ["0\nalong", "45", "90\nacross"])
    ax.set_title(f"{subset} cells (n = {rm['n']}): RF vs row", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[0, 2])
x = np.arange(3)
for k, (rowname, col, off) in enumerate([("measured rows", ACROSS, -0.2), ("ONH-radial rows", "0.55", 0.2)]):
    r = res[("all", rowname)]
    ax.bar(x + off, [r["along"], r["oblique"], r["across"]], 0.4, color=col, edgecolor="k", label=rowname)
ax.scatter(x, [100 / 6, 100 / 3, 100 / 6], marker="_", s=900, color="k", lw=2, zorder=3, label="chance")
ax.set_xticks(x, ["along\n0-15°", "oblique\n30-60°", "across\n75-90°"])
ax.set_ylabel("% of cells")
ax.set_ylim(0, 50)
ax.legend(fontsize=8, frameon=False)
ax.set_title("All cells: where do RF axes fall?", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[:, 3]); ax.axis("off")
lines = [f"15/05/2026, rows traced by hand (radius {args.radius_um:.0f} µm around each RF centre)\n",
         f"{len(c)} of {len(cells)} cells have a traced row nearby.",
         f"Traced vs ONH-radial rows at the cells: median {c.measured_vs_radial.median():.0f}° apart.\n"]
for subset in ["all", "elongated"]:
    for rowname in ["measured rows", "ONH-radial rows"]:
        r = res[(subset, rowname)]
        lines.append(f"{subset}, vs {rowname} (n = {r['n']}):\n  along {r['along']:.0f} %, oblique {r['oblique']:.0f} %, "
                     f"across {r['across']:.0f} %\n  across > along: {r['s2']:+.2f}, p = {r['p2']:.3f}\n"
                     f"  along OR across: {r['s4']:+.2f}, p = {r['p4']:.3f}")
lines.append("\nScores: across > along = mean -cos(2 x angle) (+1 all across, -1 all along);\nalong OR across = "
             "mean cos(4 x angle) (+1 all along or across, -1 all at 45°).\np: RF axes shuffled among the cells "
             f"({args.n_perm}x). Chance bands: 16.7 / 33.3 / 16.7 %.")
ax.text(0, 1, "\n".join(lines), va="top", fontsize=9.5, linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef5fd", edgecolor=ACROSS))
fig.suptitle("RF orientation vs MEASURED cone rows (15/05/2026, rows traced on the cone image)", fontsize=13,
             x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_vs_measured_rows_15_05_2026.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_vs_measured_rows_15_05_2026.png")
