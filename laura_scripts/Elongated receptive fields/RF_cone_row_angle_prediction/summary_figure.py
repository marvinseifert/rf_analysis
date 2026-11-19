"""
Summary figure: RF orientation vs the predicted major cone rows, assumed to radiate out
from the optic nerve head (ONH). RF elongation tends to run ACROSS these radiating rows.
Clear across recordings, weak within single recordings.

Cone row angle: for each cell, the direction of the straight line from the clicked ONH to
the RF centre (one point per cell, not averaged over the RF; the rows fan out slowly, so
across one RF the direction changes by only a few degrees unless it is very close to the
ONH). Compared with the RF long axis: 0° = RF along the row, 90° = across it.

One row per recording, ordered by how far the recording's main RF direction is from the
row direction (most across the rows at the top, as the bottom panel):
  1. Cone image + population covariance RFs + ONH
  2. Each RF's long axis on the (grey) cone image, coloured by its angle to the radiating
     row through its centre; thick = elongated; orange-red lines = radiating rows; faded = edge cells
  3. Histogram: angle between each RF long axis and the radiating row. Edge cells left out.
Last row: across-recording test. Each recording's main RF direction (elongation-weighted
mean axis) vs the row direction at its cells (mean row axis). Unrelated directions differ
by 45° on average. p: each recording's RF direction paired with every other recording's
row direction (all permutations), one-sided.

Why across recordings: within one array the radiating rows hardly change direction, so
"across the rows" and "all RFs pointing one way" look the same. The ONH sits in a different
place in each recording, so the row direction differs; the RF direction changes with it.

Cells: quality >= QUALITY_MIN, RF centre inside the displayed stimulus, more than
EDGE_EXCLUSION_UM from its edge (cut-off RFs take the edge's orientation).

Run from the repo root (one recording at a time):
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/summary_figure.py"
Output: OUTPUT_FOLDER/rf_vs_radiating_cone_rows_summary.png/.pdf

The summary box uses the across-recording numbers computed here; the other numbers come
from supporting_analyses/ and rf_cone_row_angle.py (run 19/11/2026) and are set in
SUMMARY_FROM_OTHER_ANALYSES - update them if those analyses change.
"""

import itertools
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
from PIL import Image

import rf_cone_row_angle as rca  # same folder
from laura_scripts.Alignment_pipeline.alignment_recordings import cone_image_file

Image.MAX_IMAGE_PIXELS = None

OUT_NAME = "rf_vs_radiating_cone_rows_summary"
BIN_WIDTH_DEG = 10
N_RANDOM_PERMUTATIONS = 20000  # used only if there are too many recordings for all permutations
ROW_LINE_STEP_DEG = 10  # radiating rows drawn every ... degrees around the ONH
ROW_LINE_COLOUR = "#ff6a2b"  # orange-red: stands out from the grey cones and the RF line colours
RF_AXIS_LENGTH_UM = 80
COVARIANCE_ALPHA_ROW1 = 0.5  # covariance RF overlay in the first row (rf_cone_row_angle uses 0.6)

# Colours: along the row = yellow, across = blue (via green). The middle panels show the cone
# image in grey so the blue/green lines stay visible (blue is hard to see on the cyan image).
ALONG_COLOUR = "#e0a800"
ACROSS_COLOUR = "#1e6fd9"
ROW_CMAP = LinearSegmentedColormap.from_list("row_angle", ["#ffd23f", "#3bb273", ACROSS_COLOUR])

# Results of the other analyses (run 19/11/2026), quoted in the summary box:
#   15/05 p = 0.003: rf_cone_row_angle.py (RF axis turns with the row, per recording)
#   14/05 p = 0.027: supporting_analyses/rf_orientation_field.py
#   positions not in arcs: supporting_analyses/rf_position_arc_test.py
#   angle reliability 9-23° / 23-46°: supporting_analyses/check_rf_angle_reliability.py
SUMMARY_FROM_OTHER_ANALYSES = [
    "Weak within single recordings: only 15/05 (p = 0.003) and 14/05 (p = 0.027) show it beyond "
    "one shared RF direction.",
    "RF positions are not arranged in arcs around the optic nerve head.",
    "RF angles are reliable: the same cell measured with a different noise stimulus differs by "
    "9-23° (different cells: 23-46°).",
    "Caveats: rows are assumed to radiate from the optic nerve head, not measured; flattening the "
    "retina could elongate RFs the same way; 7 recordings.",
]


def axis_diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


def mean_axis(angles_deg, weights=None):
    """Weighted mean of axes (0-180°) and its strength (0 = no shared axis, 1 = all equal)."""
    w = np.ones(len(angles_deg)) if weights is None else np.asarray(weights)
    z = (w * np.exp(2j * np.deg2rad(angles_deg))).sum() / w.sum()
    return float(np.degrees(np.angle(z)) / 2 % 180), float(np.abs(z))


def main_rf_vs_row(table):
    """Main RF direction (weighted by elongation), its strength, and the row direction."""
    t = table[~table.near_edge]
    w = np.clip(1 - t.tilt.values, 0, 1)
    rf_dir, strength = mean_axis(t.rf_axis_projected_deg.values, w)
    row_dir, _ = mean_axis(t.cone_row_axis_projected_deg.values)  # ONH -> RF centre
    return rf_dir, strength, row_dir


def across_recording_test(rf_dirs, row_dirs):
    """Mean |RF - row| over recordings and its one-sided permutation p."""
    rf_dirs, row_dirs = np.asarray(rf_dirs), np.asarray(row_dirs)
    obs = axis_diff(rf_dirs, row_dirs).mean()
    n = len(rf_dirs)
    if math.factorial(n) <= 50000:
        perms = itertools.permutations(range(n))
    else:
        rng = np.random.default_rng(0)
        perms = (rng.permutation(n) for _ in range(N_RANDOM_PERMUTATIONS))
    null = np.array([axis_diff(rf_dirs, row_dirs[list(p)]).mean() for p in perms])
    p = np.mean(null >= obs) if obs >= 45 else np.mean(null <= obs)
    return float(obs), float(p), float(null.mean())


ELONGATION_MIN = 1.3  # 'elongated' in all panels: elongation = 1/tilt = long / short axis > this


def plot_row_histogram(ax, r, first=False):
    t = r["table"][~r["table"].near_edge]
    to_row = t.rf_vs_row_difference_deg.values  # 0 = along the row, 90 = across
    el = (1 / t.tilt.values) > ELONGATION_MIN
    bins = np.arange(0, 90 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.hist(to_row, bins=bins, color=ACROSS_COLOUR, alpha=0.75, edgecolor="white", label="all cells")
    ax.hist(to_row[el], bins=bins, histtype="step", color="k", lw=1.5,
            label=f"elongated (elongation > {ELONGATION_MIN})")
    ax.axhline(len(t) * BIN_WIDTH_DEG / 90, color="k", ls="--", lw=1, label="evenly spread")
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 45, 90])
    ax.tick_params(labelsize=8)
    if first:
        ax.set_ylabel("cells", fontsize=9)
        ax.set_xlabel("difference between RF angle and cone row angle (°)\n0 = along, 90 = across", fontsize=8.5)
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)
    ax.set_title(f"Difference between RF and cone row angle (n = {len(t)})", loc="left", fontsize=8.5)


def plot_rf_axes_vs_rows(ax, r):
    """Each RF's long axis, coloured by its angle to the radiating row (0 = along, 90 = across)."""
    geo, t = r["geo"], r["table"]
    onh_x, onh_y = r["onh_xy"]
    hw, hh = r["half_w"], r["half_h"]
    img = Image.open(cone_image_file(r["recording"]))
    iw, ih = img.size
    small = np.asarray(img.convert("L").reduce(4))  # grey, so blue/green lines stay visible
    img.close()
    ax.set_facecolor("black")
    # ORIENTATION [CONE IMAGE, fitted]: as the summary plots, no extra flip.
    ax.imshow(small, cmap="gray", extent=[0, iw, ih, 0], origin="upper", alpha=0.7,
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData)
    ax.plot([-hw, hw, hw, -hw, -hw], [-hh, -hh, hh, hh, -hh], "w--", lw=1, zorder=3)
    far = np.hypot(np.abs(onh_x) + hw, np.abs(onh_y) + hh) + 100
    for deg in np.arange(0, 360, ROW_LINE_STEP_DEG):
        a = np.deg2rad(deg)
        ax.plot([onh_x, onh_x + far * np.cos(a)], [onh_y, onh_y + far * np.sin(a)], color=ROW_LINE_COLOUR,
                lw=0.9, alpha=0.75, zorder=4)

    L = RF_AXIS_LENGTH_UM / 2
    outline = [patheffects.Stroke(linewidth=3.4, foreground="black"), patheffects.Normal()]
    outline_thin = [patheffects.Stroke(linewidth=2.0, foreground="black"), patheffects.Normal()]
    for _, c in t.iterrows():
        a = np.deg2rad(c.rf_axis_projected_deg)
        ax.plot([c.x_projected_um - L * np.cos(a), c.x_projected_um + L * np.cos(a)],
                [c.y_projected_um - L * np.sin(a), c.y_projected_um + L * np.sin(a)],
                color=ROW_CMAP(c.rf_vs_row_difference_deg / 90), lw=2.0 if 1 / c.tilt > ELONGATION_MIN else 0.8,
                solid_capstyle="round", alpha=0.3 if c.near_edge else 1.0, zorder=6,
                path_effects=None if c.near_edge else (outline if 1 / c.tilt > ELONGATION_MIN else outline_thin))
    ax.plot(onh_x, onh_y, "o", color="red", ms=10, mec="white", mew=1.5, zorder=8)
    xs, ys = np.r_[-hw, hw, onh_x], np.r_[-hh, hh, onh_y]
    ax.set_xlim(xs.min() - 60, xs.max() + 60)
    ax.set_ylim(ys.max() + 60, ys.min() - 60)  # ORIENTATION [DISPLAY only]: y down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("RF axes and radiating cone rows (orange)", loc="left", fontsize=8.5)


def plot_summary(ax, rows, obs, p, null_mean):
    rows = sorted(rows, key=lambda d: -d["diff"])  # most across the rows at the top
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0, 45, color=ALONG_COLOUR, alpha=0.07, zorder=0)
    ax.axvspan(45, 90, color=ACROSS_COLOUR, alpha=0.07, zorder=0)
    for yi, d in zip(y, rows):
        ax.scatter(d["diff"], yi, s=110, color=ACROSS_COLOUR, alpha=0.85, edgecolors="k", zorder=3)
    ax.axvline(45, color="k", ls="--", lw=1)
    ax.axvline(obs, color=ACROSS_COLOUR, lw=2.5)
    ax.text(obs, len(rows) - 0.4, f"mean {obs:.0f}°", fontsize=9, ha="center", va="bottom",
            color=ACROSS_COLOUR, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([d["recording"].replace("_", "/") for d in rows], fontsize=9)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.7, len(rows) + 0.4)
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(["0\nalong rows", "45", "90\nacross rows"])
    ax.set_xlabel("difference between main RF angle and cone row angle (°)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Across recordings (p = {p:.3f})", loc="left", fontsize=11, fontweight="bold")
    ax.text(0, -0.30,
            textwrap.fill(f"p = {p:.3f}: each recording's main RF direction was also paired with every OTHER "
                          f"recording's cone rows (all {math.factorial(len(rows))} combinations). Those "
                          f"mismatched pairings give {null_mean:.0f}° on average; the real pairing ({obs:.0f}°) "
                          f"is larger than {100 * (1 - p):.1f}% of them. So each recording's RFs lie across "
                          "their OWN cone rows, not just in some fixed direction.", 95),
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef5fd", edgecolor=ACROSS_COLOUR))


def summary_text(rows, obs, p, null_mean, width=105):
    n = len(rows)
    across = sum(d["diff"] >= 60 for d in rows)
    bullets = [
        f"RFs tend to be elongated ACROSS the predicted radiating cone rows: the main RF direction is "
        f"{obs:.0f}° from the rows on average, 60° or more in {across} of {n} recordings "
        f"(pairing RFs with other recordings' rows gives {null_mean:.0f}°; p = {p:.3f}).",
        *SUMMARY_FROM_OTHER_ANALYSES,
    ]
    return "\n".join(textwrap.fill(b, width=width, initial_indent="•  ", subsequent_indent="    ")
                     for b in bullets)


POOLED_ELONGATED_MIN = ELONGATION_MIN  # pooled histograms: same elongated threshold
POOLED_ROUND_MAX = 1.1  # pooled histograms: nearly round cells


def plot_pooled_hist(ax, cells, elongated):
    """All recordings pooled (edge cells left out): RF axis vs row for elongated or round cells.
    Labels say what fraction of cells lie more across than along their row (> 45°)."""
    from scipy.stats import binomtest

    cells = cells[~cells.near_edge]
    elong = 1 / cells.tilt
    if elongated:
        g = cells[elong > POOLED_ELONGATED_MIN]
        name, colour = f"Elongated cells (elongation > {POOLED_ELONGATED_MIN})", ACROSS_COLOUR
    else:
        g = cells[elong < POOLED_ROUND_MAX]
        name, colour = f"Nearly round cells (elongation < {POOLED_ROUND_MAX})", "0.55"
    a = g.rf_vs_row_difference_deg.values
    n, k = len(a), int((a > 45).sum())
    pval = binomtest(k, n, 0.5).pvalue if n else np.nan
    bins = np.arange(0, 90 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.axvspan(45, 90, color=ACROSS_COLOUR, alpha=0.07, zorder=0)
    counts, _, _ = ax.hist(a, bins=bins, color=colour, edgecolor="white", zorder=2)
    ax.set_ylim(0, counts.max() * 1.55)  # room for the label box above the bars
    ax.axhline(n * BIN_WIDTH_DEG / 90, color="k", ls="--", lw=1, zorder=3)
    ax.axvline(45, color="k", lw=0.6, zorder=3)
    ax.text(0.98, 0.97, f"{k / n * 100:.0f}% of cells ({k} of {n})\nlie more ACROSS than\nalong their row (> 45°)\n"
            f"50% if evenly spread; p = {pval:.2g}", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"), zorder=5)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(["0\nalong row", "45", "90\nacross row"])
    ax.tick_params(labelsize=8)
    ax.set_ylabel("cells", fontsize=9)
    ax.set_xlabel("difference between RF angle and cone row angle (°)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{name}, all {cells.recording.nunique()} recordings pooled", loc="left",
                 fontsize=9.5, fontweight="bold")


HOW_TO_READ = (
    "Red circle = optic nerve head. Recordings ordered from\nmost to least across the rows (left to right).\n\n"
    "Each recording's main RF direction (mean long axis,\n"
    "elongated cells count more) is compared with the\n"
    "radiating row direction there (line from the optic\n"
    "nerve head to each RF centre, averaged over cells).\n\n"

    f"Cells: quality >= {rca.QUALITY_MIN}, more than {rca.EDGE_EXCLUSION_UM} µm inside\n"
    "the stimulus edge (faded lines are left out).\n"
    f"Elongated: elongation (long / short axis = 1/tilt)\n> {ELONGATION_MIN} "
    "(thick lines, black histogram outline)."
)


def main():
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    recordings, onh_positions = rca.recordings_with_onh()
    n = len(recordings)
    # Load one recording at a time (the per-cell maps are freed inside analyse_recording; what
    # is kept - the cell table and the combined covariance map - is small).
    results, summary_rows = [], []
    for rec, ch in recordings:
        res = rca.analyse_recording(rec, ch, onh_positions[rec])
        rf_dir, strength, row_dir = main_rf_vs_row(res["table"])
        summary_rows.append(dict(recording=rec, rf=rf_dir, row=row_dir, strength=strength,
                                 diff=float(axis_diff(rf_dir, row_dir))))
        results.append(res)
    obs, p, null_mean = across_recording_test([d["rf"] for d in summary_rows],
                                              [d["row"] for d in summary_rows])
    print(f"Across recordings: mean RF-row difference {obs:.1f}° (mismatched pairings {null_mean:.1f}°), p = {p:.4f}")

    # Landscape: 3 rows (cone image + covariance RFs | RF axes vs rows | histogram) x one
    # column per recording, ordered most across the rows on the left; summary column on the right.
    panel_in = 3.5
    fig = plt.figure(figsize=(panel_in * n + 7.5 + 5.2, panel_in * 3 + 2.2), facecolor="white")
    fig_h, fig_w = fig.get_figheight(), fig.get_figwidth()
    outer = GridSpec(1, 3, figure=fig, width_ratios=[panel_in * n, 6.2, 4.6], wspace=0.13,
                     top=1 - 0.35 / fig_h, bottom=0.55 / fig_h, left=0.3 / fig_w, right=1 - 0.3 / fig_w)
    grid = GridSpecFromSubplotSpec(3, n, subplot_spec=outer[0], hspace=0.22, wspace=0.08)
    right = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], height_ratios=[1.05, 1.25, 0.75],
                                    hspace=0.85)

    order = np.argsort([-d["diff"] for d in summary_rows], kind="stable")
    for k, i in enumerate(order):
        res, d = results[i], summary_rows[i]
        ax0 = fig.add_subplot(grid[0, k])
        rca.plot_maps(ax0, None, res, colorbar=False, onh_lines=False, onh_marker="o",
                      cov_alpha=COVARIANCE_ALPHA_ROW1)
        if ax0.get_legend() is not None:
            ax0.get_legend().remove()
        ax0.set_title(f"{d['recording'].replace('_', '/')}\n{res['channel']}", loc="left",
                      fontsize=9, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_xlabel("")
        ax0.set_ylabel("")
        ax1 = fig.add_subplot(grid[1, k])
        plot_rf_axes_vs_rows(ax1, res)
        plot_row_histogram(fig.add_subplot(grid[2, k]), res, first=(k == 0))
    pooled = pd.concat([r["table"] for r in results], ignore_index=True)
    del results

    # One vertical colour bar at the right-hand end of the middle row
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x1 + 0.004, pos.y0, 0.006, pos.height])
    sm = plt.cm.ScalarMappable(cmap=ROW_CMAP, norm=plt.Normalize(0, 90))
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 45, 90])
    cb.ax.set_yticklabels(["0° along row", "45°", "90° across row"], fontsize=8)

    # Right-hand column: summary text, across-recording plot, how to read
    ax_txt = fig.add_subplot(right[0])
    ax_txt.axis("off")
    ax_txt.text(0, 1, "Summary\n" + summary_text(summary_rows, obs, p, null_mean, width=62), va="top", ha="left",
                fontsize=9, linespacing=1.35, transform=ax_txt.transAxes,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef5fd", edgecolor=ACROSS_COLOUR))
    plot_summary(fig.add_subplot(right[1]), summary_rows, obs, p, null_mean)

    # Far-right column: all recordings pooled, elongated vs nearly round cells
    far = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[2], height_ratios=[1, 1, 0.35], hspace=0.45)
    plot_pooled_hist(fig.add_subplot(far[0]), pooled, elongated=True)
    plot_pooled_hist(fig.add_subplot(far[1]), pooled, elongated=False)
    ax_note = fig.add_subplot(far[2])
    ax_note.axis("off")
    ax_note.text(0, 1, "Dashed line: count if angles were evenly spread. Each cell counts once\nhere, but cells from "
                 "the same recording are not independent (they tend to share\n"
                 "one direction), so the across-recordings test is the main result.",
                 va="top", fontsize=8.3, transform=ax_note.transAxes)
    notes = fig.add_subplot(right[2])
    notes.axis("off")
    notes.text(0, 1.0, HOW_TO_READ, va="top", fontsize=8.5, linespacing=1.3, transform=notes.transAxes)

    for fmt in ("png", "pdf"):
        out = rca.OUTPUT_FOLDER / f"{OUT_NAME}.{fmt}"
        fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
