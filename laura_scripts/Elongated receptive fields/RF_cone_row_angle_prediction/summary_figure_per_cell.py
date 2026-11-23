"""
Corrected summary figure (23/11/2026): RF orientation vs the predicted radiating cone rows, with PER-CELL tests.
Same panels and style as summary_figure.py (which is left unchanged); only the main test is replaced.

Why a new test: summary_figure.py compares each recording's MEAN RF direction with its MEAN row direction (the
rows at its cells averaged). That assumes the rows hardly change direction across one array. Where the optic
nerve head (ONH) is on or near the array the rows fan out (row spread = |mean of exp(2i*row angle)| over the
recording's cells: 1 = parallel, 0 = all directions; 0.08-0.35 in 4 of 7 recordings), so the mean row direction
is not meaningful and the test can both miss and fake an effect.

Tests here compare each cell with ITS OWN row (cells as summary_figure.py: quality >= rca.QUALITY_MIN, not near
the stimulus edge). Statistic: mean over cells of |RF long axis - own row| (0 = along, 90 = across).
  1. Within-recording shuffle (main test): RF axes permuted among cells of the same recording (N_PERM x).
     Asks whether RFs follow the LOCAL row direction as the rows fan out; a recording-wide bias (flattening,
     projector) cannot produce that. Recordings whose rows are nearly parallel contribute little power.
  2. Fake ONH positions: each recording's ONH moved to a random position within FAKE_ONH_RADIUS_UM of its cells'
     centroid, rows recomputed (N_PERM x). Asks whether the real ONH position is special.

Run from the repo root (loads each recording like summary_figure.py; ~15-20 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/summary_figure_per_cell.py"
Output: rca.OUTPUT_FOLDER / rf_vs_radiating_cone_rows_summary_per_cell.png/.pdf
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import rf_cone_row_angle as rca  # same folder
import summary_figure as sf  # same folder: plotting functions reused unchanged

OUT_NAME = "rf_vs_radiating_cone_rows_summary_per_cell"
N_PERM = 5000
FAKE_ONH_RADIUS_UM = 1500.0
ACROSS, ALONG = sf.ACROSS_COLOUR, sf.ALONG_COLOUR


def cell_table(res):
    return res["table"][~res["table"].near_edge]


def per_cell_tests(tables, onh, sel=lambda t: t, seed=1):
    """Statistic + null distributions of test 1 (within-recording shuffle) and test 2 (fake ONH)."""
    rng = np.random.default_rng(seed)
    groups = [(r, sel(t)) for r, t in tables.items()]
    groups = [(r, t) for r, t in groups if len(t)]
    real = np.concatenate([sf.axis_diff(t.rf_axis_projected_deg, t.cone_row_axis_projected_deg) for _, t in groups]).mean()
    null1 = np.array([np.concatenate([sf.axis_diff(rng.permutation(t.rf_axis_projected_deg.values),
                                                   t.cone_row_axis_projected_deg.values) for _, t in groups]).mean()
                      for _ in range(N_PERM)])
    null2 = []
    for _ in range(N_PERM):
        parts = []
        for _, t in groups:
            P = np.c_[t.x_projected_um, t.y_projected_um]
            a, rad = rng.uniform(0, 2 * np.pi), FAKE_ONH_RADIUS_UM * np.sqrt(rng.uniform())
            o = P.mean(0) + rad * np.array([np.cos(a), np.sin(a)])
            rows = np.degrees(np.arctan2(P[:, 1] - o[1], P[:, 0] - o[0])) % 180
            parts.append(sf.axis_diff(t.rf_axis_projected_deg.values, rows))
        null2.append(np.concatenate(parts).mean())
    null2 = np.array(null2)
    n = sum(len(t) for _, t in groups)
    return dict(real=float(real), null1=null1, null2=null2, p1=float(np.mean(null1 >= real)),
                p2=float(np.mean(null2 >= real)), n=n)


def plot_per_recording(ax, stats):
    stats = sorted(stats, key=lambda d: -d["pct_across"])
    y = np.arange(len(stats))[::-1]
    ax.axvspan(0, 50, color=ALONG, alpha=0.07, zorder=0)
    ax.axvspan(50, 100, color=ACROSS, alpha=0.07, zorder=0)
    for yi, d in zip(y, stats):
        ax.scatter(d["pct_across"], yi, s=40 + 3 * d["n"] ** 0.5 * 10, color=ACROSS, alpha=0.85, edgecolors="k",
                   zorder=3)
        ax.text(2, yi, f"n = {d['n']}, row spread {d['spread']:.2f}", va="center", fontsize=8, color="0.3")
    ax.axvline(50, color="k", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([d["recording"].replace("_", "/") for d in stats], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(stats) - 0.3)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0 %", "50 %\n(no preference)", "100 %"])
    ax.set_xlabel("% of cells whose RF lies more ACROSS than along its own row (> 45°)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Each recording, every cell vs its OWN row", loc="left", fontsize=11, fontweight="bold")


def plot_test(ax, t, title):
    for null, p, col, lab in [(t["null1"], t["p1"], "0.45", "1: RFs shuffled within recording"),
                              (t["null2"], t["p2"], "#d98c3a", "2: ONH at random positions")]:
        ax.hist(null, bins=50, histtype="stepfilled", alpha=0.35, color=col, label=f"{lab} (p = {p:.3f})")
    ax.axvline(t["real"], color=ACROSS, lw=2.5)
    ax.text(t["real"], ax.get_ylim()[1] * 1.02, f"real {t['real']:.1f}°", color=ACROSS, fontsize=8.5,
            va="bottom", ha="center", fontweight="bold")
    ax.set_xlabel("mean |RF axis - own row| (°)\nhigher = more across", fontsize=8.5)
    ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"{title} (n = {t['n']})", loc="left", fontsize=9.5, fontweight="bold", pad=14)


def summary_text(t_all, t_el, stats, width=62):
    k_across = sum(d["pct_across"] > 50 for d in stats)
    lo, hi = min(d["pct_across"] for d in stats), max(d["pct_across"] for d in stats)
    bullets = [
        f"RFs tend to lie ACROSS their own radiating cone row: in {k_across} of {len(stats)} recordings more cells "
        f"lie across than along ({lo:.0f}-{hi:.0f} %).",
        f"They follow the LOCAL row direction as the rows fan out: RF axes shuffled within each recording give "
        f"{t_all['null1'].mean():.1f}° vs real {t_all['real']:.1f}° (all cells, p = {t_all['p1']:.3f}; elongated "
        f"cells p = {t_el['p1']:.3f}). A recording-wide bias (flattening, projector) cannot produce this.",
        f"The real ONH position fits better than random positions, but only borderline (all p = {t_all['p2']:.3f}, "
        f"elongated p = {t_el['p2']:.3f}).",
        "Effect size is modest (a few degrees on average; ~60 % rather than 50 % of cells across).",
        *sf.SUMMARY_FROM_OTHER_ANALYSES[1:],
        "Replaces the mean-direction test of the original figure (p = 0.007), which is not meaningful where the "
        "rows fan out (row spread 0.08-0.35 in 4 of 7 recordings).",
    ]
    return "\n".join(textwrap.fill(b, width=width, initial_indent="•  ", subsequent_indent="    ") for b in bullets)


HOW_TO_READ = (
    "Red circle = optic nerve head. Recordings ordered from\nmost to least across the rows (left to right),\n"
    "by % of cells lying across their own row.\n\n"
    "Each cell's RF long axis is compared with the radiating\nrow through its own centre (line from the optic nerve\n"
    "head to the RF centre). Row spread: 1 = the rows at\nthe recording's cells are parallel, 0 = all directions.\n\n"
    f"Cells: quality >= {rca.QUALITY_MIN}, more than {rca.EDGE_EXCLUSION_UM} µm inside\n"
    "the stimulus edge (faded lines are left out).\n"
    f"Elongated: elongation (long / short axis = 1/tilt)\n> {sf.ELONGATION_MIN} (thick lines, black histogram outline)."
)


def main():
    recordings, onh_positions = rca.recordings_with_onh()
    n = len(recordings)
    results, tables, stats = [], {}, []
    for rec, ch in recordings:
        res = rca.analyse_recording(rec, ch, onh_positions[rec])
        t = cell_table(res)
        d = sf.axis_diff(t.rf_axis_projected_deg, t.cone_row_axis_projected_deg)
        spread = float(abs(np.mean(np.exp(2j * np.deg2rad(t.cone_row_axis_projected_deg)))))
        stats.append(dict(recording=rec, pct_across=float((d > 45).mean() * 100), n=len(t), spread=spread))
        tables[rec] = t
        results.append(res)
    t_all = per_cell_tests(tables, onh_positions)
    t_el = per_cell_tests(tables, onh_positions, sel=lambda t: t[(1 / t.tilt) > sf.ELONGATION_MIN], seed=2)
    for name, t in [("all", t_all), ("elongated", t_el)]:
        print(f"{name}: n {t['n']}, real {t['real']:.1f}°, test 1 null {t['null1'].mean():.1f}° p = {t['p1']:.4f}; "
              f"test 2 null {t['null2'].mean():.1f}° p = {t['p2']:.4f}")

    panel_in = 3.5
    fig = plt.figure(figsize=(panel_in * n + 7.5 + 5.2, panel_in * 3 + 2.2), facecolor="white")
    fig_h, fig_w = fig.get_figheight(), fig.get_figwidth()
    outer = GridSpec(1, 3, figure=fig, width_ratios=[panel_in * n, 6.2, 4.6], wspace=0.13,
                     top=1 - 0.35 / fig_h, bottom=0.55 / fig_h, left=0.3 / fig_w, right=1 - 0.3 / fig_w)
    grid = GridSpecFromSubplotSpec(3, n, subplot_spec=outer[0], hspace=0.22, wspace=0.08)
    right = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], height_ratios=[1.25, 0.95, 0.75], hspace=0.6)

    order = np.argsort([-d["pct_across"] for d in stats], kind="stable")
    for k, i in enumerate(order):
        res, d = results[i], stats[i]
        ax0 = fig.add_subplot(grid[0, k])
        rca.plot_maps(ax0, None, res, colorbar=False, onh_lines=False, onh_marker="o",
                      cov_alpha=sf.COVARIANCE_ALPHA_ROW1)
        if ax0.get_legend() is not None:
            ax0.get_legend().remove()
        ax0.set_title(f"{d['recording'].replace('_', '/')}\n{res['channel']}", loc="left", fontsize=9,
                      fontweight="bold")
        ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_xlabel(""); ax0.set_ylabel("")
        ax1 = fig.add_subplot(grid[1, k])
        sf.plot_rf_axes_vs_rows(ax1, res)
        sf.plot_row_histogram(fig.add_subplot(grid[2, k]), res, first=(k == 0))
    pooled = pd.concat([r["table"] for r in results], ignore_index=True)
    del results

    pos = ax1.get_position()
    cax = fig.add_axes([pos.x1 + 0.004, pos.y0, 0.006, pos.height])
    sm = plt.cm.ScalarMappable(cmap=sf.ROW_CMAP, norm=plt.Normalize(0, 90))
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 45, 90])
    cb.ax.set_yticklabels(["0° along row", "45°", "90° across row"], fontsize=8)

    ax_txt = fig.add_subplot(right[0])
    ax_txt.axis("off")
    ax_txt.text(0, 1, "Summary (per-cell tests)\n" + summary_text(t_all, t_el, stats), va="top", ha="left",
                fontsize=9, linespacing=1.35, transform=ax_txt.transAxes,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef5fd", edgecolor=ACROSS))
    plot_per_recording(fig.add_subplot(right[1]), stats)
    tests = GridSpecFromSubplotSpec(1, 2, subplot_spec=right[2], wspace=0.3)
    plot_test(fig.add_subplot(tests[0]), t_all, "All cells")
    plot_test(fig.add_subplot(tests[1]), t_el, "Elongated cells")

    far = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[2], height_ratios=[1, 1, 0.55], hspace=0.45)
    sf.plot_pooled_hist(fig.add_subplot(far[0]), pooled, elongated=True)
    sf.plot_pooled_hist(fig.add_subplot(far[1]), pooled, elongated=False)
    ax_note = fig.add_subplot(far[2])
    ax_note.axis("off")
    ax_note.text(0, 1, "Dashed line: count if angles were evenly spread. Each cell counts once\nhere, but cells from "
                 "the same recording are not independent, and a\nrecording-wide bias could also skew these. The "
                 "within-recording\nshuffle test (middle column) is the main result.\n\n" + HOW_TO_READ,
                 va="top", fontsize=8.3, linespacing=1.25, transform=ax_note.transAxes)

    for fmt in ("png", "pdf"):
        out = rca.OUTPUT_FOLDER / f"{OUT_NAME}.{fmt}"
        fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
