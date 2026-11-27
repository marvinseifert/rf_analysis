"""
Visual explainer for the RF / moving-bar / cone-row results (23/11/2026).

A  Schematic, local view: Laura's row mosaic (UV-B row, R/G, G/R, B-UV row ...) with the rows horizontal, an RF
   elongated ACROSS the rows (the RF result), and the two bar alternatives being tested: bar moving across the rows
   (edge along them) or along the rows (edge across them).
B  Schematic, whole retina: rows radiate from the optic nerve head (ONH); RFs are elongated around the ONH.
NOTE (23/11/2026): per cell (each cell vs its OWN local row) the bar preference is mixed: across the rows in 22/07,
14/08, 15/10, along them in 14/05 and 15/05, 50 % pooled. The earlier "all 5 across, p = 0.025" came from
comparing a recording's mean bar axis with a MEAN row axis, which is meaningless where the rows fan out.
C  Real data, every tuned cell rotated so its local cone row is horizontal: preferred bar motion axis
   (line length = OSI) and RF long axis (elongated cells); plus each recording's main axes.
Data: bar_orientation_vs_cone_rows_cells_peak_response.csv (bar_orientation_vs_cone_rows.py).
Not to scale: real RFs are ~100-150 µm across (~10 cone rows); the schematic RF is shrunk to show cones.

Run from the repo root:
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/plot_bar_rows_schematic.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
df = pd.read_csv(FIG / "bar_orientation_vs_cone_rows_cells_peak_response.csv")
tuned = df[(df.osi >= 0.1) & (df.n_spikes >= 100) & (~df.near_edge.astype(bool))].copy()
COL = {"UV": "#c040c0", "Blue": "#3070e0", "Red": "#e03030", "Green": "#30a040"}
P, HS, HD = 6.0, 6.0, 3.0  # UV-B pitch, single-row height, R/G row height (µm)


def mosaic(width, height):
    """Laura's row mosaic, rows along x. Returns list of (x, y, type, single?)."""
    rows = [(0.0, [("UV", 0), ("Blue", P)], True), (HS / 2 + HD / 2, [("Green", 0), ("Red", P)], False),
            (HS / 2 + 1.5 * HD, [("Red", 0), ("Green", P)], False), (HS + 2 * HD, [("Blue", 0), ("UV", P)], True),
            (1.5 * HS + 2.5 * HD, [("Red", 0), ("Green", P)], False),
            (1.5 * HS + 3.5 * HD, [("Green", 0), ("Red", P)], False)]
    period = 2 * HS + 4 * HD
    out = []
    for j in range(int(height // period) + 2):
        for i in range(int(width // (2 * P)) + 2):
            for y, cones, single in rows:
                for t, x in cones:
                    out.append((i * 2 * P + x, j * period + y, t, single))
    return out


def rel_angle(a, b):
    """Signed angle of axis a relative to axis b, in (-90, 90] (polar frame, clockwise)."""
    return (a - b + 90) % 180 - 90


fig = plt.figure(figsize=(20, 8.5))
gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.25, 1, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

# ------------------------------------------------------------------ A: local schematic (two bar conditions)
for k, (preferred, ttl) in enumerate([(True, "Alternative 1: bar edge ALONG the rows, bar moves\nACROSS the rows "
                                             "(22/07, 14/08, 15/10 lean this way)"),
                                      (False, "Alternative 2: bar edge ACROSS the rows, bar moves\nALONG the rows "
                                              "(14/05, 15/05 lean this way)")]):
    ax = fig.add_subplot(gs[k, 0])
    W, H = 72, 60
    for x, y, t, single in mosaic(W, H):
        if x > W or y > H:
            continue
        if single:
            ax.add_patch(Circle((x, y), 2.4, color=COL[t], alpha=0.9, lw=0))
        else:
            ax.add_patch(Ellipse((x, y), 4.8, 2.4, color=COL[t], alpha=0.9, lw=0))
    cx, cy = 36, 30
    ax.add_patch(Ellipse((cx, cy), 26, 52, fill=True, fc=(1, 0.85, 0.2, 0.25), ec="darkorange", lw=2.5, zorder=5))
    if preferred:
        ax.add_patch(Rectangle((-2, 14), W + 4, 5, color="white", alpha=0.85, zorder=6, ec="k", lw=1))
        ax.add_patch(FancyArrowPatch((W + 6, 16.5), (W + 6, 46), arrowstyle="-|>", mutation_scale=25, lw=3,
                                     color="k", zorder=7, clip_on=False))
    else:
        ax.add_patch(Rectangle((14, -2), 5, H + 4, color="white", alpha=0.85, zorder=6, ec="k", lw=1))
        ax.add_patch(FancyArrowPatch((16.5, H + 6), (52, H + 6), arrowstyle="-|>", mutation_scale=25, lw=3,
                                     color="k", zorder=7, clip_on=False))
    ax.set_xlim(-4, W + 12); ax.set_ylim(H + 10, -4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl, fontsize=10, loc="left")
    if k == 0:
        ax.text(-4, H + 9, "rows run left-right; orange = RF (elongated across the rows; not to scale)",
                fontsize=8, color="0.3")

# legend for cones
ax = fig.add_subplot(gs[:, 1])
ax.axis("off")
# ------------------------------------------------------------------ B: whole-retina schematic
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
ax.add_patch(Circle((0, 0), 1.0, fc="#f4f0e8", ec="0.5", lw=1.5))
for th in np.linspace(0, 2 * np.pi, 36, endpoint=False):
    ax.plot([0.08 * np.cos(th), 0.98 * np.cos(th)], [0.08 * np.sin(th), 0.98 * np.sin(th)], color="#e08040", lw=0.8,
            alpha=0.7)
ax.add_patch(Circle((0, 0), 0.07, color="k", zorder=5))
ax.text(0.09, -0.12, "ONH", fontsize=9)
for r, th in [(0.45, 0.4), (0.7, 1.9), (0.55, 3.3), (0.8, 4.4), (0.4, 5.4), (0.75, 0.0), (0.6, 2.7)]:
    x, y = r * np.cos(th), r * np.sin(th)
    ax.add_patch(Ellipse((x, y), 0.08, 0.2, angle=np.degrees(th), fc=(1, 0.85, 0.2, 0.5), ec="darkorange", lw=1.8,
                         zorder=6))
ax.set_title("Whole retina (schematic):\ncone rows (orange lines) radiate from the optic\nnerve head (ONH); "
             "RFs (orange ellipses) tend to be\nelongated AROUND the ONH, i.e. across the rows\n"
             "(your RF result: 64 % of elongated cells, p = 0.014)", fontsize=10, loc="left")
for i, (t, c) in enumerate(COL.items()):
    ax.add_patch(Circle((-1.1 + i * 0.32, -1.12), 0.035, color=c, clip_on=False))
    ax.text(-1.06 + i * 0.32, -1.135, t, fontsize=8, va="center")

# ------------------------------------------------------------------ C: real data in the row frame
tuned["bar_rel"] = rel_angle(tuned.bar_axis_polar, tuned.row_axis_polar)
tuned["rf_rel"] = rel_angle(tuned.rf_axis_polar, tuned.row_axis_polar)


def axis_line(ax, rel_deg, length, **kw):
    # row frame: rows horizontal; an axis at rel_deg from the rows (polar convention: clockwise positive)
    a = np.deg2rad(rel_deg)
    dx, dy = np.cos(a) * length, np.sin(a) * length
    ax.plot([-dx, dx], [-dy, dy], **kw)


ax = fig.add_subplot(gs[0, 2])
for _, r in tuned.iterrows():
    axis_line(ax, r.bar_rel, r.osi, color="k", lw=1, alpha=0.5)
ax.axhline(0, color="#e08040", lw=3, alpha=0.6, zorder=0)
ax.set_xlim(-0.8, 0.8); ax.set_ylim(0.8, -0.8); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.78, 0.06, "cone row", color="#e08040", ha="right", fontsize=8)
ax.set_title(f"Real data: preferred bar MOTION axis of every\ntuned cell (n = {len(tuned)}), rotated so its row is "
             "horizontal\n(line length = OSI)", fontsize=9, loc="left")

ax = fig.add_subplot(gs[1, 2])
el = tuned[tuned.elongated.astype(bool)]
for _, r in el.iterrows():
    axis_line(ax, r.rf_rel, 1 - min(r.tilt, 1), color="darkorange", lw=2, alpha=0.8)
ax.axhline(0, color="#e08040", lw=3, alpha=0.6, zorder=0)
ax.set_xlim(-0.6, 0.6); ax.set_ylim(0.6, -0.6); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Real data: RF long axis of the tuned cells with\nelongated RFs (n = {len(el)}), same frame\n"
             "(line length = elongation)", fontsize=9, loc="left")

ax = fig.add_subplot(gs[:, 3])


def axial_mean(deg, w):
    z = np.sum(np.asarray(w) * np.exp(2j * np.deg2rad(deg)))
    return (np.rad2deg(np.angle(z)) / 2 + 90) % 180 - 90


cols = plt.cm.tab10(np.arange(10))
for i, (rec, d) in enumerate(tuned.groupby("recording")):
    m = axial_mean(d.bar_rel, d.osi)
    axis_line(ax, m, 1.0, color=cols[i], lw=4, solid_capstyle="round", label=f"{rec} (n = {len(d)}): {abs(m):.0f}°")
ax.axhline(0, color="#e08040", lw=6, alpha=0.5, zorder=0)
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="0.8", lw=0.8))
ax.set_xlim(-1.2, 1.2); ax.set_ylim(1.2, -1.2); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.text(1.15, 0.08, "rows", color="#e08040", ha="right", fontsize=9)
ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
ax.set_title("Each recording's main preferred motion axis\nrelative to each cell's OWN row: mixed\n"
             "(3 across, 2 along; pooled 50 % across)", fontsize=9, loc="left")

fig.suptitle("Cone rows, RF shape and moving-bar preference in adult zebrafish RGCs (5 recordings with moving bars; "
             "rows assumed to radiate from the optic nerve head). RF elongation follows the rows; bar preference "
             "does not consistently.", fontsize=12, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"bar_rows_schematic.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "bar_rows_schematic.png")
