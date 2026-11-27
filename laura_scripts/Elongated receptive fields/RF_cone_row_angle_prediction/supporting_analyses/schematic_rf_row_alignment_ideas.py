"""
Schematic figure (no data): the RF-vs-cone-row alignment we see and the tentative computational ideas
(23/11/2026), with the tentative conclusions of the RF-orientation analyses as notes.

Run from the repo root:
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/schematic_rf_row_alignment_ideas.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
COL = {"UV": "#c040c0", "B": "#3070e0", "R": "#e03030", "G": "#30a040"}
RF_FC, RF_EC = (1, 0.82, 0.2, 0.30), "darkorange"
ROWC = "#e08040"
P, HS, HD = 6.0, 6.0, 3.0
rng = np.random.default_rng(3)


def mosaic(w, h):
    rows = [(0.0, [("UV", 0), ("B", P)], 1), (HS / 2 + HD / 2, [("G", 0), ("R", P)], 0),
            (HS / 2 + 1.5 * HD, [("R", 0), ("G", P)], 0), (HS + 2 * HD, [("B", 0), ("UV", P)], 1),
            (1.5 * HS + 2.5 * HD, [("R", 0), ("G", P)], 0), (1.5 * HS + 3.5 * HD, [("G", 0), ("R", P)], 0)]
    per = 2 * HS + 4 * HD
    out = []
    for j in range(int(h // per) + 2):
        for i in range(int(w // (2 * P)) + 2):
            for y, cs, single in rows:
                for t, x in cs:
                    if i * 2 * P + x <= w and j * per + y <= h:
                        out.append((i * 2 * P + x, j * per + y, t, single))
    return out


def draw_cones(ax, cones, alpha=0.9):
    for x, y, t, single in cones:
        if single:
            ax.add_patch(Circle((x, y), 2.3, color=COL[t], alpha=alpha, lw=0))
        else:
            ax.add_patch(Ellipse((x, y), 4.6, 2.3, color=COL[t], alpha=alpha, lw=0))


def finish(ax, w, h, title):
    ax.set_xlim(-4, w + 4); ax.set_ylim(h + 4, -4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=kw.pop("style", "-|>"), mutation_scale=kw.pop("ms", 18),
                                 lw=kw.pop("lw", 2.5), color=kw.pop("color", "k"), zorder=8, **kw))


fig = plt.figure(figsize=(24, 15))
gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.12)

# ---------------- A: the observation, local view
W, H = 84, 66
ax = fig.add_subplot(gs[0, 0])
draw_cones(ax, mosaic(W, H))
ax.add_patch(Ellipse((20, 33), 22, 50, fc=RF_FC, ec=RF_EC, lw=2.5, zorder=5))
ax.text(20, 33, "common", ha="center", va="center", fontsize=9, zorder=6, weight="bold")
ax.add_patch(Ellipse((59, 33), 44, 18, fc=(0.8, 0.8, 0.8, 0.25), ec="0.4", lw=2, ls="--", zorder=5))
ax.text(59, 33, "rare", ha="center", va="center", fontsize=9, zorder=6, color="0.3", weight="bold")
arrow(ax, (2, H + 6), (W - 2, H + 6), style="<|-|>", lw=2.5, ms=12, color=ROWC)
ax.text(W / 2, H + 8, "cone row direction", ha="center", va="top", color=ROWC, fontsize=8.5)
finish(ax, W, H + 12, "A  Observation: over CRYSTALLINE mosaic, RFs avoid lying\nALONG the cone rows (~5 % along) and "
                 "tend to lie ACROSS them")

# ---------------- B: disrupted mosaic and larval patch
ax = fig.add_subplot(gs[0, 1])
cones = mosaic(W, H)
dis = []
for x, y, t, s in cones:
    dis.append((x + rng.normal(0, 2.2), y + rng.normal(0, 2.2), t, s))
draw_cones(ax, dis, alpha=0.75)
for (cx, cy, ang) in [(22, 20, 20), (60, 26, 115), (38, 50, 70)]:
    ax.add_patch(Ellipse((cx, cy), 18, 30, angle=ang, fc=RF_FC, ec=RF_EC, lw=2, zorder=5))
finish(ax, W, H, "B  Over DISRUPTED mosaic (and in the non-crystalline\nlarval patch): no RF bias relative to the "
                 "ONH-radial\ndirection, i.e. where crystalline rows would run")

# ---------------- C: whole retina
ax = fig.add_subplot(gs[0, 2:])
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0, 0), 1.0, fc="#f4f0e8", ec="0.5", lw=1.5))
for th in np.linspace(0, 2 * np.pi, 40, endpoint=False):
    sw = 0.25  # rows twist slightly away from radial (15/05: ~10-15°)
    rr = np.linspace(0.3, 0.98, 30)
    ax.plot(rr * np.cos(th + sw * (rr - 0.3)), rr * np.sin(th + sw * (rr - 0.3)), color=ROWC, lw=0.8, alpha=0.7)
ax.add_patch(Circle((0, 0), 0.28, fc="#ece4d4", ec="0.55", ls="--", lw=1.2, zorder=3))
for _ in range(60):
    r, a = 0.27 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    ax.plot(r * np.cos(a), r * np.sin(a), ".", color=rng.choice(list(COL.values())), ms=3, zorder=4)
ax.add_patch(Circle((0, 0), 0.05, color="k", zorder=6))
ax.text(0.0, -0.11, "ONH", ha="center", fontsize=9, zorder=7)
ax.text(0.0, 0.17, "larval patch\n(non-crystalline)", ha="center", fontsize=8, color="0.3", zorder=7)
for r, th in [(0.5, 0.5), (0.72, 1.8), (0.6, 3.2), (0.82, 4.3), (0.5, 5.4), (0.78, 0.0), (0.62, 2.5), (0.45, 3.9)]:
    x, y = r * np.cos(th), r * np.sin(th)
    ax.add_patch(Ellipse((x, y), 0.07, 0.2, angle=np.degrees(th), fc=RF_FC, ec=RF_EC, lw=1.8, zorder=6))
for (x, y, a) in [(0.12, 0.05, 30), (-0.1, -0.15, 100)]:
    ax.add_patch(Ellipse((x, y), 0.06, 0.13, angle=a, fc=RF_FC, ec=RF_EC, lw=1.5, zorder=6))
ax.add_patch(Rectangle((0.38, -0.62), 0.32, 0.22, angle=20, fc="0.75", ec="0.4", alpha=0.6, zorder=5))
ax.text(0.62, -0.72, "disrupted patch", fontsize=8, color="0.3")
ax.set_title("C  Whole retina: rows radiate from the optic nerve head (and twist slightly), so RFs lying across "
             "the rows are elongated\nAROUND the ONH; in the larval patch and disrupted patches, no bias relative to "
             "the radial direction", loc="left",
             fontsize=10)

# ---------------- D: rings vs spokes (H2)
for k, (kind, ttl) in enumerate([("rings", "D  Idea: ring vs spoke patterns (population layout)\nrings around the ONH "
                                           "drive the population a little more (~6 %)"), ]):
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
    for r in np.arange(0.15, 1.05, 0.15):
        ax.add_patch(Circle((-0.5, 0), r * 0.5, fill=False, ec="k", lw=2.2 if int(r / 0.15) % 2 else 0.5))
    for th in np.linspace(0, np.pi, 12, endpoint=False):
        ax.plot([0.5 - 0.48 * np.cos(th), 0.5 + 0.48 * np.cos(th)], [-0.48 * np.sin(th), 0.48 * np.sin(th)], "k",
                lw=1.6)
    for x0 in (-0.5, 0.5):
        for th in (0.3, 2.0, 4.0):
            x, y = x0 + 0.33 * np.cos(th), 0.33 * np.sin(th)
            ax.add_patch(Ellipse((x, y), 0.07, 0.17, angle=np.degrees(th), fc=RF_FC, ec=RF_EC, lw=1.8, zorder=5))
    ax.text(-0.5, -0.62, "rings: stronger", ha="center", fontsize=9, weight="bold")
    ax.text(0.5, -0.62, "spokes: weaker", ha="center", fontsize=9, color="0.35")
    ax.set_title(ttl, loc="left", fontsize=10)

# ---------------- E: radial motion (H3)
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
    ax.plot([0.1 * np.cos(th), 1.0 * np.cos(th)], [0.1 * np.sin(th), 1.0 * np.sin(th)], color=ROWC, lw=0.7, alpha=0.6)
ax.add_patch(Circle((0, 0), 0.04, color="k"))
for r in (0.45, 0.62, 0.79):
    ax.add_patch(Circle((0, 0), r, fill=False, ec="k", lw=2.5 if r == 0.62 else 0.8, ls="-" if r == 0.62 else ":"))
for th in (0.6, 2.4, 4.2):
    x, y = 0.62 * np.cos(th), 0.62 * np.sin(th)
    ax.add_patch(Ellipse((x, y), 0.08, 0.2, angle=np.degrees(th), fc=RF_FC, ec=RF_EC, lw=1.8, zorder=5))
    arrow(ax, (0.98 * np.cos(th), 0.98 * np.sin(th)), (0.74 * np.cos(th), 0.74 * np.sin(th)), lw=2.4, ms=15)  # contracting
    arrow(ax, (0.3 * np.cos(th), 0.3 * np.sin(th)), (0.5 * np.cos(th), 0.5 * np.sin(th)), lw=1.2, ms=10,
          color="0.6")  # expanding (less preferred), pointing outward
ax.text(0, -1.08, "black: contracting (preferred)   grey: expanding", ha="center", fontsize=8)
ax.set_title("E  Idea: expanding vs contracting (radial) motion\ntuned crystalline cells prefer edges around the ONH moving\n"
             "ALONG the rows (radially); preferred direction mostly INWARD =\nCONTRACTING (14/05, ONH central: 12 of 13 "
             "DS cells)", loc="left", fontsize=10)

# ---------------- F: resolution (H4)
ax = fig.add_subplot(gs[1, 2])
W2, H2 = 84, 66
draw_cones(ax, mosaic(W2, H2), alpha=0.35)
ax.add_patch(Ellipse((42, 33), 22, 46, fc=RF_FC, ec=RF_EC, lw=2.5, zorder=5))
arrow(ax, (31, 62), (53, 62), style="<|-|>", lw=2, ms=12)
ax.text(42, 66, "narrow along the row -> finer detail\nresolved for patterns varying along rows", ha="center",
        va="top", fontsize=8)
arrow(ax, (64, 10), (64, 56), style="<|-|>", lw=2, ms=12, color="0.4")
ax.text(66, 33, "wide\nacross", fontsize=8, va="center", color="0.3")
finish(ax, W2, H2 + 6, "F  Idea: direction-dependent resolution\nRFs ~5 % wider across rows -> ~5 % finer resolution "
                       "along rows")

# ---------------- G: anti-aliasing (H1)
ax = fig.add_subplot(gs[1, 3])
pts = [(x, y) for x, y, t, s in mosaic(W2, H2) if t == "B"]
for x, y in pts:
    ax.add_patch(Circle((x, y), 2.3, color=COL["B"]))
arrow(ax, (6, 70), (30, 70), style="<|-|>", lw=1.8, ms=10)
ax.text(18, 74, "sampled finer\nalong rows", ha="center", va="top", fontsize=8)
arrow(ax, (78, 8), (78, 32), style="<|-|>", lw=1.8, ms=10, color="0.4")
ax.text(75, 20, "coarser\nacross", ha="right", va="center", fontsize=8, color="0.3")
ax.add_patch(Ellipse((44, 34), 30, 52, fc=RF_FC, ec=RF_EC, lw=2, ls="--", zorder=5))
finish(ax, W2, H2 + 10, "G  Idea: matched anti-aliasing - unlikely at RGC scale\nlattice samples ~1.2x coarser "
                        "across rows; RFs wider across\n(right direction) but RFs are ~5x larger than the lattice")

# ---------------- H: colour sampling (H5)
ax = fig.add_subplot(gs[2, 0])
cones = mosaic(W2, H2)
types = [t for _, _, t, _ in cones]
shuffled = rng.permutation(types)
off = W2 + 10
draw_cones(ax, cones, alpha=0.85)
draw_cones(ax, [(x + off, y, t2, t2 in ("UV", "B")) for (x, y, _, _), t2 in zip(cones, shuffled)], alpha=0.85)
for x0 in (42, 42 + off):
    ax.add_patch(Circle((x0, 33), 22, fill=False, ec=RF_EC, lw=2.5, zorder=5))
ax.text(42, H2 + 8, "regular: every RF gets\nthe same cone mix", ha="center", va="top", fontsize=8.5)
ax.text(42 + off, H2 + 8, "random: mix varies\nfrom RF to RF", ha="center", va="top", fontsize=8.5)
finish(ax, 2 * W2 + 10, H2 + 14, "H  Idea: even colour sampling (claim 2) - small at RGC scale\nregular mosaic removes only "
                                 "1-3 % variation in the cone mix of an RF")

# ---------------- I: developmental by-product
ax = fig.add_subplot(gs[2, 1])
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
for r, a in [(0.3, 0.9), (0.5, 0.6), (0.7, 0.4), (0.9, 0.25)]:
    ax.add_patch(Circle((0, 0), r, fill=False, ec="0.5", lw=6, alpha=a))
for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
    ax.plot([0.3 * np.cos(th), 0.95 * np.cos(th)], [0.3 * np.sin(th), 0.95 * np.sin(th)], color=ROWC, lw=1)
arrow(ax, (0.7, -0.75), (0.95, -0.95), lw=2)
ax.text(0.55, -0.98, "new tissue added\nat the margin", fontsize=8, ha="right")
ax.add_patch(Circle((0, 0), 0.28, fc="#ece4d4", ec="0.55", ls="--"))
ax.text(0, 0, "oldest\n(larval)", ha="center", va="center", fontsize=8)
ax.set_title("I  Possible origin: growth in rings at the margin\nrows radiate, RGCs may elongate around the eye -> the\n"
             "alignment could be a developmental by-product", loc="left", fontsize=10)

# ---------------- notes
ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
notes = (
    "Tentative conclusions (RF orientation vs cone mosaic, 23/11/2026)\n\n"
    "- Over crystalline mosaic, RGC RFs avoid lying along the cone rows and tend to lie across them. Over disrupted\n"
    "  mosaic and in the larval patch, RFs show no bias relative to the ONH-radial direction (where crystalline rows\n"
    "  run): the radial geometry is still there, so the bias seems to need crystalline rows (not circular: same reference).\n"
    "- Robust so far: not explained by RF size, quality, elongation or distance from the ONH; present in every cell type\n"
    "  tested - a property of the mosaic under the cell, not of cell types.\n"
    "- With measured rows (15/05), RFs lie squarely across the real rows; rows twist ~10-15° from the ONH-radial\n"
    "  assumption, which explains the 60-75° peak against radial rows (the 'lattice diagonal' idea is probably a\n"
    "  coincidence).\n"
    "- Graded with mosaic order (15/05): crystalline cells align, mildly irregular cells (rows still traceable) align\n"
    "  more weakly (best with rows predicted from the surroundings), strongly disrupted cells do not. Timing of the\n"
    "  disruption (during vs after recording) is unresolved.\n"
    "- The original summary figure's main test was flawed (averaging fanning rows); per-cell tests give the same\n"
    "  conclusion with more defensible statistics.\n"
    "- RFs do not snap to the lattice in two orientations ('along or across'); cells not across their row may still\n"
    "  use the mosaic in other ways.\n\n"
    "Caveats: crystalline/disrupted split chosen after seeing the data (crude median split); rows measured in one\n"
    "recording; larval patch 19 cells, lower RF quality, and near the ONH the radial direction changes fast across an\n"
    "RF (noisier angles); in disrupted patches the reference itself is less certain; many analyses - replicate.\n"
    "Computational ideas (D-H) are exploratory.")
ax.text(0, 1, notes, va="top", fontsize=9.8, linespacing=1.35, family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#eef5fd", edgecolor="#1e6fd9"))

handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in
           [("UV", COL["UV"]), ("blue", COL["B"]), ("red", COL["R"]), ("green", COL["G"])]]
handles += [plt.Line2D([], [], color=ROWC, lw=2, label="cone row"),
            Ellipse((0, 0), 1, 1, fc=RF_FC, ec=RF_EC, label="RGC receptive field")]
fig.legend(handles=handles, loc="upper right", ncol=6, fontsize=9, frameon=False)
fig.suptitle("RGC receptive fields and the cone row mosaic: what we see (A-C) and tentative computational ideas "
             "(D-I) - schematic, not to scale", fontsize=14, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"schematic_rf_row_alignment_ideas.{fmt}", dpi=150 if fmt == "png" else None,
                bbox_inches="tight")
print("saved", FIG / "schematic_rf_row_alignment_ideas.png")
