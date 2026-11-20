"""
Design sketch: random-bar stimuli for fine RF / subunit mapping (20/11/2026). Nothing is read from data.

Schematic cone mosaic (Laura's description): cones in rows, blue cones 12 µm apart along a row with a red/green
double cone between each pair; rows 10 µm apart (row spacing is an assumption for the sketch).
Bars 4 µm wide, 610 nm (bright = red light, dark = off), 80 µm patch, 15 Hz.

Rows of the figure:
  A  one bar per frame at a random position (Laura's idea)          -> 1D sensitivity profile, no interactions
  B  a few bars per frame (here 4) at random positions              -> profile + some pairs together
  C  every bar random bright/dark each frame, bars ACROSS the rows  -> 1D noise: sNMF/covariance possible;
                                                                       separates cones along a row
  D  same, bars ALONG the rows                                      -> separates the rows
Last column: the stimulus over time (position x frames) - the whole stimulus is just this small matrix.

Run:  MPLBACKEND=Agg poetry run python laura_scripts/SNMF/plot_bar_stimulus_design.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/SNMF/stimulus_design_random_bars.png")
PATCH, BAR = 80.0, 4.0          # µm
N_BARS = int(PATCH / BAR)
BLUE_SPACING, ROW_SPACING = 12.0, 10.0
LIGHT, DARK = "#e8590c", "#111111"
cmap = ListedColormap([DARK, LIGHT])
rng = np.random.default_rng(3)
N_FRAMES_SHOW, N_TIME = 4, 60


def mosaic():
    """Schematic cone positions: (x, y, colour)."""
    cones = []
    for r, y in enumerate(np.arange(ROW_SPACING / 2, PATCH, ROW_SPACING)):
        shift = (r % 2) * BLUE_SPACING / 2
        for x in np.arange(shift + 2, PATCH, BLUE_SPACING):
            cones.append((x, y, "tab:blue"))
            if x + BLUE_SPACING / 2 < PATCH:
                cones += [(x + BLUE_SPACING / 2 - 1.2, y, "tab:red"), (x + BLUE_SPACING / 2 + 1.2, y, "tab:green")]
    return cones


CONES = mosaic()


def frame_values(design):
    if design == "single":
        v = np.zeros(N_BARS, int); v[rng.integers(N_BARS)] = 1
    elif design == "few":
        v = np.zeros(N_BARS, int); v[rng.choice(N_BARS, 4, replace=False)] = 1
    else:
        v = rng.integers(0, 2, N_BARS)
    return v


def draw_frame(ax, v, along_rows, background):
    ax.add_patch(Rectangle((0, 0), PATCH, PATCH, color=background, zorder=0))
    for i, on in enumerate(v):
        if background == DARK and not on:
            continue
        colour = LIGHT if on else DARK
        if along_rows:   # bars horizontal = parallel to the rows
            ax.add_patch(Rectangle((0, i * BAR), PATCH, BAR, color=colour, zorder=1))
        else:            # bars vertical = across the rows
            ax.add_patch(Rectangle((i * BAR, 0), BAR, PATCH, color=colour, zorder=1))
    for x, y, c in CONES:
        ax.plot(x, y, "o", ms=3.2, color=c, mec="white", mew=0.4, zorder=3)
    ax.add_patch(Circle((PATCH / 2, PATCH / 2), 30, fill=False, ec="white", ls="--", lw=1, zorder=4))
    ax.set_xlim(0, PATCH); ax.set_ylim(PATCH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


designs = [
    ("single", False, DARK, "A  One bar per frame, random position (your idea)",
     "clean 1D sensitivity profile, strong responses\nbut never two places at once -> cannot show subunits"),
    ("few", False, DARK, "B  A few bars per frame (here 4), random positions",
     "strong responses + some pairs together\n-> subunits possible, needs more frames than C"),
    ("dense", False, None, "C  Every bar random bright/dark each frame - bars ACROSS the rows",
     "1D noise: sNMF / covariance work (bars independent)\nseparates cones ALONG a row"),
    ("dense", True, None, "D  Same as C - bars ALONG the rows",
     "separates the ROWS\nC + D together give 2D positions on the mosaic"),
]

fig = plt.figure(figsize=(16, 15.5))
gs = fig.add_gridspec(len(designs), N_FRAMES_SHOW + 2, width_ratios=[1] * N_FRAMES_SHOW + [0.15, 1.5],
                      hspace=0.38, wspace=0.08)
for r, (design, along, bg, title, note) in enumerate(designs):
    for f in range(N_FRAMES_SHOW):
        ax = fig.add_subplot(gs[r, f])
        draw_frame(ax, frame_values(design), along, bg or DARK)
        ax.set_title(f"frame {f + 1}", fontsize=8, loc="left")
        if f == 0:
            ax.text(0, 1.16, title, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
            if r == 0:
                ax.plot([4, 24], [PATCH - 4, PATCH - 4], "w-", lw=3, zorder=5)
                ax.text(14, PATCH - 7, "20 µm", color="white", ha="center", fontsize=7, zorder=5)
    ax = fig.add_subplot(gs[r, N_FRAMES_SHOW + 1])
    m = np.array([frame_values(design) for _ in range(N_TIME)]).T
    ax.imshow(m, cmap=cmap, aspect="auto", interpolation="nearest", extent=[0, N_TIME / 15, PATCH, 0])
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel(("bar position along the row (µm)" if not along else "bar position across rows (µm)"), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("the stimulus over time (all of it = this matrix)", fontsize=8, loc="left")
    ax.text(1.02, 0.5, note, transform=ax.transAxes, fontsize=8.5, va="center")

handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="white", label=l, ms=7)
           for c, l in [("tab:blue", "blue cone (labelled)"), ("tab:red", "red cone"), ("tab:green", "green cone")]]
handles += [Rectangle((0, 0), 1, 1, color=LIGHT, label="bar on (610 nm)"), Rectangle((0, 0), 1, 1, color=DARK, label="bar off"),
            plt.Line2D([], [], ls="--", color="grey", label="example RF centre (60 µm)")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.45, 0.005))
fig.suptitle("Random-bar stimuli on a schematic cone mosaic (4 µm bars, 610 nm, 80 µm patch shown, 15 Hz)",
             fontsize=13, x=0.01, ha="left", y=0.93)
fig.text(0.01, 0.025,
         "Schematic mosaic from your description (blue cones 12 µm apart, red/green double cone between; row spacing "
         "10 µm assumed). For red-only (610 nm) only the red cones respond.\n"
         "Size: 3 h at 15 Hz = 162,000 frames x 20 bars per 80 µm (about 500 bars across the whole screen) "
         "= tens of MB - no memory problem.", fontsize=9)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
