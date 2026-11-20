"""
Do neighbouring RGCs share RF spots (possible shared subunits / cone input)? (20/11/2026)

One cell and its nearest neighbours (quality >= SUBUNIT_QUALITY_MIN), zoomed onto the cone mosaic:
  top row:    the cell alone, then + 1, + 2, + 3 nearest neighbours. Each RF = its actual covariance
              map (cm_most_important, scaled to its own peak, NOT smoothed), drawn in its own colour on
              the grey cone image; colour strength = RF strength (positive part; below
              OVERLAY_FLOOR of the peak transparent so the noise does not cover the cones).
              (Until 20/11/2026 the top row showed smoothed 50/75 % outlines + detected peaks.)
  bottom row: each of those RFs alone (covariance map on the cones), same zoom.
Second figure (..._overlap_pixels_...), same cells and zoom, added 20/11/2026:
  row 1: all RFs in greyscale (each scaled to its own peak; brightest RF per pixel), pixels inside
         >= 2 RFs coloured by how many (an RF = pixels >= --overlap-threshold of its own peak)
  row 2: the same overlap pixels on the grey cone image
Neighbours = nearest RF centres (`positions`) in the projected frame. Same zoom for all panels.
View from the door; stimulus -> projection only via A_raw_rf_to_projected_um.

Default: 14/08/2026 6px, centre cell = highest-quality cell in the middle-right box
(--box xmin xmax ymin ymax in projected µm, + right / + down). Or give --cell.

Run from the repo root:
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/plot_neighbour_rf_overlap.py
"""

import argparse
import contextlib
import io
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from PIL import Image

from loading.noise_data import load_noise_data  # v2 RF dataset, checked
from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import (
    alignment_cache_v2_file as alignment_cache_file,
    cone_image_file,
    noise_data_v2_file as noise_data_file,
)
from matplotlib.colors import to_rgb

import sys as _sys
from pathlib import Path as _Path
# This folder's name starts with a digit, so it can't be imported as a package:
# load subunit_tools.py from the script's own folder.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from subunit_tools import SUBUNIT_QUALITY_MIN, subunit_score

Image.MAX_IMAGE_PIXELS = None
OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
COLOURS = ["tab:green", "yellow", "tab:orange", "tab:cyan"]  # centre cell, then neighbours 1-3
OVERLAY_FLOOR = 0.2  # fraction of each RF's peak below which the overlay is transparent
OVERLAY_MAX_ALPHA = 0.85

parser = argparse.ArgumentParser()
parser.add_argument("--recording", default="14_08_2026")
parser.add_argument("--channel", default="6px_15Hz_30mins_shuffle_x6")
parser.add_argument("--cell", type=int, default=None)
parser.add_argument("--box", type=float, nargs=4, default=[200, 750, -350, 150],
                    help="projected µm: xmin xmax ymin ymax, used when --cell is not given")
parser.add_argument("--n-neighbours", type=int, default=3)
parser.add_argument("--rf-alpha", type=float, default=0.6)
parser.add_argument("--overlap-threshold", type=float, default=0.3,
                    help="a pixel belongs to an RF if it is >= this fraction of that RF's own peak (overlap figure)")
args = parser.parse_args()

# ------------------------------------------------------------ geometry
cache_file = alignment_cache_file(args.recording)
cache_data = np.load(cache_file)
with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(cache_file)
# ORIENTATION [stimulus -> projection, date-dependent]: before 8 Oct 2026 (x, y) -> (-y, -x);
# from 8 Oct 2026 (x, y) -> (y, -x). Applied only through this matrix, never to the data.
A = geo.A_raw_rf_to_projected_um
half_w = int(cache_data["displayed_stim_width_px"]) * geo.corrected_stim_um_per_px / 2
half_h = int(cache_data["displayed_stim_height_px"]) * geo.corrected_stim_um_per_px / 2
cones = iio.imread(cone_image_file(args.recording))
if cones.ndim == 3 and cones.shape[2] == 4:
    cones = cones[..., :3]
cones_grey = cones.mean(axis=2) if cones.ndim == 3 else cones

# ------------------------------------------------------------ cells
data_file = noise_data_file(args.recording)
with load_noise_data(data_file, lazy=True) as ds:
    g.require_v2_rf_frame(ds)
    q = ds.quality.sel(channel=args.channel).load()
    good = [int(c) for c in q.cell_index.values if np.isfinite(q.sel(cell_index=c)) and q.sel(cell_index=c) >= SUBUNIT_QUALITY_MIN]
    pos = ds.positions.sel(channel=args.channel, cell_index=good).load()
    centre_r = ds.center_outline_um.sel(channel=args.channel, cell_index=good).max("degree").load()
    x = ds.x.values * geo.rf_scale_factor  # measured / nominal stimulus pixel size
    y = ds.y.values * geo.rf_scale_factor

    def proj(c):
        px, py = (int(v) for v in pos.sel(cell_index=c).values)  # pos_dim = [x, y], canvas indices
        return A[:2, :2] @ np.array([x[px], y[py]]) + A[:2, 2]

    centres = {c: proj(c) for c in good}
    if args.cell is None:
        x0, x1, y0, y1 = args.box
        in_box = [c for c in good if x0 <= centres[c][0] <= x1 and y0 <= centres[c][1] <= y1]
        focal = max(in_box, key=lambda c: float(q.sel(cell_index=c)))
    else:
        focal = args.cell
        if focal not in centres:
            raise SystemExit(f"cell {focal} has quality < {SUBUNIT_QUALITY_MIN} in {args.channel}")
    others = sorted((c for c in good if c != focal), key=lambda c: np.hypot(*(centres[c] - centres[focal])))
    cells = [focal] + others[:args.n_neighbours]
    cm = ds.cm_most_important.sel(channel=args.channel, cell_index=cells).transpose("cell_index", "y", "x").load()

step = abs(x[1] - x[0])
extent = [x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2]
X, Y = np.meshgrid(x, y)
proj_x = A[0, 0] * X + A[0, 1] * Y + A[0, 2]
proj_y = A[1, 0] * X + A[1, 1] * Y + A[1, 2]
inside = (np.abs(proj_x) <= half_w) & (np.abs(proj_y) <= half_h)


def radius(c):
    r = float(centre_r.sel(cell_index=c))
    return r if np.isfinite(r) and r > 0 else 50.0


maps, peaks = {}, {}
for c in cells:
    raw = np.asarray(cm.sel(cell_index=c).values, float)
    m = np.where(inside & np.isfinite(raw), raw / np.nanmax(np.abs(raw)), np.nan)
    maps[c] = m
    peaks[c] = subunit_score(m, step, radius(c))[2]
for c in cells:
    d = np.hypot(*(centres[c] - centres[focal]))
    print(f"cell {c}: QI {float(q.sel(cell_index=c)):.0f}, {d:.0f} µm from cell {focal}, "
          f"centre radius {radius(c):.0f} µm, {len(peaks[c]) - 1} extra peaks")

# one zoom for every panel: all centres + their largest centre radius
P = np.array([centres[c] for c in cells])
pad = max(radius(c) for c in cells) * 1.1
lo, hi = P.min(axis=0) - pad, P.max(axis=0) + pad
mid, half = (lo + hi) / 2, max(hi - lo) / 2


# ------------------------------------------------------------ plot
def draw_cones(ax, grey=False):
    ax.set_facecolor("black")
    # ORIENTATION [cone image -> projection]: from the electrode clicks (door-view image).
    ax.imshow(cones_grey if grey else cones, cmap="gray" if grey else None,
              extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData, zorder=2)


def frame(ax):
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] + half, mid[1] - half)  # door view: +y down
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])


def draw_rf_colour(ax, c, colour):
    """Actual RF (positive part) in one colour; opacity grows with RF strength."""
    v = np.nan_to_num(maps[c])
    rgba = np.zeros(v.shape + (4,))
    rgba[..., :3] = to_rgb(colour)
    rgba[..., 3] = np.clip((v - OVERLAY_FLOOR) / (1 - OVERLAY_FLOOR), 0, 1) * OVERLAY_MAX_ALPHA
    # same placement as the bottom-row imshow; ORIENTATION only via A.
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest",
              transform=Affine2D(A) + ax.transData, zorder=10)


n = len(cells)
fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.9), facecolor="white")
for k in range(n):
    ax = axes[0, k]
    draw_cones(ax, grey=True)
    for c, colour in zip(cells[:k + 1], COLOURS):
        draw_rf_colour(ax, c, colour)
    frame(ax)
    ax.set_title(f"cell {focal} alone" if k == 0 else f"+ {k} neighbour{'s' if k > 1 else ''}", fontsize=10, loc="left")

    c = cells[k]
    ax = axes[1, k]
    draw_cones(ax)
    ax.imshow(maps[c], extent=extent, origin="upper", cmap="coolwarm", vmin=-1, vmax=1, alpha=args.rf_alpha,
              interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=10)
    frame(ax)
    for side in ax.spines.values():
        side.set_edgecolor(COLOURS[k]); side.set_linewidth(3)
    d = np.hypot(*(centres[c] - centres[focal]))
    ax.set_title(f"cell {c}  ·  QI {float(q.sel(cell_index=c)):.0f}" + ("" if k == 0 else f"  ·  {d:.0f} µm away"),
                 fontsize=9, loc="left")

ax = axes[1, 0]
x0, x1 = ax.get_xlim(); y0, _ = ax.get_ylim(); w = x1 - x0
ax.plot([x0 + 0.06 * w, x0 + 0.06 * w + 50], [y0 - 0.06 * w] * 2, "w-", lw=3, zorder=30)
ax.text(x0 + 0.06 * w + 25, y0 - 0.1 * w, "50 µm", color="white", ha="center", fontsize=8, zorder=30)

handles = [Line2D([], [], color=col, lw=6, label=f"cell {c}") for c, col in zip(cells, COLOURS)]
fig.legend(handles=handles, loc="lower center", ncol=len(cells), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
           title=f"Top row: each cell's own RF map in its colour; stronger colour = stronger RF; "
                 f"below {OVERLAY_FLOOR:.0%} of its peak not shown. Bottom row: each RF alone.", title_fontsize=9)
fig.suptitle(f"{args.recording.replace('_', '/')}  ·  {args.channel}  ·  neighbouring RFs on the cones "
             f"(covariance, each RF scaled to its own peak)  ·  view from the door", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.05, hspace=0.12)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
stem = f"neighbour_rf_overlap_cell_{focal}_{args.recording}_{args.channel}"
for fmt in ("png", "pdf"):
    out = OUTPUT_FOLDER / f"{stem}.{fmt}"
    fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
    print("saved", out)


# ============================================================
# Second figure: RFs in greyscale, overlapping pixels in colour
# ============================================================
from matplotlib.colors import ListedColormap  # noqa: E402

OVERLAP_COLOURS = {2: "yellow", 3: "tab:orange", 4: "red"}  # number of RFs sharing a pixel
thr = args.overlap_threshold
pos_maps = {c: np.clip(np.nan_to_num(maps[c]), 0, None) for c in cells}

fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.9), facecolor="white")
for k in range(n):
    incl = cells[:k + 1]
    grey = np.max([pos_maps[c] for c in incl], axis=0)
    count = np.sum([pos_maps[c] >= thr for c in incl], axis=0)
    count_img = np.where(count >= 2, np.minimum(count, 4), np.nan).astype(float)
    cmap_count = ListedColormap([OVERLAP_COLOURS[i] for i in (2, 3, 4)])

    ax = axes[0, k]
    ax.set_facecolor("black")
    # RF maps sit on the stimulus grid; ORIENTATION only via A (same placement as the other figures).
    ax.imshow(np.where(inside, grey, np.nan), extent=extent, origin="upper", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=5)
    ax.imshow(count_img, extent=extent, origin="upper", cmap=cmap_count, vmin=1.5, vmax=4.5,
              interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=10)
    frame(ax)
    n_px = int(np.sum(count >= 2))
    ax.set_title((f"cell {focal} alone" if k == 0 else f"+ {k} neighbour{'s' if k > 1 else ''}")
                 + (f"  ·  overlap {n_px * step ** 2:.0f} µm²" if k else ""), fontsize=10, loc="left")

    ax = axes[1, k]
    draw_cones(ax, grey=True)
    ax.imshow(count_img, extent=extent, origin="upper", cmap=cmap_count, vmin=1.5, vmax=4.5, alpha=0.75,
              interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=10)
    frame(ax)
    ax.set_title("overlap on the cones", fontsize=9, loc="left")

ax = axes[1, 0]
x0, x1 = ax.get_xlim(); y0, _ = ax.get_ylim(); w = x1 - x0
ax.plot([x0 + 0.06 * w, x0 + 0.06 * w + 50], [y0 - 0.06 * w] * 2, "w-", lw=3, zorder=30)
ax.text(x0 + 0.06 * w + 25, y0 - 0.1 * w, "50 µm", color="white", ha="center", fontsize=8, zorder=30)
handles = [Line2D([], [], color=col, lw=6, label=f"{i} RFs overlap") for i, col in OVERLAP_COLOURS.items()][:max(1, n - 1)]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
           title=f"Cells in order: {', '.join(str(c) for c in cells)}.  Grey = RF strength (each cell scaled to its own peak).  "
                 f"A pixel is part of an RF if it is ≥ {thr:.0%} of that RF's peak.", title_fontsize=9)
fig.suptitle(f"{args.recording.replace('_', '/')}  ·  {args.channel}  ·  where neighbouring RFs overlap  ·  view from the door",
             fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.05, hspace=0.12)
stem = f"neighbour_rf_overlap_pixels_cell_{focal}_{args.recording}_{args.channel}"
for fmt in ("png", "pdf"):
    out = OUTPUT_FOLDER / f"{stem}.{fmt}"
    fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
    print("saved", out)
