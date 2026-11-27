"""
Single cells on the cone mosaic (v2 alignment), each with its STAs next to it.

Makes three figures for one recording / noise channel:
  1. top-N quality cells, covariance RF (cm_most_important) on the cone mosaic
  2. the same cells, RMS RF on the cone mosaic
  3. the N_SUBUNIT "most subunit-like" cells (see SUBUNIT SCORE below), covariance RF
     on the cone mosaic + the cone image alone with the RF drawn as contour lines

Next to every cell: the STA time course
  * single pixel   = sta_single_pixel from noise_data_v2.nc
  * top 3 %        = weighted STA from laura_scripts/RF_analysis/weighted_STA.py,
                     weight_method="percentile", percentile 97 (top 3 % of positive covariance pixels)
  Each trace is drawn as (STA - 0.5) scaled to its own largest deviation, so the shapes can be
  compared (the weighted STA is an average over many pixels, so its raw amplitude is smaller).

RF display = plot_top_quality_cells.py: each RF normalised to its own peak, masked to the
displayed stimulus, all panels in the projected frame, view from the door, same zoom everywhere.

SUBUNIT SCORE (cells with quality >= SUBUNIT_QUALITY_MIN only):
  covariance map lightly smoothed (Gaussian, SUBUNIT_SMOOTH_UM); peaks inside the RF centre
  (within the cell's largest centre-outline radius of the main peak) that are
    - at least SUBUNIT_MIN_PEAK x the main peak,
    - at least SUBUNIT_MIN_SEPARATION_UM away from any higher peak, and
    - separated from the main peak by a dip (lowest value on the straight line between
      them <= SUBUNIT_MAX_DIP x the smaller peak), and
    - that dip is deeper than SUBUNIT_MIN_DIP_NOISE_SD x the noise SD of the smoothed map
      (measured > SUBUNIT_NOISE_DISTANCE_UM from the main peak). Added 20/11/2026: without it
      the noisiest maps ranked highest (bumps of ~1 noise SD on a broad plateau).
  are counted. Cells are ranked by number of such extra peaks, then by how deep the dips are.

Run from the repo root:
    poetry run python laura_scripts/6px_subunit_exploration/plot_cells_with_stas.py \
        --recording 14_08_2026 --channel 6px_15Hz_30mins_shuffle_x6 --stimulus-id 10

Heavy (opens the 8 GB noise_data_v2.nc lazily + reloads ~40 raw STAs): run on its own.
"""

import argparse
import ast
import contextlib
import functools
import io
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from PIL import Image
from scipy import ndimage

from loading.noise_data import load_noise_data  # v2 RF dataset, checked
from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import (
    alignment_cache_v2_file as alignment_cache_file,
    cone_image_file,
    noise_data_v2_file as noise_data_file,
)

Image.MAX_IMAGE_PIXELS = None

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/6px_subunit_exploration")
REPO = Path(__file__).resolve().parents[2]

STYLE = {
    "cm_most_important": dict(cmap="coolwarm", signed=True, name="covariance"),
    "rms": dict(cmap="Reds", signed=False, name="RMS"),
}
SINGLE_COLOUR = "tab:blue"
WEIGHTED_COLOUR = "tab:green"
STA_TIME_WINDOW_MS = (-600, 100)

import sys as _sys
from pathlib import Path as _Path
# This folder's name starts with a digit, so it can't be imported as a package:
# load subunit_tools.py from the script's own folder.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from subunit_tools import (  # noqa: E402  shared with compare_6px_8px_subunit_cells.py
    SUBUNIT_MAX_DIP, SUBUNIT_MIN_DIP_NOISE_SD, SUBUNIT_MIN_PEAK, SUBUNIT_MIN_SEPARATION_UM,
    SUBUNIT_NOISE_DISTANCE_UM, SUBUNIT_QUALITY_MIN, SUBUNIT_SMOOTH_UM, smooth_normalised, subunit_score,
)

parser = argparse.ArgumentParser()
parser.add_argument("--recording", required=True, help="DD_MM_YYYY")
parser.add_argument("--channel", required=True)
parser.add_argument("--stimulus-id", type=int, required=True, help="stimulus index of the channel (folder _idx_N)")
parser.add_argument("--top", type=int, default=30)
parser.add_argument("--n-subunit", type=int, default=10)
parser.add_argument("--percentile", type=float, default=97, help="weighted STA: keep pixels above this percentile")
parser.add_argument("--zoom-half-width-um", type=float, default=None,
                    help="half-width of every panel (default: 1.5 x median largest centre radius, 60-150 µm)")
parser.add_argument("--n-cols", type=int, default=5)
parser.add_argument("--rf-alpha", type=float, default=0.6)
parser.add_argument("--cache", type=Path, default=None, help="npz to cache weighted STAs (optional)")
args = parser.parse_args()


# ============================================================
# Weighted STA: use calculate_weighted_sta_for_cells from weighted_STA.py unchanged.
# That file runs an example at import time, so only its imports + this function are executed.
# ============================================================
def load_weighted_sta_function():
    src_file = REPO / "laura_scripts" / "RF_analysis" / "weighted_STA.py"
    tree = ast.parse(src_file.read_text())
    func = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calculate_weighted_sta_for_cells"]
    first = func[0].lineno
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno < first]
    ns = {}
    exec(compile(ast.Module(body=imports + func, type_ignores=[]), str(src_file), "exec"), ns)
    ns["load_noise_data"] = functools.partial(ns["load_noise_data"], lazy=True)  # 8 GB file
    return ns["calculate_weighted_sta_for_cells"]


def weighted_stas(cell_ids, data_file):
    cache = {}
    if args.cache and args.cache.exists():
        z = np.load(args.cache)
        cache = {int(c): v for c, v in zip(z["cells"], z["stas"])}
        time = z["time"]
    todo = [c for c in cell_ids if c not in cache]
    if todo:
        from location.x_array import x_y_and_scale
        from organize.configs import Collapse_2d_Config, Recording_Config

        rec = Recording_Config(root_path=data_file.parents[1])
        rec.add_channel(stimulus_id=args.stimulus_id, name=args.channel, colour="blue")
        # cut size as in the noise_analysis folder's collapse_2d_config.json
        import json
        cut = json.loads((data_file.parent / "collapse_2d_config.json").read_text())["cut_size_um"]["data"]
        col = Collapse_2d_Config(recording_config=rec, cut_size_um=x_y_and_scale(*cut))
        func = load_weighted_sta_function()
        da = func(data_file, rec, col, todo, [args.channel], weight_method="percentile",
                  percentile_by_channel={args.channel: args.percentile}, verbose=False)
        time = da.time_max.values
        for c in todo:
            cache[c] = da.sel(cell_index=c, channel=args.channel).values
        if args.cache:
            np.savez(args.cache, cells=np.array(list(cache)), stas=np.array(list(cache.values())), time=time)
    return {c: cache[c] for c in cell_ids}, time


# ============================================================
# Subunit score
# ============================================================
# subunit_score: see subunit_tools.py


# ============================================================
# Geometry + cone image
# ============================================================
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

# ============================================================
# Load RFs, STAs, choose cells
# ============================================================
data_file = noise_data_file(args.recording)
with load_noise_data(data_file, lazy=True) as ds:
    g.require_v2_rf_frame(ds)
    quality = ds.quality.sel(channel=args.channel).load()
    quality = quality.where(np.isfinite(quality), drop=True).sortby(quality, ascending=False)
    top_ids = [int(c) for c in quality.cell_index.values[:args.top]]
    sub_pool = [int(c) for c in quality.cell_index.values if float(quality.sel(cell_index=c)) >= SUBUNIT_QUALITY_MIN]
    all_ids = sorted(set(top_ids) | set(sub_pool))
    cm = ds.cm_most_important.sel(channel=args.channel).sel(cell_index=all_ids).transpose("cell_index", "y", "x").load()
    rms = ds.rms.sel(channel=args.channel).sel(cell_index=top_ids).transpose("cell_index", "y", "x").load()
    sta_single = ds.sta_single_pixel.sel(channel=args.channel).sel(cell_index=all_ids).load()
    # RF centre from the pipeline (indices into the x/y canvas, verified 18/11/2026); every panel of
    # every figure is centred here, so covariance and RMS panels of one cell show the same cones.
    # (Before 20/11/2026 each panel was centred on its own map's peak: up to 108 µm apart.)
    pos_idx = ds.positions.sel(channel=args.channel).sel(cell_index=all_ids).load()
    centre_r = ds.center_outline_um.sel(channel=args.channel).sel(cell_index=all_ids).max("degree").load()
    x = ds.x.values * geo.rf_scale_factor  # measured / nominal stimulus pixel size
    y = ds.y.values * geo.rf_scale_factor
    time_ms = ds.time_max.values
    # stimulus pixel in µm = (N in "Npx_...") x grid step (grid = 1 stimulus-file pixel)
    stim_px_um = int(args.channel.split("px")[0]) * float(abs(ds.x.values[1] - ds.x.values[0]))

step = x[1] - x[0]
extent = [x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2]
X, Y = np.meshgrid(x, y)
proj_x = A[0, 0] * X + A[0, 1] * Y + A[0, 2]
proj_y = A[1, 0] * X + A[1, 1] * Y + A[1, 2]
inside = (np.abs(proj_x) <= half_w) & (np.abs(proj_y) <= half_h)  # displayed-stimulus mask


def normalised(rf, signed):
    rf = np.asarray(rf, dtype=float)
    peak = np.nanmax(np.abs(rf) if signed else rf)
    if np.isfinite(peak) and peak > 0:
        rf = rf / peak
    return np.where(inside, np.nan_to_num(rf, nan=0.0), np.nan)


# Subunit ranking
scores = []
for c in sub_pool:
    raw = np.asarray(cm.sel(cell_index=c).values, float)
    # keep NaN outside the cell's stored 300x300 window (normalised() fills it with 0, which
    # would make the noise SD look ~0), and outside the displayed stimulus
    img = np.where(inside & np.isfinite(raw), raw / np.nanmax(np.abs(raw)), np.nan)
    r = float(centre_r.sel(cell_index=c))
    r = r if np.isfinite(r) and r > 0 else 50.0
    # signed map (not clipped) so the noise SD far from the cell is measured on the real values
    n_extra, dip_sum, peaks = subunit_score(img, abs(step), r)
    scores.append((n_extra, dip_sum, c, peaks))
    print(f"  cell {c}: {n_extra} extra peaks")
scores.sort(key=lambda t: (t[0], t[1]), reverse=True)
subunit = scores[:args.n_subunit]
print("subunit ranking (extra peaks, dip sum, cell):", [(s[0], round(s[1], 2), s[2]) for s in scores[:15]])

zoom = args.zoom_half_width_um
if zoom is None:
    r = centre_r.sel(cell_index=top_ids).values
    zoom = float(np.clip(1.5 * np.nanmedian(r), 60, 150))
print(f"panel half-width {zoom:.0f} µm")

w_sta, w_time = weighted_stas(sorted(set(top_ids) | {s[2] for s in subunit}), data_file)


# ============================================================
# Plot helpers
# ============================================================
def to_proj(ix, iy):
    return A[:2, :2] @ np.array([x[ix], y[iy]]) + A[:2, 2]


def cell_centre(c):
    px, py = pos_idx.sel(cell_index=c).values  # pos_dim = [x, y]
    return to_proj(int(px), int(py))


def draw_cones(ax, grey=False):
    ax.set_facecolor("black")
    # ORIENTATION [cone image -> projection]: from the electrode clicks (door-view image).
    ax.imshow(cones_grey if grey else cones, cmap="gray" if grey else None,
              extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData, zorder=2)


def frame(ax, centre):
    ax.set_xlim(centre[0] - zoom, centre[0] + zoom)
    ax.set_ylim(centre[1] + zoom, centre[1] - zoom)  # door view: +y down
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])


def draw_rf(ax, img, style):
    vmin, vmax = (-1.0, 1.0) if style["signed"] else (0.0, 1.0)
    ax.imshow(img, extent=extent, origin="upper", cmap=style["cmap"], vmin=vmin, vmax=vmax,
              alpha=args.rf_alpha, interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=10)


def draw_sta(ax, cell, first):
    for t, trace, colour, label in ((time_ms, sta_single.sel(cell_index=cell).values, SINGLE_COLOUR, "single pixel"),
                                    (w_time, w_sta[cell], WEIGHTED_COLOUR, f"top {100 - args.percentile:.0f} %")):
        d = np.asarray(trace, float) - 0.5
        sel = (t >= STA_TIME_WINDOW_MS[0]) & (t <= STA_TIME_WINDOW_MS[1])
        peak = np.nanmax(np.abs(d[sel])) if np.isfinite(d[sel]).any() else np.nan
        if np.isfinite(peak) and peak > 0:
            ax.plot(t[sel], d[sel] / peak, color=colour, lw=1.3, label=label)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlim(*STA_TIME_WINDOW_MS); ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=7)
    if first:
        ax.set_xlabel("time to spike (ms)", fontsize=7)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left", handlelength=1.2)
    else:
        ax.set_xticklabels([])


def scale_bar(ax):
    x0, x1 = ax.get_xlim(); y0, _ = ax.get_ylim()
    w = x1 - x0
    ax.plot([x0 + 0.08 * w, x0 + 0.08 * w + 50], [y0 - 0.08 * w] * 2, "w-", lw=3, zorder=30)
    ax.text(x0 + 0.08 * w + 25, y0 - 0.13 * w, "50 µm", color="white", ha="center", fontsize=7, zorder=30)


def save(fig, stem):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        out = OUTPUT_FOLDER / f"{stem}.{fmt}"
        fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


# ============================================================
# Figures 1 + 2: top-N, covariance and RMS
# ============================================================
rec_label = args.recording.replace("_", "/")
for variable, maps in (("cm_most_important", cm), ("rms", rms)):
    style = STYLE[variable]
    n = len(top_ids)
    n_rows = int(np.ceil(n / args.n_cols))
    fig = plt.figure(figsize=(4.6 * args.n_cols, 2.9 * n_rows), facecolor="white")
    gs = fig.add_gridspec(n_rows, 2 * args.n_cols, width_ratios=[1, 0.6] * args.n_cols, wspace=0.08, hspace=0.3,
                          top=0.95, bottom=0.03, left=0.02, right=0.98)
    for k, c in enumerate(top_ids):
        r, col = divmod(k, args.n_cols)
        ax = fig.add_subplot(gs[r, 2 * col]); ax_t = fig.add_subplot(gs[r, 2 * col + 1])
        img = normalised(maps.sel(cell_index=c).values, style["signed"])
        strength = np.abs(img) if style["signed"] else img
        draw_cones(ax); draw_rf(ax, img, style); frame(ax, cell_centre(c))
        ax.set_title(f"cell {c}  ·  QI {float(quality.sel(cell_index=c)):.0f}", fontsize=8, loc="left")
        draw_sta(ax_t, c, k == 0)
        if k == 0:
            scale_bar(ax)
    fig.suptitle(f"{rec_label}  ·  {args.channel}  ·  top {n} cells by quality  ·  {style['name']} RF on the cones "
                 f"(each cell scaled to its own peak)  ·  view from the door", fontsize=11, y=0.995)
    save(fig, f"top_{n}_cells_{style['name'].lower()}_with_STAs_{args.recording}_{args.channel}")

# ============================================================
# Figure 3: most subunit-like cells
# ============================================================
style = STYLE["cm_most_important"]
n = len(subunit)
n_cols = 2
n_rows = int(np.ceil(n / n_cols))
fig = plt.figure(figsize=(7.2 * n_cols, 3.0 * n_rows + 0.8), facecolor="white")
gs = fig.add_gridspec(n_rows, 3 * n_cols, width_ratios=[1, 1, 0.75] * n_cols, wspace=0.08, hspace=0.3,
                      top=0.94, bottom=0.07, left=0.02, right=0.98)
for k, (n_extra, dip_sum, c, peaks) in enumerate(subunit):
    r, col = divmod(k, n_cols)
    ax = fig.add_subplot(gs[r, 3 * col]); ax_c = fig.add_subplot(gs[r, 3 * col + 1])
    ax_t = fig.add_subplot(gs[r, 3 * col + 2])
    img = normalised(cm.sel(cell_index=c).values, True)
    centre = cell_centre(c)
    draw_cones(ax); draw_rf(ax, img, style); frame(ax, centre)
    ax.set_title(f"cell {c}  ·  QI {float(quality.sel(cell_index=c)):.0f}  ·  {n_extra + 1} peaks", fontsize=8, loc="left")
    # cones alone (grey) with the RF as contour lines + peaks
    draw_cones(ax_c, grey=True)
    smooth = smooth_normalised(img, abs(step))
    # smooth[i, j] sits at stimulus (x[j], y[i]), same as the imshow; ORIENTATION only via A.
    ax_c.contour(x, y, smooth, levels=[0.5, 0.75], colors=["tab:green", "yellow"], linewidths=0.9,
                 transform=Affine2D(A) + ax_c.transData, zorder=10)
    pk = np.array([to_proj(px, py) for py, px in peaks])
    ax_c.plot(pk[:, 0], pk[:, 1], "+", color="yellow", ms=7, mew=1.3, zorder=11)
    frame(ax_c, centre)
    draw_sta(ax_t, c, k == 0)
    if k == 0:
        scale_bar(ax)
fig.suptitle(f"{rec_label}  ·  {args.channel}  ·  {n} cells with the most separate peaks in the RF centre "
             f"(quality ≥ {SUBUNIT_QUALITY_MIN})", fontsize=11, y=0.995)
fig.text(0.01, 0.005,
         f"Left: covariance RF on the cones. Middle: cones (grey) with the RF outlined at 50 % (green) and 75 % (yellow) "
         f"of its peak; + = peaks counted.\nA peak counts if it is ≥ {SUBUNIT_MIN_PEAK:.0%} of the main peak, "
         f"≥ {SUBUNIT_MIN_SEPARATION_UM:.0f} µm from a higher peak, inside the RF centre, and the RF dips to "
         f"≤ {SUBUNIT_MAX_DIP:.0%} of it on the way to the main peak,\nwith a dip at least {SUBUNIT_MIN_DIP_NOISE_SD:.0f}× the map's noise "
         f"(noise = spread of the map > {SUBUNIT_NOISE_DISTANCE_UM:.0f} µm from the cell; map smoothed by {SUBUNIT_SMOOTH_UM:.0f} µm). "
         f"Noise pixel = {stim_px_um:.0f} µm.",
         fontsize=8, va="bottom")
save(fig, f"subunit_like_{n}_cells_with_STAs_{args.recording}_{args.channel}")
