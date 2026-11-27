"""
Top-N quality cells, one panel each, on the cone mosaic (v2 alignment).

Same as RF_alignment.py CELL 4, without having to run CELLs 1-3 (no clicking):
  * cells ranked by quality for the chosen channel
  * each RF normalised to its own peak (|value| for covariance), NaN -> 0
  * RF masked to the displayed stimulus
  * covariance colour limits -1..1 (per-cell normalised); RMS 0..1
  * each panel centred on the strongest plotted RF pixel
No smoothing.

Run from the repo root, e.g.
    poetry run python laura_scripts/Alignment_pipeline/plot_top_quality_cells.py \
        --recording 14_08_2026 --channel 6px_15Hz_30mins_shuffle_x6 --top 20
"""

import argparse
import contextlib
import io

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loading.noise_data import load_noise_data  # v2 RF dataset, checked
from matplotlib.transforms import Affine2D
from pathlib import Path

from PIL import Image

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import (
    alignment_cache_v2_file as alignment_cache_file,
    cone_image_file,
    noise_data_v2_file as noise_data_file,
)

Image.MAX_IMAGE_PIXELS = None

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Alignment_pipeline")
STYLE = {
    "cm_most_important": dict(cmap="coolwarm", signed=True, name="covariance"),
    "rms": dict(cmap="Reds", signed=False, name="RMS"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--recording", required=True, help="DD_MM_YYYY")
parser.add_argument("--channel", required=True)
parser.add_argument("--top", type=int, default=20, help="number of highest-quality cells")
parser.add_argument("--variable", default="cm_most_important", choices=list(STYLE))
parser.add_argument("--zoom-half-width-um", type=float, default=None,
                    help="half-width of each panel in projected µm (default: 15 stimulus pixels, min 100 µm)")
parser.add_argument("--n-cols", type=int, default=5)
parser.add_argument("--rf-alpha", type=float, default=0.6)
args = parser.parse_args()

style = STYLE[args.variable]
signed = style["signed"]

# ------------------------------------------------------------
# Alignment geometry (sizes + date-dependent transform, all from the v2 cache)
# ------------------------------------------------------------
cache = alignment_cache_file(args.recording)
cache_data = np.load(cache)
with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(cache)
# ORIENTATION [stimulus -> projection, date-dependent]: before 8 Oct 2026 (x, y) -> (-y, -x);
# from 8 Oct 2026 (x, y) -> (y, -x). Applied only through this matrix, never to the data.
A = geo.A_raw_rf_to_projected_um
half_w = int(cache_data["displayed_stim_width_px"]) * geo.corrected_stim_um_per_px / 2
half_h = int(cache_data["displayed_stim_height_px"]) * geo.corrected_stim_um_per_px / 2

cones = iio.imread(cone_image_file(args.recording))
if cones.ndim == 3 and cones.shape[2] == 4:
    cones = cones[..., :3]

# ------------------------------------------------------------
# RFs of the top-N cells
# ------------------------------------------------------------
data_file = noise_data_file(args.recording)
with load_noise_data(data_file, lazy=True) as ds:
    g.require_v2_rf_frame(ds)
    quality = ds.quality.sel(channel=args.channel).load()
    quality = quality.where(np.isfinite(quality), drop=True).sortby(quality, ascending=False)
    top = quality.isel(cell_index=slice(0, args.top))
    rfs = ds[args.variable].sel(channel=args.channel).sel(cell_index=top.cell_index).load()
    x = ds.x.values * geo.rf_scale_factor  # measured / nominal stimulus pixel size
    y = ds.y.values * geo.rf_scale_factor
    stim_px_um = float(abs(ds.x.values[1] - ds.x.values[0]))

step = x[1] - x[0]
extent = [x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2]
X, Y = np.meshgrid(x, y)
proj_x = A[0, 0] * X + A[0, 1] * Y + A[0, 2]
proj_y = A[1, 0] * X + A[1, 1] * Y + A[1, 2]
inside = (np.abs(proj_x) <= half_w) & (np.abs(proj_y) <= half_h)  # displayed-stimulus mask

zoom = args.zoom_half_width_um
if zoom is None:
    # stimulus pixel size from the dataset grid (noise_params), e.g. 6px x 2 µm = 12 µm
    zoom = max(100.0, 15 * stim_px_um)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
n = len(top.cell_index)
n_rows = int(np.ceil(n / args.n_cols))
fig, axes = plt.subplots(n_rows, args.n_cols, figsize=(3.4 * args.n_cols, 3.7 * n_rows), facecolor="white")
axes = np.atleast_1d(axes).ravel()

for rank, (ax, cell_id) in enumerate(zip(axes, top.cell_index.values), start=1):
    q = float(top.sel(cell_index=cell_id))
    rf = rfs.sel(cell_index=cell_id)
    peak = float((np.abs(rf) if signed else rf).max(skipna=True))
    if np.isfinite(peak) and peak != 0:
        rf = rf / peak
    img = np.where(inside, rf.fillna(0).transpose("y", "x").values, np.nan)
    if not signed:
        m = np.nanmax(img)
        img = img / m if np.isfinite(m) and m > 0 else img
    vmin, vmax = (-1.0, 1.0) if signed else (0.0, 1.0)

    strength = np.abs(img) if signed else img
    if not np.isfinite(strength).any() or np.nanmax(strength) <= 0:
        ax.set_title(f"Rank {rank} | cell {cell_id}\nQI = {q:.2f}\nno RF map", fontsize=9)
        ax.axis("off")
        continue
    iy, ix = np.unravel_index(np.nanargmax(strength), strength.shape)
    peak_proj = A[:2, :2] @ np.array([x[ix], y[iy]]) + A[:2, 2]

    ax.set_facecolor("black")
    # ORIENTATION [cone image -> projection]: from the electrode clicks (door-view image).
    ax.imshow(cones, extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData, zorder=2)
    im = ax.imshow(img, extent=extent, origin="upper", cmap=style["cmap"], vmin=vmin, vmax=vmax,
                   alpha=args.rf_alpha, interpolation="nearest",
                   transform=Affine2D(A) + ax.transData, zorder=10)
    ax.set_xlim(peak_proj[0] - zoom, peak_proj[0] + zoom)
    ax.set_ylim(peak_proj[1] + zoom, peak_proj[1] - zoom)  # door view: +y down
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Rank {rank} | cell {cell_id}\nQI = {q:.2f}", fontsize=9)

for ax in axes[n:]:
    ax.axis("off")

# 100 µm scale bar on the first panel
ax0 = axes[0]
x0, x1 = ax0.get_xlim(); y0, _ = ax0.get_ylim()
ax0.plot([x0 + 0.08 * (x1 - x0), x0 + 0.08 * (x1 - x0) + 50], [y0 - 0.08 * (x1 - x0)] * 2, "w-", lw=3, zorder=30)
ax0.text(x0 + 0.08 * (x1 - x0) + 25, y0 - 0.12 * (x1 - x0), "50 µm", color="white", ha="center", fontsize=8, zorder=30)

fig.suptitle(f"Top {n} quality cells  ·  {args.recording.replace('_', '/')}  ·  {args.channel}  ·  "
             f"RF ({style['name']}, per-cell normalised) on the cone mosaic  ·  v2 alignment  ·  "
             f"panels ±{zoom:.0f} µm, view from the door", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.2)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
stem = f"top_{n}_quality_cells_{style['name'].lower()}_{args.recording}_{args.channel}"
for fmt in ("png", "pdf"):
    out = OUTPUT_FOLDER / f"{stem}.{fmt}"
    fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
    print("saved", out)
