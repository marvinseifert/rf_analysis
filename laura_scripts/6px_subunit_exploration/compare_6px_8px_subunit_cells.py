"""
Do the "subunit" spots of a cell repeat in a second noise stimulus? (20/11/2026)

For one recording and two noise channels (default 14/08/2026: 6px and 8px, both shuffled in 2 µm steps):
  * cells: the N most subunit-like cells in channel A (same score as plot_cells_with_stas.py,
    subunit_tools.py), with quality >= SUBUNIT_QUALITY_MIN in channel A
  * per cell: covariance RF of A on the cones | covariance RF of B on the cones |
    grey cones with both RFs outlined at 50 % of their peak and each channel's peaks marked
  * "fine structure" match: both maps smoothed by SUBUNIT_SMOOTH_UM minus smoothed by
    COARSE_UM (keeps detail finer than ~COARSE_UM, i.e. spots, removes the overall blob),
    Pearson correlation inside the RF centre. Chance level: same, with map B rotated by
    90 / 180 / 270 degrees about the RF centre (keeps how spotty B is, scrambles where the spots are).
  * last panel: that match for every cell with quality >= SUBUNIT_QUALITY_MIN in BOTH channels.

All panels centred on channel A's RF centre (`positions`). Same frame for both channels: one
stimulus grid, one transform (A_raw_rf_to_projected_um), view from the door.

Caveat: the two stimuli were played at different times, so a real RF can differ between them
(e.g. adaptation); a mismatch is not proof of noise, a match is good evidence the spots are real.

Run from the repo root:
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/compare_6px_8px_subunit_cells.py
"""

import argparse
import contextlib
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
import sys as _sys
from pathlib import Path as _Path
# This folder's name starts with a digit, so it can't be imported as a package:
# load subunit_tools.py from the script's own folder.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from subunit_tools import (
    SUBUNIT_QUALITY_MIN, SUBUNIT_SMOOTH_UM, smooth_normalised, subunit_score,
)

Image.MAX_IMAGE_PIXELS = None
OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
COARSE_UM = 15.0
COLOUR_A = "tab:green"
COLOUR_B = "yellow"

parser = argparse.ArgumentParser()
parser.add_argument("--recording", default="14_08_2026")
parser.add_argument("--channel-a", default="6px_15Hz_30mins_shuffle_x6")
parser.add_argument("--channel-b", default="8px_15Hz_35mins_shuffle_x8")
parser.add_argument("--n-cells", type=int, default=10)
parser.add_argument("--zoom-half-width-um", type=float, default=79.0, help="same as plot_cells_with_stas.py for 14/08 6px")
parser.add_argument("--rf-alpha", type=float, default=0.6)
args = parser.parse_args()
A_LABEL = args.channel_a.split("_")[0]
B_LABEL = args.channel_b.split("_")[0]

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

# ------------------------------------------------------------ data
data_file = noise_data_file(args.recording)
with load_noise_data(data_file, lazy=True) as ds:
    g.require_v2_rf_frame(ds)
    qa = ds.quality.sel(channel=args.channel_a).load()
    qb = ds.quality.sel(channel=args.channel_b).load()
    pool = [int(c) for c in qa.cell_index.values if np.isfinite(qa.sel(cell_index=c)) and qa.sel(cell_index=c) >= SUBUNIT_QUALITY_MIN]
    both = [c for c in pool if np.isfinite(qb.sel(cell_index=c)) and qb.sel(cell_index=c) >= SUBUNIT_QUALITY_MIN]
    cm_a = ds.cm_most_important.sel(channel=args.channel_a, cell_index=pool).transpose("cell_index", "y", "x").load()
    cm_b = ds.cm_most_important.sel(channel=args.channel_b, cell_index=pool).transpose("cell_index", "y", "x").load()
    pos_a = ds.positions.sel(channel=args.channel_a, cell_index=pool).load()
    centre_r = ds.center_outline_um.sel(channel=args.channel_a, cell_index=pool).max("degree").load()
    x = ds.x.values * geo.rf_scale_factor  # measured / nominal stimulus pixel size
    y = ds.y.values * geo.rf_scale_factor

step = abs(x[1] - x[0])
extent = [x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2]
X, Y = np.meshgrid(x, y)
proj_x = A[0, 0] * X + A[0, 1] * Y + A[0, 2]
proj_y = A[1, 0] * X + A[1, 1] * Y + A[1, 2]
inside = (np.abs(proj_x) <= half_w) & (np.abs(proj_y) <= half_h)


def norm_map(raw):
    raw = np.asarray(raw, float)
    peak = np.nanmax(np.abs(raw)) if np.isfinite(raw).any() else np.nan
    if not np.isfinite(peak) or peak == 0:
        return None
    return np.where(inside & np.isfinite(raw), raw / peak, np.nan)


def radius(c):
    r = float(centre_r.sel(cell_index=c))
    return r if np.isfinite(r) and r > 0 else 50.0


def fine(m):
    m0 = np.nan_to_num(m)
    return ndimage.gaussian_filter(m0, SUBUNIT_SMOOTH_UM / step) - ndimage.gaussian_filter(m0, COARSE_UM / step)


def fine_match(c, ma, mb):
    """(real correlation, mean of the 3 rotated correlations) inside the RF centre."""
    px, py = (int(v) for v in pos_a.sel(cell_index=c).values)
    r_px = int(np.ceil(radius(c) / step))
    if min(py, px) - r_px < 0 or py + r_px >= ma.shape[0] or px + r_px >= ma.shape[1]:
        return np.nan, np.nan
    fa, fb = fine(ma), fine(mb)
    sl = (slice(py - r_px, py + r_px + 1), slice(px - r_px, px + r_px + 1))
    wa, wb = fa[sl], fb[sl]
    valid = np.isfinite(ma[sl]) & np.isfinite(mb[sl])
    yy, xx = np.indices(wa.shape)
    disc = (np.hypot(yy - r_px, xx - r_px) <= r_px) & valid
    if disc.sum() < 50:
        return np.nan, np.nan

    def corr(u, v, mask):
        return float(np.corrcoef(u[mask], v[mask])[0, 1])

    real = corr(wa, wb, disc)
    rot = [corr(wa, np.rot90(wb, k), disc & np.rot90(valid, k)) for k in (1, 2, 3)]
    return real, float(np.mean(rot))


# ------------------------------------------------------------ choose cells (score in channel A)
scores, maps = [], {}
for c in pool:
    ma, mb = norm_map(cm_a.sel(cell_index=c).values), norm_map(cm_b.sel(cell_index=c).values)
    if ma is None:
        continue
    maps[c] = (ma, mb)
    n_extra, dip_sum, peaks = subunit_score(ma, step, radius(c))
    scores.append((n_extra, dip_sum, c, peaks))
scores.sort(key=lambda t: (t[0], t[1]), reverse=True)
chosen = scores[:args.n_cells]

match = {}
for c in both:
    if c in maps and maps[c][1] is not None:
        match[c] = fine_match(c, *maps[c])
print(f"cells with quality >= {SUBUNIT_QUALITY_MIN} in both channels: {len(match)}")
for c, (real, rot) in match.items():
    print(f"  cell {c}: fine-structure r = {real:.2f}, rotated = {rot:.2f}, "
          f"QI {A_LABEL} {float(qa.sel(cell_index=c)):.0f} / {B_LABEL} {float(qb.sel(cell_index=c)):.0f}")


# ------------------------------------------------------------ plot
def to_proj(ix, iy):
    return A[:2, :2] @ np.array([x[ix], y[iy]]) + A[:2, 2]


def draw_cones(ax, grey=False):
    ax.set_facecolor("black")
    # ORIENTATION [cone image -> projection]: from the electrode clicks (door-view image).
    ax.imshow(cones_grey if grey else cones, cmap="gray" if grey else None,
              extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData, zorder=2)


def frame(ax, centre):
    z = args.zoom_half_width_um
    ax.set_xlim(centre[0] - z, centre[0] + z)
    ax.set_ylim(centre[1] + z, centre[1] - z)  # door view: +y down
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])


def draw_rf(ax, img):
    ax.imshow(img, extent=extent, origin="upper", cmap="coolwarm", vmin=-1, vmax=1, alpha=args.rf_alpha,
              interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=10)


n = len(chosen)
n_cols = 2
n_rows = int(np.ceil(n / n_cols)) + 1  # + summary row
fig = plt.figure(figsize=(7.2 * n_cols, 2.9 * n_rows + 0.6), facecolor="white")
gs = fig.add_gridspec(n_rows, 3 * n_cols, width_ratios=[1, 1, 1.15] * n_cols, wspace=0.08, hspace=0.32, top=0.95, bottom=0.05, left=0.02, right=0.98)
rec_label = args.recording.replace("_", "/")

for k, (n_extra, _, c, peaks_a) in enumerate(chosen):
    r, col = divmod(k, n_cols)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[r, 3 * col + i]) for i in range(3))
    ma, mb = maps[c]
    px, py = (int(v) for v in pos_a.sel(cell_index=c).values)
    centre = to_proj(px, py)
    qb_c = float(qb.sel(cell_index=c))
    for ax, m, lab, q in ((ax_a, ma, A_LABEL, float(qa.sel(cell_index=c))), (ax_b, mb, B_LABEL, qb_c)):
        draw_cones(ax)
        if m is not None:
            draw_rf(ax, m)
        frame(ax, centre)
        ax.set_title(f"{lab}  ·  QI {q:.0f}" if np.isfinite(q) else f"{lab}  ·  QI n/a", fontsize=8, loc="left")
    ax_a.set_title(f"cell {c}  ·  " + ax_a.get_title(loc="left"), fontsize=8, loc="left")
    draw_cones(ax_c, grey=True)
    for m, colour, marker in ((ma, COLOUR_A, "+"), (mb, COLOUR_B, "x")):
        if m is None:
            continue
        ax_c.contour(x, y, smooth_normalised(m, step), levels=[0.5], colors=[colour], linewidths=1.0,
                     transform=Affine2D(A) + ax_c.transData, zorder=10)
        _, _, pk = subunit_score(m, step, radius(c))
        pk = np.array([to_proj(ix, iy) for iy, ix in pk])
        ax_c.plot(pk[:, 0], pk[:, 1], marker, color=colour, ms=7, mew=1.4, zorder=11)
    frame(ax_c, centre)
    real, rot = match.get(c, (np.nan, np.nan))
    if np.isnan(real) and mb is not None:
        real, rot = fine_match(c, ma, mb)
    ax_c.set_title(f"match r = {real:.2f}  ·  chance {rot:.2f}", fontsize=8, loc="left")
    if k == 0:
        x0, x1 = ax_a.get_xlim(); y0, _ = ax_a.get_ylim(); w = x1 - x0
        ax_a.plot([x0 + 0.08 * w, x0 + 0.08 * w + 50], [y0 - 0.08 * w] * 2, "w-", lw=3, zorder=30)
        ax_a.text(x0 + 0.08 * w + 25, y0 - 0.13 * w, "50 µm", color="white", ha="center", fontsize=7, zorder=30)

# summary: every cell good in both channels
ax_s = fig.add_subplot(gs[n_rows - 1, 0:3])
cells_m = [c for c in match if np.isfinite(match[c][0])]
real = np.array([match[c][0] for c in cells_m]); rot = np.array([match[c][1] for c in cells_m])
order = np.argsort(real)
ax_s.scatter(np.arange(len(order)), rot[order], s=18, color="0.6", label=f"{B_LABEL} map rotated (chance)", zorder=3)
ax_s.scatter(np.arange(len(order)), real[order], s=22, color=COLOUR_A, label="real", zorder=4)
for i, j in enumerate(order):
    ax_s.plot([i, i], [rot[j], real[j]], color="0.8", lw=0.8, zorder=1)
ax_s.axhline(0, color="0.5", lw=0.6, ls=":")
ax_s.set_xticks([]); ax_s.set_xlabel(f"cells with quality ≥ {SUBUNIT_QUALITY_MIN} in both stimuli (n = {len(order)})", fontsize=8)
ax_s.set_ylabel(f"fine-structure match\n{A_LABEL} vs {B_LABEL} (r)", fontsize=8)
ax_s.tick_params(labelsize=7)
ax_s.legend(fontsize=7, frameon=False, loc="upper left")
ax_s.spines[["top", "right"]].set_visible(False)
better = int(np.sum(real > rot))
ax_t = fig.add_subplot(gs[n_rows - 1, 3:6]); ax_t.axis("off")
ax_t.text(0, 1,
          f"Top rows: the {n} most subunit-like cells in {A_LABEL} (as in the subunit figure).\n"
          f"Left: {A_LABEL} covariance RF on the cones. Middle: {B_LABEL}. Right: grey cones, RFs outlined at 50 % of their\n"
          f"peak ({A_LABEL} green, {B_LABEL} yellow); + / x = peaks counted in each.\n\n"
          f"Fine-structure match: correlation of the small-scale detail (spots smaller than ~{COARSE_UM:.0f} µm)\n"
          f"inside the RF centre. Chance = the same with the {B_LABEL} map turned 90/180/270°.\n"
          f"Real above chance in {better} of {len(order)} cells; median r = {np.median(real):.2f} vs chance {np.median(rot):.2f}.\n\n"
          f"The two stimuli were played at different times, so a mismatch can also be a real change\n(e.g. adaptation), not only noise.",
          va="top", fontsize=8, transform=ax_t.transAxes)
fig.suptitle(f"{rec_label}  ·  do the RF spots repeat between {args.channel_a} and {args.channel_b}?  ·  view from the door",
             fontsize=11, y=0.99)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
stem = f"subunit_spots_{A_LABEL}_vs_{B_LABEL}_{args.recording}"
for fmt in ("png", "pdf"):
    out = OUTPUT_FOLDER / f"{stem}.{fmt}"
    fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
    print("saved", out)
