"""
Population RFs on the cone mosaic, one panel per recording (v2 alignment).

For every recording listed in RECORDINGS (each needs a v2 alignment cache made by
RF_alignment.py), this draws the population covariance (and/or RMS) RF map over the
cone image, combined the same way as RF_alignment.py CELL 3:
  * cells with quality >= QUALITY_MIN
  * each cell normalised to its own peak (|value| for covariance, value for RMS)
  * population = max over cells (covariance: the signed value with the largest |value|)
  * RF pixels outside the displayed stimulus are masked (dashed box)
  * covariance colour limits = +/- 99.5th percentile of |value|; RMS rescaled to 0-1
No smoothing.

Run from the repo root:
    poetry run python laura_scripts/Alignment_pipeline/plot_population_overlays_all_recordings.py

Loads one recording at a time (the noise_data_v2.nc files are large).
"""

import contextlib
import io
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loading.noise_data import load_noise_data  # v2 RF dataset, checked
from matplotlib.transforms import Affine2D
from PIL import Image

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import (
    NAS_LAURA,
    alignment_cache_v2_file as alignment_cache_file,
    cone_image_file,
    noise_data_v2_file as noise_data_file,
)

Image.MAX_IMAGE_PIXELS = None

# ============================================================
# SETTINGS
# ============================================================
OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")

# (recording DD_MM_YYYY, channel). Rule used on 18/11/2026: 32px if there is one,
# otherwise 12px, white if there is a choice. 30/04/2026 left out (19/11/2026): it has v2
# clicks, but its only noise channel is 4px (8 µm pixels), too fine to show RFs this way.
RECORDINGS = [
    ("07_05_2026", "12px_20Hz_20mins_shuffle_x12_white"),
    ("14_05_2026", "12px_20Hz_25mins_shuffle_x12"),
    ("15_05_2026", "12px_20Hz_25mins_shuffle_white"),
    ("22_07_2026", "32px_15Hz_20mins_shuffle_x4"),
    ("07_08_2026", "32px_15Hz_20mins_shuffle_x4"),
    ("14_08_2026", "32px_15Hz_20mins_shuffle_x4"),
    ("15_10_2026", "32px_15Hz_20mins_shuffle_x4"),  # stimulus index 11
]

QUALITY_MIN = 10
# "cm_most_important" (covariance) and/or "rms"; one figure per entry.
RF_VARIABLES = ["cm_most_important"]
RF_ALPHA = 0.6
SAVE_FORMATS = ("png", "pdf")
PNG_DPI = 200

# noise_analysis folder choice: alignment_recordings.NOISE_FOLDERS_IN_ORDER (QI10 first).

STYLE = {
    "cm_most_important": dict(cmap="coolwarm", signed=True, name="covariance",
                              file="covariance_RFs_on_cone_mosaic_v2_all_recordings"),
    "rms": dict(cmap="Reds", signed=False, name="RMS",
                file="rms_RFs_on_cone_mosaic_v2_all_recordings"),
}


# ============================================================
# HELPERS
# ============================================================
def population_map(ds, variable, channel, signed):
    """RF_alignment.py CELL 3, population mode, normalise_each_cell=True, combine_method='max'."""
    good = ds.quality.sel(channel=channel) >= QUALITY_MIN
    rfs = ds[variable].sel(channel=channel).sel(cell_index=good.cell_index[good]).load()
    peak = (np.abs(rfs) if signed else rfs).max(dim=["x", "y"], skipna=True)
    rfs = (rfs / peak.where(np.isfinite(peak) & (peak != 0))).fillna(0)
    if signed:
        combined = rfs.isel(cell_index=np.abs(rfs).argmax(dim="cell_index"))
    else:
        combined = rfs.max(dim="cell_index")
    # Stored in the v2 stimulus frame; imshow wants (y, x). No rotation here.
    return combined.transpose("y", "x").values, int(good.sum())


# ------------------------------------------------------------
# Optional command line: plot one recording / channel instead of RECORDINGS, e.g.
#   ... plot_population_overlays_all_recordings.py --recording 14_08_2026 \
#       --channel 6px_15Hz_30mins_shuffle_x6
# The file name then gets "_<recording>_<channel>" instead of "_all_recordings".
# ------------------------------------------------------------
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--recording")
_parser.add_argument("--channel")
_parser.add_argument("--variable", action="append", choices=list(STYLE))
_args = _parser.parse_args()
if _args.recording or _args.channel:
    if not (_args.recording and _args.channel):
        raise SystemExit("Give both --recording and --channel")
    RECORDINGS = [(_args.recording, _args.channel)]
    for _style in STYLE.values():
        _style["file"] = _style["file"].replace("_all_recordings", f"_{_args.recording}_{_args.channel}")
if _args.variable:
    RF_VARIABLES = _args.variable


# ============================================================
# LOAD (one recording at a time)
# ============================================================
panels = []
for recording, channel in RECORDINGS:
    data_file = noise_data_file(recording)
    with load_noise_data(data_file, lazy=True) as ds:
        g.require_v2_rf_frame(ds)
        maps = {}
        for variable in RF_VARIABLES:
            maps[variable], n_cells = population_map(ds, variable, channel, STYLE[variable]["signed"])
        x_um, y_um = ds.x.values, ds.y.values
    print(f"{recording}  {channel}: {n_cells} cells (quality >= {QUALITY_MIN}) from {data_file.parent.name}")
    panels.append(dict(recording=recording, channel=channel, maps=maps, x=x_um, y=y_um, n=n_cells))


# ============================================================
# PLOT
# ============================================================
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
n_cols = min(4, len(panels) + 1)  # +1 panel for the notes
n_rows = int(np.ceil((len(panels) + 1) / n_cols))

for variable in RF_VARIABLES:
    style = STYLE[variable]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.25 * n_cols, 6.25 * n_rows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, p in zip(axes, panels):
        cache = alignment_cache_file(p["recording"])
        cache_data = np.load(cache)
        with contextlib.redirect_stdout(io.StringIO()):
            geo = g.load_alignment_geometry(cache)  # sizes + date-dependent transform from the cache

        cones = iio.imread(cone_image_file(p["recording"]))
        if cones.ndim == 3 and cones.shape[2] == 4:
            cones = cones[..., :3]

        # RF coordinates scaled by measured / nominal stimulus pixel size
        x = p["x"] * geo.rf_scale_factor
        y = p["y"] * geo.rf_scale_factor
        step = x[1] - x[0]
        extent = [x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2]

        width_px = int(cache_data["displayed_stim_width_px"])
        height_px = int(cache_data["displayed_stim_height_px"])
        half_w = width_px * geo.corrected_stim_um_per_px / 2
        half_h = height_px * geo.corrected_stim_um_per_px / 2

        # ORIENTATION [stimulus -> projection, date-dependent]: A_raw_rf_to_projected_um is
        #   before 8 Oct 2026: (x, y) -> (-y, -x)  (OpenGL up-down flip + projector 90° ACW)
        #   from 8 Oct 2026:   (x, y) -> (y, -x)   (90° ACW only)
        # chosen by alignment_geometry.stimulus_to_projection_for_recording from the cache name.
        A = geo.A_raw_rf_to_projected_um

        # Mask RF pixels outside the displayed stimulus (clip_rf_to_displayed_stimulus = True)
        X, Y = np.meshgrid(x, y)
        proj_x = A[0, 0] * X + A[0, 1] * Y + A[0, 2]
        proj_y = A[1, 0] * X + A[1, 1] * Y + A[1, 2]
        inside = (np.abs(proj_x) <= half_w) & (np.abs(proj_y) <= half_h)
        rf_img = np.where(inside, p["maps"][variable], np.nan)

        finite = rf_img[np.isfinite(rf_img)]
        if style["signed"]:
            vmax = float(np.nanpercentile(np.abs(finite), 99.5)) if finite.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
            vmin = -vmax
        else:
            if finite.size and np.nanmax(finite) > 0:
                rf_img = rf_img / np.nanmax(finite)
            vmin, vmax = 0.0, 1.0

        ax.set_facecolor("black")
        # ORIENTATION [cone image -> projection]: A_image1_to_projected_um from the electrode
        # clicks (cone image is the door-view "flipped" image; no extra flip here).
        ax.imshow(cones, extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
                  transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData, zorder=2)
        im = ax.imshow(rf_img, extent=extent, origin="upper", cmap=style["cmap"], vmin=vmin, vmax=vmax,
                       alpha=RF_ALPHA, interpolation="nearest",
                       transform=Affine2D(A) + ax.transData, zorder=10)
        ax.plot([-half_w, half_w, half_w, -half_w, -half_w], [-half_h, -half_h, half_h, half_h, -half_h],
                "k--", lw=0.8, zorder=20)
        # Projected frame, door view: +x right, +y down.
        ax.set_xlim(-half_w - 30, half_w + 30)
        ax.set_ylim(half_h + 30, -half_h - 30)
        ax.set_aspect("equal")
        ax.set_title(f"{p['recording'].replace('_', '/')}  ·  {p['channel']}\n"
                     f"{p['n']} cells (quality ≥ {QUALITY_MIN})  ·  stimulus {width_px}×{height_px} px",
                     fontsize=11, loc="left")
        ax.set_xlabel("projected x (µm, + right)", fontsize=9)
        ax.set_ylabel("projected y (µm, + down)", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=f"{style['name']} (per-cell normalised)")

    notes_ax = axes[len(panels)]
    for ax in axes[len(panels):]:
        ax.axis("off")
    combine = ("signed largest |value| per pixel; colour limits =\n99.5th percentile of |value|"
               if style["signed"] else "max per pixel, rescaled 0-1")
    notes_ax.text(
        0, 0.95,
        f"{style['name'].capitalize()} RFs ({variable}) on the cone mosaic\n\n"
        f"Combined as RF_alignment.py CELL 3: each cell\nnormalised to its peak, population =\n{combine};\n"
        "masked to the displayed stimulus (dashed box).\n\n"
        "v2 alignment (alignment_geometry.py):\n"
        "before 8 Oct 2026  (x, y) -> (-y, -x)\n"
        "from 8 Oct 2026    (x, y) -> (y, -x)\n"
        "RF scale corrected by measured stimulus pixel size.\n"
        "View: from the door.",
        va="top", fontsize=9.5, family="monospace", transform=notes_ax.transAxes,
    )

    fig.tight_layout()
    for fmt in SAVE_FORMATS:
        out = OUTPUT_FOLDER / f"{style['file']}.{fmt}"
        fig.savefig(out, dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)
