"""
RF orientation vs the predicted "major cone row" orientation.

Assumption: the major cone rows radiate out from the optic nerve head (ONH). So at each
RF the predicted row orientation is the axis of the line ONH -> RF centre (radial axis).
For every cell this script measures the angle between the RF long axis and that radial
axis, and plots a histogram per recording (0° = RF parallel to the row, 90° = across it).

Inputs
  * the recordings and channels of the summary alignment plots
    (Alignment_pipeline/plot_population_overlays_all_recordings.py, RECORDINGS)
  * their v2 alignment caches (RF_alignment.py) and noise_data_v2.nc
  * optic_nerve_head_positions.json: the ONH in cone-image pixels, one entry per
    recording, made with click_optic_nerve_head.py. Recordings without an ONH are skipped.

Frame: everything is compared in the PROJECTED frame (the aligned cone image, door view,
x right, y down), the same frame as the summary alignment plots:
  * RF centre (positions) and RF long axis (angle) go through the same
    stimulus -> projection transform as the RF maps (geo.A_raw_rf_to_projected_um)
  * the ONH goes through the cone image -> projection affine from the electrode clicks
    (geo.A_image1_to_projected_um). An affine maps straight lines to straight lines, so the
    line ONH -> RF on the cone image is the same line in the projected frame.

Cells: quality >= QUALITY_MIN in the plotted channel, RF centre inside the displayed
stimulus (as the dashed box in the summary plots), finite angle/tilt. Histograms show all
of them, plus (outline) the elongated ones, tilt < TILT_MAX_ELONGATED
(tilt = minor/major axis, 1 = round; the angle of a round RF means little).

Run from the repo root (loads one recording at a time):
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/rf_cone_row_angle.py"
Options: --recording 14_08_2026 (only that one), --onh-file, --output-folder.

Outputs (OUTPUT_FOLDER)
  rf_vs_cone_row_angle_all_recordings.png/.pdf   ONE figure, one row per recording:
      whole cone image + ONH -> RF lines | displayed stimulus + RF axes |
      histogram, all cells | histogram, cells within EDGE_EXCLUSION_UM of the stimulus edge left out
  rf_vs_cone_row_angle_all_recordings_cells.csv  one row per cell
  (with --recording: rf_vs_cone_row_angle_<recording>.png/.pdf/_cells.csv instead)
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from PIL import Image

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import (
    alignment_cache_v2_file,
    cone_image_file,
    noise_data_v2_file,
)
from loading.noise_data import load_noise_data  # v2 RF dataset, checked

Image.MAX_IMAGE_PIXELS = None

# ============================================================
# SETTINGS
# ============================================================
HERE = Path(__file__).resolve().parent
ONH_FILE = HERE / "optic_nerve_head_positions.json"
OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")

# Same recordings / channels as the summary alignment plots
# (plot_population_overlays_all_recordings.py, 19/11/2026). 30/04 left out there too.
RECORDINGS = [
    ("07_05_2026", "12px_20Hz_20mins_shuffle_x12_white"),
    ("14_05_2026", "12px_20Hz_25mins_shuffle_x12"),
    ("15_05_2026", "12px_20Hz_25mins_shuffle_white"),
    ("22_07_2026", "32px_15Hz_20mins_shuffle_x4"),
    ("07_08_2026", "32px_15Hz_20mins_shuffle_x4"),
    ("14_08_2026", "32px_15Hz_20mins_shuffle_x4"),
    ("15_10_2026", "32px_15Hz_20mins_shuffle_x4"),
]

QUALITY_MIN = 10  # as the summary plots
TILT_MAX_ELONGATED = 0.75  # "elongated" subset (RF_alignment.py population_tilt_threshold)
ONLY_INSIDE_DISPLAYED_STIMULUS = True
# RFs centred this close to the edge of the displayed stimulus are cut off by it, and the
# cut-off part makes them look elongated ALONG the edge (14/08: 30 of 33 within 15° of the
# edge direction). The figure shows histograms with and without them.
EDGE_EXCLUSION_UM = 50
# Circular correlation test (see rf_row_correlation): number of shuffles.
N_PERMUTATIONS = 10000
RANDOM_SEED = 0
BIN_WIDTH_DEG = 10
COVARIANCE_ALPHA = 0.6  # population covariance RF overlay on the left panel (as the summary plots)
RF_AXIS_LENGTH_UM = 80  # length of the RF-axis lines on the map figure
# Colours for |RF - row| on the maps: warm colours only, so they stand out on the cyan cones
# (0° yellow -> orange -> red -> 90° magenta). Each line also gets a thin black outline.
DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "rf_row_difference", ["#ffe600", "#ff8c00", "#ff1e1e", "#ff3cf5"])
SAVE_FORMATS = ("png", "pdf")
PNG_DPI = 200


# ============================================================
# HELPERS
# ============================================================
def axis_deg(dx, dy):
    """Axis orientation 0-180° in the projected frame: 0° = right, increasing towards
    down (clockwise on screen, as the stimulus-frame `angle`)."""
    return np.degrees(np.arctan2(dy, dx)) % 180.0


def signed_axis_difference_deg(a_deg, b_deg):
    """a - b for axes, in [-90, 90). |result| = alignment_geometry.orientation_difference_deg."""
    return (np.asarray(a_deg) - np.asarray(b_deg) + 90.0) % 180.0 - 90.0


def rayleigh_axial(signed_diff_deg):
    """Rayleigh test on doubled angles (axial data): is there ANY preferred RF-vs-row
    angle? Returns (mean signed difference in deg, resultant length 0-1, p)."""
    th = np.deg2rad(2 * np.asarray(signed_diff_deg))
    n = th.size
    if n < 2:
        return np.nan, np.nan, np.nan
    c = np.exp(1j * th).mean()
    r = np.abs(c)
    rn = n * r
    # Zar, Biostatistical Analysis, eq. 27.4
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - rn ** 2)) - (1 + 2 * n)))
    return float(np.degrees(np.angle(c)) / 2), float(r), min(p, 1.0)


def rf_row_correlation(rf_axis_deg, row_axis_deg, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Does the RF axis ROTATE WITH the predicted row axis across cells?

    Circular-circular correlation (Fisher & Lee 1983 / Jammalamadaka & SenGupta 2001) on
    doubled angles (axes): r > 0 = cells whose row axis is rotated further also have their
    RF axis rotated further (r = 1: RF = row + a constant offset). Unlike the histograms, a
    single orientation shared by all RFs, a stimulus-grid bias or a constant offset from an
    imprecise ONH click gives r ~ 0, not a false positive.
    p (one-sided, r > 0): RF angles shuffled across cells N_PERMUTATIONS times.
    Power is low when the row axis hardly varies across the array (ONH far away);
    `row_spread_deg` (circular SD of the row axis) shows how much it varies.
    Returns (r, p, row_spread_deg)."""
    a = np.deg2rad(2 * np.asarray(rf_axis_deg, float))
    b = np.deg2rad(2 * np.asarray(row_axis_deg, float))
    n = a.size
    if n < 4:
        return np.nan, np.nan, np.nan

    def centred_sin(x):
        return np.sin(x - np.angle(np.exp(1j * x).mean(axis=-1, keepdims=True)))

    sa, sb = centred_sin(a), centred_sin(b)
    den_b = np.sqrt((sb ** 2).sum())
    r = float((sa * sb).sum() / (np.sqrt((sa ** 2).sum()) * den_b))
    rng = np.random.default_rng(seed)
    shuffled = centred_sin(np.stack([rng.permutation(a) for _ in range(n_perm)]))
    r_null = (shuffled * sb).sum(axis=1) / (np.sqrt((shuffled ** 2).sum(axis=1)) * den_b)
    p = float((1 + np.sum(r_null >= r)) / (1 + n_perm))
    R_b = np.abs(np.exp(1j * b).mean())
    row_spread = float(np.degrees(np.sqrt(-2 * np.log(max(R_b, 1e-12)))) / 2)
    return r, p, row_spread


def load_onh_positions(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def apply(A, x, y):
    """3x3 affine applied to point arrays."""
    return A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2]


def population_covariance_map(ds, channel):
    """As plot_population_overlays_all_recordings.population_map (covariance): cells with
    quality >= QUALITY_MIN, each normalised to its peak |value|, population = the signed
    value with the largest |value| per pixel. Stimulus frame, (y, x)."""
    good = ds.quality.sel(channel=channel) >= QUALITY_MIN
    rfs = ds["cm_most_important"].sel(channel=channel).sel(cell_index=good.cell_index[good]).load()
    peak = np.abs(rfs).max(dim=["x", "y"], skipna=True)
    rfs = (rfs / peak.where(np.isfinite(peak) & (peak != 0))).fillna(0)
    combined = rfs.isel(cell_index=np.abs(rfs).argmax(dim="cell_index"))
    return combined.transpose("y", "x").values


def analyse_recording(recording, channel, onh):
    with contextlib.redirect_stdout(io.StringIO()):
        geo = g.load_alignment_geometry(alignment_cache_v2_file(recording))
    cache = np.load(alignment_cache_v2_file(recording))
    half_w = int(cache["displayed_stim_width_px"]) * geo.corrected_stim_um_per_px / 2
    half_h = int(cache["displayed_stim_height_px"]) * geo.corrected_stim_um_per_px / 2

    data_file = noise_data_v2_file(recording)
    with load_noise_data(data_file, lazy=True) as ds:
        g.require_v2_rf_frame(ds)
        d = ds[["quality", "tilt", "angle", "positions"]].sel(channel=channel).load()
        x_um, y_um = ds.x.values, ds.y.values
        cov_map = population_covariance_map(ds, channel)

    cell = d.cell_index.values
    quality = d.quality.values
    tilt = d.tilt.values
    angle = d.angle.values
    # positions = (x, y) indices into the x / y coordinates, stimulus frame
    # (checked 19/11/2026 on 14/08: they hit the map peak; `angle` matches the map moments).
    pos = d.positions.transpose("cell_index", "pos_dim").values

    keep = (quality >= QUALITY_MIN) & np.isfinite(angle) & np.isfinite(tilt)
    cell, quality, tilt, angle, pos = cell[keep], quality[keep], tilt[keep], angle[keep], pos[keep]

    # RF centre, stimulus frame µm, scaled by measured / nominal stimulus pixel size
    sx = x_um[pos[:, 0].astype(int)] * geo.rf_scale_factor
    sy = y_um[pos[:, 1].astype(int)] * geo.rf_scale_factor

    # ORIENTATION [RF ROTATION, date-dependent]: stimulus -> projected, same transform as
    # the RF maps in the summary plots (before 8 Oct 2026 (x, y) -> (-y, -x); from 8 Oct (y, -x)).
    A = geo.A_raw_rf_to_projected_um
    px, py = apply(A, sx, sy)
    # RF long axis: `angle` is stimulus frame (0° = +x right, towards +y down) -> projected
    a = np.deg2rad(angle)
    vx, vy = A[0, 0] * np.cos(a) + A[0, 1] * np.sin(a), A[1, 0] * np.cos(a) + A[1, 1] * np.sin(a)
    rf_axis = axis_deg(vx, vy)

    # ORIENTATION [CONE IMAGE, fitted]: ONH from cone-image px -> projected µm with the
    # electrode-click affine (the same one that places the cone image in the summary plots).
    onh_x, onh_y = apply(geo.A_image1_to_projected_um, onh["x_px"], onh["y_px"])
    radial_axis = axis_deg(px - onh_x, py - onh_y)  # predicted major-cone-row axis
    signed = signed_axis_difference_deg(rf_axis, radial_axis)

    inside = (np.abs(px) <= half_w) & (np.abs(py) <= half_h)
    edge_distance = np.minimum(half_w - np.abs(px), half_h - np.abs(py))  # µm, < 0 = outside
    table = pd.DataFrame(dict(
        recording=recording, channel=channel, cell_index=cell, quality=quality, tilt=tilt,
        elongated=tilt < TILT_MAX_ELONGATED,
        rf_angle_stimulus_deg=angle,
        x_projected_um=px, y_projected_um=py, inside_displayed_stimulus=inside,
        distance_to_stimulus_edge_um=edge_distance, near_edge=edge_distance < EDGE_EXCLUSION_UM,
        distance_to_onh_um=np.hypot(px - onh_x, py - onh_y),
        rf_axis_projected_deg=rf_axis, cone_row_axis_projected_deg=radial_axis,
        rf_minus_row_deg=signed, rf_vs_row_difference_deg=np.abs(signed),
    ))
    if ONLY_INSIDE_DISPLAYED_STIMULUS:
        n_out = int((~inside).sum())
        table = table[inside].reset_index(drop=True)
    else:
        n_out = 0
    print(f"{recording}  {channel}: {len(table)} cells (quality >= {QUALITY_MIN}"
          f"{f', {n_out} outside the displayed stimulus dropped' if n_out else ''}), "
          f"{int(table.elongated.sum())} with tilt < {TILT_MAX_ELONGATED}, "
          f"{int(table.near_edge.sum())} within {EDGE_EXCLUSION_UM} µm of the stimulus edge; "
          f"ONH at ({onh_x:.0f}, {onh_y:.0f}) µm, "
          f"{np.hypot(onh_x, onh_y):.0f} µm from the stimulus centre")
    return dict(recording=recording, channel=channel, table=table, geo=geo,
                cov_map=cov_map, x_um=x_um, y_um=y_um,
                half_w=half_w, half_h=half_h, onh_xy=(onh_x, onh_y), data_file=data_file)


# ============================================================
# PLOTS: one figure, one row per recording:
#   [whole cone image + ONH -> RF lines] [displayed stimulus + RF axes] [histogram]
# ============================================================
def plot_maps(ax0, ax1, r, colorbar=True, onh_lines=True, onh_marker="*", cov_alpha=None):
    """Left = whole cone image with ONH, stimulus box and ONH -> RF lines;
    right = displayed stimulus with each RF axis (coloured by the difference) and the
    radial (predicted row) axis through its centre (white dashes).
    ax1=None draws only the left panel (used by summary_figure.py)."""
    geo, t = r["geo"], r["table"]
    img = Image.open(cone_image_file(r["recording"]))
    full_w, full_h = img.size
    onh_x, onh_y = r["onh_xy"]
    hw, hh = r["half_w"], r["half_h"]
    box = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])

    for ax, down in ((ax0, 8), (ax1, 4)) if ax1 is not None else ((ax0, 8),):
        small = np.asarray(img.convert("RGB").reduce(down))
        ax.set_facecolor("black")
        # ORIENTATION [CONE IMAGE, fitted]: as the summary plots, no extra flip.
        ax.imshow(small, extent=[0, full_w, full_h, 0], origin="upper",
                  transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData)
        ax.plot(box[:, 0], box[:, 1], "w--", lw=1)
        ax.set_aspect("equal")
        ax.set_xlabel("projected x (µm, + right)", fontsize=9)
        ax.set_ylabel("projected y (µm, + down)", fontsize=9)
        ax.tick_params(labelsize=8)
    img.close()

    # Left: population covariance RFs, drawn exactly as the summary alignment plots.
    A = geo.A_raw_rf_to_projected_um
    x = r["x_um"] * geo.rf_scale_factor
    y = r["y_um"] * geo.rf_scale_factor
    step = x[1] - x[0]
    extent = [x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2]
    X, Y = np.meshgrid(x, y)
    proj_x, proj_y = apply(A, X, Y)
    rf_img = np.where((np.abs(proj_x) <= hw) & (np.abs(proj_y) <= hh), r["cov_map"], np.nan)
    finite = rf_img[np.isfinite(rf_img)]
    vmax = float(np.nanpercentile(np.abs(finite), 99.5)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    # ORIENTATION [RF ROTATION, date-dependent]: stimulus -> projected, as the summary plots.
    im = ax0.imshow(rf_img, extent=extent, origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    alpha=COVARIANCE_ALPHA if cov_alpha is None else cov_alpha, interpolation="nearest",
                    transform=Affine2D(A) + ax0.transData, zorder=3)
    if colorbar:
        cb0 = plt.colorbar(im, ax=ax0, fraction=0.04, pad=0.02)
        cb0.set_label("covariance (per-cell normalised)", fontsize=8)
        cb0.ax.tick_params(labelsize=7)

    # Left: overview including the ONH
    corners = np.array([[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], float)
    cx, cy = apply(geo.A_image1_to_projected_um, corners[:, 0], corners[:, 1])
    xs, ys = np.r_[cx, onh_x], np.r_[cy, onh_y]
    pad = 0.05 * max(np.ptp(xs), np.ptp(ys))
    ax0.set_xlim(xs.min() - pad, xs.max() + pad)
    ax0.set_ylim(ys.max() + pad, ys.min() - pad)  # ORIENTATION [DISPLAY only]: y down
    if onh_lines:
        ax0.add_collection(LineCollection(
            [[(onh_x, onh_y), (x, y)] for x, y in zip(t.x_projected_um, t.y_projected_um)],
            colors="yellow", lw=0.4, alpha=0.5, zorder=5))
    ax0.plot(onh_x, onh_y, onh_marker, color="red", ms=16 if onh_marker == "*" else 11, mec="white",
             label="optic nerve head", zorder=6)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(f"{r['recording'].replace('_', '/')}  ·  {r['channel']}\n"
                  f"covariance RFs (quality ≥ {QUALITY_MIN}), ONH -> RF centres", loc="left", fontsize=10)

    if ax1 is None:
        return

    # Right: zoom on the displayed stimulus
    L = RF_AXIS_LENGTH_UM / 2
    cmap = DIFFERENCE_CMAP
    outline = [patheffects.Stroke(linewidth=4.2, foreground="black"), patheffects.Normal()]
    outline_thin = [patheffects.Stroke(linewidth=2.4, foreground="black"), patheffects.Normal()]
    for _, c in t.iterrows():
        ra = np.deg2rad(c.cone_row_axis_projected_deg)
        ax1.plot([c.x_projected_um - L * np.cos(ra), c.x_projected_um + L * np.cos(ra)],
                 [c.y_projected_um - L * np.sin(ra), c.y_projected_um + L * np.sin(ra)],
                 color="white", lw=0.8, ls=(0, (2, 2)), zorder=5)
        a = np.deg2rad(c.rf_axis_projected_deg)
        ax1.plot([c.x_projected_um - L * np.cos(a), c.x_projected_um + L * np.cos(a)],
                 [c.y_projected_um - L * np.sin(a), c.y_projected_um + L * np.sin(a)],
                 color=cmap(c.rf_vs_row_difference_deg / 90), lw=2.6 if c.elongated else 1.0,
                 solid_capstyle="round", zorder=6, alpha=0.35 if c.near_edge else 1.0,
                 path_effects=None if c.near_edge else (outline if c.elongated else outline_thin))
    e = EDGE_EXCLUSION_UM
    ax1.plot([-hw + e, hw - e, hw - e, -hw + e, -hw + e], [-hh + e, -hh + e, hh - e, hh - e, -hh + e],
             color="0.75", ls=":", lw=1.2, zorder=4)
    ax1.plot(onh_x, onh_y, "r*", ms=14, mec="white", zorder=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90))
    cb = plt.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label("|RF axis − row axis| (°)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax1.set_xlim(-hw - 40, hw + 40)
    ax1.set_ylim(hh + 40, -hh - 40)  # ORIENTATION [DISPLAY only]: y down
    ax1.set_title(f"RF long axis (thick: tilt < {TILT_MAX_ELONGATED}), white dashes = predicted row\n"
                  f"faded: within {EDGE_EXCLUSION_UM} µm of the stimulus edge (grey dotted box)",
                  loc="left", fontsize=10)


def plot_histogram(ax, r, exclude_edge=False):
    bins = np.arange(0, 90 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    t = r["table"]
    if exclude_edge:
        n_edge = int(t.near_edge.sum())
        t = t[~t.near_edge]
        heading = f"Edge cells EXCLUDED ({n_edge} within {EDGE_EXCLUSION_UM} µm left out)\n"
    else:
        heading = "All cells (including edge cells)\n"
    all_d = t.rf_vs_row_difference_deg.values
    el = t[t.elongated]
    ax.hist(all_d, bins=bins, color="#7aa6c2", edgecolor="white", label=f"all (n={len(t)})")
    ax.hist(el.rf_vs_row_difference_deg.values, bins=bins, histtype="step", color="k", lw=1.6,
            label=f"tilt < {TILT_MAX_ELONGATED} (n={len(el)})")
    ax.axhline(len(t) * BIN_WIDTH_DEG / 90, color="#7aa6c2", ls="--", lw=1, label="uniform")
    _, _, p_all = rayleigh_axial(t.rf_minus_row_deg.values)
    _, _, p_el = rayleigh_axial(el.rf_minus_row_deg.values)
    med_all = np.median(all_d) if len(t) else np.nan
    med_el = np.median(el.rf_vs_row_difference_deg) if len(el) else np.nan
    r_c, p_c, spread = rf_row_correlation(t.rf_axis_projected_deg, t.cone_row_axis_projected_deg)
    r_ce, p_ce, _ = rf_row_correlation(el.rf_axis_projected_deg, el.cone_row_axis_projected_deg)
    print(f"  {r['recording']} {'edge excluded' if exclude_edge else 'all cells    '}: "
          f"RF-row correlation r={r_c:+.2f} p={p_c:.2g} (n={len(t)}, row-axis spread {spread:.0f}°); "
          f"tilt<{TILT_MAX_ELONGATED}: r={r_ce:+.2f} p={p_ce:.2g} (n={len(el)})")
    ax.set_title(heading + f"median all {med_all:.0f}° (Rayleigh p={p_all:.2g})\n"
                 f"median tilt<{TILT_MAX_ELONGATED} {med_el:.0f}° (p={p_el:.2g})\n"
                 f"RF rotates with row: r={r_c:+.2f}, p={p_c:.2g} (row spread {spread:.0f}°)\n"
                 f"   tilt<{TILT_MAX_ELONGATED}: r={r_ce:+.2f}, p={p_ce:.2g}",
                 loc="left", fontsize=10)
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel("|RF long axis − cone row axis| (°)\n0 = along the row, 90 = across", fontsize=9)
    ax.set_ylabel("cells", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)  # same height as the map panels


def plot_all(results, out_folder, name):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(25, 6.2 * n + 1.2), facecolor="white", squeeze=False)
    for row, r in zip(axes, results):  # one recording at a time (cone images are large)
        plot_maps(row[0], row[1], r)
        plot_histogram(row[2], r)
        plot_histogram(row[3], r, exclude_edge=True)
    fig.suptitle(
        "RF long axis vs predicted major cone row (rows assumed to radiate from the optic nerve head; "
        "row axis = line ONH -> RF centre)\n"
        f"Aligned cone-image frame (projected, door view). Cells: quality >= {QUALITY_MIN}, RF centre "
        "inside the displayed stimulus (dashed box). Rayleigh p: doubled signed difference, "
        "any preferred angle.\n"
        "'RF rotates with row': circular correlation of RF axis vs row axis across cells (doubled angles), "
        f"one-sided permutation p ({N_PERMUTATIONS} shuffles); not fooled by one shared orientation or a constant offset. "
        "Row spread = circular SD of the row axis (small = little power).",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / fig.get_figheight()))  # room for the title
    for fmt in SAVE_FORMATS:
        out = out_folder / f"{name}.{fmt}"
        fig.savefig(out, dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def recordings_with_onh(recording=None, onh_file=ONH_FILE):
    """[(recording, channel)] from RECORDINGS (or just `recording`) that have a clicked ONH,
    and the ONH positions. Prints the click command for any without one."""
    recordings = [rc for rc in RECORDINGS if recording in (None, rc[0])]
    if not recordings:
        raise SystemExit(f"{recording} is not in RECORDINGS")
    onh_positions = load_onh_positions(onh_file)
    missing = [rec for rec, _ in recordings if rec not in onh_positions]
    if missing:
        print("No optic nerve head yet for: " + ", ".join(missing) + " (skipped). Click it with:")
        for rec in missing:
            print(f'  poetry run python "{HERE.relative_to(Path.cwd()) if HERE.is_relative_to(Path.cwd()) else HERE}'
                  f'/click_optic_nerve_head.py" --recording {rec}')
    recordings = [rc for rc in recordings if rc[0] in onh_positions]
    if not recordings:
        raise SystemExit("Nothing to analyse.")
    return recordings, onh_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", help="only this recording (DD_MM_YYYY)")
    parser.add_argument("--onh-file", type=Path, default=ONH_FILE)
    parser.add_argument("--output-folder", type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()
    recordings, onh_positions = recordings_with_onh(args.recording, args.onh_file)

    args.output_folder.mkdir(parents=True, exist_ok=True)
    results = [analyse_recording(rec, ch, onh_positions[rec]) for rec, ch in recordings]

    # Only the recordings with an ONH are in the figure; the name says which when it is one.
    name = (("rf_vs_cone_row_angle_" + recordings[0][0]) if len(recordings) == 1
            else "rf_vs_cone_row_angle_all_recordings")
    plot_all(results, args.output_folder, name)
    csv = args.output_folder / f"{name}_cells.csv"
    pd.concat([r["table"] for r in results], ignore_index=True).to_csv(csv, index=False)
    print("saved", csv)


# Importable (rf_position_arc_test.py uses the loading code); runs only as a script.
if __name__ == "__main__":
    main()
