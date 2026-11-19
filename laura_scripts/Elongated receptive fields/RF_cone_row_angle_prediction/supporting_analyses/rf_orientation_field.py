"""
Smoothed RF orientation field: do RF orientations, averaged over space, form circles around
the optic nerve head (ONH)?

No assumption about where the rows go is used to MAKE the field; the circles around the
ONH are only drawn on top and used to score it.

  1. Cells as rf_cone_row_angle.py (quality >= QUALITY_MIN, inside the displayed stimulus),
     minus those within EDGE_EXCLUSION_UM of the stimulus edge (cut-off RFs get the edge's
     orientation).
  2. Each cell's long axis as a doubled-angle vector (axes: 0° and 180° are the same),
     weighted by elongation, w = 1 - tilt (round RFs have no meaningful axis).
  3. Smoothed field: at each grid point, Gaussian-weighted (SMOOTH_SIGMA_UM) sum of nearby
     cells' vectors. Direction = local average RF orientation; coherence (0-1) = how much
     nearby cells agree. Drawn as short lines (longer = more coherent) where enough cells
     are nearby (MIN_LOCAL_WEIGHT).
  4. Circle score S: coherence-weighted mean of cos(2 x (field angle - circle tangent))
     over the grid. +1 = field follows circles around the ONH, -1 = follows spokes, 0 neither.
  5. Shuffle test: the same RF angles (with their weights) randomly reassigned to the cell
     positions N_PERMUTATIONS times; p = fraction of shuffles with S >= the real S. A single
     orientation shared by all cells is in every shuffle too, so it cannot fake a result.
  6. Picture: S for every possible centre on the cone image; if RF orientations form
     circles around the ONH, the best centre should be near the clicked ONH. (A uniform
     field also matches circles around a far-away centre, so don't over-read this.)

Everything in the projected frame (aligned cone image, door view), as the summary plots.

Run from the repo root (one recording at a time):
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_orientation_field.py"
Options: --recording 14_08_2026, --output-folder.

Outputs (OUTPUT_FOLDER of rf_cone_row_angle.py)
  rf_orientation_field_all_recordings.png/.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from PIL import Image

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # rf_cone_row_angle.py is one folder up
import rf_cone_row_angle as rca  # noqa: E402
from laura_scripts.Alignment_pipeline.alignment_recordings import cone_image_file

Image.MAX_IMAGE_PIXELS = None

# ============================================================
# SETTINGS
# ============================================================
SMOOTH_SIGMA_UM = 150  # Gaussian width; nearest-neighbour RF distances are ~75-125 µm
GRID_STEP_UM = 50  # spacing of the drawn / scored field
MIN_LOCAL_WEIGHT = 1.0  # need at least ~1 fully elongated cell's worth of weight nearby
WEIGHT_BY_ELONGATION = True  # w = 1 - tilt; False = every cell counts the same
N_PERMUTATIONS = 2000
RANDOM_SEED = 0
CENTRE_GRID_STEP_UM = 50  # candidate-centre grid for the best-centre picture
CIRCLE_SPACING_UM = 200  # ideal circles drawn around the ONH
FIELD_LINE_MAX_UM = 45  # length of a fully coherent field line
SAVE_FORMATS = ("png", "pdf")
PNG_DPI = 200
CMAP = rca.DIFFERENCE_CMAP  # here: 0° yellow = along the circle -> 90° magenta = along the spoke


# ============================================================
# ANALYSIS
# ============================================================
def smoothed_field(cells_xy, axis_deg, weights, grid_xy):
    """Kernel matrix and smoothed doubled-angle field at grid points.
    Returns K (g, n), V (g,) complex sum, local weight W (g,)."""
    d2 = ((grid_xy[:, None, :] - cells_xy[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * SMOOTH_SIGMA_UM ** 2))
    z = weights * np.exp(2j * np.deg2rad(axis_deg))
    return K, K @ z, K @ weights


def circle_score(V, W, use, grid_xy, centre):
    """Coherence-weighted agreement of the field with circles around `centre`.
    Field angle phi = arg(V)/2, coherence = |V|/W; tangent of the circle = spoke + 90°.
    S = sum(|V| cos(2(phi - tangent))) / sum(W) over the used grid points
      = sum(Re(V e^{-2i tangent})) / sum(W)."""
    gx, gy = grid_xy[use, 0], grid_xy[use, 1]
    tangent = np.arctan2(gy - centre[1], gx - centre[0]) + np.pi / 2
    return float(np.real(V[use] * np.exp(-2j * tangent)).sum() / W[use].sum())


def analyse(res):
    t = res["table"]
    t = t[~t.near_edge].reset_index(drop=True)
    xy = t[["x_projected_um", "y_projected_um"]].values
    axis = t.rf_axis_projected_deg.values
    w = np.clip(1 - t.tilt.values, 0, 1) if WEIGHT_BY_ELONGATION else np.ones(len(t))
    onh = np.array(res["onh_xy"])

    hw = res["half_w"] - rca.EDGE_EXCLUSION_UM
    hh = res["half_h"] - rca.EDGE_EXCLUSION_UM
    gx = np.arange(-hw, hw + 1e-9, GRID_STEP_UM)
    gy = np.arange(-hh, hh + 1e-9, GRID_STEP_UM)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.c_[GX.ravel(), GY.ravel()]
    K, V, W = smoothed_field(xy, axis, w, grid_xy)
    use = W >= MIN_LOCAL_WEIGHT
    if not use.any():
        print(f"{res['recording']}: {len(t)} cells (edge excluded) - too few elongated cells for a "
              f"smoothed field (no grid point with local weight >= {MIN_LOCAL_WEIGHT}); skipped")
        return dict(cells=t, xy=xy, axis=axis, w=w, grid_xy=grid_xy, V=V, W=W, use=use, S=np.nan,
                    p=np.nan, S_null=np.array([]), S_map=None, best=(np.nan, np.nan))
    S = circle_score(V, W, use, grid_xy, onh)

    # Shuffle test: angles + weights move together to random cell positions
    rng = np.random.default_rng(RANDOM_SEED)
    z = w * np.exp(2j * np.deg2rad(axis))
    tangent = np.arctan2(grid_xy[use, 1] - onh[1], grid_xy[use, 0] - onh[0]) + np.pi / 2
    rot = np.exp(-2j * tangent)
    S_null = np.empty(N_PERMUTATIONS)
    for k in range(N_PERMUTATIONS):
        perm = rng.permutation(len(t))
        Vp, Wp = K[use] @ z[perm], K[use] @ w[perm]
        S_null[k] = np.real(Vp * rot).sum() / Wp.sum()
    p = float((1 + np.sum(S_null >= S)) / (1 + N_PERMUTATIONS))

    # Best-centre picture over the cone-image footprint
    geo = res["geo"]
    iw, ih = res["image_size"]
    corners = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], float)
    cx, cy = rca.apply(geo.A_image1_to_projected_um, corners[:, 0], corners[:, 1])
    ccx = np.arange(cx.min(), cx.max() + CENTRE_GRID_STEP_UM, CENTRE_GRID_STEP_UM)
    ccy = np.arange(cy.min(), cy.max() + CENTRE_GRID_STEP_UM, CENTRE_GRID_STEP_UM)
    CX, CY = np.meshgrid(ccx, ccy)
    on_image = MplPath(np.c_[cx, cy]).contains_points(np.c_[CX.ravel(), CY.ravel()])
    tang_all = (np.arctan2(grid_xy[use, 1][None, :] - CY.ravel()[:, None],
                           grid_xy[use, 0][None, :] - CX.ravel()[:, None]) + np.pi / 2)
    S_map = np.real(V[use][None, :] * np.exp(-2j * tang_all)).sum(1) / W[use].sum()
    S_map = np.where(on_image, S_map, np.nan).reshape(CX.shape)
    b = np.nanargmax(S_map)
    best = (CX.ravel()[b], CY.ravel()[b])

    print(f"{res['recording']}: {len(t)} cells (edge excluded), {int(use.sum())} grid points used; "
          f"circle score S={S:+.2f}, shuffle p={p:.3g} (null mean {S_null.mean():+.2f}); best centre "
          f"{np.hypot(best[0] - onh[0], best[1] - onh[1]):.0f} µm from the ONH")
    return dict(cells=t, xy=xy, axis=axis, w=w, grid_xy=grid_xy, V=V, W=W, use=use, S=S, p=p,
                S_null=S_null, S_map=S_map, ccx=ccx, ccy=ccy, best=best)


# ============================================================
# PLOTS
# ============================================================
def plot_field(ax, res, a):
    geo = res["geo"]
    onh_x, onh_y = res["onh_xy"]
    hw, hh = res["half_w"], res["half_h"]
    img = Image.open(cone_image_file(res["recording"]))
    iw, ih = img.size
    small = np.asarray(img.convert("RGB").reduce(4))
    img.close()
    ax.set_facecolor("black")
    # ORIENTATION [CONE IMAGE, fitted]: as the summary plots, no extra flip.
    ax.imshow(small, extent=[0, iw, ih, 0], origin="upper", alpha=0.55,
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData)
    ax.plot([-hw, hw, hw, -hw, -hw], [-hh, -hh, hh, hh, -hh], "w--", lw=1, zorder=3)

    # Ideal circles around the ONH (comparison only)
    far = np.hypot(np.abs(onh_x) + hw, np.abs(onh_y) + hh)
    th = np.linspace(0, 2 * np.pi, 361)
    for r in np.arange(CIRCLE_SPACING_UM, far + CIRCLE_SPACING_UM, CIRCLE_SPACING_UM):
        ax.plot(onh_x + r * np.cos(th), onh_y + r * np.sin(th), color="white", lw=0.6,
                ls=(0, (3, 4)), alpha=0.5, zorder=4)

    # Individual RF axes (thin, grey), for reference
    L = 25
    rad = np.deg2rad(a["axis"])
    ax.add_collection(LineCollection(
        np.stack([a["xy"] - L * np.c_[np.cos(rad), np.sin(rad)],
                  a["xy"] + L * np.c_[np.cos(rad), np.sin(rad)]], axis=1),
        colors="0.85", lw=0.8, alpha=0.8, zorder=5))

    # Smoothed field: direction phi, length ~ coherence, colour = angle to the circle
    use = a["use"]
    g = a["grid_xy"][use]
    V, W = a["V"][use], a["W"][use]
    phi = np.angle(V) / 2
    coh = np.abs(V) / W
    tangent = np.arctan2(g[:, 1] - onh_y, g[:, 0] - onh_x) + np.pi / 2
    diff = np.degrees(np.abs(np.angle(np.exp(2j * (phi - tangent)))) / 2)  # 0-90
    half = FIELD_LINE_MAX_UM / 2 * np.clip(coh, 0.15, 1)
    seg = np.stack([g - half[:, None] * np.c_[np.cos(phi), np.sin(phi)],
                    g + half[:, None] * np.c_[np.cos(phi), np.sin(phi)]], axis=1)
    ax.add_collection(LineCollection(
        seg, colors=CMAP(diff / 90), lw=2.4, zorder=6, capstyle="round",
        path_effects=[patheffects.Stroke(linewidth=3.8, foreground="black"), patheffects.Normal()]))
    ax.plot(onh_x, onh_y, "r*", ms=16, mec="white", zorder=8, label="optic nerve head")
    ax.legend(loc="upper right", fontsize=8)

    xs, ys = np.r_[-hw, hw, onh_x], np.r_[-hh, hh, onh_y]
    ax.set_xlim(xs.min() - 80, xs.max() + 80)
    ax.set_ylim(ys.max() + 80, ys.min() - 80)  # ORIENTATION [DISPLAY only]: y down
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 90))
    cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("smoothed field vs circle (°): 0 along circle, 90 along spoke", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("projected x (µm, + right)", fontsize=9)
    ax.set_ylabel("projected y (µm, + down)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{res['recording'].replace('_', '/')}  ·  {res['channel']}\n"
                 f"smoothed RF orientation (σ={SMOOTH_SIGMA_UM} µm, {len(a['cells'])} cells)\n"
                 "longer = cells agree more; grey = single RFs", loc="left", fontsize=10)


def plot_null(ax, a):
    if not a["use"].any():
        ax.axis("off")
        ax.text(0.5, 0.5, "too few elongated cells\nfor a smoothed field", ha="center", va="center",
                fontsize=11, transform=ax.transAxes)
        return
    ax.hist(a["S_null"], bins=40, color="#7aa6c2", edgecolor="white",
            label=f"angles shuffled across cells ({N_PERMUTATIONS}x)")
    ax.axvline(a["S"], color="#d62728", lw=2.5, label=f"real: S = {a['S']:+.2f}")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    lim = max(0.3, np.nanmax(np.abs(np.r_[a["S_null"], a["S"]])) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("circle score S\n(+1 field along circles, -1 along spokes)", fontsize=9)
    ax.set_ylabel("shuffles", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)
    ax.set_title(f"More circular than the same angles placed at random?\n"
                 f"p = {a['p']:.3g} (one-sided, S >= shuffles)", loc="left", fontsize=10)


def plot_centre_map(ax, res, a):
    if a["S_map"] is None:
        ax.axis("off")
        return
    onh_x, onh_y = res["onh_xy"]
    hw, hh = res["half_w"], res["half_h"]
    m, gx, gy = a["S_map"], a["ccx"], a["ccy"]
    lim = np.nanmax(np.abs(m))
    step = gx[1] - gx[0]
    im = ax.imshow(m, extent=[gx[0] - step / 2, gx[-1] + step / 2, gy[-1] + step / 2, gy[0] - step / 2],
                   origin="upper", cmap="PuOr_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.plot([-hw, hw, hw, -hw, -hw], [-hh, -hh, hh, hh, -hh], "k--", lw=1)
    ax.plot(onh_x, onh_y, "r*", ms=16, mec="white", zorder=7, label="optic nerve head (clicked)")
    ax.plot(*a["best"], "x", color="lime", ms=13, mew=3, zorder=7,
            path_effects=[patheffects.Stroke(linewidth=5, foreground="black"), patheffects.Normal()],
            label="best circle centre")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[-1], gy[0])  # ORIENTATION [DISPLAY only]: y down
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("circle score if the centre were here", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("projected x (µm, + right)", fontsize=9)
    ax.set_ylabel("projected y (µm, + down)", fontsize=9)
    ax.tick_params(labelsize=8)
    d = np.hypot(a["best"][0] - onh_x, a["best"][1] - onh_y)
    ax.set_title(f"Circle score for every possible centre (picture, not a test)\n"
                 f"best centre {d:.0f} µm from the clicked ONH", loc="left", fontsize=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", help="only this recording (DD_MM_YYYY)")
    parser.add_argument("--output-folder", type=Path, default=rca.OUTPUT_FOLDER)
    args = parser.parse_args()
    recordings, onh_positions = rca.recordings_with_onh(args.recording)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    n = len(recordings)
    fig, axes = plt.subplots(n, 3, figsize=(20, 6.4 * n + 1.4), facecolor="white", squeeze=False)
    for row, (rec, ch) in zip(axes, recordings):  # one recording at a time (large files)
        res = rca.analyse_recording(rec, ch, onh_positions[rec])
        with Image.open(cone_image_file(rec)) as im:
            res["image_size"] = im.size
        a = analyse(res)
        plot_field(row[0], res, a)
        plot_null(row[1], a)
        plot_centre_map(row[2], res, a)
        del res

    weight = "weighted by elongation (1 - tilt)" if WEIGHT_BY_ELONGATION else "unweighted"
    fig.suptitle(
        "Smoothed RF orientation field vs circles around the optic nerve head\n"
        f"Each RF's long axis {weight}, Gaussian-smoothed (σ = {SMOOTH_SIGMA_UM} µm). Cells: quality >= "
        f"{rca.QUALITY_MIN}, more than {rca.EDGE_EXCLUSION_UM} µm inside the stimulus edge. "
        "Dashed circles: ideal circles around the clicked ONH (comparison only).\n"
        "Circle score S: +1 field follows circles, -1 follows spokes. p: real S vs the same RF angles "
        "shuffled across cell positions (a shared orientation cannot fake it).",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / fig.get_figheight()))
    name = ("rf_orientation_field_" + recordings[0][0]) if n == 1 else "rf_orientation_field_all_recordings"
    for fmt in SAVE_FORMATS:
        out = args.output_folder / f"{name}.{fmt}"
        fig.savefig(out, dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
