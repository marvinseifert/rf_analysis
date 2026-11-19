"""
Do RF CENTRES line up along arcs around the optic nerve head (ONH)?

(rf_cone_row_angle.py tests RF ORIENTATION. This tests RF POSITIONS.)

Idea: if RF centres sit along arcs (circles) around the ONH, the line joining two
NEIGHBOURING cells runs around the ONH (along the arc), not towards it (along a spoke).

  1. Pairs: every two RF centres closer than NEIGHBOUR_MAX_UM (and further apart than
     NEIGHBOUR_MIN_UM: closer ones are probably the same cell recorded twice).
  2. For each pair: angle between the line joining the two cells and the spoke
     ONH -> pair midpoint. 0° = along the spoke, 90° = along an arc.
  3. Arc index = -mean(cos(2 x angle)): +1 = all pairs along arcs, -1 = all along
     spokes, ~0 = no preference.
  4. Test: does the pair line TURN WITH the spoke across the array? Circular correlation of
     pair axis vs spoke axis (doubled angles), as rf_cone_row_angle.rf_row_correlation,
     one-sided permutation p (pair axes shuffled across pairs; pairs share cells, so p is
     somewhat optimistic). Cells on a square electrode grid, or at random, give r ~ 0;
     cells arranged around the ONH give r > 0. Then the arc index says HOW: > 0 along
     arcs, < 0 along spokes. Arcs only show up if cells are closer together ALONG an arc
     than BETWEEN arcs (with equal spacing, arcs and spokes are the same pattern).
  5. Picture only (not a test): the arc index for every possible centre on the cone image.
     If cells form arcs around the ONH the best centre should be near the clicked ONH, but
     the electrode grid can also make points away from it score high (checked on a
     synthetic grid), so don't read it as a p-value.

Cells: as rf_cone_row_angle.py (quality >= QUALITY_MIN, RF centre inside the displayed
stimulus), minus those within EDGE_EXCLUSION_UM of the stimulus edge (cut-off RFs get
their centres pushed against the edge and would form a fake line along it).
Everything in the projected (aligned cone image, door view) frame.

Run from the repo root (one recording at a time):
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_position_arc_test.py"
Options: --recording 14_08_2026, --output-folder.

Outputs (OUTPUT_FOLDER of rf_cone_row_angle.py)
  rf_position_arcs_all_recordings.png/.pdf   one row per recording:
      RF centres + neighbour pairs on the cone image | pair-angle histogram | arc-index map
  rf_position_arcs_all_recordings_pairs.csv  one row per pair
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# Pair distance limit. Checked on synthetic data, 19/11/2026 (40 runs each, ~100 cells in a
# 1400 µm square, arcs 130 µm apart with cells ~70 µm apart along them, half the cells kept):
#   max 100 µm: arcs found 92%, but RANDOM positions "significant" 10-12% (should be 5%)
#   max 120 µm: arcs found 85%, random 0-2%, square grid 0%      <- used
#   max 150 µm: arcs found 2% (pairs ACROSS arcs swamp pairs along them), random 0-5%
# So arcs are only detectable if they are further apart than this limit and cells are
# closer together along an arc than that. (Real nearest-neighbour distances ~75-125 µm.)
NEIGHBOUR_MAX_UM = 120
NEIGHBOUR_MIN_UM = 20  # closer pairs: probably the same cell twice
EXCLUDE_EDGE_CELLS = True  # uses rca.EDGE_EXCLUSION_UM (50 µm)
CENTRE_GRID_STEP_UM = 25  # candidate-centre grid for the arc-index map
BIN_WIDTH_DEG = 10
COVARIANCE_ALPHA = 0.4  # faint covariance overlay under the RF centres
SAVE_FORMATS = ("png", "pdf")
PNG_DPI = 200
CMAP = rca.DIFFERENCE_CMAP  # 0° yellow (along spoke) -> 90° magenta (along arc)


# ============================================================
# ANALYSIS
# ============================================================
def neighbour_pairs(t, region_half_w, region_half_h):
    """Index pairs (i, j), i < j, of cells NEIGHBOUR_MIN_UM <= distance < NEIGHBOUR_MAX_UM
    whose MIDPOINT is at least NEIGHBOUR_MAX_UM / 2 inside the region the cells can be in.

    Why the midpoint rule: near a region edge, two close cells can only sit along the edge
    (both must be inside), so pairs there are biased to run parallel to the edge. Seen from
    an ONH inside the array that looks like arcs: random positions gave p < 0.05 in 20-30%
    of synthetic runs. With the midpoint at least half the maximum pair length inside,
    every pair orientation fits, so the edge can no longer bias it.
    """
    P = t[["x_projected_um", "y_projected_um"]].values
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    i, j = np.nonzero(np.triu((D >= NEIGHBOUR_MIN_UM) & (D < NEIGHBOUR_MAX_UM), k=1))
    mid = (P[i] + P[j]) / 2
    m = NEIGHBOUR_MAX_UM / 2
    ok = (np.abs(mid[:, 0]) <= region_half_w - m) & (np.abs(mid[:, 1]) <= region_half_h - m)
    return i[ok], j[ok], P


def pair_angles_vs_centre(p1, p2, cx, cy):
    """Angle (0-90°) between each pair line and the spoke from centre(s) (cx, cy) to the
    pair midpoint. cx, cy may be arrays of shape (k, 1) for many centres -> (k, m)."""
    pair_axis = np.arctan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    mx, my = (p1[:, 0] + p2[:, 0]) / 2, (p1[:, 1] + p2[:, 1]) / 2
    spoke = np.arctan2(my - cy, mx - cx)
    d = np.degrees(pair_axis - spoke) % 180.0
    return np.minimum(d, 180.0 - d)


def arc_index(angle_deg, axis=-1):
    """-mean(cos 2θ): +1 = all along arcs (90°), -1 = all along spokes (0°)."""
    return -np.mean(np.cos(np.deg2rad(2 * angle_deg)), axis=axis)


def analyse(res):
    t = res["table"]
    region_w, region_h = res["half_w"], res["half_h"]  # cells are inside the displayed stimulus
    if EXCLUDE_EDGE_CELLS:
        t = t[~t.near_edge].reset_index(drop=True)
        region_w, region_h = region_w - rca.EDGE_EXCLUSION_UM, region_h - rca.EDGE_EXCLUSION_UM
    i, j, P = neighbour_pairs(t, region_w, region_h)
    p1, p2 = P[i], P[j]
    onh_x, onh_y = res["onh_xy"]
    ang = pair_angles_vs_centre(p1, p2, onh_x, onh_y)
    ai = float(arc_index(ang)) if ang.size else np.nan
    pair_axis = np.degrees(np.arctan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])) % 180
    mid = (p1 + p2) / 2
    spoke_axis = np.degrees(np.arctan2(mid[:, 1] - onh_y, mid[:, 0] - onh_x)) % 180
    r_c, p_c, spread = rca.rf_row_correlation(pair_axis, spoke_axis)

    # Arc index for every candidate centre over the cone-image footprint
    geo = res["geo"]
    w, h = res["image_size"]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    cx, cy = rca.apply(geo.A_image1_to_projected_um, corners[:, 0], corners[:, 1])
    gx = np.arange(cx.min(), cx.max() + CENTRE_GRID_STEP_UM, CENTRE_GRID_STEP_UM)
    gy = np.arange(cy.min(), cy.max() + CENTRE_GRID_STEP_UM, CENTRE_GRID_STEP_UM)
    GX, GY = np.meshgrid(gx, gy)
    on_image = MplPath(np.c_[cx, cy]).contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    if ang.size:
        grid = arc_index(pair_angles_vs_centre(p1, p2, GX.reshape(-1, 1), GY.reshape(-1, 1))).reshape(GX.shape)
    else:
        grid = np.full(GX.shape, np.nan)
    grid = np.where(on_image, grid, np.nan)
    valid = np.isfinite(grid)
    b = np.nanargmax(grid) if valid.any() else None
    best = (GX.ravel()[b], GY.ravel()[b]) if b is not None else (np.nan, np.nan)

    pairs = pd.DataFrame(dict(
        recording=res["recording"], channel=res["channel"],
        cell_a=t.cell_index.values[i], cell_b=t.cell_index.values[j],
        distance_um=np.hypot(*(p2 - p1).T),
        mid_x_projected_um=(p1[:, 0] + p2[:, 0]) / 2, mid_y_projected_um=(p1[:, 1] + p2[:, 1]) / 2,
        distance_mid_to_onh_um=np.hypot((p1[:, 0] + p2[:, 0]) / 2 - onh_x, (p1[:, 1] + p2[:, 1]) / 2 - onh_y),
        pair_vs_spoke_deg=ang,
    ))
    print(f"{res['recording']}: {len(t)} cells, {len(pairs)} neighbour pairs; arc index {ai:+.2f}; "
          f"turns with spoke r={r_c:+.2f} p={p_c:.2g} (spoke spread {spread:.0f}°); best centre "
          f"{np.hypot(best[0] - onh_x, best[1] - onh_y):.0f} µm from the ONH")
    return dict(cells=t, pairs=pairs, p1=p1, p2=p2, arc_index=ai, r=r_c, p=p_c, spread=spread, best=best,
                grid=grid, gx=gx, gy=gy)


# ============================================================
# PLOTS
# ============================================================
def plot_positions(ax, res, a):
    geo = res["geo"]
    onh_x, onh_y = res["onh_xy"]
    hw, hh = res["half_w"], res["half_h"]
    img = Image.open(cone_image_file(res["recording"]))
    w, h = img.size
    small = np.asarray(img.convert("RGB").reduce(4))
    img.close()
    ax.set_facecolor("black")
    # ORIENTATION [CONE IMAGE, fitted]: as the summary plots, no extra flip.
    ax.imshow(small, extent=[0, w, h, 0], origin="upper",
              transform=Affine2D(geo.A_image1_to_projected_um) + ax.transData)

    # Faint population covariance overlay, as rf_cone_row_angle.py / the summary plots
    A = geo.A_raw_rf_to_projected_um
    x, y = res["x_um"] * geo.rf_scale_factor, res["y_um"] * geo.rf_scale_factor
    step = x[1] - x[0]
    X, Y = np.meshgrid(x, y)
    px, py = rca.apply(A, X, Y)
    rf_img = np.where((np.abs(px) <= hw) & (np.abs(py) <= hh), res["cov_map"], np.nan)
    finite = rf_img[np.isfinite(rf_img)]
    vmax = float(np.nanpercentile(np.abs(finite), 99.5)) if finite.size else 1.0
    # ORIENTATION [RF ROTATION, date-dependent]: stimulus -> projected, as the summary plots.
    ax.imshow(rf_img, extent=[x[0] - step / 2, x[-1] + step / 2, y[-1] + step / 2, y[0] - step / 2],
              origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=COVARIANCE_ALPHA,
              interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=3)

    box = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
    ax.plot(box[:, 0], box[:, 1], "w--", lw=1, zorder=4)
    e = rca.EDGE_EXCLUSION_UM
    ax.plot([-hw + e, hw - e, hw - e, -hw + e, -hw + e], [-hh + e, -hh + e, hh - e, hh - e, -hh + e],
            color="0.75", ls=":", lw=1.2, zorder=4)

    # Neighbour pairs coloured by their angle to the spoke, then the RF centres
    ang = a["pairs"].pair_vs_spoke_deg.values
    lc = LineCollection(np.stack([a["p1"], a["p2"]], axis=1), colors=CMAP(ang / 90), lw=2.2, zorder=6,
                        path_effects=[patheffects.Stroke(linewidth=3.6, foreground="black"),
                                      patheffects.Normal()])
    ax.add_collection(lc)
    ax.scatter(a["cells"].x_projected_um, a["cells"].y_projected_um, s=14, c="white",
               edgecolors="black", linewidths=0.6, zorder=7)
    ax.plot(onh_x, onh_y, "r*", ms=16, mec="white", zorder=8, label="optic nerve head")
    ax.legend(loc="upper right", fontsize=8)

    xs, ys = np.r_[-hw, hw, onh_x], np.r_[-hh, hh, onh_y]
    ax.set_xlim(xs.min() - 80, xs.max() + 80)
    ax.set_ylim(ys.max() + 80, ys.min() - 80)  # ORIENTATION [DISPLAY only]: y down
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 90))
    cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("pair line vs spoke (°): 0 spoke, 90 arc", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("projected x (µm, + right)", fontsize=9)
    ax.set_ylabel("projected y (µm, + down)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{res['recording'].replace('_', '/')}  ·  {res['channel']}\n"
                 f"RF centres (white, n={len(a['cells'])}) and neighbour pairs "
                 f"({NEIGHBOUR_MIN_UM}-{NEIGHBOUR_MAX_UM} µm, n={len(a['pairs'])})",
                 loc="left", fontsize=10)


def plot_pair_histogram(ax, a):
    ang = a["pairs"].pair_vs_spoke_deg.values
    bins = np.arange(0, 90 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    ax.hist(ang, bins=bins, color="#7aa6c2", edgecolor="white", label=f"neighbour pairs (n={len(ang)})")
    ax.axhline(len(ang) * BIN_WIDTH_DEG / 90, color="k", ls="--", lw=1, label="uniform (no arcs)")
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel("angle between pair line and spoke (°)\n0 = along the spoke, 90 = along an arc",
                  fontsize=9)
    ax.set_ylabel("pairs", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)
    med = np.median(ang) if ang.size else np.nan
    ax.set_title(f"pair line turns with spoke: r={a['r']:+.2f}, p={a['p']:.2g} "
                 f"(spoke spread {a['spread']:.0f}°)\n"
                 f"median {med:.0f}°, arc index {a['arc_index']:+.2f} "
                 "(+ arcs, - spokes, 0 neither)", loc="left", fontsize=10)


def plot_centre_map(ax, res, a):
    onh_x, onh_y = res["onh_xy"]
    hw, hh = res["half_w"], res["half_h"]
    g, gx, gy = a["grid"], a["gx"], a["gy"]
    lim = np.nanmax(np.abs(g)) if np.isfinite(g).any() else 1.0
    step = gx[1] - gx[0]
    im = ax.imshow(g, extent=[gx[0] - step / 2, gx[-1] + step / 2, gy[-1] + step / 2, gy[0] - step / 2],
                   origin="upper", cmap="PuOr_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.plot([-hw, hw, hw, -hw, -hw], [-hh, -hh, hh, hh, -hh], "k--", lw=1)
    ax.scatter(a["cells"].x_projected_um, a["cells"].y_projected_um, s=4, c="k", zorder=5)
    ax.plot(onh_x, onh_y, "r*", ms=16, mec="white", zorder=7, label="optic nerve head (clicked)")
    ax.plot(*a["best"], "x", color="lime", ms=13, mew=3, zorder=7,
            path_effects=[patheffects.Stroke(linewidth=5, foreground="black"), patheffects.Normal()],
            label="best arc centre")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[-1], gy[0])  # ORIENTATION [DISPLAY only]: y down
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("arc index if the centre were here", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("projected x (µm, + right)", fontsize=9)
    ax.set_ylabel("projected y (µm, + down)", fontsize=9)
    ax.tick_params(labelsize=8)
    d_best = np.hypot(a["best"][0] - onh_x, a["best"][1] - onh_y)
    ax.set_title(f"Arc index for every possible centre (picture, not a test)\n"
                 f"best centre {d_best:.0f} µm from the clicked ONH",
                 loc="left", fontsize=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", help="only this recording (DD_MM_YYYY)")
    parser.add_argument("--output-folder", type=Path, default=rca.OUTPUT_FOLDER)
    args = parser.parse_args()
    recordings, onh_positions = rca.recordings_with_onh(args.recording)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    n = len(recordings)
    fig, axes = plt.subplots(n, 3, figsize=(20, 6.4 * n + 1.4), facecolor="white", squeeze=False)
    all_pairs = []
    for row, (rec, ch) in zip(axes, recordings):  # one recording at a time (large files)
        res = rca.analyse_recording(rec, ch, onh_positions[rec])
        with Image.open(cone_image_file(rec)) as im:
            res["image_size"] = im.size
        a = analyse(res)
        plot_positions(row[0], res, a)
        plot_pair_histogram(row[1], a)
        plot_centre_map(row[2], res, a)
        all_pairs.append(a["pairs"])
        del res  # free the covariance map before the next recording

    fig.suptitle(
        "Do RF centres line up along arcs around the optic nerve head?\n"
        f"Neighbour pairs ({NEIGHBOUR_MIN_UM}-{NEIGHBOUR_MAX_UM} µm apart): angle between the line joining "
        "the two cells and the spoke from the ONH to their midpoint (90° = along an arc). "
        f"Cells: quality >= {rca.QUALITY_MIN}, more than {rca.EDGE_EXCLUSION_UM} µm inside the stimulus edge.\n"
        "Test: does the pair line turn with the spoke across the array (circular correlation r > 0, "
        "permutation p)? A square electrode grid or random positions give r ~ 0. Arc index: + arcs, - spokes.\n"
        "Right: arc index for every possible centre on the cone image (picture only: the electrode grid "
        "can also make other points score high).",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / fig.get_figheight()))
    name = ("rf_position_arcs_" + recordings[0][0]) if n == 1 else "rf_position_arcs_all_recordings"
    for fmt in SAVE_FORMATS:
        out = args.output_folder / f"{name}.{fmt}"
        fig.savefig(out, dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)
    csv = args.output_folder / f"{name}_pairs.csv"
    pd.concat(all_pairs, ignore_index=True).to_csv(csv, index=False)
    print("saved", csv)


if __name__ == "__main__":
    main()
