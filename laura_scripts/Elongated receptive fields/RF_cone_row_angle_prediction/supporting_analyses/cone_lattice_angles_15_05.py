"""
15/05/2026: angles of the labelled-cone lattice relative to the traced cone rows, to compare with the RF-vs-row
angle peak (folded 60-75°; signed 105-120° with ONH-radial rows). (23/11/2026)

1. Labelled cones detected on 15_05_2026_p0_cones_flipped.jpg (green channel; background removed with a 6 µm
   Gaussian, smoothed 1 µm, local maxima >= --min-sep-um apart and above the --threshold-pct percentile).
2. Only cones within 15 µm of a traced row point (15_05_2026_cones_rows.csv); local row direction = principal axis
   of the traced row points within 25 µm.
3. Neighbour vectors: every pair of cones closer than --max-nn x the median nearest-neighbour distance, expressed in
   the cone's local row frame (rows horizontal), in the projected µm frame used for the RF angles
   (A_image1_to_projected_um), so angles compare directly with rf_axis_projected_deg - row.
   Signed angle = (vector angle - row angle) mod 180: 0/180 = along the row, 90 = across (same convention as the RF
   signed angles); folded angle = min(signed, 180 - signed).

Run from the repo root (~3 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/cone_lattice_angles_15_05.py"
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file

parser = argparse.ArgumentParser()
parser.add_argument("--min-sep-um", type=float, default=5.0)
parser.add_argument("--threshold-pct", type=float, default=80.0)
parser.add_argument("--max-nn", type=float, default=1.8)
args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = None
REC = "15_05_2026"
IMG_DIR = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Experiments/Imaging/20260515_dragonfly_microscope")
FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(alignment_cache_v2_file(REC))
Ai = geo.A_image1_to_projected_um
um_px = float(np.sqrt(abs(np.linalg.det(Ai[:2, :2]))))

img = np.asarray(Image.open(cone_image_file(REC)))[..., 1].astype(np.float32)
bg = gaussian_filter(img, 6 / um_px)
sm = gaussian_filter(img - bg, 1 / um_px)
thr = np.percentile(sm, args.threshold_pct)
pk = peak_local_max(sm, min_distance=int(args.min_sep_um / um_px), threshold_abs=thr)
cones_px = pk[:, ::-1].astype(float)  # (x, y)
C = np.c_[cones_px, np.ones(len(cones_px))] @ Ai.T
cones = C[:, :2]
print(f"{len(cones)} cone candidates detected ({um_px:.3f} µm/px)")

rows = pd.read_csv(IMG_DIR / "15_05_2026_cones_rows.csv")
R = np.c_[rows.x_pixels, rows.y_pixels, np.ones(len(rows))] @ Ai.T
rpts = R[:, :2]
rtree = cTree = cKDTree(rpts)
d_row, _ = rtree.query(cones)
keep = d_row <= 15
cones = cones[keep]
print(f"{len(cones)} cones within 15 µm of a traced row")
row_dir = np.full(len(cones), np.nan)
for i, p in enumerate(cones):
    idx = rtree.query_ball_point(p, 25)
    if len(idx) < 10:
        continue
    X = rpts[idx] - rpts[idx].mean(0)
    w, v = np.linalg.eigh(X.T @ X)
    row_dir[i] = np.degrees(np.arctan2(v[1, -1], v[0, -1])) % 180
ok = np.isfinite(row_dir)
cones, row_dir = cones[ok], row_dir[ok]
tree = cKDTree(cones)
dnn, _ = tree.query(cones, 2)
nn = float(np.median(dnn[:, 1]))
pairs = np.array(sorted(tree.query_pairs(args.max_nn * nn)))
vec = cones[pairs[:, 1]] - cones[pairs[:, 0]]
dist = np.hypot(*vec.T)
ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 180
rd = row_dir[pairs[:, 0]]
signed = (ang - rd) % 180
folded = np.minimum(signed, 180 - signed)
# vectors in the row frame (rows horizontal), both directions
th = np.deg2rad(rd)
rx = vec[:, 0] * np.cos(th) + vec[:, 1] * np.sin(th)
ry = -vec[:, 0] * np.sin(th) + vec[:, 1] * np.cos(th)
rx, ry = np.r_[rx, -rx], np.r_[ry, -ry]
print(f"median nearest-neighbour distance {nn:.1f} µm; {len(pairs)} neighbour pairs (< {args.max_nn} x NN)")

bins = np.arange(0, 181, 5)
h, _ = np.histogram(signed, bins)
centres = bins[:-1] + 2.5
top = np.argsort(h)[::-1][:6]
print("most common signed neighbour angles (°, 5° bins):", [(int(centres[i]), int(h[i])) for i in sorted(top)])
for lo_, hi_ in [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 105), (105, 120), (120, 135),
                 (135, 150), (150, 165), (165, 180)]:
    sel = (signed >= lo_) & (signed < hi_)
    print(f"  {lo_:3d}-{hi_:3d}°: {sel.mean() * 100:5.1f} % of pairs, median distance "
          f"{np.median(dist[sel]) if sel.any() else np.nan:5.1f} µm")


def peak_angle(lo_, hi_):
    sel = (signed >= lo_) & (signed < hi_)
    hh, ee = np.histogram(signed[sel], np.arange(lo_, hi_ + 1, 2))
    return float(ee[np.argmax(hh)] + 1), float(np.median(signed[sel])), float(np.median(dist[sel]))


diag1, diag2 = peak_angle(30, 85), peak_angle(95, 150)
print(f"diagonal neighbours: peak {diag1[0]:.0f}° (median {diag1[1]:.0f}°, {diag1[2]:.1f} µm) and "
      f"{diag2[0]:.0f}° (median {diag2[1]:.0f}°, {diag2[2]:.1f} µm); folded {180 - diag2[0]:.0f}° and {diag1[0]:.0f}°")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(20, 6.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1.4, 1], wspace=0.3)
ax = fig.add_subplot(gs[0])
cx, cy = np.median(cones, 0)
crop = (np.abs(cones[:, 0] - cx) < 60) & (np.abs(cones[:, 1] - cy) < 60)
inv = np.linalg.inv(Ai)
corner = (inv @ np.r_[cx - 60, cy - 60, 1])[:2].astype(int)
size = int(120 / um_px)
sub = img[corner[1]:corner[1] + size, corner[0]:corner[0] + size]
ext = [cx - 60, cx - 60 + size * um_px, cy - 60 + size * um_px, cy - 60]
ax.imshow(sub, cmap="gray", extent=ext)
ax.plot(cones[crop, 0], cones[crop, 1], "o", mfc="none", mec="cyan", ms=6)
rsel = (np.abs(rpts[:, 0] - cx) < 60) & (np.abs(rpts[:, 1] - cy) < 60)
ax.plot(rpts[rsel, 0], rpts[rsel, 1], ",", color="#ff6a2b", alpha=0.5)
ax.set_xlim(cx - 60, cx + 60); ax.set_ylim(cy + 60, cy - 60)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("120 µm crop: detected labelled cones (cyan)\nand traced rows (orange)", loc="left", fontsize=10)

ax = fig.add_subplot(gs[1])
ax.hist2d(rx, ry, bins=80, range=[[-2.2 * nn, 2.2 * nn]] * 2, cmap="magma")
ax.axhline(0, color="#ff6a2b", lw=1, alpha=0.7)
ax.set_aspect("equal")
ax.set_xlabel("along the local row (µm)"); ax.set_ylabel("across (µm)")
ax.set_title("Neighbour positions in the row frame\n(each cone at the centre, row horizontal)", loc="left", fontsize=10)

ax = fig.add_subplot(gs[2])
ax.hist(signed, bins=bins, color="0.55", edgecolor="white", label="cone neighbour directions")
ax.axvspan(105, 120, color="#1e6fd9", alpha=0.18, label="RF excess, signed vs ONH-radial rows (105-120°)")
ax.axvspan(60, 75, color="#3bb273", alpha=0.18, label="mirror bin (60-75°)")
for d in (diag1[0], diag2[0]):
    ax.axvline(d, color="k", ls="--", lw=1)
ax.set_xticks(np.arange(0, 181, 15)); ax.set_xlim(0, 180)
ax.set_xlabel("signed angle of neighbour direction vs local row (°)\n0/180 = along the row, 90 = across")
ax.set_ylabel("neighbour pairs")
ax.legend(fontsize=8, frameon=False, loc="upper center")
ax.set_title(f"Lattice directions: diagonal neighbours at {diag1[0]:.0f}° and {diag2[0]:.0f}° "
             f"(folded {diag1[0]:.0f}° / {180 - diag2[0]:.0f}°)", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[3])
ax.scatter(folded, dist, s=2, alpha=0.2, color="0.3")
ax.axvspan(60, 75, color="#3bb273", alpha=0.18, label="RF excess, folded (60-75°)")
ax.set_xlabel("folded neighbour angle vs row (°)"); ax.set_ylabel("distance (µm)")
ax.set_xlim(0, 90)
ax.legend(fontsize=8, frameon=False)
ax.set_title(f"Distance vs angle (median NN {nn:.1f} µm)", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"15/05/2026 labelled-cone lattice relative to the traced rows ({len(cones)} cones, {len(pairs)} "
             "neighbour pairs; projected frame, same as the RF angles)", fontsize=12, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"cone_lattice_angles_15_05_2026.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "cone_lattice_angles_15_05_2026.png")
