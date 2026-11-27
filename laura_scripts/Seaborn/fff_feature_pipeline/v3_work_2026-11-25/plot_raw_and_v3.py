"""Left: raw cone image. Right: v3 lattice regularity map (200 µm window every 50 µm).
No marked regions and no RFs; ONH (+) and noise stimulus outline (dashed) only."""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator
REPO = Path("/home/laura/PycharmProjects/rf_analysis")
sys.path[:0] = [str(REPO), str(REPO / "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction")]
import mark_mosaic_regions as m
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file
rec, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(npz); xs, ys, res = z["xs"], z["ys"], z["res3"]
with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(alignment_cache_v2_file(rec))
Ainv = np.linalg.inv(geo.A_image1_to_projected_um)
st = xs[1] - xs[0]
EX, EY = np.meshgrid(np.r_[xs - st / 2, xs[-1] + st / 2], np.r_[ys - st / 2, ys[-1] + st / 2])
P = Ainv @ np.vstack([EX.ravel(), EY.ravel(), np.ones(EX.size)])
PX, PY = (P[0] / P[2]).reshape(EX.shape), (P[1] / P[2]).reshape(EX.shape)
img = Image.open(cone_image_file(rec)); W, H = img.size; small = np.asarray(img.convert("RGB").reduce(5))
onh = json.load(open(m.ONH_FILE)).get(rec); onh = None if onh is None else (onh["x_px"], onh["y_px"])
outline = m._stimulus_outline_px(rec); upp = m._cone_um_per_px(rec)
bounds = [-0.001, 0.001, .1, .2, .3, .4, .5, .6]
cols = ["#000000", "#d7301f", "#fc8d59", "#fdcc8a", "#91bfdb", "#4575b4", "#253494"]
cmap, norm = ListedColormap(cols), BoundaryNorm(bounds, len(cols))
fig, axs = plt.subplots(1, 2, figsize=(22, 11))
for k, ax in enumerate(axs):
    ax.imshow(small, extent=[0, W, H, 0], origin="upper", interpolation="antialiased")
    if k == 1:
        ax.imshow(np.zeros((1, 1, 3)), extent=[0, W, H, 0], alpha=0.35)  # darken so colours read
        mesh = ax.pcolormesh(PX, PY, np.ma.masked_invalid(res), cmap=cmap, norm=norm, alpha=0.6, shading="flat")
    if outline is not None: ax.plot(outline[:, 0], outline[:, 1], "--", color="yellow", lw=1.3)
    if onh: ax.plot(*onh, "c+", ms=22, mew=2.5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    sp = 200 / upp; ox, oy = onh if onh else (0, 0)
    for axis, o in ((ax.xaxis, ox), (ax.yaxis, oy)):
        axis.set_major_locator(MultipleLocator(sp, offset=o % sp)); axis.set_major_formatter(FuncFormatter(lambda v, _, o=o: f"{(v - o) * upp:.0f}"))
    ax.tick_params(labelsize=8); ax.set_xlabel("µm from ONH" if onh else "µm"); ax.grid(color="white", alpha=0.15, lw=0.6)
axs[0].set_title(f"{rec}: cone image (+ = ONH, dashed = noise stimulus)", fontsize=13)
axs[1].set_title("lattice regularity v3 (200 µm window every 50 µm; blank = window off the image)", fontsize=13)
cb = fig.colorbar(mesh, ax=axs[1], fraction=0.035, pad=0.02, ticks=[0, .05, .15, .25, .35, .45, .55])
cb.ax.set_yticklabels(["0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "≥0.5"])
fig.tight_layout(); fig.savefig(out, dpi=75)
v = res[np.isfinite(res)]
print(rec, "grid points", v.size, "| % <0.1:", round(100 * (v < .1).mean()), " 0.1-0.2:", round(100 * ((v >= .1) & (v < .2)).mean()), " >=0.2:", round(100 * (v >= .2).mean()))
