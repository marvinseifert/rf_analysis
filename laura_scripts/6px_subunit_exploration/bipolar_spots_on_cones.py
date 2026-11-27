"""
The bipolar-scale (15-50 µm band) spots of selected cells on the aligned cone image, and are they
cone-sized or bipolar-sized? (23/11/2026)

For each cell (default: the 6 shown in bipolar_scale_split_half_14_08_2026_6px.png):
  * RF map (frame STA collapsed with the cell's time course, 100 x 100 file px window), its two independent
    sequence halves, and band(map) = Gaussian 3 µm - Gaussian 8 µm (as bipolar_scale_split_half.py).
  * Reproducible spots: local maxima of band(half A) and band(half B) inside the RF (>= 30 % of that map's
    band maximum, >= 8 µm apart) that have a partner in the other half within --match-um.
  * Spot size: FWHM of a 2D Gaussian fitted to band(full map) around each reproducible spot.
  * Reference widths through the SAME pipeline (ideal 6 px blur = shuffled checker autocorrelation, then the
    band filter): a single cone (bump sigma 1.5 µm), two cones 6.7 / 12 / 20 µm apart, and bipolar-sized
    Gaussian subunits (FWHM 15 / 25 µm). A spot as wide as the single-cone reference is unresolved:
    it could be one cone or a small bipolar cell. Only spots clearly wider than that are extended.
  * Cone image: v2 alignment (alignment_geometry.load_alignment_geometry), RF and band contours drawn in the
    projected frame on the cone image. Local cone spacing from the cone image's autocorrelation.

Map coordinates: window pixel -> file px -> raw RF µm = 2 x (file px - 400) (checked: cell 282's stored
covariance-map peak), x rf_scale_factor, then A_raw_rf_to_projected_um (as plot_cells_with_stas.py).

Run from the repo root (~5 min):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/bipolar_spots_on_cones.py
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from polarspike import Overview
from scipy.ndimage import convolve, gaussian_filter, map_coordinates
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from loading.noise_data import load_noise_data
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file

parser = argparse.ArgumentParser()
parser.add_argument("--cells", type=int, nargs="+", default=[282, 224, 198, 243, 218, 256])
parser.add_argument("--match-um", type=float, default=6.0)
parser.add_argument("--zoom-um", type=float, default=70.0)
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
STIM_FILE = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5")
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/6px_subunit_exploration")
RES = OUT / "cone_bump_power_and_split_half_14_08_2026_6px.json"
UM, SEQ, N_REP, STIM_INDEX, CUT = 1.9476, 27000, 3, 10, 50
W, HALF = 2 * CUT, 13500
LO, HI = 3.0, 8.0
res = json.loads(RES.read_text())
L = res["args"]["n_lags"]
real = {int(c): v for c, v in res["real"].items()}

with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(alignment_cache_v2_file("14_08_2026"))
A = geo.A_raw_rf_to_projected_um
cones = iio.imread(cone_image_file("14_08_2026"))
cones = cones[..., :3] if cones.ndim == 3 and cones.shape[2] == 4 else cones
cones_grey = cones.mean(axis=2) if cones.ndim == 3 else cones.astype(float)
Ai = geo.A_image1_to_projected_um
um_per_img_px = float(np.sqrt(abs(np.linalg.det(Ai[:2, :2]))))
print(f"cone image {cones_grey.shape}, {um_per_img_px:.3f} µm per image px")

ds = load_noise_data(PROJECT / "noise_analysis", lazy=True)
CHANNEL = "6px_15Hz_30mins_shuffle_x6"
gx_raw = ds.x.values * geo.rf_scale_factor
gy_raw = ds.y.values * geo.rf_scale_factor
step_raw = float(gx_raw[1] - gx_raw[0])
rec = Overview.Recording.load(PROJECT / "overview")
rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @STIM_INDEX")
triggers = np.asarray(rec.dataframes["stimulus_df"].query("stimulus_index == @STIM_INDEX")
                      ["trigger_fr_relative"].to_numpy()[0])
f = h5py.File(STIM_FILE, "r")
noise = f["Noise"]
H, Wst = noise.shape[1:]
fi = np.arange(SEQ)
IN_A, IN_B = (fi >= L) & (fi < HALF), fi >= HALF + L


def band(m):
    return gaussian_filter(m, LO / UM) - gaussian_filter(m, HI / UM)


def fwhm_um(img, py, px, rad_um=10.0):
    r = int(round(rad_um / UM))
    y0, y1, x0, x1 = max(py - r, 0), min(py + r + 1, img.shape[0]), max(px - r, 0), min(px + r + 1, img.shape[1])
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]

    def gauss(xy, a, cx, cy, s, b):
        return a * np.exp(-((xy[0] - cx) ** 2 + (xy[1] - cy) ** 2) / (2 * s ** 2)) + b
    try:
        p, _ = curve_fit(gauss, (xx.ravel(), yy.ravel()), sub.ravel(), p0=[sub.max(), px, py, 3, 0], maxfev=4000)
        return float(abs(p[3]) * 2.355 * UM)
    except RuntimeError:
        return np.nan


# ------------------------------------------------------------------ reference widths through the same pipeline
_t = 1 - np.abs(np.arange(-5, 6)) / 6
TRI = np.outer(_t, _t)
gy, gx = np.mgrid[:W, :W].astype(float)
c0 = W / 2


def ref_map(points_um, sigma_um):
    rf = sum(np.exp(-((gx - c0 - x / UM) ** 2 + (gy - c0 - y / UM) ** 2) / (2 * (sigma_um / UM) ** 2))
             for x, y in points_um)
    return band(convolve(rf, TRI, mode="constant"))


refs = {}
b = ref_map([(0, 0)], 1.5)
refs["single cone"] = fwhm_um(b, W // 2, W // 2)
for fw in (15, 25):
    b = ref_map([(0, 0)], fw / 2.355)
    refs[f"bipolar subunit, FWHM {fw} µm"] = fwhm_um(b, W // 2, W // 2)
sep = {}
for d in (6.7, 12.0, 20.0, 25.0):
    b = ref_map([(-d / 2, 0), (d / 2, 0)], 1.5)
    pk = peak_local_max(b, min_distance=1, threshold_rel=0.3)
    sep[d] = int(len(pk))
print("reference spot FWHM (µm):", {k: round(v, 1) for k, v in refs.items()})
print("two cones d µm apart -> number of separate spots after 6px blur + band:", sep)

# ------------------------------------------------------------------ cells
results = {}
fig, axs = plt.subplots(len(args.cells), 5, figsize=(22.5, 4.4 * len(args.cells)))
for r_i, cell in enumerate(args.cells):
    v = real[cell]
    cx, cy = v["centre"]
    oy, ox = int(cy) - CUT, int(cx) - CUT
    y0, y1, x0, x1 = max(oy, 0), min(oy + W, H), max(ox, 0), min(ox + W, Wst)
    S = np.zeros((SEQ, W, W), np.float32)
    S[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox] = noise[:, y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    S = S.reshape(SEQ, -1)
    valid = np.zeros((W, W), bool); valid[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = True
    sp = rec.get_spikes_triggered([{"stimulus_index": STIM_INDEX, "cell_index": cell}], time="frames", pandas=False)
    t = np.sort(sp["times_relative"].to_numpy().astype(np.int64).ravel())
    fr = np.searchsorted(triggers, t, side="right") - 1
    fr = fr[(fr >= 0) & (fr < SEQ * N_REP)]
    c = np.bincount(fr % SEQ, minlength=SEQ).astype(float)
    k = np.array(v["k"])

    def rf_map(cc):
        kc = sum(k[lag] * np.roll(cc, -lag) for lag in range(L))
        return ((kc @ S) / cc.sum()).reshape(W, W)

    m, ma, mb = rf_map(c), rf_map(c * IN_A), rf_map(c * IN_B)
    sm = gaussian_filter(m, 3 / UM)
    mask = (sm >= 0.2 * sm[valid].max()) & valid
    bf, ba, bb = band(m), band(ma), band(mb)

    def peaks(bm):
        pk = peak_local_max(np.where(mask, bm, -np.inf), min_distance=int(8 / UM / 2) + 1,
                            threshold_abs=0.3 * bm[mask].max(), exclude_border=False)
        return pk
    pa, pb = peaks(ba), peaks(bb)
    matched = []
    for p in pa:
        if len(pb) and np.min(np.hypot(*(pb - p).T)) * UM <= args.match_um:
            q_ = pb[np.argmin(np.hypot(*(pb - p).T))]
            matched.append((p + q_) / 2)
    matched = np.array(matched)
    widths = [fwhm_um(bf, int(round(p[0])), int(round(p[1]))) for p in matched]
    nn = []
    if len(matched) > 1:
        D = np.hypot(*(matched[:, None, :] - matched[None, :, :]).transpose(2, 0, 1)) * UM
        np.fill_diagonal(D, np.inf)
        nn = D.min(1).tolist()

    # window px -> projected µm
    def to_proj(row, col):
        fx, fy = ox + col, oy + row
        rx, ry = 2.0 * (fx - 400) * geo.rf_scale_factor, 2.0 * (fy - 400) * geo.rf_scale_factor
        return A[0, 0] * rx + A[0, 1] * ry + A[0, 2], A[1, 0] * rx + A[1, 1] * ry + A[1, 2]
    RR, CC = np.mgrid[:W, :W]
    PX, PY = to_proj(RR, CC)
    pk_full = np.unravel_index(np.argmax(np.where(valid, sm, -np.inf)), sm.shape)
    centre = np.array(to_proj(*pk_full))
    spots_proj = np.array([to_proj(p[0], p[1]) for p in matched]) if len(matched) else np.zeros((0, 2))

    # local cone spacing from the cone image autocorrelation (crop around the RF centre)
    inv = np.linalg.inv(Ai)
    ic = inv @ np.r_[centre, 1]
    half = int(40 / um_per_img_px)
    crop = cones_grey[int(ic[1]) - half:int(ic[1]) + half, int(ic[0]) - half:int(ic[0]) + half].astype(float)
    spacing = np.nan
    if crop.size and min(crop.shape) > 10:
        crop = crop - gaussian_filter(crop, 8 / um_per_img_px)
        F = np.fft.fft2(crop - crop.mean())
        ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
        yy, xx = np.indices(ac.shape)
        rr = np.hypot(yy - ac.shape[0] // 2, xx - ac.shape[1] // 2) * um_per_img_px
        prof = np.array([ac[(rr >= a) & (rr < a + 0.5)].mean() for a in np.arange(0, 25, 0.5)])
        rs = np.arange(0, 25, 0.5) + 0.25
        loc = [i for i in range(2, len(prof) - 1) if rs[i] > 4 and prof[i] > prof[i - 1] and prof[i] >= prof[i + 1]]
        spacing = float(rs[loc[0]]) if loc else np.nan

    results[cell] = dict(n_spots_A=len(pa), n_spots_B=len(pb), n_reproducible=len(matched),
                         spot_fwhm_um=[round(w, 1) for w in widths], spot_nn_um=[round(x, 1) for x in nn],
                         local_cone_spacing_um=spacing)
    print(f"cell {cell}: spots A {len(pa)}, B {len(pb)}, reproducible {len(matched)}; FWHM "
          f"{[round(w, 1) for w in widths]} µm; nearest-neighbour {[round(x, 1) for x in nn]} µm; "
          f"local cone spacing (image autocorr.) {spacing:.1f} µm")

    # ---- panels
    z = args.zoom_um
    for j in range(5):
        ax = axs[r_i, j]
        ax.imshow(cones, extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
                  transform=Affine2D(Ai) + ax.transData, zorder=1)
        zz = z / 2 if j == 3 else z
        ax.set_xlim(centre[0] - zz, centre[0] + zz)
        ax.set_ylim(centre[1] + zz, centre[1] - zz)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    lvl = lambda img: [0.3 * np.nanmax(np.where(mask, img, np.nan)), 0.6 * np.nanmax(np.where(mask, img, np.nan))]
    # covariance map (cm_most_important) of this cell, cropped around the RF, in the projected frame
    ix0, iy0 = int(np.searchsorted(gx_raw, 2.0 * (ox - 400) * geo.rf_scale_factor)), \
        int(np.searchsorted(gy_raw, 2.0 * (oy - 400) * geo.rf_scale_factor))
    cm = ds.cm_most_important.sel(channel=CHANNEL, cell_index=cell).isel(
        y=slice(max(iy0, 0), iy0 + W), x=slice(max(ix0, 0), ix0 + W)).transpose("y", "x").values.astype(float)
    xs_, ys_ = gx_raw[max(ix0, 0):ix0 + W], gy_raw[max(iy0, 0):iy0 + W]
    cm = cm / np.nanmax(np.abs(cm))
    ext_cm = [xs_[0] - step_raw / 2, xs_[-1] + step_raw / 2, ys_[-1] + step_raw / 2, ys_[0] - step_raw / 2]
    ax = axs[r_i, 0]
    ax.imshow(np.ma.masked_invalid(cm), extent=ext_cm, origin="upper", cmap="coolwarm", vmin=-1, vmax=1,
              alpha=0.6, interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=5)
    ax.set_title(f"cell {cell} (q {v['quality']:.0f}): covariance RF (cm_most_important),\n"
                 "normalised to its peak, 60 % opaque", fontsize=8, loc="left")
    ax.set_ylabel(f"cell {cell}")
    # its 15-50 µm band, for column 5
    cmb = gaussian_filter(np.nan_to_num(cm), LO / step_raw) - gaussian_filter(np.nan_to_num(cm), HI / step_raw)
    CXr, CYr = np.meshgrid(xs_, ys_)
    CPX, CPY = A[0, 0] * CXr + A[0, 1] * CYr + A[0, 2], A[1, 0] * CXr + A[1, 1] * CYr + A[1, 2]
    cms = gaussian_filter(np.nan_to_num(cm), 3 / step_raw)
    cmask = cms >= 0.2 * cms.max()
    ax = axs[r_i, 1]
    ax.contour(PX, PY, np.where(mask, ba, np.nan), levels=lvl(ba), colors="cyan", linewidths=0.9, zorder=5)
    ax.contour(PX, PY, np.where(mask, bb, np.nan), levels=lvl(bb), colors="magenta", linewidths=0.9, zorder=5)
    ax.set_title("15-50 µm band: half A (cyan), half B (magenta)", fontsize=9, loc="left")
    ax = axs[r_i, 2]
    ax.contour(PX, PY, np.where(mask, bf, np.nan), levels=lvl(bf), colors="yellow", linewidths=1, zorder=5)
    for (sx, sy), w in zip(spots_proj, widths):
        ax.add_patch(plt.Circle((sx, sy), w / 2, fill=False, ec="red", lw=1.4, zorder=6))
    ax.set_title(f"reproducible spots (red circle = FWHM): {len(matched)}\n"
                 f"FWHM {', '.join(f'{w:.0f}' for w in widths)} µm (single cone would be "
                 f"{refs['single cone']:.0f} µm)", fontsize=8, loc="left")
    ax = axs[r_i, 3]
    for (sx, sy), w in zip(spots_proj, widths):
        ax.add_patch(plt.Circle((sx, sy), w / 2, fill=False, ec="red", lw=1.4, zorder=6))
        ax.plot(sx, sy, "r+", ms=8, zorder=7)
    ax.plot([centre[0] - z / 2 + 4, centre[0] - z / 2 + 14], [centre[1] + z / 2 - 5] * 2, "w-", lw=3, zorder=8)
    ax.text(centre[0] - z / 2 + 9, centre[1] + z / 2 - 8, "10 µm", color="w", ha="center", fontsize=7, zorder=8)
    ax.set_title(f"zoom: cones under the spots\nlocal cone spacing ~{spacing:.0f} µm", fontsize=8, loc="left")
    ax = axs[r_i, 4]
    vb = np.nanmax(np.abs(np.where(cmask, cmb, np.nan)))
    ax.imshow(np.where(cmask, cmb, np.nan) / vb, extent=ext_cm, origin="upper", cmap="coolwarm", vmin=-1, vmax=1,
              alpha=0.65, interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=5)
    for (sx, sy), w in zip(spots_proj, widths):
        ax.add_patch(plt.Circle((sx, sy), w / 2, fill=False, ec="k", lw=1.4, zorder=6))
    ax.set_title("covariance map, 15-50 µm band (inside its 20 % area);\nblack circles = reproducible STA spots "
                 "(col. 3)", fontsize=8, loc="left")

fig.suptitle("14/08/2026 6px: bipolar-scale (15-50 µm band) spots on the aligned cone image (v2 alignment). "
             f"Reference FWHM through the same blur + band: single cone {refs['single cone']:.0f} µm, "
             f"bipolar FWHM 15 µm -> {refs['bipolar subunit, FWHM 15 µm']:.0f} µm, "
             f"25 µm -> {refs['bipolar subunit, FWHM 25 µm']:.0f} µm.\nTwo cones produce separate spots only "
             f"if >= {min([d for d, n in sep.items() if n >= 2], default=np.nan):.0f} µm apart. "
             "Alignment error of a few µm is likely, so cone-by-cone matching is not reliable.",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"bipolar_spots_on_cones.{fmt}", dpi=140 if fmt == "png" else None, bbox_inches="tight")
(OUT / "bipolar_spots_on_cones.json").write_text(json.dumps(dict(refs=refs, two_cone_separation=sep,
                                                                 cells=results), indent=1, default=float))
print("saved", OUT / "bipolar_spots_on_cones.png")
