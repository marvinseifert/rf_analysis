"""
Are the cone-sized blobs in the 15-50 µm band of an RF map real? Same display for the real cell and controls
(23/11/2026). For each cell (default 282, 218), all on the aligned cone image, same band filter
(Gaussian 3 µm - Gaussian 8 µm), same colour scale rule (each panel normalised to its max inside the RF area):

  1. covariance map (cm_most_important), band       - the map you looked at
  2. our STA map, all spikes, band                  - what we can split and simulate
  3. half A (frames 0-13499), band                  - independent of 4
  4. half B (frames 13500-26999), band
  5. shape twin, band: simulated cell whose RF = the cell's own map smoothed 10 µm (same shape, NO structure
     at this scale), same stimulus window, time course, spike count and map peak
  6. shifted spike train, band: the cell's spikes moved in time (>= 5 min): no stimulus link, pure noise

If 2 shows blobs that also appear in 3 AND 4, and that 5 and 6 do not have, they are real. If 5 or 6 look
just as blobby, the blobs are what the filter makes of RF shape or noise.

Run from the repo root (~5 min):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/band_controls_on_cones.py
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
from scipy.ndimage import gaussian_filter

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file
from loading.noise_data import load_noise_data

parser = argparse.ArgumentParser()
parser.add_argument("--cells", type=int, nargs="+", default=[282, 218])
parser.add_argument("--zoom-um", type=float, default=70.0)
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
STIM_FILE = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5")
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
RES = OUT / "cone_bump_power_and_split_half_14_08_2026_6px.json"
CHANNEL = "6px_15Hz_30mins_shuffle_x6"
UM, SEQ, N_REP, STIM_INDEX, CUT = 1.9476, 27000, 3, 10, 50
W, HALF = 2 * CUT, 13500
LO, HI, TWIN_SIGMA = 3.0, 8.0, 10.0
res = json.loads(RES.read_text())
L = res["args"]["n_lags"]
real = {int(c): v for c, v in res["real"].items()}

with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(alignment_cache_v2_file("14_08_2026"))
A = geo.A_raw_rf_to_projected_um
Ai = geo.A_image1_to_projected_um
cones = iio.imread(cone_image_file("14_08_2026"))
cones = cones[..., :3] if cones.ndim == 3 and cones.shape[2] == 4 else cones
ds = load_noise_data(PROJECT / "noise_analysis", lazy=True)
gx_raw, gy_raw = ds.x.values * geo.rf_scale_factor, ds.y.values * geo.rf_scale_factor
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


def band(m, step):
    return gaussian_filter(m, LO / step) - gaussian_filter(m, HI / step)


titles = ["1 covariance map (cm_most_important)", "2 STA map, all spikes", "3 half A (independent of B)",
          "4 half B", f"5 shape twin (own RF smoothed {TWIN_SIGMA:.0f} µm)", "6 shifted spikes, half A (pure noise)",
          "7 shifted spikes, half B"]
N_SHIFT = 20
fig, axs = plt.subplots(len(args.cells), 7, figsize=(31, 4.8 * len(args.cells)), squeeze=False)
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
    shift = np.random.default_rng(cell).integers(SEQ * N_REP // 6, SEQ * N_REP * 5 // 6)
    c_sh = np.bincount(((fr + shift) % (SEQ * N_REP)) % SEQ, minlength=SEQ).astype(float)
    k = np.array(v["k"])

    def rf_map(cc):
        kc = sum(k[lag] * np.roll(cc, -lag) for lag in range(L))
        return ((kc @ S) / cc.sum()).reshape(W, W)

    m = rf_map(c)
    sm = gaussian_filter(m, 3 / UM)
    mask = (sm >= 0.2 * sm[valid].max()) & valid
    # shape twin
    rf = gaussian_filter(np.where(valid, m, 0.0), TWIN_SIGMA / UM) * valid
    u = S @ rf.ravel()
    drive = np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(np.r_[k, np.zeros(SEQ - L)])))
    drive /= drive.std()
    lo, hi = 0.0, 3.0
    for _ in range(22):
        beta = (lo + hi) / 2
        lam = np.exp(beta * drive); lam *= len(fr) / (N_REP * lam.sum())
        lo, hi = (beta, hi) if rf_map(lam * N_REP)[valid].max() < m[valid].max() else (lo, beta)
    lam = np.exp(beta * drive); lam *= len(fr) / (N_REP * lam.sum())
    c_tw = np.random.default_rng(cell + 1).poisson(lam, size=(N_REP, SEQ)).sum(axis=0).astype(float)

    maps = [rf_map(c), rf_map(c * IN_A), rf_map(c * IN_B), rf_map(c_tw), rf_map(c_sh * IN_A), rf_map(c_sh * IN_B)]
    bands = [band(mm, UM) for mm in maps]

    # covariance map crop, its band and mask
    ix0 = int(np.searchsorted(gx_raw, 2.0 * (ox - 400) * geo.rf_scale_factor))
    iy0 = int(np.searchsorted(gy_raw, 2.0 * (oy - 400) * geo.rf_scale_factor))
    cm = ds.cm_most_important.sel(channel=CHANNEL, cell_index=cell).isel(
        y=slice(max(iy0, 0), iy0 + W), x=slice(max(ix0, 0), ix0 + W)).transpose("y", "x").values.astype(float)
    xs_, ys_ = gx_raw[max(ix0, 0):ix0 + W], gy_raw[max(iy0, 0):iy0 + W]
    cmb = band(np.nan_to_num(cm), step_raw)
    cms = gaussian_filter(np.nan_to_num(cm), 3 / step_raw)
    cmask = cms >= 0.2 * cms.max()
    ext_cm = [xs_[0] - step_raw / 2, xs_[-1] + step_raw / 2, ys_[-1] + step_raw / 2, ys_[0] - step_raw / 2]
    # our window grid extent in raw µm (window px -> file px -> raw)
    rx0, ry0 = 2.0 * (ox - 400) * geo.rf_scale_factor, 2.0 * (oy - 400) * geo.rf_scale_factor
    st = 2.0 * geo.rf_scale_factor
    ext_w = [rx0 - st / 2, rx0 + (W - 0.5) * st, ry0 + (W - 0.5) * st, ry0 - st / 2]
    pk = np.unravel_index(np.argmax(np.where(valid, sm, -np.inf)), sm.shape)
    prx, pry = rx0 + pk[1] * st, ry0 + pk[0] * st
    centre = (A[0, 0] * prx + A[0, 1] * pry + A[0, 2], A[1, 0] * prx + A[1, 1] * pry + A[1, 2])

    # the real (all-spike) band's reproducible-blob check: correlation of bands A and B inside the RF
    def corr(a_, b_):
        a_, b_ = a_[mask] - a_[mask].mean(), b_[mask] - b_[mask].mean()
        return (a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum())
    r_ab = corr(bands[1], bands[2])
    r_sh = corr(bands[4], bands[5])
    rng_s = np.random.default_rng(1000 + cell)
    null = []
    for _ in range(N_SHIFT):
        shf = rng_s.integers(SEQ * N_REP // 6, SEQ * N_REP * 5 // 6)
        cs = np.bincount(((fr + shf) % (SEQ * N_REP)) % SEQ, minlength=SEQ).astype(float)
        null.append(corr(band(rf_map(cs * IN_A), UM), band(rf_map(cs * IN_B), UM)))
    null = np.array(null)
    for j in range(7):
        ax = axs[r_i, j]
        ax.imshow(cones, extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
                  transform=Affine2D(Ai) + ax.transData, zorder=1)
        if j == 0:
            img, ext, msk = cmb, ext_cm, cmask
        else:
            img, ext, msk = bands[j - 1], ext_w, mask
        vb = np.nanmax(np.abs(np.where(msk, img, np.nan)))
        ax.imshow(np.where(msk, img, np.nan) / vb, extent=ext, origin="upper", cmap="coolwarm", vmin=-1, vmax=1,
                  alpha=0.65, interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=5)
        z = args.zoom_um
        ax.set_xlim(centre[0] - z, centre[0] + z); ax.set_ylim(centre[1] + z, centre[1] - z)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ttl = titles[j]
        if j == 3:
            ttl += f"\nreal A vs B: r = {r_ab:+.2f}"
        if j == 6:
            ttl += (f"\nthis shift A vs B: r = {r_sh:+.2f}; {N_SHIFT} shifts: median {np.median(null):+.2f}, "
                    f"95% {np.percentile(null, 95):+.2f}")
        ax.set_title(ttl, fontsize=9, loc="left")
        if j == 0:
            ax.set_ylabel(f"cell {cell} (q {v['quality']:.0f})", fontsize=10)
            ax.plot([centre[0] - z + 6, centre[0] - z + 26], [centre[1] + z - 8] * 2, "w-", lw=3, zorder=8)
            ax.text(centre[0] - z + 16, centre[1] + z - 13, "20 µm", color="w", ha="center", fontsize=8, zorder=8)
    print(f"cell {cell}: band A vs B r = {r_ab:+.2f}; shifted-spike halves r = {r_sh:+.2f}; "
          f"{N_SHIFT} shifts median {np.median(null):+.2f}, 95th pct {np.percentile(null, 95):+.2f}, "
          f"max {null.max():+.2f}")
fig.suptitle("15-50 µm band (Gaussian 3 µm - 8 µm) of the RF map on the aligned cone image, inside each map's RF "
             "area (20 % of peak); red = positive, blue = negative, each panel scaled to its own max.\n"
             "Real blobs must appear in 3 AND 4, and 3 vs 4 must agree better than the noise halves 6 vs 7 "
             "(chance agreement; numbers over 20 shifts).", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"band_controls_on_cones.{fmt}", dpi=130 if fmt == "png" else None, bbox_inches="tight")
print("saved", OUT / "band_controls_on_cones.png")
