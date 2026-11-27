"""
Does cell 282's 6px hot spot also appear in the 8px noise of the same recording? (23/11/2026)

14/08/2026: 6px_15Hz_30mins_shuffle_x6 (stim 10, 27000 frames x 3) and 8px_15Hz_35mins_shuffle_x8
(stim 4, 31500 frames x 3), both shuffled, same 2 µm-nominal file grid. For each stimulus: frame STA collapsed
with that stimulus's own time course (RF-centre pixels), 100 x 100 file px window at the cell's 6px RF centre;
the 15-50 µm band (Gaussian 3 - 8 µm); independent sequence halves; chance from 20 shifted spike trains.
All panels on the aligned cone image (v2), same zoom. File px -> raw µm = 2 x (px - 400) for both files
(6px file 798 px, 8px file 800 px wide: the 1 px difference in centring = 2 µm, ignored).

Run from the repo root (~5 min):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/compare_6px_8px_hotspot.py --cell 282
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

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, default=282)
parser.add_argument("--zoom-um", type=float, default=70.0)
parser.add_argument("--n-shift", type=int, default=20)
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
STIMS = {
    "6px": dict(file=Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5"), index=10,
                quality=PROJECT / "6px_15Hz_30mins_shuffle_x6_idx_10" / "quality.npy"),
    "8px": dict(file=Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/8px_15Hz_35mins_shuffle_x8.h5"), index=4,
                quality=PROJECT / "8px_15Hz_35mins_shuffle_x8_idx_4" / "quality.npy"),
}
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/6px_subunit_exploration")
UM, N_REP, L, CUT = 1.9476, 3, 11, 50
W = 2 * CUT
LO, HI = 3.0, 8.0

with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(alignment_cache_v2_file("14_08_2026"))
A, Ai = geo.A_raw_rf_to_projected_um, geo.A_image1_to_projected_um
cones = iio.imread(cone_image_file("14_08_2026"))
cones = cones[..., :3] if cones.ndim == 3 and cones.shape[2] == 4 else cones
rec = Overview.Recording.load(PROJECT / "overview")
q6 = np.load(STIMS["6px"]["quality"])
cx, cy = q6[q6[:, 1] == args.cell, 2:4][0]  # 6px RF centre (file px), used for both stimuli
oy, ox = int(cy) - CUT, int(cx) - CUT


def band(m):
    return gaussian_filter(m, LO / UM) - gaussian_filter(m, HI / UM)


out = {}
for name, St in STIMS.items():
    qq = np.load(St["quality"])
    qual = float(qq[qq[:, 1] == args.cell, 0][0])
    with h5py.File(St["file"], "r") as f:
        d = f["Noise"]
        seq, H, Wst = d.shape
        y0, y1, x0, x1 = max(oy, 0), min(oy + W, H), max(ox, 0), min(ox + W, Wst)
        S = np.zeros((seq, W, W), np.float32)
        S[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox] = d[:, y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    S = S.reshape(seq, -1)
    valid = np.zeros((W, W), bool); valid[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = True
    rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @St['index']")
    trig = np.asarray(rec.dataframes["stimulus_df"].query("stimulus_index == @St['index']")
                      ["trigger_fr_relative"].to_numpy()[0])
    sp = rec.get_spikes_triggered([{"stimulus_index": St["index"], "cell_index": args.cell}], time="frames",
                                  pandas=False)
    t = np.sort(sp["times_relative"].to_numpy().astype(np.int64).ravel())
    fr = np.searchsorted(trig, t, side="right") - 1
    fr = fr[(fr >= 0) & (fr < seq * N_REP)]
    c = np.bincount(fr % seq, minlength=seq).astype(float)
    fi = np.arange(seq)
    IN_A, IN_B = (fi >= L) & (fi < seq // 2), fi >= seq // 2 + L
    sta = np.stack([np.roll(c, -lag) @ S for lag in range(L)]) / c.sum()
    st = (sta ** 2).max(0)
    k = sta[:, st >= 0.5 * st.max()].mean(1); k /= np.linalg.norm(k)

    def rf_map(cc):
        kc = sum(k[lag] * np.roll(cc, -lag) for lag in range(L))
        return ((kc @ S) / cc.sum()).reshape(W, W)
    m, ma, mb = rf_map(c), rf_map(c * IN_A), rf_map(c * IN_B)
    sm = gaussian_filter(m, 3 / UM)
    mask = (sm >= 0.2 * sm[valid].max()) & valid

    def corr(a_, b_, msk):
        a_, b_ = a_[msk] - a_[msk].mean(), b_[msk] - b_[msk].mean()
        return float((a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum()))
    r_ab = corr(band(ma), band(mb), mask)
    rng = np.random.default_rng(args.cell)
    null = []
    for _ in range(args.n_shift):
        shf = rng.integers(seq * N_REP // 6, seq * N_REP * 5 // 6)
        cs = np.bincount(((fr + shf) % (seq * N_REP)) % seq, minlength=seq).astype(float)
        null.append(corr(band(rf_map(cs * IN_A)), band(rf_map(cs * IN_B)), mask))
    # map SNR: peak / SD of shifted-spike maps
    cs = np.bincount(((fr + seq) % (seq * N_REP)) % seq, minlength=seq).astype(float)
    snr = float(m[valid].max() / rf_map(cs)[valid].std())
    out[name] = dict(quality=qual, n_spikes=int(len(fr)), snr=snr, r_halves=r_ab, null=np.array(null),
                     m=m, ma=ma, mb=mb, mask=mask, valid=valid)
    print(f"{name}: quality {qual:.1f}, {len(fr)} spikes, map SNR {snr:.1f}, band halves r {r_ab:+.2f} "
          f"(shifted: median {np.median(null):+.2f}, 95% {np.percentile(null, 95):+.2f}, max {max(null):+.2f})")

# 6px vs 8px: correlation of the full maps and of their bands inside the 6px RF area
m6, m8, msk6 = out["6px"]["m"], out["8px"]["m"], out["6px"]["mask"]


def corr2(a_, b_):
    a_, b_ = a_[msk6] - a_[msk6].mean(), b_[msk6] - b_[msk6].mean()
    return float((a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum()))
r_maps, r_bands = corr2(m6, m8), corr2(band(m6), band(m8))
print(f"6px vs 8px inside the 6px RF area: full maps r = {r_maps:+.2f}, 15-50 µm bands r = {r_bands:+.2f}")

# ------------------------------------------------------------------ figure
rx0, ry0 = 2.0 * (ox - 400) * geo.rf_scale_factor, 2.0 * (oy - 400) * geo.rf_scale_factor
stp = 2.0 * geo.rf_scale_factor
ext_w = [rx0 - stp / 2, rx0 + (W - 0.5) * stp, ry0 + (W - 0.5) * stp, ry0 - stp / 2]
sm6 = gaussian_filter(m6, 3 / UM)
pk = np.unravel_index(np.argmax(np.where(out["6px"]["valid"], sm6, -np.inf)), sm6.shape)
prx, pry = rx0 + pk[1] * stp, ry0 + pk[0] * stp
centre = (A[0, 0] * prx + A[0, 1] * pry + A[0, 2], A[1, 0] * prx + A[1, 1] * pry + A[1, 2])
fig, axs = plt.subplots(2, 4, figsize=(19, 10.5))
for r_i, name in enumerate(["6px", "8px"]):
    o = out[name]
    panels = [(o["m"], o["valid"], "RF map (all spikes)"), (band(o["m"]), o["mask"], "15-50 µm band, all spikes"),
              (band(o["ma"]), o["mask"], "band, half A"), (band(o["mb"]), o["mask"], "band, half B")]
    for j, (img, msk, ttl) in enumerate(panels):
        ax = axs[r_i, j]
        ax.imshow(cones, extent=[0, cones.shape[1], cones.shape[0], 0], origin="upper",
                  transform=Affine2D(Ai) + ax.transData, zorder=1)
        vb = np.nanmax(np.abs(np.where(msk, img, np.nan)))
        ax.imshow(np.where(msk, img, np.nan) / vb, extent=ext_w, origin="upper", cmap="coolwarm", vmin=-1, vmax=1,
                  alpha=0.6, interpolation="nearest", transform=Affine2D(A) + ax.transData, zorder=5)
        ax.plot(*centre, "+", color="yellow", ms=14, mew=2, zorder=7)  # 6px RF peak, same point in every panel
        z = args.zoom_um
        ax.set_xlim(centre[0] - z, centre[0] + z); ax.set_ylim(centre[1] + z, centre[1] - z)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if j == 3:
            ttl += (f"\nhalves r = {o['r_halves']:+.2f} (shifted spikes: 95% {np.percentile(o['null'], 95):+.2f}, "
                    f"max {o['null'].max():+.2f})")
        ax.set_title(ttl, fontsize=9, loc="left")
        if j == 0:
            ax.set_ylabel(f"{name}: quality {o['quality']:.1f}, {o['n_spikes']} spikes, map SNR {o['snr']:.1f}",
                          fontsize=9)
            ax.plot([centre[0] - z + 6, centre[0] - z + 26], [centre[1] + z - 8] * 2, "w-", lw=3, zorder=8)
            ax.text(centre[0] - z + 16, centre[1] + z - 13, "20 µm", color="w", ha="center", fontsize=8, zorder=8)
fig.suptitle(f"Cell {args.cell}, 14/08/2026: does the 6px hot spot appear in the independent 8px noise? "
             f"(yellow + = 6px RF peak in every panel; 6px vs 8px inside the 6px RF: maps r = {r_maps:+.2f}, "
             f"15-50 µm bands r = {r_bands:+.2f})", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"hotspot_6px_vs_8px_cell_{args.cell}.{fmt}", dpi=130 if fmt == "png" else None,
                bbox_inches="tight")
print("saved", OUT / f"hotspot_6px_vs_8px_cell_{args.cell}.png")
