"""
Half-split reproducibility at BIPOLAR-CELL scale, with a shape-twin control (23/11/2026).

Same as cone_bump_power_and_split_half.py part 1 + shape_twin_control.py, but for structure of roughly
15-50 µm (bipolar-sized subunits / lobes) instead of < 10 µm (cones):
  band(map) = Gaussian(map, --band-lo-um) - Gaussian(map, --band-hi-um)   (default 3 and 8 µm sigma)
  passes ~0.75 at a 25 µm period, ~0.3 at 12 µm and at 60 µm, ~0.05 at 150 µm.
Real cell: r_band = correlation of band(map) between independent noise-sequence halves (frames 0-13499 vs
13500-26999 of every repeat), inside the RF (smoothed map >= --rf-fraction of its peak).
Chance: the same with the spike train shifted in time (>= 5 min).
Shape twin: planted RF = the cell's own full map smoothed with --twin-sigma-um (default 10 µm: passes < 5 % at a
25 µm period), pushed through the same LN model / estimator (real stimulus, time course, spike count, peak),
--n-twin times. A cell passes if r_band(real) > 95th percentile of its twins: its 15-50 µm structure reproduces
better than a cell with the same overall RF shape but no structure at that scale.
Window: 100 x 100 file px (195 µm) so whole RFs and their lobes fit.

All cells with quality >= 10 (from the JSON of the main run). Loads the 17 GB stimulus once.
Run from the repo root (~40 min):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/bipolar_scale_split_half.py
"""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from polarspike import Overview
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("--band-lo-um", type=float, default=3.0)
parser.add_argument("--band-hi-um", type=float, default=8.0)
parser.add_argument("--twin-sigma-um", type=float, default=10.0)
parser.add_argument("--rf-fraction", type=float, default=0.2)
parser.add_argument("--n-twin", type=int, default=20)
parser.add_argument("--cut-half", type=int, default=50)
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
STIM_FILE = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5")
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
RES = OUT / "cone_bump_power_and_split_half_14_08_2026_6px.json"
UM, SEQ, N_REP, STIM_INDEX = 1.9476, 27000, 3, 10
HALF = SEQ // 2
res = json.loads(RES.read_text())
L = res["args"]["n_lags"]
CUT = args.cut_half
W = 2 * CUT
real = {int(c): v for c, v in res["real"].items()}
cells = sorted(real, key=lambda c: -real[c]["quality"])
print(f"{len(cells)} cells")

rec = Overview.Recording.load(PROJECT / "overview")
rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @STIM_INDEX")
triggers = np.asarray(rec.dataframes["stimulus_df"].query("stimulus_index == @STIM_INDEX")
                      ["trigger_fr_relative"].to_numpy()[0])
print("loading stimulus ...", flush=True)
with h5py.File(STIM_FILE, "r") as f:
    d = f["Noise"]
    stim = np.empty(d.shape, np.uint8)
    for s in range(0, d.shape[0], 4220):
        stim[s:s + 4220] = d[s:s + 4220]
H, Wst = stim.shape[1:]
fi = np.arange(SEQ)
IN_A, IN_B = (fi >= L) & (fi < HALF), fi >= HALF + L
rng0 = np.random.default_rng(0)


def window(cx, cy):
    oy, ox = int(cy) - CUT, int(cx) - CUT
    y0, y1, x0, x1 = max(oy, 0), min(oy + W, H), max(ox, 0), min(ox + W, Wst)
    out = np.zeros((SEQ, W, W), np.float32)
    out[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox] = stim[:, y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    valid = np.zeros((W, W), bool)
    valid[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = True
    return out.reshape(SEQ, -1), valid


def rf_map(c, S, k):
    kc = sum(k[lag] * np.roll(c, -lag) for lag in range(L))
    return ((kc @ S) / c.sum()).reshape(W, W)


def band(m):
    return gaussian_filter(m, args.band_lo_um / UM) - gaussian_filter(m, args.band_hi_um / UM)


def rf_mask(m, valid):
    sm = gaussian_filter(m, 3 / UM)
    return (sm >= args.rf_fraction * sm[valid].max()) & valid


def corr(x, y, mask):
    x, y = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def r_band(c, S, k, mask):
    return corr(band(rf_map(c * IN_A, S, k)), band(rf_map(c * IN_B, S, k)), mask)


rows, keep_maps = {}, {}
for n_c, cell in enumerate(cells):
    v = real[cell]
    S, valid = window(*v["centre"])
    sp = rec.get_spikes_triggered([{"stimulus_index": STIM_INDEX, "cell_index": cell}], time="frames", pandas=False)
    t = np.sort(sp["times_relative"].to_numpy().astype(np.int64).ravel())
    fr = np.searchsorted(triggers, t, side="right") - 1
    fr = fr[(fr >= 0) & (fr < SEQ * N_REP)]
    c = np.bincount(fr % SEQ, minlength=SEQ).astype(float)
    shift = rng0.integers(SEQ * N_REP // 6, SEQ * N_REP * 5 // 6)
    c_sh = np.bincount(((fr + shift) % (SEQ * N_REP)) % SEQ, minlength=SEQ).astype(float)
    k = np.array(v["k"])
    m = rf_map(c, S, k)
    mask = rf_mask(m, valid)
    if mask.sum() < 20:
        print(f"  cell {cell}: RF mask too small, skipped"); continue
    r_real = r_band(c, S, k, mask)
    r_shift = r_band(c_sh, S, k, mask)
    peak = m[valid].max()
    rf = gaussian_filter(np.where(valid, m, 0.0), args.twin_sigma_um / UM) * valid
    u = S @ rf.ravel()
    drive = np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(np.r_[k, np.zeros(SEQ - L)])))
    drive /= drive.std()
    lo, hi = 0.0, 3.0
    for _ in range(22):
        beta = (lo + hi) / 2
        lam = np.exp(beta * drive); lam *= len(fr) / (N_REP * lam.sum())
        lo, hi = (beta, hi) if rf_map(lam * N_REP, S, k)[valid].max() < peak else (lo, beta)
    lam = np.exp(beta * drive); lam *= len(fr) / (N_REP * lam.sum())
    rng = np.random.default_rng(cell)
    twins = []
    for _ in range(args.n_twin):
        cf = rng.poisson(lam, size=(N_REP, SEQ)).sum(axis=0).astype(float)
        twins.append(r_band(cf, S, k, rf_mask(rf_map(cf, S, k), valid)))
    twins = np.array(twins)
    p95 = float(np.percentile(twins, 95))
    rows[cell] = dict(quality=v["quality"], n_spikes=int(len(fr)), r_real=r_real, r_shift=r_shift,
                      twin_median=float(np.median(twins)), twin_p95=p95, twins=twins.tolist(),
                      passes=bool(r_real > p95), margin=float(r_real - p95))
    keep_maps[cell] = (m, rf_map(c * IN_A, S, k), rf_map(c * IN_B, S, k), valid, mask)
    print(f"  [{n_c + 1}/{len(cells)}] cell {cell} (q {v['quality']:.0f}): r_band {r_real:+.2f} | shifted "
          f"{r_shift:+.2f} | shape twin median {np.median(twins):+.2f}, 95% {p95:+.2f} -> "
          f"{'PASS' if r_real > p95 else 'fail'}", flush=True)

n_pass = sum(r["passes"] for r in rows.values())
from scipy.stats import binomtest
p_pop = binomtest(n_pass, len(rows), 0.05, alternative="greater").pvalue
print(f"\n{n_pass}/{len(rows)} cells beat their shape twin's 95th percentile (chance ~{0.05 * len(rows):.1f}; "
      f"binomial p = {p_pop:.3g}): {[c for c, r in rows.items() if r['passes']]}")
print(f"median r_band: real {np.median([r['r_real'] for r in rows.values()]):+.2f}, shifted "
      f"{np.median([r['r_shift'] for r in rows.values()]):+.2f}, twin {np.median([r['twin_median'] for r in rows.values()]):+.2f}")
stem = "bipolar_scale_split_half_14_08_2026_6px"
(OUT / f"{stem}.json").write_text(json.dumps(dict(args=vars(args), n_pass=n_pass, p_population=p_pop,
                                                  cells=rows), indent=1))

# ------------------------------------------------------------------ figure
best = sorted(rows, key=lambda c: -rows[c]["margin"])[:6]
fig = plt.figure(figsize=(18, 4.2 + 2.6 * len(best)))
gs = fig.add_gridspec(1 + len(best), 6, height_ratios=[1.6] + [1] * len(best))
ax = fig.add_subplot(gs[0, 0:3])
order = sorted(rows, key=lambda c: -rows[c]["quality"])
for i, cell in enumerate(order):
    r = rows[cell]
    tw = np.array(r["twins"])
    ax.scatter(np.full(len(tw), i), tw, s=3, color="0.75", zorder=1)
    ax.plot([i - 0.35, i + 0.35], [r["twin_p95"]] * 2, color="0.3", lw=1, zorder=2)
    ax.scatter(i, r["r_shift"], s=10, color="tab:orange", zorder=2)
    ax.scatter(i, r["r_real"], s=28, marker="D", color="tab:red" if r["passes"] else "white", edgecolor="tab:red",
               zorder=3)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xticks(range(len(order)), [str(c) for c in order], fontsize=6, rotation=90)
ax.set_xlabel("cell (sorted by quality, high -> low)")
ax.set_ylabel("split-half r, 15-50 µm band")
ax.set_title(f"Bipolar-scale structure: real (diamond, filled = above the shape twin's 95th pct), shape twins "
             f"(grey, line = 95th pct), shifted spikes (orange). {n_pass}/{len(rows)} pass "
             f"(chance {0.05 * len(rows):.1f}, binomial p = {p_pop:.2g})", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax = fig.add_subplot(gs[0, 3:5])
x = np.array([rows[c]["twin_p95"] for c in rows]); y = np.array([rows[c]["r_real"] for c in rows])
ax.scatter(x, y, s=14, color=["tab:red" if rows[c]["passes"] else "0.5" for c in rows])
for c in rows:
    if rows[c]["passes"]:
        ax.annotate(str(c), (rows[c]["twin_p95"], rows[c]["r_real"]), fontsize=7, xytext=(3, 2),
                    textcoords="offset points")
lim = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("shape twin, 95th percentile"); ax.set_ylabel("real cell")
ax.spines[["top", "right"]].set_visible(False)
ext = [-CUT * UM, CUT * UM, CUT * UM, -CUT * UM]
for r_i, cell in enumerate(best):
    m, ma, mb, valid, mask = keep_maps[cell]
    panels = [(m, "full map"), (ma, "half A"), (mb, "half B"), (band(ma), "15-50 µm band, half A"),
              (band(mb), "15-50 µm band, half B")]
    for j, (img, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + r_i, j])
        img = np.where(valid, img, np.nan)
        lim_ = np.nanpercentile(np.abs(img[mask]), 99) if j >= 3 else np.nanmax(np.abs(img))
        ax.imshow(img, cmap="coolwarm", vmin=-lim_, vmax=lim_, extent=ext)
        ax.contour(np.linspace(ext[0], ext[1], W), np.linspace(ext[3], ext[2], W)[::-1], mask.astype(float),
                   levels=[0.5], colors="k", linewidths=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        if r_i == 0:
            ax.set_title(ttl, fontsize=9)
        if j == 0:
            r = rows[cell]
            ax.set_ylabel(f"cell {cell}\nq {r['quality']:.0f}", fontsize=8)
            ax.plot([ext[0] + 8, ext[0] + 58], [ext[2] - 10] * 2, "k-", lw=2.5)
            ax.text(ext[0] + 33, ext[2] - 22, "50 µm", ha="center", fontsize=7)
    ax = fig.add_subplot(gs[1 + r_i, 5]); ax.axis("off")
    r = rows[cell]
    ax.text(0, 0.5, f"r = {r['r_real']:+.2f}\nshape twin median {r['twin_median']:+.2f}\n95%: {r['twin_p95']:+.2f}\n"
            f"shifted spikes {r['r_shift']:+.2f}\n{'PASS' if r['passes'] else 'fail'}", fontsize=9, va="center")
fig.suptitle("14/08/2026 6px: does structure at bipolar-cell scale (~15-50 µm) reproduce across independent "
             "sequence halves beyond a smooth twin with the same RF shape (own map smoothed 10 µm)?\n"
             "Bottom: the 6 cells with the largest margin over their twin (maps in stimulus orientation; "
             "contour = RF area used).", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"{stem}.{fmt}", dpi=140 if fmt == "png" else None, bbox_inches="tight")
print("saved", OUT / f"{stem}.png")
