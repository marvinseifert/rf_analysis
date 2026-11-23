"""
Shape-twin control for the split-half fine-structure test (23/11/2026).

The elliptical-Gaussian smooth twin in cone_bump_power_and_split_half.py is beaten by any RF that is not an
ellipse (arms, corners, edge clipping). The shape twin keeps the cell's own RF shape and removes only fine
detail: planted RF = the cell's full RF map smoothed with a Gaussian (sigma --shape-sigma-um, default 4 µm,
FWHM 9.4 µm; passes < 11 % at a 12 µm period, < 1 % at 8 µm). That RF is pushed through the same LN model and
estimator as before (actual stimulus window, real time course, real spike count, beta set so the expected map
peak equals the real map peak), --n-twin times, and its split-half r_fine is computed exactly as for the real
cell. A cell passes if its real r_fine is above the shape twin's 95th percentile: its fine structure then
reproduces better than a cell with the same RF shape but no detail finer than ~10 µm.

Caveat: the real full map contains noise; smoothing keeps its coarse part, which the twin then treats as
signal. That makes the twin, if anything, slightly MORE reproducible (a conservative control).

Default cells = those that passed the elliptical twin (read from the JSON). --all-cells runs every cell.
Run from the repo root (~10 min for 13 cells):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/shape_twin_control.py
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
parser.add_argument("--shape-sigma-um", type=float, default=4.0)
parser.add_argument("--n-twin", type=int, default=20)
parser.add_argument("--all-cells", action="store_true")
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
STIM_FILE = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5")
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
RES = OUT / "cone_bump_power_and_split_half_14_08_2026_6px.json"
UM, SEQ, N_REP, STIM_INDEX = 1.9476, 27000, 3, 10
HALF = SEQ // 2

res = json.loads(RES.read_text())
a = res["args"]
L, CUT, HP = a["n_lags"], a["cut_half"], a["hp_um"]
W = 2 * CUT
real = {int(c): v for c, v in res["real"].items()}
if args.all_cells:
    cells = sorted(real)
else:
    cells = sorted([c for c, v in real.items() if v["r_fine"] > v["r_fine_smooth_twin_p95"]],
                   key=lambda c: -(real[c]["r_fine"] - real[c]["r_fine_smooth_twin_p95"]))
print(f"{len(cells)} cells: {cells}")

rec = Overview.Recording.load(PROJECT / "overview")
rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @STIM_INDEX")
triggers = np.asarray(rec.dataframes["stimulus_df"].query("stimulus_index == @STIM_INDEX")
                      ["trigger_fr_relative"].to_numpy()[0])
f = h5py.File(STIM_FILE, "r")
noise = f["Noise"]
H, Wst = noise.shape[1:]
fi = np.arange(SEQ)
IN_A, IN_B = (fi >= L) & (fi < HALF), fi >= HALF + L


def window(cx, cy):
    oy, ox = int(cy) - CUT, int(cx) - CUT
    y0, y1, x0, x1 = max(oy, 0), min(oy + W, H), max(ox, 0), min(ox + W, Wst)
    out = np.zeros((SEQ, W, W), np.float32)
    out[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox] = noise[:, y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    valid = np.zeros((W, W), bool)
    valid[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = True
    return out.reshape(SEQ, -1), valid


def rf_map(c, S, k):
    kc = sum(k[lag] * np.roll(c, -lag) for lag in range(L))
    return ((kc @ S) / c.sum()).reshape(W, W)


def fine(m):
    return m - gaussian_filter(m, HP / UM)


def corr(x, y, mask):
    x, y = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def r_fine(c, S, k, mask):
    return corr(fine(rf_map(c * IN_A, S, k)), fine(rf_map(c * IN_B, S, k)), mask)


rows = {}
for cell in cells:
    v = real[cell]
    S, valid = window(*v["centre"])
    sp = rec.get_spikes_triggered([{"stimulus_index": STIM_INDEX, "cell_index": cell}], time="frames", pandas=False)
    t = np.sort(sp["times_relative"].to_numpy().astype(np.int64).ravel())
    fr = np.searchsorted(triggers, t, side="right") - 1
    fr = fr[(fr >= 0) & (fr < SEQ * N_REP)]
    c = np.bincount(fr % SEQ, minlength=SEQ).astype(float)
    k = np.array(v["k"])
    m = rf_map(c, S, k)
    sm = gaussian_filter(m, 3 / UM)
    mask = (sm >= 0.3 * sm[valid].max()) & valid
    r_real = r_fine(c, S, k, mask)
    peak = m[valid].max()

    rf = gaussian_filter(np.where(valid, m, 0.0), args.shape_sigma_um / UM) * valid  # shape twin RF
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
        mt = rf_map(cf, S, k)
        smt = gaussian_filter(mt, 3 / UM)
        twins.append(r_fine(cf, S, k, (smt >= 0.3 * smt[valid].max()) & valid))
    twins = np.array(twins)
    # the twin's r with noise-free halves = how much the smoothed shape alone reproduces
    r_shape_ceiling = r_fine(lam * N_REP, S, k, mask)
    rows[cell] = dict(r_real=r_real, shape_twin_median=float(np.median(twins)),
                      shape_twin_p95=float(np.percentile(twins, 95)), shape_twins=twins.tolist(),
                      ell_twin_median=v["r_fine_smooth_twin"], ell_twin_p95=v["r_fine_smooth_twin_p95"],
                      shape_noise_free=r_shape_ceiling, quality=v["quality"], n_spikes=int(len(fr)),
                      passes=bool(r_real > np.percentile(twins, 95)))
    print(f"cell {cell}: real r {r_real:+.2f} | shape twin median {np.median(twins):+.2f}, "
          f"95% {np.percentile(twins, 95):+.2f} | elliptical twin 95% {v['r_fine_smooth_twin_p95']:+.2f} "
          f"-> {'PASS' if rows[cell]['passes'] else 'fail'}", flush=True)

n_pass = sum(r["passes"] for r in rows.values())
print(f"\n{n_pass}/{len(rows)} cells beat their shape twin's 95th percentile "
      f"(expected by chance ~{0.05 * len(rows):.1f}): {[c for c, r in rows.items() if r['passes']]}")
stem = f"shape_twin_control_14_08_2026_6px{'_all' if args.all_cells else ''}"
(OUT / f"{stem}.json").write_text(json.dumps(dict(args=vars(args), cells=rows), indent=1))

fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(rows) + 2), 4.8))
for i, (cell, r) in enumerate(rows.items()):
    tw = np.array(r["shape_twins"])
    ax.scatter(np.full(len(tw), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(tw)), tw, s=8,
               color="0.6", zorder=1)
    ax.plot([i - 0.25, i + 0.25], [r["shape_twin_p95"]] * 2, color="0.3", lw=1.5, zorder=2)
    ax.plot([i - 0.25, i + 0.25], [r["ell_twin_p95"]] * 2, color="tab:blue", lw=1.5, ls=":", zorder=2)
    ax.scatter(i, r["r_real"], s=60, marker="D", color="tab:red" if r["passes"] else "white",
               edgecolor="tab:red", zorder=3)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xticks(range(len(rows)), [str(c) for c in rows], fontsize=8)
ax.set_xlabel("cell (sorted by margin over the elliptical twin)")
ax.set_ylabel("split-half r of fine structure")
ax.set_title(f"Shape-twin control: real cell (diamond; filled = above the shape twin's 95th pct) vs {args.n_twin} "
             f"shape twins (grey; line = 95th pct).\nBlue dotted = elliptical twin 95th pct. "
             f"Shape twin = own RF map smoothed sigma {args.shape_sigma_um:.0f} µm. {n_pass}/{len(rows)} pass.",
             fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"{stem}.{fmt}", dpi=170 if fmt == "png" else None, bbox_inches="tight")
print("saved", OUT / f"{stem}.png")
