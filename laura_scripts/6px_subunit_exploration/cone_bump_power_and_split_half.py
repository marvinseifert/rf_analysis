"""
Is fine RF structure at the cone scale real, and could we even detect it? (23/11/2026)

NOTE: the simulated mosaic here is an early schematic (blue every 12 µm, UV 10 µm across, R/G on the B-UV links),
NOT Laura's row mosaic (UV-B row, R/G row, G/R row, B-UV row, ...). Use cone_bump_checker_size.py for simulations
with the correct mosaic. This script's JSON (real-cell maps, time courses, spike counts, elliptical-twin results)
is read by shape_twin_control.py, bipolar_scale_split_half.py and the other follow-up scripts.
14/08/2026, 6px_15Hz_30mins_shuffle_x6 (stimulus index 10). Neither part uses the measured cone mosaic.

PART 1  Half-split reproducibility (real cells, quality >= --min-quality)
  Two RF maps per cell from independent halves of the noise SEQUENCE (frames 0-13499 vs 13500-26999 of each
  of the 3 identical repeats; spikes whose STA window crosses a half boundary are dropped). The halves share no
  stimulus frames. Fine structure F = map - Gaussian(map, --hp-um) (removes the RF blob, keeps spots).
  Reproducibility r_fine = correlation of F_A and F_B inside the RF centre. Chance = the same with the spike
  train shifted in time (circularly by >= 5 min; keeps the spike count and bursts, no link to the stimulus).
  Also: Fourier ring correlation of the half-maps averaged over 10-15 µm periods (cone band).

PART 2  Power check by simulation (cells chosen from the real ones, spanning the quality range)
  Fake RF = Gaussian envelope (centre and size of the real cell's map) x bumps (Gaussian, sigma --bump-um)
  on a zebrafish-like row mosaic with ALL cone types:
     blue cones every 12 µm along a row; UV cones in the rows +-10 µm across (each blue paired with a UV, as in
     cone_rf_connectivity_functions.build_orthogonal_blue_uv_pairs); a red/green double cone on every blue-UV
     segment (red near blue, green near UV, 3 µm from each). 6 cones per 12 x 20 µm unit cell.
     The mosaic gets a random rotation and translation for every simulated cell (no registration assumed).
  Conditions: smooth (no bumps, the false-positive control) / all cones equal weight / all cones random weight
  (uniform 0-1) / UV cones only (sparse, 12 x 20 µm lattice).
  Spikes: LN model through the ACTUAL stimulus window of the real cell: drive = real STA time course k
  (frames) x RF; rate = exp(beta * drive / sd); Poisson per frame, 3 repeats. beta is set so that the
  expected RF map has the same peak as the real cell's map, total spikes = the real cell's spike count
  ("x1"), or 4x more ("x4", = a 4x longer recording), or no Poisson noise at all ("inf", the expected map:
  what the stimulus allows with unlimited data).
  Estimator: the same frame-STA collapsed with k as in Part 1.
  Position test: T = mean of F at the planted cones inside the RF centre (weighted by the smoothed map),
  compared with the same mosaic translated to --n-null random positions within its unit cell.
  p = (1 + #null >= T) / (1 + n_null). Power = fraction of simulated cells with p < 0.05.
  The split-half r_fine is also computed for every simulated cell (x1), to see if Part 1 can see planted bumps.

Map grid = stimulus-file pixels, 1.9476 µm (corrected pixel size for 14/08/2026).
Run from the repo root (loads the 17 GB stimulus into RAM once; ~10-15 min):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/cone_bump_power_and_split_half.py
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from polarspike import Overview
from scipy.ndimage import convolve, gaussian_filter, map_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("--min-quality", type=float, default=10.0)
parser.add_argument("--n-sim-cells", type=int, default=8)
parser.add_argument("--n-seeds", type=int, default=20)
parser.add_argument("--n-null", type=int, default=200)
parser.add_argument("--n-twin", type=int, default=10, help="smooth-twin simulations per real cell")
parser.add_argument("--bump-um", type=float, default=1.5, help="sigma of one cone's bump")
parser.add_argument("--hp-um", type=float, default=6.0, help="sigma of the blob removed to get fine structure")
parser.add_argument("--n-lags", type=int, default=11, help="STA length in frames (15 Hz)")
parser.add_argument("--cut-half", type=int, default=30, help="window half width, file px (as the snippets)")
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
STIM_FOLDER = PROJECT / "6px_15Hz_30mins_shuffle_x6_idx_10"
STIM_FILE = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5")
STIM_INDEX = 10
OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
UM = 1.9476
SEQ, N_REP = 27000, 3
HALF = SEQ // 2
L = args.n_lags
W = 2 * args.cut_half
BLUE_ROW_UM, ROW_UM, DOUBLE_OFFSET_UM = 12.0, 10.0, 3.0
CONDITIONS = ["smooth", "all equal", "all random", "UV only"]
rng_master = np.random.default_rng(0)
t0 = time.time()

# ------------------------------------------------------------------ data
q = np.load(STIM_FOLDER / "quality.npy")  # quality, cell_index, centre_x, centre_y (file px)
rec = Overview.Recording.load(PROJECT / "overview")
rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @STIM_INDEX")
triggers = np.asarray(rec.dataframes["stimulus_df"].query("stimulus_index == @STIM_INDEX")
                      ["trigger_fr_relative"].to_numpy()[0])
cells_with_spikes = set(rec.dataframes["noise_analysis"]["cell_index"].unique())
cells = [int(c) for c in q[q[:, 0] >= args.min_quality, 1] if int(c) in cells_with_spikes]
print(f"{len(cells)} cells with quality >= {args.min_quality}")

print("loading stimulus ...")
with h5py.File(STIM_FILE, "r") as f:
    d = f["Noise"]
    stim = np.empty(d.shape, np.uint8)
    for s in range(0, d.shape[0], 4220):
        stim[s:s + 4220] = d[s:s + 4220]
H_STIM, W_STIM = stim.shape[1:]
print(f"  done ({time.time() - t0:.0f} s)")


def window(cx, cy):
    """(SEQ, W*W) float32 contrast (-1/+1, 0 outside the stimulus) and the valid mask (W, W)."""
    oy, ox = int(cy) - args.cut_half, int(cx) - args.cut_half
    y0, y1, x0, x1 = max(oy, 0), min(oy + W, H_STIM), max(ox, 0), min(ox + W, W_STIM)
    out = np.zeros((SEQ, W, W), np.float32)
    out[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox] = stim[:, y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    valid = np.zeros((W, W), bool)
    valid[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = True
    return out.reshape(SEQ, -1), valid


def spike_frames(cell):
    sp = rec.get_spikes_triggered([{"stimulus_index": STIM_INDEX, "cell_index": cell}], time="frames", pandas=False)
    t = np.sort(sp["times_relative"].to_numpy().astype(np.int64).ravel())
    fr = np.searchsorted(triggers, t, side="right") - 1
    return fr[(fr >= 0) & (fr < SEQ * N_REP)]


def counts_from_frames(fr):
    """spike counts per sequence frame (repeats summed): full, half A, half B."""
    c = np.bincount(fr % SEQ, minlength=SEQ).astype(np.float64)
    f = np.arange(SEQ)
    ok_a = (f >= L) & (f < HALF)
    ok_b = (f >= HALF + L)
    return c, c * ok_a, c * ok_b


def sta(c, S):
    """(L, W*W) STA from counts per sequence frame (circular: the sequence repeats)."""
    n = c.sum()
    return np.stack([np.roll(c, -lag) @ S for lag in range(L)]) / n


def rf_map(c, S, k):
    kc = sum(k[lag] * np.roll(c, -lag) for lag in range(L))
    return ((kc @ S) / c.sum()).reshape(W, W)


def time_course(sta_full):
    strength = (sta_full ** 2).max(axis=0)
    centre = strength >= 0.5 * strength.max()
    k = sta_full[:, centre].mean(axis=1)
    return k / np.linalg.norm(k)


hp_px = args.hp_um / UM
taper = np.outer(np.hanning(W), np.hanning(W))
ffx = np.fft.fftfreq(W, d=UM)
FR = np.hypot(*np.meshgrid(ffx, ffx, indexing="ij"))
BAND = (FR >= 1 / 15) & (FR <= 1 / 10)


def fine(m):
    return m - gaussian_filter(m, hp_px)


def centre_mask(m, valid):
    sm = gaussian_filter(m, 3 / UM)
    return (sm >= 0.3 * sm[valid].max()) & valid


def corr(a, b, mask):
    a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def frc_band(a, b):
    A = np.fft.fft2((a - a.mean()) * taper); B = np.fft.fft2((b - b.mean()) * taper)
    return float(np.real((A[BAND] * np.conj(B[BAND])).sum()) /
                 np.sqrt((np.abs(A[BAND]) ** 2).sum() * (np.abs(B[BAND]) ** 2).sum()))


def split_half_stats(ma, mb, mask):
    return corr(fine(ma), fine(mb), mask), frc_band(ma, mb)


# ------------------------------------------------------------------ PART 1: real cells
real = {}
for i, cell in enumerate(cells):
    cx, cy = q[q[:, 1] == cell, 2:4][0]
    S, valid = window(cx, cy)
    fr = spike_frames(cell)
    shift = rng_master.integers(SEQ * N_REP // 6, SEQ * N_REP * 5 // 6)  # >= 5 min
    c, ca, cb = counts_from_frames(fr)
    s_c, s_a, s_b = counts_from_frames((fr + shift) % (SEQ * N_REP))
    k = time_course(sta(c, S))
    m, ma, mb = rf_map(c, S, k), rf_map(ca, S, k), rf_map(cb, S, k)
    sa, sb = rf_map(s_a, S, k), rf_map(s_b, S, k)
    mask = centre_mask(m, valid)
    r_real, frc_real = split_half_stats(ma, mb, mask)
    r_null, frc_null = split_half_stats(sa, sb, mask)
    sm = gaussian_filter(m, 3 / UM)
    peak = float(sm[valid].max())
    noise_sd = float(np.std(rf_map(s_c, S, k)[valid]))
    real[cell] = dict(quality=float(q[q[:, 1] == cell, 0][0]), n_spikes=int(len(fr)), r_fine=r_real,
                      r_fine_null=r_null, frc_band=frc_real, frc_band_null=frc_null, peak=float(m[valid].max()),
                      snr=float(m[valid].max() / noise_sd), k=k.tolist(), centre=(cx, cy),
                      env_area_px=int(mask.sum()))
    real[cell]["_maps"] = (m, ma, mb, valid)
    print(f"  [{i + 1}/{len(cells)}] cell {cell}: q {real[cell]['quality']:.0f}, {len(fr)} spikes, "
          f"SNR {real[cell]['snr']:.1f}, r_fine {r_real:+.2f} (chance {r_null:+.2f}), "
          f"FRC 10-15 µm {frc_real:+.2f} (chance {frc_null:+.2f})")
print(f"PART 1 done ({time.time() - t0:.0f} s)")


bad = [c for c in real if not np.isfinite([real[c]["r_fine"], real[c]["r_fine_null"]]).all()]
for c in bad:
    print(f"  dropping cell {c} (NaN: too few spikes / empty centre)")
    del real[c]
cells = [c for c in cells if c in real]


# ------------------------------------------------------------------ PART 2: simulations
def mosaic(kind, extent=90.0):
    """Cone positions (µm, mosaic frame) and unit cell (a, b)."""
    a, b = BLUE_ROW_UM, 2 * ROW_UM
    pts = []
    for j in range(-int(extent // b) - 1, int(extent // b) + 2):
        for i in range(-int(extent // a) - 1, int(extent // a) + 2):
            x, y = i * a, j * b
            uv = [(x, y + ROW_UM)]
            if kind == "UV only":
                pts += uv
                continue
            pts += [(x, y)] + uv  # blue, UV
            # double cones on the blue->UV segments above and below the blue (red near blue, green near UV)
            pts += [(x, y + DOUBLE_OFFSET_UM), (x, y + ROW_UM - DOUBLE_OFFSET_UM),
                    (x, y - DOUBLE_OFFSET_UM), (x, y - ROW_UM + DOUBLE_OFFSET_UM)]
    return np.array(pts, float), (a, b)


def place(pts_um, theta, offset_um, centre_px):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (pts_um + offset_um) @ R.T / UM + centre_px  # (x, y) in window px


SUB = 4  # subpixel samples per axis (bumps are smaller than a pixel)
sub = (np.arange(SUB) + 0.5) / SUB - 0.5
gy, gx = np.mgrid[:W, :W].astype(float)
S_PX = 6  # checker size, file px
_tri = 1 - np.abs(np.arange(-S_PX + 1, S_PX)) / S_PX
TRI = np.outer(_tri, _tri)  # autocorrelation of the shuffled 6 px checker (ideal, infinite stimulus)


def gauss2(x, y, c, Sig):
    P = np.linalg.inv(Sig)
    dx, dy = x - c[0], y - c[1]
    return np.exp(-0.5 * (P[0, 0] * dx * dx + 2 * P[0, 1] * dx * dy + P[1, 1] * dy * dy))


def render(cones_px, weights, sigma_px):
    keep = weights > 1e-3
    cones_px, weights = cones_px[keep], weights[keep]
    rf = np.zeros((W, W))
    for dy in sub:
        for dx in sub:
            X, Y = (gx + dx)[..., None], (gy + dy)[..., None]
            rf += (weights * np.exp(-((X - cones_px[:, 0]) ** 2 + (Y - cones_px[:, 1]) ** 2)
                                    / (2 * sigma_px ** 2))).sum(-1)
    return rf / SUB ** 2


def position_test(m, valid, cones_px_fn, unit, rng):
    """T at the planted mosaic vs the mosaic translated within its unit cell."""
    F = fine(m)
    w_map = gaussian_filter(m, 3 / UM)
    w_map = w_map / w_map[valid].max()

    def stat(cpx):
        inside = (cpx[:, 0] >= 0) & (cpx[:, 0] <= W - 1) & (cpx[:, 1] >= 0) & (cpx[:, 1] <= W - 1)
        cpx = cpx[inside]
        w = map_coordinates(w_map, [cpx[:, 1], cpx[:, 0]], order=1)
        v = map_coordinates(valid.astype(float), [cpx[:, 1], cpx[:, 0]], order=0) > 0
        keep = (w >= 0.3) & v
        if keep.sum() == 0:
            return np.nan
        f = map_coordinates(F, [cpx[keep, 1], cpx[keep, 0]], order=1)
        return float((w[keep] * f).sum() / w[keep].sum())

    T = stat(cones_px_fn(np.zeros(2)))
    null = np.array([stat(cones_px_fn(rng.uniform([0, 0], unit))) for _ in range(args.n_null)])
    return T, float((1 + np.sum(null >= T)) / (1 + len(null)))


def envelope(cell, S, valid):
    """Centre and covariance (px) of an elliptical Gaussian matched to the real cell's map."""
    m_real = real[cell]["_maps"][0]
    sm = gaussian_filter(m_real, 3 / UM)
    mask = centre_mask(m_real, valid)
    wpos = np.clip(sm, 0, None) * mask
    yy, xx = np.mgrid[:W, :W]
    c = np.array([(wpos * xx).sum(), (wpos * yy).sum()]) / wpos.sum()
    dx, dy = xx - c[0], yy - c[1]
    cov = np.array([[(wpos * dx * dx).sum(), (wpos * dx * dy).sum()],
                    [(wpos * dx * dy).sum(), (wpos * dy * dy).sum()]]) / wpos.sum()
    return c, cov / 0.484  # self-weighted Gaussian truncated at 30 % of peak has 0.484 x its variance


def simulate(cell, S, valid, cond, seed, mults, keep_maps=False):
    R = real[cell]
    k = np.array(R["k"])
    c, Sig = envelope(cell, S, valid)
    rng = np.random.default_rng([cell, CONDITIONS.index(cond), seed])
    pts, unit = mosaic("UV only" if cond == "UV only" else "all")
    theta = rng.uniform(0, np.pi)
    off0 = rng.uniform([0, 0], unit)
    cones_fn = lambda extra: place(pts, theta, off0 + extra, c)
    cpx = cones_fn(np.zeros(2))
    bump_px = args.bump_um / UM
    if cond == "smooth":
        rf = gauss2(gx, gy, c, Sig + bump_px ** 2 * np.eye(2))
    else:
        w = rng.uniform(0, 1, len(cpx)) if cond == "all random" else np.ones(len(cpx))
        rf = render(cpx, w * gauss2(cpx[:, 0], cpx[:, 1], c, Sig), bump_px)
    rf = rf * valid
    u = S @ rf.ravel()
    drive = np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(np.r_[k, np.zeros(SEQ - L)])))  # circular
    drive /= drive.std()
    lo, hi = 0.0, 3.0  # beta: expected map peak = real map peak (same SNR)
    for _ in range(22):
        beta = (lo + hi) / 2
        lam = np.exp(beta * drive); lam *= R["n_spikes"] / (N_REP * lam.sum())
        pk = rf_map(lam * N_REP, S, k)[valid].max()
        lo, hi = (beta, hi) if pk < R["peak"] else (lo, beta)
    lam = np.exp(beta * drive); lam *= R["n_spikes"] / (N_REP * lam.sum())
    row = dict(cell=cell, cond=cond, seed=seed, beta=beta)
    maps = {}
    for mult in mults:
        if mult == "ideal":
            m = convolve(rf, TRI, mode="constant") * valid
        elif mult == "inf":
            m = rf_map(lam * N_REP, S, k)
        else:
            fct = {"x1": 1, "x4": 4}[mult]
            cf = rng.poisson(lam * fct, size=(N_REP, SEQ)).sum(axis=0).astype(float)
            m = rf_map(cf, S, k)
            if mult == "x1":
                f = np.arange(SEQ)
                ma = rf_map(cf * ((f >= L) & (f < HALF)), S, k)
                mb = rf_map(cf * (f >= HALF + L), S, k)
                row["r_fine"], row["frc_band"] = split_half_stats(ma, mb, centre_mask(m, valid))
                row["n_spikes_x1"] = int(cf.sum())
        row[f"T_{mult}"], row[f"p_{mult}"] = position_test(m, valid, cones_fn, unit, rng)
        maps[mult] = m
    if keep_maps:
        row["_maps"], row["_rf"], row["_cones"], row["_valid"] = maps, rf, cpx, valid
    return row


# smooth twin for every real cell: the split-half null for "no structure finer than the RF"
print("smooth twins for all real cells ...")
for cell in cells:
    S, valid = window(*real[cell]["centre"])
    rs = [simulate(cell, S, valid, "smooth", 100 + s, ["x1"])["r_fine"] for s in range(args.n_twin)]
    real[cell]["r_fine_smooth_twin"] = float(np.median(rs))
    real[cell]["r_fine_smooth_twin_p95"] = float(np.percentile(rs, 95))
print(f"  done ({time.time() - t0:.0f} s)")

order = sorted(cells, key=lambda c: real[c]["quality"])
pick = [order[int(i)] for i in np.linspace(0, len(order) - 1, args.n_sim_cells).round()]
pick = list(dict.fromkeys(pick))
print(f"simulating cells {pick}")
MULTS = ["x1", "x4", "inf", "ideal"]
sims = []
for cell in pick:
    S, valid = window(*real[cell]["centre"])
    for cond in CONDITIONS:
        rows = [simulate(cell, S, valid, cond, seed, MULTS, keep_maps=(cell == pick[-1] and seed == 0))
                for seed in range(args.n_seeds)]
        sims += rows
        print(f"  cell {cell} (q {real[cell]['quality']:.0f}) {cond:10s}: power "
              + ", ".join(f"{mu} {np.mean([r[f'p_{mu}'] < 0.05 for r in rows]):.2f}" for mu in MULTS)
              + f"; split-half r_fine {np.median([r['r_fine'] for r in rows]):+.2f}  ({time.time() - t0:.0f} s)")

# ------------------------------------------------------------------ summary + save
print("\nPOWER of the position test (fraction of simulated cells with p < 0.05), simulated cells pooled:")
for cond in CONDITIONS:
    rows = [s for s in sims if s["cond"] == cond]
    print(f"  {cond:10s}: " + "   ".join(f"{mu} {np.mean([s[f'p_{mu}'] < 0.05 for s in rows]):.2f}" for mu in MULTS)
          + f"   | split-half r_fine median {np.median([s['r_fine'] for s in rows]):+.2f}, "
          f"FRC 10-15 µm median {np.median([s['frc_band'] for s in rows]):+.2f}")
rv = np.array([[real[c]["r_fine"], real[c]["r_fine_null"], real[c]["r_fine_smooth_twin"],
                real[c]["r_fine_smooth_twin_p95"], real[c]["frc_band"], real[c]["frc_band_null"]] for c in cells])
above = rv[:, 0] > rv[:, 3]
print(f"\nREAL cells (n = {len(cells)}): split-half r_fine median {np.median(rv[:, 0]):+.2f}; "
      f"shifted-spike chance {np.median(rv[:, 1]):+.2f}; smooth twin {np.median(rv[:, 2]):+.2f}")
print(f"  above the 95th percentile of their own smooth twin: {above.sum()}/{len(cells)} "
      f"(expected by chance ~{0.05 * len(cells):.1f}): {[c for c, a in zip(cells, above) if a]}")
print(f"  FRC 10-15 µm median {np.median(rv[:, 4]):+.2f} vs chance {np.median(rv[:, 5]):+.2f}")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
stem = "cone_bump_power_and_split_half_14_08_2026_6px"
(OUTPUT_FOLDER / f"{stem}.json").write_text(json.dumps(dict(
    args=vars(args), real={c: {k: v for k, v in r.items() if not k.startswith("_")} for c, r in real.items()},
    sims=[{k: v for k, v in s.items() if not k.startswith("_")} for s in sims]), indent=1, default=float))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(18, 10.5))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
ext = [-0.5, W - 0.5, W - 0.5, -0.5]
for r_i, cond in enumerate(["all random"]):
    ex = [s for s in sims if s["cell"] == pick[-1] and s["seed"] == 0 and s["cond"] == cond][0]
    panels = [(ex["_rf"], f"planted RF: bumps on all cones,\nrandom weights (cell {pick[-1]} envelope)"),
              (ex["_maps"]["ideal"], "ideal map: planted RF x checker\nautocorrelation (unlimited stimulus)"),
              (ex["_maps"]["inf"], "expected map: this 30-min sequence,\nno spike noise"),
              (ex["_maps"]["x1"], f"estimated map, real spike count\n({ex['n_spikes_x1']} spikes)"),
              (fine(ex["_maps"]["x1"]), f"fine structure of the estimate\n(map - {args.hp_um:.0f} µm blur)")]
    for i, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        v = np.abs(img[ex["_valid"]]).max()
        ax.imshow(img, cmap="coolwarm", vmin=-v, vmax=v, extent=ext)
        cp = ex["_cones"]
        ax.plot(cp[:, 0], cp[:, 1], ".", ms=1.5, color="k")
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9, loc="left")
        if i == 0:
            ax.plot([2, 2 + 20 / UM], [W - 3, W - 3], "k-", lw=3)
            ax.text(2 + 10 / UM, W - 5, "20 µm", ha="center", fontsize=8)

ax = fig.add_subplot(gs[1, 0:2])
xs = np.arange(len(CONDITIONS))
cols = {"x1": "tab:blue", "x4": "tab:orange", "inf": "tab:green", "ideal": "tab:purple"}
labs = {"x1": "real spike count", "x4": "4x spikes", "inf": "no spike noise (this sequence)",
        "ideal": "ideal (unlimited stimulus + spikes)"}
for j, mu in enumerate(MULTS):
    pw = [np.mean([s[f"p_{mu}"] < 0.05 for s in sims if s["cond"] == c]) for c in CONDITIONS]
    ax.bar(xs + (j - 1.5) * 0.2, pw, 0.2, color=cols[mu], label=labs[mu])
ax.axhline(0.05, color="k", ls="--", lw=0.8)
ax.set_xticks(xs, ["smooth RF\n(no bumps)", "bumps on all cones\nequal weight", "bumps on all cones\nrandom weight",
                   "bumps on UV\ncones only"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("fraction of simulated cells\nwith cone-position test p < 0.05")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"Position-test power ({len(pick)} cells x {args.n_seeds} random mosaic placements; dashed = 5 %)",
             fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1, 2:4])
groups = [("real cells:\nshifted spikes", rv[:, 1], "0.6"), ("real cells:\nsmooth twins", rv[:, 2], "0.3"),
          ("real cells", rv[:, 0], "tab:red")]
groups += [(f"sim:\n{c}", np.array([s["r_fine"] for s in sims if s["cond"] == c]), "tab:blue") for c in CONDITIONS]
jit = np.random.default_rng(5)
for i, (lab, dd, col) in enumerate(groups):
    ax.scatter(i + jit.uniform(-0.15, 0.15, len(dd)), dd, s=8, alpha=0.6, color=col)
    ax.plot([i - 0.25, i + 0.25], [np.median(dd)] * 2, color="k", lw=2)
ax.axhline(0, color="grey", lw=0.6)
ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
ax.set_ylabel("split-half reproducibility of fine structure\n(r between independent sequence halves)")
ax.set_title("Half-split (real spike counts and SNR in the simulations)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1, 4])
ax.errorbar(rv[:, 2], rv[:, 0], xerr=[np.zeros(len(rv)), rv[:, 3] - rv[:, 2]], fmt="o", ms=3,
            color="tab:red", ecolor="0.7", lw=0.8)
for c, x, y, a in zip(cells, rv[:, 2], rv[:, 0], above):
    if a:
        ax.annotate(str(c), (x, y), fontsize=7, xytext=(3, 2), textcoords="offset points")
lim = [min(rv[:, [0, 2]].min(), -0.3) - 0.05, max(rv[:, [0, 3]].max(), 0.3) + 0.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("same cell's smooth twin (median; bar to 95th pct)")
ax.set_ylabel("real cell")
ax.set_title(f"Real vs matched smooth twin:\n{above.sum()}/{len(cells)} above the twin's 95th percentile",
             fontsize=10, loc="left")
fig.suptitle("14/08/2026, 6px noise: could cone-scale bumps be detected, and does fine structure in real RFs "
             f"reproduce beyond a smooth RF of the same shape? (fine structure = map - {args.hp_um:.0f} µm blur)",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=180 if fmt == "png" else None, bbox_inches="tight")
print(f"saved {OUTPUT_FOLDER / stem}.png/.pdf/.json  ({time.time() - t0:.0f} s)")
