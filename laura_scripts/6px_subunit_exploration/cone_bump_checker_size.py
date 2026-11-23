"""
Cone-bump simulation for 2, 4 and 6 px shuffled checkers (23/11/2026): what would the maps look like, and how
much power would the position test have?

Same model as cone_bump_power_and_split_half.py, but the stimulus is SYNTHETIC shuffled binary noise generated
here for each checker size s (every frame: random +-1 checkers of s x s file px, grid shifted by a random
0..s-1 px in x and y, like the real *_shuffle files), 27000 frames played 3x, 1.9476 µm per file px.

SAME CELL across checker sizes (the fair comparison): the planted RF, time course and output nonlinearity are
fixed. beta and the drive scale are calibrated once on the 6 px stimulus so that the 6 px map has the real
cell's peak (as before); the 2 and 4 px stimuli then use the same beta and the same drive scale, so a finer
checker drives the cell less (more checkers inside the RF average out) and its map SNR drops as it would in a
real experiment. The total spike count is kept equal to the real cell's.

Figure: rows = planted RF (smooth / all equal / all random / one cone type: UV, blue, red, green), same placement per row;
for each checker: ideal map (checker blur only) | estimated map (real spike count) | fine structure.
Then: position-test power (--n-seeds placements per condition and checker) and map SNR per checker.

Run from the repo root (~10 min):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/cone_bump_checker_size.py --cell 282
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from polarspike import Overview
from scipy.ndimage import convolve, gaussian_filter, map_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, default=282)
parser.add_argument("--checkers", type=int, nargs="+", default=[2, 4, 6])
parser.add_argument("--n-seeds", type=int, default=10)
parser.add_argument("--n-null", type=int, default=200)
args = parser.parse_args()

PROJECT = Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00")
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
RES = OUT / "cone_bump_power_and_split_half_14_08_2026_6px.json"
UM, SEQ, N_REP, STIM_INDEX = 1.9476, 27000, 3, 10
# Row mosaic (Laura, 23/11/2026): single-cone row UV-B-UV-B, then two red/green rows (red always next to blue;
# the second R/G row reversed, so each column is one R/G double cone), then a B-UV row shifted by half a step
# (blue next to red), then R/G, G/R and back. Two R/G rows take the space of one UV/B row.
CONE_PITCH_UM = 6.0                  # UV-B centre spacing along a row (B-B = UV-UV = 12 µm)
SINGLE_ROW_UM = CONE_PITCH_UM        # height of a single-cone row
DOUBLE_ROW_UM = SINGLE_ROW_UM / 2    # height of one R/G row
CONDITIONS = ["smooth", "all equal", "all random", "UV only", "Blue only", "Red only", "Green only"]

res = json.loads(RES.read_text())
a = res["args"]
L, CUT, HP, BUMP = a["n_lags"], a["cut_half"], a["hp_um"], a["bump_um"]
W = 2 * CUT
v = {int(c): x for c, x in res["real"].items()}[args.cell]
k = np.array(v["k"])
valid = np.ones((W, W), bool)  # synthetic stimulus covers the whole window


def shuffled_checker(s, seed):
    """(SEQ, W*W) float32 +-1 shuffled checkerboard, checker s file px, random grid offset 0..s-1 per frame."""
    rng = np.random.default_rng(seed)
    n = W // s + 2
    out = np.empty((SEQ, W, W), np.float32)
    for i0 in range(0, SEQ, 3000):
        i1 = min(i0 + 3000, SEQ)
        cells = rng.integers(0, 2, size=(i1 - i0, n, n)).astype(np.float32) * 2 - 1
        big = np.repeat(np.repeat(cells, s, axis=1), s, axis=2)
        oy, ox = rng.integers(0, s, i1 - i0), rng.integers(0, s, i1 - i0)
        for j in range(i1 - i0):
            out[i0 + j] = big[j, oy[j]:oy[j] + W, ox[j]:ox[j] + W]
    return out.reshape(SEQ, -1)


def tri_kernel(s):
    t = 1 - np.abs(np.arange(-s + 1, s)) / s
    return np.outer(t, t)


def rf_map(c, S):
    kc = sum(k[lag] * np.roll(c, -lag) for lag in range(L))
    return ((kc @ S) / c.sum()).reshape(W, W)


def fine(m):
    return m - gaussian_filter(m, HP / UM)


# real cell: spike count and map peak on the REAL 6 px stimulus (calibration target), envelope from the JSON run
rec = Overview.Recording.load(PROJECT / "overview")
rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @STIM_INDEX")
sp = rec.get_spikes_triggered([{"stimulus_index": STIM_INDEX, "cell_index": args.cell}], time="frames", pandas=False)
n_spikes = int(v["n_spikes"])
peak_real = float(v["peak"])

# envelope from the real map (recomputed on the real stimulus would need the 17 GB file; use the stored centre
# size instead: elliptical Gaussian with the same area as the real RF centre, centred in the window)
r_eff = np.sqrt(v["env_area_px"] / np.pi)
sig = r_eff / np.sqrt(2 * np.log(1 / 0.3))
c0 = np.array([W / 2 - 0.5, W / 2 - 0.5])
gy, gx = np.mgrid[:W, :W].astype(float)
SUB = 4
sub = (np.arange(SUB) + 0.5) / SUB - 0.5


def mosaic_rows():
    """One 6-row period: list of (y_centre_um, [(type, x_um), ...]) for x within one 12 µm period."""
    p, hs, hd = CONE_PITCH_UM, SINGLE_ROW_UM, DOUBLE_ROW_UM
    y_s1, y_d2, y_d3 = 0.0, hs / 2 + hd / 2, hs / 2 + 1.5 * hd
    y_s4 = hs / 2 + 2 * hd + hs / 2
    y_d5, y_d6 = y_s4 + hs / 2 + hd / 2, y_s4 + hs / 2 + 1.5 * hd
    return [(y_s1, [("UV", 0.0), ("Blue", p)]),
            (y_d2, [("Green", 0.0), ("Red", p)]),      # red below blue
            (y_d3, [("Red", 0.0), ("Green", p)]),      # reversed: each column = one R/G double cone
            (y_s4, [("Blue", 0.0), ("UV", p)]),        # blue below red
            (y_d5, [("Red", 0.0), ("Green", p)]),      # red below blue
            (y_d6, [("Green", 0.0), ("Red", p)])], y_s4 + hs / 2 + 2 * hd + hs / 2  # period in y


def mosaic(kind, extent=90.0):
    """Cone positions (µm, mosaic frame) of all types or of one type ("UV only", ...); unit cell (12, period_y)."""
    rows, period_y = mosaic_rows()
    A = 2 * CONE_PITCH_UM
    pts = []
    for j in range(-int(extent // period_y) - 1, int(extent // period_y) + 2):
        for i in range(-int(extent // A) - 1, int(extent // A) + 2):
            for y, cones in rows:
                pts += [(i * A + x, j * period_y + y) for t, x in cones if kind == "all" or kind == f"{t} only"]
    return np.array(pts, float), (A, period_y)


def place(pts, theta, off):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (pts + off) @ R.T / UM + c0


def env(x, y):
    return np.exp(-((x - c0[0]) ** 2 + (y - c0[1]) ** 2) / (2 * sig ** 2))


def render(cpx, w, s):
    keep = w > 1e-3
    cpx, w = cpx[keep], w[keep]
    rf = np.zeros((W, W))
    for ddy in sub:
        for ddx in sub:
            X, Y = (gx + ddx)[..., None], (gy + ddy)[..., None]
            rf += (w * np.exp(-((X - cpx[:, 0]) ** 2 + (Y - cpx[:, 1]) ** 2) / (2 * s ** 2))).sum(-1)
    return rf / SUB ** 2


def planted(cond, seed):
    rng = np.random.default_rng([args.cell, CONDITIONS.index(cond), seed])
    pts, unit = mosaic(cond if cond.endswith(" only") else "all")
    theta, off0 = rng.uniform(0, np.pi), rng.uniform([0, 0], unit)
    fn = lambda extra: place(pts, theta, off0 + extra)
    cpx = fn(np.zeros(2))
    if cond == "smooth":
        sb = np.hypot(sig, BUMP / UM)
        rf = np.exp(-((gx - c0[0]) ** 2 + (gy - c0[1]) ** 2) / (2 * sb ** 2))
    else:
        w = rng.uniform(0, 1, len(cpx)) if cond == "all random" else np.ones(len(cpx))
        rf = render(cpx, w * env(cpx[:, 0], cpx[:, 1]), BUMP / UM)
    return rf, fn, unit, cpx, rng


def drive_of(rf, S):
    u = S @ rf.ravel()
    return np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(np.r_[k, np.zeros(SEQ - L)])))


def pos_test(m, fn, unit, rng):
    F = fine(m)
    wm = gaussian_filter(m, 3 / UM); wm = wm / wm.max()

    def stat(cpx):
        ins = (cpx[:, 0] >= 0) & (cpx[:, 0] <= W - 1) & (cpx[:, 1] >= 0) & (cpx[:, 1] <= W - 1)
        cpx = cpx[ins]
        w = map_coordinates(wm, [cpx[:, 1], cpx[:, 0]], order=1)
        ok = w >= 0.3
        if not ok.any():
            return np.nan
        return float((w[ok] * map_coordinates(F, [cpx[ok, 1], cpx[ok, 0]], order=1)).sum() / w[ok].sum())

    T = stat(fn(np.zeros(2)))
    null = np.array([stat(fn(rng.uniform([0, 0], unit))) for _ in range(args.n_null)])
    return float((1 + np.sum(null >= T)) / (1 + len(null)))


print("generating stimuli ...")
STIM = {s: shuffled_checker(s, 1000 + s) for s in args.checkers}
S6 = STIM[6] if 6 in STIM else shuffled_checker(6, 1006)


def simulate(cond, seed, keep=False):
    rf, fn, unit, cpx, rng = planted(cond, seed)
    d6 = drive_of(rf, S6)
    scale = d6.std()  # fixed drive scale of THIS cell, set on 6 px
    lo, hi = 0.0, 3.0
    for _ in range(22):
        beta = (lo + hi) / 2
        lam = np.exp(beta * d6 / scale); lam *= n_spikes / (N_REP * lam.sum())
        lo, hi = (beta, hi) if rf_map(lam * N_REP, S6).max() < peak_real else (lo, beta)
    out = {}
    for s in args.checkers:
        S = STIM[s]
        d = drive_of(rf, S) / scale
        lam = np.exp(beta * d); lam *= n_spikes / (N_REP * lam.sum())
        cf = rng.poisson(lam, size=(N_REP, SEQ)).sum(axis=0).astype(float)
        m = rf_map(cf, S)
        # map SNR: peak / SD of the same estimator on spikes with no stimulus link (shuffled counts)
        noise_sd = rf_map(rng.permutation(cf), S).std()
        out[s] = dict(p=pos_test(m, fn, unit, rng), snr=float(m.max() / noise_sd), drive_sd=float(d.std()))
        if keep:
            out[s].update(m=m, ideal=convolve(rf, tri_kernel(s), mode="constant"), rf=rf, cones=cpx)
    return out


rows = {cond: [simulate(cond, sd, keep=(sd == 0)) for sd in range(args.n_seeds)] for cond in CONDITIONS}
print(f"\ncell {args.cell} envelope, {n_spikes} spikes, same cell (RF + nonlinearity fixed on 6 px)")
print("power (fraction of placements with p < 0.05) | median map SNR | drive SD relative to 6 px")
summary = {}
for cond in CONDITIONS:
    for s in args.checkers:
        pw = np.mean([r[s]["p"] < 0.05 for r in rows[cond]])
        snr = np.median([r[s]["snr"] for r in rows[cond]])
        dsd = np.median([r[s]["drive_sd"] for r in rows[cond]])
        summary[f"{cond}|{s}"] = dict(power=float(pw), snr=float(snr), drive_sd=float(dsd))
        print(f"  {cond:10s} {s} px: power {pw:.2f} | SNR {snr:5.1f} | drive {dsd:.2f}")

stem = f"cone_bump_checker_size_cell_{args.cell}"
(OUT / f"{stem}.json").write_text(json.dumps(dict(args=vars(args), summary=summary), indent=1))

# ------------------------------------------------------------------ figure
nc = len(args.checkers)
NR = len(CONDITIONS)
fig = plt.figure(figsize=(2.1 * (1 + 3 * nc) + 1, 2.3 * NR + 3.6))
gs = fig.add_gridspec(NR + 1, 1 + 3 * nc, height_ratios=[1] * NR + [1.45], hspace=0.35, wspace=0.08)
ext = [-0.5, W - 0.5, W - 0.5, -0.5]


def show(ax, img, title=None, cones=None):
    lim = np.percentile(np.abs(img), 99.5)
    ax.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim, extent=ext)
    if cones is not None:
        ax.plot(cones[:, 0], cones[:, 1], ".", ms=1.2, color="k", alpha=0.6)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


for r, cond in enumerate(CONDITIONS):
    ex = rows[cond][0]
    s0 = args.checkers[0]
    ax = fig.add_subplot(gs[r, 0])
    show(ax, ex[s0]["rf"], "planted RF" if r == 0 else None, ex[s0]["cones"])
    ax.set_ylabel(cond, fontsize=10)
    if r == 0:
        ax.plot([3, 3 + 20 / UM], [W - 4, W - 4], "k-", lw=3)
        ax.text(3 + 10 / UM, W - 7, "20 µm", ha="center", fontsize=7)
    for i, s in enumerate(args.checkers):
        e = ex[s]
        t = [f"{s}px ({s * UM:.1f} µm): ideal", f"{s}px: estimate, SNR {e['snr']:.0f}",
             f"{s}px: fine structure"] if r == 0 else [None, f"SNR {e['snr']:.0f}", None]
        show(fig.add_subplot(gs[r, 1 + 3 * i]), e["ideal"], t[0])
        show(fig.add_subplot(gs[r, 2 + 3 * i]), e["m"], t[1])
        axf = fig.add_subplot(gs[r, 3 + 3 * i])
        show(axf, fine(e["m"]), t[2], e["cones"])
        axf.text(0.97, 0.03, f"p = {e['p']:.3f}", transform=axf.transAxes, ha="right", fontsize=7,
                 bbox=dict(fc="w", ec="none", alpha=0.7))

sub_gs = gs[NR, :].subgridspec(1, 3, wspace=0.3, width_ratios=[1.2, 1, 0.8])
ax = fig.add_subplot(sub_gs[0])
xs = np.arange(len(CONDITIONS))
for i, s in enumerate(args.checkers):
    ax.bar(xs + (i - (nc - 1) / 2) * 0.8 / nc, [summary[f"{c}|{s}"]["power"] for c in CONDITIONS], 0.8 / nc,
           label=f"{s}px ({s * UM:.1f} µm)")
ax.axhline(0.05, color="k", ls="--", lw=0.8)
ax.set_xticks(xs, CONDITIONS, fontsize=8, rotation=20)
ax.set_ylim(0, 1.05)
ax.set_ylabel(f"position-test power\n({args.n_seeds} placements, real spike count)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax = fig.add_subplot(sub_gs[1])
for i, s in enumerate(args.checkers):
    ax.bar(xs + (i - (nc - 1) / 2) * 0.8 / nc, [summary[f"{c}|{s}"]["snr"] for c in CONDITIONS], 0.8 / nc,
           label=f"{s}px")
ax.set_xticks(xs, CONDITIONS, fontsize=8, rotation=20)
ax.set_ylabel("map SNR (peak / noise SD)")
ax.set_title("same cell: finer checkers drive it less", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False)
# schematic of the simulated mosaic (unrotated, 2 x 2 periods)
from matplotlib.patches import Ellipse
ax = fig.add_subplot(sub_gs[2])
COL = {"UV": "magenta", "Blue": "tab:blue", "Red": "tab:red", "Green": "tab:green"}
rows_, py = mosaic_rows()
for jj in range(2):
    for ii in range(-1, 3):
        for y, cones in rows_:
            for t, x in cones:
                X, Y = ii * 2 * CONE_PITCH_UM + x, jj * py + y
                single = t in ("UV", "Blue")
                h = SINGLE_ROW_UM * 0.9 if single else DOUBLE_ROW_UM * 0.9
                ax.add_patch(Ellipse((X, Y), CONE_PITCH_UM * 0.9, h, color=COL[t], alpha=0.85, lw=0))
                ax.text(X, Y, t[0] if t != "UV" else "UV", ha="center", va="center", fontsize=6, color="w")
ax.set_xlim(-CONE_PITCH_UM * 0.6, 3 * 2 * CONE_PITCH_UM - CONE_PITCH_UM * 0.4)
ax.set_ylim(2 * py - SINGLE_ROW_UM / 2, -SINGLE_ROW_UM / 2 - 0.5)
ax.set_aspect("equal")
ax.set_xlabel("µm along the row"); ax.set_ylabel("µm across rows")
ax.annotate("", xy=(CONE_PITCH_UM, -SINGLE_ROW_UM / 2 - 0.3), xytext=(0, -SINGLE_ROW_UM / 2 - 0.3),
            arrowprops=dict(arrowstyle="<->", lw=0.8), annotation_clip=False)
ax.text(CONE_PITCH_UM / 2, -SINGLE_ROW_UM / 2 - 1.0, f"{CONE_PITCH_UM:.0f} µm", ha="center", fontsize=7)
ax.set_title(f"Simulated mosaic (before random rotation): UV/B row, R/G, G/R rows (red next to blue),\n"
             f"B/UV row shifted half a step; single row {SINGLE_ROW_UM:.0f} µm, each R/G row "
             f"{DOUBLE_ROW_UM:.0f} µm; period {2 * CONE_PITCH_UM:.0f} x {py:.0f} µm", fontsize=8, loc="left")
ax.tick_params(labelsize=7)

fig.suptitle(f"Cone-bump simulation with synthetic shuffled checkers of {', '.join(map(str, args.checkers))} px: "
             f"same cell (envelope size, time course, {n_spikes} spikes and nonlinearity of real cell {args.cell}, "
             f"calibrated on 6 px).\nSame mosaic placement in every column of a row; p = position test on that "
             f"example; dots = cones.", fontsize=10, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"{stem}.{fmt}", dpi=140 if fmt == "png" else None, bbox_inches="tight")
print("saved", OUT / f"{stem}.png")
