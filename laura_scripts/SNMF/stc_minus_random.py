"""
Spike-triggered covariance MINUS random-time covariance (20/11/2026).

Why: in the shuffled noise, neighbouring pixels are correlated (12 µm squares on a 2 µm grid). That
correlation is in the stimulus at ANY time, so it is in the random-time covariance too. Subtracting the
random-time covariance from the spike-triggered covariance cancels it; what is left is how the cell's
spikes change the way pixels vary together (e.g. a subunit that responds to both bright and dark ->
MORE variance in that pattern before spikes).

Steps (one cell):
  1. Collapse every snippet to one image with ONE time course k for all pixels (STA time course of the RF
     centre, as snmf_one_time_course.py). Zero-filled edge columns/rows cropped; first 100 snippets dropped.
  2. Remove the STA pattern from every image (standard in STC: the STA is already known; keeps it from
     showing up again as a covariance pattern).
  3. Covariance (pixels x pixels) of: real spikes, random times seed 0, random times seed 1.
  4. Difference = real - random(seed 0). Chance level = random(seed 1) - random(seed 0): same number of
     snippets, no spikes, so any pattern it shows is noise.
  5. Eigen-decomposition: each eigenvector = a spatial pattern, its eigenvalue = how much MORE (+) or LESS
     (-) that pattern varies before spikes than at random times, as a fraction of the stimulus variance.
     A pattern counts if its eigenvalue is beyond the largest one of the chance difference.

Needs: random_control_snippets.py --cells <cell>  AND  --cells <cell> --seed 1
Outputs (Box, Figures/SNMF): figure (png + pdf) + npz.
Run from the repo root, ONE AT A TIME (~2 min):
    MPLBACKEND=Agg poetry run python laura_scripts/SNMF/stc_minus_random.py --cell 252
"""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, required=True)
parser.add_argument("--n-show", type=int, default=4, help="patterns shown per sign")
parser.add_argument("--control", choices=["random", "shifted"], default="random",
                    help="random: random times (seeds 0 and 1). shifted: the real spike train moved in time "
                         "(seeds 1 and 2; keeps bursts) - made by random_control_snippets.py --mode shifted")
parser.add_argument("--crop-um", type=float, default=None,
                    help="analyse only a square of this size (µm) centred on the RF centre (default: whole window)")
parser.add_argument("--centre-fraction", type=float, default=0.5)
parser.add_argument("--stim-folder", type=Path, default=Path(
    "/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00/6px_15Hz_30mins_shuffle_x6_idx_10"))
parser.add_argument("--drop-first", type=int, default=100)
parser.add_argument("--block", type=int, default=1000)
args = parser.parse_args()

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/SNMF")
REAL, NULL = "tab:blue", "tab:green"
UM_PER_PX = 2.0
cell_dir = args.stim_folder / f"cell_{args.cell}"
if args.control == "random":
    control_files = ["snippets_random_control.h5", "snippets_random_control_seed1.h5"]
    control_label = "random times"
else:
    control_files = ["snippets_shifted_seed1.h5", "snippets_shifted_seed2.h5"]
    control_label = "shifted spike train"
# "random0"/"random1" = the two control sets (random times or shifted spike trains)
files = {"real": cell_dir / "snippets.h5", "random0": cell_dir / control_files[0],
         "random1": cell_dir / control_files[1]}
for name, f in files.items():
    if not f.exists():
        raise SystemExit(f"missing {f}: make it with random_control_snippets.py --cells {args.cell} "
                         f"--mode {args.control} --seed ...")


def valid_slices(dset):
    a = dset.attrs
    ys = slice(int(a["cut_begin_y"] - a["window_origin_y"]), int(a["cut_end_y"] - a["window_origin_y"]))
    xs = slice(int(a["cut_begin_x"] - a["window_origin_x"]), int(a["cut_end_x"] - a["window_origin_x"]))
    return ys, xs


def blocks(dset, ys, xs):
    for i in range(args.drop_first, dset.shape[0], args.block):
        yield dset[i:min(i + args.block, dset.shape[0]), :, ys, xs].astype(np.float32)


with h5py.File(files["real"], "r") as f:
    d = f["snippets"]
    ys, xs = valid_slices(d)
    total, n_real = 0.0, 0
    for b in blocks(d, ys, xs):
        total = total + b.sum(axis=0)
        n_real += len(b)
sta = total / n_real - 0.5
strength = np.max(sta ** 2, axis=0)
centre = strength >= args.centre_fraction * strength.max()
k = sta[:, centre].mean(axis=1)
k /= np.linalg.norm(k)

# optional crop to a square around the RF centre (strength-weighted centre of the centre pixels)
if args.crop_um:
    yy, xx = np.mgrid[:strength.shape[0], :strength.shape[1]]
    w_c = strength * centre
    cy, cx = (w_c * yy).sum() / w_c.sum(), (w_c * xx).sum() / w_c.sum()
    half = int(round(args.crop_um / UM_PER_PX / 2))
    cy0 = int(np.clip(round(cy) - half, 0, strength.shape[0] - 2 * half))
    cx0 = int(np.clip(round(cx) - half, 0, strength.shape[1] - 2 * half))
    crop = (slice(cy0, cy0 + 2 * half), slice(cx0, cx0 + 2 * half))
else:
    crop = (slice(None), slice(None))
strength, centre = strength[crop], centre[crop]
H, W = strength.shape


def images(path):
    """(N, H*W): each snippet collapsed with the same time course k."""
    with h5py.File(path, "r") as f:
        d = f["snippets"]
        if valid_slices(d) != (ys, xs):
            raise SystemExit(f"{path.name}: window differs from the real snippets")
        return np.concatenate([np.einsum("nthw,t->nhw", b - 0.5, k, optimize=True)[:, crop[0], crop[1]]
                               .reshape(len(b), -1)
                               for b in blocks(d, ys, xs)]).astype(np.float64)


X = {name: images(path) for name, path in files.items()}
sta_img = X["real"].mean(axis=0)
u = sta_img / np.linalg.norm(sta_img)


def covariance(x):
    x = x - np.outer(x @ u, u)          # remove the STA pattern
    x = x - x.mean(axis=0)
    return x.T @ x / (len(x) - 1)


C = {name: covariance(x) for name, x in X.items()}
n_each = {name: len(x) for name, x in X.items()}
del X
scale = np.mean(np.diag(C["random0"]))   # average stimulus variance per pixel -> eigenvalues as a fraction
diff = (C["real"] - C["random0"]) / scale
chance = (C["random1"] - C["random0"]) / scale
ev, vec = np.linalg.eigh(diff)
ev_c, vec_c = np.linalg.eigh(chance)
lim_hi, lim_lo = ev_c.max(), ev_c.min()
more = np.where(ev > lim_hi)[0][::-1]     # strongest first
less = np.where(ev < lim_lo)[0]
print(f"cell {args.cell}: {n_each} images of {H} x {W} pixels")
print(f"chance difference: eigenvalues {lim_lo:+.3f} to {lim_hi:+.3f}")
print(f"real - random: top {np.round(ev[::-1][:6], 3)}  bottom {np.round(ev[:6], 3)}")
print(f"{len(more)} patterns vary MORE before spikes than chance allows, {len(less)} vary LESS")

# ---- figure ----
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
n = args.n_show
fig = plt.figure(figsize=(4 + 2.6 * n, 9.5))
gs = fig.add_gridspec(3, n + 1, width_ratios=[1.6] + [1] * n)

ax = fig.add_subplot(gs[:, 0])
m = 40
ax.plot(np.arange(1, m + 1), ev[::-1][:m], "o", ms=4, color=REAL, label=f"real spikes - {control_label}")
ax.plot(np.arange(1, m + 1), ev_c[::-1][:m], "o", ms=4, color=NULL, label=f"chance (two {control_label} sets)")
ax.plot(np.arange(1, m + 1), ev[:m], "o", ms=4, color=REAL, mfc="white")
ax.plot(np.arange(1, m + 1), ev_c[:m], "o", ms=4, color=NULL, mfc="white")
ax.axhline(0, color="grey", lw=0.5)
ax.axhspan(lim_lo, lim_hi, color=NULL, alpha=0.12, lw=0)
ax.set_xlabel("pattern rank (filled: largest +, open: largest -)")
ax.set_ylabel("change in variance before spikes\n(relative to the variance of one pixel)")
ax.set_title(f"{len(more)} patterns vary MORE, {len(less)} LESS\nthan chance (shaded = chance range)",
             fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.spines[["top", "right"]].set_visible(False)


def show(ax, v, title, colour):
    img = v.reshape(H, W)
    lim = np.abs(img).max()
    ax.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
    ax.contour(centre, levels=[0.5], colors="black", linewidths=0.7, linestyles=":")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(2)
    ax.set_title(title, fontsize=9, loc="left")


rows = [("more", np.argsort(ev)[::-1][:n], ev, vec, REAL, "more"),
        ("less", np.argsort(ev)[:n], ev, vec, REAL, "less"),
        ("chance", np.argsort(np.abs(ev_c))[::-1][:n], ev_c, vec_c, NULL, "chance")]
for r_i, (_, idx, e, V, colour, kind) in enumerate(rows):
    for j, i in enumerate(idx):
        ax = fig.add_subplot(gs[r_i, j + 1])
        passed = (kind == "more" and e[i] > lim_hi) or (kind == "less" and e[i] < lim_lo)
        label = {"more": "varies more", "less": "varies less", "chance": "chance pattern"}[kind]
        show(ax, V[:, i], f"{label}: {e[i]:+.3f}" + (" ✓" if passed else ""), colour)
        if r_i == 2 and j == 0:
            bar = 10 if W < 40 else 25  # pixels (2 µm each)
            ax.plot([1, 1 + bar], [1, 1], "k-", lw=3)
            ax.text(1 + bar / 2, 2, f"{bar * UM_PER_PX:.0f} µm", ha="center", fontsize=8)

fig.suptitle(f"Cell {args.cell}: covariance before spikes minus covariance with a {control_label} "
             f"(one time course for all pixels, STA removed"
             + (f", {W * UM_PER_PX:.0f} µm square around the RF centre)" if args.crop_um else ", whole window)"),
             fontsize=12, x=0.01, ha="left")
fig.text(0.01, -0.01,
         "Each map = a spatial pattern; red and blue parts move in opposite directions. ✓ = beyond chance. "
         "Dotted = RF centre. Bottom row: strongest patterns of the chance difference (two control sets), "
         "for comparison.\nA pattern that varies MORE before spikes = the cell responds to it whichever way it "
         "goes (e.g. two subunits that each respond to dark: either one dark is enough).", fontsize=8.5)
fig.tight_layout()
stem = (f"stc_minus_random_cell_{args.cell}" if args.control == "random" and not args.crop_um else
        f"stc_minus_{args.control}_cell_{args.cell}" + (f"_crop{args.crop_um:.0f}um" if args.crop_um else ""))
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=200 if fmt == "png" else None, bbox_inches="tight")
np.savez(OUTPUT_FOLDER / f"{stem}.npz", eigenvalues=ev, eigenvalues_chance=ev_c, k=k, strength=strength,
         patterns_more=vec[:, more].T.reshape(-1, H, W), patterns_less=vec[:, less].T.reshape(-1, H, W),
         chance_limits=[lim_lo, lim_hi])
print("saved", OUTPUT_FOLDER / f"{stem}.png")
