"""
How many sNMF subunits are reproducible, compared with random times? (20/11/2026)

The Moran's I cut-off does not work for the shuffled noise: real spikes and random times both give r of r
"subunits" (see snmf_real_vs_random.py). Instead, ask whether a subunit comes back when sNMF is run on a
DIFFERENT half of the data (the idea used by Liu, Gollisch et al. 2017 to choose r):

  for each r, for each repeat:
      split the snippets at random into two halves -> sNMF on each half
      pair each module of half A with its most similar module of half B (one-to-one, best total match)
      similarity of a pair = correlation of the two spatial maps (1 = identical, 0 = unrelated)
  same for the random-time snippets (snippets_random_control.h5) = what noise gives.

A real module counts as a REPRODUCIBLE SUBUNIT only if its similarity is higher than that of EVERY random-time
module at the same r (all repeats). The r where that number stops growing is the number of subunits the data
support.

Both ensembles are projected with the REAL cell's STA time course, and zero-filled edge columns/rows are cropped,
exactly as in snmf_real_vs_random.py. First 100 snippets dropped (as gollisch_pipeline.py).

Outputs (Box, Figures/SNMF): figure (png + pdf) + npz with every similarity.

Run from the repo root, ONE AT A TIME (2 x repeats x 2 sNMF runs per r; ~15-25 min with the defaults):
    MPLBACKEND=Agg poetry run python laura_scripts/SNMF/snmf_consistency.py --cell 252
Needs random_control_snippets.py --cells <cell> first.
"""

import argparse
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from stnmf import STNMF

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, required=True)
parser.add_argument("--r-values", type=int, nargs="+", default=[5, 10, 15, 20, 30])
parser.add_argument("--repeats", type=int, default=1, help="random half-splits per r")
parser.add_argument("--sparsity", type=float, default=0.5, help="gollisch_pipeline.py uses 0.5 (stnmf default 1.7)")
parser.add_argument("--stim-folder", type=Path, default=Path(
    "/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00/6px_15Hz_30mins_shuffle_x6_idx_10"))
parser.add_argument("--drop-first", type=int, default=100)
parser.add_argument("--block", type=int, default=1000)
args = parser.parse_args()

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/SNMF")
REAL_COLOUR, RANDOM_COLOUR = "tab:blue", "tab:green"
cell_dir = args.stim_folder / f"cell_{args.cell}"
files = {"real": cell_dir / "snippets.h5", "random": cell_dir / "snippets_random_control.h5"}
for f in files.values():
    if not f.exists():
        raise SystemExit(f"missing {f} (random file: run random_control_snippets.py --cells {args.cell})")


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
weights = total / n_real - 0.5  # real STA, signed


def project(path):
    with h5py.File(path, "r") as f:
        d = f["snippets"]
        if valid_slices(d) != (ys, xs):
            raise SystemExit(f"{path.name}: window differs from the real snippets")
        return np.concatenate([np.einsum("nthw,thw->hwn", b - 0.5, weights, optimize=True)
                               for b in blocks(d, ys, xs)], axis=2)


def similarity(mods_a, mods_b):
    """Correlation between every module of A and every module of B (flattened spatial maps)."""
    a = mods_a.reshape(len(mods_a), -1).astype(np.float64)
    b = mods_b.reshape(len(mods_b), -1).astype(np.float64)
    a = np.nan_to_num(a - a.mean(1, keepdims=True)); b = np.nan_to_num(b - b.mean(1, keepdims=True))
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return a @ b.T


ensembles = {name: project(path) for name, path in files.items()}
n = ensembles["real"].shape[2]
print(f"cell {args.cell}: {n} snippets each, window {ensembles['real'].shape[0]} x {ensembles['real'].shape[1]}")

sims = {name: {r: [] for r in args.r_values} for name in files}  # matched-pair similarities
t0 = time.time()
for r in args.r_values:
    for rep in range(args.repeats):
        order = np.random.default_rng(1000 * r + rep).permutation(n)  # same split for real and random
        halves = (order[: n // 2], order[n // 2:])
        for name, ste in ensembles.items():
            mods = [STNMF(ste[:, :, h], r=r, sparsity=args.sparsity).modules for h in halves]
            s = similarity(*mods)
            rows, cols = linear_sum_assignment(-s)
            sims[name][r].extend(s[rows, cols].tolist())
        print(f"r = {r}, repeat {rep + 1}: median pair similarity real {np.median(sims['real'][r][-r:]):.2f}, "
              f"random {np.median(sims['random'][r][-r:]):.2f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

n_reproducible = {}
for r in args.r_values:
    best_random = max(sims["random"][r])
    n_reproducible[r] = sum(v > best_random for v in sims["real"][r]) / args.repeats
    print(f"r = {r}: {n_reproducible[r]:.1f} real modules more similar across halves than the best random one "
          f"({best_random:.2f})")

# ---- figure ----
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
rng = np.random.default_rng(0)
for name, colour, dx, label in [("real", REAL_COLOUR, -0.6, "real spikes"), ("random", RANDOM_COLOUR, 0.6, "random times")]:
    for r in args.r_values:
        v = np.asarray(sims[name][r])
        ax.scatter(r + dx + rng.uniform(-0.35, 0.35, v.size), v, s=8, color=colour, alpha=0.35, lw=0)
    ax.plot(args.r_values, [np.median(sims[name][r]) for r in args.r_values], "-o", color=colour, label=f"{label} (median)")
ax.set_xlabel("number of sNMF modules (r)")
ax.set_ylabel("similarity of each module between the two halves\n(1 = same map found twice, 0 = unrelated)")
ax.set_xticks(args.r_values)
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(args.r_values, [n_reproducible[r] for r in args.r_values], "-o", color=REAL_COLOUR)
ax.plot(args.r_values, args.r_values, ":", color="grey", lw=1)
ax.text(args.r_values[-1], args.r_values[-1], " every module", color="grey", fontsize=8, va="center")
ax.set_xlabel("number of sNMF modules (r)")
ax.set_ylabel("reproducible subunits\n(more similar across halves than ANY random-time module)")
ax.set_xticks(args.r_values)
ax.set_ylim(bottom=0)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Cell {args.cell}: do sNMF subunits come back in the other half of the data? "
             f"(sparsity {args.sparsity}, {args.repeats} split{'s' if args.repeats > 1 else ''} per r)",
             fontsize=11, x=0.01, ha="left")
fig.text(0.01, -0.03,
         "Each dot = one module paired with its best match from sNMF on the other half of the snippets. Random times "
         "= same number of snippets at times unrelated to the cell's spikes (what the stimulus alone gives).\n"
         "Right: where the line stops rising, extra modules are no more reproducible than noise.", fontsize=8.5)
fig.tight_layout()
stem = f"snmf_consistency_cell_{args.cell}_sparsity{args.sparsity}"
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=200 if fmt == "png" else None, bbox_inches="tight")
np.savez(OUTPUT_FOLDER / f"{stem}.npz", r_values=args.r_values, repeats=args.repeats, sparsity=args.sparsity,
         **{f"sim_{name}_r{r}": np.asarray(sims[name][r]) for name in sims for r in args.r_values},
         n_reproducible=[n_reproducible[r] for r in args.r_values])
print("saved", OUTPUT_FOLDER / f"{stem}.png")
