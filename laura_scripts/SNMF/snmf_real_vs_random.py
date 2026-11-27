"""
sNMF on real spikes vs random times for one cell (20/11/2026).

Based on s_nmf/gollisch_pipeline.py (not edited). Question: does the shuffled noise alone make sNMF find a
full set of blob-shaped "subunits"? If the random-time snippets (made by random_control_snippets.py) give as
many "subunits" with Moran's I as high as the real spikes, the count from the real spikes means nothing.

Same steps as gollisch_pipeline.py, with three differences (all stated on the figure):
  1. Zero-filled columns/rows (window running off the stimulus edge, e.g. cell 152) are cropped away first.
  2. Real AND random snippets are projected with the REAL cell's STA (time course weighting), so the only
     difference between the two runs is real spikes vs random times.
  3. Snippets are read in blocks (low RAM) instead of loading the whole file.
As in gollisch_pipeline.py, the first 100 snippets are dropped (both runs).

"Subunit" = sNMF module with Moran's I > threshold (stnmf default 0.25). Moran's I measures how smooth/blobby
a module is; it was designed for uncorrelated white noise.

Outputs (Box, Figures/SNMF): figure (png + pdf) and an .npz with Moran's I of every module for both runs.

Run from the repo root, after random_control_snippets.py, ONE AT A TIME (sNMF takes several minutes):
    MPLBACKEND=Agg poetry run python laura_scripts/SNMF/snmf_real_vs_random.py --cell 252 --r 30
"""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from stnmf import STNMF

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, required=True)
parser.add_argument("--r", type=int, default=30, help="number of sNMF modules")
parser.add_argument("--sparsity", type=float, default=0.5, help="gollisch_pipeline.py uses 0.5 (stnmf default 1.7)")
parser.add_argument("--moransi-threshold", type=float, default=0.25)
parser.add_argument("--stim-folder", type=Path, default=Path(
    "/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00/6px_15Hz_30mins_shuffle_x6_idx_10"))
parser.add_argument("--drop-first", type=int, default=100)
parser.add_argument("--block", type=int, default=1000, help="snippets read per block")
args = parser.parse_args()

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/SNMF")
REAL_COLOUR, RANDOM_COLOUR = "tab:blue", "tab:green"
cell_dir = args.stim_folder / f"cell_{args.cell}"
files = {"real": cell_dir / "snippets.h5", "random": cell_dir / "snippets_random_control.h5"}
for f in files.values():
    if not f.exists():
        raise SystemExit(f"missing {f} (random file: run random_control_snippets.py --cells {args.cell})")


def valid_slices(dset):
    """Rows/columns of the 60x60 window that lie on the real stimulus (not zero-filled)."""
    a = dset.attrs
    ys = slice(int(a["cut_begin_y"] - a["window_origin_y"]), int(a["cut_end_y"] - a["window_origin_y"]))
    xs = slice(int(a["cut_begin_x"] - a["window_origin_x"]), int(a["cut_end_x"] - a["window_origin_x"]))
    return ys, xs


def blocks(dset, ys, xs):
    for i in range(args.drop_first, dset.shape[0], args.block):
        yield dset[i:min(i + args.block, dset.shape[0]), :, ys, xs].astype(np.float32)


# ---- STA of the real spikes (pass 1) ----
with h5py.File(files["real"], "r") as f:
    d = f["snippets"]
    ys, xs = valid_slices(d)
    total, n_real = 0.0, 0
    for b in blocks(d, ys, xs):
        total = total + b.sum(axis=0)
        n_real += len(b)
sta = total / n_real  # (T, H, W), values 0..1
weights = sta - 0.5   # signed time-course weighting, as in gollisch_pipeline.py
strength = np.max((sta - 0.5) ** 2, axis=0)  # the black/white "mse" map of gollisch_pipeline.py
H, W = strength.shape
print(f"cell {args.cell}: {n_real} real snippets, window on stimulus {H} x {W} pixels "
      f"(cropped from 60 x 60)" if (H, W) != (60, 60) else f"cell {args.cell}: {n_real} real snippets, 60 x 60")


def project(path):
    """(H, W, N) ensemble: each snippet collapsed over time with the REAL STA weights."""
    with h5py.File(path, "r") as f:
        d = f["snippets"]
        ys2, xs2 = valid_slices(d)
        if (ys2, xs2) != (ys, xs):
            raise SystemExit(f"{path.name}: window differs from the real snippets")
        out = [np.einsum("nthw,thw->hwn", b - 0.5, weights, optimize=True) for b in blocks(d, ys, xs)]
    return np.concatenate(out, axis=2)


results = {}
for name, path in files.items():
    ste = project(path)
    print(f"{name}: {ste.shape[2]} snippets -> sNMF (r = {args.r}, sparsity = {args.sparsity}) ...", flush=True)
    model = STNMF(ste, r=args.r, sparsity=args.sparsity, moransi_threshold=args.moransi_threshold)
    del ste
    mi = np.asarray(model.moransi, float)
    areas = [int((m > 0.5 * np.nanmax(m)).sum()) for m in model.subunits]
    results[name] = dict(moransi=mi, n_sub=int(model.num_subunits), outlines=list(model.outlines), areas=areas)
    print(f"  {name}: {model.num_subunits} of {args.r} modules pass Moran's I > {args.moransi_threshold}; "
          f"Moran's I median {np.median(mi):.2f} (range {mi.min():.2f}-{mi.max():.2f}); "
          f"subunit area >50% of peak: median {np.median(areas) if areas else float('nan'):.0f} pixels")

# ---- figure ----
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))

ax = axes[0]
rng = np.random.default_rng(0)
for k, (name, colour) in enumerate([("real", REAL_COLOUR), ("random", RANDOM_COLOUR)]):
    mi = results[name]["moransi"]
    ax.scatter(k + rng.uniform(-0.15, 0.15, mi.size), mi, s=18, color=colour, alpha=0.8)
ax.axhline(args.moransi_threshold, color="grey", ls="--", lw=1)
ax.text(1.45, args.moransi_threshold + 0.02, f"cut-off {args.moransi_threshold}\n(above = counted\nas a subunit)",
        va="bottom", fontsize=8, color="grey")
ax.set_xticks([0, 1], [f"real spikes\n{results['real']['n_sub']} of {args.r} counted",
                       f"random times\n{results['random']['n_sub']} of {args.r} counted"])
ax.set_xlim(-0.5, 2.1)
ax.set_ylabel("Moran's I of each sNMF module\n(how smooth / blob-like it is)")
ax.spines[["top", "right"]].set_visible(False)

for ax, name, colour in [(axes[1], "real", REAL_COLOUR), (axes[2], "random", RANDOM_COLOUR)]:
    ax.imshow(strength, cmap="gray", origin="lower")  # gollisch_pipeline.py flips y the same way
    for c in results[name]["outlines"]:
        ax.plot(c[:, 1], c[:, 0], color=colour, lw=1.2)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)  # outlines can run past the edge
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{'Real spikes' if name == 'real' else 'Random times (no real spikes)'}: "
                 f"{results[name]['n_sub']} subunits", fontsize=10, loc="left")
    ax.plot([2, 2 + 25], [2, 2], "w-", lw=3)  # 25 snippet pixels = 50 µm (2 µm per pixel)
    ax.text(2 + 12.5, 3.5, "50 µm", color="white", ha="center", fontsize=8)

fig.suptitle(f"Cell {args.cell}: sNMF on real spikes vs the same number of random times "
             f"(r = {args.r}, sparsity = {args.sparsity})", fontsize=11, x=0.01, ha="left")
fig.text(0.01, -0.02,
         "Grey background: real cell's RF strength. Both runs use the real cell's STA time course and the same "
         "window. If random times give as many smooth subunits as real spikes, the subunit count reflects the "
         "stimulus, not the cell.\nRandom subunits can still sit on the RF: the time-course weighting uses the real "
         "STA, which is strongest there.", fontsize=8.5)
fig.tight_layout()
stem = f"snmf_real_vs_random_cell_{args.cell}_r{args.r}_sparsity{args.sparsity}"
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=200 if fmt == "png" else None, bbox_inches="tight")
np.savez(OUTPUT_FOLDER / f"{stem}.npz", moransi_real=results["real"]["moransi"],
         moransi_random=results["random"]["moransi"], n_sub_real=results["real"]["n_sub"],
         n_sub_random=results["random"]["n_sub"], r=args.r, sparsity=args.sparsity)
print("saved", OUTPUT_FOLDER / f"{stem}.png")
