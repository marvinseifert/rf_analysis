"""
sNMF with ONE time course for all pixels, real spikes vs random times, and two ways of judging subunits
(20/11/2026).

Step 1 (option 1): each snippet (100 bins x H x W) is collapsed to one image with the SAME time course k[t]
    for every pixel (Liu, Gollisch et al. 2017), instead of each pixel's own STA (gollisch_pipeline.py /
    snmf_real_vs_random.py). k = STA time course averaged over the RF centre (pixels >= 50 % of the peak of
    the RF strength map), scaled to length 1. The per-pixel weighting stamped the RF shape onto random data
    too; with one k, random-time images have the same variance everywhere.
    sNMF (r modules) is run on the real-spike images and on the random-time images.

Step 2, two ways of deciding which real modules are more than the stimulus alone gives:
  (a) SHAPE: for each real module, its correlation with the most similar random-time module
      (1 = the stimulus alone gives the same map).
  (b) USE BEFORE SPIKES: project every image onto the module (unit length) and ask how far the average is
      from 0, in standard errors (z). Random times: stimulus unbiased -> z ~ 0 +- 1. A pattern that is
      pushed one way before spikes has a large |z|.
      Three groups: real modules on real spikes; random modules on random times (the noise level);
      random modules on real spikes (does ANY blob inside the RF score high? -> shows what (b) can and
      cannot tell).

Zero-filled edge columns/rows are cropped; first 100 snippets dropped (both, as gollisch_pipeline.py).
Outputs (Box, Figures/SNMF): figure (png + pdf) + npz.

Run from the repo root, ONE AT A TIME (~3 min):
    MPLBACKEND=Agg poetry run python laura_scripts/SNMF/snmf_one_time_course.py --cell 252
Needs random_control_snippets.py --cells <cell> first.
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
parser.add_argument("--r", type=int, default=30)
parser.add_argument("--sparsity", type=float, default=0.5, help="gollisch_pipeline.py uses 0.5 (stnmf default 1.7)")
parser.add_argument("--centre-fraction", type=float, default=0.5, help="RF centre = strength >= this x peak")
parser.add_argument("--stim-folder", type=Path, default=Path(
    "/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00/6px_15Hz_30mins_shuffle_x6_idx_10"))
parser.add_argument("--drop-first", type=int, default=100)
parser.add_argument("--block", type=int, default=1000)
args = parser.parse_args()

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/SNMF")
REAL, RANDOM, CROSS = "tab:blue", "tab:green", "grey"
BIN_MS, POST_BINS = 10, 20  # 10 ms bins; last 20 bins are after the spike
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


# ---- real STA -> RF strength map and ONE time course k ----
with h5py.File(files["real"], "r") as f:
    d = f["snippets"]
    ys, xs = valid_slices(d)
    total, n_real = 0.0, 0
    for b in blocks(d, ys, xs):
        total = total + b.sum(axis=0)
        n_real += len(b)
sta = total / n_real - 0.5                     # (T, H, W), signed
strength = np.max(sta ** 2, axis=0)            # same map as gollisch_pipeline.py
centre = strength >= args.centre_fraction * strength.max()
k = sta[:, centre].mean(axis=1)
k /= np.linalg.norm(k)
H, W = strength.shape
peak_bin = int(np.argmax(np.abs(k)))
print(f"cell {args.cell}: {n_real} snippets, window {H} x {W}; RF centre = {centre.sum()} pixels; "
      f"time course peaks {(peak_bin - (len(k) - POST_BINS - 1)) * BIN_MS} ms from the spike")


def project(path):
    """(H, W, N): every pixel's time course weighted by the SAME k."""
    with h5py.File(path, "r") as f:
        d = f["snippets"]
        if valid_slices(d) != (ys, xs):
            raise SystemExit(f"{path.name}: window differs from the real snippets")
        return np.concatenate([np.einsum("nthw,t->hwn", b - 0.5, k, optimize=True)
                               for b in blocks(d, ys, xs)], axis=2)


def unit(mods):
    m = np.nan_to_num(mods.reshape(len(mods), -1).astype(np.float64))
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def usage_z(mods, ste):
    """How far the average projection onto each (unit) module is from 0, in standard errors."""
    proj = unit(mods) @ ste.reshape(-1, ste.shape[2]).astype(np.float64)  # (r, N)
    return proj.mean(axis=1) / (proj.std(axis=1, ddof=1) / np.sqrt(proj.shape[1]))


ste, res = {}, {}
for name, path in files.items():
    ste[name] = project(path)
    print(f"{name}: sNMF (r = {args.r}, sparsity = {args.sparsity}) ...", flush=True)
    m = STNMF(ste[name], r=args.r, sparsity=args.sparsity)
    res[name] = dict(modules=np.asarray(m.modules), moransi=np.asarray(m.moransi, float),
                     outlines=list(spatial_outlines) if (spatial_outlines := m.outlines) is not None else [],
                     n_moransi=int(m.num_subunits))
    print(f"  {name}: Moran's I median {np.median(res[name]['moransi']):.2f}; "
          f"{res[name]['n_moransi']} of {args.r} pass the 0.25 cut-off")

# (a) shape: each real module vs its most similar random module
def centred_unit(mods):
    m = np.nan_to_num(mods.reshape(len(mods), -1).astype(np.float64))
    m -= m.mean(axis=1, keepdims=True)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


best_match = (centred_unit(res["real"]["modules"]) @ centred_unit(res["random"]["modules"]).T).max(axis=1)

# (b) use before spikes
z_real = usage_z(res["real"]["modules"], ste["real"])
z_null = usage_z(res["random"]["modules"], ste["random"])
z_cross = usage_z(res["random"]["modules"], ste["real"])
null_max = np.abs(z_null).max()
n_b = int((np.abs(z_real) > null_max).sum())
n_b_cross = int((np.abs(z_cross) > null_max).sum())
print(f"(a) real module vs most similar random module: median r = {np.median(best_match):.2f} "
      f"(range {best_match.min():.2f}-{best_match.max():.2f}); {int((best_match < 0.5).sum())} below 0.5")
print(f"(b) |z| random modules on random times: max {null_max:.1f} (noise level)")
print(f"    real modules on real spikes: median |z| {np.median(np.abs(z_real)):.1f}; {n_b} of {args.r} above noise")
print(f"    random modules on real spikes: median |z| {np.median(np.abs(z_cross)):.1f}; {n_b_cross} of {args.r} above noise")

# ---- figure ----
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
t_ms = (np.arange(len(k)) - (len(k) - POST_BINS - 1)) * BIN_MS

ax = axes[0, 0]
ax.plot(t_ms, k, color="black")
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_xlabel("time from spike (ms)"); ax.set_ylabel("weight")
ax.set_title("The one time course used for every pixel\n(STA time course of the RF centre)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

for ax, name, colour, label in [(axes[0, 1], "real", REAL, "Real spikes"),
                                (axes[0, 2], "random", RANDOM, "Random times")]:
    ax.imshow(strength, cmap="gray", origin="lower")
    ax.contour(centre, levels=[0.5], colors="white", linewidths=0.8, linestyles=":")
    for c in res[name]["outlines"]:
        ax.plot(c[:, 1], c[:, 0], color=colour, lw=1.2)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}: sNMF modules (r = {args.r})\nMoran's I cut-off would count "
                 f"{res[name]['n_moransi']} of {args.r}", fontsize=10, loc="left")
    ax.plot([2, 27], [2, 2], "w-", lw=3)
    ax.text(14.5, 3.5, "50 µm", color="white", ha="center", fontsize=8)
axes[0, 2].text(W - 1, 1, "dotted = RF centre", color="white", ha="right", fontsize=8)

ax = axes[1, 0]
order = np.argsort(best_match)
ax.bar(np.arange(args.r), best_match[order], color=REAL)
ax.set_ylim(0, 1)
ax.set_xlabel("real modules (sorted)"); ax.set_ylabel("correlation with the most similar\nrandom-time module")
ax.set_title("(a) Shape: does the stimulus alone give the same map?\n1 = identical map found in random times",
             fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 1]
rng = np.random.default_rng(0)
groups = [(z_real, REAL, "real modules\non real spikes"), (z_null, RANDOM, "random modules\non random times"),
          (z_cross, CROSS, "random modules\non real spikes")]
for i, (z, colour, _) in enumerate(groups):
    ax.scatter(i + rng.uniform(-0.15, 0.15, z.size), np.abs(z), s=18, color=colour, alpha=0.8)
ax.axhline(null_max, color=RANDOM, ls="--", lw=1)
ax.text(2.45, null_max, "highest\nnoise value", color=RANDOM, fontsize=8, va="center")
ax.set_yscale("symlog", linthresh=5)
ax.set_xticks(range(3), [g[2] for g in groups], fontsize=9)
ax.set_xlim(-0.5, 3.0)
ax.set_ylabel("how strongly the pattern is pushed before spikes\n(|z| = distance of the average from 0, in standard errors)")
ax.set_title(f"(b) Use before spikes: {n_b} of {args.r} real modules above noise\n"
             f"(but {n_b_cross} of {args.r} random-time blobs too, on real spikes)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 2]
ax.scatter(best_match, np.abs(z_real), color=REAL, s=22)
ax.axhline(null_max, color=RANDOM, ls="--", lw=1)
ax.set_yscale("symlog", linthresh=5)
ax.set_xlim(0, 1)
ax.set_xlabel("(a) correlation with most similar random-time module")
ax.set_ylabel("(b) |z| before spikes")
ax.set_title("Each real module: (a) against (b)\nWanted: left (unlike noise) AND high (used before spikes)",
             fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Cell {args.cell}: sNMF with one time course for all pixels - real spikes vs random times "
             f"(r = {args.r}, sparsity = {args.sparsity})", fontsize=12, x=0.01, ha="left")
fig.tight_layout()
stem = f"snmf_one_time_course_cell_{args.cell}_r{args.r}_sparsity{args.sparsity}"
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=200 if fmt == "png" else None, bbox_inches="tight")
np.savez(OUTPUT_FOLDER / f"{stem}.npz", k=k, best_match=best_match, z_real=z_real, z_null=z_null,
         z_cross=z_cross, moransi_real=res["real"]["moransi"], moransi_random=res["random"]["moransi"],
         modules_real=res["real"]["modules"], modules_random=res["random"]["modules"])
print("saved", OUTPUT_FOLDER / f"{stem}.png")
