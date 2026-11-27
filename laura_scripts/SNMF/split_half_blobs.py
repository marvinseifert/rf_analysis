"""
Do the blobs in one cell's RF map come back in independent halves of its spikes? (20/11/2026)

Split the cell's snippets (spikes) into two halves, make the RF map from each half, and compare:
  - split "time":   first half of the recording vs second half. CAVEAT: the 30-min noise sequence is played
                    3 times identically, so both halves contain the same stimulus frames (repeat 1 vs repeat 3):
                    the halves are NOT independent and look alike even where there is only noise.
  - split "stim":   spikes during the first half of the noise SEQUENCE (minutes 0-15, any repeat) vs the second
                    half (15-30). The halves share no stimulus frames -> independent. Spikes within 1 s of a
                    half boundary are left out (their 0.8 s snippet would reach into the other half).
  - chance:         the same "stim" split of a SHIFTED spike train (random_control_snippets.py --mode shifted
                    --seed 1): same spikes and bursts, no link to the stimulus, so every blob is noise.

RF map of a half = its STA collapsed over time with ONE time course k (the full-data STA time course of the
RF centre, as in snmf_one_time_course.py), so the map is positive where the cell responds (ON or OFF).
Blobs = local peaks of the map (lightly smoothed, sigma --smooth-um) that are >= --peak-fraction of that map's
maximum and >= --min-sep-um apart. A blob "comes back" if the other half has a blob within --match-um.
Similarity = correlation of the two half-maps inside the RF (pixels >= 20 % of the full map's peak).

Zero-filled edge columns/rows cropped; first 100 snippets dropped (as gollisch_pipeline.py).
Outputs (Box, Figures/SNMF): figure (png + pdf).

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python laura_scripts/SNMF/split_half_blobs.py --cell 252
"""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, required=True)
parser.add_argument("--smooth-um", type=float, default=2.0)
parser.add_argument("--peak-fraction", type=float, default=0.5)
parser.add_argument("--min-sep-um", type=float, default=8.0)
parser.add_argument("--match-um", type=float, default=4.0)
parser.add_argument("--centre-fraction", type=float, default=0.5)
parser.add_argument("--rf-fraction", type=float, default=0.2, help="RF area for the correlation")
parser.add_argument("--stim-folder", type=Path, default=Path(
    "/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00/6px_15Hz_30mins_shuffle_x6_idx_10"))
parser.add_argument("--drop-first", type=int, default=100)
parser.add_argument("--block", type=int, default=1000)
args = parser.parse_args()

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/SNMF")
UM_PER_PX = 2.0
COL_A, COL_B = "tab:green", "tab:blue"
cell_dir = args.stim_folder / f"cell_{args.cell}"
real_file = cell_dir / "snippets.h5"
shifted_file = cell_dir / "snippets_shifted_seed1.h5"
for f in (real_file, shifted_file):
    if not f.exists():
        raise SystemExit(f"missing {f}")


def valid_slices(dset):
    a = dset.attrs
    ys = slice(int(a["cut_begin_y"] - a["window_origin_y"]), int(a["cut_end_y"] - a["window_origin_y"]))
    xs = slice(int(a["cut_begin_x"] - a["window_origin_x"]), int(a["cut_end_x"] - a["window_origin_x"]))
    return ys, xs


def per_snippet_sums(path, groups):
    """Sum of snippets (T, H, W) for each group of snippet indices (after drop_first)."""
    with h5py.File(path, "r") as f:
        d = f["snippets"]
        ys, xs = valid_slices(d)
        n = d.shape[0] - args.drop_first
        sums = {g: 0.0 for g in groups}
        counts = {g: 0 for g in groups}
        labels = np.full(n, -1)
        for g, idx in groups.items():
            labels[idx] = list(groups).index(g)
        for i in range(0, n, args.block):
            b = d[args.drop_first + i:args.drop_first + min(i + args.block, n), :, ys, xs].astype(np.float32)
            lab = labels[i:i + len(b)]
            for gi, g in enumerate(groups):
                sel = lab == gi
                if sel.any():
                    sums[g] = sums[g] + b[sel].sum(axis=0)
                    counts[g] += int(sel.sum())
    return {g: sums[g] / counts[g] - 0.5 for g in groups}, counts


# ---- spike times of the snippet rows (same selection and order as the snippet scripts) ----
from polarspike import Overview  # noqa: E402

project_root = args.stim_folder.parent
stim_index = int(args.stim_folder.name.rsplit("_idx_", 1)[1])
recording = Overview.Recording.load(project_root / "overview")
recording.dataframes["noise_analysis"] = recording.dataframes["spikes_df"].query("stimulus_index == @stim_index")
triggers = (recording.dataframes["stimulus_df"].query("stimulus_index == @stim_index")["trigger_fr_relative"]
            .to_numpy())[0]
spikes = recording.get_spikes_triggered([{"stimulus_index": stim_index, "cell_index": args.cell}],
                                        time="frames", pandas=False)
if len(spikes) > 50000:
    spikes = spikes.sample(n=50000, seed=0)
t_real = np.sort(spikes["times_relative"].to_numpy().astype(np.int64).ravel())
noise_params = json.loads((args.stim_folder / "noise_params.json").read_text())
fs, per_repeat, n_rep = noise_params["sample_freq"], noise_params["trigger_per_repeat"], noise_params["stim_repeats"]
repeat_starts = np.asarray(triggers)[np.arange(n_rep) * per_repeat]
repeat_len = float(np.median(np.diff(np.append(repeat_starts, triggers.max()))))
shift_params = json.loads((cell_dir / "params_snippets_shifted_seed1.json").read_text())
t_shift = np.sort((t_real + shift_params["offset_samples"]) % int(triggers.max()))


def stim_halves(t):
    """Rows (after drop_first) in the first / second half of the noise sequence, 1 s from the boundaries."""
    t = t[args.drop_first:]
    rep_i = np.clip(np.searchsorted(repeat_starts, t, side="right") - 1, 0, n_rep - 1)
    phase = (t - repeat_starts[rep_i]) / fs           # s into the sequence
    half = repeat_len / fs / 2
    a = np.where((phase >= 1) & (phase < half - 1))[0]
    b = np.where((phase >= half + 1) & (phase < 2 * half - 1))[0]
    return a, b


with h5py.File(real_file, "r") as f:
    n = f["snippets"].shape[0] - args.drop_first
if n != len(t_real) - args.drop_first:
    raise SystemExit(f"snippet count {n + args.drop_first} != spike count {len(t_real)}: spike selection differs")
sa, sb = stim_halves(t_real)
xa, xb = stim_halves(t_shift)
sta_t, cnt_t = per_snippet_sums(real_file, {"time_A": np.arange(n // 2), "time_B": np.arange(n // 2, n)})
sta_r, cnt_r = per_snippet_sums(real_file, {"stim_A": sa, "stim_B": sb})
sta_full = (sta_t["time_A"] * cnt_t["time_A"] + sta_t["time_B"] * cnt_t["time_B"]) / (cnt_t["time_A"] + cnt_t["time_B"])
sta_s, cnt_s = per_snippet_sums(shifted_file, {"shift_A": xa, "shift_B": xb})
print(f"spikes per half: time {cnt_t}, stim {cnt_r}, shifted {cnt_s}")

strength = np.max(sta_full ** 2, axis=0)
centre = strength >= args.centre_fraction * strength.max()
k = sta_full[:, centre].mean(axis=1)
k /= np.linalg.norm(k)


def rf_map(sta):
    return np.tensordot(k, sta, axes=(0, 0))  # (H, W), positive where the cell responds


full = rf_map(sta_full)
rf_area = full >= args.rf_fraction * full.max()
maps = {g: rf_map(s) for g, s in {**sta_t, **sta_r, **sta_s}.items()}
H, W = full.shape
sig = args.smooth_um / UM_PER_PX


def blobs(m, restrict=None):
    s = gaussian_filter(m, sig)
    pk = peak_local_max(s, min_distance=max(1, int(round(args.min_sep_um / UM_PER_PX))),
                        threshold_abs=args.peak_fraction * s.max(), exclude_border=False)
    if restrict is not None:
        pk = pk[restrict[pk[:, 0], pk[:, 1]]]
    return pk


def matched(pa, pb):
    if len(pa) == 0 or len(pb) == 0:
        return np.zeros(len(pa), bool)
    d = np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1))
    return d.min(axis=1) * UM_PER_PX <= args.match_um


def corr_in(a, b, mask):
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


rows = [("time", "time_A", "time_B", "First vs second half of the recording\n(share stimulus frames: not independent)"),
        ("stim", "stim_A", "stim_B", "First vs second half of the noise sequence\n(no shared frames: independent)"),
        ("shift", "shift_A", "shift_B", "Chance: shifted spike train,\nsame split as the row above")]
summary = {}
for key, a, b, _ in rows:
    # chance maps have no RF: look for blobs in the same RF area as the real cell
    pa, pb = blobs(maps[a], rf_area), blobs(maps[b], rf_area)
    ma, mb = matched(pa, pb), matched(pb, pa)
    summary[key] = dict(pa=pa, pb=pb, ma=ma, mb=mb, r=corr_in(maps[a], maps[b], rf_area),
                        r_out=corr_in(maps[a], maps[b], ~rf_area))
    print(f"{key}: blobs in RF A {len(pa)} ({ma.sum()} come back), B {len(pb)} ({mb.sum()} come back); "
          f"correlation of the two maps inside RF {summary[key]['r']:.2f}, outside RF {summary[key]['r_out']:.2f}")

# ---- figure ----
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 3, figsize=(13, 13.5))
for r_i, (key, a, b, title) in enumerate(rows):
    s = summary[key]
    lim = max(np.abs(maps[a]).max(), np.abs(maps[b]).max()) if key != "shift" else None
    for c_i, (g, pk, mk, colour, lab) in enumerate([(a, s["pa"], s["ma"], COL_A, "half A"),
                                                    (b, s["pb"], s["mb"], COL_B, "half B")]):
        ax = axes[r_i, c_i]
        m = gaussian_filter(maps[g], sig)
        v = lim if lim is not None else np.abs(m).max()
        ax.imshow(m, cmap="gray", vmin=0, vmax=v * 0.9, origin="lower")
        ax.contour(rf_area, levels=[0.5], colors="white", linewidths=0.6, linestyles=":")
        for (py, px), ok in zip(pk, mk):
            ax.plot(px, py, "o", ms=9, mfc="none", mec=colour, mew=2 if ok else 1, ls="")
            if not ok:
                ax.plot(px, py, "x", ms=6, color=colour)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: {len(pk)} blobs, {int(mk.sum())} come back", fontsize=9, loc="left")
        if c_i == 0:
            ax.set_ylabel(title, fontsize=10)
    ax = axes[r_i, 2]
    ax.imshow(gaussian_filter(full, sig), cmap="gray", origin="lower", vmin=0)
    for pk, mk, colour in [(s["pa"], s["ma"], COL_A), (s["pb"], s["mb"], COL_B)]:
        ax.plot(pk[mk, 1], pk[mk, 0], "o", ms=9, mfc="none", mec=colour, mew=2, ls="")
        ax.plot(pk[~mk, 1], pk[~mk, 0], "x", ms=6, color=colour, ls="")
    ax.contour(rf_area, levels=[0.5], colors="white", linewidths=0.6, linestyles=":")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Both halves on the full map\nmaps alike inside RF: r = {s['r']:.2f} "
                 f"(outside: {s['r_out']:.2f})", fontsize=9, loc="left")
for ax in axes[:, 0]:
    ax.plot([1, 1 + 10], [1, 1], "w-", lw=3)
    ax.text(6, 2.2, "20 µm", color="white", ha="center", fontsize=8)
fig.suptitle(f"Cell {args.cell}: do the RF blobs come back in independent halves of the spikes?",
             fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.0,
         f"Blob = local peak >= {args.peak_fraction:.0%} of that map's maximum, >= {args.min_sep_um:.0f} µm apart, "
         f"inside the RF (dotted, >= {args.rf_fraction:.0%} of the full map's peak); maps smoothed "
         f"{args.smooth_um:.0f} µm. Thick circle = a blob within {args.match_um:.0f} µm in the other half; "
         f"x = no match.\nr = how alike the two half-maps are (1 = identical). Bottom row: same analysis on a shifted "
         f"spike train - how often blobs 'come back' by chance.", fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
stem = f"split_half_blobs_cell_{args.cell}"
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=180 if fmt == "png" else None, bbox_inches="tight")
print("saved", OUTPUT_FOLDER / f"{stem}.png")
