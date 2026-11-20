"""
How fine is the reliable detail in one cell's RF map? Fourier ring correlation of two independent halves
(20/11/2026).

Two RF maps from independent halves of the spikes (first vs second half of the noise SEQUENCE, any repeat,
spikes within 1 s of a boundary left out - same split as split_half_blobs.py; the halves share no stimulus
frames). Each map = STA collapsed with ONE time course k (STA time course of the RF centre).

Fourier ring correlation (FRC, standard resolution test in microscopy): both maps are split into detail sizes
(spatial frequencies); for each size, how alike are the two halves (1 = identical, 0 = unrelated)?
Detail size = period of the pattern (e.g. 16 µm = a light-dark pattern repeating every 16 µm; a blob of
~8 µm across is detail of ~16 µm). The reliable resolution = the finest detail size where the halves still
agree above the usual 1/7 criterion. Chance: same analysis on a shifted spike train (random_control_snippets.py
--mode shifted --seed 1).

Maps: largest square window on the stimulus (edge columns cropped), mean removed, Hann taper (so the window edge
does not create fake fine detail).

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python laura_scripts/SNMF/rf_resolution_test.py --cell 150
"""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from polarspike import Overview

parser = argparse.ArgumentParser()
parser.add_argument("--cell", type=int, required=True)
parser.add_argument("--n-rings", type=int, default=12)
parser.add_argument("--threshold", type=float, default=1 / 7)
parser.add_argument("--cone-spacing-um", type=float, nargs=2, default=[15, 20])
parser.add_argument("--centre-fraction", type=float, default=0.5)
parser.add_argument("--stim-folder", type=Path, default=Path(
    "/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00/6px_15Hz_30mins_shuffle_x6_idx_10"))
parser.add_argument("--drop-first", type=int, default=100)
parser.add_argument("--block", type=int, default=1000)
args = parser.parse_args()

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/SNMF")
UM_PER_PX = 2.0
REAL, NULL = "tab:blue", "tab:green"
cell_dir = args.stim_folder / f"cell_{args.cell}"
real_file, shifted_file = cell_dir / "snippets.h5", cell_dir / "snippets_shifted_seed1.h5"
for f in (real_file, shifted_file):
    if not f.exists():
        raise SystemExit(f"missing {f}")

# ---- spike times of the snippet rows (same selection/order as the snippet scripts) ----
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
offset = json.loads((cell_dir / "params_snippets_shifted_seed1.json").read_text())["offset_samples"]
t_shift = np.sort((t_real + offset) % int(triggers.max()))


def seq_halves(t):
    t = t[args.drop_first:]
    rep_i = np.clip(np.searchsorted(repeat_starts, t, side="right") - 1, 0, n_rep - 1)
    phase = (t - repeat_starts[rep_i]) / fs
    half = repeat_len / fs / 2
    return (np.where((phase >= 1) & (phase < half - 1))[0],
            np.where((phase >= half + 1) & (phase < 2 * half - 1))[0])


def valid_slices(dset):
    a = dset.attrs
    ys = slice(int(a["cut_begin_y"] - a["window_origin_y"]), int(a["cut_end_y"] - a["window_origin_y"]))
    xs = slice(int(a["cut_begin_x"] - a["window_origin_x"]), int(a["cut_end_x"] - a["window_origin_x"]))
    return ys, xs


def half_stas(path, idx_a, idx_b):
    with h5py.File(path, "r") as f:
        d = f["snippets"]
        ys, xs = valid_slices(d)
        n = d.shape[0] - args.drop_first
        if n != len(t_real) - args.drop_first:
            raise SystemExit(f"{path.name}: {d.shape[0]} snippets but {len(t_real)} spikes")
        lab = np.full(n, -1); lab[idx_a] = 0; lab[idx_b] = 1
        sums, cnt = [0.0, 0.0], [0, 0]
        for i in range(0, n, args.block):
            b = d[args.drop_first + i:args.drop_first + min(i + args.block, n), :, ys, xs].astype(np.float32)
            for g in (0, 1):
                sel = lab[i:i + len(b)] == g
                sums[g] = sums[g] + b[sel].sum(axis=0); cnt[g] += int(sel.sum())
    return [s / c - 0.5 for s, c in zip(sums, cnt)], cnt


real_a, real_b = seq_halves(t_real)
sh_a, sh_b = seq_halves(t_shift)
(sta_a, sta_b), cnt = half_stas(real_file, real_a, real_b)
(shs_a, shs_b), cnt_s = half_stas(shifted_file, sh_a, sh_b)
sta_full = (sta_a * cnt[0] + sta_b * cnt[1]) / sum(cnt)
strength = np.max(sta_full ** 2, axis=0)
centre = strength >= args.centre_fraction * strength.max()
k = sta_full[:, centre].mean(axis=1)
k /= np.linalg.norm(k)

# largest square, centred on the RF centre as far as the window allows
H, W = strength.shape
S = min(H, W)
yy, xx = np.mgrid[:H, :W]
wc = strength * centre
cy, cx = (wc * yy).sum() / wc.sum(), (wc * xx).sum() / wc.sum()
y0 = int(np.clip(round(cy - S / 2), 0, H - S)); x0 = int(np.clip(round(cx - S / 2), 0, W - S))
sq = (slice(y0, y0 + S), slice(x0, x0 + S))


def rf_map(sta):
    return np.tensordot(k, sta, axes=(0, 0))[sq]


maps = {"real_a": rf_map(sta_a), "real_b": rf_map(sta_b), "sh_a": rf_map(shs_a), "sh_b": rf_map(shs_b)}
taper = np.outer(np.hanning(S), np.hanning(S))
fy = np.fft.fftfreq(S, d=UM_PER_PX)
fr = np.hypot(*np.meshgrid(fy, fy, indexing="ij"))         # cycles per µm
edges = np.linspace(0, fr[fr > 0].min() * 0 + 1 / (2 * UM_PER_PX), args.n_rings + 1)  # up to the grid limit
edges[0] = fy[1] * 0.5                                        # leave out the mean


def frc(a, b):
    A = np.fft.fft2((a - a.mean()) * taper); B = np.fft.fft2((b - b.mean()) * taper)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fr >= lo) & (fr < hi)
        out.append(np.real((A[m] * np.conj(B[m])).sum()) / np.sqrt((np.abs(A[m]) ** 2).sum() * (np.abs(B[m]) ** 2).sum()))
    return np.array(out)


f_mid = (edges[:-1] + edges[1:]) / 2
period = 1 / f_mid
frc_real = frc(maps["real_a"], maps["real_b"])
frc_null = frc(maps["sh_a"], maps["sh_b"])
above = frc_real >= args.threshold
# resolution = finest detail before the curve FIRST drops below the threshold
first_drop = np.argmax(~above) if (~above).any() else len(above)
resolution = period[first_drop - 1] if first_drop > 0 else np.nan
print(f"cell {args.cell}: halves {cnt} spikes (shifted {cnt_s}); square {S} x {S} px = {S * UM_PER_PX:.0f} µm")
for p, r1, r0 in zip(period, frc_real, frc_null):
    print(f"  detail {p:5.1f} µm: halves agree {r1:+.2f}   chance {r0:+.2f}")
print(f"reliable down to about {resolution:.0f} µm detail (threshold {args.threshold:.2f})")

# ---- figure ----
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
lim = max(np.abs(maps["real_a"]).max(), np.abs(maps["real_b"]).max())
for i, (key, lab) in enumerate([("real_a", "half A"), ("real_b", "half B")]):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(maps[key], cmap="gray", vmin=-lim * 0.2, vmax=lim, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"RF map, {lab}\n({cnt[i]} spikes)", fontsize=10, loc="left")
    ax.plot([1, 11], [1, 1], "w-", lw=3); ax.text(6, 2.2, "20 µm", color="white", ha="center", fontsize=8)
ax = fig.add_subplot(gs[0, 2])
ax.axvspan(*args.cone_spacing_um, color="tab:orange", alpha=0.15, lw=0)
ax.text(np.mean(args.cone_spacing_um), 0.97, "cone\nspacing", ha="center", va="top", fontsize=8, color="tab:orange",
        transform=ax.get_xaxis_transform())
ax.plot(period, frc_real, "-o", color=REAL, label="two halves of the real spikes")
ax.plot(period, frc_null, "-o", color=NULL, label="chance (shifted spike train)")
ax.axhline(args.threshold, color="grey", ls="--", lw=1)
ax.text(period.max(), args.threshold + 0.02, "1/7 criterion", color="grey", fontsize=8, ha="right")
ax.axhline(0, color="grey", lw=0.5)
if np.isfinite(resolution):
    ax.axvline(resolution, color=REAL, ls=":", lw=1)
    ax.text(resolution, -0.05, f" reliable down to\n ~{resolution:.0f} µm", color=REAL, fontsize=9,
            transform=ax.get_xaxis_transform(), va="top")
ax.set_xscale("log")
ax.invert_xaxis()  # coarse left, fine right
ticks = [64, 32, 16, 8, 4]
ax.set_xticks(ticks, [str(t) for t in ticks])
ax.set_xlabel("size of detail (µm, period of the pattern)  -  coarse to fine")
ax.set_ylabel("how alike the two halves are\nat this detail size (1 = identical)")
ax.set_ylim(-0.4, 1.0)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Fourier ring correlation", fontsize=10, loc="left")
fig.suptitle(f"Cell {args.cell}: how fine is the reliable detail in the RF map? "
             f"(independent halves: first vs second half of the noise sequence)", fontsize=11, x=0.01, ha="left")
fig.text(0.01, -0.04,
         "Detail size = period: a pattern repeating every 16 µm (blobs ~8 µm across) is 16 µm detail. Above the 1/7 line "
         "the halves agree on that detail; the shifted spike train shows the chance level.", fontsize=8.5)
fig.tight_layout()
stem = f"rf_resolution_cell_{args.cell}"
for fmt in ("png", "pdf"):
    fig.savefig(OUTPUT_FOLDER / f"{stem}.{fmt}", dpi=180 if fmt == "png" else None, bbox_inches="tight")
print("saved", OUTPUT_FOLDER / f"{stem}.png")
